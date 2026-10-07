"""
brokers.py
==========
Where the live runner's orders go. Two implementations behind one interface:

  SimBroker    DEFAULT. Live NSE prices in, simulated fills out. Books fills the
               way the backtest does (signal close plus adverse slippage, stop
               first on an ambiguous bar, flat before the close) and charges the
               same contract-note costs, so live results are comparable with the
               backtest. Nothing is ever sent to a broker.

  AngelBroker  REAL MONEY on Angel One. Angel has no paper-trading sandbox, so
               every order this class places is a real order. It refuses to
               exist unless --real-money is passed AND the environment says
               ANGEL_REAL_MONEY=I_ACCEPT_REAL_LOSSES, and it enforces its own
               caps on order value and orders per day.

Exchange rules this respects (NSE/SEBI retail algo framework, April 2026): API
orders come only from the static IP registered on the SmartAPI app, and market
and IOC orders are not allowed. Entries are marketable LIMIT orders, priced a
few basis points through the signal close, and the stop is part of the ROBO
bracket, so a filled position is protected at the broker from its first second.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

from market import NSE, Market, order_charges

REAL_MONEY_ENV = "ANGEL_REAL_MONEY"
REAL_MONEY_VALUE = "I_ACCEPT_REAL_LOSSES"

Log = Callable[[str], None]


# ------------------------------------------------------------------ clock

class MarketClock:
    """Trading days from the exchange calendar, session hours from the Market."""

    def __init__(self, market: Market = NSE):
        self.market = market
        self._cal = None
        try:
            import pandas_market_calendars as mcal
            self._cal = mcal.get_calendar(market.name)
        except Exception:
            pass                          # weekdays only; holidays then look like dead sessions

    def now(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz=self.market.tz)

    def is_trading_day(self, day: dt.date) -> bool:
        if day.weekday() >= 5:
            return False
        if self._cal is None:
            return True
        return len(self._cal.schedule(start_date=day, end_date=day)) > 0

    def _at(self, day: dt.date, t: dt.time) -> pd.Timestamp:
        return pd.Timestamp(dt.datetime.combine(day, t), tz=self.market.tz)

    def is_open(self, now: Optional[pd.Timestamp] = None) -> bool:
        now = now or self.now()
        day = now.date()
        return (self.is_trading_day(day)
                and self._at(day, self.market.open) <= now < self._at(day, self.market.close))

    def next_open(self, now: Optional[pd.Timestamp] = None) -> pd.Timestamp:
        now = now or self.now()
        day = now.date()
        if self.is_trading_day(day) and now < self._at(day, self.market.open):
            return self._at(day, self.market.open)
        for k in range(1, 15):
            d = day + dt.timedelta(days=k)
            if self.is_trading_day(d):
                return self._at(d, self.market.open)
        return self._at(day + dt.timedelta(days=1), self.market.open)


# ------------------------------------------------------------------ interface

class Broker:
    name = "base"
    fills = "broker"           # what the dashboard card says: broker | simulated

    def __init__(self, market: Market = NSE, log: Log = print, dry_run: bool = False):
        self.market = market
        self.clock = MarketClock(market)
        self.log = log
        self.dry_run = dry_run

    def account(self) -> dict:
        raise NotImplementedError

    def positions(self) -> Dict[str, int]:
        raise NotImplementedError

    def is_open(self) -> bool:
        return self.clock.is_open()

    def next_open(self) -> pd.Timestamp:
        return self.clock.next_open()

    def on_bar(self, bars: Dict[str, pd.DataFrame], now: pd.Timestamp) -> List[dict]:
        """Bars completed since the last call. Only the simulator needs them."""
        return []

    def submit_bracket(self, symbol: str, side: int, qty: int, stop: float,
                       target: float, ref_price: float,
                       bar_ts: Optional[pd.Timestamp] = None) -> Optional[dict]:
        """bar_ts: the completed bar whose close triggered this entry."""
        raise NotImplementedError

    def flatten_all(self, reason: str = "FLATTEN",
                    now: Optional[pd.Timestamp] = None) -> None:
        raise NotImplementedError


def _slip(price: float, is_buy: bool, bps: float) -> float:
    return price * (1 + (bps / 10_000) * (1 if is_buy else -1))


# ------------------------------------------------------------------ simulator

class SimBroker(Broker):
    """
    Paper trading on live prices, with the backtest's fill rules.

    State lives in a JSON file so a restart mid-session neither forgets an open
    position nor enters it twice. Every closed trade is appended to a JSONL file
    in the backtest's trade schema, so live and backtest results line up.
    """
    name = "sim"
    fills = "simulated"

    def __init__(self, state_path: Path, trades_path: Path, equity: float = 100_000.0,
                 slippage_bps: float = 2.0, cost_model: str = "india_intraday",
                 leverage: float = 1.0, market: Market = NSE, log: Log = print,
                 dry_run: bool = False):
        super().__init__(market, log, dry_run)
        self.state_path = Path(state_path)
        self.trades_path = Path(trades_path)
        self.slippage_bps = slippage_bps
        self.cost_model = cost_model
        self.leverage = leverage
        self.state = {"cash": float(equity), "start_equity": float(equity),
                      "positions": {}, "last_price": {}}
        if self.state_path.exists():
            try:
                self.state.update(json.loads(self.state_path.read_text()))
            except json.JSONDecodeError:
                pass

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, default=str))
        tmp.replace(self.state_path)

    # -- account ---------------------------------------------------

    def account(self) -> dict:
        cash = self.state["cash"]
        unreal = 0.0
        for sym, p in self.state["positions"].items():
            last = self.state["last_price"].get(sym, p["entry"])
            unreal += p["side"] * (last - p["entry"]) * p["qty"]
        equity = cash + unreal
        return {"equity": equity, "cash": cash, "buying_power": equity * self.leverage,
                "multiplier": self.leverage}

    def positions(self) -> Dict[str, int]:
        return {s: p["side"] * p["qty"] for s, p in self.state["positions"].items()}

    # -- orders ----------------------------------------------------

    def submit_bracket(self, symbol, side, qty, stop, target, ref_price,
                       bar_ts=None) -> Optional[dict]:
        if symbol in self.state["positions"]:
            self.log(f"  {symbol}: SIM already holds a position, not entering twice")
            return None
        fill = _slip(ref_price, side > 0, self.slippage_bps)
        charges = order_charges(self.cost_model, side > 0, qty, fill)
        if self.dry_run:
            self.log(f"  DRY RUN (sim) would fill {'BUY' if side > 0 else 'SELL'} {qty} "
                     f"{symbol} @ {fill:.2f}, stop {stop:.2f}, target {target:.2f}")
            return None
        when = str(bar_ts) if bar_ts is not None else str(self.clock.now())
        self.state["cash"] -= charges
        self.state["positions"][symbol] = {
            "side": side, "qty": qty, "entry": fill, "raw_entry": ref_price,
            "stop": stop, "target": target, "charges": charges,
            # Exits are only checked on bars AFTER the one we entered on.
            "entry_time": when, "entry_bar": when,
        }
        self.state["last_price"][symbol] = ref_price
        self._save()
        return {"id": f"SIM-{symbol}-{int(time.time())}", "symbol": symbol, "qty": qty,
                "side": "buy" if side > 0 else "sell", "status": "filled (simulated)",
                "fill": round(fill, 2), "charges": round(charges, 2)}

    def _close(self, symbol: str, raw_exit: float, reason: str, ts) -> dict:
        p = self.state["positions"].pop(symbol)
        side, qty = p["side"], p["qty"]
        # A target is a resting limit and fills at its price; anything else slips.
        fill = raw_exit if reason.startswith("TARGET") else _slip(raw_exit, side < 0,
                                                                  self.slippage_bps)
        exit_charges = order_charges(self.cost_model, side < 0, qty, fill)
        gross = side * (fill - p["entry"]) * qty
        net = gross - p["charges"] - exit_charges
        risk = abs(p["entry"] - p["stop"]) * qty
        self.state["cash"] += gross - exit_charges
        self.state["last_price"][symbol] = fill
        self._save()
        rec = {"symbol": symbol, "side": "long" if side > 0 else "short", "shares": qty,
               "entry_time": p["entry_time"], "exit_time": str(ts),
               "entry": round(p["entry"], 4), "exit": round(fill, 4),
               "stop": p["stop"], "target": p["target"], "reason": reason,
               "gross_pnl": round(gross, 2), "fees": round(p["charges"] + exit_charges, 2),
               "net_pnl": round(net, 2), "R": round(net / risk, 3) if risk > 0 else 0.0,
               "fills": "simulated"}
        self.trades_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.trades_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.log(f"  {symbol}: SIM {reason} @ {fill:.2f}  net {self.market.currency}"
                 f"{net:,.2f} ({rec['R']:+.2f}R)")
        return rec

    def on_bar(self, bars: Dict[str, pd.DataFrame], now: pd.Timestamp) -> List[dict]:
        """Walk every completed bar after the entry bar: stop first, as the backtest does."""
        closed = []
        for sym in list(self.state["positions"]):
            df = bars.get(sym)
            if df is None or df.empty:
                continue
            p = self.state["positions"][sym]
            after = pd.Timestamp(p["entry_bar"])
            # `now` is when the newest COMPLETE bar closed: a bar stamped t covers
            # [t, t + bar), so only bars with t + bar <= now are done.
            step = df.index.to_series().diff().min() if len(df) > 1 else pd.Timedelta(0)
            done = df[(df.index > after) & (df.index + step <= now)]
            for ts, bar in done.iterrows():
                self.state["last_price"][sym] = float(bar["close"])
                p["entry_bar"] = str(ts)                  # never re-check a bar
                hi, lo = float(bar["high"]), float(bar["low"])
                if p["side"] > 0:
                    hit_stop, hit_tgt = lo <= p["stop"], hi >= p["target"]
                else:
                    hit_stop, hit_tgt = hi >= p["stop"], lo <= p["target"]
                if hit_stop:
                    closed.append(self._close(sym, p["stop"], "STOP*" if hit_tgt else "STOP", ts))
                    break
                if hit_tgt:
                    closed.append(self._close(sym, p["target"], "TARGET", ts))
                    break
            else:
                self._save()
        return closed

    def flatten_all(self, reason: str = "EOD", now=None) -> None:
        held = list(self.state["positions"])
        if not held:
            return
        self.log(f"SIM flattening {len(held)} position(s): {held}")
        if self.dry_run:
            return
        for sym in held:
            p = self.state["positions"][sym]
            self._close(sym, self.state["last_price"].get(sym, p["entry"]), reason,
                        now if now is not None else self.clock.now())


# ------------------------------------------------------------------ real money

class RealMoneyRefused(SystemExit):
    pass


def require_real_money_consent() -> None:
    if os.environ.get(REAL_MONEY_ENV) != REAL_MONEY_VALUE:
        raise RealMoneyRefused(
            f"--real-money places REAL orders on Angel One (it has no paper sandbox).\n"
            f"Refusing to start. To proceed knowingly, set {REAL_MONEY_ENV}={REAL_MONEY_VALUE} "
            f"in the environment as well.")


class AngelBroker(Broker):
    """
    Real orders on Angel One through SmartAPI. See the module docstring first.

    Entry is a ROBO (bracket) order: a LIMIT entry with a target leg and a
    stop-loss leg attached, so the stop exists at the broker the moment the
    entry fills. If the bracket is rejected, the trade is skipped; there is no
    fallback to an unprotected entry.

    Open question pinned by --robo-units: SmartAPI documents `squareoff` and
    `stoploss` as "Only For ROBO" without units. Angel's bracket order takes
    them as distances in rupees from the entry ("points"), which is the default
    here. Confirm it on your account with one 1-share order before trusting it
    (run.md walks through this).
    """
    name = "angel"
    fills = "broker"

    def __init__(self, state_path: Path, max_order_value: float = 20_000.0,
                 max_orders_per_day: int = 3, entry_band_bps: float = 10.0,
                 fill_timeout_s: float = 30.0, robo_units: str = "points",
                 market: Market = NSE, log: Log = print, dry_run: bool = False):
        require_real_money_consent()
        super().__init__(market, log, dry_run)
        import angel
        self.angel = angel
        self.s = angel.session()
        self.state_path = Path(state_path)
        self.max_order_value = max_order_value
        self.max_orders_per_day = max_orders_per_day
        self.entry_band_bps = entry_band_bps
        self.fill_timeout_s = fill_timeout_s
        if robo_units not in ("points", "price"):
            raise ValueError("robo_units must be 'points' or 'price'")
        self.robo_units = robo_units
        self.state = {"date": "", "orders": 0}
        if self.state_path.exists():
            try:
                self.state.update(json.loads(self.state_path.read_text()))
            except json.JSONDecodeError:
                pass

    # -- small helpers ---------------------------------------------

    def _today_orders(self) -> int:
        today = self.clock.now().date().isoformat()
        if self.state.get("date") != today:
            self.state = {"date": today, "orders": 0}
        return int(self.state["orders"])

    def _count_order(self) -> None:
        self._today_orders()
        self.state["orders"] += 1
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state))

    @staticmethod
    def _guard(params: dict) -> dict:
        """The exchange rules, enforced in code: limit orders only, never IOC."""
        if params.get("ordertype") not in ("LIMIT", "STOPLOSS_LIMIT"):
            raise ValueError(f"refusing {params.get('ordertype')} order: API orders must be limit")
        if params.get("duration", "DAY") != "DAY":
            raise ValueError("refusing non-DAY order: IOC is not allowed for API orders")
        return params

    def _ok(self, resp) -> bool:
        return isinstance(resp, dict) and bool(resp.get("status"))

    def _data(self, resp):
        return resp.get("data") if self._ok(resp) else None

    # -- account ---------------------------------------------------

    def account(self) -> dict:
        d = self._data(self.s.call("rms", lambda c: c.rmsLimit())) or {}

        def f(k):
            try:
                return float(d.get(k) or 0)
            except (TypeError, ValueError):
                return 0.0
        net = f("net")
        return {"equity": net, "cash": f("availablecash"),
                "buying_power": f("availablecash") + f("availableintradaypayin"),
                "multiplier": 1.0}

    def _positions_raw(self) -> List[dict]:
        return self._data(self.s.call("position", lambda c: c.position())) or []

    def positions(self) -> Dict[str, int]:
        out = {}
        for p in self._positions_raw():
            if str(p.get("producttype", "")).upper() not in ("INTRADAY", "BO"):
                continue
            qty = int(float(p.get("netqty") or 0))
            if qty:
                out[str(p.get("tradingsymbol", "")).removesuffix("-EQ")] = qty
        return out

    def _order_book(self) -> List[dict]:
        return self._data(self.s.call("orderbook", lambda c: c.orderBook())) or []

    # -- entry -----------------------------------------------------

    def bracket_params(self, symbol: str, side: int, qty: int, stop: float,
                       target: float, ref_price: float) -> dict:
        """The exact ROBO order this broker would send. Pure, so it can be tested."""
        sc = self.angel.scrip(symbol)
        tick = sc.tick_size
        band = self.entry_band_bps / 10_000
        limit = self.angel.round_to_tick(ref_price * (1 + band if side > 0 else 1 - band), tick)
        if self.robo_units == "points":
            squareoff = self.angel.round_to_tick(abs(target - limit), tick)
            stoploss = self.angel.round_to_tick(abs(limit - stop), tick)
        else:
            squareoff = self.angel.round_to_tick(target, tick)
            stoploss = self.angel.round_to_tick(stop, tick)
        tag = (self.clock.now().strftime("%y%m%d") + symbol[:10]
               + ("L" if side > 0 else "S"))[:19]
        return self._guard({
            "variety": "ROBO", "tradingsymbol": sc.tradingsymbol, "symboltoken": sc.token,
            "transactiontype": "BUY" if side > 0 else "SELL", "exchange": "NSE",
            "ordertype": "LIMIT", "producttype": "BO", "duration": "DAY",
            "price": f"{limit:.2f}", "squareoff": f"{squareoff:.2f}",
            "stoploss": f"{stoploss:.2f}", "trailingStopLoss": "0",
            "quantity": str(int(qty)), "ordertag": tag,
        })

    def submit_bracket(self, symbol, side, qty, stop, target, ref_price,
                       bar_ts=None) -> Optional[dict]:
        value = qty * ref_price
        if value > self.max_order_value:
            self.log(f"  {symbol}: REFUSED, {self.market.currency}{value:,.0f} exceeds "
                     f"--max-order-value {self.market.currency}{self.max_order_value:,.0f}")
            return None
        if self._today_orders() >= self.max_orders_per_day:
            self.log(f"  {symbol}: REFUSED, already {self.max_orders_per_day} orders today "
                     f"(--max-orders-per-day)")
            return None
        params = self.bracket_params(symbol, side, qty, stop, target, ref_price)
        if self.dry_run:
            self.log(f"  DRY RUN would send to Angel One: {json.dumps(params)}")
            return None

        # Never send the same bracket twice: a retry after a timeout is how a bot
        # ends up with two positions. Check the day's book for our tag first.
        for o in self._order_book():
            if o.get("ordertag") == params["ordertag"]:
                self.log(f"  {symbol}: order {params['ordertag']} already exists "
                         f"({o.get('status')}), not resending")
                return None

        self._count_order()
        try:
            resp = self.s.call("order", lambda c: c.placeOrderFullResponse(dict(params)),
                               idempotent=False)
        except Exception as e:
            self.log(f"  {symbol}: ROBO not accepted ({str(e)[:140]}); no position taken")
            return None
        if not self._ok(resp):
            msg = resp.get("message") if isinstance(resp, dict) else resp
            self.log(f"  {symbol}: ROBO rejected ({msg}); no position taken")
            return None
        oid = str(resp["data"].get("orderid"))
        uid = str(resp["data"].get("uniqueorderid"))
        return self._await_fill(symbol, oid, uid, params)

    def _status(self, uid: str) -> dict:
        resp = self.s.call("details", lambda c: c.individual_order_details(uid))
        return (resp or {}).get("data") or {}

    def _await_fill(self, symbol, oid, uid, params) -> Optional[dict]:
        deadline = time.monotonic() + self.fill_timeout_s
        st = {}
        while time.monotonic() < deadline:
            st = self._status(uid)
            status = str(st.get("orderstatus") or st.get("status") or "").lower()
            if status == "complete":
                break
            if status in ("rejected", "cancelled"):
                self.log(f"  {symbol}: ROBO {status}: {st.get('text', '')[:140]}")
                return None
            time.sleep(2.0)
        else:
            # Not filled in time: the setup is stale. Cancel, then trust the
            # broker's word on what (if anything) filled before the cancel.
            self.s.call("order", lambda c: c.cancelOrder(oid, "ROBO"), idempotent=False)
            time.sleep(1.0)
            st = self._status(uid)
        filled = int(float(st.get("filledshares") or 0))
        if filled <= 0:
            self.log(f"  {symbol}: ROBO not filled within {self.fill_timeout_s:.0f}s, cancelled")
            return None
        avg = float(st.get("averageprice") or params["price"])
        return {"id": oid, "unique_id": uid, "symbol": symbol, "qty": filled,
                "side": params["transactiontype"].lower(), "status": "filled",
                "fill": avg, "order": params}

    # -- exit ------------------------------------------------------

    EXIT_TAG = "XIT"

    def _working(self) -> List[dict]:
        live = ("open", "trigger pending", "open pending", "validation pending",
                "modify pending")
        return [o for o in self._order_book() if str(o.get("status", "")).lower() in live]

    def _cancel(self, orders: List[dict]) -> None:
        for o in orders:
            try:
                self.s.call("order", lambda c, o=o: c.cancelOrder(str(o["orderid"]),
                                                                  o.get("variety", "NORMAL")),
                            idempotent=False)
            except Exception as e:
                self.log(f"  cancel {o.get('orderid')} failed: {str(e)[:120]}")

    def _held_raw(self) -> List[dict]:
        return [p for p in self._positions_raw()
                if str(p.get("producttype", "")).upper() in ("INTRADAY", "BO")
                and int(float(p.get("netqty") or 0)) != 0]

    def flatten_all(self, reason: str = "EOD", now=None) -> None:
        """
        Cancel what is working, then work the remaining quantity out with limit orders.

        Cancelling an open ROBO leg is how Angel exits a bracket. Whatever is still
        held afterwards is closed with LIMIT orders priced through the last price.
        Each round cancels our previous exit orders and re-reads the position
        BEFORE placing new ones: re-pricing without cancelling first can fill twice
        and turn a long into a short. If this has not finished by 15:12, Angel's
        own 15:15 square-off closes it, at a fee, and the log says so loudly.
        """
        held, working = self._held_raw(), self._working()
        if not held and not working:
            return
        self.log(f"flattening: {len(held)} position(s) "
                 f"{[p.get('tradingsymbol') for p in held]}, {len(working)} working order(s)")
        if self.dry_run:
            return
        self._cancel(working)

        deadline = (self.market.open.hour * 60 + self.market.open.minute
                    + self.market.rth_minutes - 18)          # 15:12 on NSE
        while True:
            time.sleep(3.0)
            ours = [o for o in self._working()
                    if str(o.get("ordertag", "")).startswith(self.EXIT_TAG)]
            if ours:
                self._cancel(ours)
                time.sleep(2.0)
            held = self._held_raw()
            if not held:
                self.log("flat")
                return
            t = self.clock.now()
            if t.hour * 60 + t.minute >= deadline:
                self.log(f"!!! STILL HOLDING {[p.get('tradingsymbol') for p in held]} at "
                         f"{t:%H:%M}. Angel's 15:15 auto square-off will close it "
                         f"(Rs 50 + GST per position). Check the account now.")
                return
            for p in held:
                self._exit_with_limit(p)
            time.sleep(5.0)

    def _exit_with_limit(self, p: dict) -> None:
        qty = int(float(p.get("netqty") or 0))
        sym = str(p.get("tradingsymbol", "")).removesuffix("-EQ")
        sc = self.angel.scrip(sym)
        ltp_resp = self.s.call("ltp", lambda c: c.ltpData("NSE", sc.tradingsymbol, sc.token))
        ltp = float(((ltp_resp or {}).get("data") or {}).get("ltp") or 0)
        if ltp <= 0:
            return
        sell = qty > 0
        band = self.entry_band_bps / 10_000 * 3        # cross the spread decisively
        price = self.angel.round_to_tick(ltp * (1 - band if sell else 1 + band), sc.tick_size)
        params = self._guard({
            "variety": "NORMAL", "tradingsymbol": sc.tradingsymbol, "symboltoken": sc.token,
            "transactiontype": "SELL" if sell else "BUY", "exchange": "NSE",
            "ordertype": "LIMIT", "producttype": str(p.get("producttype", "INTRADAY")).upper(),
            "duration": "DAY", "price": f"{price:.2f}", "quantity": str(abs(qty)),
            "ordertag": (self.EXIT_TAG + self.clock.now().strftime("%H%M%S") + sym[:10])[:19],
        })
        try:
            self.s.call("order", lambda c: c.placeOrderFullResponse(dict(params)),
                        idempotent=False)
        except Exception as e:
            self.log(f"  exit {sym} failed: {str(e)[:120]}")
