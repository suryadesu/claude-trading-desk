#!/usr/bin/env python3
"""
live.py
=======
Runs the validated ORB strategy live on NSE, with Angel One (default) or
INDmoney (--broker indmoney) for data and, only with --real-money, orders.

It imports `ORBStrategy` and `ModelDecider` directly from the backtest. Nothing is
reimplemented here, because a second copy of the logic drifts from the one you
measured and then your backtest no longer describes what is running.

  python3 live.py --once --dry-run      # one cycle, no fills, safe any time
  python3 live.py --replay 2026-10-06   # walk a past session through this code, offline fills
  python3 live.py                       # SIMULATED fills on live prices (the default)
  python3 live.py --broker indmoney     # the same, with INDmoney data and charges
  python3 live.py --real-money          # REAL orders on the broker. Read run.md first.

Simulated by default. Neither Angel One nor INDmoney has a paper-trading
sandbox, so the only safe default is to read live prices and book fills locally
with the backtest's own rules (core/brokers.py SimBroker). --real-money switches
to AngelBroker (needs ANGEL_REAL_MONEY=I_ACCEPT_REAL_LOSSES) or, with --broker
indmoney, IndBroker (needs INDSTOCKS_REAL_MONEY=I_ACCEPT_REAL_LOSSES). Both
enforce their own caps on order value and orders per day.

Safety, all on by default:
  - simulated fills unless --real-money AND the environment variable agree
  - a kill switch file (out/STOP) halts new entries without killing the process
  - a daily loss limit flattens and stands down
  - real entries carry their stop at the broker (Angel ROBO bracket, INDmoney
    smart order with stop and target legs)
  - refuses any order whose stop is on the wrong side of the fill
  - flattens at 15:05 IST, before the broker's own intraday square-off

Every decision and fill is logged in the backtest's schema so live results can be
reconciled against the backtest's distribution. Realised slippage versus the 2 bp
assumed is the number most likely to invalidate the whole thing, and it is
measurable from the first fill.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from brokers import AngelBroker, Broker, IndBroker, SimBroker      # noqa: E402
from contracts import Action, ENTRIES                                # noqa: E402
from data import fetch_universe                                      # noqa: E402
from decision import ModelDecider, RuleDecider                       # noqa: E402
from market import NSE, intraday_cost_model                          # noqa: E402
from strategy import ORBConfig, ORBStrategy                          # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
KILL_SWITCH = OUT / "STOP"
STATE_PATH = OUT / "live_state.json"
MKT = NSE
TZ = MKT.tz
CUR = MKT.currency


def load_env(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def log(msg: str) -> None:
    stamp = dt.datetime.now(dt.timezone.utc).astimezone().strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def jlog(path: Path, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


# ----------------------------------------------------------------- state

class State:
    """
    Survives restarts so a crash mid-session cannot double-enter. Keyed by
    session date, and reset automatically when the date rolls over.
    """

    def __init__(self, path: Path = STATE_PATH):
        self.path = path
        self.data = {"date": "", "entered": [], "start_equity": None, "halted": False}
        if path.exists():
            try:
                self.data.update(json.loads(path.read_text()))
            except json.JSONDecodeError:
                pass

    def roll(self, today: str, equity: float) -> None:
        if self.data.get("date") != today:
            self.data = {"date": today, "entered": [], "start_equity": equity, "halted": False}
            self.save()

    def has_entered(self, symbol: str) -> bool:
        return symbol in self.data["entered"]

    def mark(self, symbol: str) -> None:
        self.data["entered"].append(symbol)
        self.save()

    def halt(self) -> None:
        self.data["halted"] = True
        self.save()

    @property
    def halted(self) -> bool:
        return bool(self.data.get("halted"))

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2))


# ----------------------------------------------------------------- runner

class LiveRunner:
    def __init__(self, args):
        self.args = args
        self.cfg = ORBConfig(or_minutes=args.or_minutes, exec_minutes=args.exec_minutes,
                             rr=args.rr, last_entry_min=args.last_entry_min)
        self.strat = ORBStrategy(self.cfg)
        self.broker = make_broker(args)
        self.state = State()
        self.decisions = OUT / "live_decisions.jsonl"
        self.fills = OUT / "live_fills.jsonl"

        if args.decider in ("laya", "jev"):
            self.decider = ModelDecider(
                prompt=self.strat.model_prompt(), name=args.decider,
                threshold=args.model_threshold, temperature=args.model_temperature,
                log_path=OUT / f"live_{args.decider}_stream.jsonl",
            )
        else:
            self.decider = RuleDecider()

    # -- data ------------------------------------------------------

    def bars(self, as_of: Optional[dt.date] = None) -> Dict[str, pd.DataFrame]:
        """
        Always a fresh pull. The disk cache is keyed by date range and would
        happily hand back a stale frame mid-session, which is exactly how a live
        bot ends up trading yesterday's setup.

        SmartAPI's candles are real time, and the newest one may still be
        forming; cycle() only ever looks at bars that have closed.
        """
        ref = as_of or dt.date.today()
        start = (ref - dt.timedelta(days=self.args.warmup_days)).isoformat()
        return fetch_universe(symbols=self.args.symbols, start=start,
                              end=(ref + dt.timedelta(days=1)).isoformat(),
                              minutes=self.args.exec_minutes,
                              use_cache=bool(as_of), verbose=False)

    # -- sizing ----------------------------------------------------

    def size(self, equity: float, price: float, stop: float) -> int:
        risk_per_share = abs(price - stop)
        if risk_per_share <= 0:
            return 0
        by_risk = (equity * self.args.risk_pct) / risk_per_share
        by_notional = (equity * self.args.max_notional_pct) / price
        return int(max(0, min(by_risk, by_notional)))

    # -- one cycle -------------------------------------------------

    def cycle(self, now_override: Optional[pd.Timestamp] = None,
              bars_override: Optional[Dict[str, pd.DataFrame]] = None) -> None:
        now_et = now_override or pd.Timestamp.now(tz=TZ)
        today = now_et.strftime("%Y-%m-%d")
        bars = bars_override if bars_override is not None else self.bars()
        # Let the simulator see every bar that closed since the last cycle, so a
        # stop or target that traded in between is booked before anything else.
        for rec in self.broker.on_bar(bars, now_et):
            jlog(self.fills, {"kind": "exit", **rec})
        acct = self.broker.account()
        self.state.roll(today, acct["equity"])

        # --- guardrails, checked every cycle before anything else -----
        if KILL_SWITCH.exists():
            log("KILL SWITCH present (out/STOP). No new entries.")
            return
        if self.state.halted:
            log("halted for the day by the loss limit. No new entries.")
            return

        start_eq = self.state.data.get("start_equity") or acct["equity"]
        dd = (acct["equity"] / start_eq - 1) * 100
        if dd <= -self.args.daily_loss_pct:
            log(f"DAILY LOSS LIMIT hit ({dd:+.2f}% vs limit -{self.args.daily_loss_pct}%). "
                f"Flattening and standing down.")
            self.broker.flatten_all("LOSS_LIMIT", now_et)
            self.state.halt()
            return

        held = self.broker.positions()
        mfo = (now_et - now_et.normalize()
               - pd.Timedelta(hours=MKT.open.hour, minutes=MKT.open.minute)).total_seconds() / 60

        # --- flatten before the close ---------------------------------
        if mfo >= self.args.flat_at_minute:
            if held:
                self.broker.flatten_all("EOD", now_et)
            return

        if now_override is None:
            log(f"cycle: equity {CUR}{acct['equity']:,.2f} ({dd:+.2f}% today)  "
                f"bp {CUR}{acct['buying_power']:,.0f}  positions {len(held)}  "
                f"{mfo:.0f} min from open  [{self.broker.name}]")

        for sym, df in bars.items():
            if sym in held or self.state.has_entered(sym):
                continue
            if len(held) >= self.args.max_positions:
                log(f"  {sym}: at max positions ({self.args.max_positions}), skipping")
                continue

            plan = self.strat.prepare(df)
            today_bars = plan[plan.index.normalize() == pd.Timestamp(today, tz=TZ)]
            # Only bars that have CLOSED. A bar stamped t covers [t, t + bar), and
            # SmartAPI hands back the one still forming; trading it is lookahead.
            today_bars = today_bars[today_bars.index
                                    + pd.Timedelta(minutes=self.args.exec_minutes) <= now_et]
            if today_bars.empty:
                continue
            last = today_bars.iloc[-1]
            age_min = (now_et - today_bars.index[-1]).total_seconds() / 60
            if age_min > self.args.exec_minutes * 3:
                log(f"  {sym}: latest bar is {age_min:.0f} min old, data feed is lagging. Skipping.")
                continue
            if not str(last.get("signal") or ""):
                continue

            snap = self.strat.snapshot(sym, today_bars.index[-1], last)
            dec = self.decider.decide(snap)
            rec = {"ts": str(today_bars.index[-1]), "symbol": sym, "signal": str(last["signal"]),
                   "price": float(last["close"]), "stop": float(last["stop"]),
                   "target": float(last["target"]), "action": dec.action.value,
                   "prob": dec.probabilities.get(dec.action.value, dec.confidence),
                   "probabilities": dec.probabilities, "note": dec.note, "aux": dec.aux}

            if dec.action not in ENTRIES:
                log(f"  {sym}: {last['signal']} breakout -> STAND ASIDE "
                    f"({dec.note or 'below threshold'})")
                jlog(self.decisions, {**rec, "submitted": False})
                continue

            side = 1 if str(last["signal"]) == "long" else -1
            price, stop, target = float(last["close"]), float(last["stop"]), float(last["target"])

            # A stop on the wrong side of the entry means something is wrong upstream.
            if (side > 0 and stop >= price) or (side < 0 and stop <= price):
                log(f"  {sym}: REFUSED, stop {stop:.2f} is on the wrong side of {price:.2f}")
                jlog(self.decisions, {**rec, "submitted": False, "refused": "bad_stop"})
                continue

            qty = self.size(acct["equity"], price, stop)
            if qty < 1:
                log(f"  {sym}: size rounds to 0 shares, skipping")
                jlog(self.decisions, {**rec, "submitted": False, "refused": "zero_size"})
                continue
            if qty * price > acct["buying_power"]:
                log(f"  {sym}: {CUR}{qty * price:,.0f} exceeds buying power, skipping")
                jlog(self.decisions, {**rec, "submitted": False, "refused": "buying_power"})
                continue

            log(f"  {sym}: {last['signal'].upper()} approved p={rec['prob']:.2f} -> "
                f"{qty} shares @ ~{price:.2f}, stop {stop:.2f}, target {target:.2f} "
                f"(risk {CUR}{abs(price - stop) * qty:.0f})")
            try:
                order = self.broker.submit_bracket(sym, side, qty, stop, target, price,
                                                   bar_ts=today_bars.index[-1])
            except Exception as e:
                log(f"  {sym}: ORDER REJECTED {type(e).__name__}: {str(e)[:140]}")
                jlog(self.decisions, {**rec, "submitted": False, "error": str(e)[:200]})
                continue

            if order is None:
                jlog(self.decisions, {**rec, "submitted": False, "refused": "broker"})
                continue
            self.state.mark(sym)
            jlog(self.decisions, {**rec, "submitted": True, "qty": qty, "order": order})
            # signal_close vs actual fill is how you measure real slippage
            jlog(self.fills, {"ts": str(today_bars.index[-1]), "symbol": sym,
                              "side": "long" if side > 0 else "short", "qty": qty,
                              "signal_close": price, "stop": stop, "target": target,
                              "order": order})

    # -- loop ------------------------------------------------------

    def replay(self, day: str) -> None:
        """
        Walk one past session through the LIVE code path, bar by bar, submitting
        nothing. This is how you check that live.py and the backtest agree: any
        difference here is a bug in one of them, not a market condition.
        """
        # Replays never touch a broker: a fresh simulator, its own state files.
        for f in (OUT / "replay_sim_state.json", OUT / "replay_sim_trades.jsonl",
                  OUT / "replay_state.json"):
            f.unlink(missing_ok=True)
        self.broker = SimBroker(OUT / "replay_sim_state.json", OUT / "replay_sim_trades.jsonl",
                                equity=self.args.sim_equity,
                                slippage_bps=self.args.slippage_bps,
                                cost_model=intraday_cost_model(self.args.broker), log=log)
        self.state = State(OUT / "replay_state.json")
        ref = dt.date.fromisoformat(day)
        bars = self.bars(as_of=ref)
        day_ts = pd.Timestamp(day, tz=TZ)
        stamps = sorted({t for df in bars.values()
                         for t in df.index if t.normalize() == day_ts})
        if not stamps:
            log(f"no bars for {day} (market holiday or weekend?)")
            return
        log(f"replaying {day}: {len(stamps)} bars, {len(bars)} symbols, simulated fills")
        for ts in stamps:
            # the clock stands just after this bar closed
            self.cycle(now_override=ts + pd.Timedelta(minutes=self.args.exec_minutes),
                       bars_override=bars)
        self.broker.flatten_all("EOD", stamps[-1] + pd.Timedelta(minutes=self.args.exec_minutes))
        acct = self.broker.account()
        log(f"replay complete: equity {CUR}{acct['equity']:,.2f}; "
            f"trades in {OUT / 'replay_sim_trades.jsonl'}")

    def run(self) -> None:
        if self.args.replay:
            self.replay(self.args.replay)
            return
        if self.args.once:
            self.cycle()
            return
        log(f"live runner started. decider={self.args.decider} "
            f"dry_run={self.args.dry_run} symbols={' '.join(self.args.symbols)}")
        log(f"kill switch: touch {KILL_SWITCH}")
        while True:
            try:
                if not self.broker.is_open():
                    nxt = self.broker.next_open()
                    wait = max(30.0, (nxt - pd.Timestamp.now(tz=TZ)).total_seconds())
                    log(f"market closed. next open {nxt:%Y-%m-%d %H:%M} {MKT.tz_label}. "
                        f"sleeping {min(wait, 3600)/60:.0f} min")
                    time.sleep(min(wait, 3600))
                    continue
                self.cycle()
            except KeyboardInterrupt:
                log("interrupted. open positions are left alone (real brackets stay at "
                    "the broker; simulated ones in out/sim_state.json). Use --flatten to close.")
                return
            except Exception as e:
                log(f"cycle error {type(e).__name__}: {str(e)[:160]}")
            # wake a few seconds after the next bar boundary so the bar is complete
            now = dt.datetime.now(dt.timezone.utc)
            m = self.args.exec_minutes
            nxt = (now.replace(second=0, microsecond=0)
                   + dt.timedelta(minutes=m - (now.minute % m)))
            time.sleep(max(5.0, (nxt - now).total_seconds() + self.args.bar_buffer_sec))


def make_broker(args) -> Broker:
    if args.real_money:
        if args.broker == "indmoney":
            b = IndBroker(OUT / "indmoney_state.json", max_order_value=args.max_order_value,
                          max_orders_per_day=args.max_orders_per_day,
                          entry_band_bps=args.entry_band_bps,
                          sl_limit_band_bps=args.sl_limit_band_bps,
                          log=log, dry_run=args.dry_run)
            label, ip_where = "INDmoney", "your INDstocks API settings"
        else:
            b = AngelBroker(OUT / "angel_state.json", max_order_value=args.max_order_value,
                            max_orders_per_day=args.max_orders_per_day,
                            entry_band_bps=args.entry_band_bps, robo_units=args.robo_units,
                            log=log, dry_run=args.dry_run)
            label, ip_where = "Angel One", "your SmartAPI app"
        acct = b.account()
        log("=" * 72)
        log(f"REAL MONEY MODE: orders go to {label} and are NOT simulated.")
        log(f"  account equity {CUR}{acct['equity']:,.2f}, available {CUR}{acct['cash']:,.2f}")
        log(f"  caps: {CUR}{args.max_order_value:,.0f} per order, "
            f"{args.max_orders_per_day} orders per day, daily loss {args.daily_loss_pct}%")
        log(f"  orders are only accepted from the static IP registered in {ip_where}")
        log(f"  dry run: {args.dry_run}.  Ctrl+C now to abort; starting in 10 s.")
        log("=" * 72)
        time.sleep(10)
        return b
    # Angel keeps the original file names; INDmoney's simulator has its own books.
    sfx = "" if args.broker == "angel" else f"_{args.broker}"
    return SimBroker(OUT / f"sim_state{sfx}.json", OUT / f"sim_trades{sfx}.jsonl",
                     equity=args.sim_equity, slippage_bps=args.slippage_bps,
                     cost_model=intraday_cost_model(args.broker),
                     log=log, dry_run=args.dry_run)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the ORB strategy on NSE: simulated fills by default, "
                    "real Angel One or INDmoney orders only with --real-money.")
    p.add_argument("--broker", default=os.environ.get("BROKER") or "angel",
                   choices=["angel", "indmoney"],
                   help="where bars come from and whose brokerage is charged: Angel One "
                        "SmartAPI or INDmoney INDstocks (default: BROKER in .env, else angel)")
    p.add_argument("--symbols", nargs="+",
                   default=["RELIANCE", "HDFCBANK", "ICICIBANK", "INFY",
                            "TCS", "SBIN", "AXISBANK", "BHARTIARTL"])
    p.add_argument("--decider", default="laya", choices=["laya", "jev", "rules"],
                   help="laya: the local laya-serve sidecar at MODEL_URL. jev: the same "
                        "client pointed at hosted Jev (JEV_URL/JEV_MODEL/TYPESAFE_API_KEY)")
    p.add_argument("--model-threshold", "--jev-threshold", dest="model_threshold",
                   type=float, default=0.30,
                   help="a placeholder: refit it with core/calibrate.py on in-sample "
                        "NSE decisions before trusting it")
    p.add_argument("--model-temperature", type=float, default=1.0,
                   help="temperature fitted by core/calibrate.py; 1.0 = raw model output")
    p.add_argument("--or-minutes", type=int, default=15)
    p.add_argument("--exec-minutes", type=int, default=5)
    p.add_argument("--rr", type=float, default=2.0)
    p.add_argument("--last-entry-min", type=int, default=135, help="135 = 11:30 IST")
    p.add_argument("--flat-at-minute", type=int, default=MKT.flat_at_minute,
                   help=f"{MKT.flat_at_minute} = 15:05 IST, before the broker's square-off")
    p.add_argument("--risk-pct", type=float, default=0.005)
    p.add_argument("--max-notional-pct", type=float, default=1.0)
    p.add_argument("--max-positions", type=int, default=3)
    p.add_argument("--daily-loss-pct", type=float, default=2.0)
    p.add_argument("--warmup-days", type=int, default=60)
    p.add_argument("--bar-buffer-sec", type=float, default=20.0)
    p.add_argument("--once", action="store_true", help="one cycle then exit")
    p.add_argument("--replay", default=None, metavar="YYYY-MM-DD",
                   help="walk a past session through the live code path, simulated fills")
    p.add_argument("--dry-run", action="store_true", help="log intended orders, fill nothing")
    p.add_argument("--flatten", action="store_true", help="close everything and exit")

    sim = p.add_argument_group("simulated fills (default)")
    sim.add_argument("--sim-equity", type=float, default=100_000.0,
                     help="starting capital in rupees for the simulator")
    sim.add_argument("--slippage-bps", type=float, default=2.0)

    real = p.add_argument_group("REAL MONEY on Angel One or INDmoney")
    real.add_argument("--real-money", action="store_true",
                      help="place real orders. Also needs ANGEL_REAL_MONEY (or, with --broker "
                           "indmoney, INDSTOCKS_REAL_MONEY) = I_ACCEPT_REAL_LOSSES")
    real.add_argument("--max-order-value", type=float, default=20_000.0,
                      help="refuse any single order above this many rupees")
    real.add_argument("--max-orders-per-day", type=int, default=3)
    real.add_argument("--entry-band-bps", type=float, default=10.0,
                      help="how far through the signal close the entry LIMIT is priced")
    real.add_argument("--robo-units", default="points", choices=["points", "price"],
                      help="Angel only: how ROBO squareoff/stoploss are sent: rupees from "
                           "entry (points) or absolute prices. Confirm on your account first.")
    real.add_argument("--sl-limit-band-bps", type=float, default=30.0,
                      help="INDmoney only: how far beyond the stop trigger the stop-loss "
                           "leg's limit price sits")
    args = p.parse_args()
    os.environ["BROKER"] = args.broker          # core/data.py reads it
    return args


def main() -> None:
    load_env()
    args = parse_args()
    if args.flatten:
        make_broker(args).flatten_all("FLATTEN")
        return
    LiveRunner(args).run()


if __name__ == "__main__":
    main()
