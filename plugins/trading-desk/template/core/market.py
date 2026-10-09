"""
market.py
=========
Everything that differs between exchanges, in one place: the session clock, the
currency, and what a trade costs.

The engine, the strategies and the live runner read these instead of hard-coding
09:30 ET and the SEC fee, so moving the harness to another exchange is a matter
of picking a Market and a cost model, not editing the logic you measured.

Costs are charged per order side, because that is how every component of them is
levied: a round trip is one buy order plus one sell order.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True)
class Market:
    name: str
    tz: str
    tz_label: str                 # what the context text the model reads calls it
    open: dt.time
    close: dt.time
    flat_at_minute: int           # minutes from the open at which intraday positions are closed
    currency: str

    @property
    def rth_minutes(self) -> int:
        o = self.open.hour * 60 + self.open.minute
        c = self.close.hour * 60 + self.close.minute
        return c - o


US = Market(name="US", tz="America/New_York", tz_label="ET",
            open=dt.time(9, 30), close=dt.time(16, 0),
            flat_at_minute=385,           # 15:55
            currency="$")

# NSE cash equities. Angel One squares off open intraday positions itself at
# 15:15 and charges for it; 15:05 leaves ten minutes to work limit exits, which
# matters because API market orders are not allowed (see core/brokers.py).
NSE = Market(name="NSE", tz="Asia/Kolkata", tz_label="IST",
             open=dt.time(9, 15), close=dt.time(15, 30),
             flat_at_minute=350,          # 15:05
             currency="₹")

MARKETS = {"US": US, "NSE": NSE}


# ------------------------------------------------------------------ US costs

# Regulatory costs that apply to SALES only (2025 rates). Commissions are zero at
# most US retail brokers, which is exactly why people forget the rest exists.
SEC_FEE_RATE = 27.80 / 1_000_000      # per dollar of principal sold
TAF_PER_SHARE = 0.000166              # FINRA trading activity fee
TAF_CAP = 8.30


def us_equity_charges(is_buy: bool, shares: int, price: float) -> float:
    if is_buy:
        return 0.0
    return shares * price * SEC_FEE_RATE + min(shares * TAF_PER_SHARE, TAF_CAP)


# --------------------------------------------------------------- India costs
#
# NSE cash equity statutory charges plus a broker's brokerage, rates as
# published in October 2024 and current when this was written (October 2026). Statutory rates change with budgets and
# exchange circulars, and broker schedules change without notice: check them
# against the broker's own charges page before trusting a net figure.

BROKERAGE_FLAT = 20.0                 # Angel: Rs per executed order...
BROKERAGE_PCT = 0.001                 # ...or 0.1% of turnover, whichever is lower

# Brokerage per executed order, per broker: (flat rupees, cap as a fraction of
# turnover or None). INDmoney's INDstocks FAQ lists a flat Rs 10 per order;
# confirm it on the contract note before trusting a net figure.
BROKERS = {
    "angel": (BROKERAGE_FLAT, BROKERAGE_PCT),
    "indmoney": (10.0, None),
}
# Angel lists equity delivery as free. Charging delivery like intraday instead
# overstates its cost by at most Rs 20 an order, which is the safe direction.
DELIVERY_BROKERAGE = True

STT_INTRADAY_SELL = 0.00025           # 0.025%, sell side only
STT_DELIVERY = 0.001                  # 0.1%, both sides
NSE_TXN = 0.0000297                   # 0.00297% of turnover, both sides
SEBI_FEE = 10 / 10_000_000            # Rs 10 per crore, both sides
STAMP_INTRADAY_BUY = 0.00003          # 0.003%, buy side only
STAMP_DELIVERY_BUY = 0.00015          # 0.015%, buy side only
GST = 0.18                            # on brokerage + exchange + SEBI charges
DP_CHARGE_PER_SELL = 20.0             # delivery sells only, per scrip per day (+GST)


def _brokerage(turnover: float, broker: str = "angel") -> float:
    flat, pct = BROKERS[broker]
    return flat if pct is None else min(flat, turnover * pct)


def india_charges_breakdown(is_buy: bool, shares: int, price: float,
                            delivery: bool = False, broker: str = "angel") -> dict:
    """Every line of a contract note for one executed order, in rupees."""
    turnover = shares * price
    brokerage = (_brokerage(turnover, broker)
                 if (not delivery or DELIVERY_BROKERAGE) else 0.0)
    if delivery:
        stt = turnover * STT_DELIVERY
        stamp = turnover * STAMP_DELIVERY_BUY if is_buy else 0.0
        dp = 0.0 if is_buy else DP_CHARGE_PER_SELL
    else:
        stt = 0.0 if is_buy else turnover * STT_INTRADAY_SELL
        stamp = turnover * STAMP_INTRADAY_BUY if is_buy else 0.0
        dp = 0.0
    txn = turnover * NSE_TXN
    sebi = turnover * SEBI_FEE
    gst = (brokerage + txn + sebi + dp) * GST
    parts = {"brokerage": brokerage, "stt": stt, "exchange": txn, "sebi": sebi,
             "stamp": stamp, "dp": dp, "gst": gst}
    parts["total"] = sum(parts.values())
    return parts


def india_intraday_charges(is_buy: bool, shares: int, price: float) -> float:
    return india_charges_breakdown(is_buy, shares, price, delivery=False)["total"]


def india_delivery_charges(is_buy: bool, shares: int, price: float) -> float:
    return india_charges_breakdown(is_buy, shares, price, delivery=True)["total"]


def india_intraday_charges_ind(is_buy: bool, shares: int, price: float) -> float:
    return india_charges_breakdown(is_buy, shares, price, delivery=False,
                                   broker="indmoney")["total"]


def india_delivery_charges_ind(is_buy: bool, shares: int, price: float) -> float:
    return india_charges_breakdown(is_buy, shares, price, delivery=True,
                                   broker="indmoney")["total"]


COST_MODELS = {
    "us_equity": us_equity_charges,
    "india_intraday": india_intraday_charges,          # Angel One brokerage
    "india_delivery": india_delivery_charges,
    "india_intraday_ind": india_intraday_charges_ind,  # INDmoney brokerage
    "india_delivery_ind": india_delivery_charges_ind,
    "none": lambda is_buy, shares, price: 0.0,
}


def intraday_cost_model(broker: str = "angel") -> str:
    """The intraday cost model for a broker name: angel | indmoney."""
    return {"angel": "india_intraday", "indmoney": "india_intraday_ind"}[broker]


def order_charges(model: str, is_buy: bool, shares: int, price: float) -> float:
    """Charges for one executed order under a named cost model."""
    try:
        fn = COST_MODELS[model]
    except KeyError:
        raise ValueError(f"unknown cost model {model!r}; one of {sorted(COST_MODELS)}")
    return fn(is_buy, shares, price)
