"""
data.py
=======
NSE bars from a broker's historical endpoint, cached to disk. Two sources:

  angel      Angel One SmartAPI (the default)             core/angel.py
  indmoney   INDmoney's INDstocks API                     core/indstocks.py

Pick one with BROKER=indmoney in the environment, or --broker on the runners
(which set it). The bars are the same NSE prints either way; each source has its
own cache files so the two never mix.

Why a broker API and not yfinance here: yfinance caps 5-minute history at about
60 days, roughly twelve trading weeks, which is not enough to survive one bad
regime, and its NSE quotes run about 15 minutes late, which is the whole life of
a 5-minute breakout. SmartAPI is free with an Angel One account, real time, and
serves years of intraday candles, up to 100 days of 5-minute bars per request.
INDstocks is also free, but serves 7 days of 5-minute bars per request, so a
first download takes about fifteen times as many calls.

Everything is returned in Asia/Kolkata time and filtered to the regular session
(09:15-15:30) by default. The pre-open auction (09:00-09:08) prints one
discovered price, not a market, and would flatter any opening-range strategy.

Credentials: see core/angel.py and core/indstocks.py.
"""

from __future__ import annotations

import datetime as dt
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from angel import AngelError, scrip, session
from indstocks import IndError
from market import NSE

SOURCES = ("angel", "indmoney")
API_ERRORS = (AngelError, IndError)

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

TZ = NSE.tz
SESSION_OPEN = NSE.open
SESSION_CLOSE = NSE.close
COLUMNS = ["open", "high", "low", "close", "volume"]

# Liquid NIFTY 50 names with tight spreads. Intraday strategies pay the spread
# thousands of times, so this list is deliberately boring.
DEFAULT_UNIVERSE = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "SBIN", "AXISBANK", "BHARTIARTL",
    "KOTAKBANK", "LT", "ITC", "HINDUNILVR", "BAJFINANCE", "MARUTI", "SUNPHARMA", "NTPC",
]

# SmartAPI interval names, and the most days one request may span for each.
INTERVALS = {1: ("ONE_MINUTE", 30), 3: ("THREE_MINUTE", 60), 5: ("FIVE_MINUTE", 100),
             10: ("TEN_MINUTE", 100), 15: ("FIFTEEN_MINUTE", 200),
             30: ("THIRTY_MINUTE", 200), 60: ("ONE_HOUR", 400), 0: ("ONE_DAY", 2000)}


def source() -> str:
    """Which broker supplies bars: BROKER in the environment, Angel One by default."""
    s = (os.environ.get("BROKER") or "angel").strip().lower()
    if s not in SOURCES:
        raise ValueError(f"BROKER={s!r}: use one of {SOURCES}")
    return s


# INDstocks interval names. Intraday requests span at most 7 days, hourly 15,
# daily a year; a wider window is silently cut short, so stay under the cap.
IND_INTERVALS = {1: ("1minute", 6), 2: ("2minute", 6), 3: ("3minute", 6),
                 5: ("5minute", 6), 10: ("10minute", 6), 15: ("15minute", 6),
                 30: ("30minute", 6), 60: ("60minute", 14), 0: ("1day", 360)}


def _interval(minutes: int):
    try:
        return INTERVALS[minutes]
    except KeyError:
        raise ValueError(f"SmartAPI has no {minutes}-minute candles; "
                         f"use one of {sorted(INTERVALS)} (0 = daily)") from None


def _cache_path(symbol: str, minutes: int, start: str, end: str) -> Path:
    tf = "1D" if minutes == 0 else f"{minutes}T"
    prefix = "IND" if source() == "indmoney" else "NSE"     # Angel keeps its old names
    return CACHE_DIR / f"{prefix}_{symbol}_{tf}_{start}_{end}.pkl"


def _chunks(start: dt.date, end: dt.date, max_days: int, margin: int = 10):
    """[start, end) split into windows a single request can carry, a little under the cap."""
    step = dt.timedelta(days=max(1, max_days - margin))
    a = start
    while a < end:
        b = min(a + step, end)
        yield a, b
        a = b


def _frame(rows: list) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(rows, columns=["ts"] + COLUMNS)
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("ts"), utc=True)).tz_convert(TZ)
    df.index.name = None
    return df.astype(float)


def _to_session(df: pd.DataFrame, minutes: int, rth_only: bool) -> pd.DataFrame:
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if minutes > 0 and rth_only and not df.empty:
        t = df.index.time
        df = df[(t >= SESSION_OPEN) & (t < SESSION_CLOSE)]
    return df


def fetch_bars(
    symbol: str,
    start: str,
    end: str,
    minutes: int = 5,
    feed: Optional[str] = None,       # accepted for old callers; SmartAPI has one feed
    rth_only: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    One NSE symbol, [start, end), as a DataFrame indexed by IST timestamp with
    columns open, high, low, close, volume. `end` is exclusive, like a slice.
    """
    path = _cache_path(symbol, minutes, start, end)
    if use_cache and path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)

    fetch = _fetch_indstocks if source() == "indmoney" else _fetch_angel
    df = _to_session(fetch(symbol, start, end, minutes), minutes, rth_only)

    if use_cache:
        with open(path, "wb") as f:
            pickle.dump(df, f)
    return df


def _fetch_angel(symbol: str, start: str, end: str, minutes: int) -> pd.DataFrame:
    name, max_days = _interval(minutes)
    sc = scrip(symbol)
    s = session()
    frames = []
    for a, b in _chunks(dt.date.fromisoformat(start), dt.date.fromisoformat(end), max_days):
        params = {"exchange": "NSE", "symboltoken": sc.token, "interval": name,
                  "fromdate": f"{a.isoformat()} 09:00",
                  "todate": f"{(b - dt.timedelta(days=1)).isoformat()} 15:30"}
        resp = s.call("candles", lambda c, p=params: c.getCandleData(p))
        if not isinstance(resp, dict) or resp.get("status") is False:
            msg = resp.get("message") if isinstance(resp, dict) else resp
            raise AngelError(f"{symbol} {a}..{b}: {msg}")
        frames.append(_frame(resp.get("data") or []))   # holidays come back empty
    return pd.concat(frames) if frames else pd.DataFrame(columns=COLUMNS)


def _ind_frame(candles: list) -> pd.DataFrame:
    """INDstocks candles: {ts (open time, epoch SECONDS), o, h, l, c, v}."""
    if not candles:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(candles)
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("ts").astype("int64"), unit="s",
                                               utc=True)).tz_convert(TZ)
    df.index.name = None
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    return df[COLUMNS].astype(float)


def _fetch_indstocks(symbol: str, start: str, end: str, minutes: int) -> pd.DataFrame:
    import indstocks
    try:
        name, max_days = IND_INTERVALS[minutes]
    except KeyError:
        raise ValueError(f"INDstocks has no {minutes}-minute candles; "
                         f"use one of {sorted(IND_INTERVALS)} (0 = daily)") from None
    code = indstocks.scrip_code(indstocks.scrip(symbol))
    s = indstocks.session()
    frames = []
    for a, b in _chunks(dt.date.fromisoformat(start), dt.date.fromisoformat(end),
                        max_days, margin=0):
        t0 = pd.Timestamp(a, tz=TZ)
        t1 = pd.Timestamp(b, tz=TZ)
        resp = s.call("data", "GET", f"/market/historical/{name}",
                      params={"scrip-codes": code,
                              "start_time": int(t0.timestamp() * 1000),
                              "end_time": int(t1.timestamp() * 1000)})
        if (not isinstance(resp, dict) or resp.get("success") is False
                or resp.get("status") == "error"):
            msg = resp.get("message") if isinstance(resp, dict) else resp
            raise IndError(f"{symbol} {a}..{b}: {msg}")
        candles = ((resp.get("data") or {}).get(code) or {}).get("candles") or []
        frames.append(_ind_frame(candles))               # holidays come back empty
    return pd.concat(frames) if frames else pd.DataFrame(columns=COLUMNS)


def fetch_universe(
    symbols: Optional[List[str]] = None,
    start: str = "2023-01-01",
    end: Optional[str] = None,
    minutes: int = 5,
    feed: Optional[str] = None,
    rth_only: bool = True,
    use_cache: bool = True,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch every symbol, one at a time so the cache is per-symbol and a failure on
    one name doesn't cost you the whole download.
    """
    symbols = symbols or DEFAULT_UNIVERSE
    end = end or (dt.date.today() + dt.timedelta(days=1)).isoformat()
    out: Dict[str, pd.DataFrame] = {}

    for i, sym in enumerate(symbols, 1):
        cached = _cache_path(sym, minutes, start, end).exists()
        try:
            df = fetch_bars(sym, start, end, minutes, feed, rth_only, use_cache)
        except API_ERRORS as e:
            if "credentials missing" in str(e) or "login failed" in str(e):
                raise                      # every symbol would fail the same way
            if verbose:
                print(f"[data] {sym}: FAILED ({str(e)[:100]}) - skipping")
            continue
        if df.empty:
            if verbose:
                print(f"[data] {sym}: no bars returned - skipping")
            continue
        out[sym] = df
        if verbose:
            tag = "cache" if cached else "fetch"
            print(f"[data] {i}/{len(symbols)} {sym}: {len(df):,} bars "
                  f"({df.index.min().date()} -> {df.index.max().date()}) [{tag}]")

    if not out:
        raise RuntimeError("No data for any symbol. Check credentials, symbols and dates.")
    return out


if __name__ == "__main__":
    # Smoke test: needs the four ANGEL_* variables (see core/angel.py), or with
    # --broker indmoney the INDSTOCKS_* ones (see core/indstocks.py).
    import sys
    if "--broker" in sys.argv:
        os.environ["BROKER"] = sys.argv[sys.argv.index("--broker") + 1]
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    try:
        bars = fetch_universe(["RELIANCE", "SBIN"], start="2024-01-01", end="2024-03-01",
                              minutes=5)
    except (*API_ERRORS, RuntimeError, ValueError) as e:
        sys.exit(f"[data] {e}")
    print(f"[data] source: {source()}")
    rel = bars["RELIANCE"]
    print(rel.head(3))
    print(f"\nbars/day check: {len(rel) / rel.index.normalize().nunique():.1f} "
          f"(expect 75 for 5-min bars over 09:15-15:30)")
