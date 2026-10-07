"""
data.py
=======
NSE bars from Angel One's SmartAPI historical endpoint, cached to disk.

Why a broker API and not yfinance here: yfinance caps 5-minute history at about
60 days, roughly twelve trading weeks, which is not enough to survive one bad
regime, and its NSE quotes run about 15 minutes late, which is the whole life of
a 5-minute breakout. SmartAPI is free with an Angel One account, real time, and
serves years of intraday candles, up to 100 days of 5-minute bars per request.

Everything is returned in Asia/Kolkata time and filtered to the regular session
(09:15-15:30) by default. The pre-open auction (09:00-09:08) prints one
discovered price, not a market, and would flatter any opening-range strategy.

Credentials: see core/angel.py.
"""

from __future__ import annotations

import datetime as dt
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from angel import AngelError, scrip, session
from market import NSE

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


def _interval(minutes: int):
    try:
        return INTERVALS[minutes]
    except KeyError:
        raise ValueError(f"SmartAPI has no {minutes}-minute candles; "
                         f"use one of {sorted(INTERVALS)} (0 = daily)") from None


def _cache_path(symbol: str, minutes: int, start: str, end: str) -> Path:
    tf = "1D" if minutes == 0 else f"{minutes}T"
    return CACHE_DIR / f"NSE_{symbol}_{tf}_{start}_{end}.pkl"


def _chunks(start: dt.date, end: dt.date, max_days: int):
    """[start, end) split into windows a single request can carry, a little under the cap."""
    step = dt.timedelta(days=max(1, max_days - 10))
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

    df = pd.concat(frames) if frames else pd.DataFrame(columns=COLUMNS)
    df = _to_session(df, minutes, rth_only)

    if use_cache:
        with open(path, "wb") as f:
            pickle.dump(df, f)
    return df


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
        except AngelError as e:
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
    # Smoke test: needs the four ANGEL_* variables (see core/angel.py).
    import os
    import sys
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    try:
        bars = fetch_universe(["RELIANCE", "SBIN"], start="2024-01-01", end="2024-03-01",
                              minutes=5)
    except (AngelError, RuntimeError) as e:
        sys.exit(f"[data] {e}")
    rel = bars["RELIANCE"]
    print(rel.head(3))
    print(f"\nbars/day check: {len(rel) / rel.index.normalize().nunique():.1f} "
          f"(expect 75 for 5-min bars over 09:15-15:30)")
