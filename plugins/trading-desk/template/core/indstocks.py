"""
indstocks.py
============
The INDmoney alternative to core/angel.py: one INDstocks API session for
historical bars, live prices, account state and (only with --real-money) orders.
Pick it with --broker indmoney or BROKER=indmoney.

  INDSTOCKS_CLIENT_ID      shown on indstocks.com -> access tokens, after TOTP setup
  INDSTOCKS_MPIN           your INDmoney MPIN
  INDSTOCKS_TOTP_SECRET    the secret behind the TOTP QR (shown once, at setup)
  INDSTOCKS_ACCESS_TOKEN   optional: a token copied from the dashboard instead of
                           the three above. It lasts 24 hours.

Docs: https://api-docs.indstocks.com. Plain REST, no SDK.

Two behaviours shape this module:

  - Only ONE token is live at a time: generating a new one kills the previous
    one, and generation is limited to once a minute. So the token is cached on
    disk (core/cache/, gitignored) and shared by the backtest, the live bot and
    the smoke test, instead of each process logging the others out. Generating
    a token on the website also kills ours; the session notices the 401 and
    makes a new one.
  - Historical intraday candles come 7 days per request at most, and a wider
    window is silently cut short rather than refused. core/data.py pages in
    6-day windows for that reason.

Rate limits (API Conventions page): orders 10/s, data and quotes 5/s, other
read endpoints 15/s; over the limit is HTTP 429. Credentials and tokens are
never printed.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from angel import Scrip, round_to_tick  # noqa: F401  (shared types; no SDK import)

BASE_URL = "https://api.indstocks.com"
CACHE_DIR = Path(__file__).parent / "cache"
TOKEN_PATH = CACHE_DIR / "indstocks_token.json"
TOKEN_MAX_AGE_S = 23 * 3600            # tokens last 24 h; renew a little early
ENV_KEYS = ("INDSTOCKS_CLIENT_ID", "INDSTOCKS_MPIN", "INDSTOCKS_TOTP_SECRET")
TOKEN_ENV = "INDSTOCKS_ACCESS_TOKEN"
ALGO_ID_NSE = "99999"                  # required on every NSE order (exchange algo id)

# Minimum seconds between calls, per endpoint class, a little inside the limits.
SPACING = {
    "token": 61.0,         # 1 per minute
    "data": 0.22,          # 5/s: instruments, historical
    "quote": 0.22,         # 5/s
    "order": 0.11,         # 10/s: place, modify, cancel
    "read": 0.07,          # 15/s: funds, positions, order book
}


class IndError(RuntimeError):
    pass


def credentials_present() -> bool:
    return bool(os.environ.get(TOKEN_ENV)) or all(os.environ.get(k) for k in ENV_KEYS)


def scrip_code(sc: Scrip) -> str:
    """The id the market-data endpoints take: NSE_2885."""
    return f"NSE_{sc.token}"


# ------------------------------------------------------------------ session

class Session:
    """Thread-safe, paced HTTP client. Use session() for the shared one."""

    def __init__(self, http=None):
        if not credentials_present():
            missing = [k for k in ENV_KEYS if not os.environ.get(k)]
            raise IndError(
                "INDstocks credentials missing: " + ", ".join(missing)
                + f" (or set {TOKEN_ENV}). Put them in .env (see deploy/.env.example).")
        if http is None:
            import requests
            http = requests.Session()
        self.http = http
        self._token: Optional[str] = None
        self._lock = threading.Lock()          # token state
        self._pace_lock = threading.Lock()
        self._last: Dict[str, float] = {}
        self._last_generated = 0.0

    # -- auth ------------------------------------------------------

    def _read_cached(self) -> Optional[str]:
        try:
            d = json.loads(TOKEN_PATH.read_text())
        except (OSError, ValueError):
            return None
        if time.time() - float(d.get("created", 0)) > TOKEN_MAX_AGE_S:
            return None
        return d.get("token") or None

    def _write_cached(self, token: str) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = TOKEN_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps({"token": token, "created": time.time()}))
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        tmp.replace(TOKEN_PATH)

    def _generate(self) -> str:
        if not all(os.environ.get(k) for k in ENV_KEYS):
            raise IndError("INDstocks token expired or rejected, and no TOTP credentials "
                           f"to make a new one. Set {', '.join(ENV_KEYS)} in .env, or paste "
                           f"a fresh {TOKEN_ENV}.")
        import pyotp
        wait = self._last_generated + SPACING["token"] - time.monotonic()
        if self._last_generated and wait > 0:
            time.sleep(wait)                    # the server allows one per minute
        self._last_generated = time.monotonic()
        r = self.http.post(BASE_URL + "/generate/token", timeout=30,
                           headers={"x-api-key": os.environ["INDSTOCKS_CLIENT_ID"],
                                    "Content-Type": "application/json"},
                           json={"mpin": os.environ["INDSTOCKS_MPIN"],
                                 "totp": pyotp.TOTP(os.environ["INDSTOCKS_TOTP_SECRET"]).now()})
        body = _json(r)
        token = (body.get("token") or (body.get("data") or {}).get("token")
                 if isinstance(body, dict) else None)
        if r.status_code >= 400 or not token:
            msg = body.get("message", "") if isinstance(body, dict) else ""
            raise IndError(f"INDstocks login failed: HTTP {r.status_code} {msg}".strip())
        self._write_cached(token)
        return token

    def token(self, fresh: bool = False) -> str:
        with self._lock:
            if fresh:
                self._token = None
                if TOKEN_PATH.exists():
                    TOKEN_PATH.unlink(missing_ok=True)
            if self._token is None and not fresh:
                self._token = os.environ.get(TOKEN_ENV) or self._read_cached()
            if self._token is None:
                self._token = self._generate()
            return self._token

    # -- pacing and retries ----------------------------------------

    def _pace(self, kind: str) -> None:
        gap = SPACING.get(kind, 0.2)
        while True:
            with self._pace_lock:
                now = time.monotonic()
                wait = self._last.get(kind, 0.0) + gap - now
                if wait <= 0:
                    self._last[kind] = now
                    return
            time.sleep(wait)

    def call(self, kind: str, method: str, path: str, params: Optional[dict] = None,
             body: Optional[dict] = None, retries: int = 5, idempotent: bool = True,
             raw: bool = False) -> Any:
        """
        One paced request with backoff on 429 and network errors and one token
        refresh on 401/403. Returns the decoded JSON (or text, with raw=True).

        idempotent=False (order placement) never retries once the request may have
        reached the broker. The caller reconciles by the order's `remarks` tag.
        """
        refreshed = False
        for attempt in range(retries + 1):
            self._pace(kind)
            headers = {"Authorization": self.token(), "Content-Type": "application/json"}
            try:
                r = self.http.request(method, BASE_URL + path, params=params, json=body,
                                      headers=headers, timeout=30)
            except Exception as e:                          # network error
                if not idempotent:
                    raise IndError(f"{path}: {type(e).__name__}: {str(e)[:120]}") from None
                if attempt == retries:
                    raise IndError(f"{path}: giving up after {attempt + 1} tries: "
                                   f"{type(e).__name__}") from None
                time.sleep(min(30.0, 1.0 * 2 ** attempt))
                continue

            if r.status_code in (401, 403) and not refreshed and not os.environ.get(TOKEN_ENV):
                refreshed = True
                self.token(fresh=True)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                if idempotent and attempt < retries:
                    time.sleep(min(30.0, 1.0 * 2 ** attempt))
                    continue
            if r.status_code >= 400:
                b = _json(r)
                msg = (b.get("message") or b.get("error_code") or "") if isinstance(b, dict) else ""
                raise IndError(f"{path}: HTTP {r.status_code} {msg}".strip())
            return r.text if raw else _json(r)
        raise IndError(f"{path}: no response")


def _json(r) -> Any:
    try:
        return r.json()
    except Exception:
        return {}


_session: Optional[Session] = None
_session_lock = threading.Lock()


def session() -> Session:
    global _session
    with _session_lock:
        if _session is None:
            _session = Session()
        return _session


# ------------------------------------------------------------------ scrips

_scrips: Optional[Dict[str, Scrip]] = None
_scrip_lock = threading.Lock()


def _load_master() -> str:
    """The day's equity instrument CSV, downloaded once a day."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"indstocks_equity_{dt.date.today().isoformat()}.csv"
    if not path.exists():
        text = session().call("data", "GET", "/market/instruments",
                              params={"source": "equity"}, raw=True)
        path.write_text(text, encoding="utf-8")
        for old in CACHE_DIR.glob("indstocks_equity_*.csv"):
            if old != path:
                old.unlink(missing_ok=True)
    return path.read_text(encoding="utf-8")


def parse_master(text: str) -> Dict[str, Scrip]:
    """NSE cash equities keyed by plain symbol: RELIANCE -> security id 2885."""
    rows = list(csv.DictReader(io.StringIO(text)))
    rows = [{(k or "").strip().upper(): (v or "").strip() for k, v in r.items()} for r in rows]

    # TICK_SIZE units are not documented. If any row has a fractional tick the
    # column is in rupees; if every tick is a whole number it is in paise (as in
    # Angel's master, where 5 means Rs 0.05).
    ticks = []
    for r in rows:
        try:
            ticks.append(float(r.get("TICK_SIZE") or 0))
        except ValueError:
            pass
    in_paise = bool(ticks) and all(t == int(t) for t in ticks if t > 0)

    out: Dict[str, Scrip] = {}
    for r in rows:
        if r.get("EXCH", "").upper() != "NSE":
            continue
        series = r.get("SERIES", "").upper()
        if series and series != "EQ":
            continue
        tsym = r.get("TRADING_SYMBOL") or r.get("SYMBOL_NAME") or ""
        name = tsym.upper().removesuffix("-EQ")
        sid = r.get("SECURITY_ID", "")
        if not name or not sid:
            continue
        try:
            tick = float(r.get("TICK_SIZE") or 0) or 0.05
            lot = int(float(r.get("LOT_UNITS") or 1)) or 1
        except ValueError:
            continue
        if in_paise:
            tick /= 100.0
        out.setdefault(name, Scrip(symbol=name, tradingsymbol=tsym, token=sid,
                                   tick_size=tick, lot_size=lot))
    return out


def scrips() -> Dict[str, Scrip]:
    global _scrips
    with _scrip_lock:
        if _scrips is None:
            _scrips = parse_master(_load_master())
        return _scrips


def scrip(symbol: str) -> Scrip:
    sym = symbol.upper().removesuffix("-EQ")
    try:
        return scrips()[sym]
    except KeyError:
        raise IndError(f"{symbol}: not an NSE equity in the INDstocks instrument master") from None


def ltp(symbols) -> Dict[str, float]:
    """Last traded price for plain symbols, one request for all of them."""
    codes = {scrip_code(scrip(s)): s for s in symbols}
    resp = session().call("quote", "GET", "/market/quotes/ltp",
                          params={"scrip-codes": ",".join(codes)})
    data = (resp or {}).get("data") or {}
    out = {}
    for code, sym in codes.items():
        try:
            out[sym] = float((data.get(code) or {}).get("live_price") or 0)
        except (TypeError, ValueError):
            out[sym] = 0.0
    return out
