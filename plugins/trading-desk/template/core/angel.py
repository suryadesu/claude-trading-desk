"""
angel.py
========
One Angel One SmartAPI session for everything: historical bars, live bars,
account state and (only with --real-money) orders.

  ANGEL_API_KEY        SmartAPI app key (smartapi.angelone.in -> My Apps)
  ANGEL_CLIENT_CODE    your Angel One client ID
  ANGEL_MPIN           your 4-digit MPIN (SmartAPI login takes the MPIN, not a password)
  ANGEL_TOTP_SECRET    the secret behind the TOTP QR (smartapi.angelone.in/enable-totp)

Login is fully automatic with TOTP, so a bot can re-authenticate itself every
morning. The session (JWT) is good for the trading day; we log in once per day
per process and again on any token error.

Rate limits (smartapi docs, "RateLimit") are per client code, and the server
answers an over-limit call with 403 "Access denied because of exceeding rate
limit". Each endpoint gets its own minimum spacing below, and every call goes
through backoff, because the forum also reports that limit firing on traffic
well under it.

Credentials and tokens are never printed. The SDK itself writes a daily
logs/<date>/app.log in the working directory and, on a network error, logs
the request headers there, bearer token included. We switch that file
logging off as soon as the client exists.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

SCRIP_MASTER_URL = ("https://margincalculator.angelone.in/OpenAPI_File/files/"
                    "OpenAPIScripMaster.json")
CACHE_DIR = Path(__file__).parent / "cache"
ENV_KEYS = ("ANGEL_API_KEY", "ANGEL_CLIENT_CODE", "ANGEL_MPIN", "ANGEL_TOTP_SECRET")

# Minimum seconds between calls, per endpoint, a little inside the published limits.
SPACING = {
    "login": 1.2,          # 1/s
    "candles": 0.45,       # 3/s and 150/min -> 0.4 s sustained
    "orderbook": 1.1,      # 1/s
    "position": 1.1,       # 1/s
    "rms": 0.6,            # 2/s
    "order": 0.15,         # 9/s across place, modify and cancel together
    "details": 0.15,       # 10/s
    "ltp": 0.15,           # 10/s
}


class AngelError(RuntimeError):
    pass


@dataclass(frozen=True)
class Scrip:
    symbol: str            # RELIANCE
    tradingsymbol: str     # RELIANCE-EQ
    token: str             # 2885
    tick_size: float       # rupees (the master lists paise)
    lot_size: int


def credentials_present() -> bool:
    return all(os.environ.get(k) for k in ENV_KEYS)


def _quiet_sdk_logging() -> None:
    """Stop the SDK writing request headers (bearer token included) to logs/."""
    try:
        import logzero
        logzero.logfile(None)
        logzero.loglevel(logging.CRITICAL)
    except Exception:
        pass
    logging.getLogger("SmartApi").setLevel(logging.CRITICAL)


# ------------------------------------------------------------------ scrips

_scrips: Optional[Dict[str, Scrip]] = None
_scrip_lock = threading.Lock()


def _load_master() -> list:
    """The day's instrument master, downloaded once a day. It is public; no login needed."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"angel_scrips_{dt.date.today().isoformat()}.json"
    if not path.exists():
        with urllib.request.urlopen(SCRIP_MASTER_URL, timeout=60) as r:
            path.write_bytes(r.read())
        for old in CACHE_DIR.glob("angel_scrips_*.json"):
            if old != path:
                old.unlink(missing_ok=True)
    return json.loads(path.read_text(encoding="utf-8"))


def scrips() -> Dict[str, Scrip]:
    """NSE cash equities keyed by plain symbol: RELIANCE -> RELIANCE-EQ, token 2885."""
    global _scrips
    with _scrip_lock:
        if _scrips is None:
            out: Dict[str, Scrip] = {}
            for row in _load_master():
                seg = str(row.get("exch_seg", "")).upper()
                tsym = str(row.get("symbol", ""))
                if seg not in ("NSE", "NSE_CM") or not tsym.endswith("-EQ"):
                    continue
                try:
                    tick = float(row.get("tick_size") or 5) / 100.0   # paise -> rupees
                    lot = int(float(row.get("lotsize") or 1))
                except ValueError:
                    continue
                name = tsym[:-3]
                out[name] = Scrip(symbol=name, tradingsymbol=tsym, token=str(row["token"]),
                                  tick_size=tick, lot_size=lot)
            _scrips = out
        return _scrips


def scrip(symbol: str) -> Scrip:
    sym = symbol.upper().removesuffix("-EQ")
    try:
        return scrips()[sym]
    except KeyError:
        raise AngelError(f"{symbol}: not an NSE equity in Angel's instrument master") from None


def round_to_tick(price: float, tick: float) -> float:
    return round(round(price / tick) * tick, 2)


# ------------------------------------------------------------------ session

class Session:
    """Thread-safe, paced SmartConnect wrapper. Use session() for the shared one."""

    def __init__(self):
        missing = [k for k in ENV_KEYS if not os.environ.get(k)]
        if missing:
            raise AngelError(
                "Angel One credentials missing: " + ", ".join(missing)
                + ". Put them in .env (see deploy/.env.example).")
        self._client = None
        self._login_day: Optional[dt.date] = None
        self._lock = threading.Lock()          # login state
        self._pace_lock = threading.Lock()     # call spacing (taken inside login too)
        self._last: Dict[str, float] = {}

    # -- auth ------------------------------------------------------

    def _login(self) -> None:
        import pyotp
        from SmartApi import SmartConnect

        client = SmartConnect(api_key=os.environ["ANGEL_API_KEY"])
        _quiet_sdk_logging()
        self._pace("login")
        resp = client.generateSession(os.environ["ANGEL_CLIENT_CODE"],
                                      os.environ["ANGEL_MPIN"],
                                      pyotp.TOTP(os.environ["ANGEL_TOTP_SECRET"]).now())
        if not isinstance(resp, dict) or not resp.get("status"):
            msg = resp.get("message", "unknown error") if isinstance(resp, dict) else str(resp)
            code = resp.get("errorcode", "") if isinstance(resp, dict) else ""
            raise AngelError(f"Angel One login failed: {msg} {code}".strip())
        self._client = client
        self._login_day = dt.date.today()

    def client(self):
        with self._lock:
            if self._client is None or self._login_day != dt.date.today():
                self._login()
            return self._client

    def _relogin(self) -> None:
        with self._lock:
            self._client = None
        self.client()

    # -- pacing and retries ------------------------------------------

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

    @staticmethod
    def _is_rate_limit(text: str) -> bool:
        t = text.lower()
        return "rate" in t and ("exceed" in t or "limit" in t) or "access denied" in t

    @staticmethod
    def _is_auth(text: str) -> bool:
        t = text.lower()
        return "token" in t and ("invalid" in t or "expired" in t) or "ag8001" in t

    def call(self, kind: str, fn: Callable[[Any], Any], retries: int = 5,
             idempotent: bool = True) -> Any:
        """
        Run fn(client) paced for `kind`, with backoff on rate limits and network
        errors and one re-login on a token error.

        idempotent=False (order placement) never retries after the request may
        have reached the broker: a retried place-order after a timeout is how a
        bot ends up with two positions. The caller reconciles by ordertag instead.
        """
        relogged = False
        for attempt in range(retries + 1):
            self._pace(kind)
            try:
                resp = fn(self.client())
            except AngelError:
                raise
            except Exception as e:                       # SDK raises a zoo of types
                text = f"{type(e).__name__}: {e}"
                if self._is_auth(text) and not relogged:
                    relogged = True
                    self._relogin()
                    continue
                if not idempotent:
                    raise AngelError(f"{kind}: {text[:160]}") from None
                if attempt == retries:
                    raise AngelError(f"{kind}: giving up after {attempt + 1} tries: "
                                     f"{text[:160]}") from None
                time.sleep(min(30.0, 1.0 * 2 ** attempt))
                continue

            if isinstance(resp, dict) and resp.get("status") is False:
                text = f"{resp.get('message', '')} {resp.get('errorcode', '')}"
                if self._is_auth(text) and not relogged:
                    relogged = True
                    self._relogin()
                    continue
                if self._is_rate_limit(text) and idempotent and attempt < retries:
                    time.sleep(min(30.0, 1.0 * 2 ** attempt))
                    continue
            return resp
        raise AngelError(f"{kind}: no response")


_session: Optional[Session] = None
_session_lock = threading.Lock()


def session() -> Session:
    global _session
    with _session_lock:
        if _session is None:
            _session = Session()
        return _session
