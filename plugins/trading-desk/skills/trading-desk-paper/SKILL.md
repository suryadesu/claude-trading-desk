---
name: trading-desk-paper
description: Take a backtested strategy live on NSE with Angel One or INDmoney - simulated fills by default, replay validation, kill switches, state files, real-money safeguards (ROBO brackets, limit-only API orders, static IP, 3:15 PM square-off) and the failure modes that only appear live. Use when wiring a live loop or debugging one.
version: 2.0.0
tags: [trading, nse, angel-one, smartapi, indmoney, indstocks, paper-trading, live]
allowed-tools: Read, Grep, Glob, Bash, Write, Edit
---

# Paper trading on NSE

A live loop is not a backtest with the dates removed. These are the parts that
only exist live.

## There is no broker sandbox, so simulate by default

Angel One (like Groww, Zerodha and most Indian brokers) has no paper-trading
environment: every API order is a real order. The live runner therefore
defaults to `SimBroker` (`core/brokers.py`):

- live prices from SmartAPI, fills booked locally
- fills at the signal bar's close plus adverse slippage, stop first on an
  ambiguous bar, flat at 15:05 IST: the backtest engine's own rules
- the same contract-note charges as the backtest (`core/market.py`)
- state in `orb/out/sim_state.json`, so a restart cannot double-enter
- the dashboard card says `fills: "simulated"`

Simulated results are hypothetical. They show what the rules would have done,
not what the market would have given you. The gap between the two is mostly
slippage, and only real fills can measure it.

## Validate by replay before trusting it

Run the live loop against a past date with `--replay YYYY-MM-DD` and check it
produces the same trades the backtest did for that day. A replay always uses a
fresh simulator. If replay and backtest disagree, one of them is wrong and you do
not yet know which.

This catches the whole class of bug where the live path computes a feature
differently from the backtest path. Share the feature code between them.

Only act on bars that have closed. SmartAPI returns the candle that is still
forming; `live.py` drops any bar whose `start + bar length` is after now.

## A kill switch you can reach without a deploy

Check for a file on every cycle and stand down if it exists:

```python
if (OUT / "STOP").exists():
    log("kill switch present, standing down")
    return
```

`touch orb/out/STOP` (or `docker compose exec orb touch /app/orb/out/STOP`)
stops new entries from an ssh session in one command. You want this before you
need it.

## Write state the same way every bot does

Use `core/botstate.py`, which gives every bot two files:

- `state/<bot>.json`, a snapshot replaced **atomically** (tempfile then rename,
  so a reader never sees half a file)
- `state/<bot>.events.jsonl`, append-only, one JSON object per line

That is the whole contract with the dashboard. Do not invent a second format.
Bots may write other things into the state directory, so anything reading it must
identify a snapshot by its `bot` field rather than assuming every `.json` is one.

## Data has sharp edges

- SmartAPI candles are free and real time, up to 100 days of 5-minute bars per
  request, 3 requests a second and 150 a minute. The forum also reports the rate
  limit firing well under that, so every call goes through backoff
  (`core/angel.py`).
- Bars come back in IST. The regular session is 09:15-15:30; drop the 09:00-09:08
  pre-open auction, which prints one discovered price, not a market.
- yfinance `.NS` daily bars are fine for a smoke test. Its intraday history is
  about 60 days and runs about 15 minutes late, so it cannot run an intraday
  strategy live or test one properly.
- NSE holidays are real closures. `MarketClock` reads them from
  `pandas_market_calendars`; a weekday-only clock would wait on Diwali.
- Indian cash equities cannot be held short overnight. Intraday (MIS) shorts are
  fine; anything held for days is long-only.

## Real money: only with --real-money

`AngelBroker` refuses to exist unless `--real-money` is passed **and** the
environment has `ANGEL_REAL_MONEY=I_ACCEPT_REAL_LOSSES`. It also enforces
`--max-order-value` and `--max-orders-per-day` before any order leaves.

What the exchange requires of API orders (NSE/SEBI retail algo framework, in
force since April 2026):

- **Static IP.** Orders are accepted only from the IP registered on the SmartAPI
  app. On Oracle, reserve a public IP and register it.
- **No market or IOC orders.** Entries are marketable LIMIT orders priced
  `--entry-band-bps` through the signal close, and exits are LIMIT orders re-priced
  through the last price. `AngelBroker._guard` refuses anything else.
- Under 10 orders a second needs no strategy registration with the exchange.

How positions stay protected:

- **Entry is a ROBO bracket order**: LIMIT entry, target leg and stop-loss leg in
  one request, so a crash cannot leave a naked position. If the bracket is
  rejected the trade is skipped; there is no fallback to an unprotected entry.
- `squareoff` and `stoploss` are sent as rupee distances from the entry
  (`--robo-units points`). SmartAPI does not document the unit. **Place one
  1-share bracket by hand through the API and check the legs in the Angel app
  before trusting it**; switch to `--robo-units price` if the legs land in the
  wrong place.
- Unfilled within 30 seconds: cancelled. A stale breakout is not the trade you
  measured.
- Never retry a place-order call after a timeout. Retrying is how a bot ends up
  with two positions. The session refuses to retry order calls, and each
  bracket carries an `ordertag` that is checked against the order book before
  sending.
- Flatten at 15:05: cancel working legs, then exit with limit orders, cancelling
  the previous exit before re-pricing so two exits can never both fill. Angel
  squares off open intraday positions itself at 15:15 and charges Rs 50 + GST
  per position for it.

## INDmoney instead of Angel One

`--broker indmoney` (or `BROKER=indmoney`) switches data, brokerage and the
real-money broker to INDmoney's INDstocks API (`core/indstocks.py`,
`IndBroker`). What differs:

- **Consent** is `INDSTOCKS_REAL_MONEY=I_ACCEPT_REAL_LOSSES`. Angel's variable
  does not unlock it.
- **Entry is a smart order**: a LIMIT entry with stop-loss and target legs in one
  request. The stop leg is a stop-limit with `--sl-limit-band-bps` of room. Its
  legs' arming isn't documented, so **one 1-share order by hand first**.
- **One token at a time.** A new token, from the website or another process,
  ends the old one. The client caches one token in `core/cache/` and shares it
  between processes, and regenerates on a 401, at most once a minute.
- **History is 7 days per request**, silently truncated beyond that, so
  `core/data.py` pages in 6-day windows.
- **Brokerage is ₹10 flat per order** (`india_intraday_ind`); statutory charges
  are identical.

Reconcile open positions from the broker on startup rather than from your own
state file: the broker is the authority on what you own.

## What to watch on day one

- did the bot wake at 09:15 IST, and sleep through the holiday you expected
- did it find the candidates the backtest would have found
- is achieved risk per trade what you configured
- in simulation: do the logged decisions match a `--replay` of the same day
- with real money: do fills land near the prices the backtest assumed, and if
  not, the friction formula in `trading-desk-method` tells you what that costs
