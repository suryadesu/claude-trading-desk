# Testing the NSE setup (Angel One or INDmoney + Laya)

How to check that the trading desk works on Indian stocks: NSE data from Angel
One's SmartAPI or INDmoney's INDstocks API, the open-source
[Laya](https://huggingface.co/convaiinnovations/laya) decision model, and the
live bot's simulated fills. It goes from a first smoke
test through to the full Docker stack. Real-money trading is covered last, and
only as an option.

Each level depends on the one before it. Levels 1 to 3 need no API keys at all.

| Level | What it proves | Needs |
| --- | --- | --- |
| 1. Model server | Laya runs and answers requests | Python 3.10+ |
| 2. Keyless backtest | The `laya` arm works end to end on real NSE data, with Indian charges | Level 1 |
| 3. Calibration and replay | Threshold fitting works, and results are reproducible | Level 2 |
| 4. Broker data and ORB | The real strategy on NSE intraday data | A free Angel One SmartAPI key, or INDmoney INDstocks API access |
| 5. Live bot, simulated fills | The live loop on live NSE prices, no orders | Level 4, market hours |
| 6. Docker | The deployed stack | Docker |
| 7. Real money (optional) | Real orders on Angel One or INDmoney | Static IP, explicit consent |

All commands run from the template folder:

```bash
cd plugins/trading-desk/template
```

They use Git Bash syntax with explicit paths into each virtual environment, so
nothing needs activating. On macOS or Linux, use `bin/` instead of `Scripts/` in
the venv paths.

**Neither broker has a paper-trading sandbox.** Every order their APIs accept
is real. That is why the live bot books simulated fills by default, and why real
orders need both a flag and an environment variable (Level 7).

### Choosing a broker

Angel One is the default. To use INDmoney instead, add `--broker indmoney` to
any command in Levels 4 to 7, or put `BROKER=indmoney` in `.env`. The switch
picks both where bars come from and whose brokerage the backtest and simulator
charge.

| | Angel One SmartAPI (default) | INDmoney INDstocks API |
| --- | --- | --- |
| API cost | free | free |
| Brokerage per executed order | ₹20 or 0.1%, whichever is lower | ₹10 flat |
| 5-minute history per request | 100 days | 7 days (a first download makes about 15 times as many calls) |
| Protected entry | ROBO bracket order | smart order with stop-loss and target legs |
| Login | TOTP, one session per day | TOTP; only **one** token is live at a time |
| Sandbox | none | none |

Statutory charges (STT, exchange, SEBI, stamp duty, GST) are the same for both.
The bars are the same NSE prices, so a strategy's edge before costs doesn't
depend on the broker.

---

## Level 1: Start the Laya model server

Laya runs as its own server, separate from the bots. It needs Python 3.10 or
newer and torch, and the bots need neither.

**Use a short path for this virtual environment on Windows.** Torch has deeply
nested files, and a long path such as one under `AppData\Local\Temp` fails with
`WinError 206: The filename or extension is too long`.

In a first terminal:

```bash
python -m venv C:/lv
```
```bash
C:/lv/Scripts/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
```
```bash
C:/lv/Scripts/python -m pip install "laya[serve]"
```
```bash
LAYA_PRELOAD=1 LAYA_MODELS=typed-decisions LAYA_DEFAULT_MODEL=typed-decisions C:/lv/Scripts/laya-serve
```

The first start downloads the model weights from Hugging Face into
`~/.cache/huggingface`, which takes about 30 seconds. Leave this terminal running.

**Check it is ready** from a second terminal:

```bash
curl -s localhost:8000/health
```

Expected: `{"status":"ok","loaded":["typed-decisions"],...,"device":"cpu",...}`.

On startup, the server may warn that the checkpoint ships invalid temperatures
and that confidence is uncalibrated. That is expected, and it is why Level 3
exists.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL_URL` | `http://localhost:8000/v1/systemone` | Where the bots send requests |
| `MODEL_NAME` | `typed-decisions` | Laya checkpoint: `typed-decisions` (1024 tokens), `english` (512), `multilingual` (1024) |
| `MODEL_MAX_LEN` | unset | Raise the model's token limit if descriptions get cut |
| `LAYA_API_KEY` | unset | Only needed if you started the server with one |

---

## Level 2: Run a backtest with no API keys

This uses the `example_sma` strategy, a moving-average crossover on two NSE ETFs
(`NIFTYBEES.NS` tracks the NIFTY 50, `BANKBEES.NS` the Bank NIFTY), using free
daily data from Yahoo. It is a harness test, not a strategy to trade.

The positions are held for days, so they are delivery (CNC) trades: they pay
delivery charges, and they are **long-only**, because Indian cash equities
cannot be held short overnight.

In a second terminal:

```bash
python -m venv C:/tdv
```
```bash
C:/tdv/Scripts/python -m pip install -r deploy/requirements.txt
```
```bash
C:/tdv/Scripts/python example_sma/run_backtest.py --arms rules gated laya --split 2022-01-01
```

**Expected output:**

- `=== data ===` shows about 2,900 daily bars for each ETF.
- `=== signals ===` shows about 110 crossovers in total.
- A line like
  `[laya] prefetch done: 112 decisions in 2.1 min (0.9/s), 0 errors, 0 truncated`.
- Result rows for `rules` and `gated`, each with `train` and `test` halves.
- At the default threshold of 0.55, `laya` shows
  `no trades (every candidate was vetoed)`. That is expected: before
  calibration, Laya's probabilities sit around 0.3 to 0.4 for every option.

**What to check:**

- `0 errors`. Otherwise the server isn't reachable at `MODEL_URL`.
- `0 truncated`. Otherwise candidate descriptions are longer than the model can
  read.

**Output files** go in `example_sma/out/`: `decisions_laya.jsonl` (every
question and answer), `trades_<arm>.csv` and `summary.json`. Model answers are
cached in `core/cache/laya_cache.jsonl`.

---

## Level 3: Calibrate, then read the out-of-sample half once

The threshold must be chosen on the **in-sample** half only. Fitting it on the
out-of-sample (test) half turns that half into training data, and the result
proves nothing.

### 3a. Fit on the in-sample half

```bash
C:/tdv/Scripts/python core/calibrate.py --decisions example_sma/out/decisions_laya.jsonl --trades example_sma/out/trades_rules.csv --before 2022-01-01 --tolerance 5D
```

`--tolerance 5D` suits daily bars. Use `1D` for intraday strategies such as ORB.

**Read these lines in order:**

1. **`spearman(p_side, R) = ...`** shows whether the model's probability ranks
   trades at all. If it is near zero or negative, stop here: no threshold will
   help.
2. **`fitted temperature T = ...`** is the adjustment to the model's
   probabilities.
3. **The reliability tables** show whether win rates rise with the model's
   probability.
4. **The threshold sweep** shows how many trades each threshold keeps, and their
   mean R and t-stat. Pick a threshold here.

On this NSE example, when this branch was set up, the result was:

```
spearman(p_side, R) = +0.028  (t = +0.15, n = 31): indistinguishable from no ranking at this sample size
```

So on this toy strategy Laya does not rank NSE trades, and the honest conclusion
is that the model adds nothing here. Some decisions are dropped from the fit
because the rules arm cannot take short-side crossovers on NSE.

### 3b. Read the out-of-sample half once

Only if 3a showed real ranking, use its T and your chosen threshold:

```bash
C:/tdv/Scripts/python example_sma/run_backtest.py --arms rules gated laya --split 2022-01-01 --model-temperature <T> --model-threshold <threshold>
```

Compare the `test` row of `laya` against `gated`, not against `rules`.

### 3c. Check that results are reproducible

Stop the Laya server (Ctrl+C in its terminal), then rerun 3b with
`--model-offline`. The trades taken, their sides and the model probabilities
should be identical, answered entirely from the cache. Yahoo re-adjusts
historical prices between downloads, so prices can differ in the 4th decimal.

---

## Level 4: Broker data and the ORB strategy

ORB trades 5-minute breakouts on 8 NIFTY 50 stocks. Its data comes from Angel
One's SmartAPI (4a) or, with `--broker indmoney`, INDmoney's INDstocks API
(4a-alt). Both are free with an account. You need only one of them.

### 4a. Get SmartAPI credentials

1. Open an Angel One account if you don't have one.
2. At [smartapi.angelone.in](https://smartapi.angelone.in), sign in, go to
   **My Apps** and create an app. Choose a Trading APIs app if you may ever
   trade for real (Level 7). Copy its **API key**.
3. At [smartapi.angelone.in/enable-totp](https://smartapi.angelone.in/enable-totp),
   enable TOTP. Copy the **text secret** shown under the QR code. It is a long
   string, not the 6-digit code.
4. Create `.env` in the template folder (copy the Angel section of
   `deploy/.env.example`) and fill in:

| Variable | What it is |
| --- | --- |
| `ANGEL_API_KEY` | the app's API key |
| `ANGEL_CLIENT_CODE` | your Angel One client ID |
| `ANGEL_MPIN` | your 4-digit MPIN. SmartAPI logs in with the MPIN, not your password |
| `ANGEL_TOTP_SECRET` | the TOTP text secret |

Never commit `.env`; it is in `.gitignore`. The code never prints these values,
and it turns off the SDK's own log file, which would otherwise record the
session token on a network error.

### 4a-alt. Or: get INDmoney (INDstocks) credentials

1. Open an INDmoney account with Indian stocks enabled, if you don't have one.
2. Sign in at [indstocks.com](https://indstocks.com) and open the API
   **access tokens** page.
3. Click **Setup TOTP**, scan the QR code into an authenticator app, and copy
   the **text secret** shown with it. It is shown only once. Confirm with one
   6-digit code within 5 minutes.
4. The page then shows your **Client ID**.
5. Add to `.env` in the template folder (see the INDmoney section of
   `deploy/.env.example`):

| Variable | What it is |
| --- | --- |
| `INDSTOCKS_CLIENT_ID` | the Client ID from the access tokens page |
| `INDSTOCKS_MPIN` | your INDmoney MPIN |
| `INDSTOCKS_TOTP_SECRET` | the TOTP text secret |
| `BROKER` | optional: `indmoney` to make it the default instead of passing `--broker indmoney` |

Instead of the three credentials, you can paste a token generated on the website
as `INDSTOCKS_ACCESS_TOKEN`. It lasts 24 hours, after which you paste a new one.

**Only one INDstocks token is live at a time.** Generating a new one, here or on
the website, ends the previous one. The code generates at most one token,
caches it in `core/cache/indstocks_token.json` (gitignored), and shares it
between the backtest, the live bot and the smoke test. If you generate a token
on the website while the bot runs, the bot's next call fails, and it logs in
again by itself.

### 4b. Smoke-test the data

```bash
C:/tdv/Scripts/python core/data.py
```

For INDmoney:

```bash
C:/tdv/Scripts/python core/data.py --broker indmoney
```

The INDmoney run also prints `[data] source: indmoney`.

**Expected:** RELIANCE and SBIN 5-minute bars for January to February 2024,
timestamps in `+05:30`, and the line `bars/day check: 75.0`. A login failure
prints the broker's message: check the values, and that your system clock
is right (TOTP codes are time-based).

### 4c. Backtest, calibrate, then read the test half once

With the Laya server running:

```bash
C:/tdv/Scripts/python orb/run_backtest.py --arms rules gated laya --start 2023-01-01 --end 2026-09-01
```
```bash
C:/tdv/Scripts/python core/calibrate.py --decisions orb/out/decisions_laya.jsonl --trades orb/out/trades_rules.csv --before 2025-03-01 --tolerance 1D
```
```bash
C:/tdv/Scripts/python orb/run_backtest.py --arms rules gated laya --start 2025-03-01 --end 2026-09-01 --model-temperature <T> --model-threshold <threshold>
```

The first run downloads about 3.5 years of 5-minute bars for 8 stocks, in pages
of 90 days, paced under SmartAPI's limit of 3 requests a second. Expect several
minutes; bars are cached in `core/cache/`, so later runs are fast. Costs are NSE
intraday charges: Angel brokerage of ₹20 or 0.1% per order, plus STT, exchange
and SEBI fees, stamp duty and GST (`core/market.py`).

**With INDmoney**, add `--broker indmoney` to all three commands, and point
calibration at the INDmoney files. Each run's output files get an `indmoney`
tag, so they never overwrite the Angel results:

```bash
C:/tdv/Scripts/python orb/run_backtest.py --broker indmoney --arms rules gated laya --start 2023-01-01 --end 2026-09-01
```
```bash
C:/tdv/Scripts/python core/calibrate.py --decisions orb/out/decisions_laya_indmoney.jsonl --trades orb/out/trades_rules_indmoney.csv --before 2025-03-01 --tolerance 1D
```

The download pages 6 days at a time (INDstocks serves at most 7 days of 5-minute
bars per request), so the first run makes about 220 requests per stock and takes
roughly 10 minutes for 8 stocks. Bars are cached separately (`IND_*.pkl`).
Brokerage is ₹10 flat per order; everything else is the same.

To compare the two brokers' costs on identical trades, run the same backtest
with each `--broker` and compare `summary.json` with `summary_indmoney.json`.
The `pnl_before_costs` lines should be nearly identical; the difference is all
brokerage.

**The parameter sweep** takes the same switch:

```bash
C:/tdv/Scripts/python orb/sweep.py --broker indmoney
```

Results go to `orb/out/sweep_indmoney.csv`.

| Flag | Default | Use |
| --- | --- | --- |
| `--equity` | 100000 | Starting capital in rupees |
| `--slippage-bps` | 2.0 | Adverse slippage per side |
| `--max-model-calls` | 20000 | Refuses runs with more candidates than this |
| `--prefetch-batch` | 16 | Candidates per Laya batch request (max 64) |

---

## Level 5: The live bot, simulated fills

The live bot reads live NSE prices from the broker and books fills locally, with
the backtest's own rules. No order ever reaches a broker in this mode. Add
`--broker indmoney` to any command below to use INDmoney's prices and
brokerage. The INDmoney simulator keeps its own books
(`sim_state_indmoney.json`, `sim_trades_indmoney.jsonl`), so switching brokers
never mixes the two.

### 5a. Replay a past session (any time, no market needed)

```bash
C:/tdv/Scripts/python orb/live.py --replay 2026-10-06 --decider rules
```

This walks one past session through the live code path, bar by bar, with a fresh
simulator. Trades go to `orb/out/replay_sim_trades.jsonl`. They should match the
backtest's trades for that day; if they don't, one of the two has a bug.

### 5b. One live cycle (market hours, 09:15-15:30 IST)

```bash
C:/tdv/Scripts/python orb/live.py --once --dry-run --decider laya --model-threshold <threshold> --model-temperature <T>
```

**Expected:** a `cycle:` line with equity in ₹ and `[sim]`, then one line per
breakout candidate. `--dry-run` logs what the simulator would fill without
booking it.

### 5c. Run it for a session

```bash
C:/tdv/Scripts/python orb/live.py --decider laya --model-threshold <threshold> --model-temperature <T>
```

It sleeps until 09:15 IST, skips NSE holidays, trades until 11:30, and flattens
at 15:05.

| File in `orb/out/` | What it holds |
| --- | --- |
| `sim_state.json` | simulated cash and open positions. Survives restarts |
| `sim_trades.jsonl` | every closed simulated trade, in the backtest's schema |
| `live_decisions.jsonl` | every candidate and what was decided |
| `live_laya_stream.jsonl` | every question and answer from Laya |
| `STOP` | create this file to stop new entries |

| Flag | Default | Use |
| --- | --- | --- |
| `--sim-equity` | 100000 | Simulator starting capital in rupees |
| `--slippage-bps` | 2.0 | Adverse slippage per simulated fill |
| `--daily-loss-pct` | 2.0 | Flatten and stand down after this loss in a day |
| `--flat-at-minute` | 350 | 15:05 IST, before the broker's own intraday square-off |
| `--broker` | `angel` | `indmoney` for INDmoney prices and brokerage |

---

## Level 6: Docker

1. Create `deploy/.env` from `deploy/.env.example`. Set the four `ANGEL_*`
   values, or `BROKER=indmoney` and the three `INDSTOCKS_*` values. Also set
   `ORB_MODEL_THRESHOLD` and `ORB_MODEL_TEMPERATURE` from your calibration.
2. Build and start:

```bash
cd deploy && docker compose --env-file .env up -d --build
```
```bash
docker compose ps
```
```bash
docker compose logs -f orb
```

**Expected:**

- four services: `laya`, `orb`, `dashboard`, `keepalive`.
- `laya` turns `healthy` once its weights download, then `orb` starts.
- `orb` logs `cycle:` lines with `[sim]` during market hours, and
  `market closed. next open ... IST` outside them.
- The dashboard at http://localhost:8080 shows amounts in ₹ and the ORB card
  tagged as simulated.

Stop with `docker compose down`. Simulator state lives in the `orb-out` volume
and survives restarts.

---

## Level 7: Real money (optional, at your own risk)

Only do this after Levels 4 to 6 look right and you have read
`skills/trading-desk-paper/SKILL.md`. Real orders lose real money, and nothing
here is financial advice.

### 7a. Exchange requirements

- **Static IP.** Since April 2026, NSE/SEBI rules accept API orders only from a
  registered static IP. For Angel, register it on your SmartAPI app (up to 5
  IPs). For INDmoney, register it on the indstocks.com API page (a primary and a
  secondary slot). Both allow changes only once a week. On Oracle, reserve a
  public IP for the VM.
- **Limit orders only.** API market and IOC orders are not allowed. The bot uses
  marketable LIMIT entries and exits, and refuses anything else.

### 7b. See the exact orders without sending them

```bash
ANGEL_REAL_MONEY=I_ACCEPT_REAL_LOSSES C:/tdv/Scripts/python orb/live.py --once --real-money --dry-run
```

It prints your account margin and the caps, waits 10 seconds, then logs the
exact SmartAPI order for any approved candidate as `DRY RUN would send to Angel
One: {...}`.

### 7c. Confirm the bracket's units with one share

SmartAPI doesn't document whether a ROBO bracket's `squareoff` and `stoploss`
are rupee distances from the entry or absolute prices. The bot sends distances
(`--robo-units points`). Before trusting it, place one 1-share ROBO order on a
cheap, liquid stock with those parameters, through the API from your static IP.
Then check in the Angel app that the target and stop legs sit where you
intended, and cancel it. If they're wrong, use `--robo-units price`.

### 7d. Go live

```bash
ANGEL_REAL_MONEY=I_ACCEPT_REAL_LOSSES C:/tdv/Scripts/python orb/live.py --real-money --max-order-value 20000 --max-orders-per-day 3
```

| Safeguard | Default |
| --- | --- |
| Refuses to start without `--real-money` and `ANGEL_REAL_MONEY=I_ACCEPT_REAL_LOSSES` | always |
| `--max-order-value` | ₹20,000 per order |
| `--max-orders-per-day` | 3 |
| Entry is a ROBO bracket, so the stop exists at the broker | always; a rejected bracket is skipped, never replaced by an unprotected entry |
| Unfilled entry | cancelled after 30 s |
| Order calls | never retried, and checked by `ordertag` against the order book first |
| Flatten | 15:05 IST, limit exits re-priced until flat; Angel squares off at 15:15 (₹50 + GST per position) |
| `orb/out/STOP`, daily loss limit | as in Level 5 |

To close everything immediately:

```bash
ANGEL_REAL_MONEY=I_ACCEPT_REAL_LOSSES C:/tdv/Scripts/python orb/live.py --flatten --real-money
```

### 7e. Real money on INDmoney instead

The same steps, with INDmoney's own consent variable. `ANGEL_REAL_MONEY` does
not unlock INDmoney, and the reverse is also true.

**See the exact order without sending it:**

```bash
INDSTOCKS_REAL_MONEY=I_ACCEPT_REAL_LOSSES C:/tdv/Scripts/python orb/live.py --broker indmoney --once --real-money --dry-run
```

It logs `DRY RUN would send to INDmoney: {...}`: one smart order with a LIMIT
entry, a stop-loss leg (`sl_trigger_price` at the stop, `sl_limit_price`
`--sl-limit-band-bps` beyond it) and a target leg (`tgt_trigger_price` and
`tgt_limit_price` at the target).

**Confirm the legs with one share.** The INDstocks docs don't say exactly when a
smart order's legs arm, or how they show in the order book. Before trusting the
bot, send one 1-share smart order on a cheap, liquid stock from your static IP.
Use the same fields the dry run printed. Check in the INDmoney app that the
entry fills, and that the stop and target legs then appear at the right prices.
Then cancel the legs and exit.

**Go live:**

```bash
INDSTOCKS_REAL_MONEY=I_ACCEPT_REAL_LOSSES C:/tdv/Scripts/python orb/live.py --broker indmoney --real-money --max-order-value 20000 --max-orders-per-day 3
```

| Safeguard | INDmoney behaviour |
| --- | --- |
| Consent | `--real-money` and `INDSTOCKS_REAL_MONEY=I_ACCEPT_REAL_LOSSES` |
| Caps | `--max-order-value`, `--max-orders-per-day`, as for Angel |
| Protected entry | one smart order with stop and target legs; a rejected order is skipped, never replaced by an unprotected entry |
| Stop leg | a stop-limit (API orders can't be market orders) with `--sl-limit-band-bps` (default 30) of room. A gap through that band can leave it unfilled; the 15:05 flatten is the backstop |
| Unfilled entry | entry cancelled after 30 s, and its legs too if nothing filled. A partial fill keeps its legs |
| Order calls | never retried, and checked by `remarks` tag against the order book first |
| Flatten | 15:05 IST, limit exits re-priced until flat. INDmoney's own square-off time isn't documented, so check it in the app |

To close everything immediately:

```bash
INDSTOCKS_REAL_MONEY=I_ACCEPT_REAL_LOSSES C:/tdv/Scripts/python orb/live.py --broker indmoney --flatten --real-money
```

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `WinError 206 ... filename too long` during install | Venv path too deep for torch on Windows | Use a short path such as `C:/lv` |
| `Angel One credentials missing: ...` | `.env` not found or incomplete | Create `.env` in the template folder with all four `ANGEL_*` values |
| `Angel One login failed: ...` | Wrong value, or clock skew breaking TOTP | Check the values; sync your system clock |
| `... exceeding rate limit` repeatedly | Too many requests, or SmartAPI's false positives | The client already backs off; wait a minute, or cache more |
| `not an NSE equity in Angel's instrument master` | Symbol typo, or not an `-EQ` series stock | Use the plain NSE symbol, e.g. `RELIANCE` |
| `[laya] giving up after 5 tries: URLError` | Laya server not running | Start `laya-serve`; check `curl localhost:8000/health` |
| `laya` arm takes no trades | Uncalibrated threshold | Run Level 3 / 4c |
| `market closed. next open ...` on a weekday | NSE holiday | Expected |
| `REFUSED, ... exceeds --max-order-value` | Real-money cap working | Lower size, or raise the cap knowingly |
| Real orders rejected for the IP | Static IP not registered with the broker | Register the server's static IP (SmartAPI app, or indstocks.com API page) |
| `INDstocks credentials missing: ...` | `--broker indmoney` without the INDSTOCKS values | Add them to `.env` (Level 4a-alt), or set `INDSTOCKS_ACCESS_TOKEN` |
| `INDstocks login failed: HTTP 4xx` | Wrong Client ID, MPIN or TOTP secret, clock skew, or a lockout after 5 bad codes | Check the values and your clock; after a lockout, wait 15 minutes |
| INDmoney calls start failing with 401 | A token was generated elsewhere (website or another machine), which ended ours | Handled automatically: the client logs in again (at most once a minute) |
| `HTTP 429` from INDstocks | Over 5 data requests a second | The client already paces and backs off; just rerun |
| `not an NSE equity in the INDstocks instrument master` | Symbol typo, or not an `EQ` series stock | Use the plain NSE symbol, e.g. `RELIANCE` |

---

## Cleanup

```bash
rm -rf example_sma/out orb/out core/cache logs
```

To remove the two virtual environments and the downloaded model weights:

```bash
rm -rf C:/lv C:/tdv ~/.cache/huggingface/hub/models--convaiinnovations--laya
```
