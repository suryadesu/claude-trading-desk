# Testing the Laya setup

How to check that the trading desk works with the open-source
[Laya](https://huggingface.co/convaiinnovations/laya) decision model, from a
first smoke test through to the full Docker stack.

Each level depends on the one before it. Levels 1 to 3 need no API keys at all.

| Level | What it proves | Needs |
| --- | --- | --- |
| 1. Model server | Laya runs and answers requests | Python 3.10+ |
| 2. Keyless backtest | The `laya` arm works end to end on real market data | Level 1 |
| 3. Calibration and replay | Threshold fitting works, and results are reproducible | Level 2 |
| 4. ORB and Docker | The real strategy and the deployed stack | A free Alpaca paper key, Docker |

All commands run from the template folder:

```bash
cd plugins/trading-desk/template
```

They use Git Bash syntax with explicit paths into each virtual environment,
so nothing needs activating. On macOS or Linux, use `bin/` instead of
`Scripts/` in the venv paths.

---

## Level 1: Start the Laya model server

Laya runs as its own server, separate from the bots. It needs Python 3.10 or
newer and torch, and the bots need neither.

**Use a short path for this virtual environment on Windows.** Torch has
deeply nested files, and a long path such as one under `AppData\Local\Temp`
fails with `WinError 206: The filename or extension is too long`.

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
`~/.cache/huggingface`, which takes about 30 seconds. Leave this terminal
running.

**Check it is ready** from a second terminal:

```bash
curl -s localhost:8000/health
```

Expected: `{"status":"ok","loaded":["typed-decisions"],...,"device":"cpu",...}`.

On startup, the server may warn that the checkpoint ships invalid temperatures
and that confidence is uncalibrated. That is expected, and it is why Level 3
exists.

### Settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL_URL` | `http://localhost:8000/v1/systemone` | Where the bots send requests |
| `MODEL_NAME` | `typed-decisions` | Laya checkpoint: `typed-decisions` (1024 tokens), `english` (512), `multilingual` (1024) |
| `MODEL_MAX_LEN` | unset | Raise the model's token limit if descriptions get cut |
| `LAYA_API_KEY` | unset | Only needed if you started the server with one |

---

## Level 2: Run a backtest with no API keys

This uses the `example_sma` strategy, a moving-average crossover on SPY and QQQ
using free Yahoo data. It is a harness test, not a strategy to trade.

In a second terminal:

```bash
python -m venv C:/tdv
```
```bash
C:/tdv/Scripts/python -m pip install pandas numpy scipy yfinance
```
```bash
C:/tdv/Scripts/python example_sma/run_backtest.py --arms rules gated laya --split 2022-01-01
```

**Expected output:**

- `=== signals ===` shows about 110 crossovers across SPY and QQQ.
- A line like
  `[laya] prefetch done: 110 decisions in 2.1 min (0.9/s), 0 errors, 0 truncated`.
- Three result rows (`rules`, `gated`, `laya`), each with `train` and `test`
  halves.

**What to check:**

- `0 errors`. Otherwise the server isn't reachable at `MODEL_URL`.
- `0 truncated`. Otherwise candidate descriptions are longer than the model
  can read, so it decides on less than the gates see. Shorten the strategy's
  `context_lines`, or set `MODEL_MAX_LEN`.
- At the default threshold the `laya` row takes almost every trade. That is
  expected before calibration: zero-shot, the model's probabilities sit around
  0.3 to 0.4 for every option.

**Speed:** on CPU, expect about 6 seconds per decision on its own, or about
1 second each when batched. Backtests batch automatically.

**Output files** go in `example_sma/out/`:

- `decisions_laya.jsonl` holds every question and answer, including
  `raw_probabilities`.
- `trades_<arm>.csv` holds each arm's trades.
- `summary.json` holds the headline numbers.

Model answers are cached in `core/cache/laya_cache.jsonl`.

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
   trades at all. If it is zero or negative, stop here: no threshold will help.
2. **`fitted temperature T = ...`** is the adjustment to the model's
   probabilities. If it reports `T hit the edge of the grid`, the probabilities
   barely relate to outcomes.
3. **The reliability tables** show whether, after tempering, win rates rise
   with the model's probability.
4. **The threshold sweep** shows how many trades each threshold keeps, and
   their mean R and t-stat. Pick a threshold here.

A run from when this was set up, for reference:

```
spearman(p_side, R) = +0.362  (t = +3.01, n = 62): ranks trades in-sample; ...
fitted temperature T = 0.561  (sharpens the model)
```

### 3b. Read the out-of-sample half once

Use the T from 3a and the threshold you chose:

```bash
C:/tdv/Scripts/python example_sma/run_backtest.py --arms rules gated laya --split 2022-01-01 --model-temperature 0.561 --model-threshold 0.45
```

Compare the `test` row of `laya` against `gated`, not against `rules`. Beating
`rules` only shows that filtering helps. Beating `gated` shows the model is
better than three hand-written if-statements.

That reference run gave these `test` results:

| Arm | Trades | R per trade |
| --- | --- | --- |
| rules | 42 | +0.034 |
| gated | 13 | +0.014 |
| laya | 18 | +0.391 |

That is 18 trades on a toy strategy: encouraging, not evidence.

### 3c. Check that results are reproducible

Stop the Laya server (Ctrl+C in its terminal), then:

```bash
C:/tdv/Scripts/python example_sma/run_backtest.py --arms laya --split 2022-01-01 --model-temperature 0.561 --model-threshold 0.45 --model-offline
```

**Expected:** the same trades as 3b, answered entirely from the cache. Yahoo
re-adjusts historical prices between downloads, so stop prices can differ in
the 4th decimal. The trades taken, their sides and the model probabilities
should not differ.

---

## Level 4: The ORB strategy and the full stack

### 4a. ORB backtest

ORB uses Alpaca's free market data, so it needs a free Alpaca paper-trading
key.

1. Create `.env` in the template folder from `deploy/.env.example`, and fill in
   `ALPACA_PAPER_KEY` and `ALPACA_PAPER_SECRET`. Never commit `.env`; it is in
   `.gitignore`.
2. Install the Alpaca client and run the backtest, with the Laya server
   running again:

```bash
C:/tdv/Scripts/python -m pip install alpaca-py
```
```bash
C:/tdv/Scripts/python orb/run_backtest.py --arms rules gated laya --start 2023-01-01 --end 2026-09-01
```

3. Calibrate on the in-sample half, then read the test half once:

```bash
C:/tdv/Scripts/python core/calibrate.py --decisions orb/out/decisions_laya.jsonl --trades orb/out/trades_rules.csv --before 2025-03-01 --tolerance 1D
```
```bash
C:/tdv/Scripts/python orb/run_backtest.py --arms rules gated laya --start 2025-03-01 --end 2026-09-01 --model-temperature <T> --model-threshold <threshold>
```

Useful flags:

| Flag | Default | Use |
| --- | --- | --- |
| `--max-model-calls` | 20000 | Refuses runs with more candidates than this |
| `--prefetch-batch` | 16 | Candidates per batch request (max 64) |
| `--prefetch-workers` | 1 | Parallel requests; more doesn't help a single CPU server |
| `--model-fallback` | `wait` | What to do if the model can't answer: `wait` or `rule` |
| `--arms ... jev` | | Runs hosted TypeSafe Jev as well, if `TYPESAFE_API_KEY` is set |

A multi-year ORB run can have thousands of candidates, which takes hours on
CPU. Narrow the date range or symbols for a first run.

### 4b. Live paper trading, dry run

`--dry-run` places no orders:

```bash
C:/tdv/Scripts/python orb/live.py --once --dry-run --decider laya --model-threshold <threshold> --model-temperature <T>
```

Decisions are written to `orb/out/live_laya_stream.jsonl`.

### 4c. Full stack in Docker

This starts the Laya service, the bots, the dashboard and the keepalive. The
Laya service has no public port: only the bots can reach it.

1. Create `deploy/.env` from `deploy/.env.example`, and set the Alpaca keys,
   plus `ORB_MODEL_THRESHOLD` and `ORB_MODEL_TEMPERATURE` from your
   calibration.
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

- `laya` turns `healthy` once its weights download.
- `orb` waits for it, then logs `decider=laya`.
- The dashboard at http://localhost:8080 shows the ORB card tagged
  "laya gates trades".

Stop everything with `docker compose down`.

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `WinError 206 ... filename too long` during install | Venv path too deep for torch on Windows | Use a short path such as `C:/lv` |
| `[laya] giving up after 5 tries: URLError` | Server not running or wrong URL | Start `laya-serve`; check `curl localhost:8000/health` and `MODEL_URL` |
| `[laya] HTTP 401` | Server has `LAYA_API_KEY` set | Set the same `LAYA_API_KEY` for the backtest |
| `[laya] WARNING: state truncated ...` | Description longer than the model's limit | Shorten `context_lines`, or set `MODEL_MAX_LEN` |
| `laya` arm takes every trade | Uncalibrated threshold | Run Level 3 |
| `laya` arm takes no trades | Threshold too high, or `entry_criteria` not keyed by `enter_long` / `enter_short` / `wait` | Lower the threshold; fix the strategy's criteria keys |
| `no_answer` notes in `--model-offline` mode | Decision not in cache: the description, model name or `MODEL_MAX_LEN` changed | Run once with the server up |
| Different results after changing `MODEL_NAME` or upgrading `laya` | It is a different model | Recalibrate; pin `LAYA_VERSION` in `deploy/.env` once settled |

---

## Cleanup

```bash
rm -rf example_sma/out orb/out core/cache
```

To remove the two virtual environments and the downloaded model weights:

```bash
rm -rf C:/lv C:/tdv ~/.cache/huggingface/hub/models--convaiinnovations--laya
```
