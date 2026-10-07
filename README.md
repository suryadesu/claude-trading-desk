# Trading Desk

A Claude Code plugin for building algorithmic trading bots, testing them honestly,
paper trading them, and hosting them 24/7 for nothing.

Everything in this stack has a free tier. There is no paid step anywhere.

> **This branch (`laya-angel`) trades Indian stocks on NSE.** Data comes from
> Angel One's free SmartAPI, the decision model is the open-source
> [Laya](https://huggingface.co/convaiinnovations/laya), and the live bot books
> **simulated** fills by default. Angel One has no paper-trading sandbox, so real
> orders happen only with `--real-money` plus an explicit environment
> confirmation. See [run.md](run.md) for setup and testing.

## Install

```
/plugin marketplace add aabrole/claude-trading-desk
/plugin install trading-desk@trading-desk
```

Then, in the directory you want the workspace:

```
/trading-desk-init
```

## What you get

**Six skills** Claude loads when the work calls for them:

| Skill | What it carries |
| --- | --- |
| `trading-desk-method` | The testing discipline, and ten named bugs that each made results look better than they were |
| `trading-desk-strategy` | The strategy contract, and the causality rules that keep a backtest honest |
| `trading-desk-laya` | Wiring the open-source Laya decision model as a filter, calibrating it, and how to tell whether it added anything |
| `trading-desk-paper` | Going live on NSE: simulated fills by default, replay validation, kill switches, Angel One real-money safeguards |
| `trading-desk-dashboard` | The live desk, server-sent events, animated backtest curves |
| `trading-desk-oracle` | Free 24/7 hosting, capacity retries, and the two firewalls that catch everyone |

**Three commands:** `/trading-desk-init`, `/trading-desk-new-strategy`,
`/trading-desk-deploy`.

**A working harness** in `template/`: a portfolio backtest engine with real
slippage and fees, a three-arm decision layer, a live dashboard, and a scripted
deploy to a free VM.

## The part that makes it worth using

Most trading repos help you find a strategy that works. This one is built to tell
you when one **does not**, because that is the far more common answer and the
expensive thing to get wrong.

Every strategy runs three ways over an identical candidate set:

| Arm | What it is |
| --- | --- |
| `rules` | Take every signal. The strategy as advertised. |
| `gated` | Hand-written if-statements filter the signals. **The control.** |
| `laya` | A decision model ([Laya](https://huggingface.co/convaiinnovations/laya), open source, self-hosted) filters the same signals. |

The control arm is the one people skip, and skipping it is what makes "AI improved
my strategy" impossible to disprove. If the hand-written gates match the model,
you did not need the model, and that is a real finding worth reporting.

The kit also insists on out-of-sample splits, a t-stat on mean return per trade,
gross reported next to net, and a mechanical audit of every trade against the
rules as written.

## Start here

```bash
# no API key at all: proves the harness works end to end (NSE ETFs via Yahoo)
python3 example_sma/run_backtest.py

# the worked example: opening range breakout on 8 NIFTY 50 stocks
# (needs a free Angel One SmartAPI key: see run.md)
python3 orb/run_backtest.py --arms rules gated

# the live bot, simulated fills on live NSE prices
python3 orb/live.py --once --dry-run

# add the model arm: Laya, open source, served locally (Python >= 3.10, own venv is fine)
pip install "laya[serve]" && laya-serve          # http://localhost:8000/v1/systemone
python3 orb/run_backtest.py --arms rules gated laya

# the live desk
python3 dashboard/server.py            # http://localhost:8080
```

`example_sma` is a moving-average crossover. It is a template for wiring the
pipeline, not an edge, and the harness will tell you so.

## What each piece costs

| Thing | Tier | Cost |
| --- | --- | --- |
| Angel One SmartAPI: NSE candles, live data, order API | free with an Angel One account | ₹0 |
| Simulated fills (the default) | local | ₹0 |
| Real orders (only with `--real-money`) | Angel brokerage + statutory charges | ₹20 or 0.1% per order, plus STT, exchange, SEBI, stamp duty and GST |
| Oracle Cloud Always Free VM (2 ARM cores, 12 GB) | free forever | ₹0 |
| Yahoo data via yfinance (daily example only) | free | ₹0 |
| Laya decision model (Apache-2.0, runs on the same VM) | open source | ₹0 |

## Layout

```
core/         engine, decision layer, metrics, state files. No strategy logic.
orb/          worked example: opening range breakout
example_sma/  keyless smoke test: moving-average crossover
dashboard/    server.py + index.html. Standard library and one HTML file.
deploy/       Dockerfile, compose, and the Oracle provisioning scripts
```

Strategies live one level under the root because each resolves `core/` as
`../core`. Moving them deeper breaks those imports.

## Read this part

This is research tooling, not advice, and not a product that makes money.

- Backtested and paper results are hypothetical. Paper fills are simulated, and a
  paper fill at the midpoint is not evidence you would have been filled there.
- Most strategies do not work. The kit is designed to show you that clearly, and
  the correct response to a t-stat under 2 is to stop, not to tune.
- Nothing here is financial advice. You are responsible for anything you deploy
  with real money, and the default configuration trades paper accounts only.
- The example strategies are examples. Do not trade them.

## Licence

MIT. See `LICENSE`.
