---
name: trading-desk-laya
description: Wire the open-source Laya decision model (System One, self-hosted via laya-serve) into the strategy harness as a filter, including context limits, calibrating its temperature and threshold in-sample, caching, and how to test whether it added anything. Use when adding or evaluating model-based trade filtering.
version: 2.0.0
tags: [trading, laya, ai-decisions, calibration]
allowed-tools: Read, Grep, Glob, Bash, Write, Edit
---

# Model-filtered trades

The model does not hunt for trades. The strategy proposes every entry and the
model answers "take it or stand aside". That is the only arrangement where the
A/B against the control arm means anything, because both arms then see an
identical candidate set.

## The model

[Laya](https://huggingface.co/convaiinnovations/laya) (Apache-2.0) is a
non-generative decision model: a ModernBERT encoder with a decision head that
scores each option and returns a probability distribution in one forward pass.
It never writes text, so there is nothing to parse and nothing to hallucinate.

Checkpoints, chosen with the request's `model` field (`MODEL_NAME`):

| name | context | use |
| --- | --- | --- |
| `typed-decisions` | 1024 tokens | **default.** Fine-tuned on typed decisions |
| `english` | 512 tokens | base checkpoint, near chance zero-shot on typed decisions |
| `multilingual` | 1024 (up to 8192 with `max_len`) | non-English state, long state |

## Running it

Laya needs Python >= 3.10 and torch, so it runs as a sidecar and the bots talk
to it over HTTP with stdlib `urllib`:

```bash
pip install "laya[serve]"           # own venv is fine
LAYA_PRELOAD=1 LAYA_MODELS=typed-decisions laya-serve     # :8000
curl -s localhost:8000/health
```

`laya-serve` speaks the `/v1/systemone` wire protocol (originally TypeSafe
Jev's), so `ModelDecider` in `core/decision.py` is the same client either way:

```
POST /v1/systemone
{ "model": "typed-decisions", "state": "<the situation in words>", "questions": {...} }
POST /v1/systemone/batch
{ "model": ..., "states": ["...", ...up to 64], "questions": {...} }  ->  {"results": [...]}
```

Question types: `choice` (a `criteria` map), `score` (a `criteria` list) and
`noul` (a single yes-likelihood). Each answer gives `.choice`, `.confidence` and
`.probabilities`.

Env: `MODEL_URL` (default `http://localhost:8000/v1/systemone`), `MODEL_NAME`,
`MODEL_MAX_LEN` (optional), and `LAYA_API_KEY` (only if the server sets one).
In `deploy/` the `laya` service runs on the compose network with **no published
port**. Keep it that way: the server has no auth unless `LAYA_API_KEY` is set.

The `jev` arm still exists and uses the same client against hosted Jev
(`JEV_URL`, `JEV_MODEL`, `TYPESAFE_API_KEY`). `--arms gated laya jev` compares
the two models directly.

## Context length is a silent failure

A state longer than the checkpoint's window is cut, and then the model decides
on less than the gates saw. That breaks the fairness of the A/B without any
error. Laya reports `usage.truncated`; `ModelDecider` counts it in `stats()`,
logs `truncated` per decision, and prints a warning the first time it happens.
If it fires, shorten `context_lines` or set `MODEL_MAX_LEN`. Do not ignore it.

## Calibrate before you threshold

Laya's own card says its checkpoints are over-confident until fitted on your
data, and a threshold tuned for another model means nothing here. Fit both
numbers on the **in-sample half only**:

```bash
python3 orb/run_backtest.py --arms rules laya --start 2023-01-01 --end 2026-09-01
python3 core/calibrate.py --decisions orb/out/decisions_laya.jsonl \
                          --trades orb/out/trades_rules.csv --before 2025-03-01
python3 orb/run_backtest.py --arms rules gated laya \
        --model-temperature <T> --model-threshold <t>        # then read OOS once
```

`calibrate.py` joins every decision to the outcome the `rules` arm got on the
same candidate, so the model is judged on the full candidate set and not only
on the trades it approved. It fits a temperature T (p_i^(1/T), renormalised),
prints reliability bins before and after, the Spearman rank correlation of the
probability against R, and a threshold sweep. If the rank correlation is near
zero, no threshold will help. Stop there.

Tempering does not change the argmax, only how confident the probability on it
is. The decision log keeps `raw_probabilities`, and the cache stores raw
responses, so refitting T never needs a new model call.

## Threshold on the probability, not the confidence

Gate on `probabilities[chosen_action]`, not the `confidence` field. Measure which
of the two ranks your outcomes before trusting either. With Jev, in one measured
case, `confidence` correlated with outcome in the **wrong direction** while the
probability ranked correctly.

A high threshold looks disciplined and mostly removes sample size. Sweep it and
look at the whole curve.

## Entry criteria keys are Action values

`entry_criteria` must be keyed `enter_long` / `enter_short` / `wait`. The decider
reads the probability of the proposed side by that key, so a strategy that
names its options `take_it` / `stand_aside` vetoes every trade without any error.

## Never let the model flip the side

The strategy decided the direction. The model can only approve or stand aside.
A model that can reverse a trade is a different strategy with an unmeasurable
candidate set.

## Record extra questions, act on none of them

Ask for a risk score alongside the decision. One call answers all questions in
parallel, so it is free, and afterwards you can check whether the stated risk
actually predicted outcomes. Never let those answers affect a trade.

## Cache and prefetch

The cache key is a hash of the provider (`laya` / `jev`) plus the exact
request, including the checkpoint name. The URL is not part of it, so a cache
built against `localhost:8000` also serves the bot that reaches `laya:8000`
inside docker. A re-run costs nothing, Jev and Laya answers never mix, and
every decision can be audited later. `--model-offline` replays from cache with the server stopped.
Changing `MODEL_NAME` or upgrading `laya` changes the model, so recalibrate.
Pin `LAYA_VERSION` once you have a calibration you trust.

Prefetch groups states into `/v1/systemone/batch` requests of 16 (max 64) and
caches each answer under its single-request key, so `decide()` is unchanged.
This is legitimate **only when each question is independent**. If your strategy
can hold a position across candidates, the snapshot depends on earlier outcomes,
and prefetching silently feeds the model the wrong state. Verify independence
before batching.

Retries catch `urllib.error.HTTPError` (5xx and 429 retry; `laya-serve` answers
503 when it is at `LAYA_MAX_CONCURRENT`) and then
`(OSError, http.client.HTTPException, json.JSONDecodeError)`. A bare
`RemoteDisconnected` escaping the retry loop would kill a long run near the end.

## How to tell whether it helped

Compare the model arm to the **gated** arm, not to the unfiltered arm. Beating
the unfiltered arm only shows that filtering helps. Then:

- t-test the difference in mean R between the two arms
- check the model arm's ranking power over the **full** candidate set, not just
  the accepted band, or restriction of range will hide it
- report the stand-aside rate: a model that approves 97% is not filtering

A likely finding, worth stating plainly: the model is a **combiner, not a
source**. It can weigh evidence you already computed, but it does not create an
edge your features don't contain. Across six strategies in the project this kit
came from, a hosted model never beat hand-written rules by a statistically
significant margin. Publish that kind of result rather than hiding it.

## Cost

$0 in money, paid in time. Measured on a laptop CPU with ORB's prompt (one
decision plus two recorded questions, about 1,100 tokens): **about 6 s per
decision on its own, about 4 s each when batched.** Expect similar or slower on
2 Ampere cores. That is fine live, where a few candidates arrive per 5-minute
bar. A multi-year backtest with a few thousand candidates is hours on CPU, so
run the prefetch overnight or on a GPU box (Laya's card quotes ~33 ms per
question on a T4), then ship `core/cache/laya_cache.jsonl`. Every extra question
costs another forward pass, so drop record-only questions you don't analyse.

## What zero-shot looked like

On hand-written ORB states, `typed-decisions` returned near-flat distributions
(every option between 0.27 and 0.41) and sometimes preferred the opposite side.
Untuned, it hardly discriminates on this domain. Do not read anything into
its probabilities before `calibrate.py` shows a non-zero rank correlation
in-sample. If it never does, the honest options are to fine-tune Laya on
in-sample candidates (as a new arm, compared against `gated` again) or to report
that the model added nothing.
