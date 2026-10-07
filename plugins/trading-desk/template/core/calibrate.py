#!/usr/bin/env python3
"""
calibrate.py
============
Fit the model arm's temperature and threshold on the IN-SAMPLE half only.

Laya's checkpoints are over-confident out of the box (its own model card says
so), and a threshold tuned for Jev means nothing for Laya. This script reads the
decision log a model arm wrote, joins every decision to the outcome the `rules`
arm got on the same candidate, and fits one temperature T so that the tempered
probability of the proposed side behaves like P(trade wins).

  python3 orb/run_backtest.py --arms rules laya --start 2023-01-01 --end 2026-09-01
  python3 core/calibrate.py --decisions orb/out/decisions_laya.jsonl \\
                            --trades orb/out/trades_rules.csv --before 2025-03-01

Then pass --model-temperature and --model-threshold to the runner and look at
the out-of-sample half ONCE. Fitting either number on data after --before turns
the test half into a second training half, and the result into a description of
it.

Why the rules arm: it takes every candidate, so its trades are outcomes for the
full candidate set. Judging the model only on trades it approved is restriction
of range, and it hides whether the probability ranks anything at all.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decision import temper   # noqa: E402

GRID = np.exp(np.linspace(math.log(0.25), math.log(8.0), 61))


def load_decisions(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("proposed") not in ("enter_long", "enter_short"):
                continue
            raw = r.get("raw_probabilities") or r.get("probabilities") or {}
            if not raw:
                continue
            rows.append({"symbol": r["symbol"], "ts": pd.Timestamp(r["timestamp"]),
                         "proposed": r["proposed"], "raw": raw,
                         "truncated": bool(r.get("truncated", False))})
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit(f"no entry decisions with probabilities in {path}")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def join_outcomes(dec: pd.DataFrame, trades_path: Path, tolerance: str) -> pd.DataFrame:
    """Match each decision to the rules-arm trade on that symbol entered at or after it."""
    tr = pd.read_csv(trades_path)
    if tr.empty or "R" not in tr:
        sys.exit(f"{trades_path} has no trades with an R column")
    tr["entry_time"] = pd.to_datetime(tr["entry_time"], utc=True)
    tr = tr.sort_values("entry_time")[["symbol", "entry_time", "R"]]
    dec = dec.sort_values("ts")
    out = pd.merge_asof(dec, tr, left_on="ts", right_on="entry_time", by="symbol",
                        direction="forward", tolerance=pd.Timedelta(tolerance))
    missing = int(out["R"].isna().sum())
    if missing:
        print(f"[calibrate] {missing} of {len(out)} decisions have no rules-arm trade within "
              f"{tolerance} (position cap, or a candidate the engine skipped); dropped")
    return out.dropna(subset=["R"]).reset_index(drop=True)


def p_side(df: pd.DataFrame, T: float) -> np.ndarray:
    return np.array([temper(r, T).get(side, 0.0) for r, side in zip(df["raw"], df["proposed"])])


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 5) -> pd.DataFrame:
    q = pd.qcut(p, q=min(bins, len(np.unique(p))), duplicates="drop")
    g = pd.DataFrame({"p": p, "y": y, "bin": q}).groupby("bin", observed=True)
    return pd.DataFrame({"n": g.size(), "mean_p": g["p"].mean().round(3),
                         "win_rate": g["y"].mean().round(3)})


def threshold_sweep(p: np.ndarray, R: np.ndarray, grid: List[float]) -> pd.DataFrame:
    rows = []
    for t in grid:
        sel = R[p >= t]
        n = len(sel)
        mean = float(sel.mean()) if n else float("nan")
        sd = float(sel.std(ddof=1)) if n > 1 else float("nan")
        tstat = mean / (sd / math.sqrt(n)) if n > 1 and sd > 0 else float("nan")
        rows.append({"threshold": round(t, 2), "taken": n,
                     "take_rate": round(n / len(R), 3) if len(R) else 0.0,
                     "mean_R": round(mean, 3), "t_stat": round(tstat, 2)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit model temperature and threshold in-sample.")
    ap.add_argument("--decisions", required=True, type=Path,
                    help="decisions_<arm>.jsonl written by a model arm")
    ap.add_argument("--trades", required=True, type=Path,
                    help="trades_rules.csv from the same run: outcomes for every candidate")
    ap.add_argument("--before", required=True,
                    help="in-sample cutoff. Only decisions before this date are used.")
    ap.add_argument("--tolerance", default="1D",
                    help="max gap from decision to rules-arm entry (1D intraday, 5D daily)")
    args = ap.parse_args()

    dec = load_decisions(args.decisions)
    cut = pd.Timestamp(args.before, tz="UTC")
    n_all = len(dec)
    dec = dec[dec["ts"] < cut]
    print(f"[calibrate] {len(dec)} of {n_all} entry decisions are before {args.before} "
          f"(in-sample); the rest are not looked at")
    if dec["truncated"].any():
        print(f"[calibrate] WARNING: {int(dec['truncated'].sum())} decisions were made on a "
              f"truncated state. Fix the context length before trusting any of this.")

    df = join_outcomes(dec, args.trades, args.tolerance)
    if len(df) < 30:
        sys.exit(f"only {len(df)} in-sample decisions with outcomes; too few to fit anything")
    y = (df["R"].to_numpy() > 0).astype(float)
    R = df["R"].to_numpy(dtype=float)

    losses = [log_loss(p_side(df, T), y) for T in GRID]
    T = float(GRID[int(np.argmin(losses))])
    p0, p1 = p_side(df, 1.0), p_side(df, T)

    print(f"\nin-sample decisions with outcomes: {len(df)}   base win rate: {y.mean():.3f}")
    print(f"log loss  raw {log_loss(p0, y):.4f}   tempered {log_loss(p1, y):.4f}")
    print(f"brier     raw {np.mean((p0 - y) ** 2):.4f}   tempered {np.mean((p1 - y) ** 2):.4f}")
    print(f"fitted temperature T = {T:.3f}  ({'softens' if T > 1 else 'sharpens'} the model)")
    if T in (GRID[0], GRID[-1]):
        print("  T hit the edge of the grid: the probability barely relates to outcome.")

    # Ranking power over the FULL candidate set. Tempering is monotone per
    # decision, so it cannot change this; if it is near zero, no threshold helps.
    rho = pd.Series(p0).corr(pd.Series(R), method="spearman")
    n = len(R)
    t_rho = rho * math.sqrt((n - 2) / max(1e-12, 1 - rho ** 2)) if n > 2 else 0.0
    if rho <= 0:
        verdict = "the model does not rank these trades, or ranks them backwards: stop here"
    elif abs(t_rho) < 2:
        verdict = "indistinguishable from no ranking at this sample size"
    else:
        verdict = "ranks trades in-sample; only the out-of-sample half can confirm it"
    print(f"spearman(p_side, R) = {rho:+.3f}  (t = {t_rho:+.2f}, n = {n}): {verdict}")

    print("\nreliability, raw:")
    print(reliability(p0, y).to_string())
    print("\nreliability, tempered:")
    print(reliability(p1, y).to_string())

    grid = sorted(set(np.round(np.quantile(p1, np.linspace(0.0, 0.9, 10)), 2).tolist()))
    print("\nthreshold sweep on tempered p (in-sample):")
    print(threshold_sweep(p1, R, grid).to_string(index=False))
    print(f"\nall candidates: mean R {R.mean():+.3f}. A threshold that does not beat this, "
          f"and the gated arm, is not adding anything.")
    print(f"\nnext: --model-temperature {T:.3f} --model-threshold <chosen above>, then read "
          f"the out-of-sample half once.")


if __name__ == "__main__":
    main()
