#!/usr/bin/env python3
"""Train + walk-forward backtest the Premarket Trader strategy and save a report.

Example:
    python scripts/run_walkforward.py --start 2017-01-01
    python scripts/run_walkforward.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from premarket_trader.config import Params
from premarket_trader.data import load_universe, download_prices
from premarket_trader import walkforward, report
from premarket_trader import metrics as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="YAML params file")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--universe-csv")
    ap.add_argument("--no-optimize", action="store_true")
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the rules-only (no-ML) comparison")
    ap.add_argument("--no-risk", action="store_true",
                    help="disable the v3 risk layer (regime sizing + circuit breakers)")
    ap.add_argument("--name", default="walkforward")
    args = ap.parse_args()

    p = Params.from_yaml(args.config) if args.config else Params()
    for k in ("start", "end", "universe_csv"):
        v = getattr(args, k if k != "universe_csv" else "universe_csv")
        if v:
            setattr(p, k, v)
    if args.no_risk:                                   # v2 ablation: turn the risk layer off
        p.regime_sizing, p.icebreaker, p.three_strike = False, False, 0
        p.daily_max_loss = 10.0                        # effectively disables the loss/goal breaker

    tickers = load_universe(p.universe_csv)
    print(f"Downloading {len(tickers)} tickers from {p.start} ...")
    prices = download_prices(tickers, p.start, p.end)
    print(f"  price rows: {len(prices):,}  tickers with data: {prices['ticker'].nunique() if len(prices) else 0}")

    print("\n=== Rules + ML (walk-forward) ===")
    wf = walkforward.run(prices, p, optimize=not args.no_optimize)
    if "error" in wf:
        print("ERROR:", wf["error"])
        return
    bp = wf["best_params"]
    print("Selected params:",
          {k: getattr(bp, k) for k in ("gap_min", "rvol_min", "ml_min_proba", "max_positions")})
    print("  TRAIN:", M.summary_line(wf["metrics_train"]))
    print("  TEST :", M.summary_line(wf["metrics_test"]), "   <-- out-of-sample headline")
    print("  FULL :", M.summary_line(wf["metrics_full"]))
    out = report.save_report(wf, name=args.name)
    print(f"  saved report -> {out}")

    if not args.no_baseline:
        print("\n=== Rules only (no ML) ===")
        wf2 = walkforward.run(prices, dataclasses.replace(p, use_ml=False),
                              optimize=not args.no_optimize)
        if "error" not in wf2:
            print("  TEST :", M.summary_line(wf2["metrics_test"]), "   <-- out-of-sample")
            report.save_report(wf2, name=args.name + "_rules_only")


if __name__ == "__main__":
    main()
