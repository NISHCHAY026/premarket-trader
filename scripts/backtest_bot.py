#!/usr/bin/env python3
"""Backtest the BOT (replays the live decision/execution code path over history).

This validates that the exact selection + sizing + execution logic the live bot would
run reproduces the research backtest. Uses the walk-forward-tuned parameters.

    python scripts/backtest_bot.py --start 2018-01-01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from premarket_trader.config import Params
from premarket_trader.data import load_universe, download_prices
from premarket_trader import walkforward, report
from premarket_trader import metrics as M
from premarket_trader.bot.broker import PaperBroker
from premarket_trader.bot.risk import RiskManager
from premarket_trader.bot.engine import BotEngine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--universe-csv")
    ap.add_argument("--daily-max-loss", type=float, default=0.06,
                    help="halt for the day after this fractional loss")
    ap.add_argument("--name", default="bot_backtest")
    args = ap.parse_args()

    p = Params.from_yaml(args.config) if args.config else Params()
    for k in ("start", "end", "universe_csv"):
        if getattr(args, k):
            setattr(p, k, getattr(args, k))

    tickers = load_universe(p.universe_csv)
    print(f"Downloading {len(tickers)} tickers from {p.start} ...")
    prices = download_prices(tickers, p.start, p.end)

    print("Tuning selection (walk-forward) ...")
    wf = walkforward.run(prices, p)
    if "error" in wf:
        print("ERROR:", wf["error"]); return
    bp, ev = wf["best_params"], wf["eval_events"]

    broker = PaperBroker(bp.account, bp.slippage_bps, bp.commission_per_share)
    engine = BotEngine(bp, broker, RiskManager(bp, daily_max_loss_frac=args.daily_max_loss))
    res = engine.backtest(ev)
    m = M.compute(res, bp.account)

    print("\n=== Bot replay vs. research backtest (same OOF period) ===")
    print("  research :", M.summary_line(wf["metrics_full"]))
    print("  bot loop :", M.summary_line(m), "  <-- live code path")
    print(f"  daily-loss halts triggered: {int(res['equity']['halted'].sum())} day(s)")

    report.save_report({"best_params": bp, "metrics_full": m,
                        "metrics_train": {}, "metrics_test": {}, "result_full": res},
                       name=args.name)
    print(f"  saved -> results/{args.name}_*")


if __name__ == "__main__":
    main()
