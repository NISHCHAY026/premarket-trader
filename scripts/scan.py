#!/usr/bin/env python3
"""Run the premarket scanner and print today's ranked gap-and-go watchlist.

Example:
    python scripts/scan.py
    python scripts/scan.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from premarket_trader.config import Params
from premarket_trader import scanner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--universe-csv")
    args = ap.parse_args()

    p = Params.from_yaml(args.config) if args.config else Params()
    if args.universe_csv:
        p.universe_csv = args.universe_csv

    wl = scanner.scan(p)
    if wl is None or wl.empty:
        print("No qualifying setups in the latest session.")
        return
    print("\nPremarket watchlist (ranked):\n")
    for _, r in wl.iterrows():
        proba = f"{r['ml_proba']:.2f}" if r["ml_proba"] == r["ml_proba"] else "  - "
        print(f"  {r['ticker']:6s}  entry ${r['entry']:.2f}  stop ${r['stop']:.2f}  "
              f"target ${r['target']:.2f}  gap {r['gap']*100:5.1f}%  "
              f"rvol {r['rvol']:.1f}x  p(win) {proba}")
    print("\nTrade management: 2:1 profit-to-loss, exit by close, breakout-or-bailout.")


if __name__ == "__main__":
    main()
