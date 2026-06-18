#!/usr/bin/env python3
"""Run the Premarket Trader bot.

SAFE BY DEFAULT — paper mode places no real orders:
    python scripts/run_bot.py                      # paper: print the next session's plan

LIVE Robinhood mode is HARD-GATED behind multiple explicit confirmations and is
dry-run unless you also pass --execute:
    set RH_USERNAME=... & set RH_PASSWORD=...
    python scripts/run_bot.py --mode live --approve --confirm "I APPROVE LIVE TRADING"
    #   ...add --execute to actually send orders (real money).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from premarket_trader.config import Params
from premarket_trader.bot.broker import PaperBroker, RobinhoodBroker
from premarket_trader.bot.risk import RiskManager
from premarket_trader.bot.engine import BotEngine


def print_plans(plans):
    if not plans:
        print("No qualifying setups for the next session.")
        return
    print("\nPlanned trades (next session):")
    for pl in plans:
        notional = pl.shares * pl.entry
        proba = f"{pl.proba:.2f}" if pl.proba == pl.proba else " - "
        print(f"  BUY {pl.shares:>6} {pl.symbol:6s} @ ~${pl.entry:.2f}  "
              f"stop ${pl.stop:.2f}  target ${pl.target:.2f}  "
              f"(${notional:,.0f}, p(win) {proba})")
    print("\nManagement: 2:1 bracket, exit by close, breakout-or-bailout.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["paper", "live"], default="paper")
    ap.add_argument("--config")
    ap.add_argument("--approve", action="store_true", help="(live) operator approval")
    ap.add_argument("--confirm", default="",
                    help='(live) must equal exactly: I APPROVE LIVE TRADING')
    ap.add_argument("--execute", action="store_true",
                    help="(live) actually send orders; omit for a dry run")
    ap.add_argument("--max-order-usd", type=float, default=300.0)
    args = ap.parse_args()

    p = Params.from_yaml(args.config) if args.config else Params()

    if args.mode == "paper":
        broker = PaperBroker(p.account, p.slippage_bps, p.commission_per_share)
        engine = BotEngine(p, broker, RiskManager(p))
        print(f"[PAPER]  starting equity ${broker.equity():,.0f}  —  no real orders placed.")
        print_plans(engine.plan_today())
        return

    # ---- live: refuse unless every gate is satisfied ----
    print("=" * 66)
    print(" LIVE ROBINHOOD MODE — REAL MONEY AT RISK")
    print("=" * 66)
    if not (args.approve and args.confirm == RobinhoodBroker.CONFIRM_PHRASE):
        print("Refused. Live trading requires BOTH:")
        print('  --approve   and   --confirm "I APPROVE LIVE TRADING"')
        print("Plus RH_USERNAME / RH_PASSWORD in the environment. Aborting.")
        return

    broker = RobinhoodBroker(approved=True, confirm_phrase=args.confirm,
                             max_order_usd=args.max_order_usd, dry_run=not args.execute,
                             dry_run_equity=p.account)
    engine = BotEngine(p, broker, RiskManager(p))
    mode = "EXECUTE (sending real orders)" if args.execute else "DRY-RUN (no orders sent)"
    print(f"[LIVE — {mode}]  per-order cap ${args.max_order_usd:.0f}")
    plans = engine.plan_today()
    print_plans(plans)
    if not plans:
        return

    if args.execute:  # final, interactive, order-time confirmation
        total = sum(pl.shares * pl.entry for pl in plans)
        print(f"\nAbout to send {len(plans)} REAL order(s), total notional ~${total:,.0f}.")
        typed = input(f'Type exactly "{RobinhoodBroker.CONFIRM_PHRASE}" to send, '
                      "anything else to abort: ")
        if typed != RobinhoodBroker.CONFIRM_PHRASE:
            print("Aborted at order-time confirmation. No orders sent.")
            return

    for pl in plans:
        try:                                    # one bad order must not abort the batch
            fill = broker.execute_day_trade(pl, None)
            print(f"  -> {fill.reason}: {pl.symbol} x{pl.shares}")
        except RuntimeError as e:
            print(f"  -> SKIPPED {pl.symbol}: {e}")


if __name__ == "__main__":
    main()
