"""Bot safety + risk tests: live-gate, sizing caps, kill switch, paper outcomes."""
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from premarket_trader.config import Params
from premarket_trader.bot.broker import PaperBroker, RobinhoodBroker, TradePlan
from premarket_trader.bot.risk import RiskManager


def test_live_broker_is_blocked_without_approval():
    for kwargs in ({}, {"approved": True}, {"approved": True, "confirm_phrase": "nope"},
                   {"approved": False, "confirm_phrase": "I APPROVE LIVE TRADING"}):
        blocked = False
        try:
            RobinhoodBroker(**kwargs)
        except PermissionError:
            blocked = True
        assert blocked, f"live broker should be blocked for {kwargs}"


def test_paper_broker_target_and_stop():
    p = Params(stop_pct=0.10, profit_ratio=2.0)  # stop 9.0, target 12.0 on a $10 entry
    b = PaperBroker(10_000, slippage_bps=0, commission_per_share=0)
    f = b.execute_day_trade(TradePlan("AAA", 100, 10.0, 9.0, 12.0),
                            {"open": 10, "high": 12.5, "low": 9.5, "close": 11, "date": None})
    assert f.reason == "target" and abs(f.pnl - 200) < 1e-6
    assert abs(b.equity() - 10_200) < 1e-6
    # both stop and target inside range -> pessimistic stop
    f2 = b.execute_day_trade(TradePlan("BBB", 100, 10.0, 9.0, 12.0),
                             {"open": 10, "high": 12.5, "low": 8.5, "close": 11, "date": None})
    assert f2.reason == "stop" and abs(f2.pnl + 100) < 1e-6


def test_daily_max_loss_kill_switch():
    rm = RiskManager(Params(max_positions=10), daily_max_loss_frac=0.05)
    st = rm.new_day(10_000)
    assert rm.can_trade(st)
    rm.record(st, -600)            # -6% exceeds the -5% daily limit
    assert not rm.can_trade(st) and st.halted


def test_sizing_respects_caps_and_no_leverage():
    rm = RiskManager(Params(risk_per_trade=0.01, max_pos_frac=0.5))
    # risk-based: 1% of 10k / $1 risk = 100 shares (below the concentration + cash caps)
    assert rm.size(10_000, 10.0, 9.0, 10_000) == 100
    # constrained by available buying power (no leverage): $500 / $10 = 50
    assert rm.size(10_000, 10.0, 9.0, 500) == 50
    # zero/negative risk per share -> no position
    assert rm.size(10_000, 10.0, 10.0, 10_000) == 0


def test_planned_risk_budget_caps_daily_loss():
    # daily budget = 2% of 10k = $200; each trade risks $1/share, sized to 100 shares
    # ($100 risk), so at most 2 trades fit before the budget is exhausted.
    rm = RiskManager(Params(risk_per_trade=0.01, max_pos_frac=0.5, max_positions=10),
                     daily_max_loss_frac=0.02)
    cands = pd.DataFrame({
        "ticker": [f"T{i}" for i in range(5)],
        "entry": [10.0] * 5, "stop_px": [9.0] * 5, "target_px": [12.0] * 5,
        "ml_proba": [0.6] * 5,
    })
    st = rm.new_day(10_000)
    plans = rm.build_plans(cands, 10_000, st)
    planned_risk = sum(pl.shares * (pl.entry - pl.stop) for pl in plans)
    assert planned_risk <= 0.02 * 10_000 + 1e-6
    assert len(plans) == 2


def test_live_dry_run_is_side_effect_free_and_capped():
    os.environ["RH_USERNAME"], os.environ["RH_PASSWORD"] = "x", "y"
    try:
        # dry-run equity returns the configured value WITHOUT importing/contacting robin_stocks
        b = RobinhoodBroker(approved=True, confirm_phrase="I APPROVE LIVE TRADING",
                            dry_run=True, dry_run_equity=12_345.0)
        assert b.equity() == 12_345.0
        f = b.execute_day_trade(TradePlan("AAA", 10, 5.0, 4.5, 6.0), None)
        assert f.reason == "dry_run" and f.pnl == 0.0
        # hard ceiling on per-order notional cannot be lifted past HARD_MAX_ORDER_USD
        blocked = False
        try:
            RobinhoodBroker(approved=True, confirm_phrase="I APPROVE LIVE TRADING",
                            max_order_usd=999_999.0, dry_run=True)
        except ValueError:
            blocked = True
        assert blocked
    finally:
        os.environ.pop("RH_USERNAME", None)
        os.environ.pop("RH_PASSWORD", None)


if __name__ == "__main__":
    test_live_broker_is_blocked_without_approval()
    test_paper_broker_target_and_stop()
    test_daily_max_loss_kill_switch()
    test_sizing_respects_caps_and_no_leverage()
    test_planned_risk_budget_caps_daily_loss()
    test_live_dry_run_is_side_effect_free_and_capped()
    print("All bot tests passed.")
