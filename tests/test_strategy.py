"""Correctness tests: outcome logic, no-lookahead features, backtest sanity."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from premarket_trader.config import Params
from premarket_trader.features import _simulate_outcome, build_events, FEATURE_COLS
from premarket_trader.backtest import run_backtest
from premarket_trader import metrics as M


def test_outcome_cases():
    p = Params(stop_pct=0.10, profit_ratio=2.0)  # stop -10% (9.0), target +20% (12.0)
    df = pd.DataFrame({
        "ticker": ["A"] * 4,
        "date": pd.date_range("2020-01-01", periods=4),
        "open": [10, 10, 10, 10],
        "high": [12.1, 10.5, 12.1, 10.5],
        "low": [9.5, 8.9, 8.9, 9.5],
        "close": [11, 9.5, 11, 10],
    })
    out = _simulate_outcome(df, p)
    assert out["exit_reason"].tolist() == ["target", "stop", "stop", "close"]
    # target row gains +20%, stop row loses -10%
    assert abs(out["trade_ret_gross"].iloc[0] - 0.20) < 1e-9
    assert abs(out["trade_ret_gross"].iloc[1] + 0.10) < 1e-9
    assert out["win"].tolist() == [1, 0, 0, 0]  # close at 10 == entry -> not a win


def _synthetic_prices(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in ["AAA", "BBB"]:
        price = 5.0
        for d in pd.bdate_range("2020-01-01", periods=90):
            ret = rng.normal(0, 0.03)
            o = price * (1 + rng.normal(0, 0.02))
            c = o * (1 + ret)
            hi = max(o, c) * (1 + abs(rng.normal(0, 0.02)))
            lo = min(o, c) * (1 - abs(rng.normal(0, 0.02)))
            vol = rng.integers(500_000, 5_000_000)
            rows.append([d, t, o, hi, lo, c, vol])
            price = c
    return pd.DataFrame(rows, columns=["date", "ticker", "open", "high", "low", "close", "volume"])


def test_no_lookahead_features():
    """Same-day high/low/close must not influence any feature — only the outcome."""
    prices = _synthetic_prices()
    p = Params(event_gap_min=0.0, price_min=0.1, price_max=1e6, min_dollar_vol=0)
    ev1 = build_events(prices, p)
    assert not ev1.empty

    # perturb the LAST day's intraday high/low/close for every ticker
    tampered = prices.copy()
    last = tampered.groupby("ticker")["date"].transform("max") == tampered["date"]
    tampered.loc[last, "high"] *= 1.5
    tampered.loc[last, "low"] *= 0.5
    tampered.loc[last, "close"] *= 1.2
    ev2 = build_events(tampered, p)

    key = ["date", "ticker"]
    a = ev1.set_index(key)[FEATURE_COLS]
    b = ev2.set_index(key).reindex(a.index)[FEATURE_COLS]
    # features identical despite tampered same-day H/L/C
    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)


def test_backtest_runs_and_scores():
    prices = _synthetic_prices(seed=1)
    p = Params(event_gap_min=0.0, gap_min=0.0, rvol_min=0.0, price_min=0.1,
               price_max=1e6, min_dollar_vol=0, use_ml=False, max_positions=3)
    ev = build_events(prices, p)
    res = run_backtest(ev, p)
    m = M.compute(res, p.account)
    assert m["trades"] > 0
    assert "sharpe" in m and "max_drawdown" in m
    # equity is finite and positive-ish (no NaNs)
    assert res["equity"]["equity"].notna().all()


if __name__ == "__main__":
    test_outcome_cases()
    test_no_lookahead_features()
    test_backtest_runs_and_scores()
    print("All tests passed.")
