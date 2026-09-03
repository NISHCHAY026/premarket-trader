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


def _rich_prices(seed=0, n=320, tickers=("AAA", "BBB", "CCC", "DDD", "EEE")):
    """A fixture long and lively enough that no feature is degenerate.

    >=300 bars so the 200-day SMA warms up, and periodic explosive days so the
    move-sequence features (day_of_move, former_runner_count, ...) actually fire.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n)
    rows = []
    for t in tickers:
        price = 8.0
        for d in dates:
            shock = abs(rng.normal(0.18, 0.07)) if rng.random() < 0.08 else 0.0
            o = price * (1 + rng.normal(0, 0.02) + shock)
            c = o * (1 + rng.normal(0, 0.04))
            hi = max(o, c) * (1 + abs(rng.normal(0, 0.03)) + (0.15 if shock else 0.0))
            lo = min(o, c) * (1 - abs(rng.normal(0, 0.03)))
            vol = int(rng.integers(500_000, 5_000_000) * (3 if shock else 1))
            rows.append([d, t, o, hi, lo, c, vol])
            price = max(c, 0.5)
    return pd.DataFrame(rows, columns=["date", "ticker", "open", "high", "low",
                                       "close", "volume"])


def _leak_audit(prices, p, probes=4):
    """Perturb bar T's high/low/close/volume; features dated <= T must not move.

    Returns (offending_feature_names, rows_compared). A feature that changes is
    reading same-day (or future) data and is therefore a lookahead leak.
    """
    ev1 = build_events(prices, p)
    assert not ev1.empty, "fixture produced no events"
    key = ["date", "ticker"]
    all_dates = sorted(prices["date"].unique())
    # probe bars spread across the second half, where history is warm
    picks = [all_dates[int(len(all_dates) * f)] for f in (0.55, 0.7, 0.85, 0.95)][:probes]

    offenders, compared = set(), 0
    for T in picks:
        tam = prices.copy()
        m = tam["date"] == T
        # tamper same-day H/L/C/V while keeping OHLC coherent; `open` is untouched
        tam.loc[m, "close"] = tam.loc[m, "open"] * 1.25
        tam.loc[m, "high"] = tam.loc[m, ["open", "close"]].max(axis=1) * 1.6
        tam.loc[m, "low"] = tam.loc[m, ["open", "close"]].min(axis=1) * 0.6
        tam.loc[m, "volume"] = (tam.loc[m, "volume"] * 4).astype("int64")
        ev2 = build_events(tam, p)

        a = ev1[ev1["date"] <= T].set_index(key)[FEATURE_COLS].sort_index()
        b = ev2[ev2["date"] <= T].set_index(key)[FEATURE_COLS].sort_index()
        # membership itself must not change (a changed pool is also a leak)
        if not a.index.equals(b.index):
            offenders.add("__pool_membership__")
            common = a.index.intersection(b.index)
            a, b = a.loc[common], b.loc[common]
        compared += len(a)
        for col in FEATURE_COLS:
            if not np.allclose(a[col].to_numpy(dtype=float),
                               b[col].to_numpy(dtype=float), equal_nan=True):
                offenders.add(col)
    return offenders, compared


def test_no_lookahead_features():
    """Same-day high/low/close/volume must not influence ANY feature."""
    prices = _rich_prices()
    p = Params(event_gap_min=0.0, price_min=0.1, price_max=1e6, min_dollar_vol=0)

    # the audit must actually compare something, or it proves nothing
    offenders, compared = _leak_audit(prices, p)
    assert compared > 0, "audit compared zero rows - it would pass vacuously"
    assert not offenders, f"lookahead leak in: {sorted(offenders)}"

    # and no feature may be constant, or it is untestable by construction
    ev = build_events(prices, p)
    degenerate = [c for c in FEATURE_COLS if ev[c].nunique(dropna=False) <= 1]
    assert not degenerate, f"degenerate (untestable) features: {degenerate}"


def test_leak_audit_detects_an_injected_leak():
    """Mutation guard: if the audit cannot catch a planted leak, it tests nothing."""
    import premarket_trader.features as fmod
    prices = _rich_prices(seed=3)
    p = Params(event_gap_min=0.0, price_min=0.1, price_max=1e6, min_dollar_vol=0)

    original = fmod._per_ticker
    def leaky(g):
        out = original(g)
        out["vol_trend"] = out["close"] / out["open"] - 1.0      # same-day close
        out["ret_1d"] = out["close"].shift(-1) / out["close"] - 1.0  # FUTURE bar
        return out

    fmod._per_ticker = leaky
    try:
        offenders, _ = _leak_audit(prices, p)
    finally:
        fmod._per_ticker = original
    assert "vol_trend" in offenders, "audit missed a same-day leak"
    assert offenders, "audit missed an injected leak entirely"


def test_backtest_runs_and_scores():
    prices = _rich_prices(seed=1)
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
    # auto-discover so a new test can never be silently left un-run
    import sys as _sys
    _tests = [(n, f) for n, f in sorted(globals().items())
              if n.startswith("test_") and callable(f)]
    for _name, _fn in _tests:
        _fn()
        print(f"  ok  {_name}")
    print(f"All {len(_tests)} tests passed.")
