"""Live-ish premarket scanner: rank today's gap setups with the trained model.

Note: free daily data (yfinance) has no reliable premarket feed. This scans using the
most recent session's bar as the candidate day. For genuine premarket use, supply
current premarket prices via `premarket_overrides` (ticker -> premarket price) to set
the entry/gap; otherwise the latest available open is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Params
from .data import load_universe, download_prices
from .features import build_events, FEATURE_COLS
from . import model as model_mod


def scan(p: Params, lookback_days: int = 600,
         premarket_overrides: dict[str, float] | None = None) -> pd.DataFrame:
    tickers = load_universe(p.universe_csv)
    start = (pd.Timestamp.today() - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    prices = download_prices(tickers, start=start, end=None, cache=False)
    if prices.empty:
        return pd.DataFrame()

    if premarket_overrides:
        last_idx = prices.groupby("ticker")["date"].idxmax()
        for i in last_idx:
            t = prices.loc[i, "ticker"]
            if t in premarket_overrides:
                prices.loc[i, "open"] = float(premarket_overrides[t])

    events = build_events(prices, p)
    if events.empty:
        return pd.DataFrame()

    events["date"] = pd.to_datetime(events["date"])
    scan_day = events["date"].max()
    # Match the backtest's embargo so live probabilities align with the validated
    # out-of-fold model (the backtest trains on data <= test_start - embargo).
    train = events[events["date"] <= scan_day - pd.Timedelta(days=p.embargo_days)]
    today = events[events["date"] == scan_day].copy()
    if today.empty:
        return pd.DataFrame()

    # Freshness guard: the "next session" plan is only valid on a current bar.
    staleness = (pd.Timestamp.today().normalize() - scan_day.normalize()).days
    if staleness > 4:
        print(f"[scanner] WARNING: latest completed session is {staleness} days old "
              f"({scan_day.date()}); this plan may be stale. Use premarket_overrides "
              "for a genuine pre-open scan.")

    if p.use_ml and train["win"].nunique() == 2 and len(train) >= 200:
        mdl = model_mod.fit_full(train)
        today["ml_proba"] = mdl.predict_proba(today[FEATURE_COLS])[:, 1]
    else:
        today["ml_proba"] = np.nan

    m = (
        (today["gap"] >= p.gap_min) & (today["rvol"] >= p.rvol_min)
        & (today["open"] >= p.price_min) & (today["open"] <= p.price_max)
        & (today["dollar_vol_prev"] >= p.min_dollar_vol)
    )
    if p.use_ml:
        m = m & today["ml_proba"].notna() & (today["ml_proba"] >= p.ml_min_proba)
    cand = today[m].copy()
    if cand.empty:
        return cand

    cand["entry"] = cand["open"]
    cand["stop"] = cand["entry"] * (1 - p.stop_pct)
    cand["target"] = cand["entry"] * (1 + p.target_pct)
    rank = "ml_proba" if p.use_ml else "gap"
    cols = ["date", "ticker", "entry", "stop", "target", "gap", "rvol", "ml_proba"]
    return cand.sort_values(rank, ascending=False)[cols].head(p.max_positions).reset_index(drop=True)
