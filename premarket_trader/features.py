"""Build the gap-event table: per (ticker, day) features + fixed-management outcome.

Strict no-lookahead rule: every *feature* is knowable at the OPEN of day T
(data through T-1, plus that morning's open). The *outcome* (used as the training
label and for PnL) uses day T's high/low/close, which is only known after entry —
so outcomes are never used as features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Params

FEATURE_COLS = [
    "gap", "rvol", "log_price", "atr_pct", "range_prev",
    "ret_1d", "ret_5d", "ret_20d", "gap_vs_atr", "dist_high20",
    "vol_trend", "market_breadth", "dow", "month",
    # ---- corpus-driven additions (all lookahead-safe; see docs/STRATEGY.md) ----
    "gap_pctile", "rvol_pctile", "n_gappers",            # cross-sectional leadership
    "close_location_prev", "upper_wick_prev", "lower_wick_prev", "retrace_depth_prev",
    "extension_vs_ema9", "consec_up_days", "blue_sky_dist",
    # ---- v3: full-corpus delta-mine, HIGH-priority subset only (trend-MA + prior-day-high + move-seq) ----
    "dist_to_sma200", "above_sma200", "dist_to_prev_day_high",
    "day_of_move", "days_since_last_gap", "gap_count_10d", "former_runner_count", "max_prior_1d_range",
]


def _per_ticker(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    prev_close = g["close"].shift(1)
    # True range -> ATR(14), shifted so it reflects info up to T-1
    tr = pd.concat([
        g["high"] - g["low"],
        (g["high"] - prev_close).abs(),
        (g["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean().shift(1)
    avg_vol20 = g["volume"].rolling(20).mean().shift(1)

    g["prev_close"] = prev_close
    g["gap"] = g["open"] / prev_close - 1.0
    g["rvol"] = g["volume"].shift(1) / avg_vol20                # prior-day volume surge
    g["dollar_vol_prev"] = (g["close"] * g["volume"]).shift(1)
    g["log_price"] = np.log(g["open"].clip(lower=0.01))
    g["atr_pct"] = atr14 / prev_close
    g["range_prev"] = ((g["high"] - g["low"]) / g["close"]).shift(1)
    g["ret_1d"] = g["close"].shift(1) / g["close"].shift(2) - 1.0
    g["ret_5d"] = g["close"].shift(1) / g["close"].shift(6) - 1.0
    g["ret_20d"] = g["close"].shift(1) / g["close"].shift(21) - 1.0
    g["gap_vs_atr"] = g["gap"] / g["atr_pct"].replace(0, np.nan)
    g["dist_high20"] = g["prev_close"] / g["high"].rolling(20).max().shift(1) - 1.0
    g["vol_trend"] = avg_vol20 / g["volume"].rolling(60).mean().shift(1)

    # ---- corpus-driven features (all from PRIOR completed bars; no lookahead) ----
    prev_open, prev_high, prev_low = (g["open"].shift(1), g["high"].shift(1), g["low"].shift(1))
    rng = (prev_high - prev_low).replace(0, np.nan)
    body_hi = pd.concat([prev_open, prev_close], axis=1).max(axis=1)
    body_lo = pd.concat([prev_open, prev_close], axis=1).min(axis=1)
    # where the prior bar closed in its range + wick anatomy (topping/bottoming tails)
    g["close_location_prev"] = ((prev_close - prev_low) / rng).fillna(0.5)
    g["upper_wick_prev"] = ((prev_high - body_hi) / rng).fillna(0.0)
    g["lower_wick_prev"] = ((body_lo - prev_low) / rng).fillna(0.0)
    g["retrace_depth_prev"] = ((prev_high - prev_close) / rng).fillna(0.5)  # 50%-retrace proxy
    # over-extension / exhaustion
    ema9 = g["close"].ewm(span=9, adjust=False).mean().shift(1)
    g["extension_vs_ema9"] = (prev_close - ema9) / ema9
    up = (g["close"] > g["close"].shift(1)).astype(int)
    run = up.groupby((up != up.shift()).cumsum()).cumcount() + 1
    g["consec_up_days"] = (run * up).shift(1).clip(upper=6)
    # blue-sky: open above the long-horizon high = no overhead resistance
    hist_high = g["high"].rolling(252, min_periods=20).max().shift(1)
    g["blue_sky_dist"] = g["open"] / hist_high - 1.0

    # ---- v3 (full-corpus delta-mine); all from prior bars + today's open, no lookahead ----
    # 200-day SMA of closes = the slow trend resistance level the corpus cites constantly
    sma200 = g["close"].rolling(200, min_periods=100).mean().shift(1)
    # NaN must propagate: fabricating 0.0 here would defeat the dropna guard below
    # and tell the model "the open sits exactly on the 200-day SMA".
    g["dist_to_sma200"] = g["open"] / sma200 - 1.0
    g["above_sma200"] = np.where(sma200.notna(), (g["open"] > sma200).astype(float), np.nan)
    # distance to the single prior-day high = the day-2 continuation trigger
    g["dist_to_prev_day_high"] = g["open"] / prev_high - 1.0
    # move-day sequence: a "move day" = big up-close / big gap / wide range (prior bars only)
    _G, _R = 0.10, 0.20
    mv = (((g["close"] / prev_close - 1.0) >= _G)
          | ((g["open"] / prev_close - 1.0) >= _G)
          | (((g["high"] - g["low"]) / prev_close) >= _R)).astype(int)
    mv_run = (mv.groupby((mv != mv.shift()).cumsum()).cumcount() + 1) * mv
    g["day_of_move"] = 1.0 + mv_run.shift(1).fillna(0.0)      # which day of an ongoing run is today
    _idx = pd.Series(np.arange(len(g)), index=g.index)
    g["days_since_last_gap"] = (_idx - _idx.where(mv == 1).ffill().shift(1)).clip(upper=60).fillna(60.0)
    g["gap_count_10d"] = mv.rolling(10).sum().shift(1)
    g["former_runner_count"] = mv.rolling(252, min_periods=20).sum().shift(1)   # historical propensity
    g["max_prior_1d_range"] = (((g["high"] - g["low"]) / prev_close)
                               .rolling(252, min_periods=20).max().shift(1))
    return g


def _simulate_outcome(events: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Fixed 2:1 intraday day-trade outcome (pessimistic when both stop+target hit)."""
    entry = events["open"].to_numpy()
    stop = entry * (1.0 - p.stop_pct)
    target = entry * (1.0 + p.target_pct)
    high = events["high"].to_numpy()
    low = events["low"].to_numpy()
    close = events["close"].to_numpy()

    exit_px = close.copy()
    reason = np.full(len(events), "close", dtype=object)
    hit_stop = low <= stop
    hit_tgt = high >= target
    # pessimistic: if both could trigger same day, assume the stop filled first
    exit_px = np.where(hit_stop, stop, np.where(hit_tgt, target, close))
    reason = np.where(hit_stop, "stop", np.where(hit_tgt, "target", "close"))

    out = events.copy()
    out["entry"] = entry
    out["stop_px"] = stop
    out["target_px"] = target
    out["exit_px"] = exit_px
    out["exit_reason"] = reason
    out["trade_ret_gross"] = exit_px / entry - 1.0
    out["win"] = (out["trade_ret_gross"] > 0).astype(int)
    return out


def build_events(prices: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Return the candidate gap-event table with features, labels and outcomes."""
    prices = prices.sort_values(["ticker", "date"]).copy()
    feat = pd.concat([_per_ticker(g) for _, g in prices.groupby("ticker", sort=False)],
                     ignore_index=True)

    # Daily market breadth (how many universe names gap up >=5% that day) = "hot tape".
    # Known at the open, so it is a legitimate feature.
    breadth = (feat.assign(big=(feat["gap"] >= 0.05).astype(int))
                   .groupby("date")["big"].transform("sum"))
    feat["market_breadth"] = breadth
    feat["dow"] = pd.to_datetime(feat["date"]).dt.dayofweek
    feat["month"] = pd.to_datetime(feat["date"]).dt.month

    # Candidate pool: a real gap, in the methodology's price band, tradable liquidity.
    pool = feat[
        (feat["gap"] >= p.event_gap_min)
        & (feat["open"] >= p.price_min)
        & (feat["open"] <= p.price_max)
        & (feat["dollar_vol_prev"] >= p.min_dollar_vol)
    ].copy()
    # cross-sectional "leading gainer" features over the day's candidate pool (open-time)
    pool["gap_pctile"] = pool.groupby("date")["gap"].rank(pct=True)
    pool["rvol_pctile"] = pool.groupby("date")["rvol"].rank(pct=True)
    pool["n_gappers"] = pool.groupby("date")["gap"].transform("count")
    pool = pool.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS)
    pool = _simulate_outcome(pool, p)
    return pool.sort_values("date").reset_index(drop=True)
