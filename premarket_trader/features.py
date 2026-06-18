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
    pool = pool.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS)
    pool = _simulate_outcome(pool, p)
    return pool.sort_values("date").reset_index(drop=True)
