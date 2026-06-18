"""LightGBM win-probability model, trained walk-forward (out-of-fold predictions).

The model predicts P(trade is profitable) from pre-open features only. Predictions
for each test window come from a model trained exclusively on earlier data (with an
embargo gap), so the probabilities used in the backtest are never in-sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Params
from .features import FEATURE_COLS


def _new_model():
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=31,
        max_depth=-1, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        min_child_samples=40, reg_lambda=1.0, random_state=42, n_jobs=-1,
        verbose=-1,
    )


def walk_forward_predict(events: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Add an out-of-fold `ml_proba` column via rolling-origin training."""
    ev = events.sort_values("date").reset_index(drop=True).copy()
    ev["date"] = pd.to_datetime(ev["date"])
    ev["ml_proba"] = np.nan

    dmin, dmax = ev["date"].min(), ev["date"].max()
    train_win = pd.Timedelta(days=int(p.ml_train_years * 365))
    test_win = pd.Timedelta(days=int(p.ml_test_months * 30))
    embargo = pd.Timedelta(days=p.embargo_days)

    test_start = dmin + train_win
    while test_start < dmax:
        test_end = test_start + test_win
        tr = ev[ev["date"] <= test_start - embargo]
        te_mask = (ev["date"] > test_start) & (ev["date"] <= test_end)
        # need both classes and enough rows to train
        if tr["win"].nunique() == 2 and len(tr) >= 200 and te_mask.any():
            model = _new_model()
            model.fit(tr[FEATURE_COLS], tr["win"])
            ev.loc[te_mask, "ml_proba"] = model.predict_proba(
                ev.loc[te_mask, FEATURE_COLS])[:, 1]
        test_start = test_end
    return ev


def first_oof_date(events: pd.DataFrame, p: Params):
    """The earliest date that has out-of-fold predictions (start of evaluable period)."""
    ev = events.dropna(subset=["ml_proba"])
    return None if ev.empty else pd.to_datetime(ev["date"]).min()


def fit_full(events: pd.DataFrame):
    """Train one model on ALL events (for the live scanner). Returns the fitted model."""
    model = _new_model()
    model.fit(events[FEATURE_COLS], events["win"])
    return model
