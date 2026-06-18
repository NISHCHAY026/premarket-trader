"""Walk-forward training + honest out-of-sample evaluation.

Two layers guard against overfitting:
  1. The ML win-probabilities are out-of-fold (model.walk_forward_predict).
  2. Selection thresholds are grid-searched on a chronological TRAIN slice and then
     applied, unchanged, to a later TEST slice. The TEST metrics are the headline.
"""
from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import pandas as pd

from .config import Params
from .features import build_events
from . import model as model_mod
from .backtest import run_backtest
from . import metrics as metrics_mod

GRID = {
    "gap_min": [0.03, 0.05, 0.08, 0.12],
    "rvol_min": [1.0, 1.5, 2.5],
    "ml_min_proba": [0.45, 0.50, 0.55, 0.60],
    "max_positions": [3, 5, 10],
}
MIN_TRADES_TRAIN = 60


def _date_split(ev: pd.DataFrame, frac: float):
    dates = np.sort(ev["date"].unique())
    if len(dates) < 4:
        return ev.iloc[0:0], ev
    cut = dates[int(len(dates) * frac)]
    return ev[ev["date"] <= cut], ev[ev["date"] > cut]


def _objective(m: dict) -> float:
    if m.get("trades", 0) < MIN_TRADES_TRAIN:
        return -1e9
    # reward risk-adjusted return; small nudge from total return as tie-breaker
    return m["sharpe"] + 0.25 * np.tanh(m["total_return"])


def grid_search(train: pd.DataFrame, p: Params) -> tuple[Params, dict]:
    best_p, best_obj, best_m = p, -np.inf, {}
    keys = list(GRID)
    for combo in itertools.product(*(GRID[k] for k in keys)):
        cand = dataclasses.replace(p, **dict(zip(keys, combo)))
        m = metrics_mod.compute(run_backtest(train, cand), cand.account)
        obj = _objective(m)
        if obj > best_obj:
            best_p, best_obj, best_m = cand, obj, m
    return best_p, best_m


def run(prices: pd.DataFrame, p: Params, optimize: bool = True) -> dict:
    events = build_events(prices, p)
    if events.empty:
        return {"error": "no gap events built — check universe/date range"}

    if p.use_ml:
        events = model_mod.walk_forward_predict(events, p)
        oof_start = model_mod.first_oof_date(events, p)
        ev = events[pd.to_datetime(events["date"]) >= oof_start].copy() if oof_start is not None else events.iloc[0:0]
    else:
        ev = events.copy()
    if ev.empty:
        return {"error": "no out-of-fold period — extend the date range"}

    train, test = _date_split(ev, p.train_frac)
    best_p, m_train = (grid_search(train, p) if optimize and not train.empty else (p, {}))

    res_train = run_backtest(train, best_p)
    res_test = run_backtest(test, best_p)
    res_full = run_backtest(ev, best_p)
    return {
        "events": events,
        "eval_events": ev,
        "best_params": best_p,
        "metrics_train": metrics_mod.compute(res_train, best_p.account),
        "metrics_test": metrics_mod.compute(res_test, best_p.account),
        "metrics_full": metrics_mod.compute(res_full, best_p.account),
        "result_full": res_full,
        "result_test": res_test,
    }
