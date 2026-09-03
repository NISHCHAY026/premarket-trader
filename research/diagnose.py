"""Diagnostics for the self-improvement round: importances + OOS error analysis."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd

from premarket_trader.config import Params
from premarket_trader.data import load_universe, download_prices
from premarket_trader.features import build_events, FEATURE_COLS
from premarket_trader import model as M, walkforward as W
from premarket_trader.backtest import select_candidates

p = Params(start="2018-01-01")
prices = download_prices(load_universe(p.universe_csv), p.start, p.end)
events = build_events(prices, p)
events = M.walk_forward_predict(events, p)
oof = M.first_oof_date(events, p)
ev = events[pd.to_datetime(events["date"]) >= oof].copy()
train, test = W._date_split(ev, p.train_frac)
best, _ = W.grid_search(train, p)

print("=== dataset ===")
print(f"events total={len(events)}  oof={len(ev)}  train={len(train)}  test={len(test)}")
print(f"base win rate (all oof events): {ev['win'].mean():.3f}")
print(f"tuned: gap_min={best.gap_min} rvol_min={best.rvol_min} "
      f"proba={best.ml_min_proba} maxpos={best.max_positions}")

# --- feature importances from a model fit on the train slice ---
import lightgbm as lgb
tr = train.dropna(subset=FEATURE_COLS)
m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                       min_child_samples=30, subsample=0.9, colsample_bytree=0.8,
                       verbose=-1, random_state=7)
m.fit(tr[FEATURE_COLS], tr["win"])
imp = pd.Series(m.booster_.feature_importance("gain"), index=FEATURE_COLS)
imp = (imp / imp.sum() * 100).sort_values(ascending=False)
print("\n=== feature importances (% of gain) ===")
for k, v in imp.items():
    print(f"  {k:24s} {v:6.2f}")

# --- OOS selected-trade error analysis ---
# per-day selection, exactly as run_backtest does (whole-frame would give a global top-N)
sel = pd.concat([select_candidates(g, best) for _, g in test.groupby("date")], ignore_index=True)
sel["month_p"] = pd.to_datetime(sel["date"]).dt.to_period("M")
print("\n=== TEST selected trades ===")
print(f"n={len(sel)}  win={sel['win'].mean():.3f}  "
      f"mean gross ret={sel['trade_ret_gross'].mean()*100:.2f}%")
print("\nby exit reason:")
print(sel.groupby("exit_reason").agg(n=("win", "size"), win=("win", "mean"),
      ret=("trade_ret_gross", "mean")).to_string())
print("\nworst 8 trades:")
cols = ["date", "ticker", "gap", "rvol", "ml_proba", "exit_reason", "trade_ret_gross"]
print(sel.nsmallest(8, "trade_ret_gross")[cols].to_string(index=False))
print("\nby ml_proba bucket (calibration):")
sel["pb"] = pd.cut(sel["ml_proba"], [0.5, 0.55, 0.6, 0.65, 1.0])
print(sel.groupby("pb", observed=True).agg(n=("win", "size"), win=("win", "mean"),
      ret=("trade_ret_gross", "mean")).to_string())

# stash for the workflow agents
out = {"importances": imp.round(2).to_dict(),
       "test_by_reason": sel.groupby("exit_reason")["win"].agg(["size", "mean"]).to_dict(),
       "tuned": {"gap_min": best.gap_min, "rvol_min": best.rvol_min,
                  "ml_min_proba": best.ml_min_proba, "max_positions": best.max_positions}}
Path("results/diagnostics.json").write_text(json.dumps(out, indent=1, default=str))
print("\nsaved -> results/diagnostics.json")
