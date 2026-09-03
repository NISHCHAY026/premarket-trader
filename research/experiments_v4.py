"""Self-improvement round: 3 experiment families, winners picked on TRAIN only.

Families:
  G) label geometry     — stop_pct x profit_ratio (labels + brackets change)
  P) feature pruning    — drop dead features (importance ~0) and seasonality
  L) LGBM hyperparams   — never tuned before

Protocol: for each family, every variant is tuned (grid_search) and scored on the
TRAIN slice; the family winner is chosen by TRAIN Sharpe. Only the final combined
winner gets ONE evaluation on TEST + FULL (vs the v3 baseline) to limit test burn.
"""
import sys, dataclasses, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd

from premarket_trader.config import Params
from premarket_trader.data import load_universe, download_prices
from premarket_trader import features as F, model as M, walkforward as W
from premarket_trader import metrics as Met
from premarket_trader.backtest import run_backtest

BASE = Params(start="2018-01-01")
prices = download_prices(load_universe(BASE.universe_csv), BASE.start, BASE.end)

FULL_COLS = list(F.FEATURE_COLS)
DEAD = ["day_of_move", "above_sma200", "gap_count_10d", "consec_up_days"]
SEASON = ["month", "dow"]

def set_cols(cols):
    F.FEATURE_COLS[:] = cols          # single shared list object
    M.FEATURE_COLS = F.FEATURE_COLS

def set_lgbm(kw):
    import lightgbm as lgb
    def _nm():
        d = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, max_depth=-1,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                 min_child_samples=40, reg_lambda=1.0, random_state=42,
                 n_jobs=-1, verbose=-1)
        d.update(kw)
        return lgb.LGBMClassifier(**d)
    M._new_model = _nm

def evaluate(p, cols, lgbm_kw, label):
    """Build -> OOF predict -> tune on train -> return train metrics (+handles)."""
    set_cols(cols); set_lgbm(lgbm_kw)
    ev = F.build_events(prices, p)
    ev = M.walk_forward_predict(ev, p)
    oof = M.first_oof_date(ev, p)
    ev = ev[pd.to_datetime(ev["date"]) >= oof].copy()
    train, test = W._date_split(ev, p.train_frac)
    best, _ = W.grid_search(train, p)
    mtr = Met.compute(run_backtest(train, best), best.account)
    print(f"  {label:34s} TRAIN Sharpe {mtr['sharpe']:5.2f}  Ret {mtr['total_return']*100:6.1f}%  "
          f"DD {mtr['max_drawdown']*100:6.1f}%  Trades {mtr['trades']}", flush=True)
    return {"label": label, "train_sharpe": mtr["sharpe"], "params": best,
            "train": train, "test": test, "ev": ev}

results = {}

print("=== family G: label geometry (TRAIN selection) ===", flush=True)
GEOM = [(0.05, 2.0), (0.08, 2.0), (0.12, 2.0), (0.08, 1.5), (0.08, 3.0), (0.05, 3.0)]
gruns = []
for stop, ratio in GEOM:
    p = dataclasses.replace(BASE, stop_pct=stop, profit_ratio=ratio)
    gruns.append(evaluate(p, FULL_COLS, {}, f"G stop={stop:.2f} ratio={ratio:.1f}"))
results["G"] = max(gruns, key=lambda r: r["train_sharpe"])
print(f"  -> G winner: {results['G']['label']}")

print("=== family P: feature pruning (TRAIN selection, base geometry) ===", flush=True)
PRUNE = {
    "P0 full 32": FULL_COLS,
    "P1 drop dead4": [c for c in FULL_COLS if c not in DEAD],
    "P2 dead4+seasonality": [c for c in FULL_COLS if c not in DEAD + SEASON],
}
pruns = [evaluate(BASE, cols, {}, lab) for lab, cols in PRUNE.items()]
results["P"] = max(pruns, key=lambda r: r["train_sharpe"])
print(f"  -> P winner: {results['P']['label']}")

print("=== family L: LGBM hyperparams (TRAIN selection, base geometry, P-winner cols) ===", flush=True)
pcols = PRUNE[results["P"]["label"]]
LGBM = {
    "L0 default": {},
    "L1 shallow": dict(num_leaves=15, min_child_samples=60, learning_rate=0.05, n_estimators=300),
    "L2 regularized": dict(reg_lambda=5.0, colsample_bytree=0.6, num_leaves=15),
    "L3 deep-slow": dict(num_leaves=63, learning_rate=0.02, n_estimators=600),
}
lruns = [evaluate(BASE, pcols, kw, lab) for lab, kw in LGBM.items()]
results["L"] = max(lruns, key=lambda r: r["train_sharpe"])
print(f"  -> L winner: {results['L']['label']}")

# ---- combined winner: ONE evaluation on TEST + FULL ----
print("\n=== COMBINED winner -> single TEST/FULL evaluation ===", flush=True)
gstop, gratio = [(s, r) for (s, r) in GEOM
                 if f"stop={s:.2f} ratio={r:.1f}" in results["G"]["label"]][0]
p = dataclasses.replace(BASE, stop_pct=gstop, profit_ratio=gratio)
lkw = LGBM[results["L"]["label"]]
run = evaluate(p, pcols, lkw, "COMBINED (train)")
best = run["params"]
mte = Met.compute(run_backtest(run["test"], best), best.account)
mfl = Met.compute(run_backtest(run["ev"], best), best.account)
print(f"\n  v3 baseline : TEST Sharpe 1.28  Ret +10.1%  PF 1.80  DD -3.2% | FULL Sharpe 0.91")
print(f"  v4 combined : TEST Sharpe {mte['sharpe']:.2f}  Ret {mte['total_return']*100:+.1f}%  "
      f"PF {mte['profit_factor']:.2f}  DD {mte['max_drawdown']*100:.1f}% | FULL Sharpe {mfl['sharpe']:.2f}")
print(f"  combined config: stop={gstop} ratio={gratio}  cols={results['P']['label']}  lgbm={results['L']['label']}")
Path("results/experiments_v4.json").write_text(json.dumps(
    {"G": results["G"]["label"], "P": results["P"]["label"], "L": results["L"]["label"],
     "test": {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
              for k, v in mte.items()},
     "full_sharpe": float(mfl["sharpe"])}, indent=1))
print("saved -> results/experiments_v4.json")
