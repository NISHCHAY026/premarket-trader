"""Does higher ml_proba actually mean better outcomes? Compare slices properly."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from scipy import stats

from premarket_trader.config import Params
from premarket_trader.data import load_universe, download_prices
from premarket_trader.features import build_events
from premarket_trader import model as M, walkforward as W
from premarket_trader.backtest import select_candidates

p = Params(start="2018-01-01")
prices = download_prices(load_universe(p.universe_csv), p.start, p.end)
ev = M.walk_forward_predict(build_events(prices, p), p)
ev = ev[pd.to_datetime(ev["date"]) >= M.first_oof_date(ev, p)].copy()
train, test = W._date_split(ev, p.train_frac)
best, _ = W.grid_search(train, p)
per_day = lambda f: pd.concat([select_candidates(g, best) for _, g in f.groupby("date")],
                              ignore_index=True)

for name, frame in (("TRAIN", train), ("TEST", test)):
    s = per_day(frame)
    rho, pv = stats.spearmanr(s["ml_proba"], s["trade_ret_gross"])
    hi = s[s["ml_proba"] > 0.65]["trade_ret_gross"]
    rest = s[s["ml_proba"] <= 0.65]["trade_ret_gross"]
    t, tp = stats.ttest_ind(hi, rest, equal_var=False) if len(hi) > 2 else (float("nan"),) * 2
    print(f"{name:6s} n={len(s):3d}  spearman(proba, return) = {rho:+.3f} (p={pv:.2f})   "
          f">0.65 vs rest: diff {hi.mean()*100:+.2f}% vs {rest.mean()*100:+.2f}%  p={tp:.2f}")

print("\nIf the sign of spearman flips across slices and no p-value is significant,")
print("the 'high-proba is bad' pattern is NOISE and must not drive a model change.")
