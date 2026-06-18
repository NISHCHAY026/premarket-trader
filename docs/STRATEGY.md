# Strategy, Model & Backtest Methodology

This document explains how the [playbook](PLAYBOOK.md) is turned into a concrete,
**trainable, leakage-free** model — and, just as importantly, what the backtest can and
cannot honestly claim.

---

## 1. From an intraday method to a daily-bar model

The playbook is an **intraday** method (first-pullback entries on 1–5 minute candles in
low-float small-caps). Faithfully backtesting that needs minute bars, premarket prints,
float, halts and borrow data — none of which are available from free daily sources.

So we test the **data-honest adjacent hypothesis**: *does buying low-priced names that
gap up on high relative volume, and selecting them well, carry a positive intraday edge?*
This is the **selection edge** of the methodology, evaluated on daily OHLCV.

**Entry:** buy at the day's **open** when a name qualifies.
**Exit (same day):** a fixed **2:1** bracket — stop at `open·(1−stop_pct)`, target at
`open·(1+2·stop_pct)`; otherwise exit at the **close**. If both the stop and target fall
inside the day's range, we **assume the stop filled first** (pessimistic).

## 2. Features — strictly no lookahead

Every feature is knowable at the **open** of the trading day (built from data through the
prior close, plus that morning's open):

`gap`, `rvol` (prior-day volume vs its 20-day average), `log_price`, `atr_pct`,
`range_prev`, `ret_1d/5d/20d`, `gap_vs_atr`, `dist_high20`, `vol_trend`,
`market_breadth` (how many universe names gap ≥5% that day), `dow`, `month`.

The trade **outcome** (used as the training label and for PnL) uses the day's high/low/
close — known only *after* entry — so it is **never** used as a feature. A unit test
([`tests/test_strategy.py`](../tests/test_strategy.py)) tampers with each day's intraday
bars and asserts the features are unchanged.

## 3. The model and how it is trained

A **LightGBM** classifier predicts `P(trade is profitable)` from the pre-open features.
Training is **walk-forward / out-of-fold**: predictions for each test window come only
from a model fit on *earlier* data, with an embargo gap. The strategy then trades only
candidates whose predicted win-probability clears a threshold, ranked best-first.

## 4. Honest out-of-sample evaluation

Two guards against overfitting:

1. **Out-of-fold ML** — probabilities are never in-sample.
2. **Chronological train/test split** — selection thresholds (`gap_min`, `rvol_min`,
   `ml_min_proba`, `max_positions`) are grid-searched on the earlier **train** slice and
   then applied unchanged to the later **test** slice. **Test metrics are the headline.**

## 5. What the demo shows

Default 135-name universe, 2018→present, $30k account, 1% risk/trade, 10 bps slippage/side:

| Variant (out-of-sample TEST) | Return | CAGR | Sharpe | Max DD | Win | Profit factor |
|---|---|---|---|---|---|---|
| **Rules + ML** | **+8.9%** | **+4.3%** | **0.93** | **−4.1%** | **58.5%** | **1.55** |
| Rules only (no ML) | −35.8% | −13.2% | −1.51 | −35.6% | 31.9% | 0.60 |

The takeaway mirrors the playbook: **naively chasing every gap loses money; disciplined,
model-driven selection is the edge.** (Your numbers will move with the universe, dates and
data vintage — re-run to reproduce.)

## 6. Limitations (read these)

- **Daily-bar fidelity.** We approximate an intraday method with open→close trades; the
  real first-pullback timing, scaling, and "breakout-or-bailout" exits are not captured.
- **Survivorship bias.** The default universe is *currently listed* names; delisted
  blow-ups are missing, which flatters results. Supply a point-in-time universe via
  `--universe-csv` to reduce this.
- **Float & catalyst** are not in free daily data, so two of the five pillars are proxied
  (by price/volatility and by the gap/volume itself).
- **Slippage/liquidity.** Low-float names can gap through stops; the flat 10 bps/side
  assumption is optimistic for the cheapest names.
- **Single split.** The headline uses one chronological train/test split; a rolling
  walk-forward of the thresholds would be more robust (the ML layer already rolls).
- **Back-adjusted prices.** Data is pulled with `auto_adjust=True`, which prevents fake
  split-induced gaps but bakes future split/dividend factors into past bars — so pre-split
  gap magnitudes are mildly non-causal. A point-in-time (as-of) price feed would remove
  this; for low-priced names where splits are common it is a real (if small) bias.

Treat this as a **rigorous research framework**, not a turnkey money printer. The honest
edge lives in execution and in data (intraday + float) that this free-data demo can't see.
