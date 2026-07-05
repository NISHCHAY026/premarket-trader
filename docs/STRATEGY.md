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
`market_breadth` (how many universe names gap ≥5% that day), `dow`, `month`,
plus the **corpus-driven** additions (see §5b): `gap_pctile`, `rvol_pctile`, `n_gappers`
(cross-sectional "leading gainer" rank), `close_location_prev`, `upper_wick_prev`,
`lower_wick_prev`, `retrace_depth_prev` (prior-bar candle shape / 50%-retrace proxy),
`extension_vs_ema9`, `consec_up_days` (over-extension), and `blue_sky_dist` (open vs
the 252-day high) — plus the **v3** additions (see §5d): `dist_to_sma200`, `above_sma200`,
`dist_to_prev_day_high`, and the move-sequence family (`day_of_move`, `days_since_last_gap`,
`gap_count_10d`, `former_runner_count`, `max_prior_1d_range`).

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

## 5b. Corpus-driven feature upgrade (v2)

A deep mine of the full transcript corpus surfaced concrete, *daily-computable* signals the
original model ignored. The highest-value, leakage-safe ones were added (the rest — float,
reverse-split, news catalysts — were correctly dropped as not derivable from adjusted daily
OHLCV):

- **Cross-sectional leadership** (`gap_pctile`, `rvol_pctile`, `n_gappers`) — the corpus's
  strongest, most-repeated idea ("trade the #1 leading gainer"). The original model scored
  every name in isolation with no notion of whether it was *the* leader or the 8th-best.
- **Prior-bar candle shape** (`close_location_prev`, wick ratios, `retrace_depth_prev`) —
  encodes the topping-tail / strong-close vocabulary and the 50%-retracement rule.
- **Over-extension** (`extension_vs_ema9`, `consec_up_days`) — the "too extended after 4–5
  green candles" disqualifier.
- **Blue-sky** (`blue_sky_dist`) — open above the long-horizon high = no overhead supply.

Effect on the **held-out test slice** (params tuned only on the earlier train half):

| OOS TEST | v1 (baseline) | v2 (corpus features) |
|---|---|---|
| Sharpe | 0.93 | **1.14** |
| Max drawdown | −4.1% | **−2.8%** |
| Profit factor | 1.55 | **1.80** |
| Trades | 53 | 36 (more selective) |

**Honest caveat:** the gains concentrate in the recent test slice on a **small sample
(36 trades)**; over the full out-of-fold period the result is comparable (Sharpe ~0.80).
The features are well-motivated and leakage-free, but on this small, survivorship-biased
universe treat this as a *modest* improvement, not a step-change. The corpus is a hypothesis
source, not ground truth.

## 5c. Risk layer (v3) — implemented, validated, kept opt-in

The corpus also suggested a **risk layer**: regime sizing (trade bigger on a hot tape,
stand aside on a cold one), an "icebreaker" ladder (quarter→half→full size on the first
picks), a daily-goal cap, and three-strike / daily-max-loss circuit breakers. We implemented
all of it ([backtest.py](../premarket_trader/backtest.py), gated by `Params`) and ran a
**component ablation** (selection + ML held fixed):

| Variant | TEST Sharpe | TEST Return | FULL Sharpe | FULL Max DD |
|---|---|---|---|---|
| **v2 (risk layer off)** | 1.14 | +8.5% | 0.80 | −7.3% |
| v3 full | 0.83 | +2.3% | 0.69 | −3.6% |
| regime sizing only | 1.27 | +12.3% | 0.85 | −10.7% |
| circuit breakers only | 1.06 | +7.5% | 0.77 | −7.3% |

**Findings:** the *full* layer **hurts** risk-adjusted return — the icebreaker sizes down the
highest-probability trades (backwards for a model whose edge *is* the probability ranking)
and the daily-goal cap chops winning days. Only **regime sizing** lifts headline return/Sharpe,
and it does so by *leveraging up to 1.5× on hot days*, which raises full-period drawdown
(−7.3% → −10.7%) on a small sample — conviction-leverage, not a free lunch.

**Decision:** the risk layer ships as an **opt-in** (all of it **off by default**); the
validated **v2 configuration remains the shipped model.** Enable any piece via `Params`
(`regime_sizing`, `daily_max_loss`, `three_strike`, `icebreaker`) if you want that trade-off.
This is the point of validation: a corpus suggestion is a hypothesis, and this one did not earn
a default-on slot. (The live bot keeps its own hard daily-loss kill switch regardless — see
[BOT.md](BOT.md).)

## 5d. Full-corpus refresh (v3 features)

After the transcript scrape completed (**2,098 transcripts / 8.0M words**, vs. the 1,105 used
for v2), a second delta-focused mine — handed the v2 feature set and asked only *"what's still
missing?"* — surfaced a genuinely new, distinct signal cluster:

- **`dist_to_sma200` / `above_sma200`** — distance to the 200-day SMA of closes, cited across
  40+ transcripts as *the* daily overhead-resistance ceiling (distinct from the 20/252-day
  *highs* v2 already had).
- **`dist_to_prev_day_high`** — distance to the single prior-day high, the actual day-2
  continuation trigger.
- **move-sequence family** (`day_of_move`, `days_since_last_gap`, `gap_count_10d`,
  `former_runner_count`, `max_prior_1d_range`) — where in an ongoing run the name sits, and its
  *historical* propensity for explosive single-day moves ("former runner"). Distinct from
  `consec_up_days`, which counts up-*closes*, not move-days.

**The ablation was decisive.** Adding *all* 14 mined features **overfit** and hurt the model
(TEST Sharpe 1.14 → 0.94). Keeping only the a-priori **HIGH-priority** subset (the 8 above)
improved it on train, test, *and* the full period:

| OOS TEST | v2 | v3 (HIGH subset) |
|---|---|---|
| Sharpe | 1.14 | **1.28** |
| Return | +8.5% | **+10.1%** |
| Profit factor | 1.80 | 1.80 |
| Max drawdown | −2.8% | −3.2% |
| Full-period Sharpe | 0.80 | **0.91** |

The MED-priority signals (overhead-supply, directional volume, lagged breadth) were dropped —
they diluted the signal on this small, survivorship-biased universe. Two agent claims of
"leaks" were also checked against the code and found **false** (breadth uses the open, not the
close; a dollar-volume floor already exists). **v3 is the shipped model.** Same honest caveat:
~43 OOS trades is a small sample — the edge is real but modest.

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
