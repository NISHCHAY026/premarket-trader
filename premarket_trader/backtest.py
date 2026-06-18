"""Event-driven daily backtester for the gap-and-go day-trade strategy.

Each trading day: rank that day's qualifying gap events, take the best N, size each
by fixed-fractional risk, simulate the intraday outcome, net out slippage/commission,
and compound equity. All positions are intraday (open->exit same day); total notional
is capped at 1x equity (cash, no leverage) for conservatism.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Params


def select_candidates(day: pd.DataFrame, p: Params) -> pd.DataFrame:
    m = (
        (day["gap"] >= p.gap_min)
        & (day["rvol"] >= p.rvol_min)
        & (day["open"] >= p.price_min)
        & (day["open"] <= p.price_max)
        & (day["dollar_vol_prev"] >= p.min_dollar_vol)
    )
    if p.use_ml:
        m = m & day["ml_proba"].notna() & (day["ml_proba"] >= p.ml_min_proba)
    cand = day[m].copy()
    rank = "ml_proba" if (p.use_ml and cand["ml_proba"].notna().any()) else "gap"
    return cand.sort_values(rank, ascending=False).head(p.max_positions)


def _regime_temp(ev: pd.DataFrame, lookback: int):
    """Market 'temperature' per day = breadth / its trailing median (prior days only).

    >1 = hotter than recent, <1 = colder. Uses shift(1) so it is knowable at the open.
    """
    bday = ev.groupby("date")["market_breadth"].first().sort_index()
    med = bday.rolling(lookback, min_periods=5).median().shift(1)
    return (bday / med.replace(0, np.nan)).to_dict()


def run_backtest(events: pd.DataFrame, p: Params) -> dict:
    ev = events.sort_values("date").copy()
    ev["date"] = pd.to_datetime(ev["date"])
    equity = p.account
    slip = p.slippage_bps / 1e4
    temp_map = _regime_temp(ev, p.regime_lookback) if p.regime_sizing else {}

    trades, curve = [], []
    for day, grp in ev.groupby("date"):
        cand = select_candidates(grp, p)

        # ---- regime sizing: stand aside on cold tape, scale size with temperature ----
        risk_scale, halted = 1.0, False
        if p.regime_sizing:
            t = temp_map.get(day, float("nan"))
            if t == t:                                  # not NaN
                if t < 0.5:                             # much colder than recent -> no trades
                    curve.append({"date": day, "equity": equity, "day_pnl": 0.0,
                                  "n_trades": 0, "halted": True})
                    continue
                risk_scale = min(max(t, 0.5), 1.5)

        buying_power, day_pnl, reds, taken = equity, 0.0, 0, 0
        daily_max_loss = p.daily_max_loss * equity
        daily_goal = p.daily_goal_mult * daily_max_loss

        for i, (_, r) in enumerate(cand.iterrows()):
            # ---- circuit breakers (rank order, same-day realized PnL) ----
            if (p.three_strike and reds >= p.three_strike) \
                    or day_pnl <= -daily_max_loss or day_pnl >= daily_goal:
                halted = True
                break
            ladder = (0.25 if i == 0 else 0.5 if i == 1 else 1.0) if p.icebreaker else 1.0
            entry = r["entry"]
            risk_per_share = entry * p.stop_pct
            if risk_per_share <= 0:
                continue
            shares = int(min(
                (equity * p.risk_per_trade * risk_scale * ladder) / risk_per_share,
                (equity * p.max_pos_frac) / entry,
                buying_power / entry,
            ))
            if shares <= 0:
                continue
            buying_power -= shares * entry
            gross = shares * (r["exit_px"] - entry)
            costs = shares * (entry + r["exit_px"]) * slip + 2 * p.commission_per_share * shares
            net = gross - costs
            day_pnl += net
            taken += 1
            if net <= 0:
                reds += 1
            trades.append({
                "date": day, "ticker": r["ticker"], "entry": entry,
                "exit": r["exit_px"], "shares": shares, "reason": r["exit_reason"],
                "gap": r["gap"], "rvol": r["rvol"],
                "ml_proba": r.get("ml_proba", np.nan),
                "ret": r["trade_ret_gross"], "pnl": net,
            })
        equity += day_pnl
        curve.append({"date": day, "equity": equity, "day_pnl": day_pnl,
                      "n_trades": taken, "halted": halted})

    return {"trades": pd.DataFrame(trades), "equity": pd.DataFrame(curve), "params": p}
