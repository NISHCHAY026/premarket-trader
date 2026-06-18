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


def run_backtest(events: pd.DataFrame, p: Params) -> dict:
    ev = events.sort_values("date").copy()
    ev["date"] = pd.to_datetime(ev["date"])
    equity = p.account
    slip = p.slippage_bps / 1e4

    trades, curve = [], []
    for day, grp in ev.groupby("date"):
        cand = select_candidates(grp, p)
        buying_power = equity                       # 1x notional cap for the day
        day_pnl = 0.0
        for _, r in cand.iterrows():
            entry = r["entry"]
            risk_per_share = entry * p.stop_pct
            if risk_per_share <= 0:
                continue
            shares = int(min(
                (equity * p.risk_per_trade) / risk_per_share,   # risk-based
                (equity * p.max_pos_frac) / entry,              # concentration cap
                buying_power / entry,                           # cash cap
            ))
            if shares <= 0:
                continue
            buying_power -= shares * entry
            gross = shares * (r["exit_px"] - entry)
            costs = shares * (entry + r["exit_px"]) * slip + 2 * p.commission_per_share * shares
            net = gross - costs
            day_pnl += net
            trades.append({
                "date": day, "ticker": r["ticker"], "entry": entry,
                "exit": r["exit_px"], "shares": shares, "reason": r["exit_reason"],
                "gap": r["gap"], "rvol": r["rvol"],
                "ml_proba": r.get("ml_proba", np.nan),
                "ret": r["trade_ret_gross"], "pnl": net,
            })
        equity += day_pnl
        curve.append({"date": day, "equity": equity, "day_pnl": day_pnl,
                      "n_trades": int((cand.shape[0]) if len(cand) else 0)})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(curve)
    return {"trades": trades_df, "equity": equity_df, "params": p}
