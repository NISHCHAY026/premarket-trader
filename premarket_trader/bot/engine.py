"""BotEngine: the daily trading cycle, usable for backtest replay or live/paper runs.

The SAME selection + sizing code path drives both the historical backtest and the live
run, so a green backtest exercises the exact logic that would trade — only the broker
(simulated vs. real) and the data (historical bar vs. today's scan) differ.
"""
from __future__ import annotations

import pandas as pd

from ..config import Params
from ..backtest import select_candidates
from .broker import Broker, TradePlan
from .risk import RiskManager


class BotEngine:
    def __init__(self, params: Params, broker: Broker, risk: RiskManager | None = None):
        self.p = params
        self.broker = broker
        self.risk = risk or RiskManager(params)

    # ---- historical replay (backtest of the live logic) ----
    def backtest(self, events: pd.DataFrame) -> dict:
        ev = events.sort_values("date").copy()
        ev["date"] = pd.to_datetime(ev["date"])
        curve, fills_all = [], []
        for day, grp in ev.groupby("date"):
            st = self.risk.new_day(self.broker.equity())
            cand = select_candidates(grp, self.p)
            plans = self.risk.build_plans(cand, self.broker.equity(), st)
            bars = {r["ticker"]: {"open": r["open"], "high": r["high"],
                                  "low": r["low"], "close": r["close"], "date": day}
                    for _, r in cand.iterrows()}
            day_pnl = 0.0
            for plan in plans:
                fill = self.broker.execute_day_trade(plan, bars[plan.symbol])
                self.risk.record(st, fill.pnl)
                day_pnl += fill.pnl
                fills_all.append(fill.__dict__)
            curve.append({"date": day, "equity": self.broker.equity(),
                          "day_pnl": day_pnl, "n_trades": len(plans),
                          "halted": st.halted})
        return {"trades": pd.DataFrame(fills_all), "equity": pd.DataFrame(curve),
                "params": self.p}

    # ---- live / paper: what would the bot do for the next session? ----
    def plan_today(self) -> list[TradePlan]:
        from ..scanner import scan
        wl = scan(self.p)
        if wl is None or wl.empty:
            return []
        wl = wl.rename(columns={"stop": "stop_px", "target": "target_px"})
        equity = self.broker.equity()
        st = self.risk.new_day(equity)
        return self.risk.build_plans(wl, equity, st)
