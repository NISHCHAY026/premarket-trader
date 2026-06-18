"""Risk manager: position sizing + hard safety limits with a daily kill switch."""
from __future__ import annotations

from dataclasses import dataclass

from ..config import Params
from .broker import TradePlan


@dataclass
class RiskState:
    day_start_equity: float
    realized_today: float = 0.0
    trades_today: int = 0
    halted: bool = False
    halt_reason: str = ""


class RiskManager:
    def __init__(self, p: Params, daily_max_loss_frac: float = 0.06):
        self.p = p
        self.daily_max_loss_frac = daily_max_loss_frac

    def new_day(self, equity: float) -> RiskState:
        return RiskState(day_start_equity=equity)

    def can_trade(self, st: RiskState) -> bool:
        if st.halted:
            return False
        if st.realized_today <= -self.daily_max_loss_frac * st.day_start_equity:
            st.halted, st.halt_reason = True, "daily max loss hit"
            return False
        if st.trades_today >= self.p.max_positions:
            return False
        return True

    def size(self, equity: float, entry: float, stop: float, buying_power: float) -> int:
        risk_per_share = entry - stop
        if risk_per_share <= 0 or entry <= 0:
            return 0
        shares = min(
            (equity * self.p.risk_per_trade) / risk_per_share,   # fixed-fractional risk
            (equity * self.p.max_pos_frac) / entry,              # concentration cap
            buying_power / entry,                                # no leverage
        )
        return int(max(0, shares))

    def build_plans(self, candidates, equity: float, st: RiskState) -> list[TradePlan]:
        """candidates: rows with entry/stop_px/target_px/ticker/ml_proba (best-first).

        Enforces the daily-loss budget UP FRONT: the sum of each plan's maximum loss
        (shares * (entry - stop)) is capped at daily_max_loss_frac * equity. Because every
        position carries a hard stop, this bounds the worst-case loss for the whole day
        before a single order is placed — the kill switch is structural, not reactive.
        """
        plans, buying_power = [], equity
        risk_budget = self.daily_max_loss_frac * equity
        used_risk = 0.0
        for _, r in candidates.iterrows():
            if not self.can_trade(st):
                break
            entry = float(r["entry"])
            stop = float(r["stop_px"])
            target = float(r["target_px"])
            per_share_risk = entry - stop
            if per_share_risk <= 0:
                continue
            shares = self.size(equity, entry, stop, buying_power)
            if shares <= 0:
                continue
            # clamp to the remaining daily-loss budget
            remaining = risk_budget - used_risk
            if shares * per_share_risk > remaining:
                shares = int(remaining / per_share_risk)
            if shares <= 0:
                break
            used_risk += shares * per_share_risk
            buying_power -= shares * entry
            st.trades_today += 1
            plans.append(TradePlan(r["ticker"], shares, entry, stop, target,
                                   float(r.get("ml_proba", float("nan")))))
        return plans

    def record(self, st: RiskState, pnl: float) -> None:
        st.realized_today += pnl
