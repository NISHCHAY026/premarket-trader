"""Broker abstraction: a safe simulated broker (default) and a hard-gated live one.

Execution model for this daily-bar strategy: enter at the session open, attach a fixed
2:1 stop/target bracket, and exit by the close if neither triggers. Backtest and paper
modes resolve the outcome from that day's OHLC (pessimistic: stop assumed first if both
the stop and target fall inside the day's range). The live (Robinhood) broker places real
orders and is disabled unless the operator explicitly approves it.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class TradePlan:
    symbol: str
    shares: int
    entry: float
    stop: float
    target: float
    proba: float = float("nan")


@dataclass
class Fill:
    date: object
    symbol: str
    shares: int
    entry: float
    exit: float
    reason: str
    pnl: float
    proba: float = float("nan")


class Broker(ABC):
    is_live: bool = False

    @abstractmethod
    def equity(self) -> float: ...

    @abstractmethod
    def execute_day_trade(self, plan: TradePlan, bar: dict) -> Fill:
        """Execute one day-trade. `bar` carries the day's open/high/low/close."""


def resolve_outcome(entry: float, stop: float, target: float, bar: dict) -> tuple[float, str]:
    """Pessimistic intraday resolution from a daily OHLC bar."""
    hit_stop = bar["low"] <= stop
    hit_tgt = bar["high"] >= target
    if hit_stop:                      # assume the stop fills first if both are in range
        return stop, "stop"
    if hit_tgt:
        return target, "target"
    return bar["close"], "close"


# ---------------------------------------------------------------------------
# Simulated broker (used for backtest + paper trading). No external connection.
# ---------------------------------------------------------------------------
class PaperBroker(Broker):
    is_live = False

    def __init__(self, starting_equity: float, slippage_bps: float = 10.0,
                 commission_per_share: float = 0.0):
        self._equity = float(starting_equity)
        self.slip = slippage_bps / 1e4
        self.commission = commission_per_share
        self.fills: list[Fill] = []

    def equity(self) -> float:
        return self._equity

    def execute_day_trade(self, plan: TradePlan, bar: dict) -> Fill:
        exit_px, reason = resolve_outcome(plan.entry, plan.stop, plan.target, bar)
        gross = plan.shares * (exit_px - plan.entry)
        costs = (plan.shares * (plan.entry + exit_px) * self.slip
                 + 2 * self.commission * plan.shares)
        pnl = gross - costs
        self._equity += pnl
        fill = Fill(bar.get("date"), plan.symbol, plan.shares, plan.entry,
                    exit_px, reason, pnl, plan.proba)
        self.fills.append(fill)
        return fill


# ---------------------------------------------------------------------------
# Live broker (Robinhood). HARD-GATED: never constructed without explicit approval.
# ---------------------------------------------------------------------------
class RobinhoodBroker(Broker):
    is_live = True
    CONFIRM_PHRASE = "I APPROVE LIVE TRADING"
    HARD_MAX_ORDER_USD = 1000.0       # absolute ceiling; --max-order-usd cannot exceed this

    def __init__(self, approved: bool = False, confirm_phrase: str = "",
                 max_order_usd: float = 300.0, max_daily_orders: int = 5,
                 dry_run: bool = True, dry_run_equity: float = 25_000.0):
        # Triple gate: a code flag, the exact typed phrase, and credentials in env.
        if not approved or confirm_phrase != self.CONFIRM_PHRASE:
            raise PermissionError(
                "Robinhood live trading is NOT approved. Construction blocked. "
                "Live trading requires explicit operator approval (see scripts/run_bot.py)."
            )
        if max_order_usd > self.HARD_MAX_ORDER_USD:
            raise ValueError(
                f"max_order_usd ${max_order_usd:.0f} exceeds the hard ceiling "
                f"${self.HARD_MAX_ORDER_USD:.0f}. Lower it or edit HARD_MAX_ORDER_USD.")
        if not (os.environ.get("RH_USERNAME") and os.environ.get("RH_PASSWORD")):
            raise RuntimeError("RH_USERNAME / RH_PASSWORD not set in environment.")
        self.max_order_usd = max_order_usd
        self.max_daily_orders = max_daily_orders
        self.dry_run = dry_run
        self.dry_run_equity = float(dry_run_equity)
        self._orders_today = 0
        self._notional_today = 0.0
        self._rh = None

    def _login(self):
        if self._rh is None:
            import robin_stocks.robinhood as rh  # lazy import; only when truly live
            rh.login(os.environ["RH_USERNAME"], os.environ["RH_PASSWORD"])
            self._rh = rh
        return self._rh

    def equity(self) -> float:
        # A dry run must be side-effect-free: never authenticate or hit the network.
        if self.dry_run:
            return self.dry_run_equity
        rh = self._login()
        prof = rh.profiles.load_portfolio_profile()
        val = prof.get("equity") or prof.get("market_value")
        if not val or float(val) <= 0:
            raise RuntimeError("Robinhood returned no/zero equity; refusing to size blind.")
        return float(val)

    def execute_day_trade(self, plan: TradePlan, bar: dict | None) -> Fill:
        if self._orders_today >= self.max_daily_orders:
            raise RuntimeError("Daily live-order cap reached; refusing further orders.")
        notional = plan.shares * plan.entry
        if notional > self.max_order_usd:
            raise RuntimeError(
                f"Order ${notional:.0f} exceeds the ${self.max_order_usd:.0f} per-order cap.")
        if self._notional_today + notional > self.max_order_usd * self.max_daily_orders:
            raise RuntimeError("Daily notional cap reached; refusing further orders.")
        self._orders_today += 1
        self._notional_today += notional
        # Dry run returns BEFORE any login/network — a genuinely safe rehearsal.
        if self.dry_run:
            return Fill(bar.get("date") if bar else None, plan.symbol, plan.shares,
                        plan.entry, plan.entry, "dry_run", 0.0, plan.proba)
        # Real order: market buy + protective stop. (Reached only with dry_run=False.)
        rh = self._login()
        rh.orders.order_buy_market(plan.symbol, plan.shares)
        rh.orders.order_sell_stop_loss(plan.symbol, plan.shares, round(plan.stop, 2))
        return Fill(None, plan.symbol, plan.shares, plan.entry, float("nan"),
                    "submitted", float("nan"), plan.proba)
