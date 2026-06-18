"""Strategy configuration.

Defaults encode a premarket-momentum ("gap-and-go") day-trading methodology:
low-priced names gapping up on high relative volume with a catalyst, entered at
the open, managed intraday with a fixed 2:1 profit-to-loss ratio, exited same day.
Values are tunable; the walk-forward optimizer searches the selection thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


@dataclass
class Params:
    # ---- data / universe ----
    start: str = "2017-01-01"
    end: str | None = None
    universe_csv: str | None = None          # None -> built-in sample universe

    # ---- what counts as a "gap event" worth modeling ----
    event_gap_min: float = 0.02              # >=2% overnight gap to enter the candidate pool
    price_min: float = 1.0                   # pillar 1: low-priced
    price_max: float = 20.0
    min_dollar_vol: float = 1_000_000.0      # prior-day $ volume liquidity floor

    # ---- selection filters (tuned by walk-forward) ----
    gap_min: float = 0.04                    # min overnight gap to actually trade
    rvol_min: float = 1.5                    # prior-day relative-volume surge proxy
    ml_min_proba: float = 0.50               # min model win-probability to trade
    max_positions: int = 5                   # max trades taken per day (best-ranked)

    # ---- trade management (fixed at the methodology's signature 2:1) ----
    stop_pct: float = 0.08                   # stop = open*(1-stop_pct)
    profit_ratio: float = 2.0                # target = open*(1+stop_pct*profit_ratio)

    # ---- sizing / risk ----
    account: float = 30_000.0
    risk_per_trade: float = 0.01             # fraction of equity risked per trade
    max_pos_frac: float = 0.50               # cap a position at this fraction of equity

    # ---- costs ----
    slippage_bps: float = 10.0               # per side (basis points of notional)
    commission_per_share: float = 0.0

    # ---- risk layer (v3): regime sizing + circuit breakers ----
    # VALIDATED AS OPT-IN, DEFAULT OFF. Ablation (docs/STRATEGY.md §5c) showed the full
    # layer hurts risk-adjusted return; only regime_sizing helps headline return/Sharpe,
    # at the cost of higher drawdown. Defaults below reproduce the shipped v2 behaviour.
    regime_sizing: bool = False              # scale size by market "temperature" (breadth)
    regime_lookback: int = 20                # trailing window for the breadth median
    icebreaker: bool = False                 # quarter -> half -> full size on the day's first picks
    daily_max_loss: float = 10.0             # halt past this fractional loss (10.0 = effectively off)
    daily_goal_mult: float = 2.0             # also halt once up daily_goal_mult * daily_max_loss
    three_strike: int = 0                    # halt the day after this many losing trades (0 = off)

    # ---- model ----
    use_ml: bool = True

    # ---- walk-forward ----
    train_frac: float = 0.6                  # chronological train fraction for param search
    ml_train_years: float = 3.0              # rolling ML train window
    ml_test_months: float = 6.0              # ML test (out-of-fold) window
    embargo_days: int = 3                    # gap between ML train and test

    @property
    def target_pct(self) -> float:
        return self.stop_pct * self.profit_ratio

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Params":
        import yaml
        raw = yaml.safe_load(Path(path).read_text()) or {}
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in valid})
