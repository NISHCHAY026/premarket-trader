"""Premarket Trader — a premarket-momentum (gap-and-go) day-trading model.

Pipeline: data -> features (no-lookahead) -> walk-forward ML -> event-driven
backtest -> honest out-of-sample evaluation -> live scanner.
"""
__version__ = "0.1.0"

from .config import Params  # noqa: F401
