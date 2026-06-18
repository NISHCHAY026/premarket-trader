"""Live/paper trading bot built on the Premarket Trader model.

Safety model: the bot trades PAPER by default. The Robinhood (real-money) broker is
hard-gated and never constructed unless the operator passes explicit approval flags.
See `broker.RobinhoodBroker` and `scripts/run_bot.py`.
"""
