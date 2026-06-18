# The Trading Bot

The bot wraps the trained model in an execution loop that can **paper-trade** (default) or,
once explicitly approved, place **real orders on Robinhood**. The same selection + sizing
code drives the historical backtest and the live run, so a green backtest exercises the
exact logic that would trade.

> ⚠️ Real money is dangerous. Live mode is **off by default** and gated behind several
> deliberate confirmations. Read this whole page before considering it.

## Architecture

```
premarket_trader/bot/
  broker.py   Broker interface + PaperBroker (simulated) + RobinhoodBroker (hard-gated)
  risk.py     RiskManager: sizing, max positions, no-leverage, daily max-loss kill switch
  engine.py   BotEngine: daily cycle (scan -> size -> execute); backtest replay + plan_today
scripts/
  run_bot.py        paper (default) | live (gated)
  backtest_bot.py   replay the live code path over history
```

## Safety model (defense in depth)

1. **Paper by default.** `run_bot.py` with no flags uses `PaperBroker` — it never imports
   or contacts Robinhood and places no orders.
2. **Live construction is blocked** unless `RobinhoodBroker` receives `approved=True` **and**
   the exact phrase `I APPROVE LIVE TRADING`; otherwise it raises `PermissionError`.
3. **Credentials only from the environment** (`RH_USERNAME` / `RH_PASSWORD`) — never stored
   in code or committed.
4. **Dry-run even when live.** Live mode is a dry-run (validates the path, sends nothing)
   unless you *also* pass `--execute`.
5. **Hard caps.** Per-order USD cap, a max daily live-order count, fixed-fractional risk per
   trade, a concentration cap, no leverage (notional ≤ equity), and a **daily max-loss kill
   switch** that halts trading for the day.

## Backtesting the bot

`backtest_bot.py` replays the live decision/execution path through the engine over history
and cross-checks it against the research backtest:

```
research : Return 27.1% | CAGR 4.7% | Sharpe 0.92 | MaxDD -5.0% | Win 57.5% | PF 1.64 | Trades 120
bot loop : Return 23.1% | CAGR 4.1% | Sharpe 0.89 | MaxDD -5.0% | Win 57.5% | PF 1.56 | Trades 120
```

Same trades and win rate. The bot's slightly lower return is the **daily planned-risk
budget** (the kill switch) clamping position size on the most aggressive days — the safety
control working, not a logic discrepancy.

## Safety review

This bot's safety was checked by an adversarial multi-agent review (unauthorized-order
paths, risk-control enforcement, live-decision leakage). Outcome: **paper is safe; no
approval-gate bypass exists.** The HIGH findings it surfaced — the daily kill switch not
being enforced live, dry-run performing a real login — have been **fixed** (enforced
up-front planned-risk budget; dry-run is now genuinely network-free) and are covered by
tests in `tests/test_bot.py`.

## Usage

```bash
# Paper: print what the bot would trade next session (safe, no orders)
python scripts/run_bot.py

# Backtest the bot's live code path
python scripts/backtest_bot.py --start 2018-01-01

# LIVE (real money) — only after you've decided to, with eyes open:
set RH_USERNAME=you@example.com
set RH_PASSWORD=...
python scripts/run_bot.py --mode live --approve --confirm "I APPROVE LIVE TRADING"
#   dry-run by default; add --execute to actually send orders, --max-order-usd to cap size
```

## About linking Robinhood

Linking is **opt-in and manual**. When you decide to, you set your credentials as
environment variables and pass the approval flags above; nothing connects before that.
Even then, start in dry-run, keep `--max-order-usd` small, and watch it. This is a
daily-bar research strategy with real execution caveats (see
[STRATEGY.md §6](STRATEGY.md#6-limitations-read-these)) — treat live trading as an
experiment with money you can afford to lose, not a passive income system.
