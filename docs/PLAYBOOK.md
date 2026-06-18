# The Premarket Momentum Playbook

A generic, widely-taught approach to **premarket / opening-momentum day trading**
("gap-and-go"). This document states the methodology in neutral terms; the code in
this repo turns it into a backtestable, trainable model.

> Educational use only. Nothing here is financial advice. Day trading is risky and
> most participants lose money.

---

## 1. Stock selection — the "five pillars"

A name is a candidate only when it is genuinely *in play*. The classic filters:

1. **Low price** — roughly **$1–$20**. Cheap enough to move fast in percentage terms.
2. **Large percentage gain** — already **up strongly on the day** (a real gap), not drifting.
3. **High relative volume** — trading **many multiples** of its normal volume; unusual interest.
4. **Low float** — a **small tradable share count** (often < ~20M); thin supply amplifies moves.
5. **A catalyst** — **breaking news** (earnings, FDA, contract, offering, etc.) explaining the move.

The single best tell is being the **#1 percentage gainer** in the market — the most
*obvious* stock, where the crowd's attention concentrates.

## 2. Market environment (regime)

The same setup performs very differently depending on the tape. A practical gauge:
**how many names are up >100% today?** Several = a *hot* tape that rewards aggression;
none = a *cold* tape to trade small or sit out. Read the environment before sizing up.

## 3. Entry — the "first pullback"

Do **not** chase the initial vertical squeeze (that is buying a lottery ticket at the top).
Instead:

1. Let the stock squeeze up on its catalyst.
2. Wait for the **first pullback** (1–3 red candles) that **holds the 50% retracement** of the move.
3. Enter on the **first candle to make a new high** — the moment the trend shifts back up.
4. Confirm with: candlestick shape (bottoming tail bullish, topping tail bearish),
   a supportive momentum indicator (e.g. MACD positive), and order-flow on Level 2.

Entries cluster near **half-dollar and whole-dollar** levels (psychological strikes).
The second and third pullbacks can work too, but the **first** is highest-probability.

## 4. Exit and risk — the 2:1 rule

- **Stop loss:** the **low of the pullback** (or a tight arbitrary stop if that is too far).
- **Profit target:** a **2:1 profit-to-loss ratio** — aim to make twice what you risk
  (often a retest of the high of day). At 2:1 you only need to be right **~33%** of the
  time to break even, so even a 45–55% win rate is strongly profitable.
- **Breakout-or-bailout:** when the setup is right it works *immediately*. If it stalls
  right after entry, exit at once — don't wait for the full stop.
- **Scale out** into strength; sell into the extension, not the panic.

## 5. Position sizing and discipline

- **Match size to skill.** Beginners trade small (even smaller than feels right) and
  scale size up only as a documented edge is proven.
- **Starter position first.** Take a small "feeler" each day; if it works, press; if it
  fails, the caution flag is out — the tape is telling you it's a cold day.
- **Cut losses fast; never average down** a loser. One undisciplined add can erase weeks
  of gains.
- **Daily max loss / no-trade days.** Survive the cold stretches; protect the account so
  you're around when conditions turn hot.
- **Consistency over home runs.** The edge is a repeatable checklist executed with
  discipline, not any single trade.

---

## How this maps to the model

| Playbook concept | In the model |
|---|---|
| Five pillars (price, gap, RVOL, float\*, catalyst) | candidate filters + ML features |
| "Trade the obvious / highest-quality name" | ML win-probability ranking |
| Market regime ("how many up >100%") | `market_breadth` feature |
| First-pullback entry (intraday) | approximated by entry at the open (daily-bar proxy) |
| 2:1 profit-to-loss, breakout-or-bailout | fixed stop/target trade simulation |
| Starter size, risk per trade, cut losses | fixed-fractional sizing + hard stop |

\* Float and intraday pullback timing require data beyond free daily bars — see
[STRATEGY.md](STRATEGY.md) for what the backtest can and cannot capture.
