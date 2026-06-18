"""Performance metrics from a backtest result."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute(result: dict, account: float) -> dict:
    eq = result["equity"]
    tr = result["trades"]
    if eq is None or eq.empty:
        return {"trades": 0, "note": "no equity curve"}

    eq = eq.sort_values("date").reset_index(drop=True)
    equity = eq["equity"].to_numpy()
    start, end = account, equity[-1]
    days = max((eq["date"].iloc[-1] - eq["date"].iloc[0]).days, 1)
    years = days / 365.25

    # daily returns on equity for risk stats
    rets = pd.Series(equity, index=eq["date"]).pct_change().dropna()
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    downside = rets[rets < 0].std()
    sortino = (rets.mean() / downside * np.sqrt(252)) if downside and downside > 0 else 0.0

    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = dd.min() if len(dd) else 0.0

    cagr = (end / start) ** (1 / years) - 1 if start > 0 and end > 0 else -1.0

    n = len(tr)
    wins = tr[tr["pnl"] > 0] if n else tr
    losses = tr[tr["pnl"] <= 0] if n else tr
    gross_win = wins["pnl"].sum() if n else 0.0
    gross_loss = -losses["pnl"].sum() if n else 0.0
    win_rate = (len(wins) / n) if n else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    avg_win = wins["pnl"].mean() if len(wins) else 0.0
    avg_loss = losses["pnl"].mean() if len(losses) else 0.0
    expectancy = tr["pnl"].mean() if n else 0.0

    return {
        "start_equity": round(start, 2),
        "end_equity": round(end, 2),
        "total_return": round(end / start - 1, 4),
        "cagr": round(cagr, 4),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "max_drawdown": round(float(max_dd), 4),
        "years": round(years, 2),
        "trades": int(n),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(float(profit_factor), 3),
        "avg_win": round(float(avg_win), 2),
        "avg_loss": round(float(avg_loss), 2),
        "expectancy_per_trade": round(float(expectancy), 2),
    }


def summary_line(m: dict) -> str:
    if m.get("trades", 0) == 0:
        return "No trades."
    return (f"Return {m['total_return']*100:6.1f}% | CAGR {m['cagr']*100:5.1f}% | "
            f"Sharpe {m['sharpe']:.2f} | MaxDD {m['max_drawdown']*100:5.1f}% | "
            f"Win {m['win_rate']*100:4.1f}% | PF {m['profit_factor']:.2f} | "
            f"Trades {m['trades']}")
