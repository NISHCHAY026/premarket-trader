"""Persist backtest artifacts: metrics JSON, trades/equity CSVs, equity-curve PNG."""
from __future__ import annotations

import json
from pathlib import Path

from .config import RESULTS_DIR
from . import metrics as metrics_mod


def save_report(wf: dict, name: str = "walkforward", outdir: Path | None = None) -> Path:
    outdir = Path(outdir or RESULTS_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    payload = {
        "best_params": wf["best_params"].to_dict(),
        "metrics_train": wf.get("metrics_train"),
        "metrics_test": wf.get("metrics_test"),
        "metrics_full": wf.get("metrics_full"),
    }
    (outdir / f"{name}_metrics.json").write_text(json.dumps(payload, indent=2, default=str))

    res = wf["result_full"]
    if res["trades"] is not None and not res["trades"].empty:
        res["trades"].to_csv(outdir / f"{name}_trades.csv", index=False)
    if res["equity"] is not None and not res["equity"].empty:
        res["equity"].to_csv(outdir / f"{name}_equity.csv", index=False)
        _plot_equity(res, wf, outdir / f"{name}_equity.png")
    return outdir


def _plot_equity(res: dict, wf: dict, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    eq = res["equity"].sort_values("date")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eq["date"], eq["equity"], color="#1b9e77", lw=1.6)
    m = wf.get("metrics_full", {})
    ax.set_title(f"Premarket Trader — equity curve  ({metrics_mod.summary_line(m)})", fontsize=9)
    ax.set_ylabel("Equity ($)")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
