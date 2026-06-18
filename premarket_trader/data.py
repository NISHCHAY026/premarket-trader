"""Universe definition and OHLCV download (yfinance), cached to parquet."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATA_DIR

# Default universe tilted toward the methodology's target population: low-priced,
# high-volatility, frequently-gapping US names (EV/clean-energy, crypto miners,
# meme/retail, SPAC/growth, small biotech, China ADRs), plus liquid anchors.
# Survivorship-biased (current listings only); delisted symbols are skipped with a
# warning. Override with --universe-csv for a larger / point-in-time universe.
SAMPLE_UNIVERSE = [
    # liquid anchors / occasional gappers
    "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "AMZN", "META", "GOOGL", "NFLX", "INTC",
    "MU", "AVGO", "QCOM", "PYPL", "BAC", "F", "T", "BABA", "SMCI", "ARM",
    # EV / clean energy (mostly low-priced)
    "NIO", "XPEV", "LI", "LCID", "RIVN", "NKLA", "GOEV", "RIDE", "CHPT", "BLNK",
    "EVGO", "PLUG", "FCEL", "BE", "RUN", "SPWR", "QS", "MVIS", "LAZR", "OUST",
    "ACHR", "JOBY", "ENPH", "FSLR",
    # crypto / miners
    "COIN", "MARA", "RIOT", "CLSK", "HUT", "BITF", "BTBT", "CIFR", "WULF", "HIVE",
    "GREE", "MSTR", "BTDR",
    # meme / retail / cannabis
    "GME", "AMC", "BB", "NOK", "KOSS", "EXPR", "SNDL", "TLRY", "CGC", "ACB",
    "CLOV", "WISH", "SPCE", "SIRI",
    # SPAC / growth / AI / recent
    "SOFI", "HOOD", "AFRM", "UPST", "OPEN", "DKNG", "PLTR", "RBLX", "PATH", "AI",
    "IONQ", "RGTI", "QUBT", "SOUN", "BBAI", "NNDM", "RKLB", "ASTS", "LUNR", "DNA",
    "RDW", "U", "SNAP", "PINS", "ROKU", "CVNA", "BYND", "PTON",
    # small biotech / pharma (gap-prone)
    "OCGN", "SAVA", "NVAX", "MRNA", "BNTX", "CYTK", "VKTX", "BCRX", "ATAI", "CRBP",
    # China ADRs / intl
    "GRAB", "NU", "BILI", "IQ", "FUTU", "TIGR", "TME",
    # commodities / miners / energy (volatile)
    "RIG", "CLF", "AA", "VALE", "GOLD", "KGC", "HL", "AG", "CDE", "UEC",
    "CCJ", "SMR", "OKLO", "DNN", "UUUU", "OXY", "AAL", "CCL", "NCLH",
]


def load_universe(csv: str | None = None) -> list[str]:
    if csv:
        lines = Path(csv).read_text(encoding="utf-8").splitlines()
        return [ln.strip().upper() for ln in lines if ln.strip() and not ln.startswith("#")]
    return list(dict.fromkeys(SAMPLE_UNIVERSE))


def _to_long(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Normalise yfinance output (single- or multi-ticker) to a long table."""
    frames = []
    if isinstance(df.columns, pd.MultiIndex):
        level0 = set(df.columns.get_level_values(0))
        for t in tickers:
            if t not in level0:
                continue
            sub = df[t].copy()
            sub["ticker"] = t
            frames.append(sub)
        long = pd.concat(frames) if frames else pd.DataFrame()
    else:
        long = df.copy()
        long["ticker"] = tickers[0]
    if long.empty:
        return long
    long = long.reset_index().rename(columns={c: str(c).lower() for c in long.columns})
    long = long.rename(columns={"index": "date", "datetime": "date"})
    keep = ["date", "ticker", "open", "high", "low", "close", "volume"]
    long = long[[c for c in keep if c in long.columns]].dropna(subset=["open", "close"])
    return long.sort_values(["ticker", "date"]).reset_index(drop=True)


def download_prices(tickers: list[str], start: str, end: str | None = None,
                    cache: bool = True, batch: int = 40) -> pd.DataFrame:
    """Daily OHLCV (split/dividend-adjusted) for the universe, cached to parquet."""
    import yfinance as yf

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    key = f"prices_{start}_{end or 'latest'}_{len(tickers)}.parquet"
    cache_file = DATA_DIR / key
    if cache and cache_file.exists():
        return pd.read_parquet(cache_file)

    frames = []
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        raw = yf.download(chunk, start=start, end=end, auto_adjust=True,
                          progress=False, group_by="ticker", threads=True)
        if raw is not None and not raw.empty:
            frames.append(_to_long(raw, chunk))
    long = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not long.empty:
        long["date"] = pd.to_datetime(long["date"])
        long = long.drop_duplicates(["ticker", "date"]).reset_index(drop=True)
        if cache:
            long.to_parquet(cache_file)
    return long
