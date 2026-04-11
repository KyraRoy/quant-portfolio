"""
data_fetcher.py
---------------
Downloads adjusted closing prices for a diversified 20-stock portfolio
plus SPY as benchmark.

Portfolio covers 10 GICS sectors so the correlation heatmap shows
meaningful cross-sector structure.

Outputs (data/):
  - prices.csv      — adjusted close prices (date × ticker)
  - log_returns.csv — daily log returns (date × ticker)
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf

# ── Configuration ──────────────────────────────────────────────────────────────
START_DATE = "2018-01-01"
END_DATE   = "2024-12-31"
DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")

# 20-stock diversified portfolio + SPY benchmark, one or two per GICS sector
PORTFOLIO = {
    "Technology":        ["AAPL", "MSFT", "NVDA"],
    "Communication":     ["GOOGL", "META"],
    "Consumer Discr.":   ["AMZN", "TSLA"],
    "Consumer Staples":  ["PG", "KO"],
    "Financials":        ["JPM", "GS"],
    "Healthcare":        ["JNJ", "UNH"],
    "Industrials":       ["CAT", "HON"],
    "Energy":            ["XOM", "CVX"],
    "Utilities":         ["NEE"],
    "Real Estate":       ["AMT"],
    "Benchmark":         ["SPY"],
}

ALL_TICKERS = [t for tickers in PORTFOLIO.values() for t in tickers]


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    print(f"Downloading {len(tickers)} tickers ({start} → {end})...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=True)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    prices.index = pd.to_datetime(prices.index)
    # Reorder columns to match PORTFOLIO order
    prices = prices[[t for t in tickers if t in prices.columns]]
    return prices


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    missing = prices.isna().sum() / len(prices)
    to_drop = missing[missing > 0.10].index.tolist()
    if to_drop:
        print(f"Dropping {to_drop} (>10% missing)")
        prices = prices.drop(columns=to_drop)
    prices = prices.ffill().dropna(how="all")
    print(f"Clean price matrix: {prices.shape}")
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna(how="all")


def save(prices: pd.DataFrame, log_returns: pd.DataFrame) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    prices.to_csv(os.path.join(DATA_DIR, "prices.csv"))
    log_returns.to_csv(os.path.join(DATA_DIR, "log_returns.csv"))
    print(f"Saved to {DATA_DIR}/")


def main():
    prices      = download_prices(ALL_TICKERS, START_DATE, END_DATE)
    prices      = clean_prices(prices)
    log_returns = compute_log_returns(prices)
    save(prices, log_returns)

    print("\n── Summary ──────────────────────────────────────")
    print(f"  Date range : {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"  Trading days: {len(prices)}")
    print(f"  Tickers    : {prices.columns.tolist()}")
    print("─────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
