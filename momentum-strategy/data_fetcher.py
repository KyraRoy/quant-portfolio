"""
data_fetcher.py
---------------
Fetches S&P 500 price history and prepares clean data for the momentum strategy.

Pipeline:
  1. Scrape current S&P 500 tickers from Wikipedia
  2. Download adjusted close prices via yfinance (2018-2024)
  3. Clean: drop tickers with >20% missing data, forward-fill the rest
  4. Compute log returns
  5. Save prices and log returns to data/

Outputs (data/):
  - sp500_prices.csv   — adjusted close prices (date × ticker)
  - sp500_log_returns.csv — daily log returns (date × ticker)
"""

import os
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from io import StringIO

# ── Configuration ──────────────────────────────────────────────────────────────
START_DATE = "2018-01-01"
END_DATE   = "2024-12-31"
MAX_MISSING_FRAC = 0.20        # drop tickers missing more than this fraction of days
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ── 1. Fetch S&P 500 tickers from Wikipedia ────────────────────────────────────

def fetch_sp500_tickers() -> list[str]:
    """
    Scrapes the S&P 500 constituent list from Wikipedia.

    Returns
    -------
    list[str]
        Sorted list of ticker symbols with '.' replaced by '-' to match
        yfinance conventions (e.g. 'BRK.B' → 'BRK-B').
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    print("Fetching S&P 500 tickers from Wikipedia...")

    # Use a browser-like User-Agent so Wikipedia doesn't return 403
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    # pandas read_html pulls all <table> elements; the first one is the constituent table
    tables = pd.read_html(StringIO(response.text))
    sp500_table = tables[0]

    tickers = (
        sp500_table["Symbol"]
        .str.replace(".", "-", regex=False)   # yfinance uses '-' not '.'
        .sort_values()
        .tolist()
    )
    print(f"  Found {len(tickers)} tickers.")
    return tickers


# ── 2. Download adjusted closing prices ────────────────────────────────────────

def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Downloads adjusted closing prices for all tickers via yfinance.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols compatible with yfinance.
    start, end : str
        Date range in 'YYYY-MM-DD' format (inclusive).

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame indexed by date, one column per ticker.
        Tickers that yfinance could not retrieve are silently dropped.
    """
    print(f"\nDownloading prices for {len(tickers)} tickers ({start} → {end})...")
    print("  This may take a few minutes on the first run.\n")

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,   # returns adjusted prices directly in 'Close'
        progress=True,
    )

    # yfinance returns a MultiIndex when >1 ticker; extract just 'Close'
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers[:1]

    prices.index = pd.to_datetime(prices.index)
    print(f"\n  Downloaded shape: {prices.shape}  (trading days × tickers)")
    return prices


# ── 3. Clean missing data ──────────────────────────────────────────────────────

def clean_prices(prices: pd.DataFrame, max_missing_frac: float = MAX_MISSING_FRAC) -> pd.DataFrame:
    """
    Removes tickers with excessive missing data and forward-fills the rest.

    Parameters
    ----------
    prices : pd.DataFrame
        Raw price DataFrame (date × ticker).
    max_missing_frac : float
        Tickers missing more than this fraction of trading days are dropped.

    Returns
    -------
    pd.DataFrame
        Cleaned price DataFrame.
    """
    n_days = len(prices)
    missing_frac = prices.isna().sum() / n_days

    # Drop high-missing tickers
    to_drop = missing_frac[missing_frac > max_missing_frac].index.tolist()
    if to_drop:
        print(f"\nDropping {len(to_drop)} tickers with >{max_missing_frac*100:.0f}% missing data:")
        print(f"  {to_drop}")
    else:
        print("\nNo tickers exceeded the missing-data threshold.")

    prices = prices.drop(columns=to_drop)

    # Forward-fill remaining gaps (e.g. halted trading days)
    prices = prices.ffill()

    # Drop any rows that are still all-NaN (e.g. leading rows before any ticker listed)
    prices = prices.dropna(how="all")

    print(f"  Clean price matrix: {prices.shape}  (trading days × tickers)")
    return prices


# ── 4. Compute log returns ─────────────────────────────────────────────────────

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Computes daily log returns from adjusted closing prices.

    log_return(t) = ln( price(t) / price(t-1) )

    The first row is dropped (NaN from the diff).

    Parameters
    ----------
    prices : pd.DataFrame
        Clean price DataFrame (date × ticker).

    Returns
    -------
    pd.DataFrame
        Log returns DataFrame, same shape minus the first row.
    """
    log_returns = np.log(prices / prices.shift(1)).dropna(how="all")
    print(f"\nLog returns shape: {log_returns.shape}")
    return log_returns


# ── 5. Save to CSV ─────────────────────────────────────────────────────────────

def save_data(prices: pd.DataFrame, log_returns: pd.DataFrame, data_dir: str) -> None:
    """
    Saves price and log-return DataFrames to CSV files.

    Parameters
    ----------
    prices : pd.DataFrame
        Clean adjusted close prices.
    log_returns : pd.DataFrame
        Daily log returns.
    data_dir : str
        Directory to write CSVs into (created if it doesn't exist).
    """
    os.makedirs(data_dir, exist_ok=True)

    prices_path      = os.path.join(data_dir, "sp500_prices.csv")
    log_returns_path = os.path.join(data_dir, "sp500_log_returns.csv")

    prices.to_csv(prices_path)
    log_returns.to_csv(log_returns_path)

    print(f"\nSaved:")
    print(f"  {prices_path}")
    print(f"  {log_returns_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    tickers     = fetch_sp500_tickers()
    raw_prices  = download_prices(tickers, START_DATE, END_DATE)
    prices      = clean_prices(raw_prices)
    log_returns = compute_log_returns(prices)
    save_data(prices, log_returns, DATA_DIR)

    # Quick sanity-check summary
    print("\n── Summary ──────────────────────────────────────────")
    print(f"  Date range  : {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"  Trading days: {len(prices)}")
    print(f"  Tickers kept: {prices.shape[1]}")
    print(f"  Log returns — mean: {log_returns.stack().mean():.6f}  "
          f"std: {log_returns.stack().std():.6f}")
    print("─────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
