"""
data_fetcher.py
---------------
Downloads Fama-French factor data and prepares the return series to explain.

Factor data comes from Kenneth French's Data Library via pandas_datareader.
We fetch three datasets:
  - F-F_Research_Data_5_Factors_2x3_daily  (FF5: Mkt-RF, SMB, HML, RMW, CMA + RF)
  - F-F_Momentum_Factor_daily               (UMD: Up Minus Down momentum factor)

The dependent return series are pulled from Project 1 (momentum strategy):
  - long       : top-decile momentum long portfolio
  - long_short : dollar-neutral long-short portfolio

We also run the model on the individual stocks from Project 2 so the
dashboard can show factor exposures for each ticker.

Outputs (data/):
  - ff_factors.csv      — daily FF5 + UMD factors and risk-free rate
  - portfolio_returns.csv — momentum long and long-short daily returns
  - stock_returns.csv   — individual stock returns (from risk-dashboard)
"""

import io
import os
import zipfile
import numpy as np
import pandas as pd
import requests

DATA_DIR        = os.path.join(os.path.dirname(__file__), "data")
START           = "2018-01-01"
END             = "2024-12-31"

# Paths to sibling project data (relative to this file's directory)
_BASE           = os.path.dirname(os.path.dirname(__file__))
MOMENTUM_RETURNS = os.path.join(_BASE, "momentum-strategy", "data", "backtest_returns.csv")
STOCK_RETURNS    = os.path.join(_BASE, "risk-dashboard",    "data", "log_returns.csv")

# Kenneth French Data Library — direct ZIP URLs (no third-party library needed)
FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
UMD_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_daily_CSV.zip"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _download_french_zip(url: str, factor_names: list[str]) -> pd.DataFrame:
    """
    Downloads a ZIP from Kenneth French's website, extracts the CSV inside,
    skips the descriptive header rows, and returns a clean daily DataFrame.
    """
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
        raw = zf.read(csv_name).decode("utf-8", errors="replace")

    # The CSV has descriptive text rows before the data; data lines are
    # comma-separated and start with an 8-digit date (YYYYMMDD).
    lines = raw.splitlines()
    data_lines = [l for l in lines if l and l.split(",")[0].strip().isdigit()
                  and len(l.split(",")[0].strip()) == 8]

    df = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        header=None,
        names=["Date"] + factor_names,
    )
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df = df.set_index("Date")
    df = df.apply(pd.to_numeric, errors="coerce") / 100   # percent → decimal
    return df


# ── 1. Download FF factors ─────────────────────────────────────────────────────

def fetch_ff_factors(start: str, end: str) -> pd.DataFrame:
    """
    Downloads daily FF5 factors + momentum (UMD) directly from Ken French's
    Data Library ZIPs.

    Factors returned (all in decimal, not percent):
      Mkt-RF  — excess market return
      SMB     — Small Minus Big (size)
      HML     — High Minus Low (value)
      RMW     — Robust Minus Weak (profitability)
      CMA     — Conservative Minus Aggressive (investment)
      UMD     — Up Minus Down (momentum)
      RF      — daily risk-free rate
    """
    print("Downloading FF5 factors from Ken French's Data Library...")
    ff5 = _download_french_zip(FF5_URL, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])

    print("Downloading momentum factor (UMD)...")
    umd = _download_french_zip(UMD_URL, ["UMD"])

    factors = ff5.join(umd, how="inner")
    factors = factors.loc[start:end]
    factors.index.name = "Date"

    print(f"  Factors shape : {factors.shape}")
    print(f"  Columns       : {factors.columns.tolist()}")
    print(f"  Date range    : {factors.index[0].date()} → {factors.index[-1].date()}")
    return factors


# ── 2. Load portfolio returns ──────────────────────────────────────────────────

def load_portfolio_returns() -> pd.DataFrame:
    """
    Loads the momentum strategy's backtest returns from Project 1.
    Trims leading zeros (pre-signal warm-up period).
    """
    returns = pd.read_csv(MOMENTUM_RETURNS, index_col=0, parse_dates=True)
    first_active = (returns["long"] != 0).idxmax()
    returns = returns.loc[first_active:]
    print(f"Portfolio returns: {returns.shape}  ({first_active.date()} → {returns.index[-1].date()})")
    return returns


def load_stock_returns() -> pd.DataFrame:
    """Loads individual stock log returns from Project 2."""
    returns = pd.read_csv(STOCK_RETURNS, index_col=0, parse_dates=True)
    print(f"Stock returns: {returns.shape}")
    return returns


# ── 3. Save ───────────────────────────────────────────────────────────────────

def save(factors: pd.DataFrame,
         port_returns: pd.DataFrame,
         stock_returns: pd.DataFrame) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    factors.to_csv(os.path.join(DATA_DIR, "ff_factors.csv"))
    port_returns.to_csv(os.path.join(DATA_DIR, "portfolio_returns.csv"))
    stock_returns.to_csv(os.path.join(DATA_DIR, "stock_returns.csv"))
    print(f"\nSaved to {DATA_DIR}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    factors       = fetch_ff_factors(START, END)
    port_returns  = load_portfolio_returns()
    stock_returns = load_stock_returns()
    save(factors, port_returns, stock_returns)

    print("\n── Summary ─────────────────────────────────────────")
    print(f"  FF factors    : {factors.shape[0]} days, {factors.shape[1]} factors")
    print(f"  Portfolio legs: {port_returns.columns.tolist()}")
    print(f"  Stocks        : {stock_returns.shape[1]} tickers")
    print("────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
