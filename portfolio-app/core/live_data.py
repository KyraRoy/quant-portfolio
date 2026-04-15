"""
core/live_data.py
-----------------
Live market data integration via yfinance.

All public fetch functions are wrapped with @st.cache_data(ttl=86400) so
data refreshes once per calendar day on first access. Each function falls
back gracefully to the static CSVs in portfolio-app/data/ if the network
call fails, so the dashboard always renders even when yfinance is down.

Public API
----------
get_portfolio_prices(start)         -> prices for the 20-stock portfolio + SPY
get_portfolio_log_returns(start)    -> daily log returns for that portfolio
get_sp500_tickers()                 -> current S&P 500 constituent list
get_momentum_universe_prices(...)   -> last N months of prices for full S&P 500
compute_current_momentum_signal(prices, lookback, skip)  -> JT signal Series
get_pair_live_prices(a, b, years)   -> recent prices for a cointegrated pair
last_updated_badge()                -> HTML "● LIVE · YYYY-MM-DD" badge
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import requests
from io import StringIO

# ── Constants ──────────────────────────────────────────────────────────────────

# The 20 stocks used across the Risk and Monte Carlo dashboards, plus SPY
PORTFOLIO_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    "AMZN", "TSLA", "PG",   "KO",    "JPM",
    "GS",   "JNJ",  "UNH",  "CAT",   "HON",
    "XOM",  "CVX",  "NEE",  "AMT",   "SPY",
]

# Path to static fallback CSVs (portfolio-app/data/)
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ── Internal helpers ───────────────────────────────────────────────────────────

def _extract_close(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise yfinance output into a plain (date × ticker) DataFrame.
    Handles both the old single-level and the newer MultiIndex column layouts
    introduced in yfinance ≥0.2.38.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        # MultiIndex format: (field, ticker) — pull the 'Close' level
        prices = raw["Close"].copy()
    else:
        prices = raw.copy()

    # Ensure column names are plain strings (yfinance may return Ticker objects)
    prices.columns = [str(c) for c in prices.columns]
    prices.index = pd.to_datetime(prices.index)
    return prices


def _yf_download(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Thin, defensive wrapper around yf.download."""
    kwargs = dict(
        tickers=tickers,
        start=start,
        auto_adjust=True,
        progress=False,
    )
    if end:
        kwargs["end"] = end
    raw = yf.download(**kwargs)
    return _extract_close(raw)


# ── Portfolio prices (20 stocks + SPY) ────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner="Fetching live price data…")
def get_portfolio_prices(start: str = "2018-01-01") -> pd.DataFrame:
    """
    Adjusted close prices for the 20-stock portfolio + SPY from `start` to today.
    Falls back to stock_prices.csv if the download fails.
    """
    try:
        prices = _yf_download(PORTFOLIO_TICKERS, start)
        prices = prices.ffill().dropna(how="all")
        if prices.empty or len(prices) < 10:
            raise ValueError("Download returned insufficient data.")
        return prices
    except Exception:
        fallback = os.path.join(_DATA_DIR, "stock_prices.csv")
        return pd.read_csv(fallback, index_col=0, parse_dates=True)


@st.cache_data(ttl=86400, show_spinner=False)
def get_portfolio_log_returns(start: str = "2018-01-01") -> pd.DataFrame:
    """
    Daily log returns for the 20-stock portfolio.
    Derived from get_portfolio_prices — shares the same cache TTL.
    """
    prices = get_portfolio_prices(start)
    return np.log(prices / prices.shift(1)).dropna(how="all")


# ── S&P 500 universe ───────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner="Fetching S&P 500 constituent list…")
def get_sp500_tickers() -> list[str]:
    """
    Scrape the current S&P 500 constituent list from Wikipedia.
    Returns ticker symbols formatted for yfinance ('.' → '-').
    Raises on failure — callers should catch.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )}
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    return (
        tables[0]["Symbol"]
        .str.replace(".", "-", regex=False)
        .tolist()
    )


@st.cache_data(ttl=86400, show_spinner="Downloading S&P 500 price history for momentum signal — this runs once per day…")
def get_momentum_universe_prices(lookback_months: int = 14) -> pd.DataFrame | None:
    """
    Download the last `lookback_months` of price data for the full S&P 500
    universe. Used to compute today's JT momentum signal.

    Returns None if the download fails (caller should display a warning).
    """
    try:
        tickers = get_sp500_tickers()
    except Exception:
        return None

    end   = pd.Timestamp.today()
    start = (end - pd.DateOffset(months=lookback_months)).strftime("%Y-%m-%d")

    try:
        prices = _yf_download(tickers, start)
        # Drop tickers with more than 20% missing data, then forward-fill
        min_rows = int(len(prices) * 0.80)
        prices = prices.dropna(thresh=min_rows, axis=1).ffill()
        if prices.empty:
            return None
        return prices
    except Exception:
        return None


# ── Momentum signal computation ────────────────────────────────────────────────

def compute_current_momentum_signal(
    prices: pd.DataFrame,
    lookback: int = 12,
    skip: int = 1,
) -> pd.Series:
    """
    Compute the Jegadeesh-Titman momentum signal from a price DataFrame.

    Signal = cumulative log return over the formation window:
      [-(lookback + skip) months, -skip months]
    i.e. trailing 12 months, skipping the most recent month.

    Parameters
    ----------
    prices : pd.DataFrame
        (date × ticker) adjusted close prices.
    lookback : int
        Formation window in months (default 12).
    skip : int
        Most-recent months to skip, avoiding short-term reversal (default 1).

    Returns
    -------
    pd.Series
        Cross-sectional percentile ranks [0, 1], sorted descending by score.
        Index = ticker symbols.
    """
    # Monthly log returns
    log_ret    = np.log(prices / prices.shift(1))
    monthly    = log_ret.resample("ME").sum()

    needed = lookback + skip + 1
    if len(monthly) < needed:
        return pd.Series(dtype=float)

    # Formation window: skip most recent `skip` months, use prior `lookback` months
    window  = monthly.iloc[-(lookback + skip):-skip]
    cum_ret = window.sum()

    # Cross-sectional percentile rank then sort best → worst
    scores = cum_ret.rank(pct=True).sort_values(ascending=False)
    return scores


# ── Pairs live data ────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def get_pair_live_prices(
    ticker_a: str,
    ticker_b: str,
    lookback_years: int = 3,
) -> pd.DataFrame | None:
    """
    Download the last `lookback_years` of daily adjusted close prices for a pair.
    Returns a two-column DataFrame (ticker_a, ticker_b), or None on failure.
    """
    start = (pd.Timestamp.today() - pd.DateOffset(years=lookback_years)).strftime("%Y-%m-%d")
    try:
        prices = _yf_download([ticker_a, ticker_b], start)
        # Subset to just the two columns we need and drop NaN rows
        cols_present = [c for c in [ticker_a, ticker_b] if c in prices.columns]
        if len(cols_present) < 2:
            return None
        return prices[cols_present].ffill().dropna()
    except Exception:
        return None


# ── UI helpers ─────────────────────────────────────────────────────────────────

def last_updated_badge() -> str:
    """
    Return an HTML badge showing today's date in green.
    Usage:  st.markdown(page_header_html + ld.last_updated_badge(), ...)
    """
    ts = pd.Timestamp.today().strftime("%Y-%m-%d")
    return (
        f'<span style="display:inline-block;background:rgba(52,211,153,0.15);'
        f'color:#34D399;border:1px solid rgba(52,211,153,0.3);border-radius:20px;'
        f'padding:2px 10px;font-size:0.75rem;font-weight:600;margin-left:8px;">'
        f'&#9679; LIVE &middot; {ts}</span>'
    )
