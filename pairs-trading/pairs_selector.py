"""
pairs_selector.py
-----------------
Finds statistically cointegrated pairs from the S&P 500 universe.

Pipeline:
  1. Load adjusted prices from Project 1 (484 tickers, 2018-2024)
  2. Fetch GICS sector labels from Wikipedia
  3. For each sector: screen pairs with Pearson correlation > CORR_THRESHOLD
  4. Run Engle-Granger cointegration test on log prices for each screened pair
  5. Keep pairs with p-value < PVAL_THRESHOLD and half-life in a tradeable range
  6. Save top pairs ranked by p-value

Why log prices?
  Cointegration is tested on log prices (not returns) because we want to model
  a stationary *spread* between the levels. Returns are already stationary by
  construction and contain no useful long-run relationship information.

Why within-sector only?
  Pairs from the same sector share common fundamental drivers (commodity prices,
  regulatory environment, consumer trends). Cross-sector pairs that pass
  statistical tests often reflect spurious correlation that breaks down
  out-of-sample.

Outputs (data/):
  - sector_labels.csv  — ticker → GICS sector mapping
  - candidate_pairs.csv — all pairs passing correlation screen
  - top_pairs.csv       — final pairs passing cointegration test, ranked by p-value
"""

import io
import os
import time
import warnings
import numpy as np
import pandas as pd
import requests
from itertools import combinations
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
CORR_THRESHOLD  = 0.85    # minimum Pearson correlation to screen pairs
PVAL_THRESHOLD  = 0.05    # Engle-Granger p-value cutoff
MIN_HALF_LIFE   = 5       # trading days — too fast = noise
MAX_HALF_LIFE   = 126     # trading days — too slow = untradeable (~6 months)
TOP_N           = 20      # number of pairs to keep in final output
FORMATION_END   = "2021-12-31"   # use first 4 years to find pairs
DATA_DIR        = os.path.join(os.path.dirname(__file__), "data")

_BASE           = os.path.dirname(os.path.dirname(__file__))
PRICES_PATH     = os.path.join(_BASE, "momentum-strategy", "data", "sp500_prices.csv")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# ── 1. Load prices ─────────────────────────────────────────────────────────────

def load_prices() -> pd.DataFrame:
    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    # Use formation period only for pair selection (avoids look-ahead bias)
    prices = prices.loc[:FORMATION_END]
    # Drop any column with >5% missing in formation period
    prices = prices.dropna(thresh=int(len(prices) * 0.95), axis=1)
    prices = prices.ffill()
    print(f"Prices loaded: {prices.shape}  (formation period: {prices.index[0].date()} → {prices.index[-1].date()})")
    return prices


# ── 2. Fetch sector labels ─────────────────────────────────────────────────────

def fetch_sector_labels() -> pd.Series:
    """Returns a Series: ticker → GICS sector."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    print("Fetching sector labels from Wikipedia...")
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    table = pd.read_html(io.StringIO(resp.text))[0]
    table["Symbol"] = table["Symbol"].str.replace(".", "-", regex=False)
    sector_map = table.set_index("Symbol")["GICS Sector"]
    print(f"  Got {len(sector_map)} tickers across {sector_map.nunique()} sectors.")
    return sector_map


# ── 3. Correlation screen ──────────────────────────────────────────────────────

def correlation_screen(log_prices: pd.DataFrame,
                        sector_map: pd.Series,
                        threshold: float = CORR_THRESHOLD) -> list[tuple]:
    """
    For each sector, compute pairwise return correlations.
    Returns all (ticker_A, ticker_B, correlation) tuples above threshold.
    """
    returns     = log_prices.diff().dropna()
    candidates  = []
    sectors     = sector_map.unique()

    for sector in sectors:
        tickers = [t for t in sector_map[sector_map == sector].index
                   if t in log_prices.columns]
        if len(tickers) < 2:
            continue

        corr = returns[tickers].corr()
        for a, b in combinations(tickers, 2):
            c = corr.loc[a, b]
            if c >= threshold:
                candidates.append((a, b, round(c, 4)))

    print(f"  Correlation screen: {len(candidates)} pairs with ρ ≥ {threshold}")
    return candidates


# ── 4. Cointegration test ──────────────────────────────────────────────────────

def estimate_half_life(spread: pd.Series) -> float:
    """
    Estimates the mean-reversion half-life of a spread via AR(1) regression.
    Regresses Δspread on lagged spread; half-life = -ln(2) / ln(β).
    """
    lag    = spread.shift(1).dropna()
    delta  = spread.diff().dropna()
    aligned = pd.concat([delta, lag], axis=1).dropna()
    X = sm.add_constant(aligned.iloc[:, 1])
    res = sm.OLS(aligned.iloc[:, 0], X).fit()
    beta = res.params.iloc[1]
    if beta >= 0 or beta <= -1:
        return np.nan
    return -np.log(2) / np.log(1 + beta)


def cointegration_screen(log_prices: pd.DataFrame,
                          candidates: list[tuple]) -> pd.DataFrame:
    """
    Runs Engle-Granger cointegration test on each candidate pair.
    Keeps pairs with p-value < PVAL_THRESHOLD and half-life in tradeable range.
    """
    results = []
    print(f"\nRunning cointegration tests on {len(candidates)} pairs...")

    for i, (a, b, corr) in enumerate(candidates):
        if i % 50 == 0 and i > 0:
            print(f"  {i}/{len(candidates)} tested...")

        pa = log_prices[a].dropna()
        pb = log_prices[b].dropna()
        common = pa.index.intersection(pb.index)
        if len(common) < 252:
            continue

        pa, pb = pa[common], pb[common]

        # Engle-Granger: regress pa on pb, test residuals for stationarity
        _, pval, _ = coint(pa, pb)

        if pval >= PVAL_THRESHOLD:
            continue

        # Estimate hedge ratio and spread
        X = sm.add_constant(pb)
        res = sm.OLS(pa, X).fit()
        hedge_ratio = res.params.iloc[1]
        spread = pa - hedge_ratio * pb

        half_life = estimate_half_life(spread)
        if np.isnan(half_life) or not (MIN_HALF_LIFE <= half_life <= MAX_HALF_LIFE):
            continue

        results.append({
            "ticker_A":    a,
            "ticker_B":    b,
            "correlation": corr,
            "coint_pval":  round(pval, 5),
            "hedge_ratio": round(hedge_ratio, 4),
            "half_life":   round(half_life, 1),
            "spread_mean": round(spread.mean(), 4),
            "spread_std":  round(spread.std(), 4),
        })

    df = pd.DataFrame(results).sort_values("coint_pval")
    print(f"  Passed cointegration filter: {len(df)} pairs")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    prices      = load_prices()
    log_prices  = np.log(prices)
    sector_map  = fetch_sector_labels()

    candidates  = correlation_screen(log_prices, sector_map)

    cand_df = pd.DataFrame(candidates, columns=["ticker_A", "ticker_B", "correlation"])

    pairs_df = cointegration_screen(log_prices, candidates)

    top_pairs = pairs_df.head(TOP_N)

    os.makedirs(DATA_DIR, exist_ok=True)
    sector_map.to_csv(os.path.join(DATA_DIR, "sector_labels.csv"), header=True)
    cand_df.to_csv(os.path.join(DATA_DIR, "candidate_pairs.csv"), index=False)
    top_pairs.to_csv(os.path.join(DATA_DIR, "top_pairs.csv"), index=False)

    print(f"\n── Top {len(top_pairs)} Cointegrated Pairs ──────────────────────────")
    print(top_pairs[["ticker_A","ticker_B","correlation","coint_pval","half_life"]].to_string(index=False))
    print("─────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
