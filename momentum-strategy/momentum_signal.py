"""
momentum_signal.py
------------------
Computes cross-sectional momentum signals for S&P 500 constituents.

Signal definition (Jegadeesh & Titman 1993):
  - Lookback : trailing 12-month total log return
  - Skip     : most recent 1 month (avoids short-term reversal)
  - Ranking  : cross-sectional percentile rank each rebalance date

Outputs (data/):
  - momentum_signals.csv  — monthly signal scores (date × ticker)
  - long_portfolio.csv    — top-decile tickers per month
  - short_portfolio.csv   — bottom-decile tickers per month
"""

import os
import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
LOOKBACK_MONTHS = 12     # formation period
SKIP_MONTHS     = 1      # skip most recent N months (short-term reversal filter)
TOP_QUANTILE    = 0.10   # long the top 10%
BOT_QUANTILE    = 0.10   # short the bottom 10%
DATA_DIR        = os.path.join(os.path.dirname(__file__), "data")


def load_log_returns(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "sp500_log_returns.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"Loaded log returns: {df.shape}")
    return df


def compute_momentum_scores(log_returns: pd.DataFrame,
                             lookback: int = LOOKBACK_MONTHS,
                             skip: int = SKIP_MONTHS) -> pd.DataFrame:
    """
    Computes cross-sectional momentum scores at each month-end.

    For each rebalance date t:
      score = cumulative log return over [t - lookback - skip, t - skip]

    Returns
    -------
    pd.DataFrame
        Monthly momentum scores (rebalance date × ticker), ranked 0-1.
    """
    # Resample to month-end cumulative returns for signal construction
    monthly_returns = log_returns.resample("ME").sum()

    scores = {}
    dates = monthly_returns.index[lookback + skip:]   # need enough history

    for date in dates:
        # Window: [t - lookback - skip + 1 : t - skip]  (skip the last `skip` months)
        end_idx   = monthly_returns.index.get_loc(date) - skip
        start_idx = end_idx - lookback

        if start_idx < 0:
            continue

        window = monthly_returns.iloc[start_idx:end_idx]
        cumret = window.sum()                          # total log return over window

        # Cross-sectional rank → percentile [0, 1]
        scores[date] = cumret.rank(pct=True)

    signals = pd.DataFrame(scores).T
    signals.index.name = "date"
    print(f"Momentum signal matrix: {signals.shape}  (months × tickers)")
    return signals


def build_portfolios(signals: pd.DataFrame,
                     top_q: float = TOP_QUANTILE,
                     bot_q: float = BOT_QUANTILE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separates signals into long (top decile) and short (bottom decile) portfolios.

    Returns
    -------
    long_df, short_df : pd.DataFrame
        Boolean masks, True where the ticker is in the portfolio that month.
    """
    long_mask  = signals >= (1 - top_q)
    short_mask = signals <= bot_q
    return long_mask, short_mask


def main():
    log_returns = load_log_returns(DATA_DIR)
    signals     = compute_momentum_scores(log_returns)
    long_mask, short_mask = build_portfolios(signals)

    # Save outputs
    signals.to_csv(os.path.join(DATA_DIR, "momentum_signals.csv"))
    long_mask.to_csv(os.path.join(DATA_DIR, "long_portfolio.csv"))
    short_mask.to_csv(os.path.join(DATA_DIR, "short_portfolio.csv"))

    print("\nSaved: momentum_signals.csv, long_portfolio.csv, short_portfolio.csv")

    # Quick sanity check
    avg_long  = long_mask.sum(axis=1).mean()
    avg_short = short_mask.sum(axis=1).mean()
    print(f"Avg long  positions/month : {avg_long:.1f}")
    print(f"Avg short positions/month : {avg_short:.1f}")


if __name__ == "__main__":
    main()
