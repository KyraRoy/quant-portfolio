# Momentum Strategy with Backtester

A cross-sectional momentum strategy applied to S&P 500 constituents, with a full event-driven backtester and performance analytics.

## Strategy Overview

**Cross-sectional momentum** ranks stocks by their past 12-month return (skipping the most recent month to avoid short-term reversal) and goes long the top decile, short the bottom decile. Portfolios are rebalanced monthly.

| Parameter | Value |
|-----------|-------|
| Universe | S&P 500 |
| Lookback | 12 months (skip last 1 month) |
| Rebalance | Monthly |
| Long/Short | Top / Bottom 10% |
| Data | 2018 – 2024 |

## Project Structure

```
momentum-strategy/
├── data/
│   ├── sp500_prices.csv          # Adjusted closing prices
│   └── sp500_log_returns.csv     # Daily log returns
├── data_fetcher.py               # Fetch, clean, and save price data
├── momentum_signal.py            # Compute cross-sectional momentum rankings
├── backtester.py                 # Event-driven portfolio backtester
├── performance.py                # Sharpe, drawdown, factor attribution
├── requirements.txt
└── README.md
```

## Quickstart

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# 1. Fetch and clean data
python data_fetcher.py

# 2. Compute signals
python momentum_signal.py

# 3. Run backtest
python backtester.py

# 4. View performance report
python performance.py
```

## Results

Backtest period: 2018-01-02 → 2024-12-30 · Transaction costs: 10 bps round-trip · Universe: 484 S&P 500 constituents

| Metric | Long-Only | Short-Only | Long-Short |
|--------|-----------|------------|------------|
| CAGR | 18.98% | 15.86% | -1.04% |
| Annualized Vol | 23.56% | 27.24% | 23.43% |
| Sharpe Ratio | 0.686 | 0.529 | -0.096 |
| Max Drawdown | -38.58% | -44.85% | -47.97% |
| Calmar Ratio | 0.492 | 0.354 | -0.022 |
| Daily Hit Rate | 47.1% | 44.1% | 44.8% |

> **Note on long-short performance:** The 2018–2024 window included two major momentum crashes — the COVID reversal (2020) and the 2022 rate shock — both of which hit crowded momentum longs while lifting beaten-down shorts. The long-only leg captured most of the underlying equity risk premium (18.98% CAGR, Sharpe 0.69), consistent with the academic literature on momentum in individual stocks.

## Dependencies

- `pandas`, `numpy` — data manipulation
- `yfinance` — price data
- `scipy`, `statsmodels` — statistical analysis
- `matplotlib` — visualization
- `beautifulsoup4`, `requests` — Wikipedia scraping
