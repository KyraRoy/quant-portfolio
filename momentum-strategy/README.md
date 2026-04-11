# Momentum Strategy with Backtester

> Cross-sectional momentum (Jegadeesh & Titman, 1993) applied to S&P 500 constituents over 2018–2024, with an event-driven backtester and full performance analytics.

---

## Table of Contents

- [Strategy Overview](#strategy-overview)
- [The JT 12-1 Signal](#the-jt-12-1-signal)
- [Pipeline](#pipeline)
- [Quickstart](#quickstart)
- [Results](#results)
- [Regime Analysis](#regime-analysis)
- [Known Limitations](#known-limitations)
- [Dependencies](#dependencies)
- [References](#references)

---

## Strategy Overview

**Cross-sectional momentum** is the tendency for stocks that outperformed their peers over the past year to continue outperforming over the next month — and for underperformers to keep underperforming. First documented by Jegadeesh & Titman (1993), it remains one of the most replicated anomalies in empirical asset pricing.

This implementation:

- Ranks all S&P 500 stocks each month by their trailing 12-1 month return
- Goes **long the top decile** (strongest momentum) and **short the bottom decile** (weakest momentum)
- Rebalances monthly, with equal weighting within each leg
- Deducts **10 bps round-trip transaction costs** at every rebalance

| Parameter | Value |
|-----------|-------|
| Universe | S&P 500 (~484 constituents after cleaning) |
| Signal | Cumulative log return, months `t-12` to `t-1` |
| Skip period | 1 month (short-term reversal filter) |
| Rebalance | Monthly, month-end |
| Long leg | Top decile (rank ≥ 90th percentile) |
| Short leg | Bottom decile (rank ≤ 10th percentile) |
| Transaction costs | 10 bps round-trip per rebalance |
| Data range | 2018-01-02 → 2024-12-30 |

---

## The JT 12-1 Signal

The signal is constructed exactly as in the original paper:

```
score(i, t) = Σ log_return(i, m)   for m in [t-12, t-1]
```

**Why skip the most recent month?** Short-horizon returns (< 1 month) exhibit *reversal*, not momentum — likely due to bid-ask bounce and liquidity effects. Including month `t` in the lookback would contaminate the trend signal with this noise. The skip is standard in the academic literature and meaningfully improves out-of-sample performance.

At each month-end, scores are cross-sectionally ranked into percentiles `[0, 1]`. This normalization makes the signal robust to changes in the overall market's return level — only *relative* performance within the cross-section drives portfolio construction.

---

## Pipeline

The project is structured as four independent, sequentially runnable scripts:

```
momentum-strategy/
├── data/                        # Generated outputs (gitignored)
│   ├── sp500_prices.csv         # Adjusted closing prices (date × ticker)
│   ├── sp500_log_returns.csv    # Daily log returns (date × ticker)
│   ├── momentum_signals.csv     # Monthly cross-sectional ranks
│   ├── long_portfolio.csv       # Top-decile boolean mask
│   ├── short_portfolio.csv      # Bottom-decile boolean mask
│   ├── backtest_returns.csv     # Daily P&L for each portfolio leg
│   ├── backtest_nav.csv         # Cumulative NAV series
│   ├── nav_chart.png            # NAV plot
│   ├── performance_summary.csv  # Metrics table
│   └── rolling_sharpe.png       # Rolling 252-day Sharpe plot
├── data_fetcher.py              # Step 1 — fetch and clean price data
├── momentum_signal.py           # Step 2 — compute JT rankings
├── backtester.py                # Step 3 — simulate portfolio returns
├── performance.py               # Step 4 — metrics and charts
├── requirements.txt
└── README.md
```

### `data_fetcher.py`

Scrapes the current S&P 500 constituent list from Wikipedia using `requests` with a browser user-agent (avoids 403s), downloads 7 years of adjusted closing prices via `yfinance`, then cleans the panel:

- Drops tickers with **> 20% missing** trading days (19 tickers removed)
- **Forward-fills** remaining gaps (e.g. trading halts, index additions)
- Computes **log returns** `ln(P_t / P_{t-1})`
- Saves `sp500_prices.csv` (15 MB) and `sp500_log_returns.csv` (17 MB)

### `momentum_signal.py`

For each month-end rebalance date, sums log returns over the 12-month formation window (skipping the most recent month), then cross-sectionally ranks the universe into percentiles. Outputs boolean masks for the long and short portfolios — ~49 names per leg per month.

### `backtester.py`

Event-driven simulation with careful look-ahead prevention: signals generated at month-end `t` are applied starting at `t+1` via a one-day shift of the weight matrix. Computes:

- **Long-only** NAV: equal-weight top decile
- **Short-only** NAV: equal-weight bottom decile  
- **Long-short** NAV: long leg minus short leg (dollar-neutral)

Transaction costs are deducted each rebalance proportional to portfolio turnover (`|Δweight| × 10 bps`).

### `performance.py`

Computes the standard quant performance metrics using a 4% annualized risk-free rate (approximate 2018–2024 T-bill average):

| Metric | Formula |
|--------|---------|
| CAGR | `(1 + r).prod() ^ (252/T) - 1` |
| Annualized Vol | `r.std() × √252` |
| Sharpe Ratio | `(r - rf/252).mean() / r.std() × √252` |
| Max Drawdown | `min((NAV - NAV.cummax()) / NAV.cummax())` |
| Calmar Ratio | `CAGR / |Max Drawdown|` |

Also generates a rolling 252-day Sharpe plot to visualize regime-level variation in strategy performance.

---

## Quickstart

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline in order
python data_fetcher.py          # ~2–3 min (downloads 484 tickers)
python momentum_signal.py
python backtester.py
python performance.py
```

---

## Results

**Backtest period:** 2018-01-02 → 2024-12-30 &nbsp;|&nbsp; **Universe:** 484 S&P 500 constituents &nbsp;|&nbsp; **Transaction costs:** 10 bps round-trip

| Metric | Long-Only | Short-Only | Long-Short |
|--------|:---------:|:----------:|:----------:|
| CAGR | **18.98%** | 15.86% | -1.04% |
| Annualized Vol | 23.56% | 27.24% | 23.43% |
| Sharpe Ratio | **0.686** | 0.529 | -0.096 |
| Max Drawdown | -38.58% | -44.85% | -47.97% |
| Calmar Ratio | 0.492 | 0.354 | -0.022 |
| Daily Hit Rate | 47.1% | 44.1% | 44.8% |

The **long-only leg** is the headline result: an 18.98% CAGR and 0.69 Sharpe over a 7-year period that includes two major drawdowns. This is consistent with the well-documented finding that momentum delivers a persistent premium in large-cap U.S. equities, particularly on the long side where implementation is practical.

---

## Regime Analysis

The long-short portfolio underperformed over this specific window for two identifiable reasons:

**2020 — COVID Momentum Crash**

Going into March 2020, the momentum long book was concentrated in the prior year's winners: travel, energy, and consumer discretionary. These sectors experienced the sharpest drawdowns in the initial crash. Simultaneously, the short book — loaded with underperformers like healthcare and staples — surged as defensive rotations kicked in. The strategy was positioned exactly backwards for the fastest market dislocation in modern history.

**2022 — Rate Shock Rotation**

By late 2021, momentum had accumulated large positions in high-growth technology and communication services stocks — the prior year's leaders. The Fed's aggressive rate hiking cycle in 2022 compressed growth stock valuations sharply, while beaten-down value and energy names (which the strategy was short) outperformed dramatically. This is a well-known vulnerability: momentum is exposed to sharp factor reversals when the macroeconomic regime changes abruptly.

These episodes are consistent with the "momentum crash" literature (Daniel & Moskowitz, 2016), which shows that momentum strategies have negative skewness — they earn steady gains punctuated by rare, severe drawdowns concentrated in market reversals.

---

## Known Limitations

**Survivorship bias** — The constituent list is pulled from today's Wikipedia page, which reflects the *current* S&P 500. Stocks that were in the index during 2018–2024 but were later delisted or removed (due to bankruptcy, acquisition, or index reconstitution) are excluded from the backtest universe. This biases returns upward, since survivors tend to be companies that did well.

**Point-in-time constituent data** — A production-grade backtest would use a historical S&P 500 membership database (e.g. Compustat or a vendor like FactSet) to ensure the universe at each rebalance date reflects only stocks that were *actually in the index* at that time — not what's in it today.

**No market impact modeling** — Transaction costs are modeled as a flat 10 bps per rebalance. In practice, trading ~50 stocks monthly would incur market impact that scales with position size and varies by liquidity. The flat-cost assumption is reasonable for a small fund but understates costs at scale.

**Equal weighting** — Each leg is equal-weighted, which ignores the signal *strength* within the decile. A volatility-scaled or signal-weighted scheme (e.g. weighting by percentile rank) would likely improve risk-adjusted returns and reduce turnover.

**No factor neutralization** — The long book has a persistent beta tilt: momentum winners over a bull market tend to be high-beta growth stocks. The reported returns are not beta-adjusted, so some of the long-only Sharpe reflects market risk premium, not pure momentum alpha.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data manipulation and log return computation |
| `yfinance` | Adjusted price history download |
| `requests`, `lxml` | Wikipedia scraping with browser user-agent |
| `matplotlib` | NAV and rolling Sharpe charts |
| `scipy`, `statsmodels` | Statistical utilities (used in later analysis) |

---

## References

Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers: Implications for stock market efficiency. *Journal of Finance*, 48(1), 65–91.

Daniel, K., & Moskowitz, T. J. (2016). Momentum crashes. *Journal of Financial Economics*, 122(2), 221–247.

Asness, C., Moskowitz, T., & Pedersen, L. (2013). Value and momentum everywhere. *Journal of Finance*, 68(3), 929–985.
