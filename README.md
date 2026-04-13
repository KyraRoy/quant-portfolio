# Quantitative Finance Portfolio

A collection of five end-to-end quantitative finance projects built in Python on real S&P 500 market data from 2018–2024. Each project covers a distinct area of quant finance — strategy backtesting, risk measurement, statistical arbitrage, factor modeling, and simulation — and is packaged together as a single interactive web app.

**Live app:** [portfolio-app/app.py](portfolio-app/app.py) · Run locally with `streamlit run portfolio-app/app.py`

---

## Projects

### 1. Momentum Strategy
**Location:** `momentum-strategy/`

Implements the Jegadeesh-Titman (1993) cross-sectional momentum strategy on the full S&P 500 universe (484 stocks). Stocks are ranked monthly by their 12-1 month return (12-month lookback, skipping the most recent month to avoid short-term reversal). The top decile is bought long, the bottom decile is sold short, and the portfolio is rebalanced monthly with 10 basis points of transaction costs per trade.

| Metric | Long-Only | Long-Short |
|---|---|---|
| CAGR | 18.98% | −1.04% |
| Sharpe | 0.686 | −0.096 |
| Max Drawdown | −38.6% | −48.0% |
| Volatility | 23.56% | 23.43% |

The long-short portfolio underperforms due to momentum crashes in 2020 (COVID reversal) and 2022 (rate shock), where the short book — concentrated in beaten-down value stocks — rallied sharply against the position.

**Pipeline:** `data_fetcher.py` → `momentum_signal.py` → `backtester.py` → `performance.py`

---

### 2. Risk Dashboard
**Location:** `risk-dashboard/`

Computes institutional-grade risk metrics on a 20-stock diversified portfolio across sectors (Technology, Financials, Healthcare, Energy, Consumer, Utilities, Real Estate). Covers Value-at-Risk and Conditional Value-at-Risk via three methods, rolling volatility regimes, drawdown analysis, and cross-asset correlation structure.

**Metrics computed:**
- **VaR / CVaR** — Historical simulation, parametric (Gaussian), and Monte Carlo (100k paths)
- **Risk-adjusted returns** — Sharpe, Sortino, Calmar ratios
- **Drawdown** — full drawdown curve and episode table
- **Beta** — regression-based market beta to SPY
- **Correlation heatmap** — sorted by GICS sector

**Pipeline:** `data_fetcher.py` → `risk_metrics.py` → `dashboard.py`

---

### 3. Pairs Trading / Statistical Arbitrage
**Location:** `pairs-trading/`

Market-neutral mean-reversion strategy using Engle-Granger cointegration. Pairs are screened within GICS sectors (correlation ≥ 0.85), tested for cointegration, and filtered by Ornstein-Uhlenbeck half-life (5–126 days). 12 cointegrated pairs are found. The strategy is formed on 2018–2021 data and traded out-of-sample on 2022–2024.

**Signal mechanics:**
- Hedge ratio estimated via OLS on log prices
- 60-day rolling z-score of the spread
- Entry at |z| > 2.0, exit at |z| < 0.5, stop-loss at |z| > 3.0
- OU process fit via AR(1) to estimate mean-reversion speed κ and half-life

| Metric | Value |
|---|---|
| Total trades | 226 |
| Win rate | 53.5% |
| Avg holding period | 22 days |
| Portfolio CAGR | −0.44% |
| Sharpe | −1.29 |

**Pipeline:** `pairs_selector.py` → `spread_model.py` → `backtester.py` → `dashboard.py`

---

### 4. Fama-French Factor Model
**Location:** `fama-french/`

Decomposes the momentum strategy's returns through four nested OLS factor models to measure alpha and identify risk exposures. Uses HC3 heteroskedasticity-robust standard errors throughout. Factor data is sourced directly from Ken French's data library.

**Models estimated:**
- **CAPM** — market beta only
- **FF3** — market, size (SMB), value (HML)
- **FF5** — adds profitability (RMW) and investment (CMA)
- **FF5+Momentum** — adds the UMD momentum factor

Alpha shrinks from CAPM → FF3 → FF5 as each model absorbs more systematic variance, but remains positive at ~6.9% annualized in the full six-factor model (R² = 88.7%). Rolling 126-day regressions show how factor exposures shift through COVID and rate shock regimes.

**Pipeline:** `data_fetcher.py` → `factor_model.py` → `dashboard.py`

---

### 5. Monte Carlo Portfolio Simulator
**Location:** `monte-carlo/`

Simulates 10,000 portfolio paths over a user-selected horizon (1–10 years) using two methods and computes tail risk statistics and outcome probabilities. Includes a Markowitz mean-variance optimizer to construct the efficient frontier.

**Simulation methods:**
- **GBM** — correlated geometric Brownian motion via Cholesky decomposition, Itô drift correction (−½σ²)
- **Block Bootstrap** — historical resampling with block size 10 to preserve autocorrelation structure

**Outputs:** VaR / CVaR at the horizon, probability of reaching a target multiple, fan charts (5th–95th percentile bands), terminal wealth distribution, and a probability table by sub-horizon.

**Optimizer:** SLSQP-based mean-variance optimization producing the efficient frontier with labeled portfolios — minimum variance, maximum Sharpe, and equal weight.

**Pipeline:** `simulator.py` + `optimizer.py` → `dashboard.py`

---

## Unified Web App

**Location:** `portfolio-app/`

All five projects are combined into a single Streamlit multi-page application with a shared dark-theme design system.

```
portfolio-app/
├── app.py                    # Home page — project cards, hero, stats
├── pages/
│   ├── 1_📈_Momentum.py
│   ├── 2_🛡️_Risk_Dashboard.py
│   ├── 3_⚖️_Pairs_Trading.py
│   ├── 4_📊_Factor_Model.py
│   └── 5_🎲_Monte_Carlo.py
├── core/
│   ├── style.py              # Shared color palette, Plotly template, KPI cards
│   ├── risk_metrics.py
│   ├── factor_model.py
│   ├── spread_model.py
│   ├── simulator.py
│   └── optimizer.py
├── data/                     # Pre-computed CSVs (~4 MB)
├── .streamlit/config.toml    # Dark navy theme
└── requirements.txt
```

### Running locally

```bash
cd portfolio-app
pip install -r requirements.txt
streamlit run app.py
```

### Deploying to Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
2. Click **Create app**
3. Set repository to `KyraRoy/quant-portfolio`, branch `main`, main file `portfolio-app/app.py`
4. Click **Deploy**

---

## Stack

| Library | Purpose |
|---|---|
| `pandas` / `numpy` | Data manipulation and numerical computing |
| `scipy` | Portfolio optimization (SLSQP solver) |
| `statsmodels` | OLS regression with HC3 robust standard errors |
| `plotly` | All interactive charts |
| `streamlit` | Web app framework |
| `yfinance` | Historical price data download |
| `requests` / `lxml` | S&P 500 constituent scraping, Ken French data |
| `matplotlib` | Pandas Styler gradient coloring |

---

## Data

All raw price data is sourced from Yahoo Finance via `yfinance`. Fama-French factor data is downloaded directly from [Kenneth French's data library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html). The portfolio app ships with pre-computed result CSVs so no data download is needed at runtime.

**Coverage:** January 2018 – December 2024 · S&P 500 universe (484 stocks after cleaning)
