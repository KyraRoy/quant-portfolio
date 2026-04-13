"""
risk_metrics.py
---------------
All risk metric calculations for the dashboard.

Metrics implemented:
  - VaR  : Historical, Parametric (Gaussian), Monte Carlo  @ 95% and 99%
  - CVaR : Expected Shortfall (average loss beyond VaR threshold)
  - Rolling volatility : 21-day (monthly) and 63-day (quarterly)
  - Sharpe, Sortino, Calmar ratios
  - Max drawdown + drawdown duration (in trading days)
  - Beta and correlation to benchmark (SPY)
  - Per-ticker risk table

All functions take a pd.Series or pd.DataFrame of *daily* returns.
Annualization uses 252 trading days throughout.
"""

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252
RISK_FREE    = 0.04   # annualized; approximate 2018-2024 T-bill average


# ── VaR ───────────────────────────────────────────────────────────────────────

def var_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical VaR at the given confidence level.
    Simply the (1 - confidence) quantile of the empirical return distribution.
    No distributional assumptions — fully data-driven.
    """
    return float(np.percentile(returns.dropna(), (1 - confidence) * 100))


def var_parametric(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Parametric (Gaussian) VaR.
    Assumes returns are normally distributed: VaR = μ - z * σ
    Tends to underestimate tail risk because real return distributions
    have heavier tails than a Gaussian (excess kurtosis > 0).
    """
    mu, sigma = returns.mean(), returns.std()
    z = stats.norm.ppf(1 - confidence)
    return float(mu + z * sigma)


def var_monte_carlo(returns: pd.Series, confidence: float = 0.95,
                    n_sims: int = 100_000) -> float:
    """
    Monte Carlo VaR.
    Simulates n_sims random returns drawn from a fitted normal distribution,
    then takes the (1-confidence) quantile of the simulated distribution.
    With large n_sims this converges to the parametric result, but can be
    extended to non-normal distributions (e.g. Student-t, GBM paths).
    """
    mu, sigma = returns.mean(), returns.std()
    rng = np.random.default_rng(seed=42)
    simulated = rng.normal(mu, sigma, n_sims)
    return float(np.percentile(simulated, (1 - confidence) * 100))


def cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Conditional VaR (Expected Shortfall).
    The average return in the worst (1-confidence)% of days.
    A coherent risk measure — preferred over VaR for regulatory and
    portfolio optimization purposes (Basel III uses 97.5% ES).
    """
    threshold = var_historical(returns, confidence)
    tail = returns[returns <= threshold]
    return float(tail.mean()) if len(tail) > 0 else threshold


# ── Volatility ────────────────────────────────────────────────────────────────

def rolling_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
    """Annualized rolling volatility over a given window (trading days)."""
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


# ── Return ratios ─────────────────────────────────────────────────────────────

def sharpe_ratio(returns: pd.Series, rf: float = RISK_FREE) -> float:
    """Annualized Sharpe ratio: excess return per unit of total volatility."""
    excess = returns - rf / TRADING_DAYS
    return float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns: pd.Series, rf: float = RISK_FREE) -> float:
    """
    Annualized Sortino ratio: excess return per unit of *downside* volatility.
    Only penalizes negative deviations, unlike Sharpe which treats upside
    and downside volatility symmetrically.
    """
    excess    = returns - rf / TRADING_DAYS
    downside  = returns[returns < 0].std() * np.sqrt(TRADING_DAYS)
    ann_ret   = excess.mean() * TRADING_DAYS
    return float(ann_ret / downside) if downside > 0 else np.nan


def calmar_ratio(returns: pd.Series) -> float:
    """CAGR divided by absolute max drawdown. Popular with CTA / trend funds."""
    mdd  = max_drawdown(returns)
    cagr = (1 + returns).prod() ** (TRADING_DAYS / len(returns)) - 1
    return float(cagr / abs(mdd)) if mdd != 0 else np.nan


# ── Drawdown ──────────────────────────────────────────────────────────────────

def drawdown_series(returns: pd.Series) -> pd.Series:
    """Full drawdown time series (fraction below previous peak)."""
    nav  = (1 + returns).cumprod()
    peak = nav.cummax()
    return (nav - peak) / peak


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown_series(returns).min())


def drawdown_durations(returns: pd.Series) -> pd.DataFrame:
    """
    Returns a DataFrame of every distinct drawdown episode:
      start, trough, end, depth (%), duration (trading days).
    """
    dd   = drawdown_series(returns)
    nav  = (1 + returns).cumprod()

    episodes = []
    in_dd    = False
    start    = None
    trough_date = None
    trough_val  = 0.0

    for date, val in dd.items():
        if not in_dd and val < 0:
            in_dd  = True
            start  = date
            trough_date = date
            trough_val  = val
        elif in_dd:
            if val < trough_val:
                trough_val  = val
                trough_date = date
            if val == 0:
                episodes.append({
                    "Start":    start,
                    "Trough":   trough_date,
                    "End":      date,
                    "Depth (%)": round(trough_val * 100, 2),
                    "Duration (days)": (date - start).days,
                })
                in_dd = False

    return pd.DataFrame(episodes)


# ── Market relationship ───────────────────────────────────────────────────────

def beta(returns: pd.Series, benchmark: pd.Series) -> float:
    """
    OLS beta to benchmark.
    beta > 1: more volatile than market
    beta < 1: less volatile (defensive)
    beta < 0: moves opposite to market
    """
    aligned = pd.concat([returns, benchmark], axis=1).dropna()
    cov     = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return float(cov[0, 1] / cov[1, 1])


def correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    return returns_df.corr()


# ── CAGR ──────────────────────────────────────────────────────────────────────

def cagr(returns: pd.Series) -> float:
    return float((1 + returns).prod() ** (TRADING_DAYS / len(returns)) - 1)


# ── Per-ticker summary table ──────────────────────────────────────────────────

def ticker_risk_table(returns_df: pd.DataFrame,
                      benchmark_col: str = "SPY") -> pd.DataFrame:
    """
    Builds a per-ticker risk summary table. If benchmark_col is present in
    returns_df, computes beta; otherwise omits it.
    """
    rows = []
    bench = returns_df[benchmark_col] if benchmark_col in returns_df.columns else None

    for col in returns_df.columns:
        r = returns_df[col].dropna()
        row = {
            "Ticker":       col,
            "CAGR (%)":     round(cagr(r) * 100, 2),
            "Ann. Vol (%)": round(r.std() * np.sqrt(TRADING_DAYS) * 100, 2),
            "Sharpe":       round(sharpe_ratio(r), 3),
            "Sortino":      round(sortino_ratio(r), 3),
            "Max DD (%)":   round(max_drawdown(r) * 100, 2),
            "VaR 95% (%)":  round(var_historical(r, 0.95) * 100, 2),
            "CVaR 95% (%)": round(cvar(r, 0.95) * 100, 2),
        }
        if bench is not None and col != benchmark_col:
            row["Beta"] = round(beta(r, bench.reindex(r.index).dropna()), 3)
        rows.append(row)

    return pd.DataFrame(rows).set_index("Ticker")
