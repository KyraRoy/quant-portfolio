"""
simulator.py
------------
Monte Carlo simulation engine for a multi-asset portfolio.

Two simulation methods are supported:

  1. Geometric Brownian Motion (GBM)
     Assumes log-normally distributed returns. Each asset's daily log return
     is drawn from a multivariate normal distribution parameterised by the
     historical mean vector (μ) and covariance matrix (Σ). The Cholesky
     decomposition of Σ is used to preserve cross-asset correlations.

       log(S_t / S_{t-1}) ~ N(μ - σ²/2, Σ)   (Itô correction applied)

  2. Historical Bootstrap
     Resamples blocks of actual daily returns (with replacement). Makes no
     distributional assumption and naturally captures fat tails, skewness,
     and volatility clustering present in real data.

Both methods produce a (n_sims × n_days) matrix of portfolio NAV paths,
from which we derive:
  - Percentile fan chart (5th, 25th, 50th, 75th, 95th)
  - Terminal wealth distribution
  - Probability of reaching a return target
  - Simulated VaR and CVaR at multiple horizons
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


TRADING_DAYS = 252


@dataclass
class SimulationResult:
    method:          str
    n_sims:          int
    horizon_days:    int
    weights:         np.ndarray
    tickers:         list[str]
    paths:           np.ndarray          # shape: (n_sims, horizon_days + 1)
    terminal_wealth: np.ndarray          # shape: (n_sims,)  — final NAV values
    percentiles:     dict[int, np.ndarray]  # {5: array, 25: ..., 50: ..., 75: ..., 95: ...}


# ── Parameter estimation ───────────────────────────────────────────────────────

def estimate_parameters(returns: pd.DataFrame,
                         weights: np.ndarray) -> tuple:
    """
    Estimates the mean return vector and covariance matrix from historical data.

    Returns (mu, cov, port_mu, port_vol):
      mu       — daily mean log return per asset (annualized drift corrected)
      cov      — daily covariance matrix
      port_mu  — annualized portfolio expected return
      port_vol — annualized portfolio volatility
    """
    mu  = returns.mean().values                            # daily mean vector
    cov = returns.cov().values                             # daily covariance matrix

    port_mu  = float(weights @ mu) * TRADING_DAYS
    port_vol = float(np.sqrt(weights @ cov @ weights) * np.sqrt(TRADING_DAYS))

    return mu, cov, port_mu, port_vol


# ── GBM simulation ─────────────────────────────────────────────────────────────

def simulate_gbm(returns: pd.DataFrame,
                  weights: np.ndarray,
                  n_sims: int = 1_000,
                  horizon_days: int = 252,
                  seed: int = 42) -> SimulationResult:
    """
    Simulates portfolio paths via Geometric Brownian Motion.

    Each day's log return vector is drawn from:
      N(μ - 0.5 * diag(Σ), Σ)
    where the -0.5σ² term is the Itô correction converting from geometric
    to arithmetic mean (ensures E[S_T] = S_0 * exp(μT)).
    """
    mu, cov, port_mu, port_vol = estimate_parameters(returns, weights)

    # Itô drift correction: subtract half-variance from each asset's drift
    drift = mu - 0.5 * np.diag(cov)

    # Cholesky factorisation preserves correlation structure
    L = np.linalg.cholesky(cov + 1e-10 * np.eye(len(mu)))   # small jitter for stability

    rng = np.random.default_rng(seed)
    n_assets = len(mu)

    # Simulate log returns: shape (n_sims, horizon_days, n_assets)
    z        = rng.standard_normal((n_sims, horizon_days, n_assets))
    log_rets = drift + (z @ L.T)   # broadcast drift, apply Cholesky

    # Portfolio log return each day = weighted sum of asset log returns
    port_log_rets = log_rets @ weights   # shape: (n_sims, horizon_days)

    # Cumulative NAV paths (starting at 1.0)
    paths = np.ones((n_sims, horizon_days + 1))
    paths[:, 1:] = np.exp(np.cumsum(port_log_rets, axis=1))

    return _package_result("GBM", n_sims, horizon_days, weights,
                            returns.columns.tolist(), paths)


# ── Historical bootstrap ───────────────────────────────────────────────────────

def simulate_bootstrap(returns: pd.DataFrame,
                        weights: np.ndarray,
                        n_sims: int = 1_000,
                        horizon_days: int = 252,
                        block_size: int = 10,
                        seed: int = 42) -> SimulationResult:
    """
    Simulates portfolio paths by resampling blocks of historical returns.

    Block bootstrap (block_size > 1) preserves short-term autocorrelation
    and volatility clustering better than i.i.d. resampling.
    """
    rng = np.random.default_rng(seed)

    # Pre-compute daily portfolio returns from the historical series
    port_hist = (returns * weights).sum(axis=1).values   # shape: (T,)
    T = len(port_hist)

    # Number of blocks needed to cover horizon_days
    n_blocks = int(np.ceil(horizon_days / block_size))

    paths = np.ones((n_sims, horizon_days + 1))

    for sim in range(n_sims):
        # Sample random starting indices for each block
        starts    = rng.integers(0, T - block_size, size=n_blocks)
        sampled   = np.concatenate([port_hist[s : s + block_size] for s in starts])
        sampled   = sampled[:horizon_days]    # trim to exact horizon
        paths[sim, 1:] = np.exp(np.cumsum(sampled))

    return _package_result("Bootstrap", n_sims, horizon_days, weights,
                            returns.columns.tolist(), paths)


# ── Result packaging ───────────────────────────────────────────────────────────

def _package_result(method, n_sims, horizon_days, weights, tickers, paths):
    terminal = paths[:, -1]
    pcts = {p: np.percentile(paths, p, axis=0) for p in [5, 25, 50, 75, 95]}
    return SimulationResult(
        method=method, n_sims=n_sims, horizon_days=horizon_days,
        weights=weights, tickers=tickers,
        paths=paths, terminal_wealth=terminal, percentiles=pcts,
    )


# ── Derived statistics ─────────────────────────────────────────────────────────

def prob_reach_target(result: SimulationResult, target: float) -> float:
    """Fraction of simulations where terminal wealth ≥ target (e.g. 2.0 = double)."""
    return float((result.terminal_wealth >= target).mean())


def simulated_var(result: SimulationResult, confidence: float = 0.95) -> float:
    """VaR from terminal wealth distribution (loss relative to starting NAV of 1.0)."""
    return float(np.percentile(result.terminal_wealth, (1 - confidence) * 100)) - 1.0


def simulated_cvar(result: SimulationResult, confidence: float = 0.95) -> float:
    """CVaR: average terminal wealth in the worst (1-confidence)% of simulations."""
    threshold = np.percentile(result.terminal_wealth, (1 - confidence) * 100)
    tail = result.terminal_wealth[result.terminal_wealth <= threshold]
    return float(tail.mean()) - 1.0 if len(tail) > 0 else simulated_var(result, confidence)


def horizon_statistics(result: SimulationResult) -> pd.DataFrame:
    """
    Summary statistics at multiple horizons (1M, 3M, 6M, 1Y, end).
    """
    horizons = {
        "1 Month":  21,
        "3 Months": 63,
        "6 Months": 126,
        "1 Year":   252,
        f"{result.horizon_days // 252} Years": result.horizon_days,
    }

    rows = []
    for label, days in horizons.items():
        if days > result.horizon_days:
            continue
        nav_at = result.paths[:, days]
        rows.append({
            "Horizon":       label,
            "Median NAV":    round(float(np.median(nav_at)), 4),
            "5th Pct":       round(float(np.percentile(nav_at, 5)), 4),
            "95th Pct":      round(float(np.percentile(nav_at, 95)), 4),
            "P(loss)":       round(float((nav_at < 1.0).mean() * 100), 1),
            "P(+20%)":       round(float((nav_at >= 1.2).mean() * 100), 1),
            "P(double)":     round(float((nav_at >= 2.0).mean() * 100), 1),
        })

    return pd.DataFrame(rows)
