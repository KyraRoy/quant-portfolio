"""
optimizer.py
------------
Mean-variance portfolio optimization (Markowitz, 1952).

Computes the efficient frontier by solving:
  minimize    w' Σ w           (portfolio variance)
  subject to  w' μ = μ_target  (target return)
              Σ w_i = 1        (fully invested)
              w_i ≥ 0          (long-only)

Special portfolios on the frontier:
  - Minimum Variance Portfolio (MVP): lowest possible volatility
  - Maximum Sharpe Portfolio   (MSP): highest risk-adjusted return
  - Equal Weight Portfolio     (EWP): 1/N benchmark

The efficient frontier is swept by solving the optimization across a grid
of target returns between MVP return and the maximum individual asset return.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass

TRADING_DAYS = 252
RISK_FREE    = 0.04


@dataclass
class PortfolioPoint:
    weights:    np.ndarray
    ret:        float      # annualized expected return
    vol:        float      # annualized volatility
    sharpe:     float


@dataclass
class EfficientFrontier:
    frontier_points:    list[PortfolioPoint]
    min_variance:       PortfolioPoint
    max_sharpe:         PortfolioPoint
    equal_weight:       PortfolioPoint
    tickers:            list[str]
    mu:                 np.ndarray    # daily mean returns
    cov:                np.ndarray    # daily covariance matrix


# ── Core helpers ───────────────────────────────────────────────────────────────

def _port_stats(weights: np.ndarray, mu: np.ndarray,
                cov: np.ndarray) -> tuple[float, float, float]:
    ret    = float(weights @ mu) * TRADING_DAYS
    vol    = float(np.sqrt(weights @ cov @ weights)) * np.sqrt(TRADING_DAYS)
    sharpe = (ret - RISK_FREE) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def _to_point(weights, mu, cov) -> PortfolioPoint:
    r, v, s = _port_stats(weights, mu, cov)
    return PortfolioPoint(weights=weights, ret=r, vol=v, sharpe=s)


# ── Optimization problems ──────────────────────────────────────────────────────

def min_variance_portfolio(mu: np.ndarray, cov: np.ndarray,
                            n: int) -> PortfolioPoint:
    """Global minimum variance portfolio (no return target)."""
    result = minimize(
        fun=lambda w: w @ cov @ w,
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    return _to_point(result.x, mu, cov)


def max_sharpe_portfolio(mu: np.ndarray, cov: np.ndarray,
                          n: int, rf: float = RISK_FREE) -> PortfolioPoint:
    """Maximum Sharpe ratio portfolio."""
    def neg_sharpe(w):
        r, v, _ = _port_stats(w, mu, cov)
        return -(r - rf) / v if v > 0 else 0.0

    result = minimize(
        fun=neg_sharpe,
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    return _to_point(result.x, mu, cov)


def target_return_portfolio(mu: np.ndarray, cov: np.ndarray,
                             n: int, target_ret: float) -> PortfolioPoint | None:
    """Minimum variance portfolio for a given target annual return."""
    daily_target = target_ret / TRADING_DAYS

    result = minimize(
        fun=lambda w: w @ cov @ w,
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[
            {"type": "eq", "fun": lambda w: w.sum() - 1},
            {"type": "eq", "fun": lambda w: w @ mu - daily_target},
        ],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        return None
    return _to_point(result.x, mu, cov)


# ── Efficient frontier sweep ───────────────────────────────────────────────────

def compute_efficient_frontier(returns: pd.DataFrame,
                                n_points: int = 80) -> EfficientFrontier:
    """
    Builds the full efficient frontier.

    Sweeps target returns from the MVP return to the maximum single-asset
    return, solving a constrained optimization at each point.
    """
    tickers = returns.columns.tolist()
    n       = len(tickers)
    mu      = returns.mean().values
    cov     = returns.cov().values

    mvp = min_variance_portfolio(mu, cov, n)
    msp = max_sharpe_portfolio(mu, cov, n)
    ewp = _to_point(np.ones(n) / n, mu, cov)

    # Sweep target returns between MVP and max asset return
    ret_min = mvp.ret
    ret_max = float(mu.max()) * TRADING_DAYS * 0.95   # cap slightly below max

    frontier = []
    for target in np.linspace(ret_min, ret_max, n_points):
        pt = target_return_portfolio(mu, cov, n, target)
        if pt is not None:
            frontier.append(pt)

    return EfficientFrontier(
        frontier_points=frontier,
        min_variance=mvp,
        max_sharpe=msp,
        equal_weight=ewp,
        tickers=tickers,
        mu=mu,
        cov=cov,
    )


def weights_to_series(weights: np.ndarray, tickers: list[str]) -> pd.Series:
    return pd.Series(weights, index=tickers).round(4)
