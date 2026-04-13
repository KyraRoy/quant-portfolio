"""
spread_model.py
---------------
Models the spread between a cointegrated pair.

For a pair (A, B) with hedge ratio β:
  spread(t) = log(price_A(t)) − β × log(price_B(t))

The spread is assumed to follow an Ornstein-Uhlenbeck (OU) process:
  dS = κ(μ − S)dt + σ dW

where:
  κ  = mean-reversion speed
  μ  = long-run equilibrium level
  σ  = volatility of the spread
  half-life = ln(2) / κ  (time for spread to halve its deviation from μ)

Trading signals are generated from a rolling z-score of the spread:
  z(t) = (spread(t) − rolling_mean(t)) / rolling_std(t)

Entry:  |z| > ENTRY_Z  (spread has deviated enough to be worth trading)
Exit:   |z| < EXIT_Z   (spread has reverted — take profit)
Stop:   |z| > STOP_Z   (spread keeps moving against us — cut loss)
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dataclasses import dataclass


# ── Configuration ──────────────────────────────────────────────────────────────
ENTRY_Z  = 2.0    # open trade when |z-score| exceeds this
EXIT_Z   = 0.5    # close trade when |z-score| falls below this
STOP_Z   = 3.0    # stop-loss if |z-score| exceeds this
Z_WINDOW = 60     # rolling window (trading days) for z-score normalisation


@dataclass
class PairSpread:
    ticker_a:    str
    ticker_b:    str
    hedge_ratio: float
    spread:      pd.Series    # full spread series
    z_score:     pd.Series    # rolling z-score
    half_life:   float
    ou_kappa:    float        # mean-reversion speed
    ou_mu:       float        # long-run mean
    ou_sigma:    float        # spread volatility


# ── Spread construction ────────────────────────────────────────────────────────

def compute_hedge_ratio(log_price_a: pd.Series,
                         log_price_b: pd.Series) -> float:
    """
    OLS regression of log(A) on log(B) over the full price series.
    The slope is the hedge ratio β: how many units of B offset one unit of A.
    """
    common = log_price_a.index.intersection(log_price_b.index)
    X = sm.add_constant(log_price_b[common])
    res = sm.OLS(log_price_a[common], X).fit()
    return float(res.params.iloc[1])


def compute_spread(log_price_a: pd.Series,
                   log_price_b: pd.Series,
                   hedge_ratio: float) -> pd.Series:
    common = log_price_a.index.intersection(log_price_b.index)
    return log_price_a[common] - hedge_ratio * log_price_b[common]


def rolling_zscore(spread: pd.Series, window: int = Z_WINDOW) -> pd.Series:
    mu  = spread.rolling(window).mean()
    sig = spread.rolling(window).std()
    return (spread - mu) / sig


# ── OU process fitting ─────────────────────────────────────────────────────────

def fit_ou(spread: pd.Series) -> tuple[float, float, float]:
    """
    Fits an OU process via discrete AR(1) regression:
      spread(t) = a + b * spread(t-1) + ε

    Returns (κ, μ, σ_OU):
      κ         = -ln(b)                   mean-reversion speed (per day)
      μ         = a / (1 - b)              long-run mean
      σ_OU      = std(residuals)            daily volatility of spread
    """
    s   = spread.dropna()
    lag = s.shift(1).dropna()
    s   = s.loc[lag.index]

    X   = sm.add_constant(lag)
    res = sm.OLS(s, X).fit()
    a, b = res.params

    kappa = -np.log(b) if b > 0 else np.nan
    mu    = a / (1 - b) if abs(1 - b) > 1e-8 else s.mean()
    sigma = res.resid.std()

    return float(kappa), float(mu), float(sigma)


def half_life_from_kappa(kappa: float) -> float:
    return float(np.log(2) / kappa) if kappa > 0 else np.nan


# ── Main model builder ─────────────────────────────────────────────────────────

def build_pair_spread(ticker_a: str,
                       ticker_b: str,
                       prices: pd.DataFrame,
                       hedge_ratio: float | None = None) -> PairSpread:
    """
    Constructs a full PairSpread object for a given pair.

    If hedge_ratio is None it is estimated from the data.
    Uses the hedge ratio from pairs_selector (formation period) if provided.
    """
    log_a = np.log(prices[ticker_a].dropna())
    log_b = np.log(prices[ticker_b].dropna())

    if hedge_ratio is None:
        hedge_ratio = compute_hedge_ratio(log_a, log_b)

    spread  = compute_spread(log_a, log_b, hedge_ratio)
    z_score = rolling_zscore(spread, Z_WINDOW)

    kappa, mu, sigma = fit_ou(spread)
    hl = half_life_from_kappa(kappa)

    return PairSpread(
        ticker_a    = ticker_a,
        ticker_b    = ticker_b,
        hedge_ratio = hedge_ratio,
        spread      = spread,
        z_score     = z_score,
        half_life   = hl,
        ou_kappa    = kappa,
        ou_mu       = mu,
        ou_sigma    = sigma,
    )


# ── Signal generation ─────────────────────────────────────────────────────────

def generate_signals(z_score: pd.Series,
                      entry_z: float = ENTRY_Z,
                      exit_z:  float = EXIT_Z,
                      stop_z:  float = STOP_Z) -> pd.Series:
    """
    Generates a position signal from the z-score:
      +1 → long spread (A underperformed: buy A, sell B)
      -1 → short spread (A outperformed: sell A, buy B)
       0 → flat

    State machine: position opens on entry signal, closes on exit or stop.
    """
    signals  = pd.Series(0, index=z_score.index, dtype=float)
    position = 0

    for i, (date, z) in enumerate(z_score.items()):
        if np.isnan(z):
            signals[date] = 0
            continue

        if position == 0:
            # Entry
            if z < -entry_z:
                position = 1    # spread too low → long
            elif z > entry_z:
                position = -1   # spread too high → short

        elif position == 1:
            # Exit or stop-loss on long
            if z > -exit_z or z < -stop_z:
                position = 0

        elif position == -1:
            # Exit or stop-loss on short
            if z < exit_z or z > stop_z:
                position = 0

        signals[date] = position

    return signals
