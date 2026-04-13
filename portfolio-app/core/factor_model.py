"""
factor_model.py
---------------
Runs OLS factor regressions and computes all derived outputs.

Models implemented (each is a nested extension of the previous):
  CAPM  : R - RF = α + β₁(Mkt-RF) + ε
  FF3   : + β₂(SMB) + β₃(HML)
  FF5   : + β₄(RMW) + β₅(CMA)
  FF5+M : + β₆(UMD)   ← six-factor model including momentum

For each model we report:
  - Alpha (annualized), t-stat, p-value
  - Factor loadings (β), t-stats, p-values
  - R², Adjusted R², Information Ratio
  - Rolling 126-day (6-month) alpha and factor betas
  - Return decomposition: how much each factor contributed

All regressions are run on *excess* returns (portfolio return minus RF).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dataclasses import dataclass, field

TRADING_DAYS = 252

MODELS = {
    "CAPM":  ["Mkt-RF"],
    "FF3":   ["Mkt-RF", "SMB", "HML"],
    "FF5":   ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
    "FF5+M": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"],
}

FACTOR_LABELS = {
    "Mkt-RF": "Market (Mkt-RF)",
    "SMB":    "Size (SMB)",
    "HML":    "Value (HML)",
    "RMW":    "Profitability (RMW)",
    "CMA":    "Investment (CMA)",
    "UMD":    "Momentum (UMD)",
}


@dataclass
class RegressionResult:
    model_name:  str
    portfolio:   str
    alpha_daily: float          # raw daily alpha from OLS intercept
    alpha_ann:   float          # annualized alpha (× 252)
    alpha_tstat: float
    alpha_pval:  float
    betas:       dict           # factor → loading
    tstats:      dict           # factor → t-stat
    pvals:       dict           # factor → p-value
    r2:          float
    adj_r2:      float
    n_obs:       int
    residuals:   pd.Series
    fitted:      pd.Series


# ── Core regression ────────────────────────────────────────────────────────────

def align(portfolio_returns: pd.Series, factors: pd.DataFrame) -> pd.DataFrame:
    """
    Inner-join portfolio excess returns with factor returns.
    Excess return = portfolio return − daily RF.
    """
    excess = portfolio_returns - factors["RF"]
    df = pd.concat([excess.rename("excess_ret"), factors], axis=1).dropna()
    return df


def run_regression(portfolio_returns: pd.Series,
                   factors: pd.DataFrame,
                   factor_cols: list[str],
                   model_name: str,
                   portfolio_name: str) -> RegressionResult:
    """
    OLS regression of excess portfolio returns on the specified factors.
    Includes a constant (alpha).
    """
    df = align(portfolio_returns, factors)
    y  = df["excess_ret"]
    X  = sm.add_constant(df[factor_cols])

    model  = sm.OLS(y, X).fit(cov_type="HC3")   # heteroskedasticity-robust SEs

    alpha_d = model.params["const"]
    betas   = {f: model.params[f]    for f in factor_cols}
    tstats  = {f: model.tvalues[f]   for f in factor_cols}
    pvals   = {f: model.pvalues[f]   for f in factor_cols}

    return RegressionResult(
        model_name  = model_name,
        portfolio   = portfolio_name,
        alpha_daily = alpha_d,
        alpha_ann   = alpha_d * TRADING_DAYS,
        alpha_tstat = model.tvalues["const"],
        alpha_pval  = model.pvalues["const"],
        betas       = betas,
        tstats      = tstats,
        pvals       = pvals,
        r2          = model.rsquared,
        adj_r2      = model.rsquared_adj,
        n_obs       = int(model.nobs),
        residuals   = model.resid,
        fitted      = model.fittedvalues,
    )


def run_all_models(portfolio_returns: pd.Series,
                   factors: pd.DataFrame,
                   portfolio_name: str) -> dict[str, RegressionResult]:
    """Runs all four models for a single portfolio return series."""
    return {
        name: run_regression(portfolio_returns, factors, fcols, name, portfolio_name)
        for name, fcols in MODELS.items()
    }


# ── Summary table ──────────────────────────────────────────────────────────────

def results_to_table(results: dict[str, RegressionResult]) -> pd.DataFrame:
    """
    Builds a wide summary DataFrame: one row per model, columns for alpha,
    each factor loading, R², etc. Appends t-stats in parentheses.
    """
    all_factors = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]
    rows = []

    for name, r in results.items():
        row = {
            "Model":      name,
            "Alpha (ann %)": f"{r.alpha_ann * 100:.3f}",
            "α t-stat":   f"({r.alpha_tstat:.2f})",
            "α p-value":  f"{r.alpha_pval:.3f}",
            "R²":         f"{r.r2:.4f}",
            "Adj. R²":    f"{r.adj_r2:.4f}",
            "N":          r.n_obs,
        }
        for f in all_factors:
            if f in r.betas:
                row[f"{FACTOR_LABELS[f]} β"] = f"{r.betas[f]:.4f}"
                row[f"{FACTOR_LABELS[f]} t"] = f"({r.tstats[f]:.2f})"
            else:
                row[f"{FACTOR_LABELS[f]} β"] = "—"
                row[f"{FACTOR_LABELS[f]} t"] = ""
        rows.append(row)

    return pd.DataFrame(rows).set_index("Model")


# ── Rolling factor exposures ───────────────────────────────────────────────────

def rolling_regression(portfolio_returns: pd.Series,
                        factors: pd.DataFrame,
                        factor_cols: list[str],
                        window: int = 126) -> pd.DataFrame:
    """
    Rolls a window of `window` trading days and re-runs OLS at each step.
    Returns a DataFrame with columns: alpha_ann, and one beta per factor.
    """
    df = align(portfolio_returns, factors)
    y  = df["excess_ret"]
    X  = df[factor_cols]

    records = {}
    dates   = df.index[window - 1:]

    for i, date in enumerate(dates):
        y_w = y.iloc[i : i + window]
        X_w = sm.add_constant(X.iloc[i : i + window])
        res = sm.OLS(y_w, X_w).fit()
        record = {"alpha_ann": res.params["const"] * TRADING_DAYS}
        record.update({f: res.params[f] for f in factor_cols})
        records[date] = record

    return pd.DataFrame(records).T


# ── Return decomposition ───────────────────────────────────────────────────────

def return_decomposition(result: RegressionResult,
                          factors: pd.DataFrame) -> pd.DataFrame:
    """
    Decomposes the portfolio's daily excess return into:
      alpha contribution + each factor's contribution + residual.

    factor_contribution(t) = β_i × factor_i(t)
    """
    df = align(pd.Series(dtype=float), factors).reindex(result.residuals.index)
    # Re-align factors to the regression sample
    factor_data = factors.loc[result.residuals.index]

    decomp = {}
    decomp["Alpha"] = pd.Series(result.alpha_daily, index=result.residuals.index)
    for f, b in result.betas.items():
        decomp[FACTOR_LABELS[f]] = factor_data[f] * b
    decomp["Residual (α-specific)"] = result.residuals

    return pd.DataFrame(decomp)


# ── Per-ticker table ───────────────────────────────────────────────────────────

def ticker_factor_table(stock_returns: pd.DataFrame,
                         factors: pd.DataFrame,
                         model: str = "FF5+M") -> pd.DataFrame:
    """
    Runs the chosen model for each ticker and returns a summary table.
    """
    factor_cols = MODELS[model]
    rows = []
    for ticker in stock_returns.columns:
        r = run_regression(stock_returns[ticker], factors, factor_cols, model, ticker)
        row = {
            "Ticker":       ticker,
            "Alpha (ann %)": round(r.alpha_ann * 100, 3),
            "α p-value":    round(r.alpha_pval, 3),
            "R²":           round(r.r2, 4),
        }
        for f in factor_cols:
            row[f"{f} β"] = round(r.betas[f], 3)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Ticker")
