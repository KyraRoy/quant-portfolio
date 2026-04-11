"""
performance.py
--------------
Computes and displays performance metrics for the momentum strategy backtest.

Metrics:
  - CAGR, Annualized Volatility, Sharpe Ratio
  - Max Drawdown, Calmar Ratio
  - Monthly hit rate (% of months with positive return)
  - Rolling 12-month Sharpe (plotted)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
RISK_FREE = 0.04      # annualized risk-free rate (approximate 2018-2024 avg T-bill)
TRADING_DAYS = 252


def load_returns(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "backtest_returns.csv")
    return pd.read_csv(path, index_col=0, parse_dates=True)


# ── Metric helpers ─────────────────────────────────────────────────────────────

def cagr(returns: pd.Series) -> float:
    """Compound annual growth rate."""
    total     = (1 + returns).prod()
    n_years   = len(returns) / TRADING_DAYS
    return total ** (1 / n_years) - 1


def annualized_vol(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe(returns: pd.Series, rf: float = RISK_FREE) -> float:
    excess = returns - rf / TRADING_DAYS
    return excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown of the NAV curve."""
    nav = (1 + returns).cumprod()
    peak = nav.cummax()
    dd   = (nav - peak) / peak
    return dd.min()


def calmar(returns: pd.Series) -> float:
    mdd = max_drawdown(returns)
    return cagr(returns) / abs(mdd) if mdd != 0 else np.nan


def hit_rate(returns: pd.Series) -> float:
    """Fraction of trading days with positive returns."""
    return (returns > 0).mean()


def summary_table(returns_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in returns_df.columns:
        r = returns_df[col].dropna()
        rows.append({
            "Portfolio"   : col,
            "CAGR (%)"    : round(cagr(r) * 100, 2),
            "Vol (%)"     : round(annualized_vol(r) * 100, 2),
            "Sharpe"      : round(sharpe(r), 3),
            "Max DD (%)"  : round(max_drawdown(r) * 100, 2),
            "Calmar"      : round(calmar(r), 3),
            "Hit Rate (%)" : round(hit_rate(r) * 100, 1),
        })
    return pd.DataFrame(rows).set_index("Portfolio")


def plot_rolling_sharpe(returns_df: pd.DataFrame, window: int = 252,
                        save_path: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = {"long": "steelblue", "long_short": "seagreen", "short": "tomato"}

    for col in returns_df.columns:
        r = returns_df[col].dropna()
        rolling = r.rolling(window).apply(
            lambda x: sharpe(pd.Series(x)), raw=False
        )
        ax.plot(rolling.index, rolling,
                label=col.replace("_", "-").title(),
                color=colors.get(col, "gray"), linewidth=1.3)

    ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
    ax.set_title(f"Rolling {window}-Day Sharpe Ratio")
    ax.set_ylabel("Sharpe")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    else:
        plt.show()


def main():
    returns_df = load_returns(DATA_DIR)
    print(f"Loaded returns: {returns_df.shape}\n")

    table = summary_table(returns_df)
    print("══ Performance Summary ══════════════════════════════════")
    print(table.to_string())
    print("═════════════════════════════════════════════════════════\n")

    table.to_csv(os.path.join(DATA_DIR, "performance_summary.csv"))
    print("Saved: performance_summary.csv")

    plot_rolling_sharpe(
        returns_df,
        save_path=os.path.join(DATA_DIR, "rolling_sharpe.png")
    )


if __name__ == "__main__":
    main()
