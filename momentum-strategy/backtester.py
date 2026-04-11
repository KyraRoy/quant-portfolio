"""
backtester.py
-------------
Event-driven backtester for the cross-sectional momentum strategy.

Mechanics:
  - Signals are computed at month-end t and acted on at open of t+1
    (avoids look-ahead bias).
  - Equal-weighted within each leg (long / short).
  - Transaction costs modeled as a flat round-trip bps per rebalance.
  - Supports long-only and long-short portfolio variants.

Outputs (data/):
  - backtest_returns.csv  — daily strategy returns
  - backtest_nav.csv      — cumulative NAV series
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────────
TRANSACTION_COST_BPS = 10    # round-trip basis points per trade
INITIAL_CAPITAL      = 1.0   # normalized starting NAV
DATA_DIR             = os.path.join(os.path.dirname(__file__), "data")


def load_inputs(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices      = pd.read_csv(os.path.join(data_dir, "sp500_prices.csv"),
                               index_col=0, parse_dates=True)
    long_mask   = pd.read_csv(os.path.join(data_dir, "long_portfolio.csv"),
                               index_col=0, parse_dates=True)
    short_mask  = pd.read_csv(os.path.join(data_dir, "short_portfolio.csv"),
                               index_col=0, parse_dates=True)
    return prices, long_mask, short_mask


def build_weights(mask: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight within each leg; rows sum to 1 (or 0 if no positions)."""
    counts = mask.sum(axis=1).replace(0, np.nan)
    return mask.div(counts, axis=0).fillna(0.0)


def run_backtest(prices: pd.DataFrame,
                 long_mask: pd.DataFrame,
                 short_mask: pd.DataFrame,
                 tc_bps: float = TRANSACTION_COST_BPS) -> dict[str, pd.Series]:
    """
    Simulates daily P&L for long-only, short-only, and long-short portfolios.

    Signals at month-end t → weights applied from t+1 onward until next signal.
    Transaction costs are deducted at each rebalance proportional to turnover.

    Parameters
    ----------
    prices : pd.DataFrame        Adjusted close prices (date × ticker)
    long_mask : pd.DataFrame     Boolean long membership (month × ticker)
    short_mask : pd.DataFrame    Boolean short membership (month × ticker)
    tc_bps : float               Round-trip cost in basis points

    Returns
    -------
    dict mapping portfolio name → daily return Series
    """
    daily_returns = prices.pct_change().dropna(how="all")
    tc_rate = tc_bps / 10_000

    results = {}

    for name, mask in [("long", long_mask), ("short", short_mask)]:
        weights_monthly = build_weights(mask.astype(float))

        # Align signal dates to trading calendar: each signal is valid from the
        # next trading day after the signal date until the next signal date.
        daily_weights = weights_monthly.reindex(daily_returns.index, method="ffill")

        # Shift by 1 day: signal known at t, trade at t+1
        daily_weights = daily_weights.shift(1).fillna(0.0)

        # Align tickers between weights and returns
        common = daily_weights.columns.intersection(daily_returns.columns)
        w = daily_weights[common]
        r = daily_returns[common]

        # Portfolio return (before costs)
        port_ret = (w * r).sum(axis=1)

        # Transaction costs: turnover at each rebalance date
        turnover = w.diff().abs().sum(axis=1)
        costs = turnover * tc_rate

        results[name] = port_ret - costs

    # Long-short: long leg minus short leg (dollar-neutral)
    results["long_short"] = results["long"] - results["short"]

    return results


def compute_nav(returns: pd.Series, initial: float = INITIAL_CAPITAL) -> pd.Series:
    """Computes cumulative NAV from a daily return series."""
    return initial * (1 + returns).cumprod()


def plot_nav(nav_dict: dict[str, pd.Series], save_path: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    styles = {"long": ("steelblue", "-"), "long_short": ("seagreen", "--"),
              "short": ("tomato", ":")}

    for name, nav in nav_dict.items():
        color, ls = styles.get(name, ("gray", "-"))
        ax.plot(nav.index, nav, label=name.replace("_", "-").title(),
                color=color, linestyle=ls, linewidth=1.5)

    ax.axhline(1, color="black", linewidth=0.7, linestyle=":")
    ax.set_title("Momentum Strategy — Cumulative NAV (2018–2024)")
    ax.set_ylabel("NAV (starting = 1.0)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved chart: {save_path}")
    else:
        plt.show()


def main():
    prices, long_mask, short_mask = load_inputs(DATA_DIR)
    print(f"Prices      : {prices.shape}")
    print(f"Long mask   : {long_mask.shape}")
    print(f"Short mask  : {short_mask.shape}")

    ret_dict = run_backtest(prices, long_mask, short_mask)

    nav_dict = {name: compute_nav(ret) for name, ret in ret_dict.items()}

    # Save returns and NAV
    returns_df = pd.DataFrame(ret_dict)
    nav_df     = pd.DataFrame(nav_dict)

    returns_df.to_csv(os.path.join(DATA_DIR, "backtest_returns.csv"))
    nav_df.to_csv(os.path.join(DATA_DIR, "backtest_nav.csv"))
    print("\nSaved: backtest_returns.csv, backtest_nav.csv")

    plot_nav(nav_dict, save_path=os.path.join(DATA_DIR, "nav_chart.png"))

    # Print final NAV for each leg
    print("\nFinal NAV:")
    for name, nav in nav_dict.items():
        print(f"  {name:12s}: {nav.iloc[-1]:.4f}")


if __name__ == "__main__":
    main()
