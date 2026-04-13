"""
backtester.py
-------------
Event-driven backtester for the pairs trading strategy.

For each pair:
  - Signal at day t → position held from t+1 (no look-ahead bias)
  - Daily P&L = position(t) × [return_A(t+1) − hedge_ratio × return_B(t+1)]
  - Transaction costs: 10 bps per leg per trade (20 bps total round-trip)

Portfolio:
  - Equal capital allocated to each pair
  - All pair P&Ls summed to get portfolio return

Train / Test split:
  - Formation  : 2018-01-01 → 2021-12-31  (pairs selected on this window)
  - Trading    : 2022-01-01 → 2024-12-31  (strategy run out-of-sample)

Outputs (data/):
  - pair_returns.csv   — daily P&L per pair (trading period)
  - portfolio_nav.csv  — equal-weight portfolio cumulative NAV
  - trade_log.csv      — every entry/exit event with P&L
"""

import os
import numpy as np
import pandas as pd

import spread_model as sm_mod

# ── Configuration ──────────────────────────────────────────────────────────────
FORMATION_START  = "2018-01-01"
FORMATION_END    = "2021-12-31"
TRADING_START    = "2022-01-01"
TRADING_END      = "2024-12-31"
TC_BPS_PER_LEG   = 10          # basis points per leg per trade
DATA_DIR         = os.path.join(os.path.dirname(__file__), "data")

_BASE            = os.path.dirname(os.path.dirname(__file__))
PRICES_PATH      = os.path.join(_BASE, "momentum-strategy", "data", "sp500_prices.csv")


# ── Load data ──────────────────────────────────────────────────────────────────

def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    prices  = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    pairs   = pd.read_csv(os.path.join(DATA_DIR, "top_pairs.csv"))
    return prices, pairs


# ── Per-pair backtest ─────────────────────────────────────────────────────────

def backtest_pair(ticker_a: str,
                  ticker_b: str,
                  hedge_ratio: float,
                  prices: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """
    Runs the backtest for a single pair over the trading period.

    Returns
    -------
    daily_pnl : pd.Series
        Daily P&L as a fraction of capital (both legs combined).
    trade_log : pd.DataFrame
        One row per closed trade.
    """
    # Build spread on FULL price history (formation + trading)
    pair_spread = sm_mod.build_pair_spread(ticker_a, ticker_b, prices, hedge_ratio)
    z_score     = pair_spread.z_score

    # Generate signals on full history, then slice to trading period
    signals_full = sm_mod.generate_signals(z_score)
    signals      = signals_full.loc[TRADING_START:TRADING_END]

    # Shift by 1: signal at t → position at t+1
    position = signals.shift(1).fillna(0)

    # Daily log returns
    log_prices = np.log(prices[[ticker_a, ticker_b]].ffill())
    ret_a = log_prices[ticker_a].diff()
    ret_b = log_prices[ticker_b].diff()

    # Align to trading period
    ret_a = ret_a.loc[TRADING_START:TRADING_END]
    ret_b = ret_b.loc[TRADING_START:TRADING_END]
    position = position.reindex(ret_a.index).fillna(0)

    # Spread P&L: long spread = long A, short β units of B
    spread_return = ret_a - hedge_ratio * ret_b
    daily_pnl     = position * spread_return

    # Transaction costs: deduct on every position change
    turnover   = position.diff().abs()
    tc_rate    = TC_BPS_PER_LEG * 2 / 10_000   # both legs
    daily_pnl -= turnover * tc_rate

    # ── Build trade log ────────────────────────────────────────────────────────
    trades = []
    in_trade   = False
    entry_date = None
    entry_pos  = 0
    cumulative  = 0.0

    for date in position.index:
        pos = position[date]
        pnl = daily_pnl[date]

        if not in_trade and pos != 0:
            in_trade   = True
            entry_date = date
            entry_pos  = pos
            trade_pnl  = 0.0

        if in_trade:
            trade_pnl += pnl
            if pos == 0:
                trades.append({
                    "pair":       f"{ticker_a}/{ticker_b}",
                    "side":       "long spread" if entry_pos == 1 else "short spread",
                    "entry_date": entry_date,
                    "exit_date":  date,
                    "duration":   (date - entry_date).days,
                    "pnl (%)":    round(trade_pnl * 100, 4),
                })
                in_trade = False

    trade_df = pd.DataFrame(trades)
    return daily_pnl, trade_df


# ── Portfolio aggregation ──────────────────────────────────────────────────────

def run_portfolio(prices: pd.DataFrame, pairs: pd.DataFrame):
    """
    Runs all pairs and aggregates into an equal-weight portfolio.
    """
    all_returns = {}
    all_trades  = []

    print(f"Backtesting {len(pairs)} pairs over {TRADING_START} → {TRADING_END}...\n")

    for _, row in pairs.iterrows():
        a, b, hr = row["ticker_A"], row["ticker_B"], row["hedge_ratio"]

        # Skip if tickers not in price data
        if a not in prices.columns or b not in prices.columns:
            print(f"  Skipping {a}/{b} — not in price data")
            continue

        try:
            pnl, trades = backtest_pair(a, b, hr, prices)
            all_returns[f"{a}/{b}"] = pnl
            all_trades.append(trades)
            n_trades = len(trades)
            total_pnl = pnl.sum() * 100
            print(f"  {a:6s}/{b:6s}  trades={n_trades:3d}  total P&L={total_pnl:+.2f}%")
        except Exception as e:
            print(f"  {a}/{b} failed: {e}")

    if not all_returns:
        raise RuntimeError("No pairs backtested successfully.")

    # Equal-weight portfolio
    returns_df   = pd.DataFrame(all_returns).fillna(0)
    port_returns = returns_df.mean(axis=1)
    port_nav     = (1 + port_returns).cumprod()

    trade_log = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    return returns_df, port_returns, port_nav, trade_log


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    prices, pairs = load_inputs()
    print(f"Loaded {len(pairs)} pairs and {prices.shape[1]} tickers.\n")

    returns_df, port_returns, port_nav, trade_log = run_portfolio(prices, pairs)

    # ── Save outputs ───────────────────────────────────────────────────────────
    returns_df.to_csv(os.path.join(DATA_DIR, "pair_returns.csv"))
    port_nav.to_frame("nav").to_csv(os.path.join(DATA_DIR, "portfolio_nav.csv"))
    if not trade_log.empty:
        trade_log.to_csv(os.path.join(DATA_DIR, "trade_log.csv"), index=False)

    # ── Summary ────────────────────────────────────────────────────────────────
    cagr    = (port_nav.iloc[-1]) ** (252 / len(port_returns)) - 1
    vol     = port_returns.std() * np.sqrt(252)
    sharpe  = (port_returns.mean() - 0.04/252) / port_returns.std() * np.sqrt(252)
    mdd     = ((port_nav - port_nav.cummax()) / port_nav.cummax()).min()
    n_trades = len(trade_log) if not trade_log.empty else 0
    win_rate = (trade_log["pnl (%)"] > 0).mean() * 100 if n_trades > 0 else 0

    print(f"\n── Portfolio Summary ({TRADING_START} → {TRADING_END}) ────────────")
    print(f"  Pairs traded  : {returns_df.shape[1]}")
    print(f"  Total trades  : {n_trades}")
    print(f"  Win rate      : {win_rate:.1f}%")
    print(f"  CAGR          : {cagr*100:.2f}%")
    print(f"  Ann. Vol      : {vol*100:.2f}%")
    print(f"  Sharpe        : {sharpe:.3f}")
    print(f"  Max Drawdown  : {mdd*100:.2f}%")
    print("──────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
