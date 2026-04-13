"""
dashboard.py
------------
Interactive Streamlit dashboard for the pairs trading strategy.

Sections:
  1. Sidebar  — pair selector, z-score thresholds, date range
  2. KPI cards — total trades, win rate, CAGR, Sharpe, Max DD
  3. Spread chart — price ratio, spread, z-score with entry/exit signals
  4. Cumulative NAV — portfolio and individual pairs
  5. Trade log table — every trade with P&L and duration
  6. P&L distribution — histogram of trade returns
  7. Holding period distribution

Run with:
    streamlit run dashboard.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import spread_model as sm_mod

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pairs Trading",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#FAFAFA", family="Inter, sans-serif", size=12),
    xaxis=dict(gridcolor="#2D3748", zeroline=False),
    yaxis=dict(gridcolor="#2D3748", zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    margin=dict(l=0, r=0, t=40, b=0),
)

_BASE       = "../momentum-strategy/data/sp500_prices.csv"
TRADING_START = "2022-01-01"
TRADING_END   = "2024-12-31"


# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data
def load_all():
    import os
    base = os.path.dirname(os.path.dirname(__file__))

    prices    = pd.read_csv(
        os.path.join(base, "momentum-strategy", "data", "sp500_prices.csv"),
        index_col=0, parse_dates=True,
    )
    top_pairs = pd.read_csv("data/top_pairs.csv")
    pair_ret  = pd.read_csv("data/pair_returns.csv",   index_col=0, parse_dates=True)
    port_nav  = pd.read_csv("data/portfolio_nav.csv",  index_col=0, parse_dates=True)
    trade_log = pd.read_csv("data/trade_log.csv") if _file_exists("data/trade_log.csv") else pd.DataFrame()

    return prices, top_pairs, pair_ret, port_nav, trade_log


def _file_exists(path):
    import os
    return os.path.exists(path)


prices, top_pairs, pair_ret, port_nav, trade_log = load_all()

pair_labels = [f"{r.ticker_A} / {r.ticker_B}" for _, r in top_pairs.iterrows()
               if f"{r.ticker_A}/{r.ticker_B}" in pair_ret.columns]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Pair Inspector")

    selected_label = st.selectbox("Select pair", pair_labels)
    a, b = [t.strip() for t in selected_label.split("/")]

    pair_row = top_pairs[(top_pairs.ticker_A == a) & (top_pairs.ticker_B == b)].iloc[0]
    hedge_ratio = pair_row["hedge_ratio"]

    st.divider()
    st.markdown("### Signal Thresholds")
    entry_z = st.slider("Entry |z|",  1.0, 3.0, sm_mod.ENTRY_Z, 0.1)
    exit_z  = st.slider("Exit |z|",   0.1, 1.5, sm_mod.EXIT_Z,  0.1)
    stop_z  = st.slider("Stop |z|",   2.0, 5.0, sm_mod.STOP_Z,  0.1)

    st.divider()
    st.markdown("### Pair Stats")
    st.markdown(f"""
| | |
|--|--|
| **Hedge ratio β** | {hedge_ratio:.4f} |
| **Half-life** | {pair_row['half_life']:.1f} days |
| **Coint. p-value** | {pair_row['coint_pval']:.5f} |
| **Correlation** | {pair_row['correlation']:.4f} |
| **Spread mean** | {pair_row['spread_mean']:.4f} |
| **Spread σ** | {pair_row['spread_std']:.4f} |
""")
    st.divider()
    st.markdown(
        "Formation: 2018–2021 · Trading: 2022–2024 · "
        "Costs: 10 bps/leg · Equal-weight portfolio"
    )

# ── Build selected pair spread ──────────────────────────────────────────────────
@st.cache_data
def get_pair_spread(a, b, hedge_ratio, entry_z, exit_z, stop_z):
    pair_spread = sm_mod.build_pair_spread(a, b, prices, hedge_ratio)
    signals     = sm_mod.generate_signals(pair_spread.z_score, entry_z, exit_z, stop_z)
    return pair_spread, signals

pair_spread, signals = get_pair_spread(a, b, hedge_ratio, entry_z, exit_z, stop_z)

# Slice to trading period for display
z_trade  = pair_spread.z_score.loc[TRADING_START:TRADING_END]
sig_trade = signals.loc[TRADING_START:TRADING_END]
sp_trade  = pair_spread.spread.loc[TRADING_START:TRADING_END]

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# Pairs Trading / Statistical Arbitrage")
st.markdown(
    f"Market-neutral mean-reversion strategy · Engle-Granger cointegration · "
    f"Trading period: `{TRADING_START}` → `{TRADING_END}`"
)
st.divider()

# ── Section 1: KPI cards ───────────────────────────────────────────────────────
st.markdown("### Portfolio Performance")

port_returns = port_nav["nav"].pct_change().dropna()
cagr    = port_nav["nav"].iloc[-1] ** (252 / len(port_returns)) - 1
vol     = port_returns.std() * np.sqrt(252)
sharpe  = (port_returns.mean() - 0.04/252) / port_returns.std() * np.sqrt(252)
mdd     = ((port_nav["nav"] - port_nav["nav"].cummax()) / port_nav["nav"].cummax()).min()

pair_trades = trade_log[trade_log["pair"] == f"{a}/{b}"] if not trade_log.empty else pd.DataFrame()
total_trades = len(trade_log) if not trade_log.empty else 0
win_rate     = (trade_log["pnl (%)"] > 0).mean() * 100 if total_trades > 0 else 0
avg_duration = trade_log["duration"].mean() if total_trades > 0 else 0

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Trades",   f"{total_trades}")
c2.metric("Win Rate",       f"{win_rate:.1f}%")
c3.metric("CAGR",           f"{cagr*100:.2f}%")
c4.metric("Sharpe",         f"{sharpe:.3f}")
c5.metric("Max Drawdown",   f"{mdd*100:.2f}%")
c6.metric("Avg Hold (days)",f"{avg_duration:.0f}")

st.divider()

# ── Section 2: Spread chart ────────────────────────────────────────────────────
st.markdown(f"### Spread Analysis — {a} / {b}")
st.caption(
    f"Hedge ratio β = {hedge_ratio:.4f} · "
    f"Half-life = {pair_spread.half_life:.1f} days · "
    f"Formation period hedge ratio applied to full sample"
)

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    row_heights=[0.35, 0.25, 0.40],
    vertical_spacing=0.04,
    subplot_titles=[
        f"Normalised Price — {a} (blue) vs {b} (orange)",
        "Log-Price Spread",
        "Rolling Z-Score with Trade Signals",
    ],
)

# Row 1: normalised prices
pa_norm = (prices[a].loc[TRADING_START:TRADING_END] /
           prices[a].loc[TRADING_START:TRADING_END].iloc[0])
pb_norm = (prices[b].loc[TRADING_START:TRADING_END] /
           prices[b].loc[TRADING_START:TRADING_END].iloc[0])

fig.add_trace(go.Scatter(x=pa_norm.index, y=pa_norm,
    name=a, line=dict(color="#2196F3", width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=pb_norm.index, y=pb_norm,
    name=b, line=dict(color="#FF9800", width=1.5)), row=1, col=1)

# Row 2: spread
fig.add_trace(go.Scatter(x=sp_trade.index, y=sp_trade,
    name="Spread", line=dict(color="#9C27B0", width=1.3),
    fill="tozeroy", fillcolor="rgba(156,39,176,0.08)"), row=2, col=1)

# Row 3: z-score + signals
fig.add_trace(go.Scatter(x=z_trade.index, y=z_trade,
    name="Z-score", line=dict(color="#FAFAFA", width=1.2),
    hovertemplate="%{x|%Y-%m-%d}<br>z = %{y:.2f}<extra></extra>"), row=3, col=1)

# Threshold lines
for val, color, dash in [
    ( entry_z, "#4CAF50", "dash"),
    (-entry_z, "#4CAF50", "dash"),
    ( exit_z,  "#FFC107", "dot"),
    (-exit_z,  "#FFC107", "dot"),
    ( stop_z,  "#EF5350", "longdash"),
    (-stop_z,  "#EF5350", "longdash"),
]:
    fig.add_hline(y=val, line_dash=dash, line_color=color,
                  line_width=1, opacity=0.7, row=3, col=1)

# Entry/exit markers
long_entries  = z_trade[(sig_trade.diff() > 0) & (sig_trade == 1)]
short_entries = z_trade[(sig_trade.diff() < 0) & (sig_trade == -1)]
exits         = z_trade[sig_trade.diff().abs() > 0 & (sig_trade == 0)]

for dates, marker_color, symbol, label in [
    (long_entries.index,  "#4CAF50", "triangle-up",   "Long entry"),
    (short_entries.index, "#EF5350", "triangle-down",  "Short entry"),
    (exits.index,         "#FFC107", "circle",          "Exit"),
]:
    if len(dates):
        fig.add_trace(go.Scatter(
            x=dates, y=z_trade.reindex(dates),
            mode="markers", name=label,
            marker=dict(color=marker_color, size=8, symbol=symbol),
        ), row=3, col=1)

fig.update_layout(**LAYOUT, height=680, showlegend=True)
fig.update_yaxes(gridcolor="#2D3748")
st.plotly_chart(fig, width="stretch")

st.divider()

# ── Section 3: Portfolio NAV ───────────────────────────────────────────────────
col_nav, col_pairs = st.columns([3, 2], gap="large")

with col_nav:
    st.markdown("### Portfolio Cumulative NAV")
    fig_nav = go.Figure()

    # Individual pair NAVs (muted)
    for col in pair_ret.columns:
        pair_nav = (1 + pair_ret[col].loc[TRADING_START:TRADING_END]).cumprod()
        fig_nav.add_trace(go.Scatter(
            x=pair_nav.index, y=pair_nav,
            name=col, line=dict(width=0.8),
            opacity=0.3, showlegend=False,
            hovertemplate=f"<b>{col}</b><br>%{{x|%Y-%m-%d}}<br>NAV=%{{y:.3f}}<extra></extra>",
        ))

    # Portfolio NAV (bold)
    fig_nav.add_trace(go.Scatter(
        x=port_nav.index, y=port_nav["nav"],
        name="Portfolio (equal-weight)",
        line=dict(color="#4CAF50", width=2.5),
        hovertemplate="%{x|%Y-%m-%d}<br>NAV=%{y:.3f}<extra></extra>",
    ))
    fig_nav.add_hline(y=1, line_dash="dot", line_color="#555", line_width=1)
    fig_nav.update_layout(**LAYOUT, height=360, title="Cumulative NAV — All Pairs + Portfolio")
    st.plotly_chart(fig_nav, width="stretch")

with col_pairs:
    st.markdown("### Top Pairs by Total P&L")
    if not trade_log.empty:
        pair_pnl = (trade_log.groupby("pair")["pnl (%)"].sum()
                              .sort_values(ascending=False)
                              .reset_index())
        colors = ["#4CAF50" if v > 0 else "#EF5350" for v in pair_pnl["pnl (%)"]]
        fig_pnl = go.Figure(go.Bar(
            x=pair_pnl["pnl (%)"],
            y=pair_pnl["pair"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.2f}%" for v in pair_pnl["pnl (%)"]],
            textposition="outside",
            textfont=dict(color="#FAFAFA"),
        ))
        fig_pnl.update_layout(**LAYOUT, height=360, title="Total P&L by Pair (%)")
        fig_pnl.update_xaxes(ticksuffix="%")
        st.plotly_chart(fig_pnl, width="stretch")

st.divider()

# ── Section 4: Trade log ───────────────────────────────────────────────────────
col_log, col_dist = st.columns(2, gap="large")

with col_log:
    st.markdown("### Trade Log")
    if not trade_log.empty:
        display = trade_log.copy()

        def color_pnl(v):
            try:
                return "color: #4CAF50" if float(v) > 0 else "color: #EF5350"
            except Exception:
                return ""

        st.dataframe(
            display.style.map(color_pnl, subset=["pnl (%)"]),
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("No trades recorded.")

with col_dist:
    st.markdown("### P&L Distribution")
    if not trade_log.empty:
        pnls = trade_log["pnl (%)"]
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=pnls,
            nbinsx=40,
            marker_color="#2196F3",
            marker_line_color="rgba(33,150,243,0.5)",
            marker_line_width=0.5,
            name="Trade P&L",
        ))
        fig_hist.add_vline(x=0, line_color="#9E9E9E", line_width=1)
        fig_hist.add_vline(x=pnls.mean(), line_dash="dash",
                           line_color="#FFC107", line_width=1.5,
                           annotation_text=f"Mean: {pnls.mean():.3f}%",
                           annotation_font=dict(color="#FFC107"))
        fig_hist.update_layout(**LAYOUT, height=300,
                               title="Distribution of Trade P&L (%)")
        fig_hist.update_xaxes(ticksuffix="%")
        st.plotly_chart(fig_hist, width="stretch")

    st.markdown("### Holding Period Distribution")
    if not trade_log.empty:
        fig_dur = go.Figure(go.Histogram(
            x=trade_log["duration"],
            nbinsx=30,
            marker_color="#9C27B0",
        ))
        fig_dur.add_vline(
            x=trade_log["duration"].mean(), line_dash="dash",
            line_color="#FFC107", line_width=1.5,
            annotation_text=f"Mean: {trade_log['duration'].mean():.0f}d",
            annotation_font=dict(color="#FFC107"),
        )
        fig_dur.update_layout(**LAYOUT, height=260, title="Holding Period (days)")
        st.plotly_chart(fig_dur, width="stretch")

st.divider()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.caption(
    "Pairs selected via Engle-Granger cointegration (formation 2018–2021) · "
    "Traded out-of-sample (2022–2024) · "
    "10 bps/leg transaction costs · "
    "Data: Yahoo Finance via yfinance"
)
