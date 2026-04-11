"""
dashboard.py
------------
Interactive Streamlit dashboard for the S&P 500 momentum strategy.

Run with:
    streamlit run dashboard.py

Sections:
  1. Strategy parameters sidebar
  2. KPI cards — CAGR, Sharpe, Max Drawdown
  3. Cumulative NAV vs SPY benchmark
  4. Drawdown (underwater equity curve)
  5. Monthly returns heatmap
  6. Rolling 252-day Sharpe ratio
  7. Momentum leaderboard — current top/bottom signals
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Momentum Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Color palette ──────────────────────────────────────────────────────────────
COLORS = {
    "long":            "#2196F3",              # blue
    "long_fill":       "rgba(33,150,243,0.15)",
    "long_short":      "#4CAF50",              # green
    "long_short_fill": "rgba(76,175,80,0.15)",
    "short":           "#EF5350",              # red
    "short_fill":      "rgba(239,83,80,0.15)",
    "spy":             "#9E9E9E",              # gray
    "positive":        "#4CAF50",
    "negative":        "#EF5350",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#FAFAFA", family="Inter, sans-serif"),
    xaxis=dict(gridcolor="#2D3748", zeroline=False),
    yaxis=dict(gridcolor="#2D3748", zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    margin=dict(l=0, r=0, t=40, b=0),
)

# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    nav     = pd.read_csv("data/backtest_nav.csv",     index_col=0, parse_dates=True)
    returns = pd.read_csv("data/backtest_returns.csv", index_col=0, parse_dates=True)
    signals = pd.read_csv("data/momentum_signals.csv", index_col=0, parse_dates=True)
    perf    = pd.read_csv("data/performance_summary.csv", index_col=0)

    # Trim leading zero-return rows (before first signal kicks in)
    first_active = (returns["long"] != 0).idxmax()
    nav     = nav.loc[first_active:]
    returns = returns.loc[first_active:]

    # Re-base NAV to 1.0 at start of active period
    nav = nav / nav.iloc[0]

    return nav, returns, signals, perf


@st.cache_data
def load_spy(start: str, end: str) -> pd.Series:
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    prices = spy["Close"].squeeze()
    return (prices / prices.iloc[0]).rename("SPY")


def compute_drawdown(nav: pd.Series) -> pd.Series:
    peak = nav.cummax()
    return (nav - peak) / peak


def monthly_returns_pivot(daily_returns: pd.Series) -> pd.DataFrame:
    """Returns a (year × month) pivot of monthly P&L as percentages."""
    monthly = (1 + daily_returns).resample("ME").prod() - 1
    monthly = monthly * 100
    pivot = monthly.groupby([monthly.index.year, monthly.index.month]).first().unstack()
    pivot.index.name = "Year"
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]
    return pivot


def rolling_sharpe(returns: pd.Series, window: int = 252, rf: float = 0.04) -> pd.Series:
    excess = returns - rf / 252
    mu  = excess.rolling(window).mean()
    sig = excess.rolling(window).std()
    return (mu / sig * np.sqrt(252)).dropna()


# ── Load everything ────────────────────────────────────────────────────────────
nav, returns, signals, perf = load_data()
start_str = nav.index[0].strftime("%Y-%m-%d")
end_str   = nav.index[-1].strftime("%Y-%m-%d")
spy_nav   = load_spy(start_str, end_str).reindex(nav.index).ffill()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Strategy Parameters")
    st.markdown("""
| Parameter | Value |
|-----------|-------|
| Universe | S&P 500 |
| Signal | JT 12-1 momentum |
| Lookback | 12 months |
| Skip | 1 month |
| Rebalance | Monthly |
| Long leg | Top decile |
| Short leg | Bottom decile |
| Costs | 10 bps round-trip |
| Period | 2018 – 2024 |
| Tickers | 484 |
""")
    st.divider()
    st.markdown("## About")
    st.markdown(
        "Implements the **Jegadeesh & Titman (1993)** cross-sectional "
        "momentum strategy. Stocks are ranked monthly by their trailing "
        "12-1 month return; the top and bottom deciles form the long and "
        "short legs respectively."
    )
    st.divider()

    # Portfolio selector for drill-down charts
    selected = st.radio(
        "Highlight portfolio",
        ["long", "long_short"],
        format_func=lambda x: "Long-Only" if x == "long" else "Long-Short",
    )

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# S&P 500 Momentum Strategy")
st.markdown(
    f"Cross-sectional momentum · Jegadeesh & Titman (1993) · "
    f"`{start_str}` → `{end_str}`"
)
st.divider()

# ── Section 1: KPI cards ───────────────────────────────────────────────────────
st.markdown("### Performance at a Glance")

col1, col2, col3, col4, col5 = st.columns(5)

long_perf  = perf.loc["long"]
ls_perf    = perf.loc["long_short"]

with col1:
    st.metric(
        "CAGR (Long-Only)",
        f"{long_perf['CAGR (%)']:.2f}%",
        delta=f"{long_perf['CAGR (%)'] - ls_perf['CAGR (%)']:.2f}% vs L/S",
    )
with col2:
    st.metric(
        "Sharpe (Long-Only)",
        f"{long_perf['Sharpe']:.3f}",
        delta=f"{long_perf['Sharpe'] - ls_perf['Sharpe']:.3f} vs L/S",
    )
with col3:
    st.metric(
        "Max Drawdown",
        f"{long_perf['Max DD (%)']:.2f}%",
        delta=f"{long_perf['Max DD (%)'] - ls_perf['Max DD (%)']:.2f}% vs L/S",
        delta_color="inverse",
    )
with col4:
    st.metric(
        "Calmar Ratio",
        f"{long_perf['Calmar']:.3f}",
    )
with col5:
    st.metric(
        "Ann. Volatility",
        f"{long_perf['Vol (%)']:.2f}%",
    )

st.divider()

# ── Section 2: Cumulative NAV ──────────────────────────────────────────────────
st.markdown("### Cumulative NAV vs SPY Benchmark")

fig_nav = go.Figure()

for col, label, color, dash in [
    ("long",       "Long-Only",  COLORS["long"],       "solid"),
    ("long_short", "Long-Short", COLORS["long_short"], "dash"),
    ("short",      "Short-Only", COLORS["short"],      "dot"),
]:
    fig_nav.add_trace(go.Scatter(
        x=nav.index, y=nav[col],
        name=label,
        line=dict(color=color, width=2, dash=dash),
        hovertemplate=f"<b>{label}</b><br>Date: %{{x|%Y-%m-%d}}<br>NAV: %{{y:.3f}}<extra></extra>",
    ))

fig_nav.add_trace(go.Scatter(
    x=spy_nav.index, y=spy_nav,
    name="SPY (benchmark)",
    line=dict(color=COLORS["spy"], width=1.5, dash="longdash"),
    hovertemplate="<b>SPY</b><br>Date: %{x|%Y-%m-%d}<br>NAV: %{y:.3f}<extra></extra>",
))

fig_nav.add_hline(y=1.0, line_dash="dot", line_color="#555", line_width=1)

# Annotate key events
for date, label in [("2020-03-23", "COVID bottom"), ("2022-01-03", "Rate hike cycle")]:
    fig_nav.add_vline(
        x=date, line_dash="dash", line_color="#FFA726", line_width=1, opacity=0.6,
    )
    fig_nav.add_annotation(
        x=date, y=nav["long"].max() * 0.95,
        text=label, showarrow=False,
        font=dict(color="#FFA726", size=11),
        xanchor="left", xshift=6,
    )

fig_nav.update_layout(**PLOTLY_LAYOUT, height=420, title="Cumulative NAV (rebased to 1.0)")
st.plotly_chart(fig_nav, width="stretch")

st.divider()

# ── Section 3: Drawdown + Monthly heatmap ─────────────────────────────────────
col_dd, col_heat = st.columns([1, 1], gap="large")

with col_dd:
    st.markdown("### Drawdown (Underwater Curve)")

    highlight_color = COLORS["long"] if selected == "long" else COLORS["long_short"]
    highlight_label = "Long-Only" if selected == "long" else "Long-Short"

    fig_dd = go.Figure()

    # Muted traces for the others
    for col, label, color, fill in [
        ("long",       "Long-Only",  COLORS["long"],       COLORS["long_fill"]),
        ("long_short", "Long-Short", COLORS["long_short"], COLORS["long_short_fill"]),
    ]:
        dd = compute_drawdown(nav[col]) * 100
        opacity = 1.0 if col == selected else 0.25
        width   = 2   if col == selected else 1
        fig_dd.add_trace(go.Scatter(
            x=dd.index, y=dd,
            name=label,
            fill="tozeroy",
            line=dict(color=color, width=width),
            fillcolor=fill,
            opacity=opacity,
            hovertemplate=f"<b>{label}</b><br>%{{x|%Y-%m-%d}}<br>DD: %{{y:.1f}}%<extra></extra>",
        ))

    fig_dd.update_layout(**PLOTLY_LAYOUT, height=340, title=f"Max DD: {long_perf['Max DD (%)']:.1f}% (Long-Only)")
    fig_dd.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig_dd, width="stretch")

with col_heat:
    st.markdown("### Monthly Returns Heatmap")

    pivot = monthly_returns_pivot(returns[selected])
    # Drop months with no data
    pivot = pivot.dropna(how="all")

    # Compute annual total for the rightmost column
    annual = pivot.sum(axis=1).rename("Full Year")
    pivot_display = pd.concat([pivot, annual], axis=1)

    fig_heat = go.Figure(go.Heatmap(
        z=pivot_display.values,
        x=pivot_display.columns.tolist(),
        y=pivot_display.index.astype(str).tolist(),
        colorscale=[
            [0.0,  "#C62828"],
            [0.35, "#EF5350"],
            [0.5,  "#263238"],
            [0.65, "#66BB6A"],
            [1.0,  "#1B5E20"],
        ],
        zmid=0,
        text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row]
              for row in pivot_display.values],
        texttemplate="%{text}",
        textfont=dict(size=10),
        showscale=False,
        hovertemplate="<b>%{y} %{x}</b><br>Return: %{z:.2f}%<extra></extra>",
    ))

    label = "Long-Only" if selected == "long" else "Long-Short"
    fig_heat.update_layout(**PLOTLY_LAYOUT, height=340, title=f"Monthly P&L — {label}")
    st.plotly_chart(fig_heat, width="stretch")

st.divider()

# ── Section 4: Rolling Sharpe ──────────────────────────────────────────────────
st.markdown("### Rolling 252-Day Sharpe Ratio")

fig_rs = go.Figure()

for col, label, color, dash in [
    ("long",       "Long-Only",  COLORS["long"],       "solid"),
    ("long_short", "Long-Short", COLORS["long_short"], "dash"),
]:
    rs = rolling_sharpe(returns[col])
    fig_rs.add_trace(go.Scatter(
        x=rs.index, y=rs,
        name=label,
        line=dict(color=color, width=2, dash=dash),
        hovertemplate=f"<b>{label}</b><br>%{{x|%Y-%m-%d}}<br>Sharpe: %{{y:.2f}}<extra></extra>",
    ))

fig_rs.add_hline(y=0, line_dash="dot", line_color="#555", line_width=1)
fig_rs.add_hline(y=1, line_dash="dot", line_color="#4CAF50", line_width=0.8, opacity=0.4,
                 annotation_text="Sharpe = 1", annotation_position="right")

# Shade the COVID and rate-shock windows
for x0, x1, label in [
    ("2020-02-19", "2020-05-01", "COVID crash"),
    ("2022-01-01", "2022-12-31", "Rate shock"),
]:
    fig_rs.add_vrect(
        x0=x0, x1=x1,
        fillcolor="#FFA726", opacity=0.07,
        line_width=0,
        annotation_text=label,
        annotation_position="top left",
        annotation_font=dict(color="#FFA726", size=10),
    )

fig_rs.update_layout(**PLOTLY_LAYOUT, height=340, title="Rolling Sharpe (252-day window, rf = 4%)")
st.plotly_chart(fig_rs, width="stretch")

st.divider()

# ── Section 5: Momentum leaderboard ───────────────────────────────────────────
st.markdown("### Momentum Signal Leaderboard")
st.caption(f"Cross-sectional ranks as of last signal date: `{signals.index[-1].strftime('%Y-%m-%d')}`")

last_signal = signals.iloc[-1].dropna().sort_values()

col_top, col_bot = st.columns(2, gap="large")

with col_top:
    st.markdown("#### Top 10 — Long Portfolio")
    top10 = last_signal.tail(10).iloc[::-1]
    top_df = pd.DataFrame({
        "Ticker": top10.index,
        "Momentum Score": (top10.values * 100).round(1),
    })
    top_df["Momentum Score"] = top_df["Momentum Score"].apply(lambda x: f"{x:.1f}th pct")
    st.dataframe(
        top_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
            "Momentum Score": st.column_config.TextColumn("Rank (Percentile)"),
        },
    )

with col_bot:
    st.markdown("#### Bottom 10 — Short Portfolio")
    bot10 = last_signal.head(10)
    bot_df = pd.DataFrame({
        "Ticker": bot10.index,
        "Momentum Score": (bot10.values * 100).round(1),
    })
    bot_df["Momentum Score"] = bot_df["Momentum Score"].apply(lambda x: f"{x:.1f}th pct")
    st.dataframe(
        bot_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
            "Momentum Score": st.column_config.TextColumn("Rank (Percentile)"),
        },
    )

st.divider()

# ── Section 6: Full performance table ─────────────────────────────────────────
st.markdown("### Full Performance Summary")

display_perf = perf.copy()
display_perf.index = ["Long-Only", "Short-Only", "Long-Short"]
display_perf.columns = ["CAGR (%)", "Ann. Vol (%)", "Sharpe", "Max DD (%)", "Calmar", "Hit Rate (%)"]

st.dataframe(
    display_perf.style
        .format({
            "CAGR (%)":    "{:.2f}%",
            "Ann. Vol (%)": "{:.2f}%",
            "Sharpe":      "{:.3f}",
            "Max DD (%)":  "{:.2f}%",
            "Calmar":      "{:.3f}",
            "Hit Rate (%)": "{:.1f}%",
        })
        .background_gradient(subset=["CAGR (%)","Sharpe","Calmar"], cmap="RdYlGn")
        .background_gradient(subset=["Max DD (%)"], cmap="RdYlGn_r"),
    width="stretch",
)

st.divider()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.caption(
    "Strategy: Jegadeesh & Titman (1993) · "
    "Data: Yahoo Finance via yfinance · "
    "Universe: S&P 500 (484 constituents after cleaning) · "
    "Transaction costs: 10 bps round-trip"
)
