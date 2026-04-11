"""
dashboard.py
------------
Interactive Streamlit risk dashboard.

Sections:
  1. Sidebar  — portfolio selector, date range, confidence level
  2. KPI cards — VaR, CVaR, Sharpe, Sortino, Max Drawdown
  3. Return distribution with VaR / CVaR overlays
  4. VaR comparison — Historical vs Parametric vs Monte Carlo
  5. Rolling volatility (21-day and 63-day)
  6. Drawdown chart
  7. Correlation heatmap
  8. Per-ticker risk table

Run with:
    streamlit run dashboard.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import streamlit as st

import risk_metrics as rm

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Risk Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared Plotly theme ────────────────────────────────────────────────────────
LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#FAFAFA", family="Inter, sans-serif", size=12),
    xaxis=dict(gridcolor="#2D3748", zeroline=False),
    yaxis=dict(gridcolor="#2D3748", zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    margin=dict(l=0, r=0, t=40, b=0),
)

SECTOR_MAP = {
    "AAPL": "Technology",    "MSFT": "Technology",   "NVDA": "Technology",
    "GOOGL": "Communication","META": "Communication",
    "AMZN": "Cons. Discr.",  "TSLA": "Cons. Discr.",
    "PG":   "Cons. Staples", "KO":   "Cons. Staples",
    "JPM":  "Financials",    "GS":   "Financials",
    "JNJ":  "Healthcare",    "UNH":  "Healthcare",
    "CAT":  "Industrials",   "HON":  "Industrials",
    "XOM":  "Energy",        "CVX":  "Energy",
    "NEE":  "Utilities",
    "AMT":  "Real Estate",
    "SPY":  "Benchmark",
}

SECTOR_COLORS = {
    "Technology":    "#2196F3",
    "Communication": "#9C27B0",
    "Cons. Discr.":  "#FF9800",
    "Cons. Staples": "#4CAF50",
    "Financials":    "#F44336",
    "Healthcare":    "#00BCD4",
    "Industrials":   "#FF5722",
    "Energy":        "#FFC107",
    "Utilities":     "#8BC34A",
    "Real Estate":   "#795548",
    "Benchmark":     "#9E9E9E",
}

# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    prices  = pd.read_csv("data/prices.csv",      index_col=0, parse_dates=True)
    returns = pd.read_csv("data/log_returns.csv",  index_col=0, parse_dates=True)
    return prices, returns


prices, all_returns = load_data()
all_tickers = [t for t in all_returns.columns if t != "SPY"]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Portfolio Settings")

    selected_tickers = st.multiselect(
        "Select holdings",
        options=all_tickers,
        default=all_tickers,
    )

    date_range = st.date_input(
        "Date range",
        value=(all_returns.index[0].date(), all_returns.index[-1].date()),
        min_value=all_returns.index[0].date(),
        max_value=all_returns.index[-1].date(),
    )

    confidence = st.select_slider(
        "VaR confidence level",
        options=[0.90, 0.95, 0.99],
        value=0.95,
        format_func=lambda x: f"{int(x*100)}%",
    )

    weight_scheme = st.radio(
        "Portfolio weighting",
        ["Equal weight", "Market cap proxy"],
        index=0,
    )

    st.divider()
    st.markdown("## About")
    st.markdown(
        "Risk metrics for a diversified 20-stock portfolio spanning 10 GICS "
        "sectors. Benchmark: SPY. Risk-free rate: 4% (2018–2024 T-bill avg)."
    )

# ── Filter data by selections ──────────────────────────────────────────────────
if not selected_tickers:
    st.warning("Select at least one ticker in the sidebar.")
    st.stop()

start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
returns = all_returns.loc[start:end, selected_tickers + ["SPY"]].dropna(how="all")
spy_ret = returns["SPY"]

# ── Build equal-weight portfolio return series ─────────────────────────────────
port_returns = returns[selected_tickers].mean(axis=1).rename("Portfolio")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# Risk Metrics Dashboard")
st.markdown(
    f"Equal-weight portfolio · {len(selected_tickers)} holdings · "
    f"`{start.date()}` → `{end.date()}` · "
    f"VaR confidence: **{int(confidence*100)}%**"
)
st.divider()

# ── Section 1: KPI cards ───────────────────────────────────────────────────────
st.markdown("### Key Risk Metrics")

var_h  = rm.var_historical(port_returns, confidence)
cvar_v = rm.cvar(port_returns, confidence)
sh     = rm.sharpe_ratio(port_returns)
so     = rm.sortino_ratio(port_returns)
mdd    = rm.max_drawdown(port_returns)
cal    = rm.calmar_ratio(port_returns)
ann_vol= port_returns.std() * np.sqrt(252) * 100
beta_v = rm.beta(port_returns, spy_ret)

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    st.metric(
        f"VaR {int(confidence*100)}% (daily)",
        f"{var_h*100:.2f}%",
        help=f"Historical: worst expected daily loss {int(confidence*100)}% of the time",
    )
with c2:
    st.metric(
        f"CVaR {int(confidence*100)}% (daily)",
        f"{cvar_v*100:.2f}%",
        help="Expected Shortfall: avg loss on the worst days beyond VaR",
    )
with c3:
    st.metric("Sharpe Ratio", f"{sh:.3f}",
              help="Annualized excess return per unit of total volatility")
with c4:
    st.metric("Sortino Ratio", f"{so:.3f}",
              help="Annualized excess return per unit of downside volatility only")
with c5:
    st.metric("Max Drawdown", f"{mdd*100:.2f}%",
              help="Largest peak-to-trough decline over the period")
with c6:
    st.metric("Beta to SPY", f"{beta_v:.3f}",
              help="Sensitivity of portfolio returns to SPY moves")

st.divider()

# ── Section 2: Return distribution + VaR comparison ───────────────────────────
st.markdown("### Return Distribution & VaR Methods")

col_dist, col_var = st.columns([3, 2], gap="large")

with col_dist:
    # KDE histogram via plotly figure_factory
    hist_data = [port_returns.dropna().values.tolist()]
    fig_dist = ff.create_distplot(
        hist_data, ["Portfolio"],
        bin_size=0.002, show_rug=False,
        colors=["#2196F3"],
    )
    fig_dist.update_traces(
        selector=dict(type="histogram"),
        marker_color="rgba(33,150,243,0.25)",
        marker_line_color="rgba(33,150,243,0.5)",
    )

    # VaR lines
    var_p  = rm.var_parametric(port_returns, confidence)
    var_mc = rm.var_monte_carlo(port_returns, confidence)

    for val, label, color in [
        (var_h,  f"VaR Historical ({int(confidence*100)}%)",  "#FFA726"),
        (var_p,  f"VaR Parametric ({int(confidence*100)}%)",  "#EF5350"),
        (cvar_v, f"CVaR ({int(confidence*100)}%)",            "#B71C1C"),
    ]:
        fig_dist.add_vline(
            x=val, line_dash="dash", line_color=color, line_width=1.8,
            annotation_text=label,
            annotation_position="top right",
            annotation_font=dict(color=color, size=10),
        )

    # Shade the tail
    x_fill = np.linspace(port_returns.min(), var_h, 200)
    fig_dist.add_trace(go.Scatter(
        x=np.concatenate([x_fill, x_fill[::-1]]),
        y=np.concatenate([np.zeros(200),
                          [fig_dist.data[1].y[
                              np.argmin(np.abs(np.array(fig_dist.data[1].x) - v))]
                           for v in x_fill[::-1]]]),
        fill="toself",
        fillcolor="rgba(183,28,28,0.15)",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig_dist.update_layout(**LAYOUT, height=360,
                           title="Daily Return Distribution with VaR/CVaR")
    fig_dist.update_xaxes(tickformat=".1%", title="Daily Return")
    fig_dist.update_yaxes(title="Density")
    st.plotly_chart(fig_dist, width="stretch")

with col_var:
    st.markdown("#### VaR Method Comparison")
    st.caption(f"All at {int(confidence*100)}% confidence level")

    var_methods = {
        "Historical":    var_h,
        "Parametric":    var_p,
        "Monte Carlo":   var_mc,
        f"CVaR (ES)":    cvar_v,
    }

    fig_bar = go.Figure(go.Bar(
        x=[v * 100 for v in var_methods.values()],
        y=list(var_methods.keys()),
        orientation="h",
        marker_color=["#FFA726", "#EF5350", "#FF7043", "#B71C1C"],
        text=[f"{v*100:.3f}%" for v in var_methods.values()],
        textposition="outside",
        textfont=dict(color="#FAFAFA"),
    ))
    fig_bar.update_layout(**LAYOUT, height=200, title="Daily VaR (negative = loss)")
    fig_bar.update_xaxes(ticksuffix="%", title_text="Return (%)")
    st.plotly_chart(fig_bar, width="stretch")

    st.markdown("**Interpretation**")
    st.markdown(f"""
- **Historical**: reads directly from data — no assumptions
- **Parametric**: assumes normal returns — typically *underestimates* tail risk
- **Monte Carlo**: simulates 100k paths from fitted distribution
- **CVaR**: avg loss *beyond* VaR — always worse than VaR
    """)

st.divider()

# ── Section 3: Rolling volatility ─────────────────────────────────────────────
st.markdown("### Rolling Volatility")

vol_21 = rm.rolling_volatility(port_returns, 21)   # ~1 month
vol_63 = rm.rolling_volatility(port_returns, 63)   # ~1 quarter

fig_vol = go.Figure()
fig_vol.add_trace(go.Scatter(
    x=vol_21.index, y=vol_21 * 100,
    name="21-day (monthly)",
    line=dict(color="#2196F3", width=1.5),
    hovertemplate="%{x|%Y-%m-%d}<br>Vol: %{y:.1f}%<extra>21-day</extra>",
))
fig_vol.add_trace(go.Scatter(
    x=vol_63.index, y=vol_63 * 100,
    name="63-day (quarterly)",
    line=dict(color="#FF9800", width=2),
    hovertemplate="%{x|%Y-%m-%d}<br>Vol: %{y:.1f}%<extra>63-day</extra>",
))

# Shade COVID and rate-shock regimes
for x0, x1, label in [
    ("2020-02-19", "2020-06-01", "COVID crash"),
    ("2022-01-01", "2022-12-31", "Rate shock"),
]:
    fig_vol.add_vrect(
        x0=x0, x1=x1, fillcolor="#FFA726", opacity=0.08, line_width=0,
        annotation_text=label, annotation_position="top left",
        annotation_font=dict(color="#FFA726", size=10),
    )

long_run_avg = port_returns.std() * np.sqrt(252) * 100
fig_vol.add_hline(
    y=long_run_avg, line_dash="dot", line_color="#9E9E9E", line_width=1,
    annotation_text=f"Full-period avg: {long_run_avg:.1f}%",
    annotation_position="right",
    annotation_font=dict(color="#9E9E9E", size=10),
)

fig_vol.update_layout(**LAYOUT, height=320, title="Annualized Rolling Volatility")
fig_vol.update_yaxes(ticksuffix="%")
st.plotly_chart(fig_vol, width="stretch")

st.divider()

# ── Section 4: Drawdown chart ──────────────────────────────────────────────────
col_dd, col_dur = st.columns([3, 2], gap="large")

with col_dd:
    st.markdown("### Drawdown (Underwater Curve)")
    dd_series = rm.drawdown_series(port_returns) * 100

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dd_series.index, y=dd_series,
        fill="tozeroy",
        fillcolor="rgba(239,83,80,0.2)",
        line=dict(color="#EF5350", width=1.5),
        name="Drawdown",
        hovertemplate="%{x|%Y-%m-%d}<br>DD: %{y:.2f}%<extra></extra>",
    ))
    fig_dd.add_hline(
        y=mdd * 100, line_dash="dot", line_color="#B71C1C", line_width=1.2,
        annotation_text=f"Max DD: {mdd*100:.1f}%",
        annotation_position="right",
        annotation_font=dict(color="#B71C1C"),
    )
    fig_dd.update_layout(**LAYOUT, height=320, title="Portfolio Drawdown")
    fig_dd.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig_dd, width="stretch")

with col_dur:
    st.markdown("### Top Drawdown Episodes")
    episodes = rm.drawdown_durations(port_returns)
    if not episodes.empty:
        top5 = (episodes.sort_values("Depth (%)")
                        .head(5)
                        .reset_index(drop=True))
        top5["Start"]  = pd.to_datetime(top5["Start"]).dt.strftime("%Y-%m-%d")
        top5["Trough"] = pd.to_datetime(top5["Trough"]).dt.strftime("%Y-%m-%d")
        top5["End"]    = pd.to_datetime(top5["End"]).dt.strftime("%Y-%m-%d")
        st.dataframe(top5, hide_index=True, width="stretch")
    else:
        st.info("No completed drawdown episodes in selected period.")

st.divider()

# ── Section 5: Correlation heatmap ────────────────────────────────────────────
st.markdown("### Correlation Heatmap")
st.caption("Pairwise Pearson correlation of daily log returns. High correlation = less diversification benefit.")

corr = rm.correlation_matrix(returns[selected_tickers])

# Sort by sector for visual grouping
ticker_order = sorted(
    selected_tickers,
    key=lambda t: (SECTOR_MAP.get(t, "Z"), t)
)
corr = corr.loc[ticker_order, ticker_order]

fig_corr = go.Figure(go.Heatmap(
    z=corr.values,
    x=corr.columns.tolist(),
    y=corr.index.tolist(),
    colorscale=[
        [0.0,  "#1565C0"],
        [0.5,  "#263238"],
        [1.0,  "#B71C1C"],
    ],
    zmin=-1, zmax=1, zmid=0,
    text=[[f"{v:.2f}" for v in row] for row in corr.values],
    texttemplate="%{text}",
    textfont=dict(size=9),
    colorbar=dict(
        title="ρ",
        tickvals=[-1, -0.5, 0, 0.5, 1],
        ticktext=["-1", "-0.5", "0", "0.5", "1"],
    ),
    hovertemplate="%{y} × %{x}<br>ρ = %{z:.3f}<extra></extra>",
))

n = len(ticker_order)
fig_corr.update_layout(**LAYOUT, height=max(400, n * 35),
                       title="Return Correlation Matrix (sorted by sector)")
fig_corr.update_xaxes(tickangle=-45, gridcolor="rgba(0,0,0,0)")
fig_corr.update_yaxes(gridcolor="rgba(0,0,0,0)", autorange="reversed")
st.plotly_chart(fig_corr, width="stretch")

st.divider()

# ── Section 6: Per-ticker risk table ──────────────────────────────────────────
st.markdown("### Per-Ticker Risk Summary")

table = rm.ticker_risk_table(returns[selected_tickers + ["SPY"]])

# Add sector column
table.insert(0, "Sector", [SECTOR_MAP.get(t, "—") for t in table.index])

# Color-code Sharpe: green if > 0.5, red if < 0
def style_sharpe(v):
    if isinstance(v, float):
        if v > 0.5:  return "color: #4CAF50"
        if v < 0:    return "color: #EF5350"
    return ""

def style_dd(v):
    if isinstance(v, float) and v < -20:
        return "color: #EF5350"
    return ""

st.dataframe(
    table.style
        .format({
            "CAGR (%)":     "{:.2f}%",
            "Ann. Vol (%)": "{:.2f}%",
            "Sharpe":       "{:.3f}",
            "Sortino":      "{:.3f}",
            "Max DD (%)":   "{:.2f}%",
            "VaR 95% (%)":  "{:.2f}%",
            "CVaR 95% (%)": "{:.2f}%",
            "Beta":         "{:.3f}",
        })
        .map(style_sharpe, subset=["Sharpe", "Sortino"])
        .map(style_dd, subset=["Max DD (%)"]),
    width="stretch",
)

st.divider()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.caption(
    "VaR / CVaR computed on daily log returns · "
    "Annualization: 252 trading days · "
    "Risk-free rate: 4% · "
    "Data: Yahoo Finance via yfinance"
)
