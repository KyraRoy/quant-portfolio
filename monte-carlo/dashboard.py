"""
dashboard.py
------------
Interactive Streamlit dashboard for the Monte Carlo portfolio simulator.

Sections:
  1. Sidebar  — portfolio builder, simulation settings
  2. Efficient frontier — with MVP, Max Sharpe, Equal Weight, and current portfolio
  3. Portfolio weights — donut chart
  4. Simulation fan chart — percentile paths over the horizon
  5. Terminal wealth distribution — histogram with VaR/CVaR
  6. Probability table — P(loss), P(+20%), P(double) at multiple horizons
  7. GBM vs Bootstrap comparison
  8. Horizon statistics table

Run with:
    streamlit run dashboard.py
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

import simulator as sim
import optimizer as opt

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Monte Carlo Simulator",
    page_icon="🎲",
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

SECTOR_COLORS = {
    "AAPL":"#2196F3","MSFT":"#2196F3","NVDA":"#2196F3",
    "GOOGL":"#9C27B0","META":"#9C27B0",
    "AMZN":"#FF9800","TSLA":"#FF9800",
    "PG":"#4CAF50","KO":"#4CAF50",
    "JPM":"#F44336","GS":"#F44336",
    "JNJ":"#00BCD4","UNH":"#00BCD4",
    "CAT":"#FF5722","HON":"#FF5722",
    "XOM":"#FFC107","CVX":"#FFC107",
    "NEE":"#8BC34A","AMT":"#795548","SPY":"#9E9E9E",
}

_BASE = os.path.dirname(os.path.dirname(__file__))
RETURNS_PATH = os.path.join(_BASE, "risk-dashboard", "data", "log_returns.csv")

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_returns():
    df = pd.read_csv(RETURNS_PATH, index_col=0, parse_dates=True)
    return df.drop(columns=["SPY"], errors="ignore")

all_returns = load_returns()
all_tickers = all_returns.columns.tolist()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Portfolio Builder")

    selected = st.multiselect(
        "Select assets",
        options=all_tickers,
        default=["AAPL","MSFT","JPM","JNJ","XOM","PG"],
    )

    if not selected:
        st.warning("Select at least 2 assets.")
        st.stop()

    st.divider()
    st.markdown("### Weights")
    weight_mode = st.radio(
        "Weighting",
        ["Equal weight", "Max Sharpe", "Min Variance", "Custom"],
    )

    returns_sel = all_returns[selected].dropna()

    # Pre-compute frontier to get special portfolios
    @st.cache_data
    def get_frontier(tickers_key):
        r = all_returns[list(tickers_key)].dropna()
        return opt.compute_efficient_frontier(r)

    frontier = get_frontier(tuple(selected))

    if weight_mode == "Equal weight":
        weights = np.ones(len(selected)) / len(selected)
    elif weight_mode == "Max Sharpe":
        weights = frontier.max_sharpe.weights
    elif weight_mode == "Min Variance":
        weights = frontier.min_variance.weights
    else:
        # Custom sliders
        raw = {}
        for t in selected:
            raw[t] = st.slider(t, 0, 100, 100 // len(selected), key=f"w_{t}")
        total = sum(raw.values()) or 1
        weights = np.array([raw[t] / total for t in selected])

    st.divider()
    st.markdown("### Simulation Settings")
    n_sims       = st.select_slider("Simulations",
                                    [500, 1_000, 5_000, 10_000], value=1_000)
    horizon_yrs  = st.slider("Horizon (years)", 1, 10, 3)
    horizon_days = horizon_yrs * 252
    method       = st.radio("Method", ["GBM", "Bootstrap", "Both"])
    target_mult  = st.slider("Target wealth multiple (×)", 1.0, 5.0, 2.0, 0.1,
                              help="Used in probability calculations")

    st.divider()
    st.caption("Based on historical daily log returns 2018–2024.")

# ── Run simulations ────────────────────────────────────────────────────────────
@st.cache_data
def run_sims(tickers, weights_tuple, n_sims, horizon_days, method):
    w = np.array(weights_tuple)
    r = all_returns[list(tickers)].dropna()
    results = {}
    if method in ("GBM", "Both"):
        results["GBM"] = sim.simulate_gbm(r, w, n_sims, horizon_days)
    if method in ("Bootstrap", "Both"):
        results["Bootstrap"] = sim.simulate_bootstrap(r, w, n_sims, horizon_days)
    return results

sim_results = run_sims(tuple(selected), tuple(weights), n_sims, horizon_days, method)
primary_result = sim_results.get("GBM") or sim_results.get("Bootstrap")

mu, cov, port_mu, port_vol = sim.estimate_parameters(returns_sel, weights)
port_sharpe = (port_mu - 0.04) / port_vol

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# Monte Carlo Portfolio Simulator")
st.markdown(
    f"{n_sims:,} simulations · {horizon_yrs}-year horizon · "
    f"{weight_mode} · {method}"
)
st.divider()

# ── Section 1: KPI cards ───────────────────────────────────────────────────────
st.markdown("### Portfolio Parameters")

var_95  = sim.simulated_var(primary_result,  0.95)
cvar_95 = sim.simulated_cvar(primary_result, 0.95)
p_target = sim.prob_reach_target(primary_result, target_mult)
median_tw = float(np.median(primary_result.terminal_wealth))

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Exp. Annual Return",  f"{port_mu*100:.2f}%")
c2.metric("Annual Volatility",   f"{port_vol*100:.2f}%")
c3.metric("Sharpe Ratio",        f"{port_sharpe:.3f}")
c4.metric(f"Simulated VaR 95% ({horizon_yrs}Y)",  f"{var_95*100:.1f}%")
c5.metric(f"Simulated CVaR 95% ({horizon_yrs}Y)", f"{cvar_95*100:.1f}%")
c6.metric(f"P(reach {target_mult:.1f}×)",         f"{p_target*100:.1f}%")

st.divider()

# ── Section 2: Efficient frontier + weights ────────────────────────────────────
col_ef, col_w = st.columns([3, 2], gap="large")

with col_ef:
    st.markdown("### Efficient Frontier")
    st.caption("Long-only, fully-invested. Each dot is a minimum-variance portfolio for a given return target.")

    f_vols = [p.vol * 100 for p in frontier.frontier_points]
    f_rets = [p.ret * 100 for p in frontier.frontier_points]
    f_sh   = [p.sharpe     for p in frontier.frontier_points]

    fig_ef = go.Figure()

    # Frontier curve
    fig_ef.add_trace(go.Scatter(
        x=f_vols, y=f_rets, mode="lines",
        line=dict(color="#2196F3", width=2),
        name="Efficient Frontier",
        hovertemplate="Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<extra>Frontier</extra>",
    ))

    # Individual assets
    asset_vols = [float(np.sqrt(cov[i,i])) * np.sqrt(252) * 100 for i in range(len(selected))]
    asset_rets = [float(mu[i]) * 252 * 100 for i in range(len(selected))]
    fig_ef.add_trace(go.Scatter(
        x=asset_vols, y=asset_rets, mode="markers+text",
        marker=dict(size=8, color=[SECTOR_COLORS.get(t,"#9E9E9E") for t in selected]),
        text=selected, textposition="top right", textfont=dict(size=9),
        name="Individual Assets",
        hovertemplate="%{text}<br>Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<extra></extra>",
    ))

    # Special portfolios
    for label, pt, color, sym in [
        ("Min Variance",  frontier.min_variance,  "#FFC107", "diamond"),
        ("Max Sharpe",    frontier.max_sharpe,    "#4CAF50", "star"),
        ("Equal Weight",  frontier.equal_weight,  "#9E9E9E", "circle"),
        ("Your Portfolio",_to_pt := opt._to_point(weights, mu/252, cov/252), "#EF5350", "x"),
    ]:
        fig_ef.add_trace(go.Scatter(
            x=[pt.vol*100], y=[pt.ret*100], mode="markers+text",
            marker=dict(size=14, color=color, symbol=sym),
            text=[label], textposition="bottom right", textfont=dict(size=10, color=color),
            name=label, showlegend=True,
        ))

    fig_ef.update_layout(**LAYOUT, height=400,
                          title="Mean-Variance Efficient Frontier")
    fig_ef.update_xaxes(ticksuffix="%", title_text="Annualized Volatility")
    fig_ef.update_yaxes(ticksuffix="%", title_text="Annualized Return")
    st.plotly_chart(fig_ef, width="stretch")

with col_w:
    st.markdown("### Portfolio Weights")

    w_series = opt.weights_to_series(weights, selected)
    nonzero  = w_series[w_series > 0.001]

    fig_donut = go.Figure(go.Pie(
        labels=nonzero.index.tolist(),
        values=nonzero.values.tolist(),
        hole=0.55,
        marker_colors=[SECTOR_COLORS.get(t, "#9E9E9E") for t in nonzero.index],
        textinfo="label+percent",
        textfont=dict(size=11),
        hovertemplate="%{label}: %{percent}<extra></extra>",
    ))
    fig_donut.update_layout(**LAYOUT, height=300, showlegend=False,
                             title=f"{weight_mode} Weights")
    st.plotly_chart(fig_donut, width="stretch")

    # Weight table
    wdf = pd.DataFrame({"Ticker": nonzero.index, "Weight": (nonzero.values * 100).round(1)})
    wdf["Weight"] = wdf["Weight"].apply(lambda x: f"{x:.1f}%")
    st.dataframe(wdf, hide_index=True, width="stretch")

st.divider()

# ── Section 3: Fan chart ───────────────────────────────────────────────────────
st.markdown("### Simulation Paths — Fan Chart")
st.caption(f"Shaded bands show 5th–95th and 25th–75th percentiles across {n_sims:,} simulations.")

days_axis  = np.arange(horizon_days + 1)
years_axis = days_axis / 252

METHOD_COLORS = {"GBM": "#2196F3", "Bootstrap": "#FF9800"}

for mname, result in sim_results.items():
    color = METHOD_COLORS[mname]

    fig_fan = go.Figure()

    # Sample of individual paths (muted)
    sample_idx = np.random.default_rng(0).integers(0, n_sims, min(80, n_sims))
    for idx in sample_idx:
        fig_fan.add_trace(go.Scatter(
            x=years_axis, y=result.paths[idx],
            line=dict(color=color, width=0.4),
            opacity=0.08, showlegend=False, hoverinfo="skip",
        ))

    # Percentile bands
    pcts = result.percentiles
    fig_fan.add_trace(go.Scatter(
        x=np.concatenate([years_axis, years_axis[::-1]]),
        y=np.concatenate([pcts[95], pcts[5][::-1]]),
        fill="toself", fillcolor=color.replace(")", ",0.10)").replace("rgb","rgba"),
        line=dict(width=0), name="5th–95th pct", showlegend=True,
        hoverinfo="skip",
    ))
    fig_fan.add_trace(go.Scatter(
        x=np.concatenate([years_axis, years_axis[::-1]]),
        y=np.concatenate([pcts[75], pcts[25][::-1]]),
        fill="toself", fillcolor=color.replace(")", ",0.20)").replace("rgb","rgba"),
        line=dict(width=0), name="25th–75th pct", showlegend=True,
        hoverinfo="skip",
    ))

    # Median and key percentiles
    for pct, lw, label in [(50,2.5,"Median"), (5,1.2,"5th pct"), (95,1.2,"95th pct")]:
        fig_fan.add_trace(go.Scatter(
            x=years_axis, y=pcts[pct],
            line=dict(color=color, width=lw,
                      dash="solid" if pct==50 else "dot"),
            name=label,
            hovertemplate=f"Year %{{x:.1f}}<br>{label}: %{{y:.3f}}<extra></extra>",
        ))

    fig_fan.add_hline(y=1.0, line_dash="dot", line_color="#555", line_width=1,
                      annotation_text="Starting NAV", annotation_position="right",
                      annotation_font=dict(color="#9E9E9E", size=10))

    fig_fan.update_layout(**LAYOUT, height=420,
                           title=f"Portfolio NAV — {mname}  ({horizon_yrs}-year horizon)")
    fig_fan.update_xaxes(title_text="Years", tickformat=".1f")
    fig_fan.update_yaxes(title_text="Portfolio NAV")
    st.plotly_chart(fig_fan, width="stretch")

st.divider()

# ── Section 4: Terminal wealth + probability table ─────────────────────────────
col_tw, col_prob = st.columns(2, gap="large")

with col_tw:
    st.markdown(f"### Terminal Wealth Distribution ({horizon_yrs}Y)")

    fig_tw = go.Figure()
    for mname, result in sim_results.items():
        fig_tw.add_trace(go.Histogram(
            x=result.terminal_wealth,
            nbinsx=60,
            name=mname,
            opacity=0.7,
            marker_color=METHOD_COLORS[mname],
            hovertemplate="NAV: %{x:.3f}<br>Count: %{y}<extra>" + mname + "</extra>",
        ))

    # VaR and median lines
    var_v  = sim.simulated_var(primary_result, 0.95) + 1.0
    med_v  = float(np.median(primary_result.terminal_wealth))
    cvar_v = sim.simulated_cvar(primary_result, 0.95) + 1.0

    for val, label, color in [
        (1.0,   "Break-even",     "#9E9E9E"),
        (var_v, "VaR 95%",        "#FFA726"),
        (cvar_v,"CVaR 95%",       "#EF5350"),
        (med_v, "Median",         "#4CAF50"),
        (target_mult, f"Target ({target_mult:.1f}×)", "#FFC107"),
    ]:
        fig_tw.add_vline(x=val, line_dash="dash", line_color=color, line_width=1.5,
                         annotation_text=label, annotation_position="top",
                         annotation_font=dict(color=color, size=10))

    fig_tw.update_layout(**LAYOUT, height=380, barmode="overlay",
                          title="Distribution of Final Portfolio Value")
    fig_tw.update_xaxes(title_text=f"NAV after {horizon_yrs} years")
    st.plotly_chart(fig_tw, width="stretch")

with col_prob:
    st.markdown("### Probability Table by Horizon")
    st.caption("Based on primary simulation method.")

    stats_df = sim.horizon_statistics(primary_result)
    st.dataframe(
        stats_df.style.format({
            "Median NAV": "{:.3f}",
            "5th Pct":    "{:.3f}",
            "95th Pct":   "{:.3f}",
            "P(loss)":    "{:.1f}%",
            "P(+20%)":    "{:.1f}%",
            "P(double)":  "{:.1f}%",
        }),
        hide_index=True,
        width="stretch",
    )

    st.divider()
    st.markdown("### GBM vs Bootstrap — Terminal Wealth")
    if len(sim_results) == 2:
        comp_data = {
            mname: {
                "Median":     round(float(np.median(r.terminal_wealth)), 4),
                "5th Pct":    round(float(np.percentile(r.terminal_wealth, 5)), 4),
                "95th Pct":   round(float(np.percentile(r.terminal_wealth, 95)), 4),
                f"P(>{target_mult:.1f}×)": f"{sim.prob_reach_target(r, target_mult)*100:.1f}%",
                "VaR 95%":   f"{sim.simulated_var(r, 0.95)*100:.1f}%",
                "CVaR 95%":  f"{sim.simulated_cvar(r, 0.95)*100:.1f}%",
            }
            for mname, r in sim_results.items()
        }
        st.dataframe(pd.DataFrame(comp_data), width="stretch")
    else:
        st.info("Select 'Both' in the sidebar to compare methods.")

st.divider()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.caption(
    "GBM: Geometric Brownian Motion with Cholesky correlation · "
    "Bootstrap: block resampling of historical returns (block=10 days) · "
    "Based on 2018–2024 daily log returns · "
    "Risk-free rate: 4%"
)
