import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.style import apply_style, page_header, section, kpi_row, CHART, C
import core.simulator as sim
import core.optimizer as opt

st.set_page_config(page_title="Monte Carlo Simulator", page_icon="🎲", layout="wide")
apply_style()

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

@st.cache_data
def load_returns():
    df = pd.read_csv(f"{DATA}/stock_log_returns.csv", index_col=0, parse_dates=True)
    return df.drop(columns=["SPY"], errors="ignore")

all_returns = load_returns()
all_tickers = all_returns.columns.tolist()

SECTOR_COLORS = {
    "AAPL":C["primary"],"MSFT":C["primary"],"NVDA":C["primary"],
    "GOOGL":C["purple"],"META":C["purple"],
    "AMZN":C["amber"],"TSLA":C["amber"],
    "PG":C["green"],"KO":C["green"],
    "JPM":C["red"],"GS":C["red"],
    "JNJ":C["cyan"],"UNH":C["cyan"],
    "CAT":"#FB923C","HON":"#FB923C",
    "XOM":"#FCD34D","CVX":"#FCD34D",
    "NEE":"#86EFAC","AMT":"#A78BFA",
}

with st.sidebar:
    st.markdown("### Portfolio Builder")
    selected = st.multiselect("Assets", all_tickers,
                               default=["AAPL","MSFT","JPM","JNJ","XOM","PG"])
    if not selected:
        st.warning("Select at least 2 assets.")
        st.stop()

    st.divider()
    weight_mode = st.radio("Weighting",
        ["Equal weight", "Max Sharpe", "Min Variance", "Custom"])

    returns_sel = all_returns[selected].dropna()

    @st.cache_data
    def get_frontier(tickers_key):
        return opt.compute_efficient_frontier(all_returns[list(tickers_key)].dropna())

    frontier = get_frontier(tuple(selected))

    if weight_mode == "Equal weight":
        weights = np.ones(len(selected)) / len(selected)
    elif weight_mode == "Max Sharpe":
        weights = frontier.max_sharpe.weights
    elif weight_mode == "Min Variance":
        weights = frontier.min_variance.weights
    else:
        raw = {t: st.slider(t, 0, 100, 100//len(selected), key=f"w_{t}") for t in selected}
        total = sum(raw.values()) or 1
        weights = np.array([raw[t]/total for t in selected])

    st.divider()
    st.markdown("### Simulation")
    n_sims      = st.select_slider("Simulations", [500,1_000,5_000,10_000], value=1_000)
    horizon_yrs = st.slider("Horizon (years)", 1, 10, 3)
    horizon_days = horizon_yrs * 252
    method      = st.radio("Method", ["GBM", "Bootstrap", "Both"])
    target_mult = st.slider("Target multiple (×)", 1.0, 5.0, 2.0, 0.1)

@st.cache_data
def run_sims(tickers, weights_t, n_sims, horizon_days, method):
    w = np.array(weights_t)
    r = all_returns[list(tickers)].dropna()
    out = {}
    if method in ("GBM","Both"):
        out["GBM"] = sim.simulate_gbm(r, w, n_sims, horizon_days)
    if method in ("Bootstrap","Both"):
        out["Bootstrap"] = sim.simulate_bootstrap(r, w, n_sims, horizon_days)
    return out

sim_results   = run_sims(tuple(selected), tuple(weights), n_sims, horizon_days, method)
primary       = sim_results.get("GBM") or sim_results.get("Bootstrap")
mu, cov, port_mu, port_vol = sim.estimate_parameters(returns_sel, weights)
port_sharpe   = (port_mu - 0.04) / port_vol

page_header("Monte Carlo Portfolio Simulator",
            f"{n_sims:,} simulations · {horizon_yrs}-year horizon · {weight_mode} · {method}")

kpi_row([
    {"label": "Exp. Annual Return", "value": f"{port_mu*100:.2f}%",               "color": C["green"]},
    {"label": "Annual Volatility",  "value": f"{port_vol*100:.2f}%",              "color": C["amber"]},
    {"label": "Sharpe Ratio",       "value": f"{port_sharpe:.3f}",                "color": C["primary"]},
    {"label": f"VaR 95% ({horizon_yrs}Y)",  "value": f"{sim.simulated_var(primary,.95)*100:.1f}%",  "color": C["red"]},
    {"label": f"CVaR 95% ({horizon_yrs}Y)", "value": f"{sim.simulated_cvar(primary,.95)*100:.1f}%", "color": C["red"]},
    {"label": f"P(reach {target_mult:.1f}×)", "value": f"{sim.prob_reach_target(primary,target_mult)*100:.1f}%", "color": C["cyan"]},
])

# ── Efficient frontier + Weights ───────────────────────────────────────────────
section("Efficient Frontier & Portfolio Weights")
c1, c2 = st.columns([3, 2], gap="large")

with c1:
    f_vols = [p.vol*100 for p in frontier.frontier_points]
    f_rets = [p.ret*100 for p in frontier.frontier_points]

    fig_ef = go.Figure()
    fig_ef.add_trace(go.Scatter(
        x=f_vols, y=f_rets, mode="lines",
        line=dict(color=C["primary"], width=2.5),
        name="Efficient Frontier",
        hovertemplate="Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<extra>Frontier</extra>",
    ))

    # Individual assets
    asset_vols = [float(np.sqrt(cov[i,i])) * np.sqrt(252) * 100 for i in range(len(selected))]
    asset_rets = [float(mu[i]) * 252 * 100 for i in range(len(selected))]
    fig_ef.add_trace(go.Scatter(
        x=asset_vols, y=asset_rets, mode="markers+text",
        marker=dict(size=9, color=[SECTOR_COLORS.get(t, C["muted"]) for t in selected],
                    line=dict(color="#0F172A", width=1)),
        text=selected, textposition="top right", textfont=dict(size=9, color="#94A3B8"),
        name="Individual Assets",
    ))

    my_pt = opt._to_point(weights, mu, cov)
    for label, pt, color, sym, size in [
        ("Min Variance", frontier.min_variance, C["amber"],  "diamond", 14),
        ("Max Sharpe",   frontier.max_sharpe,   C["green"],  "star",    16),
        ("Equal Weight", frontier.equal_weight, C["muted"],  "circle",  12),
        ("Your Portfolio", my_pt,               C["red"],    "x",       14),
    ]:
        fig_ef.add_trace(go.Scatter(
            x=[pt.vol*100], y=[pt.ret*100], mode="markers+text",
            marker=dict(size=size, color=color, symbol=sym,
                        line=dict(color="#0F172A", width=1.5)),
            text=[label], textposition="bottom right",
            textfont=dict(size=10, color=color), name=label,
        ))

    fig_ef.update_layout(**CHART, height=380, title="Mean-Variance Efficient Frontier (long-only)")
    fig_ef.update_xaxes(ticksuffix="%", title_text="Annualized Volatility")
    fig_ef.update_yaxes(ticksuffix="%", title_text="Annualized Return")
    st.plotly_chart(fig_ef, width="stretch")

with c2:
    w_s     = opt.weights_to_series(weights, selected)
    nonzero = w_s[w_s > 0.001]
    fig_d   = go.Figure(go.Pie(
        labels=nonzero.index.tolist(),
        values=nonzero.values.tolist(),
        hole=0.60,
        marker_colors=[SECTOR_COLORS.get(t, C["muted"]) for t in nonzero.index],
        textinfo="label+percent",
        textfont=dict(size=11, color="#F1F5F9"),
        hovertemplate="%{label}: %{percent}<extra></extra>",
    ))
    fig_d.update_layout(**CHART, height=280, showlegend=False, title=f"{weight_mode}")
    st.plotly_chart(fig_d, width="stretch")

    wdf = pd.DataFrame({"Ticker": nonzero.index, "Weight": (nonzero.values*100).round(1)})
    wdf["Weight"] = wdf["Weight"].apply(lambda x: f"{x:.1f}%")
    st.dataframe(wdf, hide_index=True, width="stretch")

# ── Fan charts ─────────────────────────────────────────────────────────────────
section("Simulation Fan Chart",
        f"{n_sims:,} paths · shaded = 5th–95th and 25th–75th percentile bands")

days_ax  = np.arange(horizon_days + 1)
years_ax = days_ax / 252
METHOD_C = {"GBM": C["primary"], "Bootstrap": C["amber"]}

for mname, result in sim_results.items():
    color = METHOD_C[mname]
    pcts  = result.percentiles

    fig_fan = go.Figure()
    # Ghost paths
    sample_idx = np.random.default_rng(0).integers(0, n_sims, min(60, n_sims))
    for idx in sample_idx:
        fig_fan.add_trace(go.Scatter(
            x=years_ax, y=result.paths[idx],
            line=dict(color=color, width=0.5), opacity=0.07,
            showlegend=False, hoverinfo="skip",
        ))

    # Bands
    rgba = lambda a: f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},{a})"
    for lo, hi, alpha, name in [(5,95,0.12,"5–95th"),(25,75,0.22,"25–75th")]:
        fig_fan.add_trace(go.Scatter(
            x=np.concatenate([years_ax, years_ax[::-1]]),
            y=np.concatenate([pcts[hi], pcts[lo][::-1]]),
            fill="toself", fillcolor=rgba(alpha),
            line=dict(width=0), name=name, hoverinfo="skip",
        ))

    for pct, lw, dash, lbl in [(50,2.5,"solid","Median"),(5,1.2,"dot","5th"),(95,1.2,"dot","95th")]:
        fig_fan.add_trace(go.Scatter(
            x=years_ax, y=pcts[pct], name=lbl,
            line=dict(color=color, width=lw, dash=dash),
            hovertemplate=f"Year %{{x:.1f}}<br>{lbl}: %{{y:.3f}}<extra></extra>",
        ))

    fig_fan.add_hline(y=1, line_dash="dot", line_color="#334155", line_width=1,
                      annotation_text="Starting NAV", annotation_position="right",
                      annotation_font=dict(color=C["muted"], size=10))
    fig_fan.add_hline(y=target_mult, line_dash="dash", line_color=C["amber"],
                      line_width=1, annotation_text=f"Target {target_mult:.1f}×",
                      annotation_position="right",
                      annotation_font=dict(color=C["amber"], size=10))

    fig_fan.update_layout(**CHART, height=420,
                           title=f"Portfolio NAV Paths — {mname} ({horizon_yrs}-year horizon)")
    fig_fan.update_xaxes(title_text="Years", tickformat=".1f")
    fig_fan.update_yaxes(title_text="Portfolio NAV")
    st.plotly_chart(fig_fan, width="stretch")

# ── Terminal wealth + probability table ────────────────────────────────────────
section("Terminal Wealth & Outcome Probabilities")
c3, c4 = st.columns(2, gap="large")

with c3:
    fig_tw = go.Figure()
    for mname, result in sim_results.items():
        fig_tw.add_trace(go.Histogram(
            x=result.terminal_wealth, nbinsx=60,
            name=mname, opacity=0.75,
            marker_color=METHOD_C[mname],
        ))

    med   = float(np.median(primary.terminal_wealth))
    var_v = sim.simulated_var(primary, .95) + 1
    cv_v  = sim.simulated_cvar(primary, .95) + 1

    for val, lbl, col in [(1.0,"Break-even",C["muted"]),(var_v,"VaR 95%",C["amber"]),
                           (cv_v,"CVaR 95%",C["red"]),(med,"Median",C["green"]),
                           (target_mult,f"{target_mult:.1f}×",C["cyan"])]:
        fig_tw.add_vline(x=val, line_dash="dash", line_color=col, line_width=1.5,
                         annotation_text=lbl, annotation_position="top",
                         annotation_font=dict(color=col, size=10))

    fig_tw.update_layout(**CHART, height=340, barmode="overlay",
                          title=f"Terminal NAV Distribution ({horizon_yrs}Y)")
    fig_tw.update_xaxes(title_text="Portfolio NAV at horizon")
    st.plotly_chart(fig_tw, width="stretch")

with c4:
    stats_df = sim.horizon_statistics(primary)
    st.markdown('<div class="section">Probability Table by Horizon</div>', unsafe_allow_html=True)
    st.dataframe(stats_df.style.format({
        "Median NAV": "{:.3f}", "5th Pct": "{:.3f}", "95th Pct": "{:.3f}",
        "P(loss)": "{:.1f}%", "P(+20%)": "{:.1f}%", "P(double)": "{:.1f}%",
    }), hide_index=True, width="stretch")

    if len(sim_results) == 2:
        comp = {m: {
            "Median":       round(float(np.median(r.terminal_wealth)), 4),
            "5th Pct":      round(float(np.percentile(r.terminal_wealth, 5)), 4),
            "95th Pct":     round(float(np.percentile(r.terminal_wealth, 95)), 4),
            f"P(>{target_mult:.1f}×)": f"{sim.prob_reach_target(r,target_mult)*100:.1f}%",
            "VaR 95%":      f"{sim.simulated_var(r,.95)*100:.1f}%",
        } for m, r in sim_results.items()}
        st.markdown('<div class="section" style="margin-top:1rem;">GBM vs Bootstrap</div>',
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(comp), width="stretch")
