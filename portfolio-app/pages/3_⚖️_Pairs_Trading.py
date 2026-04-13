import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core.style import apply_style, page_header, section, kpi_row, CHART, C
import core.spread_model as sm_mod

st.set_page_config(page_title="Pairs Trading", page_icon="⚖️", layout="wide")
apply_style()

DATA     = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
BASE     = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PRICES_PATH = os.path.join(BASE, "momentum-strategy", "data", "sp500_prices.csv")

@st.cache_data
def load():
    top_pairs  = pd.read_csv(f"{DATA}/top_pairs.csv")
    pair_ret   = pd.read_csv(f"{DATA}/pair_returns.csv",  index_col=0, parse_dates=True)
    port_nav   = pd.read_csv(f"{DATA}/pairs_nav.csv",     index_col=0, parse_dates=True)
    trade_log  = pd.read_csv(f"{DATA}/trade_log.csv")
    prices     = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    return top_pairs, pair_ret, port_nav, trade_log, prices

top_pairs, pair_ret, port_nav, trade_log, prices = load()
pair_labels = [f"{r.ticker_A} / {r.ticker_B}" for _, r in top_pairs.iterrows()
               if f"{r.ticker_A}/{r.ticker_B}" in pair_ret.columns]

with st.sidebar:
    st.markdown("### Pair Inspector")
    selected_label = st.selectbox("Select pair", pair_labels)
    a, b = [t.strip() for t in selected_label.split("/")]
    row  = top_pairs[(top_pairs.ticker_A == a) & (top_pairs.ticker_B == b)].iloc[0]
    hr   = row["hedge_ratio"]

    st.divider()
    entry_z = st.slider("Entry |z|", 1.0, 3.0, float(sm_mod.ENTRY_Z), 0.1)
    exit_z  = st.slider("Exit |z|",  0.1, 1.5, float(sm_mod.EXIT_Z),  0.1)
    stop_z  = st.slider("Stop |z|",  2.0, 5.0, float(sm_mod.STOP_Z),  0.1)

    st.divider()
    st.markdown(f"""
| | |
|--|--|
| **β (hedge ratio)** | {hr:.4f} |
| **Half-life** | {row['half_life']:.1f} days |
| **Coint p-val** | {row['coint_pval']:.5f} |
| **Correlation** | {row['correlation']:.4f} |
""")

@st.cache_data
def get_spread(a, b, hr, entry_z, exit_z, stop_z):
    ps = sm_mod.build_pair_spread(a, b, prices, hr)
    sg = sm_mod.generate_signals(ps.z_score, entry_z, exit_z, stop_z)
    return ps, sg

pair_spread, signals = get_spread(a, b, hr, entry_z, exit_z, stop_z)
TRADE_START, TRADE_END = "2022-01-01", "2024-12-31"
z_t   = pair_spread.z_score.loc[TRADE_START:TRADE_END]
sig_t = signals.loc[TRADE_START:TRADE_END]
sp_t  = pair_spread.spread.loc[TRADE_START:TRADE_END]

port_ret  = port_nav["nav"].pct_change().dropna()
cagr      = port_nav["nav"].iloc[-1] ** (252 / len(port_ret)) - 1
sharpe    = (port_ret.mean() - 0.04/252) / port_ret.std() * np.sqrt(252)
mdd       = ((port_nav["nav"] - port_nav["nav"].cummax()) / port_nav["nav"].cummax()).min()
win_rate  = (trade_log["pnl (%)"] > 0).mean() * 100 if len(trade_log) else 0
avg_dur   = trade_log["duration"].mean() if len(trade_log) else 0

page_header("Pairs Trading / Stat Arb",
            "Engle-Granger cointegration · Formation 2018–2021 · Out-of-sample 2022–2024")

kpi_row([
    {"label": "Total Trades",        "value": str(len(trade_log)),      "color": C["primary"]},
    {"label": "Win Rate",            "value": f"{win_rate:.1f}%",        "color": C["green"]},
    {"label": "Portfolio CAGR",      "value": f"{cagr*100:.2f}%",        "color": C["amber"]},
    {"label": "Sharpe",              "value": f"{sharpe:.3f}",           "color": C["primary"]},
    {"label": "Max Drawdown",        "value": f"{mdd*100:.2f}%",         "color": C["red"]},
    {"label": "Avg Hold (days)",     "value": f"{avg_dur:.0f}",          "color": C["cyan"]},
])

# ── 3-panel spread chart ───────────────────────────────────────────────────────
section(f"Spread Analysis — {a} / {b}")
fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.30, 0.22, 0.48], vertical_spacing=0.03,
                    subplot_titles=[
                        f"Normalised Price — {a} vs {b}",
                        "Log-Price Spread",
                        "Z-Score with Trade Signals",
                    ])

pa = (prices[a].loc[TRADE_START:TRADE_END] / prices[a].loc[TRADE_START:TRADE_END].iloc[0])
pb = (prices[b].loc[TRADE_START:TRADE_END] / prices[b].loc[TRADE_START:TRADE_END].iloc[0])
fig.add_trace(go.Scatter(x=pa.index, y=pa, name=a,
    line=dict(color=C["primary"], width=1.8)), row=1, col=1)
fig.add_trace(go.Scatter(x=pb.index, y=pb, name=b,
    line=dict(color=C["amber"], width=1.8)), row=1, col=1)

fig.add_trace(go.Scatter(x=sp_t.index, y=sp_t, name="Spread",
    fill="tozeroy", fillcolor="rgba(192,132,252,0.10)",
    line=dict(color=C["purple"], width=1.3)), row=2, col=1)

fig.add_trace(go.Scatter(x=z_t.index, y=z_t, name="Z-score",
    line=dict(color="#E2E8F0", width=1.2)), row=3, col=1)

for val, color, dash in [
    (entry_z, C["green"], "dash"), (-entry_z, C["green"], "dash"),
    (exit_z,  C["amber"], "dot"),  (-exit_z,  C["amber"], "dot"),
    (stop_z,  C["red"],   "longdash"), (-stop_z, C["red"], "longdash"),
]:
    fig.add_hline(y=val, line_dash=dash, line_color=color,
                  line_width=1, opacity=0.7, row=3, col=1)

# Entry/exit markers
long_e  = z_t[(sig_t.diff() > 0) & (sig_t == 1)]
short_e = z_t[(sig_t.diff() < 0) & (sig_t == -1)]
exits   = z_t[sig_t.diff().abs() > 0 & (sig_t == 0)]
for dates, color, sym, lbl in [
    (long_e.index,  C["green"], "triangle-up",   "Long entry"),
    (short_e.index, C["red"],   "triangle-down",  "Short entry"),
    (exits.index,   C["amber"], "circle",          "Exit"),
]:
    if len(dates):
        fig.add_trace(go.Scatter(x=dates, y=z_t.reindex(dates), mode="markers",
            name=lbl, marker=dict(color=color, size=9, symbol=sym,
            line=dict(color="#0F172A", width=1))), row=3, col=1)

fig.update_layout(**CHART, height=660, showlegend=True)
fig.update_yaxes(gridcolor="rgba(51,65,85,0.4)")
st.plotly_chart(fig, width="stretch")

# ── NAV + P&L ─────────────────────────────────────────────────────────────────
c1, c2 = st.columns([3, 2], gap="large")

with c1:
    section("Portfolio NAV", "Bold = equal-weight portfolio. Muted = individual pairs.")
    fig_nav = go.Figure()
    for col in pair_ret.columns:
        pn = (1 + pair_ret[col].loc[TRADE_START:TRADE_END]).cumprod()
        fig_nav.add_trace(go.Scatter(x=pn.index, y=pn, name=col,
            line=dict(width=0.8), opacity=0.25, showlegend=False))
    fig_nav.add_trace(go.Scatter(x=port_nav.index, y=port_nav["nav"],
        name="Portfolio", line=dict(color=C["green"], width=2.5)))
    fig_nav.add_hline(y=1, line_dash="dot", line_color="#334155", line_width=1)
    fig_nav.update_layout(**CHART, height=320, title="Cumulative NAV — All Pairs")
    st.plotly_chart(fig_nav, width="stretch")

with c2:
    section("P&L by Pair")
    pair_pnl = (trade_log.groupby("pair")["pnl (%)"].sum()
                          .sort_values().reset_index())
    fig_pnl = go.Figure(go.Bar(
        x=pair_pnl["pnl (%)"], y=pair_pnl["pair"], orientation="h",
        marker_color=[C["green"] if v > 0 else C["red"] for v in pair_pnl["pnl (%)"]],
        text=[f"{v:+.1f}%" for v in pair_pnl["pnl (%)"]],
        textposition="outside", textfont=dict(color="#F1F5F9"),
    ))
    fig_pnl.update_layout(**CHART, height=320, title="Total P&L by Pair (%)")
    fig_pnl.update_xaxes(ticksuffix="%")
    st.plotly_chart(fig_pnl, width="stretch")

# ── Distributions ──────────────────────────────────────────────────────────────
section("Trade Analytics")
c3, c4 = st.columns(2, gap="large")
with c3:
    fig_h = go.Figure(go.Histogram(
        x=trade_log["pnl (%)"], nbinsx=40,
        marker_color=C["primary"], marker_line_color="rgba(129,140,248,0.4)",
        marker_line_width=0.5,
    ))
    fig_h.add_vline(x=0, line_color="#334155", line_width=1)
    fig_h.add_vline(x=trade_log["pnl (%)"].mean(), line_dash="dash",
                    line_color=C["amber"], line_width=1.5,
                    annotation_text=f"Mean: {trade_log['pnl (%)'].mean():.3f}%",
                    annotation_font=dict(color=C["amber"]))
    fig_h.update_layout(**CHART, height=280, title="Trade P&L Distribution (%)")
    st.plotly_chart(fig_h, width="stretch")

with c4:
    fig_d = go.Figure(go.Histogram(
        x=trade_log["duration"], nbinsx=30, marker_color=C["purple"]))
    fig_d.add_vline(x=trade_log["duration"].mean(), line_dash="dash",
                    line_color=C["amber"], line_width=1.5,
                    annotation_text=f"Mean: {trade_log['duration'].mean():.0f}d",
                    annotation_font=dict(color=C["amber"]))
    fig_d.update_layout(**CHART, height=280, title="Holding Period Distribution (days)")
    st.plotly_chart(fig_d, width="stretch")
