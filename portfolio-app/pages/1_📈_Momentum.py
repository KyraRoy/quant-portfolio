import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.style import apply_style, page_header, section, kpi_row, CHART, C

st.set_page_config(page_title="Momentum Strategy", page_icon="📈", layout="wide")
apply_style()
page_header("Momentum Strategy",
            "Jegadeesh-Titman (1993) · S&P 500 · 2018–2024 · 10 bps transaction costs")

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

@st.cache_data
def load():
    nav  = pd.read_csv(f"{DATA}/backtest_nav.csv",     index_col=0, parse_dates=True)
    ret  = pd.read_csv(f"{DATA}/backtest_returns.csv", index_col=0, parse_dates=True)
    perf = pd.read_csv(f"{DATA}/performance_summary.csv", index_col=0)
    first = (ret["long"] != 0).idxmax()
    nav  = nav.loc[first:] / nav.loc[first:].iloc[0]
    ret  = ret.loc[first:]
    return nav, ret, perf

@st.cache_data
def load_spy():
    # SPY is included in stock_prices.csv — no live download needed
    prices = pd.read_csv(f"{DATA}/stock_prices.csv", index_col=0, parse_dates=True)
    p = prices["SPY"].dropna()
    return p

nav, ret, perf = load()
_spy_raw = load_spy()
# Normalize SPY to 1.0 at strategy start, aligned to NAV index
spy = (_spy_raw / _spy_raw.reindex(nav.index).dropna().iloc[0]).reindex(nav.index).ffill().rename("SPY")

long_p  = perf.loc["long"]
ls_p    = perf.loc["long_short"]

# ── KPIs ───────────────────────────────────────────────────────────────────────
kpi_row([
    {"label": "CAGR (Long-Only)",    "value": f"{long_p['CAGR (%)']:.2f}%",  "color": C["green"]},
    {"label": "Sharpe (Long-Only)",  "value": f"{long_p['Sharpe']:.3f}",      "color": C["primary"]},
    {"label": "Max Drawdown",        "value": f"{long_p['Max DD (%)']:.1f}%", "color": C["red"]},
    {"label": "Calmar Ratio",        "value": f"{long_p['Calmar']:.3f}",      "color": C["amber"]},
    {"label": "Ann. Volatility",     "value": f"{long_p['Vol (%)']:.2f}%",    "color": C["cyan"]},
    {"label": "Sharpe (L/S)",        "value": f"{ls_p['Sharpe']:.3f}",        "color": C["muted"]},
])

# ── NAV Chart ─────────────────────────────────────────────────────────────────
section("Cumulative NAV vs SPY",
        "Signal at month-end t → trade at t+1. Equal-weight decile portfolios.")

fig = go.Figure()
traces = [
    ("long",       "Long-Only",  C["green"],   "solid",   2.5),
    ("long_short", "Long-Short", C["primary"], "dash",    2.0),
    ("short",      "Short-Only", C["red"],     "dot",     1.5),
]
for col, label, color, dash, width in traces:
    fig.add_trace(go.Scatter(
        x=nav.index, y=nav[col], name=label,
        line=dict(color=color, width=width, dash=dash),
        hovertemplate=f"<b>{label}</b><br>%{{x|%Y-%m-%d}}<br>NAV: %{{y:.3f}}<extra></extra>",
    ))

fig.add_trace(go.Scatter(
    x=spy.index, y=spy, name="SPY",
    line=dict(color=C["spy"], width=1.5, dash="longdash"),
    hovertemplate="<b>SPY</b><br>%{x|%Y-%m-%d}<br>NAV: %{y:.3f}<extra></extra>",
))
fig.add_hline(y=1, line_dash="dot", line_color="#1E293B", line_width=1)

for date, label, xanchor in [
    ("2020-03-23", "COVID bottom", "left"),
    ("2022-01-03", "Rate hikes begin", "left"),
]:
    fig.add_vline(x=date, line_dash="dash", line_color=C["amber"], line_width=1, opacity=0.5)
    fig.add_annotation(x=date, y=nav["long"].max() * 0.92, text=label,
                       showarrow=False, font=dict(color=C["amber"], size=10),
                       xanchor=xanchor, xshift=6)

fig.update_layout(**CHART, height=420, title="Cumulative NAV (rebased to 1.0 at strategy start)")
st.plotly_chart(fig, width="stretch")

# ── Drawdown + Monthly Heatmap ─────────────────────────────────────────────────
c1, c2 = st.columns(2, gap="large")

with c1:
    section("Drawdown")
    dd_long = ((nav["long"] - nav["long"].cummax()) / nav["long"].cummax()) * 100
    dd_ls   = ((nav["long_short"] - nav["long_short"].cummax()) / nav["long_short"].cummax()) * 100

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dd_long.index, y=dd_long, name="Long-Only",
        fill="tozeroy", fillcolor="rgba(52,211,153,0.12)",
        line=dict(color=C["green"], width=1.5),
    ))
    fig_dd.add_trace(go.Scatter(
        x=dd_ls.index, y=dd_ls, name="Long-Short",
        fill="tozeroy", fillcolor="rgba(129,140,248,0.10)",
        line=dict(color=C["primary"], width=1.5, dash="dash"),
    ))
    fig_dd.update_layout(**CHART, height=320, title="Drawdown (%)")
    fig_dd.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig_dd, width="stretch")

with c2:
    section("Monthly Returns Heatmap", "Long-Only portfolio")
    monthly = (1 + ret["long"]).resample("ME").prod() - 1
    monthly = monthly * 100
    pivot   = monthly.groupby([monthly.index.year, monthly.index.month]).first().unstack()
    pivot.index.name = "Year"
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.columns = month_labels[:len(pivot.columns)]

    annual = pivot.sum(axis=1)
    pivot_d = pd.concat([pivot, annual.rename("Full Yr")], axis=1).dropna(how="all")

    fig_h = go.Figure(go.Heatmap(
        z=pivot_d.values,
        x=pivot_d.columns.tolist(),
        y=pivot_d.index.astype(str).tolist(),
        colorscale=[[0,"#7F1D1D"],[0.35,"#F87171"],[0.5,"#1E293B"],
                    [0.65,"#34D399"],[1,"#065F46"]],
        zmid=0,
        text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row]
              for row in pivot_d.values],
        texttemplate="%{text}", textfont=dict(size=9),
        showscale=False,
        hovertemplate="%{y} %{x}<br>%{z:.2f}%<extra></extra>",
    ))
    fig_h.update_layout(**CHART, height=320, title="Monthly P&L — Long-Only")
    fig_h.update_xaxes(gridcolor="rgba(0,0,0,0)")
    fig_h.update_yaxes(gridcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_h, width="stretch")

# ── Rolling Sharpe ─────────────────────────────────────────────────────────────
section("Rolling 252-Day Sharpe Ratio",
        "Orange shading marks the 2020 COVID crash and 2022 rate-shock windows.")

def rolling_sharpe(r, window=252, rf=0.04):
    excess = r - rf/252
    return (excess.rolling(window).mean() / excess.rolling(window).std() * np.sqrt(252)).dropna()

fig_rs = go.Figure()
for col, label, color, dash in [
    ("long", "Long-Only", C["green"], "solid"),
    ("long_short", "Long-Short", C["primary"], "dash"),
]:
    rs = rolling_sharpe(ret[col])
    fig_rs.add_trace(go.Scatter(
        x=rs.index, y=rs, name=label,
        line=dict(color=color, width=2, dash=dash),
        hovertemplate=f"<b>{label}</b><br>%{{x|%Y-%m-%d}}<br>Sharpe: %{{y:.2f}}<extra></extra>",
    ))

fig_rs.add_hline(y=0, line_color="#334155", line_width=1)
fig_rs.add_hline(y=1, line_dash="dot", line_color=C["green"], line_width=0.8, opacity=0.4)
for x0, x1, label in [("2020-02-19","2020-05-01","COVID crash"),
                        ("2022-01-01","2022-12-31","Rate shock")]:
    fig_rs.add_vrect(x0=x0, x1=x1, fillcolor=C["amber"], opacity=0.07, line_width=0,
                     annotation_text=label, annotation_position="top left",
                     annotation_font=dict(color=C["amber"], size=10))

fig_rs.update_layout(**CHART, height=300, title="Rolling Sharpe (252-day, rf=4%)")
st.plotly_chart(fig_rs, width="stretch")

# ── Performance Table ──────────────────────────────────────────────────────────
section("Full Performance Summary")
disp = perf.copy()
disp.index = ["Long-Only", "Short-Only", "Long-Short"]

# Render as a Plotly table to avoid any matplotlib dependency
col_labels = ["Strategy"] + disp.columns.tolist()
cell_vals   = [disp.index.tolist()] + [disp[c].round(2).tolist() for c in disp.columns]

# Color CAGR and Sharpe cells green/red by sign
def cell_colors(col, vals):
    if col in ("CAGR (%)", "Sharpe", "Calmar"):
        return [C["green"] if v > 0 else C["red"] for v in vals]
    if col == "Max DD (%)":
        return [C["red"] if v < 0 else C["muted"] for v in vals]
    return [C["muted"]] * len(vals)

fill_colors = [["#1E293B"] * len(disp.index)]
for col in disp.columns:
    fill_colors.append(cell_colors(col, disp[col].tolist()))

fig_tbl = go.Figure(go.Table(
    header=dict(
        values=[f"<b>{c}</b>" for c in col_labels],
        fill_color="#0F172A", font=dict(color="#F1F5F9", size=12),
        align="left", line_color="#334155", height=36,
    ),
    cells=dict(
        values=cell_vals,
        fill_color=fill_colors,
        font=dict(color="#F1F5F9", size=12),
        align=["left"] + ["right"] * len(disp.columns),
        line_color="#334155", height=32,
        format=[None] + [".2f"] * len(disp.columns),
    ),
))
fig_tbl.update_layout(**CHART, height=160)
fig_tbl.update_layout(margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(fig_tbl, width="stretch")

with st.sidebar:
    st.markdown("### Strategy Parameters")
    st.markdown("""
| | |
|--|--|
| Universe | S&P 500 (484) |
| Signal | JT 12-1 month |
| Rebalance | Monthly |
| Long | Top decile |
| Short | Bottom decile |
| Costs | 10 bps/trade |
| Period | 2018–2024 |
""")
