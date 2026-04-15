import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st

from core.style import apply_style, page_header, section, kpi_row, CHART, C
import core.risk_metrics as rm
import core.live_data as ld

st.set_page_config(page_title="Risk Dashboard", page_icon="🛡️", layout="wide")
apply_style()

# Live prices + log returns for the 20-stock portfolio (updates daily via yfinance;
# falls back to the static CSV if the download fails).
prices_all  = ld.get_portfolio_prices()
returns_all = ld.get_portfolio_log_returns()
all_tickers = [t for t in returns_all.columns if t != "SPY"]

SECTOR = {
    "AAPL":"Technology","MSFT":"Technology","NVDA":"Technology",
    "GOOGL":"Communication","META":"Communication",
    "AMZN":"Cons. Discr.","TSLA":"Cons. Discr.",
    "PG":"Cons. Staples","KO":"Cons. Staples",
    "JPM":"Financials","GS":"Financials",
    "JNJ":"Healthcare","UNH":"Healthcare",
    "CAT":"Industrials","HON":"Industrials",
    "XOM":"Energy","CVX":"Energy",
    "NEE":"Utilities","AMT":"Real Estate",
}

with st.sidebar:
    st.markdown("### Portfolio Settings")
    selected = st.multiselect("Holdings", all_tickers, default=all_tickers)
    confidence = st.select_slider("VaR confidence", [0.90, 0.95, 0.99], value=0.95,
                                  format_func=lambda x: f"{int(x*100)}%")
    date_range = st.date_input("Date range",
        value=(returns_all.index[0].date(), returns_all.index[-1].date()),
        min_value=returns_all.index[0].date(), max_value=returns_all.index[-1].date())

if not selected:
    st.warning("Select at least one ticker.")
    st.stop()

start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
returns = returns_all.loc[start:end, selected + (["SPY"] if "SPY" in returns_all.columns else [])].dropna(how="all")
spy_ret = returns["SPY"] if "SPY" in returns.columns else None
port    = returns[selected].mean(axis=1)

var_h   = rm.var_historical(port, confidence)
cvar_v  = rm.cvar(port, confidence)
sh      = rm.sharpe_ratio(port)
so      = rm.sortino_ratio(port)
mdd     = rm.max_drawdown(port)
beta_v  = rm.beta(port, spy_ret) if spy_ret is not None else float("nan")

end_date = returns_all.index[-1].strftime("%Y-%m-%d")
page_header("Risk Dashboard",
            f"Equal-weight portfolio · {len(selected)} holdings · "
            f"{int(confidence*100)}% VaR confidence · 2018–{end_date}")

kpi_row([
    {"label": f"VaR {int(confidence*100)}% (daily)", "value": f"{var_h*100:.2f}%", "color": C["amber"]},
    {"label": f"CVaR {int(confidence*100)}% (daily)","value": f"{cvar_v*100:.2f}%","color": C["red"]},
    {"label": "Sharpe Ratio",   "value": f"{sh:.3f}",              "color": C["green"]},
    {"label": "Sortino Ratio",  "value": f"{so:.3f}",              "color": C["primary"]},
    {"label": "Max Drawdown",   "value": f"{mdd*100:.2f}%",        "color": C["red"]},
    {"label": "Beta to SPY",    "value": f"{beta_v:.3f}",          "color": C["cyan"]},
])

# ── Distribution + VaR comparison ─────────────────────────────────────────────
section("Return Distribution & VaR Methods")
c1, c2 = st.columns([3, 2], gap="large")

with c1:
    hist_data = [port.dropna().values.tolist()]
    fig_d = ff.create_distplot(hist_data, ["Portfolio"], bin_size=0.002, show_rug=False,
                                colors=[C["primary"]])
    fig_d.update_traces(selector=dict(type="histogram"),
                        marker_color="rgba(129,140,248,0.2)",
                        marker_line_color="rgba(129,140,248,0.5)")

    for val, label, color in [
        (var_h,                               f"VaR Hist {int(confidence*100)}%", C["amber"]),
        (rm.var_parametric(port, confidence), f"VaR Param",                      C["red"]),
        (cvar_v,                              f"CVaR",                            "#EF4444"),
    ]:
        fig_d.add_vline(x=val, line_dash="dash", line_color=color, line_width=1.8,
                        annotation_text=label, annotation_position="top right",
                        annotation_font=dict(color=color, size=10))

    fig_d.update_layout(**CHART, height=360, title="Daily Return Distribution")
    fig_d.update_xaxes(tickformat=".1%", title_text="Daily Return")
    st.plotly_chart(fig_d, width="stretch")

with c2:
    methods = {
        "Historical":  rm.var_historical(port, confidence) * 100,
        "Parametric":  rm.var_parametric(port, confidence) * 100,
        "Monte Carlo": rm.var_monte_carlo(port, confidence) * 100,
        "CVaR (ES)":   cvar_v * 100,
    }
    fig_bar = go.Figure(go.Bar(
        x=list(methods.values()), y=list(methods.keys()), orientation="h",
        marker_color=[C["amber"], C["red"], C["purple"], "#EF4444"],
        text=[f"{v:.3f}%" for v in methods.values()],
        textposition="outside", textfont=dict(color="#F1F5F9"),
    ))
    fig_bar.update_layout(**CHART, height=220, title="Daily VaR by Method")
    fig_bar.update_xaxes(ticksuffix="%")
    st.plotly_chart(fig_bar, width="stretch")

    st.markdown("""
<div style="background:#1E293B;border-radius:10px;padding:14px;font-size:0.82rem;color:#94A3B8;line-height:1.6;">
<b style="color:#F1F5F9">Historical</b> — reads directly from data, no assumptions<br>
<b style="color:#F1F5F9">Parametric</b> — assumes normal returns, underestimates tails<br>
<b style="color:#F1F5F9">Monte Carlo</b> — 100k simulated paths from fitted distribution<br>
<b style="color:#F1F5F9">CVaR</b> — avg loss <i>beyond</i> VaR — always worse
</div>""", unsafe_allow_html=True)

# ── Rolling vol ────────────────────────────────────────────────────────────────
section("Rolling Volatility", "Annualized. Orange bands = COVID crash and 2022 rate shock.")
vol21 = rm.rolling_volatility(port, 21)
vol63 = rm.rolling_volatility(port, 63)
avg   = port.std() * np.sqrt(252) * 100

fig_v = go.Figure()
fig_v.add_trace(go.Scatter(x=vol21.index, y=vol21*100, name="21-day",
    line=dict(color=C["primary"], width=1.5)))
fig_v.add_trace(go.Scatter(x=vol63.index, y=vol63*100, name="63-day",
    line=dict(color=C["amber"], width=2.2)))
fig_v.add_hline(y=avg, line_dash="dot", line_color=C["muted"], line_width=1,
    annotation_text=f"Avg: {avg:.1f}%", annotation_position="right",
    annotation_font=dict(color=C["muted"], size=10))
for x0, x1, lbl in [("2020-02-19","2020-06-01","COVID"),("2022-01-01","2022-12-31","Rate shock")]:
    fig_v.add_vrect(x0=x0, x1=x1, fillcolor=C["amber"], opacity=0.08, line_width=0,
                    annotation_text=lbl, annotation_position="top left",
                    annotation_font=dict(color=C["amber"], size=10))
fig_v.update_layout(**CHART, height=300, title="Annualized Rolling Volatility")
fig_v.update_yaxes(ticksuffix="%")
st.plotly_chart(fig_v, width="stretch")

# ── Drawdown + Correlation ─────────────────────────────────────────────────────
c3, c4 = st.columns([1, 1], gap="large")

with c3:
    section("Drawdown Curve")
    dd_s = rm.drawdown_series(port) * 100
    fig_dd = go.Figure(go.Scatter(x=dd_s.index, y=dd_s, fill="tozeroy",
        fillcolor="rgba(248,113,113,0.15)", line=dict(color=C["red"], width=1.5)))
    fig_dd.add_hline(y=mdd*100, line_dash="dot", line_color="#EF4444", line_width=1,
        annotation_text=f"Max: {mdd*100:.1f}%", annotation_position="right",
        annotation_font=dict(color="#EF4444"))
    fig_dd.update_layout(**CHART, height=300, title="Portfolio Drawdown")
    fig_dd.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig_dd, width="stretch")

with c4:
    section("Correlation Heatmap", "Sorted by sector.")
    corr = returns[selected].corr()
    ordered = sorted(selected, key=lambda t: (SECTOR.get(t, "Z"), t))
    corr = corr.loc[ordered, ordered]
    fig_c = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale=[[0,"#1D4ED8"],[0.5,"#1E293B"],[1,"#DC2626"]],
        zmin=-1, zmax=1, zmid=0,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}", textfont=dict(size=8),
        hovertemplate="%{y} × %{x}<br>ρ = %{z:.3f}<extra></extra>",
    ))
    fig_c.update_layout(**CHART, height=300, title="Return Correlations (sector-sorted)")
    fig_c.update_xaxes(tickangle=-45, gridcolor="rgba(0,0,0,0)")
    fig_c.update_yaxes(gridcolor="rgba(0,0,0,0)", autorange="reversed")
    st.plotly_chart(fig_c, width="stretch")
