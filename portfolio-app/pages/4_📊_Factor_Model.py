import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st

from core.style import apply_style, page_header, section, kpi_row, CHART, C
import core.factor_model as fm

st.set_page_config(page_title="Factor Model", page_icon="📊", layout="wide")
apply_style()

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

@st.cache_data
def load():
    factors  = pd.read_csv(f"{DATA}/ff_factors.csv",        index_col=0, parse_dates=True)
    port_ret = pd.read_csv(f"{DATA}/portfolio_returns.csv", index_col=0, parse_dates=True)
    stocks   = pd.read_csv(f"{DATA}/stock_log_returns.csv", index_col=0, parse_dates=True)
    first    = (port_ret["long"] != 0).idxmax()
    port_ret = port_ret.loc[first:]
    return factors, port_ret, stocks

factors, port_returns, stock_returns = load()

FACTOR_COLORS = {
    "Mkt-RF": C["primary"], "SMB": C["amber"],   "HML": C["green"],
    "RMW":    C["purple"],  "CMA": C["cyan"],     "UMD": C["red"],
    "Alpha":  C["amber"],   "Residual (α-specific)": C["muted"],
}

with st.sidebar:
    st.markdown("### Settings")
    portfolio = st.radio("Portfolio", ["long","long_short"],
                         format_func=lambda x: "Long-Only" if x=="long" else "Long-Short")
    model     = st.selectbox("Primary model", ["FF5+M","FF5","FF3","CAPM"])
    window    = st.slider("Rolling window (days)", 63, 252, 126, 21)

    st.divider()
    st.markdown("""
| Factor | Captures |
|--------|---------|
| Mkt-RF | Market premium |
| SMB | Small-cap |
| HML | Value |
| RMW | Profitability |
| CMA | Investment |
| UMD | Momentum |
""")

@st.cache_data
def get_results(portfolio, model_key):
    series  = port_returns[portfolio]
    results = fm.run_all_models(series, factors, portfolio)
    return results

results = get_results(portfolio, model)
port_series   = port_returns[portfolio]
res_primary   = results[model]

page_header("Fama-French Factor Model",
            f"Return decomposition · {portfolio.replace('_',' ').title()} portfolio · 2018–2024")

kpi_row([
    {"label": f"Alpha CAPM (ann)",  "value": f"{results['CAPM'].alpha_ann*100:.3f}%",  "color": C["muted"]},
    {"label": f"Alpha FF3 (ann)",   "value": f"{results['FF3'].alpha_ann*100:.3f}%",   "color": C["cyan"]},
    {"label": f"Alpha FF5 (ann)",   "value": f"{results['FF5'].alpha_ann*100:.3f}%",   "color": C["amber"]},
    {"label": f"Alpha FF5+M (ann)", "value": f"{results['FF5+M'].alpha_ann*100:.3f}%", "color": C["green"],
     "delta": "★ six-factor alpha"},
    {"label": f"FF5+M R²",          "value": f"{results['FF5+M'].r2*100:.1f}%",        "color": C["primary"]},
    {"label": "Mkt-RF β",           "value": f"{results['FF5+M'].betas['Mkt-RF']:.3f}","color": C["primary"]},
])

# ── Alpha shrinkage + R² ───────────────────────────────────────────────────────
section("Alpha Shrinkage vs Model Complexity",
        "Each additional set of factors absorbs some of the 'alpha' — the residual is the true unexplained return.")
c1, c2 = st.columns(2, gap="large")

with c1:
    alpha_vals = {name: r.alpha_ann * 100 for name, r in results.items()}
    fig_a = go.Figure(go.Bar(
        x=list(alpha_vals.keys()), y=list(alpha_vals.values()),
        marker_color=[C["green"] if v > 0 else C["red"] for v in alpha_vals.values()],
        marker_line_color="rgba(0,0,0,0)",
        text=[f"{v:.3f}%" for v in alpha_vals.values()],
        textposition="outside", textfont=dict(color="#F1F5F9"),
    ))
    fig_a.add_hline(y=0, line_color="#334155", line_width=1)
    fig_a.update_layout(**CHART, height=300, title="Annualized Alpha by Model")
    fig_a.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig_a, width="stretch")

with c2:
    r2s    = {n: r.r2 * 100 for n, r in results.items()}
    adjr2s = {n: r.adj_r2 * 100 for n, r in results.items()}
    fig_r = go.Figure()
    fig_r.add_trace(go.Bar(x=list(r2s.keys()), y=list(r2s.values()),
        name="R²", marker_color=C["primary"], opacity=0.9,
        text=[f"{v:.1f}%" for v in r2s.values()],
        textposition="outside", textfont=dict(color="#F1F5F9")))
    fig_r.add_trace(go.Bar(x=list(adjr2s.keys()), y=list(adjr2s.values()),
        name="Adj. R²", marker_color=C["cyan"], opacity=0.7))
    fig_r.update_layout(**CHART, height=300, barmode="overlay", title="R² vs Adjusted R²")
    fig_r.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig_r, width="stretch")

# ── Factor loadings ────────────────────────────────────────────────────────────
section(f"Factor Loadings — {model}", "Error bars = 1 standard error (HC3 robust SEs)")

factor_cols = fm.MODELS[model]
df_reg = fm.align(port_series, factors)
X_reg  = sm.add_constant(df_reg[factor_cols])
ols    = sm.OLS(df_reg["excess_ret"], X_reg).fit(cov_type="HC3")

betas  = [res_primary.betas[f] for f in factor_cols]
ses    = [ols.bse[f] for f in factor_cols]
ts     = [res_primary.tstats[f] for f in factor_cols]
labels = [fm.FACTOR_LABELS[f] for f in factor_cols]
colors = [FACTOR_COLORS.get(f, C["muted"]) for f in factor_cols]

fig_b = go.Figure(go.Bar(
    x=labels, y=betas,
    error_y=dict(type="data", array=ses, visible=True, color="#F1F5F9", thickness=1.5),
    marker_color=colors,
    marker_line_color="rgba(0,0,0,0)",
    text=[f"β={b:.3f}<br>t={t:.2f}" for b, t in zip(betas, ts)],
    textposition="outside", textfont=dict(size=10, color="#F1F5F9"),
))
fig_b.add_hline(y=0, line_color="#334155", line_width=1)
fig_b.update_layout(**CHART, height=340, title=f"Factor Betas — {model}")
st.plotly_chart(fig_b, width="stretch")

# ── Rolling alpha + betas ──────────────────────────────────────────────────────
section("Rolling Factor Exposures", f"{window}-day window — shows how exposures shift across regimes")
rolling = fm.rolling_regression(port_series, factors, factor_cols, window)

c3, c4 = st.columns(2, gap="large")
with c3:
    fig_ra = go.Figure()
    fig_ra.add_trace(go.Scatter(
        x=rolling.index, y=rolling["alpha_ann"]*100,
        name="Rolling Alpha", line=dict(color=C["amber"], width=2),
        fill="tozeroy", fillcolor="rgba(251,191,36,0.08)",
    ))
    fig_ra.add_hline(y=res_primary.alpha_ann*100, line_dash="dot", line_color=C["muted"],
        annotation_text=f"Full-period: {res_primary.alpha_ann*100:.3f}%",
        annotation_position="right", annotation_font=dict(color=C["muted"], size=10))
    fig_ra.add_hline(y=0, line_color="#334155", line_width=1)
    for x0, x1, lbl in [("2020-02-19","2020-06-01","COVID"),("2022-01-01","2022-12-31","Rate shock")]:
        fig_ra.add_vrect(x0=x0, x1=x1, fillcolor=C["red"], opacity=0.06, line_width=0,
                         annotation_text=lbl, annotation_position="top left",
                         annotation_font=dict(color=C["red"], size=10))
    fig_ra.update_layout(**CHART, height=300, title=f"Rolling {window}-Day Alpha (ann.)")
    fig_ra.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig_ra, width="stretch")

with c4:
    fig_rb = go.Figure()
    for f in factor_cols:
        fig_rb.add_trace(go.Scatter(
            x=rolling.index, y=rolling[f],
            name=fm.FACTOR_LABELS[f],
            line=dict(color=FACTOR_COLORS.get(f, C["muted"]), width=1.8),
        ))
    fig_rb.add_hline(y=0, line_color="#334155", line_width=1)
    fig_rb.update_layout(**CHART, height=300, title=f"Rolling {window}-Day Factor Betas")
    st.plotly_chart(fig_rb, width="stretch")

# ── Return decomposition ───────────────────────────────────────────────────────
section("Cumulative Return Decomposition", "Stacked area — how much each factor contributed.")
decomp    = fm.return_decomposition(res_primary, factors)
cumdecomp = (1 + decomp).cumprod() - 1

comp_order = ["Alpha"] + [fm.FACTOR_LABELS[f] for f in factor_cols] + ["Residual (α-specific)"]
comp_order = [c for c in comp_order if c in cumdecomp.columns]

fig_dc = go.Figure()
for comp in comp_order:
    fig_dc.add_trace(go.Scatter(
        x=cumdecomp.index, y=cumdecomp[comp]*100,
        name=comp, stackgroup="one",
        line=dict(width=0.5, color=FACTOR_COLORS.get(comp, C["muted"])),
        fillcolor=FACTOR_COLORS.get(comp, C["muted"]),
        hovertemplate=f"<b>{comp}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.2f}}%<extra></extra>",
    ))
fig_dc.update_layout(**CHART, height=340, title="Cumulative Return Attribution (stacked)")
fig_dc.update_yaxes(ticksuffix="%")
st.plotly_chart(fig_dc, width="stretch")
