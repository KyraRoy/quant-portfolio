"""
dashboard.py
------------
Interactive Streamlit dashboard for the Fama-French factor model.

Sections:
  1. Sidebar  — portfolio selector, model selector, rolling window
  2. Model comparison table — alpha, betas, R² across CAPM/FF3/FF5/FF5+M
  3. Alpha decomposition — how much alpha survives each model
  4. Factor loadings bar chart
  5. Rolling alpha (6-month window)
  6. Rolling factor betas
  7. Cumulative return decomposition (stacked area)
  8. Per-ticker factor exposure table

Run with:
    streamlit run dashboard.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

import factor_model as fm

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fama-French Factor Model",
    page_icon="📊",
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

MODEL_COLORS = {
    "CAPM":  "#9E9E9E",
    "FF3":   "#2196F3",
    "FF5":   "#FF9800",
    "FF5+M": "#4CAF50",
}

FACTOR_COLORS = {
    "Mkt-RF": "#2196F3",
    "SMB":    "#FF9800",
    "HML":    "#4CAF50",
    "RMW":    "#9C27B0",
    "CMA":    "#00BCD4",
    "UMD":    "#F44336",
    "Alpha":  "#FFC107",
    "Residual (α-specific)": "#607D8B",
}

# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    factors       = pd.read_csv("data/ff_factors.csv",        index_col=0, parse_dates=True)
    port_returns  = pd.read_csv("data/portfolio_returns.csv", index_col=0, parse_dates=True)
    stock_returns = pd.read_csv("data/stock_returns.csv",     index_col=0, parse_dates=True)

    # Trim leading zeros from portfolio returns
    first_active = (port_returns["long"] != 0).idxmax()
    port_returns = port_returns.loc[first_active:]

    return factors, port_returns, stock_returns


@st.cache_data
def compute_all_results(portfolio_name: str):
    factors, port_returns, stock_returns = load_data()
    returns = port_returns[portfolio_name]
    results = fm.run_all_models(returns, factors, portfolio_name)
    return results, factors, port_returns, stock_returns


factors_raw, port_raw, stocks_raw = load_data()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Settings")

    portfolio_choice = st.radio(
        "Portfolio to explain",
        ["long", "long_short"],
        format_func=lambda x: "Momentum Long-Only" if x == "long" else "Momentum Long-Short",
    )

    primary_model = st.selectbox(
        "Primary model (for rolling/decomposition)",
        ["FF5+M", "FF5", "FF3", "CAPM"],
        index=0,
    )

    roll_window = st.slider(
        "Rolling window (trading days)",
        min_value=63, max_value=252, value=126, step=21,
        help="126 ≈ 6 months, 252 ≈ 1 year",
    )

    st.divider()
    st.markdown("## Factor Glossary")
    st.markdown("""
| Factor | What it captures |
|--------|-----------------|
| **Mkt-RF** | Excess market return |
| **SMB** | Small-cap premium |
| **HML** | Value premium |
| **RMW** | Profitability premium |
| **CMA** | Conservative investment premium |
| **UMD** | Momentum premium |
""")
    st.divider()
    st.markdown("*Source: Kenneth French Data Library*")

# ── Compute ────────────────────────────────────────────────────────────────────
results, factors, port_returns, stock_returns = compute_all_results(portfolio_choice)
port_series = port_returns[portfolio_choice]

# ── Header ─────────────────────────────────────────────────────────────────────
port_label = "Momentum Long-Only" if portfolio_choice == "long" else "Momentum Long-Short"
st.markdown("# Fama-French Factor Model")
st.markdown(
    f"Return decomposition for **{port_label}** · "
    f"Jegadeesh-Titman momentum strategy · 2018–2024"
)
st.divider()

# ── Section 1: KPI cards ───────────────────────────────────────────────────────
st.markdown("### Alpha Across Models")
st.caption("How much return remains unexplained after controlling for each set of factors. "
           "True alpha shrinks (or disappears) as we add more factors.")

cols = st.columns(4)
for col, (name, res) in zip(cols, results.items()):
    sig = res.alpha_pval < 0.05
    delta_label = "significant ✓" if sig else "not significant"
    col.metric(
        label=f"{name} Alpha (ann.)",
        value=f"{res.alpha_ann * 100:.3f}%",
        delta=f"t = {res.alpha_tstat:.2f} · {delta_label}",
        delta_color="normal" if sig else "off",
    )

st.divider()

# ── Section 2: Model comparison table ─────────────────────────────────────────
st.markdown("### Regression Results — All Models")
st.caption("t-statistics in parentheses. HC3 heteroskedasticity-robust standard errors.")

table = fm.results_to_table(results)

# Highlight alpha row and R²
def highlight_alpha(val):
    """Color alpha values: green if positive, red if negative."""
    try:
        v = float(str(val).replace("(", "").replace(")", ""))
        if "%" in str(val):
            return "color: #4CAF50" if v > 0 else "color: #EF5350"
    except ValueError:
        pass
    return ""

st.dataframe(table, width="stretch")

st.divider()

# ── Section 3: Alpha shrinkage bar chart ──────────────────────────────────────
col_alpha, col_r2 = st.columns(2, gap="large")

with col_alpha:
    st.markdown("### Alpha Shrinkage")
    st.caption("Alpha falls as more systematic factors are controlled for.")

    alpha_vals = {name: res.alpha_ann * 100 for name, res in results.items()}
    colors = [FACTOR_COLORS["Alpha"] if v > 0 else "#EF5350"
              for v in alpha_vals.values()]

    fig_alpha = go.Figure(go.Bar(
        x=list(alpha_vals.keys()),
        y=list(alpha_vals.values()),
        marker_color=colors,
        text=[f"{v:.3f}%" for v in alpha_vals.values()],
        textposition="outside",
        textfont=dict(color="#FAFAFA"),
    ))
    fig_alpha.add_hline(y=0, line_color="#9E9E9E", line_width=1)
    fig_alpha.update_layout(**LAYOUT, height=320, title="Annualized Alpha by Model")
    fig_alpha.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig_alpha, width="stretch")

with col_r2:
    st.markdown("### Explanatory Power (R²)")
    st.caption("R² rises as each model explains more of the portfolio's variance.")

    r2_vals    = {name: res.r2     * 100 for name, res in results.items()}
    adj_r2vals = {name: res.adj_r2 * 100 for name, res in results.items()}

    fig_r2 = go.Figure()
    fig_r2.add_trace(go.Bar(
        name="R²",
        x=list(r2_vals.keys()), y=list(r2_vals.values()),
        marker_color="#2196F3",
        text=[f"{v:.2f}%" for v in r2_vals.values()],
        textposition="outside", textfont=dict(color="#FAFAFA"),
    ))
    fig_r2.add_trace(go.Bar(
        name="Adj. R²",
        x=list(adj_r2vals.keys()), y=list(adj_r2vals.values()),
        marker_color="#FF9800",
        text=[f"{v:.2f}%" for v in adj_r2vals.values()],
        textposition="inside", textfont=dict(color="#FAFAFA"),
    ))
    fig_r2.update_layout(**LAYOUT, height=320, barmode="overlay",
                          title="R² vs Adjusted R² by Model")
    fig_r2.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig_r2, width="stretch")

st.divider()

# ── Section 4: Factor loadings ────────────────────────────────────────────────
st.markdown("### Factor Loadings — Primary Model")
st.caption(f"Model: **{primary_model}**  ·  Bars show β ± 1 standard error  ·  "
           f"β > 0: portfolio loads positively on the factor premium")

res_primary = results[primary_model]
factor_cols = fm.MODELS[primary_model]

# Get standard errors by re-running with statsmodels exposed
import statsmodels.api as sm

df_reg = fm.align(port_series, factors)
y_reg  = df_reg["excess_ret"]
X_reg  = sm.add_constant(df_reg[factor_cols])
ols    = sm.OLS(y_reg, X_reg).fit(cov_type="HC3")

betas = [res_primary.betas[f] for f in factor_cols]
ses   = [ols.bse[f]            for f in factor_cols]
ts    = [res_primary.tstats[f] for f in factor_cols]
labels= [fm.FACTOR_LABELS[f]   for f in factor_cols]
colors= [FACTOR_COLORS.get(f, "#9E9E9E") for f in factor_cols]

fig_betas = go.Figure()
fig_betas.add_trace(go.Bar(
    x=labels, y=betas,
    error_y=dict(type="data", array=ses, visible=True, color="#FAFAFA"),
    marker_color=colors,
    text=[f"β={b:.3f}<br>t={t:.2f}" for b, t in zip(betas, ts)],
    textposition="outside",
    textfont=dict(size=10, color="#FAFAFA"),
))
fig_betas.add_hline(y=0, line_color="#9E9E9E", line_width=1)
fig_betas.update_layout(**LAYOUT, height=360, title=f"Factor Betas ({primary_model})")
st.plotly_chart(fig_betas, width="stretch")

st.divider()

# ── Section 5: Rolling alpha ───────────────────────────────────────────────────
st.markdown("### Rolling Alpha")
st.caption(f"Rolling {roll_window}-day window · annualized · shows how alpha varies by regime")

rolling = fm.rolling_regression(port_series, factors, fm.MODELS[primary_model], roll_window)

fig_ra = go.Figure()
fig_ra.add_trace(go.Scatter(
    x=rolling.index, y=rolling["alpha_ann"] * 100,
    name="Rolling Alpha",
    line=dict(color="#FFC107", width=2),
    fill="tozeroy",
    fillcolor="rgba(255,193,7,0.1)",
    hovertemplate="%{x|%Y-%m-%d}<br>Alpha: %{y:.3f}%<extra></extra>",
))
fig_ra.add_hline(
    y=res_primary.alpha_ann * 100,
    line_dash="dot", line_color="#9E9E9E", line_width=1,
    annotation_text=f"Full-period: {res_primary.alpha_ann*100:.3f}%",
    annotation_position="right",
    annotation_font=dict(color="#9E9E9E", size=10),
)
fig_ra.add_hline(y=0, line_color="#555", line_width=0.8)

for x0, x1, label in [
    ("2020-02-19", "2020-06-01", "COVID crash"),
    ("2022-01-01", "2022-12-31", "Rate shock"),
]:
    fig_ra.add_vrect(
        x0=x0, x1=x1, fillcolor="#EF5350", opacity=0.07, line_width=0,
        annotation_text=label, annotation_position="top left",
        annotation_font=dict(color="#EF5350", size=10),
    )

fig_ra.update_layout(**LAYOUT, height=320,
                     title=f"Rolling {roll_window}-Day Alpha ({primary_model}, annualized)")
fig_ra.update_yaxes(ticksuffix="%")
st.plotly_chart(fig_ra, width="stretch")

st.divider()

# ── Section 6: Rolling factor betas ───────────────────────────────────────────
st.markdown("### Rolling Factor Betas")
st.caption("Factor exposures shift over time — especially around regime changes")

factor_cols_primary = fm.MODELS[primary_model]
fig_rb = go.Figure()

for f in factor_cols_primary:
    fig_rb.add_trace(go.Scatter(
        x=rolling.index, y=rolling[f],
        name=fm.FACTOR_LABELS[f],
        line=dict(color=FACTOR_COLORS.get(f, "#9E9E9E"), width=1.8),
        hovertemplate=f"<b>{fm.FACTOR_LABELS[f]}</b><br>%{{x|%Y-%m-%d}}<br>β=%{{y:.3f}}<extra></extra>",
    ))

fig_rb.add_hline(y=0, line_color="#555", line_width=0.8)
fig_rb.update_layout(**LAYOUT, height=360,
                     title=f"Rolling {roll_window}-Day Factor Betas ({primary_model})")
st.plotly_chart(fig_rb, width="stretch")

st.divider()

# ── Section 7: Cumulative return decomposition ─────────────────────────────────
st.markdown("### Cumulative Return Decomposition")
st.caption(f"Stacked area: how much each factor contributed to total return ({primary_model})")

decomp = fm.return_decomposition(res_primary, factors)

# Cumulate each component
cumdecomp = (1 + decomp).cumprod() - 1

component_order = ["Alpha"] + [fm.FACTOR_LABELS[f] for f in factor_cols_primary] + ["Residual (α-specific)"]
component_order = [c for c in component_order if c in cumdecomp.columns]

fig_decomp = go.Figure()
for comp in component_order:
    if comp not in cumdecomp.columns:
        continue
    fig_decomp.add_trace(go.Scatter(
        x=cumdecomp.index, y=cumdecomp[comp] * 100,
        name=comp,
        stackgroup="one",
        line=dict(width=0.5, color=FACTOR_COLORS.get(comp, "#9E9E9E")),
        fillcolor=FACTOR_COLORS.get(comp, "#9E9E9E"),
        hovertemplate=f"<b>{comp}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.2f}}%<extra></extra>",
    ))

fig_decomp.update_layout(**LAYOUT, height=380,
                          title="Cumulative Return Attribution (stacked, % contribution)")
fig_decomp.update_yaxes(ticksuffix="%")
st.plotly_chart(fig_decomp, width="stretch")

st.divider()

# ── Section 8: Per-ticker table ────────────────────────────────────────────────
st.markdown("### Per-Ticker Factor Exposures")
st.caption(f"Individual stock regressions · Model: {primary_model}")

with st.spinner("Running per-ticker regressions..."):
    ticker_table = fm.ticker_factor_table(
        stock_returns.drop(columns=["SPY"], errors="ignore"),
        factors,
        model=primary_model,
    )

def color_alpha(v):
    try:
        return "color: #4CAF50" if float(v) > 0 else "color: #EF5350"
    except Exception:
        return ""

def color_pval(v):
    try:
        return "font-weight: bold" if float(v) < 0.05 else "color: #9E9E9E"
    except Exception:
        return ""

st.dataframe(
    ticker_table.style
        .format({"Alpha (ann %)": "{:.3f}%", "α p-value": "{:.3f}", "R²": "{:.4f}"}
                | {f"{f} β": "{:.3f}" for f in fm.MODELS[primary_model]})
        .map(color_alpha, subset=["Alpha (ann %)"])
        .map(color_pval,  subset=["α p-value"]),
    width="stretch",
)

st.divider()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.caption(
    "Factor data: Kenneth French Data Library · "
    "Portfolio returns: Jegadeesh-Titman momentum strategy (Project 1) · "
    "OLS with HC3 heteroskedasticity-robust standard errors"
)
