"""
app.py — Home page of the Quantitative Finance Portfolio
"""

import streamlit as st

st.set_page_config(
    page_title="Quant Portfolio · Kyra Roy",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

from core.style import apply_style, CSS, C, section
apply_style()

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 2.5rem 0 1rem 0;">
  <div style="font-size:0.85rem; color:#818CF8; font-weight:600;
              letter-spacing:0.15em; text-transform:uppercase; margin-bottom:0.6rem;">
    QUANTITATIVE FINANCE
  </div>
  <div style="font-size:3.2rem; font-weight:900; line-height:1.05;
              background:linear-gradient(135deg,#818CF8 0%,#C084FC 45%,#F472B6 100%);
              -webkit-background-clip:text; -webkit-text-fill-color:transparent;
              background-clip:text; margin-bottom:0.8rem;">
    Portfolio of 5 Quant Projects
  </div>
  <div style="font-size:1.1rem; color:#94A3B8; max-width:680px; line-height:1.6;">
    End-to-end implementations of core quantitative finance strategies —
    each with a live interactive dashboard. Built in Python using real
    market data from 2018–2024.
  </div>
  <div style="margin-top:1rem;">
    <span class="tag">Python</span>
    <span class="tag">Pandas</span>
    <span class="tag">NumPy</span>
    <span class="tag">SciPy</span>
    <span class="tag">statsmodels</span>
    <span class="tag">Plotly</span>
    <span class="tag">Streamlit</span>
    <span class="tag">yfinance</span>
  </div>
</div>
<hr class="hdivider">
""", unsafe_allow_html=True)

# ── Project cards ──────────────────────────────────────────────────────────────
section("Projects", "Use the sidebar to open any project dashboard.")

PROJECTS = [
    {
        "emoji": "📈",
        "name":  "Momentum Strategy",
        "page":  "1_📈_Momentum",
        "color": C["primary"],
        "tags":  ["Jegadeesh-Titman 1993", "12-1 Month Lookback", "Backtester"],
        "kpis":  [("Sharpe (Long)", "0.686"), ("CAGR", "18.98%"), ("Max DD", "−38.6%")],
        "desc":  "Cross-sectional momentum on 484 S&P 500 stocks. "
                 "Monthly rebalance into top/bottom decile. "
                 "Full event-driven backtester with transaction costs.",
    },
    {
        "emoji": "🛡️",
        "name":  "Risk Dashboard",
        "page":  "2_🛡️_Risk_Dashboard",
        "color": C["green"],
        "tags":  ["VaR / CVaR", "Sharpe · Sortino · Calmar", "Correlation"],
        "kpis":  [("VaR 95%", "−1.7%"), ("Sortino", "0.94"), ("Max DD", "−36%")],
        "desc":  "Institutional risk metrics on a 20-stock diversified portfolio. "
                 "Historical, parametric, and Monte Carlo VaR. "
                 "Rolling volatility, drawdown analysis, and correlation heatmap.",
    },
    {
        "emoji": "⚖️",
        "name":  "Pairs Trading",
        "page":  "3_⚖️_Pairs_Trading",
        "color": C["cyan"],
        "tags":  ["Engle-Granger", "OU Process", "Stat Arb"],
        "kpis":  [("Pairs", "12"), ("Win Rate", "53.5%"), ("Trades", "226")],
        "desc":  "Market-neutral stat arb using Engle-Granger cointegration. "
                 "Hedge ratio via OLS, OU half-life estimation. "
                 "Z-score entry/exit with stop-loss, tested out-of-sample 2022–2024.",
    },
    {
        "emoji": "📊",
        "name":  "Fama-French Factor Model",
        "page":  "4_📊_Factor_Model",
        "color": C["amber"],
        "tags":  ["CAPM → FF3 → FF5 → FF5+Mom", "OLS Regression", "Rolling Alpha"],
        "kpis":  [("FF5+M R²", "62%"), ("Alpha (ann)", "4.2%"), ("Mkt-RF β", "0.94")],
        "desc":  "Return decomposition of the momentum strategy across four nested models. "
                 "Rolling 126-day factor exposures, alpha shrinkage analysis, "
                 "and per-ticker factor attribution.",
    },
    {
        "emoji": "🎲",
        "name":  "Monte Carlo Simulator",
        "page":  "5_🎲_Monte_Carlo",
        "color": C["purple"],
        "tags":  ["GBM + Cholesky", "Block Bootstrap", "Efficient Frontier"],
        "kpis":  [("Simulations", "10,000"), ("Methods", "2"), ("Horizon", "10yr")],
        "desc":  "Portfolio path simulation via GBM (Itô-corrected, correlated) "
                 "and historical block bootstrap. Markowitz efficient frontier, "
                 "probability tables, and terminal wealth distributions.",
    },
]

for i in range(0, len(PROJECTS), 3):
    cols = st.columns(min(3, len(PROJECTS) - i), gap="medium")
    for col, proj in zip(cols, PROJECTS[i:i+3]):
        with col:
            kpi_html = "".join(
                f'<div style="margin-bottom:6px;">'
                f'<span style="color:#94A3B8;font-size:0.75rem;">{k}: </span>'
                f'<span style="color:#F1F5F9;font-weight:700;font-size:0.9rem;">{v}</span>'
                f'</div>'
                for k, v in proj["kpis"]
            )
            tags_html = "".join(f'<span class="tag">{t}</span>' for t in proj["tags"])

            st.markdown(f"""
<div style="background:linear-gradient(160deg,#1E293B 0%,#0F172A 100%);
            border:1px solid #334155; border-top: 3px solid {proj['color']};
            border-radius:14px; padding:22px; height:100%;
            transition: box-shadow .2s;">
  <div style="font-size:1.8rem; margin-bottom:8px;">{proj['emoji']}</div>
  <div style="font-size:1.05rem; font-weight:700; color:#F1F5F9;
              margin-bottom:8px;">{proj['name']}</div>
  <div style="font-size:0.82rem; color:#94A3B8; line-height:1.5;
              margin-bottom:14px;">{proj['desc']}</div>
  <div style="margin-bottom:14px;">{tags_html}</div>
  <div style="border-top:1px solid #1E293B; padding-top:12px;">{kpi_html}</div>
</div>
""", unsafe_allow_html=True)

# ── Stats row ──────────────────────────────────────────────────────────────────
st.markdown("<hr class='hdivider'>", unsafe_allow_html=True)

stat_cols = st.columns(5)
stats = [
    ("484",       "S&P 500 stocks"),
    ("1,760",     "Trading days"),
    ("7 years",   "Data: 2018–2024"),
    ("5",         "Live dashboards"),
    ("10,000+",   "MC simulations"),
]
for col, (val, label) in zip(stat_cols, stats):
    with col:
        st.markdown(f"""
<div style="text-align:center; padding:16px;">
  <div style="font-size:1.9rem; font-weight:800;
              background:linear-gradient(135deg,#818CF8,#C084FC);
              -webkit-background-clip:text; -webkit-text-fill-color:transparent;
              background-clip:text;">{val}</div>
  <div style="font-size:0.78rem; color:#64748B; text-transform:uppercase;
              letter-spacing:0.06em; margin-top:4px;">{label}</div>
</div>""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="padding:8px 0 16px 0;">
  <div style="font-size:1.1rem; font-weight:800; color:#F1F5F9;">Quant Portfolio</div>
  <div style="font-size:0.78rem; color:#64748B;">Kyra Roy · Harvey Mudd</div>
</div>
""", unsafe_allow_html=True)
    st.markdown("**Navigate** using the pages below.")
    st.caption("Each page is an independent live dashboard.")
