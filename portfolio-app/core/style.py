"""
core/style.py
-------------
Shared visual design system for the portfolio app.
Import and call apply_style() at the top of every page.
"""

import streamlit as st

# ── Color palette ──────────────────────────────────────────────────────────────
C = {
    "primary":  "#818CF8",   # indigo — main accent
    "green":    "#34D399",   # emerald — positive / long
    "amber":    "#FBBF24",   # amber — neutral / warning
    "red":      "#F87171",   # red — negative / short
    "cyan":     "#67E8F9",   # cyan — secondary series
    "purple":   "#C084FC",   # purple — tertiary series
    "muted":    "#64748B",   # slate — muted elements
    "spy":      "#94A3B8",   # benchmark lines
    "bg_card":  "#1E293B",   # card background
    "border":   "#334155",   # border color
}

# ── Plotly layout template ─────────────────────────────────────────────────────
CHART = dict(
    paper_bgcolor  = "rgba(0,0,0,0)",
    plot_bgcolor   = "rgba(15,23,42,0.4)",
    font           = dict(color="#94A3B8", family="Inter, system-ui, sans-serif", size=12),
    title_font     = dict(color="#F1F5F9", size=15, family="Inter, system-ui, sans-serif"),
    xaxis          = dict(gridcolor="rgba(51,65,85,0.5)", zeroline=False,
                          linecolor="#1E293B", tickcolor="#475569"),
    yaxis          = dict(gridcolor="rgba(51,65,85,0.5)", zeroline=False,
                          linecolor="#1E293B", tickcolor="#475569"),
    legend         = dict(bgcolor="rgba(0,0,0,0)", borderwidth=0,
                          font=dict(color="#94A3B8")),
    margin         = dict(l=0, r=0, t=48, b=0),
    hoverlabel     = dict(bgcolor="#1E293B", bordercolor="#475569",
                          font=dict(color="#F1F5F9", size=12)),
    hovermode      = "x unified",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
CSS = """
<style>
/* ── Global ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; max-width: 1400px; }

/* ── Gradient text ── */
.grad {
    background: linear-gradient(135deg, #818CF8 0%, #C084FC 50%, #F472B6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 800;
}

/* ── Page title ── */
.page-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818CF8, #C084FC);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
    line-height: 1.1;
}
.page-subtitle {
    color: #64748B;
    font-size: 0.95rem;
    margin-top: 0;
    margin-bottom: 1.5rem;
}

/* ── Metric cards ── */
.kpi-grid { display: flex; gap: 12px; margin-bottom: 1.5rem; flex-wrap: wrap; }
.kpi {
    background: #1E293B;
    border-radius: 12px;
    padding: 16px 20px;
    border-left: 3px solid;
    flex: 1; min-width: 120px;
}
.kpi-value { font-size: 1.7rem; font-weight: 700; color: #F1F5F9; line-height: 1; }
.kpi-label { font-size: 0.72rem; color: #64748B; text-transform: uppercase;
             letter-spacing: 0.08em; margin-top: 4px; }
.kpi-delta { font-size: 0.78rem; margin-top: 6px; }

/* ── Section header ── */
.section { font-size: 1.1rem; font-weight: 700; color: #E2E8F0;
           margin: 1.5rem 0 0.4rem 0; }
.caption { font-size: 0.8rem; color: #64748B; margin-bottom: 0.8rem; }

/* ── Divider ── */
.hdivider { border: none; border-top: 1px solid #1E293B; margin: 1.5rem 0; }

/* ── Tag pills ── */
.tag {
    display: inline-block;
    background: rgba(129,140,248,0.15);
    color: #818CF8;
    border: 1px solid rgba(129,140,248,0.3);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
}

/* ── Streamlit overrides ── */
div[data-testid="stMetric"] {
    background: #1E293B;
    border-radius: 12px;
    padding: 14px 18px;
    border: 1px solid #334155;
}
div[data-testid="stMetricValue"] { color: #F1F5F9 !important; font-weight: 700; }
div[data-testid="stMetricLabel"] { color: #64748B !important; font-size: 0.78rem !important; }

[data-testid="stSidebar"] {
    background: #0A1120 !important;
    border-right: 1px solid #1E293B;
}
</style>
"""


def apply_style():
    """Inject global CSS using st.html() so it never renders as a visible block."""
    st.html(CSS)


def page_header(title: str, subtitle: str = ""):
    """Render a gradient page title + subtitle."""
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def section(label: str, caption: str = ""):
    st.markdown(f'<div class="section">{label}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="caption">{caption}</div>', unsafe_allow_html=True)


def kpi_row(metrics: list[dict]):
    """
    Render a row of colored KPI cards.
    Each dict: {label, value, delta="", color=C["primary"]}
    """
    cards_html = '<div class="kpi-grid">'
    for m in metrics:
        color   = m.get("color", C["primary"])
        delta   = m.get("delta", "")
        delta_html = f'<div class="kpi-delta" style="color:{color}">{delta}</div>' if delta else ""
        cards_html += f"""
        <div class="kpi" style="border-color:{color}">
            <div class="kpi-value">{m["value"]}</div>
            <div class="kpi-label">{m["label"]}</div>
            {delta_html}
        </div>"""
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)
