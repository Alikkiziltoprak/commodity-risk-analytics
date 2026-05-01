"""
app.py
------
Commodity Risk Analytics Dashboard
Built with Streamlit + Plotly

Run locally:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from data_loader import load_commodity_data
from risk_metrics import (
    compute_returns,
    risk_summary,
    var_summary,
    rolling_vol,
    correlation_matrix,
    drawdown_series,
    historical_var,
    cvar,
    annualised_vol,
    portfolio_var,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Commodity Risk Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    
    .main { background-color: #0d1117; }
    
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }
    .metric-label {
        font-size: 11px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-family: 'IBM Plex Mono', monospace;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 600;
        color: #e6edf3;
        font-family: 'IBM Plex Mono', monospace;
        margin-top: 4px;
    }
    .metric-delta-pos { color: #3fb950; font-size: 13px; }
    .metric-delta-neg { color: #f85149; font-size: 13px; }
    
    .section-header {
        font-size: 11px;
        font-weight: 600;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        border-bottom: 1px solid #21262d;
        padding-bottom: 8px;
        margin: 24px 0 16px 0;
        font-family: 'IBM Plex Mono', monospace;
    }
    
    div[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #21262d;
    }
    
    .stSelectbox label, .stSlider label, .stMultiSelect label {
        color: #8b949e !important;
        font-size: 12px !important;
        font-family: 'IBM Plex Mono', monospace !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
</style>
""", unsafe_allow_html=True)

COLORS = {
    "WTI Crude Oil ($/bbl)":   "#f0883e",
    "Natural Gas ($/MMBtu)":   "#58a6ff",
    "Coal ($/ton)":            "#8b949e",
    "Baltic Dry Index":        "#3fb950",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ COMMODITY RISK")
    st.markdown("---")

    data_source = st.selectbox(
        "Data Source",
        ["Demo (Synthetic)", "EIA API (Live)"],
        index=0,
    )

    api_key = None
    if data_source == "EIA API (Live)":
        api_key = st.text_input("EIA API Key", type="password",
                                help="Free key at eia.gov/opendata")

    lookback = st.slider("Lookback Period (days)", 60, 504, 252, step=21)
    conf_level = st.selectbox("VaR Confidence Level", [0.90, 0.95, 0.99],
                               index=1, format_func=lambda x: f"{int(x*100)}%")
    vol_window = st.slider("Rolling Vol Window (days)", 10, 60, 30)

    st.markdown("---")
    st.markdown("**Portfolio Weights**")
    w_crude = st.slider("WTI Crude", 0, 100, 40)
    w_gas   = st.slider("Natural Gas", 0, 100, 30)
    w_coal  = st.slider("Coal", 0, 100, 15)
    w_bdi   = st.slider("Baltic Dry", 0, 100, 15)

    total_w = w_crude + w_gas + w_coal + w_bdi
    if total_w == 0:
        total_w = 1

    weights = {
        "WTI Crude Oil ($/bbl)":  w_crude / total_w,
        "Natural Gas ($/MMBtu)":  w_gas   / total_w,
        "Coal ($/ton)":           w_coal  / total_w,
        "Baltic Dry Index":       w_bdi   / total_w,
    }

    st.caption(f"Normalised: {int(weights['WTI Crude Oil ($/bbl)']*100)}% / "
               f"{int(weights['Natural Gas ($/MMBtu)']*100)}% / "
               f"{int(weights['Coal ($/ton)']*100)}% / "
               f"{int(weights['Baltic Dry Index']*100)}%")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_data(source, key, days):
    src = "eia" if source == "EIA API (Live)" else "demo"
    return load_commodity_data(source=src, api_key=key, n_days=days)

try:
    prices_full = get_data(data_source, api_key, 504)
except Exception as e:
    st.error(f"Data load error: {e}")
    st.stop()

prices = prices_full.tail(lookback)
returns = compute_returns(prices)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# Commodity Risk Analytics")
st.markdown(
    f"<span style='color:#8b949e;font-family:IBM Plex Mono,monospace;font-size:13px'>"
    f"Period: {prices.index[0].date()} → {prices.index[-1].date()} &nbsp;|&nbsp; "
    f"{len(prices)} trading days &nbsp;|&nbsp; "
    f"{'Live (EIA)' if data_source != 'Demo (Synthetic)' else 'Synthetic data'}"
    f"</span>",
    unsafe_allow_html=True,
)

# ── KPI cards ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Market Snapshot</div>', unsafe_allow_html=True)
cols = st.columns(4)
for i, (col_name, color) in enumerate(COLORS.items()):
    last = prices[col_name].iloc[-1]
    prev = prices[col_name].iloc[-2]
    chg  = (last - prev) / prev
    sign = "+" if chg >= 0 else ""
    delta_class = "metric-delta-pos" if chg >= 0 else "metric-delta-neg"
    with cols[i]:
        short_name = col_name.split(" (")[0]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{short_name}</div>
            <div class="metric-value">{last:,.2f}</div>
            <div class="{delta_class}">{sign}{chg:.2%} 1d</div>
        </div>
        """, unsafe_allow_html=True)

# ── Price chart ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Normalised Price Performance (Base = 100)</div>',
            unsafe_allow_html=True)

fig_price = go.Figure()
for col, color in COLORS.items():
    normalised = prices[col] / prices[col].iloc[0] * 100
    fig_price.add_trace(go.Scatter(
        x=prices.index, y=normalised,
        name=col.split(" (")[0],
        line=dict(color=color, width=1.8),
        hovertemplate="%{y:.1f}<extra>" + col.split(" (")[0] + "</extra>",
    ))

fig_price.update_layout(
    height=320,
    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
    font=dict(color="#8b949e", family="IBM Plex Mono"),
    legend=dict(bgcolor="#0d1117", bordercolor="#30363d", borderwidth=1,
                orientation="h", yanchor="bottom", y=1.02),
    xaxis=dict(gridcolor="#21262d", showline=False),
    yaxis=dict(gridcolor="#21262d", showline=False),
    margin=dict(l=0, r=0, t=10, b=0),
    hovermode="x unified",
)
st.plotly_chart(fig_price, use_container_width=True)

# ── Risk summary table ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Risk Summary</div>', unsafe_allow_html=True)
summary = risk_summary(prices)
st.dataframe(
    summary,
    use_container_width=True,
    height=200,
)

# ── VaR / CVaR + Rolling Vol ──────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-header">Historical Return Distribution</div>',
                unsafe_allow_html=True)
    commodity_sel = st.selectbox("Select Commodity", list(COLORS.keys()), label_visibility="collapsed")
    r = returns[commodity_sel].dropna()
    var95 = historical_var(r, conf_level)
    cvar95 = cvar(r, conf_level)

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=r, nbinsx=60,
        marker_color=COLORS[commodity_sel],
        opacity=0.7,
        name="Returns",
    ))
    fig_hist.add_vline(x=-var95,  line_color="#f85149", line_dash="dash",
                       annotation_text=f"VaR {int(conf_level*100)}%: {var95:.2%}",
                       annotation_font_color="#f85149")
    fig_hist.add_vline(x=-cvar95, line_color="#ff7b72", line_dash="dot",
                       annotation_text=f"CVaR: {cvar95:.2%}",
                       annotation_font_color="#ff7b72")
    fig_hist.update_layout(
        height=280, paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", family="IBM Plex Mono"),
        xaxis=dict(gridcolor="#21262d", tickformat=".1%"),
        yaxis=dict(gridcolor="#21262d"),
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    st.markdown('<div class="section-header">Rolling Annualised Volatility</div>',
                unsafe_allow_html=True)
    fig_vol = go.Figure()
    for col, color in COLORS.items():
        rv = rolling_vol(returns[col], window=vol_window).dropna()
        fig_vol.add_trace(go.Scatter(
            x=rv.index, y=rv,
            name=col.split(" (")[0],
            line=dict(color=color, width=1.5),
        ))
    fig_vol.update_layout(
        height=280, paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", family="IBM Plex Mono"),
        legend=dict(bgcolor="#0d1117", bordercolor="#30363d", borderwidth=1,
                    orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d", tickformat=".0%"),
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig_vol, use_container_width=True)

# ── Correlation + Drawdown ────────────────────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.markdown('<div class="section-header">Correlation Matrix</div>', unsafe_allow_html=True)
    corr = correlation_matrix(returns)
    short_names = [c.split(" (")[0] for c in corr.columns]
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values,
        x=short_names, y=short_names,
        colorscale=[[0, "#f85149"], [0.5, "#0d1117"], [1, "#3fb950"]],
        zmin=-1, zmax=1,
        text=corr.round(2).values,
        texttemplate="%{text}",
        textfont=dict(size=12, family="IBM Plex Mono"),
    ))
    fig_corr.update_layout(
        height=280, paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", family="IBM Plex Mono"),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

with col4:
    st.markdown('<div class="section-header">Drawdown Analysis</div>', unsafe_allow_html=True)
    fig_dd = go.Figure()
    for col, color in COLORS.items():
        dd = drawdown_series(prices[col])
        fig_dd.add_trace(go.Scatter(
            x=dd.index, y=dd,
            name=col.split(" (")[0],
            line=dict(color=color, width=1.5),
            fill="tozeroy",
            fillcolor="rgba({},{},{},0.07)".format(int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)),
        ))
    fig_dd.update_layout(
        height=280, paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", family="IBM Plex Mono"),
        legend=dict(bgcolor="#0d1117", bordercolor="#30363d", borderwidth=1,
                    orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d", tickformat=".0%"),
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig_dd, use_container_width=True)

# ── Portfolio VaR ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Portfolio Risk</div>', unsafe_allow_html=True)

port_returns = returns[list(weights.keys())].dot(pd.Series(weights))
port_var  = historical_var(port_returns, conf_level)
port_cvar = cvar(port_returns, conf_level)
port_vol  = annualised_vol(port_returns)

pc1, pc2, pc3, pc4 = st.columns(4)
with pc1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Portfolio VaR ({int(conf_level*100)}%, 1-day)</div>
        <div class="metric-value" style="color:#f85149">{port_var:.2%}</div>
    </div>""", unsafe_allow_html=True)
with pc2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Portfolio CVaR ({int(conf_level*100)}%, 1-day)</div>
        <div class="metric-value" style="color:#ff7b72">{port_cvar:.2%}</div>
    </div>""", unsafe_allow_html=True)
with pc3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Annualised Volatility</div>
        <div class="metric-value" style="color:#f0883e">{port_vol:.1%}</div>
    </div>""", unsafe_allow_html=True)
with pc4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Diversification Benefit</div>
        <div class="metric-value" style="color:#3fb950">
            {sum(historical_var(returns[c], conf_level) * w for c, w in weights.items()) - port_var:.2%}
        </div>
    </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<span style='color:#484f58;font-size:11px;font-family:IBM Plex Mono,monospace'>"
    "commodity-risk-analytics · github.com/your-username/commodity-risk-analytics · "
    "Data: EIA / Synthetic GBM · Risk: Historical VaR/CVaR"
    "</span>",
    unsafe_allow_html=True,
)
