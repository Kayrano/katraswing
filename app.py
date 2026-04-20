"""
KATRASWING — 5m NQ Futures Signal Dashboard
Real-time trading signals driven by chart analysis + breaking news.

Run with: streamlit run app.py
"""

import time
import streamlit as st

st.set_page_config(
    page_title="Katraswing — 5m Signal Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  body, .stApp { background-color: #0e1117; color: #fafafa; }
  .stTextInput > div > div > input { background: #1e2130; color: #fafafa; border-color: #2a2d3e; }
  .stButton > button { background: #1e2130; color: #fafafa; border: 1px solid #2a2d3e; }
  .stButton > button:hover { border-color: #42a5f5; color: #42a5f5; }
  section[data-testid="stSidebar"] { background: #0e1117; }
  div[data-testid="stMetric"] { background: #1e2130; border-radius: 8px; padding: 10px; }
  hr { border-color: #2a2d3e; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Katraswing")
    st.markdown("**5m Signal Dashboard**")
    st.markdown("---")

    ticker_input = st.text_input(
        "Ticker",
        value=st.session_state.get("ticker", "NQ=F"),
        placeholder="e.g. NQ=F, AAPL, TSLA",
        help="Use NQ=F for Nasdaq E-mini futures, MNQ=F for Micro.",
    ).strip().upper()

    finnhub_key = st.text_input(
        "Finnhub API Key",
        value=st.session_state.get("finnhub_key", ""),
        type="password",
        help="Get a free key at finnhub.io",
    ).strip()

    st.markdown("---")

    account_size = st.number_input(
        "Account Size ($)",
        value=st.session_state.get("account_size", 100_000),
        min_value=1_000,
        step=5_000,
    )
    risk_pct = st.slider(
        "Risk per trade (%)",
        min_value=0.25, max_value=3.0, step=0.25,
        value=st.session_state.get("risk_pct", 1.0),
    )

    st.markdown("---")

    auto_refresh = st.checkbox(
        "Auto-refresh (5 min)",
        value=st.session_state.get("auto_refresh", False),
    )
    run_btn = st.button("🔄 Run Signal", use_container_width=True, type="primary")

    st.markdown("---")
    st.caption("Data: yfinance + Finnhub  \nCharts: Plotly  \nv2.0 NQ Edition")

# Persist settings
st.session_state["ticker"] = ticker_input
st.session_state["finnhub_key"] = finnhub_key
st.session_state["account_size"] = account_size
st.session_state["risk_pct"] = risk_pct
st.session_state["auto_refresh"] = auto_refresh

# Auto-refresh: rerun every 5 minutes
if auto_refresh:
    last = st.session_state.get("last_refresh_ts", 0)
    if time.time() - last > 300:
        st.session_state["last_refresh_ts"] = time.time()
        st.session_state["needs_run"] = True

needs_run = run_btn or st.session_state.pop("needs_run", False)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    f"# ⚡ {ticker_input} — 5m Signal Dashboard",
)

if not needs_run and "signal_result" not in st.session_state:
    st.info("Enter a ticker in the sidebar and click **Run Signal** to load the dashboard.")
    st.stop()

# ── Run signal pipeline ───────────────────────────────────────────────────────
if needs_run:
    if not ticker_input:
        st.warning("Enter a ticker symbol in the sidebar.")
        st.stop()

    with st.spinner(f"Fetching 5m data and news for {ticker_input}…"):
        from agents.signal_engine import run_signal
        result = run_signal(
            ticker=ticker_input,
            finnhub_api_key=finnhub_key,
            account_size=account_size,
            risk_pct=risk_pct,
        )
    st.session_state["signal_result"] = result
    st.session_state["last_refresh_ts"] = time.time()
else:
    result = st.session_state["signal_result"]

# ── Error guard ───────────────────────────────────────────────────────────────
if result.error:
    st.error(f"Error: {result.error}")
    st.stop()

# ── Last refresh timestamp ────────────────────────────────────────────────────
from datetime import datetime
st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}  |  Ticker: {result.ticker}")

# ── Main layout: chart (left) + signal+news (right) ──────────────────────────
from ui.chart import (
    render_5m_chart,
    render_signal_box,
    render_news_feed,
    render_indicators,
    render_pattern_summary,
)

col_chart, col_panel = st.columns([6, 4])

with col_chart:
    render_5m_chart(result)

with col_panel:
    render_signal_box(result)
    st.markdown("---")
    st.markdown("#### 📰 Breaking News")
    render_news_feed(result)

# ── Bottom row: indicators + patterns ────────────────────────────────────────
st.markdown("---")
ind_col, pat_col = st.columns([1, 1])

with ind_col:
    st.markdown("#### 📊 Indicators")
    render_indicators(result)

with pat_col:
    st.markdown("#### 🔍 Chart Patterns")
    render_pattern_summary(result)
