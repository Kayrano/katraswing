"""
KATRASWING — 5m Signal Dashboard
Three fixed instruments: NQ Mini, ES Mini, Gram Gold / USD

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

st.markdown("""
<style>
  body, .stApp { background-color: #0e1117; color: #fafafa; }
  .stTextInput > div > div > input { background: #1e2130; color: #fafafa; border-color: #2a2d3e; }
  .stButton > button { background: #1e2130; color: #fafafa; border: 1px solid #2a2d3e; }
  .stButton > button:hover { border-color: #42a5f5; color: #42a5f5; }
  section[data-testid="stSidebar"] { background: #0e1117; }
  .stTabs [data-baseweb="tab-list"] { gap: 8px; background: #0e1117; }
  .stTabs [data-baseweb="tab"] {
    background: #1e2130; border-radius: 8px 8px 0 0;
    color: #aaa; padding: 8px 20px; font-weight: 600;
  }
  .stTabs [aria-selected="true"] { background: #2a2d3e; color: #fafafa; }
  hr { border-color: #2a2d3e; }
</style>
""", unsafe_allow_html=True)

# ── Instruments ───────────────────────────────────────────────────────────────
INSTRUMENTS = [
    {"ticker": "NQ=F",     "label": "🔵 NQ Mini",    "name": "Nasdaq 100 E-mini Futures"},
    {"ticker": "ES=F",     "label": "🟢 ES Mini",    "name": "S&P 500 E-mini Futures"},
    {"ticker": "XAUUSD=X", "label": "🟡 Gold / g",   "name": "Gram Gold Spot vs USD"},
]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Katraswing")
    st.markdown("**5m Signal Dashboard**")
    st.markdown("---")

    finnhub_key = st.text_input(
        "Finnhub API Key",
        value=st.session_state.get("finnhub_key", ""),
        type="password",
        help="Free key at finnhub.io — required for news feed.",
    ).strip()

    st.markdown("---")

    account_size = st.number_input(
        "Account Size ($)",
        value=st.session_state.get("account_size", 100_000),
        min_value=1_000, step=5_000,
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
    run_btn = st.button("🔄 Run All Signals", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("**Instruments**")
    for inst in INSTRUMENTS:
        st.markdown(f"{inst['label']} `{inst['ticker']}`  \n<span style='font-size:11px;color:#555;'>{inst['name']}</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Data: yfinance + Finnhub  \nv2.0 Multi-Instrument")

# Persist settings
st.session_state["finnhub_key"] = finnhub_key
st.session_state["account_size"] = account_size
st.session_state["risk_pct"] = risk_pct
st.session_state["auto_refresh"] = auto_refresh

# Auto-refresh trigger
if auto_refresh:
    last = st.session_state.get("last_refresh_ts", 0)
    if time.time() - last > 300:
        st.session_state["last_refresh_ts"] = time.time()
        st.session_state.pop("results", None)  # clear cache → re-fetch

needs_run = run_btn or ("results" not in st.session_state)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# ⚡ Katraswing — 5m Signal Dashboard")

# ── Fetch all signals ─────────────────────────────────────────────────────────
if needs_run:
    from agents.signal_engine import run_signal
    results = {}
    progress = st.progress(0, text="Fetching signals…")
    for i, inst in enumerate(INSTRUMENTS):
        progress.progress((i + 1) / len(INSTRUMENTS),
                          text=f"Loading {inst['name']}…")
        results[inst["ticker"]] = run_signal(
            ticker=inst["ticker"],
            display_name=inst["name"],
            finnhub_api_key=finnhub_key,
            account_size=account_size,
            risk_pct=risk_pct,
        )
    progress.empty()
    st.session_state["results"] = results
    st.session_state["last_refresh_ts"] = time.time()
else:
    results = st.session_state["results"]

from datetime import datetime
st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

# ── Tabs ──────────────────────────────────────────────────────────────────────
from ui.chart import (
    render_5m_chart, render_signal_box, render_news_feed,
    render_indicators, render_pattern_summary,
)

tabs = st.tabs([inst["label"] for inst in INSTRUMENTS])

for tab, inst in zip(tabs, INSTRUMENTS):
    with tab:
        result = results[inst["ticker"]]

        st.markdown(f"### {inst['name']} &nbsp; `{inst['ticker']}`", unsafe_allow_html=True)

        if result.error:
            st.error(f"Error loading {inst['ticker']}: {result.error}")
            continue

        col_chart, col_panel = st.columns([6, 4])

        with col_chart:
            render_5m_chart(result)

        with col_panel:
            render_signal_box(result)
            st.markdown("---")
            st.markdown("#### 📰 Breaking News")
            render_news_feed(result)

        st.markdown("---")
        ind_col, pat_col = st.columns([1, 1])
        with ind_col:
            st.markdown("#### 📊 Indicators")
            render_indicators(result)
        with pat_col:
            st.markdown("#### 🔍 Chart Patterns")
            render_pattern_summary(result)
