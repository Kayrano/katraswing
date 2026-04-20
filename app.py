"""
KATRASWING — 5m Signal Dashboard
Three fixed instruments: NQ Mini, ES Mini, Gram Gold / USD

Run locally with: streamlit run app.py
MT5 auto-trading requires MetaTrader5 to be open on the same Windows machine.
"""

import threading
import time
from datetime import date, datetime

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
  .mt5-card { background: #1e2130; border-radius: 10px; padding: 12px 16px; margin-bottom: 8px; }
  .mt5-pos  { background: #111827; border-radius: 6px; padding: 6px 10px; font-size: 12px; margin: 3px 0; }
</style>
""", unsafe_allow_html=True)

# ── Instruments ───────────────────────────────────────────────────────────────
INSTRUMENTS = [
    {"ticker": "NQ=F",     "label": "🔵 NQ Mini",   "name": "Nasdaq 100 E-mini Futures"},
    {"ticker": "ES=F",     "label": "🟢 ES Mini",   "name": "S&P 500 E-mini Futures"},
    {"ticker": "XAUUSD=X", "label": "🟡 Gold / g",  "name": "Gram Gold Spot vs USD"},
]

# ── MT5 background thread state (module-level → survives Streamlit reruns) ───
# Written by the background thread, read by Streamlit on each rerun.
_MT5: dict = {
    "thread":      None,
    "stop_event":  None,
    "running":     False,
    "connected":   False,
    "last_check":  None,       # datetime of last poll
    "last_signal": None,       # dict with last fired signal info
    "positions":   [],         # list of MT5Position objects
    "log":         [],         # last 20 log lines
    "error":       "",
}


def _mt5_log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    _MT5["log"].append(f"{ts}  {msg}")
    _MT5["log"] = _MT5["log"][-20:]


def _mt5_loop(stop_event: threading.Event, config: dict):
    """Background thread: polls signal engine and sends orders to MT5."""
    from agents.signal_engine import run_signal
    from utils.mt5_bridge import (
        connect, disconnect, ensure_connected,
        send_from_signal_result, get_open_positions,
    )

    _mt5_log("Connecting to MT5 terminal…")
    if not connect():
        _MT5["error"] = "Cannot connect to MT5. Is the terminal open?"
        _mt5_log(f"ERROR: {_MT5['error']}")
        _MT5["running"] = False
        return

    _MT5["connected"] = True
    _mt5_log("Connected ✓")

    tickers      = config["tickers"]
    min_conf     = config["min_conf"]
    account_size = config["account_size"]
    risk_pct     = config["risk_pct"]
    finnhub_key  = config["finnhub_key"]
    interval     = config["interval"]

    sent: set[str] = set()   # dedup: (ticker:direction:date)

    while not stop_event.is_set():
        today = date.today()
        # clear sent signals from prior sessions
        sent = {k for k in sent if k.endswith(str(today))}

        for ticker in tickers:
            if stop_event.is_set():
                break

            _mt5_log(f"Polling {ticker}…")
            try:
                sr = run_signal(
                    ticker=ticker,
                    finnhub_api_key=finnhub_key,
                    account_size=account_size,
                    risk_pct=risk_pct,
                )
            except Exception as exc:
                _mt5_log(f"ERROR {ticker}: {exc}")
                continue

            if sr.error:
                _mt5_log(f"{ticker}: {sr.error}")
                continue

            if sr.direction not in ("LONG", "SHORT"):
                _mt5_log(f"{ticker}: no trade (conf={sr.confidence:.0%})")
                continue

            if sr.confidence < min_conf:
                _mt5_log(
                    f"{ticker}: {sr.direction} {sr.confidence:.0%} "
                    f"< threshold {min_conf:.0%} — skip"
                )
                continue

            key = f"{ticker}:{sr.direction}:{today}"
            if key in sent:
                _mt5_log(f"{ticker}: already in trade today — skip")
                continue

            _mt5_log(
                f"SIGNAL ★ {sr.direction} {ticker} "
                f"conf={sr.confidence:.0%} entry={sr.entry:.4f}"
            )

            if not ensure_connected():
                _mt5_log("MT5 reconnect failed — skipping order")
                continue

            result = send_from_signal_result(sr)
            if result.success:
                sent.add(key)
                _MT5["last_signal"] = {
                    "ticker":     ticker,
                    "direction":  sr.direction,
                    "confidence": sr.confidence,
                    "entry":      result.entry,
                    "sl":         result.sl,
                    "tp":         result.tp,
                    "ticket":     result.ticket,
                    "time":       datetime.now().strftime("%H:%M:%S"),
                }
                _mt5_log(f"Order #{result.ticket} placed ✓")
            else:
                _mt5_log(f"Order rejected: {result.error}")

        try:
            _MT5["positions"] = get_open_positions()
        except Exception:
            pass

        _MT5["last_check"] = datetime.now()
        stop_event.wait(interval)

    disconnect()
    _MT5["connected"] = False
    _MT5["running"]   = False
    _mt5_log("MT5 thread stopped.")


def _start_mt5(config: dict):
    if _MT5["running"]:
        return
    stop_event = threading.Event()
    t = threading.Thread(target=_mt5_loop, args=(stop_event, config), daemon=True)
    _MT5["stop_event"] = stop_event
    _MT5["thread"]     = t
    _MT5["running"]    = True
    _MT5["error"]      = ""
    _MT5["log"]        = []
    t.start()


def _stop_mt5():
    if _MT5["stop_event"]:
        _MT5["stop_event"].set()
    _MT5["running"] = False


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

    # ── MT5 Auto-Trading Panel ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🤖 MT5 Auto-Trading")
    st.caption("Runs locally — MT5 terminal must be open on this machine.")

    mt5_tickers = st.multiselect(
        "Trade these signals",
        options=["NQ=F", "ES=F", "XAUUSD=X"],
        default=st.session_state.get("mt5_tickers", ["NQ=F", "ES=F"]),
    )
    mt5_min_conf = st.slider(
        "Min confidence to enter",
        min_value=0.50, max_value=0.95, step=0.05,
        value=st.session_state.get("mt5_min_conf", 0.65),
    )
    mt5_interval = st.selectbox(
        "Poll interval",
        options=[30, 60, 120, 300],
        index=1,
        format_func=lambda x: f"{x}s" if x < 60 else f"{x//60}m",
    )

    is_running = _MT5["running"]

    if not is_running:
        if st.button("▶ Start Auto-Trading", use_container_width=True, type="primary"):
            _start_mt5({
                "tickers":      mt5_tickers,
                "min_conf":     mt5_min_conf,
                "account_size": account_size,
                "risk_pct":     risk_pct,
                "finnhub_key":  finnhub_key,
                "interval":     mt5_interval,
            })
            st.rerun()
    else:
        if st.button("⏹ Stop Auto-Trading", use_container_width=True):
            _stop_mt5()
            st.rerun()

    # Status indicator
    if is_running:
        conn_icon = "🟢" if _MT5["connected"] else "🟡"
        st.markdown(f"{conn_icon} **Running** — polling every {mt5_interval}s")
        if _MT5["last_check"]:
            st.caption(f"Last poll: {_MT5['last_check'].strftime('%H:%M:%S')}")
    elif _MT5["error"]:
        st.error(_MT5["error"])

    # Emergency close-all
    if is_running and _MT5["connected"]:
        if st.button("🚨 Close All Positions", use_container_width=True):
            from utils.mt5_bridge import close_all_positions
            close_all_positions()
            st.success("All positions closed.")

    st.markdown("---")
    st.markdown("**Instruments**")
    for inst in INSTRUMENTS:
        st.markdown(
            f"{inst['label']} `{inst['ticker']}`  \n"
            f"<span style='font-size:11px;color:#555;'>{inst['name']}</span>",
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.caption("Data: yfinance + Finnhub  \nv2.1 MT5 Integrated")

# Persist settings
st.session_state["finnhub_key"]  = finnhub_key
st.session_state["account_size"] = account_size
st.session_state["risk_pct"]     = risk_pct
st.session_state["auto_refresh"] = auto_refresh
st.session_state["mt5_tickers"]  = mt5_tickers
st.session_state["mt5_min_conf"] = mt5_min_conf

# Auto-refresh trigger
if auto_refresh:
    last = st.session_state.get("last_refresh_ts", 0)
    if time.time() - last > 300:
        st.session_state["last_refresh_ts"] = time.time()
        st.session_state.pop("results", None)

needs_run = run_btn or ("results" not in st.session_state)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# ⚡ Katraswing — 5m Signal Dashboard")

# ── MT5 Live Status Bar (shown when auto-trading is active) ──────────────────
if _MT5["running"] or _MT5["last_signal"]:
    with st.container():
        mt5_cols = st.columns([1, 1, 1])

        with mt5_cols[0]:
            sig = _MT5["last_signal"]
            if sig:
                color = "#00c851" if sig["direction"] == "LONG" else "#ff4444"
                st.markdown(
                    f"<div class='mt5-card'>"
                    f"<b>Last Signal</b><br>"
                    f"<span style='color:{color};font-size:18px;'>{sig['direction']}</span> "
                    f"{sig['ticker']} — <b>{sig['confidence']:.0%}</b> conf<br>"
                    f"<small>Entry {sig['entry']:.2f} | SL {sig['sl']:.2f} | TP {sig['tp']:.2f}</small><br>"
                    f"<small style='color:#888;'>Ticket #{sig['ticket']} @ {sig['time']}</small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='mt5-card'><b>Last Signal</b><br>"
                    "<span style='color:#888;'>Waiting for signal…</span></div>",
                    unsafe_allow_html=True,
                )

        with mt5_cols[1]:
            positions = _MT5["positions"]
            pos_html = "<div class='mt5-card'><b>Open Positions</b><br>"
            if positions:
                for p in positions:
                    pnl_color = "#00c851" if p.profit >= 0 else "#ff4444"
                    dir_color = "#00c851" if p.direction == "LONG" else "#ff4444"
                    pos_html += (
                        f"<div class='mt5-pos'>"
                        f"<span style='color:{dir_color};'>{p.direction}</span> "
                        f"{p.symbol} vol={p.volume} "
                        f"<span style='color:{pnl_color};'>P&L {p.profit:+.2f}</span>"
                        f"</div>"
                    )
            else:
                pos_html += "<span style='color:#888;'>No open positions</span>"
            pos_html += "</div>"
            st.markdown(pos_html, unsafe_allow_html=True)

        with mt5_cols[2]:
            log_lines = _MT5["log"][-8:]
            log_html = "<div class='mt5-card'><b>Activity Log</b><br><small style='color:#888;font-family:monospace;'>"
            log_html += "<br>".join(log_lines) if log_lines else "No activity yet"
            log_html += "</small></div>"
            st.markdown(log_html, unsafe_allow_html=True)

        st.markdown("---")

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
