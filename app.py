"""
KATRASWING — 5m Signal Dashboard
Instruments: NQ Mini, ES Mini

Run locally: streamlit run app.py
MT5 approval-trading requires MetaTrader5 open on the same Windows machine.
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
  .signal-card { border-radius: 12px; padding: 16px 20px; margin-bottom: 10px; }
  .mt5-pos { background:#111827; border-radius:6px; padding:6px 10px; font-size:12px; margin:3px 0; }
  .log-box  { background:#111827; border-radius:8px; padding:10px; font-size:11px;
               font-family:monospace; color:#9ca3af; max-height:160px; overflow-y:auto; }
</style>
""", unsafe_allow_html=True)

# ── Instruments (gold removed — Yahoo Finance unavailable) ────────────────────
INSTRUMENTS = [
    {"ticker": "NQ=F", "label": "🔵 NQ Mini", "name": "Nasdaq 100 E-mini Futures"},
    {"ticker": "ES=F", "label": "🟢 ES Mini", "name": "S&P 500 E-mini Futures"},
]

# ── MT5 shared state (module-level — survives Streamlit reruns) ───────────────
_MT5: dict = {
    "thread":     None,
    "stop_event": None,
    "running":    False,
    "connected":  False,
    "last_check": None,
    "pending":    [],      # signals awaiting your approval — list of dicts
    "sent":       set(),   # "ticker:direction:date" keys already approved+sent
    "rejected":   set(),   # keys you dismissed (skip for today)
    "last_sent":  None,    # info dict of last successfully placed order
    "positions":  [],      # open MT5Position objects
    "log":        [],      # last 20 activity lines
    "error":      "",
}


def _log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    _MT5["log"].append(f"{ts}  {msg}")
    _MT5["log"] = _MT5["log"][-20:]


def _mt5_loop(stop_event: threading.Event, config: dict):
    """
    Background thread: polls signal engine, queues signals for manual approval.
    Does NOT place any order automatically.
    """
    from agents.signal_engine import run_signal
    from utils.mt5_bridge import connect, disconnect, get_open_positions

    _log("Connecting to MT5 terminal…")
    if not connect():
        _MT5["error"] = "Cannot connect to MT5. Is the terminal open and logged in?"
        _log(f"ERROR: {_MT5['error']}")
        _MT5["running"] = False
        return

    _MT5["connected"] = True
    _MT5["error"] = ""
    _log("Connected ✓  —  monitoring for signals…")

    tickers      = config["tickers"]
    min_conf     = config["min_conf"]
    account_size = config["account_size"]
    risk_pct     = config["risk_pct"]
    finnhub_key  = config["finnhub_key"]
    interval     = config["interval"]

    while not stop_event.is_set():
        today = date.today()
        # drop stale sent/rejected from prior sessions
        _MT5["sent"]     = {k for k in _MT5["sent"]     if k.endswith(str(today))}
        _MT5["rejected"]  = {k for k in _MT5["rejected"] if k.endswith(str(today))}

        for ticker in tickers:
            if stop_event.is_set():
                break

            _log(f"Polling {ticker}…")
            try:
                sr = run_signal(
                    ticker=ticker,
                    finnhub_api_key=finnhub_key,
                    account_size=account_size,
                    risk_pct=risk_pct,
                )
            except Exception as exc:
                _log(f"ERROR {ticker}: {exc}")
                continue

            if sr.error:
                _log(f"{ticker}: {sr.error}")
                continue

            if sr.direction not in ("LONG", "SHORT"):
                _log(f"{ticker}: no trade (conf={sr.confidence:.0%})")
                continue

            if sr.confidence < min_conf:
                _log(f"{ticker}: {sr.direction} {sr.confidence:.0%} < {min_conf:.0%} — skip")
                continue

            key = f"{ticker}:{sr.direction}:{today}"

            if key in _MT5["sent"]:
                _log(f"{ticker}: already sent today — skip")
                continue
            if key in _MT5["rejected"]:
                _log(f"{ticker}: rejected by you today — skip")
                continue

            # Check if already in pending queue
            pending_keys = {p["key"] for p in _MT5["pending"]}
            if key in pending_keys:
                _log(f"{ticker}: already awaiting your approval")
                continue

            # Queue for manual approval
            top_patterns = [(p.name, p.win_rate) for p in sr.patterns.patterns[:3]]
            _MT5["pending"].append({
                "sr":          sr,
                "key":         key,
                "ticker":      ticker,
                "direction":   sr.direction,
                "confidence":  sr.confidence,
                "entry":       sr.entry,
                "sl":          sr.sl,
                "tp":          sr.tp,
                "patterns":    top_patterns,
                "news":        sr.news_sentiment,
                "time":        datetime.now().strftime("%H:%M:%S"),
            })
            _log(f"★ AWAITING APPROVAL — {sr.direction} {ticker} {sr.confidence:.0%}")

        try:
            _MT5["positions"] = get_open_positions()
        except Exception:
            pass

        _MT5["last_check"] = datetime.now()
        stop_event.wait(interval)

    disconnect()
    _MT5["connected"] = False
    _MT5["running"]   = False
    _log("MT5 thread stopped.")


def _start_mt5(config: dict):
    if _MT5["running"]:
        return
    stop_event = threading.Event()
    t = threading.Thread(target=_mt5_loop, args=(stop_event, config), daemon=True)
    _MT5.update({"stop_event": stop_event, "thread": t, "running": True,
                 "error": "", "log": [], "pending": []})
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
        "Auto-refresh signals (5 min)",
        value=st.session_state.get("auto_refresh", False),
    )
    run_btn = st.button("🔄 Run All Signals", width="stretch", type="primary")

    # ── MT5 Panel ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🤖 MT5 Signal Monitor")
    st.caption("Finds signals and asks for your approval before placing any order.")

    mt5_tickers = st.multiselect(
        "Monitor these tickers",
        options=["NQ=F", "ES=F"],
        default=st.session_state.get("mt5_tickers", ["NQ=F", "ES=F"]),
    )
    mt5_min_conf = st.slider(
        "Min confidence to alert",
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
        if st.button("▶ Start Monitoring", width="stretch", type="primary"):
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
        if st.button("⏹ Stop Monitoring", width="stretch"):
            _stop_mt5()
            st.rerun()

    if is_running:
        icon = "🟢" if _MT5["connected"] else "🟡"
        st.markdown(f"{icon} **Monitoring** — every {mt5_interval}s")
        if _MT5["last_check"]:
            st.caption(f"Last poll: {_MT5['last_check'].strftime('%H:%M:%S')}")
        pending_count = len(_MT5["pending"])
        if pending_count:
            st.warning(f"⚠️ {pending_count} signal(s) awaiting approval")
    elif _MT5["error"]:
        st.error(_MT5["error"])

    if _MT5["connected"] and _MT5["positions"]:
        if st.button("🚨 Close All Positions", width="stretch"):
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
    st.caption("Data: yfinance + Finnhub  \nv2.2 — Approval Mode")

# Persist settings
st.session_state.update({
    "finnhub_key":  finnhub_key,
    "account_size": account_size,
    "risk_pct":     risk_pct,
    "auto_refresh": auto_refresh,
    "mt5_tickers":  mt5_tickers,
    "mt5_min_conf": mt5_min_conf,
})

# Auto-refresh trigger
if auto_refresh:
    last = st.session_state.get("last_refresh_ts", 0)
    if time.time() - last > 300:
        st.session_state["last_refresh_ts"] = time.time()
        st.session_state.pop("results", None)

needs_run = run_btn or ("results" not in st.session_state)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# ⚡ Katraswing — 5m Signal Dashboard")

# ── Pending approval signals ──────────────────────────────────────────────────
pending = _MT5["pending"]
if pending:
    st.markdown("---")
    st.markdown(f"## 🔔 Trade Signals Awaiting Your Approval  ({len(pending)})")

    to_remove = []   # indices approved or rejected this rerun

    for i, item in enumerate(pending):
        direction = item["direction"]
        ticker    = item["ticker"]
        conf      = item["confidence"]

        bg    = "#0d2b1a" if direction == "LONG" else "#2b0d0d"
        color = "#00c851" if direction == "LONG" else "#ff4444"
        arrow = "▲" if direction == "LONG" else "▼"

        pattern_str = "  |  ".join(
            f"{n} ({w:.0%})" for n, w in item["patterns"]
        ) or "—"

        st.markdown(
            f"<div class='signal-card' style='background:{bg};border:1px solid {color};'>"
            f"<span style='color:{color};font-size:22px;font-weight:700;'>"
            f"{arrow} {direction} &nbsp; {ticker}</span>"
            f"&nbsp;&nbsp;<span style='color:#ccc;font-size:14px;'>confidence <b>{conf:.0%}</b> "
            f"&nbsp;·&nbsp; detected {item['time']}</span><br><br>"
            f"<b>Entry</b> {item['entry']:.2f} &nbsp;&nbsp;"
            f"<b>Stop Loss</b> {item['sl']:.2f} &nbsp;&nbsp;"
            f"<b>Take Profit</b> {item['tp']:.2f}<br>"
            f"<small style='color:#aaa;'>Patterns: {pattern_str}"
            f" &nbsp;·&nbsp; News: {item['news']}</small>"
            f"</div>",
            unsafe_allow_html=True,
        )

        btn_col1, btn_col2, btn_col3 = st.columns([2, 2, 6])

        with btn_col1:
            if st.button(f"✅ Approve & Send", key=f"approve_{item['key']}"):
                from utils.mt5_bridge import ensure_connected, send_from_signal_result
                if ensure_connected():
                    result = send_from_signal_result(item["sr"])
                    if result.success:
                        _MT5["sent"].add(item["key"])
                        _MT5["last_sent"] = {
                            "ticker":     ticker,
                            "direction":  direction,
                            "confidence": conf,
                            "entry":      result.entry,
                            "sl":         result.sl,
                            "tp":         result.tp,
                            "ticket":     result.ticket,
                            "time":       datetime.now().strftime("%H:%M:%S"),
                        }
                        _log(f"Order #{result.ticket} placed ✓ (approved by user)")
                        st.success(f"Order #{result.ticket} placed in MT5!")
                    else:
                        st.error(f"MT5 rejected: {result.error}")
                else:
                    st.error("MT5 not connected.")
                to_remove.append(i)

        with btn_col2:
            if st.button(f"❌ Reject", key=f"reject_{item['key']}"):
                _MT5["rejected"].add(item["key"])
                _log(f"Signal rejected by user: {direction} {ticker}")
                to_remove.append(i)

        st.markdown("")  # spacing

    # Remove processed items (reverse order to keep indices valid)
    for i in sorted(set(to_remove), reverse=True):
        if i < len(_MT5["pending"]):
            _MT5["pending"].pop(i)

    if to_remove:
        st.rerun()

    st.markdown("---")

# ── Last sent order + open positions ─────────────────────────────────────────
if _MT5["last_sent"] or _MT5["positions"]:
    cols = st.columns([1, 1])

    with cols[0]:
        sig = _MT5["last_sent"]
        if sig:
            color = "#00c851" if sig["direction"] == "LONG" else "#ff4444"
            st.markdown(
                f"<div style='background:#1e2130;border-radius:10px;padding:12px 16px;'>"
                f"<b>Last Approved Order</b><br>"
                f"<span style='color:{color};font-size:18px;'>{sig['direction']}</span> "
                f"{sig['ticker']} — <b>{sig['confidence']:.0%}</b> conf<br>"
                f"<small>Entry {sig['entry']:.2f} | SL {sig['sl']:.2f} | TP {sig['tp']:.2f}</small><br>"
                f"<small style='color:#888;'>Ticket #{sig['ticket']} @ {sig['time']}</small>"
                f"</div>",
                unsafe_allow_html=True,
            )

    with cols[1]:
        positions = _MT5["positions"]
        if positions:
            html = "<div style='background:#1e2130;border-radius:10px;padding:12px 16px;'><b>Open Positions</b><br>"
            for p in positions:
                pnl_color = "#00c851" if p.profit >= 0 else "#ff4444"
                dir_color = "#00c851" if p.direction == "LONG" else "#ff4444"
                html += (
                    f"<div class='mt5-pos'>"
                    f"<span style='color:{dir_color};'>{p.direction}</span> "
                    f"{p.symbol} vol={p.volume} "
                    f"<span style='color:{pnl_color};'>P&L {p.profit:+.2f}</span></div>"
                )
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)

    st.markdown("---")

# ── Activity log (compact) ────────────────────────────────────────────────────
if _MT5["running"] and _MT5["log"]:
    with st.expander("📋 MT5 Activity Log", expanded=False):
        st.markdown(
            "<div class='log-box'>" +
            "<br>".join(_MT5["log"][-15:]) +
            "</div>",
            unsafe_allow_html=True,
        )

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

# ── Instrument Tabs ───────────────────────────────────────────────────────────
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
