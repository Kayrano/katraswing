"""
KATRASWING — Trading Signal Dashboard
Instruments: NQ Mini, ES Mini

Run locally: streamlit run app.py
MT5 approval-trading requires MetaTrader5 open on the same Windows machine.
"""

import threading
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import streamlit as st

_ET = ZoneInfo("America/New_York")

# Markets that follow regular stock exchange hours (NYSE/NASDAQ)
_STOCK_TICKERS = {"AAPL", "MSFT", "AMZN"}

# Futures that trade nearly 24/7 on CME Globex
_FUTURES_TICKERS = {"NQ=F", "ES=F", "NKD=F"}


def _market_status(ticker: str, df) -> tuple[bool, str]:
    """
    Returns (is_closed, status_label).
    Uses the last bar timestamp as primary signal; time-based rules as fallback.
    """
    # Primary: check data staleness — most reliable cross-instrument signal
    if df is not None and not df.empty:
        last_ts = df.index[-1]
        try:
            if last_ts.tzinfo is None:
                last_ts = last_ts.tz_localize("UTC")
            age_min = (datetime.now(timezone.utc) - last_ts).total_seconds() / 60
            # Stocks: stale after 20 min; futures: stale after 10 min (high liquidity)
            threshold = 20 if ticker in _STOCK_TICKERS else 10
            if age_min > threshold:
                return True, f"Market closed  ·  last bar {int(age_min)}m ago"
        except Exception:
            pass

    # Fallback: explicit time rules
    now = datetime.now(_ET)
    wd  = now.weekday()   # 0 Mon … 6 Sun
    hm  = now.hour * 60 + now.minute

    if ticker in _STOCK_TICKERS:
        if wd >= 5:
            return True, "Market closed  ·  weekend"
        if hm < 9 * 60 + 30:
            opens_in = 9 * 60 + 30 - hm
            return True, f"Pre-market  ·  opens in {opens_in}m (09:30 ET)"
        if hm >= 16 * 60:
            return True, "After hours  ·  closed at 16:00 ET"

    elif ticker in _FUTURES_TICKERS:
        # CME Globex closed: Fri 17:00 → Sun 18:00 ET; daily break 16:00-17:00 ET
        if wd == 5:  # Saturday
            return True, "Market closed  ·  CME weekend break"
        if wd == 6 and hm < 18 * 60:  # Sunday before 18:00 ET
            opens_in = 18 * 60 - hm
            return True, f"Market closed  ·  CME opens Sun 18:00 ET (in {opens_in}m)"
        if wd == 4 and hm >= 17 * 60:  # Friday after 17:00
            return True, "Market closed  ·  CME weekend break starts"
        if 16 * 60 <= hm < 17 * 60:
            return True, "Daily maintenance  ·  CME break 16:00-17:00 ET"

    return False, ""

st.set_page_config(
    page_title="Katraswing",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  body, .stApp { background-color: #0b0e17; color: #e0e0e0; }
  section[data-testid="stSidebar"] { background: #0e1117; border-right: 1px solid #1e2330; }
  .stTabs [data-baseweb="tab-list"] { gap: 4px; background: transparent; border-bottom: 1px solid #1e2330; }
  .stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 6px 6px 0 0;
    color: #6b7280; padding: 10px 24px; font-weight: 600; font-size: 13px;
  }
  .stTabs [aria-selected="true"] { background: #1e2330; color: #ffffff; }
  hr { border-color: #1e2330; }
  .stButton > button {
    background: #1a1f2e; color: #c9d1d9;
    border: 1px solid #2a3044; border-radius: 6px;
    font-size: 13px; padding: 6px 16px;
  }
  .stButton > button:hover { border-color: #3b82f6; color: #3b82f6; }
  .log-box { background:#0d1117; border-radius:6px; padding:10px 14px; font-size:11px;
              font-family:monospace; color:#6b7280; max-height:200px; overflow-y:auto;
              border: 1px solid #1e2330; }
  .sig-card { border-radius:10px; padding:20px 24px; margin-bottom:8px; }
  .tbl { width:100%; border-collapse:collapse; font-size:13px; }
  .tbl td { padding: 5px 8px; }
  .tbl td:last-child { text-align:right; font-weight:600; color:#e0e0e0; }
  .tbl td:first-child { color:#6b7280; }
  .ind-box { background:#111827; border-radius:8px; padding:12px; text-align:center; }
  .pos-row { background:#111827; border-radius:6px; padding:8px 14px; margin:4px 0;
             font-size:13px; display:flex; justify-content:space-between; }
</style>
""", unsafe_allow_html=True)

INSTRUMENTS = [
    {"ticker": "NQ=F",  "label": "NQ Mini",   "name": "Nasdaq 100 E-mini Futures"},
    {"ticker": "ES=F",  "label": "ES Mini",   "name": "S&P 500 E-mini Futures"},
    {"ticker": "NKD=F", "label": "Nikkei 225","name": "Nikkei 225 Futures"},
    {"ticker": "AAPL",  "label": "Apple",     "name": "Apple Inc."},
    {"ticker": "MSFT",  "label": "Microsoft", "name": "Microsoft Corp."},
    {"ticker": "AMZN",  "label": "Amazon",    "name": "Amazon.com Inc."},
]

# ── MT5 shared state ──────────────────────────────────────────────────────────
_MT5: dict = {
    "thread":     None,
    "stop_event": None,
    "running":    False,
    "connected":  False,
    "last_check": None,
    "pending":    [],
    "sent":       set(),
    "rejected":   set(),
    "last_sent":  None,
    "positions":  [],
    "log":        [],
    "error":      "",
}


def _log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    _MT5["log"].append(f"{ts}  {msg}")
    _MT5["log"] = _MT5["log"][-30:]


def _mt5_loop(stop_event: threading.Event, config: dict):
    try:
        _mt5_loop_inner(stop_event, config)
    except Exception as exc:
        import traceback
        _MT5["error"] = f"Thread crashed: {exc}\n{traceback.format_exc()}"
        _log(f"CRASH: {exc}")
        _MT5["connected"] = False
        _MT5["running"]   = False


def _mt5_loop_inner(stop_event: threading.Event, config: dict):
    from agents.signal_engine import run_signal
    from utils.mt5_bridge import connect, disconnect, get_open_positions, is_available

    if not is_available():
        _MT5["error"] = "MetaTrader5 package not installed. Run: python -m pip install MetaTrader5"
        _MT5["running"] = False
        return

    _log("Connecting to MT5 terminal…")
    try:
        ok = connect()
    except Exception as exc:
        _MT5["error"] = f"connect() raised: {exc}"
        _MT5["running"] = False
        return

    if not ok:
        _MT5["error"] = "Cannot connect to MT5. Is the terminal open and logged in?"
        _MT5["running"] = False
        return

    _MT5["connected"] = True
    _MT5["error"] = ""
    _log("Connected — monitoring for signals…")

    tickers      = config["tickers"]
    min_conf     = config["min_conf"]
    account_size = config["account_size"]
    risk_pct     = config["risk_pct"]
    finnhub_key  = config["finnhub_key"]
    interval     = config["interval"]

    while not stop_event.is_set():
        today = date.today()
        _MT5["sent"]     = {k for k in _MT5["sent"]     if k.endswith(str(today))}
        _MT5["rejected"] = {k for k in _MT5["rejected"] if k.endswith(str(today))}

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
                _log(f"{ticker}: no trade signal (conf={sr.confidence:.0%})")
                continue
            if sr.confidence < min_conf:
                _log(f"{ticker}: {sr.direction} {sr.confidence:.0%} below threshold")
                continue

            key = f"{ticker}:{sr.direction}:{today}"
            if key in _MT5["sent"] or key in _MT5["rejected"]:
                continue
            if key in {p["key"] for p in _MT5["pending"]}:
                continue

            top_patterns = [(p.name, p.win_rate) for p in sr.patterns.patterns[:3]]
            _MT5["pending"].append({
                "sr":         sr,
                "key":        key,
                "ticker":     ticker,
                "direction":  sr.direction,
                "confidence": sr.confidence,
                "entry":      sr.entry,
                "sl":         sr.sl,
                "tp":         sr.tp,
                "atr":        sr.atr,
                "patterns":   top_patterns,
                "news":       sr.news_sentiment,
                "indicators": sr.indicators,
                "time":       datetime.now().strftime("%H:%M:%S"),
            })
            _log(f"★ APPROVAL NEEDED — {sr.direction} {ticker} {sr.confidence:.0%}")

        try:
            _MT5["positions"] = get_open_positions()
        except Exception:
            pass

        _MT5["last_check"] = datetime.now()
        stop_event.wait(interval)

    disconnect()
    _MT5["connected"] = False
    _MT5["running"]   = False
    _log("Monitoring stopped.")


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


def _fetch_mt5_history(days: int = 30):
    try:
        import MetaTrader5 as mt5  # type: ignore[import]
        from_dt = datetime.now() - timedelta(days=days)
        deals = mt5.history_deals_get(from_dt, datetime.now())
        if deals is None:
            return []
        rows = []
        for d in deals:
            if d.entry == 0:  # entry deal (open)
                continue
            rows.append({
                "ticket":  d.deal,
                "symbol":  d.symbol,
                "type":    "BUY" if d.type == 0 else "SELL",
                "volume":  d.volume,
                "price":   d.price,
                "profit":  d.profit,
                "time":    datetime.fromtimestamp(d.time).strftime("%Y-%m-%d %H:%M"),
                "comment": d.comment,
            })
        return sorted(rows, key=lambda x: x["time"], reverse=True)
    except Exception:
        return []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Katraswing")
    st.caption("Signal-first trading dashboard")
    st.markdown("---")

    # Settings
    finnhub_key = st.text_input(
        "Finnhub API Key",
        value=st.session_state.get("finnhub_key", "d7j16r1r01qn2qavovt0d7j16r1r01qn2qavovtg"),
        type="password",
    ).strip()
    account_size = st.number_input(
        "Account ($)",
        value=st.session_state.get("account_size", 100_000),
        min_value=1_000, step=5_000,
    )
    risk_pct = st.slider(
        "Risk per trade (%)",
        min_value=0.25, max_value=3.0, step=0.25,
        value=st.session_state.get("risk_pct", 1.0),
    )
    st.markdown("---")

    # Signal scan controls
    run_btn = st.button("🔄 Scan Signals Now", use_container_width=True, type="primary")
    auto_refresh = st.checkbox(
        "Auto-scan every 5 min",
        value=st.session_state.get("auto_refresh", False),
    )
    st.markdown("---")

    # MT5 monitor
    st.markdown("### MT5 Monitor")
    mt5_tickers = st.multiselect(
        "Watch",
        options=["NQ=F", "ES=F", "NKD=F", "AAPL", "MSFT", "AMZN"],
        default=st.session_state.get("mt5_tickers", ["NQ=F", "ES=F"]),
    )
    mt5_min_conf = st.slider(
        "Min confidence",
        min_value=0.50, max_value=0.95, step=0.05,
        value=st.session_state.get("mt5_min_conf", 0.65),
    )
    mt5_interval = st.selectbox(
        "Poll interval",
        options=[30, 60, 120, 300],
        index=1,
        format_func=lambda x: f"{x}s" if x < 60 else f"{x//60}m",
    )

    from utils.mt5_bridge import is_available as _mt5_pkg_ok
    if not _mt5_pkg_ok():
        st.warning("MetaTrader5 not installed.\nRun: `python -m pip install MetaTrader5`")

    is_running = _MT5["running"]
    if not is_running:
        if st.button("▶ Start Monitoring", use_container_width=True, type="primary"):
            st.session_state["_mt5_action"] = {
                "action": "start",
                "cfg": {
                    "tickers":      mt5_tickers,
                    "min_conf":     mt5_min_conf,
                    "account_size": account_size,
                    "risk_pct":     risk_pct,
                    "finnhub_key":  finnhub_key,
                    "interval":     mt5_interval,
                },
            }
    else:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⏹ Stop", use_container_width=True):
                st.session_state["_mt5_action"] = {"action": "stop"}
        with col2:
            if _MT5["connected"] and st.button("🚨 Close All", use_container_width=True):
                from utils.mt5_bridge import close_all_positions
                close_all_positions()

    # Status
    if is_running:
        if _MT5["connected"]:
            st.markdown("🟢 **Connected**")
            if _MT5["last_check"]:
                st.caption(f"Last poll: {_MT5['last_check'].strftime('%H:%M:%S')}")
        else:
            st.markdown("🟡 **Connecting…**")
    if _MT5["error"]:
        st.error(_MT5["error"])

    # Activity log
    if _MT5["log"]:
        with st.expander("Activity log", expanded=False):
            st.markdown(
                "<div class='log-box'>" + "<br>".join(_MT5["log"][-20:]) + "</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.caption("Data: yfinance + Finnhub  |  v3.0")

# ── Persist settings ──────────────────────────────────────────────────────────
st.session_state.update({
    "finnhub_key":  finnhub_key,
    "account_size": account_size,
    "risk_pct":     risk_pct,
    "auto_refresh": auto_refresh,
    "mt5_tickers":  mt5_tickers,
    "mt5_min_conf": mt5_min_conf,
})

# ── MT5 action handler (outside sidebar) ─────────────────────────────────────
_mt5_action = st.session_state.pop("_mt5_action", None)
if _mt5_action:
    if _mt5_action["action"] == "start":
        _start_mt5(_mt5_action["cfg"])
    else:
        _stop_mt5()
    st.rerun()

# ── Auto-refresh trigger ──────────────────────────────────────────────────────
if auto_refresh:
    last = st.session_state.get("last_refresh_ts", 0)
    if time.time() - last > 300:
        st.session_state.pop("results", None)

needs_run = run_btn or ("results" not in st.session_state)

# ── Fetch signals ─────────────────────────────────────────────────────────────
if needs_run:
    from agents.signal_engine import run_signal
    results = {}
    with st.spinner("Scanning signals…"):
        for inst in INSTRUMENTS:
            results[inst["ticker"]] = run_signal(
                ticker=inst["ticker"],
                display_name=inst["name"],
                finnhub_api_key=finnhub_key,
                account_size=account_size,
                risk_pct=risk_pct,
            )
    st.session_state["results"] = results
    st.session_state["last_refresh_ts"] = time.time()
else:
    results = st.session_state["results"]


# ── Helper: render one signal card ───────────────────────────────────────────
def _signal_card(result, pending_keys: set):
    ticker = result.ticker
    direction = result.direction
    conf = int(result.confidence * 100)

    # Market closed check
    closed, closed_label = _market_status(ticker, result.df_5m)
    if closed:
        st.markdown(
            f"<div style='background:#111827;border:1px solid #374151;border-radius:10px;"
            f"padding:20px 24px;text-align:center;'>"
            f"<div style='font-size:15px;font-weight:700;color:#6b7280;'>⏸ MARKET CLOSED</div>"
            f"<div style='font-size:12px;color:#4b5563;margin-top:6px;'>{closed_label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        return

    if direction == "LONG":
        bg, border, color, arrow = "#0a1f14", "#16a34a", "#22c55e", "▲ LONG"
    elif direction == "SHORT":
        bg, border, color, arrow = "#1f0a0a", "#dc2626", "#ef4444", "▼ SHORT"
    else:
        bg, border, color, arrow = "#111827", "#374151", "#6b7280", "— FLAT"

    is_pending = ticker in pending_keys
    pending_badge = " &nbsp;<span style='background:#f59e0b;color:#000;border-radius:4px;padding:1px 6px;font-size:11px;'>AWAITING APPROVAL</span>" if is_pending else ""

    # Direction + confidence
    st.markdown(
        f"<div class='sig-card' style='background:{bg};border:1px solid {border};'>"
        f"<div style='font-size:26px;font-weight:800;color:{color};'>"
        f"{arrow}{pending_badge}</div>"
        f"<div style='margin:10px 0 4px;background:#1a1a2e;border-radius:4px;height:8px;'>"
        f"<div style='background:{color};width:{conf}%;height:100%;border-radius:4px;'></div></div>"
        f"<div style='font-size:13px;color:#9ca3af;margin-bottom:14px;'>"
        f"Confidence: <b style='color:{color};'>{conf}%</b>"
        f"&nbsp;·&nbsp;Base {int(result.base_confidence*100)}%"
        f"&nbsp;·&nbsp;News boost {result.news_boost:+.0%}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if direction in ("LONG", "SHORT") and result.entry > 0:
        risk   = abs(result.entry - result.sl)
        reward = abs(result.tp - result.entry)
        rr     = reward / risk if risk > 0 else 0

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"<table class='tbl'>"
                f"<tr><td>Entry</td><td style='color:#60a5fa;'>{result.entry:.2f}</td></tr>"
                f"<tr><td>Stop Loss</td><td style='color:#ef4444;'>{result.sl:.2f}</td></tr>"
                f"<tr><td>Take Profit</td><td style='color:#22c55e;'>{result.tp:.2f}</td></tr>"
                f"<tr><td>Risk / Reward</td><td>{risk:.2f} / {reward:.2f}</td></tr>"
                f"<tr><td>R:R Ratio</td><td style='color:{'#22c55e' if rr >= 2 else '#f59e0b'};'>"
                f"1 : {rr:.1f}</td></tr>"
                f"<tr><td>ATR</td><td>{result.atr:.2f}</td></tr>"
                f"</table>",
                unsafe_allow_html=True,
            )

        with c2:
            ind = result.indicators
            if ind:
                rsi_color = "#ef4444" if (ind.rsi or 50) > 70 else "#22c55e" if (ind.rsi or 50) < 30 else "#9ca3af"
                macd_color = "#22c55e" if (ind.macd_histogram or 0) > 0 else "#ef4444"
                st.markdown(
                    f"<table class='tbl'>"
                    f"<tr><td>RSI (14)</td><td style='color:{rsi_color};'>"
                    f"{f'{ind.rsi:.1f}' if ind.rsi else '—'}</td></tr>"
                    f"<tr><td>MACD Hist</td><td style='color:{macd_color};'>"
                    f"{f'{ind.macd_histogram:+.3f}' if ind.macd_histogram else '—'}</td></tr>"
                    f"<tr><td>ATR</td><td>{f'{ind.atr:.2f}' if ind.atr else '—'}</td></tr>"
                    f"<tr><td>BB Squeeze</td><td style='color:{'#f59e0b' if getattr(ind,'bb_squeeze',False) else '#6b7280'};'>"
                    f"{'YES' if getattr(ind,'bb_squeeze',False) else 'NO'}</td></tr>"
                    f"</table>",
                    unsafe_allow_html=True,
                )

    # Patterns
    if result.patterns and result.patterns.patterns:
        bias_c = {"BULLISH": "#22c55e", "BEARISH": "#ef4444", "NEUTRAL": "#6b7280"}
        c = bias_c.get(result.patterns.dominant_bias, "#6b7280")
        parts = []
        for p in result.patterns.patterns[:5]:
            pc = bias_c.get(p.bias, "#6b7280")
            parts.append(
                f"<span style='background:#1a1f2e;border:1px solid {pc}33;"
                f"border-radius:4px;padding:2px 8px;font-size:12px;color:{pc};'>"
                f"{p.name} <b>{p.win_rate:.0%}</b></span>"
            )
        st.markdown(
            f"<div style='margin-top:12px;'>"
            f"<span style='color:#6b7280;font-size:12px;'>Patterns — bias: </span>"
            f"<span style='color:{c};font-size:12px;font-weight:700;'>{result.patterns.dominant_bias}</span>"
            f"<br><div style='margin-top:6px;display:flex;gap:6px;flex-wrap:wrap;'>"
            + "".join(parts) +
            f"</div></div>",
            unsafe_allow_html=True,
        )

    # Active strategies
    if result.chart_signals:
        strat_lines = []
        for s in result.chart_signals[:4]:
            sc = "#22c55e" if s.signal == "LONG" else "#ef4444"
            strat_lines.append(
                f"<tr><td style='color:{sc};font-weight:600;padding:3px 6px;'>{s.strategy}</td>"
                f"<td style='color:#9ca3af;padding:3px 6px;font-size:12px;'>{s.reason[:70]}</td>"
                f"<td style='color:#6b7280;padding:3px 6px;font-size:12px;'>{int(s.confidence*100)}%</td></tr>"
            )
        st.markdown(
            f"<div style='margin-top:12px;'>"
            f"<span style='color:#6b7280;font-size:12px;'>Active strategies</span>"
            f"<table style='width:100%;margin-top:4px;border-collapse:collapse;'>"
            + "".join(strat_lines) +
            f"</table></div>",
            unsafe_allow_html=True,
        )

    ts = st.session_state.get("last_refresh_ts")
    if ts:
        st.caption(f"Scanned {datetime.fromtimestamp(ts).strftime('%H:%M:%S')}")


# ── Header ────────────────────────────────────────────────────────────────────
col_h, col_status = st.columns([3, 1])
with col_h:
    st.markdown("# ⚡ Katraswing")
with col_status:
    if _MT5["running"] and _MT5["connected"]:
        lc = _MT5["last_check"].strftime("%H:%M:%S") if _MT5["last_check"] else "—"
        st.markdown(
            f"<div style='text-align:right;padding-top:18px;'>"
            f"<span style='color:#22c55e;font-weight:700;'>● LIVE</span>"
            f"<span style='color:#6b7280;font-size:11px;'> last poll {lc}</span></div>",
            unsafe_allow_html=True,
        )
    elif _MT5["running"]:
        st.markdown(
            "<div style='text-align:right;padding-top:18px;'>"
            "<span style='color:#f59e0b;font-weight:700;'>● CONNECTING</span></div>",
            unsafe_allow_html=True,
        )
    elif _MT5["error"]:
        st.markdown(
            "<div style='text-align:right;padding-top:18px;'>"
            "<span style='color:#ef4444;font-weight:700;'>● MT5 ERROR</span></div>",
            unsafe_allow_html=True,
        )

# ── Pending approvals (always at top) ────────────────────────────────────────
pending = _MT5["pending"]
if pending:
    st.markdown("---")
    for item in pending:
        direction = item["direction"]
        ticker    = item["ticker"]
        conf      = item["confidence"]
        color     = "#22c55e" if direction == "LONG" else "#ef4444"
        arrow     = "▲" if direction == "LONG" else "▼"
        bg        = "#0a1f14" if direction == "LONG" else "#1f0a0a"
        border    = "#16a34a" if direction == "LONG" else "#dc2626"
        risk      = abs(item["entry"] - item["sl"])
        reward    = abs(item["tp"] - item["entry"])

        pattern_tags = "  ".join(
            f"<b>{n}</b> ({w:.0%})" for n, w in item["patterns"]
        ) or "—"

        st.markdown(
            f"<div style='background:{bg};border:2px solid {border};"
            f"border-radius:10px;padding:16px 20px;margin-bottom:6px;'>"
            f"<div style='font-size:20px;font-weight:800;color:{color};'>"
            f"🔔 {arrow} {direction} &nbsp; {ticker} &nbsp;"
            f"<span style='font-size:14px;font-weight:400;color:#9ca3af;'>"
            f"conf {conf:.0%} &nbsp;·&nbsp; {item['time']}</span></div>"
            f"<div style='margin-top:8px;font-size:13px;color:#e0e0e0;'>"
            f"Entry <b style='color:#60a5fa;'>{item['entry']:.2f}</b> &nbsp;&nbsp;"
            f"SL <b style='color:#ef4444;'>{item['sl']:.2f}</b> &nbsp;&nbsp;"
            f"TP <b style='color:#22c55e;'>{item['tp']:.2f}</b> &nbsp;&nbsp;"
            f"R:R <b>1:{reward/risk:.1f}</b></div>"
            f"<div style='margin-top:6px;font-size:12px;color:#6b7280;'>"
            f"Patterns: {pattern_tags} &nbsp;·&nbsp; News: {item['news']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        a_col, r_col, _ = st.columns([1, 1, 4])
        with a_col:
            if st.button("✅ Approve & Send", key=f"approve_{item['key']}", type="primary"):
                from utils.mt5_bridge import ensure_connected, send_from_signal_result
                if ensure_connected():
                    res = send_from_signal_result(item["sr"])
                    if res.success:
                        _MT5["sent"].add(item["key"])
                        _MT5["last_sent"] = {**item, "ticket": res.ticket}
                        _log(f"Order #{res.ticket} placed ✓")
                        st.success(f"Order #{res.ticket} sent to MT5!")
                        _MT5["pending"].remove(item)
                        st.rerun()
                    else:
                        st.error(f"Rejected by MT5: {res.error}")
                else:
                    st.error("MT5 not connected.")
        with r_col:
            if st.button("❌ Reject", key=f"reject_{item['key']}"):
                _MT5["rejected"].add(item["key"])
                _log(f"Rejected: {direction} {ticker}")
                _MT5["pending"].remove(item)
                st.rerun()

    st.markdown("---")

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_signals, tab_trades, tab_history, tab_journal = st.tabs([
    "📊  Signals",
    "📈  Open Trades",
    "🕐  Past Trades",
    "📓  Journal",
])

# ── Tab 1: Signals ────────────────────────────────────────────────────────────
with tab_signals:
    if _MT5["error"] and not _MT5["running"]:
        st.error(_MT5["error"])

    pending_tickers = {p["ticker"] for p in _MT5["pending"]}
    cols = st.columns(len(INSTRUMENTS))

    for col, inst in zip(cols, INSTRUMENTS):
        with col:
            result = results.get(inst["ticker"])
            closed, _ = _market_status(inst["ticker"], result.df_5m if result else None)
            status_badge = (
                "<span style='background:#1f2937;color:#6b7280;border-radius:4px;"
                "padding:1px 7px;font-size:10px;font-weight:600;margin-left:6px;'>CLOSED</span>"
                if closed else
                "<span style='background:#052e16;color:#22c55e;border-radius:4px;"
                "padding:1px 7px;font-size:10px;font-weight:600;margin-left:6px;'>LIVE</span>"
            )
            st.markdown(
                f"<div style='font-size:15px;font-weight:700;color:#e0e0e0;"
                f"margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid #1e2330;'>"
                f"{inst['label']}{status_badge}"
                f"&nbsp;<span style='color:#6b7280;font-size:12px;'>{inst['ticker']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if result is None:
                st.caption("Not loaded — click Scan Signals Now")
            elif result.error:
                st.error(result.error)
            else:
                _signal_card(result, pending_tickers)

# ── Tab 2: Open Trades ────────────────────────────────────────────────────────
with tab_trades:
    positions = _MT5["positions"]

    if not _MT5["connected"]:
        st.info("Start MT5 monitoring from the sidebar to see live positions.")
    elif not positions:
        st.markdown(
            "<div style='text-align:center;color:#6b7280;padding:60px 0;font-size:15px;'>"
            "No open positions</div>",
            unsafe_allow_html=True,
        )
    else:
        total_pnl = sum(p.profit for p in positions)
        pnl_color = "#22c55e" if total_pnl >= 0 else "#ef4444"
        st.markdown(
            f"<div style='font-size:13px;color:#6b7280;margin-bottom:16px;'>"
            f"{len(positions)} open position{'s' if len(positions)!=1 else ''} &nbsp;·&nbsp; "
            f"Total P&L: <b style='color:{pnl_color};'>{total_pnl:+.2f}</b></div>",
            unsafe_allow_html=True,
        )
        for p in positions:
            dir_c = "#22c55e" if p.direction == "LONG" else "#ef4444"
            pnl_c = "#22c55e" if p.profit >= 0 else "#ef4444"
            arrow = "▲" if p.direction == "LONG" else "▼"
            st.markdown(
                f"<div style='background:#111827;border-radius:8px;padding:14px 18px;"
                f"margin-bottom:8px;border:1px solid #1e2330;'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                f"<div>"
                f"<span style='color:{dir_c};font-size:16px;font-weight:700;'>{arrow} {p.direction}</span>"
                f"&nbsp; <span style='color:#e0e0e0;font-size:15px;font-weight:600;'>{p.symbol}</span>"
                f"&nbsp; <span style='color:#6b7280;font-size:13px;'>vol {p.volume}</span>"
                f"</div>"
                f"<div style='text-align:right;'>"
                f"<span style='color:{pnl_c};font-size:18px;font-weight:700;'>{p.profit:+.2f}</span>"
                f"</div></div>"
                f"<div style='margin-top:8px;font-size:12px;color:#6b7280;'>"
                f"Entry {p.open_price:.2f} &nbsp;·&nbsp; SL {p.sl:.2f} &nbsp;·&nbsp; TP {p.tp:.2f}"
                f" &nbsp;·&nbsp; Ticket #{p.ticket}"
                f"</div></div>",
                unsafe_allow_html=True,
            )
        if st.button("🚨 Close All Positions", type="primary"):
            from utils.mt5_bridge import close_all_positions
            close_all_positions()
            st.success("All positions closed.")
            st.rerun()

# ── Tab 3: Past Trades ────────────────────────────────────────────────────────
with tab_history:
    days_back = st.selectbox("Show last", [7, 14, 30, 60, 90], index=2,
                             format_func=lambda x: f"{x} days")

    if st.button("Load History", type="primary"):
        st.session_state["trade_history"] = _fetch_mt5_history(days_back)

    history = st.session_state.get("trade_history")

    if history is None:
        st.info("Click **Load History** to fetch closed trades from MT5.")
    elif not history:
        st.caption("No closed trades found in this period.")
    else:
        wins   = [t for t in history if t["profit"] > 0]
        losses = [t for t in history if t["profit"] <= 0]
        total  = sum(t["profit"] for t in history)
        wr     = len(wins) / len(history) * 100 if history else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Trades", len(history))
        m2.metric("Win Rate",     f"{wr:.1f}%")
        m3.metric("Total P&L",    f"{total:+.2f}")
        m4.metric("Avg P&L",      f"{total/len(history):+.2f}" if history else "—")

        st.markdown("---")
        for t in history[:50]:
            pnl_c = "#22c55e" if t["profit"] > 0 else "#ef4444"
            type_c = "#22c55e" if t["type"] == "BUY" else "#ef4444"
            st.markdown(
                f"<div style='background:#111827;border-radius:6px;padding:10px 16px;"
                f"margin-bottom:4px;border:1px solid #1e2330;"
                f"display:flex;justify-content:space-between;'>"
                f"<span style='color:{type_c};font-weight:600;'>{t['type']}</span>"
                f"&nbsp; <span style='color:#e0e0e0;'>{t['symbol']}</span>"
                f"&nbsp; <span style='color:#6b7280;font-size:12px;'>vol {t['volume']} @ {t['price']:.2f}</span>"
                f"&nbsp; <span style='color:#6b7280;font-size:12px;'>{t['time']}</span>"
                f"&nbsp; <span style='color:{pnl_c};font-weight:700;'>{t['profit']:+.2f}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

# ── Tab 4: Journal ────────────────────────────────────────────────────────────
with tab_journal:
    import pathlib
    import streamlit.components.v1 as components
    _journal_path = pathlib.Path(__file__).parent / "static" / "trading-journal.html"
    components.html(_journal_path.read_text(encoding="utf-8"), height=4800, scrolling=True)

# ── Auto-refresh while monitoring ─────────────────────────────────────────────
if _MT5["running"]:
    time.sleep(1)
    st.rerun()
