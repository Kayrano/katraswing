"""
KATRASWING — MT5 Signal Dashboard
Run: streamlit run app.py
Requires MetaTrader5 terminal open on the same Windows machine.
"""

import threading
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Katraswing", page_icon="⚡", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
  body, .stApp { background-color: #0b0e17; color: #e0e0e0; }
  section[data-testid="stSidebar"] { display: none; }
  .stTabs [data-baseweb="tab-list"] { gap: 4px; background: transparent; border-bottom: 1px solid #1e2330; }
  .stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 6px 6px 0 0;
    color: #6b7280; padding: 10px 20px; font-weight: 600; font-size: 13px;
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
              font-family:monospace; color:#6b7280; max-height:160px; overflow-y:auto;
              border: 1px solid #1e2330; }
</style>
""", unsafe_allow_html=True)

# ── Backtest cache ────────────────────────────────────────────────────────────
_BT_CACHE: dict = {"rates": {}, "ts": {}, "running": set()}
_BT_LOCK = threading.Lock()

_ET  = ZoneInfo("America/New_York")
_JST = ZoneInfo("Asia/Tokyo")


def _bt_background(ticker: str) -> None:
    try:
        from agents.intraday_backtester import run_intraday_backtest
        summary = run_intraday_backtest(ticker, timeframe="5m")
        rates = {r.strategy: r.win_rate for r in summary.results if r.total_trades >= 5}
        with _BT_LOCK:
            _BT_CACHE["rates"][ticker] = rates or None
            _BT_CACHE["ts"][ticker]    = time.time()
    except Exception:
        with _BT_LOCK:
            _BT_CACHE["rates"][ticker] = None
            _BT_CACHE["ts"][ticker]    = time.time()
    finally:
        with _BT_LOCK:
            _BT_CACHE["running"].discard(ticker)


# ── MT5 shared state (persists across reruns via session_state) ───────────────
if "_MT5" not in st.session_state:
    st.session_state["_MT5"] = {
        "thread": None, "stop_event": None,
        "running": False, "connected": False,
        "last_check": None, "pending": [],
        "sent": set(), "rejected": set(),
        "last_sent": None, "positions": [],
        "log": [], "error": "",
        "live_win_rates": {},
    }
_MT5: dict = st.session_state["_MT5"]


def _log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    _MT5["log"].append(f"{ts}  {msg}")
    _MT5["log"] = _MT5["log"][-30:]


# ── MT5 background monitoring loop ───────────────────────────────────────────

def _mt5_loop(stop_event, config):
    try:
        _mt5_loop_inner(stop_event, config)
    except Exception as exc:
        import traceback
        _MT5["error"] = f"Thread crashed: {exc}\n{traceback.format_exc()}"
        _log(f"CRASH: {exc}")
        _MT5["connected"] = False
        _MT5["running"]   = False


def _mt5_loop_inner(stop_event, config):
    from agents.signal_engine import run_signal
    from utils.mt5_bridge import connect, disconnect, get_open_positions, is_available

    if not is_available():
        _MT5["error"] = "MetaTrader5 package not installed — run: pip install MetaTrader5"
        _MT5["running"] = False
        return

    _log("Connecting to MT5…")
    if not connect():
        _MT5["error"] = "Cannot connect to MT5. Is the terminal open and logged in?"
        _MT5["running"] = False
        return

    _MT5["connected"] = True
    _MT5["error"] = ""
    _log("Connected — auto-trade active")

    instruments  = config["instruments"]   # list of {ticker, label, mt5_symbol}
    min_conf     = config["min_conf"]
    account_size = config["account_size"]
    risk_pct     = config["risk_pct"]
    finnhub_key  = config["finnhub_key"]
    auto_trade   = config.get("auto_trade", True)
    use_daily    = config.get("use_daily", True)
    _dt_cache: dict[str, tuple[dict, float]] = {}

    while not stop_event.is_set():
        today = date.today()
        _MT5["sent"]     = {k for k in _MT5["sent"]     if k.endswith(str(today))}
        _MT5["rejected"] = {k for k in _MT5["rejected"] if k.endswith(str(today))}

        for inst in instruments:
            if stop_event.is_set():
                break
            ticker     = inst["ticker"]
            mt5_symbol = inst.get("mt5_symbol")
            _log(f"Polling {ticker}…")

            daily_trend = None
            if use_daily:
                try:
                    from data.fetcher_intraday import fetch_daily_trend
                    cached = _dt_cache.get(ticker)
                    if cached is None or time.time() - cached[1] > 900:
                        daily_trend = fetch_daily_trend(ticker)
                        _dt_cache[ticker] = (daily_trend, time.time())
                    else:
                        daily_trend = cached[0]
                except Exception:
                    pass

            try:
                sr = run_signal(
                    ticker=ticker,
                    finnhub_api_key=finnhub_key,
                    account_size=account_size,
                    risk_pct=risk_pct,
                    daily_trend=daily_trend,
                    backtest_win_rates=_MT5.get("live_win_rates") or None,
                    mt5_symbol=mt5_symbol,
                )
            except Exception as exc:
                _log(f"ERROR {ticker}: {exc}")
                continue

            if sr.error:
                _log(f"{ticker}: {sr.error}")
                continue
            if sr.direction not in ("LONG", "SHORT"):
                _log(f"{ticker}: no signal (conf={sr.confidence:.0%})")
                continue
            if sr.confidence < min_conf:
                _log(f"{ticker}: {sr.direction} {sr.confidence:.0%} below threshold")
                continue

            key = f"{ticker}:{sr.direction}:{today}"
            if key in _MT5["sent"] or key in _MT5["rejected"]:
                continue

            if auto_trade:
                try:
                    from utils.mt5_bridge import send_from_signal_result
                    res = send_from_signal_result(sr)
                    if res.success:
                        _MT5["sent"].add(key)
                        _MT5["last_sent"] = {"ticker": ticker, "ticket": res.ticket,
                                             "direction": sr.direction, "conf": sr.confidence}
                        _log(f"✅ #{res.ticket} {sr.direction} {ticker} {sr.confidence:.0%}")
                        try:
                            from data.trade_outcomes import record_trade
                            strategy = sr.chart_signals[0].strategy if sr.chart_signals else "UNKNOWN"
                            record_trade(res.ticket, ticker, strategy, sr.direction,
                                         sr.confidence, sr.entry, sr.sl, sr.tp)
                        except Exception:
                            pass
                    else:
                        _log(f"⚠ Rejected: {res.error}")
                except Exception as exc:
                    _log(f"⚠ Error: {exc}")

        try:
            _MT5["positions"] = get_open_positions()
        except Exception:
            pass

        try:
            from data.trade_outcomes import update_outcomes_from_mt5, compute_win_rates
            updated = update_outcomes_from_mt5()
            if updated:
                _log(f"📚 {updated} outcome(s) recorded")
            _MT5["live_win_rates"] = compute_win_rates()
        except Exception:
            pass

        _MT5["last_check"] = datetime.now()
        stop_event.wait(900)  # 15-minute scan cycle

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
        import MetaTrader5 as mt5
        from_dt = datetime.now() - timedelta(days=days)
        deals = mt5.history_deals_get(from_dt, datetime.now())
        if deals is None:
            return []
        rows = []
        for d in deals:
            if d.entry == 0:
                continue
            rows.append({
                "ticket": d.deal, "symbol": d.symbol,
                "type": "BUY" if d.type == 0 else "SELL",
                "volume": d.volume, "price": d.price,
                "profit": d.profit,
                "time": datetime.fromtimestamp(d.time).strftime("%Y-%m-%d %H:%M"),
                "comment": d.comment,
            })
        return sorted(rows, key=lambda x: x["time"], reverse=True)
    except Exception:
        return []


# ── Helpers ───────────────────────────────────────────────────────────────────

def _refresh_backtest_rates(ticker: str):
    with _BT_LOCK:
        last_ts = _BT_CACHE["ts"].get(ticker, 0)
        stale   = time.time() - last_ts > 3600
        running = ticker in _BT_CACHE["running"]
        cached  = _BT_CACHE["rates"].get(ticker)
    if stale and not running:
        t = threading.Thread(target=_bt_background, args=(ticker,), daemon=True)
        with _BT_LOCK:
            _BT_CACHE["running"].add(ticker)
        t.start()
    return cached


@st.cache_data(ttl=300, show_spinner=False)
def _get_broker_symbols() -> list[dict]:
    """Fetch available symbols from MT5 broker (cached 5 min)."""
    try:
        from utils.mt5_bridge import get_tradeable_symbols
        return get_tradeable_symbols()
    except Exception:
        return []


# ── Read persisted settings ───────────────────────────────────────────────────
finnhub_key  = st.session_state.get("finnhub_key",  "d7j16r1r01qn2qavovt0d7j16r1r01qn2qavovtg")
account_size = st.session_state.get("account_size", 100_000.0)
risk_pct     = st.session_state.get("risk_pct",     1.0)
min_conf     = st.session_state.get("min_conf",     0.65)
auto_trade   = st.session_state.get("auto_trade",   True)
use_daily    = st.session_state.get("use_daily",    True)
use_bt_cal   = st.session_state.get("use_bt_cal",   True)
auto_refresh = st.session_state.get("auto_refresh", False)

# Selected instruments: list of {ticker, label, mt5_symbol}
# Defaults: major forex + futures using MT5 symbols directly
_DEFAULT_INSTRUMENTS = [
    {"ticker": "EURUSD",    "label": "EUR/USD",    "mt5_symbol": "EURUSD"},
    {"ticker": "GBPUSD",    "label": "GBP/USD",    "mt5_symbol": "GBPUSD"},
    {"ticker": "USDJPY",    "label": "USD/JPY",    "mt5_symbol": "USDJPY"},
    {"ticker": "#US100_M26","label": "NAS100",     "mt5_symbol": "#US100_M26"},
    {"ticker": "#US500_M26","label": "S&P500",     "mt5_symbol": "#US500_M26"},
]
instruments = st.session_state.get("instruments", _DEFAULT_INSTRUMENTS)

# ── Handle MT5 action (start/stop) ────────────────────────────────────────────
_mt5_action = st.session_state.pop("_mt5_action", None)
if _mt5_action:
    if _mt5_action["action"] == "start":
        _start_mt5(_mt5_action["cfg"])
    else:
        _stop_mt5()
    st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
h_left, h_mid, h_right = st.columns([3, 3, 2])
with h_left:
    st.markdown("# ⚡ Katraswing")

with h_mid:
    if _MT5["running"] and _MT5["connected"]:
        lc = _MT5["last_check"].strftime("%H:%M") if _MT5["last_check"] else "—"
        mode = "🤖 AUTO" if auto_trade else "👁 WATCH"
        st.markdown(
            f"<div style='padding-top:20px;'>"
            f"<span style='color:#22c55e;font-weight:700;font-size:14px;'>● LIVE {mode}</span>"
            f"<span style='color:#6b7280;font-size:12px;'> · last scan {lc}</span></div>",
            unsafe_allow_html=True)
    elif _MT5["running"]:
        st.markdown("<div style='padding-top:20px;color:#f59e0b;font-weight:700;'>● CONNECTING…</div>",
                    unsafe_allow_html=True)
    elif _MT5["error"]:
        st.markdown(f"<div style='padding-top:20px;color:#ef4444;font-size:12px;'>⚠ {_MT5['error'][:80]}</div>",
                    unsafe_allow_html=True)

with h_right:
    st.markdown("<div style='padding-top:12px;'>", unsafe_allow_html=True)
    scan_btn = st.button("🔄 Scan Now", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Settings & Instruments expander ──────────────────────────────────────────
with st.expander("⚙️  Settings & Instruments", expanded=False):
    st.markdown("#### Trading Parameters")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        account_size = st.number_input("Account ($)", value=account_size, step=1000.0, format="%.0f")
        risk_pct     = st.number_input("Risk per trade (%)", value=risk_pct, step=0.1, format="%.1f",
                                       min_value=0.1, max_value=10.0)
    with c2:
        min_conf   = st.slider("Min confidence", 0.50, 0.95, min_conf, 0.05)
        finnhub_key = st.text_input("Finnhub API key", value=finnhub_key, type="password")
    with c3:
        auto_trade  = st.checkbox("🤖 Auto-trade", value=auto_trade,
                                   help="Send orders automatically without approval")
        use_daily   = st.checkbox("Daily trend gate", value=use_daily,
                                   help="Veto signals that oppose the daily trend")
        use_bt_cal  = st.checkbox("Backtest calibration", value=use_bt_cal,
                                   help="Adjust confidence from historical win rates")
    with c4:
        auto_refresh = st.checkbox("Auto-scan every 5 min", value=auto_refresh)

    st.markdown("---")
    st.markdown("#### MT5 Instruments")

    from utils.mt5_bridge import is_available as _mt5_avail, is_connected as _mt5_ic

    mt5_col1, mt5_col2 = st.columns([1, 2])
    with mt5_col1:
        if not _MT5["running"]:
            if st.button("▶ Start Auto-Trade", type="primary", use_container_width=True):
                st.session_state["_mt5_action"] = {
                    "action": "start",
                    "cfg": {
                        "instruments": instruments,
                        "min_conf":     min_conf,
                        "account_size": account_size,
                        "risk_pct":     risk_pct,
                        "finnhub_key":  finnhub_key,
                        "interval":     900,
                        "auto_trade":   auto_trade,
                        "use_daily":    use_daily,
                    },
                }
        else:
            c_stop, c_close = st.columns(2)
            with c_stop:
                if st.button("⏹ Stop", use_container_width=True):
                    st.session_state["_mt5_action"] = {"action": "stop"}
            with c_close:
                if _MT5["connected"] and st.button("🚨 Close All", use_container_width=True):
                    from utils.mt5_bridge import close_all_positions
                    close_all_positions()
                    st.success("All positions closed.")

        if not _mt5_avail():
            st.warning("Install MT5:\n`pip install MetaTrader5`")

    with mt5_col2:
        broker_syms = _get_broker_symbols() if _mt5_ic() else []
        if broker_syms:
            sym_names   = [s["name"] for s in broker_syms]
            sym_descs   = {s["name"]: s["description"] for s in broker_syms}
            current_sel = [i["mt5_symbol"] for i in instruments if i.get("mt5_symbol") in sym_names]
            selected    = st.multiselect(
                "Select instruments from your broker",
                options=sym_names,
                default=current_sel,
                format_func=lambda n: f"{n}  —  {sym_descs.get(n, '')}",
            )
            if selected:
                instruments = [
                    {"ticker": s, "label": sym_descs.get(s, s), "mt5_symbol": s}
                    for s in selected
                ]
        else:
            st.info("Connect MT5 to pick instruments from your broker.\n"
                    "Currently using defaults: " + ", ".join(i["label"] for i in instruments))
            custom = st.text_input(
                "Or type MT5 symbol names (comma-separated)",
                value=", ".join(i["mt5_symbol"] for i in instruments),
                placeholder="EURUSD, GBPUSD, #US100_M26",
            )
            if custom:
                syms = [s.strip() for s in custom.split(",") if s.strip()]
                instruments = [{"ticker": s, "label": s, "mt5_symbol": s} for s in syms]

    if _MT5["log"]:
        with st.expander("Activity log", expanded=False):
            st.markdown(
                "<div class='log-box'>" + "<br>".join(_MT5["log"][-15:]) + "</div>",
                unsafe_allow_html=True)

# ── Persist settings ──────────────────────────────────────────────────────────
st.session_state.update({
    "finnhub_key": finnhub_key, "account_size": account_size,
    "risk_pct": risk_pct, "min_conf": min_conf,
    "auto_trade": auto_trade, "use_daily": use_daily,
    "use_bt_cal": use_bt_cal, "auto_refresh": auto_refresh,
    "instruments": instruments,
})

# ── Auto-refresh trigger ──────────────────────────────────────────────────────
if auto_refresh:
    if time.time() - st.session_state.get("last_refresh_ts", 0) > 300:
        st.session_state.pop("results", None)

needs_run = scan_btn or ("results" not in st.session_state)

# ── Signal scan (parallel) ────────────────────────────────────────────────────
if needs_run:
    from agents.signal_engine import run_signal
    from concurrent.futures import ThreadPoolExecutor

    def _refresh_daily(ticker):
        k_val, k_ts = f"_dt_{ticker}", f"_dt_ts_{ticker}"
        if time.time() - st.session_state.get(k_ts, 0) > 900:
            try:
                from data.fetcher_intraday import fetch_daily_trend
                trend = fetch_daily_trend(ticker)
                st.session_state[k_val] = trend
                st.session_state[k_ts]  = time.time()
            except Exception:
                st.session_state[k_val] = None
                st.session_state[k_ts]  = time.time()
        return st.session_state.get(k_val)

    # Pre-fetch inputs on main thread
    ticker_inputs = {}
    for inst in instruments:
        t = inst["ticker"]
        ticker_inputs[t] = {
            "inst":       inst,
            "daily":      _refresh_daily(t) if use_daily else None,
            "bt_rates":   _refresh_backtest_rates(t) if use_bt_cal else None,
        }

    def _scan_one(ticker):
        inp = ticker_inputs[ticker]
        inst = inp["inst"]
        return ticker, run_signal(
            ticker=ticker,
            display_name=inst.get("label", ticker),
            finnhub_api_key=finnhub_key,
            account_size=account_size,
            risk_pct=risk_pct,
            daily_trend=inp["daily"],
            backtest_win_rates=inp["bt_rates"],
            mt5_symbol=inst.get("mt5_symbol"),
        )

    results = {}
    with st.spinner(f"Scanning {len(instruments)} instruments…"):
        with ThreadPoolExecutor(max_workers=max(len(instruments), 1)) as ex:
            for ticker, sr in ex.map(_scan_one, [i["ticker"] for i in instruments]):
                results[ticker] = sr

    st.session_state["results"] = results
    st.session_state["last_refresh_ts"] = time.time()
else:
    results = st.session_state.get("results", {})

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_signals, tab_trades, tab_history, tab_journal, tab_learning = st.tabs([
    "📊  Signals",
    "📈  Open Trades",
    "🕐  History",
    "📓  Journal",
    "🧠  Learning",
])

# ── Tab 1: Signals ────────────────────────────────────────────────────────────
with tab_signals:
    if _MT5["error"] and not _MT5["running"]:
        st.error(_MT5["error"])

    # Separate active signals from no-signal results
    active, inactive = [], []
    for inst in instruments:
        t  = inst["ticker"]
        sr = results.get(t)
        if sr is None:
            continue
        if sr.error:
            inactive.append((inst, sr, "error"))
        elif sr.direction in ("LONG", "SHORT"):
            active.append((inst, sr))
        else:
            inactive.append((inst, sr, "flat"))

    active.sort(key=lambda x: x[1].confidence, reverse=True)

    # ── Active signals ────────────────────────────────────────────────────────
    if active:
        st.markdown(f"### 🎯  {len(active)} Active Signal{'s' if len(active) != 1 else ''}")
        # Column headers
        hc = st.columns([1, 2, 1, 4, 1])
        for col, lbl in zip(hc, ["Dir", "Symbol", "Conf", "Entry / SL / TP  ·  Strategy", ""]):
            col.markdown(f"<span style='color:#6b7280;font-size:11px;font-weight:700;"
                         f"text-transform:uppercase;letter-spacing:1px;'>{lbl}</span>",
                         unsafe_allow_html=True)
        st.markdown("<div style='border-top:1px solid #1e2330;margin:4px 0 8px;'></div>",
                    unsafe_allow_html=True)

        for inst, sr in active:
            dir_c  = "#22c55e" if sr.direction == "LONG" else "#ef4444"
            bg     = "#0a1f14" if sr.direction == "LONG" else "#1f0a0a"
            border = "#16a34a" if sr.direction == "LONG" else "#dc2626"
            arrow  = "▲" if sr.direction == "LONG" else "▼"
            conf   = int(sr.confidence * 100)
            risk   = abs(sr.entry - sr.sl)
            reward = abs(sr.tp - sr.entry)
            rr     = f"1:{reward/risk:.1f}" if risk > 0 else "—"
            strat  = sr.chart_signals[0].strategy if sr.chart_signals else "—"
            label  = inst.get("label", inst["ticker"])

            # Confidence bar (simple HTML)
            bar_w  = conf
            bar_bg = "#22c55e" if sr.direction == "LONG" else "#ef4444"

            c_dir, c_sym, c_conf, c_detail, c_btn = st.columns([1, 2, 1, 4, 1])

            with c_dir:
                st.markdown(
                    f"<div style='background:{bg};border:1px solid {border};border-radius:6px;"
                    f"padding:10px 12px;text-align:center;'>"
                    f"<span style='color:{dir_c};font-size:15px;font-weight:800;'>{arrow} {sr.direction}</span>"
                    f"</div>",
                    unsafe_allow_html=True)

            with c_sym:
                st.markdown(
                    f"<div style='padding:10px 0;'>"
                    f"<span style='color:#e0e0e0;font-size:15px;font-weight:700;'>{label}</span><br>"
                    f"<span style='color:#6b7280;font-size:11px;'>{inst['ticker']}</span>"
                    f"</div>",
                    unsafe_allow_html=True)

            with c_conf:
                st.markdown(
                    f"<div style='padding:10px 0;'>"
                    f"<span style='color:{dir_c};font-size:18px;font-weight:700;'>{conf}%</span><br>"
                    f"<div style='background:#1e2330;border-radius:2px;height:4px;width:100%;margin-top:4px;'>"
                    f"<div style='background:{bar_bg};width:{bar_w}%;height:4px;border-radius:2px;'></div></div>"
                    f"</div>",
                    unsafe_allow_html=True)

            with c_detail:
                adx_badge = ""
                if sr.adx_regime != "NEUTRAL":
                    adx_c = "#f59e0b" if sr.adx_regime == "TRENDING" else "#6b7280"
                    adx_badge = (f"<span style='background:#1e2330;color:{adx_c};"
                                 f"font-size:10px;padding:1px 5px;border-radius:3px;"
                                 f"margin-left:6px;'>{sr.adx_regime}</span>")
                veto = ""
                if sr.daily_trend_vetoed:
                    veto = "<span style='color:#ef4444;font-size:10px;'> ⚠ trend veto</span>"
                st.markdown(
                    f"<div style='padding:6px 0;font-size:13px;'>"
                    f"<span style='color:#6b7280;'>Entry</span> "
                    f"<b style='color:#60a5fa;'>{sr.entry:.4f}</b>&nbsp;&nbsp;"
                    f"<span style='color:#6b7280;'>SL</span> "
                    f"<b style='color:#ef4444;'>{sr.sl:.4f}</b>&nbsp;&nbsp;"
                    f"<span style='color:#6b7280;'>TP</span> "
                    f"<b style='color:#22c55e;'>{sr.tp:.4f}</b>&nbsp;&nbsp;"
                    f"<span style='color:#6b7280;'>R:R</span> <b>{rr}</b><br>"
                    f"<span style='color:#9ca3af;font-size:12px;'>{strat}{adx_badge}{veto}</span>"
                    f"</div>",
                    unsafe_allow_html=True)

            with c_btn:
                from utils.mt5_bridge import is_available as _ok
                if _ok():
                    key = f"send_{inst['ticker']}_{sr.direction}"
                    if st.button("Send →", key=key, type="primary"):
                        from utils.mt5_bridge import ensure_connected, send_from_signal_result, get_open_positions as _gop3
                        with st.spinner("…"):
                            ok = ensure_connected()
                        if ok:
                            res = send_from_signal_result(sr)
                            if res.success:
                                _MT5["connected"]  = True
                                _MT5["positions"]  = _gop3()
                                _MT5["last_sent"]  = {"ticker": inst["ticker"], "ticket": res.ticket}
                                _log(f"Quick-send #{res.ticket} ✓")
                                st.success(f"#{res.ticket} sent!")
                                try:
                                    from data.trade_outcomes import record_trade
                                    strat_name = sr.chart_signals[0].strategy if sr.chart_signals else "UNKNOWN"
                                    record_trade(res.ticket, inst["ticker"], strat_name, sr.direction,
                                                 sr.confidence, sr.entry, sr.sl, sr.tp)
                                except Exception:
                                    pass
                                st.rerun()
                            else:
                                st.error(res.error)
                        else:
                            st.error("MT5 not connected")

            st.markdown("<div style='border-bottom:1px solid #1e2330;margin:2px 0;'></div>",
                        unsafe_allow_html=True)

    else:
        st.markdown(
            "<div style='text-align:center;color:#6b7280;padding:60px 0;font-size:15px;'>"
            "No signals above threshold right now.<br>"
            "<span style='font-size:13px;'>Click <b>Scan Now</b> to refresh.</span></div>",
            unsafe_allow_html=True)

    # ── No-signal summary (collapsed) ─────────────────────────────────────────
    if inactive:
        with st.expander(f"{len(inactive)} instruments — no signal", expanded=False):
            for inst, sr, reason in inactive:
                label = inst.get("label", inst["ticker"])
                if reason == "error":
                    st.caption(f"⚠ {label}: {sr.error[:80]}")
                else:
                    conf_str = f"{sr.confidence:.0%}" if sr.confidence else "—"
                    st.caption(f"— {label}: NO TRADE  (conf {conf_str})")

    if results and st.session_state.get("last_refresh_ts"):
        ts = datetime.fromtimestamp(st.session_state["last_refresh_ts"]).strftime("%H:%M:%S")
        st.caption(f"Last scan: {ts}")


# ── Tab 2: Open Trades ────────────────────────────────────────────────────────
with tab_trades:
    from utils.mt5_bridge import is_available as _mt5_avail2, is_connected as _mt5_ic2, get_open_positions as _gop2
    if _mt5_avail2():
        _pos_ts = st.session_state.get("_pos_fetch_ts", 0)
        if _mt5_ic2() and time.time() - _pos_ts > 30:
            _MT5["positions"] = _gop2()
            st.session_state["_pos_fetch_ts"] = time.time()

        c_ref, _ = st.columns([1, 5])
        with c_ref:
            if st.button("🔄 Refresh", key="refresh_positions"):
                from utils.mt5_bridge import ensure_connected
                if ensure_connected():
                    _MT5["connected"] = True
                    _MT5["positions"] = _gop2()
                    st.session_state["_pos_fetch_ts"] = time.time()
                st.rerun()

    positions = _MT5["positions"]
    if not _mt5_avail2():
        st.info("Install MetaTrader5: `pip install MetaTrader5`")
    elif not positions:
        st.markdown("<div style='text-align:center;color:#6b7280;padding:60px 0;'>No open positions</div>",
                    unsafe_allow_html=True)
    else:
        total_pnl = sum(p.profit for p in positions)
        pnl_color = "#22c55e" if total_pnl >= 0 else "#ef4444"
        st.markdown(
            f"<div style='font-size:13px;color:#6b7280;margin-bottom:16px;'>"
            f"{len(positions)} position{'s' if len(positions)!=1 else ''} &nbsp;·&nbsp; "
            f"P&L: <b style='color:{pnl_color};'>{total_pnl:+.2f}</b></div>",
            unsafe_allow_html=True)
        for p in positions:
            dir_c = "#22c55e" if p.direction == "LONG" else "#ef4444"
            pnl_c = "#22c55e" if p.profit >= 0 else "#ef4444"
            arrow = "▲" if p.direction == "LONG" else "▼"
            st.markdown(
                f"<div style='background:#111827;border-radius:8px;padding:12px 18px;"
                f"margin-bottom:6px;border:1px solid #1e2330;"
                f"display:flex;justify-content:space-between;align-items:center;'>"
                f"<div>"
                f"<span style='color:{dir_c};font-weight:700;'>{arrow} {p.direction}</span>"
                f" <span style='color:#e0e0e0;font-weight:600;'>{p.symbol}</span>"
                f" <span style='color:#6b7280;font-size:12px;'>vol {p.volume}</span><br>"
                f"<span style='color:#6b7280;font-size:12px;'>"
                f"Entry {p.open_price:.4f} · SL {p.sl:.4f} · TP {p.tp:.4f} · #{p.ticket}</span>"
                f"</div>"
                f"<span style='color:{pnl_c};font-size:18px;font-weight:700;'>{p.profit:+.2f}</span>"
                f"</div>",
                unsafe_allow_html=True)
        if st.button("🚨 Close All Positions", type="primary"):
            from utils.mt5_bridge import close_all_positions
            close_all_positions()
            st.success("Closed.")
            st.rerun()


# ── Tab 3: History ────────────────────────────────────────────────────────────
with tab_history:
    days_back = st.selectbox("Show last", [7, 14, 30, 60, 90], index=2,
                             format_func=lambda x: f"{x} days")
    if st.button("Load History", type="primary"):
        st.session_state["trade_history"] = _fetch_mt5_history(days_back)
    history = st.session_state.get("trade_history")
    if history is None:
        st.info("Click **Load History** to fetch closed trades from MT5.")
    elif not history:
        st.caption("No closed trades in this period.")
    else:
        wins  = [t for t in history if t["profit"] > 0]
        total = sum(t["profit"] for t in history)
        wr    = len(wins) / len(history) * 100 if history else 0
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Trades", len(history))
        m2.metric("Win Rate", f"{wr:.1f}%")
        m3.metric("Total P&L", f"{total:+.2f}")
        m4.metric("Avg P&L", f"{total/len(history):+.2f}" if history else "—")
        st.markdown("---")
        for t in history[:50]:
            pnl_c  = "#22c55e" if t["profit"] > 0 else "#ef4444"
            type_c = "#22c55e" if t["type"] == "BUY" else "#ef4444"
            st.markdown(
                f"<div style='background:#111827;border-radius:6px;padding:8px 14px;"
                f"margin-bottom:3px;border:1px solid #1e2330;"
                f"display:flex;justify-content:space-between;font-size:13px;'>"
                f"<span><span style='color:{type_c};font-weight:600;'>{t['type']}</span>"
                f" <span style='color:#e0e0e0;'>{t['symbol']}</span>"
                f" <span style='color:#6b7280;'>vol {t['volume']} @ {t['price']:.4f}"
                f" · {t['time']}</span></span>"
                f"<span style='color:{pnl_c};font-weight:700;'>{t['profit']:+.2f}</span>"
                f"</div>",
                unsafe_allow_html=True)


# ── Tab 4: Journal ────────────────────────────────────────────────────────────
with tab_journal:
    import pathlib
    _journal_path = pathlib.Path(__file__).parent / "static" / "trading-journal.html"
    if _journal_path.exists():
        st.html(_journal_path.read_text(encoding="utf-8"))
    else:
        st.info("Journal file not found at static/trading-journal.html")


# ── Tab 5: Learning ───────────────────────────────────────────────────────────
with tab_learning:
    from data.trade_outcomes import get_summary, compute_win_rates

    c_lref, _ = st.columns([1, 5])
    with c_lref:
        if st.button("🔄 Refresh outcomes", key="refresh_outcomes"):
            try:
                from data.trade_outcomes import update_outcomes_from_mt5
                from utils.mt5_bridge import ensure_connected as _ec
                n = update_outcomes_from_mt5() if _ec() else 0
                st.success(f"Updated {n} outcome(s).")
            except Exception as exc:
                st.error(str(exc))
            st.rerun()

    summary = get_summary()
    if summary["total_sent"] == 0:
        st.info("No auto-trades recorded yet. Outcomes appear here as positions close.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        wr      = summary["win_rate"]
        wr_str  = f"{wr:.0%}" if wr is not None else "—"
        wr_col  = "#22c55e" if (wr or 0) >= 0.55 else ("#f59e0b" if (wr or 0) >= 0.45 else "#ef4444")
        p_col   = "#22c55e" if summary["total_profit"] >= 0 else "#ef4444"
        c1.metric("Sent", summary["total_sent"])
        c2.metric("Closed", summary["total_closed"], f"{summary['total_open']} open")
        c3.markdown(f"<div style='font-size:13px;color:#6b7280;'>Win rate</div>"
                    f"<div style='font-size:28px;font-weight:700;color:{wr_col};'>{wr_str}</div>",
                    unsafe_allow_html=True)
        c4.markdown(f"<div style='font-size:13px;color:#6b7280;'>Total P&L</div>"
                    f"<div style='font-size:28px;font-weight:700;color:{p_col};'>"
                    f"{summary['total_profit']:+.2f}</div>",
                    unsafe_allow_html=True)
        st.markdown("---")
        if summary["by_strategy"]:
            st.markdown("### Strategy Performance")
            live_wr = compute_win_rates()
            for row in summary["by_strategy"]:
                s     = row["strategy"]
                wr_s  = row["win_rate"]
                col   = "#22c55e" if wr_s >= 0.55 else ("#f59e0b" if wr_s >= 0.45 else "#ef4444")
                p_col2 = "#22c55e" if row["profit"] >= 0 else "#ef4444"
                badge = (
                    f"<span style='background:#1e3a1e;color:#22c55e;font-size:10px;"
                    f"padding:2px 5px;border-radius:3px;'>✓ calibrating</span>"
                    if s in live_wr else
                    f"<span style='background:#1e2330;color:#6b7280;font-size:10px;"
                    f"padding:2px 5px;border-radius:3px;'>need {max(0,5-row['trades'])} more</span>"
                )
                st.markdown(
                    f"<div style='background:#111827;border-radius:8px;padding:10px 16px;"
                    f"margin-bottom:5px;border:1px solid #1e2330;"
                    f"display:flex;justify-content:space-between;align-items:center;'>"
                    f"<div><span style='color:#e0e0e0;font-weight:700;'>{s}</span> {badge}</div>"
                    f"<div style='display:flex;gap:20px;font-size:13px;'>"
                    f"<span style='color:#6b7280;'>Trades: <b style='color:#e0e0e0;'>{row['trades']}</b></span>"
                    f"<span style='color:#6b7280;'>Win: <b style='color:{col};'>{wr_s:.0%}</b></span>"
                    f"<span style='color:#6b7280;'>P&L: <b style='color:{p_col2};'>{row['profit']:+.2f}</b></span>"
                    f"</div></div>",
                    unsafe_allow_html=True)
        with st.expander("Full trade log", expanded=False):
            for t in summary["all_trades"][:50]:
                outcome  = t.get("outcome") or "open"
                o_col    = {"WIN":"#22c55e","LOSS":"#ef4444","BREAKEVEN":"#f59e0b","open":"#6b7280"}.get(outcome,"#6b7280")
                profit_s = f"{t['profit']:+.2f}" if t.get("profit") is not None else "—"
                st.markdown(
                    f"<div style='font-size:12px;padding:3px 0;border-bottom:1px solid #1e2330;'>"
                    f"<span style='color:{o_col};font-weight:700;min-width:70px;display:inline-block;'>{outcome.upper()}</span>"
                    f"<span style='color:#9ca3af;'>{t['direction']} {t['ticker']} · {t['strategy']}"
                    f" · {t['confidence']:.0%} · P&L <b style='color:{o_col};'>{profit_s}</b>"
                    f" · #{t['ticket']} · {t['sent_at'][:16]}</span></div>",
                    unsafe_allow_html=True)


# ── Auto-refresh while monitoring ─────────────────────────────────────────────
if _MT5["running"]:
    time.sleep(30)
    st.rerun()
