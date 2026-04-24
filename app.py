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

# ── Background backtest cache (module-level, thread-safe reads) ───────────────
# Runs one backtest thread per ticker; main thread never blocks waiting for it.
_BT_CACHE: dict = {
    "rates":    {},    # ticker -> dict[strategy, win_rate]
    "ts":       {},    # ticker -> float (unix timestamp of last completed run)
    "running":  set(), # tickers currently being computed in background
}
_BT_LOCK = threading.Lock()


def _bt_background(ticker: str) -> None:
    """Run backtest for one ticker and store results in _BT_CACHE."""
    try:
        from agents.intraday_backtester import run_intraday_backtest
        summary = run_intraday_backtest(ticker, timeframe="5m")
        rates = {
            r.strategy: r.win_rate
            for r in summary.results
            if r.total_trades >= 5
        }
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

_ET  = ZoneInfo("America/New_York")
_JST = ZoneInfo("Asia/Tokyo")

# Markets that follow regular stock exchange hours (NYSE/NASDAQ)
_STOCK_TICKERS = {"AAPL", "MSFT", "AMZN"}

# US futures on CME Globex (nearly 24/7 except weekend break)
_US_FUTURES_TICKERS = {"NQ=F", "ES=F"}

# Japanese futures — primary session is Tokyo Stock Exchange hours
_JAPAN_FUTURES_TICKERS = {"NKD=F"}

# Forex pairs — open Mon 00:00 UTC through Fri 22:00 UTC (weekdays only)
_FOREX_TICKERS = {"EURUSD=X", "GBPUSD=X", "USDJPY=X", "XAUUSD=X"}


def _market_status(ticker: str, df) -> tuple[bool, str]:
    """
    Returns (is_closed, status_label).
    Primary: calendar/clock check (always accurate).
    Secondary: data staleness check only when the clock says the market IS open
               (catches rare yfinance outages or very stale cached data).
    NKD=F uses Tokyo session hours (09:00–15:30 JST) not US ET rules.
    """
    # ── Clock-based check (primary) ───────────────────────────────────────────
    if ticker in _STOCK_TICKERS:
        now = datetime.now(_ET)
        wd, hm = now.weekday(), now.hour * 60 + now.minute
        if wd >= 5:
            return True, "Market closed  ·  weekend"
        if hm < 9 * 60 + 30:
            opens_in = 9 * 60 + 30 - hm
            return True, f"Pre-market  ·  opens in {opens_in}m (09:30 ET)"
        if hm >= 16 * 60:
            return True, "After hours  ·  closed at 16:00 ET"

    elif ticker in _US_FUTURES_TICKERS:
        now = datetime.now(_ET)
        wd, hm = now.weekday(), now.hour * 60 + now.minute
        # CME Globex closed: Fri 17:00 → Sun 18:00 ET; daily break 16:00-17:00 ET
        if wd == 5:
            return True, "Market closed  ·  CME weekend break"
        if wd == 6 and hm < 18 * 60:
            opens_in = 18 * 60 - hm
            return True, f"Market closed  ·  CME opens Sun 18:00 ET (in {opens_in}m)"
        if wd == 4 and hm >= 17 * 60:
            return True, "Market closed  ·  CME weekend break starts"
        if 16 * 60 <= hm < 17 * 60:
            return True, "Daily maintenance  ·  CME break 16:00-17:00 ET"

    elif ticker in _JAPAN_FUTURES_TICKERS:
        now = datetime.now(_JST)
        wd, hm = now.weekday(), now.hour * 60 + now.minute
        if wd >= 5:
            return True, "Market closed  ·  TSE weekend"
        # TSE morning: 09:00–11:30 JST, afternoon: 12:30–15:30 JST
        if hm < 9 * 60:
            opens_in = 9 * 60 - hm
            return True, f"Pre-market  ·  TSE opens in {opens_in}m (09:00 JST)"
        if 11 * 60 + 30 <= hm < 12 * 60 + 30:
            return True, "Lunch break  ·  TSE 11:30–12:30 JST"
        if hm >= 15 * 60 + 30:
            return True, "After hours  ·  TSE closed at 15:30 JST"

    elif ticker in _FOREX_TICKERS:
        # Forex is open Mon 00:00 → Fri 22:00 UTC
        now = datetime.now(timezone.utc)
        wd  = now.weekday()
        if wd == 5:  # Saturday
            return True, "Forex closed  ·  weekend"
        if wd == 6:  # Sunday — opens ~22:00 UTC
            if now.hour < 22:
                return True, f"Forex closed  ·  opens Sun 22:00 UTC"
        if wd == 4 and now.hour >= 22:  # Friday after 22:00 UTC
            return True, "Forex closed  ·  weekly close (Fri 22:00 UTC)"

    # ── Staleness check (secondary — only fires when clock says market is open) -
    # yfinance 5m data has a ~15 min delay; use 30 min threshold to avoid false
    # positives while still catching genuine data feed outages.
    if df is not None and not df.empty:
        try:
            last_ts = df.index[-1]
            if last_ts.tzinfo is None:
                last_ts = last_ts.tz_localize("UTC")
            age_min = (datetime.now(timezone.utc) - last_ts).total_seconds() / 60
            if age_min > 30:
                return True, f"Data stale  ·  last bar {int(age_min)}m ago"
        except Exception:
            pass

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
    # Futures
    {"ticker": "NQ=F",     "label": "NQ Mini",    "name": "Nasdaq 100 E-mini Futures", "group": "Futures"},
    {"ticker": "ES=F",     "label": "ES Mini",    "name": "S&P 500 E-mini Futures",    "group": "Futures"},
    {"ticker": "NKD=F",    "label": "Nikkei 225", "name": "Nikkei 225 Futures",        "group": "Futures"},
    # Stocks
    {"ticker": "AAPL",     "label": "Apple",      "name": "Apple Inc.",                "group": "Stocks"},
    {"ticker": "MSFT",     "label": "Microsoft",  "name": "Microsoft Corp.",           "group": "Stocks"},
    {"ticker": "AMZN",     "label": "Amazon",     "name": "Amazon.com Inc.",           "group": "Stocks"},
    # Forex
    {"ticker": "EURUSD=X", "label": "EUR/USD",    "name": "Euro / US Dollar",          "group": "Forex"},
    {"ticker": "GBPUSD=X", "label": "GBP/USD",    "name": "British Pound / US Dollar", "group": "Forex"},
]

# ── MT5 shared state ──────────────────────────────────────────────────────────
# Stored in session_state so it survives Streamlit reruns.
# The background thread mutates the same dict in-place — that's safe because
# st.session_state["_MT5"] IS the dict object (no copy on read).
if "_MT5" not in st.session_state:
    st.session_state["_MT5"] = {
        "thread":         None,
        "stop_event":     None,
        "running":        False,
        "connected":      False,
        "last_check":     None,
        "pending":        [],
        "sent":           set(),
        "rejected":       set(),
        "last_sent":      None,
        "positions":      [],
        "log":            [],
        "error":          "",
        "live_win_rates": {},   # strategy → win_rate from actual closed trades
    }
_MT5: dict = st.session_state["_MT5"]


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
    auto_trade   = config.get("auto_trade", True)

    # Per-ticker daily trend cache (15-min TTL) — avoids redundant fetches
    _dt_cache: dict[str, tuple[dict, float]] = {}

    while not stop_event.is_set():
        today = date.today()
        _MT5["sent"]     = {k for k in _MT5["sent"]     if k.endswith(str(today))}
        _MT5["rejected"] = {k for k in _MT5["rejected"] if k.endswith(str(today))}

        for ticker in tickers:
            if stop_event.is_set():
                break
            _log(f"Polling {ticker}…")

            # Refresh daily trend with 15-min TTL
            daily_trend = None
            try:
                from data.fetcher_intraday import fetch_daily_trend
                cached = _dt_cache.get(ticker)
                if cached is None or time.time() - cached[1] > 900:
                    daily_trend = fetch_daily_trend(ticker)
                    _dt_cache[ticker] = (daily_trend, time.time())
                else:
                    daily_trend = cached[0]
            except Exception:
                daily_trend = None

            try:
                sr = run_signal(
                    ticker=ticker,
                    finnhub_api_key=finnhub_key,
                    account_size=account_size,
                    risk_pct=risk_pct,
                    daily_trend=daily_trend,
                    backtest_win_rates=_MT5.get("live_win_rates") or None,
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
            item = {
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
            }

            if auto_trade:
                # Send immediately without approval
                try:
                    from utils.mt5_bridge import send_from_signal_result
                    res = send_from_signal_result(sr)
                    if res.success:
                        _MT5["sent"].add(key)
                        _MT5["last_sent"] = {**item, "ticket": res.ticket}
                        _log(f"✅ AUTO-TRADE #{res.ticket} — {sr.direction} {ticker} {sr.confidence:.0%}")
                        # Record for learning
                        try:
                            from data.trade_outcomes import record_trade
                            strategy = sr.chart_signals[0].strategy if sr.chart_signals else "UNKNOWN"
                            record_trade(res.ticket, ticker, strategy, sr.direction,
                                         sr.confidence, sr.entry, sr.sl, sr.tp)
                        except Exception:
                            pass
                    else:
                        _log(f"⚠ Auto-trade rejected: {res.error}")
                except Exception as exc:
                    _log(f"⚠ Auto-trade error: {exc}")
            else:
                _MT5["pending"].append(item)
                _log(f"★ APPROVAL NEEDED — {sr.direction} {ticker} {sr.confidence:.0%}")

        try:
            _MT5["positions"] = get_open_positions()
        except Exception:
            pass

        # Update closed trade outcomes and refresh live win rates for next cycle
        try:
            from data.trade_outcomes import update_outcomes_from_mt5, compute_win_rates
            updated = update_outcomes_from_mt5()
            if updated:
                _log(f"📚 {updated} trade outcome(s) recorded for learning")
            _MT5["live_win_rates"] = compute_win_rates()
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

    # Signal filter toggles
    st.markdown("**Signal Filters**")
    use_daily_gate = st.checkbox(
        "Daily trend gate",
        value=st.session_state.get("use_daily_gate", True),
        help="Hard-veto signals that oppose the daily EMA20/50 trend direction",
    )
    use_bt_calibration = st.checkbox(
        "Backtest calibration",
        value=st.session_state.get("use_bt_calibration", True),
        help="Adjust confidence based on recent 59-day backtest win rates per strategy",
    )
    with _BT_LOCK:
        bt_running = list(_BT_CACHE["running"])
        bt_done    = [t for t in [i["ticker"] for i in INSTRUMENTS] if _BT_CACHE["ts"].get(t, 0) > 0]
    if bt_running:
        st.caption(f"⏳ Backtest computing: {', '.join(bt_running)}")
    elif bt_done:
        st.caption(f"✓ Backtest cached: {len(bt_done)}/{len(INSTRUMENTS)} tickers")
    st.markdown("---")

    # Signal scan controls
    run_btn = st.button("🔄 Scan Signals Now", use_container_width=True, type="primary")
    auto_refresh = st.checkbox(
        "Auto-scan every 5 min",
        value=st.session_state.get("auto_refresh", False),
    )
    st.markdown("---")

    # MT5 monitor
    st.markdown("### MT5 Auto-Trade")
    mt5_tickers = st.multiselect(
        "Watch",
        options=["NQ=F", "ES=F", "NKD=F", "AAPL", "MSFT", "AMZN", "EURUSD=X", "GBPUSD=X"],
        default=st.session_state.get("mt5_tickers", ["NQ=F", "ES=F"]),
    )
    mt5_min_conf = st.slider(
        "Min confidence",
        min_value=0.50, max_value=0.95, step=0.05,
        value=st.session_state.get("mt5_min_conf", 0.65),
    )
    auto_trade = st.checkbox(
        "🤖 Auto-trade (no approval)",
        value=st.session_state.get("auto_trade", True),
        help="When ON, qualifying signals are sent to MT5 immediately. When OFF, they queue for manual approval.",
    )
    mt5_interval = 900  # fixed 15-minute scan

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
                    "auto_trade":   auto_trade,
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
            mode = "🤖 AUTO-TRADE" if st.session_state.get("auto_trade", True) else "👁 Watch-only"
            st.markdown(f"🟢 **{mode}**")
            if _MT5["last_check"]:
                next_scan = _MT5["last_check"] + timedelta(seconds=900)
                st.caption(
                    f"Last scan: {_MT5['last_check'].strftime('%H:%M:%S')}\n"
                    f"Next scan: {next_scan.strftime('%H:%M:%S')}"
                )
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
    "finnhub_key":        finnhub_key,
    "account_size":       account_size,
    "risk_pct":           risk_pct,
    "auto_refresh":       auto_refresh,
    "mt5_tickers":        mt5_tickers,
    "mt5_min_conf":       mt5_min_conf,
    "use_daily_gate":     use_daily_gate,
    "use_bt_calibration": use_bt_calibration,
    "auto_trade":         auto_trade,
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
def _refresh_daily_trend(ticker: str) -> dict | None:
    """Return cached daily trend dict (15-min TTL) or fetch fresh."""
    cache_key = f"_dt_{ticker}"
    ts_key    = f"_dt_ts_{ticker}"
    if time.time() - st.session_state.get(ts_key, 0) > 900:
        try:
            from data.fetcher_intraday import fetch_daily_trend
            trend = fetch_daily_trend(ticker)
            st.session_state[cache_key] = trend
            st.session_state[ts_key]    = time.time()
        except Exception:
            st.session_state[cache_key] = None
            st.session_state[ts_key]    = time.time()
    return st.session_state.get(cache_key)


def _refresh_backtest_rates(ticker: str) -> dict[str, float] | None:
    """
    Return cached per-strategy win rates (60-min TTL).
    Never blocks — launches a background thread if the cache is stale.
    Returns None on first call (bt_adjustment will be 0 until thread completes).
    """
    with _BT_LOCK:
        last_ts  = _BT_CACHE["ts"].get(ticker, 0)
        stale    = time.time() - last_ts > 3600
        running  = ticker in _BT_CACHE["running"]
        cached   = _BT_CACHE["rates"].get(ticker)

    if stale and not running:
        t = threading.Thread(target=_bt_background, args=(ticker,), daemon=True)
        with _BT_LOCK:
            _BT_CACHE["running"].add(ticker)
        t.start()

    return cached


if needs_run:
    from agents.signal_engine import run_signal
    results = {}
    with st.spinner("Scanning signals…"):
        for inst in INSTRUMENTS:
            ticker = inst["ticker"]
            daily_trend     = _refresh_daily_trend(ticker)     if use_daily_gate    else None
            bt_win_rates    = _refresh_backtest_rates(ticker)  if use_bt_calibration else None
            results[ticker] = run_signal(
                ticker=ticker,
                display_name=inst["name"],
                finnhub_api_key=finnhub_key,
                account_size=account_size,
                risk_pct=risk_pct,
                daily_trend=daily_trend,
                backtest_win_rates=bt_win_rates,
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
    # Build veto badge if daily trend gate suppressed the signal
    veto_badge = ""
    if getattr(result, "daily_trend_vetoed", False):
        veto_badge = " &nbsp;<span style='background:#7c3aed;color:#fff;border-radius:4px;padding:1px 6px;font-size:11px;'>VETOED</span>"

    # ADX regime badge
    _adx_regime = getattr(result, "adx_regime", "NEUTRAL")
    _adx_val    = getattr(result, "adx_value", 0.0)
    _regime_colors = {"TRENDING": "#f59e0b", "RANGING": "#60a5fa", "NEUTRAL": "#4b5563"}
    _regime_c = _regime_colors.get(_adx_regime, "#4b5563")
    regime_badge = (
        f"<span style='background:{_regime_c}22;color:{_regime_c};border:1px solid {_regime_c}55;"
        f"border-radius:4px;padding:1px 7px;font-size:10px;letter-spacing:.5px;'>"
        f"{_adx_regime} ADX {_adx_val:.0f}</span>"
    )

    # Daily trend badge
    _dtd = getattr(result, "daily_trend_direction", "NEUTRAL")
    _dt_colors = {"BULLISH": "#22c55e", "BEARISH": "#ef4444", "NEUTRAL": "#4b5563"}
    _dt_c = _dt_colors.get(_dtd, "#4b5563")
    dt_badge = (
        f"<span style='background:{_dt_c}22;color:{_dt_c};border:1px solid {_dt_c}55;"
        f"border-radius:4px;padding:1px 7px;font-size:10px;letter-spacing:.5px;'>"
        f"D {_dtd}</span>"
    ) if _dtd != "NEUTRAL" else ""

    # Consensus badge
    _agree = getattr(result, "strategy_agreement", "")
    _consensus_boost = getattr(result, "consensus_boost", 0.0)
    _cb_c = "#22c55e" if _consensus_boost > 0 else "#ef4444" if _consensus_boost < 0 else "#4b5563"
    agree_badge = (
        f"<span style='background:{_cb_c}22;color:{_cb_c};border:1px solid {_cb_c}55;"
        f"border-radius:4px;padding:1px 7px;font-size:10px;'>"
        f"{_agree}</span>"
    ) if _agree else ""

    st.markdown(
        f"<div class='sig-card' style='background:{bg};border:1px solid {border};'>"
        f"<div style='font-size:26px;font-weight:800;color:{color};'>"
        f"{arrow}{pending_badge}{veto_badge}</div>"
        f"<div style='margin:10px 0 4px;background:#1a1a2e;border-radius:4px;height:8px;'>"
        f"<div style='background:{color};width:{conf}%;height:100%;border-radius:4px;'></div></div>"
        f"<div style='font-size:13px;color:#9ca3af;margin-bottom:8px;'>"
        f"Confidence: <b style='color:{color};'>{conf}%</b>"
        f"&nbsp;·&nbsp;Base {int(result.base_confidence*100)}%"
        f"&nbsp;·&nbsp;News boost {result.news_boost:+.0%}"
        + (f"&nbsp;·&nbsp;BT {getattr(result,'bt_adjustment',0):+.0%}" if abs(getattr(result,'bt_adjustment',0)) > 0.001 else "")
        + f"</div>"
        f"<div style='display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px;'>"
        f"{regime_badge} {dt_badge} {agree_badge}"
        f"</div>"
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
                with st.spinner("Sending…"):
                    connected = ensure_connected()
                if connected:
                    res = send_from_signal_result(item["sr"])
                    if res.success:
                        _MT5["sent"].add(item["key"])
                        _MT5["last_sent"] = {**item, "ticket": res.ticket}
                        _log(f"Order #{res.ticket} placed ✓")
                        st.success(f"Order #{res.ticket} sent to MT5!")
                        _MT5["pending"].remove(item)
                        st.rerun()
                    else:
                        st.error(f"MT5 rejected: {res.error}")
                else:
                    st.error("Cannot connect to MT5. Is the terminal open and logged in?")
        with r_col:
            if st.button("❌ Reject", key=f"reject_{item['key']}"):
                _MT5["rejected"].add(item["key"])
                _log(f"Rejected: {direction} {ticker}")
                _MT5["pending"].remove(item)
                st.rerun()

    st.markdown("---")

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_signals, tab_trades, tab_history, tab_journal, tab_learning = st.tabs([
    "📊  Signals",
    "📈  Open Trades",
    "🕐  Past Trades",
    "📓  Journal",
    "🧠  Learning",
])

# ── Tab 1: Signals ────────────────────────────────────────────────────────────
with tab_signals:
    if _MT5["error"] and not _MT5["running"]:
        st.error(_MT5["error"])

    pending_tickers = {p["ticker"] for p in _MT5["pending"]}

    # ── Top Signals — best actionable trades sorted by confidence ─────────────
    active_signals = [
        (inst, results[inst["ticker"]])
        for inst in INSTRUMENTS
        if inst["ticker"] in results
        and results[inst["ticker"]]
        and not results[inst["ticker"]].error
        and results[inst["ticker"]].direction in ("LONG", "SHORT")
        and not _market_status(inst["ticker"], results[inst["ticker"]].df_5m)[0]
    ]
    active_signals.sort(key=lambda x: x[1].confidence, reverse=True)

    if active_signals:
        st.markdown("### 🎯 Top Signals")
        for inst, r in active_signals:
            dir_c  = "#22c55e" if r.direction == "LONG" else "#ef4444"
            arrow  = "▲" if r.direction == "LONG" else "▼"
            bg     = "#0a1f14" if r.direction == "LONG" else "#1f0a0a"
            border = "#16a34a" if r.direction == "LONG" else "#dc2626"
            risk   = abs(r.entry - r.sl)
            reward = abs(r.tp - r.entry)
            rr     = reward / risk if risk > 0 else 0
            strat  = r.chart_signals[0].strategy if r.chart_signals else "—"

            c_info, c_btn = st.columns([5, 1])
            with c_info:
                st.markdown(
                    f"<div style='background:{bg};border:1px solid {border};border-radius:8px;"
                    f"padding:12px 18px;display:flex;gap:24px;align-items:center;flex-wrap:wrap;'>"
                    f"<span style='color:{dir_c};font-size:20px;font-weight:800;min-width:100px;'>"
                    f"{arrow} {r.direction}</span>"
                    f"<span style='color:#e0e0e0;font-size:16px;font-weight:700;'>{inst['label']}"
                    f"<span style='color:#6b7280;font-size:12px;font-weight:400;'> {inst['ticker']}</span></span>"
                    f"<span style='color:{dir_c};font-size:18px;font-weight:700;'>{int(r.confidence*100)}%</span>"
                    f"<span style='color:#9ca3af;font-size:13px;'>Entry <b style='color:#60a5fa;'>{r.entry:.4f}</b>"
                    f"&nbsp; SL <b style='color:#ef4444;'>{r.sl:.4f}</b>"
                    f"&nbsp; TP <b style='color:#22c55e;'>{r.tp:.4f}</b>"
                    f"&nbsp; R:R <b>1:{rr:.1f}</b>"
                    f"&nbsp; · &nbsp;<span style='color:#6b7280;'>{strat}</span></span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with c_btn:
                from utils.mt5_bridge import is_available as _mt5_ok
                if _mt5_ok():
                    if st.button("Send →", key=f"quick_{inst['ticker']}", type="primary"):
                        from utils.mt5_bridge import ensure_connected, send_from_signal_result, get_open_positions as _gop2
                        with st.spinner("Sending…"):
                            connected = ensure_connected()
                        if connected:
                            res = send_from_signal_result(r)
                            if res.success:
                                _MT5["connected"] = True
                                _MT5["last_sent"] = {"ticker": inst["ticker"], "ticket": res.ticket}
                                _MT5["positions"] = _gop2()
                                _log(f"Quick-send order #{res.ticket} ✓")
                                st.success(f"Order #{res.ticket} sent!")
                                st.rerun()
                            else:
                                st.error(f"MT5 rejected: {res.error}")
                        else:
                            st.error("Cannot connect to MT5.\nIs the terminal open and logged in?")
                else:
                    st.caption("Install MT5:\npip install MetaTrader5")
        st.markdown("---")
    else:
        st.info("No active signals right now. Click **Scan Signals Now** to refresh.")

    # ── Instrument grid, grouped by category ──────────────────────────────────
    groups = {}
    for inst in INSTRUMENTS:
        groups.setdefault(inst.get("group", "Other"), []).append(inst)

    for group_name, insts in groups.items():
        st.markdown(f"<div style='color:#6b7280;font-size:11px;font-weight:700;letter-spacing:1px;"
                    f"text-transform:uppercase;margin:16px 0 8px;'>{group_name}</div>",
                    unsafe_allow_html=True)
        cols = st.columns(len(insts))
        for col, inst in zip(cols, insts):
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
    # Always fetch live positions if MT5 package is available — don't rely on
    # the background monitoring loop being active.
    from utils.mt5_bridge import is_available as _mt5_avail, ensure_connected, get_open_positions as _gop
    if _mt5_avail():
        if ensure_connected():
            _MT5["connected"] = True
            _MT5["positions"] = _gop()
        col_ref, _ = st.columns([1, 5])
        with col_ref:
            if st.button("🔄 Refresh", key="refresh_positions"):
                _MT5["positions"] = _gop()
                st.rerun()

    positions = _MT5["positions"]

    if not _mt5_avail():
        st.info("MetaTrader5 package not installed. Run: pip install MetaTrader5")
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

# ── Tab 5: Learning ───────────────────────────────────────────────────────────
with tab_learning:
    from data.trade_outcomes import get_summary, compute_win_rates

    # Allow manual outcome refresh without waiting for the monitoring loop
    col_lref, _ = st.columns([1, 5])
    with col_lref:
        if st.button("🔄 Refresh outcomes", key="refresh_outcomes"):
            try:
                from data.trade_outcomes import update_outcomes_from_mt5
                from utils.mt5_bridge import ensure_connected as _ec
                if _ec():
                    n = update_outcomes_from_mt5()
                    st.success(f"Updated {n} outcome(s).")
                else:
                    st.warning("MT5 not connected.")
            except Exception as exc:
                st.error(str(exc))
            st.rerun()

    summary = get_summary()

    if summary["total_sent"] == 0:
        st.info("No auto-trades recorded yet. Start monitoring and let the app trade — outcomes will appear here as positions close.")
    else:
        # ── Overall stats ─────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        wr = summary["win_rate"]
        wr_str = f"{wr:.0%}" if wr is not None else "—"
        wr_col = "#22c55e" if (wr or 0) >= 0.55 else ("#f59e0b" if (wr or 0) >= 0.45 else "#ef4444")
        profit_col = "#22c55e" if summary["total_profit"] >= 0 else "#ef4444"

        c1.metric("Trades sent", summary["total_sent"])
        c2.metric("Closed", summary["total_closed"],
                  f"{summary['total_open']} open")
        c3.markdown(
            f"<div style='font-size:13px;color:#6b7280;'>Win rate</div>"
            f"<div style='font-size:28px;font-weight:700;color:{wr_col};'>{wr_str}</div>",
            unsafe_allow_html=True,
        )
        c4.markdown(
            f"<div style='font-size:13px;color:#6b7280;'>Total P&L</div>"
            f"<div style='font-size:28px;font-weight:700;color:{profit_col};'>"
            f"{summary['total_profit']:+.2f}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ── Per-strategy performance ──────────────────────────────────────────
        if summary["by_strategy"]:
            st.markdown("### Strategy Performance")
            live_wr = compute_win_rates()  # only strategies with ≥5 trades
            for row in summary["by_strategy"]:
                s     = row["strategy"]
                wr_s  = row["win_rate"]
                col   = "#22c55e" if wr_s >= 0.55 else ("#f59e0b" if wr_s >= 0.45 else "#ef4444")
                p_col = "#22c55e" if row["profit"] >= 0 else "#ef4444"
                calibrated = s in live_wr
                badge = (
                    f"<span style='background:#1e3a1e;color:#22c55e;font-size:10px;"
                    f"padding:2px 6px;border-radius:4px;'>✓ calibrating</span>"
                    if calibrated else
                    f"<span style='background:#2a2a2a;color:#6b7280;font-size:10px;"
                    f"padding:2px 6px;border-radius:4px;'>need {5-row['trades']} more trades</span>"
                )
                st.markdown(
                    f"<div style='background:#111827;border-radius:8px;padding:12px 18px;"
                    f"margin-bottom:6px;border:1px solid #1e2330;'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                    f"<div>"
                    f"<span style='color:#e0e0e0;font-weight:700;'>{s}</span> &nbsp; {badge}"
                    f"</div>"
                    f"<div style='display:flex;gap:24px;text-align:right;'>"
                    f"<div><div style='font-size:11px;color:#6b7280;'>Trades</div>"
                    f"<div style='font-weight:700;'>{row['trades']}</div></div>"
                    f"<div><div style='font-size:11px;color:#6b7280;'>Win rate</div>"
                    f"<div style='font-weight:700;color:{col};'>{wr_s:.0%}</div></div>"
                    f"<div><div style='font-size:11px;color:#6b7280;'>P&L</div>"
                    f"<div style='font-weight:700;color:{p_col};'>{row['profit']:+.2f}</div></div>"
                    f"</div></div></div>",
                    unsafe_allow_html=True,
                )
            if live_wr:
                st.caption(
                    "Strategies marked **✓ calibrating** are actively adjusting signal confidence "
                    "based on their real win rate. A strategy below 62% win rate gets penalised up to −10%; "
                    "above 62% it gets boosted up to +10%."
                )

        # ── Trade log ─────────────────────────────────────────────────────────
        with st.expander("Full trade log", expanded=False):
            for t in summary["all_trades"][:50]:
                outcome = t.get("outcome") or "open"
                o_col = {"WIN": "#22c55e", "LOSS": "#ef4444",
                         "BREAKEVEN": "#f59e0b", "open": "#6b7280"}.get(outcome, "#6b7280")
                profit_str = f"{t['profit']:+.2f}" if t.get("profit") is not None else "—"
                st.markdown(
                    f"<div style='font-size:12px;padding:4px 0;border-bottom:1px solid #1e2330;'>"
                    f"<span style='color:{o_col};font-weight:700;min-width:80px;display:inline-block;'>"
                    f"{outcome.upper()}</span>"
                    f"<span style='color:#9ca3af;'>{t['direction']} {t['ticker']} "
                    f"· {t['strategy']} · conf {t['confidence']:.0%} "
                    f"· P&L <b style='color:{o_col};'>{profit_str}</b> "
                    f"· #{t['ticket']} · {t['sent_at'][:16]}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ── Auto-refresh while monitoring (5s — fast enough for approvals, not wasteful)
if _MT5["running"]:
    time.sleep(5)
    st.rerun()
