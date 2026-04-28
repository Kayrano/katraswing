"""
KATRASWING — MT5 Signal Dashboard
Run: streamlit run app.py
Requires MetaTrader5 terminal open on the same Windows machine.
"""

import logging
import threading
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import streamlit as st

log = logging.getLogger(__name__)

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
            _BT_CACHE["rates"][ticker] = rates if rates else {}
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
        "account_size": 100_000.0,
        # Trade Manager shared state — written by main thread, read by background thread
        "auto_assess": True, "live_mode": True,
        "tm_cooldown": {}, "trade_assessments": [],
    }
_MT5: dict = st.session_state["_MT5"]


def _log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    _MT5["log"].append(f"{ts}  {msg}")
    _MT5["log"] = _MT5["log"][-200:]


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
    try:
        from utils.mt5_bridge import get_account_info as _gai_bg
        _ai = _gai_bg()
        if _ai and _ai.get("balance"):
            _MT5["account_size"] = float(_ai["balance"])
    except Exception as e:
        log.warning("ctx=mt5_get_account_info_bg: %s", e)

    instruments  = config["instruments"]   # list of {ticker, label, mt5_symbol}
    min_conf     = config["min_conf"]
    account_size = config["account_size"]
    risk_pct     = config["risk_pct"]
    finnhub_key  = config["finnhub_key"]
    auto_trade   = config.get("auto_trade", True)
    use_daily    = config.get("use_daily", True)
    _dt_cache:  dict[str, tuple[dict, float]] = {}
    _h4_cache:  dict[str, tuple[dict, float]] = {}
    _opt_stops: dict = {}
    _opt_stops_ts: float = 0.0

    while not stop_event.is_set():
        today = date.today()
        _MT5["sent"]     = {k for k in _MT5["sent"]     if k.endswith(str(today))}
        _MT5["rejected"] = {k for k in _MT5["rejected"] if k.endswith(str(today))}

        # Refresh optimal stops once per hour
        if time.time() - _opt_stops_ts > 3600:
            try:
                from data.trade_outcomes import compute_optimal_stops
                _opt_stops = compute_optimal_stops()
                _opt_stops_ts = time.time()
            except Exception as e:
                log.warning("ctx=refresh_optimal_stops_bg: %s", e)

        _log(f"── Scanning {len(instruments)} instruments ──")
        for inst in instruments:
            if stop_event.is_set():
                break
            ticker     = inst["ticker"]
            mt5_symbol = inst.get("mt5_symbol")

            daily_trend = None
            h4_trend    = None
            if use_daily:
                try:
                    from data.fetcher_intraday import fetch_daily_trend, fetch_h4_trend
                    # Daily — 15-min cache
                    cached = _dt_cache.get(ticker)
                    if cached is None or time.time() - cached[1] > 900:
                        daily_trend = fetch_daily_trend(ticker)
                        _dt_cache[ticker] = (daily_trend, time.time())
                    else:
                        daily_trend = cached[0]
                    # H4 — 30-min cache
                    cached_h4 = _h4_cache.get(ticker)
                    if cached_h4 is None or time.time() - cached_h4[1] > 1800:
                        h4_trend = fetch_h4_trend(ticker, mt5_symbol=mt5_symbol)
                        _h4_cache[ticker] = (h4_trend, time.time())
                    else:
                        h4_trend = cached_h4[0]
                except Exception as e:
                    log.warning("ctx=mtf_trends_bg ticker=%s: %s", ticker, e)

            try:
                sr = run_signal(
                    ticker=ticker,
                    finnhub_api_key=finnhub_key,
                    account_size=account_size,
                    risk_pct=risk_pct,
                    daily_trend=daily_trend,
                    h4_trend=h4_trend,
                    optimal_stops=_opt_stops or None,
                    live_win_rates=_MT5.get("live_win_rates") or None,
                    mt5_symbol=mt5_symbol,
                )
            except Exception as exc:
                _log(f"ERROR {ticker}: {exc}")
                continue

            if sr.error:
                _log(f"  {ticker} ERROR: {sr.error}")
                continue
            if sr.direction not in ("LONG", "SHORT"):
                _log(f"  {ticker} — FLAT")
                continue
            if sr.confidence < min_conf:
                _log(f"  {ticker} — {sr.direction} {sr.confidence:.0%} (below {min_conf:.0%})")
                continue

            key = f"{ticker}:{sr.direction}:{today}"
            if key in _MT5["sent"] or key in _MT5["rejected"]:
                continue

            strategy_name = sr.chart_signals[0].strategy if sr.chart_signals else "UNKNOWN"
            _log(f"  {ticker} → {strategy_name} {sr.direction} {sr.confidence:.0%} | SL {sr.sl} TP {sr.tp}")

            # Skip auto-send if no broker mapping yet (early connect race)
            if auto_trade and not (mt5_symbol or "").strip():
                _log(f"  ⏳ {ticker} signal held — waiting for broker symbol mapping")
                continue

            if auto_trade:
                try:
                    from utils.mt5_bridge import send_from_signal_result
                    res = send_from_signal_result(sr, risk_pct=config.get("risk_pct", 1.0))
                    if res.success:
                        _MT5["sent"].add(key)
                        _MT5["last_sent"] = {"ticker": ticker, "ticket": res.ticket,
                                             "direction": sr.direction, "conf": sr.confidence}
                        _log(f"✅ Signal sent: #{res.ticket} {sr.direction} {ticker} {sr.confidence:.0%}")
                        try:
                            from data.trade_outcomes import record_trade
                            record_trade(res.ticket, ticker, strategy_name, sr.direction,
                                         sr.confidence, sr.entry, sr.sl, sr.tp)
                        except Exception as _rte:
                            _log(f"⚠ record_trade #{res.ticket}: {_rte}")
                    else:
                        _log(f"⚠ Rejected: {res.error}")
                except Exception as exc:
                    _log(f"⚠ Error: {exc}")

        try:
            _MT5["positions"] = get_open_positions()
        except Exception as e:
            log.warning("ctx=get_open_positions_bg: %s", e)

        try:
            from data.trade_outcomes import update_outcomes_from_mt5, compute_detailed_win_rates
            updated = update_outcomes_from_mt5()
            if updated:
                _log(f"📚 {updated} outcome(s) recorded")
            _MT5["live_win_rates"] = compute_detailed_win_rates()
        except Exception as e:
            log.warning("ctx=update_outcomes_bg: %s", e)

        # Auto-assess open trades when both toggles are ON
        # Reads/writes _MT5 dict only — no st.session_state access from background thread
        try:
            _auto_assess = _MT5.get("auto_assess", False)
            _live_mode   = _MT5.get("live_mode",   False)
            if _auto_assess and _MT5["positions"]:
                from agents.trade_manager import assess_all_open_trades as _atm
                _cd = _MT5.get("tm_cooldown", {})
                _res = _atm(
                    _MT5["positions"],
                    finnhub_key=config.get("finnhub_key", ""),
                    account_size=config.get("account_size", 100_000.0),
                    risk_pct=config.get("risk_pct", 1.0),
                    use_daily=config.get("use_daily", True),
                    cooldown_state=_cd,
                    dry_run=not _live_mode,
                )
                _MT5["trade_assessments"] = _res
                _MT5["tm_cooldown"] = _cd
                for _a in _res:
                    _pl = f"P&L ${_a.current_profit:+.2f}" if hasattr(_a, "current_profit") and _a.current_profit is not None else ""
                    _hs = f"health={_a.health_score:.2f}" if hasattr(_a, "health_score") and _a.health_score is not None else ""
                    if getattr(_a, "error", None):
                        _log(f"⚠ TM #{_a.ticket} {_a.symbol}: {_a.error}")
                    elif getattr(_a, "acted_on", False):
                        _log(f"🤖 TM #{_a.ticket} {_a.symbol} {_a.action} [{_a.urgency}] {_hs} {_pl}")
                    elif getattr(_a, "action", "HOLD") != "HOLD":
                        _log(f"⚠ TM #{_a.ticket} {_a.symbol} {_a.action} [{_a.urgency}] {_hs} — not executed")
                    else:
                        _log(f"  TM #{_a.ticket} {_a.symbol} HOLD {_hs} {_pl}")
        except Exception as _atm_exc:
            _log(f"Auto-assess error: {_atm_exc}")

        _MT5["last_check"] = datetime.now()
        # 5-min cycle when positions are open (timely TM execution), 15-min otherwise
        stop_event.wait(300 if _MT5["positions"] else 900)

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
    """Return fully paired open+close trade records from MT5 history."""
    try:
        import MetaTrader5 as mt5
        from collections import defaultdict
        from_dt = datetime.now() - timedelta(days=days)
        deals   = mt5.history_deals_get(from_dt, datetime.now())
        if deals is None:
            return []

        by_pos = defaultdict(lambda: {"in": None, "out": None})
        for d in deals:
            if d.entry == 0:
                by_pos[d.position_id]["in"]  = d
            elif d.entry == 1:
                by_pos[d.position_id]["out"] = d

        rows = []
        for pos_id, pair in by_pos.items():
            out_d = pair["out"]
            in_d  = pair["in"]
            if out_d is None:
                continue
            gross     = float(out_d.profit)
            comm      = float(getattr(out_d, "commission", 0.0))
            swap      = float(getattr(out_d, "swap",       0.0))
            net       = round(gross + comm + swap, 2)
            direction = "LONG" if (in_d and in_d.type == 0) else "SHORT"
            entry_p   = float(in_d.price) if in_d else 0.0
            open_ts   = in_d.time  if in_d else out_d.time
            dur_m     = int((out_d.time - open_ts) / 60)
            rows.append({
                "ticket":     int(pos_id),
                "symbol":     str(out_d.symbol),
                "type":       "BUY" if direction == "LONG" else "SELL",
                "direction":  direction,
                "volume":     float(out_d.volume),
                "entry":      round(entry_p,         5),
                "exit":       round(float(out_d.price), 5),
                "profit":     net,
                "gross":      round(gross, 2),
                "commission": round(comm,  2),
                "swap":       round(swap,  2),
                "open_time":  datetime.fromtimestamp(open_ts).strftime("%Y-%m-%d %H:%M"),
                "close_time": datetime.fromtimestamp(out_d.time).strftime("%Y-%m-%d %H:%M"),
                "date":       datetime.fromtimestamp(out_d.time).strftime("%Y-%m-%d"),
                "duration_m": dur_m,
                "comment":    str(getattr(out_d, "comment", "")),
            })
        return sorted(rows, key=lambda x: x["close_time"], reverse=True)
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
def _get_broker_symbols(_connected: bool = False) -> list[dict]:
    """Fetch available symbols from MT5 broker (cached 5 min, keyed by connection state)."""
    if not _connected:
        return []
    try:
        from utils.mt5_bridge import get_tradeable_symbols
        return get_tradeable_symbols()
    except Exception:
        return []


# ── Hardcoded parameters ──────────────────────────────────────────────────────
finnhub_key  = "d7j16r1r01qn2qavovt0d7j16r1r01qn2qavovtg"
account_size = _MT5.get("account_size", 100_000.0)
risk_pct     = 1.0
min_conf     = 0.65
auto_trade   = True
use_daily    = True
use_bt_cal   = True
auto_refresh = True

# Curated instrument list — matched against broker symbols when MT5 is connected,
# or used directly with yfinance tickers when MT5 is offline.
_CURATED = [
    # Forex
    {"label": "EUR/USD", "cands": ["EURUSD"],                                                          "yf": "EURUSD=X"},
    {"label": "GBP/USD", "cands": ["GBPUSD"],                                                          "yf": "GBPUSD=X"},
    {"label": "USD/JPY", "cands": ["USDJPY"],                                                          "yf": "USDJPY=X"},
    {"label": "EUR/JPY", "cands": ["EURJPY"],                                                          "yf": "EURJPY=X"},
    {"label": "GBP/JPY", "cands": ["GBPJPY"],                                                          "yf": "GBPJPY=X"},
    {"label": "AUD/USD", "cands": ["AUDUSD"],                                                          "yf": "AUDUSD=X"},
    {"label": "USD/CAD", "cands": ["USDCAD"],                                                          "yf": "USDCAD=X"},
    # Indices
    {"label": "US30",    "cands": ["#US30_M26","#US30_M27","#US30","US30","DJ30","DOWJONES"],           "yf": "YM=F"},
    {"label": "NAS100",  "cands": ["#US100_M26","#US100_M27","#US100","NAS100","US100","USTEC"],        "yf": "NQ=F"},
    {"label": "SPX500",  "cands": ["#US500_M26","#US500_M27","#US500","SPX500","US500","SP500"],        "yf": "ES=F"},
    {"label": "GER40",   "cands": ["#GER40_M26","#GER40_M27","#GER40","GER40","DAX40","GER30"],        "yf": "^GDAXI"},
    {"label": "UK100",   "cands": ["#UK100_M26","#UK100_M27","#UK100","UK100","FTSE100"],              "yf": "^FTSE"},
    # Crypto
    {"label": "BTC/USD", "cands": ["BTCUSD","BTCUSDT","BTC/USD","BTCUSD.","#BTCUSD"],                  "yf": "BTC-USD"},
    {"label": "ETH/USD", "cands": ["ETHUSD","ETHUSDT","ETH/USD","ETHUSD.","#ETHUSD"],                  "yf": "ETH-USD"},
    {"label": "SOL/USD", "cands": ["SOLUSD","SOLUSDT","SOL/USD","SOLUSD."],                            "yf": "SOL-USD"},
    {"label": "BNB/USD", "cands": ["BNBUSD","BNBUSDT","BNB/USD","BNBUSD."],                            "yf": "BNB-USD"},
    # Commodities
    {"label": "XAU/USD", "cands": ["XAUUSD","GOLD","XAU/USD","XAUUSD."],                               "yf": "GC=F"},
    {"label": "XAG/USD", "cands": ["XAGUSD","SILVER","XAG/USD","XAGUSD."],                             "yf": "SI=F"},
    {"label": "US Oil",  "cands": ["XTIUSD","USOIL","WTI","OIL","XOILUSD","CL"],                      "yf": "CL=F"},
    {"label": "Nat Gas", "cands": ["XNGUSD","NGAS","NATGAS","GASUSD","NG","NATURALGAS"],               "yf": "NG=F"},
]


def _resolve_instruments(broker_syms: list[dict]) -> list[dict]:
    """Match curated instruments against available broker symbols.
    Returns list of {ticker, label, mt5_symbol}."""
    from data.fetcher_intraday import _MT5_TO_YF
    broker_names = {s["name"] for s in broker_syms}
    broker_upper = {s["name"].upper(): s["name"] for s in broker_syms}

    result = []
    for inst in _CURATED:
        mt5_sym = None
        for cand in inst["cands"]:
            # Exact match
            if cand in broker_names:
                mt5_sym = cand
                break
            # Case-insensitive exact
            exact_ci = broker_upper.get(cand.upper())
            if exact_ci:
                mt5_sym = exact_ci
                break
            # Prefix match: broker symbol starts with candidate + non-alpha (e.g. "EURUSD.r")
            for bn_upper, bn_orig in broker_upper.items():
                if bn_upper.startswith(cand.upper()) and (
                    len(bn_upper) == len(cand) or not bn_upper[len(cand)].isalpha()
                ):
                    mt5_sym = bn_orig
                    break
            if mt5_sym:
                break

        if mt5_sym:
            yf_t = _MT5_TO_YF.get(mt5_sym.upper(),
                   _MT5_TO_YF.get(mt5_sym.split("_")[0].upper(), inst["yf"]))
            result.append({"ticker": yf_t, "label": inst["label"], "mt5_symbol": mt5_sym})
        else:
            # Not in broker — analysis works via yfinance, but mt5_symbol is empty
            # so send_from_signal_result skips the order (no broker mapping yet).
            result.append({"ticker": inst["yf"], "label": inst["label"], "mt5_symbol": ""})

    return result


# Build instruments list: curated list matched against broker when MT5 is running
_all_broker_syms = _get_broker_symbols(_connected=_MT5["connected"]) if _MT5["running"] else []
_resolved = _resolve_instruments(_all_broker_syms) if _all_broker_syms else [
    {"ticker": i["yf"], "label": i["label"], "mt5_symbol": ""} for i in _CURATED
]

# Apply saved user selection (labels used as keys — broker-agnostic)
_saved_labels = st.session_state.get("_curated_sel")
if _saved_labels:
    instruments = [i for i in _resolved if i["label"] in _saved_labels]
    if not instruments:   # saved selection wiped out (e.g. new curated list) — reset
        instruments = _resolved
else:
    instruments = _resolved

# ── Handle MT5 action (start/stop) ────────────────────────────────────────────
_mt5_action = st.session_state.pop("_mt5_action", None)
if _mt5_action:
    if _mt5_action["action"] == "start":
        _start_mt5(_mt5_action["cfg"])
    else:
        _stop_mt5()
    st.rerun()

# ── Auto-start MT5 thread on first load ──────────────────────────────────────
if not _MT5["running"] and not st.session_state.get("_mt5_autostart_attempted"):
    st.session_state["_mt5_autostart_attempted"] = True
    from utils.mt5_bridge import is_available as _mt5_avail_check
    if _mt5_avail_check():
        _start_mt5({
            "instruments": instruments,
            "min_conf":     min_conf,
            "account_size": account_size,
            "risk_pct":     risk_pct,
            "finnhub_key":  finnhub_key,
            "interval":     900,
            "auto_trade":   True,
            "use_daily":    use_daily,
        })

# ── Header ────────────────────────────────────────────────────────────────────
h_left, h_mid, h_acct, h_right = st.columns([2, 3, 3, 1])
with h_left:
    st.markdown("# ⚡ Katraswing")

with h_mid:
    if _MT5["running"] and _MT5["connected"]:
        lc = _MT5["last_check"].strftime("%H:%M") if _MT5["last_check"] else "—"
        mode = "🤖 AUTO"
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

with h_acct:
    if _MT5["connected"]:
        from utils.mt5_bridge import get_account_info as _gai
        _ai = _gai()
        if _ai:
            bal = _ai.get("balance", 0)
            eq  = _ai.get("equity",  0)
            fm  = _ai.get("free_margin", 0)
            ml  = _ai.get("margin_level")
            ml_color = ("#22c55e" if (ml or 0) > 300
                        else "#f59e0b" if (ml or 0) > 150
                        else "#ef4444")
            ml_str = f"{ml:.0f}%" if ml else "—"
            st.markdown(
                f"<div style='padding-top:16px;font-size:12px;line-height:1.6;'>"
                f"<span style='color:#6b7280;'>Balance</span> "
                f"<b style='color:#e0e0e0;'>{bal:,.2f}</b>"
                f" <span style='color:#6b7280;'>{_ai.get('currency','')}</span>"
                f"&nbsp;·&nbsp;"
                f"<span style='color:#6b7280;'>Free</span> "
                f"<b style='color:#60a5fa;'>{fm:,.2f}</b><br>"
                f"<span style='color:#6b7280;'>Equity</span> "
                f"<b style='color:#e0e0e0;'>{eq:,.2f}</b>"
                f"&nbsp;·&nbsp;"
                f"<span style='color:#6b7280;'>Margin lvl</span> "
                f"<b style='color:{ml_color};'>{ml_str}</b>"
                f"</div>",
                unsafe_allow_html=True)

with h_right:
    st.markdown("<div style='padding-top:12px;'>", unsafe_allow_html=True)
    scan_btn = st.button("🔄 Scan", type="primary", width='stretch')
    st.markdown("</div>", unsafe_allow_html=True)

# ── Instruments strip + Live Log Console ──────────────────────────────────────
from utils.mt5_bridge import is_available as _mt5_avail

_ctrl_inst, _ctrl_start, _ctrl_stop, _ctrl_close, _ctrl_clear = st.columns([4, 1, 1, 1, 1])

with _ctrl_inst:
    _all_labels = [i["label"] for i in _CURATED]
    _cur_labels = st.session_state.get("_curated_sel", _all_labels)
    _cur_labels = [l for l in _cur_labels if l in _all_labels] or _all_labels
    _sel_labels = st.multiselect(
        "Instruments",
        options=_all_labels,
        default=_cur_labels,
        label_visibility="collapsed",
    )
    st.session_state["_curated_sel"] = _sel_labels
    if _sel_labels:
        instruments = [i for i in _resolved if i["label"] in _sel_labels]

with _ctrl_start:
    if not _MT5["running"]:
        if st.button("▶ Start", type="primary", width='stretch'):
            st.session_state["_mt5_action"] = {
                "action": "start",
                "cfg": {
                    "instruments": instruments,
                    "min_conf":     min_conf,
                    "account_size": account_size,
                    "risk_pct":     risk_pct,
                    "finnhub_key":  finnhub_key,
                    "interval":     900,
                    "auto_trade":   True,
                    "use_daily":    use_daily,
                },
            }

with _ctrl_stop:
    if _MT5["running"]:
        if st.button("⏹ Stop", width='stretch'):
            st.session_state["_mt5_action"] = {"action": "stop"}

with _ctrl_close:
    if _MT5["connected"] and st.button("🚨 Close All", width='stretch'):
        from utils.mt5_bridge import close_all_positions
        close_all_positions()
        st.success("All positions closed.")

with _ctrl_clear:
    if st.button("🗑 Clear", width='stretch'):
        _MT5["log"] = []
        st.rerun()

_mt5_status_txt = ("✅ MT5 connected" if _MT5["connected"]
                   else ("⏳ Connecting…" if _MT5["running"]
                         else "⚠ MT5 not started"))
st.caption(_mt5_status_txt)

# Live Activity Log
_log_entries = list(reversed(_MT5["log"][-100:]))
_parts = []
for _e in _log_entries:
    _ts  = _e[:8]
    _msg = _e[10:] if len(_e) > 10 else _e
    _msg_esc = _msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    if   "✅" in _msg or "sent" in _msg.lower(): _col = "#22c55e"
    elif "🤖" in _msg:                            _col = "#60a5fa"
    elif "⚠" in _msg or "ERROR" in _msg:         _col = "#f59e0b"
    elif "🚨" in _msg or "CLOSE" in _msg:         _col = "#ef4444"
    elif "──" in _msg:                            _col = "#374151"
    else:                                          _col = "#9ca3af"
    _parts.append(
        f"<span style='color:#4b5563'>[{_ts}]</span> "
        f"<span style='color:{_col}'>{_msg_esc}</span>"
    )
st.markdown(
    "<div style='background:#0d1117;border-radius:6px;padding:10px 14px;"
    "font-size:11px;font-family:monospace;line-height:1.8;"
    "max-height:280px;overflow-y:auto;border:1px solid #1e2330;margin-top:4px;'>"
    + "<br>".join(_parts or ["<span style='color:#374151'>Waiting for activity…</span>"])
    + "</div>",
    unsafe_allow_html=True,
)

# ── Strategy Learning panel ───────────────────────────────────────────────────
with st.expander("🎓 Strategy Learning", expanded=False):
    try:
        import pandas as _pd_sl
        from data.strategy_params import get_all_params, adapt_all as _adapt_all_now
        from data.trade_outcomes import _load as _load_trades
        # Sync stats from app-managed trades only (MT5_IMPORT excluded inside adapt_all)
        _adapt_all_now(_load_trades())
        _sp = get_all_params()
        _rows = []
        for _sname, _p in _sp.items():
            _wr = _p.get("win_rate")
            _rows.append({
                "Strategy":   _sname,
                "Win %":      f"{_wr:.0%}" if _wr is not None else "—",
                "SL ×":       f"{_p.get('sl_mult', 1.0):.2f}",
                "TP ×":       f"{_p.get('tp_mult', 1.0):.2f}",
                "Conf floor": f"{_p.get('conf_floor', 0.60):.2f}",
                "Trades":     _p.get("trades_seen", 0),
                "Status":     ("🔴 Disabled" if not _p.get("enabled", True)
                               else ("🟢 Active" if (_wr or 0) >= 0.50
                                     else ("🟡 Learning" if _p.get("trades_seen", 0) >= 5
                                           else "🔵 New"))),
            })
        st.dataframe(_pd_sl.DataFrame(_rows).set_index("Strategy"), width='stretch')
        if any(not _p.get("enabled", True) for _p in _sp.values()):
            st.caption("🔴 Disabled strategies produce no signals until win-rate recovers above 50%.")
    except Exception as _sle:
        st.caption(f"Learning panel unavailable: {_sle}")

# ── Persist selection ─────────────────────────────────────────────────────────
st.session_state.update({"instruments": instruments})

# ── Sync TM toggle state into _MT5 on every render ────────────────────────────
# Must run outside any tab block so the background thread always gets current values.
_MT5["auto_assess"] = st.session_state.get("tm_auto_assess", True)
_MT5["live_mode"]   = st.session_state.get("tm_live_mode",   True)

# ── Auto-refresh trigger ──────────────────────────────────────────────────────
if auto_refresh:
    if time.time() - st.session_state.get("last_refresh_ts", 0) > 300:
        st.session_state.pop("results", None)

needs_run = (scan_btn or ("results" not in st.session_state)) and bool(instruments)

# ── Signal scan (parallel) ────────────────────────────────────────────────────
if not instruments:
    st.info("⏳ Waiting for MT5 broker instruments to load… Start Auto-Trade if not running.")

if needs_run:
    from agents.signal_engine import run_signal
    from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as _FutureTimeout

    def _refresh_daily(ticker):
        k_val, k_ts = f"_dt_{ticker}", f"_dt_ts_{ticker}"
        if time.time() - st.session_state.get(k_ts, 0) > 900:
            try:
                from data.fetcher_intraday import fetch_daily_trend
                trend = fetch_daily_trend(ticker)
                st.session_state[k_val] = trend
                st.session_state[k_ts]  = time.time()
            except Exception as e:
                log.warning("ctx=refresh_daily ticker=%s: %s", ticker, e)
                st.session_state[k_val] = None
                st.session_state[k_ts]  = time.time()
        return st.session_state.get(k_val)

    def _refresh_h4(ticker, mt5_symbol=None):
        k_val, k_ts = f"_h4_{ticker}", f"_h4_ts_{ticker}"
        if time.time() - st.session_state.get(k_ts, 0) > 1800:
            try:
                from data.fetcher_intraday import fetch_h4_trend
                trend = fetch_h4_trend(ticker, mt5_symbol=mt5_symbol)
                st.session_state[k_val] = trend
                st.session_state[k_ts]  = time.time()
            except Exception as e:
                log.warning("ctx=refresh_h4 ticker=%s: %s", ticker, e)
                st.session_state[k_val] = None
                st.session_state[k_ts]  = time.time()
        return st.session_state.get(k_val)

    # Load live win rates and learned optimal stops once for all instruments
    _live_wr: dict = {}
    _opt_stops: dict = {}
    try:
        from data.trade_outcomes import compute_detailed_win_rates, compute_optimal_stops
        _live_wr   = compute_detailed_win_rates()
        _opt_stops = compute_optimal_stops()
    except Exception as e:
        log.warning("ctx=load_live_wr_and_optimal_stops: %s", e)

    # Pre-fetch inputs on main thread
    ticker_inputs = {}
    for inst in instruments:
        t   = inst["ticker"]
        sym = inst.get("mt5_symbol")
        ticker_inputs[t] = {
            "inst":     inst,
            "daily":    _refresh_daily(t) if use_daily else None,
            "h4":       _refresh_h4(t, mt5_symbol=sym) if use_daily else None,
            "bt_rates": _refresh_backtest_rates(t) if use_bt_cal else None,
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
            h4_trend=inp["h4"],
            optimal_stops=_opt_stops or None,
            backtest_win_rates=inp["bt_rates"],
            live_win_rates=_live_wr or None,
            mt5_symbol=inst.get("mt5_symbol"),
        )

    results = {}
    _MAX_WORKERS    = 4
    _PER_TICKER_SEC = 30
    # Scale outer budget so it can never trip before per-future timeouts do.
    # ceil(N / workers) batches × per-ticker cap + 30s buffer.
    _outer_timeout = max(
        120,
        _PER_TICKER_SEC * ((len(instruments) + _MAX_WORKERS - 1) // _MAX_WORKERS) + 30,
    )
    with st.spinner(f"Scanning {len(instruments)} instruments…"):
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
            _fs = {ex.submit(_scan_one, i["ticker"]): i["ticker"] for i in instruments}
            from agents.signal_engine import SignalResult as _SR
            try:
                for _fut in as_completed(_fs, timeout=_outer_timeout):
                    try:
                        _tk, _sr = _fut.result(timeout=_PER_TICKER_SEC)
                        results[_tk] = _sr
                    except _FutureTimeout:
                        _tk = _fs[_fut]
                        results[_tk] = _SR(
                            ticker=_tk,
                            error=f"Scan timed out after {_PER_TICKER_SEC}s",
                        )
                    except Exception as _exc:
                        _tk = _fs[_fut]
                        results[_tk] = _SR(ticker=_tk, error=str(_exc))
            except _FutureTimeout:
                # Outer budget exhausted — record any futures that never finished
                # so the UI shows partial results instead of a Python traceback.
                _unfinished = [
                    _f for _f in _fs if _fs[_f] not in results and not _f.done()
                ]
                log.warning(
                    "ctx=manual_scan_outer_timeout budget=%ds unfinished=%d/%d",
                    _outer_timeout, len(_unfinished), len(instruments),
                )
                for _f, _tk in _fs.items():
                    if _tk in results:
                        continue
                    if _f.done():
                        try:
                            _tkr, _sr = _f.result(timeout=0)
                            results[_tkr] = _sr
                        except Exception as _exc:
                            results[_tk] = _SR(ticker=_tk, error=str(_exc))
                    else:
                        _f.cancel()
                        results[_tk] = _SR(
                            ticker=_tk, error="Scan timed out (server busy)",
                        )

    st.session_state["results"] = results
    st.session_state["last_refresh_ts"] = time.time()

    # ── Auto-send qualifying signals immediately after manual scan ────────────
    if auto_trade and _MT5["connected"]:
        today = date.today()
        from utils.mt5_bridge import send_from_signal_result as _send
        for inst in instruments:
            t  = inst["ticker"]
            sr = results.get(t)
            if sr is None or sr.error or sr.direction not in ("LONG", "SHORT"):
                continue
            if sr.confidence < min_conf:
                continue
            if not (inst.get("mt5_symbol") or "").strip():
                _log(f"⏳ {t} held — no broker symbol yet")
                continue
            key = f"{t}:{sr.direction}:{today}"
            if key in _MT5["sent"] or key in _MT5["rejected"]:
                continue
            try:
                res = _send(sr, risk_pct=risk_pct)
                if res.success:
                    _MT5["sent"].add(key)
                    _MT5["last_sent"] = {"ticker": t, "ticket": res.ticket,
                                         "direction": sr.direction, "conf": sr.confidence}
                    _log(f"✅ Auto #{res.ticket} {sr.direction} {t} {sr.confidence:.0%}")
                    try:
                        from data.trade_outcomes import record_trade
                        strat = sr.chart_signals[0].strategy if sr.chart_signals else "UNKNOWN"
                        record_trade(res.ticket, t, strat, sr.direction,
                                     sr.confidence, sr.entry, sr.sl, sr.tp)
                    except Exception as _rte:
                        _log(f"⚠ record_trade #{res.ticket}: {_rte}")
                else:
                    _MT5["rejected"].add(key)
                    _log(f"⚠ Auto-send {t}: {res.error}")
            except Exception as exc:
                _log(f"⚠ Auto-send error {t}: {exc}")

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
                _rl = getattr(sr, "risk_level", "MEDIUM")
                _rl_color = {"LOW": "#22c55e", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"}.get(_rl, "#6b7280")
                _rl_mult  = {"LOW": "1.5×", "MEDIUM": "1.0×", "HIGH": "0.5×"}.get(_rl, "1.0×")
                risk_badge = (f"<span style='background:#1e2330;color:{_rl_color};"
                              f"font-size:10px;padding:1px 5px;border-radius:3px;"
                              f"margin-left:6px;'>{_rl} RISK {_rl_mult}</span>")
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
                    f"<span style='color:#9ca3af;font-size:12px;'>{strat}{adx_badge}{risk_badge}{veto}</span>"
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
                            res = send_from_signal_result(sr, risk_pct=risk_pct)
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
                                except Exception as _rte:
                                    _log(f"⚠ record_trade #{res.ticket}: {_rte}")
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

        c_ref, c_analyze, c_livemode, c_autoassess, _ = st.columns([1, 1, 1, 1, 2])
        with c_ref:
            if st.button("🔄 Refresh", key="refresh_positions"):
                from utils.mt5_bridge import ensure_connected
                if ensure_connected():
                    _MT5["connected"] = True
                    _MT5["positions"] = _gop2()
                    st.session_state["_pos_fetch_ts"] = time.time()
                st.rerun()
        with c_analyze:
            if st.button("🔍 Analyze Trades", key="analyze_trades", type="primary"):
                _positions = _MT5["positions"]
                if _positions:
                    with st.spinner(f"Analyzing {len(_positions)} position(s)…"):
                        try:
                            from agents.trade_manager import assess_all_open_trades
                            _cooldown = _MT5.get("tm_cooldown", {})
                            _assessments = assess_all_open_trades(
                                _positions,
                                finnhub_key=finnhub_key,
                                account_size=account_size,
                                risk_pct=risk_pct,
                                use_daily=use_daily,
                                cooldown_state=_cooldown,
                                dry_run=True,
                            )
                            _MT5["trade_assessments"] = _assessments
                            _MT5["tm_cooldown"] = _cooldown
                        except Exception as _ex:
                            st.error(f"Analysis error: {_ex}")
                else:
                    st.info("No open positions to analyze.")
        with c_livemode:
            tm_live = st.toggle("⚡ Live Mode", value=True, key="tm_live_mode",
                                help="When ON, Trade Manager executes actions automatically")
        with c_autoassess:
            tm_auto = st.checkbox("Auto-assess", value=True, key="tm_auto_assess",
                                  help="Trade Manager runs on every scan cycle and acts autonomously")

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

        # Map assessments by ticket for quick lookup
        _assessments_by_ticket: dict = {
            a.ticket: a
            for a in (_MT5.get("trade_assessments") or [])
        }

        for p in positions:
            dir_c = "#22c55e" if p.direction == "LONG" else "#ef4444"
            pnl_c = "#22c55e" if p.profit >= 0 else "#ef4444"
            arrow = "▲" if p.direction == "LONG" else "▼"
            st.markdown(
                f"<div style='background:#111827;border-radius:8px;padding:12px 18px;"
                f"margin-bottom:4px;border:1px solid #1e2330;"
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

            # ── Assessment card (shown when analysis has been run) ─────────────
            assessment = _assessments_by_ticket.get(p.ticket)
            if assessment:
                _a = assessment
                health = _a.health_score
                if health > 0.70:
                    h_color, h_label = "#22c55e", "HEALTHY"
                elif health > 0.50:
                    h_color, h_label = "#eab308", "FAIR"
                elif health > 0.30:
                    h_color, h_label = "#f97316", "WEAK"
                else:
                    h_color, h_label = "#ef4444", "CRITICAL"

                action_colors = {
                    "HOLD": "#6b7280", "CLOSE": "#ef4444",
                    "PARTIAL_CLOSE": "#f97316", "MODIFY_SL": "#eab308",
                    "MODIFY_TP": "#3b82f6", "MODIFY_BOTH": "#8b5cf6",
                }
                urgency_dots = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
                act_color = action_colors.get(_a.action, "#6b7280")
                urg_dot   = urgency_dots.get(_a.urgency, "")

                # Health bar
                bar_pct = int(health * 100)
                mods = []
                if _a.new_sl is not None:
                    sl_arrow = "↑" if p.direction == "LONG" else "↓"
                    mods.append(f"New SL: <b>{_a.new_sl:.4f}</b> {sl_arrow} <span style='color:#6b7280;'>(was {p.sl:.4f})</span>")
                if _a.new_tp is not None:
                    tp_arrow = "↑" if p.direction == "LONG" else "↓"
                    mods.append(f"New TP: <b>{_a.new_tp:.4f}</b> {tp_arrow} <span style='color:#6b7280;'>(was {p.tp:.4f})</span>")
                if _a.partial_close_volume:
                    mods.append(f"Close <b>{_a.partial_close_volume}</b> lots (50%)")
                mods_html = " &nbsp;·&nbsp; ".join(mods) if mods else ""

                st.markdown(
                    f"<div style='background:#0d1117;border-radius:6px;padding:10px 16px;"
                    f"margin-bottom:8px;border:1px solid #1e2330;margin-left:16px;'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;'>"
                    f"<div>"
                    f"<span style='background:{act_color}22;color:{act_color};font-size:11px;"
                    f"font-weight:700;padding:2px 8px;border-radius:4px;border:1px solid {act_color}44;'>"
                    f"{urg_dot} {_a.action}</span>"
                    f"<span style='color:{h_color};font-size:11px;margin-left:10px;'>"
                    f"Health {health:.0%} · {h_label}</span>"
                    f"</div>"
                    f"<span style='color:#6b7280;font-size:11px;'>{_a.assessed_at[11:16]} UTC</span>"
                    f"</div>"
                    f"<div style='background:#1e2330;border-radius:2px;height:3px;margin-bottom:7px;'>"
                    f"<div style='background:{h_color};width:{bar_pct}%;height:3px;border-radius:2px;'></div></div>"
                    f"<div style='font-size:12px;color:#9ca3af;margin-bottom:4px;'>{_a.reason}</div>"
                    + (f"<div style='font-size:12px;color:#e0e0e0;margin-top:4px;'>{mods_html}</div>" if mods_html else "")
                    + f"</div>",
                    unsafe_allow_html=True)

                # Expander: full signal detail
                with st.expander(f"Signal detail — #{p.ticket}", expanded=False):
                    dc1, dc2, dc3, dc4 = st.columns(4)
                    dc1.metric("MTF Score", _a.mtf_score)
                    dc2.metric("MTF Bias", _a.mtf_bias)
                    dc3.metric("ADX Regime", _a.adx_regime)
                    dc4.metric("Signal", f"{_a.signal_direction} {_a.signal_confidence:.0%}")

                    # Strategy agreement badge
                    if _a.strategy_agreement:
                        _sa = _a.strategy_agreement
                        _sa_col = "#22c55e" if "LONG" in _sa else ("#ef4444" if "SHORT" in _sa else "#6b7280")
                        st.markdown(
                            f"<div style='margin:4px 0;font-size:12px;'>"
                            f"Strategy agreement: <b style='color:{_sa_col};'>{_sa}</b></div>",
                            unsafe_allow_html=True)

                    # Breakeven
                    be = _a.breakeven_price
                    if be:
                        st.caption(f"Breakeven price: {be:.5f}")

                    # Reversal / veto warnings
                    _warn_lines = []
                    if _a.signal_direction not in ("NO TRADE", p.direction):
                        _warn_lines.append(f"⚠ Signal reversed to <b>{_a.signal_direction}</b> ({_a.signal_confidence:.0%} conf)")
                    if _a.reason and "divergence" in _a.reason.lower():
                        _warn_lines.append("⚠ RSI momentum divergence detected")
                    if _a.reason and "daily trend" in _a.reason.lower():
                        _warn_lines.append("⚠ Daily trend vetoed position direction")
                    if _warn_lines:
                        st.markdown(
                            "<div style='background:#2d1515;border-radius:5px;padding:6px 10px;"
                            "margin:6px 0;border:1px solid #7f1d1d;font-size:12px;color:#fca5a5;'>"
                            + "<br>".join(_warn_lines) + "</div>",
                            unsafe_allow_html=True)

                    # Detected patterns (from current assessment signal)
                    try:
                        _pats = list((getattr(_a, "_signal_patterns", None) or []))
                    except Exception:
                        _pats = []
                    if not _pats:
                        # Attempt to read from session if assessments stored signal reference
                        pass
                    # News sentiment context
                    if _a.reason and ("news" in _a.reason.lower() or "bearish" in _a.reason.lower() or "bullish" in _a.reason.lower()):
                        st.markdown(
                            f"<div style='font-size:11px;color:#9ca3af;margin-top:4px;'>"
                            f"📰 News context: {_a.reason}</div>",
                            unsafe_allow_html=True)

                # Apply button
                if _a.action != "HOLD":
                    _btn_key = f"apply_{p.ticket}_{_a.assessed_at}"
                    _live = _MT5.get("live_mode", False)
                    if _live:
                        if st.button(f"⚡ Apply {_a.action}", key=_btn_key, type="primary"):
                            try:
                                from agents.trade_manager import _execute_assessment
                                _cd = _MT5.get("tm_cooldown", {})
                                _a.dry_run = False
                                ok = _execute_assessment(_a, _cd)
                                _MT5["tm_cooldown"] = _cd
                                if ok:
                                    st.success(f"✓ {_a.action} executed on #{p.ticket}")
                                    _MT5["positions"] = _gop2()
                                    _MT5["trade_assessments"] = []
                                    st.rerun()
                                else:
                                    st.error("MT5 execution failed — check logs")
                            except Exception as _ex:
                                st.error(str(_ex))
                    else:
                        st.button(f"[DRY RUN] {_a.action}", key=_btn_key, disabled=True,
                                  help="Enable Live Mode to execute")

        st.markdown("<div style='margin-top:16px;'>", unsafe_allow_html=True)
        if st.button("🚨 Close All Positions", type="primary"):
            from utils.mt5_bridge import close_all_positions
            close_all_positions()
            st.success("Closed.")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


# ── Tab 3: History ────────────────────────────────────────────────────────────
with tab_history:
    hc1, hc2 = st.columns([2, 5])
    with hc1:
        days_back = st.selectbox("Show last", [7, 14, 30, 60, 90], index=2,
                                 format_func=lambda x: f"{x} days")
    with hc2:
        st.markdown("<div style='padding-top:28px;'>", unsafe_allow_html=True)
        if st.button("Load from MT5", type="primary"):
            st.session_state["trade_history"] = _fetch_mt5_history(days_back)
        st.markdown("</div>", unsafe_allow_html=True)

    history = st.session_state.get("trade_history")
    if history is None:
        st.info("Click **Load from MT5** to fetch closed trades.")
    elif not history:
        st.caption("No closed trades in this period.")
    else:
        wins  = [t for t in history if t["profit"] > 0]
        total = sum(t["profit"] for t in history)
        wr    = len(wins) / len(history) * 100 if history else 0
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Trades", len(history))
        m2.metric("Win Rate", f"{wr:.1f}%")
        m3.metric("Total P&L", f"{total:+.2f}")
        m4.metric("Avg P&L", f"{total/len(history):+.2f}" if history else "—")
        avg_win  = sum(t["profit"] for t in wins) / len(wins) if wins else 0
        losses   = [t for t in history if t["profit"] <= 0]
        avg_loss = sum(t["profit"] for t in losses) / len(losses) if losses else 0
        m5.metric("Avg Win/Loss", f"{avg_win:+.2f} / {avg_loss:+.2f}")
        st.markdown("---")
        for t in history[:100]:
            pnl_c  = "#22c55e" if t["profit"] > 0 else "#ef4444"
            dir_c  = "#22c55e" if t["type"] == "BUY" else "#ef4444"
            dur    = f"{t['duration_m']}m" if t.get("duration_m") is not None else "—"
            st.markdown(
                f"<div style='background:#111827;border-radius:6px;padding:8px 14px;"
                f"margin-bottom:3px;border:1px solid #1e2330;font-size:13px;'>"
                f"<div style='display:flex;justify-content:space-between;'>"
                f"<span>"
                f"<span style='color:{dir_c};font-weight:700;'>{t['type']}</span>"
                f" <span style='color:#e0e0e0;font-weight:600;'>{t['symbol']}</span>"
                f" <span style='color:#6b7280;'>vol {t['volume']}"
                f" · entry {t['entry']} → exit {t['exit']}"
                f" · {dur} · {t['close_time']}</span>"
                f"</span>"
                f"<span style='color:{pnl_c};font-weight:700;font-size:15px;'>{t['profit']:+.2f}</span>"
                f"</div>"
                f"<div style='color:#4b5563;font-size:11px;margin-top:2px;'>"
                f"gross {t['gross']:+.2f} · comm {t['commission']:+.2f} · swap {t['swap']:+.2f}"
                f"&nbsp;&nbsp;#{t['ticket']}"
                + (f"&nbsp;&nbsp;{t['comment']}" if t.get("comment") else "") +
                f"</div>"
                f"</div>",
                unsafe_allow_html=True)


# ── Tab 4: Journal ────────────────────────────────────────────────────────────
with tab_journal:
    from data.trade_outcomes import (
        import_all_mt5_history, get_summary as _get_summary,
        compute_win_rates as _cwr,
    )

    # ── Import controls ───────────────────────────────────────────────────────
    jc1, jc2, jc3 = st.columns([1, 1, 4])
    with jc1:
        j_days = st.selectbox("Lookback", [30, 60, 90, 180, 365], index=2,
                              format_func=lambda x: f"{x} days", key="j_days")
    with jc2:
        st.markdown("<div style='padding-top:28px;'>", unsafe_allow_html=True)
        if st.button("⬇ Import from MT5", type="primary", key="j_import"):
            with st.spinner("Importing…"):
                n = import_all_mt5_history(days=j_days)
            st.success(f"Imported / updated {n} trade(s).")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    summary = _get_summary()

    if summary["total_sent"] == 0 and summary["total_closed"] == 0:
        st.info("No trades in the journal yet. Click **Import from MT5** to pull your history.")
    else:
        closed = [t for t in summary["all_trades"] if t.get("outcome") in ("WIN","LOSS","BREAKEVEN")]

        # ── Top-level metrics ─────────────────────────────────────────────────
        total_pnl  = sum(t["profit"] for t in closed if t.get("profit") is not None)
        wins_c     = [t for t in closed if t["outcome"] == "WIN"]
        losses_c   = [t for t in closed if t["outcome"] == "LOSS"]
        wr_c       = len(wins_c) / len(closed) if closed else 0
        avg_win_c  = sum(t["profit"] for t in wins_c)  / len(wins_c)  if wins_c  else 0
        avg_loss_c = sum(t["profit"] for t in losses_c)/ len(losses_c) if losses_c else 0
        pf         = abs(avg_win_c * len(wins_c) / (avg_loss_c * len(losses_c))) if losses_c and avg_loss_c != 0 else None

        wr_col  = "#22c55e" if wr_c >= 0.55 else ("#f59e0b" if wr_c >= 0.45 else "#ef4444")
        pnl_col = "#22c55e" if total_pnl >= 0 else "#ef4444"

        jm1, jm2, jm3, jm4, jm5, jm6 = st.columns(6)
        jm1.metric("Total trades", len(closed))
        jm2.markdown(
            f"<div style='font-size:12px;color:#6b7280;'>Win rate</div>"
            f"<div style='font-size:24px;font-weight:700;color:{wr_col};'>{wr_c:.1%}</div>",
            unsafe_allow_html=True)
        jm3.markdown(
            f"<div style='font-size:12px;color:#6b7280;'>Total P&L</div>"
            f"<div style='font-size:24px;font-weight:700;color:{pnl_col};'>{total_pnl:+.2f}</div>",
            unsafe_allow_html=True)
        jm4.metric("Avg win",  f"{avg_win_c:+.2f}")
        jm5.metric("Avg loss", f"{avg_loss_c:+.2f}")
        jm6.metric("Profit factor", f"{pf:.2f}" if pf else "—")

        st.markdown("---")

        # ── Monthly P&L chart ─────────────────────────────────────────────────
        try:
            import plotly.graph_objects as go
            from collections import defaultdict

            monthly: dict[str, float] = defaultdict(float)
            for t in closed:
                if t.get("closed_at") and t.get("profit") is not None:
                    mo = t["closed_at"][:7]   # "YYYY-MM"
                    monthly[mo] += t["profit"]

            if monthly:
                mos  = sorted(monthly.keys())
                vals = [monthly[m] for m in mos]
                colors = ["#22c55e" if v >= 0 else "#ef4444" for v in vals]
                fig = go.Figure(go.Bar(x=mos, y=vals, marker_color=colors,
                                       text=[f"{v:+.0f}" for v in vals],
                                       textposition="outside"))
                fig.update_layout(
                    title="Monthly P&L", height=260,
                    plot_bgcolor="#0b0e17", paper_bgcolor="#0b0e17",
                    font_color="#9ca3af", showlegend=False,
                    margin=dict(l=0, r=0, t=36, b=0),
                    xaxis=dict(gridcolor="#1e2330"),
                    yaxis=dict(gridcolor="#1e2330", zeroline=True, zerolinecolor="#374151"),
                )
                st.plotly_chart(fig, width='stretch')
        except Exception as e:
            log.warning("ctx=monthly_pnl_chart: %s", e)

        # ── By symbol + by strategy (side by side) ────────────────────────────
        sc1, sc2 = st.columns(2)

        with sc1:
            st.markdown("**By Symbol**")
            sym_stats: dict[str, dict] = {}
            for t in closed:
                s = t.get("ticker", "?")
                if s not in sym_stats:
                    sym_stats[s] = {"trades": 0, "wins": 0, "profit": 0.0}
                sym_stats[s]["trades"] += 1
                if t["outcome"] == "WIN":
                    sym_stats[s]["wins"] += 1
                if t.get("profit") is not None:
                    sym_stats[s]["profit"] += t["profit"]
            for sym, v in sorted(sym_stats.items(), key=lambda x: -abs(x[1]["profit"])):
                wr_s = v["wins"] / v["trades"] if v["trades"] else 0
                pc   = "#22c55e" if v["profit"] >= 0 else "#ef4444"
                wc   = "#22c55e" if wr_s >= 0.55 else ("#f59e0b" if wr_s >= 0.45 else "#ef4444")
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"font-size:13px;padding:5px 0;border-bottom:1px solid #1e2330;'>"
                    f"<span style='color:#e0e0e0;font-weight:600;'>{sym}</span>"
                    f"<span style='color:#6b7280;'>{v['trades']} trades &nbsp; "
                    f"<span style='color:{wc};'>{wr_s:.0%}</span> &nbsp; "
                    f"<span style='color:{pc};font-weight:700;'>{v['profit']:+.2f}</span></span>"
                    f"</div>",
                    unsafe_allow_html=True)

        with sc2:
            st.markdown("**By Strategy**")
            for row in summary["by_strategy"]:
                s    = row["strategy"]
                wr_s = row["win_rate"]
                pc   = "#22c55e" if row["profit"] >= 0 else "#ef4444"
                wc   = "#22c55e" if wr_s >= 0.55 else ("#f59e0b" if wr_s >= 0.45 else "#ef4444")
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"font-size:13px;padding:5px 0;border-bottom:1px solid #1e2330;'>"
                    f"<span style='color:#e0e0e0;font-weight:600;'>{s}</span>"
                    f"<span style='color:#6b7280;'>{row['trades']} trades &nbsp; "
                    f"<span style='color:{wc};'>{wr_s:.0%}</span> &nbsp; "
                    f"<span style='color:{pc};font-weight:700;'>{row['profit']:+.2f}</span></span>"
                    f"</div>",
                    unsafe_allow_html=True)

        # ── Full trade log ────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("**Trade Log**")

        # Enrich with strategy/confidence from closed if available
        log_by_ticket = {t["ticket"]: t for t in summary["all_trades"]}

        col_h = st.columns([2, 1, 1, 1, 1, 1, 1, 1, 2])
        for lbl, col in zip(["Date","Symbol","Dir","Entry","Exit","P&L","Gross","Dur","Strategy"], col_h):
            col.markdown(f"<span style='color:#6b7280;font-size:10px;font-weight:700;"
                         f"text-transform:uppercase;letter-spacing:1px;'>{lbl}</span>",
                         unsafe_allow_html=True)
        st.markdown("<div style='border-top:1px solid #1e2330;margin:2px 0 4px;'></div>",
                    unsafe_allow_html=True)

        for t in closed[:200]:
            pnl_c  = "#22c55e" if (t.get("profit") or 0) > 0 else "#ef4444"
            dir_c  = "#22c55e" if t["direction"] == "LONG" else "#ef4444"
            arrow  = "▲" if t["direction"] == "LONG" else "▼"
            profit = t.get("profit")
            gross  = t.get("gross", profit)
            dur_m  = t.get("duration_m")
            if dur_m is None and t.get("sent_at") and t.get("closed_at"):
                try:
                    dt_open  = datetime.fromisoformat(t["sent_at"])
                    dt_close = datetime.fromisoformat(t["closed_at"])
                    dur_m = int((dt_close - dt_open).total_seconds() / 60)
                except Exception:
                    dur_m = None
            dur_str = f"{dur_m}m" if dur_m is not None else "—"
            date_str = (t.get("closed_at") or t.get("sent_at") or "")[:10]
            entry_v  = t.get("entry",       0.0) or 0.0
            exit_v   = t.get("close_price", 0.0) or 0.0
            strategy = t.get("strategy", "—")

            cl = st.columns([2, 1, 1, 1, 1, 1, 1, 1, 2])
            cl[0].markdown(f"<span style='font-size:12px;color:#9ca3af;'>{date_str}</span>",
                           unsafe_allow_html=True)
            cl[1].markdown(f"<span style='font-size:12px;color:#e0e0e0;font-weight:600;'>{t.get('ticker','?')}</span>",
                           unsafe_allow_html=True)
            cl[2].markdown(f"<span style='font-size:12px;color:{dir_c};font-weight:700;'>{arrow} {t['direction']}</span>",
                           unsafe_allow_html=True)
            cl[3].markdown(f"<span style='font-size:12px;color:#9ca3af;'>{entry_v:.4f}</span>",
                           unsafe_allow_html=True)
            cl[4].markdown(f"<span style='font-size:12px;color:#9ca3af;'>{exit_v:.4f}</span>",
                           unsafe_allow_html=True)
            cl[5].markdown(f"<span style='font-size:13px;color:{pnl_c};font-weight:700;'>"
                           f"{profit:+.2f}</span>" if profit is not None else "—",
                           unsafe_allow_html=True)
            cl[6].markdown(f"<span style='font-size:12px;color:#6b7280;'>"
                           f"{gross:+.2f}</span>" if gross is not None else "—",
                           unsafe_allow_html=True)
            cl[7].markdown(f"<span style='font-size:12px;color:#6b7280;'>{dur_str}</span>",
                           unsafe_allow_html=True)
            cl[8].markdown(f"<span style='font-size:11px;color:#6b7280;'>{strategy}</span>",
                           unsafe_allow_html=True)


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
    time.sleep(3)
    st.rerun()
