"""
Katraswing → MetaTrader 5 Signal Server
========================================
Continuous CLI that polls the Katraswing signal engine and forwards
actionable signals to a live MT5 terminal.

Usage:
    python mt5_signal_server.py                         # defaults
    python mt5_signal_server.py --tickers NQ=F ES=F    # custom tickers
    python mt5_signal_server.py --min-confidence 0.65  # stricter threshold
    python mt5_signal_server.py --interval 120         # poll every 2 minutes
    python mt5_signal_server.py --dry-run              # log signals, no orders
    python mt5_signal_server.py --close-all            # emergency: close all positions

Requirements:
    pip install MetaTrader5   (Windows only, MT5 terminal must be running)

The server deduplicates signals: it will NOT re-enter the same ticker in the
same direction until the existing position is closed or the session expires.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as _FT
from datetime import datetime, date
from threading import Lock

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("mt5_signal_server.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
# Covers the three asset classes the engine has the most strategy coverage for:
#   forex majors (MSS_FOREX_15M, VWAP_RSI_5M, PDH_PDL_SWEEP_5M)
#   US index futures (ORB_5M, NR7_BREAKOUT_5M, TREND_MOM_5M)
#   gold (CAMARILLA_5M mean-reversion at S/R levels)
# 6 instruments × ~5–10s/scan = well under the 60s poll interval.
# Override at runtime with --tickers <space-separated list>.
DEFAULT_TICKERS = [
    # LIVE — ordered by historical win rate (policy: symbol_policy.py)
    "YM=F",     # Dow Jones   100% WR (3T)  — LIVE
    "SI=F",     # Silver       71% WR (7T)  — LIVE
    "USDJPY=X", # USD/JPY      44% WR (9T)  — LIVE
    "ES=F",     # S&P 500     100% WR (2T)  — LIVE
    "GC=F",     # Gold                       — LIVE (default)
    # PAPER — signals logged and learned but no live orders sent
    "EURUSD=X", # EUR/USD      25% WR (8T)  — PAPER (warming up)
    "GBPUSD=X", # GBP/USD      38% WR (8T)  — PAPER (warming up)
    "AUDUSD=X", # AUD/USD      17% WR (6T)  — PAPER (warming up)
    "USDCAD=X", # USD/CAD                   — PAPER (small sample)
    "GBPJPY=X", # GBP/JPY                   — PAPER (small sample)
    "EURJPY=X", # EUR/JPY                   — PAPER (small sample)
    "CL=F",     # WTI Crude Oil             — PAPER (new, no history)
    "BZ=F",     # Brent Crude Oil           — PAPER (new, no history)
]
DEFAULT_DISPLAY_NAMES = {
    "YM=F":     "Dow Jones",
    "SI=F":     "Silver",
    "USDJPY=X": "USD/JPY",
    "ES=F":     "ES Mini",
    "GC=F":     "Gold",
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
    "AUDUSD=X": "AUD/USD",
    "USDCAD=X": "USD/CAD",
    "GBPJPY=X": "GBP/JPY",
    "EURJPY=X": "EUR/JPY",
    "CL=F":     "WTI Oil",
    "BZ=F":     "Brent Oil",
}
DEFAULT_INTERVAL_SEC = 30       # poll every 30 seconds — doubled scan frequency
DEFAULT_MIN_CONF     = 0.60     # minimum confidence to enter
FINNHUB_API_KEY      = ""       # optional — set here or via --finnhub-key

# Maximum concurrent run_signal calls per poll. Each ticker's signal fetch is
# independent (separate yfinance/Finnhub connections); the bottleneck is
# network latency, not CPU. 6 is a balance between parallelism and being
# polite to data providers.
MAX_SCAN_WORKERS = 6

# Per-ticker run_signal timeout (seconds). The signal pipeline already has
# inner timeouts on yfinance (20s) and news (12s); this is the outer ceiling.
PER_TICKER_TIMEOUT = 60

# Bar-window cache: run_signal output is reused within the same 5-minute wall-
# clock window since the underlying 5m bar can't have changed. Keyed by ticker
# → (window_start_epoch, SignalResult). Bounded to len(args.tickers).
_SIGNAL_CACHE: dict[str, tuple[int, object]] = {}
_SIGNAL_CACHE_LOCK = Lock()

# H1 bar-window cache: same idea but for the 1-hour bar window.
_H1_SIGNAL_CACHE: dict[str, tuple[int, object]] = {}
_H1_SIGNAL_CACHE_LOCK = Lock()


def _bar_window_key(now_ts: float | None = None) -> int:
    """Floor-divide the current epoch by 300s. Two calls within the same 5-min
    window return the same key — we use this to memoise run_signal."""
    return int((now_ts or time.time()) // 300) * 300


def _h1_bar_window_key(now_ts: float | None = None) -> int:
    """Floor-divide the current epoch by 3600s (1-hour bar window)."""
    return int((now_ts or time.time()) // 3600) * 3600


def _banner():
    log.info("=" * 60)
    log.info("  Katraswing MT5 Signal Server  |  katraswing.io")
    log.info("=" * 60)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Katraswing → MT5 live signal server")
    p.add_argument("--tickers",         nargs="+",  default=DEFAULT_TICKERS,
                   help="Space-separated yfinance tickers to monitor")
    p.add_argument("--interval",        type=int,   default=DEFAULT_INTERVAL_SEC,
                   help="Poll interval in seconds (default 60)")
    p.add_argument("--min-confidence",  type=float, default=DEFAULT_MIN_CONF,
                   help="Minimum signal confidence to enter (0.0–1.0, default 0.60)")
    p.add_argument("--account-size",    type=float, default=100_000.0,
                   help="Account size for position sizing")
    p.add_argument("--risk-pct",        type=float, default=1.0,
                   help="Risk per trade as % of account (default 1.0)")
    p.add_argument("--finnhub-key",     type=str,   default=FINNHUB_API_KEY,
                   help="Finnhub API key for news sentiment")
    p.add_argument("--mt5-path",        type=str,   default="",
                   help="Full path to terminal64.exe (leave blank for auto-detect)")
    p.add_argument("--dry-run",         action="store_true",
                   help="Log signals without sending orders to MT5")
    p.add_argument("--close-all",       action="store_true",
                   help="Emergency: close all Katraswing positions and exit")
    p.add_argument("--health-port",     type=int,   default=9100,
                   help="Port for /healthz and /metrics HTTP endpoints (0 to disable, default 9100)")
    p.add_argument("--telegram-token",   type=str,  default="",
                   help="Telegram bot token (from @BotFather)")
    p.add_argument("--telegram-chat-id", type=str,  default="",
                   help="Telegram chat ID to send notifications to")
    return p.parse_args()


def _signal_key(ticker: str, direction: str, session_date: date) -> str:
    return f"{ticker}:{direction}:{session_date}"


def _format_signal(sr) -> str:
    patterns = ", ".join(
        f"{p.name} ({p.win_rate:.0%} hist)" for p in sr.patterns.patterns[:3]
    ) or "no patterns"
    return (
        f"  Ticker:     {sr.display_name} ({sr.ticker})\n"
        f"  Direction:  {sr.direction}\n"
        f"  Confidence: {sr.confidence:.1%}\n"
        f"  Entry:      {sr.entry:.4f}\n"
        f"  Stop Loss:  {sr.sl:.4f}\n"
        f"  Take Profit:{sr.tp:.4f}\n"
        f"  Patterns:   {patterns}\n"
        f"  News:       {sr.news_sentiment} ({sr.news_score:+.2f})\n"
    )


# ── Position manager: breakeven stops + partial exits ────────────────────────
# State keyed by MT5 ticket. Persists across poll cycles; stale entries are
# pruned when the position closes. Resets on server restart — state is
# recovered from the position's current SL vs open_price on first observation.

_pos_state: dict[int, dict] = {}
# {ticket: {"original_1r": float, "be_done": bool, "partial_done": bool}}

_prev_positions: dict[int, object] = {}
# {ticket: MT5Position} — snapshot from last poll, used to detect closed positions


def _pos_init(pos) -> dict:
    """Return (and lazily create) management state for `pos`."""
    if pos.ticket not in _pos_state:
        # Derive original 1R from the TP (TP = entry + 2R, set at order time,
        # never modified). Fallback to current SL if TP is zero/missing.
        if pos.tp != 0 and pos.open_price != 0:
            original_1r = abs(pos.tp - pos.open_price) / 2.0
        elif pos.sl != 0 and pos.open_price != 0:
            original_1r = abs(pos.open_price - pos.sl)
        else:
            original_1r = 0.0

        # Recovery after server restart: if SL is already at or beyond entry
        # the BE+partial cycle was completed in a previous run.
        if original_1r > 0:
            already = (
                (pos.direction == "LONG"  and pos.sl >= pos.open_price - original_1r * 0.02)
                or (pos.direction == "SHORT" and pos.sl <= pos.open_price + original_1r * 0.02)
            )
        else:
            already = False

        _pos_state[pos.ticket] = {
            "original_1r":  original_1r,
            "be_done":      already,
            "partial_done": already,
            "trail_sl":     pos.sl if already else 0.0,
        }
    return _pos_state[pos.ticket]


def _manage_positions(log, dry_run: bool = False, tg=None, display_names: dict | None = None) -> None:
    """
    Called once per poll cycle before signal scanning.

    For each open position:
      1. Partial exit  — when profit >= 1R, close 50% of the position.
      2. Breakeven SL  — when profit >= 1R, move stop-loss to entry price.

    Also detects positions that closed since the last poll and fires a
    Telegram notification with the realised P&L.
    """
    global _prev_positions

    if dry_run:
        return

    try:
        from utils.mt5_bridge import (
            get_open_positions, modify_position, partial_close_position,
            is_connected,
        )
    except ImportError:
        return

    if not is_connected():
        return

    positions   = get_open_positions()
    open_tickets = {p.ticket: p for p in positions}
    _dn          = display_names or {}

    # ── Detect closed positions ──────────────────────────────────────────────
    for ticket, prev in _prev_positions.items():
        if ticket not in open_tickets:
            sym     = getattr(prev, "symbol", "")
            display = _dn.get(sym, sym)
            profit  = float(getattr(prev, "profit", 0.0))
            dirn    = getattr(prev, "direction", "")
            log.info(f"  [CLOSED] #{ticket} {sym} {dirn} | P&L={profit:+.2f}")
            if tg:
                tg.position_closed(ticket, display, dirn, profit)

    _prev_positions = dict(open_tickets)

    # ── Breakeven + partial exit ─────────────────────────────────────────────
    for pos in positions:
        if pos.open_price == 0 or pos.tp == 0:
            continue

        state = _pos_init(pos)
        one_r = state["original_1r"]
        if one_r <= 0:
            continue

        profit_pts = (
            pos.price_current - pos.open_price
            if pos.direction == "LONG"
            else pos.open_price - pos.price_current
        )

        if profit_pts < one_r:
            continue

        sym     = pos.symbol
        display = _dn.get(sym, sym)

        # ── 1. Partial close (50%) ──────────────────────────────────────────
        if not state["partial_done"]:
            sym_info = None
            try:
                import MetaTrader5 as _mt5
                sym_info = _mt5.symbol_info(sym)
            except Exception:
                pass
            vol_step = float(getattr(sym_info, "volume_step", 0.01) or 0.01)
            half     = max(vol_step, round(int(pos.volume / 2 / vol_step) * vol_step, 8))
            if half < pos.volume:
                ok = partial_close_position(pos.ticket, half)
                if ok:
                    state["partial_done"] = True
                    log.info(
                        f"  [PARTIAL] #{pos.ticket} {sym}: "
                        f"closed {half:.2f} lots at +{profit_pts:.5g} "
                        f"(1R={one_r:.5g}). Remaining {pos.volume - half:.2f} lots -> 2R."
                    )
                    if tg:
                        tg.partial_exit(pos.ticket, display, half, pos.open_price, pos.tp)
                else:
                    log.warning(f"  [PARTIAL] #{pos.ticket} {sym}: partial close failed")
            else:
                state["partial_done"] = True

        # ── 2. Breakeven stop ───────────────────────────────────────────────
        if not state["be_done"]:
            ok = modify_position(pos.ticket, new_sl=pos.open_price)
            if ok:
                state["be_done"] = True
                state["trail_sl"] = pos.open_price
                log.info(
                    f"  [BE] #{pos.ticket} {sym}: "
                    f"SL moved to entry {pos.open_price:.5g} "
                    f"(profit={profit_pts:.5g} >= 1R={one_r:.5g})"
                )
                if tg:
                    tg.breakeven(pos.ticket, display, pos.open_price)
            else:
                log.warning(f"  [BE] #{pos.ticket} {sym}: SL modify failed")

        # ── 3. Trailing stop (active once breakeven is set) ─────────────────
        # Trail distance = 0.5×1R. Move SL only in the profitable direction.
        if state["be_done"] and one_r > 0:
            trail_dist = one_r * 0.5
            if pos.direction == "LONG":
                new_trail = pos.price_current - trail_dist
                if new_trail > state["trail_sl"] and new_trail > pos.open_price:
                    ok = modify_position(pos.ticket, new_sl=round(new_trail, 5))
                    if ok:
                        state["trail_sl"] = new_trail
                        log.info(
                            f"  [TRAIL] #{pos.ticket} {sym}: SL -> {new_trail:.5g} "
                            f"(price={pos.price_current:.5g}, dist={trail_dist:.5g})"
                        )
            else:  # SHORT
                new_trail = pos.price_current + trail_dist
                if new_trail < state["trail_sl"] and new_trail < pos.open_price:
                    ok = modify_position(pos.ticket, new_sl=round(new_trail, 5))
                    if ok:
                        state["trail_sl"] = new_trail
                        log.info(
                            f"  [TRAIL] #{pos.ticket} {sym}: SL -> {new_trail:.5g} "
                            f"(price={pos.price_current:.5g}, dist={trail_dist:.5g})"
                        )

    # Prune closed positions from state dict
    for stale in [t for t in _pos_state if t not in open_tickets]:
        del _pos_state[stale]


def run_server(args: argparse.Namespace):
    _banner()

    # MT5 bridge import
    from utils.mt5_bridge import (
        connect, disconnect, is_connected, ensure_connected,
        send_from_signal_result, close_all_positions, get_open_positions,
        is_available,
    )
    from agents.signal_engine import run_signal
    from agents.swing_engine import run_h1_signal

    # Emergency close-all and exit
    if args.close_all:
        if not is_available():
            log.error("MetaTrader5 not installed. Cannot close positions.")
            sys.exit(1)
        if not connect(args.mt5_path):
            sys.exit(1)
        log.warning("EMERGENCY: closing all Katraswing positions...")
        close_all_positions()
        disconnect()
        log.info("Done. Exiting.")
        sys.exit(0)

    # Connect to MT5
    if not args.dry_run:
        if not is_available():
            log.error(
                "MetaTrader5 package not installed.\n"
                "Install with: pip install MetaTrader5\n"
                "Note: Windows only, MT5 terminal must be open.\n"
                "Use --dry-run to test signal generation without MT5."
            )
            sys.exit(1)
        if not connect(args.mt5_path):
            log.error("Cannot connect to MT5 terminal. Is it running?")
            sys.exit(1)
    else:
        log.info("DRY RUN mode — signals will be logged but NOT sent to MT5.")

    log.info(f"Monitoring: {args.tickers}")
    log.info(f"Poll interval: {args.interval}s | Min confidence: {args.min_confidence:.0%}")
    log.info("Press Ctrl+C to stop.\n")

    # Health/metrics HTTP server on a daemon thread. Non-fatal if port is busy
    # — the trading loop continues without observability rather than aborting.
    metrics = None
    if args.health_port > 0:
        try:
            from utils.health_server import Metrics, start_http_server
            metrics = Metrics()
            metrics.mt5_connected = is_connected() if not args.dry_run else False
            start_http_server(metrics, port=args.health_port)
        except Exception as _he:
            log.warning(f"Could not start health HTTP server on :{args.health_port}: {_he}")
            metrics = None

    # Telegram notifier (no-op when token/chat_id not provided)
    from utils.telegram_notify import Notifier as _Notifier
    from utils.correlation_filter import is_correlated_duplicate as _corr_check
    tg = _Notifier(token=args.telegram_token, chat_id=args.telegram_chat_id)
    if tg.enabled():
        log.info("Telegram notifications: ENABLED")
        tg.info(f"Katraswing started -- monitoring {len(args.tickers)} symbols")
    else:
        log.info("Telegram notifications: disabled (no token/chat-id)")

    # Wire Telegram notifier into learning loop so strategy changes are reported
    try:
        from agents.learning_loop import set_notifier as _set_tg_notifier
        _set_tg_notifier(tg)
    except Exception:
        pass

    # Dedup sets: live orders vs paper-only signals tracked separately so the
    # log message accurately says "already recorded (paper)" vs "already entered"
    sent_signals: set[str] = set()   # live orders placed
    paper_signals: set[str] = set()  # paper signals logged (no MT5 order)

    _daily_summary_sent: set[str] = set()  # dates for which summary was sent

    _HEARTBEAT_INTERVAL = 15 * 60  # seconds
    _last_heartbeat: float = 0.0
    _hb_signals: list[str] = []    # signal lines accumulated since last heartbeat

    def _maybe_send_heartbeat(positions: list) -> None:
        nonlocal _last_heartbeat, _hb_signals
        now = time.time()
        if now - _last_heartbeat < _HEARTBEAT_INTERVAL:
            return
        _last_heartbeat = now

        from datetime import timezone as _tz
        ts = datetime.now(_tz.utc).strftime("%H:%M UTC")

        lines = [f"<b>Heartbeat {ts}</b>"]

        # Account equity
        try:
            from utils.mt5_bridge import get_account_info as _acct
            ai = _acct()
            if ai:
                lines.append(
                    f"Equity: <b>{ai.get('equity', '?'):.2f} {ai.get('currency','')}</b>"
                    f"  |  Balance: {ai.get('balance','?'):.2f}"
                )
        except Exception:
            pass

        # Open positions
        if positions:
            lines.append(f"\nOpen positions ({len(positions)}):")
            total_pnl = 0.0
            for p in positions:
                arrow = "^" if p.direction == "LONG" else "v"
                sym = DEFAULT_DISPLAY_NAMES.get(p.symbol, p.symbol)
                lines.append(f"  {sym} {arrow} #{p.ticket}  P&L: {p.profit:+.2f}")
                total_pnl += float(p.profit)
            lines.append(f"  Running P&L: <b>{total_pnl:+.2f}</b>")
        else:
            lines.append("\nNo open positions")

        # Recent signals since last heartbeat
        if _hb_signals:
            lines.append(f"\nSignals last 15 min ({len(_hb_signals)}):")
            for s in _hb_signals[-8:]:
                lines.append(f"  {s}")
        else:
            lines.append("\nNo new signals last 15 min")

        # 7-day strategy WR summary (top performers + worst)
        try:
            import json as _json
            from datetime import timezone as _tz2, timedelta as _td
            _cutoff = (datetime.now(_tz2.utc) - _td(days=7)).isoformat()
            with open("data/trade_log.json", encoding="utf-8") as _f:
                _trades = _json.load(_f)
            _buckets: dict[str, list] = {}
            for _t in _trades:
                if _t.get("strategy") == "MT5_IMPORT":
                    continue
                if _t.get("outcome") not in ("WIN", "LOSS"):
                    continue
                if str(_t.get("closed_at", "")) < _cutoff[:10]:
                    continue
                _s = _t["strategy"]
                _buckets.setdefault(_s, []).append(_t)
            if _buckets:
                lines.append("\n7-day strategy WR:")
                _rows = sorted(
                    [(s, sum(1 for t in ts if t["outcome"]=="WIN"), len(ts))
                     for s, ts in _buckets.items()],
                    key=lambda x: -(x[1]/x[2]) if x[2] >= 3 else -99,
                )
                for _s, _w, _n in _rows:
                    if _n < 2:
                        continue
                    _wr = _w / _n * 100
                    _bar = "+" if _wr >= 50 else "-"
                    lines.append(f"  {_bar} {_s[:20]:<20} {_w}/{_n} ({_wr:.0f}%)")
        except Exception:
            pass

        _hb_signals = []

        if tg:
            tg._send("\n".join(lines))

    def _maybe_send_daily_summary() -> None:
        """Send a daily P&L Telegram summary once per day after 22:00 UTC.

        The day is only marked as sent once a non-empty summary is dispatched,
        so the function retries on each poll cycle until trades appear or
        midnight passes (whichever comes first).
        """
        import json as _json
        from datetime import timezone as _tz
        now_utc = datetime.now(_tz.utc)
        if now_utc.hour < 22:
            return
        today_str = str(now_utc.date())
        if today_str in _daily_summary_sent:
            return

        trade_log_path = "data/trade_log.json"
        try:
            with open(trade_log_path, encoding="utf-8") as _f:
                all_trades = _json.load(_f)
        except Exception:
            all_trades = []

        today_trades = [
            t for t in all_trades
            if isinstance(t, dict)
            and str(t.get("closed_at", ""))[:10] == today_str
            and t.get("outcome") in ("WIN", "LOSS")
        ]

        if not today_trades:
            # Mark sent only at 23:00+ so we don't retry indefinitely, but still
            # allow earlier cycles (22:xx) to retry if outcomes arrive late.
            if now_utc.hour >= 23:
                _daily_summary_sent.add(today_str)
                if tg:
                    tg.info(f"Daily summary {today_str}: no closed trades today.")
            return

        wins   = sum(1 for t in today_trades if t.get("outcome") == "WIN")
        losses = sum(1 for t in today_trades if t.get("outcome") == "LOSS")
        total  = len(today_trades)
        pnl    = sum(float(t.get("profit", 0)) for t in today_trades)
        wr_pct = wins / total * 100

        lines = [f"<b>Daily P&amp;L Summary — {today_str}</b>"]
        lines.append(f"Trades: {total}  |  W: {wins}  L: {losses}  ({wr_pct:.0f}% WR)")
        lines.append(f"Net P&amp;L: <b>{'+'if pnl>=0 else ''}{pnl:.2f} USD</b>")

        by_sym: dict[str, dict] = {}
        for t in today_trades:
            sym = t.get("ticker", "?")
            if sym not in by_sym:
                by_sym[sym] = {"wins": 0, "losses": 0, "pnl": 0.0}
            if t.get("outcome") == "WIN":
                by_sym[sym]["wins"] += 1
            else:
                by_sym[sym]["losses"] += 1
            by_sym[sym]["pnl"] += float(t.get("profit", 0))
        for sym, s in sorted(by_sym.items(), key=lambda x: -abs(x[1]["pnl"])):
            sign = "+" if s["pnl"] >= 0 else ""
            lines.append(f"  {sym}: {s['wins']}W/{s['losses']}L  {sign}{s['pnl']:.2f}")

        msg = "\n".join(lines)
        log.info("[daily-summary] %s", msg.replace("<b>","").replace("</b>","").replace("&amp;","&"))
        if tg:
            tg._send(msg)
        _daily_summary_sent.add(today_str)

    def _scan_with_cache(ticker: str):
        """Fetch run_signal for one ticker, reusing the cached result if we
        already computed it in the current 5-min bar window."""
        now_window = _bar_window_key()
        with _SIGNAL_CACHE_LOCK:
            cached = _SIGNAL_CACHE.get(ticker)
        if cached and cached[0] == now_window:
            return cached[1]
        display = DEFAULT_DISPLAY_NAMES.get(ticker, ticker)
        sr = run_signal(
            ticker=ticker,
            finnhub_api_key=args.finnhub_key,
            account_size=args.account_size,
            risk_pct=args.risk_pct,
            display_name=display,
        )
        with _SIGNAL_CACHE_LOCK:
            _SIGNAL_CACHE[ticker] = (now_window, sr)
        return sr

    def _scan_h1_with_cache(ticker: str):
        """Fetch run_h1_signal for one ticker, cached per 1-hour bar window."""
        now_window = _h1_bar_window_key()
        with _H1_SIGNAL_CACHE_LOCK:
            cached = _H1_SIGNAL_CACHE.get(ticker)
        if cached and cached[0] == now_window:
            return cached[1]
        display = DEFAULT_DISPLAY_NAMES.get(ticker, ticker)
        sr = run_h1_signal(
            ticker=ticker,
            finnhub_api_key=args.finnhub_key,
            account_size=args.account_size,
            risk_pct=args.risk_pct,
            display_name=display,
        )
        with _H1_SIGNAL_CACHE_LOCK:
            _H1_SIGNAL_CACHE[ticker] = (now_window, sr)
        return sr

    # Multi-cadence learning scheduler — checked once per poll iteration.
    # Cheap when nothing is due (one JSON read + three timestamp compares);
    # spawns daemon threads for daily/weekly. See agents/learning_loop.py.
    from agents.learning_loop import tick as _learning_tick, set_watchlist as _set_watchlist
    _set_watchlist(list(args.tickers))

    try:
        while True:
            poll_start = time.time()
            today = date.today()

            try:
                _learning_tick()
            except Exception as exc:
                log.error("learning_loop tick failed: %s", exc)

            # Clear stale signals from prior sessions
            stale = {k for k in sent_signals if not k.endswith(str(today))}
            sent_signals -= stale
            stale = {k for k in paper_signals if not k.endswith(str(today))}
            paper_signals -= stale

            # Fetch open positions once per cycle — used for dedup, correlation
            # filter, and position display. Avoids repeated MT5 round-trips.
            if not args.dry_run and is_connected():
                current_positions = get_open_positions()
                open_symbols      = {p.symbol for p in current_positions}
            else:
                current_positions = []
                open_symbols      = set()

            # ── Phase 0: manage open positions (breakeven + partial exits) ──
            _manage_positions(
                log, dry_run=args.dry_run, tg=tg,
                display_names=DEFAULT_DISPLAY_NAMES,
            )

            # ── Phase 0b: daily P&L Telegram summary (after 22:00 UTC) ─────
            try:
                _maybe_send_daily_summary()
            except Exception as _exc:
                log.warning("daily summary failed: %s", _exc)

            # ── Phase 1: fetch signals in parallel ───────────────────────
            # Each run_signal is network-bound (yfinance/Finnhub) and
            # independent. Bar-level cache hits return immediately.
            workers = min(MAX_SCAN_WORKERS, len(args.tickers)) or 1
            signal_results: dict[str, object] = {}
            log.info(f"-- Scanning {len(args.tickers)} ticker(s) (workers={workers}) --")
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(_scan_with_cache, t): t for t in args.tickers}
                outer_budget = max(120, PER_TICKER_TIMEOUT * ((len(args.tickers) + workers - 1) // workers) + 30)
                try:
                    for fut in as_completed(futures, timeout=outer_budget):
                        ticker = futures[fut]
                        try:
                            signal_results[ticker] = fut.result(timeout=PER_TICKER_TIMEOUT)
                        except (_FT, Exception) as exc:
                            log.error(f"Signal engine error for {ticker}: {exc}")
                            signal_results[ticker] = None
                except _FT:
                    n_unfinished = sum(1 for f in futures if not f.done())
                    log.warning(
                        f"Outer scan timeout ({outer_budget}s); {n_unfinished} ticker(s) unfinished"
                    )
                    for f, t in futures.items():
                        if t in signal_results:
                            continue
                        if f.done():
                            try:
                                signal_results[t] = f.result(timeout=0)
                            except Exception as exc:
                                log.error(f"Signal result error for {t}: {exc}")
                                signal_results[t] = None
                        else:
                            f.cancel()
                            signal_results[t] = None

            # ── Phase 2: process signals + send orders sequentially ──────
            # MT5 order_send IPC isn't safe to call from multiple threads
            # concurrently — keep this loop single-threaded.
            for ticker in args.tickers:
                sr = signal_results.get(ticker)
                if sr is None:
                    continue
                display = DEFAULT_DISPLAY_NAMES.get(ticker, ticker)

                if sr.error:
                    log.warning(f"{ticker} error: {sr.error}")
                    continue

                if sr.direction not in ("LONG", "SHORT"):
                    veto = " [MTF vetoed]" if getattr(sr, "daily_trend_vetoed", False) else ""
                    log.info(f"{display}: NO TRADE (conf={sr.confidence:.1%}){veto}")
                    continue

                if sr.confidence < args.min_confidence:
                    log.info(
                        f"{display}: {sr.direction} @ {sr.confidence:.1%} -- "
                        f"below threshold {args.min_confidence:.0%}, skipping."
                    )
                    continue

                top_patterns = [(p.name, p.win_rate) for p in sr.patterns.patterns[:3]]
                pattern_str  = " | ".join(f"{n} {w:.0%}" for n, w in top_patterns) or "-"

                log.info(
                    f"* SIGNAL: {display} {sr.direction} | "
                    f"conf={sr.confidence:.1%} | {pattern_str}"
                )
                log.info(_format_signal(sr))

                # Dedup check — distinguish paper logs from live orders
                key = _signal_key(ticker, sr.direction, today)
                is_paper = getattr(sr, "paper_only", False)
                if key in (paper_signals if is_paper else sent_signals):
                    tag = "paper -- skipping" if is_paper else "live order placed -- skipping"
                    log.info(f"  [dedup] {sr.direction} {display} already recorded today ({tag}).")
                    continue

                # Only count in heartbeat after dedup passes (new signal this session)
                _hb_signals.append(
                    f"{display} {sr.direction} {sr.confidence:.0%}"
                    + (" [paper]" if is_paper else "")
                )

                # Open-position guard: never open a second position on a symbol
                # that already has one open (same or opposite direction).
                # Hedging (LONG + SHORT on the same symbol) wastes margin with
                # zero net exposure; opposite signals usually mean the thesis
                # changed, not that a hedge is wanted.
                if not is_paper and not args.dry_run:
                    mt5_sym = getattr(sr, "mt5_symbol", "") or ""
                    if not mt5_sym:
                        from utils.mt5_bridge import resolve_mt5_symbol as _resolve
                        mt5_sym = _resolve(ticker)
                    if mt5_sym and mt5_sym in open_symbols:
                        log.info(
                            f"  [open-pos] {display} skipped -- "
                            f"{mt5_sym} already has an open position"
                        )
                        continue

                # Economic calendar filter: block within 30 min of high-impact events.
                # Applied to both live and paper signals so calibration data stays clean.
                if args.finnhub_key:
                    from utils.economic_calendar import is_news_window as _news_win
                    import datetime as _dtmod
                    _now_utc = _dtmod.datetime.now(_dtmod.timezone.utc)
                    cal_blocked, cal_reason = _news_win(
                        ticker, _now_utc.hour, _now_utc.minute, args.finnhub_key
                    )
                    if cal_blocked:
                        log.info(f"  [calendar] {cal_reason}")
                        continue

                # Correlation filter: block if a correlated instrument is already
                # open in the same direction (redundant exposure, same thesis).
                # Paper signals are exempt — they don't use real capital.
                if not is_paper and not args.dry_run:
                    corr_blocked, corr_reason = _corr_check(
                        ticker, sr.direction, current_positions
                    )
                    if corr_blocked:
                        log.info(f"  [corr] {corr_reason}")
                        continue

                # Paper-mode strategies/symbols: signal still passes calibration
                # but the broker round-trip is skipped. trade_log records a
                # paper row so the weekly auto-promotion harness can mature it.
                if is_paper:
                    log.info(f"  [paper] {sr.paper_reason or 'strategy'} -- order not sent")
                    tg.signal(display, sr.direction, sr.confidence, sr.entry, sr.sl, sr.tp,
                              sr.chart_signals[0].strategy if sr.chart_signals else "?",
                              paper=True)
                    paper_signals.add(key)
                    # Record paper trade with synthetic ticket so the nightly
                    # promotion harness can accumulate outcome data.
                    try:
                        import time as _time
                        from data.trade_outcomes import record_trade as _rt
                        from utils.mt5_bridge import resolve_mt5_symbol as _resolve_sym
                        _syn_ticket = -(int(_time.time() * 1000) % (2**30))
                        _strat = sr.chart_signals[0].strategy if getattr(sr, "chart_signals", None) else "UNKNOWN"
                        _pats  = sr.patterns.patterns if getattr(sr, "patterns", None) else None
                        _paper_mt5_sym = getattr(sr, "mt5_symbol", "") or _resolve_sym(ticker) or ""
                        _rt(
                            _syn_ticket, ticker, _strat, sr.direction,
                            sr.confidence, sr.entry, sr.sl, sr.tp,
                            patterns=_pats,
                            adx_value=getattr(sr, "adx_value", None),
                            atr_value=getattr(sr, "atr", None),
                            h1_trend=getattr(sr, "daily_trend_direction", None),
                            vol_ratio=getattr(sr, "vol_ratio", None),
                            consensus_count=getattr(sr, "consensus_count", None),
                            pattern_boost_val=getattr(sr, "pattern_boost_val", None),
                            calibrated_conf=getattr(sr, "calibrated_conf", None),
                            paper_only=True,
                            mt5_symbol=_paper_mt5_sym,
                            base_confidence=getattr(sr, "base_confidence", None),
                            consensus_boost=getattr(sr, "consensus_boost", None),
                            bt_adjustment=getattr(sr, "bt_adjustment", None),
                            live_adjustment=getattr(sr, "live_adjustment", None),
                            news_boost=getattr(sr, "news_boost", None),
                            session_boost=getattr(sr, "session_boost", None),
                        )
                    except Exception as _pe:
                        log.debug("paper record_trade: %s", _pe)
                    continue

                if args.dry_run:
                    log.info("  [dry-run] Would send to MT5.")
                    sent_signals.add(key)
                    continue

                # ── Pre-order quality filters ────────────────────────────
                from utils.mt5_bridge import (
                    get_spread_ratio as _spread_ratio,
                    get_current_price as _cur_price,
                )

                # 1. Spread filter: skip if spread > 2× typical
                _mt5_sym_live = getattr(sr, "mt5_symbol", "") or mt5_sym or ""
                if _mt5_sym_live:
                    _sr = _spread_ratio(_mt5_sym_live)
                    if _sr > 2.0:
                        log.info(
                            f"  [spread] {display} skipped -- "
                            f"spread {_sr:.1f}x typical (threshold 2x)"
                        )
                        continue

                # 2. Minimum R:R filter: require TP/SL ratio >= 1.5
                _sl_dist = abs(sr.entry - sr.sl)
                _tp_dist = abs(sr.tp - sr.entry)
                _rr = _tp_dist / _sl_dist if _sl_dist > 0 else 0.0
                if _rr < 1.5:
                    log.info(
                        f"  [rr] {display} skipped -- "
                        f"R:R={_rr:.2f} < 1.5 (SL={sr.sl:.5g} TP={sr.tp:.5g})"
                    )
                    continue

                # 3. Stale entry filter: skip if price moved > 0.5×ATR from signal entry
                if _mt5_sym_live and sr.atr > 0:
                    _live_px = _cur_price(_mt5_sym_live, sr.direction)
                    if _live_px > 0:
                        _drift = abs(_live_px - sr.entry)
                        if _drift > 0.5 * sr.atr:
                            log.info(
                                f"  [stale] {display} skipped -- "
                                f"price drifted {_drift:.5g} from entry "
                                f"(threshold 0.5x ATR={0.5*sr.atr:.5g})"
                            )
                            continue

                # Notify signal before sending order
                tg.signal(display, sr.direction, sr.confidence, sr.entry, sr.sl, sr.tp,
                          sr.chart_signals[0].strategy if sr.chart_signals else "?")

                # Reconnect if needed
                if not ensure_connected(args.mt5_path):
                    log.error("  MT5 reconnect failed. Skipping order.")
                    tg.error(f"MT5 reconnect failed -- {display} {sr.direction} order skipped")
                    continue

                result = send_from_signal_result(sr)
                if result.success:
                    log.info(
                        f"  ✓ Order #{result.ticket} placed | "
                        f"{result.direction} {result.symbol} @ {result.entry:.4f}"
                    )
                    tg.order_placed(result.ticket, display, result.direction,
                                    result.entry, sr.sl, sr.tp)
                    sent_signals.add(key)
                    try:
                        from data.trade_outcomes import record_trade
                        _strat = sr.chart_signals[0].strategy if getattr(sr, "chart_signals", None) else "UNKNOWN"
                        _pats  = sr.patterns.patterns if getattr(sr, "patterns", None) else None
                        record_trade(
                            result.ticket, ticker, _strat, sr.direction,
                            sr.confidence, sr.entry, sr.sl, sr.tp,
                            patterns=_pats,
                            adx_value=getattr(sr, "adx_value", None),
                            atr_value=getattr(sr, "atr", None),
                            h1_trend=getattr(sr, "daily_trend_direction", None),
                            vol_ratio=getattr(sr, "vol_ratio", None),
                            consensus_count=getattr(sr, "consensus_count", None),
                            pattern_boost_val=getattr(sr, "pattern_boost_val", None),
                            calibrated_conf=getattr(sr, "calibrated_conf", None),
                            base_confidence=getattr(sr, "base_confidence", None),
                            consensus_boost=getattr(sr, "consensus_boost", None),
                            bt_adjustment=getattr(sr, "bt_adjustment", None),
                            live_adjustment=getattr(sr, "live_adjustment", None),
                            news_boost=getattr(sr, "news_boost", None),
                            session_boost=getattr(sr, "session_boost", None),
                        )
                    except Exception as _rte:
                        log.warning("record_trade #%s: %s", result.ticket, _rte)
                    if metrics is not None:
                        metrics.record_signal()
                else:
                    log.warning(f"  ✗ Order failed: {result.error}")
                    if metrics is not None:
                        # Try to extract retcode from "retcode=NNNN | ..." format
                        import re as _re
                        m = _re.search(r"retcode=(\d+)", str(result.error))
                        if m:
                            metrics.record_rejection(int(m.group(1)))
                        else:
                            metrics.record_rejection(0)   # unknown / pre-flight

            # ── Phase 3: H1 swing engine scan (parallel, then sequential send) ──
            # All H1 strategies start paper_only=True so no live orders are sent
            # until auto-promotion. The scan still runs so calibration data
            # accumulates from the first poll.
            h1_workers = min(MAX_SCAN_WORKERS, len(args.tickers)) or 1
            h1_results: dict[str, object] = {}
            with ThreadPoolExecutor(max_workers=h1_workers) as ex:
                h1_futures = {ex.submit(_scan_h1_with_cache, t): t for t in args.tickers}
                try:
                    for fut in as_completed(h1_futures, timeout=90):
                        ticker = h1_futures[fut]
                        try:
                            h1_results[ticker] = fut.result(timeout=PER_TICKER_TIMEOUT)
                        except Exception as exc:
                            log.error(f"H1 signal error for {ticker}: {exc}")
                            h1_results[ticker] = None
                except _FT:
                    log.warning("H1 scan timeout — some tickers skipped")
                    for f, t in h1_futures.items():
                        if t not in h1_results:
                            h1_results[t] = None

            for ticker in args.tickers:
                sr = h1_results.get(ticker)
                if sr is None or sr.error:
                    if sr and sr.error:
                        log.debug(f"H1 {ticker}: {sr.error}")
                    continue
                if sr.direction not in ("LONG", "SHORT"):
                    continue

                display = DEFAULT_DISPLAY_NAMES.get(ticker, ticker)
                log.info(
                    f"* H1 SIGNAL: {display} {sr.direction} | "
                    f"conf={sr.confidence:.1%} | strategy={sr.chart_signals[0].strategy if sr.chart_signals else '?'}"
                )
                _hb_signals.append(f"[H1] {display} {sr.direction} {sr.confidence:.0%}")
                key = _signal_key(f"H1:{ticker}", sr.direction, today)
                if key in sent_signals:
                    log.info(f"  [dedup] H1 {sr.direction} {display} already recorded today (paper -- skipping).")
                    continue
                # H1 strategies are paper_only at launch — no broker round-trip.
                # Record with a synthetic ticket so the nightly promotion harness
                # can accumulate outcomes and auto-promote H1 strategies that
                # earn ≥10 closed trades with WR≥0.50 and PF≥1.10 (see
                # learning_loop.run_nightly). Without this, H1 strategies fire
                # signals indefinitely but never collect the data needed to
                # graduate to live.
                if getattr(sr, "paper_only", True):
                    log.info(f"  [H1 paper] signal recorded for calibration")
                    sent_signals.add(key)
                    try:
                        import time as _time
                        from data.trade_outcomes import record_trade as _rt
                        from utils.mt5_bridge import resolve_mt5_symbol as _resolve_sym
                        _syn_ticket = -(int(_time.time() * 1000) % (2**30))
                        _strat = sr.chart_signals[0].strategy if getattr(sr, "chart_signals", None) else "UNKNOWN"
                        _pats  = sr.patterns.patterns if getattr(sr, "patterns", None) else None
                        _h1_mt5_sym = getattr(sr, "mt5_symbol", "") or _resolve_sym(ticker) or ""
                        _rt(
                            _syn_ticket, ticker, _strat, sr.direction,
                            sr.confidence, sr.entry, sr.sl, sr.tp,
                            patterns=_pats,
                            adx_value=getattr(sr, "adx_value", None),
                            atr_value=getattr(sr, "atr", None),
                            h1_trend=getattr(sr, "daily_trend_direction", None),
                            vol_ratio=getattr(sr, "vol_ratio", None),
                            consensus_count=getattr(sr, "consensus_count", None),
                            pattern_boost_val=getattr(sr, "pattern_boost_val", None),
                            calibrated_conf=getattr(sr, "calibrated_conf", None),
                            paper_only=True,
                            mt5_symbol=_h1_mt5_sym,
                            base_confidence=getattr(sr, "base_confidence", None),
                            consensus_boost=getattr(sr, "consensus_boost", None),
                            bt_adjustment=getattr(sr, "bt_adjustment", None),
                            live_adjustment=getattr(sr, "live_adjustment", None),
                            news_boost=getattr(sr, "news_boost", None),
                            session_boost=getattr(sr, "session_boost", None),
                        )
                    except Exception as _pe:
                        log.debug("H1 paper record_trade: %s", _pe)
                    continue
                # If a strategy has been auto-promoted to live, process normally.
                if args.dry_run:
                    log.info("  [H1 dry-run] Would send to MT5.")
                    sent_signals.add(key)
                    continue
                if not ensure_connected(args.mt5_path):
                    log.error("  MT5 reconnect failed. Skipping H1 order.")
                    continue
                result = send_from_signal_result(sr)
                if result.success:
                    log.info(f"  ✓ H1 Order #{result.ticket} placed")
                    sent_signals.add(key)
                    try:
                        from data.trade_outcomes import record_trade
                        _strat = sr.chart_signals[0].strategy if getattr(sr, "chart_signals", None) else "UNKNOWN"
                        _pats  = sr.patterns.patterns if getattr(sr, "patterns", None) else None
                        record_trade(
                            result.ticket, ticker, _strat, sr.direction,
                            sr.confidence, sr.entry, sr.sl, sr.tp,
                            patterns=_pats,
                            adx_value=getattr(sr, "adx_value", None),
                            atr_value=getattr(sr, "atr", None),
                            h1_trend=getattr(sr, "daily_trend_direction", None),
                            vol_ratio=getattr(sr, "vol_ratio", None),
                            consensus_count=getattr(sr, "consensus_count", None),
                            pattern_boost_val=getattr(sr, "pattern_boost_val", None),
                            calibrated_conf=getattr(sr, "calibrated_conf", None),
                            base_confidence=getattr(sr, "base_confidence", None),
                            consensus_boost=getattr(sr, "consensus_boost", None),
                            bt_adjustment=getattr(sr, "bt_adjustment", None),
                            live_adjustment=getattr(sr, "live_adjustment", None),
                            news_boost=getattr(sr, "news_boost", None),
                            session_boost=getattr(sr, "session_boost", None),
                        )
                    except Exception as _rte:
                        log.warning("record_trade H1 #%s: %s", result.ticket, _rte)
                else:
                    log.warning(f"  ✗ H1 Order failed: {result.error}")

            # Show open positions
            if not args.dry_run and is_connected():
                positions = get_open_positions()
                if positions:
                    log.info(f"Open positions ({len(positions)}):")
                    for p in positions:
                        log.info(
                            f"  #{p.ticket} {p.direction} {p.symbol} "
                            f"vol={p.volume} profit={p.profit:+.2f}"
                        )

            # ── Heartbeat ────────────────────────────────────────────────
            try:
                _maybe_send_heartbeat(current_positions)
            except Exception as _exc:
                log.warning("heartbeat failed: %s", _exc)

            elapsed = time.time() - poll_start
            if metrics is not None:
                connected = is_connected() if not args.dry_run else True
                metrics.record_poll(duration=elapsed, connected=connected)
            sleep_for = max(1, args.interval - elapsed)
            log.info(f"Next poll in {sleep_for:.0f}s...\n")
            time.sleep(sleep_for)

    except KeyboardInterrupt:
        log.info("\nShutdown requested (Ctrl+C).")
    finally:
        if not args.dry_run:
            disconnect()
        log.info("MT5 Signal Server stopped.")


if __name__ == "__main__":
    run_server(_parse_args())
