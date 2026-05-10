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
    "EURUSD=X", "GBPUSD=X", "USDJPY=X",
    "ES=F", "GC=F",
]
DEFAULT_DISPLAY_NAMES = {
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
    "USDJPY=X": "USD/JPY",
    "ES=F":     "ES Mini",
    "GC=F":     "Gold",
}
DEFAULT_INTERVAL_SEC = 60       # poll every 60 seconds
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

    # Dedup set: track which (ticker, direction, session_date) have open positions
    sent_signals: set[str] = set()

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

            # Clear stale sent signals from prior sessions
            stale = {k for k in sent_signals if not k.endswith(str(today))}
            sent_signals -= stale

            # Reconcile with MT5 open positions to avoid dedup drift
            if not args.dry_run and is_connected():
                open_symbols = {p.symbol for p in get_open_positions()}
            else:
                open_symbols = set()

            # ── Phase 1: fetch signals in parallel ───────────────────────
            # Each run_signal is network-bound (yfinance/Finnhub) and
            # independent. Bar-level cache hits return immediately.
            workers = min(MAX_SCAN_WORKERS, len(args.tickers)) or 1
            signal_results: dict[str, object] = {}
            log.info(f"── Scanning {len(args.tickers)} ticker(s) (workers={workers}) ──")
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
                    log.info(f"{display}: NO TRADE (conf={sr.confidence:.1%})")
                    continue

                if sr.confidence < args.min_confidence:
                    log.info(
                        f"{display}: {sr.direction} @ {sr.confidence:.1%} — "
                        f"below threshold {args.min_confidence:.0%}, skipping."
                    )
                    continue

                top_patterns = [(p.name, p.win_rate) for p in sr.patterns.patterns[:3]]
                pattern_str  = " | ".join(f"{n} {w:.0%}" for n, w in top_patterns) or "—"

                log.info(
                    f"★ SIGNAL: {display} {sr.direction} | "
                    f"conf={sr.confidence:.1%} | {pattern_str}"
                )
                log.info(_format_signal(sr))

                # Dedup check
                key = _signal_key(ticker, sr.direction, today)
                if key in sent_signals:
                    log.info(f"  [dedup] Already entered {sr.direction} {display} today. Skipping.")
                    continue

                # Paper-mode strategies/symbols: signal still passes calibration
                # but the broker round-trip is skipped. trade_log records a
                # paper row so the weekly auto-promotion harness can mature it.
                if getattr(sr, "paper_only", False):
                    log.info(f"  [paper] {sr.paper_reason or 'strategy'} — order not sent")
                    sent_signals.add(key)
                    continue

                if args.dry_run:
                    log.info("  [dry-run] Would send to MT5.")
                    sent_signals.add(key)
                    continue

                # Reconnect if needed
                if not ensure_connected(args.mt5_path):
                    log.error("  MT5 reconnect failed. Skipping order.")
                    continue

                result = send_from_signal_result(sr)
                if result.success:
                    log.info(
                        f"  ✓ Order #{result.ticket} placed | "
                        f"{result.direction} {result.symbol} @ {result.entry:.4f}"
                    )
                    sent_signals.add(key)
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
                    f"★ H1 SIGNAL: {display} {sr.direction} | "
                    f"conf={sr.confidence:.1%} | strategy={sr.chart_signals[0].strategy if sr.chart_signals else '?'}"
                )
                key = _signal_key(f"H1:{ticker}", sr.direction, today)
                if key in sent_signals:
                    log.info(f"  [dedup] Already entered H1 {sr.direction} {display} today.")
                    continue
                # H1 strategies are paper_only at launch — no broker round-trip.
                if getattr(sr, "paper_only", True):
                    log.info(f"  [H1 paper] signal recorded for calibration")
                    sent_signals.add(key)
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
