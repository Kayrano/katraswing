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
from datetime import datetime, date

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
DEFAULT_TICKERS      = ["NQ=F", "ES=F", "XAUUSD=X"]
DEFAULT_DISPLAY_NAMES = {"NQ=F": "NQ Mini", "ES=F": "ES Mini", "XAUUSD=X": "Gram Gold"}
DEFAULT_INTERVAL_SEC = 60       # poll every 60 seconds
DEFAULT_MIN_CONF     = 0.60     # minimum confidence to enter
FINNHUB_API_KEY      = ""       # optional — set here or via --finnhub-key


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

    # Dedup set: track which (ticker, direction, session_date) have open positions
    sent_signals: set[str] = set()

    try:
        while True:
            poll_start = time.time()
            today = date.today()

            # Clear stale sent signals from prior sessions
            stale = {k for k in sent_signals if not k.endswith(str(today))}
            sent_signals -= stale

            # Reconcile with MT5 open positions to avoid dedup drift
            if not args.dry_run and is_connected():
                open_symbols = {p.symbol for p in get_open_positions()}
            else:
                open_symbols = set()

            for ticker in args.tickers:
                display = DEFAULT_DISPLAY_NAMES.get(ticker, ticker)
                log.info(f"Polling {display} ({ticker})...")

                try:
                    sr = run_signal(
                        ticker=ticker,
                        finnhub_api_key=args.finnhub_key,
                        account_size=args.account_size,
                        risk_pct=args.risk_pct,
                        display_name=display,
                    )
                except Exception as exc:
                    log.error(f"Signal engine error for {ticker}: {exc}")
                    continue

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

                # Log patterns with win rates
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
                else:
                    log.warning(f"  ✗ Order failed: {result.error}")

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
