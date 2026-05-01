"""Shared runtime primitives used by both the Streamlit UI (app.py) and
the always-on CLI server (mt5_signal_server.py).

Lives outside both top-level modules so the CLI can import without booting
Streamlit. Everything in this module is pure-Python, thread-safe, and
imports nothing from app.py.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


# ── Backtest cache shared across UI scans + scheduled jobs ──────────────────
# Cap concurrent backtests so a cold-start scan with 20 stale tickers doesn't
# spawn 20 threads each fetching 59 days of bars. Daily/weekly learning jobs
# also acquire this semaphore so they don't trample live UI scans.
BT_SEMA = threading.BoundedSemaphore(2)
# Backtest results don't meaningfully change within a day.
BT_TTL_SEC = 86400

_BT_CACHE: dict = {"rates": {}, "ts": {}, "running": set()}
_BT_LOCK = threading.Lock()


def bt_background(ticker: str) -> None:
    """Run a 59-day intraday backtest for one ticker and cache the per-strategy
    win-rate map. Concurrency-capped via BT_SEMA. Safe to call from any thread."""
    with BT_SEMA:
        try:
            from agents.intraday_backtester import run_intraday_backtest
            summary = run_intraday_backtest(ticker, timeframe="5m")
            rates = {r.strategy: r.win_rate for r in summary.results if r.total_trades >= 5}
            with _BT_LOCK:
                _BT_CACHE["rates"][ticker] = rates if rates else {}
                _BT_CACHE["ts"][ticker]    = time.time()
        except Exception as exc:
            logger.warning("ctx=bt_background ticker=%s: %s", ticker, exc)
            with _BT_LOCK:
                _BT_CACHE["rates"][ticker] = None
                _BT_CACHE["ts"][ticker]    = time.time()
        finally:
            with _BT_LOCK:
                _BT_CACHE["running"].discard(ticker)


def refresh_backtest_rates(ticker: str) -> Optional[dict]:
    """Return cached per-strategy WR map for `ticker`. If stale and no thread
    is already working on it, queue one in the background. Returns the
    *current* cached value immediately — callers tolerate `None` on cold
    start (the calibration path treats missing rates as 'no adjustment')."""
    with _BT_LOCK:
        last_ts = _BT_CACHE["ts"].get(ticker, 0)
        stale   = time.time() - last_ts > BT_TTL_SEC
        running = ticker in _BT_CACHE["running"]
        cached  = _BT_CACHE["rates"].get(ticker)
    if stale and not running:
        with _BT_LOCK:
            _BT_CACHE["running"].add(ticker)
        threading.Thread(target=bt_background, args=(ticker,), daemon=True).start()
    return cached
