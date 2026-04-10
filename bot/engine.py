"""
Katrabot — 24/7 Automated Trading Engine
=========================================
Runs as a background thread inside the Katraswing Streamlit app.
Uses APScheduler to fire a scan cycle every SCAN_INTERVAL_MINUTES.

Cycle logic:
  1.  Check Alpaca market clock — skip if closed.
  2.  Check daily P&L vs loss limit — halt if breached.
  3.  Fetch open positions.
  4.  Re-score held positions — exit any that drop below AVOID_THRESHOLD.
  5.  Run Katraswing screener to find top candidates (fast, vectorized).
  6.  Run full run_analysis() on top N candidates.
  7.  Buy anything scoring ≥ BUY_THRESHOLD that passes all risk checks.
  8.  Place OCO (stop + take-profit) on every new position.
  9.  Log everything to bot_trades.db.
"""

import time
import threading
import logging
from datetime import datetime
from typing import Optional

# ── Bot dependencies ───────────────────────────────────────────────────────────
from bot.config import (
    SCAN_INTERVAL_MINUTES, BUY_THRESHOLD, AVOID_THRESHOLD, TOP_CANDIDATES,
)
from bot.logger  import init_db, log_trade, log_run, get_recent_trades
from bot.risk_manager import (
    compute_position_size, can_open_new_position,
    is_daily_loss_limit_hit, has_upcoming_earnings, already_in_portfolio,
)
from broker.alpaca import (
    get_account, get_positions, is_market_open,
    place_market_order, place_oco_order, close_position,
)

log = logging.getLogger("katrabot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
)

# ── Module-level singleton state ───────────────────────────────────────────────
# Using module globals so the scheduler survives Streamlit script reruns.
_scheduler = None          # APScheduler BackgroundScheduler instance
_lock      = threading.Lock()

# Shared state dict — read by the UI tab (bot_renderer.py)
state = {
    "running":            False,
    "status_msg":         "Bot not started.",
    "last_run_utc":       None,
    "daily_pnl":          0.0,
    "total_buys_today":   0,
    "total_sells_today":  0,
    "last_trades":        [],   # last 10 log entries for quick display
    "errors_this_cycle":  0,
}


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def start_bot() -> str:
    global _scheduler
    with _lock:
        if state["running"]:
            return "Bot is already running."

        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.interval    import IntervalTrigger
        except ImportError:
            return "ERROR: apscheduler not installed. Run: pip install apscheduler"

        init_db()

        _scheduler = BackgroundScheduler(daemon=True, timezone="UTC")
        _scheduler.add_job(
            _run_cycle,
            trigger=IntervalTrigger(minutes=SCAN_INTERVAL_MINUTES),
            id="trading_cycle",
            max_instances=1,
            coalesce=True,
            misfire_grace_time=60,
        )
        _scheduler.start()

        state["running"]    = True
        state["status_msg"] = f"Bot running — scanning every {SCAN_INTERVAL_MINUTES} min."
        log.info("Katrabot started.")

        # Fire first cycle immediately in a daemon thread
        threading.Thread(target=_run_cycle, daemon=True, name="katrabot-init").start()
        return state["status_msg"]


def stop_bot() -> str:
    global _scheduler
    with _lock:
        if not state["running"]:
            return "Bot is not running."
        try:
            if _scheduler:
                _scheduler.shutdown(wait=False)
        except Exception:
            pass
        _scheduler          = None
        state["running"]    = False
        state["status_msg"] = "Bot stopped."
        log.info("Katrabot stopped.")
        return state["status_msg"]


def get_state() -> dict:
    return dict(state)


# ══════════════════════════════════════════════════════════════════════════════
# Core scan cycle
# ══════════════════════════════════════════════════════════════════════════════

def _run_cycle():
    """One full analysis + execution cycle."""
    ts               = datetime.utcnow().isoformat(timespec="seconds")
    tickers_scanned  = 0
    trades_executed  = 0
    positions_closed = 0
    errors           = 0
    notes            = []

    try:
        # ── 1. Market hours ───────────────────────────────────────────────────
        market_open = is_market_open()
        state["last_run_utc"] = ts

        if not market_open:
            state["status_msg"] = f"[{ts}]  Market CLOSED — waiting."
            log_run(market_open=False, notes="Market closed.")
            return

        # ── 2. Daily P&L & loss-limit check ──────────────────────────────────
        try:
            acct      = get_account()
            daily_pnl = float(acct.get("equity", 0)) - float(acct.get("last_equity", 0))
            state["daily_pnl"] = daily_pnl
        except Exception as e:
            daily_pnl = 0.0
            errors   += 1
            notes.append(f"Account error: {e}")

        if is_daily_loss_limit_hit(daily_pnl):
            msg = f"Daily loss limit hit (${daily_pnl:,.0f}). Trading halted for today."
            state["status_msg"] = msg
            log_run(market_open=True, daily_pnl=daily_pnl, errors=0, notes=msg)
            return

        # ── 3. Current positions ──────────────────────────────────────────────
        try:
            positions = get_positions()
        except Exception as e:
            errors   += 1
            notes.append(f"Positions fetch error: {e}")
            positions = []

        # ── 4. Exit any held positions that now score AVOID ───────────────────
        for pos in list(positions):
            symbol = pos.get("symbol", "")
            if not symbol:
                continue
            try:
                from agents.orchestrator import run_analysis
                report = run_analysis(symbol)
                score  = report.score.total_score
                tickers_scanned += 1

                if score < AVOID_THRESHOLD:
                    close_position(symbol)
                    positions_closed += 1
                    state["total_sells_today"] += 1
                    log_trade(
                        ticker=symbol, action="SELL",
                        price=float(pos.get("current_price", 0)),
                        qty=int(float(pos.get("qty", 1))),
                        score=score,
                        reason=f"Score {score:.1f} fell below AVOID threshold {AVOID_THRESHOLD}",
                    )
                    notes.append(f"CLOSED {symbol} (score {score:.1f})")

            except Exception as e:
                errors += 1
                notes.append(f"Re-score error {symbol}: {e}")

        # Refresh positions after exits
        try:
            positions = get_positions()
        except Exception:
            pass

        # ── 5. Check if we can open new positions ─────────────────────────────
        can_open, no_open_reason = can_open_new_position(positions)
        if not can_open:
            notes.append(f"No new entries: {no_open_reason}")
        else:
            # ── 6. Fast screener scan to find candidates ──────────────────────
            candidates = _get_top_candidates(positions)

            # ── 7. Full analysis + execution on top candidates ────────────────
            for ticker in candidates:
                if not can_open:
                    break

                try:
                    from agents.orchestrator import run_analysis
                    report = run_analysis(ticker)
                    score  = report.score.total_score
                    setup  = report.trade_setup
                    tickers_scanned += 1

                    # Skip non-BUY scores
                    if score < BUY_THRESHOLD:
                        log_trade(ticker=ticker, action="SKIP", score=score,
                                  reason=f"Score {score:.1f} < {BUY_THRESHOLD}")
                        continue

                    # Only take LONG direction
                    if setup.direction != "LONG":
                        log_trade(ticker=ticker, action="SKIP", score=score,
                                  reason=f"Direction: {setup.direction}")
                        continue

                    # Earnings proximity check
                    has_earn, earn_days = has_upcoming_earnings(ticker)
                    if has_earn:
                        log_trade(ticker=ticker, action="SKIP", score=score,
                                  reason=f"Earnings in {earn_days}d")
                        continue

                    # Position sizing
                    qty = compute_position_size(setup.entry, setup.stop_loss)
                    if qty <= 0:
                        log_trade(ticker=ticker, action="SKIP", score=score,
                                  reason="Position size = 0 (risk params invalid)")
                        continue

                    # ── Execute BUY ───────────────────────────────────────────
                    buy_resp = place_market_order(ticker, qty, "buy")
                    order_id = buy_resp.get("id", "")
                    log.info(f"BUY {qty}x {ticker} @ ~${setup.entry:.2f}  (score {score:.1f})")

                    # Wait briefly for market order to fill, then place OCO
                    time.sleep(2)
                    try:
                        place_oco_order(ticker, qty, setup.stop_loss, setup.take_profit)
                        log.info(f"OCO set — SL ${setup.stop_loss:.2f}  TP ${setup.take_profit:.2f}")
                    except Exception as oco_err:
                        notes.append(f"OCO failed {ticker}: {oco_err}")

                    trades_executed            += 1
                    state["total_buys_today"]  += 1

                    log_trade(
                        ticker=ticker, action="BUY",
                        price=setup.entry, qty=qty,
                        stop_loss=setup.stop_loss, take_profit=setup.take_profit,
                        score=score,
                        reason=f"Score {score:.1f} — {setup.direction}",
                        order_id=order_id,
                    )
                    notes.append(f"BOUGHT {qty}x {ticker} @ ${setup.entry:.2f} (score {score:.1f})")

                    # Refresh limits
                    try:
                        positions = get_positions()
                    except Exception:
                        pass
                    can_open, no_open_reason = can_open_new_position(positions)
                    if not can_open:
                        notes.append(f"Position limit: {no_open_reason}")

                except Exception as e:
                    errors += 1
                    notes.append(f"Scan error {ticker}: {e}")
                    log.exception(f"Error scanning {ticker}")

        # ── 8. Update state & persist run log ─────────────────────────────────
        state["errors_this_cycle"] = errors
        state["last_trades"]       = get_recent_trades(10)
        state["status_msg"]        = (
            f"[{ts} UTC]  "
            f"Scanned {tickers_scanned} | "
            f"Bought {trades_executed} | "
            f"Closed {positions_closed} | "
            f"Errors {errors}"
        )

        log_run(
            market_open=True,
            tickers_scanned=tickers_scanned,
            trades_executed=trades_executed,
            positions_closed=positions_closed,
            errors=errors,
            daily_pnl=state["daily_pnl"],
            notes=" | ".join(notes) if notes else "Clean cycle.",
        )
        log.info(state["status_msg"])

    except Exception as fatal:
        state["status_msg"] = f"FATAL: {fatal}"
        log.exception("Fatal error in bot cycle")
        try:
            log_run(market_open=False, errors=1, notes=f"FATAL: {fatal}")
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _get_top_candidates(current_positions: list) -> list:
    """
    Run the fast screener and return up to TOP_CANDIDATES tickers
    that are not already held, sorted by score descending.
    """
    try:
        from data.screener import run_screener
        results = run_screener(preset_name="All (no filter)")

        held    = {p.get("symbol", "").upper() for p in current_positions}
        filtered = [
            r for r in results
            if r.get("ticker", "").upper() not in held
        ]

        # Sort by score descending
        filtered.sort(key=lambda x: x.get("score", 0), reverse=True)

        return [r["ticker"] for r in filtered[:TOP_CANDIDATES]]

    except Exception as e:
        log.warning(f"Screener failed, falling back to config universe: {e}")
        # Fallback: use config universe directly
        from bot.config import WATCHLIST_EXTRAS
        from data.screener import SWING_UNIVERSE
        held    = {p.get("symbol", "").upper() for p in current_positions}
        fallback = [t for t in (SWING_UNIVERSE + WATCHLIST_EXTRAS) if t not in held]
        return fallback[:TOP_CANDIDATES]
