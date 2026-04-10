"""
Katrabot — 24/7 Automated Trading Engine
=========================================
Runs as a background thread inside the Katraswing Streamlit app.
Uses APScheduler to fire a scan cycle every SCAN_INTERVAL_MINUTES.

Supports multiple concurrent users: each user_id gets its own scheduler
and state dict.  Alpaca credentials are passed explicitly so different
users can trade against their own accounts.

Cycle logic:
  1.  Check Alpaca market clock — skip if closed.
  2.  Check daily P&L vs loss limit — halt if breached.
  3.  Fetch open positions.
  4.  Re-score held positions — exit any that drop below AVOID_THRESHOLD.
  5.  Run Katraswing screener to find top candidates (fast, vectorised).
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
    is_daily_loss_limit_hit, has_upcoming_earnings,
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

# ── Per-user singleton state ───────────────────────────────────────────────────
# Keyed by user_id (str) so multiple users can run independent bots.
_schedulers: dict = {}
_states:     dict = {}
_lock = threading.Lock()


def _default_state() -> dict:
    return {
        "running":            False,
        "status_msg":         "Bot not started.",
        "last_run_utc":       None,
        "daily_pnl":          0.0,
        "total_buys_today":   0,
        "total_sells_today":  0,
        "last_trades":        [],
        "errors_this_cycle":  0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def start_bot(user_id: str, api_key: str, secret_key: str, is_paper: bool = True) -> str:
    """Start the bot for a specific user with their Alpaca credentials."""
    with _lock:
        if user_id not in _states:
            _states[user_id] = _default_state()

        state = _states[user_id]
        if state["running"]:
            return "Bot is already running."

        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.interval    import IntervalTrigger
        except ImportError:
            return "ERROR: apscheduler not installed. Run: pip install apscheduler"

        init_db()

        creds = {"api_key": api_key, "secret_key": secret_key, "is_paper": is_paper}

        scheduler = BackgroundScheduler(daemon=True, timezone="UTC")
        scheduler.add_job(
            _run_cycle,
            trigger=IntervalTrigger(minutes=SCAN_INTERVAL_MINUTES),
            id=f"trading_cycle_{user_id}",
            kwargs={"user_id": user_id, "creds": creds},
            max_instances=1,
            coalesce=True,
            misfire_grace_time=60,
        )
        scheduler.start()
        _schedulers[user_id] = scheduler

        state["running"]    = True
        state["status_msg"] = f"Bot running — scanning every {SCAN_INTERVAL_MINUTES} min."
        log.info(f"Katrabot started for user {user_id[:8]}…")

        # Fire first cycle immediately
        threading.Thread(
            target=_run_cycle,
            kwargs={"user_id": user_id, "creds": creds},
            daemon=True,
            name=f"katrabot-init-{user_id[:8]}",
        ).start()

        return state["status_msg"]


def stop_bot(user_id: str) -> str:
    with _lock:
        state = _states.get(user_id)
        if not state or not state["running"]:
            return "Bot is not running."
        try:
            scheduler = _schedulers.pop(user_id, None)
            if scheduler:
                scheduler.shutdown(wait=False)
        except Exception:
            pass
        state["running"]    = False
        state["status_msg"] = "Bot stopped."
        log.info(f"Katrabot stopped for user {user_id[:8]}…")
        return state["status_msg"]


def get_state(user_id: str) -> dict:
    return dict(_states.get(user_id, _default_state()))


# ══════════════════════════════════════════════════════════════════════════════
# Core scan cycle
# ══════════════════════════════════════════════════════════════════════════════

def _run_cycle(user_id: str, creds: dict):
    """One full analysis + execution cycle for one user."""
    state = _states.setdefault(user_id, _default_state())

    ak  = creds.get("api_key")
    sk  = creds.get("secret_key")
    ppr = creds.get("is_paper", True)

    ts               = datetime.utcnow().isoformat(timespec="seconds")
    tickers_scanned  = 0
    trades_executed  = 0
    positions_closed = 0
    errors           = 0
    notes            = []

    try:
        # ── 1. Market hours ───────────────────────────────────────────────────
        market_open = is_market_open(api_key=ak, secret_key=sk, is_paper=ppr)
        state["last_run_utc"] = ts

        if not market_open:
            state["status_msg"] = f"[{ts}]  Market CLOSED — waiting."
            log_run(market_open=False, notes="Market closed.")
            return

        # ── 2. Daily P&L & loss-limit check ──────────────────────────────────
        try:
            acct      = get_account(api_key=ak, secret_key=sk, is_paper=ppr)
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
            positions = get_positions(api_key=ak, secret_key=sk, is_paper=ppr)
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
                    close_position(symbol, api_key=ak, secret_key=sk, is_paper=ppr)
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
            positions = get_positions(api_key=ak, secret_key=sk, is_paper=ppr)
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

                    if score < BUY_THRESHOLD:
                        log_trade(ticker=ticker, action="SKIP", score=score,
                                  reason=f"Score {score:.1f} < {BUY_THRESHOLD}")
                        continue

                    if setup.direction != "LONG":
                        log_trade(ticker=ticker, action="SKIP", score=score,
                                  reason=f"Direction: {setup.direction}")
                        continue

                    has_earn, earn_days = has_upcoming_earnings(ticker)
                    if has_earn:
                        log_trade(ticker=ticker, action="SKIP", score=score,
                                  reason=f"Earnings in {earn_days}d")
                        continue

                    qty = compute_position_size(setup.entry, setup.stop_loss)
                    if qty <= 0:
                        log_trade(ticker=ticker, action="SKIP", score=score,
                                  reason="Position size = 0 (risk params invalid)")
                        continue

                    # ── Execute BUY ───────────────────────────────────────────
                    buy_resp = place_market_order(ticker, qty, "buy",
                                                  api_key=ak, secret_key=sk, is_paper=ppr)
                    order_id = buy_resp.get("id", "")
                    log.info(f"BUY {qty}x {ticker} @ ~${setup.entry:.2f}  (score {score:.1f})")

                    time.sleep(2)
                    try:
                        place_oco_order(ticker, qty, setup.stop_loss, setup.take_profit,
                                        api_key=ak, secret_key=sk, is_paper=ppr)
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

                    try:
                        positions = get_positions(api_key=ak, secret_key=sk, is_paper=ppr)
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
    try:
        from data.screener import run_screener
        results  = run_screener(preset_name="All (no filter)")
        held     = {p.get("symbol", "").upper() for p in current_positions}
        filtered = [r for r in results if r.get("ticker", "").upper() not in held]
        filtered.sort(key=lambda x: x.get("score", 0), reverse=True)
        return [r["ticker"] for r in filtered[:TOP_CANDIDATES]]
    except Exception as e:
        log.warning(f"Screener failed, falling back to config universe: {e}")
        from bot.config import WATCHLIST_EXTRAS
        from data.screener import SWING_UNIVERSE
        held     = {p.get("symbol", "").upper() for p in current_positions}
        fallback = [t for t in (SWING_UNIVERSE + WATCHLIST_EXTRAS) if t not in held]
        return fallback[:TOP_CANDIDATES]
