"""
Trade outcome tracker — records every live auto-trade sent to MT5,
matches closed deals back to those records, and computes per-strategy
win rates that feed back into signal confidence calibration.

Storage: data/trade_log.json  (simple JSON, human-readable)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_LOG_PATH = Path(__file__).parent / "trade_log.json"
_MIN_TRADES = 5   # minimum closed trades before a strategy's win rate is used


# ── Persistence ───────────────────────────────────────────────────────────────

def _load() -> list[dict]:
    try:
        if _LOG_PATH.exists():
            return json.loads(_LOG_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"trade_log load error: {exc}")
    return []


def _save(trades: list[dict]) -> None:
    try:
        _LOG_PATH.write_text(
            json.dumps(trades, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning(f"trade_log save error: {exc}")


# ── Public API ────────────────────────────────────────────────────────────────

def record_trade(
    ticket: int,
    ticker: str,
    strategy: str,
    direction: str,
    confidence: float,
    entry: float,
    sl: float,
    tp: float,
) -> None:
    """Record a newly sent trade. Called immediately after order_send succeeds."""
    trades = _load()
    # Avoid duplicates (e.g. rerun after reconnect)
    if any(t["ticket"] == ticket for t in trades):
        return
    trades.append({
        "ticket":     ticket,
        "ticker":     ticker,
        "strategy":   strategy,
        "direction":  direction,
        "confidence": round(confidence, 4),
        "entry":      entry,
        "sl":         sl,
        "tp":         tp,
        "sent_at":    datetime.utcnow().isoformat(timespec="seconds"),
        "closed_at":  None,
        "profit":     None,
        "outcome":    None,   # "WIN" | "LOSS" | "BREAKEVEN"
    })
    _save(trades)
    logger.info(f"Recorded trade #{ticket} {direction} {ticker} via {strategy}")


def update_outcomes_from_mt5(magic: int = 234100) -> int:
    """
    Pull MT5 history deals and fill in profit/outcome for any open records.
    Returns the number of records updated.
    Requires MetaTrader5 to be initialised (call after ensure_connected).
    """
    try:
        import MetaTrader5 as mt5  # type: ignore[import]
    except ImportError:
        return 0

    trades = _load()
    open_records = [t for t in trades if t["outcome"] is None]
    if not open_records:
        return 0

    open_tickets = {t["ticket"] for t in open_records}

    # Fetch all exit deals with our magic number (entry=1 → OUT / closing deal)
    from datetime import timedelta
    since = datetime.utcnow() - timedelta(days=90)
    raw_deals = mt5.history_deals_get(since, datetime.utcnow())
    if raw_deals is None:
        return 0

    # position_id on an exit deal matches the ticket of the opening order
    closed: dict[int, dict] = {}
    for d in raw_deals:
        if d.magic != magic:
            continue
        if d.entry != 1:   # 1 = OUT (closing deal)
            continue
        if d.position_id in open_tickets:
            closed[d.position_id] = {
                "profit":    d.profit,
                "closed_at": datetime.utcfromtimestamp(d.time).isoformat(timespec="seconds"),
            }

    if not closed:
        return 0

    updated = 0
    for t in trades:
        if t["ticket"] in closed:
            info = closed[t["ticket"]]
            t["profit"]    = round(info["profit"], 2)
            t["closed_at"] = info["closed_at"]
            p = info["profit"]
            t["outcome"]   = "WIN" if p > 0 else ("LOSS" if p < 0 else "BREAKEVEN")
            updated += 1

    if updated:
        _save(trades)
        logger.info(f"Updated {updated} trade outcome(s) from MT5 history")
    return updated


def compute_win_rates(min_trades: int = _MIN_TRADES) -> dict[str, float]:
    """
    Return {strategy: win_rate} for strategies with >= min_trades closed results.
    Used as backtest_win_rates in run_signal() to calibrate confidence scores.
    """
    trades = _load()
    from collections import defaultdict
    buckets: dict[str, list[bool]] = defaultdict(list)
    for t in trades:
        if t["outcome"] in ("WIN", "LOSS"):   # exclude BREAKEVEN from ratio
            buckets[t["strategy"]].append(t["outcome"] == "WIN")

    return {
        s: sum(results) / len(results)
        for s, results in buckets.items()
        if len(results) >= min_trades
    }


def get_summary() -> dict:
    """Return a summary dict for display in the UI."""
    trades = _load()
    closed  = [t for t in trades if t["outcome"] is not None]
    wins    = [t for t in closed  if t["outcome"] == "WIN"]
    losses  = [t for t in closed  if t["outcome"] == "LOSS"]
    total_profit = sum(t["profit"] for t in closed if t["profit"] is not None)

    from collections import defaultdict
    by_strategy: dict[str, dict] = defaultdict(lambda: {"trades": 0, "wins": 0, "profit": 0.0})
    for t in closed:
        s = t["strategy"]
        by_strategy[s]["trades"] += 1
        if t["outcome"] == "WIN":
            by_strategy[s]["wins"] += 1
        if t["profit"] is not None:
            by_strategy[s]["profit"] += t["profit"]

    strategy_stats = []
    for s, v in sorted(by_strategy.items(), key=lambda x: -x[1]["trades"]):
        wr = v["wins"] / v["trades"] if v["trades"] else 0
        strategy_stats.append({
            "strategy": s,
            "trades":   v["trades"],
            "win_rate": round(wr, 3),
            "profit":   round(v["profit"], 2),
        })

    return {
        "total_sent":   len(trades),
        "total_closed": len(closed),
        "total_open":   len(trades) - len(closed),
        "wins":         len(wins),
        "losses":       len(losses),
        "win_rate":     round(len(wins) / len(closed), 3) if closed else None,
        "total_profit": round(total_profit, 2),
        "by_strategy":  strategy_stats,
        "all_trades":   list(reversed(trades)),  # newest first
    }
