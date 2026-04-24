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


def import_all_mt5_history(days: int = 90) -> int:
    """
    Import ALL closed trades from MT5 history (any magic number / manually opened trades).

    Pairs IN deals (entry=0, opening) with OUT deals (entry=1, closing) by position_id.
    Trades already in the log have their outcome filled in if still open.
    New trades (not sent by Katraswing) are added with strategy='MT5_IMPORT'.

    Returns the number of records newly added or updated.
    """
    try:
        import MetaTrader5 as mt5  # type: ignore[import]
    except ImportError:
        return 0

    from collections import defaultdict
    from datetime import timedelta

    since    = datetime.utcnow() - timedelta(days=days)
    raw      = mt5.history_deals_get(since, datetime.utcnow())
    if raw is None:
        return 0

    # Group deals by position_id → {in: deal, out: deal}
    by_pos: dict[int, dict] = defaultdict(lambda: {"in": None, "out": None})
    for d in raw:
        if d.entry == 0:
            by_pos[d.position_id]["in"] = d
        elif d.entry == 1:
            by_pos[d.position_id]["out"] = d

    trades          = _load()
    existing_by_tk  = {t["ticket"]: t for t in trades}
    imported        = 0

    for pos_id, pair in by_pos.items():
        out_d = pair["out"]
        in_d  = pair["in"]
        if out_d is None:
            continue   # position still open

        # Ticket of the opening order = position_id in MT5
        ticket = int(pos_id)
        gross  = round(float(out_d.profit), 2)
        comm   = round(float(getattr(out_d, "commission", 0.0)), 2)
        swap   = round(float(getattr(out_d, "swap", 0.0)), 2)
        net    = round(gross + comm + swap, 2)
        outcome = "WIN" if net > 0 else ("LOSS" if net < 0 else "BREAKEVEN")
        close_iso = datetime.utcfromtimestamp(out_d.time).isoformat(timespec="seconds")

        if ticket in existing_by_tk:
            rec = existing_by_tk[ticket]
            if rec["outcome"] is None:
                rec["profit"]    = net
                rec["closed_at"] = close_iso
                rec["outcome"]   = outcome
                # enrich with actual exit price if present
                rec.setdefault("close_price", round(float(out_d.price), 5))
                imported += 1
            continue

        direction  = "LONG" if (in_d and in_d.type == 0) else "SHORT"
        entry_p    = round(float(in_d.price), 5)  if in_d else 0.0
        open_iso   = datetime.utcfromtimestamp(in_d.time).isoformat(timespec="seconds") if in_d else close_iso
        volume     = float(in_d.volume) if in_d else float(out_d.volume)
        symbol     = str(out_d.symbol)

        trades.append({
            "ticket":      ticket,
            "ticker":      symbol,
            "strategy":    "MT5_IMPORT",
            "direction":   direction,
            "confidence":  0.0,
            "entry":       entry_p,
            "sl":          0.0,
            "tp":          0.0,
            "close_price": round(float(out_d.price), 5),
            "volume":      volume,
            "gross":       gross,
            "commission":  comm,
            "swap":        swap,
            "sent_at":     open_iso,
            "closed_at":   close_iso,
            "profit":      net,
            "outcome":     outcome,
        })
        existing_by_tk[ticket] = trades[-1]
        imported += 1

    if imported:
        _save(trades)
        logger.info(f"Imported/updated {imported} trade(s) from MT5 history ({days}d)")
    return imported


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
