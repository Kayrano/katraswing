"""
Online pattern win-rate learner.

The hardcoded `win_rate` field on each PatternMatch comes from textbook
sources (Bulkowski, Quantified Strategies) — daily-chart equity-market
backtests. The user's edge is 5m FX/gold; those numbers don't transfer.

This module mines closed trades in data/trade_log.json, attributes each
outcome back to the patterns that fired at entry (recorded by
record_trade since 2026-04-29), and produces an empirical win rate per
pattern with a Beta(1,1) prior.

Effective win rate blending logic:
  - n < ramp:  blend learned with textbook,
               weight = n / ramp
  - n >= ramp: use learned posterior mean only

Posterior mean (Beta(α=1, β=1) prior + n_trades observations):
  (wins + 1) / (trades + 2)

This keeps pattern_score predictions sane during the bootstrap period
(0–30 trades per pattern) and shifts to fully data-driven once we have
a meaningful sample.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)

_DATA_DIR        = Path(__file__).parent.parent / "data"
_TRADE_LOG_PATH  = _DATA_DIR / "trade_log.json"
_STATS_PATH      = _DATA_DIR / "pattern_stats.json"

DEFAULT_RAMP = 30   # trades per pattern before using learned WR alone

_lock: Lock = Lock()


def _load_stats() -> dict[str, dict]:
    if not _STATS_PATH.exists():
        return {}
    try:
        return json.loads(_STATS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("pattern_stats load: %s", exc)
        return {}


def _save_stats(stats: dict[str, dict]) -> None:
    try:
        _STATS_PATH.write_text(
            json.dumps(stats, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("pattern_stats save: %s", exc)


def _aligned(pattern_bias: str, direction: str) -> bool:
    """A pattern only gets credit/blame for a trade when its bias matched the
    position direction. Neutral patterns and counter-direction patterns are
    excluded — we only learn from patterns we actually traded with."""
    if direction == "LONG"  and pattern_bias == "BULLISH":
        return True
    if direction == "SHORT" and pattern_bias == "BEARISH":
        return True
    return False


def recompute_from_trades(trades: list[dict]) -> dict[str, dict]:
    """Rebuild per-pattern (wins, trades, win_rate) from the closed-trade log.

    Idempotent — running it twice on the same data produces the same output.
    Excludes BREAKEVEN trades (no signal) and trades without recorded patterns
    (older trades from before patterns were captured at entry).
    """
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"wins": 0, "trades": 0})
    for t in trades:
        outcome   = t.get("outcome")
        direction = t.get("direction", "")
        patterns  = t.get("patterns") or []
        if outcome not in ("WIN", "LOSS"):
            continue
        if t.get("strategy") == "MT5_IMPORT":
            continue
        for p in patterns:
            name = p.get("name", "")
            bias = p.get("bias", "")
            if not name or not _aligned(bias, direction):
                continue
            counts[name]["trades"] += 1
            if outcome == "WIN":
                counts[name]["wins"] += 1

    stats: dict[str, dict] = {}
    for name, c in counts.items():
        n   = c["trades"]
        wr  = round(c["wins"] / n, 4) if n else 0.0
        stats[name] = {"wins": c["wins"], "trades": n, "win_rate": wr}
    return stats


def refresh(trade_log_path: Optional[Path] = None) -> dict[str, dict]:
    """Recompute stats from trade_log.json and persist to pattern_stats.json."""
    path = trade_log_path or _TRADE_LOG_PATH
    if not path.exists():
        return {}
    try:
        trades = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("pattern_stats refresh: %s", exc)
        return {}
    with _lock:
        stats = recompute_from_trades(trades)
        _save_stats(stats)
    if stats:
        logger.info("pattern_stats refreshed: %d pattern(s) tracked", len(stats))
    return stats


def get_stats(pattern_name: str) -> tuple[int, int]:
    """Return (wins, trades) for a pattern. (0, 0) if untracked."""
    with _lock:
        s = _load_stats().get(pattern_name, {})
    return int(s.get("wins", 0)), int(s.get("trades", 0))


def posterior_win_rate(wins: int, trades: int) -> float:
    """Beta(1,1) prior + binomial likelihood → posterior mean.

    With Beta(α=1, β=1) (uniform) prior and `wins` successes out of `trades`,
    the posterior is Beta(α + wins, β + trades - wins) and its mean is
    (α + wins) / (α + β + trades) = (wins + 1) / (trades + 2).
    """
    return (wins + 1) / (trades + 2)


def effective_win_rate(
    pattern_name: str,
    textbook_wr: float,
    ramp: int = DEFAULT_RAMP,
) -> float:
    """Blend learned win rate with the textbook value.

    n=0   → returns textbook_wr (pure prior)
    n=ramp → returns learned posterior mean alone
    in between: linear interpolation
    """
    wins, n = get_stats(pattern_name)
    if n <= 0:
        return float(textbook_wr)
    learned = posterior_win_rate(wins, n)
    if n >= ramp:
        return float(learned)
    w = n / ramp
    return float(learned * w + textbook_wr * (1.0 - w))


def apply_to_report(report) -> None:
    """In-place update of every PatternMatch.win_rate in a PatternReport with
    the effective (learned-or-textbook-blended) value. Called from
    signal_engine.run_signal so downstream consumers (UI, logging, future
    scoring tweaks) automatically see the calibrated rate.
    """
    if not report or not getattr(report, "patterns", None):
        return
    for m in report.patterns:
        try:
            m.win_rate = round(
                effective_win_rate(m.name, m.win_rate),
                3,
            )
        except Exception:
            continue
    # Refresh aggregate
    if report.patterns:
        report.avg_win_rate = round(
            sum(m.win_rate for m in report.patterns) / len(report.patterns),
            3,
        )
