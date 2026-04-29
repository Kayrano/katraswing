"""
Trade-manager intervention learning.

Joins data/assessment_log.json (every health-scored decision the trade
manager has made on each open position) with data/trade_log.json (the
final outcome of each ticket) to produce per-(strategy, action) quality
statistics. The trade manager reads these stats to slightly bias its
own health-score input — closing less eagerly when its recent CLOSE
calls have been wrong (closed winners), and slightly more eagerly when
they've been right (closed losers).

The bias is small (bounded to ±0.05) and only affects the *input* to
_decide_action, never the decision tree itself, so a bad learning
signal can't dramatically change exit behaviour. Surfaced via
`summarize_for_strategy` for the dashboard.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DATA_DIR        = Path(__file__).parent.parent / "data"
_ASSESSMENT_LOG  = _DATA_DIR / "assessment_log.json"
_TRADE_LOG_PATH  = _DATA_DIR / "trade_log.json"

# Learning gate: don't bias trade_manager until we have at least this many
# closed trades that received a given action for a given strategy. Below
# the threshold the offset is 0.
MIN_SAMPLES_FOR_BIAS = 10

# Maximum health-score offset the learner can apply. Bounded so a noisy
# learning signal can't reshape the decision tree.
MAX_BIAS_OFFSET = 0.05


@dataclass
class ActionStats:
    strategy:    str
    action:      str
    count:       int = 0
    wins:        int = 0
    losses:      int = 0
    total_profit: float = 0.0

    @property
    def win_rate(self) -> Optional[float]:
        n = self.wins + self.losses
        return (self.wins / n) if n else None

    @property
    def avg_profit(self) -> Optional[float]:
        return (self.total_profit / self.count) if self.count else None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["win_rate"]   = self.win_rate
        d["avg_profit"] = self.avg_profit
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> list:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception as exc:
        logger.warning("intervention_stats load %s: %s", path.name, exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Core: join + bucket
# ─────────────────────────────────────────────────────────────────────────────

# Actions we're tracking (skip HOLD — it's the no-op default and would
# dominate the counts without telling us anything useful).
_TRACKED_ACTIONS = {
    "CLOSE", "PARTIAL_CLOSE",
    "MODIFY_SL", "MODIFY_TP", "MODIFY_BOTH",
}


def compute_intervention_stats(
    assessments: Optional[list] = None,
    trades:      Optional[list] = None,
) -> dict[tuple[str, str], ActionStats]:
    """For each closed trade, attribute its outcome to every action it
    received during its lifetime. Returns {(strategy, action): ActionStats}.

    A trade may receive multiple actions over its life (e.g. MODIFY_SL
    twice, then PARTIAL_CLOSE, then a final CLOSE). Each action gets
    credited the trade's final outcome — actions are decisions, and the
    outcome is what we judge them by.
    """
    if assessments is None:
        assessments = _load_json(_ASSESSMENT_LOG)
    if trades is None:
        trades = _load_json(_TRADE_LOG_PATH)

    # Index closed trades by ticket → (strategy, outcome, profit)
    trade_by_ticket: dict[int, dict] = {}
    for t in trades:
        outcome = t.get("outcome")
        if outcome not in ("WIN", "LOSS"):
            continue
        ticket = t.get("ticket")
        if ticket is None:
            continue
        trade_by_ticket[int(ticket)] = {
            "strategy": t.get("strategy", "UNKNOWN"),
            "outcome":  outcome,
            "profit":   float(t.get("profit") or 0.0),
        }

    # Walk assessments. Skip those whose ticket has no recorded outcome (still
    # open or never logged). Skip HOLD assessments (the trade_manager isn't
    # acting; nothing to credit/blame).
    buckets: dict[tuple[str, str], ActionStats] = {}
    seen: set[tuple[int, str, str]] = set()   # dedupe per (ticket, action, assessed_at)

    for a in assessments:
        ticket = a.get("ticket")
        action = a.get("action", "HOLD")
        if ticket is None or action not in _TRACKED_ACTIONS:
            continue
        ti = int(ticket)
        if ti not in trade_by_ticket:
            continue
        # Dedupe — assessment_log can grow with retries; only count one
        # assessment per (ticket, action, timestamp).
        key = (ti, action, a.get("assessed_at", ""))
        if key in seen:
            continue
        seen.add(key)

        info = trade_by_ticket[ti]
        bk   = (info["strategy"], action)
        if bk not in buckets:
            buckets[bk] = ActionStats(strategy=info["strategy"], action=action)
        st = buckets[bk]
        st.count += 1
        if info["outcome"] == "WIN":
            st.wins += 1
        else:
            st.losses += 1
        st.total_profit += info["profit"]

    return buckets


# ─────────────────────────────────────────────────────────────────────────────
# UI / dashboard helpers
# ─────────────────────────────────────────────────────────────────────────────

def summarize_for_strategy(strategy: str) -> dict[str, dict]:
    """Return per-action stats for one strategy, suitable for UI display."""
    all_stats = compute_intervention_stats()
    return {
        action: stats.to_dict()
        for (strat, action), stats in all_stats.items()
        if strat == strategy
    }


def summarize_all() -> dict[str, dict[str, dict]]:
    """Return {strategy: {action: stats_dict}} for every tracked combination."""
    all_stats = compute_intervention_stats()
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for (strat, action), stats in all_stats.items():
        out[strat][action] = stats.to_dict()
    return dict(out)


# ─────────────────────────────────────────────────────────────────────────────
# Trade-manager learning hook: per-strategy health bias
# ─────────────────────────────────────────────────────────────────────────────

# Cached per-call so the hook is cheap when invoked from _decide_action.
# Reset by calling reset_cache() — useful for tests.
_bias_cache: dict[str, float] = {}
_bias_cache_built: bool = False


def reset_cache() -> None:
    global _bias_cache, _bias_cache_built
    _bias_cache = {}
    _bias_cache_built = False


def _build_bias_cache() -> None:
    """Compute a per-strategy health-score bias from recent CLOSE outcomes.

    Logic:
      - For each strategy with ≥ MIN_SAMPLES_FOR_BIAS CLOSE actions, compute
        the win rate of trades that received a CLOSE.
      - High WR (≥ 0.65): we've been closing winners → bias health UP (close
        less eagerly). Bias = +(MAX_BIAS_OFFSET).
      - Low WR (≤ 0.35): we've been closing losers correctly → bias DOWN
        (slightly more eager). Bias = -(MAX_BIAS_OFFSET / 2). The downside
        bias is conservative: closing losers is the trade-manager's job;
        we only want to encourage it modestly.
      - In between: no bias.
    """
    global _bias_cache, _bias_cache_built
    cache: dict[str, float] = {}
    try:
        all_stats = compute_intervention_stats()
        # Aggregate CLOSE-action win rate per strategy
        per_strategy_close: dict[str, ActionStats] = {}
        for (strat, action), stats in all_stats.items():
            if action != "CLOSE":
                continue
            per_strategy_close[strat] = stats

        for strat, stats in per_strategy_close.items():
            if stats.count < MIN_SAMPLES_FOR_BIAS:
                continue
            wr = stats.win_rate or 0.5
            if wr >= 0.65:
                cache[strat] = +MAX_BIAS_OFFSET
            elif wr <= 0.35:
                cache[strat] = -MAX_BIAS_OFFSET / 2.0
    except Exception as exc:
        logger.debug("intervention bias build skipped: %s", exc)
    _bias_cache = cache
    _bias_cache_built = True


def get_health_bias(strategy: str) -> float:
    """Return the per-strategy health-score bias in [-MAX_BIAS_OFFSET/2,
    +MAX_BIAS_OFFSET]. Always 0 when there isn't enough data.

    The trade_manager adds this to its raw health_score before threshold
    comparisons. Bounded so a noisy learning signal can't reshape the
    decision tree dramatically.
    """
    if not _bias_cache_built:
        _build_bias_cache()
    return _bias_cache.get(strategy, 0.0)
