"""
Boost-stack attribution analyzer.

The 6-component confidence boost stack (consensus + bt_adj + live_adj +
news + pattern + session) was hand-tuned. When the system shows a
confidence inversion (winners avg conf < losers avg conf), at least one
component is anti-predictive — actively rewarding setups that lose more
often than they win.

This module computes the point-biserial correlation of each boost
component with WIN outcomes across closed trades. The output identifies
which components are contributing real signal and which should be
dropped, capped tighter, or sign-flipped.

Reported per nightly retrain via Telegram. No automatic mutation — the
user reviews and manually adjusts the boost weights in signal_engine.py.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

_DATA_DIR       = Path(__file__).parent.parent / "data"
_TRADE_LOG_PATH = _DATA_DIR / "trade_log.json"

# Min trades required before correlation is meaningful. Below this we
# report "insufficient data" rather than a noisy number.
MIN_TRADES = 20

# Boost components to attribute. Keys match the field names written by
# record_trade(). Each must be a numeric per-signal value, not a category.
_COMPONENTS = [
    "base_confidence",
    "consensus_boost",
    "bt_adjustment",
    "live_adjustment",
    "news_boost",
    "pattern_boost_val",
    "session_boost",
]


def _point_biserial(x: list[float], y: list[bool]) -> float:
    """Point-biserial correlation between a continuous variable x and a
    binary variable y. Pure-python — no scipy dependency at runtime.

    Equivalent to Pearson r when one variable is 0/1 encoded. Returns 0.0
    on degenerate input (all same class or all same x).
    """
    n = len(x)
    if n < 3 or n != len(y):
        return 0.0
    n1 = sum(1 for v in y if v)
    n0 = n - n1
    if n0 == 0 or n1 == 0:
        return 0.0   # all wins or all losses → no discrimination possible
    mean_all = sum(x) / n
    var_all  = sum((xi - mean_all) ** 2 for xi in x) / n
    if var_all <= 0:
        return 0.0   # x is constant
    std_all = var_all ** 0.5
    mean_1 = sum(xi for xi, yi in zip(x, y) if yi) / n1
    mean_0 = sum(xi for xi, yi in zip(x, y) if not yi) / n0
    return (mean_1 - mean_0) / std_all * ((n1 * n0) ** 0.5 / n)


def compute_correlations(
    trades: Iterable[dict] | None = None,
    min_trades: int = MIN_TRADES,
    only_live: bool = False,
) -> dict:
    """Return per-component WR-correlation summary.

    Result schema:
        {
          "n":            <int>,                  # total closed eligible trades
          "wr":           <float>,                # overall WR on those trades
          "components": {
              "consensus_boost":  {"corr": +0.18, "n": 87,  "mean_win": 0.05, "mean_loss": 0.02},
              "bt_adjustment":    {"corr": -0.04, "n": 87,  "mean_win": 0.01, "mean_loss": 0.02},
              ...
          },
          "verdict": [
              "consensus_boost: KEEP (corr=+0.18)",
              "bt_adjustment:   REVIEW (corr=-0.04) - sign may be flipped",
              ...
          ],
        }

    When trades is None, loads from data/trade_log.json.
    When only_live=True, excludes paper_only trades.
    """
    if trades is None:
        try:
            trades = json.loads(_TRADE_LOG_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("boost_attribution: load failed: %s", exc)
            return {"n": 0, "components": {}, "verdict": []}

    closed = [
        t for t in trades
        if t.get("outcome") in ("WIN", "LOSS")
        and t.get("strategy") != "MT5_IMPORT"
        and (not only_live or not t.get("paper_only"))
    ]
    if len(closed) < min_trades:
        return {
            "n":          len(closed),
            "wr":         (sum(1 for t in closed if t["outcome"] == "WIN") / len(closed)) if closed else 0.0,
            "components": {},
            "verdict":    [f"insufficient data: n={len(closed)} < {min_trades}"],
        }

    overall_wr = sum(1 for t in closed if t["outcome"] == "WIN") / len(closed)

    component_stats: dict[str, dict] = {}
    for comp in _COMPONENTS:
        x: list[float] = []
        y: list[bool]  = []
        for t in closed:
            v = t.get(comp)
            if v is None:
                continue   # row predates the boost-attribution recording
            try:
                x.append(float(v))
            except (TypeError, ValueError):
                continue
            y.append(t["outcome"] == "WIN")
        if len(x) < min_trades:
            component_stats[comp] = {
                "corr":      None,
                "n":         len(x),
                "note":      "insufficient data (most trades pre-date attribution recording)",
            }
            continue
        wins_x   = [xi for xi, yi in zip(x, y) if yi]
        losses_x = [xi for xi, yi in zip(x, y) if not yi]
        component_stats[comp] = {
            "corr":      round(_point_biserial(x, y), 4),
            "n":         len(x),
            "mean_win":  round(sum(wins_x) / len(wins_x), 4) if wins_x else 0.0,
            "mean_loss": round(sum(losses_x) / len(losses_x), 4) if losses_x else 0.0,
        }

    # ── Verdict ───────────────────────────────────────────────────────────
    # Heuristic thresholds — these tell the user how to act. Hand-tuned to
    # avoid noise on small samples while being decisive when signal is clear.
    verdict: list[str] = []
    for comp, stat in component_stats.items():
        corr = stat.get("corr")
        if corr is None:
            verdict.append(f"{comp:18s} PENDING (n={stat['n']})")
            continue
        if corr >= 0.15:
            tag = "STRONG +"
        elif corr >= 0.05:
            tag = "KEEP +"
        elif corr > -0.05:
            tag = "WEAK ±"
        elif corr > -0.15:
            tag = "REVIEW (flip?)"
        else:
            tag = "DROP / FLIP"
        verdict.append(f"{comp:18s} {tag} corr={corr:+.3f} n={stat['n']}")

    return {
        "n":          len(closed),
        "wr":         round(overall_wr, 4),
        "components": component_stats,
        "verdict":    verdict,
    }


def format_telegram(report: dict) -> str:
    """Render a compact Telegram message from compute_correlations() output."""
    lines = [f"<b>Boost attribution</b> (n={report['n']}, WR={report.get('wr', 0):.1%})"]
    if report["n"] < MIN_TRADES:
        lines.append(f"  (need {MIN_TRADES}+ trades — currently {report['n']})")
        return "\n".join(lines)
    for v in report["verdict"]:
        lines.append(f"  {v}")
    return "\n".join(lines)


if __name__ == "__main__":
    # CLI: python -m models.boost_attribution
    # Prints the verdict for the current trade_log without sending Telegram.
    import sys
    only_live = "--live-only" in sys.argv
    report = compute_correlations(only_live=only_live)
    print(f"n={report['n']}  WR={report.get('wr', 0):.1%}"
          f"{'  (live only)' if only_live else ''}")
    for line in report["verdict"]:
        print(" ", line)
    if report["components"]:
        print()
        print("Per-component detail:")
        for comp, stat in report["components"].items():
            if stat.get("corr") is None:
                continue
            print(f"  {comp:18s} mean_win={stat['mean_win']:+.4f}  "
                  f"mean_loss={stat['mean_loss']:+.4f}  n={stat['n']}")
