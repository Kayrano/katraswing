"""
Score-condition alerts: fire when a stock's Katraswing score crosses a threshold.
Triggered automatically each time the stock is analyzed in the Analyzer tab.
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

_FILE = os.path.join(os.path.dirname(__file__), "score_alerts.json")


@dataclass
class ScoreAlert:
    ticker: str
    threshold: float
    condition: str            # "above" | "below"
    note: str
    created_at: str
    triggered: bool = False
    triggered_at: Optional[str] = None
    triggered_score: Optional[float] = None


def load_score_alerts() -> list:
    try:
        with open(_FILE, encoding="utf-8") as f:
            return [ScoreAlert(**a) for a in json.load(f)]
    except Exception:
        return []


def save_score_alerts(alerts: list) -> None:
    with open(_FILE, "w", encoding="utf-8") as f:
        json.dump([asdict(a) for a in alerts], f, indent=2)


def add_score_alert(ticker: str, threshold: float, condition: str, note: str = "") -> None:
    alerts = load_score_alerts()
    alerts.append(ScoreAlert(
        ticker=ticker.upper(),
        threshold=threshold,
        condition=condition,
        note=note,
        created_at=datetime.now().isoformat(timespec="seconds"),
    ))
    save_score_alerts(alerts)


def remove_score_alert(index: int) -> None:
    alerts = load_score_alerts()
    if 0 <= index < len(alerts):
        alerts.pop(index)
        save_score_alerts(alerts)


def check_score_alerts(ticker: str, score: float) -> list:
    """
    Check if any untriggered score alerts for `ticker` just fired.
    Called after run_analysis() in the Analyzer tab.
    Returns list of triggered alert dicts.
    """
    alerts = load_score_alerts()
    newly_triggered = []
    changed = False

    for alert in alerts:
        if alert.triggered or alert.ticker != ticker.upper():
            continue
        fired = (
            (alert.condition == "above" and score >= alert.threshold) or
            (alert.condition == "below" and score <= alert.threshold)
        )
        if fired:
            alert.triggered = True
            alert.triggered_at = datetime.now().isoformat(timespec="seconds")
            alert.triggered_score = round(score, 1)
            newly_triggered.append({
                "ticker":    alert.ticker,
                "condition": alert.condition,
                "threshold": alert.threshold,
                "score":     round(score, 1),
                "note":      alert.note,
            })
            changed = True

    if changed:
        save_score_alerts(alerts)

    return newly_triggered
