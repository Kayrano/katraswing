"""
Price Alerts persistence — stores alerts in a local JSON file.
Each alert fires when a stock crosses a target price threshold.
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional
import yfinance as yf

_FILE = os.path.join(os.path.dirname(__file__), "alerts.json")


@dataclass
class PriceAlert:
    ticker: str
    target_price: float
    condition: str          # "above" | "below"
    note: str
    created_at: str         # ISO datetime string
    triggered: bool = False
    triggered_at: Optional[str] = None
    triggered_price: Optional[float] = None


def load_alerts() -> list[PriceAlert]:
    if not os.path.exists(_FILE):
        return []
    try:
        with open(_FILE, "r") as f:
            data = json.load(f)
        return [PriceAlert(**item) for item in data]
    except Exception:
        return []


def save_alerts(alerts: list[PriceAlert]) -> None:
    with open(_FILE, "w") as f:
        json.dump([asdict(a) for a in alerts], f, indent=2)


def add_alert(ticker: str, target_price: float, condition: str, note: str = "") -> list[PriceAlert]:
    alerts = load_alerts()
    alerts.append(PriceAlert(
        ticker=ticker.upper().strip(),
        target_price=target_price,
        condition=condition,
        note=note,
        created_at=datetime.now().isoformat(timespec="seconds"),
    ))
    save_alerts(alerts)
    return alerts


def remove_alert(index: int) -> list[PriceAlert]:
    alerts = load_alerts()
    if 0 <= index < len(alerts):
        alerts.pop(index)
        save_alerts(alerts)
    return alerts


def check_alerts() -> list[dict]:
    """
    Fetch current price for each untriggered alert and mark those that fired.
    Returns list of dicts for alerts that just triggered this check.
    """
    alerts = load_alerts()
    if not alerts:
        return []

    # Batch fetch — group by ticker to minimize API calls
    tickers_needed = list({a.ticker for a in alerts if not a.triggered})
    if not tickers_needed:
        return []

    prices: dict[str, float] = {}
    for t in tickers_needed:
        try:
            info = yf.Ticker(t).fast_info
            price = getattr(info, "last_price", None)
            if price:
                prices[t] = float(price)
        except Exception:
            pass

    newly_triggered = []
    changed = False

    for alert in alerts:
        if alert.triggered or alert.ticker not in prices:
            continue
        current = prices[alert.ticker]
        fired = (
            (alert.condition == "above" and current >= alert.target_price) or
            (alert.condition == "below" and current <= alert.target_price)
        )
        if fired:
            alert.triggered = True
            alert.triggered_at = datetime.now().isoformat(timespec="seconds")
            alert.triggered_price = round(current, 4)
            newly_triggered.append({
                "ticker": alert.ticker,
                "condition": alert.condition,
                "target": alert.target_price,
                "current": current,
                "note": alert.note,
            })
            changed = True

    if changed:
        save_alerts(alerts)

    return newly_triggered
