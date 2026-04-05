"""
Portfolio positions — open trades with entry price, size, direction.
Persisted to portfolio.json in the project root.
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import date
from typing import Optional

_FILE = os.path.join(os.path.dirname(__file__), "..", "portfolio.json")


@dataclass
class Position:
    ticker: str
    entry_price: float
    shares: float
    direction: str            # "LONG" | "SHORT"
    entry_date: str           # ISO date "2026-04-05"
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    notes: str = ""


def load_positions() -> list:
    try:
        with open(_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return [Position(**p) for p in data]
    except Exception:
        return []


def add_position(
    ticker: str,
    entry_price: float,
    shares: float,
    direction: str,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    notes: str = "",
) -> None:
    positions = load_positions()
    # Replace if same ticker already in portfolio
    positions = [p for p in positions if p.ticker != ticker.upper()]
    positions.append(Position(
        ticker=ticker.upper(),
        entry_price=round(entry_price, 4),
        shares=round(shares, 4),
        direction=direction,
        entry_date=str(date.today()),
        stop_loss=round(stop_loss, 4) if stop_loss else None,
        take_profit=round(take_profit, 4) if take_profit else None,
        notes=notes.strip(),
    ))
    _save(positions)


def remove_position(ticker: str) -> None:
    positions = [p for p in load_positions() if p.ticker != ticker.upper()]
    _save(positions)


def _save(positions: list) -> None:
    with open(_FILE, "w", encoding="utf-8") as f:
        json.dump([asdict(p) for p in positions], f, indent=2)
