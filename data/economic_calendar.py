"""
Economic calendar — fetches upcoming high/medium impact events from
ForexFactory JSON feed (free, no API key).

Primary use: news hold guard in trade_manager.py — block SL/TP modifications
when a market-moving release is imminent.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# 15-minute in-memory cache for the FF feed
_CAL_CACHE: tuple[list, float] | None = None
_CAL_TTL = 900   # seconds

_FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# Map ISO country code → currency symbol (ForexFactory uses country codes)
_COUNTRY_TO_CURRENCY: dict[str, str] = {
    "USD": "USD", "US":  "USD",
    "EUR": "EUR", "EU":  "EUR",
    "GBP": "GBP", "UK":  "GBP", "GB": "GBP",
    "JPY": "JPY", "JP":  "JPY",
    "CHF": "CHF", "CH":  "CHF",
    "AUD": "AUD", "AU":  "AUD",
    "CAD": "CAD", "CA":  "CAD",
    "NZD": "NZD", "NZ":  "NZD",
    "CNY": "CNY", "CN":  "CNY",
    "SEK": "SEK", "NOK": "NOK",
}

# Map ticker prefixes/patterns to the currencies that drive them
_TICKER_CURRENCIES: list[tuple[str, list[str]]] = [
    # Explicit forex pairs (most specific, checked first)
    ("EURUSD", ["EUR", "USD"]),
    ("GBPUSD", ["GBP", "USD"]),
    ("USDJPY", ["USD", "JPY"]),
    ("USDCHF", ["USD", "CHF"]),
    ("AUDUSD", ["AUD", "USD"]),
    ("NZDUSD", ["NZD", "USD"]),
    ("USDCAD", ["USD", "CAD"]),
    ("EURGBP", ["EUR", "GBP"]),
    ("EURJPY", ["EUR", "JPY"]),
    ("GBPJPY", ["GBP", "JPY"]),
    # Indices / futures (USD-denominated)
    ("US100", ["USD"]),
    ("US500", ["USD"]),
    ("NQ",    ["USD"]),
    ("ES",    ["USD"]),
    ("YM",    ["USD"]),
    ("NKD",   ["USD", "JPY"]),
    ("JP225",  ["JPY"]),
    # Gold / silver
    ("XAU",   ["USD"]),
    ("XAG",   ["USD"]),
    ("GC",    ["USD"]),
    ("SI",    ["USD"]),
    ("GOLD",  ["USD"]),
]


@dataclass
class CalendarEvent:
    title: str
    currency: str
    impact: str          # "HIGH" | "MEDIUM" | "LOW"
    event_time: datetime  # UTC-aware
    is_upcoming: bool
    is_recent: bool
    actual: str | None
    forecast: str | None
    previous: str | None


def get_symbol_currencies(ticker: str) -> list[str]:
    """Map any Katraswing/MT5/yfinance ticker to a list of affected currency codes."""
    clean = ticker.upper().replace("=X", "").replace("=F", "").replace("#", "")
    # Strip broker suffixes like _M26
    if "_" in clean:
        clean = clean.split("_")[0]

    for pattern, currencies in _TICKER_CURRENCIES:
        if pattern in clean:
            return currencies

    # 6-char forex pair fallback: XXXYYY → [XXX, YYY]
    if len(clean) == 6 and clean.isalpha():
        return [clean[:3], clean[3:]]

    # Equity CFDs — price in USD
    return ["USD"]


def _fetch_ff_raw() -> list[dict]:
    """Fetch ForexFactory JSON, returning raw list. Cached 15 min."""
    global _CAL_CACHE
    if _CAL_CACHE is not None and time.time() - _CAL_CACHE[1] < _CAL_TTL:
        return _CAL_CACHE[0]

    try:
        resp = requests.get(_FF_URL, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            _CAL_CACHE = (data, time.time())
            return data
    except Exception as exc:
        logger.debug(f"ForexFactory fetch failed: {exc}")

    return _CAL_CACHE[0] if _CAL_CACHE else []


def _parse_event_time(date_str: str, time_str: str) -> Optional[datetime]:
    """
    Parse ForexFactory date/time strings to UTC datetime.
    date_str: "04-25-2026"   time_str: "8:30am" | "All Day" | "Tentative"
    """
    if not date_str:
        return None
    try:
        dt = datetime.strptime(date_str, "%m-%d-%Y")
    except ValueError:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return None

    if time_str and time_str.lower() not in ("all day", "tentative", ""):
        try:
            t = datetime.strptime(time_str.lower().strip(), "%I:%M%p")
            dt = dt.replace(hour=t.hour, minute=t.minute)
        except ValueError:
            try:
                t = datetime.strptime(time_str.lower().strip(), "%I%p")
                dt = dt.replace(hour=t.hour, minute=0)
            except ValueError:
                pass

    # ForexFactory JSON returns times in UTC
    return dt.replace(tzinfo=timezone.utc)


def fetch_upcoming_events(
    ticker: str,
    lookahead_min: int = 60,
    lookback_min: int = 5,
) -> list[CalendarEvent]:
    """
    Return HIGH and MEDIUM impact calendar events within the time window
    [now - lookback_min, now + lookahead_min] for currencies that affect `ticker`.
    """
    currencies = get_symbol_currencies(ticker)
    raw        = _fetch_ff_raw()
    now        = datetime.now(tz=timezone.utc)
    window_start = now - timedelta(minutes=lookback_min)
    window_end   = now + timedelta(minutes=lookahead_min)

    events: list[CalendarEvent] = []
    for item in raw:
        impact_raw = str(item.get("impact", "")).capitalize()
        if impact_raw not in ("High", "Medium"):
            continue

        impact = "HIGH" if impact_raw == "High" else "MEDIUM"

        # Resolve currency from country field
        country = str(item.get("country", "")).upper()
        currency = _COUNTRY_TO_CURRENCY.get(country, country)
        if currency not in currencies:
            continue

        # Parse event time — FF JSON provides "date" and "time" fields
        ev_time = _parse_event_time(
            item.get("date", ""),
            item.get("time", ""),
        )
        if ev_time is None:
            continue

        if not (window_start <= ev_time <= window_end):
            continue

        events.append(CalendarEvent(
            title=str(item.get("title", "")),
            currency=currency,
            impact=impact,
            event_time=ev_time,
            is_upcoming=ev_time > now,
            is_recent=ev_time <= now,
            actual=item.get("actual") or None,
            forecast=item.get("forecast") or None,
            previous=item.get("previous") or None,
        ))

    events.sort(key=lambda e: e.event_time)
    return events


def has_high_impact_event(
    ticker: str,
    within_minutes: int = 60,
) -> tuple[bool, str]:
    """
    Check whether a HIGH-impact event is imminent or just released for `ticker`.

    Returns (blocked, reason_string).
    """
    events = fetch_upcoming_events(ticker, lookahead_min=within_minutes, lookback_min=5)
    high   = [e for e in events if e.impact == "HIGH"]
    if not high:
        return False, ""

    e   = high[0]   # nearest one
    now = datetime.now(tz=timezone.utc)
    if e.is_upcoming:
        delta_min = int((e.event_time - now).total_seconds() / 60)
        reason = f"{e.title} in {delta_min} min ({e.currency}, HIGH impact)"
    else:
        reason = f"{e.title} just released ({e.currency}, HIGH impact)"

    return True, reason
