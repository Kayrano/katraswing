"""
Economic calendar filter — blocks signals around high-impact news events.

Uses Finnhub's /calendar/economic endpoint. Results are cached for 4 hours
to avoid hammering the API. Returns True (blocked) if a high-impact event
for the ticker's currency is scheduled within a 30-minute window.

Usage:
    from utils.economic_calendar import is_news_window
    blocked, reason = is_news_window("EURUSD=X", hour_utc=13, minute_utc=28, api_key="...")
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

log = logging.getLogger(__name__)

# ── Currency exposure per ticker ─────────────────────────────────────────────
# Each entry lists the ISO currency codes whose high-impact events
# should trigger a block for that ticker.
_TICKER_CURRENCIES: dict[str, list[str]] = {
    "EURUSD=X": ["EUR", "USD"],
    "GBPUSD=X": ["GBP", "USD"],
    "AUDUSD=X": ["AUD", "USD"],
    "USDJPY=X": ["USD", "JPY"],
    "USDCAD=X": ["USD", "CAD"],
    "GBPJPY=X": ["GBP", "JPY"],
    "EURJPY=X": ["EUR", "JPY"],
    "GC=F":     ["USD"],   # Gold priced in USD
    "SI=F":     ["USD"],
    "YM=F":     ["USD"],
    "ES=F":     ["USD"],
    "NQ=F":     ["USD"],
    "CL=F":     ["USD"],
    "BZ=F":     ["USD"],
}

# Finnhub country code → ISO currency (common major economies)
_COUNTRY_CURRENCY: dict[str, str] = {
    "US": "USD", "GB": "GBP", "EU": "EUR", "DE": "EUR", "FR": "EUR",
    "JP": "JPY", "AU": "AUD", "CA": "CAD", "CH": "CHF", "NZ": "NZD",
    "CN": "CNY", "IN": "INR",
}

# How many minutes before/after the event to block
_BLOCK_WINDOW_MIN = 30

# Cache lifetime in seconds
_CACHE_TTL_SEC = 4 * 3600

_cache: dict = {}  # {"fetched_at": float, "events": list[dict]}


def _fetch_events(api_key: str) -> list[dict]:
    """Fetch today's economic calendar from Finnhub. Returns list of events."""
    try:
        import requests
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        url = (
            f"https://finnhub.io/api/v1/calendar/economic"
            f"?from={today}&to={today}&token={api_key}"
        )
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            log.debug("economic_calendar: Finnhub returned %s", r.status_code)
            return []
        data = r.json()
        return data.get("economicCalendar", [])
    except Exception as exc:
        log.debug("economic_calendar: fetch failed: %s", exc)
        return []


def _get_events(api_key: str) -> list[dict]:
    """Return cached (or freshly fetched) events for today."""
    now = time.time()
    if _cache and (now - _cache.get("fetched_at", 0)) < _CACHE_TTL_SEC:
        return _cache.get("events", [])
    events = _fetch_events(api_key)
    _cache["fetched_at"] = now
    _cache["events"] = events
    return events


def is_news_window(
    ticker: str,
    hour_utc: int,
    minute_utc: int,
    api_key: str = "",
) -> tuple[bool, str]:
    """
    Return (True, reason) if a high-impact event for this ticker's currency
    is within the blocking window, (False, "") otherwise.

    No API key → always returns (False, "").
    """
    if not api_key:
        return False, ""

    currencies = _TICKER_CURRENCIES.get(ticker, [])
    if not currencies:
        return False, ""

    events = _get_events(api_key)
    now_min = hour_utc * 60 + minute_utc

    for ev in events:
        impact = str(ev.get("impact", "")).lower()
        if impact not in ("high", "1"):
            continue

        country = str(ev.get("country", "")).upper()
        ev_currency = _COUNTRY_CURRENCY.get(country, "")
        if ev_currency not in currencies:
            continue

        # Parse event time if provided (format: "HH:MM" or full ISO)
        ev_time_str = str(ev.get("time", ""))
        if not ev_time_str or ev_time_str == ev.get("date", ""):
            # Date-only event — treat as blocking all day for simplicity
            # (only applies to very high-impact events like central bank rate days)
            ev_min = -1
        else:
            try:
                # Could be "HH:MM" or "YYYY-MM-DDTHH:MM:SS"
                if "T" in ev_time_str:
                    t = datetime.fromisoformat(ev_time_str.replace("Z", "+00:00"))
                    ev_min = t.hour * 60 + t.minute
                elif ":" in ev_time_str:
                    parts = ev_time_str.split(":")
                    ev_min = int(parts[0]) * 60 + int(parts[1])
                else:
                    ev_min = -1
            except Exception:
                ev_min = -1

        if ev_min == -1:
            # No specific time — skip (too many false positives from date-only entries)
            continue

        if abs(now_min - ev_min) <= _BLOCK_WINDOW_MIN:
            event_name = ev.get("event", "economic event")
            return (
                True,
                f"{ticker} blocked: {event_name} ({country}, impact={impact}) "
                f"at {ev_min // 60:02d}:{ev_min % 60:02d} UTC "
                f"(within {_BLOCK_WINDOW_MIN}min window)",
            )

    return False, ""
