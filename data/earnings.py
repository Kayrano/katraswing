"""
Earnings & News Risk Filter
Fetches upcoming earnings date and recent news headlines for a ticker.
Warns if earnings are within N days — swing trades should be avoided around earnings.
"""

from datetime import datetime, timedelta
from typing import Optional
import yfinance as yf


EARNINGS_WARN_DAYS = 7   # warn if earnings within this many days


def get_earnings_risk(ticker: str) -> dict:
    """
    Returns a dict:
      earnings_date      : datetime | None
      days_until         : int | None
      is_risky           : bool  (True if earnings within EARNINGS_WARN_DAYS)
      risk_label         : str
      risk_color         : str  (CSS hex)
      next_earnings_str  : str  (human-readable)
    """
    result = {
        "earnings_date":    None,
        "days_until":       None,
        "is_risky":         False,
        "risk_label":       "No earnings data",
        "risk_color":       "#888888",
        "next_earnings_str": "N/A",
    }

    try:
        t = yf.Ticker(ticker)
        cal = t.calendar

        earnings_date: Optional[datetime] = None

        # calendar can be a dict or a DataFrame depending on yfinance version
        if cal is not None:
            if hasattr(cal, "columns"):
                # DataFrame format
                if "Earnings Date" in cal.columns:
                    val = cal["Earnings Date"].iloc[0] if len(cal) > 0 else None
                    if val is not None:
                        earnings_date = pd.Timestamp(val).to_pydatetime()
            elif isinstance(cal, dict):
                for key in ("Earnings Date", "earningsDate", "Earnings_Date"):
                    if key in cal:
                        val = cal[key]
                        if isinstance(val, (list, tuple)) and val:
                            val = val[0]
                        if val is not None:
                            try:
                                import pandas as pd
                                earnings_date = pd.Timestamp(val).to_pydatetime()
                            except Exception:
                                pass
                        break

        if earnings_date is None:
            # Fallback: try earnings_dates property
            try:
                import pandas as pd
                ed = t.earnings_dates
                if ed is not None and len(ed) > 0:
                    today = datetime.now().replace(tzinfo=None)
                    future = [
                        d.replace(tzinfo=None)
                        for d in ed.index
                        if d.replace(tzinfo=None) >= today
                    ]
                    if future:
                        earnings_date = min(future)
            except Exception:
                pass

        if earnings_date is None:
            return result

        earnings_date = earnings_date.replace(tzinfo=None)
        today = datetime.now().replace(tzinfo=None)
        days_until = (earnings_date - today).days

        result["earnings_date"]    = earnings_date
        result["days_until"]       = days_until
        result["next_earnings_str"] = earnings_date.strftime("%b %d, %Y")

        if days_until < 0:
            result["risk_label"] = f"Earnings passed ({abs(days_until)}d ago)"
            result["risk_color"] = "#888888"
            result["is_risky"]   = False
        elif days_until <= 2:
            result["risk_label"] = f"⛔ EARNINGS IN {days_until}d — HIGH RISK"
            result["risk_color"] = "#ff4444"
            result["is_risky"]   = True
        elif days_until <= EARNINGS_WARN_DAYS:
            result["risk_label"] = f"⚠ Earnings in {days_until} days"
            result["risk_color"] = "#ffbb33"
            result["is_risky"]   = True
        else:
            result["risk_label"] = f"✓ Earnings in {days_until} days"
            result["risk_color"] = "#00c851"
            result["is_risky"]   = False

    except Exception:
        pass

    return result


def get_news(ticker: str, max_items: int = 5) -> list[dict]:
    """
    Returns recent news headlines for the ticker.
    Each item: { title, publisher, link, published }
    """
    items = []
    try:
        t = yf.Ticker(ticker)
        news = t.news or []
        for n in news[:max_items]:
            # yfinance news item structure varies by version
            title     = n.get("title") or n.get("content", {}).get("title", "")
            publisher = n.get("publisher") or n.get("content", {}).get("provider", {}).get("displayName", "")
            link      = n.get("link") or n.get("content", {}).get("canonicalUrl", {}).get("url", "#")
            pub_ts    = n.get("providerPublishTime") or n.get("content", {}).get("pubDate", "")
            if pub_ts and isinstance(pub_ts, (int, float)):
                from datetime import timezone
                pub_dt = datetime.fromtimestamp(pub_ts, tz=timezone.utc)
                published = pub_dt.strftime("%b %d")
            else:
                published = str(pub_ts)[:10] if pub_ts else ""

            if title:
                items.append({
                    "title":     title,
                    "publisher": publisher,
                    "link":      link,
                    "published": published,
                })
    except Exception:
        pass
    return items
