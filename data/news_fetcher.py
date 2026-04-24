from dataclasses import dataclass
from datetime import datetime, timezone
import time
import requests

# ── In-memory news cache (10-min TTL) ────────────────────────────────────────
# Prevents hammering Finnhub on every scan cycle.
_NEWS_CACHE: dict[str, tuple[list, float]] = {}
_NEWS_TTL = 600  # seconds

# Keywords that indicate macro/NQ-relevant news for futures
_MACRO_KEYWORDS = [
    "fed", "federal reserve", "fomc", "cpi", "inflation", "nfp", "jobs",
    "gdp", "rate", "interest", "nasdaq", "qqq", "tech", "earnings",
    "recession", "growth", "powell", "treasury", "yield", "unemployment",
    "pce", "ppi", "retail sales", "ism", "manufacturing", "services",
]

_BEARISH_KEYWORDS = [
    "miss", "disappoint", "decline", "drop", "fall", "loss", "cut", "layoff",
    "recession", "slowdown", "weak", "below", "concern", "risk", "fear",
    "sell", "downgrade", "bearish", "crash", "plunge", "slump", "crisis",
]

_BULLISH_KEYWORDS = [
    "beat", "exceed", "surpass", "rise", "gain", "record", "strong", "above",
    "growth", "positive", "upgrade", "bullish", "rally", "surge", "jump",
    "profit", "revenue", "hire", "expand", "accelerate", "outperform",
]

_HIGH_IMPACT_KEYWORDS = [
    "fed", "fomc", "cpi", "nfp", "gdp", "rate decision", "powell",
    "earnings", "pce", "jobs report", "unemployment rate",
]


@dataclass
class NewsItem:
    headline: str
    summary: str
    sentiment: str       # BULLISH / BEARISH / NEUTRAL
    sentiment_score: float  # -1.0 to +1.0
    impact: str          # HIGH / MED / LOW
    published_at: datetime
    url: str
    source: str


def _classify_sentiment(text: str, finnhub_score: float | None) -> tuple[str, float]:
    if finnhub_score is not None and finnhub_score != 0:
        score = max(-1.0, min(1.0, finnhub_score))
        label = "BULLISH" if score > 0.1 else "BEARISH" if score < -0.1 else "NEUTRAL"
        return label, score

    lower = text.lower()
    bull = sum(1 for k in _BULLISH_KEYWORDS if k in lower)
    bear = sum(1 for k in _BEARISH_KEYWORDS if k in lower)
    total = bull + bear
    if total == 0:
        return "NEUTRAL", 0.0
    score = (bull - bear) / total
    label = "BULLISH" if score > 0.1 else "BEARISH" if score < -0.1 else "NEUTRAL"
    return label, round(score, 2)


def _classify_impact(text: str) -> str:
    lower = text.lower()
    if any(k in lower for k in _HIGH_IMPACT_KEYWORDS):
        return "HIGH"
    if any(k in lower for k in ["analyst", "upgrade", "downgrade", "target", "acquisition", "merger"]):
        return "MED"
    return "LOW"


def _is_relevant(text: str, ticker: str) -> bool:
    lower = text.lower()
    ticker_lower = ticker.lower().replace("=f", "").replace("-", "")
    if ticker_lower in lower:
        return True
    # For futures like NQ=F, always include macro news
    if ticker.upper() in ("NQ=F", "ES=F", "YM=F", "RTY=F", "MNQ=F", "MES=F"):
        return any(k in lower for k in _MACRO_KEYWORDS)
    return True  # For equities, accept all company news


def fetch_news(ticker: str, api_key: str, lookback_hours: int = 6) -> list[NewsItem]:
    """Fetch news from Finnhub. Cached for 10 minutes per ticker."""
    if not api_key:
        return []

    cache_key = f"{ticker}:{lookback_hours}"
    cached = _NEWS_CACHE.get(cache_key)
    if cached is not None and time.time() - cached[1] < _NEWS_TTL:
        return cached[0]

    headers = {"X-Finnhub-Token": api_key}
    now = int(time.time())
    from_ts = now - lookback_hours * 3600
    items: list[NewsItem] = []

    is_futures = ticker.upper().endswith("=F") or ticker.upper().startswith("M") and ticker.upper().endswith("=F")

    if is_futures:
        # General market news for macro/index futures
        try:
            resp = requests.get(
                "https://finnhub.io/api/v1/news",
                params={"category": "general"},
                headers=headers,
                timeout=10,
            )
            if resp.status_code == 200:
                for article in resp.json():
                    headline = article.get("headline", "")
                    summary = article.get("summary", "")
                    full_text = headline + " " + summary
                    if not _is_relevant(full_text, ticker):
                        continue
                    pub_ts = article.get("datetime", 0)
                    if pub_ts < from_ts:
                        continue
                    sentiment, score = _classify_sentiment(
                        full_text, article.get("sentiment")
                    )
                    items.append(NewsItem(
                        headline=headline,
                        summary=summary[:200] if summary else "",
                        sentiment=sentiment,
                        sentiment_score=score,
                        impact=_classify_impact(full_text),
                        published_at=datetime.fromtimestamp(pub_ts, tz=timezone.utc),
                        url=article.get("url", ""),
                        source=article.get("source", ""),
                    ))
        except Exception:
            pass
    else:
        # Company-specific news
        try:
            from_date = datetime.fromtimestamp(from_ts).strftime("%Y-%m-%d")
            to_date = datetime.now().strftime("%Y-%m-%d")
            resp = requests.get(
                "https://finnhub.io/api/v1/company-news",
                params={"symbol": ticker, "from": from_date, "to": to_date},
                headers=headers,
                timeout=10,
            )
            if resp.status_code == 200:
                for article in resp.json():
                    headline = article.get("headline", "")
                    summary = article.get("summary", "")
                    full_text = headline + " " + summary
                    pub_ts = article.get("datetime", 0)
                    if pub_ts < from_ts:
                        continue
                    sentiment, score = _classify_sentiment(
                        full_text, article.get("sentiment")
                    )
                    items.append(NewsItem(
                        headline=headline,
                        summary=summary[:200] if summary else "",
                        sentiment=sentiment,
                        sentiment_score=score,
                        impact=_classify_impact(full_text),
                        published_at=datetime.fromtimestamp(pub_ts, tz=timezone.utc),
                        url=article.get("url", ""),
                        source=article.get("source", ""),
                    ))
        except Exception:
            pass

    items.sort(key=lambda x: x.published_at, reverse=True)
    result = items[:30]
    _NEWS_CACHE[cache_key] = (result, time.time())
    return result


def aggregate_sentiment(news_items: list[NewsItem]) -> tuple[str, float]:
    """Weighted aggregate sentiment. HIGH impact news counts 3x, MED 2x, LOW 1x."""
    if not news_items:
        return "NEUTRAL", 0.0

    weight_map = {"HIGH": 3.0, "MED": 2.0, "LOW": 1.0}
    total_weight = 0.0
    weighted_score = 0.0
    for item in news_items[:10]:  # Use most recent 10
        w = weight_map.get(item.impact, 1.0)
        weighted_score += item.sentiment_score * w
        total_weight += w

    if total_weight == 0:
        return "NEUTRAL", 0.0

    avg = weighted_score / total_weight
    label = "BULLISH" if avg > 0.05 else "BEARISH" if avg < -0.05 else "NEUTRAL"
    return label, round(avg, 3)
