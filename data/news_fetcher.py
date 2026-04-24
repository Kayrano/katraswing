from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import time
import requests

logger = logging.getLogger(__name__)

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


_RSS_FEEDS = {
    "general": [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    ],
    "forex": [
        "https://www.forexlive.com/feed/news",
    ],
}

# yfinance ticker map — reuse same logic as fetcher_intraday
def _mt5_to_yf(ticker: str) -> str:
    try:
        from data.fetcher_intraday import _MT5_TO_YF
        return _MT5_TO_YF.get(ticker.upper(), ticker)
    except Exception:
        return ticker


def fetch_news_yfinance(ticker: str) -> list[NewsItem]:
    """Fetch news from yfinance as a free second source."""
    cache_key = f"yf:{ticker}"
    cached = _NEWS_CACHE.get(cache_key)
    if cached is not None and time.time() - cached[1] < _NEWS_TTL:
        return cached[0]

    items: list[NewsItem] = []
    try:
        import yfinance as yf
        yf_sym = _mt5_to_yf(ticker)
        news   = yf.Ticker(yf_sym).news or []
        now_ts = time.time()
        for article in news:
            headline  = article.get("title", "")
            pub_ts    = article.get("providerPublishTime", 0)
            if now_ts - pub_ts > 24 * 3600:   # skip articles older than 24h
                continue
            sentiment, score = _classify_sentiment(headline, None)
            items.append(NewsItem(
                headline=headline,
                summary="",
                sentiment=sentiment,
                sentiment_score=score,
                impact=_classify_impact(headline),
                published_at=datetime.fromtimestamp(pub_ts, tz=timezone.utc),
                url=article.get("link", ""),
                source="yfinance",
            ))
    except Exception as exc:
        logger.debug(f"fetch_news_yfinance({ticker}): {exc}")

    _NEWS_CACHE[cache_key] = (items, time.time())
    return items


def fetch_news_rss(ticker: str) -> list[NewsItem]:
    """Fetch news from RSS feeds (requires feedparser). Silently skipped if not installed."""
    try:
        import feedparser  # type: ignore[import]
    except ImportError:
        return []

    is_forex = any(c in ticker.upper() for c in ["USD", "EUR", "GBP", "JPY", "CHF", "AUD"])
    feeds    = _RSS_FEEDS["general"] + (_RSS_FEEDS["forex"] if is_forex else [])
    items: list[NewsItem] = []
    now_ts = time.time()

    for url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in (feed.entries or [])[:20]:
                headline = entry.get("title", "")
                summary  = entry.get("summary", "")[:200]
                full     = headline + " " + summary
                if not _is_relevant(full, ticker):
                    continue
                pub = entry.get("published_parsed")
                if pub:
                    import calendar
                    pub_ts = float(calendar.timegm(pub))
                    if now_ts - pub_ts > 12 * 3600:
                        continue
                    pub_dt = datetime.fromtimestamp(pub_ts, tz=timezone.utc)
                else:
                    pub_dt = datetime.now(tz=timezone.utc)
                sentiment, score = _classify_sentiment(full, None)
                items.append(NewsItem(
                    headline=headline,
                    summary=summary,
                    sentiment=sentiment,
                    sentiment_score=score,
                    impact=_classify_impact(full),
                    published_at=pub_dt,
                    url=entry.get("link", ""),
                    source=url.split("/")[2],
                ))
        except Exception as exc:
            logger.debug(f"RSS feed {url}: {exc}")

    return items


def fetch_news(ticker: str, api_key: str, lookback_hours: int = 6) -> list[NewsItem]:
    """Fetch news from Finnhub + yfinance (always) + RSS (if feedparser installed). Cached 10 min."""
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

    # Merge yfinance + RSS sources
    yf_items  = fetch_news_yfinance(ticker)
    rss_items = fetch_news_rss(ticker) if api_key else []

    # Deduplicate by URL, then by headline prefix (first 40 chars)
    seen_urls: set[str] = {i.url for i in items if i.url}
    seen_heads: set[str] = {i.headline[:40].lower() for i in items}
    for extra in yf_items + rss_items:
        if extra.url and extra.url in seen_urls:
            continue
        head_key = extra.headline[:40].lower()
        if head_key in seen_heads:
            continue
        items.append(extra)
        seen_urls.add(extra.url)
        seen_heads.add(head_key)

    items.sort(key=lambda x: x.published_at, reverse=True)
    result = items[:15]   # up from 10
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


def aggregate_sentiment_with_calendar(
    news_items: list[NewsItem],
    calendar_events: list,   # list[CalendarEvent] — avoid circular import
) -> tuple[str, float]:
    """
    Enhanced sentiment aggregation that factors in calendar events:
    - Upcoming HIGH impact event → uncertainty penalty (score pulled toward 0)
    - Just-released HIGH event with actual > forecast → bullish for that currency
    - Just-released HIGH event with actual < forecast → bearish for that currency
    """
    base_label, base_score = aggregate_sentiment(news_items)
    if not calendar_events:
        return base_label, base_score

    score = base_score
    for ev in calendar_events:
        if ev.impact != "HIGH":
            continue
        if ev.is_upcoming:
            # Pull score toward neutral (uncertainty)
            score = score * 0.5
        elif ev.is_recent and ev.actual is not None and ev.forecast is not None:
            try:
                actual   = float(str(ev.actual).replace("%", "").replace("K", "000").replace("M", "000000"))
                forecast = float(str(ev.forecast).replace("%", "").replace("K", "000").replace("M", "000000"))
                if actual > forecast:
                    score = min(1.0, score + 0.15)
                elif actual < forecast:
                    score = max(-1.0, score - 0.15)
            except (ValueError, TypeError):
                pass

    label = "BULLISH" if score > 0.05 else "BEARISH" if score < -0.05 else "NEUTRAL"
    return label, round(score, 3)
