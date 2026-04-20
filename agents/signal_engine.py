"""
5m Signal Engine — combines chart strategies, pattern detection, and news sentiment
into a single SignalResult for the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from agents.intraday_strategies import IntradaySignal, run_intraday_signals
from agents.pattern_detector import PatternReport, detect_patterns
from agents.analyzer import AnalyzerAgent
from data.news_fetcher import NewsItem, fetch_news, aggregate_sentiment
from models.report import IndicatorBundle


@dataclass
class SignalResult:
    ticker: str
    direction: str           # LONG / SHORT / NO TRADE
    confidence: float        # 0.0 – 1.0 final (after news boost)
    entry: float
    sl: float
    tp: float
    atr: float
    chart_signals: list[IntradaySignal] = field(default_factory=list)
    patterns: PatternReport = field(default_factory=PatternReport)
    news_items: list[NewsItem] = field(default_factory=list)
    news_sentiment: str = "NEUTRAL"   # BULLISH / BEARISH / NEUTRAL
    news_score: float = 0.0
    news_boost: float = 0.0
    base_confidence: float = 0.0
    indicators: IndicatorBundle | None = None
    df_5m: pd.DataFrame | None = None
    error: str = ""


def run_signal(
    ticker: str,
    finnhub_api_key: str = "",
    account_size: float = 100_000.0,
    risk_pct: float = 1.0,
) -> SignalResult:
    """
    Full 5m signal pipeline:
      1. Fetch 5m bars and run intraday strategies
      2. Detect chart patterns
      3. Compute technical indicators
      4. Fetch and score news (Finnhub)
      5. Apply news boost/penalty to chart confidence
      6. Return unified SignalResult
    """
    try:
        from data.fetcher_intraday import fetch_intraday_data

        df = fetch_intraday_data(ticker, interval="5m", days=59)
        if df is None or df.empty:
            return SignalResult(ticker=ticker, direction="NO TRADE", confidence=0.0,
                                entry=0.0, sl=0.0, tp=0.0, atr=0.0,
                                error="No 5m data available for this ticker.")

        # --- Chart strategies ---
        active_signals, _ = run_intraday_signals(
            ticker, timeframe="5m",
            account_size=account_size, risk_pct=risk_pct
        )

        # --- Pattern detection (last 100 bars) ---
        patterns = _safe_detect_patterns(df)

        # --- Technical indicators ---
        indicators = _safe_indicators(df)

        # --- News ---
        news_items = fetch_news(ticker, api_key=finnhub_api_key, lookback_hours=6)
        news_sentiment, news_score = aggregate_sentiment(news_items)

        # --- No chart signal: NO TRADE ---
        if not active_signals:
            return SignalResult(
                ticker=ticker,
                direction="NO TRADE",
                confidence=0.0,
                entry=df["Close"].iloc[-1] if not df.empty else 0.0,
                sl=0.0,
                tp=0.0,
                atr=0.0,
                chart_signals=[],
                patterns=patterns,
                news_items=news_items,
                news_sentiment=news_sentiment,
                news_score=news_score,
                news_boost=0.0,
                base_confidence=0.0,
                indicators=indicators,
                df_5m=df,
            )

        # --- Best chart signal ---
        best = active_signals[0]
        base_conf = best.confidence
        direction = best.signal  # LONG or SHORT

        # News boost: +0.10 if aligned, -0.10 if opposing
        news_boost = 0.0
        if news_sentiment != "NEUTRAL":
            aligns = (
                (direction == "LONG" and news_sentiment == "BULLISH") or
                (direction == "SHORT" and news_sentiment == "BEARISH")
            )
            news_boost = 0.10 if aligns else -0.10

        # Pattern boost: ±0.05
        pattern_boost = 0.0
        if patterns.dominant_bias != "NEUTRAL":
            p_aligns = (
                (direction == "LONG" and patterns.dominant_bias == "BULLISH") or
                (direction == "SHORT" and patterns.dominant_bias == "BEARISH")
            )
            pattern_boost = 0.05 if p_aligns else -0.05

        final_conf = max(0.0, min(1.0, base_conf + news_boost + pattern_boost))

        # Suppress signal if confidence drops too low after penalties
        if final_conf < 0.35:
            direction = "NO TRADE"

        return SignalResult(
            ticker=ticker,
            direction=direction,
            confidence=round(final_conf, 3),
            entry=best.entry,
            sl=best.stop_loss,
            tp=best.take_profit,
            atr=best.atr,
            chart_signals=active_signals,
            patterns=patterns,
            news_items=news_items,
            news_sentiment=news_sentiment,
            news_score=news_score,
            news_boost=round(news_boost + pattern_boost, 3),
            base_confidence=round(base_conf, 3),
            indicators=indicators,
            df_5m=df,
        )

    except Exception as exc:
        return SignalResult(
            ticker=ticker, direction="NO TRADE", confidence=0.0,
            entry=0.0, sl=0.0, tp=0.0, atr=0.0,
            error=str(exc),
        )


def _safe_detect_patterns(df: pd.DataFrame) -> PatternReport:
    try:
        return detect_patterns(df.tail(100).reset_index(drop=True))
    except Exception:
        return PatternReport()


def _safe_indicators(df: pd.DataFrame) -> IndicatorBundle | None:
    try:
        return AnalyzerAgent().analyze(df)
    except Exception:
        return None
