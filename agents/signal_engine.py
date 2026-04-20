"""
5m Signal Engine — combines chart strategies, pattern detection, and news sentiment
into a single SignalResult for the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from agents.intraday_strategies import (
    IntradaySignal, _STRATEGIES_5M, _flat,
)
from agents.pattern_detector import PatternReport, detect_patterns
from agents.analyzer import AnalyzerAgent
from data.news_fetcher import NewsItem, fetch_news, aggregate_sentiment
from models.report import IndicatorBundle

import utils.ta_compat as ta

# 1 troy ounce = 31.1035 grams — used to convert XAUUSD spot to per-gram price
_TROY_OZ_TO_GRAM = 31.1035

# Tickers that need gram-gold price conversion
_GRAM_GOLD_TICKERS = {"XAUUSD=X", "GC=F"}


@dataclass
class SignalResult:
    ticker: str
    display_name: str = ""
    direction: str = "NO TRADE"   # LONG / SHORT / NO TRADE
    confidence: float = 0.0       # 0.0 – 1.0 final (after news boost)
    entry: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    atr: float = 0.0
    chart_signals: list[IntradaySignal] = field(default_factory=list)
    patterns: PatternReport = field(default_factory=PatternReport)
    news_items: list[NewsItem] = field(default_factory=list)
    news_sentiment: str = "NEUTRAL"
    news_score: float = 0.0
    news_boost: float = 0.0
    base_confidence: float = 0.0
    indicators: IndicatorBundle | None = None
    df_5m: pd.DataFrame | None = None
    error: str = ""


# ── Public entry point ────────────────────────────────────────────────────────

def run_signal(
    ticker: str,
    finnhub_api_key: str = "",
    account_size: float = 100_000.0,
    risk_pct: float = 1.0,
    display_name: str = "",
) -> SignalResult:
    """
    Full 5m signal pipeline:
      1. Fetch 5m bars (apply gram conversion for gold tickers)
      2. Run intraday strategies directly on the fetched df
      3. Detect chart patterns
      4. Compute technical indicators
      5. Fetch and score Finnhub news
      6. Apply news + pattern boost/penalty to chart confidence
    """
    label = display_name or ticker
    try:
        from data.fetcher_intraday import fetch_intraday_data
        from utils.position_sizing import calculate as calc_position

        df = fetch_intraday_data(ticker, interval="5m", days=59)
        if df is None or df.empty:
            return SignalResult(ticker=ticker, display_name=label,
                                error="No 5m data available for this ticker.")

        # Gram-gold price conversion
        if ticker.upper() in _GRAM_GOLD_TICKERS:
            for col in ("Open", "High", "Low", "Close"):
                if col in df.columns:
                    df[col] = df[col] / _TROY_OZ_TO_GRAM

        # --- Run strategies directly on the (possibly converted) df ---
        all_signals: list[IntradaySignal] = []
        for fn in _STRATEGIES_5M:
            try:
                all_signals.append(fn(df))
            except Exception as exc:
                all_signals.append(_flat(fn.__name__.upper(), "5m", str(exc)))

        # Absorption confluence boost
        try:
            abs_s = ta.absorption(df["High"], df["Low"], df["Close"], df["Volume"])
            if len(abs_s) >= 3 and bool(abs_s.iloc[-3:].any()):
                for sig in all_signals:
                    if sig.signal in ("LONG", "SHORT"):
                        sig.confidence = min(1.0, sig.confidence + 0.10)
                        sig.reason += " [+absorption]"
        except Exception:
            pass

        active = sorted(
            [s for s in all_signals if s.signal in ("LONG", "SHORT")],
            key=lambda s: s.confidence, reverse=True,
        )

        # Attach position sizing
        for sig in active:
            try:
                calc_position(account_size, risk_pct, sig.entry, sig.stop_loss, sig.take_profit)
            except Exception:
                pass

        # --- Pattern detection ---
        patterns = _safe_detect_patterns(df)

        # --- Indicators ---
        indicators = _safe_indicators(df)

        # --- News ---
        news_items = fetch_news(ticker, api_key=finnhub_api_key, lookback_hours=6)
        news_sentiment, news_score = aggregate_sentiment(news_items)

        if not active:
            return SignalResult(
                ticker=ticker, display_name=label,
                direction="NO TRADE", confidence=0.0,
                entry=float(df["Close"].iloc[-1]),
                sl=0.0, tp=0.0, atr=0.0,
                chart_signals=[], patterns=patterns,
                news_items=news_items,
                news_sentiment=news_sentiment, news_score=news_score,
                indicators=indicators, df_5m=df,
            )

        best = active[0]
        base_conf = best.confidence
        direction = best.signal

        # News boost ±0.10
        news_boost = 0.0
        if news_sentiment != "NEUTRAL":
            aligns = (
                (direction == "LONG" and news_sentiment == "BULLISH") or
                (direction == "SHORT" and news_sentiment == "BEARISH")
            )
            news_boost = 0.10 if aligns else -0.10

        # Pattern boost ±0.05
        pattern_boost = 0.0
        if patterns.dominant_bias != "NEUTRAL":
            p_aligns = (
                (direction == "LONG" and patterns.dominant_bias == "BULLISH") or
                (direction == "SHORT" and patterns.dominant_bias == "BEARISH")
            )
            pattern_boost = 0.05 if p_aligns else -0.05

        final_conf = max(0.0, min(1.0, base_conf + news_boost + pattern_boost))
        if final_conf < 0.35:
            direction = "NO TRADE"

        return SignalResult(
            ticker=ticker, display_name=label,
            direction=direction,
            confidence=round(final_conf, 3),
            entry=best.entry, sl=best.stop_loss, tp=best.take_profit, atr=best.atr,
            chart_signals=active, patterns=patterns,
            news_items=news_items,
            news_sentiment=news_sentiment, news_score=news_score,
            news_boost=round(news_boost + pattern_boost, 3),
            base_confidence=round(base_conf, 3),
            indicators=indicators, df_5m=df,
        )

    except Exception as exc:
        return SignalResult(ticker=ticker, display_name=label, error=str(exc))


# ── Helpers ───────────────────────────────────────────────────────────────────

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
