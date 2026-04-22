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

# ADX regime thresholds
_ADX_TRENDING = 25.0
_ADX_RANGING  = 20.0

# Mean-reversion strategies (penalised in trending markets)
_MR_STRATEGIES = {"VWAP_RSI_5M"}
# Trend-following strategies (penalised in ranging markets)
_TREND_STRATEGIES = {"ORB_5M", "TREND_MOM_5M", "EMA_PB_15M", "SQUEEZE_15M"}
# ABSORB is order-flow based — regime-independent, never penalised

# Minimum final confidence to issue a signal (raised from 0.35 → 0.60)
_SIGNAL_FLOOR = 0.60

# Baseline win rate used for backtest calibration adjustment
_BACKTEST_BASELINE_WR = 0.62


@dataclass
class SignalResult:
    ticker: str
    display_name: str = ""
    direction: str = "NO TRADE"   # LONG / SHORT / NO TRADE
    confidence: float = 0.0       # 0.0 – 1.0 final (after all boosts)
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
    # Accuracy improvements
    consensus_boost: float = 0.0
    strategy_agreement: str = ""   # e.g. "3/3 LONG" or "2/3 LONG, 1 SHORT"
    adx_regime: str = "NEUTRAL"    # TRENDING / RANGING / NEUTRAL
    adx_value: float = 0.0
    bt_adjustment: float = 0.0     # backtest-informed calibration delta
    daily_trend_direction: str = "NEUTRAL"  # BULLISH / BEARISH / NEUTRAL
    daily_trend_vetoed: bool = False


# ── Public entry point ────────────────────────────────────────────────────────

def run_signal(
    ticker: str,
    finnhub_api_key: str = "",
    account_size: float = 100_000.0,
    risk_pct: float = 1.0,
    display_name: str = "",
    daily_trend: dict | None = None,
    backtest_win_rates: dict[str, float] | None = None,
) -> SignalResult:
    """
    Full 5m signal pipeline:
      1. Fetch 5m bars (apply gram conversion for gold tickers)
      2. Run intraday strategies
      3. Apply ADX regime penalties (MR vs trend-following routing)
      4. Rank signals; compute strategy consensus
      5. Detect chart patterns
      6. Compute technical indicators
      7. Fetch and score Finnhub news
      8. Blend: base + consensus + backtest calibration + news + pattern boosts
      9. Apply daily trend gate (hard veto if opposing higher-timeframe trend)
     10. Enforce 0.60 confidence floor
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

        # --- Run strategies ---
        all_signals: list[IntradaySignal] = []
        for fn in _STRATEGIES_5M:
            try:
                all_signals.append(fn(df))
            except Exception as exc:
                all_signals.append(_flat(fn.__name__.upper(), "5m", str(exc)))

        # Absorption confluence boost (direction-agnostic order-flow confirmation)
        try:
            abs_s = ta.absorption(df["High"], df["Low"], df["Close"], df["Volume"])
            if len(abs_s) >= 3 and bool(abs_s.iloc[-3:].any()):
                for sig in all_signals:
                    if sig.signal in ("LONG", "SHORT"):
                        sig.confidence = min(1.0, sig.confidence + 0.10)
                        sig.reason += " [+absorption]"
        except Exception:
            pass

        # ── Improvement 3: ADX regime routing ────────────────────────────────
        adx_val = 0.0
        adx_regime = "NEUTRAL"
        if len(df) > 50:
            try:
                adx_series = ta.adx(df["High"], df["Low"], df["Close"], length=14)
                valid = adx_series.dropna()
                if not valid.empty:
                    adx_val = float(valid.iloc[-1])
            except Exception:
                adx_val = 0.0

        if adx_val > _ADX_TRENDING:
            adx_regime = "TRENDING"
        elif 0 < adx_val < _ADX_RANGING:
            adx_regime = "RANGING"

        if adx_val > 0:
            for sig in all_signals:
                if sig.signal not in ("LONG", "SHORT"):
                    continue
                if adx_regime == "TRENDING" and sig.strategy in _MR_STRATEGIES:
                    sig.confidence = max(0.0, sig.confidence - 0.12)
                    sig.reason += f" [ADX={adx_val:.0f} trending→MR↓]"
                elif adx_regime == "RANGING" and sig.strategy in _TREND_STRATEGIES:
                    sig.confidence = max(0.0, sig.confidence - 0.10)
                    sig.reason += f" [ADX={adx_val:.0f} ranging→trend↓]"

        # Rank after regime adjustments
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

        # ── Improvement 2B: Strategy consensus scoring ───────────────────────
        consensus_boost = 0.0
        strategy_agreement = ""
        if len(active) >= 2:
            long_count  = sum(1 for s in active if s.signal == "LONG")
            short_count = sum(1 for s in active if s.signal == "SHORT")
            total       = len(active)
            dominant    = max(long_count, short_count)
            consensus_ratio = dominant / total  # 0.5 split → 1.0 unanimous

            # Determine what the best signal's direction is vs consensus
            best_dir = active[0].signal if active else "FLAT"
            if best_dir == "LONG":
                strategy_agreement = f"{long_count}/{total} LONG"
                if long_count >= short_count:
                    consensus_boost = (consensus_ratio - 0.5) * 0.10  # max +0.05
                else:
                    consensus_boost = -0.08  # consensus opposes best signal
            elif best_dir == "SHORT":
                strategy_agreement = f"{short_count}/{total} SHORT"
                if short_count >= long_count:
                    consensus_boost = (consensus_ratio - 0.5) * 0.10
                else:
                    consensus_boost = -0.08
        elif len(active) == 1:
            strategy_agreement = f"1/1 {active[0].signal}"

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
                adx_regime=adx_regime, adx_value=round(adx_val, 1),
                daily_trend_direction=daily_trend.get("trend_direction", "NEUTRAL") if daily_trend else "NEUTRAL",
            )

        best = active[0]
        base_conf = best.confidence
        direction = best.signal

        # ── Improvement 4: Backtest-informed calibration ─────────────────────
        bt_adjustment = 0.0
        if backtest_win_rates and best.strategy in backtest_win_rates:
            recent_wr = backtest_win_rates[best.strategy]
            raw_adj = (recent_wr - _BACKTEST_BASELINE_WR) * 0.5
            bt_adjustment = max(-0.10, min(0.10, raw_adj))

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

        final_conf = max(0.0, min(1.0,
            base_conf + consensus_boost + bt_adjustment + news_boost + pattern_boost
        ))

        # ── Improvement 1: Daily trend gate (hard veto) ──────────────────────
        daily_trend_direction = "NEUTRAL"
        daily_trend_vetoed = False
        if daily_trend:
            daily_trend_direction = daily_trend.get("trend_direction", "NEUTRAL")
            if daily_trend_direction == "BULLISH" and direction == "SHORT":
                direction = "NO TRADE"
                daily_trend_vetoed = True
            elif daily_trend_direction == "BEARISH" and direction == "LONG":
                direction = "NO TRADE"
                daily_trend_vetoed = True
            elif daily_trend_direction != "NEUTRAL" and not daily_trend_vetoed:
                # Trend alignment bonus
                final_conf = min(1.0, final_conf + 0.05)

        # ── Improvement 2A: Enforce 0.60 floor ───────────────────────────────
        if not daily_trend_vetoed and final_conf < _SIGNAL_FLOOR:
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
            consensus_boost=round(consensus_boost, 3),
            strategy_agreement=strategy_agreement,
            adx_regime=adx_regime,
            adx_value=round(adx_val, 1),
            bt_adjustment=round(bt_adjustment, 3),
            daily_trend_direction=daily_trend_direction,
            daily_trend_vetoed=daily_trend_vetoed,
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
