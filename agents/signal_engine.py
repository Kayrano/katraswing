"""
5m Signal Engine — combines chart strategies, pattern detection, and news sentiment
into a single SignalResult for the dashboard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from agents.intraday_strategies import (
    IntradaySignal, _STRATEGIES_5M, _flat, recent_liq_sweep,
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

# ADX regime thresholds (used only for the human-readable label below).
# Penalty math now scales smoothly via _trend_weight() — see below.
_ADX_TRENDING = 25.0
_ADX_RANGING  = 20.0

# Soft regime parameters: logistic centered at the midpoint of the discrete
# bands (22.5). Slope of 2.5 means ±5 ADX points covers most of the curve.
#   ADX 15 → trend_weight ≈ 0.05 (firmly ranging)
#   ADX 20 → trend_weight ≈ 0.27
#   ADX 22.5 → trend_weight = 0.50 (transitional)
#   ADX 25 → trend_weight ≈ 0.73
#   ADX 30 → trend_weight ≈ 0.95 (firmly trending)
_ADX_CENTER = 22.5
_ADX_SLOPE  = 2.5

# Maximum penalties applied at the regime extremes — same as the prior
# hard-cliff logic so the in-regime numbers stay calibrated.
_MR_MAX_PENALTY    = 0.12
_TREND_MAX_PENALTY = 0.10


def _trend_weight(adx_val: float) -> float:
    """Smooth indicator in [0, 1] of how trending the market is.

    Replaces the old hard cliffs at ADX=25/20 with a logistic curve centered
    at 22.5. ADX ≤ 0 (unknown) returns 0.5 (neutral) so any signal scoring
    that branches on regime stays balanced.
    """
    if adx_val <= 0:
        return 0.5
    return 1.0 / (1.0 + math.exp(-(adx_val - _ADX_CENTER) / _ADX_SLOPE))

# Session-aware confidence nudge: forex reacts to London/NY volume; metals
# trade around-the-clock and get no adjustment.
_SESSION_FOREX = {
    "EURUSD=X", "GBPUSD=X", "AUDUSD=X", "USDJPY=X",
    "USDCAD=X", "GBPJPY=X", "EURJPY=X", "NZDUSD=X", "USDCHF=X",
}
_SESSION_INDICES = {"YM=F", "ES=F", "NQ=F"}


def _session_boost(ticker: str, hour_utc: int) -> float:
    """Small confidence nudge based on current UTC trading session.

    London 07-12 UTC: +0.03 for forex (peak volume, tight spreads).
    NY     13-18 UTC: +0.02 for equity indices.
    Asian  01-06 UTC: -0.02 for forex only (thin FX liquidity).
    Metals/unrecognised: 0.0 (24-hour market, no session bias).
    """
    if ticker in _SESSION_FOREX:
        if 7 <= hour_utc < 12:
            return  0.03
        if 1 <= hour_utc < 6:
            return -0.02
    if ticker in _SESSION_INDICES:
        if 13 <= hour_utc < 18:
            return  0.02
    return 0.0


# Mean-reversion strategies (penalised in trending markets)
# PDH_PDL_SWEEP fades institutional sweeps — breaks down when trend is strongly directional
# CAMARILLA_5M trades S3/R3 bounces — same fade-the-edge profile, prone to fail when ADX is high
# DOUBLE_BOT_BREAKOUT and HS_BREAKDOWN are reversal patterns — they fight the prevailing
# trend and are exactly what an MR-penalty-in-trending-markets is meant to filter
_MR_STRATEGIES = {
    "VWAP_RSI_5M", "PDH_PDL_SWEEP_5M", "BB_SCALP_5M", "STOCH_CROSS_5M", "CAMARILLA_5M",
    "DOUBLE_BOT_BREAKOUT_5M", "HS_BREAKDOWN_5M",
}
# Trend-following / breakout strategies (penalised in ranging markets)
# NR7 breakouts fail in choppy low-ADX conditions
# FLAG_BREAKOUT_5M is a continuation pattern — needs an existing trend to project the pole
_TREND_STRATEGIES = {"ORB_5M", "TREND_MOM_5M", "EMA_PB_15M", "SQUEEZE_15M",
                     "NR7_BREAKOUT_5M", "MSS_FOREX_15M", "EMA_MICRO_CROSS_5M",
                     "FLAG_BREAKOUT_5M"}
# ABSORB and LIQ_SWEEP are order-flow / manipulation based — regime-independent, never penalised

# Minimum final confidence to issue a signal.
# Raised 0.60 → 0.70 after analysing 74 closed trades: the [0.60, 0.70) bucket
# delivered 21% WR / -$39 P&L while [0.70, 0.80) delivered 46% / +$8.
# Lowered 0.70 → 0.65 for VPS trial: those 74 trades used the old boost stack
# (pre-Round-4 news boost, no +0.20 cap, no regime routing). The composition
# of signals in [0.65, 0.70) today is meaningfully different. Monitor: if
# 20+ new trades in this range show WR < 0.40, revert to 0.70.
_SIGNAL_FLOOR = 0.65

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
    mt5_symbol: str = ""          # MT5 broker symbol name passed through for order routing
    # Accuracy improvements
    consensus_boost: float = 0.0
    strategy_agreement: str = ""   # e.g. "3/3 LONG" or "2/3 LONG, 1 SHORT"
    raw_confidence: float = 0.0    # blended confidence BEFORE isotonic calibration
    calibration_applied: bool = False
    adx_regime: str = "NEUTRAL"    # TRENDING / RANGING / NEUTRAL
    adx_value: float = 0.0
    bt_adjustment: float = 0.0     # backtest-informed calibration delta
    live_adjustment: float = 0.0   # adjustment from actual closed trade outcomes
    live_wr_key: str = ""          # which granularity key was matched (e.g. TREND_MOM:EURUSD:LONG)
    daily_trend_direction: str = "NEUTRAL"  # BULLISH / BEARISH / NEUTRAL
    h4_trend_direction: str = "NEUTRAL"     # BULLISH / BEARISH / NEUTRAL
    mtf_score: int = 0              # -3 … +3  (daily ×2 + H4 ×1)
    mtf_bias: str = "NEUTRAL"       # STRONG_BULLISH / BULLISH / MILD_BULLISH / NEUTRAL / …
    daily_trend_vetoed: bool = False
    sl_tp_source: str = "ATR"       # ATR | BLENDED — whether learned stops were applied
    risk_level: str = "MEDIUM"      # LOW | MEDIUM | HIGH — scales lot size in send_from_signal_result
    # paper_only signals are emitted (calibration tracks them) but the order-
    # send paths in app.py / mt5_signal_server.py skip the MT5 round-trip.
    # Stamped from the picked strategy's IntradaySignal.paper_only flag.
    paper_only: bool = False
    paper_reason: str = ""           # "strategy" | "symbol" | "" — origin of the paper flag
    session_boost: float = 0.0       # UTC-session nudge applied to this signal


# ── Public entry point ────────────────────────────────────────────────────────

def run_signal(
    ticker: str,
    finnhub_api_key: str = "",
    account_size: float = 100_000.0,
    risk_pct: float = 1.0,
    display_name: str = "",
    daily_trend: dict | None = None,
    backtest_win_rates: dict[str, float] | None = None,
    live_win_rates: dict[str, float] | None = None,
    h4_trend: dict | None = None,
    optimal_stops: dict[str, dict] | None = None,
    mt5_symbol: str | None = None,
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
        from data.symbol_policy import get_disposition as _sym_disposition

        # Symbol-level kill list: DROP exits early before any data fetch /
        # strategy compute work. PAPER continues normally and is stamped on
        # the result so order-send paths can skip cleanly.
        disposition = _sym_disposition(mt5_symbol or ticker)
        if disposition == "DROP":
            return SignalResult(ticker=ticker, display_name=label,
                                error=f"Symbol dropped by policy ({ticker})")

        # ── Round 4 B2: event-window veto ───────────────────────────────────
        # Block fresh entries 15m before and 15m after a HIGH-impact release
        # affecting this ticker's currencies. Replaces the ±0.10 sentiment
        # boost that was killed in A4 — research shows event vetoes pay
        # better than sentiment boosts at this sample size.
        try:
            from data.economic_calendar import is_event_window
            in_window, reason = is_event_window(mt5_symbol or ticker)
            if in_window:
                return SignalResult(ticker=ticker, display_name=label,
                                    error=reason)
        except Exception as _ec:
            import logging as _logging
            _logging.getLogger(__name__).debug("event_window check skipped: %s", _ec)

        df = fetch_intraday_data(ticker, interval="5m", days=59, mt5_symbol=mt5_symbol)
        if df is None or df.empty:
            return SignalResult(ticker=ticker, display_name=label,
                                error="No 5m data available for this ticker.")

        # Gram-gold price conversion
        if ticker.upper() in _GRAM_GOLD_TICKERS:
            for col in ("Open", "High", "Low", "Close"):
                if col in df.columns:
                    df[col] = df[col] / _TROY_OZ_TO_GRAM

        # --- Run strategies ---
        # Pre-compute the cleaned symbol so per-(strategy, symbol) adaptive
        # params kick in once a pair has accumulated enough trades.
        sym_for_params = ticker.replace("=X", "").replace("=F", "").upper()
        from data.strategy_params import apply_params as _apply_adaptive

        # MTF pre-filter: if BOTH daily AND H4 are fully aligned against a
        # direction, convert those strategy outputs to FLAT before the regime
        # routing step.  The authoritative MTF gate at the end of the pipeline
        # is unchanged — this is purely a CPU/noise optimisation.
        _pre_daily = (daily_trend or {}).get("trend_direction", "NEUTRAL")
        _pre_h4    = (h4_trend    or {}).get("trend_direction", "NEUTRAL")
        _mtf_pre   = (
            (2 if _pre_daily == "BULLISH" else -2 if _pre_daily == "BEARISH" else 0)
            + (1 if _pre_h4 == "BULLISH" else -1 if _pre_h4 == "BEARISH" else 0)
        )
        # Only pre-veto when BOTH daily AND H4 fully oppose (score ±3).
        # Score -2 (daily BEARISH alone) is ambiguous — H4 could still be
        # BULLISH, and we want to show the signal confidence in the log so
        # marginal cases remain visible. The end-of-pipeline MTF gate handles -2.
        _pre_veto_long  = _mtf_pre <= -3
        _pre_veto_short = _mtf_pre >= 3

        all_signals: list[IntradaySignal] = []
        _pre_vetoed_any = False
        for fn in _STRATEGIES_5M:
            try:
                sig = fn(df)
                sig = _apply_adaptive(sig, symbol=sym_for_params)   # apply learned SL/TP/conf adjustments
            except Exception as exc:
                sig = _flat(fn.__name__.upper(), "5m", str(exc))
            # MTF pre-veto: convert to FLAT if direction is already blocked
            if sig.signal == "LONG" and _pre_veto_long:
                sig = _flat(sig.strategy, sig.timeframe, "MTF pre-veto: LONG blocked by daily+H4 BEARISH")
                _pre_vetoed_any = True
            elif sig.signal == "SHORT" and _pre_veto_short:
                sig = _flat(sig.strategy, sig.timeframe, "MTF pre-veto: SHORT blocked by daily+H4 BULLISH")
                _pre_vetoed_any = True
            all_signals.append(sig)

        # Absorption confluence boost (direction-agnostic order-flow confirmation)
        # Round 4 (2026-05-02): cut +0.10 → +0.05 as part of the boost-stack
        # cap. The 0.80–0.90 confidence bucket loses more money than 0.70–0.80
        # in the live data; absorption was the single biggest stack push.
        try:
            abs_s = ta.absorption(df["High"], df["Low"], df["Close"], df["Volume"])
            if len(abs_s) >= 3 and bool(abs_s.iloc[-3:].any()):
                for sig in all_signals:
                    if sig.signal in ("LONG", "SHORT"):
                        sig.confidence = min(1.0, sig.confidence + 0.05)
                        sig.reason += " [+absorption]"
        except Exception:
            pass

        # Liquidity sweep confluence boost (direction-specific)
        # When institutions swept stops within the last 5 bars, any strategy
        # that agrees with the reversal direction gets a +0.06 boost — the sweep
        # is strong evidence that the smart-money move has already started.
        # Kept below the +0.08 consensus cap so it doesn't single-handedly
        # push marginal signals over the floor.
        try:
            sweep_result = recent_liq_sweep(df, lookback=5)
            if sweep_result is not None:
                sweep_dir, sweep_wick = sweep_result
                for sig in all_signals:
                    if sig.signal == sweep_dir:
                        sig.confidence = min(1.0, sig.confidence + 0.06)
                        sig.reason += f" [+sweep:{sweep_dir} {sweep_wick:.1f}xATR]"
        except Exception:
            pass

        # ── Improvement 3: ADX regime routing ────────────────────────────────
        adx_val = 0.0
        adx_regime = "NEUTRAL"
        composite_score_val: float | None = None
        if len(df) > 50:
            try:
                adx_series = ta.adx(df["High"], df["Low"], df["Close"], length=14)
                valid = adx_series.dropna()
                if not valid.empty:
                    adx_val = float(valid.iloc[-1])
            except Exception:
                adx_val = 0.0

        # Round 5: composite regime (Hurst + Choppiness + ADX-pct) enriches the
        # discrete label and provides a soft [-1,+1] score for the penalty blend.
        try:
            from agents.regime_classifier import classify as _classify_regime
            _regime = _classify_regime(df, ticker=sym_for_params)
            adx_regime = _regime.label          # overrides ADX-only label
            composite_score_val = _regime.composite_score
        except Exception:
            # Discrete regime label kept for UI/logging only — penalty math below
            # uses a continuous trend_weight derived from the same ADX value.
            if adx_val > _ADX_TRENDING:
                adx_regime = "TRENDING"
            elif 0 < adx_val < _ADX_RANGING:
                adx_regime = "RANGING"

        if adx_val > 0:
            tw = _trend_weight(adx_val)
            # Blend composite_score into tw: decisive scores (|score|>0.3) push
            # tw by ±0.10 so the borderline ADX 20-25 zone is correctly routed.
            if composite_score_val is not None:
                if composite_score_val > 0.3:
                    tw = min(1.0, tw + 0.10)
                elif composite_score_val < -0.3:
                    tw = max(0.0, tw - 0.10)
            for sig in all_signals:
                if sig.signal not in ("LONG", "SHORT"):
                    continue
                if sig.strategy in _MR_STRATEGIES:
                    # Mean-reversion: penalty grows with how trending we are.
                    penalty = _MR_MAX_PENALTY * tw
                    if penalty > 0.005:
                        sig.confidence = max(0.0, sig.confidence - penalty)
                        sig.reason += f" [ADX={adx_val:.0f} tw={tw:.2f}→MR-{penalty:.02f}]"
                elif sig.strategy in _TREND_STRATEGIES:
                    # Trend-following: penalty grows with how ranging we are.
                    penalty = _TREND_MAX_PENALTY * (1.0 - tw)
                    if penalty > 0.005:
                        sig.confidence = max(0.0, sig.confidence - penalty)
                        sig.reason += f" [ADX={adx_val:.0f} tw={tw:.2f}→trend-{penalty:.02f}]"
                # ABSORB and other order-flow strategies pass through unchanged

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
            # Scale cap: 3+ strategies in agreement earn up to ±0.08 vs the
            # default ±0.05 for 2-strategy agreement. Distinguishes genuine
            # multi-strategy confluence from a 2/2 coin-flip.
            _consensus_cap = 0.08 if dominant >= 3 else 0.05
            if best_dir == "LONG":
                strategy_agreement = f"{long_count}/{total} LONG"
                if long_count >= short_count:
                    consensus_boost = min(_consensus_cap, (consensus_ratio - 0.5) * 0.16)
                else:
                    consensus_boost = -0.08  # consensus opposes best signal
            elif best_dir == "SHORT":
                strategy_agreement = f"{short_count}/{total} SHORT"
                if short_count >= long_count:
                    consensus_boost = min(_consensus_cap, (consensus_ratio - 0.5) * 0.16)
                else:
                    consensus_boost = -0.08
        elif len(active) == 1:
            strategy_agreement = f"1/1 {active[0].signal}"

        # --- Pattern detection ---
        patterns = _safe_detect_patterns(df)
        # Win rates are applied after direction is known (below) so direction-
        # specific stats can be used. Kept here only for the early-return path.

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
                daily_trend_vetoed=_pre_vetoed_any,
            )

        best = active[0]
        base_conf = best.confidence
        direction = best.signal
        sym_clean = ticker.replace("=X", "").upper()

        # Apply direction-specific pattern win rates now that direction is known.
        try:
            from models.pattern_stats import apply_to_report as _apply_pattern_wrs
            _apply_pattern_wrs(patterns, direction=direction)
        except Exception as _pe:
            import logging as _logging
            _logging.getLogger(__name__).debug("pattern_stats apply skipped: %s", _pe)

        # ── SL/TP calibration from past trade outcomes ────────────────────────
        # Blends ATR-based stops (from strategy) with the median SL/TP % learned
        # from real closed trades.  Weight grows with sample size (max 70%).
        # Most specific key wins: STRATEGY:SYMBOL → STRATEGY.
        sl_tp_source = "ATR"
        if optimal_stops:
            for stop_key in [f"{best.strategy}:{sym_clean}", best.strategy]:
                if stop_key in optimal_stops:
                    stats = optimal_stops[stop_key]
                    if stats["sample"] >= 3 and best.entry > 0:
                        learned_sl_dist = best.entry * stats["sl_pct"] / 100
                        learned_tp_dist = best.entry * stats["tp_pct"] / 100
                        atr_sl_dist = abs(best.entry - best.stop_loss)
                        atr_tp_dist = abs(best.take_profit - best.entry)
                        weight = min(0.70, stats["sample"] / 30 * 0.70)
                        new_sl_dist = (1 - weight) * atr_sl_dist + weight * learned_sl_dist
                        new_tp_dist = (1 - weight) * atr_tp_dist + weight * learned_tp_dist
                        if best.signal == "LONG":
                            best.stop_loss   = round(best.entry - new_sl_dist, 5)
                            best.take_profit = round(best.entry + new_tp_dist, 5)
                        else:
                            best.stop_loss   = round(best.entry + new_sl_dist, 5)
                            best.take_profit = round(best.entry - new_tp_dist, 5)
                        sl_tp_source = "BLENDED"
                    break

        # ── Backtest-informed calibration (Round 4: ±0.10 → ±0.05) ───────────
        bt_adjustment = 0.0
        if backtest_win_rates and best.strategy in backtest_win_rates:
            recent_wr = backtest_win_rates[best.strategy]
            raw_adj = (recent_wr - _BACKTEST_BASELINE_WR) * 0.5
            bt_adjustment = max(-0.05, min(0.05, raw_adj))

        # ── Live trade outcome calibration (Round 4: ±0.15 → ±0.08) ─────────
        # 86 closed trades is too few to earn ±0.15. Halve it until we have
        # 200+ samples per (strategy, symbol, direction) bucket.
        live_adjustment = 0.0
        live_wr_key = ""
        if live_win_rates:
            for key in [
                f"{best.strategy}:{sym_clean}:{direction}",
                f"{best.strategy}:{direction}",
                f"{best.strategy}:{sym_clean}",
                best.strategy,
            ]:
                if key in live_win_rates:
                    live_wr = live_win_rates[key]
                    raw_adj = (live_wr - _BACKTEST_BASELINE_WR) * 0.8
                    live_adjustment = max(-0.08, min(0.08, raw_adj))
                    live_wr_key = key
                    break

        # ── News boost (Round 4: removed; routed to event-window veto in B2)
        # The Finnhub sentiment ±0.10 boost was likely noise at our sample
        # size. We keep the field on SignalResult for backward compatibility
        # and for upcoming B2 work to populate the event veto.
        news_boost = 0.0

        # ── Pattern boost ±0.05 (kept; smallest, posterior-driven) ───────────
        pattern_boost = 0.0
        if patterns.dominant_bias != "NEUTRAL":
            p_aligns = (
                (direction == "LONG"  and patterns.dominant_bias == "BULLISH") or
                (direction == "SHORT" and patterns.dominant_bias == "BEARISH")
            )
            pattern_boost = 0.05 if p_aligns else -0.05

        # ── Round 4: hard total-boost cap of +0.20 ───────────────────────────
        # Theoretical max stack pre-Round-4 was +0.59. Live data shows the
        # 0.80–0.90 confidence bucket loses MORE money than 0.70–0.80 — i.e.
        # the stack pushes marginal trades into a worse-performing bucket.
        # Cap *positive* contribution; negative penalties pass through unclamped
        # so a misaligned stack can still demote a signal below the floor.
        import datetime as _dt
        _hour_utc = _dt.datetime.now(_dt.timezone.utc).hour
        session_boost_val = _session_boost(ticker, _hour_utc)

        positive_boosts = sum(
            max(b, 0.0)
            for b in (consensus_boost, bt_adjustment, live_adjustment,
                      news_boost, pattern_boost, session_boost_val)
        )
        negative_boosts = sum(
            min(b, 0.0)
            for b in (consensus_boost, bt_adjustment, live_adjustment,
                      news_boost, pattern_boost, session_boost_val)
        )
        capped_positive = min(positive_boosts, 0.20)
        final_conf = max(0.0, min(1.0,
            base_conf + capped_positive + negative_boosts
        ))

        # ── Multi-timeframe trend gate (Daily × 2 + H4 × 1) ──────────────────
        # Hard veto: BOTH daily AND H4 must oppose the signal for a block.
        # If only one timeframe opposes, use a confidence adjustment instead.
        daily_trend_direction = "NEUTRAL"
        h4_trend_direction    = "NEUTRAL"
        daily_trend_vetoed    = False
        mtf_score = 0

        if daily_trend:
            daily_trend_direction = daily_trend.get("trend_direction", "NEUTRAL")
            if daily_trend_direction == "BULLISH":
                mtf_score += 2
            elif daily_trend_direction == "BEARISH":
                mtf_score -= 2

        if h4_trend:
            h4_trend_direction = h4_trend.get("trend_direction", "NEUTRAL")
            if h4_trend_direction == "BULLISH":
                mtf_score += 1
            elif h4_trend_direction == "BEARISH":
                mtf_score -= 1

        if mtf_score >= 2:
            mtf_bias = "STRONG_BULLISH" if mtf_score >= 3 else "BULLISH"
        elif mtf_score == 1:
            mtf_bias = "MILD_BULLISH"
        elif mtf_score == -1:
            mtf_bias = "MILD_BEARISH"
        elif mtf_score <= -2:
            mtf_bias = "STRONG_BEARISH" if mtf_score <= -3 else "BEARISH"
        else:
            mtf_bias = "NEUTRAL"

        # Both timeframes oppose → hard veto.
        # Round 4: dropped the soft graded ±0.09 adjustment that previously
        # ran in the else branch — it double-counted with consensus_boost
        # (consensus already encodes intra-strategy direction agreement) and
        # contributed to the 0.80–0.90 over-confidence anomaly.
        if mtf_score <= -2 and direction == "LONG":
            direction = "NO TRADE"
            daily_trend_vetoed = True
        elif mtf_score >= 2 and direction == "SHORT":
            direction = "NO TRADE"
            daily_trend_vetoed = True

        # ── Round 4 B3: suppress boost contributions on vetoed signals ──────
        # When the MTF gate has vetoed the trade, the confidence number is
        # never used to gate or size — but it still ends up in trade_log via
        # downstream consumers (UI display, calibration). Reset to base so
        # vetoed rows don't poison live-WR calibration with phantom-boosted
        # confidences for trades that never actually fired.
        if daily_trend_vetoed:
            final_conf        = base_conf
            consensus_boost   = 0.0
            bt_adjustment     = 0.0
            live_adjustment   = 0.0
            news_boost        = 0.0
            pattern_boost     = 0.0
            session_boost_val = 0.0

        # ── Isotonic confidence calibration ──────────────────────────────────
        # Maps the blended raw confidence onto an empirically-grounded
        # win probability fitted from closed trades. Identity below 50
        # samples (see models.calibration).
        #
        # IMPORTANT: the calibrated number is informational — it reflects the
        # *empirical* win rate of past trades at this raw-confidence level.
        # The 0.60 floor below gates on the *raw* blended confidence (the
        # statistical edge), not the calibrated value, because the system's
        # historical WR (~35–45% on closed trades) is far below 60% and a
        # naive calibrated-floor blocks every signal. We preserve calibration
        # as a UI hint via SignalResult.raw_confidence vs .confidence.
        raw_confidence = final_conf
        calibrated_conf = final_conf
        calibration_applied = False
        try:
            from models.calibration import get_calibrator
            _cal = get_calibrator()
            if _cal.is_fitted:
                calibrated_conf = max(0.0, min(1.0, _cal.transform(final_conf)))
                calibration_applied = True
        except Exception as _ce:
            # Never let calibration break a live signal
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "calibration skipped: %s", _ce,
            )

        # ── Enforce 0.60 confidence floor (on RAW, not calibrated) ───────────
        if not daily_trend_vetoed and raw_confidence < _SIGNAL_FLOOR:
            direction = "NO TRADE"

        # ── ML win-probability gate ───────────────────────────────────────────
        # When the ML predictor has been trained (≥40 closed trades), suppress
        # signals whose predicted win probability falls below the overall
        # historical win rate (~0.38). The gate is skipped when the model is
        # not yet fitted (early data collection phase).
        if direction not in ("NO TRADE", "FLAT") and best is not None:
            try:
                from models.ml_predictor import get_predictor
                _pred = get_predictor()
                if _pred.is_fitted:
                    _ml_prob = _pred.predict_proba(
                        strategy=best.strategy,
                        direction=direction,
                        entry=best.entry,
                        sl=best.stop_loss,
                        tp=best.take_profit,
                        confidence=raw_confidence,
                        adx_value=adx_val,
                        atr_value=best.atr,
                        h1_trend=daily_trend_direction,
                    )
                    if _ml_prob is not None and _ml_prob < 0.33:  # was 0.38; model immature (<200 samples)
                        direction = "NO TRADE"
            except Exception as _ml_exc:
                import logging as _logging
                _logging.getLogger(__name__).debug(
                    "ml_gate skipped: %s", _ml_exc,
                )

        # final_conf stays = raw_confidence here. The calibrated value is
        # surfaced via the .calibration_applied flag and is reserved for
        # future UI display ("expected WR") — gating + risk-sizing should
        # continue to use the raw blended score so behaviour matches the
        # months of trades the rest of the system was calibrated on.
        final_conf = raw_confidence

        # ── Derive risk level from confidence ─────────────────────────────────
        # HIGH confidence = LOW risk = larger position; LOW confidence = HIGH risk = smaller position
        if final_conf >= 0.80:
            risk_level = "LOW"
        elif final_conf >= 0.65:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return SignalResult(
            ticker=ticker, display_name=label,
            direction=direction,
            confidence=round(final_conf, 3),
            entry=best.entry, sl=best.stop_loss, tp=best.take_profit, atr=best.atr,
            chart_signals=active, patterns=patterns,
            news_items=news_items,
            news_sentiment=news_sentiment, news_score=news_score,
            news_boost=round(news_boost, 3),
            base_confidence=round(base_conf, 3),
            indicators=indicators, df_5m=df,
            consensus_boost=round(consensus_boost, 3),
            strategy_agreement=strategy_agreement,
            adx_regime=adx_regime,
            adx_value=round(adx_val, 1),
            bt_adjustment=round(bt_adjustment, 3),
            live_adjustment=round(live_adjustment, 3),
            live_wr_key=live_wr_key,
            daily_trend_direction=daily_trend_direction,
            h4_trend_direction=h4_trend_direction,
            mtf_score=mtf_score,
            mtf_bias=mtf_bias,
            daily_trend_vetoed=daily_trend_vetoed,
            sl_tp_source=sl_tp_source,
            mt5_symbol=mt5_symbol or "",
            risk_level=risk_level,
            raw_confidence=round(raw_confidence, 3),
            calibration_applied=calibration_applied,
            paper_only=(
                getattr(best, "paper_only", False) or disposition == "PAPER"
            ),
            paper_reason=(
                "symbol" if disposition == "PAPER"
                else ("strategy" if getattr(best, "paper_only", False) else "")
            ),
            session_boost=round(session_boost_val, 3),
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
