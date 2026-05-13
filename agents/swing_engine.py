"""
H1 Swing Signal Engine — parallel rail alongside the 5m signal_engine.
=======================================================================
Mirrors the signal_engine.run_signal API but operates on H1 bars and uses
the H1 strategy roster from agents.swing_strategies.

All H1 strategy IDs end in _H1 so:
  - live-WR calibration and strategy_params track them independently
  - trade_log entries are separate from 5m trades
  - the walk-forward auto-promotion harness (learning_loop.run_nightly)
    handles H1 paper_only promotion in the same bucket system

Ships with all four H1 strategies in paper_only=True mode until the
walk-forward gate validates them (n>=20, WR>=0.50, PF>=1.30).

Usage:
    from agents.swing_engine import run_h1_signal
    result: SignalResult = run_h1_signal("EURUSD=X", daily_trend=...)

The result is a SignalResult (same type as signal_engine), ready to be
passed to send_from_signal_result / paper-trade shadow log.
"""

from __future__ import annotations

import math

from agents.signal_engine import (
    SignalResult,
    _ADX_CENTER, _ADX_SLOPE,
    _MR_MAX_PENALTY, _TREND_MAX_PENALTY,
    _SIGNAL_FLOOR,
    _BACKTEST_BASELINE_WR,
    _safe_detect_patterns,
    _safe_indicators,
)
from agents.swing_strategies import (
    _STRATEGIES_H1,
    _STRATEGY_NAME_MAP_H1,
    _TREND_STRATEGIES_H1,
    _MR_STRATEGIES_H1,
)
from agents.intraday_strategies import IntradaySignal, _flat
import utils.ta_compat as ta


def _trend_weight_h1(adx_val: float) -> float:
    if adx_val <= 0:
        return 0.5
    return 1.0 / (1.0 + math.exp(-(adx_val - _ADX_CENTER) / _ADX_SLOPE))


def run_h1_signal(
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
    """Full H1 swing signal pipeline — mirrors signal_engine.run_signal.

    Pipeline:
      1. Symbol policy check (DROP/PAPER/LIVE)
      2. Event-window veto (same B2 calendar used by 5m engine)
      3. Fetch H1 bars
      4. Run H1 strategies; apply ADX regime routing
      5. Strategy consensus scoring
      6. Pattern detection + news fetch
      7. Confidence boost blend (same stack as 5m, same caps)
      8. MTF gate (daily × 2 + H4 × 1 hard veto)
      9. Boost suppression on vetoed signals (B3)
     10. Signal floor enforcement
    """
    label = display_name or ticker
    try:
        from data.fetcher_intraday import fetch_intraday_data
        from utils.position_sizing import calculate as calc_position
        from data.symbol_policy import get_disposition as _sym_disposition

        disposition = _sym_disposition(mt5_symbol or ticker)
        if disposition == "DROP":
            return SignalResult(ticker=ticker, display_name=label,
                                error=f"Symbol dropped by policy ({ticker})")

        # ── B2: event-window veto ────────────────────────────────────────
        try:
            from data.economic_calendar import is_event_window
            in_window, reason = is_event_window(mt5_symbol or ticker)
            if in_window:
                return SignalResult(ticker=ticker, display_name=label, error=reason)
        except Exception as _ec:
            import logging as _logging
            _logging.getLogger(__name__).debug("h1 event_window check skipped: %s", _ec)

        df = fetch_intraday_data(ticker, interval="1h", days=59, mt5_symbol=mt5_symbol)
        if df is None or df.empty:
            return SignalResult(ticker=ticker, display_name=label,
                                error="No H1 data available.")

        # ── Run H1 strategies ────────────────────────────────────────────
        sym_for_params = ticker.replace("=X", "").replace("=F", "").upper()
        from data.strategy_params import apply_params as _apply_adaptive

        # MTF pre-filter: skip H1 strategies whose direction is already vetoed
        _pre_daily_h1 = (daily_trend or {}).get("trend_direction", "NEUTRAL")
        _pre_h4_h1    = (h4_trend    or {}).get("trend_direction", "NEUTRAL")
        _mtf_pre_h1   = (
            (2 if _pre_daily_h1 == "BULLISH" else -2 if _pre_daily_h1 == "BEARISH" else 0)
            + (1 if _pre_h4_h1 == "BULLISH" else -1 if _pre_h4_h1 == "BEARISH" else 0)
        )
        _pre_veto_long_h1  = _mtf_pre_h1 <= -3
        _pre_veto_short_h1 = _mtf_pre_h1 >= 3

        all_signals: list[IntradaySignal] = []
        _pre_vetoed_any_h1 = False
        for fn in _STRATEGIES_H1:
            try:
                sig = fn(df)
                sig = _apply_adaptive(sig, symbol=sym_for_params)
            except Exception as exc:
                sig = _flat(fn.__name__.upper(), "1h", str(exc))
            if sig.signal == "LONG" and _pre_veto_long_h1:
                sig = _flat(sig.strategy, sig.timeframe, "MTF pre-veto: LONG blocked by daily+H4 BEARISH")
                _pre_vetoed_any_h1 = True
            elif sig.signal == "SHORT" and _pre_veto_short_h1:
                sig = _flat(sig.strategy, sig.timeframe, "MTF pre-veto: SHORT blocked by daily+H4 BULLISH")
                _pre_vetoed_any_h1 = True
            all_signals.append(sig)

        # ── ADX regime routing ───────────────────────────────────────────
        adx_val   = 0.0
        adx_regime = "NEUTRAL"
        composite_score_val: float | None = None
        if len(df) > 50:
            try:
                adx_s = ta.adx(df["High"], df["Low"], df["Close"], length=14)
                valid = adx_s.dropna()
                if not valid.empty:
                    adx_val = float(valid.iloc[-1])
            except Exception:
                adx_val = 0.0

        # Round 5: composite regime enriches label + soft penalty blend.
        try:
            from agents.regime_classifier import classify as _classify_regime
            _regime = _classify_regime(df, ticker=sym_for_params)
            adx_regime = _regime.label
            composite_score_val = _regime.composite_score
        except Exception:
            if adx_val > 25.0:
                adx_regime = "TRENDING"
            elif 0 < adx_val < 20.0:
                adx_regime = "RANGING"

        if adx_val > 0:
            tw = _trend_weight_h1(adx_val)
            if composite_score_val is not None:
                if composite_score_val > 0.3:
                    tw = min(1.0, tw + 0.10)
                elif composite_score_val < -0.3:
                    tw = max(0.0, tw - 0.10)
            for sig in all_signals:
                if sig.signal not in ("LONG", "SHORT"):
                    continue
                if sig.strategy in _MR_STRATEGIES_H1:
                    penalty = _MR_MAX_PENALTY * tw
                    if penalty > 0.005:
                        sig.confidence = max(0.0, sig.confidence - penalty)
                        sig.reason += f" [ADX={adx_val:.0f}→MR-{penalty:.02f}]"
                elif sig.strategy in _TREND_STRATEGIES_H1:
                    penalty = _TREND_MAX_PENALTY * (1.0 - tw)
                    if penalty > 0.005:
                        sig.confidence = max(0.0, sig.confidence - penalty)
                        sig.reason += f" [ADX={adx_val:.0f}→trend-{penalty:.02f}]"

        active = sorted(
            [s for s in all_signals if s.signal in ("LONG", "SHORT")],
            key=lambda s: s.confidence, reverse=True,
        )

        for sig in active:
            try:
                calc_position(account_size, risk_pct, sig.entry, sig.stop_loss, sig.take_profit)
            except Exception:
                pass

        # ── Consensus ────────────────────────────────────────────────────
        consensus_boost   = 0.0
        strategy_agreement = ""
        if len(active) >= 2:
            long_c  = sum(1 for s in active if s.signal == "LONG")
            short_c = sum(1 for s in active if s.signal == "SHORT")
            total   = len(active)
            dominant = max(long_c, short_c)
            ratio    = dominant / total
            best_dir = active[0].signal
            _consensus_cap_h1 = 0.08 if dominant >= 3 else 0.05
            if best_dir == "LONG":
                strategy_agreement = f"{long_c}/{total} LONG"
                consensus_boost = (
                    min(_consensus_cap_h1, (ratio - 0.5) * 0.16)
                    if long_c >= short_c else -0.08
                )
            elif best_dir == "SHORT":
                strategy_agreement = f"{short_c}/{total} SHORT"
                consensus_boost = (
                    min(_consensus_cap_h1, (ratio - 0.5) * 0.16)
                    if short_c >= long_c else -0.08
                )
        elif len(active) == 1:
            strategy_agreement = f"1/1 {active[0].signal}"

        # ── Pattern + indicators + news ──────────────────────────────────
        patterns   = _safe_detect_patterns(df)
        try:
            from models.pattern_stats import apply_to_report as _apr
            _apr(patterns)
        except Exception:
            pass
        indicators = _safe_indicators(df)
        from data.news_fetcher import fetch_news, aggregate_sentiment
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
                daily_trend_direction=(
                    daily_trend.get("trend_direction", "NEUTRAL") if daily_trend else "NEUTRAL"
                ),
                daily_trend_vetoed=_pre_vetoed_any_h1,
            )

        best      = active[0]
        base_conf = best.confidence
        direction = best.signal
        sym_clean = ticker.replace("=X", "").upper()

        # ── SL/TP from learned stops ─────────────────────────────────────
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

        # ── Backtest + live adjustments (same caps as 5m) ────────────────
        bt_adjustment = 0.0
        if backtest_win_rates and best.strategy in backtest_win_rates:
            raw_adj = (backtest_win_rates[best.strategy] - _BACKTEST_BASELINE_WR) * 0.5
            bt_adjustment = max(-0.05, min(0.05, raw_adj))

        live_adjustment = 0.0
        live_wr_key     = ""
        if live_win_rates:
            for key in [
                f"{best.strategy}:{sym_clean}:{direction}",
                f"{best.strategy}:{direction}",
                f"{best.strategy}:{sym_clean}",
                best.strategy,
            ]:
                if key in live_win_rates:
                    raw_adj = (live_win_rates[key] - _BACKTEST_BASELINE_WR) * 0.8
                    live_adjustment = max(-0.08, min(0.08, raw_adj))
                    live_wr_key = key
                    break

        news_boost   = 0.0   # removed in Round 4 A4; routed to event veto
        pattern_boost = 0.0
        if patterns.dominant_bias != "NEUTRAL":
            aligns = (
                (direction == "LONG"  and patterns.dominant_bias == "BULLISH") or
                (direction == "SHORT" and patterns.dominant_bias == "BEARISH")
            )
            pattern_boost = 0.05 if aligns else -0.05

        # ── +0.20 hard cap on positive boosts ────────────────────────────
        positive_boosts = sum(max(b, 0.0) for b in (
            consensus_boost, bt_adjustment, live_adjustment, news_boost, pattern_boost))
        negative_boosts = sum(min(b, 0.0) for b in (
            consensus_boost, bt_adjustment, live_adjustment, news_boost, pattern_boost))
        final_conf = max(0.0, min(1.0,
            base_conf + min(positive_boosts, 0.20) + negative_boosts))

        # ── MTF gate ─────────────────────────────────────────────────────
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

        if mtf_score <= -2 and direction == "LONG":
            direction          = "NO TRADE"
            daily_trend_vetoed = True
        elif mtf_score >= 2 and direction == "SHORT":
            direction          = "NO TRADE"
            daily_trend_vetoed = True

        # ── B3: suppress boosts on vetoed signals ────────────────────────
        if daily_trend_vetoed:
            final_conf      = base_conf
            consensus_boost = 0.0
            bt_adjustment   = 0.0
            live_adjustment = 0.0
            news_boost      = 0.0
            pattern_boost   = 0.0

        # ── Isotonic calibration ─────────────────────────────────────────
        raw_confidence  = final_conf
        calibrated_conf = final_conf
        calibration_applied = False
        try:
            from models.calibration import get_calibrator
            _cal = get_calibrator()
            if _cal.is_fitted:
                calibrated_conf = max(0.0, min(1.0, _cal.transform(final_conf)))
                calibration_applied = True
        except Exception:
            pass

        # ── Floor on raw confidence ───────────────────────────────────────
        if not daily_trend_vetoed and raw_confidence < _SIGNAL_FLOOR:
            direction = "NO TRADE"

        final_conf = raw_confidence

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
                "symbol"   if disposition == "PAPER"
                else "strategy" if getattr(best, "paper_only", True) else ""
            ),
        )

    except Exception as exc:
        return SignalResult(ticker=ticker, display_name=label, error=str(exc))
