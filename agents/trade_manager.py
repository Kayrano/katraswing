"""
Trade Manager Agent — post-entry intelligence for open MT5 positions.

Re-runs the full signal pipeline against each open position, computes a
"trade health" score (0–1), and recommends one of:
  HOLD | CLOSE | PARTIAL_CLOSE | MODIFY_SL | MODIFY_TP | MODIFY_BOTH

Dry-run mode is ON by default; no MT5 orders are sent unless dry_run=False.
"""

from __future__ import annotations

import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Configure a StreamHandler so logger output goes to console
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("[TM %(asctime)s] %(levelname)s %(message)s",
                                            datefmt="%H:%M:%S"))
    logger.addHandler(_handler)
    logger.setLevel(logging.DEBUG)


def _tm_log(msg: str) -> None:
    """Print a timestamped trade manager log line to stdout."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[TM {ts}] {msg}", flush=True)

_ASSESSMENT_LOG = Path(__file__).parent.parent / "data" / "assessment_log.json"
_COOLDOWN_MINUTES = 30
_MIN_BARS_MINUTES = 15


# ── Assessment dataclass ──────────────────────────────────────────────────────

@dataclass
class TradeAssessment:
    ticket: int
    symbol: str
    direction: str            # "LONG" | "SHORT"
    open_price: float
    breakeven_price: float
    current_price: float
    current_sl: float
    current_tp: float
    current_profit: float
    original_confidence: float

    action: str               # HOLD|CLOSE|PARTIAL_CLOSE|MODIFY_SL|MODIFY_TP|MODIFY_BOTH
    reason: str
    urgency: str              # LOW | MEDIUM | HIGH
    health_score: float       # 0.0 – 1.0

    new_sl: float | None = None
    new_tp: float | None = None
    partial_close_volume: float | None = None

    # Signal details for UI display
    mtf_score: int = 0
    mtf_bias: str = "NEUTRAL"
    signal_direction: str = "NO TRADE"
    signal_confidence: float = 0.0
    adx_regime: str = "NEUTRAL"
    strategy_agreement: str = ""

    assessed_at: str = ""
    dry_run: bool = True
    acted_on: bool = False
    error: str = ""


# ── Assessment log ────────────────────────────────────────────────────────────

def _append_assessment_log(assessment: TradeAssessment) -> None:
    try:
        records: list[dict] = []
        if _ASSESSMENT_LOG.exists():
            records = json.loads(_ASSESSMENT_LOG.read_text(encoding="utf-8"))
        records.append({
            "ticket":      assessment.ticket,
            "symbol":      assessment.symbol,
            "direction":   assessment.direction,
            "assessed_at": assessment.assessed_at,
            "health_score": round(assessment.health_score, 3),
            "action":      assessment.action,
            "new_sl":      assessment.new_sl,
            "new_tp":      assessment.new_tp,
            "urgency":     assessment.urgency,
            "reason":      assessment.reason,
            "acted_on":    assessment.acted_on,
            "dry_run":     assessment.dry_run,
        })
        _ASSESSMENT_LOG.write_text(
            json.dumps(records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning(f"assessment log write error: {exc}")


# ── Cooldown helpers ──────────────────────────────────────────────────────────

def _is_on_cooldown(ticket: int, cooldown_state: dict) -> bool:
    last = cooldown_state.get(ticket)
    if last is None:
        return False
    return (datetime.now(tz=timezone.utc) - last).total_seconds() < _COOLDOWN_MINUTES * 60


def _set_cooldown(ticket: int, cooldown_state: dict) -> None:
    cooldown_state[ticket] = datetime.now(tz=timezone.utc)


# ── Breakeven price ───────────────────────────────────────────────────────────

def _calc_breakeven(position) -> float:
    """
    Compute the price at which the trade covers its accumulated costs.
    Uses swap + commission from live MT5 position data.
    """
    try:
        import MetaTrader5 as mt5
        sym_info   = mt5.symbol_info(position.symbol)
        tick_value = (sym_info.trade_tick_value or 1.0) if sym_info else 1.0
        tick_size  = (sym_info.trade_tick_size  or 0.00001) if sym_info else 0.00001
        cost_per_lot = (abs(position.commission) + abs(position.swap))
        if position.volume > 0 and tick_size > 0:
            cost_pts = cost_per_lot / (tick_value / tick_size * position.volume)
        else:
            cost_pts = 0.0
        if position.direction == "LONG":
            return position.open_price + cost_pts
        return position.open_price - cost_pts
    except Exception:
        return position.open_price


# ── Health score ──────────────────────────────────────────────────────────────

def _compute_health_score(
    position,
    signal,
    original_confidence: float,
    current_price: float,
    atr: float,
) -> float:
    """
    Weighted health score 0.0–1.0.

    Weights:
      MTF alignment      30%
      Confidence delta   20%
      Momentum           20%
      R:R remaining      15%
      Pattern bias       15%
    """
    direction = position.direction  # "LONG" | "SHORT"

    # ── MTF alignment (30%) ───────────────────────────────────────────────────
    mtf_score = getattr(signal, "mtf_score", 0)
    if direction == "LONG":
        mtf_component = (mtf_score + 3) / 6.0
    else:
        mtf_component = 1.0 - (mtf_score + 3) / 6.0
    mtf_component = max(0.0, min(1.0, mtf_component))

    # ── Confidence delta (20%) ────────────────────────────────────────────────
    current_conf = signal.confidence
    delta        = current_conf - original_confidence   # positive = improved
    # Map −0.20…+0.20 → 1.0…0.0 (positive delta = higher conf = better)
    conf_component = max(0.0, min(1.0, 0.5 + delta / 0.40))
    # Fold in signal-level adjustments: news pressure, strategy agreement, live win-rate calibration
    adjustments = (
        getattr(signal, "news_boost",      0.0) +   # -0.10 to +0.10
        getattr(signal, "consensus_boost", 0.0) +   # -0.08 to +0.05
        getattr(signal, "live_adjustment", 0.0)     # -0.15 to +0.15
    )
    conf_component = max(0.0, min(1.0, conf_component + adjustments))

    # ── Momentum: RSI + MACD hist (20%) ──────────────────────────────────────
    indicators = signal.indicators
    rsi  = float(getattr(indicators, "rsi",       50.0) or 50.0) if indicators else 50.0
    macd = float(getattr(indicators, "macd_histogram",  0.0) or 0.0)  if indicators else 0.0

    if direction == "LONG":
        rsi_ok   = 40 <= rsi <= 65
        macd_ok  = macd > 0
        rsi_bad  = rsi > 75 or rsi < 30
        macd_bad = macd < 0
    else:
        rsi_ok   = 35 <= rsi <= 60
        macd_ok  = macd < 0
        rsi_bad  = rsi < 25 or rsi > 70
        macd_bad = macd > 0

    if rsi_ok and macd_ok:
        mom_component = 1.0
    elif rsi_bad and macd_bad:
        mom_component = 0.0
    else:
        # Partial score
        mom_component = 0.5
        if rsi_ok or macd_ok:
            mom_component += 0.25
        if rsi_bad or macd_bad:
            mom_component -= 0.25
    mom_component = max(0.0, min(1.0, mom_component))

    # ── R:R remaining, normalised (15%) ──────────────────────────────────────
    original_tp_dist = abs(position.tp - position.open_price) if position.tp else (atr * 2)
    dist_to_tp       = abs(position.tp - current_price) if position.tp else 0.0
    if original_tp_dist > 0:
        rr_component = max(0.0, min(1.0, dist_to_tp / original_tp_dist))
    else:
        rr_component = 0.5

    # ── Pattern bias (15%) ────────────────────────────────────────────────────
    dominant_bias = getattr(signal.patterns, "dominant_bias", "NEUTRAL") if signal.patterns else "NEUTRAL"
    if (direction == "LONG" and dominant_bias == "BULLISH") or \
       (direction == "SHORT" and dominant_bias == "BEARISH"):
        pat_component = 1.0
    elif dominant_bias == "NEUTRAL":
        pat_component = 0.5
    else:
        pat_component = 0.0

    score = (
        0.30 * mtf_component    +
        0.20 * conf_component   +
        0.20 * mom_component    +
        0.15 * rr_component     +
        0.15 * pat_component
    )
    # ── Volatility penalty (applied after weighting) ──────────────────────────
    # High ATR rank = risky conditions; reduce health proportionally
    if indicators:
        vol_pct = getattr(indicators, "volatility_percentile", 50.0)
        if vol_pct > 80:
            score *= 0.90
        elif vol_pct > 65:
            score *= 0.95
    return round(max(0.0, min(1.0, score)), 3)


# ── Per-strategy max hold duration ───────────────────────────────────────────

_STRATEGY_MAX_HOURS: dict[str, float] = {
    "VWAP_RSI_5M":        3.0,
    "RSI2_VWAP":          3.0,
    "BB_SCALP_5M":        2.0,
    "STOCH_CROSS_5M":     2.0,
    "EMA_MICRO_CROSS_5M": 1.5,
    "TREND_MOM_5M":       8.0,
    "TREND_MOM":          8.0,
    "ORB_5M":            16.0,
    "ORB":               16.0,
    "NR7_BREAKOUT_5M":   12.0,
    "NR7_BO":            12.0,
    "ABSORB_15M":         6.0,
    "ABSORB_BO":          6.0,
    "TRIPLE_A":           6.0,
    "VA_BOUNCE":          4.0,
}


# ── Decision logic ────────────────────────────────────────────────────────────

def _decide_action(
    position,
    signal,
    health_score: float,
    atr: float,
    current_price: float,
    breakeven_price: float,
    sent_at: str,
    strategy: str = "",
    live_win_rates: dict | None = None,
) -> tuple[str, str, str, float | None, float | None, float | None]:
    """
    Returns (action, reason, urgency, new_sl, new_tp, partial_volume).
    Guards are checked first; score-based logic follows.
    """
    from utils.mt5_bridge import _load_learned_min

    direction = position.direction
    sl        = position.sl
    tp        = position.tp

    # ── Time stop: strategy-aware max hold duration ───────────────────────────
    try:
        opened = datetime.fromisoformat(sent_at).replace(tzinfo=timezone.utc)
        open_hours = (datetime.now(tz=timezone.utc) - opened).total_seconds() / 3600
    except Exception:
        open_hours = 0.0

    profit = position.profit

    # ── Live win-rate awareness: tighten effective health threshold ───────────
    live_wr_key = getattr(signal, "live_wr_key", "")
    if live_win_rates and live_wr_key and live_win_rates.get(live_wr_key, 0.5) < 0.45:
        health_score = max(0.0, health_score - 0.05)

    # ── Hard closes ──────────────────────────────────────────────────────────
    mtf_score = getattr(signal, "mtf_score", 0)
    sig_dir   = signal.direction
    sig_conf  = signal.confidence

    if health_score < 0.30:
        return "CLOSE", f"Health critical ({health_score:.2f}) — exiting to protect capital", "HIGH", None, None, None

    if sig_dir not in ("NO TRADE",) and sig_dir != direction and sig_conf > 0.65:
        return "CLOSE", f"Signal reversed to {sig_dir} with {sig_conf:.0%} confidence", "HIGH", None, None, None

    if direction == "LONG" and mtf_score <= -2:
        return "CLOSE", f"MTF strongly bearish (score={mtf_score}) vs LONG position", "HIGH", None, None, None
    if direction == "SHORT" and mtf_score >= 2:
        return "CLOSE", f"MTF strongly bullish (score={mtf_score}) vs SHORT position", "HIGH", None, None, None

    # Daily trend vetoed — higher timeframe directly opposing position
    if getattr(signal, "daily_trend_vetoed", False):
        return "CLOSE", "Daily trend vetoed signal direction — exiting to align with higher timeframe", "HIGH", None, None, None

    # RSI divergence — momentum exhaustion signal
    indicators = signal.indicators
    if indicators:
        if direction == "LONG" and getattr(indicators, "rsi_divergence_bearish", False):
            return "CLOSE", "RSI bearish divergence — price made higher high but RSI lower high; momentum exhausting", "HIGH", None, None, None
        if direction == "SHORT" and getattr(indicators, "rsi_divergence_bullish", False):
            return "CLOSE", "RSI bullish divergence — price made lower low but RSI higher low; momentum exhausting", "HIGH", None, None, None

    # High-confidence reversal patterns against position → partial close
    _BEARISH_REVERSALS = {"Double Top", "Triple Top", "Head & Shoulders", "Bearish Engulfing",
                          "Evening Star", "Shooting Star", "Gravestone Doji", "Hanging Man",
                          "Bearish Harami", "Rising Wedge"}
    _BULLISH_REVERSALS = {"Double Bottom", "Triple Bottom", "Inv. Head & Shoulders", "Bullish Engulfing",
                          "Morning Star", "Hammer", "Dragonfly Doji", "Bullish Harami",
                          "Falling Wedge"}
    if signal.patterns:
        for pat in (signal.patterns.patterns or []):
            pat_name = getattr(pat, "name", "")
            pat_conf = getattr(pat, "confidence", 0.0)
            if pat_conf < 0.65:
                continue
            if direction == "LONG" and pat_name in _BEARISH_REVERSALS:
                return "PARTIAL_CLOSE", f"{pat_name} ({pat_conf:.0%} conf) — reversal against LONG; locking partial profit", "MEDIUM", None, None, None
            if direction == "SHORT" and pat_name in _BULLISH_REVERSALS:
                return "PARTIAL_CLOSE", f"{pat_name} ({pat_conf:.0%} conf) — reversal against SHORT; locking partial profit", "MEDIUM", None, None, None

    # Per-strategy time stop
    max_hours = _STRATEGY_MAX_HOURS.get(strategy, 8.0)
    if open_hours > max_hours and profit <= 0:
        return "CLOSE", f"Time stop: {strategy or 'trade'} open {open_hours:.1f}h (max {max_hours:.0f}h), profit {profit:.2f}", "MEDIUM", None, None, None

    # News sentiment against position + declining health
    news_score = getattr(signal, "news_score", 0.0)
    if health_score < 0.55:
        if direction == "LONG" and news_score < -0.35:
            return "CLOSE", f"Bearish news pressure (score {news_score:.2f}) with declining health ({health_score:.2f})", "MEDIUM", None, None, None
        if direction == "SHORT" and news_score > 0.35:
            return "CLOSE", f"Bullish news pressure (score {news_score:.2f}) with declining health ({health_score:.2f})", "MEDIUM", None, None, None

    # ── Score-based: tighten SL ───────────────────────────────────────────────
    if health_score < 0.50:
        if direction == "LONG":
            trail_sl = current_price - 2.0 * atr
            new_sl   = max(sl, trail_sl) if sl else trail_sl
        else:
            trail_sl = current_price + 2.0 * atr
            new_sl   = min(sl, trail_sl) if sl else trail_sl
        new_sl = round(new_sl, 5)
        if (direction == "LONG" and new_sl > sl) or (direction == "SHORT" and new_sl < sl):
            return "MODIFY_SL", f"Health declining ({health_score:.2f}) — tightening SL", "MEDIUM", new_sl, None, None
        return "HOLD", f"Health low ({health_score:.2f}) but SL already tight — monitoring", "LOW", None, None, None

    # ── Score-based: profit management ───────────────────────────────────────
    if health_score >= 0.70:
        sl_dist = abs(current_price - sl) if sl else atr
        one_r   = sl_dist

        # Partial close + move to breakeven when ≥ 1R profit
        if profit >= one_r and profit >= atr:
            vol_step = 0.01
            try:
                import MetaTrader5 as mt5
                si = mt5.symbol_info(position.symbol)
                if si:
                    vol_step = si.volume_step or 0.01
            except Exception:
                pass
            half_vol = round(int((position.volume / 2) / vol_step) * vol_step, 8)
            half_vol = max(vol_step, half_vol)

            new_sl = round(breakeven_price, 5)
            # Add 2× spread buffer so slippage can't trigger the breakeven SL
            try:
                import MetaTrader5 as mt5
                tick = mt5.symbol_info_tick(position.symbol)
                if tick:
                    spread = abs(tick.ask - tick.bid)
                    new_sl = new_sl + spread * 2 if direction == "LONG" else new_sl - spread * 2
                    new_sl = round(new_sl, 5)
            except Exception:
                pass
            # Only move SL if it actually improves
            if (direction == "LONG" and new_sl <= sl) or (direction == "SHORT" and new_sl >= sl):
                new_sl = None
            return (
                "PARTIAL_CLOSE",
                f"Profit ≥ 1R ({profit:.2f}) — locking half, moving SL to breakeven",
                "LOW",
                new_sl,
                None,
                half_vol,
            )

        # Progressive trailing SL — tighten multiplier as profit grows
        if profit >= 2 * atr:
            if profit >= 4 * atr:
                multiplier = 1.0
            elif profit >= 3 * atr:
                multiplier = 1.2
            else:
                multiplier = 1.5
            if direction == "LONG":
                trail_sl = current_price - multiplier * atr
                new_sl   = max(sl, trail_sl) if sl else trail_sl
            else:
                trail_sl = current_price + multiplier * atr
                new_sl   = min(sl, trail_sl) if sl else trail_sl
            new_sl = round(new_sl, 5)
            if (direction == "LONG" and new_sl > sl) or (direction == "SHORT" and new_sl < sl):
                return "MODIFY_SL", f"Trailing SL ({multiplier}×ATR) after {profit:.2f} profit", "LOW", new_sl, None, None

        # Extend TP when health > 0.85 + continuation pattern
        if health_score > 0.85 and tp:
            dominant_bias = getattr(signal.patterns, "dominant_bias", "NEUTRAL") if signal.patterns else "NEUTRAL"
            continuation  = (direction == "LONG" and dominant_bias == "BULLISH") or \
                            (direction == "SHORT" and dominant_bias == "BEARISH")
            if continuation:
                original_tp_dist = abs(tp - position.open_price)
                if direction == "LONG":
                    new_tp = round(tp + atr, 5)
                else:
                    new_tp = round(tp - atr, 5)
                new_dist = abs(new_tp - position.open_price)
                if new_dist >= original_tp_dist:
                    return "MODIFY_TP", "Strong trend continuation — extending TP by 1×ATR", "LOW", None, new_tp, None

    return "HOLD", f"Health OK ({health_score:.2f}) — holding position", "LOW", None, None, None


# ── Per-position assessment ───────────────────────────────────────────────────

def assess_trade(
    position,
    finnhub_key: str = "",
    account_size: float = 100_000.0,
    risk_pct: float = 1.0,
    use_daily: bool = True,
    cooldown_state: dict | None = None,
    dry_run: bool = True,
) -> TradeAssessment:
    """
    Full assessment of one open position.
    Returns a TradeAssessment; on error returns HOLD with error field set.
    """
    if cooldown_state is None:
        cooldown_state = {}

    now_iso = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    try:
        from agents.signal_engine import run_signal
        from data.fetcher_intraday import fetch_daily_trend, fetch_h4_trend
        from data.trade_outcomes import _load
        from data.economic_calendar import fetch_upcoming_events, has_high_impact_event
        import MetaTrader5 as mt5

        symbol    = position.symbol
        direction = position.direction
        ticket    = position.ticket

        _tm_log(f"  Assessing #{ticket} {symbol} {direction} @ open={position.open_price} profit=${position.profit:+.2f}")

        # Look up current live price
        tick = mt5.symbol_info_tick(symbol)
        current_price = (tick.bid if direction == "LONG" else tick.ask) if tick else position.open_price

        # Breakeven
        breakeven = _calc_breakeven(position)

        # ── Guard 1: Minimum bars ─────────────────────────────────────────────
        trade_log = _load()
        trade_rec = next((t for t in trade_log if t["ticket"] == ticket), None)
        sent_at   = trade_rec.get("sent_at") if trade_rec else None
        original_confidence = float(trade_rec.get("confidence", 0.5)) if trade_rec else 0.5
        strategy  = trade_rec.get("strategy", "") if trade_rec else ""

        # Fall back to MT5's own open timestamp when trade_log has no sent_at
        if not sent_at and getattr(position, "time_open", 0):
            try:
                sent_at = datetime.fromtimestamp(position.time_open, tz=timezone.utc).isoformat(timespec="seconds")
            except Exception:
                pass

        if sent_at:
            try:
                opened = datetime.fromisoformat(sent_at).replace(tzinfo=timezone.utc)
                age_min = (datetime.now(tz=timezone.utc) - opened).total_seconds() / 60
                if age_min < _MIN_BARS_MINUTES:
                    _tm_log(f"    #{ticket} too new ({age_min:.0f} min < {_MIN_BARS_MINUTES} min) — skipping")
                    return TradeAssessment(
                        ticket=ticket, symbol=symbol, direction=direction,
                        open_price=position.open_price, breakeven_price=breakeven,
                        current_price=current_price, current_sl=position.sl,
                        current_tp=position.tp, current_profit=position.profit,
                        original_confidence=original_confidence,
                        action="HOLD", urgency="LOW", health_score=0.5,
                        reason=f"Trade too new ({age_min:.0f} min) — waiting for first bars",
                        assessed_at=now_iso, dry_run=dry_run,
                    )
            except Exception:
                pass

        # ── Guard 2: Cooldown ─────────────────────────────────────────────────
        if _is_on_cooldown(ticket, cooldown_state):
            _tm_log(f"  #{ticket} {symbol} {direction} | HOLD (cooldown active — acted recently)")
            return TradeAssessment(
                ticket=ticket, symbol=symbol, direction=direction,
                open_price=position.open_price, breakeven_price=breakeven,
                current_price=current_price, current_sl=position.sl,
                current_tp=position.tp, current_profit=position.profit,
                original_confidence=original_confidence,
                action="HOLD", urgency="LOW", health_score=0.5,
                reason="Cooldown active — recently acted on this position",
                assessed_at=now_iso, dry_run=dry_run,
            )

        # ── Fresh signal ──────────────────────────────────────────────────────
        from data.fetcher_intraday import _MT5_TO_YF
        yf_ticker = _MT5_TO_YF.get(symbol.upper(), symbol)

        daily_trend = None
        h4_trend    = None
        if use_daily:
            try:
                daily_trend = fetch_daily_trend(yf_ticker)
            except Exception:
                pass
            try:
                h4_trend = fetch_h4_trend(yf_ticker, mt5_symbol=symbol)
            except Exception:
                pass

        from data.trade_outcomes import compute_optimal_stops, compute_detailed_win_rates
        opt_stops  = compute_optimal_stops()
        live_rates = compute_detailed_win_rates()

        signal = run_signal(
            ticker=yf_ticker,
            finnhub_api_key=finnhub_key,
            account_size=account_size,
            risk_pct=risk_pct,
            daily_trend=daily_trend,
            h4_trend=h4_trend,
            optimal_stops=opt_stops or None,
            live_win_rates=live_rates or None,
            mt5_symbol=symbol,
        )

        atr = signal.atr or abs(position.open_price - position.sl) or (position.open_price * 0.001)

        # ── Guard 3: News hold ────────────────────────────────────────────────
        blocked, news_reason = has_high_impact_event(symbol, within_minutes=30)

        # ── Health score ──────────────────────────────────────────────────────
        health_score = _compute_health_score(
            position, signal, original_confidence, current_price, atr
        )
        _tm_log(f"    #{ticket} {symbol} health={health_score:.2f} | signal={signal.signal} conf={signal.confidence:.2f} atr={atr:.5f}")

        # ── Decision ──────────────────────────────────────────────────────────
        action, reason, urgency, new_sl, new_tp, partial_vol = _decide_action(
            position, signal, health_score, atr, current_price, breakeven,
            sent_at or now_iso,
            strategy=strategy,
            live_win_rates=live_rates or None,
        )

        # Apply news hold guard: allow CLOSE if health critical; block other modifications
        if blocked and action in ("CLOSE", "PARTIAL_CLOSE", "MODIFY_SL", "MODIFY_TP", "MODIFY_BOTH"):
            if health_score < 0.50 and action == "CLOSE":
                reason = f"Closing before news: {news_reason} (health={health_score:.2f})"
                _tm_log(f"    #{ticket} NEWS CLOSE — {news_reason}")
            else:
                _tm_log(f"    #{ticket} NEWS HOLD overrides {action} — {news_reason}")
                action   = "HOLD"
                reason   = f"News hold — {news_reason}"
                urgency  = "MEDIUM"
                new_sl   = None
                new_tp   = None
                partial_vol = None

        return TradeAssessment(
            ticket=ticket,
            symbol=symbol,
            direction=direction,
            open_price=position.open_price,
            breakeven_price=round(breakeven, 5),
            current_price=current_price,
            current_sl=position.sl,
            current_tp=position.tp,
            current_profit=position.profit,
            original_confidence=original_confidence,
            action=action,
            reason=reason,
            urgency=urgency,
            health_score=health_score,
            new_sl=new_sl,
            new_tp=new_tp,
            partial_close_volume=partial_vol,
            mtf_score=signal.mtf_score,
            mtf_bias=signal.mtf_bias,
            signal_direction=signal.direction,
            signal_confidence=signal.confidence,
            adx_regime=signal.adx_regime,
            strategy_agreement=signal.strategy_agreement,
            assessed_at=now_iso,
            dry_run=dry_run,
        )

    except Exception as exc:
        import traceback
        logger.error(f"assess_trade #{getattr(position, 'ticket', '?')}: {exc}\n{traceback.format_exc()}")
        return TradeAssessment(
            ticket=getattr(position, "ticket", 0),
            symbol=getattr(position, "symbol", "?"),
            direction=getattr(position, "direction", "?"),
            open_price=getattr(position, "open_price", 0.0),
            breakeven_price=getattr(position, "open_price", 0.0),
            current_price=getattr(position, "open_price", 0.0),
            current_sl=getattr(position, "sl", 0.0),
            current_tp=getattr(position, "tp", 0.0),
            current_profit=getattr(position, "profit", 0.0),
            original_confidence=0.5,
            action="HOLD", reason=f"Assessment error — holding for safety",
            urgency="LOW", health_score=0.5,
            assessed_at=now_iso, dry_run=dry_run,
            error=str(exc),
        )


# ── Execute action ────────────────────────────────────────────────────────────

def _execute_assessment(assessment: TradeAssessment, cooldown_state: dict) -> bool:
    """Execute the recommended action via MT5. Returns True on success."""
    from utils.mt5_bridge import (
        close_position, partial_close_position, modify_position,
    )

    action = assessment.action
    ticket = assessment.ticket

    try:
        if action == "CLOSE":
            ok = close_position(ticket)
        elif action == "PARTIAL_CLOSE":
            ok = partial_close_position(ticket, assessment.partial_close_volume or 0.01)
            if ok and assessment.new_sl is not None:
                sl_ok = modify_position(ticket, new_sl=assessment.new_sl)
                ok = ok and sl_ok
        elif action in ("MODIFY_SL", "MODIFY_BOTH"):
            ok = modify_position(ticket, new_sl=assessment.new_sl, new_tp=assessment.new_tp)
        elif action == "MODIFY_TP":
            ok = modify_position(ticket, new_tp=assessment.new_tp)
        else:
            return False   # HOLD — nothing to do

        if ok:
            assessment.acted_on = True
            _set_cooldown(ticket, cooldown_state)
            logger.info(f"TradeManager acted: {action} #{ticket}")
        else:
            logger.warning(f"TradeManager action failed: {action} #{ticket}")
        return ok
    except Exception as exc:
        logger.error(f"_execute_assessment #{ticket}: {exc}")
        _tm_log(f"  EXECUTE ERROR #{ticket}: {exc}")
        return False


# ── Public entry point ────────────────────────────────────────────────────────

def assess_all_open_trades(
    positions: list,
    finnhub_key: str = "",
    account_size: float = 100_000.0,
    risk_pct: float = 1.0,
    use_daily: bool = True,
    cooldown_state: dict | None = None,
    dry_run: bool = True,
) -> list[TradeAssessment]:
    """
    Assess all open positions in parallel.
    When dry_run=False, executes each recommended action (except HOLD).
    Returns list sorted by urgency (HIGH first) then health_score ascending.
    """
    if cooldown_state is None:
        cooldown_state = {}

    assessments: list[TradeAssessment] = []

    def _one(pos):
        return assess_trade(
            pos,
            finnhub_key=finnhub_key,
            account_size=account_size,
            risk_pct=risk_pct,
            use_daily=use_daily,
            cooldown_state=cooldown_state,
            dry_run=dry_run,
        )

    mode = "LIVE" if not dry_run else "DRY-RUN"
    _tm_log(f"── Assessing {len(positions)} open position(s) [{mode}] ──")

    from concurrent.futures import TimeoutError as _FutureTimeout
    with ThreadPoolExecutor(max_workers=min(len(positions), 4)) as ex:
        futures = {ex.submit(_one, p): p for p in positions}
        for fut in as_completed(futures, timeout=90):
            try:
                a = fut.result(timeout=60)
                assessments.append(a)
                _append_assessment_log(a)

                profit_str = f"P&L ${a.current_profit:+.2f}"
                health_str = f"health={a.health_score:.2f}"
                if a.error:
                    _tm_log(f"  #{a.ticket} {a.symbol} {a.direction} | ERROR: {a.error}")
                elif a.action == "HOLD":
                    _tm_log(f"  #{a.ticket} {a.symbol} {a.direction} | HOLD  {health_str} {profit_str} | {a.reason}")
                else:
                    _tm_log(f"  #{a.ticket} {a.symbol} {a.direction} | {a.action} [{a.urgency}] {health_str} {profit_str} | {a.reason}")

                if not dry_run and a.action != "HOLD" and not a.error:
                    ok = _execute_assessment(a, cooldown_state)
                    _append_assessment_log(a)
                    status = "✓ executed" if ok else "✗ failed"
                    _tm_log(f"    → {a.action} #{a.ticket}: {status}")
            except _FutureTimeout:
                pos = futures[fut]
                _tm_log(f"  #{pos.ticket} {pos.symbol} | TIMEOUT — skipping assessment")
            except Exception as exc:
                logger.error(f"assess future error: {exc}")
                _tm_log(f"  ERROR assessing position: {exc}")

    urgency_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    assessments.sort(key=lambda a: (urgency_order.get(a.urgency, 2), a.health_score))
    _tm_log(f"── Assessment complete: {len(assessments)} result(s) ──")
    return assessments
