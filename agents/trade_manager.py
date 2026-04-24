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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

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

    # ── Momentum: RSI + MACD hist (20%) ──────────────────────────────────────
    indicators = signal.indicators
    rsi  = float(getattr(indicators, "rsi",       50.0) or 50.0) if indicators else 50.0
    macd = float(getattr(indicators, "macd_hist",  0.0) or 0.0)  if indicators else 0.0

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
    return round(max(0.0, min(1.0, score)), 3)


# ── Decision logic ────────────────────────────────────────────────────────────

def _decide_action(
    position,
    signal,
    health_score: float,
    atr: float,
    current_price: float,
    breakeven_price: float,
    sent_at: str,
) -> tuple[str, str, str, float | None, float | None, float | None]:
    """
    Returns (action, reason, urgency, new_sl, new_tp, partial_volume).
    Guards are checked first; score-based logic follows.
    """
    from utils.mt5_bridge import _load_learned_min

    direction = position.direction
    sl        = position.sl
    tp        = position.tp

    # ── Time stop: open > 8h with minimal profit ──────────────────────────────
    try:
        opened = datetime.fromisoformat(sent_at).replace(tzinfo=timezone.utc)
        open_hours = (datetime.now(tz=timezone.utc) - opened).total_seconds() / 3600
    except Exception:
        open_hours = 0.0

    profit = position.profit

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

    # Time stop
    if open_hours > 8 and profit < 0.5 * atr:
        return "CLOSE", f"Time stop: open {open_hours:.1f}h, profit {profit:.2f} < 0.5×ATR", "MEDIUM", None, None, None

    # ── Score-based: tighten SL ───────────────────────────────────────────────
    if health_score < 0.50:
        # Tighten SL toward ATR trail
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
        # 1R = SL distance
        sl_dist = abs(current_price - sl) if sl else atr
        one_r   = sl_dist

        # Partial close + move to breakeven when ≥ 1R profit
        if profit >= one_r and profit >= atr:
            sym_info = None
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

        # Trail SL at 2× ATR when ≥ 2×ATR profit
        if profit >= 2 * atr:
            if direction == "LONG":
                trail_sl = current_price - 1.5 * atr
                new_sl   = max(sl, trail_sl) if sl else trail_sl
            else:
                trail_sl = current_price + 1.5 * atr
                new_sl   = min(sl, trail_sl) if sl else trail_sl
            new_sl = round(new_sl, 5)
            if (direction == "LONG" and new_sl > sl) or (direction == "SHORT" and new_sl < sl):
                return "MODIFY_SL", f"Trailing SL after {profit:.2f} profit (≥2×ATR)", "LOW", new_sl, None, None

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
                if new_dist >= original_tp_dist:   # never shorten
                    return "MODIFY_TP", f"Strong trend continuation — extending TP by 1×ATR", "LOW", None, new_tp, None

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

        # Look up current live price
        tick = mt5.symbol_info_tick(symbol)
        current_price = (tick.bid if direction == "LONG" else tick.ask) if tick else position.open_price

        # Breakeven
        breakeven = _calc_breakeven(position)

        # ── Guard 1: Minimum bars ─────────────────────────────────────────────
        trade_log = _load()
        trade_rec = next((t for t in trade_log if t["ticket"] == ticket), None)
        sent_at   = trade_rec["sent_at"] if trade_rec else None
        original_confidence = float(trade_rec["confidence"]) if trade_rec else 0.5

        if sent_at:
            try:
                opened = datetime.fromisoformat(sent_at).replace(tzinfo=timezone.utc)
                age_min = (datetime.now(tz=timezone.utc) - opened).total_seconds() / 60
                if age_min < _MIN_BARS_MINUTES:
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

        # ── Decision ──────────────────────────────────────────────────────────
        action, reason, urgency, new_sl, new_tp, partial_vol = _decide_action(
            position, signal, health_score, atr, current_price, breakeven,
            sent_at or now_iso,
        )

        # Apply news hold guard: override MODIFY/CLOSE with HOLD
        if blocked and action in ("CLOSE", "PARTIAL_CLOSE", "MODIFY_SL", "MODIFY_TP", "MODIFY_BOTH"):
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
                modify_position(ticket, new_sl=assessment.new_sl)
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
        return ok
    except Exception as exc:
        logger.error(f"_execute_assessment #{ticket}: {exc}")
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

    with ThreadPoolExecutor(max_workers=min(len(positions), 4)) as ex:
        futures = {ex.submit(_one, p): p for p in positions}
        for fut in as_completed(futures):
            try:
                a = fut.result()
                assessments.append(a)
                _append_assessment_log(a)
                if not dry_run and a.action != "HOLD" and not a.error:
                    _execute_assessment(a, cooldown_state)
                    _append_assessment_log(a)   # re-log with acted_on updated
            except Exception as exc:
                logger.error(f"assess future error: {exc}")

    urgency_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    assessments.sort(key=lambda a: (urgency_order.get(a.urgency, 2), a.health_score))
    return assessments
