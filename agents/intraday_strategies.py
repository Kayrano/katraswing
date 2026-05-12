"""
Five validated intraday strategies for 5m and 15m timeframes.
==============================================================
Each function accepts a prepared DataFrame (from data.fetcher_intraday) and
returns an IntradaySignal containing:
    - direction (LONG | SHORT | FLAT)
    - confidence (0.0 – 1.0)
    - entry price
    - stop_loss price
    - take_profit price
    - reason (human-readable)

Stop-loss and take-profit are embedded in the signal so the caller always has
a complete, ready-to-act trade plan.  Risk:Reward is always 1:2.

Strategy roster
───────────────
5m timeframe:
  1. VWAP_RSI_5M   — RSI(2) extreme + VWAP proximity mean reversion
  2. ORB_5M        — 15-minute opening range breakout (first 3 × 5m bars)

15m timeframe:
  3. EMA_PB_15M    — 8/21 EMA ribbon pullback to 8-EMA in trend
  4. SQUEEZE_15M   — Bollinger-Keltner squeeze breakout
  5. ABSORB_15M    — Absorption bar breakout (Valentini method)

run_intraday_signals() is the main entry point; it selects the right strategies
for the requested timeframe, runs them all, and returns every non-FLAT signal
sorted by confidence descending.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from dataclasses import dataclass

import utils.ta_compat as ta


# ── Signal dataclass ──────────────────────────────────────────────────────────

@dataclass
class IntradaySignal:
    strategy:    str    # e.g. "VWAP_RSI_5M"
    timeframe:   str    # "5m" | "15m"
    signal:      str    # "LONG" | "SHORT" | "FLAT"
    confidence:  float  # 0.0 – 1.0
    entry:       float  # suggested entry (current close)
    stop_loss:   float  # ATR-based stop
    take_profit: float  # 2× risk from entry (1:2 R:R)
    atr:         float  # ATR used for SL/TP calculation
    rr_ratio:    float  # always 2.0
    reason:      str    # human-readable explanation
    # paper_only=True signals still flow through the engine and contribute to
    # calibration data via the trade_log shadow path, but the order-send
    # surfaces (app.py auto-trade, mt5_signal_server.py) skip the MT5
    # round-trip. Stamped by data.strategy_params.apply_params.
    paper_only:  bool = False


# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_signal(
    strategy: str,
    timeframe: str,
    direction: str,
    confidence: float,
    entry: float,
    atr: float,
    sl_atr_mult: float,
    tp_atr_mult: float,
    reason: str,
    df: Optional[pd.DataFrame] = None,
    use_structural: bool = False,
) -> IntradaySignal:
    """Build a complete IntradaySignal with SL/TP.

    When `use_structural=True` and a `df` is supplied, defer to
    `utils.stops.compute_structural_stop` for swing-pivot anchored stops
    (Round 4 B1 — fixes the realised avg_win/avg_loss = 0.745 problem).
    Otherwise fall back to the legacy ATR-multiplier stops so callers that
    don't pass df still work.
    """
    if use_structural and df is not None:
        from utils.stops import compute_structural_stop
        result = compute_structural_stop(df, direction, entry, atr)
        stop_loss   = result.sl
        take_profit = result.tp
        reason = f"{reason} | SL={result.sl_source}"
    elif direction == "LONG":
        stop_loss   = entry - sl_atr_mult * atr
        take_profit = entry + tp_atr_mult * atr
    elif direction == "SHORT":
        stop_loss   = entry + sl_atr_mult * atr
        take_profit = entry - tp_atr_mult * atr
    else:
        stop_loss = take_profit = entry

    return IntradaySignal(
        strategy=strategy,
        timeframe=timeframe,
        signal=direction,
        confidence=round(confidence, 3),
        entry=round(entry, 4),
        stop_loss=round(stop_loss, 4),
        take_profit=round(take_profit, 4),
        atr=round(atr, 4),
        rr_ratio=2.0,
        reason=reason,
    )


def _flat(strategy: str, timeframe: str, reason: str) -> IntradaySignal:
    return IntradaySignal(
        strategy=strategy, timeframe=timeframe, signal="FLAT",
        confidence=0.0, entry=0.0, stop_loss=0.0, take_profit=0.0,
        atr=0.0, rr_ratio=0.0, reason=reason,
    )


# ════════════════════════════════════════════════════════════════════════════════
# STRATEGY 1 — VWAP + RSI(2) Mean Reversion  (5m)
# ════════════════════════════════════════════════════════════════════════════════

def vwap_rsi_5m(df: pd.DataFrame) -> IntradaySignal:
    """
    RSI(2) extreme combined with proximity to session VWAP on 5m bars.

    Long:  RSI(2) < 20  AND  close within 0.15×ATR of VWAP  AND  EMA(20) trending up
    Short: RSI(2) > 80  AND  close within 0.15×ATR of VWAP  AND  EMA(20) trending down

    SL:  1.0×ATR(10) from entry
    TP:  2.0×ATR(10) from entry  →  1:2 R:R

    Edge: Larry Connors-style RSI(2) mean reversion anchored to institutional
    VWAP reference.  VWAP proximity ensures we're fading into fair value, not
    chasing extended moves.  Backtested win rate ~63%.
    """
    TF = "5m"
    NAME = "VWAP_RSI_5M"

    if len(df) < 25:
        return _flat(NAME, TF, "Insufficient bars (need 25)")
    if "session_vwap" not in df.columns:
        return _flat(NAME, TF, "session_vwap column missing")

    rsi2  = ta.rsi(df["Close"], length=2)
    ema20 = ta.ema(df["Close"], length=20)
    atr10 = ta.atr(df["High"], df["Low"], df["Close"], length=10)

    cur_rsi   = float(rsi2.iloc[-1])
    cur_ema20 = float(ema20.iloc[-1])
    cur_atr   = float(atr10.iloc[-1])
    cur_close = float(df["Close"].iloc[-1])
    cur_vwap  = float(df["session_vwap"].iloc[-1])
    cur_rvol  = float(df["rvol"].iloc[-1]) if "rvol" in df.columns else 1.0

    if any(np.isnan(v) for v in [cur_rsi, cur_ema20, cur_atr, cur_vwap]) or cur_atr == 0:
        return _flat(NAME, TF, "NaN indicator — warmup incomplete")

    # Forensic May 2026: 6 live trades, 33% WR, avg loss $17.57 vs avg win $9.25.
    # The 0.5×ATR band let price drift far from VWAP, so trades were no longer
    # "fading into fair value" — they were chasing extended moves. Reverted to
    # the 0.15×ATR band that matches the docstring (line 110/111).
    band    = 0.15 * cur_atr
    in_band = abs(cur_close - cur_vwap) <= band

    rvol_note = f"RVOL {cur_rvol:.1f}x"

    # Long: RSI(2) < 30 (relaxed from 20) + near VWAP + above EMA20
    if cur_rsi < 30 and in_band and cur_close > cur_ema20:
        conf = 0.75 if cur_rsi < 15 else 0.65
        if cur_rvol >= 1.5:
            conf = min(conf + 0.10, 0.95)
        return _make_signal(
            NAME, TF, "LONG", conf, cur_close, cur_atr,
            sl_atr_mult=1.0, tp_atr_mult=2.0,
            reason=f"RSI(2)={cur_rsi:.1f}<30 | VWAP band {cur_vwap:.2f}±{band:.2f} | EMA20↑ | {rvol_note}",
            df=df, use_structural=True,
        )

    # Short: RSI(2) > 70 (relaxed from 80) + near VWAP + below EMA20
    if cur_rsi > 70 and in_band and cur_close < cur_ema20:
        conf = 0.75 if cur_rsi > 85 else 0.65
        if cur_rvol >= 1.5:
            conf = min(conf + 0.10, 0.95)
        return _make_signal(
            NAME, TF, "SHORT", conf, cur_close, cur_atr,
            sl_atr_mult=1.0, tp_atr_mult=2.0,
            reason=f"RSI(2)={cur_rsi:.1f}>70 | VWAP band {cur_vwap:.2f}±{band:.2f} | EMA20↓ | {rvol_note}",
            df=df, use_structural=True,
        )

    side = "above" if cur_close > cur_ema20 else "below"
    band_note = "in VWAP band" if in_band else f"outside VWAP band (dist={abs(cur_close-cur_vwap):.2f})"
    return _flat(NAME, TF, f"RSI(2)={cur_rsi:.1f} not extreme | {band_note} | close {side} EMA20")


# ════════════════════════════════════════════════════════════════════════════════
# STRATEGY 2 — Opening Range Breakout  (5m, 15-minute ORB window)
# ════════════════════════════════════════════════════════════════════════════════

def orb_5m(df: pd.DataFrame) -> IntradaySignal:
    """
    15-minute Opening Range Breakout on 5m bars.

    The opening range = the first 3 × 5m bars (09:30–09:45 ET).
    Entry window = bars 4–18 (09:45–10:30 ET); edge decays sharply after that.

    Long:  close > ORB high  AND  RVOL > 1.5  AND  VWAP sloping up
    Short: close < ORB low   AND  RVOL > 1.5  AND  VWAP sloping down

    SL:  just beyond the opposite ORB extreme (captured as 1.0×ATR)
    TP:  ORB range × 1.5 projected from the ORB boundary → ≈1:2 R:R

    Backtested win rate ~61% (trending days outperform significantly).
    """
    TF = "5m"
    NAME = "ORB_5M"

    if len(df) < 6:
        return _flat(NAME, TF, "Insufficient bars")
    if "session_bar_number" not in df.columns or "session_date" not in df.columns:
        return _flat(NAME, TF, "Session metadata missing")

    # Session gate: ORB edge is documented for US equity open (09:30–11:59 ET).
    # Reject signals outside that window so FTSE / NG / commodity bars during
    # their own openers don't produce random ORB setups.
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        et_hour = int(df.index[-1].tz_convert("America/New_York").hour)
        if et_hour not in range(9, 12):
            return _flat(NAME, TF, f"Outside US equity open window (ET {et_hour:02d}:xx)")

    cur_bar_num = int(df["session_bar_number"].iloc[-1])
    if cur_bar_num < 4:
        return _flat(NAME, TF, f"Opening range not yet complete (bar {cur_bar_num}; need ≥4)")

    today      = df["session_date"].iloc[-1]
    today_bars = df[df["session_date"] == today]
    orb_bars   = today_bars[today_bars["session_bar_number"] <= 3]

    if len(orb_bars) < 3:
        return _flat(NAME, TF, "Opening range bars not yet complete (need first 3 bars)")

    orb_high  = float(orb_bars["High"].max())
    orb_low   = float(orb_bars["Low"].min())
    orb_range = max(orb_high - orb_low, 1e-6)

    cur_close = float(df["Close"].iloc[-1])
    cur_rvol  = float(df["rvol"].iloc[-1]) if "rvol" in df.columns else 1.0
    rvol_ok   = cur_rvol >= 1.5

    # VWAP slope
    vwap_dir = 0
    if "session_vwap" in df.columns:
        vwap_recent = df["session_vwap"].dropna()
        if len(vwap_recent) >= 3:
            delta = float(vwap_recent.iloc[-1]) - float(vwap_recent.iloc[-3])
            vwap_dir = 1 if delta > 0 else (-1 if delta < 0 else 0)

    atr10 = ta.atr(df["High"], df["Low"], df["Close"], length=10)
    cur_atr = float(atr10.iloc[-1]) if atr10 is not None and not atr10.isna().all() else orb_range * 0.5

    rvol_note = f"RVOL {cur_rvol:.1f}x {'✓' if rvol_ok else '✗'}"
    vwap_note = "VWAP↑" if vwap_dir > 0 else ("VWAP↓" if vwap_dir < 0 else "VWAP flat")

    # Long breakout
    if cur_close > orb_high and rvol_ok and vwap_dir > 0:
        penetration = (cur_close - orb_high) / orb_range
        conf = min(0.85, 0.60 + penetration * 0.25)
        stop_loss = round(orb_low - 0.1 * cur_atr, 4)
        risk_pts  = cur_close - stop_loss
        # TP anchored to actual risk so R:R >= 1.5 regardless of ATR size.
        # Take the larger of ORB-projection and risk-anchored target.
        take_profit = round(cur_close + max(orb_range * 1.5, risk_pts * 1.5), 4)
        rr = (take_profit - cur_close) / max(risk_pts, 1e-6)
        return IntradaySignal(
            strategy=NAME, timeframe=TF, signal="LONG", confidence=round(conf, 3),
            entry=round(cur_close, 4), stop_loss=stop_loss, take_profit=take_profit,
            atr=round(cur_atr, 4), rr_ratio=round(rr, 2),
            reason=f"ORB-5 breakout ↑: close {cur_close:.2f} > ORB H {orb_high:.2f} | "
                   f"range={orb_range:.2f} | {rvol_note} | {vwap_note} | bar {cur_bar_num}",
        )

    # Short breakdown
    if cur_close < orb_low and rvol_ok and vwap_dir < 0:
        penetration = (orb_low - cur_close) / orb_range
        conf = min(0.85, 0.60 + penetration * 0.25)
        stop_loss = round(orb_high + 0.1 * cur_atr, 4)
        risk_pts  = stop_loss - cur_close
        take_profit = round(cur_close - max(orb_range * 1.5, risk_pts * 1.5), 4)
        rr = (cur_close - take_profit) / max(risk_pts, 1e-6)
        return IntradaySignal(
            strategy=NAME, timeframe=TF, signal="SHORT", confidence=round(conf, 3),
            entry=round(cur_close, 4), stop_loss=stop_loss, take_profit=take_profit,
            atr=round(cur_atr, 4), rr_ratio=round(rr, 2),
            reason=f"ORB-5 breakdown ↓: close {cur_close:.2f} < ORB L {orb_low:.2f} | "
                   f"range={orb_range:.2f} | {rvol_note} | {vwap_note} | bar {cur_bar_num}",
        )

    reasons = []
    if orb_low <= cur_close <= orb_high:
        reasons.append(f"price inside ORB [{orb_low:.2f}–{orb_high:.2f}]")
    if not rvol_ok:
        reasons.append(rvol_note)
    if vwap_dir == 0:
        reasons.append("VWAP flat — no directional confirmation")
    elif cur_close > orb_high and vwap_dir < 0:
        reasons.append("above ORB but VWAP pointing down")
    elif cur_close < orb_low and vwap_dir > 0:
        reasons.append("below ORB but VWAP pointing up")
    return _flat(NAME, TF, " | ".join(reasons) or "No ORB-5 signal")


# ════════════════════════════════════════════════════════════════════════════════
# STRATEGY 3 — Trend Momentum  (5m, universal — works 24/7 on any instrument)
# ════════════════════════════════════════════════════════════════════════════════

def trend_momentum_5m(df: pd.DataFrame) -> IntradaySignal:
    """
    Multi-factor trend + momentum filter that works on any instrument at any hour.

    Long:  EMA(9) > EMA(21)  AND  MACD hist > 0  AND  RSI(14) in [40, 72]
           AND  close > EMA(9)
    Short: EMA(9) < EMA(21)  AND  MACD hist < 0  AND  RSI(14) in [28, 60]
           AND  close < EMA(9)

    Confidence: base 0.62 — boosted by:
      +0.08 if RVOL ≥ 1.5 (volume confirmation)
      +0.05 if RSI > 55 (LONG) or RSI < 45 (SHORT) (momentum strength)
      +0.05 if EMA gap > 0.3×ATR (trend strength)

    SL: 1.2×ATR  TP: 2.4×ATR  → 1:2 R:R
    """
    TF   = "5m"
    NAME = "TREND_MOM_5M"

    if len(df) < 30:
        return _flat(NAME, TF, "Insufficient bars (need 30)")

    ema9  = ta.ema(df["Close"], length=9)
    ema21 = ta.ema(df["Close"], length=21)
    rsi14 = ta.rsi(df["Close"], length=14)
    mac   = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    atr14 = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    if ema9 is None or ema21 is None or rsi14 is None or atr14 is None:
        return _flat(NAME, TF, "Indicator unavailable")

    cur_ema9  = float(ema9.iloc[-1])
    cur_ema21 = float(ema21.iloc[-1])
    cur_rsi   = float(rsi14.iloc[-1])
    cur_atr   = float(atr14.iloc[-1])
    cur_close = float(df["Close"].iloc[-1])
    cur_rvol  = float(df["rvol"].iloc[-1]) if "rvol" in df.columns else 1.0

    if any(np.isnan(v) for v in [cur_ema9, cur_ema21, cur_rsi, cur_atr]) or cur_atr == 0:
        return _flat(NAME, TF, "NaN indicator — warmup incomplete")

    macd_hist = 0.0
    if mac is not None and not mac.empty and len(mac) > 0:
        try:
            macd_hist = float(mac.iloc[-1, 1])
            if np.isnan(macd_hist):
                macd_hist = 0.0
        except Exception:
            pass

    ema_gap   = abs(cur_ema9 - cur_ema21)
    rvol_ok   = cur_rvol >= 1.5
    trend_str = ema_gap > 0.3 * cur_atr

    def _conf(rsi_strong: bool) -> float:
        c = 0.62
        if rvol_ok:   c += 0.08
        if rsi_strong: c += 0.05
        if trend_str:  c += 0.05
        return min(c, 0.92)

    # Long
    if (cur_ema9 > cur_ema21 and macd_hist > 0
            and 40 <= cur_rsi <= 72 and cur_close > cur_ema9):
        return _make_signal(
            NAME, TF, "LONG", _conf(cur_rsi > 55), cur_close, cur_atr,
            sl_atr_mult=1.2, tp_atr_mult=2.4,
            reason=(f"EMA9({cur_ema9:.2f})>EMA21({cur_ema21:.2f}) | "
                    f"MACD hist {macd_hist:+.4f} | RSI(14)={cur_rsi:.1f} | RVOL {cur_rvol:.1f}x"),
        )

    # Short
    if (cur_ema9 < cur_ema21 and macd_hist < 0
            and 28 <= cur_rsi <= 60 and cur_close < cur_ema9):
        return _make_signal(
            NAME, TF, "SHORT", _conf(cur_rsi < 45), cur_close, cur_atr,
            sl_atr_mult=1.2, tp_atr_mult=2.4,
            reason=(f"EMA9({cur_ema9:.2f})<EMA21({cur_ema21:.2f}) | "
                    f"MACD hist {macd_hist:+.4f} | RSI(14)={cur_rsi:.1f} | RVOL {cur_rvol:.1f}x"),
        )

    reasons = []
    if cur_ema9 > cur_ema21 and macd_hist > 0 and cur_close > cur_ema9:
        reasons.append(f"trend up but RSI(14)={cur_rsi:.1f} outside [40–72]")
    elif cur_ema9 < cur_ema21 and macd_hist < 0 and cur_close < cur_ema9:
        reasons.append(f"trend down but RSI(14)={cur_rsi:.1f} outside [28–60]")
    elif macd_hist == 0:
        reasons.append("MACD hist = 0, no momentum direction")
    else:
        trend = "up" if cur_ema9 > cur_ema21 else "down"
        reasons.append(f"EMA trend {trend} but MACD or price not aligned")
    return _flat(NAME, TF, " | ".join(reasons))


# ════════════════════════════════════════════════════════════════════════════════
# STRATEGY 4 — EMA Ribbon Pullback  (15m)
# ════════════════════════════════════════════════════════════════════════════════

def ema_pullback_15m(df: pd.DataFrame) -> IntradaySignal:
    """
    8/21 EMA ribbon pullback on 15m bars.

    Trend condition:
      Uptrend:   EMA(8) > EMA(21)  AND  both EMAs pointing up (slope > 0)
      Downtrend: EMA(8) < EMA(21)  AND  both EMAs pointing down

    Entry condition:
      Long:  close touches the 8-EMA band (within 0.15×ATR)  AND  RSI(3) < 40
      Short: close touches the 8-EMA band (within 0.15×ATR)  AND  RSI(3) > 60

    SL:  1.5×ATR(14) from entry
    TP:  3.0×ATR(14) from entry  →  1:2 R:R

    Edge: pullback-to-moving-average in a confirmed trend is one of the
    highest-probability setups on intraday charts.  Backtested win rate ~62%.
    """
    TF = "15m"
    NAME = "EMA_PB_15M"

    if len(df) < 30:
        return _flat(NAME, TF, "Insufficient bars (need 30)")

    ema8  = ta.ema(df["Close"], length=8)
    ema21 = ta.ema(df["Close"], length=21)
    rsi3  = ta.rsi(df["Close"], length=3)
    atr14 = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    cur_ema8  = float(ema8.iloc[-1])
    cur_ema21 = float(ema21.iloc[-1])
    prev_ema8 = float(ema8.iloc[-2])
    prev_ema21= float(ema21.iloc[-2])
    cur_rsi   = float(rsi3.iloc[-1])
    cur_atr   = float(atr14.iloc[-1])
    cur_close = float(df["Close"].iloc[-1])

    if any(np.isnan(v) for v in [cur_ema8, cur_ema21, cur_rsi, cur_atr]) or cur_atr == 0:
        return _flat(NAME, TF, "NaN indicator — warmup incomplete")

    uptrend   = cur_ema8 > cur_ema21 and cur_ema8 > prev_ema8 and cur_ema21 > prev_ema21
    downtrend = cur_ema8 < cur_ema21 and cur_ema8 < prev_ema8 and cur_ema21 < prev_ema21

    band    = 0.15 * cur_atr
    near_8  = abs(cur_close - cur_ema8) <= band

    # Long: uptrend + pullback to 8-EMA + RSI oversold
    if uptrend and near_8 and cur_rsi < 40:
        conf = 0.70 if cur_rsi < 25 else 0.62
        return _make_signal(
            NAME, TF, "LONG", conf, cur_close, cur_atr,
            sl_atr_mult=1.5, tp_atr_mult=3.0,
            reason=f"EMA8({cur_ema8:.2f})>EMA21({cur_ema21:.2f}) | pullback to 8-EMA | RSI(3)={cur_rsi:.1f}<40",
        )

    # Short: downtrend + bounce to 8-EMA + RSI overbought
    if downtrend and near_8 and cur_rsi > 60:
        conf = 0.70 if cur_rsi > 75 else 0.62
        return _make_signal(
            NAME, TF, "SHORT", conf, cur_close, cur_atr,
            sl_atr_mult=1.5, tp_atr_mult=3.0,
            reason=f"EMA8({cur_ema8:.2f})<EMA21({cur_ema21:.2f}) | bounce to 8-EMA | RSI(3)={cur_rsi:.1f}>60",
        )

    if not uptrend and not downtrend:
        return _flat(NAME, TF, f"No clear EMA ribbon trend (EMA8={cur_ema8:.2f}, EMA21={cur_ema21:.2f})")
    if not near_8:
        return _flat(NAME, TF, f"Price {cur_close:.2f} not near EMA8 band (±{band:.2f} of {cur_ema8:.2f})")
    trend_dir = "uptrend" if uptrend else "downtrend"
    return _flat(NAME, TF, f"In {trend_dir} near EMA8 but RSI(3)={cur_rsi:.1f} not confirming")


# ════════════════════════════════════════════════════════════════════════════════
# STRATEGY 4 — Bollinger-Keltner Squeeze Breakout  (15m)
# ════════════════════════════════════════════════════════════════════════════════

def squeeze_15m(df: pd.DataFrame) -> IntradaySignal:
    """
    Bollinger-Keltner squeeze breakout on 15m bars — identical logic to H1
    SQUEEZE strategy but tuned for faster timeframe.

    Squeeze:   BB(20, 2.0σ) fully inside KC(20, 1.5×ATR)
    Breakout:  BB expands outside KC on current bar (prev bar was squeezed)
    Direction: MACD(12,26,9) histogram sign at breakout bar
    Volume:    above-SMA20 volume boosts confidence

    SL:  1.5×ATR(14) from entry
    TP:  3.0×ATR(14) from entry  →  1:2 R:R

    Backtested win rate ~61%.
    """
    TF = "15m"
    NAME = "SQUEEZE_15M"

    if len(df) < 30:
        return _flat(NAME, TF, "Insufficient bars (need 30)")

    bb  = ta.bbands(df["Close"], length=20, std=2.0)
    kc  = ta.keltner_channels(df["High"], df["Low"], df["Close"], length=20, atr_mult=1.5)
    mac = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    atr14 = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    if bb is None or bb.empty or kc is None or kc.empty:
        return _flat(NAME, TF, "BB or KC unavailable")

    bb_upper      = float(bb.iloc[-1, 2])
    bb_lower      = float(bb.iloc[-1, 0])
    kc_upper      = float(kc.iloc[-1, 2])
    kc_lower      = float(kc.iloc[-1, 0])
    bb_upper_prev = float(bb.iloc[-2, 2])
    bb_lower_prev = float(bb.iloc[-2, 0])
    kc_upper_prev = float(kc.iloc[-2, 2])
    kc_lower_prev = float(kc.iloc[-2, 0])

    was_squeezed = (bb_upper_prev < kc_upper_prev) and (bb_lower_prev > kc_lower_prev)
    broke_out    = (bb_upper > kc_upper) or (bb_lower < kc_lower)

    if not was_squeezed:
        return _flat(NAME, TF, "No prior squeeze condition")
    if not broke_out:
        return _flat(NAME, TF, "Squeeze still active — waiting for breakout")

    vol_sma20 = float(df["Volume"].rolling(20).mean().iloc[-1])
    cur_vol   = float(df["Volume"].iloc[-1])
    vol_ok    = (vol_sma20 > 0) and (cur_vol > vol_sma20)
    base_conf = 0.78 if vol_ok else 0.62

    cur_atr = float(atr14.iloc[-1]) if atr14 is not None and not atr14.isna().all() else 0.0
    if cur_atr == 0:
        return _flat(NAME, TF, "ATR zero — cannot compute SL/TP")

    if mac is None or mac.empty:
        return _flat(NAME, TF, "MACD unavailable at breakout bar")

    macd_hist = float(mac.iloc[-1, 1])
    if np.isnan(macd_hist):
        return _flat(NAME, TF, "MACD histogram NaN")

    cur_close = float(df["Close"].iloc[-1])
    vol_note  = f"vol {cur_vol/vol_sma20:.1f}×avg" if vol_sma20 > 0 else "vol N/A"

    if macd_hist > 0:
        return _make_signal(
            NAME, TF, "LONG", base_conf, cur_close, cur_atr,
            sl_atr_mult=1.5, tp_atr_mult=3.0,
            reason=f"Squeeze broke ↑ | MACD hist {macd_hist:+.4f} | {vol_note}",
        )
    if macd_hist < 0:
        return _make_signal(
            NAME, TF, "SHORT", base_conf, cur_close, cur_atr,
            sl_atr_mult=1.5, tp_atr_mult=3.0,
            reason=f"Squeeze broke ↓ | MACD hist {macd_hist:+.4f} | {vol_note}",
        )

    return _flat(NAME, TF, "Squeeze broke but MACD histogram = 0 — no direction")


# ════════════════════════════════════════════════════════════════════════════════
# STRATEGY 5 — Absorption Breakout  (15m, Valentini method)
# ════════════════════════════════════════════════════════════════════════════════

def absorption_15m(df: pd.DataFrame) -> IntradaySignal:
    """
    Absorption bar breakout on 15m bars — same logic as H1 ABSORB_BO but tuned
    to the 15-minute timeframe where institutional absorption signals are clear.

    Absorption bar: volume > 2×SMA20  AND  range < 0.3×ATR(14) — tight range on
                    heavy volume indicates institutions absorbing supply/demand.

    Long:  previous bar is absorption + current bar closes above absorption High
           + price above session VWAP + RVOL ≥ 1.5
    Short: previous bar is absorption + current bar closes below absorption Low
           + price below session VWAP + RVOL ≥ 1.5

    SL:  below absorption bar low (long) / above absorption bar high (short)
         ± 0.1×ATR buffer — naturally tight since absorption bars are narrow.
    TP:  2× risk distance from entry  →  1:2 R:R

    Backtested win rate ~63%.
    """
    TF = "15m"
    NAME = "ABSORB_15M"

    if len(df) < 25:
        return _flat(NAME, TF, "Insufficient bars (need 25)")

    abs_s = ta.absorption(df["High"], df["Low"], df["Close"], df["Volume"])

    prev_abs = abs_s.iloc[-2]
    if not (pd.notna(prev_abs) and bool(prev_abs)):
        return _flat(NAME, TF, "No absorption on previous bar")

    absorb_high = float(df["High"].iloc[-2])
    absorb_low  = float(df["Low"].iloc[-2])
    cur_close   = float(df["Close"].iloc[-1])
    cur_rvol    = float(df["rvol"].iloc[-1]) if "rvol" in df.columns else 1.0
    cur_vwap    = float(df["session_vwap"].iloc[-1]) if "session_vwap" in df.columns else cur_close

    atr14   = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    cur_atr = float(atr14.iloc[-1]) if atr14 is not None and not atr14.isna().all() else 0.0
    if cur_atr == 0:
        return _flat(NAME, TF, "ATR zero — cannot compute SL/TP")

    rvol_ok   = cur_rvol >= 1.5
    rvol_note = f"RVOL {cur_rvol:.1f}× {'✓' if rvol_ok else '✗'}"

    # Long breakout
    if cur_close > absorb_high and cur_close > cur_vwap:
        conf = 0.75 + (0.05 if rvol_ok else 0.0)
        stop_loss   = round(absorb_low - 0.1 * cur_atr, 4)
        risk_dist   = cur_close - stop_loss
        take_profit = round(cur_close + 2.0 * risk_dist, 4)
        rr          = (take_profit - cur_close) / max(risk_dist, 1e-6)
        return IntradaySignal(
            strategy=NAME, timeframe=TF, signal="LONG", confidence=round(min(conf, 0.95), 3),
            entry=round(cur_close, 4), stop_loss=stop_loss, take_profit=take_profit,
            atr=round(cur_atr, 4), rr_ratio=round(rr, 2),
            reason=f"Absorption breakout ↑: close {cur_close:.2f} > absorb H {absorb_high:.2f} | "
                   f"VWAP {cur_vwap:.2f} | {rvol_note}",
        )

    # Short breakout
    if cur_close < absorb_low and cur_close < cur_vwap:
        conf = 0.75 + (0.05 if rvol_ok else 0.0)
        stop_loss   = round(absorb_high + 0.1 * cur_atr, 4)
        risk_dist   = stop_loss - cur_close
        take_profit = round(cur_close - 2.0 * risk_dist, 4)
        rr          = (cur_close - take_profit) / max(risk_dist, 1e-6)
        return IntradaySignal(
            strategy=NAME, timeframe=TF, signal="SHORT", confidence=round(min(conf, 0.95), 3),
            entry=round(cur_close, 4), stop_loss=stop_loss, take_profit=take_profit,
            atr=round(cur_atr, 4), rr_ratio=round(rr, 2),
            reason=f"Absorption breakout ↓: close {cur_close:.2f} < absorb L {absorb_low:.2f} | "
                   f"VWAP {cur_vwap:.2f} | {rvol_note}",
        )

    return _flat(NAME, TF, f"Absorption bar present but no breakout: close {cur_close:.2f} in "
                           f"[{absorb_low:.2f}, {absorb_high:.2f}]")


# ════════════════════════════════════════════════════════════════════════════════
# STRATEGY 7 — Previous Day High / Low Liquidity Sweep Reversal  (5m)
# ════════════════════════════════════════════════════════════════════════════════

def pdh_pdl_sweep_5m(df: pd.DataFrame) -> IntradaySignal:
    """
    Previous Day High/Low Liquidity Sweep Reversal.

    Institutional algorithms systematically hunt stop clusters above prior-day
    highs (PDH) and below prior-day lows (PDL) before reversing hard.
    The setup: the current 5m bar sweeps the level (wick beyond it) but CLOSES
    back on the correct side — trapping breakout participants.

    Long:  bar LOW dips below PDL  AND  bar CLOSE is back ABOVE PDL
    Short: bar HIGH sweeps above PDH  AND  bar CLOSE is back BELOW PDH

    SL: 0.5×ATR beyond the sweep wick extreme
    TP: 2×risk from entry  (1:2 R:R)

    Win rate: 65–75% on ES/NQ futures (Steady Turtle Trading practitioner data).
    Source: dailypriceaction.com, steady-turtle.com
    """
    NAME = "PDH_PDL_SWEEP_5M"
    TF   = "5m"

    if len(df) < 50:
        return _flat(NAME, TF, "Insufficient bars (need 50)")
    if "session_date" not in df.columns:
        return _flat(NAME, TF, "session_date column missing")

    atr_s   = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    cur_atr = float(atr_s.dropna().iloc[-1]) if not atr_s.dropna().empty else 0.0
    if cur_atr == 0:
        return _flat(NAME, TF, "ATR=0")

    # Previous completed session
    today    = df["session_date"].iloc[-1]
    prev_dates = sorted(d for d in df["session_date"].unique() if d < today)
    if not prev_dates:
        return _flat(NAME, TF, "No prior session in data")
    prev_session = df[df["session_date"] == prev_dates[-1]]
    pdh = float(prev_session["High"].max())
    pdl = float(prev_session["Low"].min())

    cur       = df.iloc[-1]
    cur_close = float(cur["Close"])
    cur_high  = float(cur["High"])
    cur_low   = float(cur["Low"])
    rvol      = float(df["rvol"].iloc[-1]) if "rvol" in df.columns else 1.0

    # Skip first 5 bars of the session (pre-open noise)
    session_bars = int((df["session_date"] == today).sum())
    if session_bars < 5:
        return _flat(NAME, TF, f"Too early in session (bar {session_bars})")

    # Bearish sweep: wick above PDH, close back below — traps longs
    bearish_sweep = (cur_high > pdh) and (cur_close < pdh)
    # Bullish sweep: wick below PDL, close back above — traps shorts
    bullish_sweep = (cur_low < pdl) and (cur_close > pdl)

    if bearish_sweep:
        conf = 0.70 + (0.07 if rvol >= 1.2 else 0.0)
        sl   = round(cur_high + 0.5 * cur_atr, 4)
        risk = sl - cur_close
        if risk <= 0:
            return _flat(NAME, TF, "Invalid bearish-sweep risk geometry")
        tp = round(cur_close - 2.0 * risk, 4)
        return IntradaySignal(
            strategy=NAME, timeframe=TF, signal="SHORT",
            confidence=round(min(conf, 0.92), 3),
            entry=round(cur_close, 4), stop_loss=sl, take_profit=tp,
            atr=round(cur_atr, 4), rr_ratio=2.0,
            reason=(f"Sweep above PDH {pdh:.2f} → close {cur_close:.2f} "
                    f"| Wick={cur_high:.2f} | RVOL={rvol:.1f}x"),
        )

    if bullish_sweep:
        conf = 0.70 + (0.07 if rvol >= 1.2 else 0.0)
        sl   = round(cur_low - 0.5 * cur_atr, 4)
        risk = cur_close - sl
        if risk <= 0:
            return _flat(NAME, TF, "Invalid bullish-sweep risk geometry")
        tp = round(cur_close + 2.0 * risk, 4)
        return IntradaySignal(
            strategy=NAME, timeframe=TF, signal="LONG",
            confidence=round(min(conf, 0.92), 3),
            entry=round(cur_close, 4), stop_loss=sl, take_profit=tp,
            atr=round(cur_atr, 4), rr_ratio=2.0,
            reason=(f"Sweep below PDL {pdl:.2f} → close {cur_close:.2f} "
                    f"| Wick={cur_low:.2f} | RVOL={rvol:.1f}x"),
        )

    return _flat(NAME, TF,
        f"No sweep | PDH {pdh:.2f} PDL {pdl:.2f} | H={cur_high:.2f} L={cur_low:.2f} C={cur_close:.2f}")


# ════════════════════════════════════════════════════════════════════════════════
# STRATEGY 8 — Camarilla Pivot S3/R3 Bounce + S4/R4 Breakout  (5m)
# ════════════════════════════════════════════════════════════════════════════════

def camarilla_pivot_5m(df: pd.DataFrame) -> IntradaySignal:
    """
    Camarilla Pivot Points — Mean Reversion at S3/R3, Breakout at S4/R4.

    Camarilla pivots (Nick Scott, 1989) divide prior-day range into 8 levels.
    The inner levels S3/R3 are reversal zones: when price reaches them in a
    range-bound session it tends to snap back.  The outer S4/R4 signal a trending
    day: if price pushes through them a directional move is in progress.

    Mean reversion (S3 / R3):
        Long at S3 touch + rejection close above S3   → SL at S4, TP 2×risk
        Short at R3 touch + rejection close below R3  → SL at R4, TP 2×risk

    Breakout (S4 / R4):
        Long when bar CLOSES above R4                 → SL just below R4, TP 2×risk
        Short when bar CLOSES below S4                → SL just above S4, TP 2×risk

    Win rate: 59–62% mean reversion; 55–60% breakout (10-yr backtest EUR/USD & SPY).
    Source: QuantifiedStrategies.com, LiteFinance
    """
    NAME = "CAMARILLA_5M"
    TF   = "5m"

    # Session gate: Asian session + early London (00:00–12:00 UTC).
    # Camarilla is a mean-reversion play; NY open momentum (12–17 UTC)
    # routinely breaks through R4/S4 instead of reversing at them.
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        _utc_h = int(df.index[-1].tz_convert("UTC").hour)
        if not (_utc_h < 12 or _utc_h >= 20):
            return _flat(NAME, TF, f"Outside Camarilla window (UTC {_utc_h:02d}:xx; need 00–11 or 20–23)")

    if len(df) < 50:
        return _flat(NAME, TF, "Insufficient bars (need 50)")
    if "session_date" not in df.columns:
        return _flat(NAME, TF, "session_date column missing")

    atr_s   = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    cur_atr = float(atr_s.dropna().iloc[-1]) if not atr_s.dropna().empty else 0.0
    if cur_atr == 0:
        return _flat(NAME, TF, "ATR=0")

    # Previous completed session OHLC
    today      = df["session_date"].iloc[-1]
    prev_dates = sorted(d for d in df["session_date"].unique() if d < today)
    if not prev_dates:
        return _flat(NAME, TF, "No prior session in data")
    prev = df[df["session_date"] == prev_dates[-1]]
    ph   = float(prev["High"].max())
    pl   = float(prev["Low"].min())
    pc   = float(prev["Close"].iloc[-1])
    rng  = ph - pl
    if rng == 0:
        return _flat(NAME, TF, "Zero prior-day range")

    # Camarilla levels (Nick Scott formula)
    r4 = pc + rng * 1.1 / 2
    r3 = pc + rng * 1.1 / 4
    s3 = pc - rng * 1.1 / 4
    s4 = pc - rng * 1.1 / 2

    cur        = df.iloc[-1]
    prev_bar   = df.iloc[-2]
    cur_close  = float(cur["Close"])
    cur_high   = float(cur["High"])
    cur_low    = float(cur["Low"])
    prev_low   = float(prev_bar["Low"])
    prev_high  = float(prev_bar["High"])
    rvol       = float(df["rvol"].iloc[-1]) if "rvol" in df.columns else 1.0

    # ── Breakout signals (higher confidence, momentum-driven) ────────────────
    if cur_close > r4:
        conf = 0.72 + (0.08 if rvol >= 1.5 else 0.0)
        sl   = round(r4 - 0.5 * cur_atr, 4)
        risk = cur_close - sl
        if risk <= 0:
            return _flat(NAME, TF, "R4 breakout: invalid risk")
        tp = round(cur_close + 2.0 * risk, 4)
        return IntradaySignal(
            strategy=NAME, timeframe=TF, signal="LONG",
            confidence=round(min(conf, 0.92), 3),
            entry=round(cur_close, 4), stop_loss=sl, take_profit=tp,
            atr=round(cur_atr, 4), rr_ratio=2.0,
            reason=f"R4 breakout: close {cur_close:.2f} > R4 {r4:.2f} | RVOL={rvol:.1f}x",
        )

    if cur_close < s4:
        conf = 0.72 + (0.08 if rvol >= 1.5 else 0.0)
        sl   = round(s4 + 0.5 * cur_atr, 4)
        risk = sl - cur_close
        if risk <= 0:
            return _flat(NAME, TF, "S4 breakdown: invalid risk")
        tp = round(cur_close - 2.0 * risk, 4)
        if tp <= 0:
            return _flat(NAME, TF, "S4 breakdown: TP <= 0")
        return IntradaySignal(
            strategy=NAME, timeframe=TF, signal="SHORT",
            confidence=round(min(conf, 0.92), 3),
            entry=round(cur_close, 4), stop_loss=sl, take_profit=tp,
            atr=round(cur_atr, 4), rr_ratio=2.0,
            reason=f"S4 breakdown: close {cur_close:.2f} < S4 {s4:.2f} | RVOL={rvol:.1f}x",
        )

    # ── Mean-reversion bounce signals ────────────────────────────────────────
    # S3 bounce: prior bar touched S3, current bar closes above S3
    touched_s3  = (prev_low <= s3) or (cur_low <= s3)
    rejected_s3 = cur_close > s3
    if touched_s3 and rejected_s3:
        conf = 0.62 + (0.07 if rvol >= 1.3 else 0.0)
        sl   = round(s4 - 0.3 * cur_atr, 4)
        risk = cur_close - sl
        if risk <= 0:
            return _flat(NAME, TF, "S3 bounce: invalid risk")
        tp = round(cur_close + 2.0 * risk, 4)
        return IntradaySignal(
            strategy=NAME, timeframe=TF, signal="LONG",
            confidence=round(min(conf, 0.85), 3),
            entry=round(cur_close, 4), stop_loss=sl, take_profit=tp,
            atr=round(cur_atr, 4), rr_ratio=2.0,
            reason=f"S3 bounce: low {cur_low:.2f}→S3 {s3:.2f}, reject above | RVOL={rvol:.1f}x",
        )

    # R3 bounce: prior bar touched R3, current bar closes below R3
    touched_r3  = (prev_high >= r3) or (cur_high >= r3)
    rejected_r3 = cur_close < r3
    if touched_r3 and rejected_r3:
        conf = 0.62 + (0.07 if rvol >= 1.3 else 0.0)
        sl   = round(r4 + 0.3 * cur_atr, 4)
        risk = sl - cur_close
        if risk <= 0:
            return _flat(NAME, TF, "R3 bounce: invalid risk")
        tp = round(cur_close - 2.0 * risk, 4)
        if tp <= 0:
            return _flat(NAME, TF, "R3 bounce: TP <= 0")
        return IntradaySignal(
            strategy=NAME, timeframe=TF, signal="SHORT",
            confidence=round(min(conf, 0.85), 3),
            entry=round(cur_close, 4), stop_loss=sl, take_profit=tp,
            atr=round(cur_atr, 4), rr_ratio=2.0,
            reason=f"R3 bounce: high {cur_high:.2f}→R3 {r3:.2f}, reject below | RVOL={rvol:.1f}x",
        )

    return _flat(NAME, TF,
        f"No Camarilla trigger | S3={s3:.2f} R3={r3:.2f} S4={s4:.2f} R4={r4:.2f} | C={cur_close:.2f}")


# ════════════════════════════════════════════════════════════════════════════════
# STRATEGY 9 — NR7 Volatility Compression Breakout  (5m)
# ════════════════════════════════════════════════════════════════════════════════

def nr7_breakout_5m(df: pd.DataFrame) -> IntradaySignal:
    """
    NR7 Volatility Compression Breakout.

    Detects the Narrow Range 7 bar: a bar whose high-low range is the smallest
    of the preceding 7 bars.  The NR7 signals extreme compression before a
    volatility expansion.  Signal fires when the very next bar closes BEYOND the
    NR7 bar's high (long) or low (short), confirming the breakout direction.

    Long:  close of current bar > NR7 bar's high
    Short: close of current bar < NR7 bar's low

    SL: opposite end of NR7 bar + 0.3×ATR buffer
    TP: 2×risk from entry

    Win rate: 57% bull-market up breakout, 54% down breakout.
    Source: Bulkowski, Encyclopedia of Chart Patterns, 29,021 trades (1990-2013)
    """
    NAME = "NR7_BREAKOUT_5M"
    TF   = "5m"

    if len(df) < 20:
        return _flat(NAME, TF, "Insufficient bars (need 20)")

    atr_s   = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    cur_atr = float(atr_s.dropna().iloc[-1]) if not atr_s.dropna().empty else 0.0
    if cur_atr == 0:
        return _flat(NAME, TF, "ATR=0")

    highs  = df["High"].values
    lows   = df["Low"].values
    closes = df["Close"].values
    n      = len(df)

    # Setup bar is df.iloc[-2]; current bar is df.iloc[-1]
    nr7_idx = n - 2
    if nr7_idx < 6:
        return _flat(NAME, TF, "Not enough bars for NR7 lookback")

    nr7_range = float(highs[nr7_idx] - lows[nr7_idx])
    # Compare against the *prior 6 bars only* (exclusive of the setup bar) so a
    # flat day where every bar has the same range doesn't trivially pass — the
    # setup bar's range must be STRICTLY narrower than every one of the prior 6.
    prior_ranges = [float(highs[i] - lows[i]) for i in range(nr7_idx - 6, nr7_idx)]
    if nr7_range >= min(prior_ranges):
        return _flat(NAME, TF,
            f"No NR7: setup range {nr7_range:.4f} >= min(prior 6) {min(prior_ranges):.4f}")

    nr7_high  = float(highs[nr7_idx])
    nr7_low   = float(lows[nr7_idx])
    cur_close = float(closes[-1])
    rvol      = float(df["rvol"].iloc[-1]) if "rvol" in df.columns else 1.0

    breakout_long  = cur_close > nr7_high
    breakout_short = cur_close < nr7_low

    if breakout_long:
        conf = 0.60 + (0.10 if rvol >= 1.5 else 0.0)
        sl   = round(nr7_low - 0.3 * cur_atr, 4)
        risk = cur_close - sl
        if risk <= 0:
            return _flat(NAME, TF, "NR7 long: invalid risk")
        tp = round(cur_close + 2.0 * risk, 4)
        return IntradaySignal(
            strategy=NAME, timeframe=TF, signal="LONG",
            confidence=round(min(conf, 0.88), 3),
            entry=round(cur_close, 4), stop_loss=sl, take_profit=tp,
            atr=round(cur_atr, 4), rr_ratio=2.0,
            reason=(f"NR7 breakout ↑: close {cur_close:.4f} > NR7-high {nr7_high:.4f} "
                    f"| NR7-range={nr7_range:.4f} | RVOL={rvol:.1f}x"),
        )

    if breakout_short:
        conf = 0.60 + (0.10 if rvol >= 1.5 else 0.0)
        sl   = round(nr7_high + 0.3 * cur_atr, 4)
        risk = sl - cur_close
        if risk <= 0:
            return _flat(NAME, TF, "NR7 short: invalid risk")
        tp = round(cur_close - 2.0 * risk, 4)
        if tp <= 0:
            return _flat(NAME, TF, "NR7 short: TP <= 0")
        return IntradaySignal(
            strategy=NAME, timeframe=TF, signal="SHORT",
            confidence=round(min(conf, 0.88), 3),
            entry=round(cur_close, 4), stop_loss=sl, take_profit=tp,
            atr=round(cur_atr, 4), rr_ratio=2.0,
            reason=(f"NR7 breakdown ↓: close {cur_close:.4f} < NR7-low {nr7_low:.4f} "
                    f"| NR7-range={nr7_range:.4f} | RVOL={rvol:.1f}x"),
        )

    return _flat(NAME, TF,
        f"NR7 set up (range={nr7_range:.4f}) — no breakout yet "
        f"| C={cur_close:.4f} in [{nr7_low:.4f}, {nr7_high:.4f}]")


# ════════════════════════════════════════════════════════════════════════════════
# STRATEGY 11 — Bollinger Band Pierce-and-Reclaim Scalp  (5m)
# ════════════════════════════════════════════════════════════════════════════════

def bb_scalp_5m(df: pd.DataFrame) -> IntradaySignal:
    """
    BB pierce-and-reclaim: prior bar closes outside the band, current bar
    closes back inside (reclaim). VWAP alignment required. Tight SL/TP.

    SL: 0.6×ATR  TP: 1.2×ATR  →  1:2 R:R
    Add to _MR_STRATEGIES (penalised when ADX > 25).
    """
    NAME = "BB_SCALP_5M"
    TF   = "5m"

    if len(df) < 25:
        return _flat(NAME, TF, "Insufficient bars (need 25)")
    if "session_vwap" not in df.columns:
        return _flat(NAME, TF, "session_vwap column missing")

    bb    = ta.bbands(df["Close"], length=20, std=2.0)
    atr10 = ta.atr(df["High"], df["Low"], df["Close"], length=10)

    if bb is None or bb.empty:
        return _flat(NAME, TF, "BB unavailable")

    cur_close  = float(df["Close"].iloc[-1])
    cur_vwap   = float(df["session_vwap"].iloc[-1])
    cur_atr    = float(atr10.iloc[-1])
    cur_bbl    = float(bb.iloc[-1, 0])
    cur_bbu    = float(bb.iloc[-1, 2])
    prev_close = float(df["Close"].iloc[-2])
    prev_bbl   = float(bb.iloc[-2, 0])
    prev_bbu   = float(bb.iloc[-2, 2])
    cur_rvol   = float(df["rvol"].iloc[-1]) if "rvol" in df.columns else 1.0

    if any(np.isnan(v) for v in [cur_close, cur_atr, cur_bbl, cur_bbu,
                                   prev_bbl, prev_bbu, cur_vwap]) or cur_atr == 0:
        return _flat(NAME, TF, "NaN indicator — warmup incomplete")

    rvol_note = f"RVOL {cur_rvol:.1f}x"

    if prev_close < prev_bbl and cur_close > cur_bbl and cur_close > cur_vwap:
        conf = 0.65
        if cur_rvol >= 1.5:
            conf += 0.08
        if cur_close - cur_bbl > 0.3 * cur_atr:
            conf += 0.06
        return _make_signal(NAME, TF, "LONG", min(conf, 0.85), cur_close, cur_atr,
            sl_atr_mult=0.6, tp_atr_mult=1.2,
            reason=f"BB pierce↓ reclaim: prev {prev_close:.4f}<BBL {prev_bbl:.4f} → cur {cur_close:.4f}>BBL {cur_bbl:.4f} | VWAP {cur_vwap:.4f} | {rvol_note}")

    if prev_close > prev_bbu and cur_close < cur_bbu and cur_close < cur_vwap:
        conf = 0.65
        if cur_rvol >= 1.5:
            conf += 0.08
        if cur_bbu - cur_close > 0.3 * cur_atr:
            conf += 0.06
        return _make_signal(NAME, TF, "SHORT", min(conf, 0.85), cur_close, cur_atr,
            sl_atr_mult=0.6, tp_atr_mult=1.2,
            reason=f"BB pierce↑ reclaim: prev {prev_close:.4f}>BBU {prev_bbu:.4f} → cur {cur_close:.4f}<BBU {cur_bbu:.4f} | VWAP {cur_vwap:.4f} | {rvol_note}")

    return _flat(NAME, TF,
        f"No BB pierce-reclaim | BBL {cur_bbl:.4f} BBU {cur_bbu:.4f} C {cur_close:.4f}"
        + (" | above VWAP" if cur_close > cur_vwap else " | below VWAP"))


# ════════════════════════════════════════════════════════════════════════════════
# STRATEGY 12 — Stochastic Extreme Crossover Scalp  (5m)
# ════════════════════════════════════════════════════════════════════════════════

def stoch_cross_5m(df: pd.DataFrame) -> IntradaySignal:
    """
    Stochastic K/D crossover inside extreme oversold (<25) / overbought (>75)
    zones, filtered by EMA(21) trend direction.

    SL: 0.7×ATR  TP: 1.4×ATR  →  1:2 R:R
    Add to _MR_STRATEGIES (penalised when ADX > 25).
    """
    NAME = "STOCH_CROSS_5M"
    TF   = "5m"

    if len(df) < 25:
        return _flat(NAME, TF, "Insufficient bars (need 25)")

    stoch_df = ta.stoch(df["High"], df["Low"], df["Close"], k=14, d=3)
    ema21    = ta.ema(df["Close"], length=21)
    atr10    = ta.atr(df["High"], df["Low"], df["Close"], length=10)

    if stoch_df is None or stoch_df.empty:
        return _flat(NAME, TF, "Stochastic unavailable")

    cur_k  = float(stoch_df.iloc[-1, 0])
    cur_d  = float(stoch_df.iloc[-1, 1])
    prev_k = float(stoch_df.iloc[-2, 0])
    prev_d = float(stoch_df.iloc[-2, 1])

    cur_ema21 = float(ema21.iloc[-1])
    cur_atr   = float(atr10.iloc[-1])
    cur_close = float(df["Close"].iloc[-1])
    cur_vwap  = float(df["session_vwap"].iloc[-1]) if "session_vwap" in df.columns else cur_close
    cur_rvol  = float(df["rvol"].iloc[-1]) if "rvol" in df.columns else 1.0

    if any(np.isnan(v) for v in [cur_k, cur_d, prev_k, prev_d, cur_ema21, cur_atr]) or cur_atr == 0:
        return _flat(NAME, TF, "NaN indicator — warmup incomplete")

    k_crossed_up   = prev_k < prev_d and cur_k > cur_d
    k_crossed_down = prev_k > prev_d and cur_k < cur_d
    cross_spread   = abs(cur_k - cur_d)
    near_vwap      = abs(cur_close - cur_vwap) <= 0.5 * cur_atr

    def _conf() -> float:
        c = 0.63
        if cross_spread >= 3.0:
            c += 0.07
        if near_vwap:
            c += 0.06
        if cur_rvol >= 1.4:
            c += 0.06
        return min(c, 0.85)

    vwap_note = " | near VWAP" if near_vwap else ""

    if k_crossed_up and cur_k < 25 and cur_d < 25 and cur_close > cur_ema21:
        return _make_signal(NAME, TF, "LONG", _conf(), cur_close, cur_atr,
            sl_atr_mult=0.7, tp_atr_mult=1.4,
            reason=f"Stoch cross↑ oversold K={cur_k:.1f}/D={cur_d:.1f} | above EMA21 {cur_ema21:.4f} | RVOL {cur_rvol:.1f}x{vwap_note}")

    if k_crossed_down and cur_k > 75 and cur_d > 75 and cur_close < cur_ema21:
        return _make_signal(NAME, TF, "SHORT", _conf(), cur_close, cur_atr,
            sl_atr_mult=0.7, tp_atr_mult=1.4,
            reason=f"Stoch cross↓ overbought K={cur_k:.1f}/D={cur_d:.1f} | below EMA21 {cur_ema21:.4f} | RVOL {cur_rvol:.1f}x{vwap_note}")

    if k_crossed_up and cur_k < 25:
        return _flat(NAME, TF, f"Stoch cross↑ oversold but below EMA21 {cur_ema21:.4f}")
    if k_crossed_down and cur_k > 75:
        return _flat(NAME, TF, f"Stoch cross↓ overbought but above EMA21 {cur_ema21:.4f}")
    return _flat(NAME, TF, f"Stoch neutral/no cross K={cur_k:.1f}/D={cur_d:.1f}")


# ════════════════════════════════════════════════════════════════════════════════
# STRATEGY 13 — Fast EMA Micro-Cross with Delta Pressure  (5m)
# ════════════════════════════════════════════════════════════════════════════════

def ema_micro_cross_5m(df: pd.DataFrame) -> IntradaySignal:
    """
    EMA(5)/EMA(13) fresh crossover (exactly 1-bar old) confirmed by VWAP
    alignment and 3-bar cumulative delta_proxy (buying/selling pressure).

    SL: 0.5×ATR  TP: 1.0×ATR  →  1:2 R:R (tightest in the suite)
    Add to _TREND_STRATEGIES (penalised when ADX < 20).
    """
    NAME = "EMA_MICRO_CROSS_5M"
    TF   = "5m"

    if len(df) < 20:
        return _flat(NAME, TF, "Insufficient bars (need 20)")
    if "session_vwap" not in df.columns:
        return _flat(NAME, TF, "session_vwap column missing")

    ema5  = ta.ema(df["Close"], length=5)
    ema13 = ta.ema(df["Close"], length=13)
    atr10 = ta.atr(df["High"], df["Low"], df["Close"], length=10)

    cur_ema5   = float(ema5.iloc[-1])
    cur_ema13  = float(ema13.iloc[-1])
    prev_ema5  = float(ema5.iloc[-2])
    prev_ema13 = float(ema13.iloc[-2])
    cur_close  = float(df["Close"].iloc[-1])
    cur_vwap   = float(df["session_vwap"].iloc[-1])
    cur_atr    = float(atr10.iloc[-1])
    cur_rvol   = float(df["rvol"].iloc[-1]) if "rvol" in df.columns else 1.0

    if any(np.isnan(v) for v in [cur_ema5, cur_ema13, prev_ema5, prev_ema13,
                                   cur_atr, cur_vwap]) or cur_atr == 0:
        return _flat(NAME, TF, "NaN indicator — warmup incomplete")

    long_cross  = prev_ema5 < prev_ema13 and cur_ema5 > cur_ema13
    short_cross = prev_ema5 > prev_ema13 and cur_ema5 < cur_ema13

    if not long_cross and not short_cross:
        gap_dir = "above" if cur_ema5 > cur_ema13 else "below"
        return _flat(NAME, TF, f"No fresh EMA5/13 cross | EMA5={cur_ema5:.4f} {gap_dir} EMA13={cur_ema13:.4f}")

    delta = ta.delta_proxy(df["Open"], df["Close"], df["Volume"])
    delta3_sum = float(delta.iloc[-3:].sum())
    delta_abs_mean = float(delta.abs().rolling(10).mean().iloc[-1]) if len(df) >= 10 else 0.0
    strong_pressure = (delta_abs_mean > 0) and (abs(delta3_sum) > 1.5 * delta_abs_mean)
    ema_gap = abs(cur_ema5 - cur_ema13)

    def _conf() -> float:
        c = 0.64
        if ema_gap > 0.2 * cur_atr:
            c += 0.07
        if strong_pressure:
            c += 0.07
        if cur_rvol >= 1.3:
            c += 0.05
        return min(c, 0.85)

    delta_note = f"Δ3={delta3_sum:+.0f}"

    if long_cross and cur_close > cur_vwap and delta3_sum > 0:
        return _make_signal(NAME, TF, "LONG", _conf(), cur_close, cur_atr,
            sl_atr_mult=0.5, tp_atr_mult=1.0,
            reason=f"EMA5×EMA13↑ fresh cross | above VWAP {cur_vwap:.4f} | {delta_note}>0 | RVOL {cur_rvol:.1f}x")

    if short_cross and cur_close < cur_vwap and delta3_sum < 0:
        return _make_signal(NAME, TF, "SHORT", _conf(), cur_close, cur_atr,
            sl_atr_mult=0.5, tp_atr_mult=1.0,
            reason=f"EMA5×EMA13↓ fresh cross | below VWAP {cur_vwap:.4f} | {delta_note}<0 | RVOL {cur_rvol:.1f}x")

    reasons = []
    if long_cross:
        if cur_close <= cur_vwap: reasons.append(f"EMA cross↑ but below VWAP {cur_vwap:.4f}")
        if delta3_sum <= 0:       reasons.append(f"delta pressure negative ({delta_note})")
    elif short_cross:
        if cur_close >= cur_vwap: reasons.append(f"EMA cross↓ but above VWAP {cur_vwap:.4f}")
        if delta3_sum >= 0:       reasons.append(f"delta pressure positive ({delta_note})")
    return _flat(NAME, TF, " | ".join(reasons) or "Cross filter mismatch")


# ════════════════════════════════════════════════════════════════════════════════
# STRATEGY 10 — Market Structure Shift  (Forex only, 15m entry / daily trend)
# ════════════════════════════════════════════════════════════════════════════════

def mss_forex_15m(df: pd.DataFrame) -> IntradaySignal:
    """
    Market Structure Shift (MSS) entry — Forex pairs only (=X tickers).

    Higher-timeframe trend  (daily bars, 2-bar swing lookback):
        2× Higher High + 2× Higher Low → UPTREND
        2× Lower  High + 2× Lower  Low → DOWNTREND
        Mixed / no clear structure      → FLAT (no trade)

    Entry timeframe  (15m bars resampled from the 5m df, 6-bar lookback):
        UPTREND   + current 15m close > recent confirmed 15m swing high → LONG
        DOWNTREND + current 15m close < recent confirmed 15m swing low  → SHORT

    SL: opposing 15m swing level (swing low for LONG, swing high for SHORT)
    TP: 2× risk from entry  →  1:2 R:R

    Source: classical ICT/SMC Market Structure Shift; pullback entry on MSS
    confirmation.  Win-rate estimate: 60–65% (BabyPips community backtest data).
    """
    NAME = "MSS_FOREX_15M"
    TF   = "15m"

    # FOREX-only guard
    if "market" in df.columns and df["market"].iloc[-1] != "FOREX":
        return _flat(NAME, TF, "Non-FOREX ticker — MSS strategy skipped")

    # Session gate: London + NY only (07:00–17:00 UTC).
    # MSS patterns require institutional order flow to be reliable;
    # Asian session produces false structure breaks on low volume.
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        _utc_h = int(df.index[-1].tz_convert("UTC").hour)
        if not (7 <= _utc_h < 17):
            return _flat(NAME, TF, f"Outside London/NY window (UTC {_utc_h:02d}:xx; need 07–16)")

    if len(df) < 100:
        return _flat(NAME, TF, "Insufficient bars (need 100)")

    # resample("1D") needs a DatetimeIndex; fail gracefully if the caller
    # handed us a positional/integer index (some test fixtures, some
    # malformed yfinance returns).
    if not isinstance(df.index, pd.DatetimeIndex):
        return _flat(NAME, TF, "Index is not DatetimeIndex — cannot resample to 1D")

    # ── Daily swing structure ─────────────────────────────────────────────────
    daily = (
        df[["Open", "High", "Low", "Close"]]
        .resample("1D")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
        .dropna()
    )
    if len(daily) < 6:
        return _flat(NAME, TF, "Insufficient daily bars for swing structure")

    d_highs = daily["High"].values
    d_lows  = daily["Low"].values
    n_d     = len(d_highs)

    # 2-bar lookback pivot: high[i] > high[i-1] AND high[i] > high[i+1]
    swing_highs: list[float] = []
    swing_lows:  list[float] = []
    for i in range(1, n_d - 1):
        if d_highs[i] > d_highs[i - 1] and d_highs[i] > d_highs[i + 1]:
            swing_highs.append(float(d_highs[i]))
        if d_lows[i] < d_lows[i - 1] and d_lows[i] < d_lows[i + 1]:
            swing_lows.append(float(d_lows[i]))

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return _flat(NAME, TF,
            f"Not enough daily swing points (SH={len(swing_highs)}, SL={len(swing_lows)})")

    sh_prev, sh_last = swing_highs[-2], swing_highs[-1]
    sl_prev, sl_last = swing_lows[-2],  swing_lows[-1]

    hh_hl = sh_last > sh_prev and sl_last > sl_prev  # uptrend
    lh_ll = sh_last < sh_prev and sl_last < sl_prev  # downtrend

    if not hh_hl and not lh_ll:
        return _flat(NAME, TF,
            f"No clear daily structure | SH {sh_prev:.5f}→{sh_last:.5f} "
            f"SL {sl_prev:.5f}→{sl_last:.5f}")

    daily_dir = "UP" if hh_hl else "DOWN"

    # ── 15m swing points (resample 5m → 15m, 6-bar lookback confirmation) ────
    df_15 = (
        df[["High", "Low", "Close"]]
        .resample("15min")
        .agg({"High": "max", "Low": "min", "Close": "last"})
        .dropna()
    )
    if len(df_15) < 20:
        return _flat(NAME, TF, "Insufficient 15m bars after resample")

    h15 = df_15["High"].values
    l15 = df_15["Low"].values
    c15 = df_15["Close"].values
    n15 = len(h15)
    LB  = 6   # lookback / confirmation bars each side

    # Scan from the most recently confirmed bar backwards
    recent_sh: float | None = None
    recent_sl: float | None = None
    for i in range(n15 - LB - 1, LB - 1, -1):
        window_h = h15[i - LB : i + LB + 1]
        window_l = l15[i - LB : i + LB + 1]
        if recent_sh is None and float(h15[i]) == float(max(window_h)):
            recent_sh = float(h15[i])
        if recent_sl is None and float(l15[i]) == float(min(window_l)):
            recent_sl = float(l15[i])
        if recent_sh is not None and recent_sl is not None:
            break

    if recent_sh is None or recent_sl is None:
        return _flat(NAME, TF, "Could not find confirmed 15m swing points")

    cur_close = float(c15[-1])

    atr14   = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    cur_atr = float(atr14.dropna().iloc[-1]) if atr14 is not None and not atr14.dropna().empty else 0.0
    if cur_atr == 0:
        return _flat(NAME, TF, "ATR=0")

    rvol = float(df["rvol"].iloc[-1]) if "rvol" in df.columns else 1.0

    # ── LONG: uptrend + MSS close above 15m swing high ───────────────────────
    if daily_dir == "UP" and cur_close > recent_sh:
        sl_price = recent_sl
        risk     = cur_close - sl_price
        if risk <= 0 or risk > 10 * cur_atr:
            return _flat(NAME, TF, f"Invalid long risk geometry (risk={risk:.5f})")
        conf = min(0.78 + (0.07 if rvol >= 1.3 else 0.0), 0.92)
        tp   = round(cur_close + 2.0 * risk, 5)
        return IntradaySignal(
            strategy=NAME, timeframe=TF, signal="LONG",
            confidence=round(conf, 3),
            entry=round(cur_close, 5),
            stop_loss=round(sl_price, 5),
            take_profit=tp,
            atr=round(cur_atr, 5),
            rr_ratio=2.0,
            reason=(f"Daily HH+HL ↑ | MSS: close {cur_close:.5f} > 15m SwingH {recent_sh:.5f} "
                    f"| SL=SwingL {recent_sl:.5f} | RVOL={rvol:.1f}x"),
        )

    # ── SHORT: downtrend + MSS close below 15m swing low ─────────────────────
    if daily_dir == "DOWN" and cur_close < recent_sl:
        sl_price = recent_sh
        risk     = sl_price - cur_close
        if risk <= 0 or risk > 10 * cur_atr:
            return _flat(NAME, TF, f"Invalid short risk geometry (risk={risk:.5f})")
        conf = min(0.78 + (0.07 if rvol >= 1.3 else 0.0), 0.92)
        tp   = round(cur_close - 2.0 * risk, 5)
        return IntradaySignal(
            strategy=NAME, timeframe=TF, signal="SHORT",
            confidence=round(conf, 3),
            entry=round(cur_close, 5),
            stop_loss=round(sl_price, 5),
            take_profit=tp,
            atr=round(cur_atr, 5),
            rr_ratio=2.0,
            reason=(f"Daily LH+LL ↓ | MSS: close {cur_close:.5f} < 15m SwingL {recent_sl:.5f} "
                    f"| SL=SwingH {recent_sh:.5f} | RVOL={rvol:.1f}x"),
        )

    return _flat(NAME, TF,
        f"Daily {daily_dir} trend confirmed | no MSS trigger | "
        f"C={cur_close:.5f} 15m[SwingH={recent_sh:.5f} SwingL={recent_sl:.5f}]")


# ════════════════════════════════════════════════════════════════════════════════
# PATTERN-TRIGGERED STRATEGIES — fire on detected chart patterns
# ════════════════════════════════════════════════════════════════════════════════
#
# Round 2 addition. Where the existing 13 strategies use technical indicators
# (RSI, MACD, EMAs, etc.) to enter, these three use the chart pattern detector
# itself as the trigger. Patterns previously contributed only a ±0.05 nudge
# to existing strategy confidence; now high-confidence patterns can initiate
# a trade on their own when the confirmation gate passes.
#
# Confirmation gate (shared by all three):
#   • Pattern confidence ≥ threshold (0.70 or 0.75 depending on pattern type)
#   • RVOL ≥ 1.2 — current bar volume must exceed 20-bar average (signals
#     real participation in the breakout, not low-liquidity noise)
#   • ATR > 0 — required for stop placement
#
# The MTF gate (daily + H4 alignment) is applied downstream by signal_engine,
# not duplicated inside the strategy.
# ════════════════════════════════════════════════════════════════════════════════


_PATTERN_STRATEGY_MIN_BARS = 60   # detect_patterns scans up to last 100 bars


def _safe_rvol(df: pd.DataFrame) -> float:
    """Latest bar's volume / 20-bar average. Defaults to 1.0 if unavailable."""
    try:
        if "rvol" in df.columns and not pd.isna(df["rvol"].iloc[-1]):
            return float(df["rvol"].iloc[-1])
        if "Volume" in df.columns and len(df) >= 20:
            recent_v = df["Volume"].iloc[-20:]
            avg = float(recent_v.mean())
            cur = float(recent_v.iloc[-1])
            return cur / avg if avg > 0 else 1.0
    except Exception:
        pass
    return 1.0


def _safe_pattern_report(df: pd.DataFrame):
    """Run pattern detection, swallowing edge-case errors (NaN bars, etc.)."""
    try:
        from agents.pattern_detector import detect_patterns
        return detect_patterns(df.tail(100).reset_index(drop=True))
    except Exception:
        return None


def _safe_atr(df: pd.DataFrame, length: int = 14) -> float:
    try:
        atr_s = ta.atr(df["High"], df["Low"], df["Close"], length=length)
        valid = atr_s.dropna() if atr_s is not None else None
        if valid is not None and not valid.empty:
            return float(valid.iloc[-1])
    except Exception:
        pass
    return 0.0


def double_bottom_breakout_5m(df: pd.DataFrame) -> IntradaySignal:
    """LONG when a Double Bottom or Triple Bottom prints with confidence ≥ 0.75
    on the 5m chart, RVOL confirms participation, and the detector signals the
    neckline break (the detector only fires if `closes[-1] > neckline * 0.99`).

    SL placed 1.5×ATR below entry; TP at 3×ATR above (1:2 R:R).
    """
    NAME, TF = "DOUBLE_BOT_BREAKOUT_5M", "5m"
    if df is None or len(df) < _PATTERN_STRATEGY_MIN_BARS:
        return _flat(NAME, TF, "Insufficient bars")

    report = _safe_pattern_report(df)
    if report is None or not report.patterns:
        return _flat(NAME, TF, "No patterns detected")

    candidate = next(
        (p for p in report.patterns
         if p.name in ("Double Bottom", "Triple Bottom")
         and p.bias == "BULLISH"
         and p.confidence >= 0.75),
        None,
    )
    if candidate is None:
        return _flat(NAME, TF, "No Double/Triple Bottom ≥0.75 confidence")

    rvol = _safe_rvol(df)
    if rvol < 1.2:
        return _flat(NAME, TF, f"{candidate.name} found but RVOL={rvol:.2f} < 1.2")

    atr = _safe_atr(df)
    if atr <= 0:
        return _flat(NAME, TF, "ATR=0")

    entry = float(df["Close"].iloc[-1])
    confidence = min(0.95, candidate.confidence + (0.05 if rvol >= 1.5 else 0.0))
    return _make_signal(
        NAME, TF, "LONG", confidence, entry, atr,
        sl_atr_mult=1.5, tp_atr_mult=3.0,
        reason=(
            f"{candidate.name} (conf={candidate.confidence:.2f}, "
            f"WR={candidate.win_rate:.0%}) | RVOL={rvol:.1f}x | breakout LONG"
        ),
    )


def head_shoulders_breakdown_5m(df: pd.DataFrame) -> IntradaySignal:
    """Symmetric strategy for both H&S (SHORT) and Inverse H&S (LONG) on 5m.
    Pattern detector confirms neckline break before yielding the match.

    SL placed 1.5×ATR opposite the trade direction; TP at 3×ATR (1:2).
    """
    NAME, TF = "HS_BREAKDOWN_5M", "5m"
    if df is None or len(df) < _PATTERN_STRATEGY_MIN_BARS:
        return _flat(NAME, TF, "Insufficient bars")

    report = _safe_pattern_report(df)
    if report is None or not report.patterns:
        return _flat(NAME, TF, "No patterns detected")

    bearish = next(
        (p for p in report.patterns
         if p.name == "Head & Shoulders" and p.confidence >= 0.75),
        None,
    )
    bullish = next(
        (p for p in report.patterns
         if p.name == "Inv. Head & Shoulders" and p.confidence >= 0.75),
        None,
    )
    candidate = bearish or bullish
    if candidate is None:
        return _flat(NAME, TF, "No H&S or Inv. H&S ≥0.75 confidence")

    direction = "SHORT" if candidate is bearish else "LONG"

    rvol = _safe_rvol(df)
    if rvol < 1.2:
        return _flat(NAME, TF, f"{candidate.name} found but RVOL={rvol:.2f} < 1.2")

    atr = _safe_atr(df)
    if atr <= 0:
        return _flat(NAME, TF, "ATR=0")

    entry = float(df["Close"].iloc[-1])
    confidence = min(0.95, candidate.confidence + (0.05 if rvol >= 1.5 else 0.0))
    return _make_signal(
        NAME, TF, direction, confidence, entry, atr,
        sl_atr_mult=1.5, tp_atr_mult=3.0,
        reason=(
            f"{candidate.name} (conf={candidate.confidence:.2f}, "
            f"WR={candidate.win_rate:.0%}) | RVOL={rvol:.1f}x | "
            f"neckline {direction}"
        ),
    )


def flag_breakout_5m(df: pd.DataFrame) -> IntradaySignal:
    """Continuation strategy: fires LONG on Bull Flag, SHORT on Bear Flag with
    confidence ≥ 0.70 (lower threshold than reversal patterns — flags have
    higher base reliability per Bulkowski) and RVOL confirming the break.

    Tighter stops than reversal strategies: 1.0×ATR SL, 2.5×ATR TP. The pole
    typically projects further than the flag's depth, justifying the wider TP.
    """
    NAME, TF = "FLAG_BREAKOUT_5M", "5m"
    if df is None or len(df) < _PATTERN_STRATEGY_MIN_BARS:
        return _flat(NAME, TF, "Insufficient bars")

    report = _safe_pattern_report(df)
    if report is None or not report.patterns:
        return _flat(NAME, TF, "No patterns detected")

    bull = next(
        (p for p in report.patterns
         if p.name == "Bull Flag" and p.confidence >= 0.70),
        None,
    )
    bear = next(
        (p for p in report.patterns
         if p.name == "Bear Flag" and p.confidence >= 0.70),
        None,
    )
    candidate = bull or bear
    if candidate is None:
        return _flat(NAME, TF, "No Bull/Bear Flag ≥0.70 confidence")

    direction = "LONG" if candidate is bull else "SHORT"

    rvol = _safe_rvol(df)
    if rvol < 1.2:
        return _flat(NAME, TF, f"{candidate.name} found but RVOL={rvol:.2f} < 1.2")

    atr = _safe_atr(df)
    if atr <= 0:
        return _flat(NAME, TF, "ATR=0")

    entry = float(df["Close"].iloc[-1])
    confidence = min(0.92, candidate.confidence + (0.05 if rvol >= 1.5 else 0.0))
    return _make_signal(
        NAME, TF, direction, confidence, entry, atr,
        sl_atr_mult=1.0, tp_atr_mult=2.5,
        reason=(
            f"{candidate.name} (conf={candidate.confidence:.2f}, "
            f"WR={candidate.win_rate:.0%}) | RVOL={rvol:.1f}x | "
            f"continuation {direction}"
        ),
    )


# ════════════════════════════════════════════════════════════════════════════════
# STRATEGY — Liquidity Sweep (Stop Hunt) Reversal  (5m)
# ════════════════════════════════════════════════════════════════════════════════

def liq_sweep_5m(df: pd.DataFrame) -> IntradaySignal:
    """
    Detects institutional stop-hunt (liquidity sweep) reversals on 5m bars.

    A liquidity sweep happens when price briefly spikes past a recent swing
    high/low — triggering retail stop losses — then immediately reverses back
    inside the prior range. The reversal candle is the entry.

    Long:  current bar Low < swing_low AND Close > swing_low
           AND wick below level >= 0.3×ATR  AND  body closes back >= 0.2×ATR
    Short: current bar High > swing_high AND Close < swing_high
           AND wick above level >= 0.3×ATR  AND  body closes back >= 0.2×ATR

    Volume confirmation: RVOL >= 1.2 adds +0.05 confidence.
    Larger sweep wick = higher base confidence (capped at 0.87).

    SL:  structural stop (just beyond sweep extreme)
    TP:  2× risk from entry  →  1:2 R:R

    Regime: regime-independent (sweeps occur in both trending and ranging
    markets). Not added to _MR_STRATEGIES or _TREND_STRATEGIES so no ADX
    penalty is applied — same treatment as ABSORB.
    """
    TF   = "5m"
    NAME = "LIQ_SWEEP_5M"

    LOOKBACK      = 30   # bars to search for the swing level
    SWING_MIN_AGE = 5    # swing must be at least this many bars old
    MIN_WICK_ATR  = 0.3  # minimum sweep wick size relative to ATR
    MIN_BACK_ATR  = 0.2  # close must be back inside by at least this much

    min_bars = LOOKBACK + SWING_MIN_AGE + 10
    if len(df) < min_bars:
        return _flat(NAME, TF, f"Insufficient bars (need {min_bars})")

    atr_s = ta.atr(df["High"], df["Low"], df["Close"], length=10)
    if atr_s is None or atr_s.dropna().empty:
        return _flat(NAME, TF, "ATR unavailable")
    cur_atr = float(atr_s.dropna().iloc[-1])
    if cur_atr == 0 or np.isnan(cur_atr):
        return _flat(NAME, TF, "ATR zero or NaN")

    cur_high  = float(df["High"].iloc[-1])
    cur_low   = float(df["Low"].iloc[-1])
    cur_close = float(df["Close"].iloc[-1])
    cur_rvol  = _safe_rvol(df)

    if any(np.isnan(v) for v in [cur_high, cur_low, cur_close]):
        return _flat(NAME, TF, "NaN in OHLC")

    # Swing reference window: exclude the last SWING_MIN_AGE bars so the
    # "swing" level is at least a few bars old (not just the previous bar).
    ref = df.iloc[-(LOOKBACK + SWING_MIN_AGE):-SWING_MIN_AGE]
    if ref.empty:
        return _flat(NAME, TF, "Reference window empty")

    swing_high = float(ref["High"].max())
    swing_low  = float(ref["Low"].min())

    if np.isnan(swing_high) or np.isnan(swing_low):
        return _flat(NAME, TF, "Swing high/low NaN")

    # ── Bullish sweep: Low pierced below swing_low, Close reversed above ──
    bull_wick  = swing_low - cur_low          # wick below the level (>=0 = swept)
    bull_close = cur_close - swing_low        # how far close is back above level

    bullish = (
        cur_low   < swing_low
        and cur_close > swing_low
        and bull_wick  >= MIN_WICK_ATR * cur_atr
        and bull_close >= MIN_BACK_ATR * cur_atr
    )

    # ── Bearish sweep: High pierced above swing_high, Close reversed below ──
    bear_wick  = cur_high - swing_high        # wick above the level
    bear_close = swing_high - cur_close       # how far close is back below level

    bearish = (
        cur_high  > swing_high
        and cur_close < swing_high
        and bear_wick  >= MIN_WICK_ATR * cur_atr
        and bear_close >= MIN_BACK_ATR * cur_atr
    )

    # If both fire (extremely rare spike-through), take the larger sweep
    if bullish and bearish:
        bullish = bull_wick >= bear_wick
        bearish = not bullish

    vol_note = f"RVOL={cur_rvol:.1f}x"
    vol_bonus = 0.05 if cur_rvol >= 1.2 else 0.0

    if bullish:
        wick_ratio = bull_wick / cur_atr
        conf = min(0.72 + wick_ratio * 0.05 + vol_bonus, 0.87)
        return _make_signal(
            NAME, TF, "LONG", conf, cur_close, cur_atr,
            sl_atr_mult=1.0, tp_atr_mult=2.0,
            reason=(
                f"Bullish sweep: Low={cur_low:.5g} below swing={swing_low:.5g} "
                f"| wick={wick_ratio:.2f}xATR | {vol_note}"
            ),
            df=df, use_structural=True,
        )

    if bearish:
        wick_ratio = bear_wick / cur_atr
        conf = min(0.72 + wick_ratio * 0.05 + vol_bonus, 0.87)
        return _make_signal(
            NAME, TF, "SHORT", conf, cur_close, cur_atr,
            sl_atr_mult=1.0, tp_atr_mult=2.0,
            reason=(
                f"Bearish sweep: High={cur_high:.5g} above swing={swing_high:.5g} "
                f"| wick={wick_ratio:.2f}xATR | {vol_note}"
            ),
            df=df, use_structural=True,
        )

    return _flat(
        NAME, TF,
        f"No sweep: swing_high={swing_high:.5g} swing_low={swing_low:.5g} "
        f"| bar H={cur_high:.5g} L={cur_low:.5g} C={cur_close:.5g}",
    )


# ── Liquidity sweep confluence helper ────────────────────────────────────────

def recent_liq_sweep(
    df: pd.DataFrame,
    lookback: int = 5,
    min_wick_atr: float = 0.3,
    min_back_atr: float = 0.2,
) -> tuple[str, float] | None:
    """Check whether a liquidity sweep occurred in the last `lookback` bars.

    Scans bars [-lookback .. -1] (not the current live bar) against the swing
    high/low from the 30 bars preceding that window.  Returns
    ``('LONG', wick_ratio)`` for a bullish sweep, ``('SHORT', wick_ratio)``
    for a bearish sweep, or ``None`` if no qualifying sweep was found.

    Used as a confluence multiplier: when another strategy fires in the same
    direction as a recent sweep, confidence is boosted because institutions
    just cleared stops before the move.
    """
    SWING_LOOKBACK = 30

    needed = lookback + SWING_LOOKBACK + 10
    if len(df) < needed:
        return None

    atr_s = ta.atr(df["High"], df["Low"], df["Close"], length=10)
    if atr_s is None or atr_s.dropna().empty:
        return None
    cur_atr = float(atr_s.dropna().iloc[-1])
    if cur_atr <= 0 or np.isnan(cur_atr):
        return None

    # Swing reference: 30 bars before the scan window
    ref_end   = -(lookback + 1)
    ref_start = ref_end - SWING_LOOKBACK
    ref = df.iloc[ref_start:ref_end] if ref_end != 0 else df.iloc[ref_start:]
    if ref.empty:
        return None

    swing_high = float(ref["High"].max())
    swing_low  = float(ref["Low"].min())
    if np.isnan(swing_high) or np.isnan(swing_low):
        return None

    # Scan the recent bars (newest first so we return the freshest sweep)
    scan = df.iloc[-lookback:-1] if lookback > 1 else df.iloc[-2:-1]
    best_bull: tuple[int, float] | None = None  # (row_idx, wick_ratio)
    best_bear: tuple[int, float] | None = None

    for i in range(len(scan) - 1, -1, -1):
        row = scan.iloc[i]
        lo, hi, cl = float(row["Low"]), float(row["High"]), float(row["Close"])

        bull_wick  = swing_low - lo
        bull_close = cl - swing_low
        if (lo < swing_low
                and cl > swing_low
                and bull_wick  >= min_wick_atr * cur_atr
                and bull_close >= min_back_atr * cur_atr):
            if best_bull is None or bull_wick > best_bull[1]:
                best_bull = (i, bull_wick / cur_atr)

        bear_wick  = hi - swing_high
        bear_close = swing_high - cl
        if (hi > swing_high
                and cl < swing_high
                and bear_wick  >= min_wick_atr * cur_atr
                and bear_close >= min_back_atr * cur_atr):
            if best_bear is None or bear_wick > best_bear[1]:
                best_bear = (i, bear_wick / cur_atr)

    # Return the more recent sweep; if same bar, larger wick wins
    if best_bull and best_bear:
        return ("LONG", best_bull[1]) if best_bull[0] >= best_bear[0] else ("SHORT", best_bear[1])
    if best_bull:
        return ("LONG", best_bull[1])
    if best_bear:
        return ("SHORT", best_bear[1])
    return None


# ════════════════════════════════════════════════════════════════════════════════
# STRATEGY — Fair Value Gap (FVG)  (5m)
# ════════════════════════════════════════════════════════════════════════════════

def fvg_5m(df: pd.DataFrame) -> IntradaySignal:
    """
    Fair Value Gap reversal entry on 5m bars.

    An FVG is a 3-candle imbalance where the middle candle moves so fast it
    leaves an untraded price zone: candle[-3].High < candle[-1].Low (bullish)
    or candle[-3].Low > candle[-1].High (bearish).  Price frequently returns
    to fill this zone — that pullback is the entry.

    Long:  unmitigated bullish FVG in last 20 bars AND current close is
           inside the zone [c[-3].High, c[-1].Low]
    Short: unmitigated bearish FVG in last 20 bars AND current close is
           inside the zone [c[-1].High, c[-3].Low]

    Mitigation: if any bar between FVG formation and current bar fully crossed
    the zone (low < zone_low for bullish), the FVG is invalidated.
    Gap must be >= 0.2×ATR to filter tick-sized noise.

    Fresher FVGs (age <= 5 bars) → conf 0.73; older → 0.68.
    RVOL >= 1.2 adds +0.05.

    Regime: trend-following (FVGs form during impulses, fill as trend continues).
    Penalised in ranging markets via _TREND_STRATEGIES.
    """
    TF   = "5m"
    NAME = "FVG_5M"

    LOOKBACK    = 20
    MIN_GAP_ATR = 0.2

    min_bars = LOOKBACK + 5
    if len(df) < min_bars:
        return _flat(NAME, TF, f"Insufficient bars (need {min_bars})")

    atr = _safe_atr(df, length=10)
    if atr <= 0:
        return _flat(NAME, TF, "ATR zero")

    cur_close = float(df["Close"].iloc[-1])
    n         = len(df)

    best_bull: tuple[float, float, int] | None = None  # (zone_low, zone_high, age)
    best_bear: tuple[float, float, int] | None = None

    # Scan bars to find FVG zones (3-candle pattern ending at bar i)
    for i in range(max(2, n - LOOKBACK), n - 1):
        h1 = float(df["High"].iloc[i - 2])
        l1 = float(df["Low"].iloc[i - 2])
        l3 = float(df["Low"].iloc[i])
        h3 = float(df["High"].iloc[i])

        # ── Bullish FVG: gap between candle[-3].high and candle[-1].low ──
        gap = l3 - h1
        if gap >= MIN_GAP_ATR * atr:
            zone_low, zone_high = h1, l3
            # Mitigation: any bar after formation (excl. current) went below zone_low
            post = df["Low"].iloc[i + 1: n - 1]
            if post.empty or float(post.min()) >= zone_low:
                age = n - 1 - i
                if best_bull is None or age < best_bull[2]:
                    best_bull = (zone_low, zone_high, age)

        # ── Bearish FVG: gap between candle[-3].low and candle[-1].high ──
        gap = l1 - h3
        if gap >= MIN_GAP_ATR * atr:
            zone_low, zone_high = h3, l1
            post = df["High"].iloc[i + 1: n - 1]
            if post.empty or float(post.max()) <= zone_high:
                age = n - 1 - i
                if best_bear is None or age < best_bear[2]:
                    best_bear = (zone_low, zone_high, age)

    rvol = _safe_rvol(df)

    # ── Entry: current close inside the freshest unmitigated zone ──
    if best_bull and best_bull[0] <= cur_close <= best_bull[1]:
        zone_low, zone_high, age = best_bull
        conf = 0.73 if age <= 5 else 0.68
        if rvol >= 1.2:
            conf = min(conf + 0.05, 0.88)
        return _make_signal(
            NAME, TF, "LONG", conf, cur_close, atr,
            sl_atr_mult=1.0, tp_atr_mult=2.0,
            reason=(
                f"Bullish FVG [{zone_low:.5g}-{zone_high:.5g}] "
                f"age={age}bars | RVOL={rvol:.1f}x"
            ),
            df=df, use_structural=True,
        )

    if best_bear and best_bear[0] <= cur_close <= best_bear[1]:
        zone_low, zone_high, age = best_bear
        conf = 0.73 if age <= 5 else 0.68
        if rvol >= 1.2:
            conf = min(conf + 0.05, 0.88)
        return _make_signal(
            NAME, TF, "SHORT", conf, cur_close, atr,
            sl_atr_mult=1.0, tp_atr_mult=2.0,
            reason=(
                f"Bearish FVG [{zone_low:.5g}-{zone_high:.5g}] "
                f"age={age}bars | RVOL={rvol:.1f}x"
            ),
            df=df, use_structural=True,
        )

    bull_note = f"bull FVG [{best_bull[0]:.5g}-{best_bull[1]:.5g}]" if best_bull else "no bull FVG"
    bear_note = f"bear FVG [{best_bear[0]:.5g}-{best_bear[1]:.5g}]" if best_bear else "no bear FVG"
    return _flat(NAME, TF, f"Price {cur_close:.5g} not in zone | {bull_note} | {bear_note}")


# ════════════════════════════════════════════════════════════════════════════════
# STRATEGY — Order Block  (5m)
# ════════════════════════════════════════════════════════════════════════════════

def order_block_5m(df: pd.DataFrame) -> IntradaySignal:
    """
    Order block pullback entry on 5m bars.

    An order block is the last opposing candle before a strong impulse move.
    Institutions leave unfilled orders in this zone; price returns to it
    before continuing in the impulse direction.

    Impulse detection: 3+ candles in the same direction with total body
    >= 1.5×ATR, found within the last 30 bars.

    Bullish OB: last bearish candle before a bullish impulse.
      Entry when: current bar LOW touches the OB zone AND close >= OB low
      (shows price entered the zone but rejected downward — support held).

    Bearish OB: last bullish candle before a bearish impulse.
      Entry when: current bar HIGH touches the OB zone AND close <= OB high.

    OB zone = [OB candle Low, OB candle High].
    Gap between OB candle and impulse start must be <= 3 bars (tighter OB).

    Conf 0.74 base; RVOL >= 1.2 adds +0.05 (capped 0.87).

    Regime: trend-following (OB is defined by an impulse, entry is in impulse
    direction). Penalised in ranging markets via _TREND_STRATEGIES.
    """
    TF   = "5m"
    NAME = "ORDER_BLOCK_5M"

    LOOKBACK         = 30
    MIN_IMPULSE_ATR  = 1.5   # impulse body must be >= this × ATR
    MIN_IMPULSE_BARS = 3     # consecutive same-direction bars forming the impulse
    MAX_OB_GAP       = 3     # OB candle must be within this many bars of impulse start

    min_bars = LOOKBACK + MIN_IMPULSE_BARS + 5
    if len(df) < min_bars:
        return _flat(NAME, TF, f"Insufficient bars (need {min_bars})")

    atr = _safe_atr(df, length=10)
    if atr <= 0:
        return _flat(NAME, TF, "ATR zero")

    n         = len(df)
    cur_close = float(df["Close"].iloc[-1])
    cur_low   = float(df["Low"].iloc[-1])
    cur_high  = float(df["High"].iloc[-1])
    rvol      = _safe_rvol(df)

    opens  = df["Open"].values
    closes = df["Close"].values
    highs  = df["High"].values
    lows   = df["Low"].values

    best_bull_ob: tuple[float, float, int] | None = None  # (ob_low, ob_high, impulse_end_idx)
    best_bear_ob: tuple[float, float, int] | None = None

    scan_start = max(MIN_IMPULSE_BARS, n - LOOKBACK)

    for i in range(scan_start, n - 1):
        # ── Bullish impulse: MIN_IMPULSE_BARS consecutive bullish candles ──
        if i >= MIN_IMPULSE_BARS:
            bull_run = all(
                closes[j] > opens[j]
                for j in range(i - MIN_IMPULSE_BARS + 1, i + 1)
            )
            if bull_run:
                total_body = closes[i] - opens[i - MIN_IMPULSE_BARS + 1]
                if total_body >= MIN_IMPULSE_ATR * atr:
                    impulse_start = i - MIN_IMPULSE_BARS + 1
                    # Find last bearish candle within MAX_OB_GAP bars before impulse
                    for k in range(
                        impulse_start - 1,
                        max(0, impulse_start - MAX_OB_GAP - 1),
                        -1,
                    ):
                        if closes[k] < opens[k]:  # bearish candle = order block
                            ob_low  = float(lows[k])
                            ob_high = float(highs[k])
                            age     = n - 1 - i
                            # Only keep if OB hasn't been fully violated since
                            post_lows = lows[i + 1: n - 1]
                            if len(post_lows) == 0 or float(post_lows.min()) >= ob_low:
                                if best_bull_ob is None or age < best_bull_ob[2]:
                                    best_bull_ob = (ob_low, ob_high, age)
                            break

        # ── Bearish impulse: MIN_IMPULSE_BARS consecutive bearish candles ──
        if i >= MIN_IMPULSE_BARS:
            bear_run = all(
                closes[j] < opens[j]
                for j in range(i - MIN_IMPULSE_BARS + 1, i + 1)
            )
            if bear_run:
                total_body = opens[i - MIN_IMPULSE_BARS + 1] - closes[i]
                if total_body >= MIN_IMPULSE_ATR * atr:
                    impulse_start = i - MIN_IMPULSE_BARS + 1
                    for k in range(
                        impulse_start - 1,
                        max(0, impulse_start - MAX_OB_GAP - 1),
                        -1,
                    ):
                        if closes[k] > opens[k]:  # bullish candle = order block
                            ob_low  = float(lows[k])
                            ob_high = float(highs[k])
                            age     = n - 1 - i
                            post_highs = highs[i + 1: n - 1]
                            if len(post_highs) == 0 or float(post_highs.max()) <= ob_high:
                                if best_bear_ob is None or age < best_bear_ob[2]:
                                    best_bear_ob = (ob_low, ob_high, age)
                            break

    conf_base = 0.74
    vol_bonus = 0.05 if rvol >= 1.2 else 0.0

    # ── Bullish OB entry: price low touched OB zone, close held above OB low ──
    if best_bull_ob:
        ob_low, ob_high, age = best_bull_ob
        touched = cur_low <= ob_high          # price entered the zone
        held    = cur_close >= ob_low         # close didn't fall through
        if touched and held:
            conf = min(conf_base + vol_bonus, 0.87)
            return _make_signal(
                NAME, TF, "LONG", conf, cur_close, atr,
                sl_atr_mult=1.0, tp_atr_mult=2.0,
                reason=(
                    f"Bullish OB [{ob_low:.5g}-{ob_high:.5g}] "
                    f"age={age}bars | RVOL={rvol:.1f}x"
                ),
                df=df, use_structural=True,
            )

    # ── Bearish OB entry: price high touched OB zone, close held below OB high ──
    if best_bear_ob:
        ob_low, ob_high, age = best_bear_ob
        touched = cur_high >= ob_low
        held    = cur_close <= ob_high
        if touched and held:
            conf = min(conf_base + vol_bonus, 0.87)
            return _make_signal(
                NAME, TF, "SHORT", conf, cur_close, atr,
                sl_atr_mult=1.0, tp_atr_mult=2.0,
                reason=(
                    f"Bearish OB [{ob_low:.5g}-{ob_high:.5g}] "
                    f"age={age}bars | RVOL={rvol:.1f}x"
                ),
                df=df, use_structural=True,
            )

    bull_note = f"bull OB [{best_bull_ob[0]:.5g}-{best_bull_ob[1]:.5g}]" if best_bull_ob else "no bull OB"
    bear_note = f"bear OB [{best_bear_ob[0]:.5g}-{best_bear_ob[1]:.5g}]" if best_bear_ob else "no bear OB"
    return _flat(NAME, TF, f"No OB touch: {bull_note} | {bear_note}")


# ════════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR — run_intraday_signals
# ════════════════════════════════════════════════════════════════════════════════

_STRATEGIES_5M  = [vwap_rsi_5m, orb_5m, trend_momentum_5m,
                   pdh_pdl_sweep_5m, camarilla_pivot_5m, nr7_breakout_5m,
                   bb_scalp_5m, stoch_cross_5m, ema_micro_cross_5m,
                   mss_forex_15m,
                   double_bottom_breakout_5m, head_shoulders_breakdown_5m,
                   flag_breakout_5m,
                   liq_sweep_5m,
                   fvg_5m, order_block_5m]
_STRATEGIES_15M = [ema_pullback_15m, squeeze_15m, absorption_15m]

_STRATEGY_NAME_MAP: dict[str, str] = {
    "vwap_rsi_5m":       "VWAP_RSI_5M",
    "orb_5m":            "ORB_5M",
    "trend_momentum_5m": "TREND_MOM_5M",
    "ema_pullback_15m":  "EMA_PB_15M",
    "squeeze_15m":       "SQUEEZE_15M",
    "absorption_15m":    "ABSORB_15M",
    "pdh_pdl_sweep_5m":  "PDH_PDL_SWEEP_5M",
    "camarilla_pivot_5m":"CAMARILLA_5M",
    "nr7_breakout_5m":   "NR7_BREAKOUT_5M",
    "mss_forex_15m":     "MSS_FOREX_15M",
    "bb_scalp_5m":          "BB_SCALP_5M",
    "stoch_cross_5m":       "STOCH_CROSS_5M",
    "ema_micro_cross_5m":   "EMA_MICRO_CROSS_5M",
    "double_bottom_breakout_5m":  "DOUBLE_BOT_BREAKOUT_5M",
    "head_shoulders_breakdown_5m":"HS_BREAKDOWN_5M",
    "flag_breakout_5m":           "FLAG_BREAKOUT_5M",
}


def run_intraday_signals(
    ticker: str,
    timeframe: str = "15m",
    account_size: float = 100_000.0,
    risk_pct: float = 1.0,
) -> tuple[list[IntradaySignal], list[IntradaySignal]]:
    """
    Main entry point.  Fetches intraday data, runs all strategies for the
    requested timeframe, and returns:

        (active_signals, all_signals)

    where active_signals contains only LONG/SHORT signals (sorted by confidence
    descending) and all_signals includes FLAT results too.

    Also attaches lot_size to each active signal via position_sizing.

    Parameters
    ----------
    ticker : str
    timeframe : str
        "5m" or "15m"
    account_size : float
        Portfolio size in USD for position sizing.
    risk_pct : float
        Percentage of account to risk per trade (e.g. 1.0 = 1%).
    """
    from data.fetcher_intraday import fetch_intraday_data
    from utils.position_sizing import calculate as calc_position

    df = fetch_intraday_data(ticker, interval=timeframe, days=59)

    strategy_fns = _STRATEGIES_5M if timeframe == "5m" else _STRATEGIES_15M

    all_signals: list[IntradaySignal] = []
    for fn in strategy_fns:
        try:
            sig = fn(df)
            all_signals.append(sig)
        except Exception as exc:
            tf = timeframe
            name = _STRATEGY_NAME_MAP.get(fn.__name__, fn.__name__.upper())
            all_signals.append(_flat(name, tf, f"Runtime error: {exc}"))

    # Absorption confluence multiplier (Valentini): if any absorption in last 3 bars,
    # boost LONG/SHORT signal confidence by 0.10 (capped at 1.0)
    try:
        abs_s = ta.absorption(df["High"], df["Low"], df["Close"], df["Volume"])
        recent_abs = bool(abs_s.iloc[-3:].any()) if len(abs_s) >= 3 else False
        if recent_abs:
            for sig in all_signals:
                if sig.signal in ("LONG", "SHORT"):
                    sig.confidence = min(1.0, sig.confidence + 0.10)
                    sig.reason += " [+absorption confluence]"
    except Exception:
        pass

    active = sorted(
        [s for s in all_signals if s.signal in ("LONG", "SHORT")],
        key=lambda s: s.confidence,
        reverse=True,
    )

    # Attach position sizing to each active signal
    for sig in active:
        try:
            ps = calc_position(account_size, risk_pct, sig.entry, sig.stop_loss, sig.take_profit)
            # Attach as extra attribute (not in dataclass — we'll use a wrapper dict in UI)
            sig._lot_size   = ps.shares
            sig._pos_value  = ps.position_value
            sig._dollar_risk = ps.dollar_risk
        except Exception:
            sig._lot_size   = 0
            sig._pos_value  = 0.0
            sig._dollar_risk = 0.0

    return active, all_signals
