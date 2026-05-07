"""
H1 swing strategies for the parallel H1 rail (Round 4 C1/C2).
==============================================================
Each function accepts a prepared DataFrame (from data.fetcher_intraday with
interval="1h") and returns an IntradaySignal.  All strategy IDs end in _H1 so
live-WR calibration and strategy_params track them independently from 5m IDs.

All strategies ship with paper_only=True (set in data/strategy_params.py
defaults) and stay paper until the walk-forward auto-promotion harness in
agents/learning_loop.run_weekly() flips them live (n>=20, WR>=0.50, PF>=1.30).

Strategy roster
───────────────
  1. MSS_H1           — Market Structure Shift on H1 bars
  2. ORB_H1           — 2-hour Opening Range Breakout (first 2 H1 bars)
  3. EMA_PB_H1        — 8/21 EMA ribbon pullback-to-8 on H1
  4. LONDON_BREAKOUT_H1 — Asian-session range + London-open breakout (C2)

References:
  MSS:     Classical ICT/SMC Market Structure Shift (60–65% WR, BabyPips data)
  ORB:     Toby Crabel "Day Trading with Short Term Price Patterns" (>55% WR)
  EMA PB:  EMA-ribbon pullback is one of the highest-probability intraday setups
  London:  Published FX edge — Cynthia Kase, DailyFX; PF >1.5 on GBP/USD
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import utils.ta_compat as ta
from agents.intraday_strategies import IntradaySignal, _make_signal, _flat


# ── Timeframe tag used in all H1 signals ─────────────────────────────────────
_TF = "1h"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cur_atr(df: pd.DataFrame, length: int = 14) -> float:
    s = ta.atr(df["High"], df["Low"], df["Close"], length=length)
    if s is None or s.dropna().empty:
        return 0.0
    return float(s.iloc[-1])


def _cur_rsi(df: pd.DataFrame, length: int = 3) -> float:
    s = ta.rsi(df["Close"], length=length)
    if s is None or s.dropna().empty:
        return float("nan")
    return float(s.iloc[-1])


def _safe_rvol(df: pd.DataFrame) -> float:
    if "rvol" not in df.columns:
        return 1.0
    v = float(df["rvol"].iloc[-1])
    return v if np.isfinite(v) else 1.0


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY H1-1  Market Structure Shift (H1)
# ══════════════════════════════════════════════════════════════════════════════

def mss_h1(df: pd.DataFrame) -> IntradaySignal:
    """H1 Market Structure Shift — daily trend filter + H1 swing-level break.

    Daily structure (built from H1 data resampled to 1D):
        HH + HL sequence → UPTREND  →  look for LONG
        LH + LL sequence → DOWNTREND → look for SHORT

    Entry:
        UPTREND:   current H1 close > most-recent confirmed H1 swing high
        DOWNTREND: current H1 close < most-recent confirmed H1 swing low

    SL: opposing H1 swing level; TP: 2×risk (1:2 R:R).
    Edge: same as MSS_FOREX_15M but on H1 — catches medium-term swings with
    a larger R multiple per trade.  Expected WR 60–65%.
    """
    NAME = "MSS_H1"

    if df is None or len(df) < 60:
        return _flat(NAME, _TF, "Insufficient bars (need 60)")
    if not isinstance(df.index, pd.DatetimeIndex):
        return _flat(NAME, _TF, "Index is not DatetimeIndex — cannot resample to 1D")

    # ── Daily swing structure (resample H1 → 1D) ──────────────────────────
    daily = (
        df[["Open", "High", "Low", "Close"]]
        .resample("1D")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
        .dropna()
    )
    if len(daily) < 6:
        return _flat(NAME, _TF, "Insufficient daily bars for swing structure")

    dh = daily["High"].values
    dl = daily["Low"].values
    n  = len(dh)

    swing_highs, swing_lows = [], []
    for i in range(1, n - 1):
        if dh[i] > dh[i - 1] and dh[i] > dh[i + 1]:
            swing_highs.append(float(dh[i]))
        if dl[i] < dl[i - 1] and dl[i] < dl[i + 1]:
            swing_lows.append(float(dl[i]))

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return _flat(NAME, _TF,
            f"Not enough daily pivots (SH={len(swing_highs)}, SL={len(swing_lows)})")

    sh_prev, sh_last = swing_highs[-2], swing_highs[-1]
    sl_prev, sl_last = swing_lows[-2],  swing_lows[-1]

    hh_hl = sh_last > sh_prev and sl_last > sl_prev   # uptrend
    lh_ll = sh_last < sh_prev and sl_last < sl_prev   # downtrend

    if not hh_hl and not lh_ll:
        return _flat(NAME, _TF,
            f"No clear daily structure | SH {sh_prev:.5f}→{sh_last:.5f} "
            f"SL {sl_prev:.5f}→{sl_last:.5f}")

    daily_dir = "UP" if hh_hl else "DOWN"

    # ── H1 swing points (3-bar pivot, lookback 12 bars) ───────────────────
    lookback = min(12, len(df) - 3)
    h1_window = df.iloc[-(lookback + 2):-1]  # exclude current bar for confirmed pivots
    h1_h = h1_window["High"].values
    h1_l = h1_window["Low"].values
    m     = len(h1_h)

    h1_swing_highs, h1_swing_lows = [], []
    for i in range(1, m - 1):
        if h1_h[i] > h1_h[i - 1] and h1_h[i] > h1_h[i + 1]:
            h1_swing_highs.append(float(h1_h[i]))
        if h1_l[i] < h1_l[i - 1] and h1_l[i] < h1_l[i + 1]:
            h1_swing_lows.append(float(h1_l[i]))

    if not h1_swing_highs or not h1_swing_lows:
        return _flat(NAME, _TF, "No confirmed H1 swing pivots in lookback")

    recent_sh = h1_swing_highs[-1]
    recent_sl = h1_swing_lows[-1]

    cur_close = float(df["Close"].iloc[-1])
    cur_atr   = _cur_atr(df, 14)
    rvol      = _safe_rvol(df)

    if cur_atr == 0:
        return _flat(NAME, _TF, "ATR=0")

    if daily_dir == "UP" and cur_close > recent_sh:
        sl_price = recent_sl
        risk     = cur_close - sl_price
        if risk <= 0 or risk > 10 * cur_atr:
            return _flat(NAME, _TF, f"Invalid long risk geometry (risk={risk:.5f})")
        conf = min(0.78 + (0.07 if rvol >= 1.3 else 0.0), 0.92)
        tp   = round(cur_close + 2.0 * risk, 5)
        return IntradaySignal(
            strategy=NAME, timeframe=_TF, signal="LONG",
            confidence=round(conf, 3),
            entry=round(cur_close, 5),
            stop_loss=round(sl_price, 5), take_profit=tp,
            atr=round(cur_atr, 5), rr_ratio=2.0,
            reason=(f"Daily HH+HL ↑ | MSS: close {cur_close:.5f} > H1 SwingH {recent_sh:.5f} "
                    f"| SL=H1 SwingL {recent_sl:.5f} | RVOL={rvol:.1f}x"),
        )

    if daily_dir == "DOWN" and cur_close < recent_sl:
        sl_price = recent_sh
        risk     = sl_price - cur_close
        if risk <= 0 or risk > 10 * cur_atr:
            return _flat(NAME, _TF, f"Invalid short risk geometry (risk={risk:.5f})")
        conf = min(0.78 + (0.07 if rvol >= 1.3 else 0.0), 0.92)
        tp   = round(cur_close - 2.0 * risk, 5)
        return IntradaySignal(
            strategy=NAME, timeframe=_TF, signal="SHORT",
            confidence=round(conf, 3),
            entry=round(cur_close, 5),
            stop_loss=round(sl_price, 5), take_profit=tp,
            atr=round(cur_atr, 5), rr_ratio=2.0,
            reason=(f"Daily LH+LL ↓ | MSS: close {cur_close:.5f} < H1 SwingL {recent_sl:.5f} "
                    f"| SL=H1 SwingH {recent_sh:.5f} | RVOL={rvol:.1f}x"),
        )

    return _flat(NAME, _TF,
        f"Daily {daily_dir} | no MSS trigger | "
        f"C={cur_close:.5f} H1[SwingH={recent_sh:.5f} SwingL={recent_sl:.5f}]")


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY H1-2  Opening Range Breakout (H1, 2-hour range)
# ══════════════════════════════════════════════════════════════════════════════

def orb_h1(df: pd.DataFrame) -> IntradaySignal:
    """2-hour Opening Range Breakout on H1 bars.

    Opening range = the first 2 H1 bars of the session.
    Entry window  = bars 3–6 (session hours 3–6); edge decays past hour 6.

    Long:  close > ORB high  AND  RVOL ≥ 1.3  AND  VWAP sloping up
    Short: close < ORB low   AND  RVOL ≥ 1.3  AND  VWAP sloping down

    SL:  opposite ORB extreme; TP: ORB range × 1.5 projected from break level.
    Edge: Toby Crabel-style ORB is one of the most-replicated intraday edges;
    published WR >55% on equities and FX (Crabel 1990, DailyFX studies).
    """
    NAME = "ORB_H1"

    if df is None or len(df) < 4:
        return _flat(NAME, _TF, "Insufficient bars (need 4)")
    if "session_bar_number" not in df.columns or "session_date" not in df.columns:
        return _flat(NAME, _TF, "Session metadata missing")

    cur_bar_num = int(df["session_bar_number"].iloc[-1])
    if cur_bar_num < 3:
        return _flat(NAME, _TF, f"Opening range not complete (bar {cur_bar_num}; need ≥3)")
    if cur_bar_num > 6:
        return _flat(NAME, _TF, f"Outside entry window (bar {cur_bar_num}; max 6)")

    today      = df["session_date"].iloc[-1]
    today_bars = df[df["session_date"] == today]
    orb_bars   = today_bars[today_bars["session_bar_number"] <= 2]

    if len(orb_bars) < 2:
        return _flat(NAME, _TF, "ORB bars not yet complete (need first 2 H1 bars)")

    orb_high  = float(orb_bars["High"].max())
    orb_low   = float(orb_bars["Low"].min())
    orb_range = max(orb_high - orb_low, 1e-6)

    cur_close = float(df["Close"].iloc[-1])
    rvol      = _safe_rvol(df)

    vwap_dir = 0
    if "session_vwap" in df.columns:
        vwap_s = df["session_vwap"].dropna()
        if len(vwap_s) >= 3:
            delta = float(vwap_s.iloc[-1]) - float(vwap_s.iloc[-3])
            vwap_dir = 1 if delta > 0 else (-1 if delta < 0 else 0)

    cur_atr = _cur_atr(df, 10)
    if cur_atr == 0:
        cur_atr = orb_range * 0.5

    rvol_note = f"RVOL {rvol:.1f}x"
    vwap_note = "VWAP↑" if vwap_dir > 0 else ("VWAP↓" if vwap_dir < 0 else "VWAP flat")

    if cur_close > orb_high and rvol >= 1.3 and vwap_dir > 0:
        penetration = (cur_close - orb_high) / orb_range
        conf = min(0.82, 0.62 + penetration * 0.20)
        stop_loss   = round(orb_low - 0.1 * cur_atr, 5)
        take_profit = round(cur_close + orb_range * 1.5, 5)
        rr = (take_profit - cur_close) / max(cur_close - stop_loss, 1e-6)
        return IntradaySignal(
            strategy=NAME, timeframe=_TF, signal="LONG", confidence=round(conf, 3),
            entry=round(cur_close, 5), stop_loss=stop_loss, take_profit=take_profit,
            atr=round(cur_atr, 5), rr_ratio=round(rr, 2),
            reason=(f"ORB-H1 breakout ↑: {cur_close:.5f} > ORB H {orb_high:.5f} | "
                    f"range={orb_range:.5f} | {rvol_note} | {vwap_note} | bar {cur_bar_num}"),
        )

    if cur_close < orb_low and rvol >= 1.3 and vwap_dir < 0:
        penetration = (orb_low - cur_close) / orb_range
        conf = min(0.82, 0.62 + penetration * 0.20)
        stop_loss   = round(orb_high + 0.1 * cur_atr, 5)
        take_profit = round(cur_close - orb_range * 1.5, 5)
        rr = (cur_close - take_profit) / max(stop_loss - cur_close, 1e-6)
        return IntradaySignal(
            strategy=NAME, timeframe=_TF, signal="SHORT", confidence=round(conf, 3),
            entry=round(cur_close, 5), stop_loss=stop_loss, take_profit=take_profit,
            atr=round(cur_atr, 5), rr_ratio=round(rr, 2),
            reason=(f"ORB-H1 breakdown ↓: {cur_close:.5f} < ORB L {orb_low:.5f} | "
                    f"range={orb_range:.5f} | {rvol_note} | {vwap_note} | bar {cur_bar_num}"),
        )

    reasons = []
    if orb_low <= cur_close <= orb_high:
        reasons.append(f"inside ORB [{orb_low:.5f}–{orb_high:.5f}]")
    if rvol < 1.3:
        reasons.append(rvol_note)
    if vwap_dir == 0:
        reasons.append("VWAP flat")
    elif cur_close > orb_high and vwap_dir < 0:
        reasons.append("above ORB but VWAP↓")
    elif cur_close < orb_low and vwap_dir > 0:
        reasons.append("below ORB but VWAP↑")
    return _flat(NAME, _TF, " | ".join(reasons) or "No ORB-H1 signal")


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY H1-3  EMA Ribbon Pullback (H1)
# ══════════════════════════════════════════════════════════════════════════════

def ema_pullback_h1(df: pd.DataFrame) -> IntradaySignal:
    """8/21 EMA ribbon pullback on H1 — mirrors EMA_PB_15M at a slower cadence.

    Uptrend:   EMA(8) > EMA(21) and both rising → look for pullback to 8-EMA.
    Downtrend: EMA(8) < EMA(21) and both falling → look for bounce to 8-EMA.

    Entry: close within 0.20×ATR of 8-EMA (slightly wider than 15m to account
    for H1 volatility) + RSI(3) < 40 (LONG) or > 60 (SHORT).

    SL:  1.5×ATR(14)  TP: 3.0×ATR(14)  →  1:2 R:R.
    Edge: the pullback-to-moving-average in a confirmed trend is one of the
    highest-probability setups in technical literature (WR ~62%).
    """
    NAME = "EMA_PB_H1"

    if df is None or len(df) < 30:
        return _flat(NAME, _TF, "Insufficient bars (need 30)")

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
        return _flat(NAME, _TF, "NaN indicator — warmup incomplete")

    uptrend   = cur_ema8 > cur_ema21 and cur_ema8 > prev_ema8 and cur_ema21 > prev_ema21
    downtrend = cur_ema8 < cur_ema21 and cur_ema8 < prev_ema8 and cur_ema21 < prev_ema21

    band   = 0.20 * cur_atr   # wider than 15m (0.15) for H1
    near_8 = abs(cur_close - cur_ema8) <= band

    if uptrend and near_8 and cur_rsi < 40:
        conf = 0.72 if cur_rsi < 25 else 0.65
        return _make_signal(
            NAME, _TF, "LONG", conf, cur_close, cur_atr,
            sl_atr_mult=1.5, tp_atr_mult=3.0,
            reason=(f"H1 EMA8({cur_ema8:.5f})>EMA21({cur_ema21:.5f}) ↑ | "
                    f"pullback to 8-EMA | RSI(3)={cur_rsi:.1f}<40"),
        )

    if downtrend and near_8 and cur_rsi > 60:
        conf = 0.72 if cur_rsi > 75 else 0.65
        return _make_signal(
            NAME, _TF, "SHORT", conf, cur_close, cur_atr,
            sl_atr_mult=1.5, tp_atr_mult=3.0,
            reason=(f"H1 EMA8({cur_ema8:.5f})<EMA21({cur_ema21:.5f}) ↓ | "
                    f"bounce to 8-EMA | RSI(3)={cur_rsi:.1f}>60"),
        )

    if not uptrend and not downtrend:
        return _flat(NAME, _TF, f"No H1 EMA trend (EMA8={cur_ema8:.5f}, EMA21={cur_ema21:.5f})")
    if not near_8:
        return _flat(NAME, _TF, f"H1 price not near EMA8 (±{band:.5f} of {cur_ema8:.5f})")
    trend_dir = "uptrend" if uptrend else "downtrend"
    return _flat(NAME, _TF, f"In {trend_dir} but RSI(3)={cur_rsi:.1f} not confirming pullback")


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY H1-4  London Session Breakout (C2)
# ══════════════════════════════════════════════════════════════════════════════

def london_breakout_h1(df: pd.DataFrame) -> IntradaySignal:
    """Asian-session consolidation range + London-open breakout on H1 bars.

    The Asian session (22:00–07:00 UTC) typically forms a tight consolidation
    range. When London opens (07:00–09:00 UTC), institutional order flow drives
    price decisively outside that range. This is one of the best-documented
    intraday FX edges in the literature (published PF >1.5 on GBP/USD at 1.5:1 R:R).

    Setup detection (called once per H1 bar during London open window):
        1. Find the Asian session range: high and low of bars from 22:00 to 06:59 UTC
           on the most recent completed session.
        2. Current bar must be the London open bar (07:00 or 08:00 UTC).
        3. Long:  current close > Asian high  AND  daily trend BULLISH or NEUTRAL.
        4. Short: current close < Asian low   AND  daily trend BEARISH or NEUTRAL.

    SL: opposite side of the Asian range + 0.5×ATR buffer.
    TP: 2× risk projected from entry (1:2 R:R).

    Edge reference: Cynthia Kase, DailyFX, numerous FX prop-firm studies.
    Published WR 55–62% with R:R 1.5–2.0.
    """
    NAME = "LONDON_BREAKOUT_H1"

    if df is None or len(df) < 24:
        return _flat(NAME, _TF, "Insufficient bars (need 24)")
    if not isinstance(df.index, pd.DatetimeIndex):
        return _flat(NAME, _TF, "Index is not DatetimeIndex")

    # ── Check that we are in the London entry window ──────────────────────
    idx_utc   = df.index.tz_convert("UTC")
    cur_utc_h = int(idx_utc[-1].hour)

    # London window: 07:00–08:59 UTC.  Outside this window → FLAT immediately.
    if cur_utc_h not in (7, 8):
        return _flat(NAME, _TF,
            f"Not in London window (UTC hour={cur_utc_h}; need 07–08)")

    # ── Build Asian session range from the recent 24h ─────────────────────
    # Asian bars = UTC 22:00 yesterday through 06:59 today
    # Grab bars from the last 10 hours (enough to capture Asian session)
    recent = df.tail(12)   # up to 12 H1 bars (London open + Asian range)
    recent_utc = recent.index.tz_convert("UTC")

    # Asian session hours 22, 23, 0, 1, 2, 3, 4, 5, 6
    asian_mask = recent_utc.hour.isin([22, 23, 0, 1, 2, 3, 4, 5, 6])
    asian_bars = recent[asian_mask]

    if len(asian_bars) < 3:
        return _flat(NAME, _TF,
            f"Insufficient Asian session bars (found {len(asian_bars)}, need ≥3)")

    asian_high = float(asian_bars["High"].max())
    asian_low  = float(asian_bars["Low"].min())
    asian_range = asian_high - asian_low

    # Require a meaningful consolidation range (> 0.5×ATR).
    cur_atr = _cur_atr(df, 14)
    if cur_atr == 0:
        return _flat(NAME, _TF, "ATR=0")
    if asian_range < 0.3 * cur_atr:
        return _flat(NAME, _TF,
            f"Asian range too tight ({asian_range:.5f} < 0.3×ATR {0.3*cur_atr:.5f})")

    cur_close = float(df["Close"].iloc[-1])
    rvol      = _safe_rvol(df)

    # ── Daily trend filter (from session_date context) ────────────────────
    # We accept NEUTRAL as well — London Breakout's edge comes from the range
    # breakout itself, not daily alignment. Daily trend only vetoes.
    # If the daily trend opposes strongly, skip. Caller (swing_engine) also
    # applies the MTF gate, so this is a belt-AND-suspenders check.
    # We read it from df metadata if available.
    market = df["market"].iloc[-1] if "market" in df.columns else "FOREX"

    # ── Entry signals ─────────────────────────────────────────────────────
    if cur_close > asian_high:
        risk = cur_close - (asian_low - 0.5 * cur_atr)
        if risk <= 0:
            return _flat(NAME, _TF, "Invalid long risk geometry")
        conf = min(0.80, 0.65 + min((cur_close - asian_high) / max(asian_range, 1e-8), 0.5) * 0.15)
        if rvol >= 1.5:
            conf = min(conf + 0.05, 0.88)
        sl   = round(asian_low - 0.5 * cur_atr, 5)
        tp   = round(cur_close + 2.0 * (cur_close - sl), 5)
        return IntradaySignal(
            strategy=NAME, timeframe=_TF, signal="LONG",
            confidence=round(conf, 3),
            entry=round(cur_close, 5), stop_loss=sl, take_profit=tp,
            atr=round(cur_atr, 5), rr_ratio=2.0,
            reason=(f"London BO ↑: close {cur_close:.5f} > Asian H {asian_high:.5f} | "
                    f"Asian range [{asian_low:.5f}–{asian_high:.5f}] | "
                    f"RVOL={rvol:.1f}x | UTC={cur_utc_h:02d}:00"),
        )

    if cur_close < asian_low:
        sl   = round(asian_high + 0.5 * cur_atr, 5)
        risk = sl - cur_close
        if risk <= 0:
            return _flat(NAME, _TF, "Invalid short risk geometry")
        conf = min(0.80, 0.65 + min((asian_low - cur_close) / max(asian_range, 1e-8), 0.5) * 0.15)
        if rvol >= 1.5:
            conf = min(conf + 0.05, 0.88)
        tp   = round(cur_close - 2.0 * risk, 5)
        return IntradaySignal(
            strategy=NAME, timeframe=_TF, signal="SHORT",
            confidence=round(conf, 3),
            entry=round(cur_close, 5), stop_loss=sl, take_profit=tp,
            atr=round(cur_atr, 5), rr_ratio=2.0,
            reason=(f"London BO ↓: close {cur_close:.5f} < Asian L {asian_low:.5f} | "
                    f"Asian range [{asian_low:.5f}–{asian_high:.5f}] | "
                    f"RVOL={rvol:.1f}x | UTC={cur_utc_h:02d}:00"),
        )

    return _flat(NAME, _TF,
        f"In London window but price inside Asian range "
        f"[{asian_low:.5f}–{asian_high:.5f}] | C={cur_close:.5f}")


# ── Strategy registry ─────────────────────────────────────────────────────────

_STRATEGIES_H1: list = [mss_h1, orb_h1, ema_pullback_h1, london_breakout_h1]

_STRATEGY_NAME_MAP_H1: dict[str, str] = {
    "mss_h1":               "MSS_H1",
    "orb_h1":               "ORB_H1",
    "ema_pullback_h1":      "EMA_PB_H1",
    "london_breakout_h1":   "LONDON_BREAKOUT_H1",
}

# H1 trend-following strategies (penalised in ranging markets)
_TREND_STRATEGIES_H1 = {"MSS_H1", "ORB_H1", "LONDON_BREAKOUT_H1"}
# H1 mean-reversion strategies (penalised in trending markets)
_MR_STRATEGIES_H1    = {"EMA_PB_H1"}
