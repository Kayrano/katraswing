"""
Expert Stock Market Analyzer — Chart Pattern Detector
Detects named swing trade and candlestick patterns from OHLCV data.

Patterns detected (22 total):
  Classic Reversal:   Head & Shoulders, Inv. H&S, Double Top/Bottom, Triple Top/Bottom
  Continuation:       Bull Flag, Bear Flag, Cup & Handle, Rectangle
  Triangle/Wedge:     Ascending, Descending, Symmetrical Triangle, Rising Wedge, Falling Wedge
  Candlestick (1-3 bars): Bullish/Bearish Engulfing, Hammer, Shooting Star,
                           Dragonfly/Gravestone Doji, Morning Star, Evening Star, Harami

Each PatternMatch includes a historical win_rate sourced from Bulkowski,
Quantified Strategies, and Liberated Stock Trader backtests.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


@dataclass
class PatternMatch:
    name: str
    bias: str           # "BULLISH" | "BEARISH" | "NEUTRAL"
    confidence: float   # 0.0 – 1.0 (how cleanly this instance fits)
    win_rate: float     # historical win rate (0.0 – 1.0) from backtests
    description: str
    bar_start: int      # index into df where pattern begins
    bar_end: int        # index where pattern ends (usually last bar)
    color: str          # display color
    timeframe_note: str = ""  # e.g. "Best on Daily/4H"
    correction: str = ""      # how to handle failure / false signals


@dataclass
class PatternReport:
    patterns: list[PatternMatch] = field(default_factory=list)
    dominant_bias: str = "NEUTRAL"
    pattern_score_adj: float = 0.0   # bonus/penalty for signal confidence
    avg_win_rate: float = 0.0        # avg win_rate of detected patterns


def detect_patterns(df: pd.DataFrame) -> PatternReport:
    """
    Run all detectors on the last ~100 bars of df.
    Returns a PatternReport with all found patterns.
    """
    window = df.iloc[-100:].copy().reset_index(drop=True)
    closes  = window["Close"].values
    highs   = window["High"].values
    lows    = window["Low"].values
    opens   = window["Open"].values if "Open" in window.columns else closes.copy()
    volumes = window["Volume"].values if "Volume" in window.columns else np.ones(len(closes))
    n       = len(window)

    matches: list[PatternMatch] = []

    # ── Classic / swing patterns (60-bar window) ──────────────────────────────
    w60 = min(60, n)
    c60, h60, l60 = closes[-w60:], highs[-w60:], lows[-w60:]
    n60 = len(c60)

    matches += _detect_bull_flag(c60, h60, l60, n60)
    matches += _detect_bear_flag(c60, h60, l60, n60)
    matches += _detect_double_bottom(c60, l60, n60)
    matches += _detect_double_top(c60, h60, n60)
    matches += _detect_triple_bottom(l60, c60, n60)
    matches += _detect_triple_top(h60, c60, n60)
    matches += _detect_head_and_shoulders(h60, l60, c60, n60)
    matches += _detect_inverse_head_and_shoulders(h60, l60, c60, n60)
    matches += _detect_cup_and_handle(c60, l60, n60)
    matches += _detect_ascending_triangle(h60, l60, n60)
    matches += _detect_descending_triangle(h60, l60, n60)
    matches += _detect_symmetrical_triangle(h60, l60, n60)
    matches += _detect_rising_wedge(h60, l60, c60, n60)
    matches += _detect_falling_wedge(h60, l60, c60, n60)
    matches += _detect_rectangle(h60, l60, c60, n60)

    # FVG and Inside Bar use last 20-bar window
    w20 = min(20, n)
    o20 = opens[-w20:]; c20 = closes[-w20:]; h20 = highs[-w20:]; l20 = lows[-w20:]
    n20 = len(c20)
    off20 = n - n20
    matches += _detect_fair_value_gap(o20, c20, h20, l20, n20, off20)
    matches += _detect_inside_bar(h20, l20, c20, n20, off20)

    # ── Candlestick patterns (last 5 bars only) ───────────────────────────────
    if n >= 3:
        c5 = closes[-5:]; h5 = highs[-5:]; l5 = lows[-5:]
        o5 = opens[-5:];  v5 = volumes[-5:]
        n5 = len(c5)
        offset = n - n5

        matches += _detect_engulfing(o5, c5, h5, l5, v5, n5, offset)
        matches += _detect_hammer(o5, c5, h5, l5, n5, offset)
        matches += _detect_shooting_star(o5, c5, h5, l5, n5, offset)
        matches += _detect_doji(o5, c5, h5, l5, n5, offset)
        matches += _detect_morning_star(o5, c5, h5, l5, v5, n5, offset)
        matches += _detect_evening_star(o5, c5, h5, l5, v5, n5, offset)
        matches += _detect_harami(o5, c5, n5, offset)

    # Dominant bias
    bull_score = sum(m.confidence for m in matches if m.bias == "BULLISH")
    bear_score = sum(m.confidence for m in matches if m.bias == "BEARISH")

    if bull_score > bear_score:
        dominant = "BULLISH"
        score_adj = min(2.5, bull_score * 1.5)
    elif bear_score > bull_score:
        dominant = "BEARISH"
        score_adj = -min(2.5, bear_score * 1.5)
    else:
        dominant = "NEUTRAL"
        score_adj = 0.0

    avg_wr = (sum(m.win_rate for m in matches) / len(matches)) if matches else 0.0

    return PatternReport(
        patterns=sorted(matches, key=lambda m: m.confidence, reverse=True),
        dominant_bias=dominant,
        pattern_score_adj=round(score_adj, 2),
        avg_win_rate=round(avg_wr, 3),
    )


# ── Pivot helpers ─────────────────────────────────────────────────────────────

def _find_local_highs(highs: np.ndarray, order: int = 3) -> list[int]:
    """Vectorized via scipy.signal.argrelextrema with `np.greater_equal` —
    matches the original `highs[i] == max(window)` semantics (plateaus with
    ties all qualify). Result is filtered to interior indices to mirror the
    original `range(order, n - order)` bound and avoid scipy's edge clipping.
    """
    n = len(highs)
    if n < 2 * order + 1:
        return []
    arr = np.asarray(highs)
    candidates = argrelextrema(arr, np.greater_equal, order=order)[0]
    interior = candidates[(candidates >= order) & (candidates < n - order)]
    return interior.tolist()


def _find_local_lows(lows: np.ndarray, order: int = 3) -> list[int]:
    """Vectorized counterpart of `_find_local_highs` for troughs."""
    n = len(lows)
    if n < 2 * order + 1:
        return []
    arr = np.asarray(lows)
    candidates = argrelextrema(arr, np.less_equal, order=order)[0]
    interior = candidates[(candidates >= order) & (candidates < n - order)]
    return interior.tolist()


def _candle_body(o, c):
    return abs(c - o)


def _candle_range(h, l):
    return h - l


# ── Classic Continuation Patterns ────────────────────────────────────────────

def _detect_bull_flag(closes, highs, lows, n) -> list[PatternMatch]:
    """Bull Flag: sharp pole up (≥5% in ≤8 bars) → tight consolidation (≤5%) → breakout."""
    matches = []
    if n < 20:
        return matches

    lookback = min(25, n - 5)
    for pole_start in range(n - lookback, n - 8):
        pole_end = pole_start + 5
        if pole_end >= n - 3:
            break
        pole_rise = (closes[pole_end] - closes[pole_start]) / closes[pole_start]
        if pole_rise < 0.05:
            continue

        cons_start = pole_end
        cons_end   = min(n - 1, cons_start + 10)
        cons_range = highs[cons_start:cons_end + 1]
        cons_lows  = lows[cons_start:cons_end + 1]
        if len(cons_range) < 3:
            continue

        cons_high = float(np.max(cons_range))
        cons_low  = float(np.min(cons_lows))
        cons_size = (cons_high - cons_low) / cons_high

        if cons_size <= 0.05 and closes[-1] >= cons_high * 0.99:
            conf = min(1.0, pole_rise * 5 + (0.05 - cons_size) * 5)
            matches.append(PatternMatch(
                name="Bull Flag",
                bias="BULLISH",
                confidence=round(conf, 2),
                win_rate=0.67,
                description=f"Pole +{pole_rise*100:.1f}% → tight consolidation ({cons_size*100:.1f}% range). Breakout zone.",
                bar_start=pole_start,
                bar_end=n - 1,
                color="#00c851",
                timeframe_note="Best on 5m–Daily",
                correction="Avoid flags >20 bars. Require volume collapse in flag + surge on breakout.",
            ))
            break

    return matches


def _detect_bear_flag(closes, highs, lows, n) -> list[PatternMatch]:
    """Bear Flag: sharp pole down → tight consolidation → breakdown."""
    matches = []
    if n < 20:
        return matches

    lookback = min(25, n - 5)
    for pole_start in range(n - lookback, n - 8):
        pole_end = pole_start + 5
        if pole_end >= n - 3:
            break
        pole_drop = (closes[pole_start] - closes[pole_end]) / closes[pole_start]
        if pole_drop < 0.05:
            continue

        cons_start = pole_end
        cons_end   = min(n - 1, cons_start + 10)
        cons_range = highs[cons_start:cons_end + 1]
        cons_lows  = lows[cons_start:cons_end + 1]
        if len(cons_range) < 3:
            continue

        cons_high = float(np.max(cons_range))
        cons_low  = float(np.min(cons_lows))
        cons_size = (cons_high - cons_low) / cons_high

        if cons_size <= 0.05 and closes[-1] <= cons_low * 1.01:
            conf = min(1.0, pole_drop * 5 + (0.05 - cons_size) * 5)
            matches.append(PatternMatch(
                name="Bear Flag",
                bias="BEARISH",
                confidence=round(conf, 2),
                win_rate=0.68,
                description=f"Pole -{pole_drop*100:.1f}% → tight consolidation. Breakdown zone.",
                bar_start=pole_start,
                bar_end=n - 1,
                color="#ff4444",
                timeframe_note="Best on 5m–Daily",
                correction="Stop above flag upper channel. Do not short into major support.",
            ))
            break

    return matches


def _detect_cup_and_handle(closes, lows, n) -> list[PatternMatch]:
    """Cup & Handle: U-shaped recovery (>20 bars) + small handle pullback."""
    matches = []
    if n < 35:
        return matches

    cup_start = max(0, n - 55)
    cup_end   = n - 8

    if cup_end - cup_start < 20:
        return matches

    left_rim  = float(closes[cup_start])
    right_rim = float(closes[cup_end])
    cup_base  = float(np.min(lows[cup_start:cup_end + 1]))

    rim_diff  = abs(left_rim - right_rim) / max(left_rim, right_rim)
    cup_depth = (min(left_rim, right_rim) - cup_base) / min(left_rim, right_rim)

    if rim_diff > 0.06 or cup_depth < 0.08:
        return matches

    handle_low  = float(np.min(lows[cup_end:n]))
    handle_drop = (right_rim - handle_low) / right_rim

    if handle_drop <= 0.0 or handle_drop > cup_depth * 0.55:
        return matches

    if closes[-1] >= right_rim * 0.97:
        conf = min(1.0, 0.4 + cup_depth * 2 + (0.06 - rim_diff) * 5)
        matches.append(PatternMatch(
            name="Cup & Handle",
            bias="BULLISH",
            confidence=round(conf, 2),
            win_rate=0.76,
            description=f"Cup base ${cup_base:.2f}, rim ~${right_rim:.2f}. Handle forming, breakout near.",
            bar_start=cup_start,
            bar_end=n - 1,
            color="#33b5e5",
            timeframe_note="Best on Daily/Weekly (7–65 week cup)",
            correction="Handle must retrace 10–30% of cup. Buy point = handle high + 0.10. Stop 7–8% below entry.",
        ))

    return matches


def _detect_rectangle(closes, highs, lows, n) -> list[PatternMatch]:
    """Rectangle: price bouncing between flat support and resistance (≥2 touches each)."""
    matches = []
    if n < 20:
        return matches

    recent_h = highs[-20:]
    recent_l = lows[-20:]

    top    = float(np.max(recent_h))
    bottom = float(np.min(recent_l))
    height = (top - bottom) / top

    if height < 0.02 or height > 0.15:
        return matches

    top_touches    = sum(1 for h in recent_h if h >= top * 0.98)
    bottom_touches = sum(1 for l in recent_l if l <= bottom * 1.02)

    if top_touches >= 2 and bottom_touches >= 2:
        conf = min(1.0, 0.45 + (top_touches + bottom_touches) * 0.07)
        # Bias = prior trend direction (use simple slope)
        slope = closes[-1] - closes[-20]
        bias  = "BULLISH" if slope >= 0 else "BEARISH"
        matches.append(PatternMatch(
            name="Rectangle",
            bias=bias,
            confidence=round(conf, 2),
            win_rate=0.77,
            description=f"Range ${bottom:.2f}–${top:.2f} ({height*100:.1f}% wide). {top_touches}+{bottom_touches} touches. Breakout expected.",
            bar_start=n - 20,
            bar_end=n - 1,
            color="#ffbb33",
            timeframe_note="Best on 4H/Daily",
            correction="Enter on retest of broken level after confirmed close outside range. Avoid entering on the initial breakout candle.",
        ))

    return matches


# ── Classic Reversal Patterns ─────────────────────────────────────────────────

def _detect_double_bottom(closes, lows, n) -> list[PatternMatch]:
    """Double Bottom (W): two lows within 4% → neckline break."""
    matches = []
    pivot_lows = _find_local_lows(lows, order=4)
    if len(pivot_lows) < 2:
        return matches

    for i in range(len(pivot_lows) - 1):
        b1, b2 = pivot_lows[i], pivot_lows[i + 1]
        if b2 - b1 < 5:
            continue
        low1, low2 = lows[b1], lows[b2]
        diff = abs(low1 - low2) / max(low1, low2)
        if diff > 0.04:
            continue

        neckline = float(np.max(closes[b1:b2 + 1]))
        if closes[-1] > neckline * 0.99 and b2 >= n - 20:
            height = (neckline - min(low1, low2)) / neckline
            conf   = min(1.0, 0.5 + height * 3)
            matches.append(PatternMatch(
                name="Double Bottom",
                bias="BULLISH",
                confidence=round(conf, 2),
                win_rate=0.88,
                description=f"Two bottoms ~${min(low1,low2):.2f}, neckline ${neckline:.2f}. Breaking out.",
                bar_start=b1,
                bar_end=n - 1,
                color="#00c851",
                timeframe_note="Best on Daily/Weekly",
                correction="Enter on neckline retest as support, not on initial breakout. Volume must expand on breakout.",
            ))

    return matches[:1]


def _detect_double_top(closes, highs, n) -> list[PatternMatch]:
    """Double Top (M): two highs within 4% → neckline break."""
    matches = []
    pivot_highs = _find_local_highs(highs, order=4)
    if len(pivot_highs) < 2:
        return matches

    for i in range(len(pivot_highs) - 1):
        h1, h2 = pivot_highs[i], pivot_highs[i + 1]
        if h2 - h1 < 5:
            continue
        high1, high2 = highs[h1], highs[h2]
        diff = abs(high1 - high2) / max(high1, high2)
        if diff > 0.04:
            continue

        neckline = float(np.min(closes[h1:h2 + 1]))
        if closes[-1] < neckline * 1.01 and h2 >= n - 20:
            height = (max(high1, high2) - neckline) / max(high1, high2)
            conf   = min(1.0, 0.5 + height * 3)
            matches.append(PatternMatch(
                name="Double Top",
                bias="BEARISH",
                confidence=round(conf, 2),
                win_rate=0.79,
                description=f"Two tops ~${max(high1,high2):.2f}, neckline ${neckline:.2f}. Breakdown risk.",
                bar_start=h1,
                bar_end=n - 1,
                color="#ff4444",
                timeframe_note="Best on Daily/4H",
                correction="Second peak must form on lower volume. Wait for full-bodied candle close below neckline.",
            ))

    return matches[:1]


def _detect_triple_bottom(lows, closes, n) -> list[PatternMatch]:
    """Triple Bottom: three lows within 3% → neckline break."""
    matches = []
    pivot_lows = _find_local_lows(lows, order=3)
    if len(pivot_lows) < 3:
        return matches

    for i in range(len(pivot_lows) - 2):
        b1, b2, b3 = pivot_lows[i], pivot_lows[i + 1], pivot_lows[i + 2]
        if b3 - b1 < 10 or b3 < n - 25:
            continue
        l1, l2, l3 = lows[b1], lows[b2], lows[b3]
        spread = (max(l1, l2, l3) - min(l1, l2, l3)) / max(l1, l2, l3)
        if spread > 0.03:
            continue

        neckline = float(np.max(closes[b1:b3 + 1]))
        if closes[-1] > neckline * 0.99:
            conf = min(1.0, 0.60 + (0.03 - spread) * 10)
            matches.append(PatternMatch(
                name="Triple Bottom",
                bias="BULLISH",
                confidence=round(conf, 2),
                win_rate=0.87,
                description=f"Three bottoms ~${min(l1,l2,l3):.2f}, neckline ${neckline:.2f}. Strong reversal signal.",
                bar_start=b1,
                bar_end=n - 1,
                color="#00c851",
                timeframe_note="Best on Daily/Weekly",
                correction="Require RSI/MACD divergence on third bottom. Accept 1–3% variation between lows.",
            ))
            break

    return matches[:1]


def _detect_triple_top(highs, closes, n) -> list[PatternMatch]:
    """Triple Top: three highs within 3% → neckline break."""
    matches = []
    pivot_highs = _find_local_highs(highs, order=3)
    if len(pivot_highs) < 3:
        return matches

    for i in range(len(pivot_highs) - 2):
        h1, h2, h3 = pivot_highs[i], pivot_highs[i + 1], pivot_highs[i + 2]
        if h3 - h1 < 10 or h3 < n - 25:
            continue
        hi1, hi2, hi3 = highs[h1], highs[h2], highs[h3]
        spread = (max(hi1, hi2, hi3) - min(hi1, hi2, hi3)) / max(hi1, hi2, hi3)
        if spread > 0.03:
            continue

        neckline = float(np.min(closes[h1:h3 + 1]))
        if closes[-1] < neckline * 1.01:
            conf = min(1.0, 0.60 + (0.03 - spread) * 10)
            matches.append(PatternMatch(
                name="Triple Top",
                bias="BEARISH",
                confidence=round(conf, 2),
                win_rate=0.87,
                description=f"Three tops ~${max(hi1,hi2,hi3):.2f}, neckline ${neckline:.2f}. Strong bearish reversal.",
                bar_start=h1,
                bar_end=n - 1,
                color="#ff4444",
                timeframe_note="Best on Daily/Weekly",
                correction="Third peak on clearly declining volume. Wait for all three peaks at similar price ±1%.",
            ))
            break

    return matches[:1]


def _detect_head_and_shoulders(highs, lows, closes, n) -> list[PatternMatch]:
    """Head & Shoulders: left shoulder < head > right shoulder → neckline break."""
    matches = []
    pivot_highs = _find_local_highs(highs, order=4)
    if len(pivot_highs) < 3:
        return matches

    for i in range(len(pivot_highs) - 2):
        ls, head, rs = pivot_highs[i], pivot_highs[i + 1], pivot_highs[i + 2]
        if rs - ls < 10:
            continue
        h_ls, h_head, h_rs = highs[ls], highs[head], highs[rs]
        if not (h_head > h_ls and h_head > h_rs):
            continue
        shoulder_diff = abs(h_ls - h_rs) / h_head
        if shoulder_diff > 0.06:
            continue

        neckline = ((float(np.min(lows[ls:head + 1])) + float(np.min(lows[head:rs + 1]))) / 2)
        if closes[-1] < neckline * 1.02 and rs >= n - 15:
            height = (h_head - neckline) / h_head
            conf   = min(1.0, 0.5 + height * 2 - shoulder_diff * 3)
            matches.append(PatternMatch(
                name="Head & Shoulders",
                bias="BEARISH",
                confidence=round(conf, 2),
                win_rate=0.83,
                description=f"Head ${h_head:.2f}, neckline ~${neckline:.2f}. Bearish reversal.",
                bar_start=ls,
                bar_end=n - 1,
                color="#ff4444",
                timeframe_note="Best on Daily/4H (intraday produces high false rate)",
                correction="Do not short until candle CLOSES below neckline. Stop above right shoulder, not head.",
            ))

    return matches[:1]


def _detect_inverse_head_and_shoulders(highs, lows, closes, n) -> list[PatternMatch]:
    """Inverse H&S: bullish reversal."""
    matches = []
    pivot_lows = _find_local_lows(lows, order=4)
    if len(pivot_lows) < 3:
        return matches

    for i in range(len(pivot_lows) - 2):
        ls, head, rs = pivot_lows[i], pivot_lows[i + 1], pivot_lows[i + 2]
        if rs - ls < 10:
            continue
        l_ls, l_head, l_rs = lows[ls], lows[head], lows[rs]
        if not (l_head < l_ls and l_head < l_rs):
            continue
        shoulder_diff = abs(l_ls - l_rs) / abs(l_head) if l_head != 0 else 1
        if shoulder_diff > 0.06:
            continue

        neckline = ((float(np.max(highs[ls:head + 1])) + float(np.max(highs[head:rs + 1]))) / 2)
        if closes[-1] > neckline * 0.98 and rs >= n - 15:
            height = (neckline - l_head) / neckline
            conf   = min(1.0, 0.5 + height * 2 - shoulder_diff * 3)
            matches.append(PatternMatch(
                name="Inv. Head & Shoulders",
                bias="BULLISH",
                confidence=round(conf, 2),
                win_rate=0.89,
                description=f"Head ${l_head:.2f}, neckline ~${neckline:.2f}. Bullish reversal.",
                bar_start=ls,
                bar_end=n - 1,
                color="#00c851",
                timeframe_note="Best on Daily/4H",
                correction="Volume must be higher on left shoulder decline and expand on neckline breakout. Wait for retest.",
            ))

    return matches[:1]


# ── Triangle / Wedge Patterns ─────────────────────────────────────────────────

def _detect_ascending_triangle(highs, lows, n) -> list[PatternMatch]:
    """Flat resistance + rising lows = ascending triangle (bullish)."""
    matches = []
    if n < 20:
        return matches

    recent_h = highs[-20:]
    recent_l = lows[-20:]

    top = float(np.max(recent_h))
    top_touches = sum(1 for h in recent_h if h >= top * 0.98)

    x = np.arange(len(recent_l))
    slope, _ = np.polyfit(x, recent_l, 1)

    if top_touches >= 2 and slope > 0 and top > 0:
        conf = min(1.0, 0.4 + top_touches * 0.1 + slope / top * 50)
        matches.append(PatternMatch(
            name="Ascending Triangle",
            bias="BULLISH",
            confidence=round(conf, 2),
            win_rate=0.83,
            description=f"Flat resistance ~${top:.2f} with rising lows ({top_touches} touches). Bullish breakout setup.",
            bar_start=n - 20,
            bar_end=n - 1,
            color="#00c851",
            timeframe_note="Best on 4H/Daily. Intraday (15m+) acceptable for futures.",
            correction="Stop below most recent higher low. Don't chase breakout >2–3% from resistance line.",
        ))

    return matches


def _detect_descending_triangle(highs, lows, n) -> list[PatternMatch]:
    """Flat support + falling highs = descending triangle (bearish)."""
    matches = []
    if n < 20:
        return matches

    recent_h = highs[-20:]
    recent_l = lows[-20:]

    bottom = float(np.min(recent_l))
    bottom_touches = sum(1 for l in recent_l if l <= bottom * 1.02)

    x = np.arange(len(recent_h))
    slope, _ = np.polyfit(x, recent_h, 1)

    if bottom_touches >= 2 and slope < 0 and bottom > 0:
        conf = min(1.0, 0.4 + bottom_touches * 0.1 + abs(slope) / bottom * 50)
        matches.append(PatternMatch(
            name="Descending Triangle",
            bias="BEARISH",
            confidence=round(conf, 2),
            win_rate=0.87,
            description=f"Flat support ~${bottom:.2f} with falling highs ({bottom_touches} touches). Bearish breakdown setup.",
            bar_start=n - 20,
            bar_end=n - 1,
            color="#ff4444",
            timeframe_note="Best on 4H/Daily",
            correction="Stop above most recent lower high. In strong bull markets, reduce size or skip.",
        ))

    return matches


def _detect_symmetrical_triangle(highs, lows, n) -> list[PatternMatch]:
    """Symmetrical triangle: converging highs and lows (continuation)."""
    matches = []
    if n < 20:
        return matches

    recent_h = highs[-20:]
    recent_l = lows[-20:]
    x = np.arange(20)

    slope_h, _ = np.polyfit(x, recent_h, 1)
    slope_l, _ = np.polyfit(x, recent_l, 1)

    # Highs falling, lows rising, converging
    if slope_h < 0 and slope_l > 0:
        convergence = abs(slope_h) + abs(slope_l)
        conf = min(1.0, 0.40 + convergence / float(np.mean(recent_h)) * 100)
        matches.append(PatternMatch(
            name="Symmetrical Triangle",
            bias="NEUTRAL",
            confidence=round(conf, 2),
            win_rate=0.65,
            description="Converging highs and lows. Breakout expected in direction of prior trend.",
            bar_start=n - 20,
            bar_end=n - 1,
            color="#ffbb33",
            timeframe_note="Best on Daily/4H",
            correction="Enter only when break occurs in first 2/3 of triangle. Near-apex breaks have 50%+ failure rate.",
        ))

    return matches


def _detect_rising_wedge(highs, lows, closes, n) -> list[PatternMatch]:
    """Rising wedge: both support and resistance rising but converging → bearish."""
    matches = []
    if n < 15:
        return matches

    recent_h = highs[-15:]
    recent_l = lows[-15:]
    x = np.arange(15)

    slope_h, _ = np.polyfit(x, recent_h, 1)
    slope_l, _ = np.polyfit(x, recent_l, 1)

    # Both rising but lows rising faster than highs (converging upward)
    if slope_h > 0 and slope_l > 0 and slope_l > slope_h:
        conf = min(1.0, 0.45 + (slope_l - slope_h) / float(np.mean(recent_l)) * 200)
        matches.append(PatternMatch(
            name="Rising Wedge",
            bias="BEARISH",
            confidence=round(conf, 2),
            win_rate=0.64,
            description="Both highs and lows rising but converging. Bearish reversal/continuation.",
            bar_start=n - 15,
            bar_end=n - 1,
            color="#ff4444",
            timeframe_note="Best on 4H/Daily",
            correction="Require RSI/MACD bearish divergence during formation. Only trade on close below lower trendline.",
        ))

    return matches


def _detect_falling_wedge(highs, lows, closes, n) -> list[PatternMatch]:
    """Falling wedge: both support and resistance falling but converging → bullish."""
    matches = []
    if n < 15:
        return matches

    recent_h = highs[-15:]
    recent_l = lows[-15:]
    x = np.arange(15)

    slope_h, _ = np.polyfit(x, recent_h, 1)
    slope_l, _ = np.polyfit(x, recent_l, 1)

    # Both falling but highs falling faster (converging downward)
    if slope_h < 0 and slope_l < 0 and slope_h < slope_l:
        conf = min(1.0, 0.45 + (slope_l - slope_h) / float(np.mean(recent_l)) * 200)
        matches.append(PatternMatch(
            name="Falling Wedge",
            bias="BULLISH",
            confidence=round(conf, 2),
            win_rate=0.71,
            description="Both highs and lows falling but converging. Bullish reversal/continuation.",
            bar_start=n - 15,
            bar_end=n - 1,
            color="#00c851",
            timeframe_note="Best on 4H/Daily",
            correction="Bullish RSI divergence on each successive lower low confirms. Target = height of wedge at widest point.",
        ))

    return matches


# ── Candlestick Patterns ──────────────────────────────────────────────────────

def _detect_engulfing(opens, closes, highs, lows, volumes, n, offset) -> list[PatternMatch]:
    """Bullish and Bearish Engulfing (2-bar pattern)."""
    matches = []
    if n < 2:
        return matches

    prev_o, prev_c = opens[-2], closes[-2]
    curr_o, curr_c = opens[-1], closes[-1]
    curr_v = volumes[-1]
    avg_v  = float(np.mean(volumes))

    prev_body = _candle_body(prev_o, prev_c)
    curr_body = _candle_body(curr_o, curr_c)

    if curr_body < prev_body * 0.5:
        return matches

    vol_boost = 0.10 if curr_v > avg_v * 1.2 else 0.0

    # Bullish engulfing: prev bearish, curr bullish + engulfs prev body
    if prev_c < prev_o and curr_c > curr_o:
        if curr_o <= prev_c and curr_c >= prev_o:
            conf = min(1.0, 0.55 + (curr_body / prev_body - 1) * 0.1 + vol_boost)
            matches.append(PatternMatch(
                name="Bullish Engulfing",
                bias="BULLISH",
                confidence=round(conf, 2),
                win_rate=0.75,
                description=f"Previous bearish bar fully engulfed by bullish bar. Strong reversal signal at support.",
                bar_start=offset + n - 2,
                bar_end=offset + n - 1,
                color="#00c851",
                timeframe_note="Best on Daily/4H",
                correction="Only trade at identifiable support zones (prior lows, VWAP, 200 EMA). Require daily close.",
            ))

    # Bearish engulfing: prev bullish, curr bearish + engulfs prev body
    if prev_c > prev_o and curr_c < curr_o:
        if curr_o >= prev_c and curr_c <= prev_o:
            conf = min(1.0, 0.55 + (curr_body / prev_body - 1) * 0.1 + vol_boost)
            matches.append(PatternMatch(
                name="Bearish Engulfing",
                bias="BEARISH",
                confidence=round(conf, 2),
                win_rate=0.65,
                description=f"Previous bullish bar fully engulfed by bearish bar. Reversal signal at resistance.",
                bar_start=offset + n - 2,
                bar_end=offset + n - 1,
                color="#ff4444",
                timeframe_note="Best on Daily/4H",
                correction="Combine with overbought RSI (>70) and negative volume divergence.",
            ))

    return matches


def _detect_hammer(opens, closes, highs, lows, n, offset) -> list[PatternMatch]:
    """Hammer (bullish) and Hanging Man (bearish) — same shape, different context."""
    matches = []
    if n < 2:
        return matches

    o, c, h, l = opens[-1], closes[-1], highs[-1], lows[-1]
    body   = _candle_body(o, c)
    rng    = _candle_range(h, l)
    if rng == 0:
        return matches

    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)

    # Hammer shape: lower wick >= 2x body, upper wick minimal
    if lower_wick < body * 2 or upper_wick > body * 0.5:
        return matches
    if body / rng < 0.1:
        return matches

    # Downtrend context (last 5 bars declining) → Hammer (bullish)
    if n >= 5:
        trend = closes[-5] - closes[-2]
        if trend > 0:   # prior downtrend
            conf = min(1.0, 0.45 + lower_wick / rng * 0.3)
            matches.append(PatternMatch(
                name="Hammer",
                bias="BULLISH",
                confidence=round(conf, 2),
                win_rate=0.63,
                description=f"Long lower wick ({lower_wick:.2f}) after downtrend. Buyers absorbed selling pressure.",
                bar_start=offset + n - 1,
                bar_end=offset + n - 1,
                color="#00c851",
                timeframe_note="Best on Daily",
                correction="Require prior downtrend of ≥5 bars. Combine with oversold RSI <30. Confirm with next bullish candle.",
            ))
        else:           # after uptrend → Hanging Man (bearish warning)
            conf = min(1.0, 0.40 + lower_wick / rng * 0.2)
            matches.append(PatternMatch(
                name="Hanging Man",
                bias="BEARISH",
                confidence=round(conf, 2),
                win_rate=0.59,
                description=f"Hammer shape after uptrend — bearish warning. Require confirmation candle.",
                bar_start=offset + n - 1,
                bar_end=offset + n - 1,
                color="#ff4444",
                timeframe_note="Best on Daily at major resistance",
                correction="Do not trade in isolation. Only at major resistance confluence. Confirm with bearish next candle.",
            ))

    return matches


def _detect_shooting_star(opens, closes, highs, lows, n, offset) -> list[PatternMatch]:
    """Shooting Star: long upper wick after uptrend → bearish reversal."""
    matches = []
    if n < 2:
        return matches

    o, c, h, l = opens[-1], closes[-1], highs[-1], lows[-1]
    body   = _candle_body(o, c)
    rng    = _candle_range(h, l)
    if rng == 0:
        return matches

    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    # Shooting star shape: upper wick >= 2x body, lower wick minimal
    if upper_wick < body * 2 or lower_wick > body * 0.5:
        return matches
    if body / rng < 0.1:
        return matches

    # Uptrend context
    if n >= 5 and closes[-2] > closes[-5]:
        conf = min(1.0, 0.45 + upper_wick / rng * 0.3)
        matches.append(PatternMatch(
            name="Shooting Star",
            bias="BEARISH",
            confidence=round(conf, 2),
            win_rate=0.65,
            description=f"Long upper wick ({upper_wick:.2f}) after uptrend. Bears rejected the highs.",
            bar_start=offset + n - 1,
            bar_end=offset + n - 1,
            color="#ff4444",
            timeframe_note="Best on Daily/4H at resistance",
            correction="Upper wick must be 2x body. Combine with bearish volume divergence.",
        ))

    return matches


def _detect_doji(opens, closes, highs, lows, n, offset) -> list[PatternMatch]:
    """Dragonfly Doji (bullish), Gravestone Doji (bearish), and Standard Doji."""
    matches = []
    if n < 1:
        return matches

    o, c, h, l = opens[-1], closes[-1], highs[-1], lows[-1]
    body = _candle_body(o, c)
    rng  = _candle_range(h, l)
    if rng == 0:
        return matches

    # Doji condition: body < 5% of total range
    if body / rng > 0.05:
        return matches

    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    if lower_wick > rng * 0.6 and upper_wick < rng * 0.1:
        # Dragonfly Doji — bullish at support
        conf = 0.50
        matches.append(PatternMatch(
            name="Dragonfly Doji",
            bias="BULLISH",
            confidence=conf,
            win_rate=0.67,
            description="Open ≈ Close ≈ High, long lower wick. Buyers absorbed selling. Bullish at support.",
            bar_start=offset + n - 1,
            bar_end=offset + n - 1,
            color="#00c851",
            timeframe_note="Best on Daily/Weekly at major lows",
            correction="Only trade at major support with oversold oscillators. Require confirmation candle.",
        ))
    elif upper_wick > rng * 0.6 and lower_wick < rng * 0.1:
        # Gravestone Doji — bearish at resistance
        conf = 0.50
        matches.append(PatternMatch(
            name="Gravestone Doji",
            bias="BEARISH",
            confidence=conf,
            win_rate=0.67,
            description="Open ≈ Close ≈ Low, long upper wick. Sellers rejected the highs. Bearish at resistance.",
            bar_start=offset + n - 1,
            bar_end=offset + n - 1,
            color="#ff4444",
            timeframe_note="Best on Daily/Weekly at major highs",
            correction="Only trade at major resistance with overbought oscillators. Require confirmation candle.",
        ))
    else:
        # Standard doji — indecision (lower confidence, context-dependent)
        conf = 0.30
        matches.append(PatternMatch(
            name="Doji",
            bias="NEUTRAL",
            confidence=conf,
            win_rate=0.52,
            description="Open ≈ Close — market indecision. Confirmation candle required before acting.",
            bar_start=offset + n - 1,
            bar_end=offset + n - 1,
            color="#aaaaaa",
            timeframe_note="Daily or Weekly only",
            correction="Doji alone has no edge. Wait for the next candle to confirm direction.",
        ))

    return matches


def _detect_morning_star(opens, closes, highs, lows, volumes, n, offset) -> list[PatternMatch]:
    """Morning Star: 3-candle bullish reversal (bearish → star → bullish)."""
    matches = []
    if n < 3:
        return matches

    o1, c1 = opens[-3], closes[-3]
    o2, c2 = opens[-2], closes[-2]
    o3, c3 = opens[-1], closes[-1]
    v3 = volumes[-1]
    avg_v = float(np.mean(volumes))

    b1 = _candle_body(o1, c1)
    b2 = _candle_body(o2, c2)
    b3 = _candle_body(o3, c3)

    if b1 == 0 or b3 == 0:
        return matches

    # Candle 1: large bearish
    if c1 >= o1:
        return matches
    # Candle 2: small body (star) gapping below candle 1 close
    if b2 > b1 * 0.4:
        return matches
    # Candle 3: bullish, closes above midpoint of candle 1
    mid1 = (o1 + c1) / 2
    if c3 <= o3 or c3 < mid1:
        return matches

    vol_boost = 0.10 if v3 > avg_v else 0.0
    conf = min(1.0, 0.55 + (b3 / b1) * 0.1 + vol_boost)
    matches.append(PatternMatch(
        name="Morning Star",
        bias="BULLISH",
        confidence=round(conf, 2),
        win_rate=0.68,
        description="3-candle bullish reversal: large bearish → small star → bullish close above midpoint.",
        bar_start=offset + n - 3,
        bar_end=offset + n - 1,
        color="#00c851",
        timeframe_note="Best on Daily/Weekly",
        correction="Require at prior support level or oversold RSI. Volume spike on 3rd candle confirms.",
    ))

    return matches


def _detect_evening_star(opens, closes, highs, lows, volumes, n, offset) -> list[PatternMatch]:
    """Evening Star: 3-candle bearish reversal (bullish → star → bearish)."""
    matches = []
    if n < 3:
        return matches

    o1, c1 = opens[-3], closes[-3]
    o2, c2 = opens[-2], closes[-2]
    o3, c3 = opens[-1], closes[-1]
    v3 = volumes[-1]
    avg_v = float(np.mean(volumes))

    b1 = _candle_body(o1, c1)
    b2 = _candle_body(o2, c2)
    b3 = _candle_body(o3, c3)

    if b1 == 0 or b3 == 0:
        return matches

    # Candle 1: large bullish
    if c1 <= o1:
        return matches
    # Candle 2: small body (star)
    if b2 > b1 * 0.4:
        return matches
    # Candle 3: bearish, closes below midpoint of candle 1
    mid1 = (o1 + c1) / 2
    if c3 >= o3 or c3 > mid1:
        return matches

    vol_boost = 0.10 if v3 > avg_v else 0.0
    conf = min(1.0, 0.58 + (b3 / b1) * 0.1 + vol_boost)
    matches.append(PatternMatch(
        name="Evening Star",
        bias="BEARISH",
        confidence=round(conf, 2),
        win_rate=0.71,
        description="3-candle bearish reversal: large bullish → small star → bearish close below midpoint.",
        bar_start=offset + n - 3,
        bar_end=offset + n - 1,
        color="#ff4444",
        timeframe_note="Best on Daily/Weekly",
        correction="Combine with overbought RSI >75 and negative MACD divergence. Appears in parabolic bull markets near ATH — fails due to momentum.",
    ))

    return matches


def _detect_harami(opens, closes, n, offset) -> list[PatternMatch]:
    """Bullish/Bearish Harami: small inside bar after large candle."""
    matches = []
    if n < 2:
        return matches

    o1, c1 = opens[-2], closes[-2]
    o2, c2 = opens[-1], closes[-1]

    b1 = _candle_body(o1, c1)
    b2 = _candle_body(o2, c2)

    if b1 == 0:
        return matches

    # Second candle body must be < 25% of first
    if b2 > b1 * 0.25:
        return matches
    # Second candle body must be inside first candle body
    if not (min(o2, c2) >= min(o1, c1) and max(o2, c2) <= max(o1, c1)):
        return matches

    if c1 < o1 and c2 > o2:  # bearish → bullish inside = bullish harami
        matches.append(PatternMatch(
            name="Bullish Harami",
            bias="BULLISH",
            confidence=0.42,
            win_rate=0.55,
            description="Small bullish inside bar after large bearish candle. Potential pause/reversal — require confirmation.",
            bar_start=offset + n - 2,
            bar_end=offset + n - 1,
            color="#00c851",
            timeframe_note="Daily only",
            correction="Do not trade as reversal alone. Use as 'watch for confirmation' signal. Edge improves to 63% with diverging oscillator at a key level.",
        ))
    elif c1 > o1 and c2 < o2:  # bullish → bearish inside = bearish harami
        matches.append(PatternMatch(
            name="Bearish Harami",
            bias="BEARISH",
            confidence=0.42,
            win_rate=0.55,
            description="Small bearish inside bar after large bullish candle. Potential pause/reversal — require confirmation.",
            bar_start=offset + n - 2,
            bar_end=offset + n - 1,
            color="#ff4444",
            timeframe_note="Daily only",
            correction="Most common candlestick false signal. In trending markets, haramis resolve in the direction of the trend 70%+ of the time.",
        ))

    return matches


# ── Fair Value Gap (FVG / Imbalance) ─────────────────────────────────────────

def _detect_fair_value_gap(opens, closes, highs, lows, n, offset) -> list[PatternMatch]:
    """
    Fair Value Gap: 3-candle imbalance where candle[i] leaves a price gap
    between candle[i-2].high and candle[i].low (bullish) or
    candle[i-2].low and candle[i].high (bearish).

    Win rate: 60-65% (edgeful.com Smart Money Concepts backtests)
    Confidence scales with gap size relative to ATR.
    Only the most recent unmitigated FVG in the lookback is returned.
    """
    matches = []
    if n < 3:
        return matches

    # Approximate ATR from recent ranges
    ranges = highs - lows
    atr = float(np.mean(ranges)) if len(ranges) > 0 else 1.0
    if atr == 0:
        return matches

    # Scan from newest to oldest, stop at first (most recent) unmitigated FVG
    for i in range(n - 1, 1, -1):
        gap_low_bull  = highs[i - 2]   # bullish FVG: gap between [i-2] high and [i] low
        gap_high_bull = lows[i]
        gap_high_bear = lows[i - 2]    # bearish FVG: gap between [i-2] low and [i] high
        gap_low_bear  = highs[i]

        # Bullish FVG: [i-2] high < [i] low — price left an upward imbalance
        if gap_low_bull < gap_high_bull:
            gap_size = gap_high_bull - gap_low_bull
            # Middle candle [i-1] must be a strong bull candle
            if closes[i - 1] > opens[i - 1]:
                conf = min(0.82, 0.55 + (gap_size / atr) * 0.15)
                matches.append(PatternMatch(
                    name="Fair Value Gap (Bull)",
                    bias="BULLISH",
                    confidence=round(conf, 2),
                    win_rate=0.63,
                    description=f"Bullish FVG: imbalance zone ${gap_low_bull:.2f}–${gap_high_bull:.2f} ({gap_size/atr:.1f}× ATR). Price may return to fill.",
                    bar_start=offset + i - 2,
                    bar_end=offset + i,
                    color="#00e5cc",
                    timeframe_note="Best on 5m–1H (Smart Money Concepts)",
                    correction="Enter on first retest of FVG zone — not on creation. FVG fails if price closes through the full gap.",
                ))
                break

        # Bearish FVG: [i-2] low > [i] high — downward imbalance
        if gap_high_bear > gap_low_bear:
            gap_size = gap_high_bear - gap_low_bear
            if closes[i - 1] < opens[i - 1]:
                conf = min(0.82, 0.55 + (gap_size / atr) * 0.15)
                matches.append(PatternMatch(
                    name="Fair Value Gap (Bear)",
                    bias="BEARISH",
                    confidence=round(conf, 2),
                    win_rate=0.63,
                    description=f"Bearish FVG: imbalance zone ${gap_low_bear:.2f}–${gap_high_bear:.2f} ({gap_size/atr:.1f}× ATR). Resistance on retest.",
                    bar_start=offset + i - 2,
                    bar_end=offset + i,
                    color="#ff6b35",
                    timeframe_note="Best on 5m–1H (Smart Money Concepts)",
                    correction="Short on first retest from below. Invalidated if price closes above the full gap.",
                ))
                break

    return matches


# ── Inside Bar / NR4 Volatility Compression ───────────────────────────────────

def _detect_inside_bar(highs, lows, closes, n, offset) -> list[PatternMatch]:
    """
    Inside Bar: current bar's high ≤ prior bar's high AND low ≥ prior bar's low.
    NR4 variant: also the narrowest range of the last 4 bars (tighter compression).

    Win rates: NR4 57.7% (Bulkowski, 29k trades), plain inside bar 54%.
    Signal fires on the LAST bar in the window.
    """
    matches = []
    if n < 2:
        return matches

    idx = n - 1
    h_cur, l_cur = highs[idx], lows[idx]
    h_prev, l_prev = highs[idx - 1], lows[idx - 1]

    # Inside bar condition
    if h_cur > h_prev or l_cur < l_prev:
        return matches

    range_cur = h_cur - l_cur
    if range_cur <= 0:
        return matches

    # NR4: narrowest range of last 4 bars
    is_nr4 = False
    if n >= 4:
        recent_ranges = highs[idx - 3: idx + 1] - lows[idx - 3: idx + 1]
        is_nr4 = bool(range_cur == float(np.min(recent_ranges)))

    # Directional bias from prior trend (last 5 bars slope)
    if n >= 6:
        prior_slope = closes[idx - 1] - closes[idx - 5]
        bias = "BULLISH" if prior_slope > 0 else "BEARISH"
    else:
        bias = "NEUTRAL"

    if is_nr4:
        conf = 0.62
        win_rate = 0.58
        name = "Inside Bar (NR4)"
        desc = f"NR4 inside bar — narrowest range of last 4 bars. Volatility squeeze, breakout imminent."
    else:
        conf = 0.50
        win_rate = 0.54
        name = "Inside Bar"
        desc = f"Inside bar: range fully within prior bar. Consolidation — trade the breakout of prior bar's high/low."

    matches.append(PatternMatch(
        name=name,
        bias=bias,
        confidence=conf,
        win_rate=win_rate,
        description=desc,
        bar_start=offset + idx - 1,
        bar_end=offset + idx,
        color="#ffd166",
        timeframe_note="Best on Daily/4H; effective on 5m with RVOL confirmation",
        correction="Enter on break of mother bar high (LONG) or low (SHORT). Stop = opposite end of mother bar. Avoid in low-volume sessions.",
    ))

    return matches
