"""
Expert Stock Market Analyzer — Chart Pattern Detector
Detects named swing trade patterns from OHLCV data using pure pandas/numpy.

Patterns detected:
  - Bull Flag
  - Bear Flag
  - Double Bottom
  - Double Top
  - Head & Shoulders (bearish)
  - Inverse Head & Shoulders (bullish)
  - Cup & Handle
  - Ascending Triangle
  - Descending Triangle
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class PatternMatch:
    name: str
    bias: str           # "BULLISH" | "BEARISH" | "NEUTRAL"
    confidence: float   # 0.0 – 1.0
    description: str
    bar_start: int      # index into df where pattern begins
    bar_end: int        # index where pattern ends (usually last bar)
    color: str          # display color


@dataclass
class PatternReport:
    patterns: list[PatternMatch] = field(default_factory=list)
    dominant_bias: str = "NEUTRAL"
    pattern_score_adj: float = 0.0   # bonus/penalty for statistician


def detect_patterns(df: pd.DataFrame) -> PatternReport:
    """
    Run all detectors on the last ~60 bars of df.
    Returns a PatternReport with all found patterns.
    """
    window = df.iloc[-60:].copy().reset_index(drop=True)
    closes  = window["Close"].values
    highs   = window["High"].values
    lows    = window["Low"].values
    n       = len(window)

    matches: list[PatternMatch] = []

    matches += _detect_bull_flag(closes, highs, lows, n)
    matches += _detect_bear_flag(closes, highs, lows, n)
    matches += _detect_double_bottom(closes, lows, n)
    matches += _detect_double_top(closes, highs, n)
    matches += _detect_head_and_shoulders(highs, lows, closes, n)
    matches += _detect_inverse_head_and_shoulders(highs, lows, closes, n)
    matches += _detect_cup_and_handle(closes, lows, n)
    matches += _detect_ascending_triangle(highs, lows, n)
    matches += _detect_descending_triangle(highs, lows, n)

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

    return PatternReport(
        patterns=sorted(matches, key=lambda m: m.confidence, reverse=True),
        dominant_bias=dominant,
        pattern_score_adj=round(score_adj, 2),
    )


# ── Individual pattern detectors ─────────────────────────────────────────────

def _find_local_highs(highs: np.ndarray, order: int = 3) -> list[int]:
    """Indices of local high pivots."""
    result = []
    for i in range(order, len(highs) - order):
        if highs[i] == max(highs[i - order: i + order + 1]):
            result.append(i)
    return result


def _find_local_lows(lows: np.ndarray, order: int = 3) -> list[int]:
    """Indices of local low pivots."""
    result = []
    for i in range(order, len(lows) - order):
        if lows[i] == min(lows[i - order: i + order + 1]):
            result.append(i)
    return result


def _detect_bull_flag(closes, highs, lows, n) -> list[PatternMatch]:
    """
    Bull Flag: sharp pole up (≥5% in ≤8 bars), then tight consolidation (≤3%).
    Breakout implied at the last bar above consolidation high.
    """
    matches = []
    if n < 20:
        return matches

    # Look for pole: rapid rise in the last 25 bars
    lookback = min(25, n - 5)
    for pole_start in range(n - lookback, n - 8):
        pole_end = pole_start + 5
        if pole_end >= n - 3:
            break
        pole_rise = (closes[pole_end] - closes[pole_start]) / closes[pole_start]
        if pole_rise < 0.05:
            continue

        # Consolidation after pole
        cons_start = pole_end
        cons_end   = min(n - 1, cons_start + 10)
        cons_range = highs[cons_start:cons_end + 1]
        cons_low_range = lows[cons_start:cons_end + 1]
        if len(cons_range) < 3:
            continue

        cons_high = float(np.max(cons_range))
        cons_low  = float(np.min(cons_low_range))
        cons_size = (cons_high - cons_low) / cons_high

        if cons_size <= 0.05 and closes[-1] >= cons_high * 0.99:
            conf = min(1.0, pole_rise * 5 + (0.05 - cons_size) * 5)
            matches.append(PatternMatch(
                name="Bull Flag",
                bias="BULLISH",
                confidence=round(conf, 2),
                description=f"Pole +{pole_rise*100:.1f}% then tight consolidation ({cons_size*100:.1f}% range). Breakout zone.",
                bar_start=pole_start,
                bar_end=n - 1,
                color="#00c851",
            ))
            break

    return matches


def _detect_bear_flag(closes, highs, lows, n) -> list[PatternMatch]:
    """Bear Flag: sharp pole down then tight consolidation."""
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
        cons_low_range = lows[cons_start:cons_end + 1]
        if len(cons_range) < 3:
            continue

        cons_high = float(np.max(cons_range))
        cons_low  = float(np.min(cons_low_range))
        cons_size = (cons_high - cons_low) / cons_high

        if cons_size <= 0.05 and closes[-1] <= cons_low * 1.01:
            conf = min(1.0, pole_drop * 5 + (0.05 - cons_size) * 5)
            matches.append(PatternMatch(
                name="Bear Flag",
                bias="BEARISH",
                confidence=round(conf, 2),
                description=f"Pole -{pole_drop*100:.1f}% then tight consolidation. Breakdown zone.",
                bar_start=pole_start,
                bar_end=n - 1,
                color="#ff4444",
            ))
            break

    return matches


def _detect_double_bottom(closes, lows, n) -> list[PatternMatch]:
    """
    Double Bottom (W shape): two lows within 3% of each other,
    separated by a peak, with price now above the neckline.
    """
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

        # Neckline = highest close between the two bottoms
        neckline = float(np.max(closes[b1:b2 + 1]))

        # Breakout: current price above neckline
        if closes[-1] > neckline * 0.99 and b2 >= n - 20:
            height  = (neckline - min(low1, low2)) / neckline
            conf    = min(1.0, 0.5 + height * 3)
            matches.append(PatternMatch(
                name="Double Bottom",
                bias="BULLISH",
                confidence=round(conf, 2),
                description=f"Two bottoms at ~${min(low1,low2):.2f}, neckline ${neckline:.2f}. Price breaking out.",
                bar_start=b1,
                bar_end=n - 1,
                color="#00c851",
            ))

    return matches[:1]   # return strongest only


def _detect_double_top(closes, highs, n) -> list[PatternMatch]:
    """Double Top (M shape): two highs within 3% of each other."""
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
                description=f"Two tops at ~${max(high1,high2):.2f}, neckline ${neckline:.2f}. Breakdown risk.",
                bar_start=h1,
                bar_end=n - 1,
                color="#ff4444",
            ))

    return matches[:1]


def _detect_head_and_shoulders(highs, lows, closes, n) -> list[PatternMatch]:
    """Head & Shoulders: left shoulder < head > right shoulder, then neckline break."""
    matches = []
    pivot_highs = _find_local_highs(highs, order=4)
    if len(pivot_highs) < 3:
        return matches

    for i in range(len(pivot_highs) - 2):
        ls, head, rs = pivot_highs[i], pivot_highs[i + 1], pivot_highs[i + 2]
        if rs - ls < 10:
            continue
        h_ls, h_head, h_rs = highs[ls], highs[head], highs[rs]
        # Head must be highest, shoulders roughly equal
        if not (h_head > h_ls and h_head > h_rs):
            continue
        shoulder_diff = abs(h_ls - h_rs) / h_head
        if shoulder_diff > 0.06:
            continue

        # Neckline = min of lows between ls-head and head-rs
        neckline_left  = float(np.min(lows[ls:head + 1]))
        neckline_right = float(np.min(lows[head:rs + 1]))
        neckline = (neckline_left + neckline_right) / 2

        if closes[-1] < neckline * 1.02 and rs >= n - 15:
            height = (h_head - neckline) / h_head
            conf   = min(1.0, 0.5 + height * 2 - shoulder_diff * 3)
            matches.append(PatternMatch(
                name="Head & Shoulders",
                bias="BEARISH",
                confidence=round(conf, 2),
                description=f"Head ${h_head:.2f}, neckline ~${neckline:.2f}. Bearish reversal pattern.",
                bar_start=ls,
                bar_end=n - 1,
                color="#ff4444",
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

        neckline = (float(np.max(highs[ls:head + 1])) + float(np.max(highs[head:rs + 1]))) / 2

        if closes[-1] > neckline * 0.98 and rs >= n - 15:
            height = (neckline - l_head) / neckline
            conf   = min(1.0, 0.5 + height * 2 - shoulder_diff * 3)
            matches.append(PatternMatch(
                name="Inv. Head & Shoulders",
                bias="BULLISH",
                confidence=round(conf, 2),
                description=f"Head ${l_head:.2f}, neckline ~${neckline:.2f}. Bullish reversal pattern.",
                bar_start=ls,
                bar_end=n - 1,
                color="#00c851",
            ))

    return matches[:1]


def _detect_cup_and_handle(closes, lows, n) -> list[PatternMatch]:
    """
    Cup & Handle: U-shaped recovery (>20 bars) followed by a small pullback handle.
    """
    matches = []
    if n < 35:
        return matches

    # Cup: look over last 40-55 bars
    cup_start = max(0, n - 55)
    cup_end   = n - 8

    if cup_end - cup_start < 20:
        return matches

    left_rim  = float(closes[cup_start])
    right_rim = float(closes[cup_end])
    cup_base  = float(np.min(lows[cup_start:cup_end + 1]))

    # Rims should be roughly equal (within 5%) and well above base
    rim_diff = abs(left_rim - right_rim) / max(left_rim, right_rim)
    cup_depth = (min(left_rim, right_rim) - cup_base) / min(left_rim, right_rim)

    if rim_diff > 0.06 or cup_depth < 0.08:
        return matches

    # Handle: last 5-8 bars should be a small pullback (≤50% of cup depth)
    handle_low  = float(np.min(lows[cup_end:n]))
    handle_high = float(np.max(closes[cup_end:n]))
    handle_drop = (right_rim - handle_low) / right_rim

    if handle_drop <= 0.0 or handle_drop > cup_depth * 0.55:
        return matches

    # Breakout: price approaching right rim
    if closes[-1] >= right_rim * 0.97:
        conf = min(1.0, 0.4 + cup_depth * 2 + (0.06 - rim_diff) * 5)
        matches.append(PatternMatch(
            name="Cup & Handle",
            bias="BULLISH",
            confidence=round(conf, 2),
            description=f"Cup base ${cup_base:.2f}, rim ~${right_rim:.2f}. Handle forming, breakout near.",
            bar_start=cup_start,
            bar_end=n - 1,
            color="#33b5e5",
        ))

    return matches


def _detect_ascending_triangle(highs, lows, n) -> list[PatternMatch]:
    """Flat top resistance + rising lows = ascending triangle (bullish)."""
    matches = []
    if n < 20:
        return matches

    recent_highs = highs[-20:]
    recent_lows  = lows[-20:]

    # Resistance: top highs within 2%
    top = float(np.max(recent_highs))
    top_touches = sum(1 for h in recent_highs if h >= top * 0.98)

    # Rising lows: fit a line
    x = np.arange(len(recent_lows))
    slope, _ = np.polyfit(x, recent_lows, 1)

    if top_touches >= 2 and slope > 0:
        conf = min(1.0, 0.4 + top_touches * 0.1 + slope / top * 50)
        matches.append(PatternMatch(
            name="Ascending Triangle",
            bias="BULLISH",
            confidence=round(conf, 2),
            description=f"Flat resistance ~${top:.2f} with rising lows. Bullish breakout setup.",
            bar_start=n - 20,
            bar_end=n - 1,
            color="#00c851",
        ))

    return matches


def _detect_descending_triangle(highs, lows, n) -> list[PatternMatch]:
    """Flat bottom support + falling highs = descending triangle (bearish)."""
    matches = []
    if n < 20:
        return matches

    recent_highs = highs[-20:]
    recent_lows  = lows[-20:]

    bottom = float(np.min(recent_lows))
    bottom_touches = sum(1 for l in recent_lows if l <= bottom * 1.02)

    x = np.arange(len(recent_highs))
    slope, _ = np.polyfit(x, recent_highs, 1)

    if bottom_touches >= 2 and slope < 0:
        conf = min(1.0, 0.4 + bottom_touches * 0.1 + abs(slope) / bottom * 50)
        matches.append(PatternMatch(
            name="Descending Triangle",
            bias="BEARISH",
            confidence=round(conf, 2),
            description=f"Flat support ~${bottom:.2f} with falling highs. Bearish breakdown setup.",
            bar_start=n - 20,
            bar_end=n - 1,
            color="#ff4444",
        ))

    return matches
