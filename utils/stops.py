"""Structure-aware stop placement.

The Round 4 forensic on 86 closed trades showed avg_win/avg_loss = 0.745 —
TPs miss while SLs hit. Root cause: every strategy uses a flat ATR multiplier
for SL regardless of where the recent swing pivot is. A 1.0×ATR stop placed
ABOVE a recent swing high gets sniped on the noise wick that retraces back
into the structure.

This module computes a structure-aware stop:

  LONG:  SL = min(recent_swing_low_lookback, entry − atr_floor) − buffer
  SHORT: SL = max(recent_swing_high_lookback, entry + atr_floor) + buffer

`atr_floor` ensures we never place a stop *closer* than the strategy intended,
so very tight pivots don't produce a 0.1×ATR stop that gets hit on noise.
A `max_stop_distance` cap prevents pathological cases where a deep swing
pivot would create an absurd 5-10×ATR risk.

Used by `_make_signal` in `agents/intraday_strategies.py` (Round 4 B1) and by
the trade manager's runner-trail logic in `agents/trade_manager.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# Per-strategy defaults — strategies can override by passing kwargs to
# `compute_structural_stop`. These match the heuristics from the Round 4
# audit: minimum stop distance ≥0.5×ATR, maximum ≤2.5×ATR.
DEFAULT_LOOKBACK_BARS = 20
DEFAULT_BUFFER_ATR    = 0.30
DEFAULT_MIN_DIST_ATR  = 0.50
DEFAULT_MAX_DIST_ATR  = 2.50
DEFAULT_TP_R_MULT     = 2.00     # take-profit at 2R (1:2 R:R)


@dataclass
class StructuralStop:
    """Result of compute_structural_stop()."""
    sl:                    float
    tp:                    float
    risk:                  float    # entry-to-sl distance in price points
    one_r:                 float    # alias for risk; used by partial-TP at 1R
    sl_source:             str      # "swing" | "atr_floor" | "atr_max"
    pivot_price:           float    # the swing pivot used (or 0 if atr-floored)
    pivot_bars_back:       int      # how far back the pivot is (or 0)


def compute_structural_stop(
    df: pd.DataFrame,
    direction: str,
    entry: float,
    atr: float,
    *,
    lookback_bars: int = DEFAULT_LOOKBACK_BARS,
    buffer_atr:    float = DEFAULT_BUFFER_ATR,
    min_dist_atr:  float = DEFAULT_MIN_DIST_ATR,
    max_dist_atr:  float = DEFAULT_MAX_DIST_ATR,
    tp_r_mult:     float = DEFAULT_TP_R_MULT,
) -> StructuralStop:
    """Compute a structure-aware SL + TP for the supplied entry.

    Args:
        df:           OHLCV DataFrame; uses High/Low for pivot detection.
        direction:    "LONG" | "SHORT".
        entry:        intended entry price.
        atr:          current ATR — drives the buffer and floor/max bounds.
        lookback_bars: how many recent bars to scan for the swing pivot.
        buffer_atr:   wedge between pivot and SL, expressed in ATR.
        min_dist_atr: hard floor on entry-to-SL distance.
        max_dist_atr: hard cap on entry-to-SL distance.
        tp_r_mult:    take-profit at this many R from entry.

    Returns:
        StructuralStop populated with the chosen SL/TP and rationale.
    """
    if atr <= 0 or len(df) < lookback_bars + 1:
        # Degenerate inputs — fall back to a pure-ATR stop the caller can use.
        risk = max(min_dist_atr, 1.0) * max(atr, 0.0)
        if direction == "LONG":
            sl = entry - risk
            tp = entry + tp_r_mult * risk
        else:
            sl = entry + risk
            tp = entry - tp_r_mult * risk
        return StructuralStop(
            sl=round(sl, 5), tp=round(tp, 5), risk=round(risk, 5),
            one_r=round(risk, 5), sl_source="atr_floor",
            pivot_price=0.0, pivot_bars_back=0,
        )

    # Look at the most recent `lookback_bars`, EXCLUDING the current (last)
    # bar — the strategy's entry decision is based on the closed-bar set.
    window  = df.iloc[-(lookback_bars + 1):-1]
    buffer  = buffer_atr * atr
    min_dist = min_dist_atr * atr
    max_dist = max_dist_atr * atr

    if direction == "LONG":
        pivot_idx = int(window["Low"].values.argmin())
        pivot_price = float(window["Low"].iloc[pivot_idx])
        candidate_sl = pivot_price - buffer
        sl_distance = entry - candidate_sl
        sl_source = "swing"
        pivot_bars_back = lookback_bars - pivot_idx

        if sl_distance < min_dist:
            sl_distance = min_dist
            sl_source = "atr_floor"
            pivot_bars_back = 0
            pivot_price = 0.0
        elif sl_distance > max_dist:
            sl_distance = max_dist
            sl_source = "atr_max"
        sl = entry - sl_distance
        tp = entry + tp_r_mult * sl_distance

    else:   # SHORT
        pivot_idx = int(window["High"].values.argmax())
        pivot_price = float(window["High"].iloc[pivot_idx])
        candidate_sl = pivot_price + buffer
        sl_distance = candidate_sl - entry
        sl_source = "swing"
        pivot_bars_back = lookback_bars - pivot_idx

        if sl_distance < min_dist:
            sl_distance = min_dist
            sl_source = "atr_floor"
            pivot_bars_back = 0
            pivot_price = 0.0
        elif sl_distance > max_dist:
            sl_distance = max_dist
            sl_source = "atr_max"
        sl = entry + sl_distance
        tp = entry - tp_r_mult * sl_distance

    return StructuralStop(
        sl=round(sl, 5), tp=round(tp, 5),
        risk=round(sl_distance, 5), one_r=round(sl_distance, 5),
        sl_source=sl_source,
        pivot_price=round(pivot_price, 5),
        pivot_bars_back=pivot_bars_back,
    )
