"""
ADX Regime Router
=================
Classifies the current H1 market regime using ADX(14) and determines which
of the five H1 strategies are enabled.  Also enforces session-window rules
for US and BIST markets so no signals are generated during low-edge periods.

Regime → strategy mapping (from research):

  ADX > 25  (TRENDING)      → ORB-60, Squeeze Breakout
  ADX < 20  (RANGING)       → RSI MR, VWAP Pullback, Z-Score MR
  ADX 20–25 (TRANSITIONAL)  → All five, size_factor = 0.5

Session windows (US / ET):
  Prime 1  09:30–10:30   all strategies, ORB forms here
  Mid  1   10:30–11:30   mean reversion only
  BLACKOUT 11:30–13:30   no signals (lunch — breakout failure 45-55%)
  Mid  2   13:30–15:00   mean reversion only
  Power    15:00–16:00   all strategies

Session windows (BIST / Istanbul):
  Session 1  10:00–13:00  all strategies
  BLACKOUT   13:00–14:00  no signals (mid-session exchange break)
  Session 2  14:00–18:00  all strategies
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import utils.ta_compat as ta


# ── Regime thresholds ─────────────────────────────────────────────────────────
ADX_TRENDING    = 25
ADX_RANGING     = 20

# ── Strategy names ────────────────────────────────────────────────────────────
TRENDING_STRATEGIES  = ["ORB_60", "SQUEEZE", "ABSORB_BO", "TRIPLE_A"]
RANGING_STRATEGIES   = ["RSI_MR", "VWAP_PB", "ZSCORE_MR", "VA_BOUNCE"]
MR_STRATEGIES        = {"RSI_MR", "VWAP_PB", "ZSCORE_MR", "VA_BOUNCE"}  # set for O(1) lookup
ALL_STRATEGIES       = ["RSI_MR", "VWAP_PB", "ORB_60", "SQUEEZE", "ZSCORE_MR",
                         "ABSORB_BO", "TRIPLE_A", "VA_BOUNCE"]

# ── Timezones ─────────────────────────────────────────────────────────────────
_US_TZ   = ZoneInfo("America/New_York")
_BIST_TZ = ZoneInfo("Europe/Istanbul")

# ── Session window tables ─────────────────────────────────────────────────────
# Each entry: (start_hour, start_min, end_hour, end_min, mr_only)
# mr_only=True  → only mean-reversion strategies allowed in that window
# mr_only=False → all regime-appropriate strategies allowed
_US_WINDOWS = [
    (9,  30, 10, 30, False),   # prime 1 — ORB forms
    (10, 30, 11, 30, True),    # mid 1  — mean reversion only
    (13, 30, 15,  0, True),    # mid 2  — mean reversion only
    (15,  0, 16,  0, False),   # power hour
]
_BIST_WINDOWS = [
    (10,  0, 13,  0, False),   # session 1
    (14,  0, 18,  0, False),   # session 2
]

# Blackout periods: (start_hour, start_min, end_hour, end_min, label)
_US_BLACKOUT   = [(11, 30, 13, 30, "US lunch 11:30–13:30 ET")]
_BIST_BLACKOUT = [(13,  0, 14,  0, "BIST mid-session break 13:00–14:00")]


@dataclass
class RegimeResult:
    regime:             str           # "TRENDING" | "RANGING" | "TRANSITIONAL"
    adx:                float         # raw ADX(14) value
    enabled_strategies: list[str]     # strategies that may fire this bar
    size_factor:        float         # 1.0 normally; 0.5 in transitional regime
    in_session_window:  bool          # False → suppress all signals
    mr_only_window:     bool          # True → only MR strategies allowed
    window_note:        str           # human-readable explanation
    market:             str           # "US" | "BIST"


def compute_regime(df_h1: pd.DataFrame) -> RegimeResult:
    """
    Compute the ADX regime and session-window status for the last bar in df_h1.

    df_h1 must include the 'market' column produced by fetch_hourly_data().
    """
    market = str(df_h1["market"].iloc[-1]) if "market" in df_h1.columns else "US"

    # ── ADX(14) on H1 ─────────────────────────────────────────────────────────
    adx_s   = ta.adx(df_h1["High"], df_h1["Low"], df_h1["Close"], length=14)
    adx_val = 0.0
    if adx_s is not None and not adx_s.isna().all():
        v = float(adx_s.iloc[-1])
        adx_val = 0.0 if np.isnan(v) else v

    # ── Regime classification ─────────────────────────────────────────────────
    if adx_val > ADX_TRENDING:
        regime      = "TRENDING"
        enabled     = list(TRENDING_STRATEGIES)
        size_factor = 1.0
    elif adx_val < ADX_RANGING:
        regime      = "RANGING"
        enabled     = list(RANGING_STRATEGIES)
        size_factor = 1.0
    else:
        regime      = "TRANSITIONAL"
        enabled     = list(ALL_STRATEGIES)
        size_factor = 0.5

    # ── Session window ────────────────────────────────────────────────────────
    last_bar = df_h1.index[-1]
    in_win, mr_only, win_note = _check_session_window(last_bar, market)

    # In MR-only windows further restrict the enabled list
    if mr_only:
        enabled = [s for s in enabled if s in MR_STRATEGIES]

    return RegimeResult(
        regime=regime,
        adx=round(adx_val, 1),
        enabled_strategies=enabled,
        size_factor=size_factor,
        in_session_window=in_win,
        mr_only_window=mr_only,
        window_note=win_note,
        market=market,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _check_session_window(
    dt: "pd.Timestamp | datetime",
    market: str,
) -> tuple[bool, bool, str]:
    """
    Returns (in_window, mr_only, note).

    in_window — False means the caller must suppress all signals.
    mr_only   — True means only mean-reversion strategies may fire.
    """
    tz    = _US_TZ if market == "US" else _BIST_TZ
    local = dt.astimezone(tz) if hasattr(dt, "astimezone") else dt
    hhmm  = local.hour * 60 + local.minute

    blackout  = _US_BLACKOUT   if market == "US" else _BIST_BLACKOUT
    windows   = _US_WINDOWS    if market == "US" else _BIST_WINDOWS

    # ── Blackout check (highest priority) ─────────────────────────────────────
    for (bh, bm, eh, em, label) in blackout:
        if bh * 60 + bm <= hhmm < eh * 60 + em:
            return False, False, f"Blackout — {label}: all signals suppressed"

    # ── Active window check ───────────────────────────────────────────────────
    for (sh, sm, eh, em, mr_only) in windows:
        start, end = sh * 60 + sm, eh * 60 + em
        if start <= hhmm < end:
            tag = "mean-reversion only" if mr_only else "all strategies"
            return True, mr_only, f"{sh:02d}:{sm:02d}–{eh:02d}:{em:02d} ({tag})"

    return False, False, f"Outside all session windows for {market}"
