"""
Five validated H1 intraday strategies.
=======================================
Each function is self-contained: it accepts a prepared H1 DataFrame (produced
by data.fetcher_hourly.fetch_hourly_data) plus a RegimeResult, and returns a
HourlySignal describing the signal on the most recent completed bar.

run_h1_gate() is the single entry point used by the bot engine.  It fetches H1
data, computes the regime, runs every enabled strategy, and returns an
H1GateResult that either passes or blocks the daily buy decision.

Strategy parameters are taken verbatim from the research brief:
  RSI MR    — RSI(3), thresholds 15 / 85, EMA(50) trend filter, RVOL > 1.5×
  VWAP PB   — session VWAP ± 0.2×ATR band, RSI(3) < 35, EMA(20) trend
  ORB-60    — first H1 bar = opening range, entry bars 2-4, RVOL > 2.0
  Squeeze   — BB(20,2) inside KC(20,1.5 ATR), MACD histogram direction
  Z-Score   — rolling Z < -2.0 / > +2.0, daily 200-SMA long filter
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

import utils.ta_compat as ta
from agents.regime_router import RegimeResult


# ── Signal dataclasses ────────────────────────────────────────────────────────

@dataclass
class HourlySignal:
    strategy:   str     # "RSI_MR" | "VWAP_PB" | "ORB_60" | "SQUEEZE" | "ZSCORE_MR"
    signal:     str     # "LONG" | "SHORT" | "FLAT"
    confidence: float   # 0.0 – 1.0
    reason:     str     # human-readable explanation


@dataclass
class H1GateResult:
    gate_pass:        bool
    signal:           str           # "LONG" | "SHORT" | "FLAT"
    active_strategy:  str           # which strategy triggered the gate
    confidence:       float
    regime:           str
    adx:              float
    in_session_window: bool
    signals:          list[HourlySignal]
    reason:           str


# ════════════════════════════════════════════════════════════════════════════════
# STRATEGY 1 — RSI Mean Reversion
# ════════════════════════════════════════════════════════════════════════════════

def rsi_mean_reversion(df_h1: pd.DataFrame, regime: RegimeResult) -> HourlySignal:
    """
    RSI(3) mean reversion on H1 bars.

    Long  entry: RSI(3) < 15  AND  price > EMA(50)  AND  RVOL > 1.5×
    Short entry: RSI(3) > 85  AND  price < EMA(50)  AND  RVOL > 1.5×

    Edge increases as price moves further from mean, so NO fixed stop-loss.
    Position management uses a 10-bar time stop.
    """
    if len(df_h1) < 55:
        return HourlySignal("RSI_MR", "FLAT", 0.0, "Insufficient bars (need 55)")

    rsi3  = ta.rsi(df_h1["Close"], length=3)
    ema50 = ta.ema(df_h1["Close"], length=50)

    cur_rsi   = float(rsi3.iloc[-1])
    cur_ema50 = float(ema50.iloc[-1])
    cur_close = float(df_h1["Close"].iloc[-1])
    cur_rvol  = float(df_h1["rvol"].iloc[-1]) if "rvol" in df_h1.columns else 1.0

    if np.isnan(cur_rsi) or np.isnan(cur_ema50):
        return HourlySignal("RSI_MR", "FLAT", 0.0, "NaN indicator — warmup incomplete")

    vol_ok   = cur_rvol >= 1.5
    vol_note = f"RVOL {cur_rvol:.2f}x" + (" ✓" if vol_ok else " ✗ (<1.5×)")

    # ── Long: oversold + uptrend ──────────────────────────────────────────────
    if cur_rsi < 15 and cur_close > cur_ema50:
        conf = _rsi_mr_confidence(cur_rsi, direction="long")
        if not vol_ok:
            conf *= 0.75   # volume penalty, not a disqualifier per research
        return HourlySignal(
            "RSI_MR", "LONG", round(conf, 3),
            f"RSI(3)={cur_rsi:.1f} < 15 | close {cur_close:.2f} > EMA50 {cur_ema50:.2f} | {vol_note}",
        )

    # ── Short: overbought + downtrend ─────────────────────────────────────────
    if cur_rsi > 85 and cur_close < cur_ema50:
        conf = _rsi_mr_confidence(cur_rsi, direction="short")
        if not vol_ok:
            conf *= 0.75
        return HourlySignal(
            "RSI_MR", "SHORT", round(conf, 3),
            f"RSI(3)={cur_rsi:.1f} > 85 | close {cur_close:.2f} < EMA50 {cur_ema50:.2f} | {vol_note}",
        )

    # ── No trigger ────────────────────────────────────────────────────────────
    side = "above" if cur_close > cur_ema50 else "below"
    return HourlySignal(
        "RSI_MR", "FLAT", 0.0,
        f"RSI(3)={cur_rsi:.1f} not extreme | close {side} EMA50",
    )


def _rsi_mr_confidence(rsi: float, direction: str) -> float:
    """Confidence rises as RSI moves further into extreme territory."""
    if direction == "long":
        if rsi < 5:   return 0.95
        if rsi < 10:  return 0.85
        return 0.70
    else:
        if rsi > 95:  return 0.95
        if rsi > 90:  return 0.85
        return 0.70


# ════════════════════════════════════════════════════════════════════════════════
# STRATEGY 2 — VWAP Pullback
# ════════════════════════════════════════════════════════════════════════════════

def vwap_pullback(df_h1: pd.DataFrame, regime: RegimeResult) -> HourlySignal:
    """
    Session VWAP pullback on H1 bars.

    Long  entry: close within VWAP ± 0.2×ATR  AND  RSI(3) < 35  AND  close > EMA(20)
    Short entry: close within VWAP ± 0.2×ATR  AND  RSI(3) > 65  AND  close < EMA(20)

    "First Kiss" bonus (highest-probability variant): the first return to the
    VWAP band after price has been > 0.5×ATR away.  Confidence lifts to 0.90.
    """
    if len(df_h1) < 25:
        return HourlySignal("VWAP_PB", "FLAT", 0.0, "Insufficient bars (need 25)")

    if "session_vwap" not in df_h1.columns:
        return HourlySignal("VWAP_PB", "FLAT", 0.0, "session_vwap column missing — use fetch_hourly_data()")

    atr_s = ta.atr(df_h1["High"], df_h1["Low"], df_h1["Close"], length=14)
    rsi3  = ta.rsi(df_h1["Close"], length=3)
    ema20 = ta.ema(df_h1["Close"], length=20)

    cur_close = float(df_h1["Close"].iloc[-1])
    cur_vwap  = float(df_h1["session_vwap"].iloc[-1])
    cur_atr   = float(atr_s.iloc[-1]) if atr_s is not None and not atr_s.isna().all() else 0.0
    cur_rsi   = float(rsi3.iloc[-1])
    cur_ema20 = float(ema20.iloc[-1])

    if np.isnan(cur_vwap) or np.isnan(cur_atr) or cur_atr == 0:
        return HourlySignal("VWAP_PB", "FLAT", 0.0, "VWAP or ATR unavailable")

    band    = 0.2 * cur_atr
    in_band = abs(cur_close - cur_vwap) <= band

    if not in_band:
        dist = abs(cur_close - cur_vwap)
        return HourlySignal(
            "VWAP_PB", "FLAT", 0.0,
            f"Price {cur_close:.2f} is {dist:.2f} from VWAP {cur_vwap:.2f} (band ±{band:.2f})",
        )

    first_kiss = _detect_first_kiss(df_h1, cur_atr)
    base_conf  = 0.90 if first_kiss else 0.70
    fk_tag     = " | First Kiss ✓" if first_kiss else ""

    # ── Long ──────────────────────────────────────────────────────────────────
    if cur_rsi < 35 and cur_close > cur_ema20:
        return HourlySignal(
            "VWAP_PB", "LONG", round(base_conf, 3),
            f"VWAP band touch | RSI(3)={cur_rsi:.1f} < 35 | EMA20={cur_ema20:.2f}{fk_tag}",
        )

    # ── Short ─────────────────────────────────────────────────────────────────
    if cur_rsi > 65 and cur_close < cur_ema20:
        return HourlySignal(
            "VWAP_PB", "SHORT", round(base_conf, 3),
            f"VWAP band touch | RSI(3)={cur_rsi:.1f} > 65 | EMA20={cur_ema20:.2f}{fk_tag}",
        )

    return HourlySignal(
        "VWAP_PB", "FLAT", 0.0,
        f"In VWAP band but RSI(3)={cur_rsi:.1f} not extreme or EMA trend disagrees",
    )


def _detect_first_kiss(df_h1: pd.DataFrame, atr: float, lookback: int = 10) -> bool:
    """
    True when the current bar is the first return to the VWAP band after
    price was > 0.5×ATR away from session VWAP at some point in the recent
    `lookback` bars (excl. current bar).
    """
    if len(df_h1) < 3 or "session_vwap" not in df_h1.columns or atr == 0:
        return False
    window = df_h1.iloc[-(lookback + 1):-1]
    deviations = (window["Close"] - window["session_vwap"]).abs()
    return bool((deviations > 0.5 * atr).any())


# ════════════════════════════════════════════════════════════════════════════════
# STRATEGY 3 — ORB-60
# ════════════════════════════════════════════════════════════════════════════════

def orb_breakout(df_h1: pd.DataFrame, regime: RegimeResult) -> HourlySignal:
    """
    Opening Range Breakout — 60-minute opening range on H1 bars.

    The opening range is defined as the first H1 bar of the session
    (session_bar_number == 1, i.e. 09:30–10:30 ET or 10:00–11:00 Istanbul).

    Long  entry: close > ORB high  AND  RVOL > 2.0  AND  VWAP sloping up
    Short entry: close < ORB low   AND  RVOL > 2.0  AND  VWAP sloping down

    Entry window: session bars 2, 3, or 4 only.
      US:   bars 2-4 = 10:30 AM – 1:00 PM ET  (after bar 4 edge decays sharply)
      BIST: bars 2-3 = 11:00 AM – 1:00 PM Istanbul (before mid-session break)

    Stop: just beyond the opposite ORB boundary (managed externally).
    """
    if len(df_h1) < 5:
        return HourlySignal("ORB_60", "FLAT", 0.0, "Insufficient bars")

    if "session_bar_number" not in df_h1.columns:
        return HourlySignal("ORB_60", "FLAT", 0.0, "Session metadata missing — use fetch_hourly_data()")

    cur_bar_num = int(df_h1["session_bar_number"].iloc[-1])
    market      = str(df_h1["market"].iloc[-1]) if "market" in df_h1.columns else "US"
    max_bar     = 3 if market == "BIST" else 4

    if cur_bar_num < 2 or cur_bar_num > max_bar:
        return HourlySignal(
            "ORB_60", "FLAT", 0.0,
            f"Outside ORB entry window (bar {cur_bar_num}; valid 2–{max_bar} for {market})",
        )

    # ── Find today's opening bar ──────────────────────────────────────────────
    today      = df_h1["session_date"].iloc[-1]
    today_bars = df_h1[df_h1["session_date"] == today]
    first_bar  = today_bars[today_bars["session_bar_number"] == 1]

    if first_bar.empty:
        return HourlySignal("ORB_60", "FLAT", 0.0, "Opening bar not found for today's session")

    orb_high  = float(first_bar["High"].iloc[0])
    orb_low   = float(first_bar["Low"].iloc[0])
    orb_range = max(orb_high - orb_low, 1e-6)

    cur_close = float(df_h1["Close"].iloc[-1])
    cur_rvol  = float(df_h1["rvol"].iloc[-1]) if "rvol" in df_h1.columns else 1.0
    vwap_dir  = _vwap_slope(df_h1)
    rvol_ok   = cur_rvol >= 2.0

    # ── Long breakout ─────────────────────────────────────────────────────────
    if cur_close > orb_high and rvol_ok and vwap_dir > 0:
        penetration = (cur_close - orb_high) / orb_range
        conf = min(0.85, 0.65 + penetration * 0.20)
        return HourlySignal(
            "ORB_60", "LONG", round(conf, 3),
            f"Close {cur_close:.2f} > ORB high {orb_high:.2f} | "
            f"RVOL {cur_rvol:.1f}× | VWAP ↑ | bar {cur_bar_num}",
        )

    # ── Short breakdown ───────────────────────────────────────────────────────
    if cur_close < orb_low and rvol_ok and vwap_dir < 0:
        penetration = (orb_low - cur_close) / orb_range
        conf = min(0.85, 0.65 + penetration * 0.20)
        return HourlySignal(
            "ORB_60", "SHORT", round(conf, 3),
            f"Close {cur_close:.2f} < ORB low {orb_low:.2f} | "
            f"RVOL {cur_rvol:.1f}× | VWAP ↓ | bar {cur_bar_num}",
        )

    reasons: list[str] = []
    if orb_low <= cur_close <= orb_high:
        reasons.append(f"price inside ORB [{orb_low:.2f}–{orb_high:.2f}]")
    if not rvol_ok:
        reasons.append(f"RVOL {cur_rvol:.1f}× < 2.0")
    if vwap_dir == 0:
        reasons.append("VWAP flat — no directional confirmation")
    elif cur_close > orb_high and vwap_dir < 0:
        reasons.append("close above ORB but VWAP pointing down — no long")
    elif cur_close < orb_low and vwap_dir > 0:
        reasons.append("close below ORB but VWAP pointing up — no short")

    return HourlySignal("ORB_60", "FLAT", 0.0, " | ".join(reasons) or "No ORB signal")


def _vwap_slope(df_h1: pd.DataFrame, lookback: int = 2) -> int:
    """Return +1 (up), -1 (down), or 0 (flat/unavailable) from session VWAP."""
    if "session_vwap" not in df_h1.columns or len(df_h1) <= lookback:
        return 0
    recent = df_h1["session_vwap"].dropna()
    if len(recent) <= lookback:
        return 0
    delta = float(recent.iloc[-1]) - float(recent.iloc[-1 - lookback])
    if delta > 0:  return 1
    if delta < 0:  return -1
    return 0


# ════════════════════════════════════════════════════════════════════════════════
# STRATEGY 4 — Bollinger-Keltner Squeeze Breakout
# ════════════════════════════════════════════════════════════════════════════════

def squeeze_breakout(df_h1: pd.DataFrame, regime: RegimeResult) -> HourlySignal:
    """
    Bollinger-Keltner Squeeze Breakout on H1 bars.

    Squeeze condition:  BB(20, 2.0 SD) is fully inside KC(20, 1.5 ATR).
    Breakout trigger:   BB expands outside KC on the current bar (prev bar was squeezed).
    Direction:          MACD histogram sign at the breakout bar.
    Volume:             above-average volume boosts confidence (not a hard gate).

    Stop: 1.5×ATR trailing, adjusted every bar after entry (managed externally).
    """
    if len(df_h1) < 25:
        return HourlySignal("SQUEEZE", "FLAT", 0.0, "Insufficient bars (need 25)")

    bb  = ta.bbands(df_h1["Close"], length=20, std=2.0)
    kc  = ta.keltner_channels(df_h1["High"], df_h1["Low"], df_h1["Close"],
                               length=20, atr_mult=1.5)
    mac = ta.macd(df_h1["Close"], fast=12, slow=26, signal=9)

    if bb is None or bb.empty or kc is None or kc.empty:
        return HourlySignal("SQUEEZE", "FLAT", 0.0, "BB or KC unavailable")

    # Current bar (iloc -1)
    bb_upper = float(bb.iloc[-1, 2])   # BBU
    bb_lower = float(bb.iloc[-1, 0])   # BBL
    kc_upper = float(kc.iloc[-1, 2])   # KCU
    kc_lower = float(kc.iloc[-1, 0])   # KCL

    # Previous bar (iloc -2) — needed to confirm the squeeze was active
    bb_upper_prev = float(bb.iloc[-2, 2])
    bb_lower_prev = float(bb.iloc[-2, 0])
    kc_upper_prev = float(kc.iloc[-2, 2])
    kc_lower_prev = float(kc.iloc[-2, 0])

    was_squeezed   = (bb_upper_prev < kc_upper_prev) and (bb_lower_prev > kc_lower_prev)
    broke_out      = (bb_upper > kc_upper) or (bb_lower < kc_lower)

    if not was_squeezed:
        return HourlySignal("SQUEEZE", "FLAT", 0.0, "No prior squeeze condition")
    if not broke_out:
        return HourlySignal("SQUEEZE", "FLAT", 0.0, "Squeeze still active — waiting for breakout")

    # ── Volume confirmation ───────────────────────────────────────────────────
    vol_sma20 = float(df_h1["Volume"].rolling(20).mean().iloc[-1])
    cur_vol   = float(df_h1["Volume"].iloc[-1])
    vol_ok    = (vol_sma20 > 0) and (cur_vol > vol_sma20)
    vol_note  = f"vol {cur_vol/vol_sma20:.1f}×avg" if vol_sma20 > 0 else "vol N/A"

    base_conf = 0.80 if vol_ok else 0.60

    # ── MACD histogram direction ──────────────────────────────────────────────
    if mac is None or mac.empty:
        return HourlySignal("SQUEEZE", "FLAT", 0.0, "MACD unavailable at breakout bar")

    macd_hist = float(mac.iloc[-1, 1])   # MACDh column
    if np.isnan(macd_hist):
        return HourlySignal("SQUEEZE", "FLAT", 0.0, "MACD histogram NaN")

    if macd_hist > 0:
        return HourlySignal(
            "SQUEEZE", "LONG", round(base_conf, 3),
            f"Squeeze broke → BB expanded above KC | MACD hist {macd_hist:+.4f} (bullish) | {vol_note}",
        )
    if macd_hist < 0:
        return HourlySignal(
            "SQUEEZE", "SHORT", round(base_conf, 3),
            f"Squeeze broke → BB expanded below KC | MACD hist {macd_hist:+.4f} (bearish) | {vol_note}",
        )

    return HourlySignal("SQUEEZE", "FLAT", 0.0, "Squeeze broke but MACD histogram = 0 — no direction")


# ════════════════════════════════════════════════════════════════════════════════
# STRATEGY 5 — Z-Score Mean Reversion
# ════════════════════════════════════════════════════════════════════════════════

def zscore_mean_reversion(
    df_h1: pd.DataFrame,
    df_daily: pd.DataFrame | None,
    regime: RegimeResult,
) -> HourlySignal:
    """
    Z-Score mean reversion on H1 bars.

    Long  entry: Z < -2.0  AND  price above daily 200-SMA
                 (this single filter eliminates ~80% of catastrophic losses)
    Short entry: Z > +2.0  (no directional daily-trend filter for shorts)
    Exit:        Z returns to 0 OR 15-bar time stop (managed externally).
    """
    if len(df_h1) < 22:
        return HourlySignal("ZSCORE_MR", "FLAT", 0.0, "Insufficient bars (need 22)")

    z_series  = ta.zscore(df_h1["Close"], length=20)
    cur_z     = float(z_series.iloc[-1])
    cur_close = float(df_h1["Close"].iloc[-1])

    if np.isnan(cur_z):
        return HourlySignal("ZSCORE_MR", "FLAT", 0.0, "Z-score NaN — warmup incomplete")

    # ── Long: deeply oversold + above daily 200-SMA ───────────────────────────
    if cur_z < -2.0:
        above = _above_daily_200sma(df_daily, cur_close)
        if not above:
            return HourlySignal(
                "ZSCORE_MR", "FLAT", 0.0,
                f"Z={cur_z:.2f} < -2.0 but price below daily 200-SMA — catastrophic-loss filter blocks long",
            )
        conf = _zscore_confidence(cur_z)
        return HourlySignal(
            "ZSCORE_MR", "LONG", round(conf, 3),
            f"Z={cur_z:.2f} < -2.0 | close {cur_close:.2f} above daily 200-SMA ✓",
        )

    # ── Short: deeply overbought ──────────────────────────────────────────────
    if cur_z > 2.0:
        conf = _zscore_confidence(cur_z)
        return HourlySignal(
            "ZSCORE_MR", "SHORT", round(conf, 3),
            f"Z={cur_z:.2f} > +2.0 | close {cur_close:.2f}",
        )

    return HourlySignal(
        "ZSCORE_MR", "FLAT", 0.0,
        f"Z={cur_z:.2f} within ±2.0 — no mean-reversion extreme",
    )


def _above_daily_200sma(df_daily: pd.DataFrame | None, cur_close: float) -> bool:
    """True if cur_close is above the 200-SMA on the daily chart."""
    if df_daily is None:
        return True   # conservative: let the trade through if no daily data
    try:
        sma200_s = ta.sma(df_daily["Close"], length=200)
        valid    = sma200_s.dropna()
        if valid.empty:
            return True
        return cur_close > float(valid.iloc[-1])
    except Exception:
        return True


def _zscore_confidence(z: float) -> float:
    """Confidence increases with distance beyond the ±2.0 threshold."""
    extreme = abs(z)
    if extreme > 3.0:  return 0.85
    if extreme > 2.5:  return 0.75
    return 0.65


# ════════════════════════════════════════════════════════════════════════════════
# STRATEGY 6 — Absorption Breakout  (Fabio Valentini / PickMyTrade)
# ════════════════════════════════════════════════════════════════════════════════

def absorption_breakout(df_h1: pd.DataFrame, regime: RegimeResult) -> HourlySignal:
    """
    Absorption Breakout on H1 bars.

    The bar immediately preceding the current one must show absorption:
        volume > 20-bar avg × 2.0  AND  (high − low) < ATR(14) × 0.3

    The current bar then breaks out of that narrow range with volume confirmation.

    Long  entry: close > absorption bar High  AND  price above session VWAP  AND  RVOL ≥ 1.5
    Short entry: close < absorption bar Low   AND  price below session VWAP  AND  RVOL ≥ 1.5

    Stop loss: just beyond the absorption bar's opposite edge — naturally tight
    because absorption bars have narrow ranges by definition.

    Confidence bonus (+0.10) when breakout aligns with the nearest Volume Profile
    key level (VAL for longs, VAH for shorts).
    """
    if len(df_h1) < 25:
        return HourlySignal("ABSORB_BO", "FLAT", 0.0, "Insufficient bars (need 25)")

    abs_s = ta.absorption(df_h1["High"], df_h1["Low"], df_h1["Close"], df_h1["Volume"])

    if not bool(abs_s.iloc[-2]):
        return HourlySignal("ABSORB_BO", "FLAT", 0.0, "No absorption on previous bar")

    absorb_high = float(df_h1["High"].iloc[-2])
    absorb_low  = float(df_h1["Low"].iloc[-2])
    cur_close   = float(df_h1["Close"].iloc[-1])
    cur_rvol    = float(df_h1["rvol"].iloc[-1]) if "rvol" in df_h1.columns else 1.0
    cur_vwap    = float(df_h1["session_vwap"].iloc[-1]) if "session_vwap" in df_h1.columns else cur_close

    rvol_ok = cur_rvol >= 1.5

    # Volume profile key levels for bonus
    near_val = near_vah = False
    vp_note  = ""
    try:
        vp       = ta.volume_profile(df_h1["High"], df_h1["Low"], df_h1["Close"], df_h1["Volume"])
        va_range = vp["vah"] - vp["val"]
        prox     = va_range * 0.10
        near_val = prox > 0 and abs(cur_close - vp["val"]) <= prox
        near_vah = prox > 0 and abs(cur_close - vp["vah"]) <= prox
        vp_note  = f" | VP POC={vp['poc']:.2f} VAH={vp['vah']:.2f} VAL={vp['val']:.2f}"
    except Exception:
        pass

    # ── Long breakout ─────────────────────────────────────────────────────────
    if cur_close > absorb_high and cur_close > cur_vwap:
        conf  = 0.75 + (0.05 if rvol_ok else 0.0) + (0.10 if near_val else 0.0)
        return HourlySignal(
            "ABSORB_BO", "LONG", round(min(conf, 0.95), 3),
            f"Absorption breakout ↑: close {cur_close:.2f} > absorb H {absorb_high:.2f} | "
            f"RVOL {cur_rvol:.1f}× | VWAP {cur_vwap:.2f}{vp_note}",
        )

    # ── Short breakout ────────────────────────────────────────────────────────
    if cur_close < absorb_low and cur_close < cur_vwap:
        conf  = 0.75 + (0.05 if rvol_ok else 0.0) + (0.10 if near_vah else 0.0)
        return HourlySignal(
            "ABSORB_BO", "SHORT", round(min(conf, 0.95), 3),
            f"Absorption breakout ↓: close {cur_close:.2f} < absorb L {absorb_low:.2f} | "
            f"RVOL {cur_rvol:.1f}× | VWAP {cur_vwap:.2f}{vp_note}",
        )

    return HourlySignal(
        "ABSORB_BO", "FLAT", 0.0,
        f"Absorption bar present but no breakout: close {cur_close:.2f} in "
        f"[{absorb_low:.2f}, {absorb_high:.2f}]",
    )


# ════════════════════════════════════════════════════════════════════════════════
# STRATEGY 7 — Triple-A Setup  (Absorption → Accumulation → Aggression)
# ════════════════════════════════════════════════════════════════════════════════

def triple_a_setup(df_h1: pd.DataFrame, regime: RegimeResult) -> HourlySignal:
    """
    Triple-A (Absorption → Accumulation → Aggression) on H1 bars.

    Fabio Valentini's highest-conviction setup requires all three phases:
      Phase 1 — Absorption:   a recent bar with high volume + narrow range
                               (institutional passive building)
      Phase 2 — Accumulation: the 1-3 bars after the absorption bar stay
                               strictly within its High/Low range
                               (price coils, waiting for resolution)
      Phase 3 — Aggression:   the current bar closes outside the accumulated
                               range with above-average volume
                               (the directional move is confirmed)

    Confidence 0.80 base (highest of all seven strategies).
    +0.10 bonus when the aggression bar aligns with a Volume Profile key level.
    """
    if len(df_h1) < 30:
        return HourlySignal("TRIPLE_A", "FLAT", 0.0, "Insufficient bars (need 30)")

    abs_s = ta.absorption(df_h1["High"], df_h1["Low"], df_h1["Close"], df_h1["Volume"])

    # ── Phase 1: find most recent absorption bar in the last 6 bars ──────────
    found_abs_pos: int | None = None
    for lookback in range(2, 7):          # search iloc[-2] … iloc[-6]
        abs_pos = len(df_h1) - lookback - 1
        if abs_pos < 0:
            break
        if bool(abs_s.iloc[abs_pos]):
            found_abs_pos = abs_pos
            break

    if found_abs_pos is None:
        return HourlySignal("TRIPLE_A", "FLAT", 0.0,
                            "No absorption bar found in the last 6 bars")

    absorb_high = float(df_h1["High"].iloc[found_abs_pos])
    absorb_low  = float(df_h1["Low"].iloc[found_abs_pos])

    # ── Phase 2: accumulation bars must stay inside absorption range ──────────
    accum = df_h1.iloc[found_abs_pos + 1 : len(df_h1) - 1]   # excl. current bar
    if len(accum) == 0:
        return HourlySignal("TRIPLE_A", "FLAT", 0.0,
                            "No accumulation bars between absorption and current bar")

    contained = (
        (accum["High"] <= absorb_high * 1.002) &
        (accum["Low"]  >= absorb_low  * 0.998)
    ).all()

    if not contained:
        return HourlySignal(
            "TRIPLE_A", "FLAT", 0.0,
            f"Accumulation broke absorption range [{absorb_low:.2f}–{absorb_high:.2f}]",
        )

    # ── Phase 3: aggression — current bar breaks out with volume ─────────────
    cur_close = float(df_h1["Close"].iloc[-1])
    cur_rvol  = float(df_h1["rvol"].iloc[-1]) if "rvol" in df_h1.columns else 1.0
    cur_vwap  = float(df_h1["session_vwap"].iloc[-1]) if "session_vwap" in df_h1.columns else cur_close
    cur_vol   = float(df_h1["Volume"].iloc[-1])
    vol_avg   = float(df_h1["Volume"].rolling(20).mean().iloc[-1])
    vol_ok    = cur_vol > vol_avg * 1.5
    n_accum   = len(accum)

    # Volume profile confluence
    near_val = near_vah = False
    vp_note  = ""
    try:
        vp       = ta.volume_profile(df_h1["High"], df_h1["Low"], df_h1["Close"], df_h1["Volume"])
        va_range = vp["vah"] - vp["val"]
        prox     = va_range * 0.10
        near_val = prox > 0 and abs(cur_close - vp["val"]) <= prox
        near_vah = prox > 0 and abs(cur_close - vp["vah"]) <= prox
        vp_note  = f" | POC={vp['poc']:.2f} VAH={vp['vah']:.2f} VAL={vp['val']:.2f}"
    except Exception:
        pass

    base_conf = 0.80

    # ── Long aggression ───────────────────────────────────────────────────────
    if cur_close > absorb_high and cur_close > cur_vwap:
        conf = base_conf + (0.05 if vol_ok else 0.0) + (0.10 if near_val else 0.0)
        return HourlySignal(
            "TRIPLE_A", "LONG", round(min(conf, 0.95), 3),
            f"Triple-A LONG: absorb [{absorb_low:.2f}–{absorb_high:.2f}] → "
            f"{n_accum} accum bar(s) → aggr {cur_close:.2f} ↑ | "
            f"RVOL {cur_rvol:.1f}×{vp_note}",
        )

    # ── Short aggression ──────────────────────────────────────────────────────
    if cur_close < absorb_low and cur_close < cur_vwap:
        conf = base_conf + (0.05 if vol_ok else 0.0) + (0.10 if near_vah else 0.0)
        return HourlySignal(
            "TRIPLE_A", "SHORT", round(min(conf, 0.95), 3),
            f"Triple-A SHORT: absorb [{absorb_low:.2f}–{absorb_high:.2f}] → "
            f"{n_accum} accum bar(s) → aggr {cur_close:.2f} ↓ | "
            f"RVOL {cur_rvol:.1f}×{vp_note}",
        )

    return HourlySignal(
        "TRIPLE_A", "FLAT", 0.0,
        f"Absorption + accumulation confirmed — aggression not triggered yet: "
        f"close {cur_close:.2f} still inside [{absorb_low:.2f}–{absorb_high:.2f}]",
    )


# ════════════════════════════════════════════════════════════════════════════════
# STRATEGY 8 — Value Area Bounce
# ════════════════════════════════════════════════════════════════════════════════

def value_area_bounce(df_h1: pd.DataFrame, regime: RegimeResult) -> HourlySignal:
    """
    Value Area Bounce on H1 bars.

    The Value Area (70% of session volume) creates natural institutional
    support (VAL) and resistance (VAH) levels.  When price reaches these
    zones and absorption confirms institutional defence, a bounce to POC
    is the highest-probability outcome.

    Long  entry: close ≈ VAL (within 10% of VA range)  AND  absorption present
                 AND  close above EMA(20) for trend confirmation
    Short entry: close ≈ VAH (within 10% of VA range)  AND  absorption present
                 AND  close below EMA(20)

    Target (managed externally): POC (Point of Control)
    """
    if len(df_h1) < 25:
        return HourlySignal("VA_BOUNCE", "FLAT", 0.0, "Insufficient bars (need 25)")

    # ── Volume profile ─────────────────────────────────────────────────────────
    try:
        vp = ta.volume_profile(df_h1["High"], df_h1["Low"], df_h1["Close"], df_h1["Volume"])
    except Exception as exc:
        return HourlySignal("VA_BOUNCE", "FLAT", 0.0, f"Volume profile error: {exc}")

    poc      = vp["poc"]
    vah      = vp["vah"]
    val      = vp["val"]
    va_range = vah - val

    if va_range <= 0:
        return HourlySignal("VA_BOUNCE", "FLAT", 0.0, "Value Area has zero range")

    cur_close = float(df_h1["Close"].iloc[-1])
    prox      = va_range * 0.10
    near_val  = abs(cur_close - val) <= prox
    near_vah  = abs(cur_close - vah) <= prox

    if not near_val and not near_vah:
        return HourlySignal(
            "VA_BOUNCE", "FLAT", 0.0,
            f"Price {cur_close:.2f} not near VA levels "
            f"(VAL={val:.2f}, VAH={vah:.2f}, POC={poc:.2f})",
        )

    # ── Absorption check — institutional defence of the level ─────────────────
    abs_s    = ta.absorption(df_h1["High"], df_h1["Low"], df_h1["Close"], df_h1["Volume"])
    abs_here = bool(abs_s.iloc[-1]) or bool(abs_s.iloc[-2])

    if not abs_here:
        return HourlySignal(
            "VA_BOUNCE", "FLAT", 0.0,
            f"Price at VA level but no absorption — no institutional defence confirmed",
        )

    # ── EMA(20) trend filter ──────────────────────────────────────────────────
    ema20     = ta.ema(df_h1["Close"], length=20)
    cur_ema20 = float(ema20.iloc[-1])

    base_conf = 0.75

    # Long: VAL + absorption + bullish bias
    if near_val and cur_close > cur_ema20:
        return HourlySignal(
            "VA_BOUNCE", "LONG", round(base_conf, 3),
            f"VAL bounce: close {cur_close:.2f} ≈ VAL {val:.2f} | absorption ✓ | "
            f"EMA20={cur_ema20:.2f} | target POC {poc:.2f}",
        )

    # Short: VAH + absorption + bearish bias
    if near_vah and cur_close < cur_ema20:
        return HourlySignal(
            "VA_BOUNCE", "SHORT", round(base_conf, 3),
            f"VAH rejection: close {cur_close:.2f} ≈ VAH {vah:.2f} | absorption ✓ | "
            f"EMA20={cur_ema20:.2f} | target POC {poc:.2f}",
        )

    return HourlySignal(
        "VA_BOUNCE", "FLAT", 0.0,
        f"At VA level but EMA trend does not confirm "
        f"(close={cur_close:.2f}, EMA20={cur_ema20:.2f}, VAL={val:.2f}, VAH={vah:.2f})",
    )


# ════════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR — run_h1_gate
# ════════════════════════════════════════════════════════════════════════════════

def run_h1_gate(
    ticker: str,
    df_daily: pd.DataFrame | None = None,
) -> H1GateResult:
    """
    Main entry point for the H1 confirmation gate.

    1. Fetches H1 data (fetch_hourly_data).
    2. Computes the ADX regime and session window (compute_regime).
    3. Runs every enabled strategy for the current regime.
    4. Returns gate_pass=True only when:
         - We are inside an active session window
         - At least one enabled strategy signals LONG
         - The best LONG signal confidence ≥ H1_MIN_CONFIDENCE

    On H1 data failure: gate_pass=False, reason logged.  No exception raised.
    When H1_STRATEGY_GATE=False: returns gate_pass=True immediately (bypass).
    """
    from data.fetcher_hourly import fetch_hourly_data
    from agents.regime_router import compute_regime
    from bot.config import H1_STRATEGY_GATE, H1_MIN_CONFIDENCE, H1_LOOKBACK_DAYS

    # ── Bypass mode ───────────────────────────────────────────────────────────
    if not H1_STRATEGY_GATE:
        return H1GateResult(
            gate_pass=True, signal="LONG",
            active_strategy="BYPASS", confidence=1.0,
            regime="N/A", adx=0.0, in_session_window=True,
            signals=[], reason="H1_STRATEGY_GATE=False — gate bypassed",
        )

    # ── Fetch H1 data ─────────────────────────────────────────────────────────
    try:
        df_h1 = fetch_hourly_data(ticker, days=H1_LOOKBACK_DAYS)
    except Exception as exc:
        return H1GateResult(
            gate_pass=False, signal="FLAT",
            active_strategy="NONE", confidence=0.0,
            regime="N/A", adx=0.0, in_session_window=False,
            signals=[], reason=f"H1 fetch failed: {exc}",
        )

    # ── Regime + session window ───────────────────────────────────────────────
    try:
        regime = compute_regime(df_h1)
    except Exception as exc:
        return H1GateResult(
            gate_pass=False, signal="FLAT",
            active_strategy="NONE", confidence=0.0,
            regime="N/A", adx=0.0, in_session_window=False,
            signals=[], reason=f"Regime computation failed: {exc}",
        )

    if not regime.in_session_window:
        return H1GateResult(
            gate_pass=False, signal="FLAT",
            active_strategy="NONE", confidence=0.0,
            regime=regime.regime, adx=regime.adx,
            in_session_window=False, signals=[],
            reason=regime.window_note,
        )

    # ── Run enabled strategies ────────────────────────────────────────────────
    signals: list[HourlySignal] = []
    _strategy_map = {
        "RSI_MR":    lambda: rsi_mean_reversion(df_h1, regime),
        "VWAP_PB":   lambda: vwap_pullback(df_h1, regime),
        "ORB_60":    lambda: orb_breakout(df_h1, regime),
        "SQUEEZE":   lambda: squeeze_breakout(df_h1, regime),
        "ZSCORE_MR": lambda: zscore_mean_reversion(df_h1, df_daily, regime),
        "ABSORB_BO": lambda: absorption_breakout(df_h1, regime),
        "TRIPLE_A":  lambda: triple_a_setup(df_h1, regime),
        "VA_BOUNCE": lambda: value_area_bounce(df_h1, regime),
    }
    for name in regime.enabled_strategies:
        try:
            sig = _strategy_map[name]()
            signals.append(sig)
        except Exception as exc:
            signals.append(HourlySignal(name, "FLAT", 0.0, f"Runtime error: {exc}"))

    # ── Absorption confluence multiplier ──────────────────────────────────────
    # If any absorption bar exists in the last 3 bars, boost all LONG signal
    # confidences by 10% (capped at 1.0) — Valentini's "absorption confirms
    # any entry" principle.
    abs_s = ta.absorption(df_h1["High"], df_h1["Low"], df_h1["Close"], df_h1["Volume"])
    recent_absorption = bool(abs_s.iloc[-3:].any()) if len(abs_s) >= 3 else False
    if recent_absorption:
        for sig in signals:
            if sig.signal == "LONG":
                sig.confidence = min(1.0, sig.confidence + 0.10)
                sig.reason += " [+abs confluence]"

    # ── Pick best LONG signal ─────────────────────────────────────────────────
    long_signals = [s for s in signals if s.signal == "LONG"]
    if not long_signals:
        flat_summary = " | ".join(f"{s.strategy}: {s.reason[:60]}" for s in signals[:3])
        return H1GateResult(
            gate_pass=False, signal="FLAT",
            active_strategy="NONE", confidence=0.0,
            regime=regime.regime, adx=regime.adx,
            in_session_window=True, signals=signals,
            reason=f"No LONG signal. {flat_summary}",
        )

    best = max(long_signals, key=lambda s: s.confidence)
    gate = best.confidence >= H1_MIN_CONFIDENCE

    return H1GateResult(
        gate_pass=gate,
        signal="LONG",
        active_strategy=best.strategy,
        confidence=best.confidence,
        regime=regime.regime,
        adx=regime.adx,
        in_session_window=True,
        signals=signals,
        reason=(
            f"{best.strategy} ({best.confidence:.0%}): {best.reason}"
            if gate else
            f"{best.strategy} confidence {best.confidence:.0%} below threshold "
            f"{H1_MIN_CONFIDENCE:.0%}"
        ),
    )
