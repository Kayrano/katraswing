"""Period-level regime classifier.

The signal engine already routes each individual bar based on ADX (per-signal,
bar-by-bar). The learning loop's daily/weekly reports need a *period* label
("the last 24h was TRENDING") summarised across many bars. That's what this
module does.

ADX-only is a known weak signal — open-web research (Macrosynergy, QuantifiedStrategies
backtests) shows composite filters (ADX + Hurst + Choppiness) outperform any
single classifier. Round 4 B5 adds Hurst (R/S exponent) and Choppiness Index,
plus a `composite_score` ∈ [-1, +1] that callers can use as a soft regime
weight (negative = ranging, positive = trending).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

import utils.ta_compat as ta

logger = logging.getLogger(__name__)

# Match the per-bar thresholds in agents.signal_engine for consistency.
_ADX_TRENDING = 25.0
_ADX_RANGING  = 20.0
_DOMINANT_FRAC = 0.55   # need >55% of bars in one bin to declare it dominant

# Hurst exponent: H>0.55 trending, ~0.50 random walk, <0.45 mean-reverting.
_HURST_TRENDING = 0.55
_HURST_RANGING  = 0.45
_HURST_LOOKBACK = 100   # rolling window for R/S calc

# Choppiness Index: CI>61.8 strongly ranging, CI<38.2 strongly trending.
# (Inverse of ADX semantics; higher = more sideways.)
_CHOP_RANGING  = 61.8
_CHOP_TRENDING = 38.2
_CHOP_LOOKBACK = 14


@dataclass
class RegimeReport:
    ticker:        str
    bars_analysed: int
    pct_trending:  float   # 0.0 – 1.0
    pct_ranging:   float
    pct_neutral:   float
    label:         str     # "TRENDING" | "RANGING" | "MIXED" | "INSUFFICIENT_DATA"
    # Round 4 B5: composite metrics. None when insufficient data.
    hurst:           float | None = None   # 0.0–1.0
    choppiness:      float | None = None   # 0–100
    composite_score: float | None = None   # −1 (ranging) to +1 (trending)


def hurst_exponent(prices: np.ndarray) -> float | None:
    """Compute the Hurst exponent (R/S method) for a price series.

    H ≈ 0.5  : random walk
    H > 0.55 : persistent / trending
    H < 0.45 : anti-persistent / mean-reverting

    Returns None if the series is too short or pathological (zero variance,
    too few non-zero increments). The R/S method is well-known to underestimate
    H on short series, so we require ≥50 points and use 5 sub-window sizes.
    """
    n = len(prices)
    if n < 50:
        return None
    log_p = np.log(np.asarray(prices, dtype=float) + 1e-12)
    increments = np.diff(log_p)
    if np.var(increments) == 0:
        return None
    # Sub-window sizes — geometric spread for stable log-log regression
    sizes = [int(n / k) for k in (10, 8, 6, 4, 2) if int(n / k) >= 8]
    if len(sizes) < 3:
        return None

    rs_values = []
    for size in sizes:
        n_chunks = n // size
        rs_chunk = []
        for i in range(n_chunks):
            chunk = increments[i*size:(i+1)*size]
            if len(chunk) < 2:
                continue
            mean = chunk.mean()
            cum  = (chunk - mean).cumsum()
            r = cum.max() - cum.min()
            s = chunk.std(ddof=0)
            if s > 0 and r > 0:
                rs_chunk.append(r / s)
        if rs_chunk:
            rs_values.append((size, float(np.mean(rs_chunk))))

    if len(rs_values) < 3:
        return None

    log_sizes = np.log([s for s, _ in rs_values])
    log_rs    = np.log([r for _, r in rs_values])
    slope, _ = np.polyfit(log_sizes, log_rs, 1)
    return float(slope)


def choppiness_index(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = _CHOP_LOOKBACK,
) -> pd.Series:
    """Choppiness Index — measures sideways vs trending behaviour.

    CI > 61.8 → strongly ranging; CI < 38.2 → strongly trending.

    Formula:  100 * log10( sum(TR, n) / (max(High, n) - min(Low, n)) ) / log10(n)

    where TR is true range. Higher values mean price has spent more total
    movement without making net progress (i.e., chop).
    """
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    sum_tr   = tr.rolling(length).sum()
    max_high = high.rolling(length).max()
    min_low  = low.rolling(length).min()
    rng      = max_high - min_low

    # Avoid log of zero / division by zero
    safe_ratio = (sum_tr / rng).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    safe_ratio = safe_ratio.clip(lower=1e-9)
    ci = 100.0 * np.log10(safe_ratio) / np.log10(length)
    return ci


def composite_score(
    pct_trending: float, hurst: float | None, last_chop: float | None,
) -> float:
    """Blend ADX-pct, Hurst, and Choppiness into a single [-1, +1] score.

    +1 = strongly trending, −1 = strongly ranging, 0 = neutral.
    Components are equal-weighted; missing components are dropped from the mean.
    """
    parts: list[float] = []
    # ADX share contributes (pct_trending - pct_ranging-equivalent)
    # We use 2*pct_trending - 1 mapped from [0,1] → [-1,+1].
    parts.append(2.0 * pct_trending - 1.0)
    if hurst is not None:
        # Map H around 0.5: H=0.55→+0.5, H=0.45→-0.5, clip [-1,+1]
        parts.append(max(-1.0, min(1.0, (hurst - 0.5) * 10.0)))
    if last_chop is not None:
        # Choppiness is INVERTED — high CI = ranging
        # Map CI=38.2→+1, CI=61.8→-1, linear in between
        cs = (50.0 - last_chop) / 11.8
        parts.append(max(-1.0, min(1.0, cs)))
    return round(sum(parts) / len(parts), 3) if parts else 0.0


def classify(df: pd.DataFrame, ticker: str = "", lookback_bars: int = 288) -> RegimeReport:
    """Classify the regime over the last `lookback_bars` of the supplied OHLCV.

    288 5m bars = exactly 24 hours of market time.

    Returns a RegimeReport. On insufficient data (under 30 valid ADX values),
    label is INSUFFICIENT_DATA and percentages are zero.
    """
    if df is None or len(df) < 30:
        return RegimeReport(ticker, 0, 0.0, 0.0, 0.0, "INSUFFICIENT_DATA")

    tail = df.tail(lookback_bars).copy()
    try:
        adx_series = ta.adx(tail["High"], tail["Low"], tail["Close"], length=14)
    except Exception as exc:
        logger.warning("ctx=regime_classify ticker=%s adx_failed: %s", ticker, exc)
        return RegimeReport(ticker, 0, 0.0, 0.0, 0.0, "INSUFFICIENT_DATA")

    valid = adx_series.dropna()
    n = len(valid)
    if n < 30:
        return RegimeReport(ticker, n, 0.0, 0.0, 0.0, "INSUFFICIENT_DATA")

    trending_n = int((valid > _ADX_TRENDING).sum())
    ranging_n  = int((valid < _ADX_RANGING).sum())
    neutral_n  = n - trending_n - ranging_n

    pct_t = trending_n / n
    pct_r = ranging_n  / n
    pct_n = neutral_n  / n

    if pct_t >= _DOMINANT_FRAC:
        label = "TRENDING"
    elif pct_r >= _DOMINANT_FRAC:
        label = "RANGING"
    else:
        label = "MIXED"

    # Round 4 B5: extra composite components (Hurst + Choppiness).
    hurst_val: float | None = None
    chop_val:  float | None = None
    try:
        hurst_val = hurst_exponent(tail["Close"].to_numpy())
    except Exception as exc:
        logger.debug("ctx=regime hurst_failed: %s", exc)
    try:
        ci_series = choppiness_index(tail["High"], tail["Low"], tail["Close"])
        ci_dropped = ci_series.dropna()
        if not ci_dropped.empty:
            chop_val = float(ci_dropped.iloc[-1])
    except Exception as exc:
        logger.debug("ctx=regime choppiness_failed: %s", exc)

    score = composite_score(pct_t, hurst_val, chop_val)

    return RegimeReport(
        ticker=ticker,
        bars_analysed=n,
        pct_trending=round(pct_t, 3),
        pct_ranging=round(pct_r, 3),
        pct_neutral=round(pct_n, 3),
        label=label,
        hurst=round(hurst_val, 3) if hurst_val is not None else None,
        choppiness=round(chop_val, 2) if chop_val is not None else None,
        composite_score=score,
    )
