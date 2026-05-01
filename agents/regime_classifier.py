"""Period-level regime classifier.

The signal engine already routes each individual bar based on ADX (per-signal,
bar-by-bar). The learning loop's daily/weekly reports need a *period* label
("the last 24h was TRENDING") summarised across many bars. That's what this
module does.

A bar is classified TRENDING when ADX > 25, RANGING when ADX < 20, otherwise
NEUTRAL. The period label is the dominant bin; if neither dominates by a
clear margin (≥55%), we call it MIXED.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

import utils.ta_compat as ta

logger = logging.getLogger(__name__)

# Match the per-bar thresholds in agents.signal_engine for consistency.
_ADX_TRENDING = 25.0
_ADX_RANGING  = 20.0
_DOMINANT_FRAC = 0.55   # need >55% of bars in one bin to declare it dominant


@dataclass
class RegimeReport:
    ticker:        str
    bars_analysed: int
    pct_trending:  float   # 0.0 – 1.0
    pct_ranging:   float
    pct_neutral:   float
    label:         str     # "TRENDING" | "RANGING" | "MIXED" | "INSUFFICIENT_DATA"


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

    return RegimeReport(
        ticker=ticker,
        bars_analysed=n,
        pct_trending=round(pct_t, 3),
        pct_ranging=round(pct_r, 3),
        pct_neutral=round(pct_n, 3),
        label=label,
    )
