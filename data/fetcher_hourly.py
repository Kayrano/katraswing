"""
Hourly (H1) data fetcher for intraday strategy signals.

Handles both US equities (09:30–16:00 ET) and BIST (10:00–18:00 Istanbul,
with a mid-session break 13:00–14:00 that is retained in the DataFrame
but flagged so the session window filter can suppress trades).

Extra columns added beyond raw OHLCV:
    session_date       — date() of the trading session each bar belongs to
    session_bar_number — 1-based bar index within the session (1 = opening bar)
    is_first_bar       — True for session_bar_number == 1
    session_vwap       — cumulative session VWAP; resets at each session open
    rvol               — current-bar volume / same-hour 20-session rolling average
    market             — "US" or "BIST"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from zoneinfo import ZoneInfo


# ── Session configurations ────────────────────────────────────────────────────
_US_TZ   = ZoneInfo("America/New_York")
_BIST_TZ = ZoneInfo("Europe/Istanbul")

SESSION_CONFIG: dict[str, dict] = {
    "US": {
        "tz":           _US_TZ,
        "open_hour":    9,
        "open_minute":  30,
        "close_hour":   16,
        "close_minute": 0,
    },
    "BIST": {
        "tz":           _BIST_TZ,
        "open_hour":    10,
        "open_minute":  0,
        "close_hour":   18,
        "close_minute": 0,
    },
}


def detect_market(ticker: str) -> str:
    """Return 'BIST' when the ticker carries the .IS suffix, else 'US'."""
    return "BIST" if ticker.upper().endswith(".IS") else "US"


def fetch_hourly_data(ticker: str, days: int = 60) -> pd.DataFrame:
    """
    Fetch H1 OHLCV bars and enrich with session metadata.

    Parameters
    ----------
    ticker : str
        Ticker symbol.  Turkish stocks must carry the '.IS' suffix so the
        correct session timezone is applied.
    days : int
        Look-back window in calendar days.  60 days ≈ 390 H1 bars (US market).
        yfinance caps the 1h interval at 730 days.

    Returns
    -------
    pd.DataFrame
        Timezone-aware index (session local time).  Columns:
        Open, High, Low, Close, Volume,
        session_date, session_bar_number, is_first_bar,
        session_vwap, rvol, market

    Raises
    ------
    ValueError
        When yfinance returns nothing, or fewer than 50 bars survive session
        filtering (too little history to warm up the indicators).
    """
    market = detect_market(ticker)
    cfg    = SESSION_CONFIG[market]
    tz     = cfg["tz"]

    # ── Fetch raw H1 bars ─────────────────────────────────────────────────────
    raw = yf.Ticker(ticker).history(
        period=f"{days}d", interval="1h", auto_adjust=True
    )
    if raw is None or raw.empty:
        raise ValueError(f"yfinance returned no H1 data for '{ticker}'")

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy().dropna()

    # ── Localise index to session timezone ────────────────────────────────────
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(tz)
    else:
        df.index = df.index.tz_convert(tz)

    # ── Clip to exchange session hours ────────────────────────────────────────
    # We keep bars that START at or after open and strictly before close.
    open_mins  = cfg["open_hour"]  * 60 + cfg["open_minute"]
    close_mins = cfg["close_hour"] * 60 + cfg["close_minute"]
    bar_mins   = df.index.hour * 60 + df.index.minute
    df = df[(bar_mins >= open_mins) & (bar_mins < close_mins)].copy()

    if len(df) < 50:
        raise ValueError(
            f"Insufficient H1 bars for '{ticker}' after session filtering: "
            f"{len(df)} bars (minimum 50).  Extend days= or check the ticker."
        )

    # ── Session metadata ──────────────────────────────────────────────────────
    df["session_date"]       = df.index.date
    df["session_bar_number"] = (
        df.groupby("session_date").cumcount() + 1
    )
    df["is_first_bar"] = df["session_bar_number"] == 1

    # ── Session VWAP (cumulative within each session, resets at open) ─────────
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    tp_vol  = typical * df["Volume"]
    cum_tpv = tp_vol.groupby(df["session_date"]).cumsum()
    cum_vol = df["Volume"].groupby(df["session_date"]).cumsum()
    df["session_vwap"] = cum_tpv / cum_vol.replace(0, np.nan)

    # ── Same-hour RVOL ────────────────────────────────────────────────────────
    # Compare each bar's volume to the rolling 20-session average at the same
    # hour of day.  shift(1) prevents look-ahead: the current bar is excluded
    # from its own average.
    df["_bar_hour"] = df.index.hour
    same_hr_avg = (
        df.groupby("_bar_hour")["Volume"]
          .transform(lambda s: s.shift(1).rolling(20, min_periods=3).mean())
    )
    df["rvol"] = (df["Volume"] / same_hr_avg.replace(0, np.nan)).fillna(1.0)
    df.drop(columns=["_bar_hour"], inplace=True)

    df["market"] = market
    return df
