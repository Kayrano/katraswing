"""
Intraday data fetcher for 5m and 15m bar intervals.

Enriches raw OHLCV with the same session metadata as fetcher_hourly:
    session_date       — date() of the trading session each bar belongs to
    session_bar_number — 1-based bar index within the session (1 = first bar)
    is_first_bar       — True for session_bar_number == 1
    session_vwap       — cumulative session VWAP; resets at each session open
    rvol               — bar volume / same-bar-number 20-session rolling average
    market             — "US" or "BIST"

yfinance limits:
    5m  — up to 60 calendar days of history
    15m — up to 60 calendar days of history
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from zoneinfo import ZoneInfo

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

VALID_INTERVALS = ("5m", "15m")


def detect_market(ticker: str) -> str:
    return "BIST" if ticker.upper().endswith(".IS") else "US"


def fetch_intraday_data(
    ticker: str,
    interval: str = "5m",
    days: int = 59,
) -> pd.DataFrame:
    """
    Fetch intraday OHLCV bars (5m or 15m) and enrich with session metadata.

    Parameters
    ----------
    ticker : str
        Ticker symbol.  Turkish stocks must carry the '.IS' suffix.
    interval : str
        "5m" or "15m".
    days : int
        Calendar-day lookback.  yfinance caps 5m/15m at 60 days; default 59
        gives a small safety margin.

    Returns
    -------
    pd.DataFrame
        Timezone-aware index.  Columns:
        Open, High, Low, Close, Volume,
        session_date, session_bar_number, is_first_bar,
        session_vwap, rvol, market

    Raises
    ------
    ValueError
        On bad interval, empty fetch, or insufficient session bars.
    """
    if interval not in VALID_INTERVALS:
        raise ValueError(f"interval must be one of {VALID_INTERVALS}, got {interval!r}")

    market = detect_market(ticker)
    cfg    = SESSION_CONFIG[market]
    tz     = cfg["tz"]

    raw = yf.Ticker(ticker).history(
        period=f"{days}d", interval=interval, auto_adjust=True
    )
    if raw is None or raw.empty:
        raise ValueError(f"yfinance returned no {interval} data for '{ticker}'")

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy().dropna()

    # Localise to session timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(tz)
    else:
        df.index = df.index.tz_convert(tz)

    # Clip to exchange hours
    open_mins  = cfg["open_hour"] * 60 + cfg["open_minute"]
    close_mins = cfg["close_hour"] * 60 + cfg["close_minute"]
    bar_mins   = df.index.hour * 60 + df.index.minute
    df = df[(bar_mins >= open_mins) & (bar_mins < close_mins)].copy()

    min_bars = 30
    if len(df) < min_bars:
        raise ValueError(
            f"Insufficient {interval} bars for '{ticker}' after session filtering: "
            f"{len(df)} bars (minimum {min_bars})."
        )

    # Session metadata
    df["session_date"]       = df.index.date
    df["session_bar_number"] = df.groupby("session_date").cumcount() + 1
    df["is_first_bar"]       = df["session_bar_number"] == 1

    # Session VWAP (cumulative, resets each session)
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    tp_vol  = typical * df["Volume"]
    cum_tpv = tp_vol.groupby(df["session_date"]).cumsum()
    cum_vol = df["Volume"].groupby(df["session_date"]).cumsum()
    df["session_vwap"] = cum_tpv / cum_vol.replace(0, np.nan)

    # RVOL: compare to rolling 20-session average at same bar number
    # shift(1) prevents look-ahead (current bar excluded from its own average)
    df["_bar_num"] = df["session_bar_number"]
    same_bn_avg = (
        df.groupby("_bar_num")["Volume"]
          .transform(lambda s: s.shift(1).rolling(20, min_periods=3).mean())
    )
    df["rvol"] = (df["Volume"] / same_bn_avg.replace(0, np.nan)).fillna(1.0)
    df.drop(columns=["_bar_num"], inplace=True)

    df["market"] = market
    return df
