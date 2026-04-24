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

import time
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from zoneinfo import ZoneInfo

# ── In-memory OHLCV cache (5-min TTL) ────────────────────────────────────────
# Avoids redundant yfinance/MT5 fetches when multiple scan cycles run close
# together (e.g. during backtest warm-up or rapid UI refreshes).
_OHLCV_CACHE: dict[str, tuple[pd.DataFrame, float]] = {}
_OHLCV_TTL = 300   # seconds — matches the 5-min bar interval

_US_TZ   = ZoneInfo("America/New_York")
_BIST_TZ = ZoneInfo("Europe/Istanbul")
_JST_TZ  = ZoneInfo("Asia/Tokyo")

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
    "JAPAN": {
        "tz":           _JST_TZ,
        "open_hour":    9,
        "open_minute":  0,
        "close_hour":   15,
        "close_minute": 30,
    },
}

VALID_INTERVALS = ("5m", "15m")

# EMA lengths used for daily trend direction
_DAILY_EMA_FAST = 20
_DAILY_EMA_SLOW = 50


def fetch_daily_trend(ticker: str) -> dict:
    """
    Fetch daily bars and return a trend summary dict:
        trend_direction : "BULLISH" | "BEARISH" | "NEUTRAL"
        ema20           : float — last EMA(20) daily value
        ema50           : float — last EMA(50) daily value
        adx_daily       : float — last ADX(14) daily value
        close           : float — last daily close

    BULLISH  = close > ema20 > ema50
    BEARISH  = close < ema20 < ema50
    NEUTRAL  = anything else (transitioning / ranging)

    Raises ValueError if fewer than 50 daily bars are available.
    """
    # Try MT5 first
    mt5_raw = _fetch_from_mt5(ticker, "1d", 90)
    if mt5_raw is not None and len(mt5_raw) >= 50:
        raw = mt5_raw
    else:
        yf_tick = _MT5_TO_YF.get(ticker.upper(), ticker)
        if yf_tick == ticker.upper() and ticker.startswith("#"):
            yf_tick = _MT5_TO_YF.get(ticker.split("_")[0].upper(), ticker)
        try:
            raw = yf.Ticker(yf_tick).history(period="90d", interval="1d", auto_adjust=True)
        except Exception:
            raw = None
        if raw is None or raw.empty or len(raw) < 50:
            raise ValueError(f"Insufficient daily data for '{ticker}' (need ≥50 bars)")

    close = raw["Close"].dropna()

    ema20 = close.ewm(span=_DAILY_EMA_FAST, adjust=False).mean()
    ema50 = close.ewm(span=_DAILY_EMA_SLOW, adjust=False).mean()

    last_close = float(close.iloc[-1])
    last_ema20 = float(ema20.iloc[-1])
    last_ema50 = float(ema50.iloc[-1])

    # ADX on daily bars
    adx_daily = 0.0
    try:
        high  = raw["High"].dropna()
        low   = raw["Low"].dropna()
        # align index lengths after dropna
        idx   = close.index.intersection(high.index).intersection(low.index)
        from utils.ta_compat import adx as _adx
        adx_s = _adx(high.loc[idx], low.loc[idx], close.loc[idx], length=14)
        valid = adx_s.dropna()
        if not valid.empty:
            adx_daily = float(valid.iloc[-1])
    except Exception:
        adx_daily = 0.0

    if last_close > last_ema20 and last_ema20 > last_ema50:
        direction = "BULLISH"
    elif last_close < last_ema20 and last_ema20 < last_ema50:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"

    return {
        "trend_direction": direction,
        "ema20": round(last_ema20, 4),
        "ema50": round(last_ema50, 4),
        "adx_daily": round(adx_daily, 1),
        "close": round(last_close, 4),
    }


_JAPAN_FUTURES = {"NKD=F"}

# Maps raw MT5 symbol names → yfinance tickers for offline fallback
_MT5_TO_YF: dict[str, str] = {
    # Forex
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X", "USDCHF": "USDCHF=X",
    "NZDUSD": "NZDUSD=X", "EURGBP": "EURGBP=X", "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X", "EURCAD": "EURCAD=X", "EURCHF": "EURCHF=X",
    "AUDCAD": "AUDCAD=X", "AUDCHF": "AUDCHF=X", "AUDNZD": "AUDNZD=X",
    "CADJPY": "CADJPY=X", "CHFJPY": "CHFJPY=X", "NZDJPY": "NZDJPY=X",
    # Metals
    "XAUUSD": "GC=F", "XAGUSD": "SI=F",
    # US indices (FxPro-style names with expiry suffix handled via startswith)
    "#US100_M26": "NQ=F",  "#US100_M27": "NQ=F",  "#US100": "NQ=F",
    "#US500_M26": "ES=F",  "#US500_M27": "ES=F",  "#US500": "ES=F",
    "#US30_M26":  "YM=F",  "#US30_M27":  "YM=F",  "#US30":  "YM=F",
    # European indices
    "#GER40_M26": "GDAXI", "#GER40": "GDAXI",
    "#UK100_M26": "^FTSE",  "#UK100": "^FTSE",
    # Japan
    "#JP225_M26": "NKD=F",  "#JP225": "NKD=F",
}

# Known forex currency codes for MT5 symbol detection
_FX_CODES = {
    "EUR","USD","GBP","JPY","AUD","NZD","CAD","CHF",
    "SGD","HKD","NOK","SEK","DKK","MXN","ZAR","TRY",
}

def detect_market(ticker: str) -> str:
    """
    Return "BIST", "FOREX", "JAPAN", or "US".
    Handles both yfinance tickers (NQ=F, EURUSD=X) and raw MT5 symbol names (EURUSD, #US100_M26).
    """
    sym = ticker.upper()
    if sym.endswith(".IS"):
        return "BIST"
    if sym.endswith("=X"):
        return "FOREX"
    if sym in _JAPAN_FUTURES or any(x in sym for x in ("JP225", "NIKKEI", "JPN225")):
        return "JAPAN"
    if sym.endswith("=F"):
        return "US"
    if "-USD" in sym or "-BTC" in sym or "-ETH" in sym:
        return "FOREX"
    # Raw MT5 forex pairs: 6 chars, both halves are known currency codes
    if len(sym) == 6 and sym[:3] in _FX_CODES and sym[3:] in _FX_CODES:
        return "FOREX"
    # Precious metals / commodities traded 24h
    if any(x in sym for x in ("XAU", "XAG", "GOLD", "SILVER", "OIL", "BRENT", "WTI")):
        return "FOREX"
    # Anything else from MT5 (indices, CFDs): treat as continuous to skip session clipping
    # MT5 already only returns bars during valid trading hours.
    if sym.startswith("#") or any(x in sym for x in ("US100","US500","UK100","GER40","AUS200","HK50")):
        return "FOREX"
    return "US"


def _fetch_from_mt5(ticker: str, interval: str, days: int,
                     mt5_symbol: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Try to fetch OHLCV bars from MT5. Returns raw DataFrame or None.

    If mt5_symbol is provided it is used directly, bypassing SYMBOL_MAP.
    """
    try:
        from utils.mt5_bridge import SYMBOL_MAP, fetch_bars, is_connected
        if not is_connected():
            return None
        symbol = mt5_symbol or SYMBOL_MAP.get(ticker.upper(), ticker)
        # Estimate bar count: 288 bars/day for 5m (24h), 96 for 15m; add buffer
        bars_per_day = {"5m": 300, "15m": 100, "1h": 25, "1d": 1}.get(interval, 300)
        count = days * bars_per_day
        return fetch_bars(symbol, interval, count)
    except Exception:
        return None


def fetch_intraday_data(
    ticker: str,
    interval: str = "5m",
    days: int = 59,
    mt5_symbol: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch intraday OHLCV bars (5m or 15m) and enrich with session metadata.

    Tries MT5 first (real-time, accurate market hours) then falls back to
    yfinance if MT5 is not connected.

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

    # ── Cache check ───────────────────────────────────────────────────────────
    cache_key = f"{ticker}:{interval}:{days}"
    _cached = _OHLCV_CACHE.get(cache_key)
    if _cached is not None and time.time() - _cached[1] < _OHLCV_TTL:
        return _cached[0]

    market = detect_market(ticker)
    tz     = SESSION_CONFIG.get(market, SESSION_CONFIG["US"])["tz"] if market != "FOREX" else ZoneInfo("America/New_York")

    # ── Try MT5 first ────────────────────────────────────────────────────────
    raw_mt5 = _fetch_from_mt5(ticker, interval, days, mt5_symbol=mt5_symbol)
    if raw_mt5 is not None and not raw_mt5.empty:
        raw = raw_mt5
        # MT5 returns UTC; convert to session tz
        raw.index = raw.index.tz_convert(tz)
    else:
        # ── Fall back to yfinance ─────────────────────────────────────────────
        # Map raw MT5 symbol names to yfinance tickers where needed
        yf_ticker = _MT5_TO_YF.get(ticker.upper(), ticker)
        # Also handle expiry-suffixed index futures like #US100_M28
        if yf_ticker == ticker.upper() and ticker.startswith("#"):
            base = ticker.split("_")[0].upper()
            yf_ticker = _MT5_TO_YF.get(base, ticker)
        try:
            yf_raw = yf.Ticker(yf_ticker).history(
                period=f"{days}d", interval=interval, auto_adjust=True
            )
        except Exception:
            yf_raw = None
        if yf_raw is None or yf_raw.empty:
            raise ValueError(f"No {interval} data for '{ticker}' (MT5 not connected, yfinance returned nothing)")
        raw = yf_raw

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy().dropna()

    # Localise to session timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(tz)
    else:
        df.index = df.index.tz_convert(tz)

    # Clip to exchange hours (equities only — FOREX trades 24/5)
    if market != "FOREX":
        cfg        = SESSION_CONFIG[market]
        open_mins  = cfg["open_hour"] * 60 + cfg["open_minute"]
        close_mins = cfg["close_hour"] * 60 + cfg["close_minute"]
        bar_mins   = df.index.hour * 60 + df.index.minute
        df = df[(bar_mins >= open_mins) & (bar_mins < close_mins)].copy()
    else:
        # Drop weekend bars
        df = df[df.index.dayofweek < 5].copy()

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
    _OHLCV_CACHE[cache_key] = (df, time.time())
    return df
