"""
Pure pandas/numpy technical indicator implementations.
Drop-in replacement for pandas_ta — same function signatures and return shapes.
Eliminates the pandas_ta → numba → llvmlite dependency chain.
"""

import pandas as pd
import numpy as np


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=length - 1, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(com=length - 1, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    result.name = f"RSI_{length}"
    return result


def ema(close: pd.Series, length: int) -> pd.Series:
    result = close.ewm(span=length, adjust=False).mean()
    result.name = f"EMA_{length}"
    return result


def sma(close: pd.Series, length: int) -> pd.Series:
    result = close.rolling(length).mean()
    result.name = f"SMA_{length}"
    return result


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    # Column order matches pandas_ta: MACD, MACDh, MACDs  (iloc 0,1,2)
    return pd.DataFrame({
        f"MACD_{fast}_{slow}_{signal}":  macd_line,
        f"MACDh_{fast}_{slow}_{signal}": histogram,
        f"MACDs_{fast}_{slow}_{signal}": signal_line,
    }, index=close.index)


def bbands(close: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(length).mean()
    stddev = close.rolling(length).std(ddof=0)
    upper = mid + std * stddev
    lower = mid - std * stddev
    # Column order matches pandas_ta: BBL, BBM, BBU  (iloc 0,1,2)
    return pd.DataFrame({
        f"BBL_{length}_{float(std)}": lower,
        f"BBM_{length}_{float(std)}": mid,
        f"BBU_{length}_{float(std)}": upper,
    }, index=close.index)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    result = tr.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    result.name = f"ATRr_{length}"
    return result


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    result = (direction * volume).cumsum()
    result.name = "OBV"
    return result


def stoch(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k: int = 14,
    d: int = 3,
) -> pd.DataFrame:
    lowest_low   = low.rolling(k).min()
    highest_high = high.rolling(k).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    k_line = 100 * (close - lowest_low) / denom
    d_line = k_line.rolling(d).mean()
    # Column order matches pandas_ta: STOCHk, STOCHd  (iloc 0,1)
    return pd.DataFrame({
        f"STOCHk_{k}_{d}_{d}": k_line,
        f"STOCHd_{k}_{d}_{d}": d_line,
    }, index=close.index)
