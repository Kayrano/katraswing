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
    stddev = close.rolling(length).std(ddof=1)
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
    result.name = f"ATR_{length}"
    return result


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    result = (direction * volume).cumsum()
    result.name = "OBV"
    return result


def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    length: int = 20,
) -> pd.Series:
    """Rolling VWAP over `length` bars — suitable for daily swing-trade charts."""
    typical = (high + low + close) / 3
    result = (typical * volume).rolling(length).sum() / volume.rolling(length).sum()
    result.name = f"VWAP_{length}"
    return result


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> pd.Series:
    """Average Directional Index (Wilder smoothing). Returns ADX series (0-100)."""
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    up_move   = high - prev_high
    down_move = prev_low - low

    plus_dm  = np.where((up_move > down_move) & (up_move > 0),   up_move,   0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    alpha = 1.0 / length
    atr_s      = tr.ewm(alpha=alpha, min_periods=length, adjust=False).mean()
    plus_dm_s  = pd.Series(plus_dm,  index=high.index).ewm(alpha=alpha, min_periods=length, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=high.index).ewm(alpha=alpha, min_periods=length, adjust=False).mean()

    safe_atr   = atr_s.replace(0, np.nan)
    plus_di    = 100 * plus_dm_s  / safe_atr
    minus_di   = 100 * minus_dm_s / safe_atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_line   = dx.ewm(alpha=alpha, min_periods=length, adjust=False).mean()
    adx_line.name = f"ADX_{length}"
    return adx_line


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


def absorption(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    length: int = 20,
    vol_mult: float = 2.0,
    range_mult: float = 0.3,
) -> pd.Series:
    """
    Absorption bar detector (Fabio Valentini / order-flow concept).

    An absorption bar shows abnormally high volume but moves very little —
    the signature of institutional passive accumulation or distribution.

    Condition (both must be true):
        volume  > rolling_mean(volume, length)  × vol_mult
        (high - low) < ATR(length)              × range_mult

    Returns a boolean Series: True on bars where absorption is detected.
    """
    vol_avg   = volume.rolling(length).mean()
    atr_s     = atr(high, low, close, length=length)
    vol_cond  = volume > (vol_avg * vol_mult)
    rng_cond  = (high - low) < (atr_s * range_mult)
    result    = (vol_cond & rng_cond).fillna(False)
    result.name = f"ABSORPTION_{length}_{vol_mult}_{range_mult}"
    return result


def delta_proxy(open_: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Signed-volume proxy for buying/selling pressure (approximate delta).

    Bullish candle (close > open): all volume attributed to buyers  (+volume)
    Bearish candle (close < open): all volume attributed to sellers (-volume)
    Doji (close == open):          neutral (0)

    This is NOT true footprint delta (which requires tick-level bid/ask data).
    It is the same approximation used by the PickMyTrade Fabio Valentini script.

    Returns a signed float Series (positive = net buying, negative = net selling).
    """
    direction  = np.sign(close - open_)
    result     = direction * volume
    result.name = "DELTA_PROXY"
    return result


def volume_profile(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    lookback: int = 50,
    n_rows: int = 24,
) -> dict:
    """
    Simplified Volume Profile for the last `lookback` bars.

    Distributes each bar's volume proportionally across the price bins it spans,
    then identifies:
        POC — Point of Control: price level with the most traded volume
        VAH — Value Area High: upper boundary of the 70%-volume zone
        VAL — Value Area Low:  lower boundary of the 70%-volume zone

    Returns a dict:
        {'poc': float, 'vah': float, 'val': float,
         'levels': list[float], 'volumes': list[float]}

    'levels' and 'volumes' are the bin-centre prices and their aggregated volumes.
    """
    n   = min(lookback, len(high))
    h   = high.iloc[-n:].values.astype(float)
    lo  = low.iloc[-n:].values.astype(float)
    cl  = close.iloc[-n:].values.astype(float)
    vol = volume.iloc[-n:].values.astype(float)

    price_min = float(np.nanmin(lo))
    price_max = float(np.nanmax(h))

    # Degenerate case: constant price
    if price_max <= price_min:
        mid = (price_min + price_max) / 2.0
        return {"poc": mid, "vah": mid, "val": mid, "levels": [mid], "volumes": [float(vol.sum())]}

    bins        = np.linspace(price_min, price_max, n_rows + 1)
    bin_centres = (bins[:-1] + bins[1:]) / 2.0
    bin_vols    = np.zeros(n_rows)

    for i in range(n):
        bar_range = h[i] - lo[i]
        if bar_range <= 0:
            # Doji / single-tick — put all volume in the bin containing close
            idx = int(np.searchsorted(bins[1:], cl[i]))
            idx = min(idx, n_rows - 1)
            bin_vols[idx] += vol[i]
        else:
            for j in range(n_rows):
                overlap_lo  = max(lo[i], bins[j])
                overlap_hi  = min(h[i],  bins[j + 1])
                if overlap_hi > overlap_lo:
                    bin_vols[j] += vol[i] * (overlap_hi - overlap_lo) / bar_range

    # POC = bin with peak volume
    poc_idx = int(np.argmax(bin_vols))
    poc     = float(bin_centres[poc_idx])

    # Value Area: expand outward from POC until 70% of total volume is included
    total_vol  = float(bin_vols.sum())
    target     = total_vol * 0.70
    included   = np.zeros(n_rows, dtype=bool)
    included[poc_idx] = True
    accumulated = float(bin_vols[poc_idx])
    lo_ptr, hi_ptr = poc_idx - 1, poc_idx + 1

    while accumulated < target:
        lo_v = float(bin_vols[lo_ptr]) if lo_ptr >= 0        else -1.0
        hi_v = float(bin_vols[hi_ptr]) if hi_ptr < n_rows    else -1.0
        if lo_v < 0 and hi_v < 0:
            break
        if hi_v >= lo_v:
            included[hi_ptr] = True
            accumulated     += bin_vols[hi_ptr]
            hi_ptr          += 1
        else:
            included[lo_ptr] = True
            accumulated     += bin_vols[lo_ptr]
            lo_ptr          -= 1

    va_idx = np.where(included)[0]
    val    = float(bin_centres[va_idx.min()])
    vah    = float(bin_centres[va_idx.max()])

    return {
        "poc":     poc,
        "vah":     vah,
        "val":     val,
        "levels":  bin_centres.tolist(),
        "volumes": bin_vols.tolist(),
    }


def keltner_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 20,
    atr_mult: float = 1.5,
) -> pd.DataFrame:
    """
    Keltner Channels: EMA(close, length) ± atr_mult × ATR(length).
    Column order: KCL (lower), KCM (mid/EMA), KCU (upper)  — iloc 0, 1, 2.
    Used for Bollinger-Keltner squeeze detection.
    """
    mid   = close.ewm(span=length, adjust=False).mean()
    atr_s = atr(high, low, close, length=length)
    upper = mid + atr_mult * atr_s
    lower = mid - atr_mult * atr_s
    return pd.DataFrame({
        f"KCL_{length}_{atr_mult}": lower,
        f"KCM_{length}_{atr_mult}": mid,
        f"KCU_{length}_{atr_mult}": upper,
    }, index=close.index)


def zscore(close: pd.Series, length: int = 20) -> pd.Series:
    """
    Rolling Z-score: (close − rolling_mean) / rolling_std over `length` bars.
    Returns values centred at 0; ±2 are the entry thresholds for Z-score MR.
    """
    mean   = close.rolling(length).mean()
    std    = close.rolling(length).std(ddof=1)
    result = (close - mean) / std.replace(0, np.nan)
    result.name = f"ZSCORE_{length}"
    return result
