"""
Fast Swing Screener — bulk-downloads a curated universe in one yfinance call,
computes lightweight technical indicators, and returns filtered results.

Avoids the full run_analysis() pipeline intentionally — this is a scan tool,
not a full analysis. Each ticker takes ~0 extra seconds (vectorized in bulk).
"""

import pandas as pd
import numpy as np
import yfinance as yf

# ── Swing universe: 40 most liquid US equities across sectors ─────────────────
SWING_UNIVERSE = [
    # Technology
    "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "QCOM", "ORCL", "CRM",
    # Communication / Media
    "GOOGL", "META", "NFLX",
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "NKE", "MCD",
    # Consumer Staples
    "WMT", "COST", "PG",
    # Healthcare
    "UNH", "LLY", "ABBV", "MRK",
    # Financials
    "JPM", "GS", "V", "MA",
    # Industrials
    "CAT", "BA", "GE",
    # Energy
    "XOM", "CVX",
    # Materials / Real Estate / Utilities
    "LIN", "AMT", "NEE",
    # Broad ETFs (useful benchmarks in results)
    "SPY", "QQQ", "IWM",
]

# ── Filter presets ─────────────────────────────────────────────────────────────
PRESETS = {
    "Strong Long Setups":   {"min_score": 68, "direction": "LONG"},
    "Short Candidates":     {"max_score": 34, "direction": "SHORT"},
    "Stage 2 Breakouts":    {"min_score": 58, "direction": "LONG",  "stage2": True},
    "Oversold Bounces":     {"max_rsi": 38,   "min_score": 45,  "direction": "LONG"},
    "Momentum Leaders":     {"min_score": 75},
    "High-Volume Surges":   {"min_vol_ratio": 1.8},
    "All (no filter)":      {},
}


def run_screener(
    preset_name: str = "All (no filter)",
    custom_filters: dict | None = None,
    tickers: list | None = None,
) -> list[dict]:
    """
    Scan the swing universe (or a custom ticker list) and return matching rows.

    Returns a list of dicts sorted by score descending.
    Each row: ticker, score, direction, rsi, ema_trend, stage2,
              vol_ratio, price, chg_1d, signal.
    """
    universe = tickers or SWING_UNIVERSE
    filters  = custom_filters or PRESETS.get(preset_name, {})

    # ── Bulk download 6 months of daily data ──────────────────────────────────
    try:
        raw = yf.download(
            universe,
            period="7mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
    except Exception as e:
        raise ValueError(f"Download failed: {e}")

    results = []
    for ticker in universe:
        try:
            df = _extract_ticker(raw, ticker, universe)
            if df is None or len(df) < 60:
                continue

            row = _score_ticker(ticker, df)
            if row and _passes(row, filters):
                results.append(row)
        except Exception:
            continue

    return sorted(results, key=lambda x: x["score"], reverse=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_ticker(raw, ticker: str, universe: list) -> pd.DataFrame | None:
    """Extract single-ticker OHLCV from the bulk download DataFrame."""
    try:
        if len(universe) == 1:
            # Single ticker: columns are flat OHLCV
            df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
        elif isinstance(raw.columns, pd.MultiIndex):
            df = raw[ticker][["Open", "High", "Low", "Close", "Volume"]].dropna()
        else:
            return None
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception:
        return None


def _score_ticker(ticker: str, df: pd.DataFrame) -> dict | None:
    """Compute lightweight indicators and a quick composite score (0-100)."""
    close  = df["Close"].values.astype(float)
    volume = df["Volume"].values.astype(float)
    n      = len(close)

    # ── RSI(14) ───────────────────────────────────────────────────────────────
    rsi = _rsi(close, 14)

    # ── EMA 20 / 50 ───────────────────────────────────────────────────────────
    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)

    # ── SMA 150 (Weinstein 30-week) ────────────────────────────────────────────
    sma150     = float(pd.Series(close).rolling(150).mean().iloc[-1]) if n >= 150 else None
    sma150_4w  = float(pd.Series(close).rolling(150).mean().iloc[-21]) if n >= 171 else None

    stage2 = False
    if sma150 and sma150_4w:
        price_above = close[-1] > sma150
        sma_rising  = sma150 > sma150_4w
        stage2 = price_above and sma_rising

    # ── Volume ratio (vs 50-day avg) ──────────────────────────────────────────
    vol_sma50 = float(pd.Series(volume).rolling(50).mean().iloc[-1])
    vol_ratio = volume[-1] / vol_sma50 if vol_sma50 > 0 else 1.0

    # ── MACD histogram ────────────────────────────────────────────────────────
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd_line   = ema12 - ema26
    signal_line = _ema(
        pd.Series(_ema_series(close, 12)) - pd.Series(_ema_series(close, 26)),
        9,
    )
    hist = macd_line - signal_line

    # ── Quick composite score (0-100) ─────────────────────────────────────────
    score = 50.0

    # RSI contribution
    if rsi < 30:   score += 12
    elif rsi < 45: score += 6
    elif rsi > 70: score -= 12
    elif rsi > 60: score -= 4

    # EMA trend
    if close[-1] > ema20 > ema50:
        score += 10
    elif close[-1] < ema20 < ema50:
        score -= 10

    # Stage 2
    if stage2:   score += 8
    elif sma150 and close[-1] < sma150:
        score -= 8

    # MACD
    if hist > 0: score += 6
    else:        score -= 6

    # Volume
    if vol_ratio > 2.0:   score += 8
    elif vol_ratio > 1.4: score += 4
    elif vol_ratio < 0.6: score -= 4

    score = max(0.0, min(100.0, round(score, 1)))

    # ── Direction ─────────────────────────────────────────────────────────────
    if score >= 60:    direction = "LONG"
    elif score <= 38:  direction = "SHORT"
    else:              direction = "NEUTRAL"

    # ── Signal label ──────────────────────────────────────────────────────────
    if score >= 80:   signal = "STRONG BUY"
    elif score >= 65: signal = "BUY"
    elif score >= 50: signal = "WEAK BUY"
    elif score >= 35: signal = "NEUTRAL"
    elif score >= 20: signal = "WEAK SELL"
    else:             signal = "STRONG SELL"

    # ── 1-day change ──────────────────────────────────────────────────────────
    chg_1d = (close[-1] / close[-2] - 1) * 100 if n >= 2 else 0.0

    return {
        "ticker":    ticker,
        "price":     round(float(close[-1]), 2),
        "chg_1d":    round(chg_1d, 2),
        "score":     score,
        "signal":    signal,
        "direction": direction,
        "rsi":       round(rsi, 1),
        "ema_trend": "UP" if close[-1] > ema20 > ema50 else ("DOWN" if close[-1] < ema20 < ema50 else "MIX"),
        "stage2":    stage2,
        "vol_ratio": round(float(vol_ratio), 2),
    }


def _passes(row: dict, filters: dict) -> bool:
    if filters.get("min_score") and row["score"] < filters["min_score"]:
        return False
    if filters.get("max_score") and row["score"] > filters["max_score"]:
        return False
    if filters.get("direction") and row["direction"] != filters["direction"]:
        return False
    if filters.get("stage2") and not row["stage2"]:
        return False
    if filters.get("max_rsi") and row["rsi"] > filters["max_rsi"]:
        return False
    if filters.get("min_vol_ratio") and row["vol_ratio"] < filters["min_vol_ratio"]:
        return False
    return True


# ── TA helpers (pure numpy — no external lib needed) ──────────────────────────

def _rsi(close: np.ndarray, period: int = 14) -> float:
    if len(close) < period + 1:
        return 50.0
    deltas = np.diff(close)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - 100 / (1 + rs), 2)


def _ema(close, period: int) -> float:
    s = _ema_series(close, period)
    return float(s[-1]) if len(s) else float(close[-1])


def _ema_series(close, period: int) -> np.ndarray:
    if hasattr(close, "values"):
        arr = close.values.astype(float)
    else:
        arr = np.asarray(close, dtype=float)
    if len(arr) < period:
        return arr
    k   = 2 / (period + 1)
    out = np.empty(len(arr))
    out[period - 1] = arr[:period].mean()
    for i in range(period, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    out[:period - 1] = np.nan
    return out
