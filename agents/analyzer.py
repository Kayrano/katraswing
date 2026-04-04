"""
Expert Stock Market Analyzer Agent
Computes all technical indicators and detects chart patterns.
"""

import pandas as pd
import numpy as np
import utils.ta_compat as ta
from models.report import IndicatorBundle


class AnalyzerAgent:
    """
    Expert Stock Market Analyzer.
    Takes raw OHLCV DataFrame and returns a fully populated IndicatorBundle.
    """

    def analyze(self, df: pd.DataFrame) -> IndicatorBundle:
        df = df.copy()

        # ── RSI(14) ──────────────────────────────────────────────────────────
        rsi_series = ta.rsi(df["Close"], length=14)
        rsi = float(rsi_series.iloc[-1]) if rsi_series is not None and not rsi_series.empty else 50.0

        # ── MACD(12,26,9) ────────────────────────────────────────────────────
        macd_df = ta.macd(df["Close"], fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty:
            macd_line = float(macd_df.iloc[-1, 0])        # MACD_12_26_9
            macd_histogram = float(macd_df.iloc[-1, 1])   # MACDh_12_26_9
            macd_signal = float(macd_df.iloc[-1, 2])      # MACDs_12_26_9
            macd_histogram_prev = float(macd_df.iloc[-2, 1]) if len(macd_df) > 1 else macd_histogram
        else:
            macd_line = macd_signal = macd_histogram = macd_histogram_prev = 0.0

        # ── Bollinger Bands(20, 2) ───────────────────────────────────────────
        bb_df = ta.bbands(df["Close"], length=20, std=2)
        if bb_df is not None and not bb_df.empty:
            bb_lower = float(bb_df.iloc[-1, 0])   # BBL_20_2.0
            bb_mid   = float(bb_df.iloc[-1, 1])   # BBM_20_2.0
            bb_upper = float(bb_df.iloc[-1, 2])   # BBU_20_2.0
        else:
            close = float(df["Close"].iloc[-1])
            bb_lower = close * 0.97
            bb_mid   = close
            bb_upper = close * 1.03

        # ── EMA 20 / EMA 50 / SMA 200 ────────────────────────────────────────
        ema20_s = ta.ema(df["Close"], length=20)
        ema50_s = ta.ema(df["Close"], length=50)
        sma200_s = ta.sma(df["Close"], length=200)

        ema20  = float(ema20_s.iloc[-1])  if ema20_s  is not None and not ema20_s.isna().all()  else float(df["Close"].iloc[-1])
        ema50  = float(ema50_s.iloc[-1])  if ema50_s  is not None and not ema50_s.isna().all()  else float(df["Close"].iloc[-1])
        sma200 = float(sma200_s.iloc[-1]) if sma200_s is not None and not sma200_s.isna().all() else None

        # ── ATR(14) ──────────────────────────────────────────────────────────
        atr_s = ta.atr(df["High"], df["Low"], df["Close"], length=14)
        if atr_s is not None and not atr_s.isna().all():
            atr = float(atr_s.iloc[-1])
            atr_5d_ago = float(atr_s.iloc[-6]) if len(atr_s) > 5 else atr
        else:
            atr = float(df["Close"].iloc[-1]) * 0.02
            atr_5d_ago = atr

        # ── ATR Volatility Percentile ─────────────────────────────────────────
        if atr_s is not None and not atr_s.isna().all():
            atr_valid = atr_s.dropna()
            if len(atr_valid) >= 20:
                volatility_percentile = float(
                    np.sum(atr_valid <= atr) / len(atr_valid) * 100
                )
            else:
                volatility_percentile = 50.0
        else:
            volatility_percentile = 50.0

        # ── OBV ──────────────────────────────────────────────────────────────
        obv_s = ta.obv(df["Close"], df["Volume"])
        obv = float(obv_s.iloc[-1]) if obv_s is not None and not obv_s.empty else 0.0
        obv_10d_ago = float(obv_s.iloc[-11]) if obv_s is not None and len(obv_s) >= 11 else obv

        # ── Stochastic(14, 3) ────────────────────────────────────────────────
        stoch_df = ta.stoch(df["High"], df["Low"], df["Close"], k=14, d=3)
        if stoch_df is not None and not stoch_df.empty:
            stoch_k = float(stoch_df.iloc[-1, 0])        # STOCHk_14_3_3
            stoch_d = float(stoch_df.iloc[-1, 1])        # STOCHd_14_3_3
            stoch_k_prev = float(stoch_df.iloc[-2, 0]) if len(stoch_df) > 1 else stoch_k
        else:
            stoch_k = stoch_d = stoch_k_prev = 50.0

        # ── Volume SMA(20) ────────────────────────────────────────────────────
        volume_sma20 = float(df["Volume"].rolling(20).mean().iloc[-1])
        current_volume = float(df["Volume"].iloc[-1])

        # ── Pattern Detection ─────────────────────────────────────────────────
        # Golden / Death Cross: EMA20 vs EMA50 crossover in last 5 bars
        # Align by date index before comparing to avoid misalignment after dropna()
        golden_cross = False
        death_cross = False
        if ema20_s is not None and ema50_s is not None:
            aligned = pd.concat([ema20_s, ema50_s], axis=1).dropna()
            if len(aligned) >= 6:
                ema20_arr = aligned.iloc[:, 0].values
                ema50_arr = aligned.iloc[:, 1].values
                for i in range(-5, 0):
                    above = ema20_arr[i] > ema50_arr[i]
                    below_prev = ema20_arr[i - 1] <= ema50_arr[i - 1]
                    if above and below_prev:
                        golden_cross = True
                    cross_down = ema20_arr[i] < ema50_arr[i]
                    above_prev = ema20_arr[i - 1] >= ema50_arr[i - 1]
                    if cross_down and above_prev:
                        death_cross = True

        # BB Squeeze: bandwidth < 8% of midband
        bb_bandwidth = (bb_upper - bb_lower) / bb_mid if bb_mid != 0 else 0
        bb_squeeze = bb_bandwidth < 0.08

        # Volume Spike
        volume_spike = current_volume > (1.5 * volume_sma20)

        # Above 200 SMA
        close = float(df["Close"].iloc[-1])
        above_200_sma = (sma200 is not None) and (close > sma200)

        # ── RSI Divergence (last 20 bars) ───────────────────────────────────────
        rsi_divergence_bearish = False
        rsi_divergence_bullish = False
        if rsi_series is not None:
            div_df = pd.concat([df["Close"], rsi_series], axis=1).dropna()
            div_df.columns = ["close", "rsi"]
            if len(div_df) >= 20:
                rec = div_df.iloc[-10:]
                old_half = div_df.iloc[-20:-10]
                # Bearish: price higher high, RSI lower high
                rec_hi = rec["close"].idxmax()
                old_hi = old_half["close"].idxmax()
                if (rec.loc[rec_hi, "close"] > old_half.loc[old_hi, "close"] and
                        rec.loc[rec_hi, "rsi"] < old_half.loc[old_hi, "rsi"] - 2):
                    rsi_divergence_bearish = True
                # Bullish: price lower low, RSI higher low
                rec_lo = rec["close"].idxmin()
                old_lo = old_half["close"].idxmin()
                if (rec.loc[rec_lo, "close"] < old_half.loc[old_lo, "close"] and
                        rec.loc[rec_lo, "rsi"] > old_half.loc[old_lo, "rsi"] + 2):
                    rsi_divergence_bullish = True

        # Handle NaN values safely
        def safe_float(val, default=0.0):
            try:
                v = float(val)
                return v if not np.isnan(v) else default
            except Exception:
                return default

        return IndicatorBundle(
            rsi=safe_float(rsi, 50.0),
            macd_line=safe_float(macd_line),
            macd_signal=safe_float(macd_signal),
            macd_histogram=safe_float(macd_histogram),
            bb_upper=safe_float(bb_upper),
            bb_mid=safe_float(bb_mid),
            bb_lower=safe_float(bb_lower),
            ema20=safe_float(ema20),
            ema50=safe_float(ema50),
            sma200=safe_float(sma200) if sma200 is not None else None,
            atr=safe_float(atr),
            obv=safe_float(obv),
            stoch_k=safe_float(stoch_k, 50.0),
            stoch_d=safe_float(stoch_d, 50.0),
            volume_sma20=safe_float(volume_sma20),
            current_volume=safe_float(current_volume),
            golden_cross=golden_cross,
            death_cross=death_cross,
            bb_squeeze=bb_squeeze,
            volume_spike=volume_spike,
            above_200_sma=above_200_sma,
            close=safe_float(close),
            macd_histogram_prev=safe_float(macd_histogram_prev),
            stoch_k_prev=safe_float(stoch_k_prev, 50.0),
            atr_5d_ago=safe_float(atr_5d_ago),
            obv_10d_ago=safe_float(obv_10d_ago),
            rsi_divergence_bearish=rsi_divergence_bearish,
            rsi_divergence_bullish=rsi_divergence_bullish,
            volatility_percentile=volatility_percentile,
        )
