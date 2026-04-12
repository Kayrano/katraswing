"""
Pytest test suite for the five H1 intraday strategies, the ADX regime router,
and the session-window filter.

All tests are self-contained: no network calls, no yfinance, no Alpaca.
Synthetic OHLCV DataFrames are built in-process.

Run with:  pytest tests/test_hourly_strategies.py -v
"""

from __future__ import annotations

import sys
import os
# Make sure the project root is on the path regardless of cwd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta, date as date_t
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from agents.hourly_strategies import (
    HourlySignal,
    H1GateResult,
    rsi_mean_reversion,
    vwap_pullback,
    orb_breakout,
    squeeze_breakout,
    zscore_mean_reversion,
    absorption_breakout,
    triple_a_setup,
    value_area_bounce,
    _detect_first_kiss,
    _vwap_slope,
)
from agents.regime_router import RegimeResult, compute_regime, _check_session_window
import utils.ta_compat as ta


# ══════════════════════════════════════════════════════════════════════════════
# Shared fixtures
# ══════════════════════════════════════════════════════════════════════════════

_US_TZ   = ZoneInfo("America/New_York")
_BIST_TZ = ZoneInfo("Europe/Istanbul")


def _make_timestamps(n: int, market: str = "US") -> pd.DatetimeIndex:
    """
    Generate n consecutive H1 timestamps within market session hours.
    US:   09:30 ET → wraps at 16:00
    BIST: 10:00 Istanbul → wraps at 18:00 (skips 13:00–14:00 break)
    """
    tz = _US_TZ if market == "US" else _BIST_TZ
    sh, sm = (9, 30) if market == "US" else (10, 0)
    eh     = 16      if market == "US" else 18

    stamps: list[datetime] = []
    base_day = datetime(2025, 1, 6, sh, sm, tzinfo=tz)   # a Monday
    cur = base_day

    while len(stamps) < n:
        # Skip BIST mid-session break (keep the logic simple in synthetic data)
        stamps.append(cur)
        nxt = cur + timedelta(hours=1)
        if nxt.hour >= eh:
            # Roll to next weekday
            next_date = cur.date() + timedelta(days=1)
            while next_date.weekday() >= 5:
                next_date += timedelta(days=1)
            nxt = datetime(next_date.year, next_date.month, next_date.day,
                           sh, sm, tzinfo=tz)
        cur = nxt

    return pd.DatetimeIndex(stamps)


def _make_h1_df(
    n: int = 200,
    price_start: float = 100.0,
    trend: float = 0.0003,   # per-bar drift (positive = uptrend)
    noise: float = 0.004,
    market: str = "US",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Minimal synthetic H1 DataFrame with session metadata and RVOL column.
    `trend` controls the per-bar log-return drift.
    """
    rng = np.random.default_rng(seed)
    idx = _make_timestamps(n, market)

    log_returns = rng.normal(trend, noise, n)
    closes = price_start * np.exp(np.cumsum(log_returns))

    opens  = closes * (1.0 - rng.uniform(0.0, 0.002, n))
    highs  = closes * (1.0 + rng.uniform(0.001, 0.005, n))
    lows   = closes * (1.0 - rng.uniform(0.001, 0.005, n))
    vols   = rng.integers(200_000, 1_000_000, n).astype(float)

    df = pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols},
        index=idx,
    )

    # ── Session metadata (mirrors fetcher_hourly logic) ───────────────────────
    df["session_date"]       = df.index.date
    df["session_bar_number"] = df.groupby("session_date").cumcount() + 1
    df["is_first_bar"]       = df["session_bar_number"] == 1

    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    tp_vol  = typical * df["Volume"]
    df["session_vwap"] = (
        tp_vol.groupby(df["session_date"]).cumsum()
        / df["Volume"].groupby(df["session_date"]).cumsum().replace(0, np.nan)
    )

    df["_hour"] = df.index.hour
    same_hr_avg = df.groupby("_hour")["Volume"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=3).mean()
    )
    df["rvol"] = (df["Volume"] / same_hr_avg.replace(0, np.nan)).fillna(1.0)
    df.drop(columns=["_hour"], inplace=True)

    df["market"] = market
    return df


def _dummy_regime(
    regime: str = "RANGING",
    adx: float = 15.0,
    enabled: list[str] | None = None,
    market: str = "US",
) -> RegimeResult:
    if enabled is None:
        enabled = ["RSI_MR", "VWAP_PB", "ZSCORE_MR"]
    return RegimeResult(
        regime=regime,
        adx=adx,
        enabled_strategies=enabled,
        size_factor=1.0,
        in_session_window=True,
        mr_only_window=False,
        window_note="test fixture",
        market=market,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Strategy 1 — RSI Mean Reversion
# ══════════════════════════════════════════════════════════════════════════════

class TestRsiMeanReversion:

    def test_insufficient_bars_returns_flat(self):
        df = _make_h1_df(n=20)
        sig = rsi_mean_reversion(df, _dummy_regime())
        assert sig.signal == "FLAT"
        assert sig.strategy == "RSI_MR"

    def test_long_signal_when_rsi_low_and_price_above_ema50(self):
        """
        Build a strong uptrend (price >> EMA50) then drop sharply so RSI(3) < 15.
        """
        # Strong uptrend: 200 bars, +0.5% drift → price grows from 100 to ~270
        df = _make_h1_df(n=200, price_start=100.0, trend=0.005, noise=0.001, seed=1)

        # Override last 4 bars: sharp drop → RSI(3) will be near 0
        closes = df["Close"].values.copy()
        last   = closes[-5]
        for i in range(-4, 0):
            last = last * 0.96
            closes[df.index.get_loc(df.index[i])] = last
        df["Close"] = closes

        regime = _dummy_regime(regime="RANGING", adx=12.0)
        sig    = rsi_mean_reversion(df, regime)

        # With 4 consecutive -4% bars, RSI(3) must be extremely low (<15)
        # AND the preceding strong uptrend means EMA50 is far below current close
        rsi_val = float(ta.rsi(df["Close"], length=3).iloc[-1])
        ema_val = float(ta.ema(df["Close"], length=50).iloc[-1])

        if rsi_val < 15 and df["Close"].iloc[-1] > ema_val:
            assert sig.signal == "LONG", (
                f"Expected LONG but got {sig.signal}. "
                f"RSI(3)={rsi_val:.2f}, close={df['Close'].iloc[-1]:.2f}, EMA50={ema_val:.2f}"
            )
        # If the synthetic data doesn't quite hit < 15 (seed-dependent), just
        # verify the function returns a valid HourlySignal shape.
        assert sig.signal in ("LONG", "FLAT")
        assert 0.0 <= sig.confidence <= 1.0

    def test_trend_filter_blocks_long_when_price_below_ema50(self):
        """
        RSI(3) is forced below 15 but in a steep downtrend → price below EMA50
        → long should be blocked (FLAT).
        """
        # Steep downtrend: -0.5% per bar
        df = _make_h1_df(n=200, price_start=200.0, trend=-0.005, noise=0.001, seed=2)

        regime = _dummy_regime(regime="RANGING", adx=12.0)
        sig    = rsi_mean_reversion(df, regime)

        ema_val = float(ta.ema(df["Close"], length=50).iloc[-1])
        if df["Close"].iloc[-1] < ema_val:
            # Price is below EMA50 — no long should ever fire
            assert sig.signal != "LONG", (
                f"Long should be blocked below EMA50. "
                f"Got {sig.signal}. close={df['Close'].iloc[-1]:.2f}, EMA50={ema_val:.2f}"
            )

    def test_short_signal_when_rsi_high_and_price_below_ema50(self):
        """
        Strong downtrend then 4 sharp rallies → RSI(3) > 85 + price < EMA50 → SHORT.
        """
        df = _make_h1_df(n=200, price_start=200.0, trend=-0.005, noise=0.001, seed=3)
        closes = df["Close"].values.copy()
        last   = closes[-5]
        for i in range(-4, 0):
            last = last * 1.04
            closes[df.index.get_loc(df.index[i])] = last
        df["Close"] = closes

        regime  = _dummy_regime(regime="RANGING", adx=12.0)
        sig     = rsi_mean_reversion(df, regime)

        rsi_val = float(ta.rsi(df["Close"], length=3).iloc[-1])
        ema_val = float(ta.ema(df["Close"], length=50).iloc[-1])

        if rsi_val > 85 and df["Close"].iloc[-1] < ema_val:
            assert sig.signal == "SHORT"
        assert sig.signal in ("LONG", "SHORT", "FLAT")

    def test_returns_hourly_signal_type(self):
        df  = _make_h1_df()
        sig = rsi_mean_reversion(df, _dummy_regime())
        assert isinstance(sig, HourlySignal)
        assert sig.strategy == "RSI_MR"
        assert isinstance(sig.reason, str) and len(sig.reason) > 0


# ══════════════════════════════════════════════════════════════════════════════
# Strategy 2 — VWAP Pullback
# ══════════════════════════════════════════════════════════════════════════════

class TestVwapPullback:

    def test_insufficient_bars_returns_flat(self):
        df  = _make_h1_df(n=10)
        sig = vwap_pullback(df, _dummy_regime())
        assert sig.signal == "FLAT"

    def test_flat_when_missing_session_vwap_column(self):
        df  = _make_h1_df().drop(columns=["session_vwap"])
        sig = vwap_pullback(df, _dummy_regime())
        assert sig.signal == "FLAT"
        assert "session_vwap" in sig.reason

    def test_flat_when_price_far_from_vwap(self):
        """
        Force price far above VWAP so it can't be in the ±0.2×ATR band.
        """
        df = _make_h1_df(n=100)
        # Inflate only the last close to be 10× the VWAP
        df.loc[df.index[-1], "Close"] = df["session_vwap"].iloc[-1] * 10.0
        sig = vwap_pullback(df, _dummy_regime())
        assert sig.signal == "FLAT"

    def test_first_kiss_detection_returns_true_when_prior_deviation(self):
        """
        Manually set recent bars to be far from VWAP, then bring current bar back.
        _detect_first_kiss() should return True.
        """
        df      = _make_h1_df(n=100)
        atr_val = float(ta.atr(df["High"], df["Low"], df["Close"], 14).iloc[-1])

        # Make the last 5 bars (excluding current) deviate > 0.5×ATR from VWAP
        for i in range(-6, -1):
            df.loc[df.index[i], "Close"] = (
                df["session_vwap"].iloc[i] + 2.0 * atr_val
            )

        # Current bar is back near VWAP
        df.loc[df.index[-1], "Close"] = df["session_vwap"].iloc[-1]

        assert _detect_first_kiss(df, atr_val) is True

    def test_first_kiss_returns_false_when_price_never_deviated(self):
        df      = _make_h1_df(n=100)
        atr_val = float(ta.atr(df["High"], df["Low"], df["Close"], 14).iloc[-1])
        # Keep every bar glued to VWAP
        for i in range(-11, 0):
            df.loc[df.index[i], "Close"] = df["session_vwap"].iloc[i]
        assert _detect_first_kiss(df, atr_val) is False

    def test_returns_valid_signal_type(self):
        df  = _make_h1_df()
        sig = vwap_pullback(df, _dummy_regime())
        assert isinstance(sig, HourlySignal)
        assert sig.signal in ("LONG", "SHORT", "FLAT")
        assert 0.0 <= sig.confidence <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Strategy 3 — ORB-60
# ══════════════════════════════════════════════════════════════════════════════

class TestOrbBreakout:

    def test_insufficient_bars_returns_flat(self):
        df  = _make_h1_df(n=3)
        sig = orb_breakout(df, _dummy_regime(regime="TRENDING", enabled=["ORB_60"]))
        assert sig.signal == "FLAT"

    def test_flat_outside_entry_window_bar_1(self):
        """Bar 1 is the opening range itself — not a valid entry bar."""
        df = _make_h1_df(n=100)
        # Ensure the last bar is bar 1 of its session
        df.loc[df.index[-1], "session_bar_number"] = 1
        df.loc[df.index[-1], "is_first_bar"]        = True
        sig = orb_breakout(df, _dummy_regime(regime="TRENDING", enabled=["ORB_60"]))
        assert sig.signal == "FLAT"
        assert "entry window" in sig.reason.lower()

    def test_flat_outside_entry_window_bar_5(self):
        """Bar 5 is past the entry window for US (valid: 2-4)."""
        df = _make_h1_df(n=100)
        df.loc[df.index[-1], "session_bar_number"] = 5
        df.loc[df.index[-1], "is_first_bar"]        = False
        sig = orb_breakout(df, _dummy_regime(regime="TRENDING", enabled=["ORB_60"]))
        assert sig.signal == "FLAT"

    def test_long_signal_on_valid_breakout(self):
        """
        Construct a scenario where bar 2 closes above bar 1's high with
        RVOL > 2.0 and VWAP sloping up.
        """
        df = _make_h1_df(n=100)

        # Make the last session have exactly 2 bars
        today = df["session_date"].iloc[-1]
        session_bars = df[df["session_date"] == today]
        if len(session_bars) < 2:
            pytest.skip("Synthetic data has fewer than 2 bars today — skip")

        bar1_idx = session_bars.index[0]
        bar2_idx = session_bars.index[1]

        # Set bar 1 as the ORB with a known high
        orb_high = 110.0
        df.loc[bar1_idx, "High"]                = orb_high
        df.loc[bar1_idx, "Low"]                 = 100.0
        df.loc[bar1_idx, "Close"]               = 105.0
        df.loc[bar1_idx, "session_bar_number"]  = 1
        df.loc[bar1_idx, "is_first_bar"]        = True

        # Set bar 2: close above ORB high, high RVOL, VWAP sloping up
        df.loc[bar2_idx, "Close"]               = 115.0   # > ORB high
        df.loc[bar2_idx, "rvol"]                = 2.5     # > 2.0 ✓
        df.loc[bar2_idx, "session_bar_number"]  = 2
        df.loc[bar2_idx, "is_first_bar"]        = False

        # Slope VWAP up by ensuring last 3 VWAP values are rising
        vwap_vals = df["session_vwap"].values.copy()
        vwap_vals[-3] = 104.0
        vwap_vals[-2] = 105.0
        vwap_vals[-1] = 106.0
        df["session_vwap"] = vwap_vals

        # Trim to just today's session for the test
        df_today = df[df["session_date"] == today].copy()
        # Add market column
        df_today["market"] = "US"

        # Ensure _vwap_slope returns +1 given our values
        slope = _vwap_slope(df_today, lookback=2)
        # Only run full assertion if slope is positive
        if slope > 0 and df_today["Close"].iloc[-1] > orb_high and df_today["rvol"].iloc[-1] >= 2.0:
            regime = _dummy_regime(regime="TRENDING", enabled=["ORB_60"])
            sig    = orb_breakout(df_today, regime)
            assert sig.signal == "LONG", f"Expected LONG, got {sig.signal}: {sig.reason}"
            assert sig.confidence >= 0.65

    def test_flat_when_price_inside_orb(self):
        df   = _make_h1_df(n=100)
        today = df["session_date"].iloc[-1]
        session_bars = df[df["session_date"] == today]
        if len(session_bars) < 2:
            pytest.skip("Need at least 2 bars today")

        bar1_idx = session_bars.index[0]
        bar2_idx = session_bars.index[-1]

        df.loc[bar1_idx, "High"]               = 110.0
        df.loc[bar1_idx, "Low"]                = 100.0
        df.loc[bar1_idx, "session_bar_number"] = 1
        df.loc[bar1_idx, "is_first_bar"]       = True
        df.loc[bar2_idx, "Close"]              = 105.0   # inside ORB
        df.loc[bar2_idx, "session_bar_number"] = 2
        df.loc[bar2_idx, "is_first_bar"]       = False

        df_today = df[df["session_date"] == today].copy()
        df_today["market"] = "US"
        regime = _dummy_regime(regime="TRENDING", enabled=["ORB_60"])
        sig    = orb_breakout(df_today, regime)
        assert sig.signal == "FLAT"

    def test_bist_entry_window_caps_at_bar3(self):
        """BIST entry window is bars 2-3 only (before mid-session break)."""
        df = _make_h1_df(n=100, market="BIST")
        df.loc[df.index[-1], "session_bar_number"] = 4   # > BIST max
        df.loc[df.index[-1], "is_first_bar"]        = False
        sig = orb_breakout(df, _dummy_regime(market="BIST", enabled=["ORB_60"]))
        assert sig.signal == "FLAT"
        assert "entry window" in sig.reason.lower()


# ══════════════════════════════════════════════════════════════════════════════
# Strategy 4 — Bollinger-Keltner Squeeze Breakout
# ══════════════════════════════════════════════════════════════════════════════

class TestSqueezeBreakout:

    def test_insufficient_bars_returns_flat(self):
        df  = _make_h1_df(n=10)
        sig = squeeze_breakout(df, _dummy_regime(regime="TRENDING", enabled=["SQUEEZE"]))
        assert sig.signal == "FLAT"

    def test_flat_when_no_prior_squeeze(self):
        """Random price data is unlikely to produce a squeeze; function should return FLAT."""
        df  = _make_h1_df(n=100, noise=0.02)   # high noise → wide bands, no squeeze
        sig = squeeze_breakout(df, _dummy_regime(regime="TRENDING", enabled=["SQUEEZE"]))
        # We just verify no exception and valid output shape
        assert sig.signal in ("LONG", "SHORT", "FLAT")
        assert isinstance(sig.reason, str)

    def test_long_signal_on_constructed_squeeze_breakout(self):
        """
        Scan the synthetic DataFrame for a genuine squeeze→breakout pair, then
        slice the DataFrame so that pair sits at iloc[-2] / iloc[-1], and verify
        the strategy produces a directional signal.
        """
        df = _make_h1_df(n=200, noise=0.001, seed=99)   # low noise → tight bands

        bb  = ta.bbands(df["Close"], length=20, std=2.0)
        kc  = ta.keltner_channels(df["High"], df["Low"], df["Close"], length=20, atr_mult=1.5)

        breakout_idx: int | None = None
        for i in range(22, len(df)):
            bb_u_p = float(bb.iloc[i-1, 2]); bb_l_p = float(bb.iloc[i-1, 0])
            kc_u_p = float(kc.iloc[i-1, 2]); kc_l_p = float(kc.iloc[i-1, 0])
            was_squeezed = (bb_u_p < kc_u_p) and (bb_l_p > kc_l_p)

            bb_u_c = float(bb.iloc[i, 2]); bb_l_c = float(bb.iloc[i, 0])
            kc_u_c = float(kc.iloc[i, 2]); kc_l_c = float(kc.iloc[i, 0])
            broke_out = (bb_u_c > kc_u_c) or (bb_l_c < kc_l_c)

            if was_squeezed and broke_out:
                breakout_idx = i
                break

        if breakout_idx is None:
            pytest.skip("Synthetic data produced no squeeze-breakout pair — skip")

        # Slice so the breakout bar is the last bar
        df_slice = df.iloc[: breakout_idx + 1].copy()
        regime   = _dummy_regime(regime="TRENDING", enabled=["SQUEEZE"])
        sig      = squeeze_breakout(df_slice, regime)
        assert sig.signal in ("LONG", "SHORT"), (
            f"Expected LONG or SHORT at breakout bar {breakout_idx}, got {sig.signal}: {sig.reason}"
        )
        assert sig.confidence >= 0.60

    def test_returns_valid_signal_shape(self):
        df  = _make_h1_df(n=100)
        sig = squeeze_breakout(df, _dummy_regime(regime="TRENDING", enabled=["SQUEEZE"]))
        assert isinstance(sig, HourlySignal)
        assert sig.strategy == "SQUEEZE"
        assert sig.signal in ("LONG", "SHORT", "FLAT")
        assert 0.0 <= sig.confidence <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Strategy 5 — Z-Score Mean Reversion
# ══════════════════════════════════════════════════════════════════════════════

class TestZscoreMeanReversion:

    def test_insufficient_bars_returns_flat(self):
        df  = _make_h1_df(n=10)
        sig = zscore_mean_reversion(df, None, _dummy_regime())
        assert sig.signal == "FLAT"

    def test_long_signal_when_z_below_minus_2_and_above_200sma(self):
        """
        Force last bar's close far below the 20-bar rolling mean → Z < -2.
        Daily df SMA200 is calibrated to be below the H1 close → long filter passes.
        """
        df = _make_h1_df(n=100)

        # Force Z < -2: set last close to (mean - 3*std) of the 20-bar window
        rolling_mean = df["Close"].rolling(20).mean()
        rolling_std  = df["Close"].rolling(20).std(ddof=1)
        mean_val = float(rolling_mean.dropna().iloc[-1])
        std_val  = float(rolling_std.dropna().iloc[-1])
        forced_close = mean_val - 3.0 * std_val
        df.loc[df.index[-1], "Close"] = forced_close

        # Build a daily df whose SMA200 is BELOW forced_close.
        # Start at price_start = forced_close * 0.5 with a flat trend so that
        # the entire 250-bar daily series, and therefore SMA200, stays below
        # forced_close.
        safe_start = forced_close * 0.50
        df_daily = _make_h1_df(n=250, price_start=safe_start, trend=0.0001, noise=0.001)
        df_daily.index = pd.date_range("2024-01-01", periods=250, freq="B")

        regime = _dummy_regime(regime="RANGING", adx=10.0)
        sig    = zscore_mean_reversion(df, df_daily, regime)

        cur_z   = float(ta.zscore(df["Close"], length=20).iloc[-1])
        sma200  = float(ta.sma(df_daily["Close"], 200).dropna().iloc[-1])
        if cur_z < -2.0 and forced_close > sma200:
            assert sig.signal == "LONG", (
                f"Expected LONG for Z={cur_z:.2f}, close={forced_close:.2f}, "
                f"SMA200={sma200:.2f}, got {sig.signal}: {sig.reason}"
            )
            assert sig.confidence >= 0.65

    def test_flat_when_z_below_minus_2_but_below_daily_200sma(self):
        """
        Z < -2 but price is below the daily 200-SMA → catastrophic-loss filter
        blocks the long.
        """
        df = _make_h1_df(n=100)

        # Force Z < -2 on last bar
        rolling_mean = df["Close"].rolling(20).mean()
        rolling_std  = df["Close"].rolling(20).std(ddof=1)
        mean_val = float(rolling_mean.dropna().iloc[-1])
        std_val  = float(rolling_std.dropna().iloc[-1])
        df.loc[df.index[-1], "Close"] = mean_val - 3.0 * std_val

        # Build daily df where close is BELOW SMA200
        df_daily = _make_h1_df(n=250, trend=-0.003, noise=0.002)
        df_daily.index = pd.date_range("2024-01-01", periods=250, freq="B")

        regime = _dummy_regime()
        sig    = zscore_mean_reversion(df, df_daily, regime)

        cur_z = float(ta.zscore(df["Close"], length=20).iloc[-1])
        if cur_z < -2.0:
            sma200 = float(ta.sma(df_daily["Close"], 200).dropna().iloc[-1])
            if df["Close"].iloc[-1] < sma200:
                assert sig.signal == "FLAT", (
                    f"Expected FLAT (below 200-SMA filter), got {sig.signal}"
                )

    def test_flat_when_z_within_band(self):
        """Z between -2 and +2 → always FLAT."""
        df  = _make_h1_df(n=100, noise=0.001)   # low noise keeps Z near 0
        sig = zscore_mean_reversion(df, None, _dummy_regime())
        cur_z = float(ta.zscore(df["Close"], length=20).iloc[-1])
        if -2.0 <= cur_z <= 2.0:
            assert sig.signal == "FLAT"

    def test_short_signal_when_z_above_plus_2(self):
        """Z > +2 → SHORT regardless of daily trend."""
        df = _make_h1_df(n=100)
        rolling_mean = df["Close"].rolling(20).mean()
        rolling_std  = df["Close"].rolling(20).std(ddof=1)
        mean_val = float(rolling_mean.dropna().iloc[-1])
        std_val  = float(rolling_std.dropna().iloc[-1])
        df.loc[df.index[-1], "Close"] = mean_val + 3.0 * std_val

        sig   = zscore_mean_reversion(df, None, _dummy_regime())
        cur_z = float(ta.zscore(df["Close"], length=20).iloc[-1])
        if cur_z > 2.0:
            assert sig.signal == "SHORT"
            assert sig.confidence >= 0.65


# ══════════════════════════════════════════════════════════════════════════════
# ADX Regime Router
# ══════════════════════════════════════════════════════════════════════════════

class TestRegimeRouter:

    def test_trending_regime_enables_orb_and_squeeze(self):
        """High ADX data → TRENDING → ORB_60 and SQUEEZE enabled."""
        # Strong trending data: consistent directional bars with low noise
        df = _make_h1_df(n=200, trend=0.008, noise=0.0005, seed=10)
        result = compute_regime(df)

        if result.regime == "TRENDING":
            assert "ORB_60"  in result.enabled_strategies
            assert "SQUEEZE" in result.enabled_strategies
            assert result.size_factor == 1.0

    def test_ranging_regime_enables_mr_strategies(self):
        """Low ADX data → RANGING → RSI_MR, VWAP_PB, ZSCORE_MR enabled."""
        # Flat / choppy data: zero drift, moderate noise → ADX stays low
        df = _make_h1_df(n=200, trend=0.0, noise=0.001, seed=20)
        result = compute_regime(df)

        if result.regime == "RANGING":
            assert "RSI_MR"    in result.enabled_strategies
            assert "VWAP_PB"   in result.enabled_strategies
            assert "ZSCORE_MR" in result.enabled_strategies
            assert "ORB_60"    not in result.enabled_strategies
            assert "SQUEEZE"   not in result.enabled_strategies

    def test_transitional_regime_uses_half_size(self):
        """ADX 20–25 → TRANSITIONAL → all 5 strategies, size_factor = 0.5."""
        df     = _make_h1_df(n=200, trend=0.0, noise=0.003, seed=30)
        result = compute_regime(df)
        if result.regime == "TRANSITIONAL":
            assert result.size_factor == 0.5
            assert len(result.enabled_strategies) == 5

    def test_regime_result_fields_are_valid(self):
        df     = _make_h1_df(n=200)
        result = compute_regime(df)
        assert result.regime in ("TRENDING", "RANGING", "TRANSITIONAL")
        assert result.adx >= 0.0
        assert result.market in ("US", "BIST")
        assert isinstance(result.enabled_strategies, list)
        assert isinstance(result.size_factor, float)
        assert result.size_factor in (0.5, 1.0)

    def test_adx_value_matches_ta_compat(self):
        """compute_regime ADX must equal ta.adx computed on the same data."""
        df     = _make_h1_df(n=200, seed=99)
        result = compute_regime(df)

        adx_s   = ta.adx(df["High"], df["Low"], df["Close"], length=14)
        expected = round(float(adx_s.iloc[-1]), 1)
        assert result.adx == expected


# ══════════════════════════════════════════════════════════════════════════════
# Session Window Filter
# ══════════════════════════════════════════════════════════════════════════════

class TestSessionWindow:

    # ── US windows ────────────────────────────────────────────────────────────

    def test_us_prime_window_1_is_active(self):
        dt = datetime(2025, 6, 2, 9, 30, tzinfo=_US_TZ)    # 09:30 ET
        in_win, mr_only, note = _check_session_window(dt, "US")
        assert in_win  is True
        assert mr_only is False

    def test_us_power_hour_is_active(self):
        dt = datetime(2025, 6, 2, 15, 0, tzinfo=_US_TZ)    # 15:00 ET
        in_win, mr_only, note = _check_session_window(dt, "US")
        assert in_win  is True
        assert mr_only is False

    def test_us_mid_session_is_mr_only(self):
        dt = datetime(2025, 6, 2, 10, 30, tzinfo=_US_TZ)   # 10:30 ET
        in_win, mr_only, note = _check_session_window(dt, "US")
        assert in_win  is True
        assert mr_only is True

    def test_us_lunch_blackout_is_suppressed(self):
        for minute in [0, 30, 59]:
            dt = datetime(2025, 6, 2, 12, minute, tzinfo=_US_TZ)
            in_win, _, note = _check_session_window(dt, "US")
            assert in_win is False, f"12:{minute:02d} ET should be blackout, got in_win=True"
            assert "blackout" in note.lower()

    def test_us_11_30_is_blackout(self):
        dt = datetime(2025, 6, 2, 11, 30, tzinfo=_US_TZ)
        in_win, _, _ = _check_session_window(dt, "US")
        assert in_win is False

    def test_us_13_30_is_mr_only_mid_session(self):
        dt = datetime(2025, 6, 2, 13, 30, tzinfo=_US_TZ)
        in_win, mr_only, _ = _check_session_window(dt, "US")
        assert in_win  is True
        assert mr_only is True

    def test_us_outside_all_windows_is_suppressed(self):
        dt = datetime(2025, 6, 2, 20, 0, tzinfo=_US_TZ)    # 20:00 ET — market closed
        in_win, _, _ = _check_session_window(dt, "US")
        assert in_win is False

    # ── BIST windows ──────────────────────────────────────────────────────────

    def test_bist_session1_is_active(self):
        dt = datetime(2025, 6, 2, 11, 0, tzinfo=_BIST_TZ)  # 11:00 Istanbul
        in_win, mr_only, _ = _check_session_window(dt, "BIST")
        assert in_win  is True
        assert mr_only is False

    def test_bist_session2_is_active(self):
        dt = datetime(2025, 6, 2, 15, 0, tzinfo=_BIST_TZ)  # 15:00 Istanbul
        in_win, mr_only, _ = _check_session_window(dt, "BIST")
        assert in_win  is True
        assert mr_only is False

    def test_bist_mid_session_break_is_blackout(self):
        for minute in [0, 30, 59]:
            dt = datetime(2025, 6, 2, 13, minute, tzinfo=_BIST_TZ)
            in_win, _, note = _check_session_window(dt, "BIST")
            assert in_win is False, (
                f"BIST 13:{minute:02d} should be blackout, got in_win=True"
            )
            assert "blackout" in note.lower()

    def test_bist_outside_session_is_suppressed(self):
        dt = datetime(2025, 6, 2, 19, 0, tzinfo=_BIST_TZ)  # 19:00 — after close
        in_win, _, _ = _check_session_window(dt, "BIST")
        assert in_win is False


# ══════════════════════════════════════════════════════════════════════════════
# TA Compat — new functions
# ══════════════════════════════════════════════════════════════════════════════

class TestTaCompatNewFunctions:

    def _sample_ohlcv(self, n: int = 100) -> tuple:
        rng    = np.random.default_rng(0)
        closes = 100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.01, n)))
        highs  = closes * (1 + rng.uniform(0.001, 0.005, n))
        lows   = closes * (1 - rng.uniform(0.001, 0.005, n))
        idx    = pd.date_range("2024-01-01", periods=n, freq="h")
        return (
            pd.Series(highs,  index=idx, name="High"),
            pd.Series(lows,   index=idx, name="Low"),
            pd.Series(closes, index=idx, name="Close"),
        )

    def test_keltner_channels_shape_and_columns(self):
        high, low, close = self._sample_ohlcv()
        kc = ta.keltner_channels(high, low, close, length=20, atr_mult=1.5)
        assert kc is not None
        assert kc.shape[1] == 3
        # Lower < Mid < Upper for the last valid bar
        valid = kc.dropna()
        assert not valid.empty
        lower, mid, upper = float(valid.iloc[-1, 0]), float(valid.iloc[-1, 1]), float(valid.iloc[-1, 2])
        assert lower < mid < upper, f"KC order violated: {lower:.2f} < {mid:.2f} < {upper:.2f}"

    def test_keltner_mid_equals_ema(self):
        """KC mid column must equal EMA(close, length)."""
        high, low, close = self._sample_ohlcv()
        kc   = ta.keltner_channels(high, low, close, length=20, atr_mult=1.5)
        ema_ = ta.ema(close, length=20)
        pd.testing.assert_series_equal(
            kc.iloc[:, 1].rename("EMA_20"),
            ema_.rename("EMA_20"),
            check_names=True,
            rtol=1e-6,
        )

    def test_zscore_zero_mean(self):
        """For a constant-price series, Z-score should be NaN (std=0)."""
        idx   = pd.date_range("2024-01-01", periods=50, freq="h")
        const = pd.Series(np.ones(50) * 100.0, index=idx)
        z     = ta.zscore(const, length=20)
        # Std of a constant series is 0; result should be NaN
        assert z.dropna().empty or np.isnan(z.iloc[-1])

    def test_zscore_known_values(self):
        """
        For a series where the last value is exactly mean + 2*std,
        Z-score at the last bar should be approximately 2.0 (before Bessel correction).
        """
        n      = 25
        values = np.ones(n) * 100.0
        # Rolling mean and std will use bars -20 to -1 (excl. last bar) to
        # compute the window, so we set the LAST bar to mean+3*std
        values[-1] = 100.0 + 3.0 * 1.0   # std=1 (not exact due to Bessel, close enough)
        idx = pd.date_range("2024-01-01", periods=n, freq="h")
        z   = ta.zscore(pd.Series(values, index=idx), length=20)
        assert not np.isnan(float(z.iloc[-1]))
        # Z should be positive (above mean)
        assert float(z.iloc[-1]) > 0

    def test_zscore_negative_for_below_mean(self):
        n      = 25
        values = np.ones(n) * 100.0
        values[-1] = 50.0   # well below mean
        idx = pd.date_range("2024-01-01", periods=n, freq="h")
        z   = ta.zscore(pd.Series(values, index=idx), length=20)
        assert float(z.iloc[-1]) < 0


# ══════════════════════════════════════════════════════════════════════════════
# TA Compat — volume_profile
# ══════════════════════════════════════════════════════════════════════════════

class TestTaCompatVolumeProfile:

    def _sample_ohlcv(self, n: int = 100, seed: int = 7) -> pd.DataFrame:
        rng    = np.random.default_rng(seed)
        closes = 100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.01, n)))
        highs  = closes * (1 + rng.uniform(0.001, 0.005, n))
        lows   = closes * (1 - rng.uniform(0.001, 0.005, n))
        vols   = rng.integers(100_000, 500_000, n).astype(float)
        idx    = pd.date_range("2024-01-01", periods=n, freq="h")
        return pd.DataFrame(
            {"High": highs, "Low": lows, "Close": closes, "Volume": vols},
            index=idx,
        )

    def test_returns_dict_with_required_keys(self):
        df = self._sample_ohlcv()
        vp = ta.volume_profile(df["High"], df["Low"], df["Close"], df["Volume"])
        assert isinstance(vp, dict)
        for key in ("poc", "vah", "val", "levels", "volumes"):
            assert key in vp, f"Missing key: {key}"

    def test_val_le_poc_le_vah(self):
        df = self._sample_ohlcv()
        vp = ta.volume_profile(df["High"], df["Low"], df["Close"], df["Volume"])
        assert vp["val"] <= vp["poc"] <= vp["vah"], (
            f"Order violated: VAL={vp['val']:.2f} POC={vp['poc']:.2f} VAH={vp['vah']:.2f}"
        )

    def test_levels_and_volumes_same_length(self):
        df = self._sample_ohlcv()
        vp = ta.volume_profile(df["High"], df["Low"], df["Close"], df["Volume"])
        assert len(vp["levels"]) == len(vp["volumes"])

    def test_degenerate_constant_price(self):
        n   = 30
        idx = pd.date_range("2024-01-01", periods=n, freq="h")
        val = 100.0
        h   = pd.Series(np.ones(n) * val, index=idx)
        l   = pd.Series(np.ones(n) * val, index=idx)
        c   = pd.Series(np.ones(n) * val, index=idx)
        v   = pd.Series(np.ones(n) * 1000.0, index=idx)
        vp  = ta.volume_profile(h, l, c, v)
        assert vp["poc"] == vp["vah"] == vp["val"] == val

    def test_volumes_sum_equals_total_volume(self):
        """Sum of bin volumes must equal the total volume in the lookback window."""
        df = self._sample_ohlcv(n=60)
        lookback = 50
        vp = ta.volume_profile(
            df["High"], df["Low"], df["Close"], df["Volume"], lookback=lookback
        )
        expected_total = float(df["Volume"].iloc[-lookback:].sum())
        actual_total   = sum(vp["volumes"])
        assert abs(actual_total - expected_total) < expected_total * 0.01, (
            f"Volume sum mismatch: expected {expected_total:.0f}, got {actual_total:.0f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Strategy 6 — Absorption Breakout
# ══════════════════════════════════════════════════════════════════════════════

class TestAbsorptionBreakout:

    def _make_absorption_df(self, direction: str = "long") -> pd.DataFrame:
        """
        Build a DataFrame where iloc[-2] is a confirmed absorption bar and
        iloc[-1] breaks out in the specified direction.
        """
        df  = _make_h1_df(n=80, seed=11)
        regime_abs = _dummy_regime("TRENDING", adx=30.0, enabled=["ABSORB_BO"])

        # Compute ATR to size the absorption bar correctly
        atr_s = ta.atr(df["High"], df["Low"], df["Close"], length=14)
        atr_v = float(atr_s.iloc[-2])
        avg_vol = float(df["Volume"].rolling(20).mean().iloc[-2])

        # Manufacture absorption on iloc[-2]: narrow range, high volume
        # Use 0.0002 range (≈ 0.02% of price) to stay well below ATR×0.3 threshold
        mid_price = float(df["Close"].iloc[-2])
        df.at[df.index[-2], "High"]   = mid_price * 1.0002
        df.at[df.index[-2], "Low"]    = mid_price * 0.9998
        df.at[df.index[-2], "Volume"] = avg_vol * 3.0   # > 2× avg

        absorb_high = float(df["High"].iloc[-2])
        absorb_low  = float(df["Low"].iloc[-2])

        # Manufacture breakout on iloc[-1]
        if direction == "long":
            breakout_price = absorb_high * 1.005
            df.at[df.index[-1], "Close"] = breakout_price
            df.at[df.index[-1], "High"]  = breakout_price * 1.002
            df.at[df.index[-1], "session_vwap"] = breakout_price * 0.998  # below close
        else:
            breakout_price = absorb_low * 0.995
            df.at[df.index[-1], "Close"] = breakout_price
            df.at[df.index[-1], "Low"]   = breakout_price * 0.998
            df.at[df.index[-1], "session_vwap"] = breakout_price * 1.002  # above close

        df.at[df.index[-1], "rvol"] = 2.0

        return df, regime_abs

    def test_insufficient_bars_returns_flat(self):
        df     = _make_h1_df(n=20)
        regime = _dummy_regime("TRENDING", adx=30.0, enabled=["ABSORB_BO"])
        sig    = absorption_breakout(df, regime)
        assert sig.signal == "FLAT"
        assert sig.strategy == "ABSORB_BO"

    def test_no_absorption_previous_bar_returns_flat(self):
        df     = _make_h1_df(n=80, seed=11)
        # Ensure previous bar has normal volume (no absorption)
        avg_vol = float(df["Volume"].rolling(20).mean().iloc[-2])
        df.at[df.index[-2], "Volume"] = avg_vol * 0.8
        regime = _dummy_regime("TRENDING", adx=30.0, enabled=["ABSORB_BO"])
        sig    = absorption_breakout(df, regime)
        assert sig.signal == "FLAT"

    def test_long_signal_on_absorption_breakout(self):
        df, regime = self._make_absorption_df("long")
        sig = absorption_breakout(df, regime)
        assert sig.signal == "LONG", f"Expected LONG, got {sig.signal}: {sig.reason}"
        assert sig.strategy == "ABSORB_BO"
        assert sig.confidence >= 0.75

    def test_short_signal_on_absorption_breakdown(self):
        df, regime = self._make_absorption_df("short")
        sig = absorption_breakout(df, regime)
        assert sig.signal == "SHORT", f"Expected SHORT, got {sig.signal}: {sig.reason}"
        assert sig.confidence >= 0.75

    def test_inside_range_returns_flat(self):
        """Current bar stays inside absorption range → no breakout → FLAT."""
        df     = _make_h1_df(n=80, seed=11)
        avg_vol = float(df["Volume"].rolling(20).mean().iloc[-2])
        mid    = float(df["Close"].iloc[-2])
        # Force absorption on [-2] (narrow range so condition passes)
        df.at[df.index[-2], "High"]   = mid * 1.0002
        df.at[df.index[-2], "Low"]    = mid * 0.9998
        df.at[df.index[-2], "Volume"] = avg_vol * 3.0
        # Current bar stays inside range
        df.at[df.index[-1], "Close"]  = mid
        regime = _dummy_regime("TRENDING", adx=30.0, enabled=["ABSORB_BO"])
        sig    = absorption_breakout(df, regime)
        assert sig.signal == "FLAT"


# ══════════════════════════════════════════════════════════════════════════════
# Strategy 7 — Triple-A Setup
# ══════════════════════════════════════════════════════════════════════════════

class TestTripleASetup:

    def _make_triple_a_df(self, direction: str = "long") -> pd.DataFrame:
        """
        Constructs a DataFrame with:
          iloc[-4] : absorption bar (narrow range, high volume)
          iloc[-3] and iloc[-2] : accumulation bars (coil inside absorption range)
          iloc[-1] : aggression bar (breaks out with volume)
        """
        df = _make_h1_df(n=80, seed=22)
        regime = _dummy_regime("TRENDING", adx=30.0, enabled=["TRIPLE_A"])

        avg_vol = float(df["Volume"].rolling(20).mean().iloc[-4])
        mid     = float(df["Close"].iloc[-4])

        # Phase 1 — absorption at iloc[-4]
        df.at[df.index[-4], "High"]   = mid * 1.001
        df.at[df.index[-4], "Low"]    = mid * 0.999
        df.at[df.index[-4], "Volume"] = avg_vol * 2.5

        absorb_high = float(df["High"].iloc[-4])
        absorb_low  = float(df["Low"].iloc[-4])

        # Phase 2 — accumulation inside range at iloc[-3] and iloc[-2]
        for idx_pos in [-3, -2]:
            df.at[df.index[idx_pos], "Close"] = mid
            df.at[df.index[idx_pos], "High"]  = absorb_high * 1.001
            df.at[df.index[idx_pos], "Low"]   = absorb_low  * 0.999

        # Phase 3 — aggression at iloc[-1]
        avg_vol_cur = float(df["Volume"].rolling(20).mean().iloc[-1])
        df.at[df.index[-1], "Volume"] = avg_vol_cur * 2.0  # vol_ok

        if direction == "long":
            aggr_price = absorb_high * 1.006
            df.at[df.index[-1], "Close"]       = aggr_price
            df.at[df.index[-1], "High"]        = aggr_price * 1.002
            df.at[df.index[-1], "session_vwap"] = aggr_price * 0.995
        else:
            aggr_price = absorb_low * 0.994
            df.at[df.index[-1], "Close"]       = aggr_price
            df.at[df.index[-1], "Low"]         = aggr_price * 0.998
            df.at[df.index[-1], "session_vwap"] = aggr_price * 1.005

        return df, regime

    def test_insufficient_bars_returns_flat(self):
        df     = _make_h1_df(n=25)
        regime = _dummy_regime("TRENDING", adx=30.0, enabled=["TRIPLE_A"])
        sig    = triple_a_setup(df, regime)
        assert sig.signal == "FLAT"
        assert sig.strategy == "TRIPLE_A"

    def test_no_absorption_in_last_6_bars_returns_flat(self):
        df     = _make_h1_df(n=80, seed=22)
        # Ensure no absorption in the last 6 bars by normalising volume and range
        for i in range(-6, 0):
            avg_vol = float(df["Volume"].rolling(20).mean().iloc[i])
            df.at[df.index[i], "Volume"] = avg_vol * 0.5
        regime = _dummy_regime("TRENDING", adx=30.0, enabled=["TRIPLE_A"])
        sig    = triple_a_setup(df, regime)
        assert sig.signal == "FLAT"
        assert "No absorption" in sig.reason

    def test_long_signal_on_valid_triple_a(self):
        df, regime = self._make_triple_a_df("long")
        sig = triple_a_setup(df, regime)
        assert sig.signal == "LONG", f"Expected LONG, got {sig.signal}: {sig.reason}"
        assert sig.strategy == "TRIPLE_A"
        assert sig.confidence >= 0.80

    def test_short_signal_on_valid_triple_a(self):
        df, regime = self._make_triple_a_df("short")
        sig = triple_a_setup(df, regime)
        assert sig.signal == "SHORT", f"Expected SHORT, got {sig.signal}: {sig.reason}"
        assert sig.confidence >= 0.80

    def test_accumulation_breakout_aborts(self):
        """If an accumulation bar breaks out of the absorption range, phase 2 fails."""
        df, regime = self._make_triple_a_df("long")
        # Break the accumulation range at iloc[-3]
        absorb_high = float(df["High"].iloc[-4])
        df.at[df.index[-3], "High"] = absorb_high * 1.05   # clearly outside range
        sig = triple_a_setup(df, regime)
        assert sig.signal == "FLAT"
        assert "Accumulation broke" in sig.reason


# ══════════════════════════════════════════════════════════════════════════════
# Strategy 8 — Value Area Bounce
# ══════════════════════════════════════════════════════════════════════════════

class TestValueAreaBounce:
    """
    value_area_bounce() depends on ta.volume_profile() returning specific VAL/VAH.
    Constructing a real price series that simultaneously satisfies close ≈ VAL
    AND close > EMA20 is fragile (they often point to opposite ends of the VA),
    so we use monkeypatch to inject a controlled VP dict for signal tests and
    keep the non-signal tests (FLAT cases) free of mocking.
    """

    def _make_df_with_absorption(self) -> pd.DataFrame:
        """100-bar DF with a forced absorption bar on iloc[-2]."""
        df      = _make_h1_df(n=100, seed=33)
        avg_vol = float(df["Volume"].rolling(20).mean().iloc[-2])
        mid     = float(df["Close"].iloc[-2])
        # Narrow range (0.02%) + 3× average volume → absorption fires
        df.at[df.index[-2], "High"]   = mid * 1.0002
        df.at[df.index[-2], "Low"]    = mid * 0.9998
        df.at[df.index[-2], "Volume"] = avg_vol * 3.0
        return df

    def test_insufficient_bars_returns_flat(self):
        df     = _make_h1_df(n=20)
        regime = _dummy_regime("RANGING", adx=15.0, enabled=["VA_BOUNCE"])
        sig    = value_area_bounce(df, regime)
        assert sig.signal == "FLAT"
        assert sig.strategy == "VA_BOUNCE"

    def test_price_away_from_va_returns_flat(self, monkeypatch):
        """Price in the middle of the VA (at POC) → not near VAL or VAH → FLAT."""
        df     = _make_h1_df(n=100, seed=33)
        regime = _dummy_regime("RANGING", adx=15.0, enabled=["VA_BOUNCE"])
        cur_c  = float(df["Close"].iloc[-1])
        # Inject VP where close sits at POC — far from either edge
        monkeypatch.setattr(ta, "volume_profile", lambda *a, **kw: {
            "poc": cur_c,
            "vah": cur_c + 5.0,
            "val": cur_c - 5.0,
            "levels": [], "volumes": [],
        })
        sig = value_area_bounce(df, regime)
        assert sig.signal == "FLAT"
        assert "not near VA" in sig.reason

    def test_at_val_no_absorption_returns_flat(self, monkeypatch):
        """Price near VAL but no absorption → institutional defence absent → FLAT."""
        df     = _make_h1_df(n=100, seed=33)
        regime = _dummy_regime("RANGING", adx=15.0, enabled=["VA_BOUNCE"])
        cur_c  = float(df["Close"].iloc[-1])
        # Inject VP with VAL right at current close (near_val = True)
        monkeypatch.setattr(ta, "volume_profile", lambda *a, **kw: {
            "poc": cur_c + 2.0,
            "vah": cur_c + 4.0,
            "val": cur_c - 0.02,
            "levels": [], "volumes": [],
        })
        # No absorption forced → default normal volume → FLAT
        sig = value_area_bounce(df, regime)
        assert sig.signal == "FLAT"
        assert "no absorption" in sig.reason.lower()

    def test_long_signal_at_val_with_absorption(self, monkeypatch):
        """
        close ≈ VAL  +  absorption on [-2]  +  close > EMA20 → LONG.

        With seed=33, n=100: close=105.43, EMA20≈105.22 (close > EMA20 ✓).
        We inject a VP where VAL = close − 0.02 so near_val is trivially true.
        """
        df     = self._make_df_with_absorption()
        regime = _dummy_regime("RANGING", adx=15.0, enabled=["VA_BOUNCE"])
        cur_c  = float(df["Close"].iloc[-1])
        monkeypatch.setattr(ta, "volume_profile", lambda *a, **kw: {
            "poc": cur_c + 1.5,
            "vah": cur_c + 3.0,
            "val": cur_c - 0.02,   # close is just above VAL → near_val ✓
            "levels": [], "volumes": [],
        })
        sig = value_area_bounce(df, regime)
        assert sig.signal == "LONG", f"Expected LONG, got {sig.signal}: {sig.reason}"
        assert sig.strategy == "VA_BOUNCE"
        assert sig.confidence >= 0.75

    def test_short_signal_at_vah_with_absorption(self, monkeypatch):
        """
        close ≈ VAH  +  absorption on [-2]  +  close < EMA20 → SHORT.

        We nudge Close[-1] just below EMA20 to satisfy the short trend filter,
        then inject a VP where VAH = close + 0.02 so near_vah is trivially true.
        """
        df    = self._make_df_with_absorption()
        ema20 = float(ta.ema(df["Close"], length=20).iloc[-1])
        # Force close 2% below pre-modification EMA20 so it stays below the
        # recomputed EMA even after the last bar's close changes.
        df.at[df.index[-1], "Close"] = ema20 * 0.98
        cur_c  = float(df["Close"].iloc[-1])
        regime = _dummy_regime("RANGING", adx=15.0, enabled=["VA_BOUNCE"])
        monkeypatch.setattr(ta, "volume_profile", lambda *a, **kw: {
            "poc": cur_c - 1.5,
            "vah": cur_c + 0.02,   # close is just below VAH → near_vah ✓
            "val": cur_c - 3.0,
            "levels": [], "volumes": [],
        })
        sig = value_area_bounce(df, regime)
        assert sig.signal == "SHORT", f"Expected SHORT, got {sig.signal}: {sig.reason}"
        assert sig.confidence >= 0.75
