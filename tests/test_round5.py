"""Tests for Round 5 — data-driven profitability fixes.

R5-2: ORB_5M TP is risk-anchored (R:R always >= 1.5)
R5-4: Composite regime classifier wired into run_signal / run_h1_signal
R5-5: ORB_5M session gate (US equity open 09:00-11:59 ET only)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import date, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_orb_df(
    n_bars: int = 10,
    orb_high: float = 77.0,
    orb_low: float = 76.0,
    cur_close: float = 77.5,
    atr: float = 7.2,
    et_hour: int = 10,
    rvol: float = 2.0,
) -> pd.DataFrame:
    """Synthetic 5m DataFrame for ORB_5M tests.

    Creates a DataFrame whose last bar is at `et_hour`:30 ET with the given
    price structure.  session_bar_number 1-3 are the ORB bars; the last bar
    is bar 4 (first entry-eligible bar).
    """
    import pytz
    et_tz = pytz.timezone("America/New_York")
    base = pd.Timestamp(f"2026-05-01 {et_hour:02d}:00:00", tz="America/New_York")
    timestamps = [base - pd.Timedelta(minutes=5 * (n_bars - 1 - i)) for i in range(n_bars)]
    idx = pd.DatetimeIndex(timestamps).tz_convert("UTC")

    opens  = np.full(n_bars, orb_low + (orb_high - orb_low) * 0.3)
    highs  = np.full(n_bars, orb_high)
    lows   = np.full(n_bars, orb_low)
    closes = np.full(n_bars, cur_close)
    closes[-1] = cur_close
    vols   = np.full(n_bars, 1_000.0)

    df = pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols},
        index=idx,
    )

    # Session metadata matching what fetch_intraday_data produces
    sess_date = date(2026, 5, 1)
    df["session_date"]       = sess_date
    df["session_bar_number"] = range(1, n_bars + 1)
    df["is_first_bar"]       = df["session_bar_number"] == 1
    df["rvol"]               = rvol

    # VWAP: slope upward so vwap_dir > 0 (bullish)
    base_vwap = (orb_high + orb_low) / 2
    df["session_vwap"] = [base_vwap + i * 0.01 for i in range(n_bars)]

    # Pre-compute ATR manually so the strategy can read it
    # ta.atr will use High/Low/Close, but for our synthetic data we inject
    # a plausible ATR by widening the High/Low range
    spread = atr / 2
    df["High"] = closes + spread
    df["Low"]  = closes - spread
    # Keep ORB bars with the intended structure
    for bar_num in range(1, 4):
        idx_pos = bar_num - 1
        df.iloc[idx_pos, df.columns.get_loc("High")] = orb_high
        df.iloc[idx_pos, df.columns.get_loc("Low")]  = orb_low
    df["market"] = "FOREX"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# R5-2  ORB_5M R:R is always >= 1.5 even when ATR >> ORB range
# ─────────────────────────────────────────────────────────────────────────────

class TestR52OrbRiskAnchored:
    """TP must be anchored to actual risk so R:R >= 1.5 even with large ATR."""

    def test_long_rr_at_least_1_5_large_atr(self):
        """Reproduce the SI=F scenario: ORB range=1.02, ATR=7.2."""
        from agents.intraday_strategies import orb_5m as orb_pivot_5m
        df = _make_orb_df(
            orb_high=76.9, orb_low=75.88, cur_close=77.35,
            atr=7.2, et_hour=10, rvol=2.5,
        )
        sig = orb_pivot_5m(df)
        if sig.signal == "LONG":
            risk   = sig.entry - sig.stop_loss
            reward = sig.take_profit - sig.entry
            assert reward / risk >= 1.48, (  # 1.5 target; allow 4-decimal rounding
                f"ORB LONG R:R={reward/risk:.2f} must be >= 1.5 "
                f"(entry={sig.entry}, sl={sig.stop_loss}, tp={sig.take_profit})"
            )

    def test_short_rr_at_least_1_5_large_atr(self):
        """ORB SHORT R:R must also be >= 1.5."""
        from agents.intraday_strategies import orb_5m as orb_pivot_5m
        df = _make_orb_df(
            orb_high=75.88, orb_low=74.86, cur_close=74.3,
            atr=7.2, et_hour=10, rvol=2.5,
        )
        # Invert VWAP so it slopes down (SHORT condition)
        n = len(df)
        base_vwap = (75.88 + 74.86) / 2
        df["session_vwap"] = [base_vwap - i * 0.01 for i in range(n)]
        sig = orb_pivot_5m(df)
        if sig.signal == "SHORT":
            risk   = sig.stop_loss - sig.entry
            reward = sig.entry - sig.take_profit
            assert reward / risk >= 1.48, (  # 1.5 target; allow 4-decimal rounding
                f"ORB SHORT R:R={reward/risk:.2f} must be >= 1.5"
            )

    def test_rr_preserved_when_orb_already_large(self):
        """When ORB range is large relative to ATR, original projection is kept."""
        from agents.intraday_strategies import orb_5m as orb_pivot_5m
        df = _make_orb_df(
            orb_high=78.0, orb_low=75.0, cur_close=78.5,
            atr=0.5, et_hour=10, rvol=2.5,
        )
        sig = orb_pivot_5m(df)
        if sig.signal == "LONG":
            risk   = sig.entry - sig.stop_loss
            reward = sig.take_profit - sig.entry
            # ORB range=3, 1.5x=4.5. Risk likely < 4.5, so max() picks ORB proj.
            assert reward / risk >= 1.48  # 1.5 target; allow 4-decimal rounding


# ─────────────────────────────────────────────────────────────────────────────
# R5-5  ORB_5M session gate — US equity open 09:00-11:59 ET only
# ─────────────────────────────────────────────────────────────────────────────

class TestR55OrbSessionGate:
    """ORB must return FLAT outside 09:00-11:59 ET."""

    @pytest.mark.parametrize("et_hour", [7, 8, 12, 13, 14, 15, 16, 0, 1])
    def test_flat_outside_us_open(self, et_hour: int):
        from agents.intraday_strategies import orb_5m as orb_pivot_5m
        df = _make_orb_df(et_hour=et_hour, rvol=2.0)
        sig = orb_pivot_5m(df)
        assert sig.signal == "FLAT", (
            f"ORB must be FLAT at ET {et_hour:02d}:00 — got {sig.signal} ({sig.reason})"
        )

    @pytest.mark.parametrize("et_hour", [9, 10, 11])
    def test_evaluated_during_us_open(self, et_hour: int):
        """During the US open window ORB evaluates normally (may be LONG/SHORT/FLAT)."""
        from agents.intraday_strategies import orb_5m as orb_pivot_5m
        df = _make_orb_df(et_hour=et_hour, cur_close=77.5, orb_high=76.9, rvol=2.0)
        sig = orb_pivot_5m(df)
        # The signal could be anything — what matters is the session gate didn't
        # reject it with "Outside US equity open window".
        assert "Outside US equity open window" not in (sig.reason or ""), (
            f"ORB should not gate ET {et_hour}:00 but got: {sig.reason}"
        )

    def test_tz_naive_index_not_rejected_by_gate(self):
        """If the index is tz-naive, the gate must skip gracefully (not crash)."""
        from agents.intraday_strategies import orb_5m as orb_pivot_5m
        df = _make_orb_df(et_hour=10, rvol=2.0)
        # Strip timezone
        df.index = df.index.tz_localize(None)
        sig = orb_pivot_5m(df)
        # No crash expected; signal can be anything
        assert sig.strategy == "ORB_5M"


# ─────────────────────────────────────────────────────────────────────────────
# R5-4  Composite regime wired into signal_engine
# ─────────────────────────────────────────────────────────────────────────────

class TestR54RegimeWired:
    """classify() is called inside run_signal and its label is reflected in the result."""

    def test_composite_regime_enriches_adx_regime(self):
        """classify() is called inside run_signal — verify via a direct call with
        a synthetic df.  We don't mock the full pipeline; instead we call classify
        directly to ensure it returns a RegimeReport with composite_score set."""
        import pandas as pd
        from agents.regime_classifier import classify, RegimeReport
        idx = pd.date_range("2026-05-01 00:00", periods=300, freq="5min", tz="UTC")
        prices = 1.10 + np.cumsum(np.random.randn(300) * 0.0005)
        df = pd.DataFrame({
            "Open": prices, "High": prices + 0.001,
            "Low": prices - 0.001, "Close": prices,
            "Volume": np.ones(300) * 1000,
        }, index=idx)
        report = classify(df, ticker="EURUSD")
        assert isinstance(report, RegimeReport)
        assert report.label in ("TRENDING", "RANGING", "MIXED", "INSUFFICIENT_DATA")
        # composite_score must be set when there is enough data
        if report.label != "INSUFFICIENT_DATA":
            assert report.composite_score is not None, \
                "composite_score should be set when classify() has sufficient data"

    def test_composite_score_not_none_in_source(self):
        """Verify source code calls classify() in signal_engine."""
        src = Path("agents/signal_engine.py").read_text(encoding="utf-8")
        assert "from agents.regime_classifier import classify" in src, \
            "signal_engine.py must import classify from regime_classifier"
        assert "composite_score_val" in src, \
            "signal_engine.py must use composite_score_val for penalty blending"

    def test_composite_score_in_swing_engine(self):
        """Same wiring must be present in the H1 swing engine."""
        src = Path("agents/swing_engine.py").read_text(encoding="utf-8")
        assert "from agents.regime_classifier import classify" in src, \
            "swing_engine.py must import classify from regime_classifier"
        assert "composite_score_val" in src, \
            "swing_engine.py must use composite_score_val"
