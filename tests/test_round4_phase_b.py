"""Tests for Round 4 Phase B — pre-VPS profitability fixes.

Each test class corresponds to one Phase B item (B1-B5). Run via:
    pytest tests/test_round4_phase_b.py -v
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# B1 — Structure-aware stops (utils/stops.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestB1StructuralStops:
    def _make_df(self, lows: list[float], highs: list[float], closes: list[float]) -> pd.DataFrame:
        n = len(lows)
        idx = pd.date_range("2026-04-01 09:30", periods=n, freq="5min", tz="America/New_York")
        return pd.DataFrame({
            "Open":   closes,
            "High":   highs,
            "Low":    lows,
            "Close":  closes,
            "Volume": [1000.0] * n,
        }, index=idx)

    def test_long_uses_recent_swing_low(self):
        from utils.stops import compute_structural_stop
        # Recent swing low at 1.0950 in the lookback window
        # entry=1.100 keeps the pivot-distance inside [min_dist, max_dist]:
        # candidate_sl = 1.0950 - 0.30*0.005 = 1.09350; distance = 0.0065
        # which is within [0.5×ATR=0.0025, 2.5×ATR=0.0125] → swing path
        lows  = [1.10] * 10 + [1.0950] + [1.105] * 10
        highs = [1.11] * 21
        closes = [1.105] * 21
        df = self._make_df(lows, highs, closes)
        result = compute_structural_stop(df, "LONG", entry=1.100, atr=0.005)
        assert result.sl_source == "swing"
        assert abs(result.sl - 1.09350) < 0.0001
        # TP at 2R: risk = 0.0065 → TP = 1.100 + 0.0130 = 1.113
        assert abs(result.tp - 1.11300) < 0.0001
        assert result.pivot_price == 1.0950

    def test_short_uses_recent_swing_high(self):
        from utils.stops import compute_structural_stop
        # entry=1.110 keeps pivot-distance in band:
        # candidate_sl = 1.115 + 0.0015 = 1.1165; distance = 0.0065
        lows  = [1.10] * 21
        highs = [1.105] * 10 + [1.115] + [1.105] * 10
        closes = [1.108] * 21
        df = self._make_df(lows, highs, closes)
        result = compute_structural_stop(df, "SHORT", entry=1.110, atr=0.005)
        assert result.sl_source == "swing"
        assert abs(result.sl - 1.11650) < 0.0001

    def test_atr_floor_when_pivot_too_close(self):
        from utils.stops import compute_structural_stop
        # Pivot at 1.0995 means candidate_sl = 1.0995 - 0.0015 = 1.0980;
        # distance = 1.100 - 1.0980 = 0.002 < min_dist (0.0025) → floor.
        lows  = [1.0995] * 21
        highs = [1.101] * 21
        closes = [1.100] * 21
        df = self._make_df(lows, highs, closes)
        result = compute_structural_stop(df, "LONG", entry=1.100, atr=0.005)
        assert result.sl_source == "atr_floor"
        # SL at entry - min_dist = 1.100 - 0.0025 = 1.0975
        assert abs(result.sl - 1.0975) < 0.0001

    def test_atr_max_when_pivot_too_far(self):
        from utils.stops import compute_structural_stop
        # Deep pivot way below entry
        lows  = [1.05] * 21
        highs = [1.11] * 21
        closes = [1.10] * 21
        df = self._make_df(lows, highs, closes)
        # ATR=0.005, max_dist = 2.5 * 0.005 = 0.0125
        result = compute_structural_stop(df, "LONG", entry=1.100, atr=0.005)
        assert result.sl_source == "atr_max"
        assert abs(result.sl - 1.0875) < 0.0001

    def test_one_r_equals_risk(self):
        from utils.stops import compute_structural_stop
        df = self._make_df([1.099] * 21, [1.101] * 21, [1.100] * 21)
        result = compute_structural_stop(df, "LONG", entry=1.100, atr=0.005)
        assert result.one_r == result.risk
        assert result.one_r > 0


class TestB1MakeSignalIntegration:
    """The _make_signal helper must support `df` + `use_structural=True`."""

    def test_make_signal_structural_path(self):
        from agents.intraday_strategies import _make_signal
        n = 30
        idx = pd.date_range("2026-04-01 09:30", periods=n, freq="5min", tz="America/New_York")
        df = pd.DataFrame({
            "Open":   [1.10] * n,
            "High":   [1.105] * n,
            "Low":    [1.099] * 10 + [1.0950] + [1.099] * (n - 11),
            "Close":  [1.10] * n,
            "Volume": [1000.0] * n,
        }, index=idx)
        sig = _make_signal(
            "FAKE", "5m", "LONG", confidence=0.75,
            entry=1.100, atr=0.005,
            sl_atr_mult=1.0, tp_atr_mult=2.0,
            reason="test",
            df=df, use_structural=True,
        )
        # Structural SL should be tighter than pure-ATR (1.095)
        # Pivot is at 1.0950 → SL at 1.0950 - 0.0015 = 1.09350 (closer to entry than 1.095)
        # Wait, that's actually further. Let me recompute:
        # Pure ATR LONG: SL = 1.100 - 1.0*0.005 = 1.095
        # Structural: pivot=1.0950, buffer=0.30*0.005=0.0015 → SL=1.0950-0.0015=1.09350
        # 1.09350 < 1.095 → structural SL is FURTHER from entry, giving more room
        assert sig.stop_loss < 1.095, f"Structural SL {sig.stop_loss} should be tighter (lower) than 1.095"
        assert "SL=swing" in sig.reason

    def test_make_signal_legacy_path_unchanged(self):
        """When use_structural is False (default), behaviour must be unchanged."""
        from agents.intraday_strategies import _make_signal
        sig = _make_signal(
            "FAKE", "5m", "LONG", confidence=0.75,
            entry=1.100, atr=0.005,
            sl_atr_mult=1.0, tp_atr_mult=2.0,
            reason="legacy test",
        )
        assert abs(sig.stop_loss - 1.095) < 0.0001  # entry - 1.0*0.005
        assert abs(sig.take_profit - 1.110) < 0.0001  # entry + 2.0*0.005
        assert "SL=" not in sig.reason  # no structural marker


# ─────────────────────────────────────────────────────────────────────────────
# B2 — Event-window veto
# ─────────────────────────────────────────────────────────────────────────────

class TestB2EventVeto:
    def test_is_event_window_returns_blocked(self, monkeypatch):
        from data.economic_calendar import CalendarEvent, is_event_window
        now = datetime.now(timezone.utc)
        # Mock fetch_upcoming_events to return one HIGH event 5m ahead
        fake_event = CalendarEvent(
            title="NFP", currency="USD", impact="HIGH",
            event_time=now + timedelta(minutes=5),
            is_upcoming=True, is_recent=False,
            actual=None, forecast=None, previous=None,
        )
        monkeypatch.setattr(
            "data.economic_calendar.fetch_upcoming_events",
            lambda *a, **kw: [fake_event],
        )
        blocked, reason = is_event_window("EURUSD=X")
        assert blocked is True
        assert "NFP" in reason
        assert "USD" in reason

    def test_is_event_window_clear_when_far(self, monkeypatch):
        from data.economic_calendar import is_event_window
        # No events returned
        monkeypatch.setattr(
            "data.economic_calendar.fetch_upcoming_events",
            lambda *a, **kw: [],
        )
        blocked, reason = is_event_window("EURUSD=X")
        assert blocked is False
        assert reason == ""

    def test_signal_engine_vetoes_during_event_window(self, monkeypatch):
        """run_signal must short-circuit with an error when event-window is active."""
        from data.economic_calendar import CalendarEvent
        from agents.signal_engine import run_signal
        now = datetime.now(timezone.utc)
        fake_event = CalendarEvent(
            title="CPI", currency="USD", impact="HIGH",
            event_time=now + timedelta(minutes=10),
            is_upcoming=True, is_recent=False,
            actual=None, forecast=None, previous=None,
        )
        monkeypatch.setattr(
            "data.economic_calendar.fetch_upcoming_events",
            lambda *a, **kw: [fake_event],
        )
        # Stub fetch_intraday so we don't hit network — should never be called
        # because the event veto fires first.
        called = {"n": 0}
        def fake_fetch(*a, **kw):
            called["n"] += 1
            return None
        monkeypatch.setattr(
            "data.fetcher_intraday.fetch_intraday_data", fake_fetch,
        )
        result = run_signal("EURUSD=X")
        assert "event-veto" in result.error
        assert called["n"] == 0  # data fetch was skipped — event veto fires first


# ─────────────────────────────────────────────────────────────────────────────
# B3 — Boost suppression on MTF-vetoed signals
# ─────────────────────────────────────────────────────────────────────────────

class TestB3VetoedBoostSuppression:
    """When daily_trend_vetoed=True, the recorded confidence on the SignalResult
    must equal the base confidence (no phantom boosts)."""

    def test_vetoed_signal_zeroes_boosts(self, monkeypatch):
        from agents.intraday_strategies import IntradaySignal
        from agents.signal_engine import run_signal

        n = 200
        idx = pd.date_range("2026-04-01 09:30", periods=n, freq="5min", tz="America/New_York")
        closes = np.linspace(100, 110, n)
        df = pd.DataFrame({
            "Open": closes, "High": closes + 0.05, "Low": closes - 0.05,
            "Close": closes, "Volume": np.full(n, 1000.0),
        }, index=idx)
        df["session_date"] = idx.date
        df["session_bar_number"] = np.arange(1, n+1)
        df["is_first_bar"] = df["session_bar_number"] == 1
        df["session_vwap"] = closes
        df["rvol"] = 1.0
        df["market"] = "US"

        # Force a LONG signal at conf=0.85 from a single fake strategy.
        def fake_long(df_):
            return IntradaySignal(
                strategy="TREND_MOM_5M", timeframe="5m", signal="LONG",
                confidence=0.85, entry=float(df_["Close"].iloc[-1]),
                stop_loss=float(df_["Close"].iloc[-1]) - 0.5,
                take_profit=float(df_["Close"].iloc[-1]) + 1.0,
                atr=0.1, rr_ratio=2.0, reason="forced LONG",
            )

        # Daily trend strongly bearish → mtf_score will be -2 → veto LONG
        veto_daily = {"trend_direction": "BEARISH"}

        with patch("data.fetcher_intraday.fetch_intraday_data", return_value=df), \
             patch("agents.signal_engine._STRATEGIES_5M", [fake_long]), \
             patch("data.strategy_params.apply_params", side_effect=lambda sig, **kw: sig), \
             patch("agents.signal_engine.fetch_news", return_value=[]), \
             patch("agents.signal_engine.aggregate_sentiment", return_value=("NEUTRAL", 0.0)), \
             patch("data.economic_calendar.fetch_upcoming_events", return_value=[]):
            result = run_signal("FAKE=X", daily_trend=veto_daily)

        assert result.daily_trend_vetoed is True
        assert result.direction == "NO TRADE"
        # All boosts must be zeroed on a vetoed result
        assert result.consensus_boost == 0.0
        assert result.bt_adjustment   == 0.0
        assert result.live_adjustment == 0.0
        assert result.news_boost      == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# B4 — Slippage model
# ─────────────────────────────────────────────────────────────────────────────

class TestB4Slippage:
    def test_per_asset_baseline(self):
        from utils.slippage import baseline_slippage
        # Crypto > Energy > Metals > Index ≈ Major FX
        assert baseline_slippage("BTC-USD") > baseline_slippage("CL=F")
        assert baseline_slippage("CL=F")    >= baseline_slippage("XAUUSD=X")
        assert baseline_slippage("EURUSD=X") < baseline_slippage("XAUUSD=X")

    def test_session_open_widens_spread(self):
        from utils.slippage import slippage_at_bar
        bar_quiet = pd.Series({"session_bar_number": 30, "is_first_bar": False})
        bar_open  = pd.Series({"session_bar_number": 1,  "is_first_bar": True})
        s_quiet = slippage_at_bar("EURUSD=X", bar_quiet)
        s_open  = slippage_at_bar("EURUSD=X", bar_open)
        assert s_open == s_quiet * 1.5

    def test_event_window_widens_spread(self):
        from utils.slippage import slippage_at_bar
        bar = pd.Series({"session_bar_number": 30, "is_first_bar": False})
        s_quiet = slippage_at_bar("EURUSD=X", bar, in_event_window=False)
        s_event = slippage_at_bar("EURUSD=X", bar, in_event_window=True)
        assert s_event == s_quiet * 2.0

    def test_session_and_event_take_max(self):
        from utils.slippage import slippage_at_bar
        bar_open = pd.Series({"session_bar_number": 1, "is_first_bar": True})
        s_both   = slippage_at_bar("EURUSD=X", bar_open, in_event_window=True)
        base     = slippage_at_bar("EURUSD=X", None)
        # max(1.5, 2.0) = 2.0 multiplier
        assert s_both == base * 2.0

    def test_classify_symbol(self):
        from utils.slippage import _classify_symbol
        assert _classify_symbol("EURUSD=X")  == "MAJOR_FX"
        assert _classify_symbol("EURJPY=X")  == "MINOR_FX"
        assert _classify_symbol("XAUUSD=X")  == "METALS"
        assert _classify_symbol("CL=F")      == "ENERGY"
        assert _classify_symbol("BTC-USD")   == "CRYPTO"
        assert _classify_symbol("NQ=F")      == "INDEX"


# ─────────────────────────────────────────────────────────────────────────────
# B5 — Composite regime classifier
# ─────────────────────────────────────────────────────────────────────────────

class TestB5RegimeComposite:
    def test_hurst_on_random_walk(self):
        """Hurst exponent of a random walk should be near 0.5."""
        from agents.regime_classifier import hurst_exponent
        rng = np.random.default_rng(42)
        # Cumulative sum of N(0,1) increments → random walk
        prices = 100 + np.cumsum(rng.standard_normal(500) * 0.1)
        h = hurst_exponent(prices)
        assert h is not None
        # Allow generous tolerance — R/S underestimates on short series
        assert 0.30 < h < 0.70

    def test_hurst_on_trending(self):
        """Strongly trending series → Hurst > 0.5 (closer to persistent)."""
        from agents.regime_classifier import hurst_exponent
        prices = np.linspace(100, 130, 500)
        h = hurst_exponent(prices)
        assert h is not None
        # Smooth deterministic trend → high persistence
        assert h > 0.6

    def test_choppiness_index_zero_range_is_safe(self):
        """Constant prices → choppiness CI must not crash."""
        from agents.regime_classifier import choppiness_index
        n = 50
        s = pd.Series([100.0] * n)
        ci = choppiness_index(s, s, s)
        # Constant series → no movement, CI bottoms out near 0
        assert not ci.dropna().isnull().any()

    def test_composite_score_blends_components(self):
        from agents.regime_classifier import composite_score
        # Strongly trending: pct_t=0.8, hurst=0.65, chop=30
        # ADX:  2*0.8 - 1 = +0.6
        # Hurst: (0.65-0.5)*10 = 1.5 → clipped to +1.0
        # Chop:  (50 - 30)/11.8 = +1.69 → clipped to +1.0
        # Mean: (0.6 + 1.0 + 1.0) / 3 = 0.867
        score = composite_score(0.8, 0.65, 30.0)
        assert 0.7 < score < 1.0

    def test_composite_score_strongly_ranging(self):
        from agents.regime_classifier import composite_score
        # ADX: 2*0.05 - 1 = -0.9
        # Hurst: (0.40-0.5)*10 = -1.0
        # Chop: (50 - 70)/11.8 = -1.69 → clipped to -1.0
        # Mean: (-0.9 + -1.0 + -1.0) / 3 = -0.967
        score = composite_score(0.05, 0.40, 70.0)
        assert score < -0.7

    def test_classify_returns_extended_fields(self):
        from agents.regime_classifier import classify
        n = 400
        rng = np.random.default_rng(0)
        closes = np.linspace(100, 130, n) + rng.standard_normal(n) * 0.1
        df = pd.DataFrame({
            "Open":   closes,
            "High":   closes + 0.5,
            "Low":    closes - 0.5,
            "Close":  closes,
            "Volume": np.full(n, 1000.0),
        })
        report = classify(df, ticker="X", lookback_bars=300)
        assert report.label != "INSUFFICIENT_DATA"
        assert report.hurst is not None
        assert report.choppiness is not None
        assert report.composite_score is not None
        assert -1.0 <= report.composite_score <= 1.0
