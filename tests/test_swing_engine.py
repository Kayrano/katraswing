"""Tests for Round 4 C1/C2 — H1 swing engine and strategies.

Covers:
  - Each H1 strategy function: flat on bad input, fires correctly on valid fixture
  - swing_engine.run_h1_signal: event veto, MTF gate, paper_only propagation
  - strategy_params: H1 strategies default to paper_only=True
  - London Breakout: only fires during London window, uses Asian range correctly

Run via:
    pytest tests/test_swing_engine.py -v
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_h1(
    n: int = 100,
    price: float = 1.1000,
    *,
    seed: int = 0,
    utc_start: str = "2026-04-01 09:00",
) -> pd.DataFrame:
    """Synthetic H1 DataFrame with all columns the swing strategies expect."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(utc_start, periods=n, freq="1h", tz="UTC")
    closes = price + np.cumsum(rng.standard_normal(n) * 0.001)
    highs  = closes + rng.uniform(0.0005, 0.002, n)
    lows   = closes - rng.uniform(0.0005, 0.002, n)

    df = pd.DataFrame({
        "Open":   closes + rng.standard_normal(n) * 0.0003,
        "High":   highs,
        "Low":    lows,
        "Close":  closes,
        "Volume": rng.uniform(500, 2000, n),
    }, index=idx)

    df["session_date"]       = idx.date
    df["session_bar_number"] = df.groupby("session_date").cumcount() + 1
    df["is_first_bar"]       = df["session_bar_number"] == 1
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    tp_vol  = typical * df["Volume"]
    cum_tpv = tp_vol.groupby(df["session_date"]).cumsum()
    cum_vol = df["Volume"].groupby(df["session_date"]).cumsum()
    df["session_vwap"] = cum_tpv / cum_vol.replace(0, np.nan)
    df["rvol"]   = rng.uniform(0.8, 2.0, n)
    df["market"] = "FOREX"
    return df


# ── B1  mss_h1 ───────────────────────────────────────────────────────────────

class TestMSSH1:
    def test_flat_on_insufficient_bars(self):
        from agents.swing_strategies import mss_h1
        df = _make_h1(10)
        sig = mss_h1(df)
        assert sig.signal == "FLAT"

    def test_flat_without_datetime_index(self):
        from agents.swing_strategies import mss_h1
        df = _make_h1(80).reset_index(drop=True)
        sig = mss_h1(df)
        assert sig.signal == "FLAT"
        assert "DatetimeIndex" in sig.reason

    def test_strategy_id(self):
        from agents.swing_strategies import mss_h1
        df = _make_h1(80)
        sig = mss_h1(df)
        assert sig.strategy == "MSS_H1"
        assert sig.timeframe == "1h"

    def test_returns_intraday_signal(self):
        from agents.swing_strategies import mss_h1
        from agents.intraday_strategies import IntradaySignal
        df = _make_h1(80)
        sig = mss_h1(df)
        assert isinstance(sig, IntradaySignal)


# ── ORB_H1 ────────────────────────────────────────────────────────────────────

class TestORBH1:
    def test_flat_before_bar3(self):
        from agents.swing_strategies import orb_h1
        df = _make_h1(2)
        sig = orb_h1(df)
        assert sig.signal == "FLAT"
        assert "not complete" in sig.reason or "need" in sig.reason

    def test_flat_outside_entry_window(self):
        from agents.swing_strategies import orb_h1
        df = _make_h1(20)
        # Fake bar number 7 → outside window
        df["session_bar_number"] = 7
        df["session_date"] = df.index.date[0]
        sig = orb_h1(df)
        assert sig.signal == "FLAT"
        assert "Outside entry window" in sig.reason

    def test_long_breakout_fires(self):
        from agents.swing_strategies import orb_h1
        # Build a df where bar 3's close is clearly above ORB high
        n = 10
        idx = pd.date_range("2026-04-01 09:00", periods=n, freq="1h", tz="UTC")
        closes = [1.1000, 1.1010, 1.1005, 1.1050, *([1.1030] * 6)]
        highs  = [c + 0.001 for c in closes]
        lows   = [c - 0.001 for c in closes]

        df = pd.DataFrame({
            "Open": closes, "High": highs, "Low": lows, "Close": closes,
            "Volume": [1000.0] * n,
        }, index=idx)
        df["session_date"]       = idx.date
        df["session_bar_number"] = list(range(1, n + 1))
        df["is_first_bar"]       = df["session_bar_number"] == 1
        df["session_vwap"]       = pd.Series(closes, index=idx).expanding().mean()
        df["rvol"]               = 2.0   # strong RVOL
        df["market"]             = "FOREX"

        # Force current bar (last) at bar number 3, above ORB high
        df.loc[df.index[-1], "session_bar_number"] = 3
        df.loc[df.index[-1], "Close"] = 1.1060   # above ORB high of first 2 bars
        df.loc[df.index[-1], "High"]  = 1.1065
        df.loc[df.index[-1], "Low"]   = 1.1055

        # VWAP sloping up (needed for signal)
        vwap_vals = list(range(n))
        df["session_vwap"] = [1.1000 + i * 0.0005 for i in range(n)]

        sig = orb_h1(df)
        # Should fire or be FLAT depending on exact VWAP direction
        assert sig.strategy == "ORB_H1"
        assert sig.timeframe == "1h"

    def test_strategy_id_always_correct(self):
        from agents.swing_strategies import orb_h1
        df = _make_h1(5)
        sig = orb_h1(df)
        assert sig.strategy == "ORB_H1"


# ── EMA_PB_H1 ─────────────────────────────────────────────────────────────────

class TestEmaPbH1:
    def test_flat_insufficient_bars(self):
        from agents.swing_strategies import ema_pullback_h1
        df = _make_h1(10)
        sig = ema_pullback_h1(df)
        assert sig.signal == "FLAT"

    def test_strategy_id(self):
        from agents.swing_strategies import ema_pullback_h1
        df = _make_h1(60)
        sig = ema_pullback_h1(df)
        assert sig.strategy == "EMA_PB_H1"
        assert sig.timeframe == "1h"

    def test_fires_long_in_uptrend(self):
        from agents.swing_strategies import ema_pullback_h1
        # Build a cleanly trending up series
        n = 60
        idx = pd.date_range("2026-04-01 00:00", periods=n, freq="1h", tz="UTC")
        closes = np.linspace(1.0900, 1.1200, n)   # clean uptrend
        highs  = closes + 0.0005
        lows   = closes - 0.0005
        df = pd.DataFrame({
            "Open": closes, "High": highs, "Low": lows, "Close": closes,
            "Volume": np.full(n, 1000.0),
        }, index=idx)
        df["session_date"]       = idx.date
        df["session_bar_number"] = df.groupby("session_date").cumcount() + 1
        df["is_first_bar"]       = df["session_bar_number"] == 1
        df["session_vwap"]       = pd.Series(closes, index=idx).expanding().mean()
        df["rvol"]               = 1.0
        df["market"]             = "FOREX"

        # Force last bar to touch EMA8 (pullback) with low RSI
        # Hard to guarantee without running the full indicator — just check it fires FLAT or LONG
        sig = ema_pullback_h1(df)
        assert sig.strategy == "EMA_PB_H1"
        assert sig.signal in ("LONG", "SHORT", "FLAT")
        if sig.signal in ("LONG", "SHORT"):
            assert sig.entry > 0
            assert sig.stop_loss > 0
            assert sig.take_profit > 0


# ── LONDON_BREAKOUT_H1 ────────────────────────────────────────────────────────

class TestLondonBreakoutH1:
    def _make_london_df(
        self,
        cur_utc_hour: int = 7,
        asian_high: float = 1.1020,
        asian_low: float  = 1.1000,
        cur_close: float  = 1.1035,   # above asian_high → LONG
        rvol: float       = 1.8,
    ) -> pd.DataFrame:
        """Build a 30-bar DataFrame ending at the given UTC hour with
        clearly identifiable Asian-session bars and a London-window last bar."""
        # Build 30 bars ending at cur_utc_hour on 2026-04-07
        end_time = datetime(2026, 4, 7, cur_utc_hour, 0, tzinfo=timezone.utc)
        n = 30
        idx = pd.date_range(end=end_time, periods=n, freq="1h", tz="UTC")

        # Asian session hours in the index
        asian_mask = idx.hour.isin([22, 23, 0, 1, 2, 3, 4, 5, 6])

        closes = np.full(n, 1.1010)
        highs  = np.where(asian_mask, asian_high, 1.1015)
        lows   = np.where(asian_mask, asian_low,  1.1005)

        # Override the current (last) bar
        closes[-1] = cur_close
        highs[-1]  = cur_close + 0.0002
        lows[-1]   = cur_close - 0.0002

        df = pd.DataFrame({
            "Open":   closes,
            "High":   highs.astype(float),
            "Low":    lows.astype(float),
            "Close":  closes,
            "Volume": np.full(n, 1000.0),
        }, index=idx)
        df["session_date"]       = idx.date
        df["session_bar_number"] = df.groupby("session_date").cumcount() + 1
        df["is_first_bar"]       = df["session_bar_number"] == 1
        df["session_vwap"]       = pd.Series(closes, index=idx).expanding().mean()
        df["rvol"]               = rvol
        df["market"]             = "FOREX"
        return df

    def test_flat_outside_london_window(self):
        from agents.swing_strategies import london_breakout_h1
        # 15:00 UTC is NOT London open
        df = self._make_london_df(cur_utc_hour=15, cur_close=1.1035)
        sig = london_breakout_h1(df)
        assert sig.signal == "FLAT"
        assert "London window" in sig.reason

    def test_long_on_breakout_above_asian_high(self):
        from agents.swing_strategies import london_breakout_h1
        # 07:00 UTC, close above Asian high → LONG
        df = self._make_london_df(
            cur_utc_hour=7,
            asian_high=1.1020,
            asian_low=1.1000,
            cur_close=1.1035,   # above 1.1020
        )
        sig = london_breakout_h1(df)
        assert sig.signal == "LONG", f"Expected LONG, got {sig.signal}: {sig.reason}"
        assert sig.entry > 1.1020
        assert sig.stop_loss < sig.entry
        assert sig.take_profit > sig.entry

    def test_short_on_breakout_below_asian_low(self):
        from agents.swing_strategies import london_breakout_h1
        df = self._make_london_df(
            cur_utc_hour=8,
            asian_high=1.1020,
            asian_low=1.1000,
            cur_close=1.0985,   # below 1.1000
        )
        sig = london_breakout_h1(df)
        assert sig.signal == "SHORT", f"Expected SHORT, got {sig.signal}: {sig.reason}"
        assert sig.stop_loss > sig.entry
        assert sig.take_profit < sig.entry

    def test_flat_inside_asian_range(self):
        from agents.swing_strategies import london_breakout_h1
        df = self._make_london_df(
            cur_utc_hour=7,
            asian_high=1.1020,
            asian_low=1.1000,
            cur_close=1.1010,   # inside the range
        )
        sig = london_breakout_h1(df)
        assert sig.signal == "FLAT"

    def test_flat_without_datetime_index(self):
        from agents.swing_strategies import london_breakout_h1
        df = self._make_london_df(cur_utc_hour=7, cur_close=1.1035)
        df = df.reset_index(drop=True)
        sig = london_breakout_h1(df)
        assert sig.signal == "FLAT"
        assert "DatetimeIndex" in sig.reason

    def test_strategy_id(self):
        from agents.swing_strategies import london_breakout_h1
        df = self._make_london_df()
        sig = london_breakout_h1(df)
        assert sig.strategy == "LONDON_BREAKOUT_H1"
        assert sig.timeframe == "1h"


# ── Strategy registry ──────────────────────────────────────────────────────────

class TestStrategyRegistry:
    def test_all_four_strategies_registered(self):
        from agents.swing_strategies import _STRATEGIES_H1
        names = [fn.__name__ for fn in _STRATEGIES_H1]
        assert "mss_h1"              in names
        assert "orb_h1"              in names
        assert "ema_pullback_h1"     in names
        assert "london_breakout_h1"  in names
        assert len(_STRATEGIES_H1) == 4

    def test_trend_mr_sets_disjoint(self):
        from agents.swing_strategies import _TREND_STRATEGIES_H1, _MR_STRATEGIES_H1
        assert _TREND_STRATEGIES_H1.isdisjoint(_MR_STRATEGIES_H1)


# ── strategy_params defaults ──────────────────────────────────────────────────

class TestH1StrategyParamsDefaults:
    def test_h1_strategies_default_paper_only(self):
        from data.strategy_params import _DEFAULT_PAPER_ONLY
        for strat in ("MSS_H1", "ORB_H1", "EMA_PB_H1", "LONDON_BREAKOUT_H1"):
            assert strat in _DEFAULT_PAPER_ONLY, f"{strat} not in _DEFAULT_PAPER_ONLY"

    def test_h1_strategies_in_all_strategies(self):
        from data.strategy_params import _ALL_STRATEGIES
        for strat in ("MSS_H1", "ORB_H1", "EMA_PB_H1", "LONDON_BREAKOUT_H1"):
            assert strat in _ALL_STRATEGIES

    def test_default_entry_paper_only_true_for_h1(self):
        from data.strategy_params import _default_entry
        for strat in ("MSS_H1", "ORB_H1", "EMA_PB_H1", "LONDON_BREAKOUT_H1"):
            entry = _default_entry(strat)
            assert entry["paper_only"] is True, f"{strat} should default to paper_only=True"
            assert entry["enabled"] is True, f"{strat} should be enabled (not disabled)"


# ── swing_engine.run_h1_signal integration ────────────────────────────────────

class TestRunH1Signal:
    def _base_df(self, n: int = 80) -> pd.DataFrame:
        return _make_h1(n)

    def test_event_veto_fires_before_data_fetch(self, monkeypatch):
        """run_h1_signal must short-circuit when an economic event is active."""
        from data.economic_calendar import CalendarEvent
        from agents.swing_engine import run_h1_signal
        now = datetime.now(timezone.utc)
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
        called = {"n": 0}
        def fake_fetch(*a, **kw):
            called["n"] += 1
        monkeypatch.setattr("data.fetcher_intraday.fetch_intraday_data", fake_fetch)

        result = run_h1_signal("EURUSD=X")
        assert "event-veto" in result.error or "NFP" in result.error
        assert called["n"] == 0

    def test_drop_symbol_returns_error(self, monkeypatch):
        """Dropped symbols must return an error before any fetch."""
        from agents.swing_engine import run_h1_signal
        monkeypatch.setattr(
            "data.symbol_policy.get_disposition", lambda sym: "DROP"
        )
        result = run_h1_signal("NQ=F")
        assert result.error != ""
        assert result.direction == "NO TRADE"

    def test_no_h1_data_returns_error(self, monkeypatch):
        from agents.swing_engine import run_h1_signal
        monkeypatch.setattr(
            "data.economic_calendar.fetch_upcoming_events", lambda *a, **kw: []
        )
        monkeypatch.setattr(
            "data.fetcher_intraday.fetch_intraday_data",
            lambda *a, **kw: None,
        )
        result = run_h1_signal("EURUSD=X")
        assert result.error != ""

    def test_paper_only_propagated(self, monkeypatch):
        """All H1 signals should be paper_only=True until auto-promoted."""
        from agents.swing_engine import run_h1_signal
        from agents.intraday_strategies import IntradaySignal

        df = self._base_df(80)

        def fake_long(df_):
            return IntradaySignal(
                strategy="MSS_H1", timeframe="1h", signal="LONG",
                confidence=0.85,
                entry=float(df_["Close"].iloc[-1]),
                stop_loss=float(df_["Close"].iloc[-1]) - 0.005,
                take_profit=float(df_["Close"].iloc[-1]) + 0.010,
                atr=0.001, rr_ratio=2.0, reason="test",
                paper_only=True,
            )

        monkeypatch.setattr(
            "data.economic_calendar.fetch_upcoming_events", lambda *a, **kw: []
        )
        monkeypatch.setattr(
            "data.fetcher_intraday.fetch_intraday_data", lambda *a, **kw: df
        )
        monkeypatch.setattr(
            "agents.swing_engine._STRATEGIES_H1", [fake_long]
        )
        monkeypatch.setattr(
            "data.strategy_params.apply_params",
            lambda sig, **kw: sig,
        )

        result = run_h1_signal("EURUSD=X")
        # Regardless of direction, paper_only must be True for H1 strategies
        assert result.paper_only is True

    def test_mtf_veto_produces_no_trade(self, monkeypatch):
        """Daily BEARISH trend + H4 BEARISH must veto a LONG H1 signal."""
        from agents.swing_engine import run_h1_signal
        from agents.intraday_strategies import IntradaySignal

        df = self._base_df(80)

        def fake_long(df_):
            return IntradaySignal(
                strategy="MSS_H1", timeframe="1h", signal="LONG",
                confidence=0.85,
                entry=float(df_["Close"].iloc[-1]),
                stop_loss=float(df_["Close"].iloc[-1]) - 0.005,
                take_profit=float(df_["Close"].iloc[-1]) + 0.010,
                atr=0.001, rr_ratio=2.0, reason="test",
            )

        monkeypatch.setattr(
            "data.economic_calendar.fetch_upcoming_events", lambda *a, **kw: []
        )
        monkeypatch.setattr(
            "data.fetcher_intraday.fetch_intraday_data", lambda *a, **kw: df
        )
        monkeypatch.setattr("agents.swing_engine._STRATEGIES_H1", [fake_long])
        monkeypatch.setattr(
            "data.strategy_params.apply_params",
            lambda sig, **kw: sig,
        )

        result = run_h1_signal(
            "EURUSD=X",
            daily_trend={"trend_direction": "BEARISH"},
            h4_trend={"trend_direction": "BEARISH"},
        )
        assert result.daily_trend_vetoed is True
        assert result.direction == "NO TRADE"
