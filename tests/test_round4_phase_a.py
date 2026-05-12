"""Tests for Round 4 Phase A — pre-VPS profitability fixes.

Each test class corresponds to one Phase A item (A1-A5). Run via:
    pytest tests/test_round4_phase_a.py -v
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# A1 — Strategy code bug fixes
# ─────────────────────────────────────────────────────────────────────────────

class TestA1VWAPBand:
    """The VWAP_RSI band must use 0.15×ATR, not 0.5×ATR."""

    def test_vwap_band_is_0_15_x_atr(self):
        """Read the source line directly to confirm the fix is in place."""
        src = Path("agents/intraday_strategies.py").read_text(encoding="utf-8")
        # Find the band assignment line and assert the multiplier
        assert "band    = 0.15 * cur_atr" in src or "band = 0.15 * cur_atr" in src, \
            "VWAP band must be 0.15×ATR (was 0.5 — too wide)"
        # And explicitly that the old 0.5 line is gone
        assert "band    = 0.5 * cur_atr" not in src, \
            "Old 0.5×ATR band must be removed"


class TestA1NR7Lookback:
    """NR7 must compare strictly less than min(prior 6 bars), excluding setup."""

    def _make_df(self, ranges: list[float]) -> pd.DataFrame:
        """Build a synthetic OHLCV df where each bar has the given range."""
        n = len(ranges)
        idx = pd.date_range("2026-04-01 09:30", periods=n, freq="5min", tz="America/New_York")
        rows = []
        for r in ranges:
            mid = 100.0
            rows.append({"Open": mid, "High": mid + r/2, "Low": mid - r/2,
                         "Close": mid + r/3, "Volume": 1000.0})
        df = pd.DataFrame(rows, index=idx)
        df["session_date"] = idx.date
        df["session_bar_number"] = np.arange(1, n+1)
        df["is_first_bar"] = df["session_bar_number"] == 1
        df["session_vwap"] = df["Close"].values
        df["rvol"] = 1.5
        df["market"] = "US"
        return df

    def test_flat_day_does_not_trivially_pass(self):
        """All bars same range → setup bar is NOT strictly narrower → NO NR7."""
        from agents.intraday_strategies import nr7_breakout_5m
        df = self._make_df([0.10] * 25)   # >20 bars, all equal range
        sig = nr7_breakout_5m(df)
        assert sig.signal == "FLAT"
        assert "No NR7" in sig.reason

    def test_genuine_nr7_fires(self):
        """Setup bar's range is strictly narrower than the prior 6 → NR7 detected."""
        from agents.intraday_strategies import nr7_breakout_5m
        # 18 normal-range bars, then a tight setup bar at idx -2, then a breakout bar at -1
        ranges = [0.20] * 18 + [0.05, 0.10]   # setup=0.05, current=0.10 (breakout-sized)
        df = self._make_df(ranges)
        sig = nr7_breakout_5m(df)
        # Either fires LONG/SHORT or fails the breakout gate, but the NR7 detection
        # itself must NOT report "No NR7".
        assert "No NR7" not in sig.reason


class TestA1MSSDatetimeIndexGuard:
    """MSS_FOREX_15M must fail gracefully without a DatetimeIndex."""

    def test_non_datetime_index_returns_flat(self):
        from agents.intraday_strategies import mss_forex_15m
        n = 200
        # Positional integer index — would crash df.resample("1D")
        df = pd.DataFrame({
            "Open":   np.full(n, 1.10),
            "High":   np.full(n, 1.105),
            "Low":    np.full(n, 1.095),
            "Close":  np.full(n, 1.10),
            "Volume": np.full(n, 1000.0),
            "market": ["FOREX"] * n,
        })
        sig = mss_forex_15m(df)
        assert sig.signal == "FLAT"
        assert "DatetimeIndex" in sig.reason


# ─────────────────────────────────────────────────────────────────────────────
# A2 — Per-direction kill list + paper_only stamping
# ─────────────────────────────────────────────────────────────────────────────

class TestA2PerDirectionKill:
    def _build_signal(self, strategy: str, direction: str, conf: float = 0.75):
        from agents.intraday_strategies import IntradaySignal
        return IntradaySignal(
            strategy=strategy, timeframe="5m", signal=direction,
            confidence=conf, entry=1.10,
            stop_loss=1.095 if direction == "LONG" else 1.105,
            take_profit=1.115 if direction == "LONG" else 1.085,
            atr=0.005, rr_ratio=2.0, reason="test",
        )

    def test_disabled_direction_returns_flat(self, monkeypatch, tmp_path):
        from data import strategy_params as sp
        params_file = tmp_path / "params.json"
        params_file.write_text(json.dumps({
            "CAMARILLA_5M": {
                **sp._DEFAULT_ENTRY,
                "enabled": True,
                "disabled_directions": ["LONG"],
            },
        }), encoding="utf-8")
        monkeypatch.setattr(sp, "_PARAMS_FILE", params_file)
        monkeypatch.setattr(sp, "_PARAMS", {})
        sp.load_params()
        sig_long  = self._build_signal("CAMARILLA_5M", "LONG")
        sig_short = self._build_signal("CAMARILLA_5M", "SHORT")
        out_long  = sp.apply_params(sig_long)
        out_short = sp.apply_params(sig_short)
        assert out_long.signal == "FLAT", "LONG must be flattened (disabled direction)"
        assert "suspended" in out_long.reason
        assert out_short.signal == "SHORT", "SHORT must pass through"

    def test_paper_only_stamps_signal(self, monkeypatch, tmp_path):
        from data import strategy_params as sp
        params_file = tmp_path / "params.json"
        params_file.write_text(json.dumps({
            "VWAP_RSI_5M": {
                **sp._DEFAULT_ENTRY,
                "enabled": True,
                "paper_only": True,
            },
        }), encoding="utf-8")
        monkeypatch.setattr(sp, "_PARAMS_FILE", params_file)
        monkeypatch.setattr(sp, "_PARAMS", {})
        sp.load_params()
        sig = self._build_signal("VWAP_RSI_5M", "LONG")
        out = sp.apply_params(sig)
        # Must keep signal alive (so calibration tracks it) AND stamp paper_only
        assert out.signal == "LONG"
        assert getattr(out, "paper_only", False) is True
        assert "[paper_only]" in out.reason


# ─────────────────────────────────────────────────────────────────────────────
# A3 — Symbol policy
# ─────────────────────────────────────────────────────────────────────────────

class TestA3SymbolPolicy:
    def test_drop_returns_drop(self):
        from data.symbol_policy import get_disposition
        assert get_disposition("NQ=F") == "DROP"

    def test_paper_returns_paper(self):
        from unittest.mock import patch
        from data.symbol_policy import get_disposition
        # Patch overrides to empty so the test exercises hardcoded policy only
        # (symbol_promotions.json may promote symbols at runtime)
        with patch("data.symbol_policy._load_overrides", return_value={}):
            for s in ("EURUSD=X", "GBPUSD=X", "AUDUSD=X"):
                assert get_disposition(s) == "PAPER", f"{s} must be PAPER"

    def test_live_default(self):
        from data.symbol_policy import get_disposition
        # Symbols not in any kill list default to LIVE
        for s in ("YM=F", "SI=F", "USDJPY=X", "FAKE_NEW_PAIR"):
            assert get_disposition(s) == "LIVE"

    def test_normalisation_strips_yfinance_suffixes(self):
        from unittest.mock import patch
        from data.symbol_policy import get_disposition
        # The kill set internally is the normalised key; both yf and broker
        # forms must hit the same disposition. Patch overrides to isolate hardcoded policy.
        with patch("data.symbol_policy._load_overrides", return_value={}):
            assert get_disposition("EURUSD") == get_disposition("EURUSD=X") == "PAPER"
            assert get_disposition("NQ") == get_disposition("NQ=F") == "DROP"


# ─────────────────────────────────────────────────────────────────────────────
# A4 — Confidence boost stack cap
# ─────────────────────────────────────────────────────────────────────────────

class TestA4BoostCap:
    """The total positive boost stack must not exceed +0.20 above base."""

    def test_news_boost_field_zeroed(self):
        """News sentiment ±0.10 was removed — verify the source no longer
        sets news_boost = 0.10/-0.10 based on sentiment alignment."""
        src = Path("agents/signal_engine.py").read_text(encoding="utf-8")
        # The whole `news_boost = 0.10 if aligns else -0.10` block must be gone
        assert "news_boost = 0.10 if aligns" not in src, \
            "News boost ±0.10 must be removed (Round 4)"
        assert "news_boost = 0.0" in src, \
            "news_boost must be defaulted to 0.0"

    def test_absorption_boost_is_0_05(self):
        src = Path("agents/signal_engine.py").read_text(encoding="utf-8")
        assert "sig.confidence + 0.05" in src, \
            "Absorption boost must be reduced to +0.05 (was +0.10)"
        assert "sig.confidence + 0.10" not in src, \
            "Old +0.10 absorption boost must be gone"

    def test_live_adjustment_clamp_is_0_08(self):
        src = Path("agents/signal_engine.py").read_text(encoding="utf-8")
        assert "max(-0.08, min(0.08, raw_adj))" in src, \
            "Live adjustment must clamp to ±0.08"

    def test_bt_adjustment_clamp_is_0_05(self):
        src = Path("agents/signal_engine.py").read_text(encoding="utf-8")
        assert "max(-0.05, min(0.05, raw_adj))" in src, \
            "BT adjustment must clamp to ±0.05"

    def test_mtf_graded_adj_removed(self):
        """The soft ±0.09 MTF graded adjustment must be removed."""
        src = Path("agents/signal_engine.py").read_text(encoding="utf-8")
        # The code path used `mtf_score * 0.03` for LONG and `-mtf_score * 0.03` for SHORT
        assert "mtf_score * 0.03" not in src, \
            "MTF graded ±0.03 per-point adjustment must be removed (Round 4)"

    def test_capped_positive_uses_min_0_20(self):
        src = Path("agents/signal_engine.py").read_text(encoding="utf-8")
        assert "min(positive_boosts, 0.20)" in src, \
            "Total positive-boost cap of +0.20 must be enforced"


# ─────────────────────────────────────────────────────────────────────────────
# A5 — close_reason backfill
# ─────────────────────────────────────────────────────────────────────────────

class TestA5Backfill:
    def test_unknown_rows_get_classified(self, monkeypatch, tmp_path):
        from data import trade_outcomes as to
        log = tmp_path / "trade_log.json"
        log.write_text(json.dumps([
            # Closed WIN with no close_reason → should infer TP_LIKELY from positive profit
            {"ticket": 1, "ticker": "EURUSD=X", "strategy": "ORB_5M",
             "direction": "LONG", "confidence": 0.75, "entry": 1.10,
             "sl": 1.095, "tp": 1.115, "sent_at": "2026-04-29T10:00:00Z",
             "closed_at": "2026-04-29T13:00:00Z",
             "profit": 5.10, "outcome": "WIN"},
            # Closed LOSS with close_price near sl → SL classification
            {"ticket": 2, "ticker": "EURUSD=X", "strategy": "ORB_5M",
             "direction": "LONG", "confidence": 0.75, "entry": 1.10,
             "sl": 1.095, "tp": 1.115, "sent_at": "2026-04-29T10:00:00Z",
             "closed_at": "2026-04-29T13:00:00Z",
             "close_price": 1.0950,    # exactly at SL
             "profit": -5.20, "outcome": "LOSS"},
            # Already-classified rows must NOT be re-stamped
            {"ticket": 3, "ticker": "EURUSD=X", "strategy": "ORB_5M",
             "direction": "LONG", "confidence": 0.75, "entry": 1.10,
             "sl": 1.095, "tp": 1.115, "sent_at": "2026-04-29T10:00:00Z",
             "closed_at": "2026-04-29T13:00:00Z",
             "profit": 4.20, "outcome": "WIN", "close_reason": "TP",
             "close_reason_source": "AT_CLOSE"},
            # Open trade — must be skipped entirely
            {"ticket": 4, "ticker": "EURUSD=X", "strategy": "ORB_5M",
             "direction": "LONG", "confidence": 0.75, "entry": 1.10,
             "sl": 1.095, "tp": 1.115, "sent_at": "2026-04-29T10:00:00Z",
             "outcome": None, "profit": 0.0},
        ]), encoding="utf-8")
        monkeypatch.setattr(to, "_LOG_PATH", log)
        monkeypatch.setattr(to, "_LOAD_CACHE", None)

        counts = to.backfill_close_reasons()
        assert counts["seen"] == 3   # 3 closed trades
        assert counts["updated"] == 2   # ticket=3 was already tagged

        rows = json.loads(log.read_text(encoding="utf-8"))
        # Ticket 1: profit > 0, no close_price → TP_LIKELY fallback
        assert rows[0]["close_reason"] == "TP_LIKELY"
        assert rows[0]["close_reason_source"] == "BACKFILL"
        # Ticket 2: close_price exactly at sl → classifier returns "SL"
        assert rows[1]["close_reason"] == "SL"
        assert rows[1]["close_reason_source"] == "BACKFILL"
        # Ticket 3: pre-existing close_reason untouched
        assert rows[2]["close_reason"] == "TP"
        assert rows[2]["close_reason_source"] == "AT_CLOSE"
        # Ticket 4: open trade — no close_reason added
        assert "close_reason" not in rows[3]

    def test_idempotent(self, monkeypatch, tmp_path):
        """Running backfill twice must be a no-op the second time."""
        from data import trade_outcomes as to
        log = tmp_path / "trade_log.json"
        log.write_text(json.dumps([
            {"ticket": 1, "ticker": "EURUSD=X", "strategy": "ORB_5M",
             "direction": "LONG", "confidence": 0.75, "entry": 1.10,
             "sl": 1.095, "tp": 1.115, "sent_at": "2026-04-29T10:00:00Z",
             "closed_at": "2026-04-29T13:00:00Z",
             "profit": 5.10, "outcome": "WIN"},
        ]), encoding="utf-8")
        monkeypatch.setattr(to, "_LOG_PATH", log)
        monkeypatch.setattr(to, "_LOAD_CACHE", None)
        first  = to.backfill_close_reasons()
        second = to.backfill_close_reasons()
        assert first["updated"] == 1
        assert second["updated"] == 0
