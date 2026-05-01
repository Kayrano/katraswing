"""Tests for the multi-cadence learning loop.

These tests freeze the clock via monkeypatch on `agents.learning_loop._now`
and redirect persistence into `tmp_path` so production state never moves.
The heavy fan-out functions (update_outcomes_from_mt5, run_intraday_backtest,
calibrator refit) are mocked so the tests stay fast and deterministic.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agents import learning_loop


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolate_state(monkeypatch, tmp_path):
    """Redirect every persistence path inside the module into tmp_path."""
    data_dir   = tmp_path / "data"
    reports    = data_dir / "reports"
    state_path = data_dir / "learning_state.json"
    audit_path = data_dir / "learning_log.jsonl"
    data_dir.mkdir()
    monkeypatch.setattr(learning_loop, "_DATA_DIR",    data_dir)
    monkeypatch.setattr(learning_loop, "_STATE_PATH",  state_path)
    monkeypatch.setattr(learning_loop, "_AUDIT_PATH",  audit_path)
    monkeypatch.setattr(learning_loop, "_REPORTS_DIR", reports)
    # Reset module locks so tests don't leak state
    monkeypatch.setattr(learning_loop, "_LOCKS", {
        "hourly": threading.Lock(),
        "daily":  threading.Lock(),
        "weekly": threading.Lock(),
    })
    monkeypatch.setattr(learning_loop, "_WATCHLIST", [])
    yield


def _frozen_clock(now: datetime):
    """Build a clock-reader that always returns `now`."""
    def _reader():
        return now
    return _reader


@pytest.fixture
def stub_hourly_dependencies(monkeypatch):
    """Stub out every external thing run_hourly touches so tests don't
    pull production data, network, or MT5."""
    calls = {"mt5": 0, "patterns": 0, "adapt": 0, "calibrator": 0, "iv": 0}

    def fake_mt5(magic=234100):
        calls["mt5"] += 1
        return 0

    def fake_patterns(*a, **kw):
        calls["patterns"] += 1
        return {}

    def fake_adapt(*a, **kw):
        calls["adapt"] += 1
        return 0

    fake_cal = MagicMock(sample_count=0, is_fitted=False)
    def fake_get_cal(*a, **kw):
        calls["calibrator"] += 1
        return fake_cal

    def fake_reset_iv():
        calls["iv"] += 1

    def fake_load():
        return []

    monkeypatch.setattr("data.trade_outcomes.update_outcomes_from_mt5", fake_mt5)
    monkeypatch.setattr("data.trade_outcomes._load", fake_load)
    monkeypatch.setattr("models.pattern_stats.refresh", fake_patterns)
    monkeypatch.setattr("data.strategy_params.adapt_all", fake_adapt)
    monkeypatch.setattr("models.calibration.get_calibrator", fake_get_cal)
    monkeypatch.setattr("models.intervention_stats.reset_cache", fake_reset_iv)
    return calls


# ── _is_due cadence rules ──────────────────────────────────────────────────

class TestCadenceRules:
    def test_hourly_due_on_first_run(self):
        now = datetime(2026, 5, 4, 14, 3, tzinfo=timezone.utc)
        assert learning_loop._is_due("hourly", now, {}) is True

    def test_hourly_not_due_after_minute_5(self):
        now = datetime(2026, 5, 4, 14, 6, tzinfo=timezone.utc)
        assert learning_loop._is_due("hourly", now, {}) is False

    def test_hourly_idempotent_within_same_hour(self):
        now = datetime(2026, 5, 4, 14, 3, tzinfo=timezone.utc)
        state = {"last_hourly_at": datetime(2026, 5, 4, 14, 1, tzinfo=timezone.utc)}
        assert learning_loop._is_due("hourly", now, state) is False

    def test_hourly_fires_after_restart(self):
        now = datetime(2026, 5, 4, 14, 3, tzinfo=timezone.utc)
        state = {"last_hourly_at": datetime(2026, 5, 4, 12, 33, tzinfo=timezone.utc)}
        assert learning_loop._is_due("hourly", now, state) is True

    def test_daily_due_only_after_23h_utc(self):
        # 22:00 — not yet due
        n22 = datetime(2026, 5, 4, 22, 0, tzinfo=timezone.utc)
        assert learning_loop._is_due("daily", n22, {}) is False
        # 23:00 — due
        n23 = datetime(2026, 5, 4, 23, 0, tzinfo=timezone.utc)
        assert learning_loop._is_due("daily", n23, {}) is True

    def test_daily_idempotent_same_day(self):
        now   = datetime(2026, 5, 4, 23, 30, tzinfo=timezone.utc)
        state = {"last_daily_at": datetime(2026, 5, 4, 23, 5, tzinfo=timezone.utc)}
        assert learning_loop._is_due("daily", now, state) is False

    def test_weekly_due_only_on_sunday_after_23h(self):
        # Mon (day=0) 23:00 → not weekly
        mon = datetime(2026, 5, 4, 23, 5, tzinfo=timezone.utc)
        assert learning_loop._is_due("weekly", mon, {}) is False
        # Sun (day=6) 23:30 → due
        sun = datetime(2026, 5, 3, 23, 30, tzinfo=timezone.utc)
        assert learning_loop._is_due("weekly", sun, {}) is True

    def test_weekly_idempotent_same_iso_week(self):
        sun   = datetime(2026, 5, 3, 23, 30, tzinfo=timezone.utc)   # ISO W18
        state = {"last_weekly_at": datetime(2026, 5, 3, 23, 5, tzinfo=timezone.utc)}
        assert learning_loop._is_due("weekly", sun, state) is False


# ── State persistence ──────────────────────────────────────────────────────

class TestStatePersistence:
    def test_round_trip(self, monkeypatch):
        # _load_state's clock-skew guard discards "future" timestamps relative
        # to _now() — so freeze the clock to *after* the saved value.
        saved = datetime(2026, 5, 4, 14, 3, tzinfo=timezone.utc)
        clock_after = datetime(2026, 5, 4, 14, 5, tzinfo=timezone.utc)
        monkeypatch.setattr(learning_loop, "_now", _frozen_clock(clock_after))
        learning_loop._save_state({"last_hourly_at": saved})
        loaded = learning_loop._load_state()
        assert loaded["last_hourly_at"] == saved

    def test_clock_skew_future_timestamp_ignored(self, monkeypatch):
        future = datetime(2099, 1, 1, tzinfo=timezone.utc)
        learning_loop._save_state({"last_hourly_at": future})
        loaded = learning_loop._load_state()
        # Future ts → treated as missing → next tick can fire
        assert "last_hourly_at" not in loaded

    def test_corrupt_state_returns_empty(self, monkeypatch):
        learning_loop._STATE_PATH.write_text("{not json", encoding="utf-8")
        loaded = learning_loop._load_state()
        assert loaded == {}

    def test_atomic_write_does_not_leave_partial(self, monkeypatch):
        """If os.replace raises mid-write, the prior state must still be readable."""
        first = datetime(2026, 5, 4, 14, 3, tzinfo=timezone.utc)
        clock_after = datetime(2026, 5, 4, 14, 5, tzinfo=timezone.utc)
        monkeypatch.setattr(learning_loop, "_now", _frozen_clock(clock_after))
        learning_loop._save_state({"last_hourly_at": first})
        original_replace = learning_loop.os.replace

        def boom(*a, **kw):
            raise OSError("disk full")

        monkeypatch.setattr(learning_loop.os, "replace", boom)
        with pytest.raises(OSError):
            second = datetime(2026, 5, 4, 15, 3, tzinfo=timezone.utc)
            learning_loop._save_state({"last_hourly_at": second})
        monkeypatch.setattr(learning_loop.os, "replace", original_replace)
        loaded = learning_loop._load_state()
        assert loaded["last_hourly_at"] == first


# ── tick() integration ─────────────────────────────────────────────────────

class TestTickIntegration:
    def test_tick_fires_hourly_on_first_run(self, monkeypatch, stub_hourly_dependencies):
        now = datetime(2026, 5, 4, 14, 3, tzinfo=timezone.utc)
        monkeypatch.setattr(learning_loop, "_now", _frozen_clock(now))
        result = learning_loop.tick()
        assert "hourly" in result["fired"]
        assert stub_hourly_dependencies["mt5"] == 1
        # State persisted
        loaded = learning_loop._load_state()
        assert "last_hourly_at" in loaded

    def test_tick_idempotent_within_hour(self, monkeypatch, stub_hourly_dependencies):
        now = datetime(2026, 5, 4, 14, 3, tzinfo=timezone.utc)
        monkeypatch.setattr(learning_loop, "_now", _frozen_clock(now))
        learning_loop.tick()
        learning_loop.tick()
        # Second call: not due (already fired this hour)
        assert stub_hourly_dependencies["mt5"] == 1

    def test_tick_audit_log_appended(self, monkeypatch, stub_hourly_dependencies):
        now = datetime(2026, 5, 4, 14, 3, tzinfo=timezone.utc)
        monkeypatch.setattr(learning_loop, "_now", _frozen_clock(now))
        learning_loop.tick()
        rows = [json.loads(line) for line in
                learning_loop._AUDIT_PATH.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 1
        assert rows[0]["kind"] == "hourly"
        assert rows[0]["errors"] == []


# ── run_weekly prune + promote ─────────────────────────────────────────────

class TestWeeklyMutations:
    """Verify the live-from-day-1 prune and promote rules write the
    expected updates into strategy_params (mocked to a tmp file)."""

    def _seed_trade_log(self, tmp_path: Path, *, strategy: str, n: int, wins: int,
                       avg_profit: float, ticker: str = "EURUSD"):
        trade_log = tmp_path / "data" / "trade_log.json"
        trade_log.parent.mkdir(parents=True, exist_ok=True)
        # closed_at stamped 5 days ago so the 30d window catches them all
        close_at = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        trades: list[dict] = []
        for i in range(n):
            outcome = "WIN" if i < wins else "LOSS"
            profit = avg_profit if outcome == "WIN" else -avg_profit
            trades.append({
                "ticket":    1000 + i,
                "ticker":    ticker,
                "strategy":  strategy,
                "direction": "LONG",
                "confidence": 0.75,
                "entry": 1.10, "sl": 1.095, "tp": 1.115,
                "sent_at":  close_at,
                "closed_at": close_at,
                "profit": round(profit, 2),
                "outcome": outcome,
            })
        trade_log.write_text(json.dumps(trades), encoding="utf-8")
        return trade_log

    def test_weekly_disables_below_threshold(self, monkeypatch, tmp_path):
        """20 trades, 27% WR, PF<1.0 → must flip enabled=False."""
        trade_log = self._seed_trade_log(
            tmp_path, strategy="LOSER_X", n=20, wins=5, avg_profit=10.0,
        )
        params_file = tmp_path / "strategy_params.json"
        params_file.write_text(json.dumps({
            "LOSER_X": {
                "sl_mult": 1.0, "tp_mult": 1.0, "conf_floor": 0.6,
                "enabled": True, "paper_only": False,
                "trades_seen": 20, "wins": 5, "win_rate": 0.25,
                "last_adapted": None, "adapt_count": 0,
            },
        }), encoding="utf-8")
        # Patch trade_log + strategy_params paths
        monkeypatch.setattr("data.trade_outcomes._LOG_PATH", trade_log)
        monkeypatch.setattr("data.trade_outcomes._LOAD_CACHE", None)
        monkeypatch.setattr("data.strategy_params._PARAMS_FILE", params_file)
        monkeypatch.setattr("data.strategy_params._PARAMS", {})
        # Block fetcher + calibrator since they hit network
        monkeypatch.setattr(
            "data.fetcher_intraday.fetch_intraday_data",
            lambda *a, **kw: None,
        )
        monkeypatch.setattr("models.calibration.reset_singleton", lambda: None)
        monkeypatch.setattr(
            "models.calibration.get_calibrator",
            lambda *a, **kw: MagicMock(sample_count=20, is_fitted=False),
        )
        now = datetime(2026, 5, 3, 23, 5, tzinfo=timezone.utc)
        monkeypatch.setattr(learning_loop, "_now", _frozen_clock(now))
        learning_loop.run_weekly(now)
        loaded = json.loads(params_file.read_text(encoding="utf-8"))
        assert loaded["LOSER_X"]["enabled"] is False, \
            "LOSER_X (n=20, WR=25%, PF<1) must be disabled"

    def test_weekly_promotes_paper_strategy(self, monkeypatch, tmp_path):
        """22 trades, 55% WR, PF>1.3 on a paper_only strategy → promotes."""
        trade_log = self._seed_trade_log(
            tmp_path, strategy="PAPER_GOOD", n=22, wins=12, avg_profit=10.0,
        )
        params_file = tmp_path / "strategy_params.json"
        params_file.write_text(json.dumps({
            "PAPER_GOOD": {
                "sl_mult": 1.0, "tp_mult": 1.0, "conf_floor": 0.6,
                "enabled": False, "paper_only": True,
                "trades_seen": 22, "wins": 12, "win_rate": 0.545,
                "last_adapted": None, "adapt_count": 0,
            },
        }), encoding="utf-8")
        monkeypatch.setattr("data.trade_outcomes._LOG_PATH", trade_log)
        monkeypatch.setattr("data.trade_outcomes._LOAD_CACHE", None)
        monkeypatch.setattr("data.strategy_params._PARAMS_FILE", params_file)
        monkeypatch.setattr("data.strategy_params._PARAMS", {})
        monkeypatch.setattr(
            "data.fetcher_intraday.fetch_intraday_data",
            lambda *a, **kw: None,
        )
        monkeypatch.setattr("models.calibration.reset_singleton", lambda: None)
        monkeypatch.setattr(
            "models.calibration.get_calibrator",
            lambda *a, **kw: MagicMock(sample_count=22, is_fitted=False),
        )
        now = datetime(2026, 5, 3, 23, 5, tzinfo=timezone.utc)
        monkeypatch.setattr(learning_loop, "_now", _frozen_clock(now))
        learning_loop.run_weekly(now)
        loaded = json.loads(params_file.read_text(encoding="utf-8"))
        assert loaded["PAPER_GOOD"]["paper_only"] is False
        assert loaded["PAPER_GOOD"]["enabled"] is True

    def test_weekly_rotates_audit_log(self, monkeypatch, tmp_path):
        """The JSONL audit log is renamed to learning_log.YYYY-WNN.jsonl."""
        learning_loop._AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        learning_loop._AUDIT_PATH.write_text(
            '{"ts":"2026-04-26T12:00Z","kind":"hourly","duration_s":1.0,"errors":[]}\n',
            encoding="utf-8",
        )
        params_file = tmp_path / "strategy_params.json"
        params_file.write_text("{}", encoding="utf-8")
        monkeypatch.setattr("data.strategy_params._PARAMS_FILE", params_file)
        monkeypatch.setattr("data.strategy_params._PARAMS", {})
        monkeypatch.setattr(
            "data.trade_outcomes._LOG_PATH", tmp_path / "trade_log.json",
        )
        monkeypatch.setattr("data.trade_outcomes._LOAD_CACHE", None)
        (tmp_path / "trade_log.json").write_text("[]", encoding="utf-8")
        monkeypatch.setattr(
            "data.fetcher_intraday.fetch_intraday_data", lambda *a, **kw: None,
        )
        monkeypatch.setattr("models.calibration.reset_singleton", lambda: None)
        monkeypatch.setattr(
            "models.calibration.get_calibrator",
            lambda *a, **kw: MagicMock(sample_count=0, is_fitted=False),
        )
        now = datetime(2026, 5, 3, 23, 5, tzinfo=timezone.utc)
        monkeypatch.setattr(learning_loop, "_now", _frozen_clock(now))
        learning_loop.run_weekly(now)
        archived = learning_loop._DATA_DIR / "learning_log.2026-W18.jsonl"
        assert archived.exists()
        assert not learning_loop._AUDIT_PATH.exists()


# ── Regime classifier ──────────────────────────────────────────────────────

class TestRegimeClassifier:
    def test_trending_label(self):
        import pandas as pd
        import numpy as np
        from agents.regime_classifier import classify
        # Strongly trending price series → high ADX
        n = 400
        closes = np.linspace(100, 130, n)
        df = pd.DataFrame({
            "Open":   closes,
            "High":   closes + 0.5,
            "Low":    closes - 0.5,
            "Close":  closes,
            "Volume": np.full(n, 1000.0),
        })
        r = classify(df, ticker="X", lookback_bars=300)
        assert r.label == "TRENDING"
        assert r.pct_trending > 0.55

    def test_insufficient_data(self):
        import pandas as pd
        from agents.regime_classifier import classify
        df = pd.DataFrame({"Open": [], "High": [], "Low": [], "Close": [], "Volume": []})
        r = classify(df, ticker="X")
        assert r.label == "INSUFFICIENT_DATA"
