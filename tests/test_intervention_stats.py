"""
Tests for models.intervention_stats — the trade-manager learning loop.

Covers:
  • Joining assessment_log + trade_log by ticket
  • Per-(strategy, action) stat accumulation
  • Filtering: open trades, HOLD actions, untracked tickets
  • Dedupe of duplicate assessment entries
  • Health-score bias logic gated by MIN_SAMPLES_FOR_BIAS
  • Bounded bias offset (cannot exceed MAX_BIAS_OFFSET)
"""
from __future__ import annotations

import json

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Fixture builders
# ─────────────────────────────────────────────────────────────────────────────

def _trade(*, ticket, strategy="X", outcome="WIN", profit=1.0):
    return {
        "ticket":   ticket,
        "ticker":   "EURUSD",
        "strategy": strategy,
        "outcome":  outcome,
        "profit":   profit,
        "direction": "LONG",
        "confidence": 0.7,
    }


def _assessment(*, ticket, action, assessed_at="2026-01-01T10:00:00", strategy_hint=""):
    """Note: real assessments don't carry strategy — it's looked up via
    trade_log by ticket. This fixture mirrors the actual schema."""
    return {
        "ticket":      ticket,
        "symbol":      "EURUSD",
        "direction":   "LONG",
        "assessed_at": assessed_at,
        "health_score": 0.5,
        "action":      action,
        "new_sl":      None,
        "new_tp":      None,
        "urgency":     "MEDIUM",
        "reason":      "test",
        "acted_on":    True,
        "dry_run":     False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core join + stats accumulation
# ─────────────────────────────────────────────────────────────────────────────

class TestJoinAndAccumulate:
    def test_single_close_on_winning_trade(self):
        from models.intervention_stats import compute_intervention_stats
        trades = [_trade(ticket=1, strategy="X", outcome="WIN", profit=10.0)]
        assessments = [_assessment(ticket=1, action="CLOSE")]
        stats = compute_intervention_stats(assessments, trades)
        bk = ("X", "CLOSE")
        assert bk in stats
        assert stats[bk].count == 1
        assert stats[bk].wins == 1
        assert stats[bk].losses == 0
        assert stats[bk].total_profit == pytest.approx(10.0)

    def test_multiple_actions_on_same_trade(self):
        from models.intervention_stats import compute_intervention_stats
        trades = [_trade(ticket=1, strategy="X", outcome="WIN", profit=20.0)]
        assessments = [
            _assessment(ticket=1, action="MODIFY_SL", assessed_at="2026-01-01T10:00:00"),
            _assessment(ticket=1, action="MODIFY_SL", assessed_at="2026-01-01T11:00:00"),
            _assessment(ticket=1, action="PARTIAL_CLOSE", assessed_at="2026-01-01T12:00:00"),
        ]
        stats = compute_intervention_stats(assessments, trades)
        # Each action attributed independently — both MODIFY_SL events
        # credit the WIN, plus the PARTIAL_CLOSE
        assert stats[("X", "MODIFY_SL")].count == 2
        assert stats[("X", "MODIFY_SL")].wins == 2
        assert stats[("X", "PARTIAL_CLOSE")].count == 1

    def test_loss_outcome_recorded(self):
        from models.intervention_stats import compute_intervention_stats
        trades = [_trade(ticket=2, strategy="X", outcome="LOSS", profit=-5.0)]
        assessments = [_assessment(ticket=2, action="CLOSE")]
        stats = compute_intervention_stats(assessments, trades)
        st = stats[("X", "CLOSE")]
        assert st.losses == 1
        assert st.wins == 0
        assert st.total_profit == pytest.approx(-5.0)

    def test_per_strategy_separation(self):
        from models.intervention_stats import compute_intervention_stats
        trades = [
            _trade(ticket=1, strategy="X", outcome="WIN", profit=10.0),
            _trade(ticket=2, strategy="Y", outcome="LOSS", profit=-3.0),
        ]
        assessments = [
            _assessment(ticket=1, action="CLOSE"),
            _assessment(ticket=2, action="CLOSE"),
        ]
        stats = compute_intervention_stats(assessments, trades)
        assert stats[("X", "CLOSE")].wins == 1
        assert stats[("Y", "CLOSE")].losses == 1


# ─────────────────────────────────────────────────────────────────────────────
# Filters: HOLD, open trades, missing tickets
# ─────────────────────────────────────────────────────────────────────────────

class TestFilters:
    def test_hold_actions_excluded(self):
        from models.intervention_stats import compute_intervention_stats
        trades = [_trade(ticket=1, strategy="X", outcome="WIN")]
        assessments = [_assessment(ticket=1, action="HOLD")]
        stats = compute_intervention_stats(assessments, trades)
        assert ("X", "HOLD") not in stats

    def test_open_trades_excluded(self):
        from models.intervention_stats import compute_intervention_stats
        # outcome=None → trade still open; assessment for it should be skipped
        trades = [_trade(ticket=1, strategy="X", outcome=None)]
        assessments = [_assessment(ticket=1, action="CLOSE")]
        stats = compute_intervention_stats(assessments, trades)
        assert stats == {}

    def test_orphan_assessment_excluded(self):
        from models.intervention_stats import compute_intervention_stats
        # Assessment for a ticket that's not in trade_log
        trades = [_trade(ticket=1, strategy="X", outcome="WIN")]
        assessments = [_assessment(ticket=999, action="CLOSE")]
        stats = compute_intervention_stats(assessments, trades)
        assert stats == {}


# ─────────────────────────────────────────────────────────────────────────────
# Dedupe: duplicate (ticket, action, timestamp) only counts once
# ─────────────────────────────────────────────────────────────────────────────

class TestDedupe:
    def test_duplicate_assessments_counted_once(self):
        from models.intervention_stats import compute_intervention_stats
        trades = [_trade(ticket=1, strategy="X", outcome="WIN", profit=5.0)]
        # Same assessment recorded twice (e.g., retry after crash)
        assessments = [
            _assessment(ticket=1, action="CLOSE", assessed_at="2026-01-01T10:00:00"),
            _assessment(ticket=1, action="CLOSE", assessed_at="2026-01-01T10:00:00"),
        ]
        stats = compute_intervention_stats(assessments, trades)
        assert stats[("X", "CLOSE")].count == 1

    def test_same_action_different_time_both_counted(self):
        from models.intervention_stats import compute_intervention_stats
        trades = [_trade(ticket=1, strategy="X", outcome="WIN", profit=5.0)]
        assessments = [
            _assessment(ticket=1, action="MODIFY_SL", assessed_at="2026-01-01T10:00:00"),
            _assessment(ticket=1, action="MODIFY_SL", assessed_at="2026-01-01T11:00:00"),
        ]
        stats = compute_intervention_stats(assessments, trades)
        assert stats[("X", "MODIFY_SL")].count == 2


# ─────────────────────────────────────────────────────────────────────────────
# ActionStats math
# ─────────────────────────────────────────────────────────────────────────────

class TestActionStatsMath:
    def test_win_rate_computed(self):
        from models.intervention_stats import ActionStats
        s = ActionStats(strategy="X", action="CLOSE", count=10, wins=7, losses=3,
                        total_profit=15.0)
        assert s.win_rate == pytest.approx(0.7)
        assert s.avg_profit == pytest.approx(1.5)

    def test_win_rate_zero_when_no_observations(self):
        from models.intervention_stats import ActionStats
        s = ActionStats(strategy="X", action="CLOSE")
        assert s.win_rate is None
        assert s.avg_profit is None


# ─────────────────────────────────────────────────────────────────────────────
# Bias hook: get_health_bias
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthBias:
    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path, monkeypatch):
        from models import intervention_stats
        monkeypatch.setattr(intervention_stats, "_ASSESSMENT_LOG", tmp_path / "assessment_log.json")
        monkeypatch.setattr(intervention_stats, "_TRADE_LOG_PATH", tmp_path / "trade_log.json")
        intervention_stats.reset_cache()
        yield
        intervention_stats.reset_cache()

    def _seed(self, tmp_path, *, n_close: int, win_rate: float, strategy: str = "X"):
        """Seed log files with N CLOSE actions for `strategy`, with given WR."""
        n_wins = int(n_close * win_rate)
        n_losses = n_close - n_wins

        trades = []
        assessments = []
        for i in range(n_wins):
            trades.append(_trade(ticket=i, strategy=strategy, outcome="WIN", profit=1.0))
            assessments.append(_assessment(ticket=i, action="CLOSE",
                                           assessed_at=f"2026-01-01T{i:02d}:00:00"))
        for i in range(n_wins, n_wins + n_losses):
            trades.append(_trade(ticket=i, strategy=strategy, outcome="LOSS", profit=-1.0))
            assessments.append(_assessment(ticket=i, action="CLOSE",
                                           assessed_at=f"2026-01-01T{i:02d}:00:00"))

        from models import intervention_stats
        intervention_stats._TRADE_LOG_PATH.write_text(json.dumps(trades), encoding="utf-8")
        intervention_stats._ASSESSMENT_LOG.write_text(json.dumps(assessments), encoding="utf-8")

    def test_below_min_samples_returns_zero(self, tmp_path):
        from models.intervention_stats import get_health_bias, MIN_SAMPLES_FOR_BIAS
        # Seed with fewer than MIN_SAMPLES_FOR_BIAS — no bias
        self._seed(tmp_path, n_close=MIN_SAMPLES_FOR_BIAS - 1, win_rate=0.9)
        assert get_health_bias("X") == 0.0

    def test_high_close_winrate_biases_health_up(self, tmp_path):
        from models.intervention_stats import get_health_bias, MAX_BIAS_OFFSET
        # We've been closing winners → bias UP (close less eagerly)
        self._seed(tmp_path, n_close=20, win_rate=0.80)
        assert get_health_bias("X") == pytest.approx(MAX_BIAS_OFFSET)

    def test_low_close_winrate_biases_health_down(self, tmp_path):
        from models.intervention_stats import get_health_bias, MAX_BIAS_OFFSET
        # We've been closing losers (good) — small downward bias to encourage
        self._seed(tmp_path, n_close=20, win_rate=0.20)
        assert get_health_bias("X") == pytest.approx(-MAX_BIAS_OFFSET / 2.0)

    def test_neutral_winrate_returns_zero(self, tmp_path):
        from models.intervention_stats import get_health_bias
        self._seed(tmp_path, n_close=20, win_rate=0.50)
        assert get_health_bias("X") == 0.0

    def test_unknown_strategy_returns_zero(self, tmp_path):
        from models.intervention_stats import get_health_bias
        self._seed(tmp_path, n_close=20, win_rate=0.80, strategy="OTHER")
        assert get_health_bias("X_NOT_PRESENT") == 0.0

    def test_bias_is_bounded(self, tmp_path):
        from models.intervention_stats import get_health_bias, MAX_BIAS_OFFSET
        # 100% close-win-rate (extreme regret) — bias must still cap at MAX
        self._seed(tmp_path, n_close=50, win_rate=1.0)
        bias = get_health_bias("X")
        assert abs(bias) <= MAX_BIAS_OFFSET + 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestSummarizers:
    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path, monkeypatch):
        from models import intervention_stats
        monkeypatch.setattr(intervention_stats, "_ASSESSMENT_LOG", tmp_path / "a.json")
        monkeypatch.setattr(intervention_stats, "_TRADE_LOG_PATH", tmp_path / "t.json")
        intervention_stats.reset_cache()
        yield

    def test_summarize_for_strategy_filters(self, tmp_path):
        from models import intervention_stats
        from models.intervention_stats import summarize_for_strategy

        trades = [
            _trade(ticket=1, strategy="X", outcome="WIN", profit=2.0),
            _trade(ticket=2, strategy="Y", outcome="LOSS", profit=-1.0),
        ]
        assessments = [
            _assessment(ticket=1, action="CLOSE"),
            _assessment(ticket=2, action="CLOSE"),
        ]
        intervention_stats._TRADE_LOG_PATH.write_text(json.dumps(trades), encoding="utf-8")
        intervention_stats._ASSESSMENT_LOG.write_text(json.dumps(assessments), encoding="utf-8")

        result = summarize_for_strategy("X")
        assert "CLOSE" in result
        assert result["CLOSE"]["wins"] == 1
        assert "Y" not in result   # only X's actions

    def test_summarize_all_groups_by_strategy(self, tmp_path):
        from models import intervention_stats
        from models.intervention_stats import summarize_all

        trades = [
            _trade(ticket=1, strategy="X", outcome="WIN", profit=2.0),
            _trade(ticket=2, strategy="Y", outcome="LOSS", profit=-1.0),
        ]
        assessments = [
            _assessment(ticket=1, action="CLOSE"),
            _assessment(ticket=2, action="MODIFY_SL"),
        ]
        intervention_stats._TRADE_LOG_PATH.write_text(json.dumps(trades), encoding="utf-8")
        intervention_stats._ASSESSMENT_LOG.write_text(json.dumps(assessments), encoding="utf-8")

        out = summarize_all()
        assert set(out.keys()) == {"X", "Y"}
        assert "CLOSE" in out["X"]
        assert "MODIFY_SL" in out["Y"]
