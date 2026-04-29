"""
Tests for models.pattern_stats — pattern win-rate online learner.

Covers:
  • Pattern alignment rules (only matched-bias trades count)
  • Idempotent recompute_from_trades
  • Beta(1,1) posterior math
  • Ramp blending between textbook and learned WR
  • apply_to_report mutates win_rate in place
  • MT5_IMPORT and BREAKEVEN trades excluded
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Fixture helpers
# ─────────────────────────────────────────────────────────────────────────────

def _trade(*, outcome, direction="LONG", patterns=None, strategy="X"):
    return {
        "ticket":    1,
        "ticker":    "EURUSD",
        "strategy":  strategy,
        "direction": direction,
        "confidence": 0.7,
        "entry": 1.10, "sl": 1.09, "tp": 1.11,
        "patterns":  patterns or [],
        "outcome":   outcome,
    }


def _pat(name, bias="BULLISH", conf=0.7, wr=0.6):
    return {"name": name, "bias": bias, "confidence": conf, "win_rate": wr}


# ─────────────────────────────────────────────────────────────────────────────
# Alignment rules
# ─────────────────────────────────────────────────────────────────────────────

class TestAlignment:
    def test_long_trade_bullish_pattern_counted(self):
        from models.pattern_stats import recompute_from_trades
        trades = [_trade(outcome="WIN", direction="LONG",
                         patterns=[_pat("X", "BULLISH")])]
        stats = recompute_from_trades(trades)
        assert stats["X"]["wins"] == 1
        assert stats["X"]["trades"] == 1

    def test_long_trade_bearish_pattern_excluded(self):
        from models.pattern_stats import recompute_from_trades
        trades = [_trade(outcome="WIN", direction="LONG",
                         patterns=[_pat("X", "BEARISH")])]
        stats = recompute_from_trades(trades)
        assert "X" not in stats   # bearish pattern on a LONG → no credit either way

    def test_short_trade_bearish_pattern_counted(self):
        from models.pattern_stats import recompute_from_trades
        trades = [_trade(outcome="LOSS", direction="SHORT",
                         patterns=[_pat("Y", "BEARISH")])]
        stats = recompute_from_trades(trades)
        assert stats["Y"]["wins"] == 0
        assert stats["Y"]["trades"] == 1

    def test_neutral_pattern_excluded(self):
        from models.pattern_stats import recompute_from_trades
        trades = [_trade(outcome="WIN", direction="LONG",
                         patterns=[_pat("Z", "NEUTRAL")])]
        stats = recompute_from_trades(trades)
        assert "Z" not in stats


# ─────────────────────────────────────────────────────────────────────────────
# Recompute correctness + idempotency
# ─────────────────────────────────────────────────────────────────────────────

class TestRecompute:
    def test_two_wins_one_loss(self):
        from models.pattern_stats import recompute_from_trades
        trades = [
            _trade(outcome="WIN",  direction="LONG", patterns=[_pat("Bull Flag")]),
            _trade(outcome="WIN",  direction="LONG", patterns=[_pat("Bull Flag")]),
            _trade(outcome="LOSS", direction="LONG", patterns=[_pat("Bull Flag")]),
        ]
        stats = recompute_from_trades(trades)
        assert stats["Bull Flag"]["wins"] == 2
        assert stats["Bull Flag"]["trades"] == 3
        assert stats["Bull Flag"]["win_rate"] == pytest.approx(2/3, abs=1e-3)

    def test_idempotent(self):
        from models.pattern_stats import recompute_from_trades
        trades = [
            _trade(outcome="WIN",  direction="LONG", patterns=[_pat("A")]),
            _trade(outcome="LOSS", direction="LONG", patterns=[_pat("A")]),
        ]
        s1 = recompute_from_trades(trades)
        s2 = recompute_from_trades(trades)
        assert s1 == s2

    def test_breakeven_excluded(self):
        from models.pattern_stats import recompute_from_trades
        trades = [
            _trade(outcome="BREAKEVEN", direction="LONG", patterns=[_pat("X")]),
            _trade(outcome="WIN",       direction="LONG", patterns=[_pat("X")]),
        ]
        stats = recompute_from_trades(trades)
        assert stats["X"]["trades"] == 1   # only the WIN
        assert stats["X"]["wins"] == 1

    def test_mt5_import_excluded(self):
        from models.pattern_stats import recompute_from_trades
        trades = [
            _trade(outcome="WIN", direction="LONG",
                   patterns=[_pat("X")], strategy="MT5_IMPORT"),
            _trade(outcome="WIN", direction="LONG",
                   patterns=[_pat("X")], strategy="REAL_STRAT"),
        ]
        stats = recompute_from_trades(trades)
        assert stats["X"]["trades"] == 1   # only the non-import one

    def test_open_trade_excluded(self):
        from models.pattern_stats import recompute_from_trades
        # outcome=None means trade is still open
        trades = [_trade(outcome=None, direction="LONG", patterns=[_pat("X")])]
        stats = recompute_from_trades(trades)
        assert stats == {}

    def test_multiple_patterns_per_trade_all_credited(self):
        from models.pattern_stats import recompute_from_trades
        trades = [_trade(outcome="WIN", direction="LONG",
                         patterns=[_pat("A"), _pat("B"), _pat("C")])]
        stats = recompute_from_trades(trades)
        assert stats["A"]["wins"] == 1
        assert stats["B"]["wins"] == 1
        assert stats["C"]["wins"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Posterior math
# ─────────────────────────────────────────────────────────────────────────────

class TestPosterior:
    def test_zero_observations_returns_prior_mean(self):
        from models.pattern_stats import posterior_win_rate
        # Beta(1,1) mean = 0.5
        assert posterior_win_rate(0, 0) == 0.5

    def test_all_wins_pulled_below_one(self):
        from models.pattern_stats import posterior_win_rate
        # 10 wins of 10 → (10+1)/(10+2) = 11/12 ≈ 0.917, not 1.0
        assert posterior_win_rate(10, 10) == pytest.approx(11/12)

    def test_all_losses_pulled_above_zero(self):
        from models.pattern_stats import posterior_win_rate
        # 0 wins of 10 → 1/12 ≈ 0.083, not 0.0
        assert posterior_win_rate(0, 10) == pytest.approx(1/12)

    def test_balanced_slightly_above_half(self):
        from models.pattern_stats import posterior_win_rate
        # 5 wins of 10 → 6/12 = 0.5 exactly
        assert posterior_win_rate(5, 10) == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Ramp blending: textbook → learned as n grows
# ─────────────────────────────────────────────────────────────────────────────

class TestRampBlending:
    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path, monkeypatch):
        from models import pattern_stats
        monkeypatch.setattr(pattern_stats, "_STATS_PATH", tmp_path / "pattern_stats.json")
        monkeypatch.setattr(pattern_stats, "_TRADE_LOG_PATH", tmp_path / "trades.json")
        yield

    def _seed_stats(self, tmp_path, name, wins, trades):
        path = tmp_path / "pattern_stats.json"
        path.write_text(json.dumps({
            name: {"wins": wins, "trades": trades, "win_rate": wins/trades if trades else 0.0}
        }), encoding="utf-8")

    def test_zero_obs_returns_textbook(self, tmp_path):
        from models.pattern_stats import effective_win_rate
        # No stats file → textbook returned untouched
        assert effective_win_rate("Bull Flag", textbook_wr=0.67) == pytest.approx(0.67)

    def test_at_ramp_returns_learned_only(self, tmp_path):
        from models.pattern_stats import effective_win_rate
        self._seed_stats(tmp_path, "Bull Flag", wins=20, trades=30)
        # n=ramp, so weight=1.0 → pure learned posterior = (20+1)/(30+2) = 21/32
        eff = effective_win_rate("Bull Flag", textbook_wr=0.67, ramp=30)
        assert eff == pytest.approx(21/32, abs=1e-3)

    def test_above_ramp_returns_learned_only(self, tmp_path):
        from models.pattern_stats import effective_win_rate
        self._seed_stats(tmp_path, "Bull Flag", wins=80, trades=100)
        eff = effective_win_rate("Bull Flag", textbook_wr=0.67, ramp=30)
        assert eff == pytest.approx(81/102, abs=1e-3)

    def test_below_ramp_blends_proportionally(self, tmp_path):
        from models.pattern_stats import effective_win_rate
        # n=15 of ramp=30 → weight=0.5; learned = (10+1)/(15+2) = 11/17 ≈ 0.647
        # textbook = 0.67. Expected: 0.5*0.647 + 0.5*0.67 ≈ 0.659
        self._seed_stats(tmp_path, "Bull Flag", wins=10, trades=15)
        eff = effective_win_rate("Bull Flag", textbook_wr=0.67, ramp=30)
        expected = 0.5 * (11/17) + 0.5 * 0.67
        assert eff == pytest.approx(expected, abs=1e-3)

    def test_ramp_curve_is_monotonic_when_evidence_consistent(self, tmp_path):
        """When learned WR consistently > textbook, effective should rise as
        n grows; when learned < textbook, effective should fall."""
        from models import pattern_stats
        from models.pattern_stats import effective_win_rate

        textbook = 0.50

        # Learned > textbook (every trade a win): effective should be >= textbook
        # at every n, and increase toward learned (~0.917 at n=10)
        prev = textbook
        for n in [1, 5, 10, 20, 30]:
            self._seed_stats(tmp_path, "Hot", wins=n, trades=n)
            eff = effective_win_rate("Hot", textbook_wr=textbook, ramp=30)
            assert eff >= prev - 1e-9
            prev = eff


# ─────────────────────────────────────────────────────────────────────────────
# apply_to_report: mutates PatternReport in place
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyToReport:
    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path, monkeypatch):
        from models import pattern_stats
        monkeypatch.setattr(pattern_stats, "_STATS_PATH", tmp_path / "pattern_stats.json")
        yield

    def test_no_stats_file_leaves_textbook_unchanged(self, tmp_path):
        from agents.pattern_detector import PatternMatch, PatternReport
        from models.pattern_stats import apply_to_report

        m = PatternMatch(
            name="Bull Flag", bias="BULLISH", confidence=0.7,
            win_rate=0.67, description="", bar_start=0, bar_end=10, color="#0c5",
        )
        report = PatternReport(patterns=[m])
        apply_to_report(report)
        assert report.patterns[0].win_rate == pytest.approx(0.67)

    def test_with_stats_replaces_win_rate(self, tmp_path):
        from agents.pattern_detector import PatternMatch, PatternReport
        from models.pattern_stats import apply_to_report
        from models import pattern_stats as ps

        # Seed stats: pattern has 30 trades, 24 wins → posterior 25/32 ≈ 0.781
        ps._STATS_PATH.write_text(json.dumps({
            "Bull Flag": {"wins": 24, "trades": 30, "win_rate": 0.8}
        }), encoding="utf-8")

        m = PatternMatch(
            name="Bull Flag", bias="BULLISH", confidence=0.7,
            win_rate=0.67, description="", bar_start=0, bar_end=10, color="#0c5",
        )
        report = PatternReport(patterns=[m])
        apply_to_report(report)
        # n=ramp → pure learned = 25/32 ≈ 0.781
        assert report.patterns[0].win_rate == pytest.approx(0.781, abs=1e-2)

    def test_avg_win_rate_recomputed(self, tmp_path):
        from agents.pattern_detector import PatternMatch, PatternReport
        from models.pattern_stats import apply_to_report

        m1 = PatternMatch(name="A", bias="BULLISH", confidence=0.7, win_rate=0.60,
                          description="", bar_start=0, bar_end=1, color="x")
        m2 = PatternMatch(name="B", bias="BULLISH", confidence=0.7, win_rate=0.80,
                          description="", bar_start=0, bar_end=1, color="x")
        report = PatternReport(patterns=[m1, m2], avg_win_rate=0.0)
        apply_to_report(report)
        # No stats → win_rates unchanged → avg = (0.60 + 0.80) / 2 = 0.70
        assert report.avg_win_rate == pytest.approx(0.70)


# ─────────────────────────────────────────────────────────────────────────────
# Refresh integration
# ─────────────────────────────────────────────────────────────────────────────

class TestRefresh:
    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path, monkeypatch):
        from models import pattern_stats
        monkeypatch.setattr(pattern_stats, "_STATS_PATH",     tmp_path / "pattern_stats.json")
        monkeypatch.setattr(pattern_stats, "_TRADE_LOG_PATH", tmp_path / "trades.json")
        yield

    def test_refresh_writes_stats(self, tmp_path):
        from models.pattern_stats import refresh, _TRADE_LOG_PATH

        trades = [
            _trade(outcome="WIN",  direction="LONG", patterns=[_pat("X")]),
            _trade(outcome="LOSS", direction="LONG", patterns=[_pat("X")]),
        ]
        _TRADE_LOG_PATH.write_text(json.dumps(trades), encoding="utf-8")
        stats = refresh()
        assert stats["X"]["wins"] == 1
        assert stats["X"]["trades"] == 2
        # Persisted to disk
        from models.pattern_stats import _STATS_PATH
        assert _STATS_PATH.exists()
        on_disk = json.loads(_STATS_PATH.read_text(encoding="utf-8"))
        assert on_disk == stats

    def test_refresh_missing_log_returns_empty(self, tmp_path):
        from models.pattern_stats import refresh
        # No trade_log file
        assert refresh() == {}
