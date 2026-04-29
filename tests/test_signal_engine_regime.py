"""
Tests for the soft ADX regime router in agents.signal_engine.

The original logic had hard cliffs at ADX=25 (trending) and ADX=20 (ranging),
producing identical penalties on either side of those boundaries and
nothing between. The new `_trend_weight` is a logistic centered at 22.5,
so penalties scale smoothly across the transitional zone.
"""
from __future__ import annotations

import math

import pytest

from agents.signal_engine import (
    _trend_weight,
    _ADX_CENTER,
    _ADX_SLOPE,
    _MR_MAX_PENALTY,
    _TREND_MAX_PENALTY,
)


# ─────────────────────────────────────────────────────────────────────────────
# _trend_weight curve
# ─────────────────────────────────────────────────────────────────────────────

class TestTrendWeight:
    def test_unknown_adx_returns_neutral(self):
        assert _trend_weight(0) == 0.5
        assert _trend_weight(-1) == 0.5

    def test_at_center_returns_half(self):
        assert _trend_weight(_ADX_CENTER) == pytest.approx(0.5)

    def test_far_below_center_pulls_toward_zero(self):
        # ADX = 10 → far ranging
        w = _trend_weight(10)
        assert w < 0.05

    def test_far_above_center_pulls_toward_one(self):
        # ADX = 35 → firmly trending
        w = _trend_weight(35)
        assert w > 0.95

    def test_monotonic_in_adx(self):
        prev = _trend_weight(5)
        for adx in range(6, 50):
            w = _trend_weight(adx)
            assert w >= prev - 1e-9, f"non-monotone at ADX={adx}"
            prev = w

    def test_symmetric_around_center(self):
        # Logistic is symmetric: weight(center+d) + weight(center-d) ≈ 1
        for d in (1.0, 2.5, 5.0, 7.5):
            assert (_trend_weight(_ADX_CENTER + d)
                    + _trend_weight(_ADX_CENTER - d)) == pytest.approx(1.0, abs=1e-3)

    def test_at_old_thresholds_in_expected_zones(self):
        # ADX=25 was the old TRENDING cliff — new weight is ~0.73
        assert 0.70 < _trend_weight(25) < 0.78
        # ADX=20 was the old RANGING cliff — new weight is ~0.27
        assert 0.22 < _trend_weight(20) < 0.30


# ─────────────────────────────────────────────────────────────────────────────
# Penalty derivation: max penalties at extremes, smooth scaling between
# ─────────────────────────────────────────────────────────────────────────────

class TestPenaltyScaling:
    def test_mr_penalty_at_extreme_trending(self):
        # ADX=40 → near-pure trending → MR penalty close to max
        tw = _trend_weight(40)
        penalty = _MR_MAX_PENALTY * tw
        assert penalty == pytest.approx(_MR_MAX_PENALTY, abs=0.005)

    def test_mr_penalty_at_extreme_ranging(self):
        # ADX=10 → near-pure ranging → MR penalty close to zero
        tw = _trend_weight(10)
        penalty = _MR_MAX_PENALTY * tw
        assert penalty < 0.01

    def test_trend_penalty_at_extreme_ranging(self):
        # ADX=10 → near-pure ranging → trend penalty close to max
        tw = _trend_weight(10)
        penalty = _TREND_MAX_PENALTY * (1.0 - tw)
        assert penalty == pytest.approx(_TREND_MAX_PENALTY, abs=0.005)

    def test_trend_penalty_at_extreme_trending(self):
        # ADX=40 → near-pure trending → trend penalty close to zero
        tw = _trend_weight(40)
        penalty = _TREND_MAX_PENALTY * (1.0 - tw)
        assert penalty < 0.01

    def test_both_penalties_balance_at_center(self):
        # At the midpoint, both strategies are penalized at half their max
        tw = _trend_weight(_ADX_CENTER)
        mr_pen    = _MR_MAX_PENALTY    * tw
        trend_pen = _TREND_MAX_PENALTY * (1.0 - tw)
        assert mr_pen    == pytest.approx(_MR_MAX_PENALTY    / 2)
        assert trend_pen == pytest.approx(_TREND_MAX_PENALTY / 2)

    def test_smooth_transition_no_cliffs(self):
        """In the old hard-cliff system the penalty for an MR strategy was
        either 0.12 (ADX>25) or 0.0 (ADX<=25) — a step function. Here it
        must change gradually across the transition zone."""
        prev_penalty = 0.0
        for adx in range(15, 35):
            tw = _trend_weight(adx)
            penalty = _MR_MAX_PENALTY * tw
            delta = abs(penalty - prev_penalty)
            assert delta < 0.04, f"jump at ADX={adx}: {prev_penalty:.3f}→{penalty:.3f}"
            prev_penalty = penalty


# ─────────────────────────────────────────────────────────────────────────────
# Verify no regressions vs old cliff behavior at deep extremes
# ─────────────────────────────────────────────────────────────────────────────

class TestExtremeBehavior:
    def test_deep_trending_mr_still_penalized(self):
        """At ADX=30, an MR strategy under the old logic took the full -0.12.
        New logic gives _MR_MAX_PENALTY × _trend_weight(30) ≈ 0.114 — same
        order of magnitude, just slightly softer."""
        tw = _trend_weight(30)
        penalty = _MR_MAX_PENALTY * tw
        assert 0.10 < penalty <= _MR_MAX_PENALTY

    def test_deep_ranging_trend_still_penalized(self):
        tw = _trend_weight(15)
        penalty = _TREND_MAX_PENALTY * (1.0 - tw)
        assert 0.08 < penalty <= _TREND_MAX_PENALTY
