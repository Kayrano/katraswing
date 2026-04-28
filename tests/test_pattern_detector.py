"""
Golden-output tests for pattern_detector pivot helpers.

The original `_find_local_highs` and `_find_local_lows` use Python for-loops
with O(N×order) cost. Tier 2.2 swaps them for `scipy.signal.argrelextrema`,
which is vectorized and an order of magnitude faster on long arrays.

These tests pin the exact pivot indices produced by the current
implementation so the rewrite cannot silently change which patterns fire
or where they're anchored. The vectorized version must produce identical
output for the same inputs.
"""
from __future__ import annotations

import numpy as np
import pytest

from agents.pattern_detector import _find_local_highs, _find_local_lows


def _python_local_highs(highs: np.ndarray, order: int = 3) -> list[int]:
    """Reference Python implementation captured before vectorization. Used as
    the source of truth — the vectorized implementation must agree with this."""
    result: list[int] = []
    for i in range(order, len(highs) - order):
        if highs[i] == max(highs[i - order: i + order + 1]):
            result.append(i)
    return result


def _python_local_lows(lows: np.ndarray, order: int = 3) -> list[int]:
    result: list[int] = []
    for i in range(order, len(lows) - order):
        if lows[i] == min(lows[i - order: i + order + 1]):
            result.append(i)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Hand-crafted fixtures: each has known pivot positions
# ─────────────────────────────────────────────────────────────────────────────

class TestSimpleFixtures:
    def test_strict_single_peak(self):
        # one clear peak at index 5
        arr = np.array([1.0, 2, 3, 4, 5, 10, 5, 4, 3, 2, 1], dtype=float)
        assert _find_local_highs(arr, order=3) == _python_local_highs(arr, order=3)
        assert _find_local_highs(arr, order=3) == [5]

    def test_strict_single_trough(self):
        arr = np.array([10.0, 9, 8, 7, 5, 1, 5, 7, 8, 9, 10], dtype=float)
        assert _find_local_lows(arr, order=3) == _python_local_lows(arr, order=3)
        assert _find_local_lows(arr, order=3) == [5]

    def test_no_pivots_in_monotonic(self):
        arr = np.arange(20, dtype=float)
        assert _find_local_highs(arr, order=3) == []
        assert _find_local_lows(arr, order=3) == []

    def test_empty_array(self):
        arr = np.array([], dtype=float)
        assert _find_local_highs(arr, order=3) == []

    def test_too_short_for_order(self):
        arr = np.array([1.0, 2, 1], dtype=float)
        # order=3 needs at least 7 bars (3 + 1 + 3)
        assert _find_local_highs(arr, order=3) == []
        assert _find_local_lows(arr, order=3) == []

    def test_plateau_with_ties_uses_max_equality(self):
        """Original uses `==` which means flat tops where every bar equals the
        max all qualify. Behavior is documented intentional — vectorized
        version must match."""
        arr = np.array([1.0, 2, 3, 5, 5, 5, 3, 2, 1, 0, 1, 2], dtype=float)
        py = _python_local_highs(arr, order=3)
        impl = _find_local_highs(arr, order=3)
        assert impl == py


# ─────────────────────────────────────────────────────────────────────────────
# Random walks — broader coverage
# ─────────────────────────────────────────────────────────────────────────────

class TestRandomWalks:
    @pytest.mark.parametrize("seed", [1, 7, 42, 100, 999])
    @pytest.mark.parametrize("order", [3, 4, 5])
    def test_random_walk_high_pivots_match_reference(self, seed, order):
        rng = np.random.default_rng(seed)
        steps = rng.normal(0, 1, size=200)
        arr = 100 + np.cumsum(steps)
        assert _find_local_highs(arr, order=order) == _python_local_highs(arr, order=order)

    @pytest.mark.parametrize("seed", [1, 7, 42, 100, 999])
    @pytest.mark.parametrize("order", [3, 4, 5])
    def test_random_walk_low_pivots_match_reference(self, seed, order):
        rng = np.random.default_rng(seed)
        steps = rng.normal(0, 1, size=200)
        arr = 100 + np.cumsum(steps)
        assert _find_local_lows(arr, order=order) == _python_local_lows(arr, order=order)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic OHLC patterns — verify pivots align with the geometric features
# ─────────────────────────────────────────────────────────────────────────────

class TestSyntheticPatterns:
    def test_double_top_pivots(self):
        # Two equal peaks: rise to peak at 10, fall back, rise to peak at 30
        arr = np.zeros(40)
        arr[:11]   = np.linspace(0, 5, 11)   # peak 1 at idx 10
        arr[11:21] = np.linspace(5, 0, 10)
        arr[20:31] = np.linspace(0, 5, 11)   # peak 2 at idx 30
        arr[31:]   = np.linspace(5, 0, 9)
        py = _python_local_highs(arr, order=4)
        impl = _find_local_highs(arr, order=4)
        assert impl == py
        # Peaks should be near 10 and 30
        assert any(8 <= p <= 12 for p in impl)
        assert any(28 <= p <= 32 for p in impl)

    def test_head_and_shoulders_pivots(self):
        # Three peaks: 10 (left shoulder), 25 (head), 40 (right shoulder)
        arr = np.full(60, 0.0)
        arr[5:11]  = np.linspace(0, 3, 6)   # left shoulder rise
        arr[11:21] = np.linspace(3, 0, 10)
        arr[21:26] = np.linspace(0, 6, 5)   # head rise
        arr[26:36] = np.linspace(6, 0, 10)
        arr[36:41] = np.linspace(0, 3, 5)   # right shoulder rise
        arr[41:]   = np.linspace(3, 0, 19)
        py = _python_local_highs(arr, order=4)
        impl = _find_local_highs(arr, order=4)
        assert impl == py
