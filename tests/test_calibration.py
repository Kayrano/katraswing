"""
Tests for models.calibration — the isotonic confidence calibrator.

Covers:
  • Identity fallback below min_samples and on empty trade log
  • Monotonicity of the fitted transform
  • Boundary clamping below/above training range
  • Persistence round-trip
  • Brier-score sanity: calibrating systematically-overconfident
    predictions reduces Brier on training data
  • Refresh-every-N-trades behavior of the singleton
"""
from __future__ import annotations

import json
import random

import numpy as np
import pytest

from models.calibration import (
    IsotonicConfidenceCalibrator,
    DEFAULT_MIN_SAMPLES,
    get_calibrator,
    reset_singleton,
)


# ─────────────────────────────────────────────────────────────────────────────
# Identity fallbacks
# ─────────────────────────────────────────────────────────────────────────────

class TestIdentityFallback:
    def test_unfitted_calibrator_is_identity(self):
        cal = IsotonicConfidenceCalibrator()
        assert cal.is_fitted is False
        assert cal.transform(0.42) == pytest.approx(0.42)
        assert cal.transform(0.0)  == pytest.approx(0.0)
        assert cal.transform(1.0)  == pytest.approx(1.0)

    def test_fit_with_too_few_samples_returns_identity(self):
        cal = IsotonicConfidenceCalibrator.fit([0.7], [True])
        assert cal.is_fitted is False

    def test_from_trade_log_below_min_samples(self, tmp_path):
        log = tmp_path / "trades.json"
        log.write_text(json.dumps([
            {"strategy": "X", "confidence": 0.7, "outcome": "WIN"},
            {"strategy": "X", "confidence": 0.6, "outcome": "LOSS"},
        ]), encoding="utf-8")
        cal = IsotonicConfidenceCalibrator.from_trade_log(
            path=log, min_samples=10,
        )
        assert cal.is_fitted is False
        assert cal.sample_count == 2

    def test_from_trade_log_missing_file(self, tmp_path):
        cal = IsotonicConfidenceCalibrator.from_trade_log(
            path=tmp_path / "does_not_exist.json",
        )
        assert cal.is_fitted is False

    def test_mt5_import_trades_excluded(self, tmp_path):
        log = tmp_path / "trades.json"
        # 60 MT5_IMPORT trades (excluded) + 5 real trades (below min_samples)
        rows = [
            {"strategy": "MT5_IMPORT", "confidence": 0.0, "outcome": "WIN"}
            for _ in range(60)
        ] + [
            {"strategy": "TREND_MOM_5M", "confidence": 0.7, "outcome": "WIN"}
            for _ in range(5)
        ]
        log.write_text(json.dumps(rows), encoding="utf-8")
        cal = IsotonicConfidenceCalibrator.from_trade_log(
            path=log, min_samples=10,
        )
        # Only 5 eligible trades — below min_samples → identity
        assert cal.is_fitted is False
        assert cal.sample_count == 5


# ─────────────────────────────────────────────────────────────────────────────
# Monotonicity & boundary behavior
# ─────────────────────────────────────────────────────────────────────────────

class TestMonotonicity:
    def _generate_well_calibrated(self, n: int = 200, seed: int = 42):
        """Predictions ~uniform[0.5, 0.9]; outcome probability = prediction."""
        rng = random.Random(seed)
        preds = [rng.uniform(0.5, 0.9) for _ in range(n)]
        outcomes = [rng.random() < p for p in preds]
        return preds, outcomes

    def test_fit_produces_monotonic_transform(self):
        preds, outcomes = self._generate_well_calibrated()
        cal = IsotonicConfidenceCalibrator.fit(preds, outcomes)
        assert cal.is_fitted is True
        xs = np.linspace(0.0, 1.0, 50)
        ys = [cal.transform(x) for x in xs]
        for a, b in zip(ys, ys[1:]):
            assert b >= a - 1e-9, f"non-monotone: {a} → {b}"

    def test_below_range_clamps_to_first(self):
        cal = IsotonicConfidenceCalibrator.fit([0.5, 0.7, 0.9], [False, True, True])
        # Anything below 0.5 should map to the same value as 0.5
        assert cal.transform(0.0)  == pytest.approx(cal.transform(0.5))
        assert cal.transform(0.49) == pytest.approx(cal.transform(0.5))

    def test_above_range_clamps_to_last(self):
        cal = IsotonicConfidenceCalibrator.fit([0.5, 0.7, 0.9], [False, True, True])
        assert cal.transform(0.95) == pytest.approx(cal.transform(0.9))
        assert cal.transform(1.0)  == pytest.approx(cal.transform(0.9))

    def test_interpolation_between_breakpoints(self):
        # Three breakpoints with explicit fitted values
        cal = IsotonicConfidenceCalibrator(
            breakpoints=[0.4, 0.6, 0.8],
            fitted_values=[0.3, 0.5, 0.9],
            sample_count=100,
        )
        # At a midpoint between two breakpoints, value should interpolate
        mid = cal.transform(0.5)
        assert mid == pytest.approx(0.4)   # halfway between 0.3 and 0.5

    def test_fitted_values_are_in_unit_interval(self):
        preds, outcomes = self._generate_well_calibrated()
        cal = IsotonicConfidenceCalibrator.fit(preds, outcomes)
        for x in np.linspace(0.0, 1.0, 25):
            y = cal.transform(x)
            assert 0.0 <= y <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Brier-score sanity check — calibration should improve overconfident preds
# ─────────────────────────────────────────────────────────────────────────────

class TestBrierImprovement:
    def test_calibrating_overconfident_predictor_reduces_brier(self):
        """
        Predictor claims confidences in [0.70, 0.90] but the true win rate is
        a flat 0.55 regardless of prediction. After fitting, predictions
        should regress toward 0.55, producing a lower Brier score on the
        training set.
        """
        rng = random.Random(7)
        n = 300
        preds    = [rng.uniform(0.70, 0.90) for _ in range(n)]
        outcomes = [rng.random() < 0.55 for _ in range(n)]

        # Brier on raw (uncalibrated) predictions
        brier_raw = sum((p - int(o)) ** 2 for p, o in zip(preds, outcomes)) / n

        cal = IsotonicConfidenceCalibrator.fit(preds, outcomes)
        assert cal.is_fitted is True

        brier_cal = sum(
            (cal.transform(p) - int(o)) ** 2 for p, o in zip(preds, outcomes)
        ) / n

        assert brier_cal < brier_raw
        # Average calibrated prediction should be near the true rate (0.55)
        avg_cal = sum(cal.transform(p) for p in preds) / n
        assert avg_cal == pytest.approx(0.55, abs=0.05)


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

class TestPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        rng = random.Random(11)
        preds = [rng.uniform(0.5, 0.9) for _ in range(120)]
        outcomes = [rng.random() < p for p in preds]
        cal = IsotonicConfidenceCalibrator.fit(preds, outcomes)

        path = tmp_path / "calibration.json"
        cal.save(path)

        loaded = IsotonicConfidenceCalibrator.load(path)
        assert loaded.is_fitted == cal.is_fitted
        assert loaded.sample_count == cal.sample_count
        # Transform should be identical at sample points
        for x in np.linspace(0.5, 0.9, 12):
            assert loaded.transform(x) == pytest.approx(cal.transform(x))

    def test_load_missing_returns_identity(self, tmp_path):
        cal = IsotonicConfidenceCalibrator.load(tmp_path / "nope.json")
        assert cal.is_fitted is False

    def test_load_corrupt_returns_identity(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not json", encoding="utf-8")
        cal = IsotonicConfidenceCalibrator.load(bad)
        assert cal.is_fitted is False


# ─────────────────────────────────────────────────────────────────────────────
# Singleton refresh logic
# ─────────────────────────────────────────────────────────────────────────────

class TestSingletonRefresh:
    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path, monkeypatch):
        from models import calibration
        # Redirect both the trade log and the persisted calibration file so
        # the test doesn't read or write production data.
        monkeypatch.setattr(calibration, "_TRADE_LOG_PATH", tmp_path / "trades.json")
        monkeypatch.setattr(calibration, "_CALIB_PATH",     tmp_path / "calibration.json")
        reset_singleton()
        yield
        reset_singleton()

    def _write_varied_trades(self, path, n: int, seed: int = 1):
        """Write `n` synthetic trades with confidence spread across [0.55, 0.90]."""
        rng = random.Random(seed)
        rows = []
        for _ in range(n):
            c = rng.uniform(0.55, 0.90)
            outcome = "WIN" if rng.random() < c else "LOSS"
            rows.append({"strategy": "X", "confidence": round(c, 3), "outcome": outcome})
        path.write_text(json.dumps(rows), encoding="utf-8")

    def test_first_call_below_threshold_is_identity(self, tmp_path):
        from models import calibration
        self._write_varied_trades(calibration._TRADE_LOG_PATH, n=4)
        cal = get_calibrator(min_samples=50)
        assert cal.is_fitted is False
        assert cal.transform(0.7) == pytest.approx(0.7)

    def test_refits_after_refresh_every_new_trades(self, tmp_path):
        from models import calibration
        self._write_varied_trades(calibration._TRADE_LOG_PATH, n=60, seed=1)
        cal_a = get_calibrator(min_samples=20, refresh_every=10)
        assert cal_a.is_fitted is True

        # Grow the dataset past refresh_every — next call should refit
        self._write_varied_trades(calibration._TRADE_LOG_PATH, n=80, seed=2)
        cal_b = get_calibrator(min_samples=20, refresh_every=10)
        assert cal_b.sample_count > cal_a.sample_count

    def test_returns_cached_instance_when_unchanged(self, tmp_path):
        from models import calibration
        self._write_varied_trades(calibration._TRADE_LOG_PATH, n=60)
        first  = get_calibrator(min_samples=20, refresh_every=10)
        second = get_calibrator(min_samples=20, refresh_every=10)
        assert second is first


# ─────────────────────────────────────────────────────────────────────────────
# Integration with run_signal — calibration is plumbed through correctly
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalEngineIntegration:
    def test_signal_result_has_raw_confidence_field(self):
        from agents.signal_engine import SignalResult
        sr = SignalResult(ticker="X")
        assert hasattr(sr, "raw_confidence")
        assert hasattr(sr, "calibration_applied")
        assert sr.raw_confidence == 0.0
        assert sr.calibration_applied is False
