"""
Isotonic-regression confidence calibrator.

The 6-component confidence blend in agents.signal_engine.run_signal
(consensus + bt_adj + live_adj + news + pattern + MTF) has no
statistical grounding — a "0.75 confidence" signal can have any actual
win rate. This module fits a monotone mapping from raw confidence onto
empirical win probability using closed trades from data/trade_log.json.

Output: a piecewise-linear function so that, for example, raw 0.60
maps to whatever fraction of past 0.60-confidence trades actually won.

Falls back to identity transform until DEFAULT_MIN_SAMPLES (50) closed
trades exist — premature calibration on tiny samples overfits.
"""
from __future__ import annotations

import bisect
import json
import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_DATA_DIR        = Path(__file__).parent.parent / "data"
_TRADE_LOG_PATH  = _DATA_DIR / "trade_log.json"
_CALIB_PATH      = _DATA_DIR / "calibration.json"

DEFAULT_MIN_SAMPLES   = 50
DEFAULT_REFRESH_EVERY = 10   # refit after this many new closed trades


class IsotonicConfidenceCalibrator:
    """Piecewise-linear monotone mapping raw_confidence → calibrated_probability.

    Below the smallest training point, returns the smallest fitted value;
    above the largest, returns the largest. Linear interpolation between.
    """

    def __init__(
        self,
        breakpoints: Optional[Sequence[float]] = None,
        fitted_values: Optional[Sequence[float]] = None,
        sample_count: int = 0,
    ):
        self._x = list(breakpoints) if breakpoints else []
        self._y = list(fitted_values) if fitted_values else []
        self.sample_count = sample_count

    @property
    def is_fitted(self) -> bool:
        return len(self._x) >= 2

    def transform(self, confidence: float) -> float:
        """Map a raw confidence to the calibrated probability. Identity if unfit."""
        if not self.is_fitted:
            return float(confidence)
        if confidence <= self._x[0]:
            return self._y[0]
        if confidence >= self._x[-1]:
            return self._y[-1]
        i = bisect.bisect_right(self._x, confidence)
        x0, x1 = self._x[i - 1], self._x[i]
        y0, y1 = self._y[i - 1], self._y[i]
        if x1 == x0:
            return float(y0)
        t = (confidence - x0) / (x1 - x0)
        return float(y0 + t * (y1 - y0))

    @classmethod
    def fit(
        cls,
        predictions: Sequence[float],
        outcomes: Sequence[bool],
    ) -> "IsotonicConfidenceCalibrator":
        if len(predictions) != len(outcomes):
            raise ValueError("predictions and outcomes must be the same length")
        if len(predictions) < 2:
            return cls()

        from scipy.optimize import isotonic_regression
        x = np.asarray(predictions, dtype=float)
        y = np.asarray(outcomes, dtype=float)

        order   = np.argsort(x, kind="stable")
        x_sorted = x[order]
        y_sorted = y[order]

        result = isotonic_regression(y_sorted, increasing=True)
        fitted = np.asarray(result.x, dtype=float)

        # Compress: one breakpoint per unique x with averaged fitted value.
        unique_x, inverse = np.unique(x_sorted, return_inverse=True)
        unique_y = np.zeros_like(unique_x, dtype=float)
        for idx in range(len(unique_x)):
            mask = inverse == idx
            unique_y[idx] = float(fitted[mask].mean())

        return cls(
            breakpoints=unique_x.tolist(),
            fitted_values=unique_y.tolist(),
            sample_count=len(predictions),
        )

    def save(self, path: Optional[Path] = None) -> None:
        path = path or _CALIB_PATH
        try:
            path.write_text(
                json.dumps(
                    {
                        "breakpoints":   self._x,
                        "fitted_values": self._y,
                        "sample_count":  self.sample_count,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("calibration save failed: %s", exc)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "IsotonicConfidenceCalibrator":
        path = path or _CALIB_PATH
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                breakpoints=data.get("breakpoints"),
                fitted_values=data.get("fitted_values"),
                sample_count=int(data.get("sample_count", 0)),
            )
        except Exception as exc:
            logger.warning("calibration load failed: %s", exc)
            return cls()

    @classmethod
    def from_trade_log(
        cls,
        path: Optional[Path] = None,
        min_samples: int = DEFAULT_MIN_SAMPLES,
    ) -> "IsotonicConfidenceCalibrator":
        path = path or _TRADE_LOG_PATH
        if not path.exists():
            return cls()
        try:
            trades = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("from_trade_log: %s", exc)
            return cls()

        predictions: list[float] = []
        outcomes:    list[bool]  = []
        for t in trades:
            outcome    = t.get("outcome")
            confidence = t.get("confidence")
            # MT5_IMPORT trades have confidence=0 (not predicted by this engine);
            # including them would bias calibrator toward "low conf always loses".
            if t.get("strategy") == "MT5_IMPORT":
                continue
            if outcome not in ("WIN", "LOSS") or confidence is None:
                continue
            predictions.append(float(confidence))
            outcomes.append(outcome == "WIN")

        if len(predictions) < min_samples:
            logger.info(
                "Calibrator: %d closed trades < min_samples=%d — identity transform",
                len(predictions), min_samples,
            )
            return cls(sample_count=len(predictions))

        return cls.fit(predictions, outcomes)


# ── Module-level singleton with refresh logic ───────────────────────────────
_singleton: Optional[IsotonicConfidenceCalibrator] = None
_singleton_baseline_n: int = -1


def _count_eligible_trades(path: Optional[Path] = None) -> int:
    # Resolve path at call time so monkeypatched _TRADE_LOG_PATH is honoured
    if path is None:
        path = _TRADE_LOG_PATH
    try:
        if not path.exists():
            return 0
        trades = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    return sum(
        1
        for t in trades
        if t.get("outcome") in ("WIN", "LOSS")
        and t.get("strategy") != "MT5_IMPORT"
        and t.get("confidence") is not None
    )


def get_calibrator(
    min_samples: int = DEFAULT_MIN_SAMPLES,
    refresh_every: int = DEFAULT_REFRESH_EVERY,
) -> IsotonicConfidenceCalibrator:
    """Return the active calibrator, refitting when enough new trades have closed.

    Cheap to call on every signal — most calls return the cached instance.
    """
    global _singleton, _singleton_baseline_n
    current_n = _count_eligible_trades()

    if _singleton is None or (current_n - _singleton_baseline_n) >= refresh_every:
        _singleton = IsotonicConfidenceCalibrator.from_trade_log(min_samples=min_samples)
        _singleton_baseline_n = current_n
        if _singleton.is_fitted:
            logger.info("Calibrator refit on n=%d samples", current_n)
    return _singleton


def reset_singleton() -> None:
    """Clear the cached calibrator. Used by tests."""
    global _singleton, _singleton_baseline_n
    _singleton = None
    _singleton_baseline_n = -1
