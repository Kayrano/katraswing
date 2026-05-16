"""
ML-based win probability predictor for trade signals.

Trains a LightGBM gradient-boosted tree on closed trade history to estimate
the probability a new signal will be a winner.  Falls back gracefully (returns
None) when fewer than MIN_SAMPLES closed trades exist.

Features derived at signal-evaluation time (all available before the trade
is placed):
    strategy_*         one-hot encoded strategy name
    is_long            1 = LONG, 0 = SHORT
    hour               UTC hour of signal (0–23)
    day_of_week        0 = Monday … 6 = Sunday
    session_*          one-hot: london / ny / asia / other
    sl_dist_pct        |entry − sl| / entry  (ATR proxy as fraction)
    designed_rr        |tp − entry| / |entry − sl|
    confidence         raw blended confidence at signal time
    vol_ratio          current ATR / 20-bar avg ATR
    consensus_count    number of strategies agreeing on direction
    pattern_boost_val  pattern alignment boost (±0.05) applied at entry
    calibrated_conf    isotonic-calibrated empirical win probability
"""
from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_MODEL_PATH = Path(__file__).parent.parent / "data" / "ml_predictor.pkl"
_MIN_SAMPLES = 30   # LightGBM with shallow trees handles smaller datasets than logistic regression
_MIN_WIN_RATE = 0.10  # sanity check — don't fit on degenerate data

# Known strategies for stable one-hot encoding across retrains
_KNOWN_STRATEGIES = [
    "VWAP_RSI_5M", "ORB_5M", "TREND_MOM_5M", "CAMARILLA_5M",
    "NR7_BREAKOUT_5M", "MSS_FOREX_15M", "BB_SCALP_5M", "STOCH_CROSS_5M",
    "EMA_MICRO_CROSS_5M", "PDH_PDL_SWEEP_5M", "EMA_PB_15M", "SQUEEZE_15M",
    "ABSORB_15M", "MSS_H1", "ORB_H1", "EMA_PB_H1", "LONDON_BREAKOUT_H1",
    "ABSORB_BO", "TRIPLE_A", "VA_BOUNCE",
]


def _session(hour: int) -> str:
    if 7 <= hour < 12:
        return "london"
    if 12 <= hour < 17:
        return "ny"
    if 0 <= hour < 4:
        return "asia"
    return "other"


def extract_features(
    strategy: str,
    direction: str,
    entry: float,
    sl: float,
    tp: float,
    confidence: float,
    sent_at: str | datetime | None = None,
    adx_value: float | None = None,
    atr_value: float | None = None,
    spread_pips: float | None = None,
    h1_trend: str | None = None,
    session: str | None = None,
    day_of_week: int | None = None,
    vol_ratio: float | None = None,
    consensus_count: int | None = None,
    pattern_boost_val: float | None = None,
    calibrated_conf: float | None = None,
) -> dict:
    """Return a flat feature dict for one signal."""
    if isinstance(sent_at, str):
        try:
            sent_at = datetime.fromisoformat(sent_at)
        except Exception:
            sent_at = None
    if sent_at is None:
        sent_at = datetime.now(tz=timezone.utc)

    hour = sent_at.hour
    dow = day_of_week if day_of_week is not None else sent_at.weekday()
    sess = session if session is not None else _session(hour)

    sl_dist = abs(entry - sl)
    tp_dist = abs(tp - entry)
    sl_dist_pct = sl_dist / entry if entry else 0.0
    designed_rr = tp_dist / sl_dist if sl_dist > 0 else 1.0

    # Trend alignment: 1 = aligned, 0 = neutral, -1 = counter-trend
    if h1_trend == "BULLISH":
        trend_align = 1 if direction == "LONG" else -1
    elif h1_trend == "BEARISH":
        trend_align = -1 if direction == "LONG" else 1
    else:
        trend_align = 0

    feats: dict = {
        "is_long": 1 if direction == "LONG" else 0,
        "hour": hour,
        "day_of_week": dow,
        "session_london": int(sess == "london"),
        "session_ny": int(sess == "ny"),
        "session_asia": int(sess == "asia"),
        "sl_dist_pct": sl_dist_pct,
        "designed_rr": min(designed_rr, 10.0),
        "confidence": confidence,
        # Enriched features — filled with neutral defaults when not yet logged
        "adx_value": adx_value if adx_value is not None else 25.0,
        "atr_value": atr_value if atr_value is not None else sl_dist_pct,
        "spread_pips": spread_pips if spread_pips is not None else 1.0,
        "trend_align": trend_align,
        # Signal quality features (new Phase 2)
        "vol_ratio": vol_ratio if vol_ratio is not None else 1.0,
        "consensus_count": float(consensus_count) if consensus_count is not None else 1.0,
        "pattern_boost_val": pattern_boost_val if pattern_boost_val is not None else 0.0,
        "calibrated_conf": calibrated_conf if calibrated_conf is not None else confidence,
    }
    for s in _KNOWN_STRATEGIES:
        feats[f"strat_{s}"] = int(strategy == s)

    return feats


def _feats_to_array(feats: dict, feature_names: list[str]) -> np.ndarray:
    return np.array([feats.get(k, 0.0) for k in feature_names], dtype=float)


class WinRatePredictor:
    """LightGBM gradient-boosted tree wrapper with train / predict / save / load."""

    def __init__(self) -> None:
        self.model = None
        self.feature_names: list[str] = []
        self.n_samples: int = 0
        self.cv_score: float | None = None
        self.trained_at: str | None = None
        self.feature_importances_: dict[str, float] = {}

    @property
    def is_fitted(self) -> bool:
        return self.model is not None and bool(self.feature_names)

    def train(self, trades: list[dict]) -> bool:
        """
        Fit on closed automated trades.  Returns True on success.
        Excludes MT5_IMPORT (manual trades) and open trades.
        """
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            logger.warning("ml_predictor: lightgbm not installed — run: pip install lightgbm>=4.0")
            return False

        from sklearn.model_selection import cross_val_score

        closed = [
            t for t in trades
            if t.get("strategy", "") != "MT5_IMPORT"
            and t.get("outcome") in ("WIN", "LOSS")
            and t.get("entry") and t.get("sl") and t.get("tp")
        ]

        if len(closed) < _MIN_SAMPLES:
            logger.info("ml_predictor: only %d samples, need %d — skipping fit",
                        len(closed), _MIN_SAMPLES)
            return False

        wins = sum(1 for t in closed if t["outcome"] == "WIN")
        if wins / len(closed) < _MIN_WIN_RATE:
            logger.warning("ml_predictor: degenerate win rate %.1f%% — skipping fit",
                           100 * wins / len(closed))
            return False

        rows = [
            extract_features(
                strategy=t["strategy"],
                direction=t.get("direction", "LONG"),
                entry=t["entry"],
                sl=t["sl"],
                tp=t["tp"],
                confidence=t.get("confidence", 0.7),
                sent_at=t.get("sent_at"),
                adx_value=t.get("adx_value"),
                atr_value=t.get("atr_value"),
                spread_pips=t.get("spread_pips"),
                h1_trend=t.get("h1_trend"),
                session=t.get("session"),
                day_of_week=t.get("day_of_week"),
                vol_ratio=t.get("vol_ratio"),
                consensus_count=t.get("consensus_count"),
                pattern_boost_val=t.get("pattern_boost_val"),
                calibrated_conf=t.get("calibrated_conf"),
            )
            for t in closed
        ]

        feature_names = list(rows[0].keys())
        self.feature_names = feature_names

        X = np.array([_feats_to_array(r, self.feature_names) for r in rows])
        y = np.array([1 if t["outcome"] == "WIN" else 0 for t in closed])

        # Shallow trees prevent overfit on small datasets; min_child_samples=5
        # ensures at least 5 trades per leaf node.
        clf = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            min_child_samples=5,
            class_weight="balanced",
            verbosity=-1,
        )

        cv_folds = min(5, max(2, len(closed) // 8))
        cv_scores = cross_val_score(clf, X, y, cv=cv_folds, scoring="roc_auc")
        self.cv_score = float(np.mean(cv_scores))
        logger.info("ml_predictor: CV AUC=%.3f (n=%d, LightGBM)", self.cv_score, len(closed))

        clf.fit(X, y)
        self.model = clf
        self.n_samples = len(closed)
        self.trained_at = datetime.now(tz=timezone.utc).isoformat()

        # Store feature importances for nightly reporting
        if hasattr(clf, "feature_importances_"):
            self.feature_importances_ = dict(zip(feature_names, clf.feature_importances_))

        return True

    def predict_proba(
        self,
        strategy: str,
        direction: str,
        entry: float,
        sl: float,
        tp: float,
        confidence: float,
        sent_at: str | datetime | None = None,
        adx_value: float | None = None,
        atr_value: float | None = None,
        spread_pips: float | None = None,
        h1_trend: str | None = None,
        vol_ratio: float | None = None,
        consensus_count: int | None = None,
        pattern_boost_val: float | None = None,
        calibrated_conf: float | None = None,
    ) -> float | None:
        """Return predicted win probability, or None if model not fitted."""
        if not self.is_fitted:
            return None
        feats = extract_features(
            strategy, direction, entry, sl, tp, confidence, sent_at,
            adx_value=adx_value, atr_value=atr_value,
            spread_pips=spread_pips, h1_trend=h1_trend,
            vol_ratio=vol_ratio, consensus_count=consensus_count,
            pattern_boost_val=pattern_boost_val, calibrated_conf=calibrated_conf,
        )
        x = _feats_to_array(feats, self.feature_names).reshape(1, -1)
        try:
            proba = self.model.predict_proba(x)
            # LGBMClassifier returns shape (1, 2); col 1 = win probability
            return float(proba[0, 1])
        except Exception as exc:
            logger.warning("ml_predictor: predict failed: %s", exc)
            return None

    def save(self, path: Path = _MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("ml_predictor: saved to %s (n=%d, AUC=%.3f)",
                    path, self.n_samples, self.cv_score or 0)

    @classmethod
    def load(cls, path: Path = _MODEL_PATH) -> "WinRatePredictor":
        if path.exists():
            try:
                with open(path, "rb") as f:
                    obj = pickle.load(f)
                if isinstance(obj, cls) and obj.is_fitted:
                    return obj
            except Exception as exc:
                logger.warning("ml_predictor: failed to load %s: %s", path, exc)
        return cls()


# Module-level singleton — loaded once, reloaded after weekly retrain
_predictor: WinRatePredictor | None = None


def get_predictor() -> WinRatePredictor:
    global _predictor
    if _predictor is None:
        _predictor = WinRatePredictor.load()
    return _predictor


def reload_predictor() -> WinRatePredictor:
    global _predictor
    _predictor = WinRatePredictor.load()
    return _predictor


_TRAINING_WINDOW_DAYS = 90   # prefer recent trades; fall back to full history if too few


def _parse_dt(s: str | None) -> datetime:
    """Parse an ISO-format datetime string; return epoch on failure."""
    if not s:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)


def retrain_from_log(trade_log_path: Path | None = None) -> bool:
    """Load trade_log.json, retrain on 90-day rolling window, save.

    Falls back to full history when the rolling window has fewer than
    _MIN_SAMPLES closed trades so the model stays fitted during slow periods.
    Returns True on success.
    """
    if trade_log_path is None:
        trade_log_path = Path(__file__).parent.parent / "data" / "trade_log.json"
    try:
        with open(trade_log_path) as f:
            trades = json.load(f)
    except Exception as exc:
        logger.error("ml_predictor: cannot load trade log: %s", exc)
        return False

    cutoff = datetime.now(timezone.utc) - timedelta(days=_TRAINING_WINDOW_DAYS)
    windowed = [
        t for t in trades
        if t.get("outcome") in ("WIN", "LOSS")
        and t.get("strategy") != "MT5_IMPORT"
        and _parse_dt(t.get("closed_at")) >= cutoff
    ]

    if len(windowed) >= _MIN_SAMPLES:
        logger.info("ml_predictor: using 90d window, n=%d", len(windowed))
        training_set = trades   # train() filters internally; pass full list but windowed will dominate
        # Override: rebuild trades to only include windowed closed + all open (for completeness)
        training_set = windowed
    else:
        logger.info("ml_predictor: 90d window only %d samples, falling back to full history", len(windowed))
        training_set = trades

    predictor = WinRatePredictor()
    if predictor.train(training_set):
        predictor.save()
        reload_predictor()
        return True
    return False
