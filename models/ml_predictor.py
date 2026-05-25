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

# Phase 6: per-strategy sub-models activate once a strategy has enough
# closed trades to fit a meaningful tree without overfitting.
_SUB_MODEL_MIN_TRADES = 30

# Phase 7: candidate thresholds swept on the last test fold. The threshold
# that maximises sum(pnl_per_R for trades the gate would have ADMITTED)
# becomes the operational floor.
_THRESHOLD_GRID = [0.25, 0.30, 0.33, 0.35, 0.40, 0.45, 0.50, 0.55]

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
    is_paper: bool | None = None,
    # ── Phase 5: real-time edge-decay features ────────────────────────────
    recent_strategy_wr: float | None = None,
    recent_symbol_wr: float | None = None,
    recent_mfe_r_avg: float | None = None,
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
        # Execution regime: 1 = paper (no spread/slippage/commission),
        # 0 = live broker fill. Lets the model learn that the same signal
        # pattern resolves differently under real execution costs — paper
        # tends to hit TP cleaner because bar-data resolution is exact.
        # At inference we default to 0 (live).
        "is_paper": 1 if is_paper else 0,
        # Phase 5: recent-edge features (computed from trade log at signal
        # time; from PRIOR trades only at training time to avoid leakage).
        # Defaults: 0.5 WR (neutral), 0.0 mfe (no excursion data yet).
        "recent_strategy_wr": recent_strategy_wr if recent_strategy_wr is not None else 0.5,
        "recent_symbol_wr":   recent_symbol_wr   if recent_symbol_wr   is not None else 0.5,
        "recent_mfe_r_avg":   recent_mfe_r_avg   if recent_mfe_r_avg   is not None else 0.5,
    }
    for s in _KNOWN_STRATEGIES:
        feats[f"strat_{s}"] = int(strategy == s)

    return feats


def _feats_to_array(feats: dict, feature_names: list[str]) -> np.ndarray:
    return np.array([feats.get(k, 0.0) for k in feature_names], dtype=float)


# ── Phase 5: recent-edge feature helpers ──────────────────────────────────

def _recent_wr_at(prior_trades: list[dict], strategy: str, symbol: str, k: int = 10) -> tuple[float, float, float]:
    """Compute the most recent (strategy_wr, symbol_wr, avg_mfe_r) using only
    `prior_trades` (already chronologically filtered to be before the current
    trade). Returns (0.5, 0.5, 0.5) defaults when no prior data exists.

    `k` = how many most-recent prior trades to look at. Smaller k = more
    responsive to recent regime change; larger k = less noisy.
    """
    norm_sym = symbol.replace("=X", "").replace("=F", "").upper() if symbol else ""

    def _wr(rows: list[dict]) -> float:
        if not rows:
            return 0.5   # neutral default
        wins = sum(1 for r in rows if r.get("outcome") == "WIN")
        return wins / len(rows)

    # Strategy-specific recent k
    by_strat = [r for r in prior_trades
                if r.get("strategy") == strategy
                and r.get("outcome") in ("WIN", "LOSS")][-k:]
    # Symbol-specific recent k
    by_sym = [r for r in prior_trades
              if (r.get("ticker", "").replace("=X", "").replace("=F", "").upper()) == norm_sym
              and r.get("outcome") in ("WIN", "LOSS")][-k:]

    strat_wr = _wr(by_strat)
    sym_wr   = _wr(by_sym)

    # Average MFE-in-R for the same strategy's last k trades (Phase 4 data).
    # Captures "this strategy has been getting close to TP lately" vs not.
    mfes = [float(r["mfe_r"]) for r in by_strat if r.get("mfe_r") is not None]
    mfe_avg = sum(mfes) / len(mfes) if mfes else 0.5

    return strat_wr, sym_wr, mfe_avg


class WinRatePredictor:
    """LightGBM gradient-boosted tree wrapper with train / predict / save / load."""

    def __init__(self) -> None:
        self.model = None
        self.feature_names: list[str] = []
        self.n_samples: int = 0
        self.cv_score: float | None = None
        self.trained_at: str | None = None
        self.feature_importances_: dict[str, float] = {}
        # Phase 6: per-strategy sub-models. Keys are strategy names; values
        # are LGBMClassifier instances fitted only on that strategy's trades.
        # Activated for strategies with >= _SUB_MODEL_MIN_TRADES samples.
        self.sub_models: dict[str, object] = {}
        # Phase 7: P&L-optimal threshold derived from the last held-out test
        # fold. Falls back to 0.33 (signal_engine default) when None.
        self.optimal_threshold: float | None = None

    @property
    def is_fitted(self) -> bool:
        return self.model is not None and bool(self.feature_names)

    def train(self, trades: list[dict]) -> bool:
        """
        Fit on closed automated trades.  Returns True on success.
        Excludes MT5_IMPORT (manual trades) and open trades.

        Phase 2: Cross-validation uses TimeSeriesSplit instead of random k-fold.
        Each test fold contains only trades chronologically after every trade
        in its train fold — no future-data leakage, AUC is honest.

        Phase 3: paper trades are kept in the training set with `is_paper`
        as a feature, so the tree can split on the execution regime. Once
        ≥30 live trades accumulate the model can learn live-specific
        patterns in the is_paper=0 leaves without losing the bulk of the
        sample size today.
        """
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            logger.warning("ml_predictor: lightgbm not installed — run: pip install lightgbm>=4.0")
            return False

        from sklearn.model_selection import TimeSeriesSplit

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

        # Chronological sort: time-series CV requires the data to be ordered
        # by close time. Use closed_at if present (more accurate); fall back
        # to sent_at for any legacy row. Open trades were filtered above.
        def _sort_key(t: dict) -> str:
            return t.get("closed_at") or t.get("sent_at") or ""
        closed.sort(key=_sort_key)

        # Build rows with prior-only recent-WR (no future-data leakage)
        rows: list[dict] = []
        for i, t in enumerate(closed):
            # Prior trades = chronologically older entries in `closed`
            # (since we already sorted by closed_at above). Pre-computing
            # in a list comprehension is fine — n is bounded by training set.
            strat_wr, sym_wr, mfe_avg = _recent_wr_at(
                closed[:i], t["strategy"], t.get("ticker", ""), k=10,
            )
            rows.append(extract_features(
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
                is_paper=bool(t.get("paper_only")),
                recent_strategy_wr=strat_wr,
                recent_symbol_wr=sym_wr,
                recent_mfe_r_avg=mfe_avg,
            ))

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

        # TimeSeriesSplit: each test fold contains only trades chronologically
        # after every trade in its train fold. The previous KFold leaked
        # future data into training and inflated CV AUC.
        n_splits = min(5, max(2, len(closed) // 20))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_aucs: list[float] = []
        from sklearn.metrics import roc_auc_score
        for train_idx, test_idx in tscv.split(X):
            # Both classes must be present in train AND test for a usable AUC
            if len(set(y[train_idx])) < 2 or len(set(y[test_idx])) < 2:
                continue
            clf_fold = LGBMClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=4,
                min_child_samples=5, class_weight="balanced", verbosity=-1,
            )
            clf_fold.fit(X[train_idx], y[train_idx])
            preds = clf_fold.predict_proba(X[test_idx])[:, 1]
            fold_aucs.append(float(roc_auc_score(y[test_idx], preds)))
        self.cv_score = float(np.mean(fold_aucs)) if fold_aucs else 0.5
        logger.info(
            "ml_predictor: TimeSeries CV AUC=%.3f (n=%d, folds=%d, LightGBM)",
            self.cv_score, len(closed), len(fold_aucs),
        )

        clf.fit(X, y)
        self.model = clf
        self.n_samples = len(closed)
        self.trained_at = datetime.now(tz=timezone.utc).isoformat()

        # Store feature importances for nightly reporting
        if hasattr(clf, "feature_importances_"):
            self.feature_importances_ = dict(zip(feature_names, clf.feature_importances_))

        # ── Phase 7: P&L-optimal threshold from the last test fold ────────
        # Sweep candidate thresholds against the most recent held-out fold;
        # pick the one maximising expected R-multiple. Falls back to None
        # (signal_engine then uses its hardcoded 0.33) if no usable fold.
        self.optimal_threshold = self._compute_optimal_threshold(
            clf, X, y, n_splits, closed,
        )
        if self.optimal_threshold is not None:
            logger.info(
                "ml_predictor: optimal threshold = %.2f (vs default 0.33)",
                self.optimal_threshold,
            )

        # ── Phase 6: per-strategy sub-models ──────────────────────────────
        # Fit a tiny LGBM per strategy with ≥ _SUB_MODEL_MIN_TRADES trades.
        # Predictions are blended 50/50 with the global model at inference.
        self.sub_models = self._fit_sub_models(clf, X, y, closed, feature_names)
        if self.sub_models:
            logger.info("ml_predictor: %d per-strategy sub-models fitted",
                        len(self.sub_models))

        return True

    def _compute_optimal_threshold(
        self,
        clf,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int,
        closed: list[dict],
    ) -> float | None:
        """Sweep _THRESHOLD_GRID on the last TimeSeriesSplit test fold; pick
        the threshold that maximises sum of R-multiples over admitted trades.

        R-multiple per trade: profit / risk if WIN, -1.0 if LOSS (full SL).
        Approximates expected utility — a low-WR strategy at 3R is admitted
        over a high-WR strategy at 0.5R.
        """
        from sklearn.model_selection import TimeSeriesSplit
        try:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            splits = list(tscv.split(X))
            if not splits:
                return None
            train_idx, test_idx = splits[-1]   # most recent fold
            if len(set(y[train_idx])) < 2 or len(set(y[test_idx])) < 2:
                return None
            from lightgbm import LGBMClassifier
            clf_eval = LGBMClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=4,
                min_child_samples=5, class_weight="balanced", verbosity=-1,
            )
            clf_eval.fit(X[train_idx], y[train_idx])
            preds = clf_eval.predict_proba(X[test_idx])[:, 1]
            # Per-trade R-multiple from the test fold's underlying trade rows
            test_trades = [closed[i] for i in test_idx]
            r_mults: list[float] = []
            for t in test_trades:
                entry, sl, tp = float(t["entry"]), float(t["sl"]), float(t["tp"])
                risk = abs(entry - sl)
                if risk <= 0:
                    r_mults.append(0.0)
                    continue
                if t["outcome"] == "WIN":
                    r_mults.append(abs(tp - entry) / risk)
                else:
                    r_mults.append(-1.0)
            best_thr, best_sum = None, -float("inf")
            for thr in _THRESHOLD_GRID:
                admitted_r = sum(r for r, p in zip(r_mults, preds) if p >= thr)
                if admitted_r > best_sum:
                    best_sum = admitted_r
                    best_thr = thr
            return best_thr
        except Exception as exc:
            logger.warning("optimal_threshold sweep failed: %s", exc)
            return None

    def _fit_sub_models(
        self,
        global_clf,
        X: np.ndarray,
        y: np.ndarray,
        closed: list[dict],
        feature_names: list[str],
    ) -> dict[str, object]:
        """Fit a per-strategy LGBM on rows where strategy matches. Skip
        strategies with < _SUB_MODEL_MIN_TRADES samples — global handles them.
        """
        from collections import defaultdict
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            return {}
        idx_by_strat: dict[str, list[int]] = defaultdict(list)
        for i, t in enumerate(closed):
            idx_by_strat[t["strategy"]].append(i)
        sub_models: dict[str, object] = {}
        for strat, idx in idx_by_strat.items():
            if len(idx) < _SUB_MODEL_MIN_TRADES:
                continue
            y_sub = y[idx]
            if len(set(y_sub)) < 2:
                continue   # degenerate (all wins or all losses)
            try:
                clf_sub = LGBMClassifier(
                    n_estimators=100, learning_rate=0.05, max_depth=3,
                    min_child_samples=3, class_weight="balanced", verbosity=-1,
                )
                clf_sub.fit(X[idx], y_sub)
                sub_models[strat] = clf_sub
            except Exception as exc:
                logger.warning("sub-model fit failed for %s: %s", strat, exc)
        return sub_models

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
        is_paper: bool = False,
        ticker: str = "",
    ) -> float | None:
        """Return predicted win probability, or None if model not fitted.

        Defaults `is_paper=False` because the gate runs on live signals.
        At training time paper trades carry is_paper=1 — trees can split on
        the regime so live predictions aren't contaminated by paper-execution
        patterns even though paper rows are in the training set.

        Phase 5: recent_strategy_wr / recent_symbol_wr / recent_mfe_r_avg are
        computed from the trade log at predict time. `ticker` is required
        for the symbol-WR computation; falls back to neutral defaults when
        omitted.

        Phase 6: If a per-strategy sub-model exists, the returned probability
        is the 50/50 blend of the global and strategy-specific models.
        """
        if not self.is_fitted:
            return None
        # Compute Phase 5 features from current trade log
        strat_wr, sym_wr, mfe_avg = 0.5, 0.5, 0.5
        try:
            from data.trade_outcomes import _load as _load_trades
            prior = [t for t in _load_trades()
                     if t.get("outcome") in ("WIN", "LOSS")
                     and t.get("strategy") != "MT5_IMPORT"]
            strat_wr, sym_wr, mfe_avg = _recent_wr_at(prior, strategy, ticker, k=10)
        except Exception:
            pass

        feats = extract_features(
            strategy, direction, entry, sl, tp, confidence, sent_at,
            adx_value=adx_value, atr_value=atr_value,
            spread_pips=spread_pips, h1_trend=h1_trend,
            vol_ratio=vol_ratio, consensus_count=consensus_count,
            pattern_boost_val=pattern_boost_val, calibrated_conf=calibrated_conf,
            is_paper=is_paper,
            recent_strategy_wr=strat_wr,
            recent_symbol_wr=sym_wr,
            recent_mfe_r_avg=mfe_avg,
        )
        x = _feats_to_array(feats, self.feature_names).reshape(1, -1)
        try:
            global_p = float(self.model.predict_proba(x)[0, 1])
            # Phase 6: blend with per-strategy sub-model when available.
            # `getattr` for forward compat — pickles saved before Phase 6
            # don't have a sub_models attribute on the unpickled instance.
            sub_models = getattr(self, "sub_models", None) or {}
            sub = sub_models.get(strategy)
            if sub is None:
                return global_p
            try:
                sub_p = float(sub.predict_proba(x)[0, 1])
                return 0.5 * global_p + 0.5 * sub_p
            except Exception:
                return global_p
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
