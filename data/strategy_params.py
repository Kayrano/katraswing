"""
Adaptive strategy parameter store.

After each trade closes, adapt_all() analyses outcomes per strategy and adjusts:
  - sl_mult   : multiplier on the ATR-derived SL distance  (0.75 – 1.50)
  - tp_mult   : multiplier on the ATR-derived TP distance  (0.75 – 1.50)
  - conf_floor: minimum confidence before the signal is suppressed (0.60 – 0.80)
  - enabled   : False = strategy disabled until win-rate recovers

Parameters persist in data/strategy_params.json and survive app restarts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_PARAMS_FILE = Path(__file__).parent / "strategy_params.json"

_ALL_STRATEGIES = [
    "VWAP_RSI_5M", "ORB_5M", "TREND_MOM_5M", "EMA_PB_15M", "SQUEEZE_15M",
    "ABSORB_15M", "PDH_PDL_SWEEP_5M", "CAMARILLA_5M", "NR7_BREAKOUT_5M",
    "MSS_FOREX_15M", "BB_SCALP_5M", "STOCH_CROSS_5M", "EMA_MICRO_CROSS_5M",
]

_DEFAULT_ENTRY = {
    "sl_mult":      1.0,
    "tp_mult":      1.0,
    "conf_floor":   0.60,
    "enabled":      True,
    "trades_seen":  0,
    "wins":         0,
    "win_rate":     None,
    "last_adapted": None,
    "adapt_count":  0,
}

# Bounds for adaptive adjustments
_SL_MULT_MIN,  _SL_MULT_MAX  = 0.75, 1.50
_TP_MULT_MIN,  _TP_MULT_MAX  = 0.75, 1.50
_CONF_FLOOR_MIN, _CONF_FLOOR_MAX = 0.60, 0.80
_MIN_TRADES_TO_ADAPT  = 5    # need at least this many closed trades
_RECENT_WINDOW        = 20   # use last N trades for win-rate calculation
_DISABLE_THRESHOLD    = 0.35 # disable if win_rate < this with ≥15 trades
_DISABLE_MIN_TRADES   = 15
_REENABLE_THRESHOLD   = 0.50 # re-enable if win_rate recovers to this

# Module-level cache (loaded once, written on every change)
_PARAMS: dict[str, dict] = {}


# ── Load / save ───────────────────────────────────────────────────────────────

def _default_entry() -> dict:
    return {k: v for k, v in _DEFAULT_ENTRY.items()}


def load_params() -> dict:
    """Load from JSON; create defaults for missing strategies. Cached in _PARAMS."""
    global _PARAMS
    if _PARAMS_FILE.exists():
        try:
            raw = json.loads(_PARAMS_FILE.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                _PARAMS = raw
        except Exception as exc:
            logger.warning(f"strategy_params.json unreadable: {exc} — using defaults")
            _PARAMS = {}

    # Ensure all known strategies have entries with all required keys
    for strat in _ALL_STRATEGIES:
        if strat not in _PARAMS:
            _PARAMS[strat] = _default_entry()
        else:
            for k, v in _DEFAULT_ENTRY.items():
                _PARAMS[strat].setdefault(k, v)

    return _PARAMS


def save_params() -> None:
    """Persist _PARAMS to JSON."""
    try:
        _PARAMS_FILE.write_text(
            json.dumps(_PARAMS, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.error(f"Failed to save strategy_params.json: {exc}")


def get_params(strategy: str) -> dict:
    """Return params for a strategy. Auto-creates defaults on first call."""
    if not _PARAMS:
        load_params()
    if strategy not in _PARAMS:
        _PARAMS[strategy] = _default_entry()
        save_params()
    return _PARAMS[strategy]


def get_all_params() -> dict:
    """Return the full params dict (all strategies)."""
    if not _PARAMS:
        load_params()
    return _PARAMS


# ── Apply params to a signal ─────────────────────────────────────────────────

def apply_params(signal) -> object:
    """
    Post-process an IntradaySignal through the adaptive parameter layer.
    Returns a (possibly modified) IntradaySignal:
      - FLAT if strategy is disabled or confidence below adaptive floor
      - Adjusted SL/TP distances if sl_mult/tp_mult differ from 1.0
    """
    from agents.intraday_strategies import _flat  # local import — avoids circular

    if signal.signal == "FLAT":
        return signal

    params = get_params(signal.strategy)

    if not params.get("enabled", True):
        return _flat(signal.strategy, signal.timeframe,
                     f"{signal.strategy} disabled by adaptive learning (win_rate={params.get('win_rate', 'N/A')})")

    conf_floor = params.get("conf_floor", 0.60)
    if signal.confidence < conf_floor:
        return _flat(signal.strategy, signal.timeframe,
                     f"conf {signal.confidence:.2f} < adaptive floor {conf_floor:.2f}")

    sl_mult = params.get("sl_mult", 1.0)
    tp_mult = params.get("tp_mult", 1.0)

    if abs(sl_mult - 1.0) > 1e-6 or abs(tp_mult - 1.0) > 1e-6:
        sl_dist = abs(signal.entry - signal.stop_loss)  * sl_mult
        tp_dist = abs(signal.entry - signal.take_profit) * tp_mult
        if signal.signal == "LONG":
            new_sl = round(signal.entry - sl_dist, 5)
            new_tp = round(signal.entry + tp_dist, 5)
        else:
            new_sl = round(signal.entry + sl_dist, 5)
            new_tp = round(signal.entry - tp_dist, 5)
        signal = replace(signal, stop_loss=new_sl, take_profit=new_tp)

    return signal


# ── Adaptation logic ──────────────────────────────────────────────────────────

def adapt_strategy(strategy: str, all_trades: list[dict]) -> bool:
    """
    Analyse closed trades for this strategy and update params if needed.
    Returns True if any parameter changed.
    """
    if not _PARAMS:
        load_params()

    closed = [
        t for t in all_trades
        if t.get("strategy") == strategy and t.get("outcome") in ("WIN", "LOSS")
    ]

    params  = get_params(strategy)
    changed = False

    # Always sync trade count/win_rate stats even if not enough trades to adapt params
    if closed:
        recent_stats = closed[-_RECENT_WINDOW:]
        wins_stats   = [t for t in recent_stats if t["outcome"] == "WIN"]
        wr_stats     = len(wins_stats) / len(recent_stats)
        if params["trades_seen"] != len(closed) or params.get("win_rate") != round(wr_stats, 3):
            params["trades_seen"] = len(closed)
            params["wins"]        = len([t for t in closed if t["outcome"] == "WIN"])
            params["win_rate"]    = round(wr_stats, 3)
            changed = True
            save_params()

    if len(closed) < _MIN_TRADES_TO_ADAPT:
        return changed

    recent  = closed[-_RECENT_WINDOW:]
    wins    = [t for t in recent if t["outcome"] == "WIN"]
    win_rate = len(wins) / len(recent)

    # ── Confidence floor ──────────────────────────────────────────────────
    if win_rate < 0.40 and params["conf_floor"] < _CONF_FLOOR_MAX:
        params["conf_floor"] = round(min(_CONF_FLOOR_MAX, params["conf_floor"] + 0.02), 3)
        changed = True
    elif win_rate > 0.65 and params["conf_floor"] > _CONF_FLOOR_MIN:
        params["conf_floor"] = round(max(_CONF_FLOOR_MIN, params["conf_floor"] - 0.01), 3)
        changed = True

    # ── SL multiplier ─────────────────────────────────────────────────────
    if win_rate < 0.40 and params["sl_mult"] < _SL_MULT_MAX:
        params["sl_mult"] = round(min(_SL_MULT_MAX, params["sl_mult"] + 0.05), 3)
        changed = True
    elif win_rate > 0.65 and params["sl_mult"] > 1.0:
        params["sl_mult"] = round(max(1.0, params["sl_mult"] - 0.05), 3)
        changed = True

    # ── TP multiplier ─────────────────────────────────────────────────────
    if win_rate < 0.40 and params["tp_mult"] > _TP_MULT_MIN:
        params["tp_mult"] = round(max(_TP_MULT_MIN, params["tp_mult"] - 0.05), 3)
        changed = True
    elif win_rate > 0.65 and params["tp_mult"] < _TP_MULT_MAX:
        params["tp_mult"] = round(min(_TP_MULT_MAX, params["tp_mult"] + 0.05), 3)
        changed = True

    # ── Disable / re-enable ───────────────────────────────────────────────
    if params.get("enabled", True):
        if len(recent) >= _DISABLE_MIN_TRADES and win_rate < _DISABLE_THRESHOLD:
            params["enabled"] = False
            changed = True
            logger.warning(f"Adaptive learning DISABLED {strategy} — win_rate={win_rate:.1%}")
    else:
        if win_rate >= _REENABLE_THRESHOLD and len(recent) >= 10:
            params["enabled"] = True
            changed = True
            logger.info(f"Adaptive learning RE-ENABLED {strategy} — win_rate={win_rate:.1%}")

    if changed:
        params["last_adapted"] = datetime.now(tz=timezone.utc).isoformat()
        params["adapt_count"]  = params.get("adapt_count", 0) + 1
        save_params()
        logger.info(
            f"Adaptive: {strategy} win={win_rate:.1%} "
            f"sl×{params['sl_mult']} tp×{params['tp_mult']} "
            f"floor={params['conf_floor']} enabled={params['enabled']}"
        )

    return changed


def adapt_all(all_trades: list[dict]) -> int:
    """
    Run adapt_strategy() for every app-managed strategy that has closed trades.
    Ignores MT5_IMPORT trades (manually opened or imported from MT5 history).
    Returns the count of strategies updated.
    """
    if not _PARAMS:
        load_params()

    # Only learn from trades the app itself sent — exclude MT5_IMPORT
    app_trades = [t for t in all_trades if t.get("strategy", "") != "MT5_IMPORT"]

    updated = 0
    strategies_in_log = {t.get("strategy") for t in app_trades if t.get("strategy")}

    for strategy in strategies_in_log:
        try:
            if adapt_strategy(strategy, app_trades):
                updated += 1
        except Exception as exc:
            logger.error(f"adapt_strategy({strategy}): {exc}")

    return updated


# Auto-load on import
load_params()
