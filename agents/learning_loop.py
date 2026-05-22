"""Multi-cadence self-learning scheduler.

Tick `tick()` once per server poll iteration. The function is O(microseconds)
when nothing is due (one JSON read + three timestamp compares). When a job
*is* due:

  - hourly   : runs synchronously in the calling thread (target <5s)
  - daily    : spawned in a daemon thread (may take minutes; report-only)
  - nightly  : spawned in a daemon thread (applies prune/promote per the
               user's 2026-05-01 decision; fires every night at 23:00 UTC)

State persists in `data/learning_state.json`. Every fire appends one JSONL
row to `data/learning_log.jsonl` for audit.

Crash-safety:
- atomic writes via tempfile + os.replace
- stored timestamp in the future (clock skew / sleep resume) → treat as
  None and fire
- per-kind threading.Lock prevents double-fire if `tick()` is invoked
  concurrently
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DATA_DIR    = Path(__file__).resolve().parent.parent / "data"
_STATE_PATH  = _DATA_DIR / "learning_state.json"
_AUDIT_PATH  = _DATA_DIR / "learning_log.jsonl"
_REPORTS_DIR = _DATA_DIR / "reports"

# Watchlist set by the CLI on first tick(); used by daily/nightly sweeps.
# Module-level so threads share the same value without lock juggling
# (write-once on startup, read-only afterwards).
_WATCHLIST: list[str] = []

# Optional Telegram notifier — set by the server via set_notifier()
_TG = None

# Per-kind locks so concurrent tick() calls don't fire the same job twice.
_LOCKS: dict[str, threading.Lock] = {
    "hourly": threading.Lock(),
    "daily":  threading.Lock(),
    "nightly": threading.Lock(),
}


# ── Clock — single helper so tests can monkeypatch ──────────────────────────

def _now() -> datetime:
    """All wall-clock reads in this module go through here."""
    return datetime.now(timezone.utc)


# ── State persistence ──────────────────────────────────────────────────────

def _load_state() -> dict:
    """Read learning_state.json. Returns empty dict on first run / corrupt file.
    Stored timestamps in the future are treated as missing (clock-skew safe)."""
    if not _STATE_PATH.exists():
        return {}
    try:
        with open(_STATE_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("ctx=load_state corrupt: %s; starting fresh", exc)
        return {}
    state: dict = {}
    now = _now()
    for kind in ("hourly", "daily", "nightly"):
        key = f"last_{kind}_at"
        ts_str = raw.get(key)
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            continue
        if ts > now:
            # Future timestamp → ignore (treat as not-yet-run).
            logger.warning("ctx=load_state %s in future (%s); ignoring", key, ts)
            continue
        state[key] = ts
    return state


def _save_state(state: dict) -> None:
    """Atomic write: tempfile + os.replace. Never leaves a half-written file."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    serializable = {
        k: (v.isoformat() if isinstance(v, datetime) else v)
        for k, v in state.items()
    }
    fd, tmp = tempfile.mkstemp(dir=str(_DATA_DIR), prefix=".learning_state.", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, sort_keys=True)
        os.replace(tmp, _STATE_PATH)
    except Exception:
        # Clean up the temp file if replace failed.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ── Audit log ──────────────────────────────────────────────────────────────

def _audit(row: dict) -> None:
    """Append one JSONL row to learning_log.jsonl. Best-effort: never raises."""
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(_AUDIT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
    except OSError as exc:
        logger.warning("ctx=audit_log_write: %s", exc)


# ── Cadence rules ──────────────────────────────────────────────────────────

def _is_due(kind: str, now: datetime, state: dict) -> bool:
    """Return True if the cadence window is open and we haven't fired in it yet.

    Hourly : now.minute < 5 and last_hourly_at is in a previous hour bucket.
    Daily  : now.hour == 23 and last_daily_at.date() < now.date().
    Nightly: every night at 23:00 UTC — prune/promote/ML-retrain.
    """
    last: Optional[datetime] = state.get(f"last_{kind}_at")

    if kind == "hourly":
        if now.minute >= 5:
            return False
        if last is None:
            return True
        # Previous hour bucket = different (year, month, day, hour).
        return (last.year, last.month, last.day, last.hour) != \
               (now.year, now.month, now.day, now.hour)

    if kind == "daily":
        if now.hour < 23:
            return False
        if last is None:
            return True
        return last.date() < now.date()

    if kind == "nightly":
        if now.hour < 23:
            return False
        if last is None:
            return True
        return last.date() < now.date()

    return False


# ── Main entry point ──────────────────────────────────────────────────────

def set_watchlist(tickers: list[str]) -> None:
    """Called once by the CLI on startup so daily/nightly sweeps know which
    tickers to backtest. Idempotent — safe to call from every tick if cheaper
    than threading args through."""
    global _WATCHLIST
    _WATCHLIST = list(tickers or [])


def set_notifier(tg) -> None:
    """Pass a utils.telegram_notify.Notifier instance so learning events are
    sent to Telegram. Call once at server startup."""
    global _TG
    _TG = tg


def _tg_send(msg: str) -> None:
    """Send a Telegram message if a notifier is configured."""
    try:
        if _TG and _TG.enabled():
            _TG._send(msg)
    except Exception as exc:
        logger.warning("ctx=tg_send: %s", exc)


def tick(now: Optional[datetime] = None, tickers: Optional[list[str]] = None) -> dict:
    """Called once per server poll iteration. Returns {'fired': [...]}.

    `tickers` (optional): the CLI's --tickers list. Stored module-locally so
    background daily/nightly threads can read it. Pass at least once on
    startup; subsequent calls without it reuse the prior value.
    """
    now = now or _now()
    if tickers is not None:
        set_watchlist(tickers)
    state = _load_state()
    fired: list[str] = []

    if _is_due("hourly", now, state):
        if _LOCKS["hourly"].acquire(blocking=False):
            try:
                _run_with_audit("hourly", run_hourly, now, state)
                fired.append("hourly")
            finally:
                _LOCKS["hourly"].release()

    if _is_due("daily", now, state):
        if _LOCKS["daily"].acquire(blocking=False):
            # Daily can take minutes — spawn in daemon thread.
            threading.Thread(
                target=_run_async_with_audit,
                args=("daily", run_daily, now, _LOCKS["daily"]),
                daemon=True,
                name="learning_loop_daily",
            ).start()
            fired.append("daily")

    if _is_due("nightly", now, state):
        if _LOCKS["nightly"].acquire(blocking=False):
            threading.Thread(
                target=_run_async_with_audit,
                args=("nightly", run_nightly, now, _LOCKS["nightly"]),
                daemon=True,
                name="learning_loop_nightly",
            ).start()
            fired.append("nightly")

    return {"fired": fired}


def _run_with_audit(kind: str, fn, now: datetime, state: dict) -> None:
    """Synchronous run + state update + audit log."""
    started = time.monotonic()
    errors: list[str] = []
    extras: dict = {}
    try:
        result = fn(now)
        if isinstance(result, dict):
            extras = result
    except Exception as exc:
        logger.exception("ctx=%s_run failed", kind)
        errors.append(f"{type(exc).__name__}: {exc}")
    duration_s = round(time.monotonic() - started, 3)
    state[f"last_{kind}_at"] = now
    try:
        _save_state(state)
    except Exception as exc:
        logger.exception("ctx=%s_save_state failed", kind)
        errors.append(f"save_state: {exc}")
    _audit({
        "ts":         now.isoformat(),
        "kind":       kind,
        "duration_s": duration_s,
        "errors":     errors,
        **extras,
    })


def _run_async_with_audit(kind: str, fn, now: datetime, lock: threading.Lock) -> None:
    """Daemon-thread wrapper: re-loads state at completion (in case hourly
    bumped it concurrently), updates own timestamp, releases lock."""
    started = time.monotonic()
    errors: list[str] = []
    extras: dict = {}
    try:
        result = fn(now)
        if isinstance(result, (dict, str, Path)):
            extras = {"output": str(result)} if not isinstance(result, dict) else result
    except Exception as exc:
        logger.exception("ctx=%s_run failed", kind)
        errors.append(f"{type(exc).__name__}: {exc}")
    duration_s = round(time.monotonic() - started, 3)
    try:
        state = _load_state()
        state[f"last_{kind}_at"] = now
        _save_state(state)
    except Exception as exc:
        logger.exception("ctx=%s_save_state failed", kind)
        errors.append(f"save_state: {exc}")
    finally:
        lock.release()
    _audit({
        "ts":         now.isoformat(),
        "kind":       kind,
        "duration_s": duration_s,
        "errors":     errors,
        **extras,
    })


# ── Hourly: refresh learning artefacts ─────────────────────────────────────

def run_hourly(now: datetime) -> dict:
    """Refresh every learning artefact even when no trade closed.

    Steps:
      1. update_outcomes_from_mt5  — pulls any closes the per-poll hook missed
      2. pattern_stats.refresh    — recompute per-pattern WR
      3. adapt_all                — run the per-strategy adapt cycle
      4. get_calibrator()         — internal _maybe_refit if ≥10 new closes
      5. intervention_stats.reset_cache — next get_health_bias rebuilds
    """
    extras: dict = {}

    # 1. Pull MT5 closes (may itself fan out to adapt_all + pattern_stats)
    try:
        from data.trade_outcomes import update_outcomes_from_mt5
        extras["mt5_updated"] = update_outcomes_from_mt5(magic=234100)
    except Exception as exc:
        logger.warning("ctx=hourly.update_outcomes: %s", exc)
        extras["mt5_updated"] = -1

    # 1b. Resolve paper trade outcomes via bar data (TP/SL hit detection)
    try:
        from data.trade_outcomes import update_paper_outcomes_from_mt5
        paper_resolved = update_paper_outcomes_from_mt5()
        extras["paper_resolved"] = paper_resolved
    except Exception as exc:
        logger.warning("ctx=hourly.paper_outcomes: %s", exc)
        extras["paper_resolved"] = -1

    # 2. Refresh pattern stats unconditionally (cheap and important)
    try:
        from models.pattern_stats import refresh as _refresh_patterns
        stats = _refresh_patterns()
        extras["pattern_count"] = len(stats)
    except Exception as exc:
        logger.warning("ctx=hourly.pattern_stats: %s", exc)

    # 3. Adapt all strategies — only mutates if buckets changed
    try:
        from data.strategy_params import adapt_all
        from data.trade_outcomes import _load as _load_trades
        extras["buckets_updated"] = adapt_all(_load_trades())
    except Exception as exc:
        logger.warning("ctx=hourly.adapt_all: %s", exc)

    # 4. Calibrator auto-refits when threshold crossed
    try:
        from models.calibration import get_calibrator
        cal = get_calibrator()
        extras["calibrator_n"] = getattr(cal, "sample_count", 0)
        extras["calibrator_fitted"] = bool(getattr(cal, "is_fitted", False))
    except Exception as exc:
        logger.warning("ctx=hourly.calibrator: %s", exc)

    # 5. Bust intervention-stats cache so next get_health_bias rebuilds
    try:
        from models.intervention_stats import reset_cache as _reset_iv
        _reset_iv()
    except Exception as exc:
        logger.warning("ctx=hourly.intervention_stats: %s", exc)

    # Telegram: notify if any new trades were pulled or buckets adapted
    new_trades = extras.get("mt5_updated", 0)
    buckets_updated = extras.get("buckets_updated", 0)
    if (new_trades and new_trades > 0) or (buckets_updated and buckets_updated > 0):
        _tg_send(
            f"<b>Learning (hourly)</b>\n"
            f"New closes pulled: {new_trades or 0}\n"
            f"Strategy buckets adapted: {buckets_updated or 0}\n"
            f"Patterns tracked: {extras.get('pattern_count', '?')}"
        )

    return extras


# ── Daily / Weekly stubs (Phase 2 + 3 will fill in) ────────────────────────

def run_daily(now: datetime) -> Path:
    """Build the daily review and write `data/reports/daily_YYYY-MM-DD.md`.

    Pure evidence-builder: walks the watchlist with `run_intraday_backtest`,
    classifies the per-ticker regime over the last 24h, mines the trade log
    for per-strategy 30d health, and proposes (but does **not** commit)
    walk-forward conf_floor changes. Returns the report path.
    """
    from agents.intraday_backtester import run_intraday_backtest
    from agents.regime_classifier import classify as _regime_classify
    from agents.runtime import BT_SEMA
    from data.fetcher_intraday import fetch_intraday_data
    from data.strategy_params import (
        _walk_forward_validate, get_all_params, _CONF_FLOOR_MIN, _CONF_FLOOR_MAX,
    )
    from data.trade_outcomes import _load as _load_trades

    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = _REPORTS_DIR / f"daily_{now.strftime('%Y-%m-%d')}.md"
    tickers = _WATCHLIST or []
    trades = _load_trades()
    today_str = now.strftime("%Y-%m-%d")

    # ── Per-ticker regime + backtest sweep ────────────────────────────────
    regime_rows: list[dict] = []
    backtest_rows: list[dict] = []
    for ticker in tickers:
        try:
            df = fetch_intraday_data(ticker, interval="5m", days=2)
            r = _regime_classify(df, ticker=ticker, lookback_bars=288)
            regime_rows.append({
                "ticker": ticker,
                "label":  r.label,
                "pct_t":  r.pct_trending,
                "pct_r":  r.pct_ranging,
                "bars":   r.bars_analysed,
            })
        except Exception as exc:
            logger.warning("ctx=daily.regime ticker=%s: %s", ticker, exc)
            regime_rows.append({"ticker": ticker, "label": "ERROR",
                                "pct_t": 0.0, "pct_r": 0.0, "bars": 0})

        # Skip + log on backtest semaphore timeout — never deadlock the daily run.
        acquired = BT_SEMA.acquire(timeout=120)
        if not acquired:
            logger.warning("ctx=daily.backtest ticker=%s sema_timeout", ticker)
            backtest_rows.append({"ticker": ticker, "skipped": True})
            continue
        try:
            summary = run_intraday_backtest(ticker, timeframe="5m")
            for r in summary.results:
                if r.total_trades < 3:
                    continue
                backtest_rows.append({
                    "ticker":   ticker,
                    "strategy": r.strategy,
                    "n":        r.total_trades,
                    "wr":       r.win_rate,
                    "pf":       r.profit_factor,
                    "max_dd":   r.max_drawdown_pct,
                })
        except Exception as exc:
            logger.warning("ctx=daily.backtest ticker=%s: %s", ticker, exc)
        finally:
            BT_SEMA.release()

    # ── Per-strategy 30d health from live trade_log ───────────────────────
    health_rows = _strategy_health_30d(trades, now)

    # ── Walk-forward proposals (report-only, no mutations) ────────────────
    wf_rows: list[dict] = []
    all_params = get_all_params()
    for strat, params in all_params.items():
        # Only top-level strategy keys (skip per-symbol composite keys)
        if ":" in strat:
            continue
        current = float(params.get("conf_floor", 0.60))
        for delta in (-0.05, 0.05):
            proposed = round(current + delta, 2)
            if not (_CONF_FLOOR_MIN <= proposed <= _CONF_FLOOR_MAX):
                continue
            try:
                accept = _walk_forward_validate(strat, trades, current, proposed)
            except Exception as exc:
                logger.warning("ctx=daily.wf strat=%s: %s", strat, exc)
                continue
            wf_rows.append({
                "strategy": strat,
                "current":  current,
                "proposed": proposed,
                "accept":   accept,
            })

    # ── Trades closed today ───────────────────────────────────────────────
    closed_today = [
        t for t in trades
        if t.get("outcome") in ("WIN", "LOSS")
        and (t.get("closed_at") or "").startswith(today_str)
    ]

    # ── Build markdown ────────────────────────────────────────────────────
    md = _render_daily_markdown(
        now=now,
        closed_today=len(closed_today),
        regime_rows=regime_rows,
        health_rows=health_rows,
        wf_rows=wf_rows,
        backtest_rows=backtest_rows,
    )
    _atomic_write_text(report_path, md)
    return report_path


def _strategy_health_30d(trades: list[dict], now: datetime) -> list[dict]:
    """Compute (n, wr, pf) per strategy from the last 30d of closed trades."""
    from collections import defaultdict
    cutoff = now.timestamp() - 30 * 86400
    cutoff_7d = now.timestamp() - 7 * 86400
    buckets: dict[str, dict] = defaultdict(lambda: {"all": [], "recent": []})
    for t in trades:
        if t.get("strategy") == "MT5_IMPORT":
            continue
        if t.get("outcome") not in ("WIN", "LOSS"):
            continue
        closed = t.get("closed_at") or ""
        try:
            ts = datetime.fromisoformat(closed.replace("Z", "+00:00")).timestamp()
        except (ValueError, AttributeError):
            continue
        if ts < cutoff:
            continue
        b = buckets[t["strategy"]]
        b["all"].append(t)
        if ts >= cutoff_7d:
            b["recent"].append(t)

    rows: list[dict] = []
    from data.strategy_params import get_all_params
    params = get_all_params()
    for strat, slices in buckets.items():
        all_t = slices["all"]
        rec   = slices["recent"]
        wins  = sum(1 for t in all_t if t["outcome"] == "WIN")
        gains = sum(t.get("profit", 0) or 0 for t in all_t if (t.get("profit") or 0) > 0)
        losses = -sum(t.get("profit", 0) or 0 for t in all_t if (t.get("profit") or 0) < 0)
        pf = (gains / losses) if losses > 0 else (99.0 if gains > 0 else 0.0)
        wr_30d = wins / len(all_t) if all_t else 0.0
        wr_7d  = (sum(1 for t in rec if t["outcome"] == "WIN") / len(rec)) if rec else 0.0
        rows.append({
            "strategy":  strat,
            "n":         len(all_t),
            "wr":        wr_30d,
            "pf":        pf,
            "wr_drift":  wr_7d - wr_30d,
            "conf_floor": params.get(strat, {}).get("conf_floor", 0.60),
            "enabled":    bool(params.get(strat, {}).get("enabled", True)),
        })
    rows.sort(key=lambda r: -r["n"])
    return rows


def _render_daily_markdown(
    *,
    now: datetime,
    closed_today: int,
    regime_rows: list[dict],
    health_rows: list[dict],
    wf_rows: list[dict],
    backtest_rows: list[dict],
) -> str:
    """Return the daily report as a markdown string. No I/O."""
    lines: list[str] = []
    lines.append(f"# Daily Review — {now.strftime('%Y-%m-%d')}")
    lines.append(f"Generated: {now.strftime('%Y-%m-%dT%H:%MZ')}   |   "
                 f"Trades closed today: {closed_today}")
    lines.append("")

    # Regime
    lines.append("## Regime")
    if regime_rows:
        lines.append("| Ticker | %Trending | %Ranging | Label | Bars |")
        lines.append("|--------|-----------|----------|-------|------|")
        for r in regime_rows:
            lines.append(f"| {r['ticker']} | {r['pct_t']:.1%} | {r['pct_r']:.1%} | "
                         f"{r['label']} | {r['bars']} |")
    else:
        lines.append("_No watchlist set (server not yet booted)._")
    lines.append("")

    # Strategy Health (30d)
    lines.append("## Strategy Health (30d)")
    if health_rows:
        lines.append("| Strategy | N | WR | PF | WR drift 7d-30d | conf_floor | enabled |")
        lines.append("|----------|---|-----|-----|------|-------|---------|")
        for r in health_rows:
            lines.append(f"| {r['strategy']} | {r['n']} | {r['wr']:.1%} | "
                         f"{r['pf']:.2f} | {r['wr_drift']:+.1%} | "
                         f"{r['conf_floor']:.2f} | {r['enabled']} |")
    else:
        lines.append("_No closed trades in the last 30 days._")
    lines.append("")

    # Walk-forward proposals
    lines.append("## Walk-forward proposals (report-only)")
    if wf_rows:
        lines.append("| Strategy | current_floor | proposed | walk-forward accept? |")
        lines.append("|----------|---------------|----------|-----------|")
        for r in wf_rows:
            mark = "yes" if r["accept"] else "no"
            lines.append(f"| {r['strategy']} | {r['current']:.2f} | {r['proposed']:.2f} | {mark} |")
    else:
        lines.append("_No proposals (no strategies meet validation pre-conditions)._")
    lines.append("")

    # Backtest summary
    lines.append("## Backtest summary (5m, 59d)")
    if backtest_rows:
        lines.append("| Ticker | Strategy | Trades | WR | PF | Max DD |")
        lines.append("|--------|----------|--------|------|------|--------|")
        for r in backtest_rows:
            if r.get("skipped"):
                lines.append(f"| {r['ticker']} | _skipped (sema timeout)_ | | | | |")
                continue
            lines.append(f"| {r['ticker']} | {r['strategy']} | {r['n']} | "
                         f"{r['wr']:.1%} | {r['pf']:.2f} | {r['max_dd']:.1f}% |")
    else:
        lines.append("_No backtest results available._")
    lines.append("")

    return "\n".join(lines) + "\n"


def _atomic_write_text(path: Path, content: str) -> None:
    """tempfile + os.replace — never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent),
                               prefix=f".{path.stem}.", suffix=".md")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ── Nightly: structural prune + promote + report ─────────────────────────

# Prune: disable strategies failing trailing-30d health (per the user's
# 2026-05-01 decision: changes apply live from day 1).
_PRUNE_MIN_TRADES = 10   # lowered from 15 — prune toxic strategies faster
_PRUNE_WR_MAX     = 0.35
_PRUNE_PF_MAX     = 1.0

# Promote: paper-only strategies graduate to live when their 30d evidence
# clears the bar.
_PROMOTE_MIN_TRADES = 10   # was 20 — promote faster with less data
_PROMOTE_WR_MIN     = 0.50
_PROMOTE_PF_MIN     = 1.10  # was 1.30 — lower profit-factor bar


def run_nightly(now: datetime) -> Path:
    """Build the nightly review (fires every night at 23:00 UTC).

    Applies prune/promote live from day 1. Every flip is logged in the
    report and in the JSONL audit log.

    Steps:
      1. Compute 30d (strategy, symbol) bucket health.
      2. Prune buckets that fail (n>=10, wr<0.35 or pf<1.0).
      3. Promote paper-only strategies whose 30d health passes the bar.
      4. Write data/reports/nightly_YYYY-MM-DD.md.
      5. Force calibrator refit so it sees the new active mix.
      6. Rotate learning_log.jsonl.
    """
    import data.strategy_params as _sp     # access _PARAMS through the module so
                                             # load_params()'s rebind is visible
    from data.strategy_params import load_params, save_params
    from data.trade_outcomes import _load as _load_trades
    from agents.regime_classifier import classify as _regime_classify
    from data.fetcher_intraday import fetch_intraday_data

    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    iso_year, iso_week, _ = now.isocalendar()
    report_path = _REPORTS_DIR / f"nightly_{now.strftime('%Y-%m-%d')}.md"

    trades = _load_trades()
    load_params()        # rebinds _sp._PARAMS to the file contents

    pruned: list[dict] = []
    promoted: list[dict] = []
    scoreboard = _strategy_health_30d(trades, now)
    sym_scoreboard = _per_symbol_health_30d(trades, now)

    # ── Prune step ────────────────────────────────────────────────────
    health_lookup = {row["strategy"]: row for row in scoreboard}
    for row in scoreboard:
        if row["n"] < _PRUNE_MIN_TRADES:
            continue
        if row["wr"] < _PRUNE_WR_MAX and row["pf"] < _PRUNE_PF_MAX:
            params = _sp._PARAMS.get(row["strategy"]) or {}
            if params.get("enabled", True):
                params["enabled"] = False
                params["last_adapted"] = now.isoformat()
                pruned.append({
                    "strategy": row["strategy"],
                    "n":  row["n"], "wr": row["wr"], "pf": row["pf"],
                })
                logger.info("nightly prune: disabled %s (n=%d wr=%.2f pf=%.2f)",
                            row["strategy"], row["n"], row["wr"], row["pf"])

    # ── Promote step (paper_only strategies) ──────────────────────────
    for strat, params in list(_sp._PARAMS.items()):
        if ":" in strat:
            continue   # composite per-symbol key — never auto-promote
        if not params.get("paper_only", False):
            continue
        h = health_lookup.get(strat)
        if not h or h["n"] < _PROMOTE_MIN_TRADES:
            continue
        if h["wr"] >= _PROMOTE_WR_MIN and h["pf"] >= _PROMOTE_PF_MIN:
            params["paper_only"] = False
            params["enabled"]    = True
            params["last_adapted"] = now.isoformat()
            promoted.append({
                "strategy": strat,
                "n": h["n"], "wr": h["wr"], "pf": h["pf"],
            })
            logger.info("nightly promote: %s went paper_only=False enabled=True "
                        "(n=%d wr=%.2f pf=%.2f)",
                        strat, h["n"], h["wr"], h["pf"])

    if pruned or promoted:
        save_params()
        lines = ["<b>Learning (nightly) — Strategy changes</b>"]
        for r in pruned:
            lines.append(f"DISABLED: {r['strategy']} (WR {r['wr']:.0%}, PF {r['pf']:.2f}, n={r['n']})")
        for r in promoted:
            lines.append(f"PROMOTED: {r['strategy']} paper->live (WR {r['wr']:.0%}, n={r['n']})")
        _tg_send("\n".join(lines))

    # ── Symbol promotion: paper → live ───────────────────────────────
    # Promote a PAPER symbol to LIVE when it accumulates enough evidence.
    _SYM_PROMOTE_MIN_TRADES = 20
    _SYM_PROMOTE_WR_MIN     = 0.40
    _SYM_PROMOTE_PF_MIN     = 1.0

    sym_promoted: list[dict] = []
    from data.symbol_policy import _normalise as _norm_sym, PAPER_SYMBOLS, save_override, _load_overrides
    existing_overrides = _load_overrides()
    for row in sym_scoreboard:
        sym = row["symbol"]
        # Only consider symbols currently designated PAPER
        in_paper_set = sym in PAPER_SYMBOLS
        in_override  = existing_overrides.get(sym)
        is_paper = in_paper_set and in_override != "LIVE"
        if not is_paper:
            continue
        if row["n"] < _SYM_PROMOTE_MIN_TRADES:
            continue
        if row["wr"] >= _SYM_PROMOTE_WR_MIN and row["pf"] >= _SYM_PROMOTE_PF_MIN:
            save_override(sym, "LIVE")
            sym_promoted.append({"symbol": sym, "n": row["n"],
                                  "wr": row["wr"], "pf": row["pf"]})
            logger.info("nightly symbol promote: %s -> LIVE (n=%d wr=%.2f pf=%.2f)",
                        sym, row["n"], row["wr"], row["pf"])
            _tg_send(
                f"<b>Symbol promoted to LIVE: {sym}</b>\n"
                f"WR {row['wr']:.0%}  |  PF {row['pf']:.2f}  |  n={row['n']}"
            )

    # ── Watchlist regime over the past week ───────────────────────────
    nightly_regime: list[dict] = []
    label_counts = {"TRENDING": 0, "RANGING": 0, "MIXED": 0,
                    "INSUFFICIENT_DATA": 0, "ERROR": 0}
    for ticker in (_WATCHLIST or []):
        try:
            df = fetch_intraday_data(ticker, interval="5m", days=7)
            r = _regime_classify(df, ticker=ticker, lookback_bars=2016)  # 7d × 288
            nightly_regime.append({
                "ticker":  ticker,
                "label":   r.label,
                "pct_t":   r.pct_trending,
                "pct_r":   r.pct_ranging,
                "bars":    r.bars_analysed,
            })
            label_counts[r.label] = label_counts.get(r.label, 0) + 1
        except Exception as exc:
            logger.warning("ctx=nightly.regime ticker=%s: %s", ticker, exc)
            nightly_regime.append({"ticker": ticker, "label": "ERROR",
                                   "pct_t": 0.0, "pct_r": 0.0, "bars": 0})
            label_counts["ERROR"] += 1
    overall_regime = (
        max(label_counts, key=label_counts.get) if nightly_regime else "UNKNOWN"
    )

    # ── ML predictor retrain ──────────────────────────────────────────
    try:
        from models.ml_predictor import retrain_from_log, get_predictor as _get_pred
        ml_retrained = retrain_from_log()
        logger.info("ctx=nightly.ml_predictor: retrained=%s", ml_retrained)
        if ml_retrained:
            _pred = _get_pred()
            if _pred.is_fitted and getattr(_pred, "feature_importances_", None):
                top = sorted(_pred.feature_importances_.items(), key=lambda x: -x[1])[:5]
                lines = [f"<b>ML top features (n={_pred.n_samples}, AUC={_pred.cv_score:.3f})</b>"]
                for _fname, _score in top:
                    lines.append(f"  {_fname}: {_score:.3f}")
                _tg_send("\n".join(lines))
    except Exception as exc:
        logger.warning("ctx=nightly.ml_predictor: %s", exc)

    # ── Boost-stack attribution (Phase 1) ─────────────────────────────
    # Reports which boost components correlate with WIN outcomes so the
    # user can spot anti-predictive components causing confidence inversion.
    # Read-only — no auto-mutation; user reviews and edits signal_engine.py.
    try:
        from models.boost_attribution import compute_correlations, format_telegram
        boost_report = compute_correlations(trades)
        if boost_report["n"] >= 5:   # always log; only Telegram-send above min_trades
            logger.info("ctx=nightly.boost_attribution: %s", boost_report["verdict"])
        if boost_report["n"] >= 20:
            _tg_send(format_telegram(boost_report))
    except Exception as exc:
        logger.warning("ctx=nightly.boost_attribution: %s", exc)

    # ── Calibration snapshot before forced refit ──────────────────────
    calibration_info = {"sample_count": 0, "fitted": False}
    try:
        from models.calibration import reset_singleton, get_calibrator
        reset_singleton()    # forces a fresh fit on next call
        cal = get_calibrator()
        calibration_info["sample_count"] = getattr(cal, "sample_count", 0)
        calibration_info["fitted"]       = bool(getattr(cal, "is_fitted", False))
    except Exception as exc:
        logger.warning("ctx=nightly.calibration: %s", exc)

    # ── Render markdown ───────────────────────────────────────────────
    md = _render_nightly_markdown(
        now=now,
        iso_year=iso_year,
        iso_week=iso_week,
        scoreboard=scoreboard,
        sym_scoreboard=sym_scoreboard,
        pruned=pruned,
        promoted=promoted,
        sym_promoted=sym_promoted,
        nightly_regime=nightly_regime,
        overall_regime=overall_regime,
        calibration_info=calibration_info,
    )
    _atomic_write_text(report_path, md)

    # ── Rotate the JSONL audit log so it doesn't grow unbounded ───────
    try:
        if _AUDIT_PATH.exists():
            archive = _DATA_DIR / f"learning_log.{now.strftime('%Y-%m-%d')}.jsonl"
            os.replace(_AUDIT_PATH, archive)
    except OSError as exc:
        logger.warning("ctx=nightly.rotate_audit: %s", exc)

    # ── Backup data files to GitHub (data-backup branch) ─────────────
    try:
        _backup_data_to_git(now)
    except Exception as exc:
        logger.warning("ctx=nightly.backup: %s", exc)

    return report_path


_BACKUP_FILES = [
    "data/trade_log.json",
    "data/strategy_params.json",
    "data/pattern_stats.json",
    "data/calibration.json",
    "data/stop_minimums.json",
    "data/symbol_promotions.json",
    "data/learning_state.json",
]

_GIT_ENV = {
    "GIT_AUTHOR_NAME":     "katraswing-server",
    "GIT_AUTHOR_EMAIL":    "server@katraswing.local",
    "GIT_COMMITTER_NAME":  "katraswing-server",
    "GIT_COMMITTER_EMAIL": "server@katraswing.local",
}


def _backup_data_to_git(now: datetime) -> None:
    """Commit gitignored data files to the data-backup branch on GitHub.

    Uses a temporary git worktree so the main working tree is never touched
    and `git pull` in start_all.ps1 continues to work without conflicts.

    Restore on a new server after cloning:
        git fetch origin data-backup:data-backup
        git checkout data-backup -- data/
        git checkout main
    """
    import shutil
    import subprocess

    install_dir = _DATA_DIR.parent
    wt_path = install_dir.parent / "_katraswing_backup_wt"
    env = {**os.environ, **_GIT_ENV}

    existing = [f for f in _BACKUP_FILES if (install_dir / f).exists()]
    if not existing:
        return

    try:
        # Clean up any stale worktree from a previous crashed run
        subprocess.run(["git", "worktree", "prune"],
                       cwd=install_dir, capture_output=True, timeout=15)
        if wt_path.exists():
            shutil.rmtree(wt_path, ignore_errors=True)

        # Create the data-backup branch on remote if it doesn't exist yet
        ls = subprocess.run(
            ["git", "ls-remote", "--heads", "origin", "data-backup"],
            cwd=install_dir, capture_output=True, timeout=20,
        )
        if not ls.stdout.strip():
            # Write an empty tree object, make a root commit, push
            empty_tree = subprocess.run(
                ["git", "hash-object", "-t", "tree", "--stdin"],
                input=b"", cwd=install_dir, capture_output=True,
                timeout=10, env=env,
            ).stdout.decode().strip()
            empty_commit = subprocess.run(
                ["git", "commit-tree", empty_tree, "-m", "init: data backup branch"],
                cwd=install_dir, capture_output=True,
                timeout=10, env=env,
            ).stdout.decode().strip()
            subprocess.run(
                ["git", "branch", "data-backup", empty_commit],
                cwd=install_dir, capture_output=True, timeout=10,
            )
            subprocess.run(
                ["git", "push", "origin", "data-backup"],
                cwd=install_dir, capture_output=True, timeout=30,
            )
            logger.info("ctx=nightly.backup: created data-backup branch on remote")
        else:
            # Remote branch exists — ensure local tracking branch also exists.
            # On a fresh clone the local branch is absent; git worktree add
            # requires a local branch, not just a remote ref.
            local_br = subprocess.run(
                ["git", "branch", "--list", "data-backup"],
                cwd=install_dir, capture_output=True, timeout=10,
            )
            if not local_br.stdout.strip():
                subprocess.run(
                    ["git", "fetch", "origin", "data-backup:data-backup"],
                    cwd=install_dir, capture_output=True, timeout=30,
                )

        # Attach a worktree to the data-backup branch
        r = subprocess.run(
            ["git", "worktree", "add", str(wt_path), "data-backup"],
            cwd=install_dir, capture_output=True, timeout=30,
        )
        if r.returncode != 0:
            logger.warning("ctx=nightly.backup: worktree add failed: %s",
                           r.stderr.decode().strip())
            return

        # Copy each data file into the worktree
        (wt_path / "data").mkdir(exist_ok=True)
        for rel in existing:
            shutil.copy2(install_dir / rel, wt_path / rel)

        # Stage and check for actual changes before committing
        subprocess.run(["git", "add", "data/"],
                       cwd=wt_path, capture_output=True)
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=wt_path, capture_output=True,
        )
        if diff.returncode == 0:
            logger.info("ctx=nightly.backup: no data changes since last backup")
            return

        date_str = now.strftime("%Y-%m-%d")
        subprocess.run(
            ["git", "commit", "-m", f"data: backup {date_str}"],
            cwd=wt_path, capture_output=True, timeout=30, env=env,
        )
        push = subprocess.run(
            ["git", "push", "origin", "data-backup"],
            cwd=wt_path, capture_output=True, timeout=60,
        )
        if push.returncode == 0:
            logger.info("ctx=nightly.backup: pushed data backup for %s", date_str)
            _tg_send(f"<b>Nightly backup pushed</b> — {date_str}\n"
                     f"Files: {', '.join(existing)}")
        else:
            logger.warning("ctx=nightly.backup: push failed: %s",
                           push.stderr.decode().strip())

    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(wt_path)],
            cwd=install_dir, capture_output=True, timeout=15,
        )


def _per_symbol_health_30d(trades: list[dict], now: datetime) -> list[dict]:
    """Per-symbol scoreboard over the trailing 30d.

    Includes MT5_IMPORT rows (backfilled historical trades) because those
    ARE system trades — just without strategy metadata. Strategy-level
    prune/promote still excludes them; only symbol-level stats use them.
    """
    from collections import defaultdict
    cutoff = now.timestamp() - 30 * 86400
    buckets: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        if t.get("outcome") not in ("WIN", "LOSS"):
            continue
        try:
            ts = datetime.fromisoformat(
                (t.get("closed_at") or "").replace("Z", "+00:00")
            ).timestamp()
        except (ValueError, AttributeError):
            continue
        if ts < cutoff:
            continue
        sym = (t.get("ticker") or "").replace("=X", "").replace("=F", "").upper()
        if sym:
            buckets[sym].append(t)
    rows: list[dict] = []
    for sym, ts in buckets.items():
        wins = sum(1 for t in ts if t["outcome"] == "WIN")
        gains = sum(t.get("profit", 0) or 0 for t in ts if (t.get("profit") or 0) > 0)
        losses = -sum(t.get("profit", 0) or 0 for t in ts if (t.get("profit") or 0) < 0)
        pf = (gains / losses) if losses > 0 else (99.0 if gains > 0 else 0.0)
        rows.append({
            "symbol": sym, "n": len(ts),
            "wr": wins / len(ts) if ts else 0.0,
            "pf": pf,
            "net":  sum(t.get("profit", 0) or 0 for t in ts),
        })
    rows.sort(key=lambda r: -r["net"])
    return rows


def _render_nightly_markdown(
    *,
    now: datetime,
    iso_year: int,
    iso_week: int,
    scoreboard: list[dict],
    sym_scoreboard: list[dict],
    pruned: list[dict],
    promoted: list[dict],
    sym_promoted: list[dict] | None = None,
    nightly_regime: list[dict],
    overall_regime: str,
    calibration_info: dict,
) -> str:
    """Render the nightly markdown report. Pure formatting; no I/O."""
    lines: list[str] = []
    lines.append(f"# Nightly Review — {now.strftime('%Y-%m-%d')}")
    lines.append(f"Regime: **{overall_regime}**")
    lines.append("")

    # Structural changes
    lines.append("## Structural changes")
    any_changes = pruned or promoted or sym_promoted
    if any_changes:
        for r in pruned:
            lines.append(f"- DISABLED: {r['strategy']} "
                         f"(WR {r['wr']:.0%}, PF {r['pf']:.2f}, n={r['n']})")
        for r in promoted:
            lines.append(f"- PROMOTED: {r['strategy']} from paper_only "
                         f"(WR {r['wr']:.0%}, PF {r['pf']:.2f}, n={r['n']})")
        for r in (sym_promoted or []):
            lines.append(f"- SYMBOL LIVE: {r['symbol']} graduated from PAPER "
                         f"(WR {r['wr']:.0%}, PF {r['pf']:.2f}, n={r['n']})")
    else:
        lines.append("_No strategies hit the prune or promote thresholds today._")
    lines.append("")

    # Strategy scoreboard
    lines.append("## Strategy scoreboard (30d)")
    if scoreboard:
        lines.append("| Strategy | N | WR | PF | WR drift 7d-30d |")
        lines.append("|----------|---|-----|------|------|")
        for r in scoreboard:
            lines.append(f"| {r['strategy']} | {r['n']} | {r['wr']:.1%} | "
                         f"{r['pf']:.2f} | {r['wr_drift']:+.1%} |")
    else:
        lines.append("_No 30d closed-trade evidence._")
    lines.append("")

    # Per-symbol scoreboard
    lines.append("## Per-symbol scoreboard (30d)")
    if sym_scoreboard:
        lines.append("| Symbol | N | WR | PF | Net P&L |")
        lines.append("|--------|---|-----|------|---------|")
        for r in sym_scoreboard:
            lines.append(f"| {r['symbol']} | {r['n']} | {r['wr']:.1%} | "
                         f"{r['pf']:.2f} | {r['net']:+.2f} |")
    else:
        lines.append("_No 30d per-symbol evidence._")
    lines.append("")

    # Calibration
    lines.append("## Calibration")
    lines.append(f"- Sample count: **{calibration_info['sample_count']}** "
                 f"(force-refit: {calibration_info['fitted']})")
    lines.append("")

    # Watchlist regime
    lines.append("## Watchlist regime breakdown (last 7d)")
    if nightly_regime:
        lines.append("| Ticker | %Trending | %Ranging | Bars | Dominant |")
        lines.append("|--------|-----------|----------|------|----------|")
        for r in nightly_regime:
            lines.append(f"| {r['ticker']} | {r['pct_t']:.1%} | {r['pct_r']:.1%} | "
                         f"{r['bars']} | {r['label']} |")
    else:
        lines.append("_No watchlist set; cannot classify regime._")
    lines.append("")

    return "\n".join(lines) + "\n"
