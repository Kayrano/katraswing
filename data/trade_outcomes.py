"""
Trade outcome tracker — records every live auto-trade sent to MT5,
matches closed deals back to those records, and computes per-strategy
win rates that feed back into signal confidence calibration.

Storage: data/trade_log.json  (simple JSON, human-readable)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_LOG_PATH = Path(__file__).parent / "trade_log.json"
_MIN_TRADES = 5   # minimum closed trades before a strategy's win rate is used

# ── mtime-keyed caches ───────────────────────────────────────────────────────
# These are read on every poll cycle (background thread) and on every UI rerun
# (Streamlit). The trade log only changes when an order is placed or an
# outcome is recorded — typically minutes apart. Cache the parsed list and
# the two derived stats dicts; invalidate when the file's mtime advances.
_LOAD_CACHE:           tuple[float, list[dict]] | None = None
_DETAILED_WR_CACHE:    dict[int, tuple[float, dict[str, float]]] = {}
_OPTIMAL_STOPS_CACHE:  dict[int, tuple[float, dict[str, dict]]]  = {}
_WIN_RATES_CACHE:      dict[int, tuple[float, dict[str, float]]] = {}


def _file_mtime() -> float:
    try:
        return _LOG_PATH.stat().st_mtime if _LOG_PATH.exists() else 0.0
    except Exception:
        return 0.0


# ── Persistence ───────────────────────────────────────────────────────────────

def _load() -> list[dict]:
    """Read trade_log.json with an mtime-keyed cache. Subsequent calls within
    the same file revision return the cached parsed list — eliminates JSON
    parse cost on hot paths (every UI rerun + every poll iteration)."""
    global _LOAD_CACHE
    mtime = _file_mtime()
    if mtime == 0.0:
        return []
    if _LOAD_CACHE is not None and _LOAD_CACHE[0] == mtime:
        return _LOAD_CACHE[1]
    try:
        data = json.loads(_LOG_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"trade_log load error: {exc}")
        return []
    _LOAD_CACHE = (mtime, data)
    return data


def _save(trades: list[dict]) -> None:
    global _LOAD_CACHE
    try:
        _LOG_PATH.write_text(
            json.dumps(trades, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        # Invalidate the load cache eagerly so the next read picks up the
        # new mtime; downstream stat caches will refresh on demand.
        _LOAD_CACHE = None
    except Exception as exc:
        logger.warning(f"trade_log save error: {exc}")


# ── Public API ────────────────────────────────────────────────────────────────

def record_trade(
    ticket: int,
    ticker: str,
    strategy: str,
    direction: str,
    confidence: float,
    entry: float,
    sl: float,
    tp: float,
    patterns: Optional[list] = None,
    *,
    adx_value: Optional[float] = None,
    atr_value: Optional[float] = None,
    spread_pips: Optional[float] = None,
    h1_trend: Optional[str] = None,
    vol_ratio: Optional[float] = None,
    consensus_count: Optional[int] = None,
    pattern_boost_val: Optional[float] = None,
    calibrated_conf: Optional[float] = None,
    paper_only: bool = False,
    mt5_symbol: Optional[str] = None,
    # ── Boost-stack attribution (Phase 1) ─────────────────────────────────
    # Recorded so models/boost_attribution.py can compute the correlation of
    # each component with WIN outcomes after enough trades close. Without
    # these we cannot tell which boost is anti-predictive.
    base_confidence: Optional[float] = None,
    consensus_boost: Optional[float] = None,
    bt_adjustment: Optional[float] = None,
    live_adjustment: Optional[float] = None,
    news_boost: Optional[float] = None,
    session_boost: Optional[float] = None,
) -> None:
    """Record a newly sent trade. Called immediately after order_send succeeds,
    OR for paper trades using a synthetic ticket so the promotion harness can
    accumulate outcome data.

    `patterns` is the list[PatternMatch] that fired at entry — recorded so the
    pattern win-rate learner (models.pattern_stats) can attribute each closed
    trade's outcome back to the patterns that triggered it.

    Optional keyword-only args capture market context at signal time so the
    ML predictor can use richer features:
      adx_value        — ADX reading at entry (regime strength)
      atr_value        — raw ATR in price units (volatility proxy)
      spread_pips      — bid-ask spread from MT5 tick (fill-cost indicator)
      h1_trend         — H1 daily-structure direction: "UP" | "DOWN" | "NEUTRAL"
      vol_ratio        — current ATR / 20-bar avg ATR (>1 = volatile, <1 = calm)
      consensus_count  — number of strategies agreeing on the dominant direction
      pattern_boost_val— pattern alignment boost applied at entry (±0.05)
      calibrated_conf  — isotonic-calibrated empirical win probability
      paper_only       — True for simulated paper trades (no MT5 ticket)
    """
    now_utc = datetime.now(timezone.utc)
    trades = _load()
    # Avoid duplicates (e.g. rerun after reconnect)
    if any(t["ticket"] == ticket for t in trades):
        return

    pattern_records = []
    if patterns:
        for p in patterns:
            try:
                pattern_records.append({
                    "name":       getattr(p, "name", ""),
                    "bias":       getattr(p, "bias", ""),
                    "confidence": float(getattr(p, "confidence", 0.0)),
                    "win_rate":   float(getattr(p, "win_rate", 0.0)),
                })
            except Exception:
                continue

    def _session(h: int) -> str:
        if 7 <= h < 12:
            return "london"
        if 12 <= h < 17:
            return "ny"
        if 0 <= h < 4:
            return "asia"
        return "other"

    trades.append({
        "ticket":     ticket,
        "ticker":     ticker,
        "strategy":   strategy,
        "direction":  direction,
        "confidence": round(confidence, 4),
        "entry":      entry,
        "sl":         sl,
        "tp":         tp,
        "patterns":   pattern_records,
        "sent_at":    now_utc.replace(tzinfo=None).isoformat(timespec="seconds"),
        "closed_at":  None,
        "profit":     None,
        "outcome":    None,   # "WIN" | "LOSS" | "BREAKEVEN"
        # ── Market context at signal time (ML features) ──────────────────
        "adx_value":         round(adx_value, 2) if adx_value is not None else None,
        "atr_value":         round(atr_value, 6) if atr_value is not None else None,
        "spread_pips":       round(spread_pips, 2) if spread_pips is not None else None,
        "h1_trend":          h1_trend,
        "session":           _session(now_utc.hour),
        "day_of_week":       now_utc.weekday(),
        "vol_ratio":         round(vol_ratio, 4) if vol_ratio is not None else None,
        "consensus_count":   int(consensus_count) if consensus_count is not None else None,
        "pattern_boost_val": round(pattern_boost_val, 4) if pattern_boost_val is not None else None,
        "calibrated_conf":   round(calibrated_conf, 4) if calibrated_conf is not None else None,
        "paper_only":        paper_only,
        "mt5_symbol":        mt5_symbol or "",
        # ── Boost components (Phase 1: per-component WR attribution) ──────
        "base_confidence":   round(base_confidence, 4) if base_confidence is not None else None,
        "consensus_boost":   round(consensus_boost, 4) if consensus_boost is not None else None,
        "bt_adjustment":     round(bt_adjustment, 4)   if bt_adjustment   is not None else None,
        "live_adjustment":   round(live_adjustment, 4) if live_adjustment is not None else None,
        "news_boost":        round(news_boost, 4)     if news_boost      is not None else None,
        "session_boost":     round(session_boost, 4)  if session_boost   is not None else None,
    })
    _save(trades)
    logger.info(f"Recorded trade #{ticket} {direction} {ticker} via {strategy}")


def _classify_close_reason(trade: dict, close_price: float, broker_comment: str) -> str:
    """Best-effort classification of how a trade exited.

    Returns one of: 'SL', 'TP', 'TRADE_MANAGER', 'MANUAL', 'UNKNOWN'.

    Priority:
      1. Broker comment hints (most reliable when present).
      2. Price proximity to the originally-recorded SL or TP (within 0.05%).
      3. Fall back to UNKNOWN.

    The trade-manager closes via the `MT5_BRIDGE` comment we set on outgoing
    `position_close` calls, so a non-empty Katraswing comment means the
    trade-manager intervened (partial-close, stop-tighten, reversal, etc.).
    """
    comment = (broker_comment or "").lower()
    if "sl " in comment or "[sl" in comment or comment.startswith("sl"):
        return "SL"
    if "tp " in comment or "[tp" in comment or comment.startswith("tp"):
        return "TP"
    if "so:" in comment or "stopout" in comment:
        return "STOPOUT"
    if "katraswing" in comment or "trade_manager" in comment:
        return "TRADE_MANAGER"

    sl_price = float(trade.get("sl") or 0)
    tp_price = float(trade.get("tp") or 0)
    if close_price > 0:
        for ref, label in ((sl_price, "SL"), (tp_price, "TP")):
            if ref > 0 and abs(close_price - ref) / max(ref, 1e-9) < 0.0005:
                return label
    return "UNKNOWN"


def backfill_close_reasons() -> dict[str, int]:
    """Walk the existing trade_log and classify any rows that lack a
    close_reason. Pure local computation — does not call MT5.

    Each row inferred this way is tagged `close_reason_source="BACKFILL"` so
    later analysis can weight classifier-confidence on stale-data fills
    differently from rows tagged at close time.

    Returns a dict with counts per resulting close_reason category, plus
    `seen` (total rows considered) and `updated` (rows newly classified).
    """
    trades = _load()
    counts: dict[str, int] = {"seen": 0, "updated": 0}
    if not trades:
        return counts

    for t in trades:
        if t.get("outcome") not in ("WIN", "LOSS"):
            continue
        counts["seen"] += 1
        # Skip rows that already carry a non-UNKNOWN close_reason from the
        # at-close classifier — we only want to upgrade UNKNOWNs and the
        # missing-field case.
        existing = (t.get("close_reason") or "").upper()
        if existing and existing != "UNKNOWN":
            continue

        close_price = float(t.get("close_price") or 0)
        # No broker_comment field on historical rows — pass empty so the
        # classifier falls through to the price-proximity heuristic.
        reason = _classify_close_reason(t, close_price, "")

        # Last-resort fallback for rows with no close_price recorded:
        # use the sign of profit to distinguish TP-like vs SL-like exits.
        if reason == "UNKNOWN":
            p = t.get("profit")
            if isinstance(p, (int, float)) and p:
                reason = "TP_LIKELY" if p > 0 else "SL_LIKELY"

        t["close_reason"] = reason
        t["close_reason_source"] = "BACKFILL"
        counts["updated"] += 1
        counts[reason] = counts.get(reason, 0) + 1

    if counts["updated"]:
        _save(trades)
        # Bust the load cache so subsequent reads see the patched rows.
        global _LOAD_CACHE
        _LOAD_CACHE = None
        logger.info("backfill_close_reasons: updated %d/%d rows", counts["updated"], counts["seen"])
    return counts


def update_paper_outcomes_from_mt5(mt5_symbol_map: dict | None = None) -> int:
    """Check open paper trades and mark WIN/LOSS if TP or SL was hit.

    Uses 5-minute MT5 bar data since the trade's sent_at to detect the first
    candle where the high/low crossed the recorded TP or SL.

    LONG: WIN if any bar's high >= tp, LOSS if any bar's low <= sl
    SHORT: WIN if any bar's low <= tp, LOSS if any bar's high >= sl

    Logs a per-call summary at INFO so the hourly loop's resolution rate is
    visible. Auto-repairs missing/wrong mt5_symbol fields using the bridge's
    resolver — older logs from before the symbol-fix can self-heal in place.

    Returns the number of paper trades whose outcome was updated.
    """
    try:
        import MetaTrader5 as mt5  # type: ignore[import]
    except ImportError:
        return 0

    # MT5 must be initialised by the caller (signal server). If it isn't,
    # copy_rates_range silently returns None for every symbol — log loudly
    # so the failure mode is obvious.
    if mt5.terminal_info() is None:
        logger.warning("update_paper_outcomes: MT5 not connected — skipping")
        return 0

    trades = _load()
    open_paper = [
        t for t in trades
        if t.get("paper_only")
        and t.get("outcome") is None
        and t.get("entry") and t.get("sl") and t.get("tp")
        and (t.get("mt5_symbol") or t.get("ticker"))
    ]
    if not open_paper:
        logger.debug("update_paper_outcomes: no open paper trades")
        return 0

    try:
        from utils.mt5_bridge import resolve_mt5_symbol
    except Exception:
        resolve_mt5_symbol = None  # type: ignore[assignment]

    updated      = 0
    no_bars      = 0
    bad_symbol   = 0
    bad_date     = 0
    self_healed  = 0

    for t in open_paper:
        mt5_sym = (t.get("mt5_symbol") or "").strip()
        # Self-heal: if stored mt5_symbol is empty or stale (looks like a
        # yfinance ticker), resolve from the bridge.
        if (not mt5_sym or "=" in mt5_sym) and resolve_mt5_symbol:
            try:
                resolved = resolve_mt5_symbol(t.get("ticker", "")) or ""
                if resolved and resolved != mt5_sym:
                    t["mt5_symbol"] = resolved
                    mt5_sym = resolved
                    self_healed += 1
            except Exception:
                pass
        if not mt5_sym:
            bad_symbol += 1
            continue

        sent_str = t.get("sent_at", "")
        try:
            sent_dt = datetime.fromisoformat(sent_str).replace(tzinfo=None)
        except Exception:
            bad_date += 1
            continue

        entry     = float(t["entry"])
        sl        = float(t["sl"])
        tp        = float(t["tp"])
        direction = t.get("direction", "LONG")
        now_utc   = datetime.now(timezone.utc).replace(tzinfo=None)

        # Ensure symbol is subscribed before querying bars — some brokers
        # return None for symbols not in Market Watch.
        try:
            mt5.symbol_select(mt5_sym, True)
        except Exception:
            pass

        bars = mt5.copy_rates_range(mt5_sym, mt5.TIMEFRAME_M5, sent_dt, now_utc)
        if bars is None or len(bars) == 0:
            no_bars += 1
            continue

        outcome = None
        close_time = None
        close_price = None
        # ── Phase 4: MFE/MAE tracking ─────────────────────────────────────
        # Walk the bars and keep track of maximum favorable/adverse excursion
        # in price units; convert to R-multiples (relative to risk) at the end.
        # MFE = highest profit the trade reached before close
        # MAE = deepest drawdown the trade reached before close
        # These give the ML feature engineering richer labels than binary
        # WIN/LOSS — e.g. a 0.7R MFE that reversed to SL tells us the entry
        # was right but the exit was wrong.
        mfe_price = entry   # best price reached in favorable direction
        mae_price = entry   # worst price reached in adverse direction
        bars_to_close = 0

        for bar in bars:
            bar_high = float(bar["high"])
            bar_low  = float(bar["low"])
            bar_time = datetime.utcfromtimestamp(int(bar["time"])).isoformat(timespec="seconds")
            bars_to_close += 1

            # Track excursion BEFORE checking SL/TP — the same bar that hit
            # SL also represents that bar's worst MAE.
            if direction == "LONG":
                if bar_high > mfe_price: mfe_price = bar_high
                if bar_low  < mae_price: mae_price = bar_low
                if bar_high >= tp:
                    outcome = "WIN"
                    close_time  = bar_time
                    close_price = tp
                    break
                if bar_low <= sl:
                    outcome = "LOSS"
                    close_time  = bar_time
                    close_price = sl
                    break
            else:  # SHORT
                if bar_low  < mfe_price: mfe_price = bar_low
                if bar_high > mae_price: mae_price = bar_high
                if bar_low <= tp:
                    outcome = "WIN"
                    close_time  = bar_time
                    close_price = tp
                    break
                if bar_high >= sl:
                    outcome = "LOSS"
                    close_time  = bar_time
                    close_price = sl
                    break

        if outcome:
            risk = abs(entry - sl)
            if direction == "LONG":
                mfe_r = (mfe_price - entry) / risk if risk > 0 else 0.0
                mae_r = (entry - mae_price) / risk if risk > 0 else 0.0
            else:
                mfe_r = (entry - mfe_price) / risk if risk > 0 else 0.0
                mae_r = (mae_price - entry) / risk if risk > 0 else 0.0

            t["outcome"]      = outcome
            t["closed_at"]    = close_time
            t["close_price"]  = close_price
            t["close_reason"] = "TP" if outcome == "WIN" else "SL"
            t["profit"]       = abs(tp - entry) if outcome == "WIN" else -abs(sl - entry)
            # MFE/MAE in R-multiples — clamped at zero on the unfavorable side
            t["mfe_r"]                 = round(max(mfe_r, 0.0), 3)
            t["mae_r"]                 = round(max(mae_r, 0.0), 3)
            t["bars_to_resolution"]    = bars_to_close
            t["time_to_resolution_min"] = bars_to_close * 5   # 5m bars
            updated += 1

    # Persist if anything changed (resolved outcomes OR self-healed symbols).
    if updated or self_healed:
        _save(trades)

    logger.info(
        "update_paper_outcomes: open=%d resolved=%d still_open=%d "
        "(no_bars=%d bad_symbol=%d bad_date=%d self_healed=%d)",
        len(open_paper), updated, len(open_paper) - updated,
        no_bars, bad_symbol, bad_date, self_healed,
    )
    return updated


def _enrich_live_trade_with_mfe_mae(t: dict, mt5) -> None:
    """Fetch 5m bars between sent_at and closed_at; compute MFE/MAE.

    Mutates `t` in place. Silent on failure — these are diagnostic-only
    fields; the trade outcome itself is already recorded.
    """
    entry = t.get("entry")
    sl    = t.get("sl")
    if not entry or not sl:
        return
    sent_str = t.get("sent_at", "")
    close_str = t.get("closed_at", "")
    if not sent_str or not close_str:
        return
    try:
        sent_dt  = datetime.fromisoformat(sent_str).replace(tzinfo=None)
        close_dt = datetime.fromisoformat(close_str).replace(tzinfo=None)
    except Exception:
        return

    # Resolve the broker symbol for this trade
    sym = (t.get("mt5_symbol") or "").strip()
    if not sym:
        try:
            from utils.mt5_bridge import resolve_mt5_symbol
            sym = resolve_mt5_symbol(t.get("ticker", "")) or ""
        except Exception:
            return
    if not sym:
        return

    try:
        mt5.symbol_select(sym, True)
    except Exception:
        pass

    bars = mt5.copy_rates_range(sym, mt5.TIMEFRAME_M5, sent_dt, close_dt)
    if bars is None or len(bars) == 0:
        return

    direction = t.get("direction", "LONG")
    entry_p   = float(entry)
    sl_p      = float(sl)
    mfe_price = entry_p
    mae_price = entry_p
    for bar in bars:
        bar_high = float(bar["high"])
        bar_low  = float(bar["low"])
        if direction == "LONG":
            if bar_high > mfe_price: mfe_price = bar_high
            if bar_low  < mae_price: mae_price = bar_low
        else:
            if bar_low  < mfe_price: mfe_price = bar_low
            if bar_high > mae_price: mae_price = bar_high

    risk = abs(entry_p - sl_p)
    if direction == "LONG":
        mfe_r = (mfe_price - entry_p) / risk if risk > 0 else 0.0
        mae_r = (entry_p - mae_price) / risk if risk > 0 else 0.0
    else:
        mfe_r = (entry_p - mfe_price) / risk if risk > 0 else 0.0
        mae_r = (mae_price - entry_p) / risk if risk > 0 else 0.0

    t["mfe_r"]                  = round(max(mfe_r, 0.0), 3)
    t["mae_r"]                  = round(max(mae_r, 0.0), 3)
    t["bars_to_resolution"]     = len(bars)
    t["time_to_resolution_min"] = len(bars) * 5


def update_outcomes_from_mt5(magic: int = 234100) -> int:
    """
    Pull MT5 history deals and fill in profit/outcome for any open records.
    Returns the number of records updated.
    Requires MetaTrader5 to be initialised (call after ensure_connected).
    """
    try:
        import MetaTrader5 as mt5  # type: ignore[import]
    except ImportError:
        return 0

    trades = _load()
    open_records = [t for t in trades if t["outcome"] is None]
    if not open_records:
        return 0

    open_tickets = {t["ticket"] for t in open_records}

    # Fetch all exit deals with our magic number (entry=1 → OUT / closing deal)
    from datetime import timedelta
    since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=90)
    raw_deals = mt5.history_deals_get(since, datetime.now(timezone.utc).replace(tzinfo=None))
    if raw_deals is None:
        return 0

    # position_id on an exit deal matches the ticket of the opening order.
    # We do NOT filter by magic here — manual closes and SL/TP closes on some
    # brokers produce closing deals with magic=0 even when the opening order
    # had our magic number. We only need to match by position_id.
    closed: dict[int, dict] = {}
    for d in raw_deals:
        if d.entry != 1:   # 1 = OUT (closing deal)
            continue
        if d.position_id in open_tickets:
            # Capture close price + the broker comment so we can later tell
            # whether SL, TP, or a manual/auto-trade-manager close ended the
            # position. MT5 sets `comment` to "[sl 1.10000]" / "[tp 1.10500]"
            # / "[so:0.00]" on stop-out, etc., depending on the broker.
            closed[d.position_id] = {
                "profit":     d.profit,
                "closed_at":  datetime.utcfromtimestamp(d.time).isoformat(timespec="seconds"),
                "close_price": getattr(d, "price", 0.0) or 0.0,
                "close_comment": (getattr(d, "comment", "") or "").strip(),
            }

    if not closed:
        return 0

    updated = 0
    for t in trades:
        if t["ticket"] in closed:
            info = closed[t["ticket"]]
            t["profit"]      = round(info["profit"], 2)
            t["closed_at"]   = info["closed_at"]
            t["close_price"] = round(info["close_price"], 5)
            # Classify close reason from the broker comment + numeric proximity
            # to the recorded SL/TP. Useful for diagnostics ("did we hit our
            # planned exit, or did the trade-manager intervene?").
            t["close_reason"] = _classify_close_reason(
                t, info["close_price"], info["close_comment"],
            )
            p = info["profit"]
            t["outcome"]   = "WIN" if p > 0 else ("LOSS" if p < 0 else "BREAKEVEN")
            # ── Phase 4: MFE/MAE for live trades ─────────────────────────
            # Fetch the 5m bars during the trade's life and compute the same
            # MFE/MAE / bars_to_resolution metrics as paper trades.
            try:
                _enrich_live_trade_with_mfe_mae(t, mt5)
            except Exception as exc:
                logger.debug("mfe/mae enrichment skipped for #%s: %s",
                             t.get("ticket"), exc)
            updated += 1

    if updated:
        _save(trades)
        logger.info(f"Updated {updated} trade outcome(s) from MT5 history")
        try:
            from data.strategy_params import adapt_all
            adapted = adapt_all(trades)
            if adapted:
                logger.info(f"Adaptive learning: {adapted} strategy param(s) updated")
        except Exception as _ae:
            logger.debug(f"adapt_all skipped: {_ae}")
        # Refresh per-pattern win rates from the new closed-trade evidence
        try:
            from models.pattern_stats import refresh as _refresh_pattern_stats
            _refresh_pattern_stats()
        except Exception as _pe:
            logger.debug(f"pattern_stats refresh skipped: {_pe}")
    return updated


def compute_win_rates(min_trades: int = _MIN_TRADES) -> dict[str, float]:
    """
    Return {strategy: win_rate} for strategies with >= min_trades closed results.
    Used as backtest_win_rates in run_signal() to calibrate confidence scores.
    Result is cached by trade_log mtime + min_trades param.
    """
    mtime = _file_mtime()
    cached = _WIN_RATES_CACHE.get(min_trades)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    trades = _load()
    from collections import defaultdict
    buckets: dict[str, list[bool]] = defaultdict(list)
    for t in trades:
        if t["outcome"] not in ("WIN", "LOSS"):   # exclude BREAKEVEN from ratio
            continue
        # Exclude MT5_IMPORT — these are manual trades, not generated by the
        # signal engine, and would never match best.strategy in run_signal().
        # Keeping them out also prevents future accidental contamination.
        if t.get("strategy") == "MT5_IMPORT":
            continue
        buckets[t["strategy"]].append(t["outcome"] == "WIN")

    result = {
        s: sum(results) / len(results)
        for s, results in buckets.items()
        if len(results) >= min_trades
    }
    _WIN_RATES_CACHE[min_trades] = (mtime, result)
    return result


def compute_detailed_win_rates(min_trades: int = 3) -> dict[str, float]:
    """
    Return win rates at multiple granularities so the signal engine can look up
    the most specific match for (strategy, symbol, direction):

      "STRATEGY"                  — overall per strategy
      "STRATEGY:SYMBOL"           — per strategy + symbol
      "STRATEGY:LONG"             — per strategy + direction
      "STRATEGY:SYMBOL:LONG"      — most specific (used first)

    min_trades=3 (lower than compute_win_rates) because granular buckets fill up slower.
    Losses drag the rate down just as much as wins raise it — no special weighting needed;
    the signal engine applies an asymmetric penalty multiplier at calibration time.

    Result is cached by trade_log mtime + min_trades param. Saves the repeated
    O(N) bucketing on every poll/UI rerun.
    """
    mtime = _file_mtime()
    cached = _DETAILED_WR_CACHE.get(min_trades)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    from collections import defaultdict
    trades = _load()
    buckets: dict[str, list[bool]] = defaultdict(list)

    for t in trades:
        if t["outcome"] not in ("WIN", "LOSS"):
            continue
        # Exclude manually-imported MT5 trades from the per-(strategy, symbol,
        # direction) calibration buckets. These never match a generated signal
        # but we keep them out for defensive consistency with compute_win_rates.
        if t.get("strategy") == "MT5_IMPORT":
            continue
        win      = t["outcome"] == "WIN"
        strat    = t.get("strategy", "UNKNOWN")
        # Normalise ticker: strip yfinance suffix so EURUSD=X and EURUSD map to same key
        sym      = t.get("ticker", "").replace("=X", "").upper()
        direction = t.get("direction", "")

        buckets[strat].append(win)
        if sym:
            buckets[f"{strat}:{sym}"].append(win)
        if direction:
            buckets[f"{strat}:{direction}"].append(win)
        if sym and direction:
            buckets[f"{strat}:{sym}:{direction}"].append(win)

    result = {
        k: round(sum(v) / len(v), 4)
        for k, v in buckets.items()
        if len(v) >= min_trades
    }
    _DETAILED_WR_CACHE[min_trades] = (mtime, result)
    return result


def compute_optimal_stops(min_trades: int = 3) -> dict[str, dict]:
    """
    Mine closed trade history to find optimal SL/TP as a percentage of entry price.

    Returns a dict keyed at two granularities:
        "STRATEGY"          — pooled across all symbols
        "STRATEGY:SYMBOL"   — symbol-specific (used first when available)

    Each entry:
        sl_pct   — median SL distance as % of entry, derived from winning trades
        tp_pct   — median TP distance as % of entry, derived from winning trades
        rr       — implied risk:reward (tp_pct / sl_pct)
        win_rate — actual win rate in this bucket
        sample   — number of closed trades used

    Logic:
      • Only Katraswing-sent trades (sl != 0 and tp != 0) are used; MT5_IMPORT
        trades are excluded because they have no SL/TP recorded.
      • Median of WINNING trades determines the target distances — these are the
        stop placements that survived to profit.
      • If losing trades have a smaller median SL than winning ones, the model
        adds a 5 % buffer to reduce premature stop-outs.

    Result is cached by trade_log mtime + min_trades param. Saves the
    O(N) median computation on every poll/UI rerun.
    """
    mtime = _file_mtime()
    cached = _OPTIMAL_STOPS_CACHE.get(min_trades)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    import statistics
    from collections import defaultdict

    trades = _load()
    usable = [
        t for t in trades
        if t.get("outcome") in ("WIN", "LOSS")
        and float(t.get("sl", 0)) != 0
        and float(t.get("tp", 0)) != 0
        and float(t.get("entry", 0)) != 0
    ]

    buckets: dict[str, list[dict]] = defaultdict(list)
    for t in usable:
        entry  = float(t["entry"])
        sl_pct = abs(entry - float(t["sl"])) / entry * 100
        tp_pct = abs(float(t["tp"]) - entry) / entry * 100
        rec = {
            "win":    t["outcome"] == "WIN",
            "sl_pct": sl_pct,
            "tp_pct": tp_pct,
        }
        strat = t.get("strategy", "")
        sym   = t.get("ticker", "").replace("=X", "").upper()
        buckets[strat].append(rec)
        if sym:
            buckets[f"{strat}:{sym}"].append(rec)

    result: dict[str, dict] = {}
    for key, records in buckets.items():
        if len(records) < min_trades:
            continue
        wins   = [r for r in records if r["win"]]
        losses = [r for r in records if not r["win"]]

        source = wins if len(wins) >= min_trades else records
        sl_vals = [r["sl_pct"] for r in source]
        tp_vals = [r["tp_pct"] for r in source]

        med_sl = statistics.median(sl_vals)
        med_tp = statistics.median(tp_vals)

        # Widen SL if losing trades had tighter stops than winning ones
        if wins and losses:
            loss_sl = statistics.median([r["sl_pct"] for r in losses])
            win_sl  = statistics.median([r["sl_pct"] for r in wins])
            if loss_sl < win_sl:
                med_sl = win_sl * 1.05   # 5% extra room

        result[key] = {
            "sl_pct":   round(med_sl, 5),
            "tp_pct":   round(med_tp, 5),
            "rr":       round(med_tp / max(med_sl, 0.0001), 2),
            "win_rate": round(len(wins) / len(records), 3),
            "sample":   len(records),
        }

    _OPTIMAL_STOPS_CACHE[min_trades] = (mtime, result)
    return result


def import_all_mt5_history(days: int = 90) -> int:
    """
    Import ALL closed trades from MT5 history (any magic number / manually opened trades).

    Pairs IN deals (entry=0, opening) with OUT deals (entry=1, closing) by position_id.
    Trades already in the log have their outcome filled in if still open.
    New trades (not sent by Katraswing) are added with strategy='MT5_IMPORT'.

    Returns the number of records newly added or updated.
    """
    try:
        import MetaTrader5 as mt5  # type: ignore[import]
    except ImportError:
        return 0

    from collections import defaultdict
    from datetime import timedelta

    since    = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
    raw      = mt5.history_deals_get(since, datetime.now(timezone.utc).replace(tzinfo=None))
    if raw is None:
        return 0

    # Group deals by position_id → {in: deal, out: deal}
    by_pos: dict[int, dict] = defaultdict(lambda: {"in": None, "out": None})
    for d in raw:
        if d.entry == 0:
            by_pos[d.position_id]["in"] = d
        elif d.entry == 1:
            by_pos[d.position_id]["out"] = d

    trades          = _load()
    existing_by_tk  = {t["ticket"]: t for t in trades}
    imported        = 0

    for pos_id, pair in by_pos.items():
        out_d = pair["out"]
        in_d  = pair["in"]
        if out_d is None:
            continue   # position still open

        # Ticket of the opening order = position_id in MT5
        ticket = int(pos_id)
        gross  = round(float(out_d.profit), 2)
        comm   = round(float(getattr(out_d, "commission", 0.0)), 2)
        swap   = round(float(getattr(out_d, "swap", 0.0)), 2)
        net    = round(gross + comm + swap, 2)
        outcome = "WIN" if net > 0 else ("LOSS" if net < 0 else "BREAKEVEN")
        close_iso = datetime.utcfromtimestamp(out_d.time).isoformat(timespec="seconds")

        if ticket in existing_by_tk:
            rec = existing_by_tk[ticket]
            if rec["outcome"] is None:
                rec["profit"]    = net
                rec["closed_at"] = close_iso
                rec["outcome"]   = outcome
                # enrich with actual exit price if present
                rec.setdefault("close_price", round(float(out_d.price), 5))
                imported += 1
            continue

        direction  = "LONG" if (in_d and in_d.type == 0) else "SHORT"
        entry_p    = round(float(in_d.price), 5)  if in_d else 0.0
        open_iso   = datetime.utcfromtimestamp(in_d.time).isoformat(timespec="seconds") if in_d else close_iso
        volume     = float(in_d.volume) if in_d else float(out_d.volume)
        symbol     = str(out_d.symbol)

        trades.append({
            "ticket":      ticket,
            "ticker":      symbol,
            "strategy":    "MT5_IMPORT",
            "direction":   direction,
            "confidence":  0.0,
            "entry":       entry_p,
            "sl":          0.0,
            "tp":          0.0,
            "close_price": round(float(out_d.price), 5),
            "volume":      volume,
            "gross":       gross,
            "commission":  comm,
            "swap":        swap,
            "sent_at":     open_iso,
            "closed_at":   close_iso,
            "profit":      net,
            "outcome":     outcome,
        })
        existing_by_tk[ticket] = trades[-1]
        imported += 1

    if imported:
        _save(trades)
        logger.info(f"Imported/updated {imported} trade(s) from MT5 history ({days}d)")
    return imported


def get_summary() -> dict:
    """Return a summary dict for display in the UI."""
    trades = _load()
    closed  = [t for t in trades if t["outcome"] is not None]
    wins    = [t for t in closed  if t["outcome"] == "WIN"]
    losses  = [t for t in closed  if t["outcome"] == "LOSS"]
    total_profit = sum(t["profit"] for t in closed if t["profit"] is not None)

    from collections import defaultdict
    by_strategy: dict[str, dict] = defaultdict(lambda: {"trades": 0, "wins": 0, "profit": 0.0})
    for t in closed:
        s = t["strategy"]
        by_strategy[s]["trades"] += 1
        if t["outcome"] == "WIN":
            by_strategy[s]["wins"] += 1
        if t["profit"] is not None:
            by_strategy[s]["profit"] += t["profit"]

    strategy_stats = []
    for s, v in sorted(by_strategy.items(), key=lambda x: -x[1]["trades"]):
        wr = v["wins"] / v["trades"] if v["trades"] else 0
        strategy_stats.append({
            "strategy": s,
            "trades":   v["trades"],
            "win_rate": round(wr, 3),
            "profit":   round(v["profit"], 2),
        })

    return {
        "total_sent":   len(trades),
        "total_closed": len(closed),
        "total_open":   len(trades) - len(closed),
        "wins":         len(wins),
        "losses":       len(losses),
        "win_rate":     round(len(wins) / len(closed), 3) if closed else None,
        "total_profit": round(total_profit, 2),
        "by_strategy":  strategy_stats,
        "all_trades":   list(reversed(trades)),  # newest first
    }
