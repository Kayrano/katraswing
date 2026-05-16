"""
One-time script: backfill historical MT5 trades into data/trade_log.json.

Run ONCE on the VPS from C:\\katraswing:
    python scripts/backfill_trades.py

What it does:
  1. Connects to MetaTrader 5
  2. Pulls all deals with magic=234100 going back LOOKBACK_DAYS
  3. Pairs each opening deal (IN) with its closing deal (OUT) by position_id
  4. Writes a trade_log entry for every ticket not already in the log
  5. Tags each entry strategy="MT5_IMPORT" (no strategy name available)
  6. Immediately triggers adapt_all + pattern_stats so learning starts

Limitation: strategy-level adaptation (adapt_all) is skipped for
MT5_IMPORT rows because the strategy name was never recorded. However
these trades DO feed per-symbol win rates, symbol promotion decisions,
and daily/nightly P&L summaries.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MAGIC        = 234100
LOG_PATH     = ROOT / "data" / "trade_log.json"
LOOKBACK_DAYS = 365   # go back 1 year; broker may cap this at 90 days


def main() -> None:
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("MetaTrader5 package not installed.  Run: pip install MetaTrader5")
        sys.exit(1)

    if not mt5.initialize():
        print(f"MT5 initialize() failed: {mt5.last_error()}")
        sys.exit(1)

    acct = mt5.account_info()
    print(f"Connected: {acct.server}  login={acct.login}")

    # ── Load existing log to skip already-recorded tickets ────────────────
    existing_tickets: set[int] = set()
    trades: list[dict] = []
    if LOG_PATH.exists():
        try:
            trades = json.loads(LOG_PATH.read_text(encoding="utf-8"))
            existing_tickets = {t["ticket"] for t in trades}
            print(f"Existing trade_log: {len(trades)} entries")
        except Exception as exc:
            print(f"Warning: could not read trade_log.json: {exc} — starting fresh")
            trades = []

    # ── Fetch deal history ────────────────────────────────────────────────
    since = datetime.now() - timedelta(days=LOOKBACK_DAYS)
    until = datetime.now()
    raw = mt5.history_deals_get(since, until)
    if raw is None:
        print(f"history_deals_get failed: {mt5.last_error()}")
        mt5.shutdown()
        sys.exit(1)

    print(f"Total deals in MT5 history (all magic numbers): {len(raw)}")

    # Separate IN (entry=0, opening) and OUT (entry=1, closing) deals
    in_deals:  dict[int, object] = {}   # position_id -> deal
    out_deals: dict[int, object] = {}   # position_id -> deal

    for d in raw:
        if d.magic != MAGIC:
            continue
        if d.entry == 0:
            in_deals[d.position_id] = d
        elif d.entry == 1:
            out_deals[d.position_id] = d

    print(f"Katraswing (magic={MAGIC})  IN={len(in_deals)}  OUT={len(out_deals)}")

    # ── Build new entries ─────────────────────────────────────────────────
    new_entries: list[dict] = []
    skipped_existing = 0

    for pos_id, in_d in in_deals.items():
        if pos_id in existing_tickets:
            skipped_existing += 1
            continue

        direction  = "LONG" if in_d.type == 0 else "SHORT"
        entry_price = round(float(in_d.price), 5)
        open_time   = datetime.utcfromtimestamp(in_d.time).isoformat(timespec="seconds")

        row: dict = {
            "ticket":       pos_id,
            "ticker":       in_d.symbol,
            "strategy":     "MT5_IMPORT",
            "direction":    direction,
            "confidence":   None,
            "entry":        entry_price,
            "stop_loss":    0.0,
            "take_profit":  0.0,
            "volume":       round(float(in_d.volume), 4),
            "patterns":     [],
            "sent_at":      open_time,
            "closed_at":    None,
            "profit":       None,
            "outcome":      None,
            "close_price":  None,
            "close_reason": None,
            "adx_value":    None,
            "atr_value":    None,
            "h1_trend":     None,
        }

        if pos_id in out_deals:
            out_d  = out_deals[pos_id]
            profit = round(float(out_d.profit), 2)
            row.update({
                "closed_at":    datetime.utcfromtimestamp(out_d.time).isoformat(timespec="seconds"),
                "profit":       profit,
                "close_price":  round(float(out_d.price), 5),
                "outcome":      "WIN" if profit > 0 else ("LOSS" if profit < 0 else "BREAKEVEN"),
                "close_reason": "TP_LIKELY" if profit > 0 else "SL_LIKELY",
            })

        new_entries.append(row)

    print(f"Skipped (already in log): {skipped_existing}")

    if not new_entries:
        print("Nothing new to import.")
        mt5.shutdown()
        return

    closed = sum(1 for e in new_entries if e["outcome"] is not None)
    still_open = len(new_entries) - closed
    wins  = sum(1 for e in new_entries if e["outcome"] == "WIN")
    losses = sum(1 for e in new_entries if e["outcome"] == "LOSS")

    print(f"\nImporting {len(new_entries)} trades:")
    print(f"  Closed: {closed}  (W={wins} L={losses})")
    print(f"  Still open / no close deal: {still_open}")

    # ── Save ──────────────────────────────────────────────────────────────
    trades.extend(new_entries)
    trades.sort(key=lambda t: t.get("sent_at") or "")
    LOG_PATH.write_text(json.dumps(trades, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved {len(trades)} total entries → {LOG_PATH}")

    # ── Trigger learning refresh ──────────────────────────────────────────
    print("\nRunning learning refresh...")

    try:
        from data.strategy_params import adapt_all
        updated = adapt_all(trades)
        print(f"  adapt_all: {updated} strategy buckets updated")
    except Exception as exc:
        print(f"  adapt_all failed: {exc}")

    try:
        from models.pattern_stats import refresh as refresh_patterns
        refresh_patterns()
        print("  pattern_stats: refreshed")
    except Exception as exc:
        print(f"  pattern_stats failed: {exc}")

    try:
        from models.calibration import reset_singleton, get_calibrator
        reset_singleton()
        cal = get_calibrator()
        print(f"  calibration: refit  samples={getattr(cal, 'sample_count', '?')}")
    except Exception as exc:
        print(f"  calibration failed: {exc}")

    mt5.shutdown()

    print("\n✓ Backfill complete.")
    print("  MT5_IMPORT trades feed: symbol win rates, P&L summaries, nightly reports.")
    print("  Strategy-level adaptation (adapt_all) skips them — strategy name unknown.")
    print("  Future trades placed by the server will have full strategy metadata.")


if __name__ == "__main__":
    main()
