"""Per-symbol live/paper/drop policy.

Symbol-level disposition decided 2026-05-02 from the 86-trade forensic:

  LIVE   — allowed to send orders to the broker
  PAPER  — signal flows through calibration; order send is skipped
  DROP   — strategy fire is converted to FLAT before reaching MT5

A symbol's normalised key strips yfinance suffixes ("=X", "=F") and uppercases
so EURUSD=X, EURUSD, and #EURUSD all hit the same bucket.
"""
from __future__ import annotations

from typing import Literal

Disposition = Literal["LIVE", "PAPER", "DROP"]

# Forensic findings (excludes MT5_IMPORT manual trades):
#   Profitable: YM=F (3T 100%, +$29), SI=F (7T 71%, +$13), USDJPY=X (9T 44%, +$14)
#   Marginal:   GBPUSD=X (8T 38%, -$30), EURUSD=X (8T 25%, -$18), AUDUSD=X (6T 17%, -$21)
#   Toxic:      NQ=F (3T 0%, -$58)
LIVE_SYMBOLS: set[str] = {
    "YM", "SI", "USDJPY", "ES",        # all > 40% WR live
    # ES is included because 2T/100% — small sample but adjacent to YM/SI cluster
}

PAPER_SYMBOLS: set[str] = {
    "EURUSD", "GBPUSD", "AUDUSD",      # 17–38% WR — marginal; let MSS/H1 warm up
    "USDCAD", "GBPJPY", "EURJPY",      # tiny samples — don't risk live until evidence
}

DROP_SYMBOLS: set[str] = {
    "NQ",   # 0/3 live, -$58. Revisit in Phase C with spread-aware backtester.
}


def _normalise(symbol: str) -> str:
    if not symbol:
        return ""
    s = symbol.replace("=X", "").replace("=F", "").upper()
    # Some brokers prefix with '#' ('#US100_M26', etc.); strip for policy lookup.
    if s.startswith("#"):
        s = s[1:].split("_")[0]
    return s


def get_disposition(symbol: str) -> Disposition:
    """Return the trading disposition for the given symbol.

    Default for unmapped symbols is LIVE — the policy is a kill-list, not an
    opt-in allowlist. Symbols not explicitly demoted continue to trade live.
    """
    sym = _normalise(symbol)
    if sym in DROP_SYMBOLS:
        return "DROP"
    if sym in PAPER_SYMBOLS:
        return "PAPER"
    return "LIVE"


def is_live(symbol: str) -> bool:
    return get_disposition(symbol) == "LIVE"


def is_paper(symbol: str) -> bool:
    return get_disposition(symbol) == "PAPER"


def is_dropped(symbol: str) -> bool:
    return get_disposition(symbol) == "DROP"
