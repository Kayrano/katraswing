"""Realistic slippage / spread-widening model for backtesting.

The Round 4 audit established that backtest-to-live PF degradation is
typically 10–20% in retail FX, driven by:

  - Wider spreads at session opens / closes (volatility = wider book)
  - Spreads expanding during high-impact news windows
  - Different baseline spreads per asset class (FX < indices < crypto)

The previous backtester used a flat 0.02% slippage which under-modelled all
three. This module replaces it with a per-symbol baseline plus a multiplier
for the bar's context (session boundary / event window / quiet period).

Used only by `agents.intraday_backtester` — live trading uses MT5's actual
fill prices, so this only matters for offline simulation.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


# Per-asset-class baseline ONE-WAY slippage (price-pct equivalent). These are
# rough estimates for retail spreads; adjust if your broker is unusually
# tight or wide. Side note: brokers that publish "1.0 pip EUR/USD" mean
# round-trip 1.0 pip; one-way is 0.5 pip ≈ 0.00005 ÷ 1.10 ≈ 0.0045% — round
# up to 0.05% per side to model variable spreads.
_BASELINE_SLIPPAGE: dict[str, float] = {
    "MAJOR_FX":   0.00050,   # EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD…
    "MINOR_FX":   0.00080,   # EURJPY, GBPJPY, AUDNZD…
    "EXOTIC_FX":  0.00150,
    "INDEX":      0.00080,   # NQ, ES, YM, DAX, FTSE — futures CFDs
    "METALS":     0.00100,   # XAUUSD, XAGUSD, GC, SI
    "ENERGY":     0.00120,   # CL (oil), NG (gas)
    "CRYPTO":     0.00250,   # BTC, ETH, SOL — wide retail spreads
    "DEFAULT":    0.00100,
}


def _classify_symbol(ticker: str) -> str:
    """Map any Katraswing/MT5 ticker to its asset class for spread lookup."""
    if not ticker:
        return "DEFAULT"
    s = ticker.upper().replace("=X", "").replace("=F", "")
    if s.startswith("#"):
        s = s[1:].split("_")[0]

    # Crypto first (suffixed with USD usually)
    if any(s.startswith(c) for c in ("BTC", "ETH", "SOL", "XRP", "BNB", "DOGE")):
        return "CRYPTO"
    # Metals
    if s.startswith(("XAU", "XAG", "GC", "SI", "GOLD", "SILVER", "PL", "PA")):
        return "METALS"
    # Energy
    if s.startswith(("CL", "NG", "BZ", "RB", "HO", "WTI", "BRENT", "OIL", "NAT.GAS", "XTI", "XBR", "XNG")):
        return "ENERGY"
    # Indices
    if any(tag in s for tag in (
        "NQ", "ES", "YM", "RTY", "US100", "US500", "US30",
        "GER40", "UK100", "FRA40", "ESP35", "EU50", "JP225", "AUS200", "HK50",
        "NAS100", "SPX500", "DAX",
    )):
        return "INDEX"
    # Major FX (6-char pair containing USD/EUR/GBP/JPY)
    majors = {"EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"}
    if s in majors or any(s.startswith(m) for m in majors):
        return "MAJOR_FX"
    # Generic 6-char alphanumeric pair → minor FX
    if len(s) == 6 and s.isalpha():
        return "MINOR_FX"
    return "DEFAULT"


def baseline_slippage(ticker: str) -> float:
    """One-way slippage as a fraction of price for `ticker`."""
    return _BASELINE_SLIPPAGE.get(_classify_symbol(ticker), _BASELINE_SLIPPAGE["DEFAULT"])


def slippage_at_bar(
    ticker: str,
    bar: Optional[pd.Series] = None,
    in_event_window: bool = False,
) -> float:
    """Return the effective one-way slippage for `ticker` at this bar.

    Multipliers:
      - Session-boundary widening (first/last 3 bars of a session): 1.5×
      - In-event window (caller-supplied flag): 2.0×
      - Both stack: max(session, event) → 2.0× cap

    Args:
        ticker:           the bar's symbol.
        bar:              the row from the OHLCV DataFrame; uses
                          `session_bar_number` and `is_first_bar` if present.
        in_event_window:  True if a high-impact economic release is within
                          ±15m of this bar (caller computes via
                          `data.economic_calendar.is_event_window`).

    Returns:
        One-way slippage as a price fraction (e.g. 0.0010 = 10 bps).
    """
    base = baseline_slippage(ticker)
    multiplier = 1.0

    if bar is not None:
        bar_num = bar.get("session_bar_number") if hasattr(bar, "get") else None
        is_first = bool(bar.get("is_first_bar")) if hasattr(bar, "get") else False
        # Treat first 3 bars of a session as session-open wide; we don't have
        # session-end markers in the OHLCV (would need `is_last_bar`), so
        # session-close widening is approximated only when bar_num >= 75
        # for US 5m sessions (78 bars/session).
        if is_first or (bar_num is not None and 1 <= bar_num <= 3):
            multiplier = max(multiplier, 1.5)
        elif bar_num is not None and bar_num >= 75:
            multiplier = max(multiplier, 1.5)

    if in_event_window:
        multiplier = max(multiplier, 2.0)

    return base * multiplier
