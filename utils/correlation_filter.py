"""
Correlation filter — blocks redundant position exposure.

When two instruments move together (e.g. EUR/USD + GBP/USD both LONG = two
bets on the same USD weakness), opening both wastes margin and doubles risk
on a single thesis.

Groups:
  USD_SELLERS   EURUSD, GBPUSD, AUDUSD          -- all sell USD when LONG
  JPY_CROSSES   GBPJPY, EURJPY                   -- both sell JPY when LONG
  METALS        Gold, Silver                      -- precious metals
  US_INDICES    YM (Dow), ES (S&P 500)           -- US equity direction
  OIL           WTI (CL), Brent (BZ)             -- crude oil price

Rule: if an open MT5 position already exists in the same group AND same
direction, the new signal is blocked. Opposite direction = allowed (hedge).

Usage:
    from utils.correlation_filter import is_correlated_duplicate
    blocked, reason = is_correlated_duplicate("EURUSD=X", "LONG", open_positions)
    if blocked:
        log.info(f"  [corr] {reason}")
        continue
"""

from __future__ import annotations

# ── Correlation group for each yfinance ticker ───────────────────────────────
_TICKER_GROUP: dict[str, str] = {
    "EURUSD=X": "USD_SELLERS",
    "GBPUSD=X": "USD_SELLERS",
    "AUDUSD=X": "USD_SELLERS",
    "GBPJPY=X": "JPY_CROSSES",
    "EURJPY=X": "JPY_CROSSES",
    "GC=F":     "METALS",
    "SI=F":     "METALS",
    "YM=F":     "US_INDICES",
    "ES=F":     "US_INDICES",
    "NQ=F":     "US_INDICES",
    "CL=F":     "OIL",
    "BZ=F":     "OIL",
}

# ── Map an MT5 broker symbol to the same group keys ──────────────────────────

def _mt5_group(symbol: str) -> str | None:
    """Return the correlation group for an MT5 broker symbol, or None."""
    s = symbol.upper().replace(".", "").replace("_", "").replace("-", "")

    # Forex — exact prefix match handles variants like EURUSD, EURUSDP, EURUSDpro
    if s.startswith("EURUSD"): return "USD_SELLERS"
    if s.startswith("GBPUSD"): return "USD_SELLERS"
    if s.startswith("AUDUSD"): return "USD_SELLERS"
    if s.startswith("GBPJPY"): return "JPY_CROSSES"
    if s.startswith("EURJPY"): return "JPY_CROSSES"

    # Gold — various broker names
    if s in ("GOLD", "XAUUSD", "XAUUSDM") or s.startswith("GOLD") or s.startswith("XAUUSD"):
        return "METALS"

    # Silver
    if s in ("SILVER", "XAGUSD") or s.startswith("SILVER") or s.startswith("XAGUSD"):
        return "METALS"

    # US indices — catches #US30_M26, #US30_U26, US30, DOWJONES etc.
    if any(x in s for x in ("US30", "US500", "YM", "DOWJONES", "DOW30")):
        return "US_INDICES"
    if any(x in s for x in ("US500", "SP500", "SPX", "ES")):
        return "US_INDICES"

    # Oil — WTI
    if s.startswith("CL") or any(x in s for x in ("WTIUSD", "USOIL", "OIL", "CRUDEOIL")):
        return "OIL"

    # Oil — Brent
    if s.startswith("BZ") or any(x in s for x in ("BRENT", "XBRUSD", "UKOIL", "BRENTOIL")):
        return "OIL"

    return None


def is_correlated_duplicate(
    ticker: str,
    direction: str,
    open_positions: list,
) -> tuple[bool, str]:
    """
    Check whether `open_positions` already contains a position in the same
    correlation group and same direction as the new signal.

    Returns ``(True, reason_string)`` if blocked, ``(False, "")`` if allowed.
    """
    new_group = _TICKER_GROUP.get(ticker)
    if new_group is None:
        return False, ""  # unknown ticker — no correlation data, allow through

    for pos in open_positions:
        existing_group = _mt5_group(pos.symbol)
        if existing_group != new_group:
            continue
        if pos.direction != direction:
            continue  # opposite direction — hedge, allow it
        return (
            True,
            f"{ticker} {direction} blocked: correlated with open "
            f"#{pos.ticket} {pos.symbol} {pos.direction} "
            f"(group={new_group})",
        )

    return False, ""
