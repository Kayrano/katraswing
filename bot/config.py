"""
Bot configuration — all trading parameters live here.
Edit these values to change bot behaviour without touching engine code.
"""

# ── Portfolio ──────────────────────────────────────────────────────────────────
PORTFOLIO_SIZE        = 100_000.0   # Total paper trading account ($)
RISK_PER_TRADE_PCT    = 0.015       # Risk 1.5% of portfolio per trade
MAX_POSITIONS         = 5           # Maximum simultaneous open positions
MAX_POSITION_SIZE_PCT = 0.25        # Single position can't exceed 25% of portfolio

# ── Score Thresholds ───────────────────────────────────────────────────────────
BUY_THRESHOLD         = 60          # Composite score ≥ 60 → BUY signal
AVOID_THRESHOLD       = 35          # Composite score < 35 → EXIT / AVOID
# Scores between 35–59 are HOLD — keep existing positions, no new entries

# ── Risk Management ───────────────────────────────────────────────────────────
DAILY_LOSS_LIMIT_PCT  = 0.03        # Halt trading for the day after −3% P&L
EARNINGS_DAYS_SKIP    = 3           # Don't open trades within N days of earnings

# ── Scheduling ────────────────────────────────────────────────────────────────
SCAN_INTERVAL_MINUTES = 15          # Full scan cycle frequency (minutes)
# Note: bot also checks market hours — no trades placed when market is closed.

# ── Pre-screening ──────────────────────────────────────────────────────────────
# Screener runs first (fast), then full analysis only on top N candidates.
# This keeps cycle time under 2–3 minutes for a 40-stock universe.
TOP_CANDIDATES        = 8           # Run full analysis on this many top screener hits

# ── Stock Universe ─────────────────────────────────────────────────────────────
# The screener uses its own 40-stock SWING_UNIVERSE (data/screener.py).
# This list is used only for checking existing positions that aren't in the screener.
WATCHLIST_EXTRAS = [
    "PLTR", "ARM", "MSTR", "COIN", "HOOD",
]
