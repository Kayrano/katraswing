"""
Risk management layer.
All pre-trade checks live here — keeps engine.py clean.
"""

import yfinance as yf
import pandas as pd
from datetime import date, timedelta

from bot.config import (
    PORTFOLIO_SIZE,
    RISK_PER_TRADE_PCT,
    MAX_POSITIONS,
    MAX_POSITION_SIZE_PCT,
    DAILY_LOSS_LIMIT_PCT,
    EARNINGS_DAYS_SKIP,
)


def compute_position_size(entry_price: float, stop_loss: float) -> int:
    """
    Kelly-inspired fixed-fractional sizing:
      shares = (portfolio × risk_pct) / |entry − stop|

    Caps at MAX_POSITION_SIZE_PCT × portfolio.
    Returns 0 if inputs are invalid.
    """
    risk_per_share = abs(entry_price - stop_loss)
    if risk_per_share <= 0 or entry_price <= 0:
        return 0

    dollar_risk  = PORTFOLIO_SIZE * RISK_PER_TRADE_PCT
    shares       = int(dollar_risk / risk_per_share)

    max_by_value = int((PORTFOLIO_SIZE * MAX_POSITION_SIZE_PCT) / entry_price)
    shares       = min(shares, max_by_value)
    return max(0, shares)


def can_open_new_position(current_positions: list) -> tuple:
    """
    Returns (True, 'OK') or (False, reason_string).
    Checks:
      - Position count limit
      - Portfolio deployment ceiling
    """
    if len(current_positions) >= MAX_POSITIONS:
        return False, f"Max {MAX_POSITIONS} positions already open"

    total_deployed = sum(abs(float(p.get("market_value", 0))) for p in current_positions)
    ceiling        = PORTFOLIO_SIZE * 0.80   # never deploy more than 80%
    if total_deployed >= ceiling:
        return False, f"Portfolio {total_deployed/PORTFOLIO_SIZE*100:.0f}% deployed (ceiling 80%)"

    return True, "OK"


def is_daily_loss_limit_hit(daily_pnl: float) -> bool:
    """True if today's P&L has breached the configured loss limit."""
    return daily_pnl <= -(PORTFOLIO_SIZE * DAILY_LOSS_LIMIT_PCT)


def has_upcoming_earnings(ticker: str) -> tuple:
    """
    Returns (True, days_until) if earnings are within EARNINGS_DAYS_SKIP days.
    Returns (False, -1) if safe or data unavailable.
    """
    try:
        t       = yf.Ticker(ticker)
        now_utc = pd.Timestamp.now(tz="UTC")

        # Primary: earnings_dates
        try:
            edates = t.earnings_dates
            if edates is not None and not edates.empty:
                future = edates[edates.index >= now_utc]
                if not future.empty:
                    earn_date = future.index[-1].date()
                    days = (earn_date - date.today()).days
                    if 0 <= days <= EARNINGS_DAYS_SKIP:
                        return True, days
        except Exception:
            pass

        # Fallback: calendar
        try:
            cal = t.calendar
            if cal is not None and isinstance(cal, pd.DataFrame) and not cal.empty:
                for col in cal.columns:
                    try:
                        d = pd.to_datetime(cal[col].iloc[0]).date()
                        today = date.today()
                        if today <= d <= today + timedelta(days=EARNINGS_DAYS_SKIP):
                            days = (d - today).days
                            return True, days
                    except Exception:
                        pass
        except Exception:
            pass

    except Exception:
        pass

    return False, -1


def already_in_portfolio(ticker: str, current_positions: list) -> bool:
    held = {p.get("symbol", "").upper() for p in current_positions}
    return ticker.upper() in held
