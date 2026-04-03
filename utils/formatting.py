"""Formatting utilities for numbers, currencies, and percentages."""


def fmt_price(value: float) -> str:
    """Format a price with 2 decimal places."""
    if value is None:
        return "N/A"
    return f"${value:,.2f}"


def fmt_pct(value: float, sign: bool = True) -> str:
    """Format a percentage with sign."""
    if value is None:
        return "N/A"
    prefix = "+" if sign and value > 0 else ""
    return f"{prefix}{value:.2f}%"


def fmt_market_cap(value: float) -> str:
    """Format market cap in T/B/M shorthand."""
    if not value:
        return "N/A"
    if value >= 1e12:
        return f"${value / 1e12:.2f}T"
    if value >= 1e9:
        return f"${value / 1e9:.2f}B"
    if value >= 1e6:
        return f"${value / 1e6:.2f}M"
    return f"${value:,.0f}"


def score_color(score: float) -> str:
    """Return a CSS color for a trade score."""
    if score >= 80:  return "#00c851"   # green
    if score >= 65:  return "#33b5e5"   # blue
    if score >= 50:  return "#ffbb33"   # amber
    if score >= 35:  return "#ff8800"   # orange
    return "#ff4444"                    # red


def direction_color(direction: str) -> str:
    if direction == "LONG":     return "#00c851"
    if direction == "SHORT":    return "#ff4444"
    return "#888888"
