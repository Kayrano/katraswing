"""
Alpaca Markets REST API wrapper.
Reads credentials from alpaca_config.py (ClaudeTrading folder) or falls back
to the constants below.

Paper trading endpoint: https://paper-api.alpaca.markets/v2
"""

import requests
import sys
import os
from typing import Optional

# ── Load credentials ───────────────────────────────────────────────────────────
# Try to import from the ClaudeTrading config file first
try:
    _cfg_path = r"C:\Users\Kayra\OneDrive\Masaüstü\ClaudeTrading"
    if _cfg_path not in sys.path:
        sys.path.insert(0, _cfg_path)
    from alpaca_config import ALPACA_API_KEY as _KEY, ALPACA_SECRET_KEY as _SECRET, ALPACA_BASE_URL as _BASE
except ImportError:
    _KEY    = "PKRDQJ3JAALM3BS7RCNRMYYXYY"
    _SECRET = "B1GjKfqebMGQmV5mMf2TjDWM7VyEYWy7MaoV3u5j59Q5"
    _BASE   = "https://paper-api.alpaca.markets/v2"

API_KEY    = _KEY
SECRET_KEY = _SECRET
BASE_URL   = _BASE.rstrip("/")

HEADERS = {
    "APCA-API-KEY-ID":     API_KEY,
    "APCA-API-SECRET-KEY": SECRET_KEY,
    "Content-Type":        "application/json",
}

_TIMEOUT = 12   # seconds per request


# ── Account ───────────────────────────────────────────────────────────────────

def get_account() -> dict:
    r = requests.get(f"{BASE_URL}/account", headers=HEADERS, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def is_market_open() -> bool:
    """Check live market status via Alpaca clock endpoint."""
    try:
        r = requests.get(f"{BASE_URL}/clock", headers=HEADERS, timeout=_TIMEOUT)
        r.raise_for_status()
        return bool(r.json().get("is_open", False))
    except Exception:
        return False


# ── Positions ─────────────────────────────────────────────────────────────────

def get_positions() -> list:
    r = requests.get(f"{BASE_URL}/positions", headers=HEADERS, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_position(symbol: str) -> Optional[dict]:
    try:
        r = requests.get(f"{BASE_URL}/positions/{symbol.upper()}", headers=HEADERS, timeout=_TIMEOUT)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def close_position(symbol: str) -> Optional[dict]:
    """Market-sell the entire position for a ticker."""
    try:
        r = requests.delete(
            f"{BASE_URL}/positions/{symbol.upper()}",
            headers=HEADERS,
            timeout=_TIMEOUT,
        )
        if r.status_code in (200, 204, 207):
            return r.json() if r.content else {}
        return None
    except Exception:
        return None


# ── Orders ────────────────────────────────────────────────────────────────────

def get_open_orders() -> list:
    r = requests.get(f"{BASE_URL}/orders?status=open", headers=HEADERS, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_order(order_id: str) -> dict:
    r = requests.get(f"{BASE_URL}/orders/{order_id}", headers=HEADERS, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def place_market_order(symbol: str, qty: int, side: str) -> dict:
    """
    Place a market order.
    side: 'buy' | 'sell'
    """
    payload = {
        "symbol":        symbol.upper(),
        "qty":           str(qty),
        "side":          side,
        "type":          "market",
        "time_in_force": "day",
    }
    r = requests.post(f"{BASE_URL}/orders", headers=HEADERS, json=payload, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def place_oco_order(
    symbol: str,
    qty: int,
    stop_price: float,
    take_profit_price: float,
) -> dict:
    """
    Place an OCO sell order linking a stop-loss and a take-profit.
    When one leg triggers, the other is automatically cancelled.
    """
    payload = {
        "symbol":        symbol.upper(),
        "qty":           str(qty),
        "side":          "sell",
        "type":          "limit",
        "time_in_force": "gtc",
        "order_class":   "oco",
        "stop_loss":     {"stop_price":  str(round(stop_price, 2))},
        "take_profit":   {"limit_price": str(round(take_profit_price, 2))},
    }
    r = requests.post(f"{BASE_URL}/orders", headers=HEADERS, json=payload, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def cancel_order(order_id: str) -> bool:
    try:
        r = requests.delete(f"{BASE_URL}/orders/{order_id}", headers=HEADERS, timeout=_TIMEOUT)
        return r.status_code in (200, 204)
    except Exception:
        return False


def cancel_all_orders() -> None:
    try:
        requests.delete(f"{BASE_URL}/orders", headers=HEADERS, timeout=_TIMEOUT)
    except Exception:
        pass
