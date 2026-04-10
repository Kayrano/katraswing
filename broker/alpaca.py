"""
Alpaca Markets REST API wrapper.

Credentials are resolved in this priority order:
  1. Explicit (api_key, secret_key) arguments passed to each function.
  2. st.session_state (set after the user saves their keys in the bot tab).
  3. Fallback constants below (for local dev / backward compat).
"""

import requests
from typing import Optional

# ── Fallback credentials (local dev only — overridden by user keys on cloud) ──
_FALLBACK_KEY    = "REPLACE_WITH_YOUR_KEY"
_FALLBACK_SECRET = "REPLACE_WITH_YOUR_SECRET"
_PAPER_BASE_URL  = "https://paper-api.alpaca.markets/v2"
_LIVE_BASE_URL   = "https://api.alpaca.markets/v2"

_TIMEOUT = 12   # seconds per request


# ── Credential resolution ─────────────────────────────────────────────────────

def _resolve(api_key: str | None, secret_key: str | None, is_paper: bool | None):
    """Pick the best available credentials and return (key, secret, base_url)."""
    # Try session state (user's saved keys from the bot tab)
    if not api_key or not secret_key:
        try:
            import streamlit as st
            api_key   = api_key   or st.session_state.get("alpaca_api_key")
            secret_key = secret_key or st.session_state.get("alpaca_secret_key")
            if is_paper is None:
                is_paper = st.session_state.get("alpaca_is_paper", True)
        except Exception:
            pass

    key    = api_key   or _FALLBACK_KEY
    secret = secret_key or _FALLBACK_SECRET
    paper  = is_paper if is_paper is not None else True
    base   = _PAPER_BASE_URL if paper else _LIVE_BASE_URL
    return key, secret, base.rstrip("/")


def _headers(api_key: str | None = None, secret_key: str | None = None, is_paper: bool | None = None) -> tuple[dict, str]:
    """Return (headers_dict, base_url)."""
    key, secret, base = _resolve(api_key, secret_key, is_paper)
    headers = {
        "APCA-API-KEY-ID":     key,
        "APCA-API-SECRET-KEY": secret,
        "Content-Type":        "application/json",
    }
    return headers, base


# ── Account ───────────────────────────────────────────────────────────────────

def get_account(api_key=None, secret_key=None, is_paper=None) -> dict:
    h, base = _headers(api_key, secret_key, is_paper)
    r = requests.get(f"{base}/account", headers=h, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def is_market_open(api_key=None, secret_key=None, is_paper=None) -> bool:
    try:
        h, base = _headers(api_key, secret_key, is_paper)
        r = requests.get(f"{base}/clock", headers=h, timeout=_TIMEOUT)
        r.raise_for_status()
        return bool(r.json().get("is_open", False))
    except Exception:
        return False


# ── Positions ─────────────────────────────────────────────────────────────────

def get_positions(api_key=None, secret_key=None, is_paper=None) -> list:
    h, base = _headers(api_key, secret_key, is_paper)
    r = requests.get(f"{base}/positions", headers=h, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_position(symbol: str, api_key=None, secret_key=None, is_paper=None) -> Optional[dict]:
    try:
        h, base = _headers(api_key, secret_key, is_paper)
        r = requests.get(f"{base}/positions/{symbol.upper()}", headers=h, timeout=_TIMEOUT)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def close_position(symbol: str, api_key=None, secret_key=None, is_paper=None) -> Optional[dict]:
    try:
        h, base = _headers(api_key, secret_key, is_paper)
        r = requests.delete(f"{base}/positions/{symbol.upper()}", headers=h, timeout=_TIMEOUT)
        if r.status_code in (200, 204, 207):
            return r.json() if r.content else {}
        return None
    except Exception:
        return None


# ── Orders ────────────────────────────────────────────────────────────────────

def get_open_orders(api_key=None, secret_key=None, is_paper=None) -> list:
    h, base = _headers(api_key, secret_key, is_paper)
    r = requests.get(f"{base}/orders?status=open", headers=h, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_order(order_id: str, api_key=None, secret_key=None, is_paper=None) -> dict:
    h, base = _headers(api_key, secret_key, is_paper)
    r = requests.get(f"{base}/orders/{order_id}", headers=h, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def place_market_order(symbol: str, qty: int, side: str,
                       api_key=None, secret_key=None, is_paper=None) -> dict:
    h, base = _headers(api_key, secret_key, is_paper)
    payload = {
        "symbol":        symbol.upper(),
        "qty":           str(qty),
        "side":          side,
        "type":          "market",
        "time_in_force": "day",
    }
    r = requests.post(f"{base}/orders", headers=h, json=payload, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def place_oco_order(symbol: str, qty: int, stop_price: float, take_profit_price: float,
                    api_key=None, secret_key=None, is_paper=None) -> dict:
    h, base = _headers(api_key, secret_key, is_paper)
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
    r = requests.post(f"{base}/orders", headers=h, json=payload, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def cancel_order(order_id: str, api_key=None, secret_key=None, is_paper=None) -> bool:
    try:
        h, base = _headers(api_key, secret_key, is_paper)
        r = requests.delete(f"{base}/orders/{order_id}", headers=h, timeout=_TIMEOUT)
        return r.status_code in (200, 204)
    except Exception:
        return False


def cancel_all_orders(api_key=None, secret_key=None, is_paper=None) -> None:
    try:
        h, base = _headers(api_key, secret_key, is_paper)
        requests.delete(f"{base}/orders", headers=h, timeout=_TIMEOUT)
    except Exception:
        pass
