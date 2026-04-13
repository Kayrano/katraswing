"""
Credential helpers — no authentication, single-user app.
Alpaca keys are read from (in priority order):
  1. st.session_state  (set by Settings tab, cleared on page refresh)
  2. data/alpaca_creds.json  (persisted by Settings tab, survives page refresh)
  3. st.secrets / environment variables
  4. bot/config.py hardcoded fallback
"""

import json
import os
import streamlit as st

_ALPACA_KEY        = "alpaca_api_key"
_ALPACA_SECRET_KEY = "alpaca_secret_key"
_ALPACA_IS_PAPER   = "alpaca_is_paper"

# Persistent credentials file — written by Settings tab, read on every load
_CREDS_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "alpaca_creds.json")


def _secret(key: str) -> str:
    """Read from st.secrets first, then os.environ."""
    try:
        return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, "")


def _load_creds_file() -> dict:
    """Load persisted credentials from file. Returns {} on any error."""
    try:
        with open(_CREDS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_creds_file(api_key: str, secret_key: str, is_paper: bool) -> None:
    """Persist credentials to file so they survive page refreshes."""
    try:
        os.makedirs(os.path.dirname(_CREDS_FILE), exist_ok=True)
        with open(_CREDS_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "alpaca_api_key":    api_key,
                "alpaca_secret_key": secret_key,
                "alpaca_is_paper":   is_paper,
            }, f)
    except Exception:
        pass


def clear_creds_file() -> None:
    """Remove the persisted credentials file."""
    try:
        if os.path.exists(_CREDS_FILE):
            os.remove(_CREDS_FILE)
    except Exception:
        pass


def render_auth_gate():
    """No-op — authentication removed."""
    pass


def get_current_user() -> dict:
    """Always returns a single owner user."""
    return {"id": "owner", "email": ""}


def get_alpaca_creds() -> tuple[str, str, bool]:
    """
    Return (api_key, secret_key, is_paper).
    Priority:
      1. st.session_state  (set by Settings tab this session)
      2. data/alpaca_creds.json  (saved by Settings tab, persists across refreshes)
      3. st.secrets / env vars   (Streamlit Cloud secrets)
      4. bot/config.py           (hardcoded fallback)
    """
    from bot.config import (
        ALPACA_API_KEY    as _CFG_KEY,
        ALPACA_SECRET_KEY as _CFG_SECRET,
        ALPACA_IS_PAPER   as _CFG_PAPER,
    )

    # 1. Session state (highest priority — user just typed them this session)
    api_key    = st.session_state.get(_ALPACA_KEY)
    secret_key = st.session_state.get(_ALPACA_SECRET_KEY)
    is_paper   = st.session_state.get(_ALPACA_IS_PAPER)

    # 2. Persisted credentials file (survives page refresh)
    if not api_key or not secret_key:
        saved = _load_creds_file()
        api_key    = api_key    or saved.get("alpaca_api_key")
        secret_key = secret_key or saved.get("alpaca_secret_key")
        if is_paper is None:
            fp = saved.get("alpaca_is_paper")
            if fp is not None:
                is_paper = fp

    # 3. st.secrets / environment variables
    if not api_key:
        api_key = _secret("ALPACA_API_KEY")
    if not secret_key:
        secret_key = _secret("ALPACA_SECRET_KEY")
    if is_paper is None:
        sv = _secret("ALPACA_IS_PAPER")
        is_paper = (sv.lower() != "false") if sv else None

    # 4. Hardcoded config.py fallback
    api_key    = api_key    or _CFG_KEY
    secret_key = secret_key or _CFG_SECRET
    if is_paper is None:
        is_paper = _CFG_PAPER

    return api_key or "", secret_key or "", bool(is_paper)
