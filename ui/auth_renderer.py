"""
Credential helpers — no authentication, single-user app.
Alpaca keys are read from st.secrets / env vars, or entered in Settings.
"""

import os
import streamlit as st

_ALPACA_KEY        = "alpaca_api_key"
_ALPACA_SECRET_KEY = "alpaca_secret_key"
_ALPACA_IS_PAPER   = "alpaca_is_paper"


def _secret(key: str) -> str:
    """Read from st.secrets first, then os.environ."""
    try:
        return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, "")


def render_auth_gate():
    """No-op — authentication removed."""
    pass


def get_current_user() -> dict:
    """Always returns a single owner user."""
    return {"id": "owner", "email": ""}


def get_alpaca_creds() -> tuple[str, str, bool]:
    """
    Return (api_key, secret_key, is_paper).
    Priority: session_state (set via Settings) > st.secrets / env vars.
    """
    api_key = (
        st.session_state.get(_ALPACA_KEY)
        or _secret("ALPACA_API_KEY")
    )
    secret_key = (
        st.session_state.get(_ALPACA_SECRET_KEY)
        or _secret("ALPACA_SECRET_KEY")
    )
    # is_paper: session_state override, then secrets, then default True
    if _ALPACA_IS_PAPER in st.session_state:
        is_paper = st.session_state[_ALPACA_IS_PAPER]
    else:
        secret_val = _secret("ALPACA_IS_PAPER")
        is_paper = (secret_val.lower() != "false") if secret_val else True
    return api_key or "", secret_key or "", is_paper
