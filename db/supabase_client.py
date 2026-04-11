"""
Supabase client singleton.
Credentials are read from st.secrets (Streamlit Cloud) first,
then from environment variables (local dev), then returns None if unconfigured.
"""

import os
import streamlit as st


def _get_secret(key: str) -> str | None:
    """Read a secret from st.secrets, falling back to os.environ."""
    try:
        return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key)


def supabase_configured() -> bool:
    """Return True if both Supabase credentials are available."""
    return bool(_get_secret("SUPABASE_URL") and _get_secret("SUPABASE_ANON_KEY"))


@st.cache_resource
def get_supabase():
    """
    Return a cached Supabase client, or None if credentials are not configured.
    Callers must check for None before using.
    """
    url = _get_secret("SUPABASE_URL")
    key = _get_secret("SUPABASE_ANON_KEY")
    if not url or not key:
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except Exception:
        return None


# ── Auth helpers ──────────────────────────────────────────────────────────────

def sign_up(email: str, password: str):
    sb = get_supabase()
    if not sb:
        raise RuntimeError("Authentication is not configured. SUPABASE_URL and SUPABASE_ANON_KEY are required.")
    return sb.auth.sign_up({"email": email, "password": password})


def sign_in(email: str, password: str):
    sb = get_supabase()
    if not sb:
        raise RuntimeError("Authentication is not configured. SUPABASE_URL and SUPABASE_ANON_KEY are required.")
    return sb.auth.sign_in_with_password({"email": email, "password": password})


def sign_out(_access_token: str = ""):
    sb = get_supabase()
    if sb:
        sb.auth.sign_out()


# ── API key storage ───────────────────────────────────────────────────────────

def save_user_keys(user_id: str, access_token: str, api_key: str, secret_key: str, is_paper: bool = True):
    """Insert or update Alpaca keys for a user."""
    sb = get_supabase()
    if not sb:
        return
    sb.postgrest.auth(access_token)
    sb.table("user_keys").upsert({
        "user_id":          user_id,
        "alpaca_api_key":   api_key,
        "alpaca_secret_key": secret_key,
        "is_paper":         is_paper,
    }, on_conflict="user_id").execute()


def load_user_keys(user_id: str, access_token: str) -> dict | None:
    """Return the stored Alpaca keys for a user, or None if not set."""
    sb = get_supabase()
    if not sb:
        return None
    sb.postgrest.auth(access_token)
    res = sb.table("user_keys").select("*").eq("user_id", user_id).maybe_single().execute()
    return res.data if res else None
