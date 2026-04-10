"""
Supabase client singleton.
Credentials are read from st.secrets (Streamlit Cloud) or environment variables.
"""

import streamlit as st


@st.cache_resource
def get_supabase():
    """Return a cached Supabase client. Called once per app run."""
    from supabase import create_client
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_ANON_KEY"]
    return create_client(url, key)


# ── Auth helpers ──────────────────────────────────────────────────────────────

def sign_up(email: str, password: str):
    sb = get_supabase()
    res = sb.auth.sign_up({"email": email, "password": password})
    return res


def sign_in(email: str, password: str):
    sb = get_supabase()
    res = sb.auth.sign_in_with_password({"email": email, "password": password})
    return res


def sign_out(access_token: str):
    sb = get_supabase()
    sb.auth.sign_out()


# ── API key storage ───────────────────────────────────────────────────────────

def save_user_keys(user_id: str, access_token: str, api_key: str, secret_key: str, is_paper: bool = True):
    """Insert or update Alpaca keys for a user."""
    sb = get_supabase()
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
    sb.postgrest.auth(access_token)
    res = sb.table("user_keys").select("*").eq("user_id", user_id).maybe_single().execute()
    return res.data if res else None
