"""
Settings tab renderer.
Shown to every authenticated user. Lets them manage their Alpaca API
credentials and account preferences.
"""

import streamlit as st
from ui.auth_renderer import get_current_user, get_access_token, get_alpaca_creds


def render_settings_tab():
    from db.supabase_client import save_user_keys

    user  = get_current_user()
    token = get_access_token()

    if not user:
        st.warning("Please sign in to access settings.")
        return

    user_id = user["id"]
    api_key, secret_key, is_paper = get_alpaca_creds()

    st.markdown("### ⚙️ Settings")
    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # Account info
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("#### 👤 Account")
    col_info, col_logout = st.columns([3, 1])
    with col_info:
        st.markdown(f"**Email:** {user.get('email', '—')}")
        st.caption(f"User ID: `{user_id}`")
    with col_logout:
        if st.button("Sign Out", use_container_width=True):
            from ui.auth_renderer import _do_sign_out
            _do_sign_out(token)
            st.rerun()

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # Alpaca credentials
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("#### 🔑 Alpaca API Credentials")

    keys_set = bool(api_key and secret_key)
    if keys_set:
        st.success("API keys are configured — your bot is ready to run.")
    else:
        st.warning("No API keys saved yet. Add them below to enable the Live Bot.")

    st.markdown(
        "Get your keys from the **[Alpaca dashboard](https://app.alpaca.markets)** "
        "under **Paper Trading → API Keys** (or Live Trading if you prefer)."
    )

    with st.form("alpaca_keys_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            new_key = st.text_input(
                "API Key ID",
                value=api_key or "",
                placeholder="PKRD…",
            )
        with col_b:
            new_secret = st.text_input(
                "Secret Key",
                value=secret_key or "",
                type="password",
                placeholder="B1Gj…",
            )

        new_paper = st.toggle(
            "Use Paper Trading account (recommended for testing)",
            value=is_paper if is_paper is not None else True,
        )

        save_btn = st.form_submit_button("Save Credentials", type="primary", use_container_width=True)

    if save_btn:
        if not new_key.strip() or not new_secret.strip():
            st.error("Both API Key ID and Secret Key are required.")
        else:
            try:
                save_user_keys(user_id, token, new_key.strip(), new_secret.strip(), new_paper)
                st.session_state["alpaca_api_key"]    = new_key.strip()
                st.session_state["alpaca_secret_key"] = new_secret.strip()
                st.session_state["alpaca_is_paper"]   = new_paper
                st.success("Credentials saved! Head to the Live Bot tab to start trading.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save: {e}")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # Danger zone — clear keys
    # ══════════════════════════════════════════════════════════════════════════
    if keys_set:
        st.markdown("#### 🗑️ Remove Credentials")
        st.caption("This removes your saved Alpaca keys from the database.")
        if st.button("Remove API Keys", use_container_width=False):
            try:
                from db.supabase_client import get_supabase
                sb = get_supabase()
                sb.postgrest.auth(token)
                sb.table("user_keys").delete().eq("user_id", user_id).execute()
                for k in ("alpaca_api_key", "alpaca_secret_key", "alpaca_is_paper"):
                    st.session_state.pop(k, None)
                st.success("API keys removed.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to remove keys: {e}")
