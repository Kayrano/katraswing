"""
Settings tab renderer — Alpaca API credentials and bot configuration.
Keys are stored in session_state for the current session.
Pre-populated from st.secrets / environment variables if available.
"""

import streamlit as st
from ui.auth_renderer import get_alpaca_creds, save_creds_file, clear_creds_file

_ALPACA_KEY        = "alpaca_api_key"
_ALPACA_SECRET_KEY = "alpaca_secret_key"
_ALPACA_IS_PAPER   = "alpaca_is_paper"


def render_settings_tab():
    api_key, secret_key, is_paper = get_alpaca_creds()

    st.markdown("### ⚙️ Settings")
    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # Alpaca credentials
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("#### 🔑 Alpaca API Credentials")

    keys_set = bool(api_key and secret_key)
    if keys_set:
        st.success("API keys are configured — your bot is ready to run.")
    else:
        st.warning("No API keys saved yet. Enter them below to enable the Live Bot.")

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
            st.session_state[_ALPACA_KEY]        = new_key.strip()
            st.session_state[_ALPACA_SECRET_KEY] = new_secret.strip()
            st.session_state[_ALPACA_IS_PAPER]   = new_paper
            # Persist to file so credentials survive page refreshes
            save_creds_file(new_key.strip(), new_secret.strip(), new_paper)
            st.success("Credentials saved! Head to the Live Bot tab to start trading.")
            st.rerun()

    if keys_set:
        st.divider()
        st.markdown("#### 🗑️ Clear Credentials")
        st.caption("Removes keys from this session. You will need to re-enter them.")
        if st.button("Clear API Keys", use_container_width=False):
            for k in (_ALPACA_KEY, _ALPACA_SECRET_KEY, _ALPACA_IS_PAPER):
                st.session_state.pop(k, None)
            clear_creds_file()
            st.success("Credentials cleared.")
            st.rerun()
