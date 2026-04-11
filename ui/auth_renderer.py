"""
Authentication UI — login / register / logout.

Call render_auth_gate() at the very top of app.py (before tabs).
It will st.stop() if the user is not logged in, so the rest of the
app only runs for authenticated sessions.
"""

import streamlit as st
from db.supabase_client import sign_in, sign_up, sign_out, load_user_keys, supabase_configured

# ── Session-state keys ────────────────────────────────────────────────────────
_USER_KEY          = "auth_user"
_TOKEN_KEY         = "auth_access_token"
_ALPACA_KEY        = "alpaca_api_key"
_ALPACA_SECRET_KEY = "alpaca_secret_key"
_ALPACA_IS_PAPER   = "alpaca_is_paper"


def is_authenticated() -> bool:
    return bool(st.session_state.get(_USER_KEY))


def get_current_user() -> dict | None:
    return st.session_state.get(_USER_KEY)


def get_access_token() -> str | None:
    return st.session_state.get(_TOKEN_KEY)


def get_alpaca_creds() -> tuple[str | None, str | None, bool]:
    return (
        st.session_state.get(_ALPACA_KEY),
        st.session_state.get(_ALPACA_SECRET_KEY),
        st.session_state.get(_ALPACA_IS_PAPER, True),
    )


def _do_sign_out(token: str | None = None):
    """Clear session state and call Supabase sign-out."""
    if token:
        try:
            sign_out(token)
        except Exception:
            pass
    for key in [_USER_KEY, _TOKEN_KEY, _ALPACA_KEY, _ALPACA_SECRET_KEY, _ALPACA_IS_PAPER]:
        st.session_state.pop(key, None)


def render_auth_gate():
    """
    Show login/register UI and st.stop() if user is not authenticated.
    Place this call BEFORE any other content in app.py.
    If Supabase is not configured (local dev), skips auth entirely.
    """
    if is_authenticated():
        return  # already in — let the rest of the app render

    if not supabase_configured():
        # No auth backend — inject a guest session so the rest of the app works normally
        st.session_state[_USER_KEY]  = {"id": "guest", "email": "guest@local"}
        st.session_state[_TOKEN_KEY] = ""
        return

    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown("""
        <div style="text-align:center; padding:30px 0 10px 0;">
            <h2 style="color:#4488ff; margin:0;">📈 KATRASWING</h2>
            <p style="color:#888; font-size:13px;">Sign in to access the trading platform</p>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_register = st.tabs(["Sign In", "Create Account"])

        with tab_login:
            _render_login_form()

        with tab_register:
            _render_register_form()

    st.stop()


# ── Private helpers ───────────────────────────────────────────────────────────

def _render_login_form():
    with st.form("login_form"):
        email     = st.text_input("Email")
        password  = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign In", use_container_width=True, type="primary")

    if submitted:
        if not email or not password:
            st.error("Please enter your email and password.")
            return
        try:
            res     = sign_in(email.strip(), password)
            user    = res.user
            session = res.session
            if user and session:
                st.session_state[_USER_KEY]  = {"id": user.id, "email": user.email}
                st.session_state[_TOKEN_KEY] = session.access_token
                _load_alpaca_keys_into_session(user.id, session.access_token)
                st.success("Signed in!")
                st.rerun()
            else:
                st.error("Login failed — check your credentials.")
        except Exception as e:
            st.error(f"Login error: {e}")


def _render_register_form():
    with st.form("register_form"):
        email     = st.text_input("Email")
        password  = st.text_input("Password", type="password", help="Minimum 6 characters")
        password2 = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Create Account", use_container_width=True, type="primary")

    if submitted:
        if not email or not password:
            st.error("Please fill in all fields.")
            return
        if password != password2:
            st.error("Passwords do not match.")
            return
        if len(password) < 6:
            st.error("Password must be at least 6 characters.")
            return
        try:
            res = sign_up(email.strip(), password)
            if res.user:
                st.success("Account created! Check your email to confirm, then sign in.")
            else:
                st.error("Registration failed.")
        except Exception as e:
            st.error(f"Registration error: {e}")


def _load_alpaca_keys_into_session(user_id: str, access_token: str):
    try:
        data = load_user_keys(user_id, access_token)
        if data:
            st.session_state[_ALPACA_KEY]        = data.get("alpaca_api_key", "")
            st.session_state[_ALPACA_SECRET_KEY] = data.get("alpaca_secret_key", "")
            st.session_state[_ALPACA_IS_PAPER]   = data.get("is_paper", True)
    except Exception:
        pass
