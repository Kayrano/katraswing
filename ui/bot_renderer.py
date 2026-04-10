"""
Bot monitoring tab renderer for Katraswing.
Shows live bot status, account metrics, positions, trade log, and controls.

Requires the user to be authenticated (handled by auth_renderer) and to have
saved their Alpaca API credentials via the key-setup form shown here.
"""

import streamlit as st
import pandas as pd
from datetime import datetime


def render_bot_tab():
    """Main entry point — renders the full 🤖 Live Bot tab."""

    from ui.auth_renderer import get_current_user, get_access_token, get_alpaca_creds
    from bot.config import (
        PORTFOLIO_SIZE, RISK_PER_TRADE_PCT, MAX_POSITIONS,
        BUY_THRESHOLD, AVOID_THRESHOLD, SCAN_INTERVAL_MINUTES,
    )

    st.markdown("### 🤖 Katrabot — Live Trading Bot")
    st.caption("Powered by Katraswing analysis pipeline · Executes on Alpaca Paper or Live Trading")
    st.divider()

    user  = get_current_user()
    token = get_access_token()

    if not user:
        st.warning("Please sign in to use the Live Bot.")
        return

    user_id = user["id"]

    # ══════════════════════════════════════════════════════════════════════════
    # Step 1 — API Key Setup
    # ══════════════════════════════════════════════════════════════════════════
    api_key, secret_key, is_paper = get_alpaca_creds()
    keys_saved = bool(api_key and secret_key)

    with st.expander("🔑 Alpaca API Keys" + (" ✅" if keys_saved else " — setup required"), expanded=not keys_saved):
        _render_key_setup(user_id, token, api_key, secret_key, is_paper)

    if not keys_saved:
        st.info("Enter your Alpaca API keys above to start the bot.")
        return

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # Step 2 — Bot Controls
    # ══════════════════════════════════════════════════════════════════════════
    from bot.engine import start_bot, stop_bot, get_state
    from bot.logger import get_recent_trades, get_recent_runs, get_trade_summary_today

    bot_state  = get_state(user_id)
    is_running = bot_state.get("running", False)

    status_color = "#00cc66" if is_running else "#ff4444"
    status_label = "● RUNNING" if is_running else "● STOPPED"

    ctrl_col, status_col = st.columns([2, 3])

    with ctrl_col:
        col_start, col_stop = st.columns(2)
        with col_start:
            if st.button("▶ Start Bot", disabled=is_running,
                         use_container_width=True, type="primary"):
                msg = start_bot(user_id, api_key, secret_key, is_paper)
                st.success(msg)
                st.rerun()
        with col_stop:
            if st.button("⏹ Stop Bot", disabled=not is_running,
                         use_container_width=True):
                msg = stop_bot(user_id)
                st.warning(msg)
                st.rerun()

    with status_col:
        st.markdown(
            f"""<div style="background:#111;border-radius:8px;padding:10px 16px;">
                <span style="color:{status_color};font-weight:700;font-size:15px;">{status_label}</span><br>
                <span style="color:#aaa;font-size:12px;">{bot_state.get('status_msg','—')}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # Live Metrics
    # ══════════════════════════════════════════════════════════════════════════
    from broker.alpaca import get_positions, get_account, is_market_open

    try:
        market_open = is_market_open(api_key=api_key, secret_key=secret_key, is_paper=is_paper)
        acct        = get_account(api_key=api_key, secret_key=secret_key, is_paper=is_paper)
        equity      = float(acct.get("equity",        0))
        last_equity = float(acct.get("last_equity",   0))
        buying_pwr  = float(acct.get("buying_power",  0))
        daily_pnl   = equity - last_equity
        daily_pnl_pct = (daily_pnl / last_equity * 100) if last_equity else 0
        positions   = get_positions(api_key=api_key, secret_key=secret_key, is_paper=is_paper)
    except Exception as e:
        st.error(f"Could not connect to Alpaca: {e}")
        equity = last_equity = buying_pwr = 0.0
        daily_pnl = daily_pnl_pct = 0.0
        positions = []
        market_open = False

    summary_today = get_trade_summary_today()
    mkt_label     = "OPEN" if market_open else "CLOSED"

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Market",        mkt_label)
    m2.metric("Equity",        f"${equity:,.0f}")
    m3.metric("Buying Power",  f"${buying_pwr:,.0f}")
    m4.metric("Daily P&L",     f"${daily_pnl:+,.0f}",
              delta=f"{daily_pnl_pct:+.2f}%",
              delta_color="normal")
    m5.metric("Open Positions", len(positions))
    m6.metric("Trades Today",
              f"{summary_today['buys']}B / {summary_today['sells']}S")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # Current Positions
    # ══════════════════════════════════════════════════════════════════════════
    col_pos, col_cfg = st.columns([3, 2])

    with col_pos:
        st.markdown("#### 📂 Open Positions")
        if not positions:
            st.info("No open positions.")
        else:
            rows = []
            for p in positions:
                unreal     = float(p.get("unrealized_pl",   0))
                unreal_pct = float(p.get("unrealized_plpc", 0)) * 100
                rows.append({
                    "Ticker":    p.get("symbol", ""),
                    "Qty":       int(float(p.get("qty", 0))),
                    "Entry $":   float(p.get("avg_entry_price", 0)),
                    "Current $": float(p.get("current_price",   0)),
                    "Mkt Value": float(p.get("market_value",    0)),
                    "P&L $":     round(unreal, 2),
                    "P&L %":     round(unreal_pct, 2),
                })
            df_pos = pd.DataFrame(rows)

            def _color_pnl(val):
                color = "#00cc66" if val >= 0 else "#ff4444"
                return f"color: {color}"

            st.dataframe(
                df_pos.style
                    .map(_color_pnl, subset=["P&L $", "P&L %"])
                    .format({
                        "Entry $":   "${:.2f}",
                        "Current $": "${:.2f}",
                        "Mkt Value": "${:,.0f}",
                        "P&L $":     "${:+.2f}",
                        "P&L %":     "{:+.2f}%",
                    }),
                use_container_width=True,
                hide_index=True,
            )

    with col_cfg:
        st.markdown("#### ⚙️ Bot Configuration")
        mode_label = "🧪 Paper Trading" if is_paper else "🔴 Live Trading"
        base_url   = "paper-api.alpaca.markets" if is_paper else "api.alpaca.markets"
        st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Portfolio size | ${PORTFOLIO_SIZE:,.0f} |
| Risk / trade | {RISK_PER_TRADE_PCT*100:.1f}% |
| Max positions | {MAX_POSITIONS} |
| Buy threshold | {BUY_THRESHOLD} |
| Avoid threshold | {AVOID_THRESHOLD} |
| Scan interval | {SCAN_INTERVAL_MINUTES} min |
        """)
        st.markdown(f"""
| Alpaca Account |   |
|---|---|
| Mode | {mode_label} |
| Endpoint | {base_url} |
        """)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # Trade Log
    # ══════════════════════════════════════════════════════════════════════════
    col_trades, col_runs = st.columns([3, 2])

    with col_trades:
        st.markdown("#### 📋 Recent Trade Log")
        trades = get_recent_trades(30)
        if not trades:
            st.info("No trades logged yet.")
        else:
            df_trades = pd.DataFrame(trades)[
                ["ts", "ticker", "action", "price", "qty", "stop_loss", "take_profit", "score", "reason"]
            ]
            df_trades.columns = ["Time (UTC)", "Ticker", "Action", "Price", "Qty", "Stop", "Target", "Score", "Reason"]

            def _color_action(val):
                colors = {"BUY": "#00cc66", "SELL": "#ff4444", "SKIP": "#888888", "ERROR": "#ff9900"}
                return f"color: {colors.get(val, '#fff')}"

            st.dataframe(
                df_trades.style
                    .map(_color_action, subset=["Action"])
                    .format({"Price": "${:.2f}", "Stop": "${:.2f}", "Target": "${:.2f}", "Score": "{:.1f}"}),
                use_container_width=True,
                hide_index=True,
                height=320,
            )

    with col_runs:
        st.markdown("#### 🔄 Scan History")
        runs = get_recent_runs(15)
        if not runs:
            st.info("No scan cycles logged yet.")
        else:
            df_runs = pd.DataFrame(runs)[
                ["ts", "market_open", "tickers_scanned", "trades_executed", "positions_closed", "errors", "daily_pnl"]
            ]
            df_runs.columns = ["Time (UTC)", "Mkt", "Scanned", "Bought", "Closed", "Errors", "Day P&L"]
            df_runs["Mkt"]     = df_runs["Mkt"].map({1: "✅", 0: "🔒"})
            df_runs["Day P&L"] = df_runs["Day P&L"].apply(lambda x: f"${x:+,.0f}")
            st.dataframe(df_runs, use_container_width=True, hide_index=True, height=320)

    st.divider()
    st.caption(
        "💡 This tab shows a snapshot. Refresh the page to see the latest data. "
        "Bot continues running in the background regardless of browser state."
    )


# ── API Key Setup Form ────────────────────────────────────────────────────────

def _render_key_setup(user_id: str, token: str, current_key: str, current_secret: str, current_paper: bool):
    """Form to save Alpaca credentials to Supabase and session state."""
    from db.supabase_client import save_user_keys

    st.markdown(
        "Enter your **[Alpaca](https://app.alpaca.markets) API credentials**. "
        "Keys are stored securely in your account — each user has their own isolated bot."
    )

    with st.form("alpaca_keys_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            new_key    = st.text_input("API Key ID",    value=current_key    or "", placeholder="PKRD…")
        with col_b:
            new_secret = st.text_input("API Secret Key", value=current_secret or "", type="password", placeholder="B1Gj…")

        new_paper = st.toggle("Paper Trading (recommended for testing)", value=current_paper if current_paper is not None else True)
        saved = st.form_submit_button("Save Keys", use_container_width=True, type="primary")

    if saved:
        if not new_key or not new_secret:
            st.error("Both API Key ID and Secret Key are required.")
            return
        try:
            save_user_keys(user_id, token, new_key.strip(), new_secret.strip(), new_paper)
            st.session_state["alpaca_api_key"]    = new_key.strip()
            st.session_state["alpaca_secret_key"] = new_secret.strip()
            st.session_state["alpaca_is_paper"]   = new_paper
            st.success("Keys saved! You can now start the bot.")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to save keys: {e}")
