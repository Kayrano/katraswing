"""
Politician Trades Panel
Renders the full Congress Trades tab including:
  - Global recent trade feed
  - Per-ticker trade history (when called from Analyzer context)
  - Top politician leaderboard
  - Sentiment gauge + score correction explanation
  - Disclosure delay warning
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, date, timedelta


# ── Color helpers ─────────────────────────────────────────────────────────────

def _party_color(party: str) -> str:
    p = (party or "").upper()
    if p in ("D", "DEMOCRAT", "DEMOCRATIC"):
        return "#4477ff"
    if p in ("R", "REPUBLICAN"):
        return "#ff4444"
    return "#888888"


def _action_color(action: str) -> str:
    return "#22cc88" if action == "BUY" else "#ff4455"


def _action_badge(action: str) -> str:
    color = _action_color(action)
    return f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700;">{action}</span>'


def _party_badge(party: str) -> str:
    color = _party_color(party)
    label = {"D": "DEM", "R": "REP"}.get((party or "?").upper(), party or "?")
    return f'<span style="background:{color};color:#fff;padding:2px 6px;border-radius:4px;font-size:10px;">{label}</span>'


def _star(is_top: bool) -> str:
    return "⭐" if is_top else ""


# ── Sentiment gauge ───────────────────────────────────────────────────────────

def _render_sentiment_gauge(sentiment: float, tp_signal: str, buy_count: int, sell_count: int):
    """Plotly gauge from -1 (sell) to +1 (buy)."""
    needle_color = "#22cc88" if sentiment > 0.1 else ("#ff4455" if sentiment < -0.1 else "#aaaaaa")

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(sentiment, 2),
        delta={"reference": 0, "valueformat": ".2f"},
        number={"font": {"size": 28, "color": needle_color}, "valueformat": ".2f"},
        gauge={
            "axis": {"range": [-1, 1], "tickvals": [-1, -0.5, 0, 0.5, 1],
                     "ticktext": ["Strong Sell", "Sell", "Neutral", "Buy", "Strong Buy"],
                     "tickfont": {"size": 10}},
            "bar": {"color": needle_color, "thickness": 0.25},
            "steps": [
                {"range": [-1.0, -0.4], "color": "#3a1a1a"},
                {"range": [-0.4, -0.1], "color": "#2a1a1a"},
                {"range": [-0.1,  0.1], "color": "#222222"},
                {"range": [ 0.1,  0.4], "color": "#1a2a1a"},
                {"range": [ 0.4,  1.0], "color": "#1a3a1a"},
            ],
            "threshold": {
                "line": {"color": "#ffffff", "width": 2},
                "thickness": 0.75,
                "value": sentiment,
            },
        },
        title={"text": "Congress Sentiment", "font": {"size": 13, "color": "#aaa"}},
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Top-performer signal label
    tp_colors = {"BULLISH": "#22cc88", "BEARISH": "#ff4455", "NEUTRAL": "#888"}
    tp_col = tp_colors.get(tp_signal, "#888")
    st.markdown(
        f"<div style='text-align:center;font-size:13px;color:{tp_col};font-weight:700;margin-top:-12px;'>"
        f"Top Performers: {tp_signal}  |  {buy_count} BUY · {sell_count} SELL"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Trade table ───────────────────────────────────────────────────────────────

def _render_trade_table(trades: list[dict], show_ticker: bool = False):
    """Render a styled HTML table of trades."""
    if not trades:
        st.caption("No trades found in this period.")
        return

    rows = ""
    for t in trades:
        star = _star(t.get("is_top_performer", False))
        action_badge = _action_badge(t.get("action", "?"))
        party_badge = _party_badge(t.get("party", "?"))
        ticker_col = f"<td style='color:#4488ff;font-weight:600;'>{t.get('ticker','')}</td>" if show_ticker else ""
        rows += (
            f"<tr>"
            f"<td style='color:#888;font-size:11px;'>{t.get('date','')}</td>"
            f"<td>{star} {t.get('politician','?')} {party_badge}</td>"
            + ticker_col +
            f"<td>{action_badge}</td>"
            f"<td style='color:#ccc;font-size:12px;'>{t.get('amount','?')}</td>"
            f"<td style='color:#888;font-size:11px;'>{t.get('chamber','?')}</td>"
            f"</tr>"
        )

    ticker_header = "<th>Ticker</th>" if show_ticker else ""
    html = f"""
    <div style="overflow-x:auto; max-height:340px; overflow-y:auto;">
    <table style="width:100%;border-collapse:collapse;font-size:12px;">
    <thead>
    <tr style="border-bottom:1px solid #333;color:#888;text-align:left;">
        <th>Date Filed</th><th>Politician</th>{ticker_header}
        <th>Action</th><th>Amount</th><th>Chamber</th>
    </tr>
    </thead>
    <tbody>{rows}</tbody>
    </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ── Score correction box ──────────────────────────────────────────────────────

def _render_correction_box(score_delta: float, sentiment: float, buy_count: int, sell_count: int):
    if score_delta == 0.0:
        color, icon, label = "#555", "⚪", "No correction applied"
    elif score_delta > 0:
        color, icon, label = "#22cc88", "▲", f"Score boosted by +{score_delta:.1f} pts"
    else:
        color, icon, label = "#ff4455", "▼", f"Score penalized by {score_delta:.1f} pts"

    st.markdown(
        f"""<div style="background:#1a1a2e;border:1px solid {color};border-radius:8px;
            padding:12px 16px;margin:8px 0;">
            <span style="color:{color};font-size:16px;">{icon}</span>
            <span style="color:{color};font-weight:700;font-size:14px;margin-left:8px;">{label}</span>
            <div style="color:#888;font-size:11px;margin-top:4px;">
            Weighted sentiment: {sentiment:+.2f} &nbsp;|&nbsp; {buy_count} buys / {sell_count} sells
            &nbsp;|&nbsp; Cap: ±8 pts max
            </div></div>""",
        unsafe_allow_html=True,
    )


# ── Delay warning ─────────────────────────────────────────────────────────────

def _render_delay_warning(note: str):
    st.markdown(
        f"""<div style="background:#1a1500;border-left:3px solid #ffaa00;
            padding:8px 14px;border-radius:4px;margin-bottom:10px;font-size:12px;color:#cca;">
            ⚠ <strong>Disclosure Delay:</strong> {note}
            <br>Use as a medium-term confirmation — not a same-day trading trigger.
        </div>""",
        unsafe_allow_html=True,
    )


# ── Buy/Sell bar chart ────────────────────────────────────────────────────────

def _render_activity_chart(trades: list[dict]):
    """Monthly buy/sell bar chart."""
    if not trades:
        return
    rows = []
    for t in trades:
        try:
            dt = datetime.strptime(t["date"], "%Y-%m-%d")
            rows.append({"month": dt.strftime("%Y-%m"), "action": t["action"]})
        except Exception:
            pass
    if not rows:
        return

    df = pd.DataFrame(rows)
    counts = df.groupby(["month", "action"]).size().reset_index(name="count")

    fig = px.bar(
        counts,
        x="month",
        y="count",
        color="action",
        color_discrete_map={"BUY": "#22cc88", "SELL": "#ff4455"},
        barmode="group",
        labels={"month": "Month (Filed)", "count": "# Trades"},
        title="Monthly Congressional Activity",
    )
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=36, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1, x=0),
        xaxis=dict(gridcolor="#2a2a2a"),
        yaxis=dict(gridcolor="#2a2a2a"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Leaderboard ───────────────────────────────────────────────────────────────

def _render_leaderboard(politicians: list[dict]):
    if not politicians:
        st.caption("Could not load politician leaderboard.")
        return

    rows = ""
    for i, p in enumerate(politicians[:15], 1):
        star = _star(p.get("is_top_performer", False))
        party_badge = _party_badge(p.get("party", "?"))
        rows += (
            f"<tr>"
            f"<td style='color:#888;'>{i}</td>"
            f"<td>{star} {p.get('name','?')} {party_badge}</td>"
            f"<td style='color:#888;font-size:11px;'>{p.get('chamber','?')}</td>"
            f"<td style='color:#4488ff;font-weight:600;'>{p.get('trade_count',0)}</td>"
            f"</tr>"
        )
    html = f"""
    <div style="overflow-y:auto; max-height:360px;">
    <table style="width:100%;border-collapse:collapse;font-size:12px;">
    <thead>
    <tr style="border-bottom:1px solid #333;color:#888;">
        <th>#</th><th>Politician</th><th>Chamber</th><th>Total Trades</th>
    </tr>
    </thead>
    <tbody>{rows}</tbody>
    </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Public entry points
# ══════════════════════════════════════════════════════════════════════════════

def render_politician_panel(report) -> None:
    """
    Render the politician trades expander inside the Analyzer tab.
    Takes a ReportData object.
    """
    pol = report.politician if hasattr(report, "politician") else None

    if pol is None:
        st.info("No congressional trade data available for this ticker.")
        return

    _render_delay_warning(pol.delay_note)

    col_gauge, col_correction = st.columns([1, 1])
    with col_gauge:
        _render_sentiment_gauge(pol.sentiment, pol.top_performer_signal, pol.buy_count, pol.sell_count)
    with col_correction:
        st.markdown("#### Score Adjustment")
        _render_correction_box(pol.score_delta, pol.sentiment, pol.buy_count, pol.sell_count)
        if pol.top_performer_trades:
            st.markdown("**Top Performer Trades:**")
            for t in pol.top_performer_trades[:3]:
                action_col = "#22cc88" if t["action"] == "BUY" else "#ff4455"
                st.markdown(
                    f"<div style='font-size:12px;color:#ccc;'>"
                    f"⭐ <b>{t['politician']}</b> — "
                    f"<span style='color:{action_col};'>{t['action']}</span> "
                    f"({t['amount']}) on {t['date']}</div>",
                    unsafe_allow_html=True,
                )

    if pol.recent_trades:
        _render_activity_chart(pol.recent_trades)
        st.markdown("**Recent Congressional Trades (filed, not actual trade date)**")
        _render_trade_table(pol.recent_trades[:20])
    else:
        st.caption(f"No trades filed in the past 120 days for {report.ticker}.")


def render_politician_tab() -> None:
    """
    Render the standalone Congress Trades tab.
    Shows global feed + leaderboard + per-ticker lookup.
    """
    st.markdown("## 🏛 Congress Trades")
    st.markdown(
        "<p style='color:#888;font-size:13px;'>Congressional stock trades aggregated from Capitol Trades. "
        "Data shown is the <strong>filing date</strong> — the actual trade occurred 30–45 days earlier.</p>",
        unsafe_allow_html=True,
    )

    # ── Disclosure delay banner ───────────────────────────────────────────────
    st.markdown(
        """<div style="background:#1a1500;border:1px solid #ffaa00;border-radius:8px;
        padding:10px 16px;margin-bottom:16px;font-size:12px;color:#cca;">
        ⚠ <strong>Important:</strong> The STOCK Act requires politicians to disclose trades within
        45 days of the transaction. Dates shown below are <em>filing dates</em>, not trade dates.
        The actual buy/sell likely happened <strong>30–45 days before</strong> what you see here.
        Use this data as a medium-term confirmation signal — look for clusters of insider buying
        as a bullish backdrop, not a same-day entry trigger.
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Tabs inside the Congress tab ─────────────────────────────────────────
    sub_feed, sub_lookup, sub_leaderboard = st.tabs([
        "📰 Live Feed", "🔍 Ticker Lookup", "🏆 Leaderboard"
    ])

    # ── Live Feed ─────────────────────────────────────────────────────────────
    with sub_feed:
        st.markdown("#### Most Recent Congressional Trades")
        with st.spinner("Loading latest trades..."):
            from data.politician_trades import fetch_recent_trades_all
            trades = fetch_recent_trades_all(page_size=100)

        if not trades:
            st.warning(
                "Could not load trades from Capitol Trades API. "
                "The API may be temporarily unavailable or the endpoint format may have changed."
            )
        else:
            # Filter controls
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                action_filter = st.selectbox("Action", ["All", "BUY", "SELL"], key="pol_feed_action")
            with col_f2:
                party_filter = st.selectbox("Party", ["All", "R", "D"], key="pol_feed_party")
            with col_f3:
                top_only = st.checkbox("Top performers only ⭐", key="pol_feed_top")

            filtered = trades
            if action_filter != "All":
                filtered = [t for t in filtered if t["action"] == action_filter]
            if party_filter != "All":
                filtered = [t for t in filtered if (t.get("party") or "").upper().startswith(party_filter)]
            if top_only:
                filtered = [t for t in filtered if t.get("is_top_performer")]

            st.caption(f"Showing {len(filtered)} trades")
            _render_trade_table(filtered, show_ticker=True)

    # ── Ticker Lookup ─────────────────────────────────────────────────────────
    with sub_lookup:
        st.markdown("#### Look Up Trades by Ticker")
        col_ticker, col_days = st.columns([2, 1])
        with col_ticker:
            lookup_ticker = st.text_input(
                "Ticker Symbol",
                placeholder="e.g. NVDA, AAPL, MSFT",
                key="pol_lookup_ticker",
            ).strip().upper()
        with col_days:
            days_back = st.slider("Days back", 30, 365, 120, step=30, key="pol_lookup_days")

        if lookup_ticker:
            with st.spinner(f"Fetching trades for {lookup_ticker}..."):
                from data.politician_trades import (
                    fetch_ticker_trades,
                    compute_politician_sentiment,
                    compute_score_correction,
                )
                trades_for_ticker = fetch_ticker_trades(lookup_ticker, days_back=days_back)
                sentiment_data = compute_politician_sentiment(trades_for_ticker)
                delta, _ = compute_score_correction(sentiment_data)

            _render_delay_warning(sentiment_data["delay_note"])

            col_g, col_c = st.columns([1, 1])
            with col_g:
                _render_sentiment_gauge(
                    sentiment_data["sentiment"],
                    sentiment_data["top_performer_signal"],
                    sentiment_data["buy_count"],
                    sentiment_data["sell_count"],
                )
            with col_c:
                st.markdown(f"#### {lookup_ticker} — Congress Signal")
                _render_correction_box(
                    delta,
                    sentiment_data["sentiment"],
                    sentiment_data["buy_count"],
                    sentiment_data["sell_count"],
                )

            if trades_for_ticker:
                _render_activity_chart(trades_for_ticker)
                st.markdown(f"**{len(trades_for_ticker)} trades filed in past {days_back} days:**")
                _render_trade_table(trades_for_ticker)
            else:
                st.info(f"No congressional trades found for **{lookup_ticker}** in the past {days_back} days.")
        else:
            st.markdown(
                "<div style='text-align:center;padding:40px;color:#555;'>"
                "<div style='font-size:40px;'>🔍</div>"
                "<p>Enter a ticker symbol above to look up congressional trades.</p>"
                "</div>",
                unsafe_allow_html=True,
            )

    # ── Leaderboard ───────────────────────────────────────────────────────────
    with sub_leaderboard:
        st.markdown("#### Most Active Politicians (all-time)")
        st.caption(
            "⭐ = included in the 'Top Performer' set — trades from these politicians "
            "are weighted 1.5× in the score correction formula."
        )

        with st.spinner("Loading leaderboard..."):
            from data.politician_trades import fetch_top_politicians, TOP_PERFORMERS
            politicians = fetch_top_politicians(limit=25)

        col_lb, col_tp = st.columns([3, 2])
        with col_lb:
            _render_leaderboard(politicians)
        with col_tp:
            st.markdown("#### Top Performer List")
            st.caption(
                "These politicians consistently outperform the S&P 500 in their personal "
                "portfolios and are on key finance/tech/defense committees."
            )
            for name in sorted(TOP_PERFORMERS):
                st.markdown(f"⭐ {name}")
