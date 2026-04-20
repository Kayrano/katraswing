"""
Dashboard UI rendering — chart, signal box, news feed, indicators.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from agents.signal_engine import SignalResult


# ── MT5-style candlestick chart ───────────────────────────────────────────────

_BG       = "#131722"   # MT5 dark background
_GRID     = "#1e2329"   # subtle grid lines
_UP       = "#089981"   # MT5 green candle
_DOWN     = "#F23645"   # MT5 red candle
_VWAP_C   = "#2196F3"   # blue VWAP line
_EMA_C    = "#FF9800"   # orange EMA line
_ENTRY_C  = "#B2B5BE"   # white/grey entry line
_SL_C     = "#F23645"   # red SL line
_TP_C     = "#089981"   # green TP line


def render_5m_chart(result: SignalResult) -> None:
    df = result.df_5m
    if df is None or df.empty:
        st.warning("No chart data available.")
        return

    df = df.tail(80).copy()
    # Use string labels on x-axis to eliminate weekend/holiday gaps
    x_labels = [str(i) for i in range(len(df))]
    tick_step = max(1, len(df) // 8)
    tickvals  = x_labels[::tick_step]
    ticktext  = [df.index[int(i)].strftime("%m/%d %H:%M") for i in tickvals]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.78, 0.22],
        vertical_spacing=0.02,
    )

    # ── Candlesticks ──────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=x_labels,
        open=df["Open"].values,
        high=df["High"].values,
        low=df["Low"].values,
        close=df["Close"].values,
        name="",
        increasing=dict(line=dict(color=_UP, width=1), fillcolor=_UP),
        decreasing=dict(line=dict(color=_DOWN, width=1), fillcolor=_DOWN),
        showlegend=False,
        hovertext=[
            f"O: {o:.2f}  H: {h:.2f}  L: {l:.2f}  C: {c:.2f}"
            for o, h, l, c in zip(df["Open"], df["High"], df["Low"], df["Close"])
        ],
        hoverinfo="text+x",
    ), row=1, col=1)

    # ── VWAP ─────────────────────────────────────────────────────────────────
    if "session_vwap" in df.columns:
        fig.add_trace(go.Scatter(
            x=x_labels, y=df["session_vwap"].values,
            name="VWAP",
            line=dict(color=_VWAP_C, width=1.5),
            hovertemplate="VWAP: %{y:.2f}<extra></extra>",
        ), row=1, col=1)

    # ── EMA 20 ────────────────────────────────────────────────────────────────
    try:
        import utils.ta_compat as ta
        ema20 = ta.ema(df["Close"], length=20)
        if ema20 is not None and not ema20.isna().all():
            fig.add_trace(go.Scatter(
                x=x_labels, y=ema20.values,
                name="EMA20",
                line=dict(color=_EMA_C, width=1.2),
                hovertemplate="EMA20: %{y:.2f}<extra></extra>",
            ), row=1, col=1)
    except Exception:
        pass

    # ── Entry / SL / TP horizontal lines ────────────────────────────────────
    if result.direction in ("LONG", "SHORT") and result.entry > 0:
        direction_label = "BUY" if result.direction == "LONG" else "SELL"
        for price, color, tag in [
            (result.entry, _ENTRY_C, direction_label),
            (result.sl,    _SL_C,    f"SL  {result.sl:.2f}"),
            (result.tp,    _TP_C,    f"TP  {result.tp:.2f}"),
        ]:
            fig.add_shape(
                type="line",
                x0=x_labels[0], x1=x_labels[-1],
                y0=price, y1=price,
                line=dict(color=color, width=1.5, dash="dash"),
                row=1, col=1,
            )
            # Label pinned to right edge of chart
            fig.add_annotation(
                x=x_labels[-1], y=price,
                text=f"<b>{tag}</b>",
                showarrow=False,
                xanchor="left",
                bgcolor=color,
                bordercolor=color,
                font=dict(color="#ffffff" if color != _ENTRY_C else "#131722", size=11),
                xshift=4,
                row=1, col=1,
            )

    # ── Volume bars ───────────────────────────────────────────────────────────
    vol_colors = [_UP if c >= o else _DOWN
                  for c, o in zip(df["Close"].values, df["Open"].values)]
    fig.add_trace(go.Bar(
        x=x_labels, y=df["Volume"].values,
        marker_color=vol_colors,
        marker_opacity=0.6,
        showlegend=False,
        hovertemplate="Vol: %{y:,.0f}<extra></extra>",
    ), row=2, col=1)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(color="#B2B5BE", family="monospace", size=11),
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=80, t=20, b=0),
        height=500,
        legend=dict(
            orientation="h", x=0, y=1.02,
            font=dict(size=11), bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
    )

    # Price axis: right side, matching MT5
    fig.update_yaxes(
        gridcolor=_GRID, gridwidth=1,
        side="right",
        showgrid=True, zeroline=False,
        tickfont=dict(size=10),
        row=1, col=1,
    )
    fig.update_yaxes(
        gridcolor=_GRID,
        side="right",
        showgrid=False, zeroline=False,
        tickfont=dict(size=9),
        row=2, col=1,
    )
    fig.update_xaxes(
        gridcolor=_GRID, gridwidth=1,
        showgrid=True, zeroline=False,
        tickvals=tickvals, ticktext=ticktext,
        tickfont=dict(size=10),
        row=2, col=1,
    )
    fig.update_xaxes(showticklabels=False, row=1, col=1)

    st.plotly_chart(fig, use_container_width=True)


# ── Signal box ────────────────────────────────────────────────────────────────

def render_signal_box(result: SignalResult) -> None:
    direction = result.direction
    conf_pct = int(result.confidence * 100)

    if direction == "LONG":
        badge_color = "#26a69a"
        badge_icon = "▲"
    elif direction == "SHORT":
        badge_color = "#ef5350"
        badge_icon = "▼"
    else:
        badge_color = "#555"
        badge_icon = "—"

    st.markdown(
        f"""
        <div style="background:#1e2130;border-radius:10px;padding:16px 20px;margin-bottom:10px;">
          <div style="font-size:28px;font-weight:700;color:{badge_color};">
            {badge_icon} {direction}
          </div>
          <div style="margin:8px 0 4px;font-size:13px;color:#aaa;">Confidence</div>
          <div style="background:#2a2d3e;border-radius:6px;height:10px;overflow:hidden;">
            <div style="background:{badge_color};width:{conf_pct}%;height:100%;border-radius:6px;"></div>
          </div>
          <div style="font-size:13px;color:#fafafa;margin-top:4px;">{conf_pct}%
            <span style="color:#888;font-size:11px;"> (base {int(result.base_confidence*100)}%
            {"+" if result.news_boost >= 0 else ""}{int(result.news_boost*100)}% news+pattern)</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if direction in ("LONG", "SHORT") and result.entry > 0:
        risk = abs(result.entry - result.sl)
        reward = abs(result.tp - result.entry)
        st.markdown(
            f"""
            <div style="background:#1e2130;border-radius:10px;padding:14px 20px;">
              <table style="width:100%;font-size:14px;color:#fafafa;border-collapse:collapse;">
                <tr><td style="color:#aaa;padding:3px 0;">Entry</td>
                    <td style="text-align:right;color:#42a5f5;">{result.entry:.2f}</td></tr>
                <tr><td style="color:#aaa;padding:3px 0;">Stop Loss</td>
                    <td style="text-align:right;color:#ef5350;">{result.sl:.2f}</td></tr>
                <tr><td style="color:#aaa;padding:3px 0;">Take Profit</td>
                    <td style="text-align:right;color:#26a69a;">{result.tp:.2f}</td></tr>
                <tr><td style="color:#aaa;padding:3px 0;">Risk / Reward</td>
                    <td style="text-align:right;">{risk:.2f} / {reward:.2f}</td></tr>
                <tr><td style="color:#aaa;padding:3px 0;">ATR</td>
                    <td style="text-align:right;">{result.atr:.2f}</td></tr>
              </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # NQ/MNQ lot info
        ticker_up = result.ticker.upper()
        if "NQ" in ticker_up:
            point_val = 2.0 if "MNQ" in ticker_up else 20.0
            contract_name = "MNQ" if "MNQ" in ticker_up else "NQ"
            st.markdown(
                f"""<div style="background:#1e2130;border-radius:10px;padding:12px 20px;margin-top:8px;font-size:13px;color:#aaa;">
                  <b style="color:#fafafa;">{contract_name} Point Value:</b> ${point_val:.0f}/pt &nbsp;|&nbsp;
                  <b style="color:#fafafa;">Risk in $:</b> {risk:.2f} pts × ${point_val:.0f} = <b style="color:#ef5350;">${risk*point_val:,.0f}</b>
                </div>""",
                unsafe_allow_html=True,
            )

    # Strategy breakdown
    if result.chart_signals:
        st.markdown("**Active strategies:**")
        for sig in result.chart_signals[:3]:
            icon = "🟢" if sig.signal == "LONG" else "🔴"
            st.markdown(
                f"{icon} `{sig.strategy}` — {sig.reason[:80]}  \n"
                f"  *conf: {int(sig.confidence*100)}%*"
            )


# ── News feed ─────────────────────────────────────────────────────────────────

def render_news_feed(result: SignalResult) -> None:
    news_items = result.news_items

    sentiment_colors = {"BULLISH": "#26a69a", "BEARISH": "#ef5350", "NEUTRAL": "#888"}
    impact_colors = {"HIGH": "#ff7043", "MED": "#ffb300", "LOW": "#555"}

    # Aggregate badge
    color = sentiment_colors.get(result.news_sentiment, "#888")
    st.markdown(
        f"""<div style="background:#1e2130;border-radius:8px;padding:10px 14px;margin-bottom:8px;">
          <span style="color:#aaa;font-size:13px;">Aggregate sentiment: </span>
          <span style="color:{color};font-weight:700;font-size:15px;">{result.news_sentiment}</span>
          <span style="color:#aaa;font-size:12px;"> (score: {result.news_score:+.2f})</span>
        </div>""",
        unsafe_allow_html=True,
    )

    if not news_items:
        st.caption("No recent news found. Check your Finnhub API key in the sidebar.")
        return

    for item in news_items[:12]:
        s_color = sentiment_colors.get(item.sentiment, "#888")
        i_color = impact_colors.get(item.impact, "#555")
        age = _time_ago(item.published_at)
        headline_display = item.headline[:110] + ("…" if len(item.headline) > 110 else "")
        url_part = f'<a href="{item.url}" target="_blank" style="color:#7e8fa6;font-size:11px;">↗</a>' if item.url else ""

        st.markdown(
            f"""<div style="background:#1e2130;border-radius:8px;padding:10px 14px;margin-bottom:6px;">
              <div style="font-size:13px;color:#fafafa;line-height:1.4;">{headline_display} {url_part}</div>
              <div style="margin-top:6px;display:flex;gap:6px;align-items:center;">
                <span style="background:{s_color}22;color:{s_color};border:1px solid {s_color}55;
                      border-radius:4px;padding:1px 7px;font-size:11px;font-weight:600;">{item.sentiment}</span>
                <span style="background:{i_color}22;color:{i_color};border:1px solid {i_color}55;
                      border-radius:4px;padding:1px 7px;font-size:11px;">{item.impact}</span>
                <span style="color:#555;font-size:11px;">{item.source} · {age}</span>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )


# ── Indicator mini-panel ──────────────────────────────────────────────────────

def render_indicators(result: SignalResult) -> None:
    ind = result.indicators
    if ind is None:
        return

    cols = st.columns(4)
    metrics = [
        ("RSI(14)", f"{ind.rsi:.1f}" if ind.rsi is not None else "—",
         "#ef5350" if (ind.rsi or 50) > 70 else "#26a69a" if (ind.rsi or 50) < 30 else "#fafafa"),
        ("ATR", f"{ind.atr:.2f}" if ind.atr is not None else "—", "#fafafa"),
        ("MACD", f"{ind.macd_histogram:+.3f}" if ind.macd_histogram is not None else "—",
         "#26a69a" if (ind.macd_histogram or 0) > 0 else "#ef5350"),
        ("BB Squeeze", "YES" if getattr(ind, "bb_squeeze", False) else "NO",
         "#ffb300" if getattr(ind, "bb_squeeze", False) else "#555"),
    ]
    for col, (label, value, color) in zip(cols, metrics):
        col.markdown(
            f"""<div style="background:#1e2130;border-radius:8px;padding:10px;text-align:center;">
              <div style="color:#aaa;font-size:11px;">{label}</div>
              <div style="color:{color};font-size:20px;font-weight:700;">{value}</div>
            </div>""",
            unsafe_allow_html=True,
        )


# ── Pattern summary ───────────────────────────────────────────────────────────

def render_pattern_summary(result: SignalResult) -> None:
    patterns = result.patterns
    if not patterns or not patterns.patterns:
        st.caption("No chart patterns detected.")
        return

    bias_color = {"BULLISH": "#26a69a", "BEARISH": "#ef5350", "NEUTRAL": "#888"}
    color = bias_color.get(patterns.dominant_bias, "#888")
    st.markdown(
        f"**Patterns** — dominant bias: <span style='color:{color};font-weight:700;'>{patterns.dominant_bias}</span>",
        unsafe_allow_html=True,
    )
    for p in patterns.patterns:
        bc = bias_color.get(p.bias, "#888")
        st.markdown(
            f"- <span style='color:{bc};'>{p.name}</span> — {p.description} "
            f"*(conf: {int(p.confidence*100)}%)*",
            unsafe_allow_html=True,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _time_ago(dt: datetime) -> str:
    now = datetime.now(tz=timezone.utc)
    delta = now - dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else now - dt
    secs = int(delta.total_seconds())
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"
