"""
UI Renderer — Streamlit + Plotly chart functions.
Called by app.py to render all sections of the analysis report.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import utils.ta_compat as ta
from models.report import ReportData, MTFResult
from utils.formatting import fmt_price, fmt_pct, fmt_market_cap, score_color, direction_color


def render_header(report: ReportData) -> None:
    """Company header with price and metadata."""
    change_color = "#00c851" if report.price_change_pct >= 0 else "#ff4444"
    arrow = "▲" if report.price_change_pct >= 0 else "▼"

    vp = getattr(report.indicators, "volatility_percentile", 50.0)
    if vp >= 80:
        vp_color, vp_label = "#ff4444", "HIGH VOL"
    elif vp >= 50:
        vp_color, vp_label = "#f0a500", "MED VOL"
    else:
        vp_color, vp_label = "#00c851", "LOW VOL"

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                padding: 20px 24px; border-radius: 12px; margin-bottom: 20px;
                border: 1px solid #2d2d4e;">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
            <div>
                <span style="font-size:28px; font-weight:700; color:#e0e0e0;">{report.ticker}</span>
                <span style="font-size:16px; color:#aaaaaa; margin-left:12px;">{report.company_name}</span>
            </div>
            <div style="text-align:right;">
                <span style="font-size:26px; font-weight:700; color:#e0e0e0;">{fmt_price(report.current_price)}</span>
                <span style="font-size:16px; color:{change_color}; margin-left:10px;">
                    {arrow} {fmt_pct(report.price_change_pct)}
                </span>
            </div>
        </div>
        <div style="margin-top:8px; color:#888888; font-size:13px;">
            Sector: <b style="color:#aaaaaa;">{report.sector}</b>
            &nbsp;&nbsp;|&nbsp;&nbsp;
            Market Cap: <b style="color:#aaaaaa;">{fmt_market_cap(report.market_cap)}</b>
            &nbsp;&nbsp;|&nbsp;&nbsp;
            Generated: <b style="color:#aaaaaa;">{report.generated_at.strftime('%Y-%m-%d %H:%M')}</b>
            &nbsp;&nbsp;|&nbsp;&nbsp;
            <span style="font-size:11px; font-weight:700; color:{vp_color};
                         background:{vp_color}22; border:1px solid {vp_color};
                         border-radius:4px; padding:2px 7px;">
                {vp_label} {vp:.0f}th pct
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _plot_colors() -> dict:
    """Return Plotly bg/font colors based on current theme."""
    light = st.session_state.get("light_theme", False)
    if light:
        return {
            "paper_bgcolor": "#f0f2f6",
            "plot_bgcolor":  "#ffffff",
            "font_color":    "#1a1a2e",
            "grid_color":    "#d0d4e0",
            "panel_bg":      "#e8eaf2",
            "panel_border":  "#b0b8d0",
            "text_muted":    "#555555",
        }
    return {
        "paper_bgcolor": "#0d0d1a",
        "plot_bgcolor":  "#111122",
        "font_color":    "#e0e0e0",
        "grid_color":    "#1e1e2e",
        "panel_bg":      "#1a1a2e",
        "panel_border":  "#2d2d4e",
        "text_muted":    "#aaaaaa",
    }


def render_score_panel(report: ReportData) -> None:
    """Trade score gauge, signal label, and component mini bar chart."""
    score = report.score.total_score
    label = report.score.signal_label
    win_prob = report.score.win_probability
    ev = report.score.expected_value
    color = score_color(score)
    th = _plot_colors()

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 48, "color": color}, "suffix": ""},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#555"},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": th["panel_bg"],
            "borderwidth": 2,
            "bordercolor": "#333",
            "steps": [
                {"range": [0, 20],   "color": "#2d0808"},
                {"range": [20, 35],  "color": "#2d1608"},
                {"range": [35, 50],  "color": "#2d2808"},
                {"range": [50, 65],  "color": "#1a2d1a"},
                {"range": [65, 80],  "color": "#0d2d1a"},
                {"range": [80, 100], "color": "#082d12"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.75,
                "value": score,
            },
        },
        title={"text": "TRADE SCORE", "font": {"size": 14, "color": th["text_muted"]}},
    ))
    fig.update_layout(
        paper_bgcolor=th["paper_bgcolor"],
        height=250,
        margin=dict(t=40, b=10, l=20, r=20),
    )
    st.plotly_chart(fig, width="stretch")

    regime = getattr(report.score, "regime", "NEUTRAL")
    regime_colors = {
        "TRENDING":      ("#00c851", "#0a2e18"),
        "CONSOLIDATING": ("#f0a500", "#2e2200"),
        "EXTENDED":      ("#ff8800", "#2e1800"),
        "VOLATILE":      ("#ff4444", "#2e0808"),
        "NEUTRAL":       ("#888888", "#1e1e1e"),
    }
    reg_fg, reg_bg = regime_colors.get(regime, ("#888888", "#1e1e1e"))

    st.markdown(f"""
    <div style="text-align:center; background:{th['panel_bg']}; padding:12px; border-radius:8px;
                border:1px solid {color}; margin-top:-10px;">
        <span style="font-size:22px; font-weight:700; color:{color};">{label}</span>
        &nbsp;
        <span style="font-size:11px; font-weight:700; color:{reg_fg};
                     background:{reg_bg}; border:1px solid {reg_fg};
                     border-radius:4px; padding:2px 7px; vertical-align:middle;">
            {regime}
        </span>
        <br/>
        <span style="color:{th['text_muted']}; font-size:13px;">
            Win Probability: <b style="color:{th['font_color']};">{win_prob*100:.1f}%</b>
            &nbsp;&nbsp;|&nbsp;&nbsp;
            EV per $100 risked: <b style="color:{'#00c851' if ev>0 else '#ff4444'};">${ev:+.2f}</b>
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Component mini bar chart ──────────────────────────────────────────────
    cs = report.score.component_scores
    comp_names  = ["RSI", "MACD", "Bollinger", "Trend", "Volume", "ATR", "Stoch", "Pattern"]
    comp_vals   = [cs.rsi, cs.macd, cs.bollinger, cs.trend,
                   cs.volume, cs.atr_momentum, cs.stochastic, cs.pattern]
    comp_weights = [15, 15, 10, 20, 10, 10, 10, 10]
    comp_colors  = [score_color(v * 10) for v in comp_vals]
    comp_labels  = [f"{v:.1f} ({w}%)" for v, w in zip(comp_vals, comp_weights)]

    fig2 = go.Figure(go.Bar(
        x=comp_vals,
        y=comp_names,
        orientation="h",
        marker_color=comp_colors,
        text=comp_labels,
        textposition="outside",
        textfont=dict(size=10, color=th["text_muted"]),
    ))
    fig2.update_layout(
        title=dict(text="Component Scores (0–10)", font=dict(size=12, color=th["text_muted"])),
        paper_bgcolor=th["paper_bgcolor"],
        plot_bgcolor=th["plot_bgcolor"],
        font=dict(color=th["font_color"], size=11),
        xaxis=dict(range=[0, 13], gridcolor=th["grid_color"], showgrid=True, zeroline=False),
        yaxis=dict(gridcolor=th["grid_color"]),
        height=240,
        margin=dict(t=30, b=10, l=10, r=80),
    )
    st.plotly_chart(fig2, width="stretch")


def render_trade_setup(report: ReportData) -> None:
    """Trade setup card: entry, stop loss, take profit."""
    ts = report.trade_setup
    if ts.direction == "NO TRADE":
        st.markdown("""
        <div style="background:#1a1a2e; padding:20px; border-radius:10px;
                    border:2px solid #888888; text-align:center;">
            <span style="font-size:20px; color:#888888;">⚠ NO TRADE</span><br/>
            <span style="color:#666; font-size:13px;">Signal is neutral. Wait for a stronger setup.</span>
        </div>
        """, unsafe_allow_html=True)
        return

    dir_color = direction_color(ts.direction)
    sl_pct = f"-{ts.stop_pct:.2f}%"
    tp_pct = f"+{ts.target_pct:.2f}%"

    st.markdown(f"""
    <div style="background:#1a1a2e; padding:20px; border-radius:10px;
                border:2px solid {dir_color};">
        <div style="font-size:18px; font-weight:700; color:{dir_color}; margin-bottom:14px;">
            {ts.direction} TRADE SETUP
        </div>
        <table style="width:100%; color:#e0e0e0; font-size:14px; border-collapse:collapse;">
            <tr style="border-bottom:1px solid #2d2d4e;">
                <td style="padding:6px 0; color:#aaaaaa;">Entry Price</td>
                <td style="text-align:right; font-weight:600;">{fmt_price(ts.entry)}</td>
            </tr>
            <tr style="border-bottom:1px solid #2d2d4e;">
                <td style="padding:6px 0; color:#ff4444;">Stop Loss</td>
                <td style="text-align:right; font-weight:600; color:#ff4444;">
                    {fmt_price(ts.stop_loss)} <span style="font-size:12px;">({sl_pct})</span>
                </td>
            </tr>
            <tr style="border-bottom:1px solid #2d2d4e;">
                <td style="padding:6px 0; color:#00c851;">Take Profit</td>
                <td style="text-align:right; font-weight:600; color:#00c851;">
                    {fmt_price(ts.take_profit)} <span style="font-size:12px;">({tp_pct})</span>
                </td>
            </tr>
            <tr style="border-bottom:1px solid #2d2d4e;">
                <td style="padding:6px 0; color:#aaaaaa;">Risk</td>
                <td style="text-align:right;">{fmt_price(ts.risk_amount)}</td>
            </tr>
            <tr style="border-bottom:1px solid #2d2d4e;">
                <td style="padding:6px 0; color:#aaaaaa;">Reward</td>
                <td style="text-align:right;">{fmt_price(ts.reward_amount)}</td>
            </tr>
            <tr>
                <td style="padding:6px 0; color:#aaaaaa;">R:R Ratio</td>
                <td style="text-align:right; font-weight:700; color:#ffbb33;">1 : {ts.rr_ratio:.1f}</td>
            </tr>
        </table>
        <div style="margin-top:12px; padding:8px; background:#0d0d1a; border-radius:6px;
                    font-size:12px; color:#888888; text-align:center;">
            ATR(14) used: {fmt_price(ts.atr_used)}
        </div>
    </div>
    """, unsafe_allow_html=True)


def _find_sr_levels(
    df: pd.DataFrame,
    n_swing: int = 5,
    n_levels: int = 5,
    tolerance: float = 0.005,
) -> list:
    """
    Find key support/resistance levels from swing highs and lows.
    Returns up to n_levels price levels, ranked by number of touches.
    """
    import numpy as _np
    highs  = df["High"].values
    lows   = df["Low"].values
    levels = []

    for i in range(n_swing, len(df) - n_swing):
        window_h = highs[i - n_swing : i + n_swing + 1]
        window_l = lows[i  - n_swing : i + n_swing + 1]
        if highs[i] == window_h.max():
            levels.append(highs[i])
        if lows[i] == window_l.min():
            levels.append(lows[i])

    if not levels:
        return []

    levels = sorted(levels)
    clusters: list = []
    group = [levels[0]]
    for lvl in levels[1:]:
        if lvl <= group[-1] * (1 + tolerance):
            group.append(lvl)
        else:
            clusters.append((_np.mean(group), len(group)))
            group = [lvl]
    clusters.append((_np.mean(group), len(group)))

    clusters.sort(key=lambda x: -x[1])
    return [c[0] for c in clusters[:n_levels]]


def render_candlestick_chart(report: ReportData) -> None:
    """Candlestick chart with Bollinger Bands and EMAs overlay."""
    df = report.last_90 if hasattr(report, 'last_90') else report.df.iloc[-90:]
    th = _plot_colors()

    fig = make_subplots(rows=1, cols=1)

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price",
        increasing_line_color="#00c851",
        decreasing_line_color="#ff4444",
    ))

    # Bollinger Bands
    bb = ta.bbands(df["Close"], length=20, std=2)
    if bb is not None and not bb.empty:
        fig.add_trace(go.Scatter(x=df.index, y=bb.iloc[:, 2], name="BB Upper",
                                  line=dict(color="#4488ff", width=1, dash="dash"), opacity=0.6))
        fig.add_trace(go.Scatter(x=df.index, y=bb.iloc[:, 1], name="BB Mid",
                                  line=dict(color="#aaaaff", width=1), opacity=0.4))
        fig.add_trace(go.Scatter(x=df.index, y=bb.iloc[:, 0], name="BB Lower",
                                  line=dict(color="#4488ff", width=1, dash="dash"), opacity=0.6,
                                  fill="tonexty", fillcolor="rgba(68,136,255,0.05)"))

    # EMAs
    ema20 = ta.ema(df["Close"], length=20)
    ema50 = ta.ema(df["Close"], length=50)
    if ema20 is not None:
        fig.add_trace(go.Scatter(x=df.index, y=ema20, name="EMA20",
                                  line=dict(color="#ffbb33", width=1.5)))
    if ema50 is not None:
        fig.add_trace(go.Scatter(x=df.index, y=ema50, name="EMA50",
                                  line=dict(color="#ff8800", width=1.5)))

    # VWAP (20-day rolling)
    vwap_s = ta.vwap(df["High"], df["Low"], df["Close"], df["Volume"], length=20)
    if vwap_s is not None and not vwap_s.isna().all():
        fig.add_trace(go.Scatter(
            x=df.index, y=vwap_s, name="VWAP(20)",
            line=dict(color="#cc88ff", width=1.5, dash="dot"),
            opacity=0.85,
        ))

    # Support / Resistance zones
    price_now = float(df["Close"].iloc[-1])
    sr_levels = _find_sr_levels(df)
    for lvl in sr_levels:
        is_support = lvl < price_now
        lvl_color  = "#44bb88" if is_support else "#bb4466"
        label      = f"S {fmt_price(lvl)}" if is_support else f"R {fmt_price(lvl)}"
        fig.add_hline(
            y=lvl,
            line_color=lvl_color, line_width=1, line_dash="dot", opacity=0.55,
            annotation_text=label, annotation_position="left",
            annotation_font_size=10, annotation_font_color=lvl_color,
        )

    # Trade levels (only if trade exists)
    ts = report.trade_setup
    entry_line_color = "#333333" if st.session_state.get("light_theme") else "#ffffff"
    if ts.direction != "NO TRADE":
        fig.add_hline(y=ts.entry, line_color=entry_line_color, line_width=1.5, line_dash="dot",
                      annotation_text=f"Entry {fmt_price(ts.entry)}", annotation_position="right")
        fig.add_hline(y=ts.stop_loss, line_color="#ff4444", line_width=1.5, line_dash="dash",
                      annotation_text=f"SL {fmt_price(ts.stop_loss)}", annotation_position="right")
        fig.add_hline(y=ts.take_profit, line_color="#00c851", line_width=1.5, line_dash="dash",
                      annotation_text=f"TP {fmt_price(ts.take_profit)}", annotation_position="right")

    fig.update_layout(
        title=f"{report.ticker} — Price Chart (90 days)",
        paper_bgcolor=th["paper_bgcolor"],
        plot_bgcolor=th["plot_bgcolor"],
        font=dict(color=th["font_color"]),
        xaxis=dict(gridcolor=th["grid_color"], showgrid=True),
        yaxis=dict(gridcolor=th["grid_color"], showgrid=True),
        legend=dict(bgcolor=th["panel_bg"], bordercolor="#333"),
        height=450,
        margin=dict(t=40, b=20, l=10, r=80),
        xaxis_rangeslider_visible=False,
    )
    st.plotly_chart(fig, width="stretch")


def render_macd_chart(report: ReportData) -> None:
    """MACD chart."""
    df = report.df.iloc[-90:]
    th = _plot_colors()
    macd_df = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    if macd_df is None or macd_df.empty:
        return

    macd_line = macd_df.iloc[:, 0]
    macd_hist = macd_df.iloc[:, 1]
    macd_sig  = macd_df.iloc[:, 2]

    colors = ["#00c851" if v >= 0 else "#ff4444" for v in macd_hist]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=macd_hist, name="Histogram",
                          marker_color=colors, opacity=0.8))
    fig.add_trace(go.Scatter(x=df.index, y=macd_line, name="MACD",
                              line=dict(color="#4488ff", width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=macd_sig, name="Signal",
                              line=dict(color="#ffbb33", width=1.5)))
    fig.add_hline(y=0, line_color="#555", line_width=1)

    fig.update_layout(
        title="MACD (12, 26, 9)",
        paper_bgcolor=th["paper_bgcolor"],
        plot_bgcolor=th["plot_bgcolor"],
        font=dict(color=th["font_color"]),
        xaxis=dict(gridcolor=th["grid_color"]),
        yaxis=dict(gridcolor=th["grid_color"]),
        height=220,
        margin=dict(t=40, b=20, l=10, r=10),
        legend=dict(bgcolor=th["panel_bg"]),
    )
    st.plotly_chart(fig, width="stretch")


def render_rsi_chart(report: ReportData) -> None:
    """RSI chart with overbought/oversold zones."""
    df = report.df.iloc[-90:]
    th = _plot_colors()
    rsi = ta.rsi(df["Close"], length=14)
    if rsi is None or rsi.empty:
        return

    fig = go.Figure()
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,68,68,0.1)", line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0,200,81,0.1)", line_width=0)
    fig.add_hline(y=70, line_color="#ff4444", line_width=1, line_dash="dash")
    fig.add_hline(y=30, line_color="#00c851", line_width=1, line_dash="dash")
    fig.add_hline(y=50, line_color="#555", line_width=1)

    fig.add_trace(go.Scatter(x=df.index, y=rsi, name="RSI(14)",
                              line=dict(color="#aa88ff", width=2)))

    fig.update_layout(
        title="RSI (14)",
        paper_bgcolor=th["paper_bgcolor"],
        plot_bgcolor=th["plot_bgcolor"],
        font=dict(color=th["font_color"]),
        xaxis=dict(gridcolor=th["grid_color"]),
        yaxis=dict(gridcolor=th["grid_color"], range=[0, 100]),
        height=200,
        margin=dict(t=40, b=20, l=10, r=10),
    )
    st.plotly_chart(fig, width="stretch")


def render_volume_chart(report: ReportData) -> None:
    """Volume chart with SMA20 overlay."""
    df = report.df.iloc[-90:]
    vol_sma = df["Volume"].rolling(20).mean()
    colors = ["#00c851" if c >= o else "#ff4444"
              for c, o in zip(df["Close"], df["Open"])]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                          marker_color=colors, opacity=0.7))
    fig.add_trace(go.Scatter(x=df.index, y=vol_sma, name="Vol SMA20",
                              line=dict(color="#ffbb33", width=1.5)))

    th = _plot_colors()
    fig.update_layout(
        title="Volume",
        paper_bgcolor=th["paper_bgcolor"],
        plot_bgcolor=th["plot_bgcolor"],
        font=dict(color=th["font_color"]),
        xaxis=dict(gridcolor=th["grid_color"]),
        yaxis=dict(gridcolor=th["grid_color"]),
        height=200,
        margin=dict(t=40, b=20, l=10, r=10),
        legend=dict(bgcolor=th["panel_bg"]),
    )
    st.plotly_chart(fig, width="stretch")


def render_indicator_breakdown(report: ReportData) -> None:
    """Indicator breakdown — one row per indicator using Streamlit columns."""
    ind = report.indicators
    cs = report.score.component_scores

    st.markdown("### Indicator Breakdown")

    rows = [
        ("RSI (14)",             f"{ind.rsi:.1f}",                                                                    cs.rsi,          15),
        ("MACD Histogram",       f"{ind.macd_histogram:+.4f}",                                                        cs.macd,         15),
        ("Bollinger Bands",      f"{'Below' if ind.ema20 < ind.bb_mid else 'Above'} Mid",                             cs.bollinger,    10),
        ("Trend (MA Alignment)", f"EMA20 {'>' if ind.ema20 > ind.ema50 else '<'} EMA50",                              cs.trend,        20),
        ("Volume",               f"{ind.current_volume/ind.volume_sma20:.2f}× avg" if ind.volume_sma20 else "N/A",   cs.volume,       10),
        ("ATR Momentum",         f"{ind.atr:.3f}",                                                                    cs.atr_momentum, 10),
        ("Stochastic %K",        f"{ind.stoch_k:.1f}",                                                               cs.stochastic,   10),
        ("Pattern Signals",      _build_pattern_str(ind),                                                             cs.pattern,      10),
    ]

    # Header row
    hcols = st.columns([2, 2, 4, 1])
    hcols[0].markdown("<span style='color:#666; font-size:12px;'>INDICATOR</span>", unsafe_allow_html=True)
    hcols[1].markdown("<span style='color:#666; font-size:12px;'>VALUE</span>", unsafe_allow_html=True)
    hcols[2].markdown("<span style='color:#666; font-size:12px;'>SIGNAL STRENGTH</span>", unsafe_allow_html=True)
    hcols[3].markdown("<span style='color:#666; font-size:12px;'>WT</span>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#2d2d4e; margin:4px 0 8px 0;'>", unsafe_allow_html=True)

    for label, value, score, weight in rows:
        filled = int(round(score))
        empty = 10 - filled
        color = score_color(score * 10)
        bar = "█" * filled + "░" * empty

        cols = st.columns([2, 2, 4, 1])
        cols[0].markdown(f"<span style='color:#aaaaaa; font-size:13px;'>{label}</span>", unsafe_allow_html=True)
        cols[1].markdown(f"<span style='color:#e0e0e0; font-size:13px;'>{value}</span>", unsafe_allow_html=True)
        cols[2].markdown(
            f"<span style='color:{color}; font-family:monospace; font-size:13px;'>{bar}</span>"
            f"<span style='color:#cccccc; font-size:13px;'> {score:.1f}/10</span>",
            unsafe_allow_html=True,
        )
        cols[3].markdown(f"<span style='color:#666; font-size:12px;'>{weight}%</span>", unsafe_allow_html=True)
        st.markdown("<div style='border-bottom:1px solid #1e1e2e; margin:2px 0;'></div>", unsafe_allow_html=True)


def _build_pattern_str(ind) -> str:
    patterns = []
    if ind.golden_cross:  patterns.append("Golden Cross")
    if ind.death_cross:   patterns.append("Death Cross")
    if ind.above_200_sma: patterns.append("Above SMA200")
    else:                 patterns.append("Below SMA200")
    if ind.volume_spike:  patterns.append("Vol Spike")
    if ind.bb_squeeze:    patterns.append("BB Squeeze")
    return ", ".join(patterns) if patterns else "None"


# ── Multi-Timeframe Renderer ──────────────────────────────────────────────────

def render_mtf_panel(report: ReportData) -> None:
    """Multi-timeframe agreement panel: daily vs weekly."""
    mtf = report.mtf
    if mtf is None:
        return

    agree_color = {"BULLISH": "#00c851", "BEARISH": "#ff4444", "MIXED": "#ffbb33", "UNKNOWN": "#888888"}
    dir_color = agree_color.get(mtf.agreement_direction, "#888888")
    agree_icon = "✅" if mtf.agreement else "⚠"

    st.markdown("### Multi-Timeframe Analysis")

    c1, c2, c3 = st.columns(3)

    with c1:
        d_color = score_color(mtf.daily_score)
        st.markdown(f"""
        <div style="background:#1a1a2e; padding:16px; border-radius:10px;
                    border:1px solid {d_color}; text-align:center;">
            <div style="color:#888; font-size:12px; margin-bottom:4px;">DAILY SCORE</div>
            <div style="font-size:32px; font-weight:700; color:{d_color};">{mtf.daily_score:.0f}</div>
            <div style="color:{d_color}; font-size:13px; margin-top:4px;">{mtf.daily_label}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        w_color = score_color(mtf.weekly_score) if mtf.weekly_score > 0 else "#888"
        w_label = mtf.weekly_label if mtf.weekly_label != "N/A" else "No Data"
        st.markdown(f"""
        <div style="background:#1a1a2e; padding:16px; border-radius:10px;
                    border:1px solid {w_color}; text-align:center;">
            <div style="color:#888; font-size:12px; margin-bottom:4px;">WEEKLY SCORE</div>
            <div style="font-size:32px; font-weight:700; color:{w_color};">{f"{mtf.weekly_score:.0f}" if mtf.weekly_score > 0 else "—"}</div>
            <div style="color:{w_color}; font-size:13px; margin-top:4px;">{w_label}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        c_color = score_color(mtf.combined_score)
        st.markdown(f"""
        <div style="background:#1a1a2e; padding:16px; border-radius:10px;
                    border:2px solid {dir_color}; text-align:center;">
            <div style="color:#888; font-size:12px; margin-bottom:4px;">MTF COMBINED</div>
            <div style="font-size:32px; font-weight:700; color:{c_color};">{mtf.combined_score:.0f}</div>
            <div style="color:{dir_color}; font-size:13px; margin-top:4px;">
                {agree_icon} {mtf.agreement_direction}
            </div>
        </div>
        """, unsafe_allow_html=True)

    if not mtf.agreement:
        st.warning("⚠ Daily and weekly timeframes disagree — consider waiting for alignment before entering.")
    else:
        st.success(f"✅ Both timeframes are **{mtf.agreement_direction}** — higher confidence setup.")


# ── Backtester Renderer ───────────────────────────────────────────────────────

def render_backtest_results(result) -> None:
    """Render backtest stats, equity curve, and trade log."""
    from agents.backtester import BacktestResult

    st.markdown(f"### Backtest Results — {result.ticker} ({result.period})")
    st.caption(f"Score threshold: ≥ {result.score_threshold} for LONG | ≤ {100 - result.score_threshold} for SHORT")

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpis = [
        (k1, "Total Trades",   str(result.total_trades),          "#e0e0e0"),
        (k2, "Win Rate",       f"{result.win_rate:.1f}%",         "#00c851" if result.win_rate >= 50 else "#ff4444"),
        (k3, "Profit Factor",  f"{result.profit_factor:.2f}",     "#00c851" if result.profit_factor >= 1.5 else "#ffbb33"),
        (k4, "Total Return",   f"{result.total_return_pct:+.1f}%","#00c851" if result.total_return_pct > 0 else "#ff4444"),
        (k5, "Max Drawdown",   f"-{result.max_drawdown_pct:.1f}%","#ff4444"),
        (k6, "Avg Bars Held",  f"{result.avg_bars_held:.1f}",     "#e0e0e0"),
    ]
    for col, label, val, color in kpis:
        col.markdown(f"""
        <div style="background:#1a1a2e; padding:12px; border-radius:8px; text-align:center;">
            <div style="color:#666; font-size:11px;">{label}</div>
            <div style="font-size:20px; font-weight:700; color:{color};">{val}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ── Win / Loss / Timeout breakdown ────────────────────────────────────────
    w2, w3 = st.columns(2)
    with w2:
        st.markdown(f"""
        <div style="background:#1a1a2e; padding:12px; border-radius:8px; margin-top:8px;">
            <div style="display:flex; justify-content:space-around; text-align:center;">
                <div><div style="color:#00c851; font-size:22px; font-weight:700;">{result.wins}</div>
                     <div style="color:#666; font-size:11px;">WINS</div></div>
                <div><div style="color:#ff4444; font-size:22px; font-weight:700;">{result.losses}</div>
                     <div style="color:#666; font-size:11px;">LOSSES</div></div>
                <div><div style="color:#ffbb33; font-size:22px; font-weight:700;">{result.timeouts}</div>
                     <div style="color:#666; font-size:11px;">TIMEOUTS</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with w3:
        avg_win_color  = "#00c851" if result.avg_win_pct  > 0 else "#ff4444"
        avg_loss_color = "#ff4444" if result.avg_loss_pct < 0 else "#00c851"
        st.markdown(f"""
        <div style="background:#1a1a2e; padding:12px; border-radius:8px; margin-top:8px;">
            <div style="display:flex; justify-content:space-around; text-align:center;">
                <div><div style="color:{avg_win_color}; font-size:20px; font-weight:700;">{result.avg_win_pct:+.1f}%</div>
                     <div style="color:#666; font-size:11px;">AVG WIN</div></div>
                <div><div style="color:{avg_loss_color}; font-size:20px; font-weight:700;">{result.avg_loss_pct:+.1f}%</div>
                     <div style="color:#666; font-size:11px;">AVG LOSS</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Equity curve ──────────────────────────────────────────────────────────
    if result.equity_curve:
        eq = result.equity_curve
        eq_pct = [(v - 1.0) * 100 for v in eq]
        colors = ["#00c851" if v >= 0 else "#ff4444" for v in eq_pct]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=eq_pct, mode="lines",
            line=dict(color="#4488ff", width=2),
            fill="tozeroy",
            fillcolor="rgba(68,136,255,0.08)",
            name="Equity %",
        ))
        th = _plot_colors()
        fig.add_hline(y=0, line_color="#555", line_width=1)
        fig.update_layout(
            title="Equity Curve (%)",
            paper_bgcolor=th["paper_bgcolor"], plot_bgcolor=th["plot_bgcolor"],
            font=dict(color=th["font_color"]),
            xaxis=dict(gridcolor=th["grid_color"], title="Bar"),
            yaxis=dict(gridcolor=th["grid_color"], title="Return %"),
            height=280, margin=dict(t=40, b=20, l=10, r=10),
        )
        st.plotly_chart(fig, width="stretch")

    # ── Performance by regime ─────────────────────────────────────────────────
    if result.trades:
        regime_order = ["TRENDING", "CONSOLIDATING", "EXTENDED", "VOLATILE", "NEUTRAL"]
        regime_colors = {
            "TRENDING":      "#00c851",
            "CONSOLIDATING": "#f0a500",
            "EXTENDED":      "#ff8800",
            "VOLATILE":      "#ff4444",
            "NEUTRAL":       "#888888",
        }
        regime_stats = {}
        for t in result.trades:
            r = getattr(t, "regime", "NEUTRAL")
            if r not in regime_stats:
                regime_stats[r] = {"wins": 0, "total": 0, "pnl": 0.0}
            regime_stats[r]["total"] += 1
            regime_stats[r]["pnl"] += t.pnl_pct
            if t.outcome == "WIN":
                regime_stats[r]["wins"] += 1

        if regime_stats:
            st.markdown("#### Performance by Market Regime")
            regime_cols = st.columns(len(regime_stats))
            for col, reg in zip(regime_cols, [r for r in regime_order if r in regime_stats]):
                s = regime_stats[reg]
                wr = s["wins"] / s["total"] * 100 if s["total"] else 0
                avg_pnl = s["pnl"] / s["total"] if s["total"] else 0
                fg = regime_colors.get(reg, "#888888")
                col.markdown(f"""
                <div style="background:#1a1a2e; padding:12px; border-radius:8px;
                            border:1px solid {fg}; text-align:center; margin-bottom:8px;">
                    <div style="font-size:11px; font-weight:700; color:{fg};
                                margin-bottom:4px;">{reg}</div>
                    <div style="font-size:20px; font-weight:700;
                                color:{'#00c851' if wr >= 50 else '#ff4444'};">{wr:.0f}%</div>
                    <div style="color:#666; font-size:10px;">win rate</div>
                    <div style="font-size:13px; font-weight:600; margin-top:4px;
                                color:{'#00c851' if avg_pnl > 0 else '#ff4444'};">
                        {avg_pnl:+.1f}%</div>
                    <div style="color:#555; font-size:10px;">avg PnL · {s['total']} trades</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Trade log ─────────────────────────────────────────────────────────────
    if result.trades:
        st.markdown("#### Trade Log")
        rows = []
        for t in result.trades[-50:]:   # show last 50
            rows.append({
                "Date":       t.entry_date,
                "Regime":     getattr(t, "regime", "—"),
                "Dir":        t.direction,
                "Score":      t.score_at_entry,
                "Entry":      f"${t.entry_price:.2f}",
                "SL":         f"${t.stop_loss:.2f}",
                "TP":         f"${t.take_profit:.2f}",
                "Exit":       f"${t.exit_price:.2f}",
                "P&L %":      f"{t.pnl_pct:+.2f}%",
                "Bars":       t.bars_held,
                "Outcome":    t.outcome,
            })
        import pandas as pd
        df_log = pd.DataFrame(rows)
        st.dataframe(df_log, use_container_width=True, hide_index=True)


# ── Earnings & News Renderer ──────────────────────────────────────────────────

def render_earnings_risk(ticker: str) -> None:
    """Earnings date warning and recent news panel."""
    from data.earnings import get_earnings_risk, get_news

    risk = get_earnings_risk(ticker)
    news = get_news(ticker)

    col_e, col_n = st.columns([1, 2], gap="medium")

    with col_e:
        st.markdown("#### Earnings Risk")
        color = risk["risk_color"]
        date_str = risk["next_earnings_str"]
        label = risk["risk_label"]
        st.markdown(f"""
        <div style="background:#1a1a2e; padding:16px; border-radius:10px;
                    border:2px solid {color}; text-align:center;">
            <div style="font-size:13px; color:#888; margin-bottom:6px;">NEXT EARNINGS</div>
            <div style="font-size:20px; font-weight:700; color:#e0e0e0;">{date_str}</div>
            <div style="margin-top:10px; font-size:13px; color:{color};">{label}</div>
        </div>
        """, unsafe_allow_html=True)
        if risk["is_risky"]:
            st.warning("Avoid new swing entries within 7 days of earnings — gap risk is very high.")

    with col_n:
        st.markdown("#### Recent News")
        if not news:
            st.caption("No recent news found.")
        else:
            for item in news:
                pub = f" · {item['publisher']}" if item['publisher'] else ""
                date = f" · {item['published']}" if item['published'] else ""
                st.markdown(
                    f"- [{item['title']}]({item['link']})"
                    f"<span style='color:#555; font-size:11px;'>{pub}{date}</span>",
                    unsafe_allow_html=True,
                )


def render_earnings_history(report) -> None:
    """Historical earnings reactions: EPS beat/miss + 1-day price move."""
    from data.earnings import get_earnings_history

    history = get_earnings_history(report.ticker, price_df=report.df)
    if not history:
        return

    beats  = [h for h in history if h["beat"] is True]
    misses = [h for h in history if h["beat"] is False]
    beat_rate = len(beats) / len(history) * 100 if history else 0

    reactions = [h["price_reaction_pct"] for h in history if h["price_reaction_pct"] is not None]
    avg_move   = sum(abs(r) for r in reactions) / len(reactions) if reactions else None

    st.markdown("#### Earnings History")

    # Summary KPIs
    k1, k2, k3 = st.columns(3)
    k1.markdown(f"""
    <div style="background:#1a1a2e; padding:10px; border-radius:8px; text-align:center;">
        <div style="color:#666; font-size:11px;">BEAT RATE</div>
        <div style="font-size:20px; font-weight:700;
                    color:{'#00c851' if beat_rate >= 60 else '#ffbb33' if beat_rate >= 40 else '#ff4444'};">
            {beat_rate:.0f}%
        </div>
        <div style="color:#555; font-size:10px;">last {len(history)} qtrs</div>
    </div>""", unsafe_allow_html=True)

    k2.markdown(f"""
    <div style="background:#1a1a2e; padding:10px; border-radius:8px; text-align:center;">
        <div style="color:#666; font-size:11px;">AVG MOVE</div>
        <div style="font-size:20px; font-weight:700; color:#e0e0e0;">
            {"±" + f"{avg_move:.1f}%" if avg_move is not None else "N/A"}
        </div>
        <div style="color:#555; font-size:10px;">day of earnings</div>
    </div>""", unsafe_allow_html=True)

    avg_beat_r  = sum(h["price_reaction_pct"] for h in beats  if h["price_reaction_pct"] is not None)
    avg_miss_r  = sum(h["price_reaction_pct"] for h in misses if h["price_reaction_pct"] is not None)
    n_beat_r    = sum(1 for h in beats  if h["price_reaction_pct"] is not None)
    n_miss_r    = sum(1 for h in misses if h["price_reaction_pct"] is not None)
    beat_avg_s  = f"{avg_beat_r/n_beat_r:+.1f}%" if n_beat_r else "N/A"
    miss_avg_s  = f"{avg_miss_r/n_miss_r:+.1f}%" if n_miss_r else "N/A"

    k3.markdown(f"""
    <div style="background:#1a1a2e; padding:10px; border-radius:8px; text-align:center;">
        <div style="color:#666; font-size:11px;">BEAT / MISS REACTION</div>
        <div style="font-size:14px; font-weight:700; color:#00c851;">Beat: {beat_avg_s}</div>
        <div style="font-size:14px; font-weight:700; color:#ff4444;">Miss: {miss_avg_s}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    # Per-quarter rows
    for h in history:
        beat_val = h["beat"]
        if beat_val is True:
            badge_color, badge_text = "#00c851", "BEAT"
        elif beat_val is False:
            badge_color, badge_text = "#ff4444", "MISS"
        else:
            badge_color, badge_text = "#888888", "N/A"

        react = h["price_reaction_pct"]
        if react is not None:
            r_color = "#00c851" if react >= 0 else "#ff4444"
            react_s = f"{react:+.1f}%"
        else:
            r_color, react_s = "#888888", "—"

        eps_e = f"{h['eps_estimate']:+.2f}" if h["eps_estimate"] is not None else "—"
        eps_a = f"{h['eps_actual']:+.2f}"   if h["eps_actual"]   is not None else "—"
        surp  = f"{h['surprise_pct']:+.1f}%" if h["surprise_pct"] is not None else ""

        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:16px; padding:5px 0;
                    border-bottom:1px solid #1e1e2e; font-size:13px; color:#e0e0e0;">
            <span style="min-width:70px; color:#888;">{h['date']}</span>
            <span style="min-width:44px; font-weight:700; color:{badge_color};
                         background:{badge_color}22; border-radius:4px; padding:1px 6px;
                         font-size:11px;">{badge_text}</span>
            <span style="min-width:120px; color:#aaa;">
                Est: <b style="color:#e0e0e0;">{eps_e}</b>
                &nbsp;Act: <b style="color:#e0e0e0;">{eps_a}</b>
                <span style="color:#666; font-size:11px;">{surp}</span>
            </span>
            <span style="color:{r_color}; font-weight:700;">{react_s}</span>
        </div>
        """, unsafe_allow_html=True)


# ── Position Sizing Renderer ──────────────────────────────────────────────────

def render_position_sizing(entry: float, stop_loss: float, take_profit: float) -> None:
    """Interactive position sizing calculator embedded in the Analyzer tab."""
    from utils.position_sizing import calculate

    st.markdown("#### Position Sizing Calculator")

    ps1, ps2 = st.columns(2)
    with ps1:
        account_size = st.number_input(
            "Account Size ($)", min_value=100.0, max_value=10_000_000.0,
            value=float(st.session_state.get("ps_account", 10000)),
            step=500.0, key="ps_account_input", label_visibility="visible",
        )
    with ps2:
        risk_pct = st.number_input(
            "Risk per Trade (%)", min_value=0.1, max_value=5.0,
            value=float(st.session_state.get("ps_risk", 1.0)),
            step=0.25, key="ps_risk_input", label_visibility="visible",
        )

    if entry > 0 and stop_loss > 0 and take_profit > 0:
        ps = calculate(account_size, risk_pct, entry, stop_loss, take_profit)

        c1, c2, c3, c4 = st.columns(4)
        kpis = [
            (c1, "Shares",         str(ps.shares),                   "#e0e0e0"),
            (c2, "Position Value", f"${ps.position_value:,.2f}",     "#e0e0e0"),
            (c3, "Dollar Risk",    f"${ps.dollar_risk:,.2f}",        "#ff4444"),
            (c4, "Dollar Reward",  f"${ps.dollar_reward:,.2f}",      "#00c851"),
        ]
        for col, label, val, color in kpis:
            col.markdown(f"""
            <div style="background:#1a1a2e; padding:10px; border-radius:8px; text-align:center;">
                <div style="color:#666; font-size:11px;">{label}</div>
                <div style="font-size:18px; font-weight:700; color:{color};">{val}</div>
            </div>
            """, unsafe_allow_html=True)

        st.caption(
            f"Actual risk: {ps.risk_pct_actual:.2f}% of account  ·  "
            f"R:R = 1:{ps.rr_ratio:.1f}  ·  "
            f"Position = {ps.position_value/account_size*100:.1f}% of account"
        )
    else:
        st.caption("Run an analysis first to populate entry/stop/target prices.")


# ── Chart Pattern Renderer ────────────────────────────────────────────────────

def render_chart_patterns(df: pd.DataFrame) -> None:
    """Detect and display chart patterns with confidence scores."""
    from agents.pattern_detector import detect_patterns

    report = detect_patterns(df)

    st.markdown("#### Chart Pattern Recognition")

    if not report.patterns:
        st.caption("No significant patterns detected in the last 60 bars.")
        return

    bias_color = {"BULLISH": "#00c851", "BEARISH": "#ff4444", "NEUTRAL": "#888888"}
    dom_color = bias_color.get(report.dominant_bias, "#888888")

    st.markdown(
        f"<span style='font-size:13px; color:{dom_color};'>Dominant bias: "
        f"<b>{report.dominant_bias}</b></span>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    for p in report.patterns:
        conf_filled = int(p.confidence * 10)
        conf_bar = "█" * conf_filled + "░" * (10 - conf_filled)
        c1, c2, c3 = st.columns([2, 3, 2])
        c1.markdown(
            f"<b style='color:{p.color};'>{p.name}</b>"
            f"<span style='color:#666; font-size:11px;'> ({p.bias})</span>",
            unsafe_allow_html=True,
        )
        c2.markdown(
            f"<span style='color:#aaaaaa; font-size:12px;'>{p.description}</span>",
            unsafe_allow_html=True,
        )
        c3.markdown(
            f"<span style='color:{p.color}; font-family:monospace; font-size:12px;'>{conf_bar}</span>"
            f"<span style='color:#888; font-size:12px;'> {p.confidence*100:.0f}%</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='border-bottom:1px solid #1e1e2e; margin:3px 0;'></div>",
            unsafe_allow_html=True,
        )


# ── Sector Heatmap Renderer ───────────────────────────────────────────────────

def render_sector_heatmap(scan_results: dict, sector_avgs: list[dict]) -> None:
    """Render sector heatmap tiles + drilldown table."""
    import plotly.graph_objects as go
    from utils.formatting import score_color

    st.markdown("#### Sector Heatmap")

    if not sector_avgs:
        st.info("No scan results yet.")
        return

    # Tile grid — one tile per sector
    cols = st.columns(4)
    for i, row in enumerate(sector_avgs):
        color = score_color(row["avg_score"])
        with cols[i % 4]:
            st.markdown(f"""
            <div style="background:#1a1a2e; padding:12px; border-radius:10px;
                        border:1px solid {color}; text-align:center; margin-bottom:10px;">
                <div style="color:#888; font-size:11px; margin-bottom:4px;">{row['sector'][:18]}</div>
                <div style="font-size:24px; font-weight:700; color:{color};">{row['avg_score']:.0f}</div>
                <div style="color:#555; font-size:11px;">{row['count']} stocks</div>
            </div>
            """, unsafe_allow_html=True)

    # Bar chart
    sectors_sorted = [r["sector"] for r in sector_avgs]
    scores_sorted  = [r["avg_score"] for r in sector_avgs]
    bar_colors = [score_color(s) for s in scores_sorted]

    fig = go.Figure(go.Bar(
        x=scores_sorted,
        y=sectors_sorted,
        orientation="h",
        marker_color=bar_colors,
        text=[f"{s:.1f}" for s in scores_sorted],
        textposition="outside",
    ))
    th = _plot_colors()
    fig.add_vline(x=50, line_color="#555", line_width=1, line_dash="dash")
    fig.update_layout(
        title="Average Trade Score by Sector",
        paper_bgcolor=th["paper_bgcolor"], plot_bgcolor=th["plot_bgcolor"],
        font=dict(color=th["font_color"]),
        xaxis=dict(range=[0, 100], gridcolor=th["grid_color"]),
        yaxis=dict(gridcolor=th["grid_color"]),
        height=380, margin=dict(t=40, b=20, l=20, r=60),
    )
    st.plotly_chart(fig, width="stretch")

    # Drilldown: top stocks per sector
    selected_sector = st.selectbox(
        "Drilldown — select sector",
        options=sectors_sorted,
        key="heatmap_sector_select",
        label_visibility="visible",
    )
    rows = scan_results.get(selected_sector, [])
    if rows:
        import pandas as pd
        df_sector = pd.DataFrame(rows)[
            ["ticker", "company", "price", "chg_pct", "score", "signal", "direction", "mtf", "win_prob"]
        ]
        df_sector.columns = ["Ticker", "Company", "Price", "Chg%", "Score", "Signal", "Dir", "MTF", "Win%"]
        df_sector = df_sector.sort_values("Score", ascending=False)
        st.dataframe(df_sector, use_container_width=True, hide_index=True)


# ── Theme CSS ─────────────────────────────────────────────────────────────────

def get_theme_css(is_light: bool = False) -> str:
    """Return CSS string for dark (default) or light theme."""
    if is_light:
        return """
<style>
    body, .stApp { background-color: #f0f2f6 !important; color: #1a1a2e !important; }
    .stTextInput > div > input, .stNumberInput > div > input,
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        color: #1a1a2e !important;
        border: 1px solid #b0b8d0 !important;
        border-radius: 8px !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #3a7bd5, #2a5fbf) !important;
        color: #ffffff !important; border: 1px solid #2a5fbf !important;
        border-radius: 8px !important; font-size: 14px !important; font-weight: 600 !important;
    }
    .stButton > button:hover { background: linear-gradient(135deg, #4a8be5, #3a6fcf) !important; }
    .stTabs [data-baseweb="tab"] { color: #444 !important; font-size: 14px; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: #2a5fbf !important; border-bottom: 2px solid #2a5fbf !important; }
    .block-container { padding-top: 1.2rem; }
    hr { border-color: #c8cfe0; }
    [data-testid="stDataFrame"] { background: #ffffff; }
    .stMarkdown p { color: #1a1a2e !important; }
    .stCaption { color: #555 !important; }
    [data-testid="stToggle"] label { color: #1a1a2e !important; }
</style>
"""
    else:
        return """
<style>
    body, .stApp { background-color: #0d0d1a; color: #e0e0e0; }
    .stTextInput > div > input, .stNumberInput > div > input,
    .stSelectbox > div > div {
        background-color: #1a1a2e !important;
        color: #e0e0e0 !important;
        border: 1px solid #3d3d5e !important;
        border-radius: 8px !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1a3a6e, #0d2a4e);
        color: #e0e0e0; border: 1px solid #3d5a8e;
        border-radius: 8px; font-size: 14px; font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover { background: linear-gradient(135deg, #2a4a8e, #1a3a6e); }
    .stTabs [data-baseweb="tab"] { color: #aaaaaa; font-size: 14px; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: #4488ff !important; border-bottom: 2px solid #4488ff; }
    .block-container { padding-top: 1.2rem; }
    hr { border-color: #2d2d4e; }
    [data-testid="stDataFrame"] { background: #1a1a2e; }
</style>
"""


# ── Export Buttons ────────────────────────────────────────────────────────────

def render_export_buttons(report: ReportData) -> None:
    """Download buttons for HTML report and CSV data."""
    from utils.export import build_html_report, build_csv_data

    html_bytes = build_html_report(report)
    csv_bytes  = build_csv_data(report)
    fname_base = f"katraswing_{report.ticker}_{report.generated_at.strftime('%Y%m%d_%H%M')}"

    st.markdown("#### Export Report")
    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        st.download_button(
            label="📄 Download HTML",
            data=html_bytes,
            file_name=f"{fname_base}.html",
            mime="text/html",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            label="📊 Download CSV",
            data=csv_bytes,
            file_name=f"{fname_base}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ── Radar Chart (single stock) ────────────────────────────────────────────────

def render_radar_chart(report: ReportData) -> None:
    """Radar chart of component scores for a single analyzed stock."""
    th = _plot_colors()
    cs = report.score.component_scores
    categories = [
        "RSI (15%)", "MACD (15%)", "Bollinger (10%)", "Trend (20%)",
        "Volume (10%)", "ATR (10%)", "Stochastic (10%)", "Pattern (10%)",
    ]
    vals = [cs.rsi, cs.macd, cs.bollinger, cs.trend,
            cs.volume, cs.atr_momentum, cs.stochastic, cs.pattern]
    vals_closed = vals + [vals[0]]
    cats_closed = categories + [categories[0]]

    color = score_color(report.score.total_score)

    fig = go.Figure()
    # Benchmark ring at 5.0 (neutral)
    fig.add_trace(go.Scatterpolar(
        r=[5.0] * (len(categories) + 1),
        theta=cats_closed,
        mode="lines",
        line=dict(color="#444466", width=1, dash="dot"),
        name="Neutral (5)",
        opacity=0.5,
    ))
    fig.add_trace(go.Scatterpolar(
        r=vals_closed,
        theta=cats_closed,
        fill="toself",
        fillcolor=f"{color}22",
        line=dict(color=color, width=2),
        name=report.ticker,
        opacity=0.9,
    ))

    fig.update_layout(
        polar=dict(
            bgcolor=th["panel_bg"],
            radialaxis=dict(
                range=[0, 10],
                tickvals=[2, 4, 6, 8, 10],
                tickfont=dict(size=9, color=th["text_muted"]),
                gridcolor=th["grid_color"],
            ),
            angularaxis=dict(
                gridcolor=th["grid_color"],
                tickfont=dict(size=10, color=th["font_color"]),
            ),
        ),
        paper_bgcolor=th["paper_bgcolor"],
        font=dict(color=th["font_color"], size=11),
        showlegend=True,
        legend=dict(font=dict(size=10, color=th["text_muted"]), bgcolor="rgba(0,0,0,0)"),
        height=380,
        margin=dict(t=50, b=30, l=60, r=60),
        title=dict(
            text=f"{report.ticker} — Component Score Radar",
            font=dict(size=13, color=th["text_muted"]),
        ),
    )
    st.plotly_chart(fig, width="stretch")


# ── Comparison Mode ───────────────────────────────────────────────────────────

def render_comparison(reports: list) -> None:
    """Side-by-side comparison of 2–3 analyzed stocks."""
    n = len(reports)
    if n == 0:
        return

    cols = st.columns(n, gap="medium")
    dir_color_map = {"LONG": "#00c851", "SHORT": "#ff4444", "NO TRADE": "#888888"}

    for col, r in zip(cols, reports):
        score = r.score.total_score
        color = score_color(score)
        dc    = dir_color_map.get(r.trade_setup.direction, "#888888")
        ts    = r.trade_setup

        with col:
            # ── Score card ────────────────────────────────────────────────────
            st.markdown(f"""
            <div style="background:#1a1a2e; padding:18px; border-radius:12px;
                        border:2px solid {color}; text-align:center; margin-bottom:10px;">
                <div style="font-size:22px; font-weight:700; color:#e0e0e0;">{r.ticker}</div>
                <div style="color:#888; font-size:12px; margin-bottom:8px;">{r.company_name[:28]}</div>
                <div style="font-size:15px; color:#e0e0e0;">{fmt_price(r.current_price)}</div>
                <div style="font-size:44px; font-weight:700; color:{color}; line-height:1.1;
                            margin:8px 0;">{score:.0f}</div>
                <div style="color:{color}; font-size:14px;">{r.score.signal_label}</div>
                <div style="margin-top:8px; color:{dc}; font-size:15px;
                            font-weight:700;">{ts.direction}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Trade levels ─────────────────────────────────────────────────
            if ts.direction != "NO TRADE":
                st.markdown(f"""
                <div style="background:#111122; padding:12px; border-radius:8px;
                            font-size:13px; margin-bottom:8px;">
                    <div style="margin-bottom:4px;">
                        Entry: <b style="color:#e0e0e0;">{fmt_price(ts.entry)}</b>
                    </div>
                    <div style="margin-bottom:4px; color:#ff4444;">
                        SL: <b>{fmt_price(ts.stop_loss)}</b>
                        <span style="color:#666; font-size:11px;"> ({ts.stop_pct:.1f}%)</span>
                    </div>
                    <div style="margin-bottom:4px; color:#00c851;">
                        TP: <b>{fmt_price(ts.take_profit)}</b>
                        <span style="color:#666; font-size:11px;"> (+{ts.target_pct:.1f}%)</span>
                    </div>
                    <div style="color:#ffbb33;">
                        Win Prob: <b>{r.score.win_probability * 100:.1f}%</b>
                        &nbsp;|&nbsp; EV: <b>${r.score.expected_value:+.2f}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── MTF badge ─────────────────────────────────────────────────────
            if r.mtf:
                mtf_c = {"BULLISH": "#00c851", "BEARISH": "#ff4444",
                          "MIXED": "#ffbb33"}.get(r.mtf.agreement_direction, "#888")
                st.markdown(f"""
                <div style="background:#111122; padding:8px; border-radius:6px;
                            text-align:center; font-size:12px; margin-bottom:6px;">
                    MTF: <span style="color:{mtf_c}; font-weight:700;">
                        {r.mtf.agreement_direction}</span>
                    &nbsp;({r.mtf.combined_score:.0f})
                </div>
                """, unsafe_allow_html=True)

            # ── Key indicators ────────────────────────────────────────────────
            ind = r.indicators
            rsi_c = "#00c851" if ind.rsi < 40 else ("#ff4444" if ind.rsi > 70 else "#ffbb33")
            st.markdown(f"""
            <div style="background:#111122; padding:10px; border-radius:8px;
                        font-size:12px; color:#aaa;">
                <div>RSI: <span style="color:{rsi_c};">{ind.rsi:.1f}</span>
                     &nbsp;|&nbsp; Stoch: {ind.stoch_k:.0f}</div>
                <div style="margin-top:4px;">
                    MACD hist: <span style="color:{'#00c851' if ind.macd_histogram >= 0 else '#ff4444'};">
                        {ind.macd_histogram:+.3f}</span>
                </div>
                <div style="margin-top:4px;">
                    Trend: EMA20 {'&gt;' if ind.ema20 > ind.ema50 else '&lt;'} EMA50
                    {'&#x2714;' if ind.golden_cross else ''}
                    {'&#x2718;' if ind.death_cross else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Radar chart — component score comparison ──────────────────────────────
    st.markdown("---")
    st.markdown("#### Component Score Comparison")
    st.caption("Scores are 0–10 per component. Higher = more bullish signal for that indicator.")

    th = _plot_colors()
    categories = [
        "RSI (15%)", "MACD (15%)", "Bollinger (10%)", "Trend (20%)",
        "Volume (10%)", "ATR (10%)", "Stochastic (10%)", "Pattern (10%)",
    ]
    palette = ["#4488ff", "#00c851", "#ffbb33"]

    fig = go.Figure()
    for i, r in enumerate(reports):
        cs = r.score.component_scores
        vals = [cs.rsi, cs.macd, cs.bollinger, cs.trend,
                cs.volume, cs.atr_momentum, cs.stochastic, cs.pattern]
        vals_closed = vals + [vals[0]]
        cats_closed = categories + [categories[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=cats_closed,
            fill="toself",
            name=r.ticker,
            line_color=palette[i % len(palette)],
            opacity=0.65,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], color=th["text_muted"],
                            gridcolor=th["panel_border"], linecolor=th["panel_border"]),
            angularaxis=dict(color=th["text_muted"]),
            bgcolor=th["plot_bgcolor"],
        ),
        paper_bgcolor=th["paper_bgcolor"],
        font=dict(color=th["font_color"]),
        showlegend=True,
        legend=dict(bgcolor=th["panel_bg"], bordercolor="#333"),
        height=400,
        margin=dict(t=20, b=20, l=40, r=40),
    )
    st.plotly_chart(fig, width="stretch")

    # ── Summary table ─────────────────────────────────────────────────────────
    st.markdown("#### Summary Table")
    import pandas as pd
    rows = []
    for r in reports:
        ts = r.trade_setup
        rows.append({
            "Ticker":    r.ticker,
            "Score":     f"{r.score.total_score:.0f}",
            "Signal":    r.score.signal_label,
            "Direction": ts.direction,
            "Win%":      f"{r.score.win_probability * 100:.1f}%",
            "EV":        f"${r.score.expected_value:+.2f}",
            "MTF":       r.mtf.agreement_direction if r.mtf else "—",
            "RSI":       f"{r.indicators.rsi:.1f}",
            "Entry":     fmt_price(ts.entry) if ts.direction != "NO TRADE" else "—",
            "SL":        fmt_price(ts.stop_loss) if ts.direction != "NO TRADE" else "—",
            "TP":        fmt_price(ts.take_profit) if ts.direction != "NO TRADE" else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── CAN SLIM Panel ────────────────────────────────────────────────────────────

def render_canslim_panel(report: ReportData) -> None:
    """Render the CAN SLIM methodology analysis panel."""
    cs = report.canslim
    if cs is None:
        st.warning("CAN SLIM analysis unavailable for this stock.")
        return

    overall = cs.overall_score
    rec = cs.recommendation
    passed = cs.criteria_passed

    # Recommendation color
    rec_colors = {
        "IDEAL":      "#00c851",
        "STRONG":     "#4488ff",
        "ACCEPTABLE": "#ffaa00",
        "AVOID":      "#ff4444",
    }
    rec_color = rec_colors.get(rec, "#aaaaaa")

    st.markdown("### CAN SLIM Analysis")

    # Header row: overall score + recommendation + pass count
    h1, h2, h3 = st.columns([1, 1, 2])
    with h1:
        score_clr = score_color(overall)
        st.markdown(
            f"<div style='background:#1a1a2e; border:1px solid #2d2d4e; border-radius:10px;"
            f"padding:14px 18px; text-align:center;'>"
            f"<div style='color:#aaaaaa; font-size:12px; margin-bottom:4px;'>CAN SLIM SCORE</div>"
            f"<div style='font-size:36px; font-weight:800; color:{score_clr};'>{overall:.0f}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with h2:
        st.markdown(
            f"<div style='background:#1a1a2e; border:1px solid #2d2d4e; border-radius:10px;"
            f"padding:14px 18px; text-align:center;'>"
            f"<div style='color:#aaaaaa; font-size:12px; margin-bottom:4px;'>RECOMMENDATION</div>"
            f"<div style='font-size:22px; font-weight:800; color:{rec_color};'>{rec}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with h3:
        st.markdown(
            f"<div style='background:#1a1a2e; border:1px solid #2d2d4e; border-radius:10px;"
            f"padding:14px 18px;'>"
            f"<div style='color:#aaaaaa; font-size:12px; margin-bottom:6px;'>CRITERIA PASSED</div>"
            f"<div style='font-size:15px; color:#e0e0e0;'>"
            f"<b style='color:{rec_color}; font-size:20px;'>{passed}</b> / 7 criteria ≥ 60"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)

    # Per-letter rows
    label_colors = {
        "Strong": "#00c851",
        "Pass":   "#4488ff",
        "Weak":   "#ffaa00",
        "Fail":   "#ff4444",
        "N/A":    "#666666",
    }

    for ls in cs.letters:
        s = ls.score
        clr = score_color(s)
        bar_filled = int(s / 10)          # 0-10 filled blocks
        bar_empty  = 10 - bar_filled
        bar_html = (
            f"<span style='color:{clr};'>{'█' * bar_filled}</span>"
            f"<span style='color:#333;'>{'░' * bar_empty}</span>"
        )
        lbl_clr = label_colors.get(ls.label, "#aaaaaa")

        st.markdown(
            f"<div style='display:flex; align-items:center; gap:14px; padding:7px 10px;"
            f"border-bottom:1px solid #1e1e2e;'>"
            f"<div style='min-width:28px; height:28px; border-radius:6px; background:{clr}22;"
            f"border:1px solid {clr}66; display:flex; align-items:center; justify-content:center;"
            f"font-weight:800; font-size:14px; color:{clr};'>{ls.letter}</div>"
            f"<div style='min-width:160px; color:#cccccc; font-size:13px;'>{ls.name}</div>"
            f"<div style='min-width:36px; font-weight:700; font-size:15px; color:{clr};'>{s:.0f}</div>"
            f"<div style='font-family:monospace; font-size:13px; letter-spacing:1px;'>{bar_html}</div>"
            f"<div style='min-width:60px;'>"
            f"<span style='font-size:11px; color:{lbl_clr}; background:{lbl_clr}22;"
            f"border-radius:4px; padding:2px 7px;'>{ls.label}</span>"
            f"</div>"
            f"<div style='color:#888888; font-size:12px; flex:1;'>{ls.detail}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div style='font-size:11px; color:#444; margin-top:8px;'>"
        "C=Current Earnings · A=Annual Growth · N=Near New High · "
        "S=Supply/Demand · L=Leader · I=Institutional · M=Market Direction"
        "</div>",
        unsafe_allow_html=True,
    )


def render_filter_notes(report: ReportData) -> None:
    """Show macro filter warnings (SPY, VIX, earnings) if any were triggered."""
    notes = getattr(report, "filter_notes", [])
    if not notes:
        return

    rows = "".join(
        f"<div style='padding:4px 0; border-bottom:1px solid #2d1800; font-size:13px;'>"
        f"⚠ {note}</div>"
        for note in notes
    )
    st.markdown(f"""
    <div style="background:#1e1000; border:1px solid #f0a500; border-radius:8px;
                padding:12px 16px; margin-top:12px;">
        <div style="font-size:12px; font-weight:700; color:#f0a500;
                    margin-bottom:6px; letter-spacing:0.5px;">MACRO FILTERS APPLIED</div>
        <div style="color:#e0c080;">{rows}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Trade Setup Checklist ─────────────────────────────────────────────────────

def render_setup_checklist(report: ReportData) -> None:
    """
    Swing trade quality checklist — evaluates 10 preconditions from existing
    indicator data and shows pass/fail for each with a summary grade.
    """
    ind = report.indicators
    score = report.score.total_score
    ts = report.trade_setup
    direction = ts.direction
    is_long = direction == "LONG"
    is_short = direction == "SHORT"
    regime = getattr(report.score, "regime", "NEUTRAL")
    mtf_dir = report.mtf.agreement_direction if report.mtf else "UNKNOWN"

    def _check(label: str, passed: bool, detail: str) -> dict:
        return {"label": label, "passed": passed, "detail": detail}

    checks = []

    # 1. Score threshold
    threshold = 65 if is_long else 35
    if is_long:
        score_ok = score >= threshold
        score_detail = f"Score {score:.0f} ≥ 65 required for LONG"
    elif is_short:
        score_ok = score <= threshold
        score_detail = f"Score {score:.0f} ≤ 35 required for SHORT"
    else:
        score_ok = False
        score_detail = "NO TRADE — no directional signal"
    checks.append(_check("Score threshold met", score_ok, score_detail))

    # 2. RSI zone
    rsi = ind.rsi
    if is_long:
        rsi_ok = 30 <= rsi <= 65
        rsi_detail = f"RSI {rsi:.1f} — ideal long zone: 30–65"
    elif is_short:
        rsi_ok = 35 <= rsi <= 70
        rsi_detail = f"RSI {rsi:.1f} — ideal short zone: 35–70"
    else:
        rsi_ok = 40 <= rsi <= 60
        rsi_detail = f"RSI {rsi:.1f}"
    checks.append(_check("RSI in ideal zone", rsi_ok, rsi_detail))

    # 3. MACD momentum
    hist = ind.macd_histogram
    hist_prev = ind.macd_histogram_prev
    if is_long:
        macd_ok = hist > 0 and hist >= hist_prev
        macd_detail = f"MACD hist {hist:.3f} — positive and rising for LONG"
    elif is_short:
        macd_ok = hist < 0 and hist <= hist_prev
        macd_detail = f"MACD hist {hist:.3f} — negative and falling for SHORT"
    else:
        macd_ok = False
        macd_detail = f"MACD hist {hist:.3f}"
    checks.append(_check("MACD momentum aligned", macd_ok, macd_detail))

    # 4. EMA alignment
    ema_aligned = ind.ema20 > ind.ema50 if is_long else ind.ema20 < ind.ema50
    ema_detail = (
        f"EMA20 {ind.ema20:.2f} {'>' if is_long else '<'} EMA50 {ind.ema50:.2f}"
    )
    checks.append(_check("EMA20/50 trend aligned", ema_aligned, ema_detail))

    # 5. Above/below SMA200
    if ind.sma200 is not None:
        price_proxy = ind.close if ind.close > 0 else ind.ema20
        sma200_ok = price_proxy > ind.sma200 if is_long else price_proxy < ind.sma200
        sma200_detail = f"Price {price_proxy:.2f} vs SMA200 {ind.sma200:.2f}"
    else:
        sma200_ok = True  # can't penalize if data unavailable
        sma200_detail = "SMA200 unavailable (< 200 bars)"
    checks.append(_check("Price on correct side of SMA200", sma200_ok, sma200_detail))

    # 6. Volume confirmation
    if ind.volume_sma20 > 0:
        vol_ratio = ind.current_volume / ind.volume_sma20
        vol_ok = vol_ratio >= 1.0
        vol_detail = f"Volume ratio {vol_ratio:.2f}× 20-day avg (need ≥ 1.0×)"
    else:
        vol_ok = True
        vol_detail = "Volume SMA unavailable"
    checks.append(_check("Volume at or above average", vol_ok, vol_detail))

    # 7. Stochastic not exhausted
    k = ind.stoch_k
    if is_long:
        stoch_ok = k < 80
        stoch_detail = f"Stoch %K {k:.1f} — below 80 (not overbought)"
    elif is_short:
        stoch_ok = k > 20
        stoch_detail = f"Stoch %K {k:.1f} — above 20 (not oversold)"
    else:
        stoch_ok = 20 < k < 80
        stoch_detail = f"Stoch %K {k:.1f}"
    checks.append(_check("Stochastic not exhausted", stoch_ok, stoch_detail))

    # 8. Market regime suitable
    good_regimes = {"TRENDING", "NEUTRAL", "CONSOLIDATING"}
    regime_ok = regime in good_regimes
    regime_detail = f"Regime: {regime} — {'OK' if regime_ok else 'caution in EXTENDED/VOLATILE'}"
    checks.append(_check("Market regime suitable", regime_ok, regime_detail))

    # 9. Multi-timeframe agreement
    if is_long:
        mtf_ok = mtf_dir == "BULLISH"
    elif is_short:
        mtf_ok = mtf_dir == "BEARISH"
    else:
        mtf_ok = False
    mtf_detail = f"Daily + Weekly alignment: {mtf_dir}"
    checks.append(_check("MTF timeframes agree", mtf_ok, mtf_detail))

    # 10. Positive expected value
    ev = report.score.expected_value
    ev_ok = ev > 0
    ev_detail = f"EV ${ev:+.2f} per $100 risked at 1:2 R:R"
    checks.append(_check("Positive expected value (EV > 0)", ev_ok, ev_detail))

    # ── Render ────────────────────────────────────────────────────────────────
    passed_count = sum(1 for c in checks if c["passed"])
    total_count = len(checks)
    grade_pct = passed_count / total_count

    if grade_pct >= 0.8:
        grade_color, grade_label = "#00c851", "STRONG SETUP"
    elif grade_pct >= 0.6:
        grade_color, grade_label = "#4488ff", "GOOD SETUP"
    elif grade_pct >= 0.4:
        grade_color, grade_label = "#f0a500", "WEAK SETUP"
    else:
        grade_color, grade_label = "#ff4444", "POOR SETUP"

    st.markdown("### Trade Setup Checklist")

    hc1, hc2 = st.columns([3, 1])
    with hc1:
        st.markdown(
            f"<div style='font-size:13px; color:#888; padding-top:10px;'>"
            f"{passed_count} of {total_count} conditions met</div>",
            unsafe_allow_html=True,
        )
    with hc2:
        st.markdown(
            f"<div style='text-align:right; font-size:15px; font-weight:700; color:{grade_color};"
            f"background:{grade_color}22; border:1px solid {grade_color}; border-radius:6px;"
            f"padding:6px 12px;'>{grade_label}</div>",
            unsafe_allow_html=True,
        )

    for c in checks:
        icon = "✅" if c["passed"] else "❌"
        row_bg = "#0a1a0a" if c["passed"] else "#1a0a0a"
        label_color = "#cccccc"
        detail_color = "#666666"
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:12px; padding:7px 12px;"
            f"border-bottom:1px solid #1e1e2e; background:{row_bg}; border-radius:4px; margin:2px 0;'>"
            f"<div style='font-size:16px; min-width:24px;'>{icon}</div>"
            f"<div style='min-width:240px; font-size:13px; font-weight:600; color:{label_color};'>"
            f"{c['label']}</div>"
            f"<div style='font-size:12px; color:{detail_color}; flex:1;'>{c['detail']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ── Relative Strength Panel ───────────────────────────────────────────────────

_SECTOR_ETFS = {
    "Technology":             "XLK",
    "Health Care":            "XLV",
    "Healthcare":             "XLV",
    "Financials":             "XLF",
    "Financial Services":     "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples":       "XLP",
    "Energy":                 "XLE",
    "Utilities":              "XLU",
    "Real Estate":            "XLRE",
    "Materials":              "XLB",
    "Industrials":            "XLI",
    "Communication Services": "XLC",
    "Basic Materials":        "XLB",
}


def render_relative_strength(report: ReportData) -> None:
    """
    Compare the stock's price returns vs. SPY and its sector ETF
    over 1-month, 3-month, and 6-month windows.
    """
    import yfinance as yf
    import numpy as np

    ticker = report.ticker
    sector = report.sector
    sector_etf = _SECTOR_ETFS.get(sector)

    symbols = [ticker, "SPY"]
    labels = [ticker, "S&P 500 (SPY)"]
    if sector_etf:
        symbols.append(sector_etf)
        labels.append(f"{sector} ({sector_etf})")

    periods = {"1M": 21, "3M": 63, "6M": 126}

    try:
        raw = yf.download(
            symbols, period="7mo", interval="1d", auto_adjust=True, progress=False,
        )["Close"]
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.dropna(how="all")
    except Exception as e:
        st.caption(f"Relative strength data unavailable: {e}")
        return

    rows = []
    for sym, lbl in zip(symbols, labels):
        if sym not in raw.columns:
            continue
        series = raw[sym].dropna()
        row = {"Symbol": lbl}
        for p_label, p_bars in periods.items():
            if len(series) >= p_bars + 1:
                ret = (series.iloc[-1] / series.iloc[-p_bars] - 1) * 100
                row[p_label] = round(float(ret), 2)
            else:
                row[p_label] = None
        rows.append(row)

    if not rows:
        return

    st.markdown("### Relative Strength vs. Market & Sector")

    th = _plot_colors()
    period_labels = list(periods.keys())

    # ── Bar chart ──────────────────────────────────────────────────────────────
    palette = ["#4488ff", "#888888", "#f0a500"]
    fig = go.Figure()
    for i, row in enumerate(rows):
        vals = [row.get(p) for p in period_labels]
        colors_bar = [
            "#00c851" if (v is not None and v >= 0) else "#ff4444"
            for v in vals
        ]
        fig.add_trace(go.Bar(
            name=row["Symbol"],
            x=period_labels,
            y=vals,
            marker_color=palette[i % len(palette)],
            text=[f"{v:+.1f}%" if v is not None else "N/A" for v in vals],
            textposition="outside",
            textfont=dict(size=10),
        ))

    fig.add_hline(y=0, line_color="#555555", line_width=1)
    fig.update_layout(
        barmode="group",
        paper_bgcolor=th["paper_bgcolor"],
        plot_bgcolor=th["plot_bgcolor"],
        font=dict(color=th["font_color"], size=11),
        xaxis=dict(gridcolor=th["grid_color"]),
        yaxis=dict(gridcolor=th["grid_color"], title="Return %"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        height=300,
        margin=dict(t=20, b=10, l=10, r=20),
    )
    st.plotly_chart(fig, width="stretch")

    # ── Summary table ─────────────────────────────────────────────────────────
    # Compute RS (stock return minus benchmark return) for each period
    stock_row = rows[0]
    spy_row = next((r for r in rows if "SPY" in r["Symbol"]), None)

    cols = st.columns(len(period_labels))
    for col, p in zip(cols, period_labels):
        stock_ret = stock_row.get(p)
        spy_ret = spy_row.get(p) if spy_row else None
        if stock_ret is not None and spy_ret is not None:
            rs = stock_ret - spy_ret
            rs_color = "#00c851" if rs >= 0 else "#ff4444"
            rs_sign = "+" if rs >= 0 else ""
            col.markdown(
                f"<div style='text-align:center; background:#1a1a2e; border-radius:8px;"
                f"padding:10px; border:1px solid #2d2d4e;'>"
                f"<div style='font-size:11px; color:#888; margin-bottom:4px;'>{p} vs SPY</div>"
                f"<div style='font-size:20px; font-weight:700; color:{rs_color};'>"
                f"{rs_sign}{rs:.1f}%</div>"
                f"<div style='font-size:10px; color:#555; margin-top:2px;'>"
                f"{ticker} {stock_ret:+.1f}% · SPY {spy_ret:+.1f}%</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ── AI Narrative ──────────────────────────────────────────────────────────────

def render_ai_narrative(report: ReportData) -> None:
    """
    On-demand AI narrative: calls Claude claude-haiku-4-5 to write a concise
    swing trade analysis summary from the current indicators and score.
    Shows a button — only fetches when clicked.
    """
    st.markdown("### AI Setup Narrative")

    key = f"narrative_{report.ticker}_{report.score.total_score:.0f}"

    if key not in st.session_state:
        if st.button("Generate AI Analysis", key=f"btn_narr_{report.ticker}", type="secondary"):
            with st.spinner("Generating narrative..."):
                narrative = _generate_narrative(report)
            st.session_state[key] = narrative
            st.rerun()
    else:
        narrative = st.session_state[key]
        th = _plot_colors()
        font_col = th["font_color"]
        panel_bg = th["panel_bg"]
        st.markdown(
            f"<div style='background:{panel_bg}; border:1px solid #2d2d4e;"
            f"border-radius:10px; padding:18px 22px; font-size:14px; line-height:1.7;"
            f"color:{font_col};'>{narrative}</div>",
            unsafe_allow_html=True,
        )
        if st.button("Regenerate", key=f"btn_regen_{report.ticker}", type="secondary"):
            del st.session_state[key]
            st.rerun()


def _generate_narrative(report: ReportData) -> str:
    """Call Anthropic API to generate a concise trade narrative."""
    try:
        import anthropic
    except ImportError:
        return (
            "Anthropic SDK not installed. Run <code>pip install anthropic</code> "
            "and add <code>anthropic</code> to requirements.txt to enable this feature."
        )

    ind = report.indicators
    ts = report.trade_setup
    cs = report.score.component_scores

    prompt = f"""You are a professional swing trader writing a concise analysis for {report.ticker} ({report.company_name}).

Key data:
- Trade score: {report.score.total_score:.0f}/100 ({report.score.signal_label})
- Direction: {ts.direction}
- Market regime: {getattr(report.score, 'regime', 'NEUTRAL')}
- Win probability: {report.score.win_probability*100:.1f}%
- Expected value: ${report.score.expected_value:+.2f} per $100 risked

Entry/Exit levels:
- Entry: ${ts.entry:.2f}, Stop Loss: ${ts.stop_loss:.2f} ({ts.stop_pct:.1f}%), Take Profit: ${ts.take_profit:.2f} ({ts.target_pct:.1f}%)

Component scores (0–10):
- RSI: {cs.rsi:.1f} (RSI={ind.rsi:.1f})
- MACD: {cs.macd:.1f} (hist={'rising' if ind.macd_histogram > ind.macd_histogram_prev else 'falling'})
- Bollinger: {cs.bollinger:.1f}
- Trend: {cs.trend:.1f} (EMA20={ind.ema20:.2f}, EMA50={ind.ema50:.2f})
- Volume: {cs.volume:.1f} (ratio={ind.current_volume/ind.volume_sma20:.2f}x avg)
- Stochastic: {cs.stochastic:.1f} (%K={ind.stoch_k:.1f})
- Pattern: {cs.pattern:.1f} (golden={'yes' if ind.golden_cross else 'no'}, above200={'yes' if ind.above_200_sma else 'no'})

MTF: {report.mtf.agreement_direction if report.mtf else 'N/A'} (daily {report.mtf.daily_score:.0f}, weekly {report.mtf.weekly_score:.0f})

Write a 3-paragraph analysis (120–180 words total):
1. What the overall setup looks like and why (1-2 sentences)
2. The 2-3 strongest signals driving the score — both bullish and bearish factors
3. Key risk factors and what to watch

Be direct, professional, and specific. No disclaimers. Plain text only."""

    try:
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except anthropic.AuthenticationError:
        return "API key not configured. Set the <code>ANTHROPIC_API_KEY</code> environment variable or in Streamlit secrets."
    except Exception as e:
        return f"Narrative generation failed: {e}"


# ── Valuation vs. Own History ─────────────────────────────────────────────────

def render_valuation_history(report: ReportData) -> None:
    """
    P/E ratio vs the stock's own 2-year history.
    Shows trailing P/E percentile + forward P/E + P/B + P/S from yfinance info,
    then a bar chart of quarterly historical P/E built from quarterly EPS data.
    """
    import yfinance as yf
    import numpy as np

    ticker = report.ticker
    current_price = report.current_price
    th = _plot_colors()

    st.markdown("### Valuation vs. Own History")

    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        pe_trailing = info.get("trailingPE")
        pe_forward  = info.get("forwardPE")
        pb          = info.get("priceToBook")
        ps          = info.get("priceToSalesTrailingTwelveMonths")

        # ── Historical quarterly P/E ──────────────────────────────────────────
        historical_pe = []
        try:
            qe = t.quarterly_earnings
            if qe is not None and len(qe) >= 4:
                qe = qe.sort_index()
                eps_col = qe.columns[0]   # first column is EPS / Earnings

                # Rolling TTM: sum of 4 consecutive quarters
                for i in range(3, len(qe)):
                    ttm_eps = float(qe[eps_col].iloc[i - 3 : i + 1].sum())
                    q_date  = pd.Timestamp(qe.index[i]).tz_localize(None)
                    if ttm_eps <= 0:
                        continue

                    # Find nearest price in report.df for that date
                    df_prices = report.df["Close"]
                    nearest_i = df_prices.index.get_indexer([q_date], method="nearest")[0]
                    price = float(df_prices.iloc[nearest_i])
                    pe    = round(price / ttm_eps, 1)
                    historical_pe.append({"date": q_date, "pe": pe})
        except Exception:
            pass

        # If no quarterly data, fall back to trailing P/E as a single point
        if not historical_pe and pe_trailing and pe_trailing > 0:
            historical_pe = [{"date": pd.Timestamp.now(), "pe": round(float(pe_trailing), 1)}]

        # Compute percentile of current P/E vs history
        current_pe = pe_trailing if (pe_trailing and pe_trailing > 0) else None
        if current_pe is None and historical_pe:
            current_pe = historical_pe[-1]["pe"]

        pe_vals = [x["pe"] for x in historical_pe if 0 < x["pe"] < 500]
        pe_pct  = None
        if current_pe and pe_vals:
            pe_pct = round(sum(1 for v in pe_vals if v < current_pe) / len(pe_vals) * 100, 0)

        # ── KPI badges ────────────────────────────────────────────────────────
        kpis = []
        if current_pe:
            if pe_pct is not None:
                if pe_pct >= 80:
                    pe_color, pe_label = "#ff4444", "EXPENSIVE"
                elif pe_pct >= 50:
                    pe_color, pe_label = "#f0a500", "FAIR"
                else:
                    pe_color, pe_label = "#00c851", "CHEAP"
            else:
                pe_color, pe_label = "#888888", ""
            pct_txt = f" · {pe_pct:.0f}th pct" if pe_pct is not None else ""
            kpis.append(("P/E (TTM)", f"{current_pe:.1f}{pct_txt}", pe_color, pe_label))
        if pe_forward and 0 < pe_forward < 500:
            kpis.append(("P/E (Fwd)", f"{pe_forward:.1f}", "#4488ff", ""))
        if pb and pb > 0:
            pb_color = "#ff4444" if pb > 5 else "#f0a500" if pb > 2 else "#00c851"
            kpis.append(("P/B", f"{pb:.2f}", pb_color, ""))
        if ps and ps > 0:
            ps_color = "#ff4444" if ps > 10 else "#f0a500" if ps > 3 else "#00c851"
            kpis.append(("P/S", f"{ps:.2f}", ps_color, ""))

        if not kpis:
            st.caption("Valuation data unavailable for this ticker.")
            return

        kpi_cols = st.columns(len(kpis))
        for col, (metric, val, color, label) in zip(kpi_cols, kpis):
            label_html = (
                f"<div style='font-size:10px; color:{color}; margin-top:2px;'>{label}</div>"
                if label else ""
            )
            col.markdown(
                f"<div style='background:{th['panel_bg']}; border:1px solid {color}66;"
                f"border-radius:8px; padding:10px 12px; text-align:center;'>"
                f"<div style='font-size:11px; color:#888; margin-bottom:4px;'>{metric}</div>"
                f"<div style='font-size:20px; font-weight:700; color:{color};'>{val}</div>"
                f"{label_html}"
                f"</div>",
                unsafe_allow_html=True,
            )

        # ── Historical P/E bar chart ──────────────────────────────────────────
        if len(historical_pe) >= 2:
            pe_dates  = [x["date"].strftime("%Y-%m") for x in historical_pe]
            pe_values = [x["pe"] for x in historical_pe]
            mean_pe   = float(np.mean([v for v in pe_values if 0 < v < 500]))

            bar_colors = []
            for v in pe_values:
                if v > mean_pe * 1.2:
                    bar_colors.append("#ff4444")
                elif v < mean_pe * 0.8:
                    bar_colors.append("#00c851")
                else:
                    bar_colors.append("#4488ff")

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=pe_dates,
                y=pe_values,
                marker_color=bar_colors,
                name="Quarterly P/E",
                text=[f"{v:.1f}" for v in pe_values],
                textposition="outside",
                textfont=dict(size=9),
            ))
            # Mean line
            fig.add_hline(
                y=mean_pe,
                line_color="#ffbb33",
                line_dash="dot",
                annotation_text=f"Avg {mean_pe:.1f}",
                annotation_font_color="#ffbb33",
            )
            fig.update_layout(
                paper_bgcolor=th["paper_bgcolor"],
                plot_bgcolor=th["plot_bgcolor"],
                font=dict(color=th["font_color"], size=11),
                xaxis=dict(gridcolor=th["grid_color"], tickangle=-30),
                yaxis=dict(gridcolor=th["grid_color"], title="P/E"),
                height=280,
                margin=dict(t=20, b=60, l=20, r=20),
                showlegend=False,
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(
                f"Green = below average · Red = above average · "
                f"Dotted line = {len(pe_values)}-quarter mean P/E {mean_pe:.1f}x"
            )

    except Exception as e:
        st.caption(f"Valuation history unavailable: {e}")


# ── Earnings Estimate Revision Tracker ────────────────────────────────────────

def render_estimate_revisions(report: ReportData) -> None:
    """
    Analyst EPS estimate revisions: up/down counts over 7d and 30d,
    plus the current consensus EPS estimate, growth rate, and analyst count.
    """
    import yfinance as yf

    ticker = report.ticker
    th = _plot_colors()

    st.markdown("### Analyst Estimate Revisions")

    try:
        t = yf.Ticker(ticker)

        # ── EPS Estimate table ────────────────────────────────────────────────
        ee = None
        try:
            ee = t.earnings_estimate
        except Exception:
            pass

        # ── EPS Revisions ─────────────────────────────────────────────────────
        rev = None
        try:
            rev = t.eps_revisions
        except Exception:
            pass

        # ── Analyst price targets ─────────────────────────────────────────────
        targets = {}
        try:
            apt = t.analyst_price_targets
            if apt is not None and isinstance(apt, dict):
                targets = apt
        except Exception:
            pass

        any_data = (ee is not None) or (rev is not None) or targets

        if not any_data:
            st.caption("Analyst estimate data unavailable for this ticker.")
            return

        # ── Price target banner ───────────────────────────────────────────────
        if targets:
            current = report.current_price
            mean_t  = targets.get("mean")
            high_t  = targets.get("high")
            low_t   = targets.get("low")
            n_anal  = targets.get("numberOfAnalysts")

            if mean_t and mean_t > 0:
                upside = (mean_t - current) / current * 100
                up_color = "#00c851" if upside >= 0 else "#ff4444"
                tc1, tc2, tc3, tc4 = st.columns(4)
                tc1.markdown(
                    f"<div style='background:{th['panel_bg']}; border-radius:8px; padding:10px;"
                    f"text-align:center; border:1px solid #2d2d4e;'>"
                    f"<div style='font-size:11px; color:#888;'>CONSENSUS TARGET</div>"
                    f"<div style='font-size:20px; font-weight:700; color:{up_color};'>${mean_t:.2f}</div>"
                    f"<div style='font-size:11px; color:{up_color};'>{upside:+.1f}% upside</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                tc2.markdown(
                    f"<div style='background:{th['panel_bg']}; border-radius:8px; padding:10px;"
                    f"text-align:center; border:1px solid #2d2d4e;'>"
                    f"<div style='font-size:11px; color:#888;'>HIGH TARGET</div>"
                    f"<div style='font-size:18px; font-weight:700; color:#e0e0e0;'>${high_t:.2f}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                ) if high_t else tc2.empty()
                tc3.markdown(
                    f"<div style='background:{th['panel_bg']}; border-radius:8px; padding:10px;"
                    f"text-align:center; border:1px solid #2d2d4e;'>"
                    f"<div style='font-size:11px; color:#888;'>LOW TARGET</div>"
                    f"<div style='font-size:18px; font-weight:700; color:#e0e0e0;'>${low_t:.2f}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                ) if low_t else tc3.empty()
                if n_anal:
                    tc4.markdown(
                        f"<div style='background:{th['panel_bg']}; border-radius:8px; padding:10px;"
                        f"text-align:center; border:1px solid #2d2d4e;'>"
                        f"<div style='font-size:11px; color:#888;'>ANALYSTS</div>"
                        f"<div style='font-size:20px; font-weight:700; color:#e0e0e0;'>{n_anal}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

        # ── EPS Revisions breakdown ───────────────────────────────────────────
        if rev is not None and not rev.empty:
            st.markdown("#### EPS Revisions")

            period_labels = {
                "0q": "Current Quarter",
                "1q": "Next Quarter",
                "0y": "Current Year",
                "1y": "Next Year",
            }

            rev_rows = []
            for period_key in ["0q", "1q", "0y", "1y"]:
                if period_key not in rev.index:
                    continue
                row = rev.loc[period_key]
                up7   = int(row.get("upLast7days",  0) or 0)
                dn7   = int(row.get("downLast7days", 0) or 0)
                up30  = int(row.get("upLast30days",  0) or 0)
                dn30  = int(row.get("downLast30days", 0) or 0)
                net7  = up7 - dn7
                net30 = up30 - dn30
                rev_rows.append({
                    "Period":   period_labels.get(period_key, period_key),
                    "↑ 7d":    up7,
                    "↓ 7d":    dn7,
                    "Net 7d":  net7,
                    "↑ 30d":   up30,
                    "↓ 30d":   dn30,
                    "Net 30d": net30,
                })

            if rev_rows:
                for rw in rev_rows:
                    net7  = rw["Net 7d"]
                    net30 = rw["Net 30d"]
                    n7c   = "#00c851" if net7 > 0 else "#ff4444" if net7 < 0 else "#888"
                    n30c  = "#00c851" if net30 > 0 else "#ff4444" if net30 < 0 else "#888"
                    n7s   = f"+{net7}" if net7 > 0 else str(net7)
                    n30s  = f"+{net30}" if net30 > 0 else str(net30)

                    rc1, rc2, rc3, rc4, rc5, rc6, rc7 = st.columns([2, 0.8, 0.8, 1, 0.8, 0.8, 1])
                    rc1.markdown(
                        f"<span style='font-size:13px; color:#ccc;'>{rw['Period']}</span>",
                        unsafe_allow_html=True,
                    )
                    rc2.markdown(
                        f"<span style='color:#00c851; font-size:13px;'>↑{rw['↑ 7d']}</span>",
                        unsafe_allow_html=True,
                    )
                    rc3.markdown(
                        f"<span style='color:#ff4444; font-size:13px;'>↓{rw['↓ 7d']}</span>",
                        unsafe_allow_html=True,
                    )
                    rc4.markdown(
                        f"<span style='color:{n7c}; font-weight:700; font-size:13px;'>Net 7d: {n7s}</span>",
                        unsafe_allow_html=True,
                    )
                    rc5.markdown(
                        f"<span style='color:#00c851; font-size:13px;'>↑{rw['↑ 30d']}</span>",
                        unsafe_allow_html=True,
                    )
                    rc6.markdown(
                        f"<span style='color:#ff4444; font-size:13px;'>↓{rw['↓ 30d']}</span>",
                        unsafe_allow_html=True,
                    )
                    rc7.markdown(
                        f"<span style='color:{n30c}; font-weight:700; font-size:13px;'>Net 30d: {n30s}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        "<div style='border-bottom:1px solid #1e1e2e; margin:3px 0;'></div>",
                        unsafe_allow_html=True,
                    )

        # ── EPS estimate table ────────────────────────────────────────────────
        if ee is not None and not ee.empty:
            st.markdown("#### EPS Estimates")

            period_labels_ee = {
                "0q": "Current Quarter",
                "1q": "Next Quarter",
                "0y": "Current Year",
                "1y": "Next Year",
            }

            for period_key in ["0q", "1q", "0y", "1y"]:
                if period_key not in ee.index:
                    continue
                row = ee.loc[period_key]

                avg_eps    = row.get("avg")
                low_eps    = row.get("low")
                high_eps   = row.get("high")
                n_analysts = row.get("numberOfAnalysts")
                yago_eps   = row.get("yearAgoEps")
                growth     = row.get("growth")

                if avg_eps is None:
                    continue

                label = period_labels_ee.get(period_key, period_key)
                gr_color = "#00c851" if (growth and growth > 0) else "#ff4444"
                gr_txt   = f"{growth*100:+.1f}% YoY" if growth else ""

                ec1, ec2, ec3, ec4, ec5 = st.columns([2, 1.2, 1.2, 1.2, 1.2])
                ec1.markdown(
                    f"<span style='font-size:13px; color:#ccc; font-weight:600;'>{label}</span>",
                    unsafe_allow_html=True,
                )
                ec2.markdown(
                    f"<span style='font-size:12px; color:#888;'>Avg</span> "
                    f"<span style='font-weight:700; color:#e0e0e0;'>${avg_eps:.2f}</span>",
                    unsafe_allow_html=True,
                )
                ec3.markdown(
                    f"<span style='font-size:12px; color:#888;'>Range</span> "
                    f"<span style='color:#e0e0e0; font-size:12px;'>${low_eps:.2f}–${high_eps:.2f}</span>"
                    if (low_eps is not None and high_eps is not None) else "",
                    unsafe_allow_html=True,
                )
                ec4.markdown(
                    f"<span style='color:{gr_color}; font-size:12px; font-weight:700;'>{gr_txt}</span>",
                    unsafe_allow_html=True,
                )
                ec5.markdown(
                    f"<span style='font-size:12px; color:#555;'>{int(n_analysts)} analysts</span>"
                    if n_analysts else "",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<div style='border-bottom:1px solid #1e1e2e; margin:3px 0;'></div>",
                    unsafe_allow_html=True,
                )

    except Exception as e:
        st.caption(f"Analyst estimate data unavailable: {e}")


# ── Weinstein Stage Badge ─────────────────────────────────────────────────────

def _compute_weinstein_stage(df: pd.DataFrame) -> dict:
    """
    Classify the stock into one of Stan Weinstein's 4 stages using daily OHLCV.

    SMA150 (≈ 30-week SMA) is the key moving average.
    Stage 2 (ADVANCING) — best for swing longs.
    Stage 4 (DECLINING) — best for swing shorts / avoidance.
    """
    import numpy as np

    if len(df) < 160:
        return {"stage": 0, "label": "N/A", "description": "Not enough data (< 160 bars)"}

    close  = df["Close"].values
    volume = df["Volume"].values

    # SMA150 (Weinstein's 30-week benchmark)
    sma150 = pd.Series(close).rolling(150).mean().values

    current_close   = close[-1]
    sma150_now      = sma150[-1]
    sma150_4w_ago   = sma150[-21]   # ~4 trading weeks back
    sma150_8w_ago   = sma150[-41]   # ~8 weeks back (check longer slope)

    if any(v != v for v in [sma150_now, sma150_4w_ago, sma150_8w_ago]):  # NaN check
        return {"stage": 0, "label": "N/A", "description": "Insufficient history for SMA150"}

    price_above_sma  = current_close > sma150_now
    sma_rising_4w    = sma150_now > sma150_4w_ago   # short slope
    sma_rising_8w    = sma150_now > sma150_8w_ago   # longer slope

    # OBV trend: compare current OBV to 20-bar-ago OBV
    obv = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]
    obv_rising = obv[-1] > obv[-21]

    # Stage classification
    if price_above_sma and sma_rising_4w:
        if obv_rising:
            stage, label = 2, "STAGE 2 — ADVANCING"
            description = "Price above rising SMA150 with volume confirmation. Weinstein's ideal BUY zone."
            color = "#00c851"
        else:
            stage, label = 2, "STAGE 2 — ADVANCING"
            description = "Price above rising SMA150. OBV lagging — watch for volume confirmation."
            color = "#4488ff"

    elif price_above_sma and not sma_rising_4w:
        stage, label = 3, "STAGE 3 — TOPPING"
        description = "Price near highs but SMA150 flattening. Distribution risk — reduce exposure."
        color = "#f0a500"

    elif not price_above_sma and sma_rising_8w:
        # SMA was rising but price fell below it — early Stage 3/4 transition or Stage 1 breakout attempt
        stage, label = 1, "STAGE 1 — BASING"
        description = "Price below SMA150 but SMA still rising. Possible base forming — wait for breakout."
        color = "#ffbb33"

    else:
        # Price below SMA, SMA declining
        stage, label = 4, "STAGE 4 — DECLINING"
        description = "Price below falling SMA150. Weinstein's AVOID zone for longs. Suitable for shorts."
        color = "#ff4444"

    return {
        "stage": stage,
        "label": label,
        "description": description,
        "color": color,
        "price_above_sma": price_above_sma,
        "sma_rising": sma_rising_4w,
        "obv_rising": obv_rising,
        "sma150": round(float(sma150_now), 2),
    }


def render_weinstein_stage(report: ReportData) -> None:
    """Weinstein Stage badge with explanation and key levels."""
    w = _compute_weinstein_stage(report.df)

    if w["stage"] == 0:
        return

    stage   = w["stage"]
    label   = w["label"]
    desc    = w["description"]
    color   = w["color"]
    sma150  = w["sma150"]

    stage_icons = {1: "🔄", 2: "📈", 3: "⚠", 4: "📉"}
    icon = stage_icons.get(stage, "")

    c1, c2 = st.columns([1, 3], gap="medium")
    with c1:
        st.markdown(
            f"<div style='background:#1a1a2e; border:2px solid {color}; border-radius:10px;"
            f"padding:16px 12px; text-align:center;'>"
            f"<div style='font-size:28px; line-height:1;'>{icon}</div>"
            f"<div style='font-size:11px; color:#888; margin-top:6px;'>WEINSTEIN</div>"
            f"<div style='font-size:13px; font-weight:800; color:{color}; margin-top:4px;'>"
            f"STAGE {stage}</div>"
            f"<div style='font-size:10px; color:#555; margin-top:4px;'>SMA150: ${sma150:.2f}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div style='background:#1a1a2e; border:1px solid {color}44; border-radius:10px;"
            f"padding:16px 18px;'>"
            f"<div style='font-size:15px; font-weight:700; color:{color}; margin-bottom:8px;'>"
            f"{label}</div>"
            f"<div style='font-size:13px; color:#cccccc; line-height:1.5;'>{desc}</div>"
            f"<div style='margin-top:10px; font-size:12px; color:#555;'>"
            f"Price {'above' if w['price_above_sma'] else 'below'} SMA150 &nbsp;·&nbsp; "
            f"SMA150 {'rising' if w['sma_rising'] else 'declining/flat'} &nbsp;·&nbsp; "
            f"OBV {'rising' if w['obv_rising'] else 'falling'}"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ── Institutional Footprint (Volume Anomaly) ──────────────────────────────────

def render_institutional_footprint(report: ReportData) -> None:
    """
    IBD-style Accumulation/Distribution rating.
    Counts high-volume up days vs high-volume down days over the last 25 sessions
    to detect institutional buying (accumulation) vs selling (distribution).
    """
    import numpy as np

    df = report.df
    if len(df) < 60:
        return

    close  = df["Close"].values
    open_  = df["Open"].values
    volume = df["Volume"].values

    # 50-day volume average for threshold
    vol_sma50 = pd.Series(volume).rolling(50).mean().values

    # Scan last 25 sessions (≈ 5 trading weeks)
    window = 25
    accum_days = []
    dist_days  = []

    for i in range(len(df) - window, len(df)):
        if vol_sma50[i] <= 0 or vol_sma50[i] != vol_sma50[i]:  # zero or NaN
            continue
        vol_ratio = volume[i] / vol_sma50[i]
        if vol_ratio < 1.4:   # only flag meaningful volume spikes (≥ 1.4× avg)
            continue
        price_up = close[i] >= open_[i]
        if price_up:
            accum_days.append({
                "date": str(df.index[i].date()),
                "ratio": round(vol_ratio, 1),
                "move":  round((close[i] - open_[i]) / open_[i] * 100, 2),
            })
        else:
            dist_days.append({
                "date": str(df.index[i].date()),
                "ratio": round(vol_ratio, 1),
                "move":  round((close[i] - open_[i]) / open_[i] * 100, 2),
            })

    n_acc  = len(accum_days)
    n_dist = len(dist_days)
    net    = n_acc - n_dist

    # IBD-style A–E rating
    if net >= 3:
        rating, rating_color, rating_label = "A", "#00c851", "ACCUMULATION"
    elif net >= 1:
        rating, rating_color, rating_label = "B", "#4488ff", "MILD ACCUMULATION"
    elif net == 0:
        rating, rating_color, rating_label = "C", "#888888", "NEUTRAL"
    elif net >= -2:
        rating, rating_color, rating_label = "D", "#f0a500", "MILD DISTRIBUTION"
    else:
        rating, rating_color, rating_label = "E", "#ff4444", "DISTRIBUTION"

    st.markdown("### Institutional Footprint")

    rc1, rc2, rc3, rc4 = st.columns([1, 1, 1, 2])
    rc1.markdown(
        f"<div style='background:#1a1a2e; border:2px solid {rating_color}; border-radius:10px;"
        f"padding:14px; text-align:center;'>"
        f"<div style='font-size:11px; color:#888; margin-bottom:4px;'>IBD RATING</div>"
        f"<div style='font-size:40px; font-weight:900; color:{rating_color};'>{rating}</div>"
        f"<div style='font-size:11px; color:{rating_color}; margin-top:4px;'>{rating_label}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    rc2.markdown(
        f"<div style='background:#0a1a0a; border:1px solid #00c85144; border-radius:10px;"
        f"padding:14px; text-align:center;'>"
        f"<div style='font-size:11px; color:#888; margin-bottom:4px;'>ACCUM DAYS</div>"
        f"<div style='font-size:28px; font-weight:700; color:#00c851;'>{n_acc}</div>"
        f"<div style='font-size:10px; color:#555;'>last 25 sessions</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    rc3.markdown(
        f"<div style='background:#1a0a0a; border:1px solid #ff444444; border-radius:10px;"
        f"padding:14px; text-align:center;'>"
        f"<div style='font-size:11px; color:#888; margin-bottom:4px;'>DIST DAYS</div>"
        f"<div style='font-size:28px; font-weight:700; color:#ff4444;'>{n_dist}</div>"
        f"<div style='font-size:10px; color:#555;'>last 25 sessions</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    with rc4:
        st.markdown(
            f"<div style='background:#1a1a2e; border:1px solid #2d2d4e; border-radius:10px;"
            f"padding:14px;'>"
            f"<div style='font-size:11px; color:#888; margin-bottom:6px;'>HOW TO READ</div>"
            f"<div style='font-size:12px; color:#aaa; line-height:1.6;'>"
            f"High-volume up/down days (≥1.4× avg) signal institutional activity. "
            f"Net {net:+d} over 25 sessions → <b style='color:{rating_color};'>{rating_label}</b>."
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Detail rows (most recent anomaly days)
    if accum_days or dist_days:
        with st.expander("View volume anomaly days", expanded=False):
            all_days = (
                [{"type": "ACCUM", "color": "#00c851", **d} for d in accum_days] +
                [{"type": "DIST",  "color": "#ff4444", **d} for d in dist_days]
            )
            all_days.sort(key=lambda x: x["date"], reverse=True)
            for d in all_days[:10]:
                st.markdown(
                    f"<span style='color:{d['color']}; font-weight:700; min-width:50px;"
                    f"display:inline-block;'>{d['type']}</span>"
                    f"<span style='color:#888; font-size:12px; margin-left:8px;'>{d['date']}</span>"
                    f"<span style='color:#e0e0e0; font-size:12px; margin-left:12px;'>"
                    f"{d['ratio']}× avg vol</span>"
                    f"<span style='color:{d['color']}; font-size:12px; margin-left:12px;'>"
                    f"{d['move']:+.2f}%</span>",
                    unsafe_allow_html=True,
                )


# ── Sector Rotation Timeline ──────────────────────────────────────────────────

_ROTATION_ETFS = {
    "Technology":    "XLK",
    "Healthcare":    "XLV",
    "Financials":    "XLF",
    "Cons. Disc.":   "XLY",
    "Cons. Staples": "XLP",
    "Energy":        "XLE",
    "Utilities":     "XLU",
    "Real Estate":   "XLRE",
    "Materials":     "XLB",
    "Industrials":   "XLI",
    "Comm. Svcs":    "XLC",
}


def render_sector_rotation() -> None:
    """
    4-week rolling return heatmap for all 11 GICS sector ETFs.
    Shows which sectors are gaining/losing momentum week by week.
    """
    import yfinance as yf
    import numpy as np

    th = _plot_colors()

    symbols = list(_ROTATION_ETFS.values())
    try:
        raw = yf.download(
            symbols, period="2mo", interval="1d",
            auto_adjust=True, progress=False, group_by="ticker",
        )
        # Flatten to Close-only DataFrame: index=date, columns=tickers
        if isinstance(raw.columns, pd.MultiIndex):
            close_df = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.xs("Close", axis=1, level=0)
        else:
            close_df = raw[["Close"]].rename(columns={"Close": symbols[0]}) if len(symbols) == 1 else raw
        close_df = close_df.dropna(how="all")
    except Exception as e:
        st.caption(f"Sector rotation data unavailable: {e}")
        return

    week_windows = {"1W": 5, "2W": 10, "3W": 15, "4W": 20}
    sectors = list(_ROTATION_ETFS.keys())
    etfs    = list(_ROTATION_ETFS.values())

    z_vals  = []
    z_text  = []

    for etf in etfs:
        col = etf if etf in close_df.columns else None
        row_z, row_t = [], []
        for w_label, bars in week_windows.items():
            if col and len(close_df[col].dropna()) > bars:
                series = close_df[col].dropna()
                ret = (series.iloc[-1] / series.iloc[-bars - 1] - 1) * 100
                row_z.append(round(float(ret), 2))
                row_t.append(f"{ret:+.2f}%")
            else:
                row_z.append(None)
                row_t.append("N/A")
        z_vals.append(row_z)
        z_text.append(row_t)

    # Build heatmap
    fig = go.Figure(go.Heatmap(
        z=z_vals,
        x=list(week_windows.keys()),
        y=sectors,
        text=z_text,
        texttemplate="%{text}",
        textfont=dict(size=12, color="#ffffff"),
        colorscale=[
            [0.0,  "#7b1515"],
            [0.35, "#c0392b"],
            [0.48, "#2d2d4e"],
            [0.52, "#2d2d4e"],
            [0.65, "#1a6e3a"],
            [1.0,  "#0a3d1f"],
        ],
        zmid=0,
        showscale=True,
        colorbar=dict(
            title="%",
            thickness=12,
            tickfont=dict(size=10, color=th["text_muted"]),
        ),
    ))
    fig.update_layout(
        paper_bgcolor=th["paper_bgcolor"],
        plot_bgcolor=th["plot_bgcolor"],
        font=dict(color=th["font_color"], size=12),
        xaxis=dict(side="top", tickfont=dict(size=12)),
        yaxis=dict(tickfont=dict(size=12)),
        height=420,
        margin=dict(t=30, b=10, l=120, r=60),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("Returns computed from daily closes. Green = outperforming, Red = underperforming.")


# ── Screener Results Renderer ─────────────────────────────────────────────────

def render_screener_results(results: list[dict]) -> None:
    """Render screener output as a styled, sortable table."""
    if not results:
        st.info("No stocks matched the selected filters.")
        return

    from utils.formatting import score_color, fmt_price

    st.caption(f"{len(results)} stocks matched")

    # Header
    hc = st.columns([1, 1.2, 1, 1, 0.8, 0.8, 0.8, 0.8, 0.8])
    for col, label in zip(hc, ["Ticker", "Signal", "Score", "Direction", "RSI", "Vol ×Avg", "Stage 2", "Trend", "1D %"]):
        col.markdown(f"<span style='font-size:11px; color:#555; font-weight:700;'>{label}</span>", unsafe_allow_html=True)

    st.markdown("<div style='border-bottom:1px solid #2d2d4e; margin:4px 0 8px 0;'></div>", unsafe_allow_html=True)

    for row in results:
        s = row["score"]
        color = score_color(s)
        chg = row["chg_1d"]
        chg_col = "#00c851" if chg >= 0 else "#ff4444"
        dir_col = "#00c851" if row["direction"] == "LONG" else "#ff4444" if row["direction"] == "SHORT" else "#888"
        stage2_txt = "✅" if row["stage2"] else "—"
        trend_col = "#00c851" if row["ema_trend"] == "UP" else "#ff4444" if row["ema_trend"] == "DOWN" else "#888"

        rc = st.columns([1, 1.2, 1, 1, 0.8, 0.8, 0.8, 0.8, 0.8])
        rc[0].markdown(f"<b style='color:#e0e0e0;'>{row['ticker']}</b>", unsafe_allow_html=True)
        rc[1].markdown(f"<span style='color:{color}; font-size:12px;'>{row['signal']}</span>", unsafe_allow_html=True)
        rc[2].markdown(f"<span style='font-size:18px; font-weight:700; color:{color};'>{s:.0f}</span>", unsafe_allow_html=True)
        rc[3].markdown(f"<span style='color:{dir_col}; font-weight:700; font-size:13px;'>{row['direction']}</span>", unsafe_allow_html=True)
        rc[4].markdown(f"<span style='color:#e0e0e0; font-size:13px;'>{row['rsi']:.1f}</span>", unsafe_allow_html=True)
        rc[5].markdown(f"<span style='color:#e0e0e0; font-size:13px;'>{row['vol_ratio']:.1f}×</span>", unsafe_allow_html=True)
        rc[6].markdown(f"<span style='font-size:14px;'>{stage2_txt}</span>", unsafe_allow_html=True)
        rc[7].markdown(f"<span style='color:{trend_col}; font-size:12px;'>{row['ema_trend']}</span>", unsafe_allow_html=True)
        rc[8].markdown(f"<span style='color:{chg_col}; font-size:13px;'>{chg:+.1f}%</span>", unsafe_allow_html=True)
        st.markdown("<div style='border-bottom:1px solid #1a1a2e; margin:2px 0;'></div>", unsafe_allow_html=True)


# ── Per-ticker Notes ──────────────────────────────────────────────────────────

def render_ticker_notes(report: ReportData) -> None:
    """Persistent trade notes per ticker — saved to notes.json."""
    from data.notes import get_note, save_note, delete_note

    ticker = report.ticker
    existing = get_note(ticker)

    st.markdown("### Trade Notes")

    # Text area pre-filled with saved note
    note_text = st.text_area(
        label=f"Notes for {ticker}",
        value=existing,
        height=120,
        placeholder="Trade thesis, key levels, observations, reminders...",
        label_visibility="collapsed",
        key=f"note_ta_{ticker}",
    )

    nc1, nc2 = st.columns([1, 5])
    with nc1:
        if st.button("💾 Save Note", key=f"note_save_{ticker}", type="primary"):
            save_note(ticker, note_text)
            st.success("Saved.")
    with nc2:
        if existing and st.button("🗑 Delete", key=f"note_del_{ticker}"):
            delete_note(ticker)
            st.success("Note deleted.")
            st.rerun()
