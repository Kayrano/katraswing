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

    # ── Trade log ─────────────────────────────────────────────────────────────
    if result.trades:
        st.markdown("#### Trade Log")
        rows = []
        for t in result.trades[-50:]:   # show last 50
            rows.append({
                "Date":       t.entry_date,
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
