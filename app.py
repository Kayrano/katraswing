"""
KATRASWING — Swing Trade Analyzer
Main Streamlit entry point with 4 tabs:
  1. Analyzer        — single-stock analysis + MTF
  2. Watchlist       — saved tickers, batch score scan
  3. Price Alerts    — set/monitor price triggers
  4. Backtester      — historical walk-forward simulation

Run with: streamlit run app.py
"""

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Katraswing — Swing Trade Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Theme injection ───────────────────────────────────────────────────────────
from ui.renderer import get_theme_css

if "light_theme" not in st.session_state:
    st.session_state["light_theme"] = False

st.markdown(get_theme_css(st.session_state["light_theme"]), unsafe_allow_html=True)

# ── App header + theme toggle ─────────────────────────────────────────────────
title_col, toggle_col = st.columns([5, 1])
with title_col:
    st.markdown("""
    <div style="padding:6px 0 14px 0;">
        <h1 style="font-size:30px; font-weight:800; color:#4488ff; margin:0;">📈 KATRASWING</h1>
        <p style="color:#555; font-size:13px; margin-top:4px;">
            AI-Powered Swing Trade Analyzer · 4-Agent System · Multi-Timeframe · 1:2 R:R
        </p>
    </div>
    """, unsafe_allow_html=True)
with toggle_col:
    st.markdown("<div style='padding-top:18px;'>", unsafe_allow_html=True)
    st.toggle("☀ Light", key="light_theme")
    st.markdown("</div>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_analyzer, tab_watchlist, tab_alerts, tab_backtest, tab_heatmap, tab_compare = st.tabs([
    "📊 Analyzer", "👁 Watchlist", "🔔 Price Alerts", "🧪 Backtester",
    "🌡 Sector Heatmap", "⚖ Compare",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_analyzer:
    from agents.orchestrator import run_analysis
    from ui.renderer import (
        render_header, render_score_panel, render_trade_setup,
        render_candlestick_chart, render_macd_chart, render_rsi_chart,
        render_volume_chart, render_indicator_breakdown, render_mtf_panel,
        render_earnings_risk, render_position_sizing, render_chart_patterns,
        render_export_buttons,
    )

    col_input, col_btn = st.columns([4, 1])
    with col_input:
        query = st.text_input(
            label="Stock Search",
            placeholder="Enter stock name or ticker  (e.g. Apple, TSLA, Nvidia...)",
            label_visibility="collapsed",
            key="stock_query",
        )
    with col_btn:
        analyze_clicked = st.button("Analyze", use_container_width=True, type="primary", key="btn_analyze")

    if analyze_clicked and query.strip():
        with st.spinner(f"Analyzing **{query.strip()}** — running 4-agent analysis + multi-timeframe..."):
            try:
                report = run_analysis(query.strip())
                st.session_state["last_report"] = report
            except ValueError as e:
                st.error(str(e))
                st.stop()
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()

    report = st.session_state.get("last_report")

    if report:
        # ── Sidebar sticky summary ────────────────────────────────────────────
        from utils.formatting import score_color, direction_color
        _sc = score_color(report.score.total_score)
        _dc = direction_color(report.trade_setup.direction)
        _mtf_dir = report.mtf.agreement_direction if report.mtf else "—"
        _mtf_score = f"{report.mtf.combined_score:.0f}" if report.mtf else "—"
        with st.sidebar:
            st.markdown(f"""
            <div style="padding:10px; border-radius:10px; background:#1a1a2e;
                        border:1px solid {_sc}; margin-bottom:10px;">
                <div style="font-size:16px; font-weight:700; color:#e0e0e0;">{report.ticker}</div>
                <div style="font-size:11px; color:#888; margin-bottom:6px;">{report.company_name[:24]}</div>
                <div style="font-size:28px; font-weight:700; color:{_sc};">{report.score.total_score:.0f}</div>
                <div style="font-size:12px; color:{_sc};">{report.score.signal_label}</div>
                <div style="margin-top:6px; font-size:13px; font-weight:700; color:{_dc};">
                    {report.trade_setup.direction}
                </div>
                <div style="margin-top:4px; font-size:11px; color:#888;">
                    Win: {report.score.win_probability*100:.1f}%
                    &nbsp;|&nbsp; EV: ${report.score.expected_value:+.2f}
                </div>
                <div style="margin-top:4px; font-size:11px; color:#888;">
                    MTF: {_mtf_dir} &nbsp;({_mtf_score})
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        render_header(report)

        col_score, col_trade = st.columns([1, 1], gap="medium")
        with col_score:
            render_score_panel(report)
        with col_trade:
            st.markdown("### Trade Setup")
            render_trade_setup(report)

        st.markdown("---")
        render_mtf_panel(report)

        st.markdown("---")
        render_candlestick_chart(report)

        col_macd, col_rsi = st.columns(2, gap="medium")
        with col_macd:
            render_macd_chart(report)
        with col_rsi:
            render_rsi_chart(report)

        render_volume_chart(report)

        st.markdown("---")
        render_chart_patterns(report.df)

        st.markdown("---")
        render_indicator_breakdown(report)

        st.markdown("---")
        from ui.renderer import render_canslim_panel
        render_canslim_panel(report)

        st.markdown("---")
        render_earnings_risk(report.ticker)

        st.markdown("---")
        ts = report.trade_setup
        if ts.direction != "NO TRADE":
            render_position_sizing(ts.entry, ts.stop_loss, ts.take_profit)

        st.markdown("---")
        render_export_buttons(report)

        st.markdown("""
        <div style="text-align:center; margin-top:30px; color:#333; font-size:12px;">
            ⚠ Educational purposes only. Not financial advice.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding:60px 20px; color:#444;">
            <div style="font-size:56px;">📊</div>
            <h3 style="color:#555;">Enter a stock name or ticker above</h3>
            <p style="font-size:13px; max-width:460px; margin:10px auto;">
                Type <b>Apple</b>, <b>Tesla</b>, <b>NVDA</b>, or any ticker and click Analyze.
            </p>
            <div style="color:#444; font-size:12px; margin-top:20px;">
                RSI · MACD · Bollinger Bands · ATR · Stochastic · Multi-Timeframe · 1:2 R:R
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — WATCHLIST
# ═══════════════════════════════════════════════════════════════════════════════
with tab_watchlist:
    from data.watchlist import load_watchlist, add_ticker, remove_ticker
    from agents.orchestrator import run_analysis as _run
    from utils.formatting import fmt_price, fmt_pct, score_color

    st.markdown("### Watchlist")

    # Add ticker
    wc1, wc2 = st.columns([4, 1])
    with wc1:
        new_ticker = st.text_input(
            label="Add Ticker",
            placeholder="Ticker or company name (e.g. AAPL, Microsoft)",
            label_visibility="collapsed",
            key="wl_input",
        )
    with wc2:
        if st.button("Add", key="wl_add", use_container_width=True):
            if new_ticker.strip():
                add_ticker(new_ticker.strip().upper())
                st.rerun()

    tickers = load_watchlist()

    if not tickers:
        st.info("Your watchlist is empty. Add tickers above.")
    else:
        scan_clicked = st.button("🔄 Scan All", key="wl_scan", type="primary")

        if scan_clicked:
            results = []
            progress = st.progress(0, text="Scanning...")
            for idx, t in enumerate(tickers):
                progress.progress((idx + 1) / len(tickers), text=f"Scanning {t}...")
                try:
                    r = _run(t)
                    results.append({
                        "ticker":     r.ticker,
                        "company":    r.company_name[:28],
                        "price":      r.current_price,
                        "chg_pct":    r.price_change_pct,
                        "score":      r.score.total_score,
                        "signal":     r.score.signal_label,
                        "direction":  r.trade_setup.direction,
                        "entry":      r.trade_setup.entry,
                        "stop_loss":  r.trade_setup.stop_loss,
                        "take_profit":r.trade_setup.take_profit,
                        "win_prob":   r.score.win_probability,
                        "ev":         r.score.expected_value,
                        "mtf":        r.mtf.agreement_direction if r.mtf else "—",
                    })
                except Exception as e:
                    results.append({"ticker": t, "company": "Error", "price": 0,
                                    "chg_pct": 0, "score": 0, "signal": str(e)[:30],
                                    "direction": "—", "entry": 0, "stop_loss": 0,
                                    "take_profit": 0, "win_prob": 0.0, "ev": 0.0, "mtf": "—"})
            progress.empty()
            st.session_state["wl_results"] = results

        results = st.session_state.get("wl_results", [])

        if results:
            import pandas as pd
            df_wl = pd.DataFrame(results)
            df_wl = df_wl.sort_values("score", ascending=False)

            # Render colored rows
            for _, row in df_wl.iterrows():
                s = float(row["score"])
                color = score_color(s)
                chg   = float(row["chg_pct"])
                chg_col = "#00c851" if chg >= 0 else "#ff4444"
                arrow   = "▲" if chg >= 0 else "▼"
                wp = float(row["win_prob"]) * 100
                ev = float(row["ev"])
                ev_col = "#00c851" if ev >= 0 else "#ff4444"

                c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([1.5, 2, 1.2, 1.2, 1, 1.2, 1.2, 0.6])
                c1.markdown(f"<b style='color:#e0e0e0;'>{row['ticker']}</b>", unsafe_allow_html=True)
                c2.markdown(f"<span style='color:#888; font-size:12px;'>{row['company']}</span>", unsafe_allow_html=True)
                c3.markdown(f"<span style='color:#e0e0e0;'>{fmt_price(row['price'])}</span> <span style='color:{chg_col}; font-size:11px;'>{arrow}{abs(chg):.1f}%</span>", unsafe_allow_html=True)
                c4.markdown(f"<span style='color:{color}; font-weight:700; font-size:18px;'>{s:.0f}</span>", unsafe_allow_html=True)
                c5.markdown(f"<span style='color:{color}; font-size:11px;'>{row['signal']}</span>", unsafe_allow_html=True)
                c6.markdown(f"<span style='color:#e0e0e0; font-size:12px;'>{wp:.1f}%</span> <span style='color:#666; font-size:10px;'>win</span>", unsafe_allow_html=True)
                c7.markdown(f"<span style='color:{ev_col}; font-size:12px;'>${ev:+.1f}</span> <span style='color:#666; font-size:10px;'>EV</span>", unsafe_allow_html=True)
                if c8.button("✕", key=f"rm_{row['ticker']}"):
                    remove_ticker(row["ticker"])
                    st.session_state.pop("wl_results", None)
                    st.rerun()
                st.markdown("<div style='border-bottom:1px solid #1e1e2e; margin:2px 0;'></div>", unsafe_allow_html=True)

        else:
            # Show static list with remove buttons
            for t in tickers:
                tc1, tc2 = st.columns([6, 1])
                tc1.markdown(f"<span style='color:#e0e0e0; font-size:15px;'>📌 {t}</span>", unsafe_allow_html=True)
                if tc2.button("✕", key=f"rm_s_{t}"):
                    remove_ticker(t)
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PRICE ALERTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_alerts:
    from data.alerts import load_alerts, add_alert, remove_alert, check_alerts
    from utils.formatting import fmt_price

    st.markdown("### Price Alerts")

    # Check alerts on load
    triggered = check_alerts()
    for t in triggered:
        cond = "rose above" if t["condition"] == "above" else "fell below"
        st.success(f"🔔 **{t['ticker']}** {cond} your target {fmt_price(t['target'])}  —  current: {fmt_price(t['current'])}"
                   + (f"  · {t['note']}" if t['note'] else ""))

    # Add alert form
    with st.expander("➕ Set New Alert", expanded=False):
        ac1, ac2, ac3, ac4 = st.columns([1.5, 1.5, 1.5, 1])
        with ac1:
            al_ticker = st.text_input("Ticker", placeholder="AAPL", key="al_ticker",
                                      label_visibility="visible")
        with ac2:
            al_price = st.number_input("Target Price ($)", min_value=0.01, step=0.5, key="al_price",
                                       label_visibility="visible")
        with ac3:
            al_cond = st.selectbox("Condition", ["above", "below"], key="al_cond",
                                   label_visibility="visible")
        with ac4:
            al_note = st.text_input("Note (optional)", key="al_note", label_visibility="visible")

        if st.button("Create Alert", key="al_create", type="primary"):
            if al_ticker.strip() and al_price > 0:
                add_alert(al_ticker.strip().upper(), al_price, al_cond, al_note)
                st.success(f"Alert created: {al_ticker.upper()} {al_cond} ${al_price:.2f}")
                st.rerun()
            else:
                st.warning("Please enter a ticker and target price.")

    # Alert list
    alerts = load_alerts()
    if not alerts:
        st.info("No alerts set. Use the form above to create one.")
    else:
        active   = [a for a in alerts if not a.triggered]
        fired    = [a for a in alerts if a.triggered]

        if active:
            st.markdown("#### Active Alerts")
            for idx, a in enumerate(alerts):
                if a.triggered:
                    continue
                real_idx = alerts.index(a)
                bc1, bc2, bc3, bc4, bc5 = st.columns([1, 1.5, 1.2, 2, 0.6])
                bc1.markdown(f"<b style='color:#e0e0e0;'>{a.ticker}</b>", unsafe_allow_html=True)
                cond_color = "#00c851" if a.condition == "above" else "#ff4444"
                bc2.markdown(f"<span style='color:{cond_color};'>{a.condition.upper()} {fmt_price(a.target_price)}</span>", unsafe_allow_html=True)
                bc3.markdown(f"<span style='color:#666; font-size:12px;'>Set {a.created_at[:10]}</span>", unsafe_allow_html=True)
                bc4.markdown(f"<span style='color:#888; font-size:12px;'>{a.note}</span>", unsafe_allow_html=True)
                if bc5.button("✕", key=f"del_al_{real_idx}"):
                    remove_alert(real_idx)
                    st.rerun()
                st.markdown("<div style='border-bottom:1px solid #1e1e2e; margin:2px 0;'></div>", unsafe_allow_html=True)

        if fired:
            with st.expander(f"✅ Triggered Alerts ({len(fired)})", expanded=False):
                for a in fired:
                    st.markdown(
                        f"**{a.ticker}** — {a.condition} ${a.target_price:.2f} "
                        f"→ triggered at **${a.triggered_price:.2f}** on {(a.triggered_at or '')[:10]}"
                        + (f" · _{a.note}_" if a.note else ""),
                        unsafe_allow_html=False,
                    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BACKTESTER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_backtest:
    from agents.backtester import run_backtest
    from ui.renderer import render_backtest_results

    st.markdown("### Historical Backtester")
    st.caption("Walk-forward simulation — no look-ahead bias. Enters trades when score crosses threshold.")

    bt1, bt2, bt3, bt4 = st.columns([2, 1, 1, 1])
    with bt1:
        bt_ticker = st.text_input(
            label="Ticker",
            placeholder="AAPL, Tesla, MSFT...",
            label_visibility="collapsed",
            key="bt_ticker",
        )
    with bt2:
        bt_period = st.selectbox("Period", ["1y", "2y", "3y"], index=1,
                                  label_visibility="visible", key="bt_period")
    with bt3:
        bt_threshold = st.slider("Long threshold", 50, 80, 65, step=5, key="bt_thresh")
    with bt4:
        bt_run = st.button("Run Backtest", type="primary", use_container_width=True, key="bt_run")

    if bt_run and bt_ticker.strip():
        with st.spinner(f"Running walk-forward backtest on **{bt_ticker.strip()}** ({bt_period})... this may take 20-40 seconds."):
            try:
                bt_result = run_backtest(
                    ticker=bt_ticker.strip(),
                    period=bt_period,
                    score_threshold=float(bt_threshold),
                    short_threshold=float(100 - bt_threshold),
                )
                st.session_state["bt_result"] = bt_result
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Backtest failed: {e}")

    bt_result = st.session_state.get("bt_result")
    if bt_result:
        render_backtest_results(bt_result)
    else:
        st.markdown("""
        <div style="text-align:center; padding:50px 20px; color:#444;">
            <div style="font-size:48px;">🧪</div>
            <h3 style="color:#555;">Enter a ticker and click Run Backtest</h3>
            <p style="font-size:13px; max-width:420px; margin:10px auto; color:#444;">
                The engine replays 1-3 years of daily data, scores each day,
                enters trades at the threshold, and tracks SL/TP outcomes.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SECTOR HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════
with tab_heatmap:
    from data.sector_scan import SECTOR_TICKERS, scan_all_sectors, sector_averages
    from agents.orchestrator import run_analysis as _run_hm
    from ui.renderer import render_sector_heatmap

    st.markdown("### Sector Heatmap")
    st.caption("Scans 80+ liquid stocks across 11 GICS sectors and ranks by average trade score.")

    # Sector selector + scan button
    hm_c1, hm_c2 = st.columns([3, 1])
    with hm_c1:
        selected_sectors = st.multiselect(
            "Sectors to scan (leave empty = all)",
            options=list(SECTOR_TICKERS.keys()),
            default=[],
            key="hm_sectors",
            label_visibility="visible",
        )
    with hm_c2:
        hm_scan = st.button("🔄 Scan Sectors", type="primary",
                             use_container_width=True, key="hm_scan")

    if hm_scan:
        sectors_to_scan = selected_sectors or list(SECTOR_TICKERS.keys())
        total_tickers = sum(len(SECTOR_TICKERS[s]) for s in sectors_to_scan)

        progress_bar = st.progress(0, text="Starting sector scan...")
        progress_state = {"count": 0}

        def _hm_progress(i, total, ticker, sector):
            progress_state["count"] = i + 1
            pct = (i + 1) / total
            progress_bar.progress(pct, text=f"[{i+1}/{total}] Scanning {ticker} ({sector})...")

        try:
            scan_data = scan_all_sectors(
                run_analysis_fn=_run_hm,
                sectors=sectors_to_scan,
                progress_callback=_hm_progress,
            )
            progress_bar.empty()
            st.session_state["hm_scan_data"] = scan_data
            st.session_state["hm_sector_avgs"] = sector_averages(scan_data)
        except Exception as e:
            progress_bar.empty()
            st.error(f"Scan failed: {e}")

    scan_data  = st.session_state.get("hm_scan_data")
    sector_avgs = st.session_state.get("hm_sector_avgs")

    if scan_data and sector_avgs:
        render_sector_heatmap(scan_data, sector_avgs)
    else:
        st.markdown("""
        <div style="text-align:center; padding:50px 20px; color:#444;">
            <div style="font-size:48px;">🌡</div>
            <h3 style="color:#555;">Select sectors and click Scan</h3>
            <p style="font-size:13px; max-width:440px; margin:10px auto; color:#444;">
                Scanning all 11 sectors (~80 tickers) takes 3-5 minutes.
                Select specific sectors for a faster partial scan.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — COMPARE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    from agents.orchestrator import run_analysis as _run_cmp
    from ui.renderer import render_comparison

    st.markdown("### Comparison Mode")
    st.caption("Analyze 2–3 stocks side by side. Scores, trade setups, and a radar chart are shown together.")

    cmp_c1, cmp_c2, cmp_c3, cmp_c4 = st.columns([2, 2, 2, 1])
    with cmp_c1:
        cmp_t1 = st.text_input(
            label="Stock 1", placeholder="AAPL",
            label_visibility="visible", key="cmp_t1",
        )
    with cmp_c2:
        cmp_t2 = st.text_input(
            label="Stock 2", placeholder="MSFT",
            label_visibility="visible", key="cmp_t2",
        )
    with cmp_c3:
        cmp_t3 = st.text_input(
            label="Stock 3 (optional)", placeholder="NVDA",
            label_visibility="visible", key="cmp_t3",
        )
    with cmp_c4:
        st.markdown("<div style='padding-top:22px;'>", unsafe_allow_html=True)
        cmp_run = st.button("Compare", type="primary", use_container_width=True, key="cmp_run")
        st.markdown("</div>", unsafe_allow_html=True)

    if cmp_run:
        queries = [q.strip() for q in [cmp_t1, cmp_t2, cmp_t3] if q.strip()]
        if len(queries) < 2:
            st.warning("Enter at least 2 tickers to compare.")
        else:
            cmp_reports = []
            with st.spinner(f"Analyzing {', '.join(queries)}..."):
                for q in queries:
                    try:
                        cmp_reports.append(_run_cmp(q))
                    except Exception as e:
                        st.error(f"Could not analyze **{q}**: {e}")
            if cmp_reports:
                st.session_state["cmp_reports"] = cmp_reports

    cmp_reports = st.session_state.get("cmp_reports")

    if cmp_reports:
        st.markdown("---")
        render_comparison(cmp_reports)
    else:
        st.markdown("""
        <div style="text-align:center; padding:60px 20px; color:#444;">
            <div style="font-size:56px;">⚖</div>
            <h3 style="color:#555;">Enter 2–3 stocks above and click Compare</h3>
            <p style="font-size:13px; max-width:460px; margin:10px auto; color:#444;">
                Scores, signals, trade levels, and a radar chart will appear side by side.
            </p>
        </div>
        """, unsafe_allow_html=True)
