"""
Report Export
Generate self-contained HTML report and OHLCV CSV download from a ReportData object.
No third-party dependencies — pure Python string formatting.
"""

import io
from models.report import ReportData
from utils.formatting import fmt_price, fmt_pct, fmt_market_cap, score_color


def build_html_report(report: ReportData) -> bytes:
    """Return a self-contained HTML report as UTF-8 bytes."""
    score  = report.score.total_score
    color  = score_color(score)
    ts     = report.trade_setup
    ind    = report.indicators
    cs     = report.score.component_scores
    mtf    = report.mtf

    dir_color = {"LONG": "#00c851", "SHORT": "#ff4444"}.get(ts.direction, "#888888")

    # ── MTF section ────────────────────────────────────────────────────────────
    mtf_rows = ""
    if mtf:
        mtf_rows = f"""
        <tr><td>Daily Score</td><td>{mtf.daily_score:.0f} &mdash; {mtf.daily_label}</td></tr>
        <tr><td>Weekly Score</td><td>{mtf.weekly_score:.0f} &mdash; {mtf.weekly_label}</td></tr>
        <tr><td>MTF Direction</td><td>{mtf.agreement_direction}</td></tr>
        <tr><td>Combined Score</td><td>{mtf.combined_score:.0f}</td></tr>
        """

    # ── Trade setup section ────────────────────────────────────────────────────
    if ts.direction != "NO TRADE":
        trade_rows = f"""
        <tr><td>Direction</td><td style="color:{dir_color}; font-weight:700;">{ts.direction}</td></tr>
        <tr><td>Entry</td><td>{fmt_price(ts.entry)}</td></tr>
        <tr><td style="color:#ff4444;">Stop Loss</td>
            <td style="color:#ff4444;">{fmt_price(ts.stop_loss)} ({ts.stop_pct:.2f}%)</td></tr>
        <tr><td style="color:#00c851;">Take Profit</td>
            <td style="color:#00c851;">{fmt_price(ts.take_profit)} (+{ts.target_pct:.2f}%)</td></tr>
        <tr><td>Risk</td><td>{fmt_price(ts.risk_amount)}</td></tr>
        <tr><td>Reward</td><td>{fmt_price(ts.reward_amount)}</td></tr>
        <tr><td>R:R Ratio</td><td style="color:#ffbb33; font-weight:700;">1 : {ts.rr_ratio:.1f}</td></tr>
        <tr><td>ATR(14)</td><td>{fmt_price(ts.atr_used)}</td></tr>
        """
    else:
        trade_rows = "<tr><td colspan='2' style='color:#888;'>NO TRADE &mdash; neutral signal</td></tr>"

    vol_ratio = (
        f"{ind.current_volume / ind.volume_sma20:.2f}&times; avg"
        if ind.volume_sma20 else "N/A"
    )
    bb_pos = "Above" if ind.ema20 > ind.bb_mid else "Below"
    ma_rel = "&gt;" if ind.ema20 > ind.ema50 else "&lt;"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Katraswing Report &mdash; {report.ticker}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #0d0d1a; color: #e0e0e0;
    max-width: 860px; margin: 0 auto; padding: 24px 16px;
  }}
  h1 {{ color: #4488ff; font-size: 26px; margin-bottom: 4px; }}
  .subtitle {{ color: #555; font-size: 12px; margin-bottom: 20px; }}
  h2 {{ color: #aaaaaa; font-size: 16px; border-bottom: 1px solid #2d2d4e;
        padding-bottom: 6px; margin: 20px 0 10px; }}
  .card {{ background: #1a1a2e; padding: 16px; border-radius: 10px;
            border: 1px solid #2d2d4e; margin-bottom: 14px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  td {{ padding: 7px 10px; border-bottom: 1px solid #252540; vertical-align: top; }}
  td:first-child {{ color: #aaaaaa; width: 46%; }}
  .score-box {{ text-align: center; padding: 20px; }}
  .score-num {{ font-size: 54px; font-weight: 700; color: {color}; line-height: 1; }}
  .score-label {{ font-size: 20px; color: {color}; margin-top: 6px; }}
  .score-sub {{ color: #aaa; font-size: 13px; margin-top: 8px; }}
  .footer {{ color: #444; font-size: 11px; text-align: center;
              margin-top: 28px; border-top: 1px solid #2d2d4e; padding-top: 14px; }}
  @media print {{
    body {{ background: white; color: #111; }}
    .card {{ background: #f5f5f5; border-color: #ccc; }}
    td {{ border-color: #ddd; }}
    td:first-child {{ color: #555; }}
    h1 {{ color: #2255cc; }}
    h2 {{ color: #333; }}
    .score-num {{ color: inherit; }}
    .footer {{ color: #999; }}
  }}
</style>
</head>
<body>

<h1>&#x1F4C8; Katraswing Swing Trade Report</h1>
<p class="subtitle">Generated {report.generated_at.strftime('%Y-%m-%d %H:%M')} &mdash; Educational purposes only. Not financial advice.</p>

<h2>Company Overview</h2>
<div class="card">
  <table>
    <tr><td>Ticker</td><td><b>{report.ticker}</b></td></tr>
    <tr><td>Company</td><td>{report.company_name}</td></tr>
    <tr><td>Sector</td><td>{report.sector}</td></tr>
    <tr><td>Market Cap</td><td>{fmt_market_cap(report.market_cap)}</td></tr>
    <tr><td>Price</td><td><b>{fmt_price(report.current_price)}</b></td></tr>
    <tr><td>Change</td><td>{fmt_pct(report.price_change_pct)}</td></tr>
  </table>
</div>

<h2>Trade Score</h2>
<div class="card score-box">
  <div class="score-num">{score:.0f} <span style="font-size:24px; color:#666;">/100</span></div>
  <div class="score-label">{report.score.signal_label}</div>
  <div class="score-sub">
    Win Probability: <b>{report.score.win_probability * 100:.1f}%</b>
    &nbsp;&nbsp;|&nbsp;&nbsp;
    EV per $100 risked: <b style="color:{'#00c851' if report.score.expected_value > 0 else '#ff4444'};">
      ${report.score.expected_value:+.2f}</b>
  </div>
</div>

<h2>Trade Setup</h2>
<div class="card">
  <table>
    {trade_rows}
  </table>
</div>

<h2>Indicator Scores</h2>
<div class="card">
  <table>
    <tr style="background:#111130;">
      <th style="text-align:left; color:#666; font-size:11px; padding:6px 10px;">INDICATOR</th>
      <th style="text-align:left; color:#666; font-size:11px; padding:6px 10px;">VALUE</th>
      <th style="text-align:left; color:#666; font-size:11px; padding:6px 10px;">SCORE</th>
      <th style="text-align:left; color:#666; font-size:11px; padding:6px 10px;">WT</th>
    </tr>
    <tr><td>RSI (14)</td><td>{ind.rsi:.1f}</td><td>{cs.rsi:.1f}/10</td><td>15%</td></tr>
    <tr><td>MACD Histogram</td><td>{ind.macd_histogram:+.4f}</td><td>{cs.macd:.1f}/10</td><td>15%</td></tr>
    <tr><td>Bollinger Bands</td><td>{bb_pos} Mid</td><td>{cs.bollinger:.1f}/10</td><td>10%</td></tr>
    <tr><td>Trend (MA Align)</td><td>EMA20 {ma_rel} EMA50</td><td>{cs.trend:.1f}/10</td><td>20%</td></tr>
    <tr><td>Volume</td><td>{vol_ratio}</td><td>{cs.volume:.1f}/10</td><td>10%</td></tr>
    <tr><td>ATR Momentum</td><td>{ind.atr:.3f}</td><td>{cs.atr_momentum:.1f}/10</td><td>10%</td></tr>
    <tr><td>Stochastic %K</td><td>{ind.stoch_k:.1f}</td><td>{cs.stochastic:.1f}/10</td><td>10%</td></tr>
    <tr><td>Pattern Signals</td><td>
      {'Golden Cross ' if ind.golden_cross else ''}
      {'Death Cross ' if ind.death_cross else ''}
      {'Vol Spike ' if ind.volume_spike else ''}
      {'BB Squeeze ' if ind.bb_squeeze else ''}
      {'Above SMA200' if ind.above_200_sma else 'Below SMA200'}
    </td><td>{cs.pattern:.1f}/10</td><td>10%</td></tr>
  </table>
</div>

{f'<h2>Multi-Timeframe Analysis</h2><div class="card"><table>{mtf_rows}</table></div>' if mtf_rows else ''}

<div class="footer">
  Generated by <b>Katraswing AI Swing Trade Analyzer</b><br>
  &#x26A0; This report is for educational purposes only and does not constitute financial advice.
  Always do your own research before making investment decisions.
</div>

</body>
</html>"""

    return html.encode("utf-8")


def build_csv_data(report: ReportData) -> bytes:
    """Return last 90 bars of OHLCV data as UTF-8 CSV bytes."""
    df = report.df.iloc[-90:].copy()
    buf = io.StringIO()
    df.to_csv(buf)
    return buf.getvalue().encode("utf-8")
