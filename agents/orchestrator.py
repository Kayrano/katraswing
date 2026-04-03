"""
Expert Software Developer Agent — Orchestrator
Coordinates the full analysis pipeline: fetch → analyze → score → trade setup → MTF → report.
"""

import yfinance as yf
import pandas as pd
from data.fetcher import fetch_stock_data
from agents.analyzer import AnalyzerAgent
from agents.statistician import StatisticianAgent
from agents.trader import TraderAgent
from models.report import ReportData, MTFResult


def run_analysis(query: str) -> ReportData:
    """
    Main pipeline entry point.
    Accepts a stock name or ticker string, returns a complete ReportData.
    """
    if not query or not query.strip():
        raise ValueError("Please enter a stock name or ticker symbol.")

    # ── Phase 1: Fetch daily market data ─────────────────────────────────────
    stock_data = fetch_stock_data(query.strip())

    df          = stock_data["df"]
    ticker      = stock_data["ticker"]
    company_name = stock_data["company_name"]
    sector      = stock_data["sector"]
    market_cap  = stock_data["market_cap"]
    current_price     = stock_data["current_price"]
    price_change_pct  = stock_data["price_change_pct"]

    analyzer     = AnalyzerAgent()
    statistician = StatisticianAgent()
    trader       = TraderAgent()

    # ── Phase 2: Daily technical analysis ────────────────────────────────────
    indicators   = analyzer.analyze(df)
    score_result = statistician.score(indicators)

    # ── Phase 3: Trade setup ──────────────────────────────────────────────────
    trade_setup = trader.compute_trade_setup(df, indicators, score_result.total_score)

    # ── Phase 4: Multi-timeframe analysis (weekly) ────────────────────────────
    mtf = _run_mtf(ticker, score_result.total_score, analyzer, statistician)

    # ── Phase 5: CAN SLIM analysis ────────────────────────────────────────────
    try:
        from agents.canslim_agent import CanSlimAgent
        canslim = CanSlimAgent().analyze(ticker, df)
    except Exception:
        canslim = None

    # ── Phase 6: Assemble report ──────────────────────────────────────────────
    return ReportData(
        ticker=ticker,
        company_name=company_name,
        sector=sector,
        market_cap=market_cap,
        current_price=current_price,
        price_change_pct=price_change_pct,
        df=df,
        indicators=indicators,
        trade_setup=trade_setup,
        score=score_result,
        mtf=mtf,
        canslim=canslim,
    )


def _run_mtf(
    ticker: str,
    daily_score: float,
    analyzer: AnalyzerAgent,
    statistician: StatisticianAgent,
) -> MTFResult:
    """Fetch weekly OHLCV and compute weekly score for multi-timeframe view."""
    try:
        weekly_df = yf.Ticker(ticker).history(period="3y", interval="1wk", auto_adjust=True)
        weekly_df = weekly_df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        weekly_df.index = pd.to_datetime(weekly_df.index)

        if len(weekly_df) < 30:
            raise ValueError("Not enough weekly bars")

        weekly_ind   = analyzer.analyze(weekly_df)
        weekly_score = statistician.score(weekly_ind)

        return _build_mtf(daily_score, weekly_score.total_score, weekly_score.signal_label, weekly_ind)

    except Exception:
        # If weekly fetch fails, return a degraded MTF with only daily data
        from agents.statistician import StatisticianAgent as _S
        label = _S()._label(daily_score)
        return MTFResult(
            daily_score=daily_score,
            weekly_score=0.0,
            daily_label=label,
            weekly_label="N/A",
            agreement=False,
            agreement_direction="UNKNOWN",
            combined_score=daily_score,
        )


def _build_mtf(
    daily_score: float,
    weekly_score: float,
    weekly_label: str,
    weekly_ind,
) -> MTFResult:
    from agents.statistician import StatisticianAgent as _S
    stat = _S()

    daily_label = stat._label(daily_score)

    daily_bull  = daily_score >= 50
    weekly_bull = weekly_score >= 50

    if daily_bull and weekly_bull:
        agreement = True
        direction = "BULLISH"
    elif not daily_bull and not weekly_bull:
        agreement = True
        direction = "BEARISH"
    else:
        agreement = False
        direction = "MIXED"

    combined = round(daily_score * 0.60 + weekly_score * 0.40, 1)

    return MTFResult(
        daily_score=daily_score,
        weekly_score=weekly_score,
        daily_label=daily_label,
        weekly_label=weekly_label,
        agreement=agreement,
        agreement_direction=direction,
        combined_score=combined,
        weekly_indicators=weekly_ind,
    )
