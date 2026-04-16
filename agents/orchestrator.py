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
from models.report import ReportData, MTFResult, PoliticianTradesData


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
    asset_class       = stock_data.get("asset_class", "EQUITY")

    # Flags for downstream gating — fundamentals only make sense for equities
    is_equity = asset_class == "EQUITY"

    analyzer     = AnalyzerAgent()
    statistician = StatisticianAgent()
    trader       = TraderAgent()

    # ── Phase 2: Daily technical analysis ────────────────────────────────────
    indicators   = analyzer.analyze(df)
    score_result = statistician.score(indicators)

    # ── Phase 3: CAN SLIM analysis (equity only) ──────────────────────────────
    canslim = None
    if is_equity:
        try:
            from agents.canslim_agent import CanSlimAgent
            canslim = CanSlimAgent().analyze(ticker, df)
        except Exception:
            canslim = None

    # ── Phase 4: Blend technical score with CAN SLIM (80/20) ─────────────────
    if canslim is not None:
        blended = round(0.80 * score_result.total_score + 0.20 * canslim.overall_score, 1)
        blended = max(0.0, min(100.0, blended))
        score_result.total_score   = blended
        score_result.signal_label  = statistician._label(blended)
        score_result.win_probability = round(statistician._win_probability(blended), 3)
        score_result.expected_value  = round(statistician._expected_value(score_result.win_probability), 2)

    # ── Phase 5: Macro regime + earnings proximity filters ───────────────
    # For non-equity assets (gold, forex, crypto) skip earnings check AND
    # the SPY/VIX equity bear-market filter (gold often rises in bear markets).
    apply_spy_vix = asset_class in ("EQUITY", "ETF", "INDEX")
    filtered_score, filter_notes = _apply_macro_filters(
        ticker, score_result.total_score,
        apply_earnings=is_equity,
        apply_spy_vix=apply_spy_vix,
    )
    if filtered_score != score_result.total_score:
        score_result.total_score     = round(filtered_score, 1)
        score_result.signal_label    = statistician._label(filtered_score)
        score_result.win_probability = round(statistician._win_probability(filtered_score), 3)
        score_result.expected_value  = round(statistician._expected_value(score_result.win_probability), 2)

    # ── Phase 5b: Politician trades signal (equity only) ─────────────────────
    politician_data = None
    if is_equity:
        politician_data = _apply_politician_signal(ticker, score_result, statistician, filter_notes)

    # ── Phase 6: Trade setup (uses filtered score for direction) ────────────
    trade_setup = trader.compute_trade_setup(df, indicators, score_result.total_score)

    # ── Phase 7: Multi-timeframe analysis (weekly) ────────────────────
    mtf = _run_mtf(ticker, score_result.total_score, analyzer, statistician)

    # ── Phase 8: Assemble report ───────────────────────────────
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
        politician=politician_data,
        filter_notes=filter_notes,
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

    # Weight weekly more heavily when both timeframes agree (max 65/35 → 60/40 default)
    # Agreement is already captured by the blend; no additive bonus to avoid double-counting
    if agreement:
        combined = round(daily_score * 0.65 + weekly_score * 0.35, 1)
    else:
        combined = round(daily_score * 0.60 + weekly_score * 0.40, 1)
    combined = max(0.0, min(100.0, combined))

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

def _apply_politician_signal(
    ticker: str,
    score_result,
    statistician,
    filter_notes: list,
) -> "PoliticianTradesData | None":
    """
    Fetch congressional trading data for the ticker and apply a score correction.
    Capitol Trades data has a 30-45 day disclosure delay — this is expected and noted in the UI.
    Correction is capped at ±8 points to keep the signal as a soft confirmation, not a driver.
    """
    try:
        from data.politician_trades import (
            fetch_ticker_trades,
            compute_politician_sentiment,
            compute_score_correction,
        )

        trades = fetch_ticker_trades(ticker, days_back=120)
        sentiment_data = compute_politician_sentiment(trades)
        delta, note = compute_score_correction(sentiment_data)

        if delta != 0.0:
            new_score = round(max(0.0, min(100.0, score_result.total_score + delta)), 1)
            score_result.total_score     = new_score
            score_result.signal_label    = statistician._label(new_score)
            score_result.win_probability = round(statistician._win_probability(new_score), 3)
            score_result.expected_value  = round(statistician._expected_value(score_result.win_probability), 2)

        filter_notes.append(note)

        return PoliticianTradesData(
            sentiment=sentiment_data["sentiment"],
            buy_count=sentiment_data["buy_count"],
            sell_count=sentiment_data["sell_count"],
            buy_volume=sentiment_data["buy_volume"],
            sell_volume=sentiment_data["sell_volume"],
            top_performer_signal=sentiment_data["top_performer_signal"],
            top_performer_trades=sentiment_data["top_performer_trades"],
            recent_trades=trades[:30],
            score_delta=delta,
            delay_note=sentiment_data["delay_note"],
        )
    except Exception:
        return None


def _apply_macro_filters(
    ticker: str,
    score: float,
    apply_earnings: bool = True,
    apply_spy_vix: bool = True,
) -> tuple:
    """
    Apply macro regime and earnings proximity filters to the combined score.
    All external fetches are wrapped in try/except; failures are silent.

    apply_spy_vix=False skips the SPY/VIX equity-bear-market penalty —
    used for FOREX, FUTURES, and CRYPTO where gold/oil/crypto often move
    inversely or independently of the US equity market.
    """
    notes = []
    is_long = score >= 50

    # SPY 200-day SMA: suppress LONG signals in a bear market (equities/ETFs only)
    if apply_spy_vix:
        try:
            spy_s = yf.Ticker("SPY").history(period="1y", interval="1d", auto_adjust=True)["Close"]
            if len(spy_s) >= 200:
                spy_sma200 = float(spy_s.rolling(200).mean().iloc[-1])
                spy_close  = float(spy_s.iloc[-1])
                if spy_close < spy_sma200 and is_long:
                    score = max(35.0, score - 10.0)
                    notes.append(
                        f"SPY {spy_close:.0f} < 200 SMA {spy_sma200:.0f} — bearish macro, LONG score −10"
                    )
        except Exception:
            pass

    # VIX dynamic filter: elevated volatility vs its own 20-day MA (equities/ETFs only)
    if apply_spy_vix:
        try:
            vix_s = yf.Ticker("^VIX").history(period="3mo", interval="1d", auto_adjust=True)["Close"]
            if len(vix_s) >= 20:
                vix_ma20 = float(vix_s.rolling(20).mean().iloc[-1])
                vix_cur  = float(vix_s.iloc[-1])
                if vix_cur > vix_ma20 and is_long:
                    score = max(0.0, score - 5.0)
                    notes.append(
                        f"VIX {vix_cur:.1f} > 20-day MA {vix_ma20:.1f} — elevated volatility, LONG score −5"
                    )
        except Exception:
            pass

    # Earnings proximity penalty (skipped for non-equity assets)
    if not apply_earnings:
        return round(score, 1), notes

    try:
        from datetime import date as _date
        t = yf.Ticker(ticker)
        earn_date = None

        # Try earnings_dates (newer yfinance)
        try:
            edates = t.earnings_dates
            if edates is not None and not edates.empty:
                now_utc = pd.Timestamp.now(tz="UTC")
                future = edates[edates.index >= now_utc]
                if not future.empty:
                    earn_date = future.index[0].date()
        except Exception:
            pass

        # Fallback: calendar
        if earn_date is None:
            try:
                cal = t.calendar
                if cal is not None and isinstance(cal, pd.DataFrame) and not cal.empty:
                    for col in cal.columns:
                        try:
                            d = pd.to_datetime(cal[col].iloc[0]).date()
                            if d >= _date.today():
                                earn_date = d
                                break
                        except Exception:
                            pass
            except Exception:
                pass

        if earn_date is not None:
            days = (earn_date - _date.today()).days
            if 0 <= days <= 3:
                score = round(score * 0.70, 1)
                notes.append(f"Earnings in {days}d — score −30%")
            elif 4 <= days <= 7:
                score = round(score * 0.85, 1)
                notes.append(f"Earnings in {days}d — score −15%")
    except Exception:
        pass

    return round(score, 1), notes
