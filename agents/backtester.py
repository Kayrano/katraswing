"""
Expert Financial Statistician — Backtesting Engine
Walk-forward simulation: scores each historical day, enters trades when score
crosses threshold, tracks SL/TP outcomes. No look-ahead bias (indicators are
causal — computed on data up to each point in time when calculated on full series).
"""

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta

from agents.statistician import StatisticianAgent
from models.report import IndicatorBundle


@dataclass
class BacktestTrade:
    entry_date: str
    exit_date: str
    direction: str          # "LONG" | "SHORT"
    entry_price: float
    stop_loss: float
    take_profit: float
    exit_price: float
    outcome: str            # "WIN" | "LOSS" | "TIMEOUT"
    pnl_pct: float          # % gain/loss on the trade
    bars_held: int
    score_at_entry: float


@dataclass
class BacktestResult:
    ticker: str
    period: str
    total_trades: int
    wins: int
    losses: int
    timeouts: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    avg_bars_held: float
    total_return_pct: float
    max_drawdown_pct: float
    profit_factor: float
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    score_threshold: float = 65.0


ATR_STOP_MULT   = 1.5
REWARD_MULT     = 2.0
MAX_BARS        = 20        # Force-exit after 20 bars if neither SL nor TP hit


def run_backtest(
    ticker: str,
    period: str = "2y",
    score_threshold: float = 65.0,
    short_threshold: float = 35.0,
) -> BacktestResult:
    """
    Walk-forward backtest for a given ticker.
    Enters LONG when score >= score_threshold, SHORT when score <= short_threshold.
    """
    # ── Fetch data ────────────────────────────────────────────────────────────
    df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)

    if len(df) < 120:
        raise ValueError(f"Not enough data for backtest (got {len(df)} bars, need 120+).")

    # ── Compute all indicators on full series (causal — no look-ahead bias) ──
    scores = _compute_scores_series(df)

    stat = StatisticianAgent()
    atr_series = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    trades: list[BacktestTrade] = []
    in_trade = False
    trade_direction = None
    entry_idx = None
    entry_price = stop_loss = take_profit = 0.0
    entry_score = 0.0

    equity = 1.0                    # start at 1.0 (normalized)
    equity_curve = [1.0]
    peak_equity = 1.0
    max_drawdown = 0.0

    warmup = 60   # skip first 60 bars (indicator warmup)

    for i in range(warmup, len(df) - 1):
        score = scores[i]
        atr = float(atr_series.iloc[i]) if not pd.isna(atr_series.iloc[i]) else 0.0
        close = float(df["Close"].iloc[i])

        if not in_trade:
            # Check for entry signal
            if score >= score_threshold and atr > 0:
                direction = "LONG"
            elif score <= short_threshold and atr > 0:
                direction = "SHORT"
            else:
                equity_curve.append(equity)
                continue

            entry_price = close
            if direction == "LONG":
                stop_loss   = entry_price - ATR_STOP_MULT * atr
                take_profit = entry_price + ATR_STOP_MULT * atr * REWARD_MULT
            else:
                stop_loss   = entry_price + ATR_STOP_MULT * atr
                take_profit = entry_price - ATR_STOP_MULT * atr * REWARD_MULT

            in_trade = True
            trade_direction = direction
            entry_idx = i
            entry_score = score

        else:
            # Check exit: next bar's high/low vs SL/TP
            next_high  = float(df["High"].iloc[i])
            next_low   = float(df["Low"].iloc[i])
            bars_held  = i - entry_idx
            exit_price = close
            outcome    = None

            if trade_direction == "LONG":
                if next_low <= stop_loss:
                    outcome = "LOSS"
                    exit_price = stop_loss
                elif next_high >= take_profit:
                    outcome = "WIN"
                    exit_price = take_profit
            else:  # SHORT
                if next_high >= stop_loss:
                    outcome = "LOSS"
                    exit_price = stop_loss
                elif next_low <= take_profit:
                    outcome = "WIN"
                    exit_price = take_profit

            # Timeout exit
            if outcome is None and bars_held >= MAX_BARS:
                outcome = "TIMEOUT"
                exit_price = close

            if outcome is not None:
                if trade_direction == "LONG":
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                else:
                    pnl_pct = ((entry_price - exit_price) / entry_price) * 100

                # Update equity (simple percentage compounding)
                equity *= (1 + pnl_pct / 100)
                peak_equity = max(peak_equity, equity)
                drawdown = (peak_equity - equity) / peak_equity * 100
                max_drawdown = max(max_drawdown, drawdown)

                trades.append(BacktestTrade(
                    entry_date=str(df.index[entry_idx].date()),
                    exit_date=str(df.index[i].date()),
                    direction=trade_direction,
                    entry_price=round(entry_price, 4),
                    stop_loss=round(stop_loss, 4),
                    take_profit=round(take_profit, 4),
                    exit_price=round(exit_price, 4),
                    outcome=outcome,
                    pnl_pct=round(pnl_pct, 2),
                    bars_held=bars_held,
                    score_at_entry=round(entry_score, 1),
                ))
                in_trade = False

        equity_curve.append(equity)

    # ── Aggregate stats ───────────────────────────────────────────────────────
    total    = len(trades)
    wins     = [t for t in trades if t.outcome == "WIN"]
    losses   = [t for t in trades if t.outcome == "LOSS"]
    timeouts = [t for t in trades if t.outcome == "TIMEOUT"]

    win_rate    = len(wins) / total * 100 if total else 0.0
    avg_win     = np.mean([t.pnl_pct for t in wins])    if wins    else 0.0
    avg_loss    = np.mean([t.pnl_pct for t in losses])  if losses  else 0.0
    avg_bars    = np.mean([t.bars_held for t in trades]) if trades  else 0.0
    total_ret   = (equity - 1.0) * 100

    gross_profit = sum(t.pnl_pct for t in wins)
    gross_loss   = abs(sum(t.pnl_pct for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return BacktestResult(
        ticker=ticker.upper(),
        period=period,
        total_trades=total,
        wins=len(wins),
        losses=len(losses),
        timeouts=len(timeouts),
        win_rate=round(win_rate, 1),
        avg_win_pct=round(float(avg_win), 2),
        avg_loss_pct=round(float(avg_loss), 2),
        avg_bars_held=round(float(avg_bars), 1),
        total_return_pct=round(total_ret, 2),
        max_drawdown_pct=round(max_drawdown, 2),
        profit_factor=round(float(profit_factor), 2) if profit_factor != float("inf") else 999.0,
        trades=trades,
        equity_curve=equity_curve,
        score_threshold=score_threshold,
    )


def _compute_scores_series(df: pd.DataFrame) -> list[float]:
    """
    Compute a trade score for every row in df using causal indicator values.
    Builds a single StatisticianAgent and scores each bar's indicator snapshot.
    """
    stat = StatisticianAgent()
    scores = []

    # Pre-compute all indicator series on full df (causal, no look-ahead)
    rsi_s     = ta.rsi(df["Close"], length=14)
    macd_df   = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    bb_df     = ta.bbands(df["Close"], length=20, std=2)
    ema20_s   = ta.ema(df["Close"], length=20)
    ema50_s   = ta.ema(df["Close"], length=50)
    sma200_s  = ta.sma(df["Close"], length=200)
    atr_s     = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    obv_s     = ta.obv(df["Close"], df["Volume"])
    stoch_df  = ta.stoch(df["High"], df["Low"], df["Close"], k=14, d=3)
    vol_sma   = df["Volume"].rolling(20).mean()

    n = len(df)

    def _safe(series, i, default=0.0):
        try:
            v = float(series.iloc[i])
            return v if not np.isnan(v) else default
        except Exception:
            return default

    for i in range(n):
        rsi        = _safe(rsi_s, i, 50.0)
        macd_line  = _safe(macd_df.iloc[:, 0], i) if macd_df is not None else 0.0
        macd_hist  = _safe(macd_df.iloc[:, 1], i) if macd_df is not None else 0.0
        macd_sig   = _safe(macd_df.iloc[:, 2], i) if macd_df is not None else 0.0
        macd_hist_prev = _safe(macd_df.iloc[:, 1], i - 1) if (macd_df is not None and i > 0) else macd_hist

        bb_lower   = _safe(bb_df.iloc[:, 0], i) if bb_df is not None else float(df["Close"].iloc[i]) * 0.97
        bb_mid     = _safe(bb_df.iloc[:, 1], i) if bb_df is not None else float(df["Close"].iloc[i])
        bb_upper   = _safe(bb_df.iloc[:, 2], i) if bb_df is not None else float(df["Close"].iloc[i]) * 1.03

        ema20      = _safe(ema20_s, i, float(df["Close"].iloc[i]))
        ema50      = _safe(ema50_s, i, float(df["Close"].iloc[i]))
        sma200_val = _safe(sma200_s, i, 0.0)
        sma200     = sma200_val if sma200_val > 0 else None

        atr        = _safe(atr_s, i, float(df["Close"].iloc[i]) * 0.02)
        atr_5ago   = _safe(atr_s, i - 5, atr) if i >= 5 else atr
        obv        = _safe(obv_s, i, 0.0)

        stoch_k    = _safe(stoch_df.iloc[:, 0], i, 50.0) if stoch_df is not None else 50.0
        stoch_d    = _safe(stoch_df.iloc[:, 1], i, 50.0) if stoch_df is not None else 50.0
        stoch_k_prev = _safe(stoch_df.iloc[:, 0], i - 1, stoch_k) if (stoch_df is not None and i > 0) else stoch_k

        v_sma      = _safe(vol_sma, i, 1.0)
        v_cur      = float(df["Volume"].iloc[i])

        bb_bw      = (bb_upper - bb_lower) / bb_mid if bb_mid > 0 else 0
        bb_squeeze = bb_bw < 0.08
        vol_spike  = v_cur > 1.5 * v_sma if v_sma > 0 else False
        above_200  = (sma200 is not None) and (float(df["Close"].iloc[i]) > sma200)

        # Golden/death cross: look back up to 5 bars
        golden = death = False
        if i >= 6:
            for j in range(max(1, i - 4), i + 1):
                e20_now  = _safe(ema20_s, j, 0)
                e50_now  = _safe(ema50_s, j, 0)
                e20_prev = _safe(ema20_s, j - 1, 0)
                e50_prev = _safe(ema50_s, j - 1, 0)
                if e20_now > e50_now and e20_prev <= e50_prev:
                    golden = True
                if e20_now < e50_now and e20_prev >= e50_prev:
                    death = True

        ind = IndicatorBundle(
            rsi=rsi, macd_line=macd_line, macd_signal=macd_sig,
            macd_histogram=macd_hist, bb_upper=bb_upper, bb_mid=bb_mid,
            bb_lower=bb_lower, ema20=ema20, ema50=ema50, sma200=sma200,
            atr=atr, obv=obv, stoch_k=stoch_k, stoch_d=stoch_d,
            volume_sma20=v_sma, current_volume=v_cur,
            golden_cross=golden, death_cross=death,
            bb_squeeze=bb_squeeze, volume_spike=vol_spike, above_200_sma=above_200,
            macd_histogram_prev=macd_hist_prev, stoch_k_prev=stoch_k_prev,
            atr_5d_ago=atr_5ago,
        )

        scores.append(stat.score(ind).total_score)

    return scores
