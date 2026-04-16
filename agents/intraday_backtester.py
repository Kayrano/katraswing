"""
Walk-forward intraday backtester for 5m and 15m strategies.
=============================================================
Tests all strategies on the last 59 calendar days of intraday data without
any look-ahead bias.  Results include per-strategy win rate, profit factor,
and the combined stats table displayed in the UI.

Design choices
──────────────
- Bar-by-bar simulation: at each bar we only have access to data up to and
  including that bar (df.iloc[:i+1]).
- Sessions are respected: a trade cannot survive overnight — any open position
  is force-closed on the last bar of the session at the session close price.
- Time stop: 20 bars max for 5m (100 min), 10 bars max for 15m (150 min).
- Slippage: 0.02% per entry/exit (market order approximation).
- Only one trade open per strategy at a time; no pyramiding.
- Only LONG signals are backtested by default (same as the H1 gate logic used
  in the live bot) because short-selling US equities intraday requires margin.
  Set long_only=False to test both directions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

import utils.ta_compat as ta
from agents.intraday_strategies import (
    IntradaySignal,
    vwap_rsi_5m,
    orb_5m,
    ema_pullback_15m,
    squeeze_15m,
    absorption_15m,
    _STRATEGIES_5M,
    _STRATEGIES_15M,
)
from data.fetcher_intraday import fetch_intraday_data


SLIPPAGE = 0.0002   # 0.02% one-way


@dataclass
class BacktestTrade:
    strategy:   str
    direction:  str
    entry_bar:  int
    exit_bar:   int
    entry_price: float
    exit_price:  float
    stop_loss:   float
    take_profit: float
    outcome:    str    # "TP" | "SL" | "TIME" | "SESSION_END"
    pnl_pct:    float  # percent gain/loss on the trade


@dataclass
class StrategyBacktestResult:
    strategy:      str
    timeframe:     str
    total_trades:  int
    wins:          int
    losses:        int
    win_rate:      float   # 0.0 – 1.0
    avg_win_pct:   float
    avg_loss_pct:  float
    profit_factor: float
    total_return_pct: float
    max_drawdown_pct: float
    trades:        list[BacktestTrade] = field(default_factory=list)
    meets_threshold: bool = False   # True when win_rate >= MIN_WIN_RATE


@dataclass
class IntradayBacktestSummary:
    ticker:     str
    timeframe:  str
    period_days: int
    results:    list[StrategyBacktestResult]
    best_strategy: Optional[str]
    overall_win_rate: float
    all_pass_threshold: bool   # True if ALL strategies meet 60% threshold
    generated_at: str = ""


MIN_WIN_RATE = 0.60   # strategies must hit this to be "validated"


# ── Single-strategy walk-forward simulation ───────────────────────────────────

def _backtest_strategy(
    df: pd.DataFrame,
    strategy_fn,
    timeframe: str,
    long_only: bool = True,
) -> StrategyBacktestResult:
    """
    Walk-forward simulation for one strategy on the full intraday DataFrame.
    Iterates bar by bar; at each bar calls the strategy on df.iloc[:i+1].
    """
    name = strategy_fn.__name__.upper()
    time_stop = 20 if timeframe == "5m" else 10

    trades: list[BacktestTrade] = []
    in_trade  = False
    entry_bar = 0
    entry_price = 0.0
    stop_loss   = 0.0
    take_profit = 0.0
    direction   = "LONG"
    entry_date  = None

    # Minimum warmup before we start generating signals
    warmup = 30

    for i in range(warmup, len(df)):
        slice_df = df.iloc[:i + 1]

        if in_trade:
            cur_high  = float(df["High"].iloc[i])
            cur_low   = float(df["Low"].iloc[i])
            cur_close = float(df["Close"].iloc[i])
            cur_date  = df["session_date"].iloc[i]
            bars_held = i - entry_bar

            # Session end force-close
            is_last_bar_of_session = (
                i + 1 >= len(df) or
                df["session_date"].iloc[i + 1] != cur_date
            )

            outcome = None
            exit_price = cur_close

            if direction == "LONG":
                if cur_low <= stop_loss:
                    outcome    = "SL"
                    exit_price = stop_loss * (1 - SLIPPAGE)
                elif cur_high >= take_profit:
                    outcome    = "TP"
                    exit_price = take_profit * (1 - SLIPPAGE)
                elif bars_held >= time_stop:
                    outcome    = "TIME"
                    exit_price = cur_close * (1 - SLIPPAGE)
                elif is_last_bar_of_session:
                    outcome    = "SESSION_END"
                    exit_price = cur_close * (1 - SLIPPAGE)
            else:  # SHORT
                if cur_high >= stop_loss:
                    outcome    = "SL"
                    exit_price = stop_loss * (1 + SLIPPAGE)
                elif cur_low <= take_profit:
                    outcome    = "TP"
                    exit_price = take_profit * (1 + SLIPPAGE)
                elif bars_held >= time_stop:
                    outcome    = "TIME"
                    exit_price = cur_close * (1 + SLIPPAGE)
                elif is_last_bar_of_session:
                    outcome    = "SESSION_END"
                    exit_price = cur_close * (1 + SLIPPAGE)

            if outcome:
                if direction == "LONG":
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price * 100

                trades.append(BacktestTrade(
                    strategy=name, direction=direction,
                    entry_bar=entry_bar, exit_bar=i,
                    entry_price=entry_price, exit_price=exit_price,
                    stop_loss=stop_loss, take_profit=take_profit,
                    outcome=outcome, pnl_pct=round(pnl_pct, 4),
                ))
                in_trade = False
            continue   # don't look for new entry while in trade

        # Look for new entry signal
        try:
            sig: IntradaySignal = strategy_fn(slice_df)
        except Exception:
            continue

        if sig.signal == "FLAT":
            continue
        if long_only and sig.signal != "LONG":
            continue
        if sig.entry <= 0 or sig.stop_loss <= 0 or sig.take_profit <= 0:
            continue
        # Sanity check: SL and TP must be on correct sides
        if sig.signal == "LONG" and (sig.stop_loss >= sig.entry or sig.take_profit <= sig.entry):
            continue
        if sig.signal == "SHORT" and (sig.stop_loss <= sig.entry or sig.take_profit >= sig.entry):
            continue

        in_trade    = True
        entry_bar   = i
        entry_price = sig.entry * (1 + SLIPPAGE)
        stop_loss   = sig.stop_loss
        take_profit = sig.take_profit
        direction   = sig.signal
        entry_date  = df["session_date"].iloc[i]

    # Force-close any open trade at end of data
    if in_trade and trades:
        pass   # already closed within loop on last SESSION_END

    # ── Compute statistics ────────────────────────────────────────────────────
    total = len(trades)
    if total == 0:
        return StrategyBacktestResult(
            strategy=name, timeframe=timeframe, total_trades=0,
            wins=0, losses=0, win_rate=0.0, avg_win_pct=0.0,
            avg_loss_pct=0.0, profit_factor=0.0, total_return_pct=0.0,
            max_drawdown_pct=0.0, trades=[], meets_threshold=False,
        )

    wins   = [t for t in trades if t.pnl_pct > 0]
    losses = [t for t in trades if t.pnl_pct <= 0]

    win_rate   = len(wins) / total
    avg_win    = float(np.mean([t.pnl_pct for t in wins])) if wins else 0.0
    avg_loss   = float(np.mean([t.pnl_pct for t in losses])) if losses else 0.0
    gross_win  = sum(t.pnl_pct for t in wins)
    gross_loss = abs(sum(t.pnl_pct for t in losses))
    pf         = gross_win / gross_loss if gross_loss > 0 else float("inf")
    total_ret  = sum(t.pnl_pct for t in trades)

    # Max drawdown on equity curve (simple sum of pnl_pct in sequence)
    equity = np.cumsum([t.pnl_pct for t in trades])
    peak   = np.maximum.accumulate(equity)
    dd     = equity - peak
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0

    return StrategyBacktestResult(
        strategy=name, timeframe=timeframe,
        total_trades=total,
        wins=len(wins),
        losses=len(losses),
        win_rate=round(win_rate, 4),
        avg_win_pct=round(avg_win, 4),
        avg_loss_pct=round(avg_loss, 4),
        profit_factor=round(pf, 2) if pf != float("inf") else 99.0,
        total_return_pct=round(total_ret, 2),
        max_drawdown_pct=round(max_dd, 2),
        trades=trades,
        meets_threshold=win_rate >= MIN_WIN_RATE,
    )


# ── Public entry point ────────────────────────────────────────────────────────

def run_intraday_backtest(
    ticker: str,
    timeframe: str = "15m",
    long_only: bool = True,
) -> IntradayBacktestSummary:
    """
    Run a walk-forward backtest of all intraday strategies for the given
    timeframe on the last ~59 days of data.

    Returns an IntradayBacktestSummary with per-strategy results.
    """
    from datetime import datetime

    df = fetch_intraday_data(ticker, interval=timeframe, days=59)
    sessions = df["session_date"].nunique()

    strategy_fns = _STRATEGIES_5M if timeframe == "5m" else _STRATEGIES_15M

    results: list[StrategyBacktestResult] = []
    for fn in strategy_fns:
        try:
            r = _backtest_strategy(df, fn, timeframe, long_only=long_only)
        except Exception as exc:
            name = fn.__name__.upper()
            r = StrategyBacktestResult(
                strategy=name, timeframe=timeframe, total_trades=0,
                wins=0, losses=0, win_rate=0.0, avg_win_pct=0.0,
                avg_loss_pct=0.0, profit_factor=0.0, total_return_pct=0.0,
                max_drawdown_pct=0.0, trades=[], meets_threshold=False,
            )
        results.append(r)

    # Overall combined win rate (all strategies pooled)
    all_trades  = [t for r in results for t in r.trades]
    total_count = len(all_trades)
    overall_wr  = (
        sum(1 for t in all_trades if t.pnl_pct > 0) / total_count
        if total_count > 0 else 0.0
    )

    traded = [r for r in results if r.total_trades > 0]
    best = (
        max(traded, key=lambda r: r.win_rate).strategy
        if traded else None
    )

    return IntradayBacktestSummary(
        ticker=ticker,
        timeframe=timeframe,
        period_days=sessions,
        results=results,
        best_strategy=best,
        overall_win_rate=round(overall_wr, 4),
        all_pass_threshold=all(r.meets_threshold for r in traded),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
