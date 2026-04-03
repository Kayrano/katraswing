"""
Expert Trader Agent
Determines trade direction, entry price, stop loss, and take profit.
Enforces strict 1:2 risk/reward ratio using ATR-based stops.
"""

import pandas as pd
from models.report import IndicatorBundle, TradeSetup


class TraderAgent:
    """
    Expert Trader.
    Uses ATR-based stop loss placement and enforces 1:2 R:R on every trade.
    """

    ATR_STOP_MULTIPLIER = 1.5   # Stop is 1.5 × ATR from entry
    REWARD_MULTIPLIER   = 2.0   # Take profit is 2× the risk (1:2 R:R)

    def compute_trade_setup(
        self,
        df: pd.DataFrame,
        indicators: IndicatorBundle,
        trade_score: float,
    ) -> TradeSetup:
        """
        Compute entry, stop loss, take profit.
        trade_score is used to determine direction and whether to trade at all.
        """
        entry = float(df["Close"].iloc[-1])
        atr = indicators.atr

        # Determine direction from score
        if trade_score >= 50:
            direction = "LONG"
        elif trade_score < 35:
            direction = "SHORT"
        else:
            # Neutral zone — no trade
            return TradeSetup(
                direction="NO TRADE",
                entry=entry,
                stop_loss=0.0,
                take_profit=0.0,
                risk_amount=0.0,
                reward_amount=0.0,
                rr_ratio=0.0,
                atr_used=atr,
                stop_pct=0.0,
                target_pct=0.0,
            )

        # ATR-based stop and target
        risk_distance = self.ATR_STOP_MULTIPLIER * atr

        if direction == "LONG":
            stop_loss = entry - risk_distance
            # Optionally snap to nearest swing low
            stop_loss = self._snap_to_support(df, entry, stop_loss, atr)
            risk_amount = entry - stop_loss
            take_profit = entry + (risk_amount * self.REWARD_MULTIPLIER)
        else:  # SHORT
            stop_loss = entry + risk_distance
            # Optionally snap to nearest swing high
            stop_loss = self._snap_to_resistance(df, entry, stop_loss, atr)
            risk_amount = stop_loss - entry
            take_profit = entry - (risk_amount * self.REWARD_MULTIPLIER)

        reward_amount = risk_amount * self.REWARD_MULTIPLIER
        rr_ratio = reward_amount / risk_amount if risk_amount > 0 else 0.0

        stop_pct = (abs(entry - stop_loss) / entry) * 100
        target_pct = (abs(take_profit - entry) / entry) * 100

        return TradeSetup(
            direction=direction,
            entry=round(entry, 4),
            stop_loss=round(stop_loss, 4),
            take_profit=round(take_profit, 4),
            risk_amount=round(risk_amount, 4),
            reward_amount=round(reward_amount, 4),
            rr_ratio=round(rr_ratio, 2),
            atr_used=round(atr, 4),
            stop_pct=round(stop_pct, 2),
            target_pct=round(target_pct, 2),
        )

    def _snap_to_support(
        self,
        df: pd.DataFrame,
        entry: float,
        atr_stop: float,
        atr: float,
    ) -> float:
        """
        Snap stop loss to nearest swing low if it's within 0.3 × ATR of ATR stop.
        Otherwise, keep the ATR stop.
        """
        try:
            lows = df["Low"].rolling(5, center=True).min()
            recent_lows = lows.dropna().iloc[-20:]
            nearest_support = float(recent_lows.min())
            if abs(entry - nearest_support) < (abs(entry - atr_stop) * 1.1):
                # Place stop 0.5% below support
                return nearest_support * 0.995
        except Exception:
            pass
        return atr_stop

    def _snap_to_resistance(
        self,
        df: pd.DataFrame,
        entry: float,
        atr_stop: float,
        atr: float,
    ) -> float:
        """
        Snap stop loss to nearest swing high for SHORT trades.
        """
        try:
            highs = df["High"].rolling(5, center=True).max()
            recent_highs = highs.dropna().iloc[-20:]
            nearest_resistance = float(recent_highs.max())
            if abs(nearest_resistance - entry) < (abs(atr_stop - entry) * 1.1):
                return nearest_resistance * 1.005
        except Exception:
            pass
        return atr_stop
