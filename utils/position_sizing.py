"""
Position Sizing Calculator
Given account size, risk %, entry and stop loss — computes exact share count,
dollar risk, dollar reward, and position value.
"""

from dataclasses import dataclass


@dataclass
class PositionSize:
    shares:          int
    position_value:  float    # shares × entry
    dollar_risk:     float    # shares × (entry - stop_loss)
    dollar_reward:   float    # shares × (take_profit - entry)
    risk_pct_actual: float    # actual % of account risked
    rr_ratio:        float
    account_size:    float
    risk_pct_input:  float


def calculate(
    account_size: float,
    risk_pct: float,          # e.g. 1.0 for 1%
    entry: float,
    stop_loss: float,
    take_profit: float,
) -> PositionSize:
    """
    Calculate position size based on fixed-dollar risk model.
    Rounds down to whole shares (no fractional shares).
    """
    if entry <= 0 or stop_loss <= 0 or account_size <= 0:
        return PositionSize(0, 0, 0, 0, 0, 0, account_size, risk_pct)

    max_dollar_risk = account_size * (risk_pct / 100.0)
    risk_per_share  = abs(entry - stop_loss)

    if risk_per_share <= 0:
        return PositionSize(0, 0, 0, 0, 0, 0, account_size, risk_pct)

    shares = int(max_dollar_risk / risk_per_share)   # floor — never over-risk

    if shares < 1:
        shares = 1

    dollar_risk     = shares * risk_per_share
    dollar_reward   = shares * abs(take_profit - entry)
    position_value  = shares * entry
    risk_pct_actual = (dollar_risk / account_size) * 100
    rr_ratio        = dollar_reward / dollar_risk if dollar_risk > 0 else 0

    return PositionSize(
        shares=shares,
        position_value=round(position_value, 2),
        dollar_risk=round(dollar_risk, 2),
        dollar_reward=round(dollar_reward, 2),
        risk_pct_actual=round(risk_pct_actual, 3),
        rr_ratio=round(rr_ratio, 2),
        account_size=account_size,
        risk_pct_input=risk_pct,
    )
