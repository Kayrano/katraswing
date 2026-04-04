from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import pandas as pd


@dataclass
class IndicatorBundle:
    rsi: float
    macd_line: float
    macd_signal: float
    macd_histogram: float
    bb_upper: float
    bb_mid: float
    bb_lower: float
    ema20: float
    ema50: float
    sma200: Optional[float]
    atr: float
    obv: float
    stoch_k: float
    stoch_d: float
    volume_sma20: float
    current_volume: float
    golden_cross: bool
    death_cross: bool
    bb_squeeze: bool
    volume_spike: bool
    above_200_sma: bool
    # Actual closing price (for trend scoring)
    close: float = 0.0
    # Previous values for direction detection
    macd_histogram_prev: float = 0.0
    stoch_k_prev: float = 50.0
    atr_5d_ago: float = 0.0
    obv_10d_ago: float = 0.0    # OBV 10 days ago — used for trend direction
    rsi_divergence_bearish: bool = False  # price higher high + RSI lower high
    rsi_divergence_bullish: bool = False  # price lower low + RSI higher low


@dataclass
class TradeSetup:
    direction: str          # "LONG" | "SHORT" | "NO TRADE"
    entry: float
    stop_loss: float
    take_profit: float
    risk_amount: float
    reward_amount: float
    rr_ratio: float         # always 2.0
    atr_used: float
    stop_pct: float
    target_pct: float


@dataclass
class ComponentScores:
    rsi: float
    macd: float
    bollinger: float
    trend: float
    volume: float
    atr_momentum: float
    stochastic: float
    pattern: float


@dataclass
class ScoreResult:
    total_score: float
    signal_label: str
    component_scores: ComponentScores
    win_probability: float
    expected_value: float
    regime: str = "NEUTRAL"   # TRENDING | CONSOLIDATING | EXTENDED | VOLATILE | NEUTRAL


@dataclass
class MTFResult:
    """Multi-timeframe analysis: daily + weekly scores and agreement."""
    daily_score: float
    weekly_score: float
    daily_label: str
    weekly_label: str
    agreement: bool          # both timeframes point same direction
    agreement_direction: str # "BULLISH" | "BEARISH" | "MIXED"
    combined_score: float    # weighted: 60% daily, 40% weekly
    weekly_indicators: Optional["IndicatorBundle"] = None


@dataclass
class CanSlimLetterScore:
    letter: str       # "C", "A", "N", "S", "L", "I", "M"
    name: str         # full criterion name
    score: float      # 0-100
    label: str        # "Strong", "Pass", "Weak", "N/A"
    detail: str       # one-line explanation for the UI


@dataclass
class CanSlimResult:
    overall_score: float
    letters: list          # list[CanSlimLetterScore], exactly 7
    recommendation: str    # "IDEAL" | "STRONG" | "ACCEPTABLE" | "AVOID"
    criteria_passed: int   # letters with score >= 60


@dataclass
class ReportData:
    ticker: str
    company_name: str
    sector: str
    market_cap: float
    current_price: float
    price_change_pct: float
    df: pd.DataFrame
    indicators: IndicatorBundle
    trade_setup: TradeSetup
    score: ScoreResult
    mtf: Optional[MTFResult] = None
    canslim: Optional[CanSlimResult] = None
    generated_at: datetime = field(default_factory=datetime.now)
    filter_notes: list = field(default_factory=list)
