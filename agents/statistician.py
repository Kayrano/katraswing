"""
Expert Financial Statistician Agent
Scores all technical indicators and computes a 0-100 trade score.
"""

import math
from models.report import IndicatorBundle, ComponentScores, ScoreResult


class StatisticianAgent:
    """
    Expert Financial Statistician.
    Converts indicator signals to a weighted trade score (0-100).
    """

    # Weights must sum to 1.0
    WEIGHTS = {
        "rsi":        0.15,
        "macd":       0.15,
        "bollinger":  0.10,
        "trend":      0.20,
        "volume":     0.10,
        "atr":        0.10,
        "stochastic": 0.10,
        "pattern":    0.10,
    }

    def score(self, indicators: IndicatorBundle) -> ScoreResult:
        components = ComponentScores(
            rsi=self._score_rsi(indicators),
            macd=self._score_macd(indicators),
            bollinger=self._score_bollinger(indicators),
            trend=self._score_trend(indicators),
            volume=self._score_volume(indicators),
            atr_momentum=self._score_atr(indicators),
            stochastic=self._score_stochastic(indicators),
            pattern=self._score_pattern(indicators),
        )

        raw = (
            components.rsi        * self.WEIGHTS["rsi"]        +
            components.macd       * self.WEIGHTS["macd"]       +
            components.bollinger  * self.WEIGHTS["bollinger"]  +
            components.trend      * self.WEIGHTS["trend"]      +
            components.volume     * self.WEIGHTS["volume"]     +
            components.atr_momentum * self.WEIGHTS["atr"]      +
            components.stochastic * self.WEIGHTS["stochastic"] +
            components.pattern    * self.WEIGHTS["pattern"]
        )

        total_score = round(raw * 10, 1)   # 0-10 raw → 0-100
        total_score = max(0.0, min(100.0, total_score))

        signal_label = self._label(total_score)
        win_prob = self._win_probability(total_score)
        ev = self._expected_value(win_prob)

        return ScoreResult(
            total_score=total_score,
            signal_label=signal_label,
            component_scores=components,
            win_probability=round(win_prob, 3),
            expected_value=round(ev, 2),
        )

    # ── Sub-scorers (each returns 0.0 – 10.0) ────────────────────────────────

    def _score_rsi(self, ind: IndicatorBundle) -> float:
        rsi = ind.rsi
        if rsi < 20:   return 9.5
        if rsi < 30:   return 8.0
        if rsi < 40:   return 6.5
        if rsi < 55:   return 5.0
        if rsi < 65:   return 4.0
        if rsi < 75:   return 2.5
        return 1.0

    def _score_macd(self, ind: IndicatorBundle) -> float:
        hist = ind.macd_histogram
        hist_prev = ind.macd_histogram_prev

        if hist > 0 and hist > hist_prev:
            score = 8.5
        elif hist > 0 and hist <= hist_prev:
            score = 5.5
        elif hist < 0 and hist > hist_prev:
            score = 4.5
        else:
            score = 1.5

        # Signal line cross bonus
        macd_crossed_up = hist > 0 and hist_prev <= 0
        macd_crossed_down = hist < 0 and hist_prev >= 0

        if macd_crossed_up:
            score = min(10.0, score + 1.5)
        elif macd_crossed_down:
            score = max(0.0, score - 1.5)

        return score

    def _score_bollinger(self, ind: IndicatorBundle) -> float:
        close = ind.bb_mid  # Use mid as proxy for current close in scoring
        # We use actual price via ema20 as close approximation
        close = ind.ema20

        bw = ind.bb_upper - ind.bb_lower
        if bw == 0:
            return 5.0

        if close < ind.bb_lower:
            return 9.0
        if close < ind.bb_lower + 0.25 * bw:
            return 7.0
        if close < ind.bb_mid:
            return 5.5
        if close == ind.bb_mid:
            return 5.0
        if close <= ind.bb_upper - 0.25 * bw:
            return 4.5
        if close <= ind.bb_upper:
            return 3.0
        return 1.0

    def _score_trend(self, ind: IndicatorBundle) -> float:
        close = ind.close if ind.close > 0 else ind.ema20
        points = 0
        if close > ind.ema20:
            points += 1
        if ind.ema20 > ind.ema50:
            points += 1
        if ind.sma200 is not None and close > ind.sma200:
            points += 1
            return (points / 3) * 10
        # If no SMA200 data, score out of 2
        return (points / 2) * 10

    def _score_volume(self, ind: IndicatorBundle) -> float:
        if ind.volume_sma20 == 0:
            return 5.0
        ratio = ind.current_volume / ind.volume_sma20
        if ratio > 2.0:   return 9.0
        if ratio > 1.5:   return 7.5
        if ratio > 1.0:   return 5.0
        if ratio > 0.75:  return 3.5
        return 2.0

    def _score_atr(self, ind: IndicatorBundle) -> float:
        if ind.atr_5d_ago == 0:
            return 5.0
        change = (ind.atr - ind.atr_5d_ago) / ind.atr_5d_ago
        if change > 0.15:   return 7.5
        if change > 0.0:    return 6.0
        if change > -0.10:  return 5.0
        return 3.5

    def _score_stochastic(self, ind: IndicatorBundle) -> float:
        k = ind.stoch_k
        d = ind.stoch_d
        k_prev = ind.stoch_k_prev

        crossing_up = k > d and k_prev <= d
        crossing_down = k < d and k_prev >= d

        if k < 20 and crossing_up:   return 9.0
        if k < 20:                    return 7.5
        if k < 40:                    return 6.0
        if k < 60:                    return 5.0
        if k < 80:                    return 4.0
        if k >= 80 and crossing_down: return 1.5
        return 2.5

    def _score_pattern(self, ind: IndicatorBundle) -> float:
        score = 3.5  # pessimistic baseline — no detected pattern = no edge
        if ind.golden_cross:     score += 2.0
        if ind.death_cross:      score -= 2.0
        if ind.above_200_sma:    score += 1.0
        else:                    score -= 1.0
        if ind.volume_spike:     score += 1.0
        if ind.bb_squeeze:       score += 0.5
        return max(0.0, min(10.0, score))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _label(self, score: float) -> str:
        if score >= 80:  return "STRONG BUY"
        if score >= 65:  return "BUY"
        if score >= 50:  return "WEAK BUY"
        if score >= 35:  return "NEUTRAL"
        if score >= 20:  return "WEAK SELL"
        return "STRONG SELL"

    def _win_probability(self, score: float) -> float:
        """
        Logistic transformation: maps 0-100 score to 15%-85% win probability.
        Score=50 → ~45% (realistic swing trade base rate).
        Steeper slope (-3.0) gives more differentiation in the 40-70 range.
        """
        x = (score - 50) / 25.0
        raw = 1 / (1 + math.exp(-3.0 * x))
        return 0.15 + (raw * 0.70)

    def _expected_value(self, win_prob: float) -> float:
        """
        Expected value per $100 risked at 1:2 R:R.
        EV = (win_prob × 200) - ((1 - win_prob) × 100)
        """
        return (win_prob * 200) - ((1 - win_prob) * 100)
