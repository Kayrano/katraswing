"""
CAN SLIM Agent
Scores a stock against William O'Neil's 7-criteria CAN SLIM methodology.
All data fetched via yfinance — no new dependencies required.

Criteria weights:
  C  Current Quarterly EPS Growth     20%
  A  Annual Earnings Growth           20%
  N  Near 52-Week High / Breakout     10%
  S  Supply & Demand (Acc/Dist)       15%
  L  Leader vs. Laggard (RS vs SPY)  15%
  I  Institutional Sponsorship        10%
  M  Market Direction (SPY trend)     10%
"""

import yfinance as yf
import pandas as pd
import numpy as np
from models.report import CanSlimLetterScore, CanSlimResult


WEIGHTS = {"C": 0.20, "A": 0.20, "N": 0.10, "S": 0.15, "L": 0.15, "I": 0.10, "M": 0.10}


class CanSlimAgent:
    """Evaluates a stock against the CAN SLIM framework and returns a CanSlimResult."""

    def analyze(self, ticker: str, df: pd.DataFrame) -> CanSlimResult:
        t = yf.Ticker(ticker)

        # Fetch SPY once — used for both L and M criteria
        spy_df = self._fetch_spy()

        letters = [
            self._score_c(t),
            self._score_a(t),
            self._score_n(df),
            self._score_s(df),
            self._score_l(df, spy_df),
            self._score_i(t),
            self._score_m(spy_df),
        ]

        overall = round(sum(ls.score * WEIGHTS[ls.letter] for ls in letters), 1)
        overall = max(0.0, min(100.0, overall))
        passed = sum(1 for ls in letters if ls.score >= 60)

        if overall >= 80:
            rec = "IDEAL"
        elif overall >= 65:
            rec = "STRONG"
        elif overall >= 50:
            rec = "ACCEPTABLE"
        else:
            rec = "AVOID"

        return CanSlimResult(
            overall_score=overall,
            letters=letters,
            recommendation=rec,
            criteria_passed=passed,
        )

    # ── C — Current Quarterly EPS Growth ─────────────────────────────────────

    def _score_c(self, t: yf.Ticker) -> CanSlimLetterScore:
        try:
            qf = t.quarterly_financials
            if qf is None or qf.empty:
                raise ValueError("no data")

            # Look for net income row (case-insensitive)
            row_key = next(
                (k for k in qf.index if "net income" in k.lower()),
                None,
            )
            if row_key is None:
                raise ValueError("no net income row")

            ni = qf.loc[row_key].dropna()
            if len(ni) < 5:
                raise ValueError("not enough quarters")

            # Most recent quarter vs. same quarter prior year (offset 4)
            q0 = float(ni.iloc[0])
            q4 = float(ni.iloc[4])

            if q4 == 0:
                raise ValueError("prior year quarter is zero")

            growth = (q0 - q4) / abs(q4) * 100

            if growth >= 40:
                score, label = 95, "Strong"
            elif growth >= 25:
                score, label = 80, "Pass"
            elif growth >= 10:
                score, label = 65, "Weak"
            elif growth >= 0:
                score, label = 45, "Weak"
            else:
                score, label = 15, "Fail"

            detail = f"Quarterly net income: {growth:+.1f}% vs. same quarter last year"
            return CanSlimLetterScore("C", "Current Earnings", score, label, detail)

        except Exception:
            return CanSlimLetterScore("C", "Current Earnings", 50, "N/A",
                                      "Quarterly earnings data unavailable")

    # ── A — Annual Earnings Growth ────────────────────────────────────────────

    def _score_a(self, t: yf.Ticker) -> CanSlimLetterScore:
        try:
            af = t.financials
            if af is None or af.empty:
                raise ValueError("no data")

            row_key = next(
                (k for k in af.index if "net income" in k.lower()),
                None,
            )
            if row_key is None:
                raise ValueError("no net income row")

            ni = af.loc[row_key].dropna()
            if len(ni) < 3:
                raise ValueError("need at least 3 annual figures")

            # Sorted newest→oldest
            values = [float(v) for v in ni.iloc[:4]]
            growths = []
            for i in range(len(values) - 1):
                prev = values[i + 1]
                if prev == 0:
                    continue
                growths.append((values[i] - prev) / abs(prev) * 100)

            if not growths:
                raise ValueError("cannot compute growth")

            years_above_25 = sum(1 for g in growths[:3] if g >= 25)

            if years_above_25 == 3:
                score, label = 95, "Strong"
            elif years_above_25 == 2:
                score, label = 75, "Pass"
            elif years_above_25 == 1:
                score, label = 50, "Weak"
            else:
                score, label = 20, "Fail"

            growth_str = ", ".join(f"{g:+.0f}%" for g in growths[:3])
            detail = f"Annual net income growth (3 yrs): {growth_str}"
            return CanSlimLetterScore("A", "Annual Growth", score, label, detail)

        except Exception:
            return CanSlimLetterScore("A", "Annual Growth", 50, "N/A",
                                      "Annual earnings data unavailable")

    # ── N — Near 52-Week High ─────────────────────────────────────────────────

    def _score_n(self, df: pd.DataFrame) -> CanSlimLetterScore:
        try:
            high_52w = df["High"].tail(252).max()
            current = float(df["Close"].iloc[-1])
            pct_below = (high_52w - current) / high_52w * 100

            if pct_below <= 5:
                score, label = 90, "Strong"
            elif pct_below <= 15:
                score, label = 70, "Pass"
            elif pct_below <= 30:
                score, label = 45, "Weak"
            else:
                score, label = 20, "Fail"

            detail = f"{pct_below:.1f}% below 52-week high (${high_52w:.2f})"
            return CanSlimLetterScore("N", "Near New High", score, label, detail)

        except Exception:
            return CanSlimLetterScore("N", "Near New High", 50, "N/A",
                                      "Price high data unavailable")

    # ── S — Supply & Demand (Accumulation/Distribution) ──────────────────────

    def _score_s(self, df: pd.DataFrame) -> CanSlimLetterScore:
        try:
            recent = df.tail(20).copy()
            vol_sma = recent["Volume"].mean()

            acc_days = int(((recent["Close"] > recent["Open"]) & (recent["Volume"] > vol_sma)).sum())
            dist_days = int(((recent["Close"] < recent["Open"]) & (recent["Volume"] > vol_sma)).sum())
            total_above_avg = acc_days + dist_days

            if total_above_avg == 0:
                score, label = 50, "N/A"
                detail = "No above-average volume days in last 20 sessions"
            else:
                ratio = acc_days / total_above_avg
                if ratio >= 0.75:
                    score, label = 90, "Strong"
                elif ratio >= 0.60:
                    score, label = 72, "Pass"
                elif ratio >= 0.45:
                    score, label = 50, "Weak"
                else:
                    score, label = 25, "Fail"

                detail = (f"{acc_days} accumulation vs {dist_days} distribution days "
                          f"(above-avg volume, last 20 sessions)")

            return CanSlimLetterScore("S", "Supply & Demand", score, label, detail)

        except Exception:
            return CanSlimLetterScore("S", "Supply & Demand", 50, "N/A",
                                      "Volume data unavailable")

    # ── L — Leader vs. Laggard (Relative Strength vs. SPY) ───────────────────

    def _score_l(self, df: pd.DataFrame, spy_df: pd.DataFrame) -> CanSlimLetterScore:
        try:
            if spy_df is None or spy_df.empty:
                raise ValueError("no SPY data")

            stock_ret = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[0]) - 1) * 100
            spy_ret = (float(spy_df["Close"].iloc[-1]) / float(spy_df["Close"].iloc[0]) - 1) * 100
            rel = stock_ret - spy_ret

            if rel >= 20:
                score, label = 90, "Strong"
            elif rel >= 5:
                score, label = 75, "Pass"
            elif rel >= -5:
                score, label = 50, "Weak"
            elif rel >= -20:
                score, label = 30, "Fail"
            else:
                score, label = 10, "Fail"

            arrow = "+" if rel >= 0 else ""
            detail = (f"12-month return vs SPY: {arrow}{rel:.1f}% "
                      f"(stock {stock_ret:+.1f}%, SPY {spy_ret:+.1f}%)")
            return CanSlimLetterScore("L", "Leader / Laggard", score, label, detail)

        except Exception:
            return CanSlimLetterScore("L", "Leader / Laggard", 50, "N/A",
                                      "Relative strength vs. SPY unavailable")

    # ── I — Institutional Sponsorship ────────────────────────────────────────

    def _score_i(self, t: yf.Ticker) -> CanSlimLetterScore:
        try:
            info = t.info
            inst_pct = info.get("heldPercentInstitutions") or 0
            if inst_pct == 0:
                raise ValueError("no inst data")
            inst_pct_display = inst_pct * 100

            if inst_pct >= 0.70:
                score, label = 90, "Strong"
            elif inst_pct >= 0.50:
                score, label = 75, "Pass"
            elif inst_pct >= 0.30:
                score, label = 60, "Weak"
            else:
                score, label = 40, "Fail"

            detail = f"Institutional ownership: {inst_pct_display:.1f}%"
            return CanSlimLetterScore("I", "Institutional Sponsorship", score, label, detail)

        except Exception:
            return CanSlimLetterScore("I", "Institutional Sponsorship", 50, "N/A",
                                      "Institutional ownership data unavailable")

    # ── M — Market Direction (SPY trend) ─────────────────────────────────────

    def _score_m(self, spy_df: pd.DataFrame) -> CanSlimLetterScore:
        try:
            if spy_df is None or len(spy_df) < 50:
                raise ValueError("not enough SPY data")

            close = spy_df["Close"]
            sma50 = float(close.rolling(50).mean().iloc[-1])
            sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
            current = float(close.iloc[-1])

            above_50 = current > sma50
            above_200 = (sma200 is None) or (current > sma200)

            if above_50 and above_200:
                score, label = 85, "Strong"
                detail = "SPY above SMA50 and SMA200 — confirmed uptrend"
            elif above_200:
                score, label = 60, "Weak"
                detail = "SPY above SMA200 but below SMA50 — weakening trend"
            else:
                score, label = 25, "Fail"
                detail = "SPY below SMA200 — bear market conditions"

            return CanSlimLetterScore("M", "Market Direction", score, label, detail)

        except Exception:
            return CanSlimLetterScore("M", "Market Direction", 50, "N/A",
                                      "Market direction data unavailable")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _fetch_spy(self) -> pd.DataFrame:
        try:
            spy = yf.Ticker("SPY").history(period="1y", interval="1d", auto_adjust=True)
            spy = spy[["Open", "High", "Low", "Close", "Volume"]].dropna()
            return spy
        except Exception:
            return pd.DataFrame()
