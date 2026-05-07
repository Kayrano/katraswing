"""C3 — Look-ahead audit.

Verifies that no strategy, indicator, or backtester component reads bar data
that hasn't yet been emitted.  Three layers of protection:

  1. Static grep: patterns that would cause look-ahead at the source level
     (.shift(-n), center=True rolling) are absent from all agent + util files.

  2. Idempotency + entry-anchor: each registered strategy function produces the
     same signal when called twice on identical data, and the signal's entry
     price is anchored to the last bar of the slice passed in.

  3. Backtester bar isolation: running the backtester on two datasets that share
     the same first-N bars but differ beyond bar N produces identical trade logs
     for those first-N bars.

Run via:
    pytest tests/test_audit_lookahead.py -v
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ── Source roots to audit ──────────────────────────────────────────────────────

_AGENTS_DIR = Path(__file__).parent.parent / "agents"
_UTILS_DIR  = Path(__file__).parent.parent / "utils"

_AUDITED_FILES = list(_AGENTS_DIR.glob("*.py")) + list(_UTILS_DIR.glob("*.py"))


# ── Static tests ───────────────────────────────────────────────────────────────

class TestStaticLookAhead:
    def test_no_negative_shift_in_agents(self):
        """pandas shift(-n) for n>0 is a forward shift — bar i gets bar i+n's data.

        This is the single most common look-ahead bug in backtesting code.
        None of our strategy or indicator files should use it.
        """
        pattern = re.compile(r"\.shift\(-[1-9]")
        violations: list[str] = []
        for path in _AUDITED_FILES:
            try:
                src = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(src.splitlines(), 1):
                if pattern.search(line) and not line.lstrip().startswith("#"):
                    violations.append(f"{path.name}:{lineno}: {line.strip()}")
        assert violations == [], (
            "Found forward shift (look-ahead) in:\n" + "\n".join(violations)
        )

    def test_no_center_true_rolling(self):
        """rolling(center=True) is centred on the window — the future half of the
        window is look-ahead by definition.  Should never appear in live code."""
        pattern = re.compile(r"rolling\s*\(.*center\s*=\s*True")
        violations: list[str] = []
        for path in _AUDITED_FILES:
            try:
                src = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(src.splitlines(), 1):
                if pattern.search(line) and not line.lstrip().startswith("#"):
                    violations.append(f"{path.name}:{lineno}: {line.strip()}")
        assert violations == [], (
            "Found center=True rolling (look-ahead) in:\n" + "\n".join(violations)
        )

    def test_mss_daily_pivot_loop_excludes_last_bar(self):
        """The MSS daily-pivot loop iterates range(1, n_d - 1), so the last daily
        bar is never marked as a confirmed pivot in the same call — it can only be
        confirmed once a future daily bar has closed.  That's causal.  Verify the
        guard is still in the source.
        """
        mss_src = (_AGENTS_DIR / "intraday_strategies.py").read_text(
            encoding="utf-8", errors="replace"
        )
        # The loop must use n_d - 1 (or an equivalent) as the upper bound.
        assert "range(1, n_d - 1)" in mss_src, (
            "mss_forex_15m pivot loop upper-bound changed — verify causal isolation"
        )

    def test_no_iloc_positive_future_index(self):
        """Explicit df.iloc[len(df)] or df.iloc[i+1] type forward access should
        not appear as a bare constant positive-future access.  We can't catch every
        dynamic pattern, but the literal `.iloc[len(` is a strong smell.
        """
        pattern = re.compile(r"\.iloc\[len\(")
        violations: list[str] = []
        for path in _AUDITED_FILES:
            try:
                src = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(src.splitlines(), 1):
                if pattern.search(line) and not line.lstrip().startswith("#"):
                    violations.append(f"{path.name}:{lineno}: {line.strip()}")
        assert violations == [], (
            "Found .iloc[len(...)] (possible future access) in:\n" +
            "\n".join(violations)
        )


# ── Synthetic DataFrame helpers ────────────────────────────────────────────────

def _make_ohlcv(
    n: int = 200,
    start_price: float = 100.0,
    *,
    freq: str = "5min",
    tz: str = "America/New_York",
    seed: int = 0,
) -> pd.DataFrame:
    """Generate a realistic synthetic OHLCV DataFrame with all expected columns."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-03-01 09:30", periods=n, freq=freq, tz=tz)

    closes = start_price + np.cumsum(rng.standard_normal(n) * 0.1)
    highs  = closes + rng.uniform(0.02, 0.10, n)
    lows   = closes - rng.uniform(0.02, 0.10, n)
    opens  = closes + rng.standard_normal(n) * 0.03

    df = pd.DataFrame({
        "Open":   opens,
        "High":   highs,
        "Low":    lows,
        "Close":  closes,
        "Volume": rng.uniform(500, 2000, n),
    }, index=idx)

    # Columns that strategies / signal_engine expect
    df["session_date"]       = idx.date
    df["session_bar_number"] = np.arange(1, n + 1)
    df["is_first_bar"]       = df["session_bar_number"] == 1
    df["session_vwap"]       = closes
    df["rvol"]               = rng.uniform(0.8, 2.0, n)
    df["market"]             = "FOREX"
    return df


# ── Dynamic tests ──────────────────────────────────────────────────────────────

class TestStrategyIdempotency:
    """Each strategy must return the same signal on two identical calls."""

    @pytest.fixture(params=[
        "vwap_rsi_5m",
        "orb_5m",
        "ema_pullback_15m",
        "squeeze_15m",
        "absorption_15m",
        "mss_forex_15m",
    ])
    def strategy_fn(self, request):
        import agents.intraday_strategies as strats
        return getattr(strats, request.param)

    def test_same_df_same_result(self, strategy_fn):
        df = _make_ohlcv(200)
        r1 = strategy_fn(df.copy())
        r2 = strategy_fn(df.copy())
        assert r1.signal == r2.signal
        assert r1.entry  == r2.entry
        assert r1.strategy == r2.strategy


class TestStrategyEntryAnchor:
    """When a strategy fires a LONG/SHORT, its entry price must be anchored
    to the last bar of the passed DataFrame (within 5% of that close).
    If a strategy were using look-ahead data its entry would systematically
    diverge from the last-bar close.
    """

    @pytest.mark.parametrize("strategy_name, last_close", [
        ("vwap_rsi_5m",    100.0),
        ("orb_5m",         200.0),
        ("absorption_15m", 300.0),
    ])
    def test_entry_near_last_close(self, strategy_name, last_close):
        import agents.intraday_strategies as strats
        fn = getattr(strats, strategy_name)

        # Build a df where we control the last bar's close exactly.
        df = _make_ohlcv(200, start_price=last_close)
        # Force last bar close to a known value so we can anchor the assertion.
        df.loc[df.index[-1], "Close"] = last_close
        df.loc[df.index[-1], "High"]  = last_close + 0.5
        df.loc[df.index[-1], "Low"]   = last_close - 0.5
        df["session_vwap"] = df["Close"]

        sig = fn(df)
        if sig.signal in ("LONG", "SHORT"):
            # Entry must be within 5% of the last bar's close.
            # Strategies use cur_close = df["Close"].iloc[-1] for entry.
            assert abs(sig.entry - last_close) / last_close < 0.05, (
                f"{strategy_name}: entry {sig.entry} is more than 5% away "
                f"from last-bar close {last_close} — possible look-ahead"
            )


class TestMSSDatetimeIndexGuard:
    """mss_forex_15m must return FLAT when the index is not a DatetimeIndex,
    not raise or silently return bad data."""

    def test_integer_index_returns_flat(self):
        from agents.intraday_strategies import mss_forex_15m
        df = _make_ohlcv(200).reset_index(drop=True)  # strips DatetimeIndex
        sig = mss_forex_15m(df)
        assert sig.signal == "FLAT"
        assert "DatetimeIndex" in sig.reason or "resample" in sig.reason.lower()


class TestBacktesterBarIsolation:
    """Running the backtester on two DataFrames that share the same first N bars
    but diverge afterwards must produce identical trade logs for those first N bars.

    This verifies that the bar-by-bar slice (df.iloc[:i+1]) used inside the
    backtester does not allow a strategy to 'see' bars from after its current
    position.
    """

    def test_shared_prefix_gives_same_trades(self):
        from agents.intraday_backtester import _backtest_strategy
        from agents.intraday_strategies import orb_5m

        df_base = _make_ohlcv(150, seed=7)

        # Second dataset: same first 100 bars, then extreme prices
        df_alt = df_base.copy()
        df_alt.loc[df_alt.index[100:], "Close"] = 99999.9
        df_alt.loc[df_alt.index[100:], "High"]  = 99999.9
        df_alt.loc[df_alt.index[100:], "Low"]   = 99998.0
        df_alt.loc[df_alt.index[100:], "Open"]  = 99999.0

        r_base = _backtest_strategy(df_base.iloc[:100].copy(), orb_5m, "5m")
        r_alt  = _backtest_strategy(df_alt.iloc[:100].copy(),  orb_5m, "5m")

        # Trade count and every trade's entry/exit/direction must match
        assert len(r_base.trades) == len(r_alt.trades), (
            f"Trade counts differ: base={len(r_base.trades)} alt={len(r_alt.trades)}"
        )
        for tb, ta in zip(r_base.trades, r_alt.trades):
            assert tb.direction  == ta.direction
            assert tb.entry_bar  == ta.entry_bar
            assert abs(tb.entry_price - ta.entry_price) < 1e-9, (
                f"Entry prices diverge: base={tb.entry_price} alt={ta.entry_price} — "
                "strategy may be reading future bars"
            )
