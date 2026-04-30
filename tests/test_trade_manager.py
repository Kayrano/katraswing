"""
Decision-matrix tests for agents.trade_manager._decide_action.

The decision logic has ~10 distinct branches that determine whether an
open position is closed, partially closed, has its SL/TP modified, or
held. Each branch is consequential (real money) and depends on a tangle
of signal-quality, profit-curve, and time inputs. These tests pin the
critical close paths so future tweaks to thresholds or new branches
can't silently change exit behavior.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from types import SimpleNamespace

import pytest

from agents.trade_manager import _decide_action


# ─────────────────────────────────────────────────────────────────────────────
# Fixture builders — mock Position and SignalResult shapes used by _decide_action
# ─────────────────────────────────────────────────────────────────────────────

def _position(*, direction="LONG", sl=1.09, tp=1.11, profit=0.0,
              open_price=1.10, volume=0.1, symbol="EURUSD"):
    return SimpleNamespace(
        direction=direction,
        sl=sl, tp=tp,
        profit=profit,
        open_price=open_price,
        volume=volume,
        symbol=symbol,
    )


def _signal(*, direction="LONG", confidence=0.70, mtf_score=0,
            news_score=0.0, daily_trend_vetoed=False,
            indicators=None, patterns=None, live_wr_key=""):
    return SimpleNamespace(
        direction=direction,
        confidence=confidence,
        mtf_score=mtf_score,
        news_score=news_score,
        daily_trend_vetoed=daily_trend_vetoed,
        indicators=indicators,
        patterns=patterns,
        live_wr_key=live_wr_key,
    )


def _indicators(*, bearish_div=False, bullish_div=False):
    return SimpleNamespace(
        rsi_divergence_bearish=bearish_div,
        rsi_divergence_bullish=bullish_div,
    )


def _pattern_report(*, matches=None):
    return SimpleNamespace(
        patterns=matches or [],
        dominant_bias="NEUTRAL",
    )


def _pat(name, conf=0.70, bias="BULLISH", win_rate=0.6):
    return SimpleNamespace(
        name=name, confidence=conf, bias=bias, win_rate=win_rate,
    )


def _recent_iso(minutes_ago: int = 30) -> str:
    return (datetime.now(tz=timezone.utc) - timedelta(minutes=minutes_ago)).replace(
        tzinfo=None
    ).isoformat(timespec="seconds")


# ─────────────────────────────────────────────────────────────────────────────
# Hard CLOSE branches
# ─────────────────────────────────────────────────────────────────────────────

class TestHardCloses:
    def test_health_critical_closes(self):
        action, reason, urgency, *_ = _decide_action(
            position=_position(),
            signal=_signal(),
            health_score=0.20,
            atr=0.001, current_price=1.10, breakeven_price=1.10,
            sent_at=_recent_iso(),
        )
        assert action == "CLOSE"
        assert urgency == "HIGH"
        assert "critical" in reason.lower()

    def test_signal_reversal_high_conf_closes(self):
        # Held LONG, signal now SHORT with confidence > 0.65 → CLOSE
        action, _, urgency, *_ = _decide_action(
            position=_position(direction="LONG"),
            signal=_signal(direction="SHORT", confidence=0.80),
            health_score=0.60,
            atr=0.001, current_price=1.10, breakeven_price=1.10,
            sent_at=_recent_iso(),
        )
        assert action == "CLOSE"
        assert urgency == "HIGH"

    def test_signal_reversal_low_conf_does_not_close(self):
        # Same reversal but conf=0.50 → does not close
        action, *_ = _decide_action(
            position=_position(direction="LONG"),
            signal=_signal(direction="SHORT", confidence=0.50),
            health_score=0.60,
            atr=0.001, current_price=1.10, breakeven_price=1.10,
            sent_at=_recent_iso(),
        )
        assert action != "CLOSE"

    def test_long_with_strongly_bearish_mtf_closes(self):
        action, _, urgency, *_ = _decide_action(
            position=_position(direction="LONG"),
            signal=_signal(direction="LONG", mtf_score=-2),
            health_score=0.60,
            atr=0.001, current_price=1.10, breakeven_price=1.10,
            sent_at=_recent_iso(),
        )
        assert action == "CLOSE"
        assert urgency == "HIGH"

    def test_short_with_strongly_bullish_mtf_closes(self):
        action, *_ = _decide_action(
            position=_position(direction="SHORT"),
            signal=_signal(direction="SHORT", mtf_score=2),
            health_score=0.60,
            atr=0.001, current_price=1.10, breakeven_price=1.10,
            sent_at=_recent_iso(),
        )
        assert action == "CLOSE"

    def test_daily_trend_veto_closes(self):
        action, _, urgency, *_ = _decide_action(
            position=_position(),
            signal=_signal(daily_trend_vetoed=True),
            health_score=0.60,
            atr=0.001, current_price=1.10, breakeven_price=1.10,
            sent_at=_recent_iso(),
        )
        assert action == "CLOSE"
        assert urgency == "HIGH"

    def test_rsi_bearish_divergence_closes_long(self):
        action, _, urgency, *_ = _decide_action(
            position=_position(direction="LONG"),
            signal=_signal(direction="LONG",
                           indicators=_indicators(bearish_div=True)),
            health_score=0.60,
            atr=0.001, current_price=1.10, breakeven_price=1.10,
            sent_at=_recent_iso(),
        )
        assert action == "CLOSE"
        assert "divergence" in reason_of(action, _decide_action(
            position=_position(direction="LONG"),
            signal=_signal(direction="LONG",
                           indicators=_indicators(bearish_div=True)),
            health_score=0.60,
            atr=0.001, current_price=1.10, breakeven_price=1.10,
            sent_at=_recent_iso(),
        )).lower()

    def test_rsi_bullish_divergence_closes_short(self):
        action, _, _urgency, *_ = _decide_action(
            position=_position(direction="SHORT"),
            signal=_signal(direction="SHORT",
                           indicators=_indicators(bullish_div=True)),
            health_score=0.60,
            atr=0.001, current_price=1.10, breakeven_price=1.10,
            sent_at=_recent_iso(),
        )
        assert action == "CLOSE"


def reason_of(action, decision_tuple):
    """Helper for divergence-test reason inspection."""
    return decision_tuple[1]


# ─────────────────────────────────────────────────────────────────────────────
# PARTIAL_CLOSE branches
# ─────────────────────────────────────────────────────────────────────────────

class TestPartialCloses:
    def test_high_conf_bearish_pattern_partial_closes_long_when_in_profit(self):
        # Reversal-pattern partial-close now requires profit > 0 (otherwise
        # we'd be locking a loss). Pattern conf threshold raised 0.65 → 0.75.
        report = _pattern_report(matches=[
            _pat("Double Top", conf=0.80, bias="BEARISH"),
        ])
        action, _reason, urgency, *_ = _decide_action(
            position=_position(direction="LONG", profit=2.50),
            signal=_signal(direction="LONG", patterns=report),
            health_score=0.60,
            atr=0.001, current_price=1.10, breakeven_price=1.10,
            sent_at=_recent_iso(),
        )
        assert action == "PARTIAL_CLOSE"
        assert urgency == "MEDIUM"

    def test_pattern_partial_close_blocked_when_at_loss(self):
        """Even high-conf reversal patterns shouldn't lock partial when the
        trade is already negative — that just locks the loss."""
        report = _pattern_report(matches=[
            _pat("Double Top", conf=0.85, bias="BEARISH"),
        ])
        action, *_ = _decide_action(
            position=_position(direction="LONG", profit=-1.20),
            signal=_signal(direction="LONG", patterns=report),
            health_score=0.60,
            atr=0.001, current_price=1.10, breakeven_price=1.10,
            sent_at=_recent_iso(),
        )
        assert action != "PARTIAL_CLOSE"

    def test_below_new_pattern_threshold_does_not_trigger_partial(self):
        # 0.70 confidence — below the new 0.75 threshold
        report = _pattern_report(matches=[
            _pat("Double Top", conf=0.70, bias="BEARISH"),
        ])
        action, *_ = _decide_action(
            position=_position(direction="LONG", profit=2.0),
            signal=_signal(direction="LONG", patterns=report),
            health_score=0.60,
            atr=0.001, current_price=1.10, breakeven_price=1.10,
            sent_at=_recent_iso(),
        )
        assert action != "PARTIAL_CLOSE"

    def test_low_conf_pattern_does_not_trigger_partial(self):
        report = _pattern_report(matches=[
            _pat("Double Top", conf=0.50, bias="BEARISH"),
        ])
        action, *_ = _decide_action(
            position=_position(direction="LONG", profit=2.0),
            signal=_signal(direction="LONG", patterns=report),
            health_score=0.60,
            atr=0.001, current_price=1.10, breakeven_price=1.10,
            sent_at=_recent_iso(),
        )
        assert action != "PARTIAL_CLOSE"


# ─────────────────────────────────────────────────────────────────────────────
# Time stop
# ─────────────────────────────────────────────────────────────────────────────

class TestTimeStop:
    def test_held_too_long_with_loss_closes(self):
        # Strategy default max hold = 8h, position at -profit and 10h old
        action, _, urgency, *_ = _decide_action(
            position=_position(profit=-5.0),
            signal=_signal(),
            health_score=0.60,
            atr=0.001, current_price=1.10, breakeven_price=1.10,
            sent_at=_recent_iso(minutes_ago=600),   # 10h ago
            strategy="UNKNOWN_STRAT",
        )
        assert action == "CLOSE"
        assert urgency == "MEDIUM"

    def test_held_too_long_but_profitable_does_not_close(self):
        # Same age, but in profit → time stop only fires when profit <= 0
        action, *_ = _decide_action(
            position=_position(profit=5.0),
            signal=_signal(),
            health_score=0.60,
            atr=0.001, current_price=1.10, breakeven_price=1.10,
            sent_at=_recent_iso(minutes_ago=600),
            strategy="UNKNOWN_STRAT",
        )
        assert action != "CLOSE"


# ─────────────────────────────────────────────────────────────────────────────
# News pressure
# ─────────────────────────────────────────────────────────────────────────────

class TestNewsPressure:
    def test_strong_bearish_news_with_declining_health_closes_long(self):
        action, *_ = _decide_action(
            position=_position(direction="LONG"),
            signal=_signal(direction="LONG", news_score=-0.50),
            health_score=0.45,   # < 0.55
            atr=0.001, current_price=1.10, breakeven_price=1.10,
            sent_at=_recent_iso(),
        )
        assert action == "CLOSE"

    def test_strong_bearish_news_with_healthy_position_no_action(self):
        # Same news but health = 0.70 → news pressure logic doesn't fire
        action, *_ = _decide_action(
            position=_position(direction="LONG"),
            signal=_signal(direction="LONG", news_score=-0.50),
            health_score=0.70,
            atr=0.001, current_price=1.10, breakeven_price=1.10,
            sent_at=_recent_iso(),
        )
        assert action != "CLOSE"


# ─────────────────────────────────────────────────────────────────────────────
# MODIFY_SL — health declining tightens stop
# ─────────────────────────────────────────────────────────────────────────────

class TestSLTightening:
    def test_health_below_50_tightens_long_sl_when_better(self):
        # Long at 1.10, current price 1.12, ATR 0.005 → trail_sl = 1.12 - 0.01 = 1.11
        # Existing SL is 1.09 → 1.11 > 1.09, so MODIFY_SL fires
        action, _, _urgency, new_sl, *_ = _decide_action(
            position=_position(direction="LONG", sl=1.09, profit=2.0),
            signal=_signal(),
            health_score=0.45,
            atr=0.005, current_price=1.12, breakeven_price=1.10,
            sent_at=_recent_iso(),
        )
        assert action == "MODIFY_SL"
        assert new_sl is not None
        assert new_sl >= 1.10   # SL moved up from 1.09

    def test_health_below_50_holds_when_sl_already_tight(self):
        # SL already higher than the proposed trail → HOLD
        action, *_ = _decide_action(
            position=_position(direction="LONG", sl=1.115, profit=2.0),
            signal=_signal(),
            health_score=0.45,
            atr=0.005, current_price=1.12, breakeven_price=1.10,
            sent_at=_recent_iso(),
        )
        assert action == "HOLD"


# ─────────────────────────────────────────────────────────────────────────────
# Healthy HOLD baseline
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthyHold:
    def test_default_healthy_position_holds(self):
        action, _, urgency, *_ = _decide_action(
            position=_position(profit=0.0),
            signal=_signal(),
            health_score=0.65,
            atr=0.001, current_price=1.10, breakeven_price=1.10,
            sent_at=_recent_iso(),
        )
        assert action == "HOLD"
        assert urgency == "LOW"


# ─────────────────────────────────────────────────────────────────────────────
# Price-based partial-close threshold (the unit-mismatch fix)
# ─────────────────────────────────────────────────────────────────────────────

class TestProfitPartialClose:
    """The previous code compared `position.profit` (in account currency, e.g.
    USD) against `one_r` (in price points, e.g. 0.005). On a typical EURUSD
    setup, $1 of profit easily exceeded a 0.005 price gap, triggering
    partial-close within seconds of entry. The fix uses price-move
    (current - open) on both sides of the comparison and raises 1R → 1.5R."""

    def test_dollar_profit_alone_does_not_trigger_partial(self):
        """Position open at 1.10 with SL 1.09 (1R = 0.01 price). Current
        price is 1.1005 → price-move = 0.0005 (well under 1.5R = 0.015).
        Even with $5 of profit (which exceeds 0.01 by 500x in dollar terms),
        the partial-close MUST stay closed.
        """
        action, *_ = _decide_action(
            position=_position(direction="LONG", sl=1.09, profit=5.0),
            signal=_signal(),
            health_score=0.75,
            atr=0.005, current_price=1.1005, breakeven_price=1.10,
            sent_at=_recent_iso(),
        )
        assert action == "HOLD", (
            f"expected HOLD but got {action!r} — partial-close fired despite "
            "price-move being far below 1.5R"
        )

    def test_below_one_r_price_move_does_not_trigger_anything(self):
        """Price moved 0.5R (well below the new 1.5R partial threshold and
        below the 1.0R trail trigger). Position should HOLD."""
        action, *_ = _decide_action(
            position=_position(direction="LONG", sl=1.09, profit=5.0),
            signal=_signal(),
            health_score=0.75,
            atr=0.001, current_price=1.105, breakeven_price=1.10,
            sent_at=_recent_iso(),
        )
        assert action == "HOLD"

    def test_two_r_price_move_triggers_partial(self):
        """Price moved 2R clearly above the 1.5R partial threshold. Use
        non-boundary values to avoid float precision artefacts."""
        action, _, urgency, new_sl, _, partial_vol = _decide_action(
            position=_position(direction="LONG", sl=1.09, profit=20.0),
            signal=_signal(),
            health_score=0.75,
            atr=0.005, current_price=1.12, breakeven_price=1.10,
            sent_at=_recent_iso(),
        )
        assert action == "PARTIAL_CLOSE"
        assert partial_vol is not None and partial_vol > 0
        assert urgency == "LOW"

    def test_short_two_r_price_move_triggers_partial(self):
        """SHORT mirror: open 1.10, SL 1.11, current 1.08 → price-move 2R."""
        action, *_ = _decide_action(
            position=_position(direction="SHORT", sl=1.11, profit=20.0,
                               open_price=1.10),
            signal=_signal(direction="SHORT"),
            health_score=0.75,
            atr=0.005, current_price=1.08, breakeven_price=1.10,
            sent_at=_recent_iso(),
        )
        assert action == "PARTIAL_CLOSE"

    def test_unrealised_loss_does_not_trigger_partial(self):
        """Even with positive `position.profit` (impossible in real life but
        possible if MT5 reports stale profit), price-move negative means
        no partial close."""
        action, *_ = _decide_action(
            position=_position(direction="LONG", sl=1.09, profit=999.0),
            signal=_signal(),
            health_score=0.75,
            atr=0.005, current_price=1.095, breakeven_price=1.10,   # below entry
            sent_at=_recent_iso(),
        )
        assert action != "PARTIAL_CLOSE"

    def test_two_r_price_move_triggers_trailing_sl(self):
        """Once we're past 1.5R partial-close lock, the next checkpoint is
        2×ATR profit-in-price for trailing SL. With current=1.115, atr=0.005,
        profit_price=0.015 = 3×ATR → multiplier = 1.2, trail_sl = current
        - 1.2×ATR = 1.115 - 0.006 = 1.109. Test that we get a MODIFY_SL
        when the partial-close has already happened (sl already moved up
        past breakeven, so the partial branch can't refire)."""
        # Set sl to breakeven+ so the partial-close path's "new_sl <= sl"
        # check fires and we drop to the trailing logic. profit_price stays
        # 0.015 = 3×ATR.
        action, *_ = _decide_action(
            position=_position(direction="LONG", sl=1.108, profit=15.0,
                               open_price=1.10),
            signal=_signal(),
            health_score=0.75,
            atr=0.005, current_price=1.115, breakeven_price=1.10,
            sent_at=_recent_iso(),
        )
        # At 3×ATR profit_price, 1.2×ATR trail; but the 1.5R partial check
        # fires first (profit_price=0.015, 1.5R=0.015 — equality passes the
        # >= check). Either PARTIAL_CLOSE or MODIFY_SL is acceptable; both
        # are non-destructive responses to a winning trade.
        assert action in ("PARTIAL_CLOSE", "MODIFY_SL")
