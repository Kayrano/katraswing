"""
Tests for utils.mt5_bridge — order routing, retcode handling, position
management, and learned-stop persistence.

These tests use a FakeMT5 fixture (see conftest.py) instead of the real
MetaTrader5 library. The goal is to lock down current behavior before we
add retry-with-backoff (plan Tier 1.1) so the change is safe to ship.
"""
from __future__ import annotations

import json

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# MT5 unavailable — every entrypoint short-circuits cleanly
# ─────────────────────────────────────────────────────────────────────────────

class TestMT5Unavailable:
    def test_is_available_false(self, mt5_unavailable):
        assert mt5_unavailable.is_available() is False

    def test_connect_returns_false(self, mt5_unavailable):
        assert mt5_unavailable.connect() is False

    def test_is_connected_false(self, mt5_unavailable):
        assert mt5_unavailable.is_connected() is False

    def test_send_signal_rejects(self, mt5_unavailable):
        result = mt5_unavailable.send_signal(
            "EURUSD=X", "LONG", 1.10, 1.09, 1.11,
        )
        assert result.success is False
        assert "not installed" in result.error.lower()

    def test_get_open_positions_empty(self, mt5_unavailable):
        assert mt5_unavailable.get_open_positions() == []

    def test_close_position_false(self, mt5_unavailable):
        assert mt5_unavailable.close_position(123) is False


# ─────────────────────────────────────────────────────────────────────────────
# Connection lifecycle
# ─────────────────────────────────────────────────────────────────────────────

class TestConnection:
    def test_connect_success(self, mt5_bridge):
        bridge, fake = mt5_bridge
        assert bridge.connect() is True

    def test_connect_failure(self, mt5_bridge):
        bridge, fake = mt5_bridge
        fake._initialize_should_succeed = False
        assert bridge.connect() is False

    def test_disconnect_calls_shutdown(self, mt5_bridge):
        bridge, fake = mt5_bridge
        bridge.disconnect()
        assert fake.shutdown_called is True

    def test_is_connected_true(self, mt5_bridge):
        bridge, fake = mt5_bridge
        fake._terminal_connected = True
        assert bridge.is_connected() is True

    def test_is_connected_false_when_terminal_disconnected(self, mt5_bridge):
        bridge, fake = mt5_bridge
        fake._terminal_connected = False
        assert bridge.is_connected() is False

    def test_ensure_connected_reconnects_when_dropped(self, mt5_bridge):
        bridge, fake = mt5_bridge
        fake._initialized = False         # connection lost
        fake._terminal_connected = False
        # initialize() in FakeMT5 sets both flags back to True
        assert bridge.ensure_connected() is True
        assert fake._terminal_connected is True


# ─────────────────────────────────────────────────────────────────────────────
# _filling_mode bitmask logic
# ─────────────────────────────────────────────────────────────────────────────

class TestFillingMode:
    def test_no_modes_returns_return(self, mt5_bridge):
        bridge, fake = mt5_bridge
        sym = fake.add_symbol("X", filling_mode=0)
        assert bridge._filling_mode(sym) == bridge.mt5.ORDER_FILLING_RETURN  # 2

    def test_fok_only_returns_fok(self, mt5_bridge):
        bridge, fake = mt5_bridge
        sym = fake.add_symbol("X", filling_mode=1)   # bit 0 set
        assert bridge._filling_mode(sym) == bridge.mt5.ORDER_FILLING_FOK    # 0

    def test_ioc_only_returns_ioc(self, mt5_bridge):
        bridge, fake = mt5_bridge
        sym = fake.add_symbol("X", filling_mode=2)   # bit 1 set
        assert bridge._filling_mode(sym) == bridge.mt5.ORDER_FILLING_IOC    # 1

    def test_both_modes_prefers_ioc(self, mt5_bridge):
        bridge, fake = mt5_bridge
        sym = fake.add_symbol("X", filling_mode=3)
        assert bridge._filling_mode(sym) == bridge.mt5.ORDER_FILLING_IOC

    def test_none_symbol_returns_return(self, mt5_bridge):
        bridge, _ = mt5_bridge
        assert bridge._filling_mode(None) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Learned stop minimums persistence
# ─────────────────────────────────────────────────────────────────────────────

class TestLearnedMin:
    def test_load_returns_zero_when_missing(self, mt5_bridge):
        bridge, _ = mt5_bridge
        assert bridge._load_learned_min("EURUSD") == 0.0

    def test_save_then_load_roundtrip(self, mt5_bridge):
        bridge, _ = mt5_bridge
        bridge._save_learned_min("EURUSD", 0.0025)
        assert bridge._load_learned_min("EURUSD") == pytest.approx(0.0025)

    def test_save_only_increases(self, mt5_bridge):
        bridge, _ = mt5_bridge
        bridge._save_learned_min("EURUSD", 0.0050)
        bridge._save_learned_min("EURUSD", 0.0020)   # smaller — should not overwrite
        assert bridge._load_learned_min("EURUSD") == pytest.approx(0.0050)

    def test_save_writes_valid_json(self, mt5_bridge):
        bridge, _ = mt5_bridge
        bridge._save_learned_min("GBPUSD", 0.0015)
        data = json.loads(bridge._STOP_LEARNING_PATH.read_text(encoding="utf-8"))
        assert "GBPUSD" in data
        assert data["GBPUSD"]["min_pts"] == pytest.approx(0.0015)
        assert "updated" in data["GBPUSD"]


# ─────────────────────────────────────────────────────────────────────────────
# send_signal — pre-flight rejections
# ─────────────────────────────────────────────────────────────────────────────

class TestSendSignalPreflight:
    def test_not_connected_rejects(self, mt5_bridge):
        bridge, fake = mt5_bridge
        fake._terminal_connected = False
        result = bridge.send_signal("EURUSD=X", "LONG", 1.10, 1.09, 1.11)
        assert result.success is False
        assert "not connected" in result.error.lower()

    def test_symbol_not_in_broker(self, mt5_bridge):
        bridge, fake = mt5_bridge
        # No symbols added → symbol_select returns False
        result = bridge.send_signal("EURUSD=X", "LONG", 1.10, 1.09, 1.11)
        assert result.success is False
        assert "SYMBOL_MAP" in result.error

    def test_no_tick(self, mt5_bridge):
        bridge, fake = mt5_bridge
        fake.add_symbol("EURUSD")
        # No tick added — symbol_info_tick returns None
        result = bridge.send_signal("EURUSD=X", "LONG", 1.10, 1.09, 1.11)
        assert result.success is False
        assert "tick" in result.error.lower()

    def test_duplicate_long_position_blocks_long(self, mt5_bridge):
        bridge, fake = mt5_bridge
        fake.add_symbol("EURUSD")
        fake.add_tick("EURUSD", bid=1.0998, ask=1.1000)
        fake.add_position(
            ticket=1, symbol="EURUSD", type_=fake.ORDER_TYPE_BUY,
        )
        result = bridge.send_signal("EURUSD=X", "LONG", 1.10, 1.09, 1.11)
        assert result.success is False
        assert "Duplicate" in result.error

    def test_duplicate_long_position_allows_short(self, mt5_bridge):
        bridge, fake = mt5_bridge
        fake.add_symbol("EURUSD")
        fake.add_tick("EURUSD", bid=1.0998, ask=1.1000)
        fake.add_position(
            ticket=1, symbol="EURUSD", type_=fake.ORDER_TYPE_BUY,
        )
        # Hedge with opposite direction — should not be blocked by dup guard
        result = bridge.send_signal("EURUSD=X", "SHORT", 1.10, 1.11, 1.09)
        assert result.success is True

    def test_insufficient_margin(self, mt5_bridge):
        bridge, fake = mt5_bridge
        fake.add_symbol("EURUSD")
        fake.add_tick("EURUSD", bid=1.0998, ask=1.1000)
        fake.account.margin_free = 100.0   # tiny — far below required×1.5
        result = bridge.send_signal(
            "EURUSD=X", "LONG", 1.10, 1.09, 1.11, lots=0.5,
        )
        assert result.success is False
        assert "margin" in result.error.lower()


# ─────────────────────────────────────────────────────────────────────────────
# send_signal — order_send retcode handling (CRITICAL — gates Tier 1.1)
# ─────────────────────────────────────────────────────────────────────────────

class TestSendSignalRetcodes:
    def _setup(self, fake):
        fake.add_symbol("EURUSD")
        fake.add_tick("EURUSD", bid=1.0998, ask=1.1000)

    def test_happy_path_returns_done(self, mt5_bridge):
        bridge, fake = mt5_bridge
        self._setup(fake)
        # Default order_send returns DONE
        result = bridge.send_signal(
            "EURUSD=X", "LONG", 1.10, 1.0950, 1.1100, lots=0.1,
        )
        assert result.success is True
        assert result.ticket == 12345
        assert result.symbol == "EURUSD"
        assert result.direction == "LONG"
        # SL/TP rebased from live ask price (1.1000) preserving the input distances
        assert result.entry == pytest.approx(1.1000, abs=1e-4)

    def test_done_persists_learned_min(self, mt5_bridge):
        bridge, fake = mt5_bridge
        self._setup(fake)
        bridge.send_signal("EURUSD=X", "LONG", 1.10, 1.0950, 1.1100, lots=0.1)
        # Successful trade should have written some min_pts for EURUSD
        assert bridge._STOP_LEARNING_PATH.exists()
        data = json.loads(bridge._STOP_LEARNING_PATH.read_text(encoding="utf-8"))
        assert "EURUSD" in data
        assert data["EURUSD"]["min_pts"] > 0

    def test_10016_triggers_fallback_retry_with_double_min(self, mt5_bridge):
        bridge, fake = mt5_bridge
        self._setup(fake)
        # Pre-load: first send rejects 10016, fallback succeeds
        fake.queue_order_send(
            fake.make_order_result(retcode=10016, comment="Invalid stops"),
            fake.make_order_result(retcode=fake.TRADE_RETCODE_DONE, order=99),
        )
        # Skip the precheck loop's expansion by making order_check pass
        fake.order_check_default = lambda req: fake.make_check_result(retcode=0)

        result = bridge.send_signal(
            "EURUSD=X", "LONG", 1.10, 1.0998, 1.1002, lots=0.1,
        )

        assert result.success is True
        assert result.ticket == 99
        # Two order_send calls — original + fallback
        assert len(fake.order_send_calls) == 2
        # Fallback request must have wider stops than the original
        first_req, retry_req = fake.order_send_calls
        first_sl_dist  = abs(first_req["price"]  - first_req["sl"])
        retry_sl_dist  = abs(retry_req["price"]  - retry_req["sl"])
        assert retry_sl_dist > first_sl_dist

    def test_10016_then_10016_returns_failure(self, mt5_bridge):
        bridge, fake = mt5_bridge
        self._setup(fake)
        fake.queue_order_send(
            fake.make_order_result(retcode=10016, comment="Invalid stops"),
            fake.make_order_result(retcode=10016, comment="Invalid stops"),
        )
        fake.order_check_default = lambda req: fake.make_check_result(retcode=0)

        result = bridge.send_signal(
            "EURUSD=X", "LONG", 1.10, 1.0998, 1.1002, lots=0.1,
        )
        assert result.success is False
        assert "10016" in result.error

    def test_generic_retcode_rejection_includes_hint(self, mt5_bridge):
        bridge, fake = mt5_bridge
        self._setup(fake)
        fake.queue_order_send(
            fake.make_order_result(retcode=10018, comment="Market closed"),
        )
        result = bridge.send_signal(
            "EURUSD=X", "LONG", 1.10, 1.0950, 1.1100, lots=0.1,
        )
        assert result.success is False
        assert "10018" in result.error
        # Hint dict should expand 10018 to a human message
        assert "Market closed" in result.error or "market closed" in result.error.lower()

    def test_order_send_returns_none(self, mt5_bridge):
        bridge, fake = mt5_bridge
        self._setup(fake)
        fake.queue_order_send(None)
        fake.last_error_value = (-1, "transport error")
        result = bridge.send_signal(
            "EURUSD=X", "LONG", 1.10, 1.0950, 1.1100, lots=0.1,
        )
        assert result.success is False
        assert "transport error" in result.error or "-1" in result.error


# ─────────────────────────────────────────────────────────────────────────────
# send_signal — Tier 1.1 retry-with-backoff for transient retcodes
# ─────────────────────────────────────────────────────────────────────────────

class TestSendSignalTransientRetries:
    def _setup(self, fake):
        fake.add_symbol("EURUSD")
        fake.add_tick("EURUSD", bid=1.0998, ask=1.1000)

    @pytest.fixture(autouse=True)
    def _no_real_sleep(self, monkeypatch):
        # Replace time.sleep inside the bridge so retries don't actually wait
        import utils.mt5_bridge as bridge
        monkeypatch.setattr(bridge.time, "sleep", lambda _s: None)

    @pytest.mark.parametrize("transient_retcode", [10004, 10006, 10021])
    def test_transient_retried_then_succeeds(self, mt5_bridge, transient_retcode):
        bridge, fake = mt5_bridge
        self._setup(fake)
        fake.queue_order_send(
            fake.make_order_result(retcode=transient_retcode, comment="transient"),
            fake.make_order_result(retcode=fake.TRADE_RETCODE_DONE, order=77),
        )
        result = bridge.send_signal(
            "EURUSD=X", "LONG", 1.10, 1.0950, 1.1100, lots=0.1,
        )
        assert result.success is True
        assert result.ticket == 77
        assert len(fake.order_send_calls) == 2

    def test_two_transient_then_done(self, mt5_bridge):
        bridge, fake = mt5_bridge
        self._setup(fake)
        fake.queue_order_send(
            fake.make_order_result(retcode=10004),
            fake.make_order_result(retcode=10021),
            fake.make_order_result(retcode=fake.TRADE_RETCODE_DONE, order=88),
        )
        result = bridge.send_signal(
            "EURUSD=X", "LONG", 1.10, 1.0950, 1.1100, lots=0.1,
        )
        assert result.success is True
        assert result.ticket == 88
        assert len(fake.order_send_calls) == 3

    def test_max_three_retries_then_gives_up(self, mt5_bridge):
        bridge, fake = mt5_bridge
        self._setup(fake)
        # 4 transient rejections — should stop after 3 retries (4 total attempts)
        fake.queue_order_send(
            fake.make_order_result(retcode=10004, comment="requote 1"),
            fake.make_order_result(retcode=10004, comment="requote 2"),
            fake.make_order_result(retcode=10004, comment="requote 3"),
            fake.make_order_result(retcode=10004, comment="requote 4"),
        )
        result = bridge.send_signal(
            "EURUSD=X", "LONG", 1.10, 1.0950, 1.1100, lots=0.1,
        )
        assert result.success is False
        assert "10004" in result.error
        # Initial attempt + 3 retries = 4 total
        assert len(fake.order_send_calls) == 4

    def test_terminal_retcode_not_retried(self, mt5_bridge):
        bridge, fake = mt5_bridge
        self._setup(fake)
        # 10018 (market closed) — single attempt, no retries
        fake.queue_order_send(
            fake.make_order_result(retcode=10018, comment="Market closed"),
        )
        result = bridge.send_signal(
            "EURUSD=X", "LONG", 1.10, 1.0950, 1.1100, lots=0.1,
        )
        assert result.success is False
        assert "10018" in result.error
        assert len(fake.order_send_calls) == 1

    def test_retry_refreshes_tick_and_rebases_stops(self, mt5_bridge):
        bridge, fake = mt5_bridge
        self._setup(fake)

        # Each retry should call symbol_info_tick to refresh price.
        # Drive the tick to move between attempts so we can verify rebasing.
        ticks = [
            (1.0998, 1.1000),   # initial fetch before first send
            (1.1010, 1.1012),   # first retry sees moved market
            (1.1020, 1.1022),   # second retry — even further
        ]
        idx = {"i": 0}
        original_tick = fake.symbol_info_tick

        def moving_tick(name):
            i = min(idx["i"], len(ticks) - 1)
            bid, ask = ticks[i]
            idx["i"] += 1
            from types import SimpleNamespace
            return SimpleNamespace(bid=bid, ask=ask, time=0.0)

        fake.symbol_info_tick = moving_tick

        fake.queue_order_send(
            fake.make_order_result(retcode=10004, comment="requote"),
            fake.make_order_result(retcode=fake.TRADE_RETCODE_DONE, order=55),
        )

        result = bridge.send_signal(
            "EURUSD=X", "LONG", 1.10, 1.0950, 1.1100, lots=0.1,
        )
        assert result.success is True
        # Second send must use a refreshed price, not the original
        first_send, retry_send = fake.order_send_calls
        assert retry_send["price"] != first_send["price"]
        # SL distance should be preserved (rebased from new price)
        first_sl_dist = abs(first_send["price"] - first_send["sl"])
        retry_sl_dist = abs(retry_send["price"] - retry_send["sl"])
        assert retry_sl_dist == pytest.approx(first_sl_dist, abs=1e-4)

    def test_retry_aborts_when_tick_disappears(self, mt5_bridge):
        bridge, fake = mt5_bridge
        self._setup(fake)
        fake.queue_order_send(
            fake.make_order_result(retcode=10004, comment="requote"),
        )

        # Initial tick fetch must succeed so the first send goes through.
        # The retry refresh (second tick read) returns None, aborting the retry.
        call_count = {"n": 0}
        original_tick = fake.symbol_info_tick

        def disappearing_tick(name):
            call_count["n"] += 1
            if call_count["n"] >= 2:
                return None
            return original_tick(name)

        fake.symbol_info_tick = disappearing_tick

        result = bridge.send_signal(
            "EURUSD=X", "LONG", 1.10, 1.0950, 1.1100, lots=0.1,
        )
        # Should bail out cleanly with the original transient retcode error
        assert result.success is False
        assert "10004" in result.error
        # Only one send — retry was aborted because tick went stale
        assert len(fake.order_send_calls) == 1

    def test_10016_still_uses_special_fallback_not_transient_retry(self, mt5_bridge):
        """10016 is NOT in _RETRYABLE_RETCODES — it falls through to the
        existing 'expand stops 2x' path, not the transient retry layer."""
        bridge, fake = mt5_bridge
        self._setup(fake)
        fake.queue_order_send(
            fake.make_order_result(retcode=10016, comment="Invalid stops"),
            fake.make_order_result(retcode=fake.TRADE_RETCODE_DONE, order=99),
        )
        result = bridge.send_signal(
            "EURUSD=X", "LONG", 1.10, 1.0998, 1.1002, lots=0.1,
        )
        assert result.success is True
        # Two sends: initial + 10016 fallback (NOT three from transient retry)
        assert len(fake.order_send_calls) == 2


# ─────────────────────────────────────────────────────────────────────────────
# send_from_signal_result wrapper
# ─────────────────────────────────────────────────────────────────────────────

class TestSendFromSignalResult:
    def _make_signal_result(self, **overrides):
        # Local import to avoid circular imports during conftest collection
        from agents.signal_engine import SignalResult
        defaults = dict(
            ticker="EURUSD=X",
            direction="LONG",
            confidence=0.75,
            entry=1.10,
            sl=1.0950,
            tp=1.1100,
            mt5_symbol="EURUSD",
            risk_level="MEDIUM",
        )
        defaults.update(overrides)
        return SignalResult(**defaults)

    def test_no_trade_direction_rejected(self, mt5_bridge):
        bridge, _ = mt5_bridge
        sr = self._make_signal_result(direction="NO TRADE")
        result = bridge.send_from_signal_result(sr)
        assert result.success is False
        assert "No trade" in result.error or "no trade" in result.error.lower()

    def test_yfinance_symbol_falls_back_to_symbol_map(self, mt5_bridge):
        bridge, fake = mt5_bridge
        fake.add_symbol("EURUSD")
        fake.add_tick("EURUSD", bid=1.0998, ask=1.1000)
        # mt5_symbol contains "=" — should be rejected, then SYMBOL_MAP is used
        sr = self._make_signal_result(mt5_symbol="EURUSD=X")
        result = bridge.send_from_signal_result(sr)
        assert result.success is True
        assert result.symbol == "EURUSD"

    def test_no_broker_mapping_returns_friendly_error(self, mt5_bridge):
        bridge, _ = mt5_bridge
        sr = self._make_signal_result(
            ticker="UNKNOWN",
            mt5_symbol="UNKNOWN=X",   # yfinance-format → fallback
        )
        result = bridge.send_from_signal_result(sr)
        assert result.success is False
        assert "broker symbol" in result.error.lower()

    def test_comment_is_sanitized_and_capped(self, mt5_bridge):
        bridge, fake = mt5_bridge
        fake.add_symbol("EURUSD")
        fake.add_tick("EURUSD", bid=1.0998, ask=1.1000)
        sr = self._make_signal_result()
        bridge.send_from_signal_result(sr)
        # Comment passed to order_send must be ≤24 chars and only [alnum _]
        sent = fake.order_send_calls[-1]
        assert len(sent["comment"]) <= 24
        assert all(ch.isalnum() or ch in " _" for ch in sent["comment"])

    @pytest.mark.parametrize(
        "risk_level,expected_multiplier",
        [("LOW", 1.5), ("MEDIUM", 1.0), ("HIGH", 0.5)],
    )
    def test_risk_level_scales_risk_pct(
        self, mt5_bridge, monkeypatch, risk_level, expected_multiplier,
    ):
        bridge, fake = mt5_bridge
        fake.add_symbol("EURUSD")
        fake.add_tick("EURUSD", bid=1.0998, ask=1.1000)
        sr = self._make_signal_result(risk_level=risk_level)

        captured = {}
        original = bridge.send_signal

        def spy(**kw):
            captured["risk_pct"] = kw.get("risk_pct")
            return original(**kw)

        monkeypatch.setattr(bridge, "send_signal", spy)
        bridge.send_from_signal_result(sr, risk_pct=1.0)

        assert captured["risk_pct"] == pytest.approx(expected_multiplier)


# ─────────────────────────────────────────────────────────────────────────────
# Position management
# ─────────────────────────────────────────────────────────────────────────────

class TestPositionManagement:
    def test_get_open_positions_filters_by_magic(self, mt5_bridge):
        bridge, fake = mt5_bridge
        fake.add_position(
            ticket=1, symbol="EURUSD", type_=fake.ORDER_TYPE_BUY,
            magic=234100,   # Katraswing magic
        )
        fake.add_position(
            ticket=2, symbol="EURUSD", type_=fake.ORDER_TYPE_SELL,
            magic=999999,   # someone else's magic
        )
        positions = bridge.get_open_positions()
        assert len(positions) == 1
        assert positions[0].ticket == 1

    def test_position_direction_mapping(self, mt5_bridge):
        bridge, fake = mt5_bridge
        fake.add_position(
            ticket=1, symbol="EURUSD", type_=fake.ORDER_TYPE_BUY, magic=234100,
        )
        fake.add_position(
            ticket=2, symbol="GBPUSD", type_=fake.ORDER_TYPE_SELL, magic=234100,
        )
        positions = bridge.get_open_positions()
        by_ticket = {p.ticket: p for p in positions}
        assert by_ticket[1].direction == "LONG"
        assert by_ticket[2].direction == "SHORT"

    def test_close_position_not_found(self, mt5_bridge):
        bridge, fake = mt5_bridge
        assert bridge.close_position(999) is False

    def test_close_position_success(self, mt5_bridge):
        bridge, fake = mt5_bridge
        fake.add_symbol("EURUSD")
        fake.add_tick("EURUSD", bid=1.0998, ask=1.1000)
        fake.add_position(
            ticket=42, symbol="EURUSD", type_=fake.ORDER_TYPE_BUY, magic=234100,
        )
        assert bridge.close_position(42) is True

    def test_close_position_fail(self, mt5_bridge):
        bridge, fake = mt5_bridge
        fake.add_symbol("EURUSD")
        fake.add_tick("EURUSD", bid=1.0998, ask=1.1000)
        fake.add_position(
            ticket=42, symbol="EURUSD", type_=fake.ORDER_TYPE_BUY, magic=234100,
        )
        fake.queue_order_send(fake.make_order_result(retcode=10006, comment="Rejected"))
        assert bridge.close_position(42) is False


# ─────────────────────────────────────────────────────────────────────────────
# modify_position SL clamping
# ─────────────────────────────────────────────────────────────────────────────

class TestModifyPosition:
    def test_modify_not_found(self, mt5_bridge):
        bridge, _ = mt5_bridge
        assert bridge.modify_position(999, new_sl=1.0) is False

    def test_long_sl_clamped_to_below_price_minus_min(self, mt5_bridge):
        bridge, fake = mt5_bridge
        fake.add_symbol(
            "EURUSD",
            digits=5, point=0.00001,
            trade_stops_level=100,   # 100 × 0.00001 = 0.001
        )
        fake.add_tick("EURUSD", bid=1.1000, ask=1.1001)
        fake.add_position(
            ticket=1, symbol="EURUSD",
            type_=fake.ORDER_TYPE_BUY, price_open=1.0950,
        )
        # Caller asks for SL=1.0999 (way too close to bid price 1.1000)
        bridge.modify_position(1, new_sl=1.0999)
        sent = fake.order_send_calls[-1]
        # SL should be clamped to ≤ price - effective_min ≈ 1.1000 - 0.001
        assert sent["sl"] <= 1.0991

    def test_short_sl_clamped_to_above_price_plus_min(self, mt5_bridge):
        bridge, fake = mt5_bridge
        fake.add_symbol(
            "EURUSD",
            digits=5, point=0.00001, trade_stops_level=100,
        )
        fake.add_tick("EURUSD", bid=1.1000, ask=1.1001)
        fake.add_position(
            ticket=1, symbol="EURUSD",
            type_=fake.ORDER_TYPE_SELL, price_open=1.1050,
        )
        # Caller asks for SL=1.1002 (too close to ask price 1.1001)
        bridge.modify_position(1, new_sl=1.1002)
        sent = fake.order_send_calls[-1]
        assert sent["sl"] >= 1.1010

    def test_modify_uses_sltp_action(self, mt5_bridge):
        bridge, fake = mt5_bridge
        fake.add_symbol("EURUSD")
        fake.add_tick("EURUSD", bid=1.1000, ask=1.1001)
        fake.add_position(
            ticket=1, symbol="EURUSD", type_=fake.ORDER_TYPE_BUY,
        )
        bridge.modify_position(1, new_sl=1.0900, new_tp=1.1100)
        sent = fake.order_send_calls[-1]
        assert sent["action"] == fake.TRADE_ACTION_SLTP


# ─────────────────────────────────────────────────────────────────────────────
# partial_close_position
# ─────────────────────────────────────────────────────────────────────────────

class TestPartialClose:
    def test_volume_clamped_to_position_size(self, mt5_bridge):
        bridge, fake = mt5_bridge
        fake.add_symbol("EURUSD", volume_step=0.01)
        fake.add_tick("EURUSD", bid=1.1000, ask=1.1001)
        fake.add_position(
            ticket=1, symbol="EURUSD",
            type_=fake.ORDER_TYPE_BUY, volume=0.30,
        )
        bridge.partial_close_position(1, volume=1.0)   # ask for too much
        sent = fake.order_send_calls[-1]
        assert sent["volume"] == pytest.approx(0.30)

    def test_volume_rounded_to_step(self, mt5_bridge):
        bridge, fake = mt5_bridge
        fake.add_symbol("EURUSD", volume_step=0.10)
        fake.add_tick("EURUSD", bid=1.1000, ask=1.1001)
        fake.add_position(
            ticket=1, symbol="EURUSD",
            type_=fake.ORDER_TYPE_BUY, volume=1.00,
        )
        bridge.partial_close_position(1, volume=0.27)
        sent = fake.order_send_calls[-1]
        # 0.27 rounded down to nearest 0.10 → 0.20
        assert sent["volume"] == pytest.approx(0.20)


# ─────────────────────────────────────────────────────────────────────────────
# _find_safe_stops precheck loop (10016 expansion)
# ─────────────────────────────────────────────────────────────────────────────

class TestFindSafeStops:
    def test_accepted_immediately_returns_input_stops(self, mt5_bridge):
        bridge, fake = mt5_bridge
        sym = fake.add_symbol("EURUSD", digits=5, point=0.00001, trade_stops_level=10)
        sl, tp, eff = bridge._find_safe_stops(
            symbol="EURUSD",
            price=1.1000,
            direction="LONG",
            sl_dist=0.0050, tp_dist=0.0100,
            effective_min=0.0001,
            sym_info=sym,
            fill_type=1,
            provisional_lots=0.1,
            order_type=fake.ORDER_TYPE_BUY,
            digits=5,
        )
        assert sl == pytest.approx(1.0950, abs=1e-4)
        assert tp == pytest.approx(1.1100, abs=1e-4)
        assert eff == pytest.approx(0.0001)

    def test_10016_expands_then_accepts(self, mt5_bridge):
        bridge, fake = mt5_bridge
        sym = fake.add_symbol("EURUSD", digits=5, point=0.00001, trade_stops_level=10)
        # First 3 checks fail with 10016, fourth passes
        fake.queue_order_check(
            fake.make_check_result(retcode=10016),
            fake.make_check_result(retcode=10016),
            fake.make_check_result(retcode=10016),
            fake.make_check_result(retcode=0),
        )
        sl, tp, eff = bridge._find_safe_stops(
            symbol="EURUSD",
            price=1.1000,
            direction="LONG",
            sl_dist=0.0001, tp_dist=0.0001,
            effective_min=0.0001,
            sym_info=sym,
            fill_type=1,
            provisional_lots=0.1,
            order_type=fake.ORDER_TYPE_BUY,
            digits=5,
        )
        # effective_min should have grown 1.2× three times: 0.0001 × 1.2³ ≈ 0.0001728
        assert eff == pytest.approx(0.0001 * (1.2 ** 3), rel=1e-3)
        # SL/TP reflect the expanded distance
        assert sl < 1.1000
        assert tp > 1.1000

    def test_short_direction_inverts_stops(self, mt5_bridge):
        bridge, fake = mt5_bridge
        sym = fake.add_symbol("EURUSD", digits=5, point=0.00001, trade_stops_level=10)
        sl, tp, _ = bridge._find_safe_stops(
            symbol="EURUSD",
            price=1.1000,
            direction="SHORT",
            sl_dist=0.0050, tp_dist=0.0100,
            effective_min=0.0001,
            sym_info=sym,
            fill_type=1,
            provisional_lots=0.1,
            order_type=fake.ORDER_TYPE_SELL,
            digits=5,
        )
        assert sl > 1.1000   # SL above entry for SHORT
        assert tp < 1.1000   # TP below entry for SHORT
