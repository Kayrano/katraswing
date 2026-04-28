"""
Test fixtures for utils.mt5_bridge.

The real MetaTrader5 library is Windows-only and requires a running terminal.
We don't import it. Instead, fixtures patch utils.mt5_bridge.mt5 and
utils.mt5_bridge.MT5_AVAILABLE with a FakeMT5 instance that exposes the
constants and methods the bridge actually uses.

Each test that needs MT5 declares the `mt5_bridge` fixture, which returns a
tuple (bridge_module, fake_mt5) — the test mutates fake_mt5's response
queues to drive the path under test.
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

# Make the project root importable so `import utils.mt5_bridge` works.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── MT5 constants — values mirror the real MetaTrader5 module ────────────────
class _MT5Constants:
    TRADE_ACTION_DEAL    = 1
    TRADE_ACTION_SLTP    = 6

    ORDER_TYPE_BUY       = 0
    ORDER_TYPE_SELL      = 1

    ORDER_TIME_GTC       = 0

    ORDER_FILLING_FOK    = 0
    ORDER_FILLING_IOC    = 1
    ORDER_FILLING_RETURN = 2

    TRADE_RETCODE_DONE   = 10009

    TIMEFRAME_M1  = 1
    TIMEFRAME_M5  = 5
    TIMEFRAME_M15 = 15
    TIMEFRAME_M30 = 30
    TIMEFRAME_H1  = 16385
    TIMEFRAME_H4  = 16388
    TIMEFRAME_D1  = 16408


class FakeMT5(_MT5Constants):
    """
    Configurable stand-in for the MetaTrader5 module.

    Defaults represent a healthy connection with reasonable symbol metadata.
    Tests override individual attributes or pre-load response queues.
    """

    def __init__(self) -> None:
        # Connection state
        self._initialized              = True
        self._terminal_connected       = True
        self._initialize_should_succeed = True

        # Account
        self.account = SimpleNamespace(
            balance=100_000.0,
            equity=100_000.0,
            margin=0.0,
            margin_free=100_000.0,
            margin_level=None,
            currency="USD",
            leverage=100,
        )

        # Symbols: name → SimpleNamespace
        self._symbols: dict[str, SimpleNamespace] = {}
        self._ticks:   dict[str, SimpleNamespace] = {}

        # Positions: list of SimpleNamespace
        self._positions: list[SimpleNamespace] = []

        # Response queues — each call pops from the left if non-empty,
        # otherwise the default behavior runs.
        self._order_send_queue:  deque = deque()
        self._order_check_queue: deque = deque()

        # Default behaviors (overridable by tests)
        self.order_send_default:  Callable[[dict], Any] = self._default_order_send
        self.order_check_default: Callable[[dict], Any] = self._default_order_check

        self.last_error_value = (0, "no error")

        # Call recording for assertions
        self.order_send_calls:  list[dict] = []
        self.order_check_calls: list[dict] = []
        self.symbol_select_calls: list[tuple[str, bool]] = []
        self.shutdown_called = False

    # ── Symbol / tick configuration helpers ─────────────────────────────────
    def add_symbol(
        self,
        name: str,
        *,
        digits: int = 5,
        point: float = 0.00001,
        trade_stops_level: int = 10,
        trade_tick_size: float = 0.00001,
        trade_tick_value: float = 1.0,
        volume_min: float = 0.01,
        volume_max: float = 100.0,
        volume_step: float = 0.01,
        filling_mode: int = 3,   # FOK | IOC supported
        trade_mode: int = 4,     # SYMBOL_TRADE_MODE_FULL
        path: str = "Forex",
        description: str | None = None,
    ) -> SimpleNamespace:
        info = SimpleNamespace(
            name=name,
            digits=digits,
            point=point,
            trade_stops_level=trade_stops_level,
            trade_tick_size=trade_tick_size,
            trade_tick_value=trade_tick_value,
            volume_min=volume_min,
            volume_max=volume_max,
            volume_step=volume_step,
            filling_mode=filling_mode,
            trade_mode=trade_mode,
            path=path,
            description=description or name,
        )
        self._symbols[name] = info
        return info

    def add_tick(
        self,
        name: str,
        *,
        bid: float = 1.0,
        ask: float = 1.0001,
        time_: float = 0.0,
    ) -> SimpleNamespace:
        import time as _t
        t = SimpleNamespace(bid=bid, ask=ask, time=time_ or _t.time())
        self._ticks[name] = t
        return t

    def add_position(
        self,
        *,
        ticket: int,
        symbol: str,
        type_: int,
        volume: float = 0.1,
        price_open: float = 1.0,
        sl: float = 0.0,
        tp: float = 0.0,
        profit: float = 0.0,
        magic: int = 234100,
        time_: int = 0,
    ) -> SimpleNamespace:
        p = SimpleNamespace(
            ticket=ticket,
            symbol=symbol,
            type=type_,
            volume=volume,
            price_open=price_open,
            sl=sl,
            tp=tp,
            profit=profit,
            magic=magic,
            swap=0.0,
            commission=0.0,
            time=time_,
            price_current=price_open,
        )
        self._positions.append(p)
        return p

    # ── Response queue helpers ───────────────────────────────────────────────
    def queue_order_send(self, *responses) -> None:
        """Pre-load order_send return values (consumed in order)."""
        for r in responses:
            self._order_send_queue.append(r)

    def queue_order_check(self, *responses) -> None:
        for r in responses:
            self._order_check_queue.append(r)

    @staticmethod
    def make_order_result(
        retcode: int = 10009,
        order: int = 12345,
        comment: str = "",
        request_id: int = 0,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            retcode=retcode,
            order=order,
            comment=comment,
            request_id=request_id,
            deal=0,
            volume=0.0,
            price=0.0,
            bid=0.0,
            ask=0.0,
            external_id="",
        )

    @staticmethod
    def make_check_result(retcode: int = 0, comment: str = "") -> SimpleNamespace:
        return SimpleNamespace(retcode=retcode, comment=comment)

    # ── Default behaviors ────────────────────────────────────────────────────
    def _default_order_send(self, request: dict) -> SimpleNamespace:
        return self.make_order_result(retcode=self.TRADE_RETCODE_DONE, order=12345)

    def _default_order_check(self, request: dict) -> SimpleNamespace:
        return self.make_check_result(retcode=0)

    # ── MetaTrader5 module surface ───────────────────────────────────────────
    def initialize(self, **kwargs) -> bool:
        if self._initialize_should_succeed:
            self._initialized = True
            self._terminal_connected = True
            return True
        return False

    def shutdown(self) -> None:
        self.shutdown_called = True
        self._initialized = False

    def terminal_info(self) -> SimpleNamespace | None:
        if not self._initialized:
            return None
        return SimpleNamespace(
            company="FakeBroker",
            build=4000,
            connected=self._terminal_connected,
        )

    def account_info(self) -> SimpleNamespace | None:
        return self.account

    def symbol_info(self, name: str) -> SimpleNamespace | None:
        return self._symbols.get(name)

    def symbol_info_tick(self, name: str) -> SimpleNamespace | None:
        return self._ticks.get(name)

    def symbol_select(self, name: str, enable: bool) -> bool:
        self.symbol_select_calls.append((name, enable))
        return name in self._symbols

    def symbols_get(self) -> list[SimpleNamespace]:
        return list(self._symbols.values())

    def positions_get(
        self,
        *,
        symbol: str | None = None,
        ticket: int | None = None,
    ) -> tuple | None:
        result = self._positions
        if symbol is not None:
            result = [p for p in result if p.symbol == symbol]
        if ticket is not None:
            result = [p for p in result if p.ticket == ticket]
        return tuple(result) if result else ()

    def order_check(self, request: dict) -> SimpleNamespace | None:
        self.order_check_calls.append(dict(request))
        if self._order_check_queue:
            r = self._order_check_queue.popleft()
            if callable(r):
                return r(request)
            return r
        return self.order_check_default(request)

    def order_send(self, request: dict) -> SimpleNamespace | None:
        self.order_send_calls.append(dict(request))
        if self._order_send_queue:
            r = self._order_send_queue.popleft()
            if callable(r):
                return r(request)
            return r
        return self.order_send_default(request)

    def order_calc_margin(
        self, order_type: int, symbol: str, lots: float, price: float,
    ) -> float:
        return 1000.0 * lots

    def copy_rates_from_pos(self, symbol, timeframe, start, count):
        return None

    def last_error(self) -> tuple[int, str]:
        return self.last_error_value


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_mt5() -> FakeMT5:
    """A fresh FakeMT5 with no symbols, ticks, or positions configured."""
    return FakeMT5()


@pytest.fixture
def mt5_bridge(monkeypatch, fake_mt5, tmp_path):
    """
    Returns the utils.mt5_bridge module patched with our FakeMT5 instance.

    Also redirects the learned-stops persistence file to a per-test temp path
    so tests don't pollute data/stop_minimums.json.
    """
    # Force a clean re-import path
    import utils.mt5_bridge as bridge

    monkeypatch.setattr(bridge, "mt5", fake_mt5)
    monkeypatch.setattr(bridge, "MT5_AVAILABLE", True)
    monkeypatch.setattr(bridge, "_STOP_LEARNING_PATH", tmp_path / "stop_minimums.json")

    # _TF_MAP is a module-level cache populated lazily — clear so the next
    # _tf() call rebuilds against fake_mt5's constants
    bridge._TF_MAP.clear()

    return bridge, fake_mt5


@pytest.fixture
def mt5_unavailable(monkeypatch):
    """Bridge with MT5_AVAILABLE=False (MetaTrader5 package not installed)."""
    import utils.mt5_bridge as bridge
    monkeypatch.setattr(bridge, "mt5", None)
    monkeypatch.setattr(bridge, "MT5_AVAILABLE", False)
    return bridge
