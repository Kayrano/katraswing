"""
MT5 Signal Bridge — sends Katraswing signals to MetaTrader 5.

Uses the official MetaTrader5 Python library (Windows only, same machine as MT5 terminal).
Install: pip install MetaTrader5

Architecture:
    Katraswing SignalResult → send_signal() → mt5.order_send() → MT5 terminal

Symbol mapping: Katraswing uses yfinance tickers (NQ=F, ES=F, XAUUSD=X).
MT5 symbol names vary by broker — edit SYMBOL_MAP to match your broker's naming.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False

# ── Symbol mapping: yfinance ticker → your MT5 broker symbol ─────────────────
# Edit these to match your broker's exact symbol names.
SYMBOL_MAP: dict[str, str] = {
    "NQ=F":     "#US100_M26",    # NQ Mini Futures (CME)
    "ES=F":     "#US500_M26",          # ES Mini Futures (CME)
    "EURUSD=X": "EURUSD",
    "GBPUSD=X": "GBPUSD",
    "USDJPY=X": "USDJPY"
}

# Default lot sizes by MT5 symbol (adjust for your account/risk tolerance)
DEFAULT_LOTS: dict[str, float] = {
    "#US100_M26": 0.20,
    "#US500_M26": 0.20,
    "EURUSD":     0.1,
    "GBPUSD":     0.1,
    "USDJPY":     0.1,
}

MAGIC_NUMBER = 234100   # unique ID for Katraswing orders


@dataclass
class MT5OrderResult:
    success: bool
    ticket: int
    symbol: str
    direction: str
    volume: float
    entry: float
    sl: float
    tp: float
    error: str = ""


@dataclass
class MT5Position:
    ticket: int
    symbol: str
    direction: str   # "LONG" | "SHORT"
    volume: float
    open_price: float
    sl: float
    tp: float
    profit: float
    magic: int


# ── Connection management ─────────────────────────────────────────────────────

def is_available() -> bool:
    """True if the MetaTrader5 package is installed."""
    return MT5_AVAILABLE


def connect(path: str = "") -> bool:
    """
    Initialize connection to the running MT5 terminal.
    path: optional full path to terminal64.exe (leave blank for auto-detect).
    Returns True on success.
    """
    if not MT5_AVAILABLE:
        logger.error(
            "MetaTrader5 package not installed. "
            "Run: pip install MetaTrader5  (Windows only, same machine as MT5 terminal)"
        )
        return False

    kwargs = {"path": path} if path else {}
    if not mt5.initialize(**kwargs):
        err = mt5.last_error()
        logger.error(f"MT5 initialize() failed: {err}")
        return False

    info = mt5.terminal_info()
    logger.info(
        f"Connected to MT5 terminal: {info.company} | "
        f"build {info.build} | connected={info.connected}"
    )
    return True


def disconnect():
    """Shutdown the MT5 connection."""
    if MT5_AVAILABLE and mt5 is not None:
        mt5.shutdown()
        logger.info("MT5 disconnected.")


def is_connected() -> bool:
    """Check if MT5 terminal is alive and connected to a broker."""
    if not MT5_AVAILABLE:
        return False
    info = mt5.terminal_info()
    return info is not None and info.connected


def ensure_connected(path: str = "") -> bool:
    """Re-connect if the connection dropped."""
    if is_connected():
        return True
    logger.warning("MT5 connection lost — attempting reconnect...")
    return connect(path)


# ── Order execution ───────────────────────────────────────────────────────────

def send_signal(
    ticker: str,
    direction: str,
    entry: float,
    sl: float,
    tp: float,
    lots: Optional[float] = None,
    magic: int = MAGIC_NUMBER,
    comment: str = "Katraswing",
) -> MT5OrderResult:
    """
    Send a market order to MT5.

    Args:
        ticker:    Katraswing/yfinance ticker (e.g. "NQ=F")
        direction: "LONG" or "SHORT"
        entry:     signal entry price (used for logging; MT5 uses live ask/bid)
        sl:        stop loss price
        tp:        take profit price
        lots:      contract volume (defaults to DEFAULT_LOTS per symbol)
        magic:     EA magic number for order tracking
        comment:   order comment visible in MT5 terminal
    """
    if not MT5_AVAILABLE:
        return MT5OrderResult(
            False, 0, ticker, direction, 0.0, entry, sl, tp,
            "MetaTrader5 package not installed. Run: pip install MetaTrader5"
        )

    if not is_connected():
        return MT5OrderResult(False, 0, ticker, direction, 0.0, entry, sl, tp, "MT5 not connected")

    symbol = SYMBOL_MAP.get(ticker.upper(), ticker)

    # Make the symbol visible in Market Watch
    if not mt5.symbol_select(symbol, True):
        return MT5OrderResult(
            False, 0, symbol, direction, 0.0, entry, sl, tp,
            f"Cannot select symbol '{symbol}' — check SYMBOL_MAP matches your broker's naming"
        )

    # Get live tick
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return MT5OrderResult(
            False, 0, symbol, direction, 0.0, entry, sl, tp,
            f"No live tick data for '{symbol}'"
        )

    sym_info = mt5.symbol_info(symbol)
    digits   = sym_info.digits if sym_info else 5
    vol      = lots if lots is not None else DEFAULT_LOTS.get(symbol, 0.1)

    order_type = mt5.ORDER_TYPE_BUY if direction == "LONG" else mt5.ORDER_TYPE_SELL
    price      = tick.ask if direction == "LONG" else tick.bid

    request = {
        "action":      mt5.TRADE_ACTION_DEAL,
        "symbol":      symbol,
        "volume":      float(vol),
        "type":        order_type,
        "price":       price,
        "sl":          round(sl, digits),
        "tp":          round(tp, digits),
        "deviation":   20,
        "magic":       magic,
        "comment":     comment,
        "type_time":   mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    # Validate before sending
    check = mt5.order_check(request)
    if check is None or check.retcode != 0:
        err = f"order_check failed: retcode={check.retcode if check else 'None'}"
        logger.warning(err)
        # Proceed anyway — some brokers skip validation

    result = mt5.order_send(request)
    if result is None:
        err = str(mt5.last_error())
        logger.error(f"order_send returned None: {err}")
        return MT5OrderResult(False, 0, symbol, direction, vol, price, sl, tp, err)

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(
            f"✓ Order #{result.order} | {direction} {symbol} "
            f"vol={vol} @ {price:.{digits}f} | SL={sl:.{digits}f} TP={tp:.{digits}f}"
        )
        return MT5OrderResult(True, result.order, symbol, direction, vol, price, sl, tp)

    err = f"retcode={result.retcode} | {result.comment}"
    logger.warning(f"Order rejected: {err}")
    return MT5OrderResult(False, 0, symbol, direction, vol, price, sl, tp, err)


def send_from_signal_result(signal_result, lots: Optional[float] = None) -> MT5OrderResult:
    """
    Convenience wrapper: accepts a Katraswing SignalResult and sends to MT5.
    Returns MT5OrderResult. Call only when signal_result.direction in ("LONG","SHORT").
    """
    from agents.signal_engine import SignalResult  # local import to avoid circular
    sr: SignalResult = signal_result

    if sr.direction not in ("LONG", "SHORT"):
        return MT5OrderResult(
            False, 0, sr.ticker, sr.direction, 0.0, sr.entry, sr.sl, sr.tp,
            f"No trade signal (direction={sr.direction})"
        )

    return send_signal(
        ticker=sr.ticker,
        direction=sr.direction,
        entry=sr.entry,
        sl=sr.sl,
        tp=sr.tp,
        lots=lots,
        comment=f"Katraswing {sr.confidence:.0%}",
    )


# ── Position management ───────────────────────────────────────────────────────

def get_open_positions(magic: int = MAGIC_NUMBER) -> list[MT5Position]:
    """Return all open Katraswing positions."""
    if not MT5_AVAILABLE or not is_connected():
        return []

    raw = mt5.positions_get()
    if raw is None:
        return []

    positions = []
    for p in raw:
        if p.magic != magic:
            continue
        direction = "LONG" if p.type == mt5.ORDER_TYPE_BUY else "SHORT"
        positions.append(MT5Position(
            ticket=p.ticket,
            symbol=p.symbol,
            direction=direction,
            volume=p.volume,
            open_price=p.price_open,
            sl=p.sl,
            tp=p.tp,
            profit=p.profit,
            magic=p.magic,
        ))
    return positions


def close_position(ticket: int, magic: int = MAGIC_NUMBER) -> bool:
    """Close a single open position by ticket number."""
    if not MT5_AVAILABLE or not is_connected():
        return False

    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        logger.warning(f"Position #{ticket} not found.")
        return False

    pos = positions[0]
    close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    tick  = mt5.symbol_info_tick(pos.symbol)
    price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

    req = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "position":     ticket,
        "symbol":       pos.symbol,
        "volume":       pos.volume,
        "type":         close_type,
        "price":        price,
        "deviation":    20,
        "magic":        magic,
        "comment":      "Katraswing close",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(req)
    ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
    if ok:
        logger.info(f"Closed position #{ticket} ({pos.symbol})")
    else:
        logger.warning(f"Failed to close #{ticket}: {result.comment if result else 'None'}")
    return ok


def close_all_positions(magic: int = MAGIC_NUMBER):
    """Close all open Katraswing positions."""
    for pos in get_open_positions(magic):
        close_position(pos.ticket, magic)
