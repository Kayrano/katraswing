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

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5  # type: ignore[import]
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False

# ── Symbol mapping: yfinance ticker → your MT5 broker symbol ─────────────────
# Edit these to match your broker's exact symbol names.
SYMBOL_MAP: dict[str, str] = {
    # Futures
    "NQ=F":     "#US100_M26",   # Nasdaq 100 E-mini (CME) — verify name with FxPro
    "ES=F":     "#US500_M26",   # S&P 500 E-mini (CME)   — verify name with FxPro
    "NKD=F":    "JP225",        # Nikkei 225 Futures      — verify name with FxPro
    # Stocks (CFD)
    "AAPL":     "#AAPL",        # Apple Inc.
    "MSFT":     "#MSFT",        # Microsoft Corp.
    "AMZN":     "#AMZN",        # Amazon.com Inc.
    # Forex
    "EURUSD=X": "EURUSD",
    "GBPUSD=X": "GBPUSD",
    "USDJPY=X": "USDJPY",
}

# Default lot sizes by MT5 symbol (adjust for your account/risk tolerance)
DEFAULT_LOTS: dict[str, float] = {
    "#US100_M26": 0.20,
    "#US500_M26": 0.20,
    "JP225":      0.10,
    "#AAPL":      1.0,
    "#MSFT":      1.0,
    "#AMZN":      1.0,
    "EURUSD":     0.1,
    "GBPUSD":     0.1,
    "USDJPY":     0.1,
}

MAGIC_NUMBER = 234100   # unique ID for Katraswing orders

# Persisted per-symbol stop minimums learned from live broker feedback
_STOP_LEARNING_PATH = Path(__file__).parent.parent / "data" / "stop_minimums.json"


def _load_learned_min(symbol: str) -> float:
    """Return the last successful effective_min for this symbol, or 0.0."""
    try:
        if _STOP_LEARNING_PATH.exists():
            data = json.loads(_STOP_LEARNING_PATH.read_text(encoding="utf-8"))
            return float(data.get(symbol, {}).get("min_pts", 0.0))
    except Exception:
        pass
    return 0.0


def _save_learned_min(symbol: str, min_pts: float) -> None:
    """Save the effective_min that worked for this symbol (only increases)."""
    try:
        data: dict = {}
        if _STOP_LEARNING_PATH.exists():
            data = json.loads(_STOP_LEARNING_PATH.read_text(encoding="utf-8"))
        existing = float(data.get(symbol, {}).get("min_pts", 0.0))
        if min_pts > existing:
            data[symbol] = {
                "min_pts": round(min_pts, 8),
                "updated": datetime.utcnow().isoformat(timespec="seconds"),
            }
            _STOP_LEARNING_PATH.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            logger.info(f"Learned min stop for {symbol}: {min_pts:.5f}")
    except Exception as exc:
        logger.debug(f"_save_learned_min: {exc}")


def _find_safe_stops(
    symbol: str,
    price: float,
    direction: str,
    sl_dist: float,
    tp_dist: float,
    effective_min: float,
    sym_info,
    fill_type: int,
    provisional_lots: float,
    order_type: int,
    digits: int,
) -> tuple[float, float, float]:
    """
    Pre-validate SL/TP with mt5.order_check() before ever calling order_send.
    Expands effective_min by 20% each iteration until the broker accepts or
    10 attempts are exhausted.  Returns (sl, tp, effective_min_used).
    """
    if not MT5_AVAILABLE:
        # No MT5 — just apply the geometric minimum and return
        cur_sl_dist = max(sl_dist, effective_min)
        cur_tp_dist = max(tp_dist, effective_min)
        if direction == "LONG":
            return (round(price - cur_sl_dist, digits),
                    round(price + cur_tp_dist, digits), effective_min)
        return (round(price + cur_sl_dist, digits),
                round(price - cur_tp_dist, digits), effective_min)

    for attempt in range(10):
        cur_sl_dist = max(sl_dist, effective_min)
        cur_tp_dist = max(tp_dist, effective_min)
        if direction == "LONG":
            trial_sl = round(price - cur_sl_dist, digits)
            trial_tp = round(price + cur_tp_dist, digits)
        else:
            trial_sl = round(price + cur_sl_dist, digits)
            trial_tp = round(price - cur_tp_dist, digits)

        check = mt5.order_check({
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       float(provisional_lots),
            "type":         order_type,
            "price":        price,
            "sl":           trial_sl,
            "tp":           trial_tp,
            "deviation":    20,
            "type_filling": fill_type,
        })

        if check is None:
            logger.debug(f"order_check returned None for {symbol} — proceeding with current stops")
            return trial_sl, trial_tp, effective_min

        if check.retcode != 10016:
            if attempt > 0:
                logger.info(
                    f"Stop precheck: {symbol} accepted after {attempt} expansion(s) "
                    f"(effective_min={effective_min:.5f})"
                )
            return trial_sl, trial_tp, effective_min

        logger.debug(
            f"Stop precheck attempt {attempt + 1}: {symbol} 10016 — "
            f"expanding min {effective_min:.5f} → {effective_min * 1.2:.5f}"
        )
        effective_min *= 1.2

    # Max iterations hit — use the last expanded values
    logger.warning(f"Stop precheck: max iterations for {symbol}, min={effective_min:.5f}")
    cur_sl_dist = max(sl_dist, effective_min)
    cur_tp_dist = max(tp_dist, effective_min)
    if direction == "LONG":
        return (round(price - cur_sl_dist, digits),
                round(price + cur_tp_dist, digits), effective_min)
    return (round(price + cur_sl_dist, digits),
            round(price - cur_tp_dist, digits), effective_min)


def _filling_mode(sym_info) -> int:
    """
    Return the best ORDER_FILLING_* constant supported by this symbol.
    Many brokers reject IOC (value 1) — query the symbol's filling_mode bitmask
    and pick the first supported mode rather than hardcoding IOC.
      filling_mode bit 0 (value 1) → FOK supported  → ORDER_FILLING_FOK = 0
      filling_mode bit 1 (value 2) → IOC supported  → ORDER_FILLING_IOC = 1
      neither bit set               → only RETURN    → ORDER_FILLING_RETURN = 2
    """
    if sym_info is None or not MT5_AVAILABLE:
        return 2  # ORDER_FILLING_RETURN as safe fallback
    fm = getattr(sym_info, "filling_mode", 0)
    if fm & 2:
        return 1  # ORDER_FILLING_IOC
    if fm & 1:
        return 0  # ORDER_FILLING_FOK
    return 2       # ORDER_FILLING_RETURN


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
    swap: float = 0.0
    commission: float = 0.0
    time_open: int = 0          # Unix timestamp when position was opened
    price_current: float = 0.0  # Live current price from MT5 (saves extra tick fetch)


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


# ── Symbol discovery ─────────────────────────────────────────────────────────

def get_tradeable_symbols() -> list[dict]:
    """
    Return all symbols available in the broker that allow trading.
    Each dict: {name, description, category}.
    Returns [] if MT5 not connected.
    """
    if not MT5_AVAILABLE or not is_connected():
        return []
    symbols = mt5.symbols_get()
    if not symbols:
        return []
    result = []
    for s in symbols:
        if s.trade_mode == 0:      # SYMBOL_TRADE_MODE_DISABLED
            continue
        path = getattr(s, "path", "") or ""
        category = path.split("\\")[0] if "\\" in path else path
        result.append({
            "name":        s.name,
            "description": getattr(s, "description", s.name),
            "category":    category,
        })
    return sorted(result, key=lambda x: x["name"])


# ── Market data ──────────────────────────────────────────────────────────────

_TF_MAP: dict[str, int] = {}   # populated lazily once mt5 is available


def _tf(interval: str) -> int:
    """Map interval string to MT5 TIMEFRAME constant."""
    if not MT5_AVAILABLE:
        raise RuntimeError("MetaTrader5 not installed")
    if not _TF_MAP:
        _TF_MAP.update({
            "1m":  mt5.TIMEFRAME_M1,
            "5m":  mt5.TIMEFRAME_M5,
            "15m": mt5.TIMEFRAME_M15,
            "30m": mt5.TIMEFRAME_M30,
            "1h":  mt5.TIMEFRAME_H1,
            "4h":  mt5.TIMEFRAME_H4,
            "1d":  mt5.TIMEFRAME_D1,
        })
    tf = _TF_MAP.get(interval)
    if tf is None:
        raise ValueError(f"Unsupported interval: {interval!r}")
    return tf


def fetch_bars(
    symbol: str,
    interval: str = "5m",
    count: int = 5000,
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV bars from MT5 terminal.

    Returns a DataFrame with columns Open, High, Low, Close, Volume
    and a UTC-aware DatetimeIndex, or None on failure.

    Args:
        symbol:   MT5 symbol name (e.g. "EURUSD", "#US100_M26")
        interval: "1m" | "5m" | "15m" | "30m" | "1h" | "4h" | "1d"
        count:    number of bars to fetch (from most recent bar backwards)
    """
    if not MT5_AVAILABLE or not is_connected():
        return None

    try:
        timeframe = _tf(interval)
    except ValueError as exc:
        logger.warning(str(exc))
        return None

    if not mt5.symbol_select(symbol, True):
        logger.debug(f"fetch_bars: symbol '{symbol}' not available in broker — falling back to yfinance")
        return None

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        logger.warning(f"fetch_bars: no data for {symbol} {interval}")
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")
    df = df.rename(columns={
        "open":        "Open",
        "high":        "High",
        "low":         "Low",
        "close":       "Close",
        "tick_volume": "Volume",
    })
    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    return df[keep].copy()


def is_market_open(symbol: str) -> tuple[bool, str]:
    """
    Check whether a symbol is currently tradeable in MT5.

    Returns (is_closed, reason_string) — matches the (closed, msg) tuple
    used by app.py's _market_status().

    Trade modes:
        SYMBOL_TRADE_MODE_DISABLED  (0) → hard closed
        SYMBOL_TRADE_MODE_CLOSEONLY (3) → session ending / closed
        SYMBOL_TRADE_MODE_FULL      (4) → open
        others (1=longonly, 2=shortonly) → partially open → treat as open
    """
    if not MT5_AVAILABLE or not is_connected():
        return False, ""   # unknown → don't block; let the old clock logic decide

    info = mt5.symbol_info(symbol)
    if info is None:
        return False, ""   # unknown symbol — don't block

    mode = info.trade_mode
    if mode == 0:   # DISABLED
        return True, f"{symbol} · market closed (broker)"
    if mode == 3:   # CLOSE_ONLY
        return True, f"{symbol} · session closing (close-only)"

    # Cross-check: if the last tick is stale (> 10 min) treat as closed
    tick = mt5.symbol_info_tick(symbol)
    if tick is not None:
        age = time.time() - tick.time
        if age > 600:
            mins = int(age / 60)
            return True, f"{symbol} · no quotes for {mins}m (market closed)"

    return False, ""   # market open


# ── Account info ─────────────────────────────────────────────────────────────

def get_account_info() -> dict:
    """
    Return live account metrics from MT5.
    Fields: balance, equity, margin, free_margin, margin_level, currency, leverage
    Returns empty dict if not connected.
    """
    if not MT5_AVAILABLE or not is_connected():
        return {}
    info = mt5.account_info()
    if info is None:
        return {}
    ml = info.margin_level if info.margin > 0 else None
    return {
        "balance":      round(info.balance,      2),
        "equity":       round(info.equity,        2),
        "margin":       round(info.margin,        2),
        "free_margin":  round(info.margin_free,   2),
        "margin_level": round(ml, 1) if ml else None,   # percent, e.g. 1523.4
        "currency":     info.currency,
        "leverage":     info.leverage,
    }


def calc_lot_size(
    symbol: str,
    direction: str,
    entry: float,
    sl: float,
    risk_pct: float,
    account_balance: Optional[float] = None,
) -> float:
    """
    Calculate the correct lot size so that hitting the SL loses at most risk_pct%
    of the account balance.

    Formula:
        dollar_risk   = balance × (risk_pct / 100)
        sl_distance   = |entry − sl|   (in price units)
        tick_value    = symbol_info.trade_tick_value   ($ per tick per 1 lot)
        tick_size     = symbol_info.trade_tick_size    (price units per tick)
        value_per_lot = (sl_distance / tick_size) × tick_value
        lots          = dollar_risk / value_per_lot

    Clamps result within [volume_min, volume_max] per broker rules.
    Falls back to DEFAULT_LOTS if any data is unavailable.
    """
    if not MT5_AVAILABLE or not is_connected():
        return DEFAULT_LOTS.get(symbol, 0.1)

    si = mt5.symbol_info(symbol)
    if si is None:
        return DEFAULT_LOTS.get(symbol, 0.1)

    if account_balance is None:
        ai = mt5.account_info()
        account_balance = ai.balance if ai else None
    if not account_balance or account_balance <= 0:
        return DEFAULT_LOTS.get(symbol, 0.1)

    sl_dist     = abs(entry - sl)
    tick_size   = si.trade_tick_size   or 0.00001
    tick_value  = si.trade_tick_value  or 1.0

    if sl_dist <= 0 or tick_size <= 0 or tick_value <= 0:
        return DEFAULT_LOTS.get(symbol, 0.1)

    dollar_risk     = account_balance * (risk_pct / 100.0)
    value_per_lot   = (sl_dist / tick_size) * tick_value
    raw_lots        = dollar_risk / value_per_lot

    # Clamp to broker's allowed range and step
    vol_min  = si.volume_min  or 0.01
    vol_max  = si.volume_max  or 100.0
    vol_step = si.volume_step or 0.01
    lots = max(vol_min, min(vol_max, raw_lots))
    # Round down to nearest volume_step
    lots = round(int(lots / vol_step) * vol_step, 8)

    logger.debug(
        f"calc_lot_size {symbol}: balance={account_balance:.2f}, "
        f"risk={risk_pct}%, sl_dist={sl_dist}, tick={tick_value}/{tick_size}, "
        f"raw_lots={raw_lots:.4f} → lots={lots}"
    )
    return lots


# ── Order execution ───────────────────────────────────────────────────────────

_RETCODE_HINTS = {
    10027: "AutoTrading disabled — click the 'Algo Trading' button in the MT5 toolbar.",
    10018: "Market closed for this symbol.",
    10019: "Not enough money in account.",
    10014: "Invalid volume — check lot size.",
    10016: "Invalid stops — SL/TP may be too close to price.",
    10030: "Filling mode not supported — broker rejected the order type.",
}


def send_signal(
    ticker: str,
    direction: str,
    entry: float,
    sl: float,
    tp: float,
    lots: Optional[float] = None,
    risk_pct: float = 1.0,
    magic: int = MAGIC_NUMBER,
    comment: str = "Katraswing",
) -> MT5OrderResult:
    """
    Send a market order to MT5 with full pre-flight safety checks:
      1. Duplicate guard  — skip if a position already exists for this symbol
      2. Lot sizing       — compute from account balance + SL distance (risk_pct%)
      3. Margin check     — reject if free margin < required margin × 1.5 safety buffer
      4. Stop rebasing    — SL/TP recalculated from live price; min-stop enforced

    Args:
        ticker:   yfinance/Katraswing ticker OR MT5 symbol name
        risk_pct: percent of account balance to risk per trade (used when lots=None)
    """
    if not MT5_AVAILABLE:
        return MT5OrderResult(False, 0, ticker, direction, 0.0, entry, sl, tp,
                              "MetaTrader5 not installed")
    if not is_connected():
        return MT5OrderResult(False, 0, ticker, direction, 0.0, entry, sl, tp,
                              "MT5 not connected")

    symbol = SYMBOL_MAP.get(ticker.upper(), ticker)

    if not mt5.symbol_select(symbol, True):
        return MT5OrderResult(False, 0, symbol, direction, 0.0, entry, sl, tp,
                              f"Symbol '{symbol}' not found in broker — update SYMBOL_MAP")

    # ── 1. Duplicate guard ────────────────────────────────────────────────────
    existing = mt5.positions_get(symbol=symbol)
    if existing:
        dirs = ["LONG" if p.type == mt5.ORDER_TYPE_BUY else "SHORT" for p in existing]
        if direction in dirs:
            return MT5OrderResult(False, 0, symbol, direction, 0.0, entry, sl, tp,
                                  f"Duplicate: {direction} position already open for {symbol}")
        # Opposite position exists — still allow (user may hedge intentionally)

    # ── Live tick & symbol info ───────────────────────────────────────────────
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return MT5OrderResult(False, 0, symbol, direction, 0.0, entry, sl, tp,
                              f"No live tick for '{symbol}'")

    sym_info  = mt5.symbol_info(symbol)
    digits    = sym_info.digits if sym_info else 5
    fill_type = _filling_mode(sym_info)

    order_type = mt5.ORDER_TYPE_BUY  if direction == "LONG" else mt5.ORDER_TYPE_SELL
    price      = tick.ask if direction == "LONG" else tick.bid

    # ── 2. Rebase SL/TP from live price ──────────────────────────────────────
    sl_dist = abs(entry - sl)
    tp_dist = abs(tp - entry)
    if direction == "LONG":
        live_sl = price - sl_dist
        live_tp = price + tp_dist
    else:
        live_sl = price + sl_dist
        live_tp = price - tp_dist

    # Compute effective minimum stop distance, blending three sources:
    #   • broker's stated stops level (trade_stops_level × point)
    #   • 2× current spread + 5 points (spread-aware safety buffer)
    #   • previously learned minimum for this symbol (from successful trades)
    # Then pre-validate with order_check loop so we never send a doomed order.
    point = (sym_info.point or 0.00001) if sym_info else 0.00001
    spread = abs(tick.ask - tick.bid)
    stops_dist = (sym_info.trade_stops_level * point) if sym_info else 0.0
    learned_min = _load_learned_min(symbol)
    effective_min = max(stops_dist, spread * 2.0 + point * 5, learned_min)

    provisional_lots = lots if lots is not None else DEFAULT_LOTS.get(symbol, 0.1)
    live_sl, live_tp, effective_min = _find_safe_stops(
        symbol, price, direction, sl_dist, tp_dist,
        effective_min, sym_info, fill_type, provisional_lots, order_type, digits,
    )

    # ── 3. Lot sizing from account balance ────────────────────────────────────
    acct   = mt5.account_info()
    balance = acct.balance if acct else None

    if lots is None:
        lots = calc_lot_size(symbol, direction, price, live_sl, risk_pct, balance)

    # ── 4. Margin pre-check ───────────────────────────────────────────────────
    order_type_for_margin = (mt5.ORDER_TYPE_BUY if direction == "LONG"
                             else mt5.ORDER_TYPE_SELL)
    required_margin = mt5.order_calc_margin(order_type_for_margin, symbol, lots, price)
    if required_margin is not None and required_margin > 0 and acct is not None:
        free_margin = acct.margin_free
        # Require 1.5× buffer so margin level stays above ~200% after entry
        if free_margin < required_margin * 1.5:
            return MT5OrderResult(
                False, 0, symbol, direction, lots, price, live_sl, live_tp,
                f"Insufficient margin: need {required_margin * 1.5:.2f}, "
                f"have {free_margin:.2f} free (balance {balance:.2f})"
            )

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       float(lots),
        "type":         order_type,
        "price":        price,
        "sl":           round(live_sl, digits),
        "tp":           round(live_tp, digits),
        "deviation":    20,
        "magic":        magic,
        "comment":      comment,
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": fill_type,
    }

    result = mt5.order_send(request)
    if result is None:
        err = str(mt5.last_error())
        logger.error(f"order_send returned None: {err}")
        return MT5OrderResult(False, 0, symbol, direction, lots, price, live_sl, live_tp, err)

    # If 10016 still fires despite pre-validation (rare: price moved between check and send),
    # do one final retry with 2× the effective_min.
    # Regardless of whether the retry succeeds, save the expanded min so the next
    # trade on this symbol starts from a proven-sufficient floor.
    if result.retcode == 10016:
        fallback_min = effective_min * 2.0
        _save_learned_min(symbol, fallback_min)   # learn from failure immediately
        fb_sl_dist = max(sl_dist, fallback_min)
        fb_tp_dist = max(tp_dist, fallback_min)
        if direction == "LONG":
            live_sl = round(price - fb_sl_dist, digits)
            live_tp = round(price + fb_tp_dist, digits)
        else:
            live_sl = round(price + fb_sl_dist, digits)
            live_tp = round(price - fb_tp_dist, digits)
        request["sl"] = live_sl
        request["tp"] = live_tp
        effective_min = fallback_min
        logger.warning(
            f"10016 after precheck — price moved? Retrying with 2× min="
            f"{fallback_min:.5f}: SL={live_sl} TP={live_tp}"
        )
        result = mt5.order_send(request)
        if result is None:
            return MT5OrderResult(False, 0, symbol, direction, lots, price, live_sl, live_tp,
                                  str(mt5.last_error()))

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        _save_learned_min(symbol, effective_min)   # remember working min for next time
        logger.info(
            f"✓ #{result.order} | {direction} {symbol} vol={lots} @ {price:.{digits}f} "
            f"SL={live_sl:.{digits}f} TP={live_tp:.{digits}f} margin≈{required_margin}"
        )
        return MT5OrderResult(True, result.order, symbol, direction, lots, price,
                              live_sl, live_tp)

    hint = _RETCODE_HINTS.get(result.retcode, "")
    err  = f"retcode={result.retcode} | {result.comment}" + (f" → {hint}" if hint else "")
    logger.warning(f"Order rejected: {err}")
    return MT5OrderResult(False, 0, symbol, direction, lots, price, live_sl, live_tp, err)


def send_from_signal_result(
    signal_result,
    lots: Optional[float] = None,
    risk_pct: float = 1.0,
) -> MT5OrderResult:
    """
    Convenience wrapper: accepts a Katraswing SignalResult and sends to MT5.
    Pass risk_pct to auto-size lots from account balance.
    """
    from agents.signal_engine import SignalResult  # local import to avoid circular
    sr: SignalResult = signal_result

    if sr.direction not in ("LONG", "SHORT"):
        return MT5OrderResult(False, 0, sr.ticker, sr.direction, 0.0, sr.entry, sr.sl, sr.tp,
                              f"No trade signal (direction={sr.direction})")

    # Scale risk_pct by signal risk level: LOW confidence = HIGH risk = smaller position
    _RISK_MULTIPLIER = {"LOW": 1.5, "MEDIUM": 1.0, "HIGH": 0.5}
    effective_risk_pct = risk_pct * _RISK_MULTIPLIER.get(getattr(sr, "risk_level", "MEDIUM"), 1.0)

    # Use mt5_symbol when broker-resolved; refuse to send if it's a yfinance ticker
    # (=X / =F / ^ / -USD), since the broker won't know that symbol name.
    ticker = getattr(sr, "mt5_symbol", "") or ""
    if not ticker or any(c in ticker for c in ("=", "^")) or ticker.endswith("-USD"):
        # Fall back to SYMBOL_MAP lookup on sr.ticker
        mapped = SYMBOL_MAP.get((sr.ticker or "").upper())
        if not mapped:
            return MT5OrderResult(
                False, 0, sr.ticker, sr.direction, 0.0, sr.entry, sr.sl, sr.tp,
                f"No broker symbol mapped for '{sr.ticker}' — waiting for MT5 connect"
            )
        ticker = mapped

    # Sanitised comment — broker rejects '%' and other punctuation; cap 24 chars.
    strat = (sr.chart_signals[0].strategy if sr.chart_signals else "")
    raw_cmt = f"KS {int(round(sr.confidence * 100))} {getattr(sr, 'risk_level', 'M')[:1]} {strat}"
    safe_cmt = "".join(ch for ch in raw_cmt if ch.isalnum() or ch in " _")[:24].strip()

    return send_signal(
        ticker=ticker,
        direction=sr.direction,
        entry=sr.entry,
        sl=sr.sl,
        tp=sr.tp,
        lots=lots,
        risk_pct=effective_risk_pct,
        comment=safe_cmt,
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
            swap=float(getattr(p, "swap", 0.0)),
            commission=float(getattr(p, "commission", 0.0)),
            time_open=int(getattr(p, "time", 0)),
            price_current=float(getattr(p, "price_current", p.price_open)),
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
    tick     = mt5.symbol_info_tick(pos.symbol)
    sym_info = mt5.symbol_info(pos.symbol)
    price    = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

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
        "type_filling": _filling_mode(sym_info),
    }
    result = mt5.order_send(req)
    ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
    if ok:
        logger.info(f"Closed position #{ticket} ({pos.symbol})")
    else:
        logger.warning(f"Failed to close #{ticket}: {result.comment if result else 'None'}")
    return ok


def modify_position(
    ticket: int,
    new_sl: float | None = None,
    new_tp: float | None = None,
) -> bool:
    """
    Modify SL and/or TP on an open position using TRADE_ACTION_SLTP.
    Validates new_sl against the learned broker minimum before sending.
    """
    if not MT5_AVAILABLE or not is_connected():
        return False

    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        logger.warning(f"modify_position: #{ticket} not found")
        return False

    pos      = positions[0]
    sym_info = mt5.symbol_info(pos.symbol)
    digits   = sym_info.digits if sym_info else 5
    tick     = mt5.symbol_info_tick(pos.symbol)

    if new_sl is not None and tick is not None:
        price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
        learned_min = _load_learned_min(pos.symbol)
        point       = (sym_info.point or 0.00001) if sym_info else 0.00001
        spread      = abs(tick.ask - tick.bid)
        stops_dist  = (sym_info.trade_stops_level * point) if sym_info else 0.0
        effective_min = max(stops_dist, spread * 2.0 + point * 5, learned_min)
        if pos.type == mt5.ORDER_TYPE_BUY:
            new_sl = min(new_sl, price - effective_min)
        else:
            new_sl = max(new_sl, price + effective_min)
        new_sl = round(new_sl, digits)

    request = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "symbol":   pos.symbol,
        "sl":       round(new_sl, digits) if new_sl is not None else pos.sl,
        "tp":       round(new_tp, digits) if new_tp is not None else pos.tp,
    }
    result = mt5.order_send(request)
    ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
    if ok:
        logger.info(f"Modified #{ticket}: SL={request['sl']} TP={request['tp']}")
    else:
        logger.warning(f"modify_position #{ticket} failed: {result.comment if result else 'None'}")
    return ok


def partial_close_position(ticket: int, volume: float, magic: int = MAGIC_NUMBER) -> bool:
    """Close `volume` lots of an open position (partial close)."""
    if not MT5_AVAILABLE or not is_connected():
        return False

    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        logger.warning(f"partial_close: #{ticket} not found")
        return False

    pos        = positions[0]
    close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    tick       = mt5.symbol_info_tick(pos.symbol)
    sym_info   = mt5.symbol_info(pos.symbol)
    price      = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

    # Clamp volume to broker step and position size
    vol_step = (sym_info.volume_step or 0.01) if sym_info else 0.01
    volume   = round(int(volume / vol_step) * vol_step, 8)
    volume   = min(volume, pos.volume)

    req = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "position":     ticket,
        "symbol":       pos.symbol,
        "volume":       float(volume),
        "type":         close_type,
        "price":        price,
        "deviation":    20,
        "magic":        magic,
        "comment":      "Katraswing partial close",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": _filling_mode(sym_info),
    }
    result = mt5.order_send(req)
    ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
    if ok:
        logger.info(f"Partial closed #{ticket} vol={volume} ({pos.symbol})")
    else:
        logger.warning(f"partial_close #{ticket} failed: {result.comment if result else 'None'}")
    return ok


def close_all_positions(magic: int = MAGIC_NUMBER):
    """Close all open Katraswing positions."""
    for pos in get_open_positions(magic):
        close_position(pos.ticket, magic)
