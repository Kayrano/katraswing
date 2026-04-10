"""
SQLite-based trade & run logger.
All bot activity is persisted to bot_trades.db in the Katraswing root.
Safe for multi-thread access (WAL mode).
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "bot_trades.db")


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    return con


def init_db() -> None:
    """Create tables if they don't already exist."""
    with _conn() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS trade_log (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ts           TEXT    NOT NULL,
                ticker       TEXT    NOT NULL,
                action       TEXT    NOT NULL,
                price        REAL    DEFAULT 0,
                qty          INTEGER DEFAULT 0,
                stop_loss    REAL    DEFAULT 0,
                take_profit  REAL    DEFAULT 0,
                score        REAL    DEFAULT 0,
                reason       TEXT    DEFAULT '',
                order_id     TEXT    DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS bot_runs (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                ts               TEXT    NOT NULL,
                market_open      INTEGER NOT NULL DEFAULT 0,
                tickers_scanned  INTEGER DEFAULT 0,
                trades_executed  INTEGER DEFAULT 0,
                positions_closed INTEGER DEFAULT 0,
                errors           INTEGER DEFAULT 0,
                daily_pnl        REAL    DEFAULT 0,
                notes            TEXT    DEFAULT ''
            );
        """)


def log_trade(
    ticker:      str,
    action:      str,           # BUY | SELL | SKIP | ERROR
    price:       float = 0.0,
    qty:         int   = 0,
    stop_loss:   float = 0.0,
    take_profit: float = 0.0,
    score:       float = 0.0,
    reason:      str   = "",
    order_id:    str   = "",
) -> None:
    with _conn() as con:
        con.execute(
            """INSERT INTO trade_log
               (ts, ticker, action, price, qty, stop_loss, take_profit, score, reason, order_id)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (datetime.utcnow().isoformat(timespec="seconds"),
             ticker.upper(), action, price, qty,
             stop_loss, take_profit, score, reason, order_id),
        )


def log_run(
    market_open:      bool  = False,
    tickers_scanned:  int   = 0,
    trades_executed:  int   = 0,
    positions_closed: int   = 0,
    errors:           int   = 0,
    daily_pnl:        float = 0.0,
    notes:            str   = "",
) -> None:
    with _conn() as con:
        con.execute(
            """INSERT INTO bot_runs
               (ts, market_open, tickers_scanned, trades_executed, positions_closed, errors, daily_pnl, notes)
               VALUES (?,?,?,?,?,?,?,?)""",
            (datetime.utcnow().isoformat(timespec="seconds"),
             int(market_open), tickers_scanned, trades_executed,
             positions_closed, errors, daily_pnl, notes),
        )


def get_recent_trades(limit: int = 50) -> list:
    try:
        with _conn() as con:
            rows = con.execute(
                "SELECT id,ts,ticker,action,price,qty,stop_loss,take_profit,score,reason,order_id "
                "FROM trade_log ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        cols = ["id","ts","ticker","action","price","qty","stop_loss","take_profit","score","reason","order_id"]
        return [dict(zip(cols, r)) for r in rows]
    except Exception:
        return []


def get_recent_runs(limit: int = 30) -> list:
    try:
        with _conn() as con:
            rows = con.execute(
                "SELECT id,ts,market_open,tickers_scanned,trades_executed,"
                "positions_closed,errors,daily_pnl,notes "
                "FROM bot_runs ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        cols = ["id","ts","market_open","tickers_scanned","trades_executed",
                "positions_closed","errors","daily_pnl","notes"]
        return [dict(zip(cols, r)) for r in rows]
    except Exception:
        return []


def get_trade_summary_today() -> dict:
    """Returns counts for today's trades."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    try:
        with _conn() as con:
            buys  = con.execute(
                "SELECT COUNT(*) FROM trade_log WHERE action='BUY'  AND ts LIKE ?",
                (f"{today}%",)
            ).fetchone()[0]
            sells = con.execute(
                "SELECT COUNT(*) FROM trade_log WHERE action='SELL' AND ts LIKE ?",
                (f"{today}%",)
            ).fetchone()[0]
            skips = con.execute(
                "SELECT COUNT(*) FROM trade_log WHERE action='SKIP' AND ts LIKE ?",
                (f"{today}%",)
            ).fetchone()[0]
        return {"buys": buys, "sells": sells, "skips": skips}
    except Exception:
        return {"buys": 0, "sells": 0, "skips": 0}
