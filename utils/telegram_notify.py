"""
Telegram notification helper for Katraswing signal server.

Usage:
    from utils.telegram_notify import Notifier
    tg = Notifier(token="...", chat_id="...")
    tg.signal("Gold", "LONG", 0.72, 3245.5, 3230.0, 3276.0, "LIQ_SWEEP_5M")
    tg.order_placed(12345, "Gold", "LONG", 3245.5, 3230.0, 3276.0)
    tg.partial_exit(12345, "Gold", 0.05, 3245.5, 3276.0)
    tg.breakeven(12345, "Gold", 3245.5)
    tg.position_closed(12345, "Gold", "LONG", 24.50)
    tg.error("MT5 reconnect failed")
"""

from __future__ import annotations
import logging

log = logging.getLogger(__name__)

_API = "https://api.telegram.org/bot{token}/sendMessage"


class Notifier:
    def __init__(self, token: str = "", chat_id: str = ""):
        self.token   = token.strip()
        self.chat_id = chat_id.strip()
        self._ok     = bool(self.token and self.chat_id)
        if not self._ok:
            log.debug("Telegram: token or chat_id missing -- notifications disabled")

    def enabled(self) -> bool:
        return self._ok

    def _send(self, text: str) -> None:
        if not self._ok:
            return
        try:
            import requests
            r = requests.post(
                _API.format(token=self.token),
                json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"},
                timeout=5,
            )
            if r.status_code != 200:
                log.warning("Telegram send failed: %s %s", r.status_code, r.text[:120])
        except Exception as e:
            log.warning("Telegram send error: %s", e)

    # ── Message types ────────────────────────────────────────────────────────

    def signal(
        self,
        display: str,
        direction: str,
        confidence: float,
        entry: float,
        sl: float,
        tp: float,
        strategy: str,
        paper: bool = False,
    ) -> None:
        tag   = "[PAPER] " if paper else ""
        arrow = "LONG ^" if direction == "LONG" else "SHORT v"
        self._send(
            f"<b>{tag}SIGNAL: {display} {arrow}</b>\n"
            f"Confidence: {confidence:.0%}\n"
            f"Entry: {entry:.5g}  |  SL: {sl:.5g}  |  TP: {tp:.5g}\n"
            f"Strategy: {strategy}"
        )

    def order_placed(
        self,
        ticket: int,
        display: str,
        direction: str,
        entry: float,
        sl: float,
        tp: float,
    ) -> None:
        arrow = "LONG ^" if direction == "LONG" else "SHORT v"
        self._send(
            f"<b>ORDER PLACED: #{ticket}</b>\n"
            f"{display} {arrow} @ {entry:.5g}\n"
            f"SL: {sl:.5g}  |  TP: {tp:.5g}"
        )

    def partial_exit(
        self,
        ticket: int,
        display: str,
        closed_lots: float,
        entry: float,
        tp: float,
    ) -> None:
        self._send(
            f"<b>PARTIAL EXIT: #{ticket} {display}</b>\n"
            f"Closed {closed_lots:.2f} lots at 1R profit\n"
            f"Remaining position -> TP: {tp:.5g}\n"
            f"Entry was: {entry:.5g}"
        )

    def breakeven(self, ticket: int, display: str, entry: float) -> None:
        self._send(
            f"<b>BREAKEVEN: #{ticket} {display}</b>\n"
            f"SL moved to entry {entry:.5g}\n"
            f"Trade is now risk-free"
        )

    def position_closed(
        self,
        ticket: int,
        display: str,
        direction: str,
        profit: float,
    ) -> None:
        sign  = "+" if profit >= 0 else ""
        emoji = "WIN" if profit >= 0 else "LOSS"
        arrow = "LONG ^" if direction == "LONG" else "SHORT v"
        self._send(
            f"<b>CLOSED [{emoji}]: #{ticket} {display}</b>\n"
            f"{arrow}  |  P&L: {sign}{profit:.2f}"
        )

    def error(self, message: str) -> None:
        self._send(f"<b>ERROR</b>\n{message}")

    def info(self, message: str) -> None:
        self._send(message)
