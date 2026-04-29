"""
Lightweight HTTP health/metrics endpoint for mt5_signal_server.

Uses only the stdlib (`http.server`, `threading`) to avoid adding aiohttp
or another dep. Serves two endpoints:

  GET /healthz  → JSON
      {
        "status":            "ok" | "stale" | "disconnected",
        "uptime_sec":        float,
        "last_poll_age_sec": float,
        "mt5_connected":     bool,
        "signals_total":     int,
      }
      HTTP 200 when healthy, 503 when stale or disconnected. UptimeRobot
      can poll this with a 5-min staleness threshold.

  GET /metrics → Prometheus text exposition format
      poll_duration_seconds       gauge
      signals_emitted_total       counter
      mt5_rejections_total{retcode="..."}  counter
      last_poll_timestamp_seconds gauge
      mt5_connected               gauge (1=true, 0=false)

Usage:
    from utils.health_server import Metrics, start_http_server
    metrics = Metrics()
    start_http_server(metrics, port=9100)
    metrics.record_poll(duration=1.2)
    metrics.record_signal()
    metrics.record_rejection(10004)
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

logger = logging.getLogger(__name__)

# A poll older than this counts as "stale" → /healthz returns 503.
STALE_THRESHOLD_SEC = 300


@dataclass
class Metrics:
    """Mutable in-memory metrics. Single instance shared with the HTTP server.

    All updates happen from the signal-server's main loop (single-threaded for
    the main poll), so explicit locking is unnecessary. The HTTP server reads
    snapshots from a different thread but Python's GIL keeps individual
    int/float reads/writes atomic.
    """
    started_at: float = field(default_factory=time.time)
    last_poll_ts: float = 0.0
    last_poll_duration_sec: float = 0.0
    mt5_connected: bool = False
    signals_emitted_total: int = 0
    rejections_by_retcode: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    last_signal_ts: float = 0.0

    def record_poll(self, duration: float, connected: bool) -> None:
        self.last_poll_ts            = time.time()
        self.last_poll_duration_sec  = float(duration)
        self.mt5_connected           = bool(connected)

    def record_signal(self) -> None:
        self.signals_emitted_total += 1
        self.last_signal_ts         = time.time()

    def record_rejection(self, retcode: int) -> None:
        self.rejections_by_retcode[int(retcode)] += 1

    # ── Renderers ───────────────────────────────────────────────────────────
    def healthz_payload(self) -> tuple[int, dict]:
        """Return (http_status, body_dict). 200 when healthy, 503 otherwise."""
        now = time.time()
        last_poll_age = (now - self.last_poll_ts) if self.last_poll_ts else float("inf")
        last_signal_age = (now - self.last_signal_ts) if self.last_signal_ts else None

        if not self.mt5_connected:
            status = "disconnected"
            http   = 503
        elif last_poll_age > STALE_THRESHOLD_SEC:
            status = "stale"
            http   = 503
        else:
            status = "ok"
            http   = 200

        body = {
            "status":               status,
            "uptime_sec":           round(now - self.started_at, 1),
            "last_poll_age_sec":    None if last_poll_age == float("inf")
                                    else round(last_poll_age, 1),
            "last_signal_age_sec":  None if last_signal_age is None
                                    else round(last_signal_age, 1),
            "mt5_connected":        self.mt5_connected,
            "signals_total":        self.signals_emitted_total,
            "last_poll_duration_sec": round(self.last_poll_duration_sec, 3),
        }
        return http, body

    def prometheus_payload(self) -> str:
        lines: list[str] = []
        lines.append("# HELP katraswing_uptime_seconds Seconds since signal server start")
        lines.append("# TYPE katraswing_uptime_seconds gauge")
        lines.append(f"katraswing_uptime_seconds {time.time() - self.started_at:.1f}")

        lines.append("# HELP katraswing_last_poll_timestamp_seconds Unix timestamp of the last completed poll")
        lines.append("# TYPE katraswing_last_poll_timestamp_seconds gauge")
        lines.append(f"katraswing_last_poll_timestamp_seconds {self.last_poll_ts:.1f}")

        lines.append("# HELP katraswing_poll_duration_seconds Wall-clock duration of the most recent poll")
        lines.append("# TYPE katraswing_poll_duration_seconds gauge")
        lines.append(f"katraswing_poll_duration_seconds {self.last_poll_duration_sec:.3f}")

        lines.append("# HELP katraswing_mt5_connected MT5 terminal connection status (1=connected, 0=not)")
        lines.append("# TYPE katraswing_mt5_connected gauge")
        lines.append(f"katraswing_mt5_connected {1 if self.mt5_connected else 0}")

        lines.append("# HELP katraswing_signals_emitted_total Total signals emitted (passed all filters and dedup)")
        lines.append("# TYPE katraswing_signals_emitted_total counter")
        lines.append(f"katraswing_signals_emitted_total {self.signals_emitted_total}")

        lines.append("# HELP katraswing_mt5_rejections_total Order rejections by MT5 retcode")
        lines.append("# TYPE katraswing_mt5_rejections_total counter")
        if self.rejections_by_retcode:
            for retcode, count in sorted(self.rejections_by_retcode.items()):
                lines.append(f'katraswing_mt5_rejections_total{{retcode="{retcode}"}} {count}')
        else:
            lines.append('katraswing_mt5_rejections_total{retcode="none"} 0')

        return "\n".join(lines) + "\n"


def _make_handler(metrics: Metrics):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 — http.server convention
            try:
                if self.path == "/healthz":
                    status, body = metrics.healthz_payload()
                    payload = json.dumps(body).encode("utf-8")
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                if self.path == "/metrics":
                    payload = metrics.prometheus_payload().encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"not found")
            except Exception as exc:
                logger.warning("ctx=health_handler path=%s: %s", self.path, exc)
                try:
                    self.send_response(500)
                    self.end_headers()
                except Exception:
                    pass

        def log_message(self, format, *args):
            # Suppress per-request stdout spam — keep server quiet by default.
            return

    return Handler


def start_http_server(
    metrics: Metrics,
    port: int = 9100,
    bind: str = "0.0.0.0",
) -> HTTPServer:
    """Start an HTTP server in a daemon thread. Returns the HTTPServer instance
    so callers can shut it down cleanly (server.shutdown() + server_close())."""
    handler  = _make_handler(metrics)
    server   = HTTPServer((bind, port), handler)
    thread   = threading.Thread(
        target=server.serve_forever,
        daemon=True,
        name="health-http",
    )
    thread.start()
    logger.info("Health/metrics HTTP listening on http://%s:%d (/healthz, /metrics)", bind, port)
    return server
