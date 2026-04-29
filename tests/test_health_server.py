"""
Tests for utils.health_server — Metrics state, healthz/metrics rendering,
and an end-to-end HTTP smoke test.
"""
from __future__ import annotations

import json
import time
import urllib.request

import pytest

from utils.health_server import (
    Metrics,
    STALE_THRESHOLD_SEC,
    start_http_server,
)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics state mutations
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricsState:
    def test_record_poll_sets_timestamp_and_duration(self):
        m = Metrics()
        m.record_poll(duration=1.234, connected=True)
        assert m.last_poll_duration_sec == pytest.approx(1.234)
        assert m.mt5_connected is True
        assert m.last_poll_ts > 0

    def test_record_signal_increments_counter(self):
        m = Metrics()
        m.record_signal()
        m.record_signal()
        assert m.signals_emitted_total == 2
        assert m.last_signal_ts > 0

    def test_record_rejection_buckets_by_retcode(self):
        m = Metrics()
        m.record_rejection(10004)
        m.record_rejection(10004)
        m.record_rejection(10018)
        assert m.rejections_by_retcode[10004] == 2
        assert m.rejections_by_retcode[10018] == 1


# ─────────────────────────────────────────────────────────────────────────────
# /healthz status logic
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthzPayload:
    def test_disconnected_returns_503(self):
        m = Metrics()
        m.mt5_connected = False
        m.last_poll_ts = time.time()
        status, body = m.healthz_payload()
        assert status == 503
        assert body["status"] == "disconnected"

    def test_recent_poll_connected_returns_200(self):
        m = Metrics()
        m.mt5_connected = True
        m.last_poll_ts  = time.time()
        status, body = m.healthz_payload()
        assert status == 200
        assert body["status"] == "ok"

    def test_stale_poll_returns_503(self):
        m = Metrics()
        m.mt5_connected = True
        m.last_poll_ts  = time.time() - (STALE_THRESHOLD_SEC + 60)
        status, body = m.healthz_payload()
        assert status == 503
        assert body["status"] == "stale"

    def test_no_poll_yet_returns_503(self):
        m = Metrics()
        m.mt5_connected = True
        # last_poll_ts == 0 (never polled)
        status, body = m.healthz_payload()
        assert status == 503

    def test_payload_includes_signal_counts(self):
        m = Metrics()
        m.mt5_connected = True
        m.last_poll_ts  = time.time()
        m.signals_emitted_total = 7
        _, body = m.healthz_payload()
        assert body["signals_total"] == 7


# ─────────────────────────────────────────────────────────────────────────────
# Prometheus rendering
# ─────────────────────────────────────────────────────────────────────────────

class TestPrometheusPayload:
    def test_includes_required_metrics(self):
        m = Metrics()
        m.mt5_connected = True
        m.signals_emitted_total = 3
        m.record_rejection(10004)
        text = m.prometheus_payload()
        # Required metric names appear
        assert "katraswing_uptime_seconds" in text
        assert "katraswing_signals_emitted_total" in text
        assert "katraswing_mt5_connected" in text
        assert "katraswing_last_poll_timestamp_seconds" in text
        assert "katraswing_poll_duration_seconds" in text

    def test_rejection_label_format(self):
        m = Metrics()
        m.record_rejection(10016)
        text = m.prometheus_payload()
        assert 'katraswing_mt5_rejections_total{retcode="10016"} 1' in text

    def test_no_rejections_emits_zero_sample(self):
        m = Metrics()
        text = m.prometheus_payload()
        # Prometheus convention: emit a zero sample so the metric is visible
        assert 'katraswing_mt5_rejections_total{retcode="none"} 0' in text

    def test_mt5_connected_gauge_is_int(self):
        m = Metrics()
        m.mt5_connected = True
        assert "katraswing_mt5_connected 1" in m.prometheus_payload()
        m.mt5_connected = False
        assert "katraswing_mt5_connected 0" in m.prometheus_payload()


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end HTTP smoke test
# ─────────────────────────────────────────────────────────────────────────────

class TestHttpEndToEnd:
    @pytest.fixture
    def server(self, unused_tcp_port_factory=None):
        """Start a real HTTP server on a port the OS picks for us."""
        # pick a free port deterministically — try 9101..9120
        import socket
        port = None
        for candidate in range(9101, 9121):
            with socket.socket() as s:
                try:
                    s.bind(("127.0.0.1", candidate))
                    port = candidate
                    break
                except OSError:
                    continue
        if port is None:
            pytest.skip("no free port in 9101-9120")
        m = Metrics()
        m.mt5_connected = True
        m.last_poll_ts  = time.time()
        m.signals_emitted_total = 5
        srv = start_http_server(m, port=port, bind="127.0.0.1")
        yield port, m
        srv.shutdown()
        srv.server_close()

    def test_healthz_returns_json(self, server):
        port, _ = server
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=5) as resp:
            assert resp.status == 200
            body = json.loads(resp.read().decode("utf-8"))
        assert body["status"] == "ok"
        assert body["signals_total"] == 5

    def test_metrics_returns_text(self, server):
        port, _ = server
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as resp:
            assert resp.status == 200
            body = resp.read().decode("utf-8")
        assert "katraswing_signals_emitted_total 5" in body
        assert "# HELP" in body   # Prometheus comments present

    def test_unknown_path_returns_404(self, server):
        port, _ = server
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/nope", timeout=5)
            assert False, "expected 404"
        except urllib.error.HTTPError as e:
            assert e.code == 404
