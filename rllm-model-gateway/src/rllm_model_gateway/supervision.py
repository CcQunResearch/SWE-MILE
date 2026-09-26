"""Out-of-band liveness service for a process-mode model gateway.

The data-plane ASGI server can be busy with hundreds of long-running proxy
requests.  Serving liveness from a separate, loopback-only thread lets the
trainer distinguish that ordinary saturation from a dead or stalled gateway
event loop without touching the session/trace stores.
"""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit


class SupervisionState:
    """Small thread-safe state shared with the supervision listener."""

    def __init__(
        self,
        *,
        generation: str,
        gateway_source_digest: str,
        runtime_code_provenance: Any,
    ) -> None:
        now = time.monotonic()
        self._lock = threading.Lock()
        self._generation = generation
        self._gateway_source_digest = gateway_source_digest
        self._runtime_code_provenance = runtime_code_provenance
        self._started_monotonic = now
        self._event_loop_heartbeat_monotonic = now
        self._last_proxy_success_monotonic: float | None = None

    def note_event_loop_heartbeat(
        self,
        *,
        heartbeat_monotonic: float,
        last_proxy_success_monotonic: float | None,
    ) -> None:
        """Publish main-loop progress without inspecting mutable gateway state."""
        with self._lock:
            self._event_loop_heartbeat_monotonic = heartbeat_monotonic
            if last_proxy_success_monotonic is not None:
                self._last_proxy_success_monotonic = last_proxy_success_monotonic

    def note_proxy_success(self, success_monotonic: float) -> None:
        """Publish a successful data-plane response immediately."""
        with self._lock:
            self._last_proxy_success_monotonic = success_monotonic

    def payload(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            heartbeat = self._event_loop_heartbeat_monotonic
            last_proxy_success = self._last_proxy_success_monotonic
            started = self._started_monotonic
        return {
            "schema_version": 1,
            "status": "ok",
            "pid": os.getpid(),
            "generation": self._generation,
            "gateway_source_digest": self._gateway_source_digest,
            "runtime_code_provenance": self._runtime_code_provenance,
            "event_loop_heartbeat_age_seconds": max(0.0, now - heartbeat),
            "seconds_since_last_proxy_success": (max(0.0, now - started) if last_proxy_success is None else max(0.0, now - last_proxy_success)),
            "uptime_seconds": max(0.0, now - started),
        }


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class SupervisionServer:
    """Dedicated loopback HTTP listener exposing only ``GET /live``."""

    def __init__(self, *, port: int, state: SupervisionState) -> None:
        if port <= 0 or port > 65535:
            raise ValueError("supervision port must be in 1..65535")
        self._state = state
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
                if urlsplit(self.path).path != "/live":
                    self.send_error(404)
                    return
                body = json.dumps(
                    owner._state.payload(),
                    separators=(",", ":"),
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format: str, *args: Any) -> None:
                return

        self._server = _ReusableThreadingHTTPServer(
            ("127.0.0.1", port),
            Handler,
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="rllm-gateway-supervision",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)
