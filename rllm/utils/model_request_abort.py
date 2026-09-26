"""Interrupt only the sockets owned by one sequential rollout HTTP client."""

from __future__ import annotations

import socket
import threading
import time

import httpx

from rllm.utils.diagnostic_events import emit_diagnostic


class ModelRequestAbort:
    """Use HTTP Core's trace extension, without patching its connection pool.

    Closing a socket in another thread need not wake an in-flight recv().
    shutdown() does; retain socket objects rather than reusable descriptor IDs.
    The owning client disables keepalive and issues requests sequentially.
    """

    def __init__(self, uid: str):
        self.uid = uid
        self._lock = threading.Lock()
        self._aborted = False
        self._sockets: list[socket.socket] = []

    def on_request(self, request: httpx.Request) -> None:
        with self._lock:
            if self._aborted:
                raise httpx.RequestError("rollout model request was stopped", request=request)
            self._sockets.clear()
        previous = request.extensions.get("trace")

        def trace(event, info):
            if event.endswith(("connect_tcp.complete", "connect_unix_socket.complete", "start_tls.complete")):
                stream = info.get("return_value")
                raw = stream.get_extra_info("socket") if stream is not None else None
                if raw is not None:
                    with self._lock:
                        if self._aborted:
                            self._shutdown(raw)
                        else:
                            self._sockets.append(raw)
            if previous is not None:
                previous(event, info)

        request.extensions["trace"] = trace

    @staticmethod
    def _shutdown(raw) -> bool:
        try:
            raw.shutdown(socket.SHUT_RDWR)
            return True
        except OSError:
            return False  # Already closed/disconnected; never use its old fd.

    def abort(self, *, reason: str = "cancelled") -> None:
        started = time.monotonic()
        with self._lock:
            if self._aborted:
                return
            self._aborted = True
            sockets, self._sockets = self._sockets, []
            stopped = sum(self._shutdown(raw) for raw in sockets)
        emit_diagnostic("model_request_abort", uid=self.uid, reason=reason,
                        tracked_sockets=len(sockets), sockets_shutdown=stopped,
                        elapsed_seconds=time.monotonic() - started)
