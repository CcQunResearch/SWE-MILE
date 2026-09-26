"""Bounded control-plane HTTP I/O independent of the trainer's socket loop.

Model proxy traffic continues to use native async I/O in the Gateway process.
Only trainer control requests use these pools. Futures are polled, so completion
also does not depend on cross-thread asyncio wakeups in the Ray driver.
"""

import asyncio
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from rllm_model_gateway.http_client import shared_ssl_context


class _Pool:
    def __init__(self, workers, name):
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=name)
        self.slots = threading.BoundedSemaphore(workers)


_CONTROL_POOL = _Pool(32, "gateway-control-http")
_TRACE_POOL = _Pool(8, "gateway-trace-http")
_PROBE_POOL = _Pool(4, "gateway-probe-http")
_WORKER_PROBE_POOL = _Pool(32, "gateway-worker-probe-http")


class _RequestAbort:
    def __init__(self):
        self.lock = threading.Lock()
        self.cancelled = False
        self.sockets = []

    @staticmethod
    def _shutdown(raw):
        try:
            raw.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def trace(self, event, info):
        if event.endswith(("connect_tcp.complete", "connect_unix_socket.complete", "start_tls.complete")):
            stream = info.get("return_value")
            raw = stream.get_extra_info("socket") if stream is not None else None
            if raw is not None:
                with self.lock:
                    if self.cancelled:
                        self._shutdown(raw)
                    else:
                        self.sockets.append(raw)
        # Never raise from response/connection close hooks: cleanup must still
        # close the socket after shutdown wakes the blocked reader.
        before_send = event == "request.start" or event.endswith((
            "connect_tcp.started", "connect_unix_socket.started",
            "start_tls.started", "send_request_headers.started",
        ))
        if before_send:
            with self.lock:
                if self.cancelled:
                    raise httpx.RequestError("control request cancelled")

    def abort(self):
        with self.lock:
            self.cancelled = True
            for raw in self.sockets:
                self._shutdown(raw)
            self.sockets.clear()


class ControlHTTPTransport(httpx.AsyncBaseTransport):
    """Keep HTTP progress off the rollout loop without replaying requests.

    Each request owns its retry-free connection. Cancellation interrupts only
    that connection; a running worker retains admission until it actually exits.
    Trace buffers are bounded by the separate eight-request pool (and the
    client's existing eight-response parse admission).
    """

    def __init__(self, *, probe=False, worker_probe=False):
        self.probe = probe
        self.worker_probe = worker_probe
        self._lock = threading.Lock()
        self._active = set()
        self._closed = False

    async def handle_async_request(self, request):
        body = await request.aread()
        is_trace = request.method == "GET" and request.url.path.endswith("/traces")
        pool = (_WORKER_PROBE_POOL if self.worker_probe else _PROBE_POOL) if (self.worker_probe or self.probe) else (_TRACE_POOL if is_trace else _CONTROL_POOL)
        metrics = request.extensions.get("rllm_trace_metrics", {})
        queued = time.perf_counter()
        while not pool.slots.acquire(blocking=False):
            await asyncio.sleep(.01)
        metrics["time/trace_http_admission_s"] = time.perf_counter() - queued
        abort = _RequestAbort()
        completed_at = [None]
        with self._lock:
            if self._closed:
                pool.slots.release()
                raise RuntimeError("control transport is closed")
            self._active.add(abort)

        def transfer():
            started = time.perf_counter()
            metrics["trace_fetch/http_started"] = 1
            extensions = dict(request.extensions)
            extensions.pop("rllm_trace_metrics", None)
            extensions["trace"] = abort.trace
            # A missing endpoint must not consume the whole 600s trace budget
            # in connect(). The engine still owns the total phase deadline.
            timeouts = dict(extensions.get("timeout") or {})
            if is_trace:
                timeouts["connect"] = min(float(timeouts.get("connect") or 10), 10)
            extensions["timeout"] = timeouts
            sync_request = httpx.Request(request.method, request.url, headers=request.headers,
                                         content=body, extensions=extensions)
            try:
                abort.trace("request.start", {})
                with httpx.Client(verify=shared_ssl_context(), trust_env=False,
                                  limits=httpx.Limits(max_keepalive_connections=0)) as client:
                    response = client.send(sync_request, stream=True)
                    metrics["time/trace_request_s"] = time.perf_counter() - started
                    receiving = time.perf_counter()
                    metrics["trace_fetch/http_response_started"] = 1
                    metrics["trace_fetch/response_bytes"] = 0
                    try:
                        chunks = []
                        for chunk in response.iter_bytes():
                            chunks.append(chunk)
                            metrics["trace_fetch/response_bytes"] += len(chunk)
                            metrics["time/trace_receive_s"] = time.perf_counter() - receiving
                        content = b"".join(chunks)
                        headers = httpx.Headers(response.headers)
                        # read() has already decoded HTTP content encodings.
                        headers.pop("content-encoding", None)
                        headers.pop("transfer-encoding", None)
                        headers["content-length"] = str(len(content))
                        return httpx.Response(response.status_code, headers=headers,
                                              stream=httpx.ByteStream(content), extensions=response.extensions)
                    finally:
                        metrics["time/trace_receive_s"] = time.perf_counter() - receiving
                        response.close()
            finally:
                completed_at[0] = time.perf_counter()
                metrics["time/trace_http_s"] = completed_at[0] - started

        def finished(_):
            with self._lock:
                self._active.discard(abort)
            pool.slots.release()

        try:
            future = pool.executor.submit(transfer)
        except BaseException:
            finished(None)
            raise
        future.add_done_callback(finished)
        try:
            while not future.done():
                await asyncio.sleep(.01)
            if completed_at[0] is not None:
                metrics["time/trace_http_delivery_s"] = time.perf_counter() - completed_at[0]
            return future.result()
        except BaseException:
            abort.abort()
            future.cancel()
            raise

    async def aclose(self):
        with self._lock:
            self._closed = True
            active = list(self._active)
        for abort in active:
            abort.abort()
