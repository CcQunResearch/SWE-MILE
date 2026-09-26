"""Training-side client for interacting with the model gateway."""

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

from rllm_model_gateway.http_client import shared_ssl_context
from rllm_model_gateway.models import TraceRecord, WorkerInfo


def _decode_traces(response: httpx.Response) -> list[TraceRecord]:
    return [TraceRecord(**trace) for trace in response.json()]


_TRACE_DECODE_WORKERS = 8
_TRACE_DECODE_EXECUTOR = ThreadPoolExecutor(max_workers=_TRACE_DECODE_WORKERS, thread_name_prefix="gateway-trace-decode")
_TRACE_DECODE_SLOTS = threading.BoundedSemaphore(_TRACE_DECODE_WORKERS)
_logger = logging.getLogger(__name__)


class TraceBatch(list):
    """List-compatible trace response carrying local phase timings."""

    def __init__(self, traces, metrics):
        super().__init__(traces)
        self.metrics = metrics


class GatewayClient:
    """Synchronous client for the rllm-model-gateway REST API.

    Intended for use by the training framework to create sessions, retrieve
    traces, and manage workers.
    """

    def __init__(
        self,
        gateway_url: str,
        timeout: float = 30.0,
    ) -> None:
        self.gateway_url = gateway_url.rstrip("/")
        # max_keepalive_connections=0 disables idle connection reuse so every
        # request opens a fresh TCP connection. Avoids the keepalive race where
        # client and uvicorn both expire idle connections at ~5s and the next
        # request hits a half-closed socket → httpx.ReadError. Per-request
        # handshake cost is negligible for the control-plane JSON calls this
        # client makes.
        self._http = httpx.Client(verify=shared_ssl_context(), timeout=timeout, limits=httpx.Limits(max_keepalive_connections=0))

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- Session lifecycle -------------------------------------------------

    def create_session(
        self,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        sampling_params: dict[str, Any] | None = None,
    ) -> str:
        """Create a session (or let the gateway generate an ID)."""
        body: dict[str, Any] = {}
        if session_id:
            body["session_id"] = session_id
        if metadata:
            body["metadata"] = metadata
        if sampling_params:
            body["sampling_params"] = sampling_params
        resp = self._http.post(f"{self.gateway_url}/sessions", json=body)
        resp.raise_for_status()
        return resp.json()["session_id"]

    def get_session_url(self, session_id: str) -> str:
        """Return the OpenAI-compatible base URL for an agent to use."""
        return f"{self.gateway_url}/sessions/{session_id}/v1"

    def get_session_info(self, session_id: str) -> dict[str, Any]:
        resp = self._http.get(f"{self.gateway_url}/sessions/{session_id}")
        resp.raise_for_status()
        return resp.json()

    def list_sessions(self, since: float | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if since is not None:
            params["since"] = since
        if limit is not None:
            params["limit"] = limit
        resp = self._http.get(f"{self.gateway_url}/sessions", params=params)
        resp.raise_for_status()
        return resp.json()

    def delete_session(self, session_id: str) -> int:
        resp = self._http.delete(f"{self.gateway_url}/sessions/{session_id}")
        resp.raise_for_status()
        return resp.json().get("deleted", 0)

    # -- Trace retrieval ---------------------------------------------------

    def get_session_traces(
        self,
        session_id: str,
        since: float | None = None,
        limit: int | None = None,
        *,
        consume: bool = False,
    ) -> list[TraceRecord]:
        params: dict[str, Any] = {}
        if since is not None:
            params["since"] = since
        if limit is not None:
            params["limit"] = limit
        if consume:
            params["consume"] = "true"
        resp = self._http.get(f"{self.gateway_url}/sessions/{session_id}/traces", params=params)
        resp.raise_for_status()
        data = resp.json()
        return [TraceRecord(**t) for t in data]

    def get_trace(self, trace_id: str) -> TraceRecord:
        resp = self._http.get(f"{self.gateway_url}/traces/{trace_id}")
        resp.raise_for_status()
        return TraceRecord(**resp.json())

    # -- Worker management -------------------------------------------------

    def add_worker(
        self,
        url: str,
        api_path: str = "/v1",
        model_name: str | None = None,
        weight: int = 1,
    ) -> str:
        """Register a worker.  Returns worker_id."""
        body: dict[str, Any] = {"url": url, "api_path": api_path, "weight": weight}
        if model_name:
            body["model_name"] = model_name
        resp = self._http.post(f"{self.gateway_url}/admin/workers", json=body)
        resp.raise_for_status()
        return resp.json()["worker_id"]

    def remove_worker(self, worker_id: str) -> None:
        resp = self._http.delete(f"{self.gateway_url}/admin/workers/{worker_id}")
        resp.raise_for_status()

    def list_workers(self) -> list[WorkerInfo]:
        resp = self._http.get(f"{self.gateway_url}/admin/workers")
        resp.raise_for_status()
        return [WorkerInfo(**w) for w in resp.json()]

    def get_routing_stats(self) -> dict[str, Any]:
        resp = self._http.get(f"{self.gateway_url}/admin/routing/stats")
        resp.raise_for_status()
        return resp.json()

    def get_session_cleanup_stats(self) -> dict[str, Any]:
        resp = self._http.get(
            f"{self.gateway_url}/admin/session_cleanup/stats"
        )
        resp.raise_for_status()
        return resp.json()

    def get_runtime_stats(self) -> dict[str, Any]:
        resp = self._http.get(f"{self.gateway_url}/admin/runtime/stats")
        resp.raise_for_status()
        return resp.json()

    def get_supervision_snapshot(self) -> dict[str, Any]:
        resp = self._http.get(
            f"{self.gateway_url}/admin/supervision/snapshot"
        )
        resp.raise_for_status()
        return resp.json()

    # -- Weight version ----------------------------------------------------

    def set_weight_version(self, weight_version: int) -> int:
        resp = self._http.post(f"{self.gateway_url}/admin/weight_version", json={"weight_version": weight_version})
        resp.raise_for_status()
        return resp.json()["weight_version"]

    def get_weight_version(self) -> int | None:
        resp = self._http.get(f"{self.gateway_url}/admin/weight_version")
        resp.raise_for_status()
        return resp.json().get("weight_version")

    def set_maintenance(
        self,
        *,
        active: bool,
        transition_id: str,
        health_grace_seconds: float | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "active": active,
            "transition_id": transition_id,
        }
        if health_grace_seconds is not None:
            body["health_grace_seconds"] = health_grace_seconds
        resp = self._http.post(
            f"{self.gateway_url}/admin/maintenance", json=body
        )
        resp.raise_for_status()
        return resp.json()

    def get_maintenance(self) -> dict[str, Any]:
        resp = self._http.get(f"{self.gateway_url}/admin/maintenance")
        resp.raise_for_status()
        return resp.json()

    # -- Lifecycle ---------------------------------------------------------

    def flush(self, timeout: float = 30.0) -> bool:
        resp = self._http.post(f"{self.gateway_url}/admin/flush", timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("status") == "flushed"

    def health(self) -> dict[str, Any]:
        resp = self._http.get(f"{self.gateway_url}/health")
        resp.raise_for_status()
        return resp.json()


class AsyncGatewayClient:
    """Async variant of :class:`GatewayClient` using ``httpx.AsyncClient``."""

    def __init__(
        self,
        gateway_url: str,
        timeout: float = 30.0,
        max_connections: int | None = 100,
        max_keepalive_connections: int | None = 0,
        keepalive_expiry: float | None = 5.0,
        control_transport: bool = False,
        probe_transport: bool = False,
    ) -> None:
        self.gateway_url = gateway_url.rstrip("/")
        # Runtime clients retain the historical no-keepalive default. The
        # process supervisor opts into one reserved persistent connection so
        # a rollout connection storm cannot starve health probes at accept().
        from rllm_model_gateway.control_transport import ControlHTTPTransport

        self._http = httpx.AsyncClient(
            transport=ControlHTTPTransport(probe=probe_transport) if control_transport else None,
            trust_env=not control_transport,
            verify=shared_ssl_context(),
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections,
                keepalive_expiry=keepalive_expiry,
            ),
        )

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()

    # -- Session lifecycle -------------------------------------------------

    async def create_session(
        self,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        sampling_params: dict[str, Any] | None = None,
    ) -> str:
        body: dict[str, Any] = {}
        if session_id:
            body["session_id"] = session_id
        if metadata:
            body["metadata"] = metadata
        if sampling_params:
            body["sampling_params"] = sampling_params
        resp = await self._http.post(f"{self.gateway_url}/sessions", json=body)
        resp.raise_for_status()
        return resp.json()["session_id"]

    def get_session_url(self, session_id: str) -> str:
        return f"{self.gateway_url}/sessions/{session_id}/v1"

    async def get_session_info(self, session_id: str) -> dict[str, Any]:
        resp = await self._http.get(f"{self.gateway_url}/sessions/{session_id}")
        resp.raise_for_status()
        return resp.json()

    async def session_exists(self, session_id: str) -> bool:
        """Return whether a session still exists without raising on HTTP 404."""
        resp = await self._http.get(f"{self.gateway_url}/sessions/{session_id}")
        if resp.status_code == 404:
            return False
        resp.raise_for_status()
        return True

    async def list_sessions(self, since: float | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if since is not None:
            params["since"] = since
        if limit is not None:
            params["limit"] = limit
        resp = await self._http.get(f"{self.gateway_url}/sessions", params=params)
        resp.raise_for_status()
        return resp.json()

    async def delete_session(self, session_id: str) -> int:
        resp = await self._http.delete(f"{self.gateway_url}/sessions/{session_id}")
        resp.raise_for_status()
        return resp.json().get("deleted", 0)

    async def delete_sessions(self, session_ids: list[str]) -> int:
        """Batch-delete sessions (and their traces) in a single round-trip."""
        if not session_ids:
            return 0
        resp = await self._http.post(
            f"{self.gateway_url}/sessions/batch_delete",
            json={"session_ids": list(session_ids)},
        )
        resp.raise_for_status()
        return resp.json().get("deleted", 0)

    # -- Trace retrieval ---------------------------------------------------

    async def get_session_traces(
        self,
        session_id: str,
        since: float | None = None,
        limit: int | None = None,
        *,
        consume: bool = False,
    ) -> list[TraceRecord]:
        params: dict[str, Any] = {}
        if since is not None:
            params["since"] = since
        if limit is not None:
            params["limit"] = limit
        if consume:
            params["consume"] = "true"
        metrics = {}
        queued = time.perf_counter()
        # Admission before reading bounds both buffered responses and decoding
        # jobs. A cancelled running decoder retains its slot until it finishes.
        acquired = False
        submitted = False
        try:
            while not _TRACE_DECODE_SLOTS.acquire(blocking=False):
                await asyncio.sleep(0.01)
            acquired = True
            metrics["time/trace_decode_admission_s"] = time.perf_counter() - queued
            started = time.perf_counter()
            async with self._http.stream("GET", f"{self.gateway_url}/sessions/{session_id}/traces", params=params, extensions={"rllm_trace_metrics": metrics}) as resp:
                resp.raise_for_status()
                metrics.setdefault("time/trace_request_s", time.perf_counter() - started)
                if "X-RLLM-Trace-Barrier-Seconds" in resp.headers:
                    metrics["time/trace_server_barrier_s"] = float(resp.headers["X-RLLM-Trace-Barrier-Seconds"])
                receiving = time.perf_counter()
                await resp.aread()
                metrics.setdefault("time/trace_receive_s", time.perf_counter() - receiving)
                metrics["trace_fetch/response_bytes"] = len(resp.content)
            queued = time.perf_counter()

            def decode():
                metrics["time/trace_decode_queue_s"] = time.perf_counter() - queued
                decoding = time.perf_counter()
                try:
                    return _decode_traces(resp)
                finally:
                    metrics["time/trace_decode_s"] = time.perf_counter() - decoding

            future = _TRACE_DECODE_EXECUTOR.submit(decode)
            future.add_done_callback(lambda _: _TRACE_DECODE_SLOTS.release())
            submitted = True
            try:
                while not future.done():
                    await asyncio.sleep(0.01)
                return TraceBatch(future.result(), metrics)
            except BaseException:
                future.cancel()
                raise
        except BaseException as exc:
            if not acquired:
                metrics["time/trace_decode_admission_s"] = time.perf_counter() - queued
            exc.trace_fetch_metrics = dict(metrics)
            raise
        finally:
            if acquired and not submitted:
                _TRACE_DECODE_SLOTS.release()
            _logger.info("trace_fetch session=%s metrics=%s", session_id, dict(metrics))

    async def get_trace(self, trace_id: str) -> TraceRecord:
        resp = await self._http.get(f"{self.gateway_url}/traces/{trace_id}")
        resp.raise_for_status()
        return TraceRecord(**resp.json())

    # -- Worker management -------------------------------------------------

    async def add_worker(
        self,
        url: str,
        api_path: str = "/v1",
        model_name: str | None = None,
        weight: int = 1,
    ) -> str:
        body: dict[str, Any] = {"url": url, "api_path": api_path, "weight": weight}
        if model_name:
            body["model_name"] = model_name
        resp = await self._http.post(f"{self.gateway_url}/admin/workers", json=body)
        resp.raise_for_status()
        return resp.json()["worker_id"]

    async def remove_worker(self, worker_id: str) -> None:
        resp = await self._http.delete(f"{self.gateway_url}/admin/workers/{worker_id}")
        resp.raise_for_status()

    async def list_workers(self) -> list[WorkerInfo]:
        resp = await self._http.get(f"{self.gateway_url}/admin/workers")
        resp.raise_for_status()
        return [WorkerInfo(**w) for w in resp.json()]

    async def get_routing_stats(self) -> dict[str, Any]:
        resp = await self._http.get(f"{self.gateway_url}/admin/routing/stats")
        resp.raise_for_status()
        return resp.json()

    async def get_session_cleanup_stats(self) -> dict[str, Any]:
        resp = await self._http.get(
            f"{self.gateway_url}/admin/session_cleanup/stats"
        )
        resp.raise_for_status()
        return resp.json()

    async def get_runtime_stats(self) -> dict[str, Any]:
        resp = await self._http.get(
            f"{self.gateway_url}/admin/runtime/stats"
        )
        resp.raise_for_status()
        return resp.json()

    async def get_supervision_snapshot(
        self,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        kwargs = {"timeout": timeout} if timeout is not None else {}
        resp = await self._http.get(
            f"{self.gateway_url}/admin/supervision/snapshot",
            **kwargs,
        )
        resp.raise_for_status()
        return resp.json()

    # -- Weight version ----------------------------------------------------

    async def set_weight_version(self, weight_version: int) -> int:
        resp = await self._http.post(f"{self.gateway_url}/admin/weight_version", json={"weight_version": weight_version})
        resp.raise_for_status()
        return resp.json()["weight_version"]

    async def get_weight_version(self) -> int | None:
        resp = await self._http.get(f"{self.gateway_url}/admin/weight_version")
        resp.raise_for_status()
        return resp.json().get("weight_version")

    async def set_maintenance(
        self,
        *,
        active: bool,
        transition_id: str,
        health_grace_seconds: float | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "active": active,
            "transition_id": transition_id,
        }
        if health_grace_seconds is not None:
            body["health_grace_seconds"] = health_grace_seconds
        resp = await self._http.post(
            f"{self.gateway_url}/admin/maintenance", json=body
        )
        resp.raise_for_status()
        return resp.json()

    async def get_maintenance(self) -> dict[str, Any]:
        resp = await self._http.get(f"{self.gateway_url}/admin/maintenance")
        resp.raise_for_status()
        return resp.json()

    # -- Lifecycle ---------------------------------------------------------

    async def flush(self, timeout: float = 30.0) -> bool:
        resp = await self._http.post(f"{self.gateway_url}/admin/flush", timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("status") == "flushed"

    async def health(self) -> dict[str, Any]:
        resp = await self._http.get(f"{self.gateway_url}/health")
        resp.raise_for_status()
        return resp.json()
