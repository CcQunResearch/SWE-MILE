"""GatewayManager + EvalGatewayManager: rllm-model-gateway lifecycle.

Two classes share most of the lifecycle (uvicorn-on-thread or subprocess,
trace store, session API). They differ in how upstream workers are
registered and what request-body injection the middleware applies.

* :class:`GatewayManager` — training. Workers come from a verl/tinker
  rollout engine via :meth:`start(rollout_engine)`. Injects ``logprobs``
  and ``return_token_ids`` into request bodies (vLLM needs both for the
  loss math downstream).

* :class:`EvalGatewayManager(GatewayManager)` — eval. Wraps a static
  upstream URL (vLLM endpoint, LiteLLM proxy, OpenAI-compatible server).
  Disables vLLM-specific param injection because external providers 400
  on ``return_token_ids``. Constructed with
  ``EvalGatewayManager(upstream_url, model)`` and started with
  ``.start()`` (no rollout engine).

Modes:

- 'process': subprocess via ``rllm-model-gateway`` CLI (for verl / distributed)
- 'thread': background thread via ``create_app`` + uvicorn (for tinker /
  single-machine / eval)

For Tinker backends, an in-process handler is injected into the gateway
(via ``local_handler``), avoiding the need for a separate HTTP backend server.
"""

from __future__ import annotations

import asyncio
import hashlib
import http.client
import importlib.util
import json
import logging
import math
import os
import re
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from omegaconf import OmegaConf
from rllm_model_gateway.client import AsyncGatewayClient, GatewayClient
from rllm_model_gateway.models import TraceRecord

from rllm.env import env_float
from rllm.gateway.session_ids import task_group_session_ids

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from rllm.engine.rollout import RolloutEngine, VerlEngine

logger = logging.getLogger(__name__)

_HEALTH_POLL_INTERVAL = 0.5
_HEALTH_POLL_TIMEOUT = env_float("RLLM_GATEWAY_HEALTH_TIMEOUT_S", 30.0)  # set env var: export RLLM_GATEWAY_HEALTH_TIMEOUT_S=xxx
_TRACE_API_TIMEOUT = 600.0
@dataclass(frozen=True)
class GatewaySupervisionConfig:
    enable: bool = True
    health_interval_seconds: float = 5.0
    health_timeout_seconds: float = 2.0
    failure_threshold: int = 3
    max_restarts: int = 1
    recovery_stall_timeout_seconds: float = 120.0


class GatewaySupervisionError(RuntimeError):
    """Base class for failures reported by the isolated liveness channel."""


class GatewaySoftFailure(GatewaySupervisionError):
    """A transport/control-plane failure that requires sustained confirmation."""

    gateway_failure_kind = "soft"


class GatewayHardFailure(GatewaySupervisionError):
    """A confirmed process or runtime-identity failure."""

    gateway_failure_kind = "hard"


class _FifoAdmissionLimiter:
    """Cancellation-safe FIFO token bucket for rollout session starts."""

    def __init__(self, *, rate_per_second: float, burst: int) -> None:
        self.rate_per_second = float(rate_per_second)
        self.burst = int(burst)
        self._tokens = float(burst)
        self._updated_at = time.monotonic()
        self._lock = asyncio.Lock()
        self._waiters: deque[tuple[asyncio.Future[None], float]] = deque()
        self._pump_task: asyncio.Task[None] | None = None
        self._admitted_total = 0
        self._wait_seconds_total = 0.0
        self._wait_seconds_max = 0.0
        self._admitted_at: deque[float] = deque()
        self._metrics_started_at = time.monotonic()

    async def acquire(self) -> None:
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[None] = loop.create_future()
        enqueued_at = time.monotonic()
        async with self._lock:
            self._waiters.append((waiter, enqueued_at))
            if self._pump_task is None or self._pump_task.done():
                self._pump_task = asyncio.create_task(
                    self._pump(),
                    name="gateway-session-start-admission",
                )
        try:
            await waiter
        except asyncio.CancelledError:
            waiter.cancel()
            raise

    async def _pump(self) -> None:
        while True:
            delay = 0.0
            async with self._lock:
                while self._waiters and self._waiters[0][0].cancelled():
                    self._waiters.popleft()
                if not self._waiters:
                    self._pump_task = None
                    return

                now = time.monotonic()
                elapsed = max(0.0, now - self._updated_at)
                self._tokens = min(
                    float(self.burst),
                    self._tokens + elapsed * self.rate_per_second,
                )
                self._updated_at = now
                # Drain every token currently available before yielding back
                # to the event loop.  The previous one-token/sleep(0) loop
                # could run orders of magnitude below the configured rate
                # when hundreds of lifecycle callbacks already occupied the
                # ready queue: a delayed wake refilled a full burst, but only
                # one waiter was released before yielding again.
                while self._waiters and self._tokens >= 1.0:
                    while self._waiters and self._waiters[0][0].cancelled():
                        self._waiters.popleft()
                    if not self._waiters:
                        break
                    self._tokens -= 1.0
                    waiter, enqueued_at = self._waiters.popleft()
                    if not waiter.cancelled():
                        waited = max(0.0, now - enqueued_at)
                        self._admitted_total += 1
                        self._wait_seconds_total += waited
                        self._wait_seconds_max = max(
                            self._wait_seconds_max,
                            waited,
                        )
                        self._admitted_at.append(now)
                        waiter.set_result(None)
                if self._waiters and self._tokens < 1.0:
                    delay = (1.0 - self._tokens) / self.rate_per_second
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                # Let newly-admitted rollouts run before draining more burst
                # capacity, while retaining FIFO order for queued waiters.
                await asyncio.sleep(0)

    def metrics(self) -> dict[str, int | float]:
        admitted = self._admitted_total
        now = time.monotonic()
        cutoff = now - 60.0
        while self._admitted_at and self._admitted_at[0] < cutoff:
            self._admitted_at.popleft()
        observed_window = min(
            60.0,
            max(now - self._metrics_started_at, 1e-9),
        )
        return {
            "pending": sum(
                not waiter.cancelled() for waiter, _ in self._waiters
            ),
            "admitted_total": admitted,
            "wait_seconds_mean": (
                self._wait_seconds_total / admitted if admitted else 0.0
            ),
            "wait_seconds_max": self._wait_seconds_max,
            "admitted_last_60s": len(self._admitted_at),
            "effective_rate_per_second_60s": (
                len(self._admitted_at) / observed_window
            ),
            "rate_per_second": self.rate_per_second,
            "burst": self.burst,
        }


def _find_free_port() -> int:
    """Ask the OS for a free TCP port on the loopback interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@lru_cache(maxsize=1)
def _installed_gateway_source_digest() -> str | None:
    """Hash the installed gateway package once for repeated identity probes."""
    spec = importlib.util.find_spec("rllm_model_gateway")
    package_locations = (
        tuple(spec.submodule_search_locations or ())
        if spec is not None
        else ()
    )
    if not package_locations:
        return None
    try:
        digest = hashlib.sha256()
        package_dir = Path(package_locations[0])
        for path in sorted(package_dir.rglob("*.py")):
            relative = path.relative_to(package_dir).as_posix().encode("utf-8")
            digest.update(relative)
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()
    except OSError:
        return None


def container_reachable_url(url: str, backend: str | None) -> str:
    """Rewrite host loopback addresses so a process inside a container can reach the gateway.

    The gateway binds to 127.0.0.1 on the host; inside a Docker container
    that loopback addresses the container itself. Docker Desktop resolves
    ``host.docker.internal`` to the host, and on Linux Docker 20.10+ the
    same hostname works when the container is started with
    ``--add-host=host.docker.internal:host-gateway``. Keyed on the backend
    that actually provisioned the task's sandbox; a no-op for everything
    but ``docker`` (remote backends get a public tunnel URL instead).
    """
    if backend != "docker":
        return url
    return re.sub(
        r"(https?://)(?:127\.0\.0\.1|localhost)(:\d+|/|$)",
        r"\1host.docker.internal\2",
        url,
    )


def _normalize_worker_url(raw_url: str) -> str:
    """Strip trailing ``/v1`` and trailing slashes from an upstream URL.

    The gateway client always sends ``api_path="/v1"`` when registering a
    worker, and the upstream's ``api_url`` is computed as
    ``url + api_path``. Without this normalization, callers passing an
    OpenAI-compatible URL like ``http://localhost:4000/v1`` would end up
    forwarding to ``http://localhost:4000/v1/v1/chat/completions``.
    """
    url = raw_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    return url


def _get_routable_ip() -> str:
    """Return the machine's routable IPv4 address.

    Strategy (adapted from slime's ``get_host_info``):
    1. UDP probe to 8.8.8.8 — queries kernel routing table without sending data
    2. Fallback: ``socket.getaddrinfo(hostname)`` filtering out loopback
    3. Last resort: ``127.0.0.1``
    """

    def _is_loopback(ip: str) -> bool:
        return ip.startswith("127.") or ip == "::1"

    # Strategy 1: UDP connect probe (most accurate)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip: str = s.getsockname()[0]
            if not _is_loopback(ip):
                return ip
    except Exception:
        pass

    # Strategy 2: hostname resolution filtering out loopback
    try:
        hostname = socket.gethostname()
        infos = socket.getaddrinfo(hostname, None, family=socket.AF_INET, type=socket.SOCK_STREAM)
        for info in infos:
            ip = str(info[4][0])
            if not _is_loopback(ip):
                return ip
    except Exception:
        pass

    return "127.0.0.1"


class GatewayManager:
    """Manages model gateway lifecycle for training.

    Supports two execution modes:
    - 'process': subprocess.Popen (for verl / distributed)
    - 'thread': background thread via create_app + uvicorn (for tinker / single-machine)
    """

    # Subclasses override these to flip vLLM-specific request-body injection
    # off when the upstream isn't vLLM (e.g. OpenAI/Anthropic via LiteLLM).
    add_logprobs: bool = True
    add_return_token_ids: bool = True

    def __init__(self, config: DictConfig, mode: str = "thread") -> None:
        gw_cfg = config.rllm.get("gateway", {})
        configured_host = gw_cfg.get("host", None)
        self.host: str = configured_host if configured_host else _get_routable_ip()
        self.port: int = gw_cfg.get("port", 9090)
        self.store: str = gw_cfg.get("store", "memory")
        self.db_path: str | None = gw_cfg.get("db_path", None)
        capture_raw_payloads = gw_cfg.get("capture_raw_payloads", True)
        if not isinstance(capture_raw_payloads, bool):
            raise ValueError(
                "rllm.gateway.capture_raw_payloads must be a boolean"
            )
        self.capture_raw_payloads = capture_raw_payloads
        if self.store not in ("memory", "sqlite"):
            raise ValueError(f"rllm.gateway.store must be 'memory' or 'sqlite', got {self.store!r}")
        if self.store == "memory" and self.db_path:
            raise ValueError("rllm.gateway.db_path is set but store='memory'; set store='sqlite' or clear db_path")
        from rllm.gateway.tunnel import parse_tunnel

        self.public_url, self.tunnel_backend = parse_tunnel(gw_cfg.get("tunnel", None))
        # The gateway always pins ``body.model`` to whatever the trainer is serving
        self.model: str | None = config.get("model", {}).get("name", None)

        # Cumulative token mode: drift-free multi-turn token forwarding. The
        # gateway loads the tokenizer from the served model path. renderers
        # cannot infer the family from a local checkpoint path, so
        # renderer_family must be set explicitly (e.g. "qwen3", "glm-5").
        self.cumulative_token_mode: bool = gw_cfg.get("cumulative_token_mode", False)
        self.routing_policy: str | None = gw_cfg.get("routing_policy", None)
        from rllm_model_gateway.models import GatewayRoutingConfig

        routing_cfg = gw_cfg.get("routing", {}) or {}
        routing_values = (
            OmegaConf.to_container(routing_cfg, resolve=True)
            if OmegaConf.is_config(routing_cfg)
            else dict(routing_cfg)
        )
        self.routing_config = GatewayRoutingConfig(**routing_values)
        if (
            self.routing_config.mode == "group_striped_adaptive"
            and not self.cumulative_token_mode
        ):
            raise ValueError(
                "rllm.gateway.routing.mode=group_striped_adaptive requires "
                "rllm.gateway.cumulative_token_mode=true"
            )
        if (
            self.routing_config.mode == "group_striped_adaptive"
            and self.routing_policy
        ):
            raise ValueError(
                "rllm.gateway.routing_policy cannot be combined with "
                "rllm.gateway.routing.mode=group_striped_adaptive"
            )
        self.dynamic_sequence_budget: bool = bool(
            config.rllm.get("data", {}).get("dynamic_sequence_budget", False)
        )
        self.renderer_family: str = gw_cfg.get("renderer_family", "auto")
        configured_max_context = gw_cfg.get("max_context_tokens", None)
        if configured_max_context is None:
            actor_rollout_ref = config.get("actor_rollout_ref", {}) or {}
            rollout_cfg = actor_rollout_ref.get("rollout", {}) or {}
            configured_max_context = rollout_cfg.get("max_model_len", None)
        self.max_context_tokens: int | None = (
            int(configured_max_context) if configured_max_context is not None else None
        )
        if self.max_context_tokens is not None and self.max_context_tokens <= 0:
            raise ValueError(
                "rllm.gateway.max_context_tokens / actor_rollout_ref.rollout.max_model_len "
                "must be a positive integer"
            )

        def _finite_number(
            name: str,
            default: float,
            *,
            allow_zero: bool = False,
        ) -> float:
            raw = gw_cfg.get(name, default)
            if isinstance(raw, bool) or not isinstance(raw, int | float):
                raise ValueError(f"rllm.gateway.{name} must be a finite number")
            value = float(raw)
            if not math.isfinite(value) or (
                value < 0 if allow_zero else value <= 0
            ):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(
                    f"rllm.gateway.{name} must be a finite {qualifier} number"
                )
            return value

        def _positive_integer(name: str, default: int) -> int:
            raw = gw_cfg.get(name, default)
            if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
                raise ValueError(
                    f"rllm.gateway.{name} must be a positive integer"
                )
            return raw

        self.worker_recovery_timeout = _finite_number(
            "worker_recovery_timeout", 600.0
        )
        self.session_cleanup_concurrency = _positive_integer(
            "session_cleanup_concurrency", 32
        )
        self.session_cleanup_phase_timeout = _finite_number(
            "session_cleanup_phase_timeout", 30.0
        )
        self.session_cleanup_shutdown_timeout = _finite_number(
            "session_cleanup_shutdown_timeout", 30.0
        )
        self.session_cleanup_max_pending = _positive_integer(
            "session_cleanup_max_pending", 4096
        )
        self.session_cleanup_stall_timeout = _finite_number(
            "session_cleanup_stall_timeout", 600.0
        )
        self.worker_health_check_timeout = _finite_number(
            "worker_health_check_timeout", 15.0
        )
        self.worker_health_check_failure_threshold = _positive_integer(
            "worker_health_check_failure_threshold", 3
        )
        self.maintenance_health_grace_seconds = _finite_number(
            "maintenance_health_grace_seconds", 120.0, allow_zero=True
        )
        self.recovery_admission_rate_per_second = _finite_number(
            "recovery_admission_rate_per_second", 64.0
        )
        self.recovery_admission_burst = _positive_integer(
            "recovery_admission_burst", 64
        )
        self.recovery_admission_jitter_seconds = _finite_number(
            "recovery_admission_jitter_seconds", 2.0, allow_zero=True
        )
        self.session_start_admission_rate_per_second = _finite_number(
            "session_start_admission_rate_per_second", 64.0
        )
        self.session_start_admission_burst = _positive_integer(
            "session_start_admission_burst", 64
        )
        self.weight_sync_control_timeout_seconds = _finite_number(
            "weight_sync_control_timeout_seconds", 30.0
        )
        self.weight_sync_control_max_attempts = _positive_integer(
            "weight_sync_control_max_attempts", 5
        )
        self.weight_sync_control_retry_backoff_seconds = _finite_number(
            "weight_sync_control_retry_backoff_seconds", 1.0
        )

        supervision_cfg = gw_cfg.get("supervision", {}) or {}
        supervision_values = (
            OmegaConf.to_container(supervision_cfg, resolve=True)
            if OmegaConf.is_config(supervision_cfg)
            else dict(supervision_cfg)
        )
        supervision_enabled = supervision_values.get("enable", True)
        if not isinstance(supervision_enabled, bool):
            raise ValueError("rllm.gateway.supervision.enable must be a boolean")

        def _supervision_positive_number(name: str, default: float) -> float:
            raw = supervision_values.get(name, default)
            if (
                isinstance(raw, bool)
                or not isinstance(raw, int | float)
                or not math.isfinite(float(raw))
                or float(raw) <= 0
            ):
                raise ValueError(
                    f"rllm.gateway.supervision.{name} must be a finite positive number"
                )
            return float(raw)

        threshold = supervision_values.get("failure_threshold", 3)
        if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold <= 0:
            raise ValueError(
                "rllm.gateway.supervision.failure_threshold must be a positive integer"
            )
        max_restarts = supervision_values.get("max_restarts", 1)
        if (
            isinstance(max_restarts, bool)
            or not isinstance(max_restarts, int)
            or max_restarts < 0
        ):
            raise ValueError(
                "rllm.gateway.supervision.max_restarts must be a non-negative integer"
            )
        self.supervision_config = GatewaySupervisionConfig(
            enable=supervision_enabled,
            health_interval_seconds=_supervision_positive_number(
                "health_interval_seconds", 5.0
            ),
            health_timeout_seconds=_supervision_positive_number(
                "health_timeout_seconds", 2.0
            ),
            failure_threshold=threshold,
            max_restarts=max_restarts,
            recovery_stall_timeout_seconds=_supervision_positive_number(
                "recovery_stall_timeout_seconds", 120.0
            ),
        )

        # Lifecycle operations use a separate httpx control-plane client.  Its
        # default 100-connection pool is smaller than common fully-async
        # frontiers (for example 288 rollouts), which can leave GET/DELETE
        # requests queued in the client long enough to trip cleanup deadlines.
        # One lifecycle operation can be active per rollout; keep a small
        # allowance for weight-version and administrative calls.
        workflow_cfg = config.rllm.get("workflow", {}) or {}
        n_parallel_tasks = int(workflow_cfg.get("n_parallel_tasks", 0) or 0)
        self.control_plane_max_connections = max(100, n_parallel_tasks + 16)

        self.mode = mode

        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._server: Any = None  # uvicorn.Server when using thread mode
        self._local_handler: Any = None  # in-process handler for tinker
        self._client: GatewayClient | None = None
        self._async_client: AsyncGatewayClient | None = None
        self._supervision_client: AsyncGatewayClient | None = None
        # Allocated immediately before spawning the process so constructing a
        # manager remains side-effect free and tests/config validation do not
        # require socket permissions.
        self._supervision_port: int | None = None
        self._supervision_generation: str | None = None
        self._supervision_executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="gateway-supervision-client",
            )
            if mode == "process"
            else None
        )
        self._supervision_reconnects = 0
        self._supervision_last_probe_reconnected = 0
        self._supervision_last_success_monotonic: float | None = None
        self._session_start_admission = _FifoAdmissionLimiter(
            rate_per_second=self.session_start_admission_rate_per_second,
            burst=self.session_start_admission_burst,
        )
        self._tunnel: Any = None  # CloudflaredTunnel when tunnel_backend is set
        self._worker_urls: list[str] = []
        self._available = False
        self._restart_count = 0
        self._restart_timestamps: deque[float] = deque()
        self._current_weight_version: int | None = None
        self._last_exit_diagnostics: dict[str, Any] = {}
        self._last_live_process_stats: dict[str, int | float] = {}
        self._last_rss_sample: tuple[float, float] | None = None
        self._bulk_tombstoned_sessions: set[str] = set()

        # Per-mode sampling params (extracted from rollout engine in start())
        self._train_sampling_params: dict[str, Any] = {}
        self._val_sampling_params: dict[str, Any] = {}

    @property
    def gateway_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def client(self) -> GatewayClient:
        """Sync client for lifecycle operations (start, stop, health polling)."""
        if self._client is None:
            self._client = GatewayClient(self.gateway_url)
        return self._client

    @property
    def async_client(self) -> AsyncGatewayClient:
        """Async client for runtime operations (sessions, traces)."""
        if self._async_client is None:
            self._async_client = AsyncGatewayClient(
                self.gateway_url,
                timeout=_TRACE_API_TIMEOUT,
                max_connections=self.control_plane_max_connections,
                control_transport=self.mode == "process",
            )
        return self._async_client

    def _new_supervision_client(self, timeout: float) -> AsyncGatewayClient:
        """Reserve probe I/O capacity independently of trace and worker reads."""
        return AsyncGatewayClient(
            self.gateway_url,
            timeout=timeout,
            max_connections=1,
            max_keepalive_connections=1,
            keepalive_expiry=30.0,
            control_transport=self.mode == "process",
            probe_transport=True,
        )

    async def _close_supervision_client(self) -> None:
        client = self._supervision_client
        self._supervision_client = None
        if client is not None:
            await client.close()

    @property
    def restarts_in_window(self) -> int:
        self._prune_restart_timestamps(time.monotonic())
        return len(self._restart_timestamps)

    def _prune_restart_timestamps(self, now: float) -> None:
        timestamps = getattr(self, "_restart_timestamps", None)
        if timestamps is None:
            self._restart_timestamps = deque()
            return
        window = float(
            getattr(
                self.supervision_config,
                "recovery_stall_timeout_seconds",
                120.0,
            )
        )
        while timestamps and now - timestamps[0] >= window:
            timestamps.popleft()

    @property
    def available(self) -> bool:
        return self._available

    @property
    def restart_count(self) -> int:
        return self._restart_count

    def mark_unavailable(self) -> None:
        self._available = False

    def consume_bulk_tombstone(self, session_id: str) -> bool:
        """Consume the local proof that a batch DELETE logically closed it."""
        if session_id not in self._bulk_tombstoned_sessions:
            return False
        self._bulk_tombstoned_sessions.discard(session_id)
        return True

    # -- Lifecycle -----------------------------------------------------------

    @staticmethod
    def _validate_gateway_runtime_identity(health: dict[str, Any]) -> None:
        """Fail before rollout when the gateway serves different source code."""
        expected_digest = _installed_gateway_source_digest()
        actual_digest = health.get("gateway_source_digest")
        if (
            expected_digest is not None
            and actual_digest is not None
            and actual_digest != expected_digest
        ):
            raise RuntimeError(
                "Gateway source digest mismatch: "
                f"driver={expected_digest}, gateway={actual_digest}"
            )

        raw_expected = os.environ.get("RLLM_RUNTIME_CODE_PROVENANCE")
        if raw_expected:
            try:
                expected_provenance = json.loads(raw_expected)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "RLLM_RUNTIME_CODE_PROVENANCE is invalid JSON"
                ) from exc
            if health.get("runtime_code_provenance") != expected_provenance:
                raise RuntimeError(
                    "Gateway runtime provenance differs from the training driver"
                )

    def start(self, rollout_engine: RolloutEngine) -> None:
        """Start the gateway and register inference workers.

        For VerlEngine: registers the existing vLLM server addresses.
        """
        engine_cls = type(rollout_engine).__name__

        if engine_cls == "VerlEngine":
            if self.mode == "process":
                self._start_process()
            else:
                self._start_thread()

            worker_urls = self._ensure_verl_engine_workers(rollout_engine)
            self._worker_urls = list(worker_urls)
            for url in worker_urls:
                worker_id = self.client.add_worker(url=url)
                logger.info("Registered worker %s -> %s", worker_id, url)
        else:
            logger.warning("Unknown engine type %s — no workers registered", engine_cls)

        # Extract per-mode sampling params from the rollout engine
        self._train_sampling_params = getattr(rollout_engine, "train_sampling_params", {})
        self._val_sampling_params = getattr(rollout_engine, "val_sampling_params", {})
        self._available = True

        if self.tunnel_backend and not self.public_url:
            self._start_tunnel()

    def _start_tunnel(self) -> None:
        """Spawn the named tunnel backend and pin its public URL onto ``self.public_url``."""
        if self.tunnel_backend != "cloudflared":
            raise ValueError(f"Unsupported gateway tunnel backend: {self.tunnel_backend!r}. Supported: 'cloudflared', or pass an http(s):// URL.")

        from rllm.gateway.tunnel import CloudflaredTunnel

        tunnel = CloudflaredTunnel(self.gateway_url)
        self.public_url = tunnel.start()
        self._tunnel = tunnel

    def stop(self) -> None:
        """Terminate the gateway (process or thread)."""
        self._available = False
        if self._tunnel is not None:
            try:
                self._tunnel.stop()
            except Exception:
                logger.exception("Error stopping cloudflared tunnel")
            self._tunnel = None

        if self._client is not None:
            self._client.close()
            self._client = None

        async_clients = tuple(
            client
            for client in (self._async_client, self._supervision_client)
            if client is not None
        )
        self._async_client = None
        self._supervision_client = None
        if async_clients:
            async def close_async_clients() -> None:
                await asyncio.gather(
                    *(client.close() for client in async_clients),
                    return_exceptions=True,
                )

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(close_async_clients())
            else:
                loop.create_task(
                    close_async_clients(),
                    name="gateway-control-clients-close",
                )

        supervision_executor = self._supervision_executor
        self._supervision_executor = None
        if supervision_executor is not None:
            supervision_executor.shutdown(wait=False, cancel_futures=True)

        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
            self._last_rss_sample = None

        if self._server is not None:
            self._server.should_exit = True
            if self._thread is not None:
                self._thread.join(timeout=5)
            self._thread = None
            self._server = None

        self._local_handler = None

    # -- Session / trace API -------------------------------------------------

    def create_session(
        self,
        session_id: str,
        is_validation: bool = False,
        sampling_params: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        sp = sampling_params if sampling_params is not None else (self._val_sampling_params if is_validation else self._train_sampling_params)
        return self.client.create_session(
            session_id=session_id,
            sampling_params=sp or None,
            metadata=metadata,
        )

    def get_session_url(self, session_id: str, *, public: bool = True) -> str:
        """Session-scoped base URL for the flow's LLM client.

        ``public=False`` returns the host-local gateway URL even when a tunnel
        is up — for flows whose LLM client runs on the host (e.g. a sandboxed
        bash loop driving the sandbox via ``exec``). The tunnel hostname only
        matters to clients *inside* the env, and free quick-tunnel hostnames
        can die mid-run; host-side clients should never depend on them.
        """
        if public and self.public_url:
            base = self.public_url.rstrip("/")
            return f"{base}/sessions/{session_id}/v1"
        return self.client.get_session_url(session_id)

    def get_traces(self, session_id: str) -> list[TraceRecord]:
        return self.client.get_session_traces(session_id)

    # -- Async session / trace API -------------------------------------------

    async def acreate_session(
        self,
        session_id: str,
        is_validation: bool = False,
        sampling_params: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        session_id, _timings = await self.acreate_session_with_timings(
            session_id,
            is_validation=is_validation,
            sampling_params=sampling_params,
            metadata=metadata,
        )
        return session_id

    async def acreate_session_with_timings(
        self,
        session_id: str,
        is_validation: bool = False,
        sampling_params: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, float]]:
        """Create a session and expose admission versus HTTP time separately."""

        admission_started = time.perf_counter()
        await self._session_start_admission.acquire()
        admission_elapsed = time.perf_counter() - admission_started
        sp = sampling_params if sampling_params is not None else (self._val_sampling_params if is_validation else self._train_sampling_params)
        create_started = time.perf_counter()
        created = await self.async_client.create_session(
            session_id=session_id,
            sampling_params=sp or None,
            metadata=metadata,
        )
        return created, {
            "time/session_admission_s": admission_elapsed,
            "time/session_create_s": time.perf_counter() - create_started,
        }

    def runtime_metrics(self) -> dict[str, int | float]:
        """Return trainer-local gateway admission and probe metrics."""
        admission = self._session_start_admission.metrics()
        now = time.monotonic()
        last_success = self._supervision_last_success_monotonic
        values = {
            f"gateway/session_start_admission/{key}": value
            for key, value in admission.items()
        }
        values.update(
            {
                "gateway/supervision/reconnects": self._supervision_reconnects,
                "gateway/supervision/last_probe_reconnected": (
                    self._supervision_last_probe_reconnected
                ),
                "gateway/supervision/seconds_since_success": (
                    max(0.0, now - last_success)
                    if last_success is not None
                    else 0.0
                ),
            }
        )
        return values

    async def aget_routing_stats(self) -> dict[str, Any]:
        return await self.async_client.get_routing_stats()

    async def aget_session_cleanup_stats(self) -> dict[str, Any]:
        return await self.async_client.get_session_cleanup_stats()

    async def aget_runtime_stats(self) -> dict[str, Any]:
        return await self.async_client.get_runtime_stats()

    async def aget_traces(self, session_id: str) -> list[TraceRecord]:
        # Retain the session until receipt and token alignment are confirmed.
        # The engine's existing cleanup path acknowledges successful recovery.
        return await self.async_client.get_session_traces(session_id, consume=False)

    async def asession_exists(self, session_id: str) -> bool:
        """Return whether the gateway still retains a session."""
        return await self.async_client.session_exists(session_id)

    async def adelete_session(self, session_id: str) -> int:
        """Delete a session and all its accumulated traces. Returns count removed."""
        return await self.async_client.delete_session(session_id)

    async def adelete_sessions(self, session_ids: list[str]) -> int:
        """Batch-delete many sessions in one session-scoped request."""
        if not session_ids:
            return 0
        return await self.async_client.delete_sessions(session_ids)

    async def abulk_tombstone_task_groups(
        self,
        task_ids: list[str],
        group_size: int,
        *,
        max_attempts: int = 1,
    ) -> None:
        """Logically close all sessions in a speculative wave with one POST."""
        session_ids = task_group_session_ids(
            task_ids,
            group_size,
            max_attempts=max_attempts,
        )
        if session_ids:
            await self.adelete_sessions(session_ids)
            self._bulk_tombstoned_sessions.update(session_ids)

    async def await_cleanup_ready(self) -> None:
        """Wait until reaper pressure is below one cleanup concurrency window."""
        loop = asyncio.get_running_loop()
        last_progress = loop.time()
        prior: tuple[int, int] | None = None
        while True:
            try:
                stats = await self.aget_session_cleanup_stats()
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                if (
                    loop.time() - last_progress
                    >= self.supervision_config.recovery_stall_timeout_seconds
                ):
                    raise TimeoutError(
                        "gateway cleanup readiness could not be observed for "
                        f"{self.supervision_config.recovery_stall_timeout_seconds:g}s"
                    ) from exc
                await asyncio.sleep(
                    self.supervision_config.health_interval_seconds
                )
                continue
            pending = int(stats.get("pending", 0))
            completed = int(stats.get("completed", 0))
            progress_age = float(stats.get("progress_age_seconds", 0.0) or 0.0)
            current = (pending, completed)
            if pending == 0 or (
                pending < self.session_cleanup_concurrency
                and progress_age
                <= self.supervision_config.health_interval_seconds * 2
            ):
                self._bulk_tombstoned_sessions.clear()
                return
            if prior is None or pending < prior[0] or completed > prior[1]:
                last_progress = loop.time()
            prior = current
            if (
                loop.time() - last_progress
                >= self.supervision_config.recovery_stall_timeout_seconds
            ):
                raise TimeoutError(
                    "gateway session cleanup made no progress for "
                    f"{self.supervision_config.recovery_stall_timeout_seconds:g}s "
                    f"(pending={pending})"
                )
            await asyncio.sleep(self.supervision_config.health_interval_seconds)

    # -- Weight version ------------------------------------------------------

    def set_weight_version(self, weight_version: int) -> None:
        self.client.set_weight_version(weight_version)
        self._current_weight_version = int(weight_version)

    async def aset_weight_version(self, weight_version: int) -> None:
        desired = int(weight_version)

        async def mutate(client: AsyncGatewayClient) -> bool:
            return await client.set_weight_version(desired) == desired

        async def confirm(client: AsyncGatewayClient) -> bool:
            return await client.get_weight_version() == desired

        await self._run_reliable_admin_operation(
            label=f"set weight version {desired}",
            mutate=mutate,
            confirm=confirm,
        )
        self._current_weight_version = desired

    def _probe_supervision_sync(self, timeout: float) -> dict[str, Any]:
        """Probe the side listener from its dedicated client thread."""
        port = self._supervision_port
        if port is None:
            raise GatewaySoftFailure("gateway supervision listener is unavailable")
        connection = http.client.HTTPConnection(
            "127.0.0.1",
            port,
            timeout=timeout,
        )
        try:
            connection.request(
                "GET",
                "/live",
                headers={"Connection": "close"},
            )
            response = connection.getresponse()
            body = response.read()
        except (OSError, TimeoutError, http.client.HTTPException) as exc:
            raise GatewaySoftFailure(
                f"gateway supervision transport failed: {exc!r}"
            ) from exc
        finally:
            connection.close()
        if response.status != 200:
            raise GatewaySoftFailure(
                "gateway supervision listener returned "
                f"HTTP {response.status}"
            )
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise GatewayHardFailure(
                "gateway supervision response is not valid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise GatewayHardFailure(
                "gateway supervision response is not an object"
            )
        return payload

    def _validate_supervision_identity(
        self,
        payload: dict[str, Any],
    ) -> None:
        if payload.get("schema_version") != 1 or payload.get("status") != "ok":
            raise GatewayHardFailure(
                "gateway supervision response has an unsupported schema or status"
            )
        try:
            self._validate_gateway_runtime_identity(payload)
        except RuntimeError as exc:
            raise GatewayHardFailure(str(exc)) from exc
        expected_generation = self._supervision_generation
        if (
            expected_generation is None
            or payload.get("generation") != expected_generation
        ):
            raise GatewayHardFailure(
                "gateway supervision generation mismatch: "
                f"expected={expected_generation!r}, "
                f"actual={payload.get('generation')!r}"
            )
        process = self._process
        if process is None or payload.get("pid") != process.pid:
            raise GatewayHardFailure(
                "gateway supervision PID mismatch: "
                f"expected={getattr(process, 'pid', None)!r}, "
                f"actual={payload.get('pid')!r}"
            )
        for field in (
            "event_loop_heartbeat_age_seconds",
            "seconds_since_last_proxy_success",
            "uptime_seconds",
        ):
            value = payload.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise GatewayHardFailure(
                    f"gateway supervision field {field!r} is invalid"
                )

    async def aprobe_liveness(
        self,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Probe process identity and main-loop progress out of band."""
        if self.mode == "process":
            if self._process is None:
                self._available = False
                raise GatewayHardFailure("gateway process is not running")
            return_code = self._process.poll()
            if return_code is not None:
                self._available = False
                raise GatewayHardFailure(
                    f"gateway process exited with return code {return_code}"
                )
        if self.mode != "process":
            raise GatewaySoftFailure(
                "out-of-band liveness is only available in process mode"
            )
        executor = self._supervision_executor
        if executor is None:
            raise GatewaySoftFailure("gateway supervision executor is closed")
        probe_timeout = float(
            timeout
            if timeout is not None
            else self.supervision_config.health_timeout_seconds
        )
        future = executor.submit(self._probe_supervision_sync, probe_timeout)
        # As with lifecycle jobs, do not rely on a cross-thread loop wakeup.
        try:
            while not future.done():
                await asyncio.sleep(.01)
            payload = future.result()
        except BaseException:
            future.cancel()
            raise
        self._validate_supervision_identity(payload)
        self._available = True
        self._supervision_last_success_monotonic = time.monotonic()
        return {
            "health": {
                key: payload.get(key)
                for key in (
                    "status",
                    "pid",
                    "generation",
                    "gateway_source_digest",
                    "runtime_code_provenance",
                )
            },
            "runtime": {
                "event_loop_heartbeat_age_seconds": payload[
                    "event_loop_heartbeat_age_seconds"
                ],
                "seconds_since_last_proxy_success": payload[
                    "seconds_since_last_proxy_success"
                ],
                "uptime_seconds": payload["uptime_seconds"],
            },
            "liveness": {"channel": "out_of_band", "snapshot_available": 0},
        }

    async def _aget_supervision_snapshot(self) -> dict[str, Any]:
        """Fetch the detailed main-server snapshot within one shared budget."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.supervision_config.health_timeout_seconds
        self._supervision_last_probe_reconnected = 0
        snapshot: dict[str, Any] | None = None
        last_error: httpx.TransportError | None = None
        for attempt in range(2):
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            reused_connection = self._supervision_client is not None
            if self._supervision_client is None:
                self._supervision_client = self._new_supervision_client(
                    remaining
                )
            try:
                snapshot = (
                    await self._supervision_client.get_supervision_snapshot(
                        timeout=remaining,
                    )
                )
                break
            except httpx.TransportError as exc:
                last_error = exc
                logger.warning(
                    "Gateway supervision transport failure phase=snapshot "
                    "attempt=%d/2 reused_client=%s remaining_budget=%.3fs: %r",
                    attempt + 1,
                    reused_connection,
                    max(0.0, deadline - loop.time()),
                    exc,
                )
                await self._close_supervision_client()
                if attempt == 0 and deadline - loop.time() > 0:
                    self._supervision_reconnects += 1
                    self._supervision_last_probe_reconnected = 1
                    continue
                raise
        if snapshot is None:
            if last_error is not None:
                raise last_error
            raise httpx.ConnectTimeout(
                "gateway supervision probe exhausted its timeout budget"
            )
        health = snapshot.get("health")
        runtime = snapshot.get("runtime")
        if not isinstance(health, dict) or not isinstance(runtime, dict):
            raise RuntimeError("gateway supervision snapshot is malformed")
        self._validate_gateway_runtime_identity(health)
        return {"health": health, "runtime": runtime}

    async def aget_supervision_snapshot_best_effort(
        self,
    ) -> dict[str, Any] | None:
        """Return detailed counters without turning metrics loss into failure."""
        try:
            return await self._aget_supervision_snapshot()
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - metrics boundary
            logger.warning(
                "Gateway supervision snapshot unavailable; liveness remains "
                "authoritative: %r",
                exc,
            )
            return None

    async def aprobe_health(self) -> dict[str, Any]:
        """Compatibility probe combining isolated liveness and best-effort stats."""
        if (
            self.mode != "process"
            or self._supervision_port is None
            or self._supervision_generation is None
        ):
            snapshot = await self._aget_supervision_snapshot()
            self._available = True
            self._supervision_last_success_monotonic = time.monotonic()
            return snapshot

        liveness = await self.aprobe_liveness()
        snapshot = await self.aget_supervision_snapshot_best_effort()
        if snapshot is not None:
            liveness_runtime = liveness["runtime"]
            liveness["health"] = {
                **snapshot["health"],
                **liveness["health"],
            }
            liveness["runtime"] = {
                **snapshot["runtime"],
                **liveness_runtime,
            }
            liveness["liveness"]["snapshot_available"] = 1
        return liveness

    def process_stats(self) -> dict[str, int | float]:
        process = self._process
        alive = int(process is not None and process.poll() is None)
        values: dict[str, int | float] = {
            "pid": process.pid if process is not None else 0,
            "alive": alive,
            "restarts": self._restart_count,
            "restarts_in_window": self.restarts_in_window,
            "rss_mb": 0.0,
            "rss_growth_mb_per_minute": 0.0,
            "open_fds": 0,
            "socket_fds": 0,
            "fd_soft_limit": 0,
            "fd_utilization": 0.0,
        }
        if process is None:
            for key in (
                "rss_mb",
                "open_fds",
                "socket_fds",
                "fd_soft_limit",
                "fd_utilization",
                "cgroup_oom",
                "cgroup_oom_kill",
            ):
                if key in self._last_live_process_stats:
                    values[key] = self._last_live_process_stats[key]
            return values
        pid = process.pid
        try:
            statm = Path(f"/proc/{pid}/statm").read_text().split()
            resident_pages = int(statm[1])
            values["rss_mb"] = (
                resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
            )
            if alive:
                now = time.monotonic()
                previous = self._last_rss_sample
                if previous is not None:
                    elapsed = now - previous[0]
                    if elapsed > 0:
                        values["rss_growth_mb_per_minute"] = (
                            float(values["rss_mb"]) - previous[1]
                        ) * 60.0 / elapsed
                self._last_rss_sample = (now, float(values["rss_mb"]))
        except (OSError, ValueError, IndexError):
            pass
        try:
            fd_paths = tuple(Path(f"/proc/{pid}/fd").iterdir())
            values["open_fds"] = len(fd_paths)
            values["socket_fds"] = sum(
                os.readlink(path).startswith("socket:[") for path in fd_paths
            )
        except OSError:
            pass
        try:
            limits = Path(f"/proc/{pid}/limits").read_text().splitlines()
            max_open_files = next(
                line for line in limits if line.startswith("Max open files")
            ).split()
            soft_limit = int(max_open_files[3])
            values["fd_soft_limit"] = soft_limit
            if soft_limit > 0:
                values["fd_utilization"] = (
                    float(values["open_fds"]) / soft_limit
                )
        except (OSError, StopIteration, ValueError, IndexError):
            pass
        for memory_events_path in (
            Path("/sys/fs/cgroup/memory.events"),
            Path(f"/proc/{pid}/root/sys/fs/cgroup/memory.events"),
        ):
            try:
                events = {
                    key: int(value)
                    for key, value in (
                        line.split() for line in memory_events_path.read_text().splitlines()
                    )
                }
            except (OSError, ValueError):
                continue
            values["cgroup_oom"] = events.get("oom", 0)
            values["cgroup_oom_kill"] = events.get("oom_kill", 0)
            break
        if process.poll() is not None:
            values["return_code"] = int(process.returncode or 0)
            if process.returncode is not None and process.returncode < 0:
                values["signal"] = -int(process.returncode)
            for key in (
                "rss_mb",
                "open_fds",
                "socket_fds",
                "fd_soft_limit",
                "fd_utilization",
                "cgroup_oom",
                "cgroup_oom_kill",
            ):
                if values.get(key, 0) == 0 and key in self._last_live_process_stats:
                    values[key] = self._last_live_process_stats[key]
        else:
            self._last_live_process_stats = dict(values)
        return values

    async def aprobe_workers(self, *, timeout: float | None = None) -> list[dict[str, Any]]:
        """Check registered inference servers directly after sidecar recovery."""
        from rllm_model_gateway.control_transport import ControlHTTPTransport

        async with httpx.AsyncClient(
            timeout=timeout or self.worker_health_check_timeout,
            transport=ControlHTTPTransport(worker_probe=True), trust_env=False,
        ) as client:
            async def probe(url: str) -> dict[str, Any]:
                endpoint = url.rstrip("/").removesuffix("/v1") + "/health"
                try:
                    async with asyncio.timeout(timeout or self.worker_health_check_timeout):
                        response = await client.get(endpoint)
                    return {"url": url, "healthy": response.status_code == 200, "status_code": response.status_code}
                except Exception as exc:
                    return {"url": url, "healthy": False, "error": repr(exc)[:1000]}

            return list(await asyncio.gather(*(probe(url) for url in self._worker_urls)))

    async def arestart(
        self,
        *,
        abandoned_session_ids: list[str],
    ) -> None:
        """Restart a failed process sidecar and restore its control state."""
        if self.mode != "process":
            raise RuntimeError("automatic gateway restart requires process mode")
        now = time.monotonic()
        self._prune_restart_timestamps(now)
        restart_window = float(
            getattr(
                self.supervision_config,
                "recovery_stall_timeout_seconds",
                120.0,
            )
        )
        if len(self._restart_timestamps) >= self.supervision_config.max_restarts:
            raise RuntimeError(
                "gateway automatic restart budget exhausted within the "
                f"rolling {restart_window:g}s "
                "window"
            )
        self._restart_timestamps.append(now)
        self._available = False
        old_process = self._process
        if old_process is not None:
            self._last_exit_diagnostics = self.process_stats()
            if old_process.poll() is None:
                old_process.terminate()
                try:
                    await asyncio.to_thread(old_process.wait, 5)
                except subprocess.TimeoutExpired:
                    old_process.kill()
                    await asyncio.to_thread(old_process.wait)
        self._process = None
        self._last_rss_sample = None
        if self._async_client is not None:
            await self._async_client.close()
            self._async_client = None
        await self._close_supervision_client()
        if self._client is not None:
            self._client.close()
            self._client = None

        await asyncio.to_thread(self._start_process)
        for url in self._worker_urls:
            worker_id = await asyncio.to_thread(self.client.add_worker, url)
            logger.info("Re-registered worker %s -> %s", worker_id, url)
        if self._current_weight_version is not None:
            await self.aset_weight_version(self._current_weight_version)
        # Process restart discards server-side tombstones along with all other
        # session state. Reapply both the newly abandoned sessions and any
        # speculative-wave tombstones whose late workflow threads may still be
        # alive; otherwise a late request could recreate them in the fresh
        # gateway process.
        tombstoned_session_ids = list(
            dict.fromkeys(
                (*self._bulk_tombstoned_sessions, *abandoned_session_ids)
            )
        )
        if tombstoned_session_ids:
            await self.adelete_sessions(tombstoned_session_ids)
            self._bulk_tombstoned_sessions.update(tombstoned_session_ids)
        self._restart_count += 1
        self._available = True

    async def aenter_weight_sync_maintenance(
        self,
        weight_version: int,
        *,
        sync_id: str | None = None,
    ) -> str:
        transition_id = sync_id or (
            f"weight-sync-{weight_version}-{uuid.uuid4().hex}"
        )
        enter_id = f"{transition_id}:enter"

        async def mutate(client: AsyncGatewayClient) -> bool:
            status = await client.set_maintenance(
                active=True,
                transition_id=enter_id,
                health_grace_seconds=self.maintenance_health_grace_seconds,
            )
            return bool(status.get("active")) and (
                status.get("transition_id") == enter_id
            )

        async def confirm(client: AsyncGatewayClient) -> bool:
            status = await client.get_maintenance()
            return bool(status.get("active")) and (
                status.get("transition_id") == enter_id
            )

        await self._run_reliable_admin_operation(
            label=f"enter maintenance {enter_id}",
            mutate=mutate,
            confirm=confirm,
        )
        return transition_id

    async def aexit_weight_sync_maintenance(self, sync_id: str) -> None:
        exit_id = f"{sync_id}:exit"

        async def mutate(client: AsyncGatewayClient) -> bool:
            status = await client.set_maintenance(
                active=False,
                transition_id=exit_id,
                health_grace_seconds=self.maintenance_health_grace_seconds,
            )
            return not bool(status.get("active")) and (
                status.get("transition_id") == exit_id
            )

        async def confirm(client: AsyncGatewayClient) -> bool:
            status = await client.get_maintenance()
            return not bool(status.get("active")) and (
                status.get("transition_id") == exit_id
            )

        await self._run_reliable_admin_operation(
            label=f"exit maintenance {exit_id}",
            mutate=mutate,
            confirm=confirm,
        )

    async def _run_reliable_admin_operation(
        self,
        *,
        label: str,
        mutate: Callable[[AsyncGatewayClient], Awaitable[bool]],
        confirm: Callable[[AsyncGatewayClient], Awaitable[bool]],
    ) -> None:
        """Run an idempotent gateway mutation on an isolated connection.

        A timed-out POST may already have committed server-side.  Confirming
        with a GET before retrying turns that ambiguous outcome into success
        and prevents a control-plane timeout from killing the trainer.
        """

        last_error: BaseException | None = None
        async with AsyncGatewayClient(
            self.gateway_url,
            timeout=self.weight_sync_control_timeout_seconds,
            max_connections=2,
        ) as client:
            for attempt in range(1, self.weight_sync_control_max_attempts + 1):
                try:
                    if await mutate(client):
                        return
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code < 500:
                        raise
                    last_error = exc
                except httpx.TransportError as exc:
                    last_error = exc

                try:
                    if await confirm(client):
                        logger.warning(
                            "Gateway control operation %s was confirmed after "
                            "an ambiguous response on attempt %d",
                            label,
                            attempt,
                        )
                        return
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code < 500:
                        raise
                    last_error = exc
                except httpx.TransportError as exc:
                    last_error = exc

                if attempt < self.weight_sync_control_max_attempts:
                    delay = min(
                        8.0,
                        self.weight_sync_control_retry_backoff_seconds
                        * (2 ** (attempt - 1)),
                    )
                    logger.warning(
                        "Gateway control operation %s attempt %d/%d did not "
                        "complete (%s); retrying in %.1fs",
                        label,
                        attempt,
                        self.weight_sync_control_max_attempts,
                        type(last_error).__name__
                        if last_error is not None
                        else "state_not_confirmed",
                        delay,
                    )
                    await asyncio.sleep(delay)

        raise RuntimeError(
            f"Gateway control operation failed after "
            f"{self.weight_sync_control_max_attempts} attempts: {label}"
        ) from last_error

    # -- Worker setup --------------------------------------------------------

    def _ensure_verl_engine_workers(self, rollout_engine: VerlEngine) -> list[str]:
        """Get or create worker URLs for the VerlEngine."""
        addresses = rollout_engine.server_addresses
        return [f"http://{addr}" if not addr.startswith("http") else addr for addr in addresses]

    # -- Internal ------------------------------------------------------------

    def _on_process_started(self) -> None:
        """Optional owner hook, called before health polling can block."""

    def _start_process(self) -> None:
        """Launch gateway as a subprocess and poll until healthy."""
        if self._supervision_port is None:
            self._supervision_port = _find_free_port()
        self._supervision_generation = uuid.uuid4().hex
        cmd = [
            sys.executable,
            "-m",
            "rllm_model_gateway",
            "--host",
            "0.0.0.0",
            "--port",
            str(self.port),
            "--supervision-port",
            str(self._supervision_port),
            "--supervision-generation",
            self._supervision_generation,
        ]
        cmd.extend(["--store", self.store])
        if not self.capture_raw_payloads:
            cmd.append("--no-capture-raw-payloads")
        cmd.extend(
            [
                "--worker-recovery-timeout",
                str(self.worker_recovery_timeout),
                "--session-cleanup-concurrency",
                str(self.session_cleanup_concurrency),
                "--session-cleanup-phase-timeout",
                str(self.session_cleanup_phase_timeout),
                "--session-cleanup-shutdown-timeout",
                str(self.session_cleanup_shutdown_timeout),
                "--session-cleanup-max-pending",
                str(self.session_cleanup_max_pending),
                "--session-cleanup-stall-timeout",
                str(self.session_cleanup_stall_timeout),
                "--worker-health-check-timeout",
                str(self.worker_health_check_timeout),
                "--worker-health-failure-threshold",
                str(self.worker_health_check_failure_threshold),
                "--maintenance-health-grace",
                str(self.maintenance_health_grace_seconds),
                "--recovery-admission-rate",
                str(self.recovery_admission_rate_per_second),
                "--recovery-admission-burst",
                str(self.recovery_admission_burst),
                "--recovery-admission-jitter",
                str(self.recovery_admission_jitter_seconds),
            ]
        )
        if self.db_path:
            cmd.extend(["--db-path", self.db_path])
        if self.model:
            cmd.extend(["--model", self.model])
        if self.cumulative_token_mode:
            cmd.append("--cumulative-token-mode")
            if self.dynamic_sequence_budget:
                cmd.append("--dynamic-sequence-budget")
            if self.max_context_tokens is not None:
                cmd.extend(["--max-context-tokens", str(self.max_context_tokens)])
            if self.renderer_family != "auto":
                cmd.extend(["--renderer-family", self.renderer_family])
        cmd.extend(["--routing-mode", self.routing_config.mode])
        if self.routing_policy:
            cmd.extend(["--routing-policy", self.routing_policy])
        cmd.extend(
            [
                "--routing-migration-min-active-gap",
                str(self.routing_config.migration_min_active_gap),
                "--routing-migration-sustain-seconds",
                str(self.routing_config.migration_sustain_seconds),
                "--routing-max-migrations-per-session",
                str(self.routing_config.max_migrations_per_session),
                "--routing-metrics-interval-seconds",
                str(self.routing_config.metrics_interval_seconds),
            ]
        )

        logger.info("Starting gateway subprocess: %s", " ".join(cmd))
        # Inherit parent's stdout/stderr so gateway logs are visible for debugging.
        # subprocess.PIPE causes problems as without an active reader, the OS pipe
        # buffer (~64KB on Linux) fills up under high-throughput logging, causing the
        # gateway process to block on write and eventually hang.
        self._process = subprocess.Popen(cmd)
        self._on_process_started()
        self._last_rss_sample = None

        # Poll health endpoint
        deadline = time.monotonic() + _HEALTH_POLL_TIMEOUT
        while time.monotonic() < deadline:
            try:
                health = self.client.health()
            except Exception as e:
                if self._process.poll() is not None:
                    raise RuntimeError(f"Gateway process exited unexpectedly (rc={self._process.returncode})") from e
                time.sleep(_HEALTH_POLL_INTERVAL)
                continue
            # A reachable gateway serving different code is not a transient
            # health failure.  Surface it immediately instead of retrying
            # until the generic startup timeout hides the real diagnosis.
            self._validate_gateway_runtime_identity(health)
            if health.get("generation") != self._supervision_generation:
                raise RuntimeError(
                    "Gateway startup generation mismatch: "
                    f"expected={self._supervision_generation!r}, "
                    f"actual={health.get('generation')!r}"
                )
            if health.get("pid") != self._process.pid:
                raise RuntimeError(
                    "Gateway startup PID mismatch: "
                    f"expected={self._process.pid}, actual={health.get('pid')!r}"
                )
            logger.info("Gateway process healthy at %s", self.gateway_url)
            return

        self._process.terminate()
        raise TimeoutError(f"Gateway did not become healthy within {_HEALTH_POLL_TIMEOUT}s")

    def _start_thread(self, local_handler: Any = None) -> None:
        """Start gateway in a background thread using create_app + uvicorn."""
        import uvicorn
        from rllm_model_gateway.models import GatewayConfig
        from rllm_model_gateway.server import create_app

        gw_config = GatewayConfig(
            host="0.0.0.0",
            port=self.port,
            db_path=self.db_path,
            store_worker=self.store,
            model=self.model,
            add_logprobs=self.add_logprobs,
            add_return_token_ids=self.add_return_token_ids,
            capture_raw_payloads=self.capture_raw_payloads,
            cumulative_token_mode=self.cumulative_token_mode,
            dynamic_sequence_budget=self.dynamic_sequence_budget,
            max_context_tokens=self.max_context_tokens,
            worker_recovery_timeout=self.worker_recovery_timeout,
            session_cleanup_concurrency=self.session_cleanup_concurrency,
            session_cleanup_phase_timeout=self.session_cleanup_phase_timeout,
            session_cleanup_shutdown_timeout=(
                self.session_cleanup_shutdown_timeout
            ),
            session_cleanup_max_pending=self.session_cleanup_max_pending,
            session_cleanup_stall_timeout=self.session_cleanup_stall_timeout,
            worker_health_check_timeout=self.worker_health_check_timeout,
            worker_health_check_failure_threshold=(
                self.worker_health_check_failure_threshold
            ),
            maintenance_health_grace_seconds=(
                self.maintenance_health_grace_seconds
            ),
            recovery_admission_rate_per_second=(
                self.recovery_admission_rate_per_second
            ),
            recovery_admission_burst=self.recovery_admission_burst,
            recovery_admission_jitter_seconds=(
                self.recovery_admission_jitter_seconds
            ),
            renderer_family=self.renderer_family,
            routing_policy=self.routing_policy,
            routing=self.routing_config,
        )
        app = create_app(config=gw_config, local_handler=local_handler)

        uvi_config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=self.port,
            log_level="warning",
            timeout_keep_alive=30,
        )
        server = uvicorn.Server(uvi_config)
        self._server = server

        self._thread = threading.Thread(target=server.run, daemon=True)
        self._thread.start()

        # Wait for server to start
        deadline = time.monotonic() + _HEALTH_POLL_TIMEOUT
        while time.monotonic() < deadline:
            if server.started:
                health = self.client.health()
                self._validate_gateway_runtime_identity(health)
                logger.info("Gateway thread healthy at %s", self.gateway_url)
                return
            time.sleep(_HEALTH_POLL_INTERVAL)

        raise TimeoutError(f"Gateway thread did not start within {_HEALTH_POLL_TIMEOUT}s")


# ---------------------------------------------------------------------------
# EvalGatewayManager — eval-side gateway with a static upstream URL
# ---------------------------------------------------------------------------


class EvalGatewayManager(GatewayManager):
    """Gateway pointing at a single static upstream URL (no rollout engine).

    Used by ``rllm eval``: the upstream is either the user's ``--base-url``
    (vLLM endpoint, OpenAI-compatible server) or the URL of a LiteLLM
    proxy started by ``EvalProxyManager``.

    Key differences vs the training-side :class:`GatewayManager`:

    * vLLM-specific request-body injection (``logprobs``,
      ``return_token_ids``) is OFF — external providers reject
      ``return_token_ids`` as an unknown parameter.
    * ``start()`` ignores any ``rollout_engine`` and registers the
      upstream URL passed at construction. URLs are normalized via
      :func:`_normalize_worker_url` (strips trailing ``/v1``).

    Example::

        gw = EvalGatewayManager(upstream_url=base_url, model="gpt-4o")
        gw.start()
        try:
            ...
        finally:
            gw.stop()
    """

    add_logprobs = False
    add_return_token_ids = False

    def __init__(
        self,
        upstream_url: str,
        model: str,
        *,
        host: str = "127.0.0.1",
        port: int | None = None,
        db_path: str | None = None,
        tunnel: str | None = None,
    ) -> None:
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "rllm": {
                    "gateway": {
                        "host": host,
                        "port": port if port is not None else _find_free_port(),
                        "db_path": db_path,
                        "tunnel": tunnel,
                    }
                },
                "model": {"name": model},
            }
        )
        super().__init__(cfg, mode="thread")
        self._upstream_urls: list[str] = [upstream_url]

    def start(self, rollout_engine: RolloutEngine | None = None) -> None:  # type: ignore[override]
        """Start gateway, register the static upstream URL(s), and bring up the tunnel if configured.

        ``rollout_engine`` is accepted for shape-compatibility with the base class but ignored.
        """
        if rollout_engine is not None:
            logger.warning("EvalGatewayManager.start ignores `rollout_engine` argument")
        self._start_thread()
        for raw_url in self._upstream_urls:
            url = _normalize_worker_url(raw_url)
            worker_id = self.client.add_worker(url=url)
            logger.info("Registered worker %s -> %s (raw=%s)", worker_id, url, raw_url)

        self._available = True

        if self.tunnel_backend and not self.public_url:
            self._start_tunnel()
