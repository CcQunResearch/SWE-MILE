"""Lifecycle helpers for standalone VERL rollout server managers.

VERL currently exposes no public close method for a partially initialized
``LLMServerManager``.  These helpers keep evaluation and fully-async training
on the same cleanup/retry path so failed vLLM starts cannot retain actors or
placement groups and poison a subsequent attempt.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class _DisabledProfilerNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage() != "agent loop only support torch and npu profiler, got None"


async def shutdown_vllm_http_server(server: Any) -> None:
    """Release vLLM's IPC and engine processes before Ray terminates the owner."""
    task = getattr(server, "_server_task", None)
    if task is not None and hasattr(task, "cancel"):
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    engine = getattr(server, "engine", None)
    shutdown = getattr(engine, "shutdown", None)
    if callable(shutdown):
        # This actor is being retired. The driver bounds the RPC and kills the
        # actor on expiry; an extra executor can itself outlive actor teardown.
        result = shutdown()
        if inspect.isawaitable(result):
            await result


def install_graceful_vllm_shutdown() -> None:
    # VERL 0.8 has no server close API. A local subclass is cloudpickled to Ray
    # replicas and leaves the installed VERL package unchanged.
    from verl.workers.rollout.vllm_rollout import vllm_async_server

    base = vllm_async_server.vLLMHttpServer
    if hasattr(base, "rllm_shutdown"):
        return

    class GracefulVLLMHttpServer(base):
        def __init__(self, *args, **kwargs):
            vllm_async_server.logger.addFilter(_DisabledProfilerNoiseFilter())
            super().__init__(*args, **kwargs)

        async def rllm_shutdown(self):
            await shutdown_vllm_http_server(self)

    vllm_async_server.vLLMHttpServer = GracefulVLLMHttpServer


def request_graceful_server_shutdown(servers: list, ray_module: Any, timeout: float = 30) -> None:
    refs = []
    for server in servers:
        # ActorHandle only exposes methods present in the remote class.
        method = getattr(server, "rllm_shutdown", None)
        if method is None:
            method = getattr(server, "shutdown", None)
        remote = getattr(method, "remote", None)
        if callable(remote):
            try:
                refs.append(remote())
            except Exception:
                logger.warning("Could not request graceful vLLM shutdown", exc_info=True)
    if refs:
        try:
            ray_module.get(refs, timeout=timeout)
            logger.info("Graceful vLLM shutdown completed for %d servers", len(refs))
        except Exception:
            logger.warning("Graceful vLLM shutdown incomplete after bounded wait; terminating owned actors", exc_info=True)


def teardown_llm_server_manager(
    manager: Any,
    ray_module: Any,
) -> dict[str, int]:
    """Best-effort release of all actors and placement groups owned by a manager."""

    if manager is None:
        return {
            "load_balancers": 0,
            "servers": 0,
            "workers": 0,
            "placement_groups": 0,
            "errors": 0,
        }

    load_balancers = []
    load_balancer = getattr(manager, "global_load_balancer", None)
    if load_balancer is not None:
        load_balancers.append(load_balancer)

    servers = []
    workers = []
    placement_groups = []
    for replica in getattr(manager, "rollout_replicas", ()) or ():
        servers.extend(getattr(replica, "servers", ()) or ())
        workers.extend(getattr(replica, "workers", ()) or ())
        resource_pool = getattr(replica, "resource_pool", None)
        placement_groups.extend(getattr(resource_pool, "pgs", ()) or ())

    errors = 0
    request_graceful_server_shutdown(servers, ray_module)
    # Server actors own the vLLM multiprocessing children. Stop them before
    # checkpoint-engine workers and before releasing placement groups.
    for kind, handles in (
        ("vLLM server", servers),
        ("vLLM worker", workers),
        ("load balancer", load_balancers),
    ):
        for handle in handles:
            try:
                ray_module.kill(handle, no_restart=True)
            except Exception:  # pragma: no cover - exercised against real Ray
                errors += 1
                logger.exception("Failed to terminate %s actor during rollout teardown", kind)

    remove_placement_group = getattr(
        getattr(ray_module, "util", None),
        "remove_placement_group",
        None,
    )
    if placement_groups and remove_placement_group is None:
        errors += len(placement_groups)
        logger.error(
            "Ray has no remove_placement_group API; %d rollout placement "
            "group(s) may retain GPUs",
            len(placement_groups),
        )
    else:
        for placement_group in placement_groups:
            try:
                remove_placement_group(placement_group)
            except Exception:  # pragma: no cover - exercised against real Ray
                errors += 1
                logger.exception("Failed to remove rollout placement group")

    summary = {
        "load_balancers": len(load_balancers),
        "servers": len(servers),
        "workers": len(workers),
        "placement_groups": len(placement_groups),
        "errors": errors,
    }
    logger.info("Requested standalone vLLM teardown: %s", summary)
    return summary


def wait_for_gpu_reclamation(
    ray_module: Any,
    baseline_available: float,
    *,
    timeout_seconds: float,
    stable_seconds: float = 0.0,
    poll_seconds: float = 2.0,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Wait until a failed manager's GPU reservations have been released."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if stable_seconds < 0:
        raise ValueError("stable_seconds must be non-negative")
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")

    deadline = monotonic() + timeout_seconds
    stable_since: float | None = None
    while monotonic() < deadline:
        available = float(ray_module.available_resources().get("GPU", 0.0))
        if available + 1e-6 >= baseline_available:
            if stable_seconds == 0:
                return
            now = monotonic()
            if stable_since is None:
                stable_since = now
            elif now - stable_since >= stable_seconds:
                return
        else:
            stable_since = None
        sleep(poll_seconds)

    available = float(ray_module.available_resources().get("GPU", 0.0))
    raise TimeoutError(
        "rollout GPU resources were not reclaimed after model teardown: "
        f"available={available:g}, expected_at_least={baseline_available:g}"
    )


def _exception_text(exc: BaseException) -> str:
    seen: set[int] = set()
    pending: list[BaseException] = [exc]
    messages: list[str] = []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        messages.append(str(current).lower())
        for nested in (current.__cause__, current.__context__):
            if isinstance(nested, BaseException):
                pending.append(nested)
    return "\n".join(messages)


def is_retryable_server_init_error(exc: BaseException) -> bool:
    """Recognize transient vLLM/TCPStore startup failures through Ray wrappers.

    Ray often only propagates ``Engine core initialization failed`` to the
    manager while the child-process ``EADDRINUSE`` appears exclusively in its
    forwarded logs. Treat that wrapper as retryable as well; attempts remain
    bounded, and each retry first proves that partial GPU reservations returned.
    """

    text = _exception_text(exc)
    return any(
        marker in text
        for marker in (
            "eaddrinuse",
            "address already in use",
            "engine core initialization failed",
            "workerproc initialization failed",
            "worker proc initialization failed",
        )
    )


def initialize_llm_server_manager(manager: Any) -> Any:
    """Initialize a manager while retaining its handle on partial failure."""

    async def initialize() -> None:
        await manager._initialize_llm_servers()
        await manager._init_global_load_balancer()

    asyncio.run(initialize())
    return manager


def create_llm_server_manager_with_retry(
    manager_class: Any,
    *,
    config: Any,
    ray_module: Any,
    model_key: str,
    max_attempts: int,
    retry_quiescence_seconds: float,
    gpu_release_timeout_seconds: float,
    teardown: Callable[[Any, Any], dict[str, int]] = teardown_llm_server_manager,
    wait_for_reclamation: Callable[..., None] = wait_for_gpu_reclamation,
) -> Any:
    """Create standalone vLLM with complete cleanup and bounded retries."""

    if isinstance(max_attempts, bool) or max_attempts <= 0:
        raise ValueError("server_init_max_attempts must be a positive integer")
    if retry_quiescence_seconds < 0:
        raise ValueError("server_init_retry_quiescence_seconds must be non-negative")
    if gpu_release_timeout_seconds <= 0:
        raise ValueError("server_init_gpu_release_timeout_seconds must be positive")

    baseline_available = float(ray_module.available_resources().get("GPU", 0.0))

    for attempt in range(1, max_attempts + 1):
        manager = None
        try:
            manager = manager_class(
                config=config,
                worker_group=None,
                rollout_resource_pool=None,
            )
            return initialize_llm_server_manager(manager)
        except BaseException as exc:
            retryable = is_retryable_server_init_error(exc)
            summary = teardown(manager, ray_module)
            try:
                wait_for_reclamation(
                    ray_module,
                    baseline_available,
                    timeout_seconds=gpu_release_timeout_seconds,
                    stable_seconds=(
                        retry_quiescence_seconds
                        if retryable and attempt < max_attempts
                        else 0.0
                    ),
                )
            except TimeoutError as cleanup_exc:
                raise RuntimeError(
                    f"{model_key} vLLM initialization failed and partial rollout "
                    "resources did not return; "
                    f"initialization_error={exc!s}; cleanup_error={cleanup_exc!s}; "
                    f"teardown={summary}"
                ) from exc

            if not retryable or attempt >= max_attempts:
                raise

            logger.warning(
                "%s standalone vLLM initialization hit a retryable engine "
                "startup failure on attempt %d/%d; partial resources were "
                "reclaimed and initialization will be retried",
                model_key,
                attempt,
                max_attempts,
            )

    raise AssertionError("unreachable")
