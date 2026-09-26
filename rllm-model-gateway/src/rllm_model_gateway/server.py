"""FastAPI application factory and CLI entrypoint for rllm-model-gateway."""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from rllm_model_gateway.middleware import SessionRoutingMiddleware
from rllm_model_gateway.models import (
    GatewayConfig,
    WorkerInfo,
)
from rllm_model_gateway.proxy import ReverseProxy
from rllm_model_gateway.session_manager import SessionManager
from rllm_model_gateway.session_router import SessionRouter
from rllm_model_gateway.store.base import TraceStore
from rllm_model_gateway.supervision import SupervisionServer, SupervisionState

logger = logging.getLogger(__name__)


def _gateway_package_digest(package_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(package_dir.rglob("*.py")):
        relative = path.relative_to(package_dir).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


_GATEWAY_SOURCE_DIGEST = _gateway_package_digest(Path(__file__).parent)

_SESSION_CLEANUP_MAX_ATTEMPTS = 3
_SESSION_CLEANUP_RETRY_DELAY_SECONDS = 0.1


# ------------------------------------------------------------------
# Access-log noise filter
# ------------------------------------------------------------------

# Paths whose access lines are filtered from uvicorn.access. These fire on
# every rollout (often dozens of times per task) and crowd out the lines
# that actually help. Successful session create/delete remain visible; routine
# reads and chat completions are represented in higher-level rollout records.
_NOISY_ACCESS_PATHS: tuple[str, ...] = (
    "/admin/flush",
    "/admin/maintenance",
    "/admin/routing/stats",
    "/admin/runtime/stats",
    "/admin/supervision/snapshot",
    "/admin/session_cleanup/stats",
    "/admin/workers",
    "/health",
    "/health/workers",
)


class _AccessLogPathFilter(logging.Filter):
    """Drop uvicorn.access records whose request path is in _NOISY_ACCESS_PATHS.

    uvicorn.access formats records with positional args:
        (client_addr, method, full_path, http_version, status_code)
    """

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not isinstance(args, tuple | list) or len(args) < 5:
            return True
        method = args[1]
        path = args[2]
        if not isinstance(method, str) or not isinstance(path, str):
            return True
        status_code = args[4]
        if not isinstance(status_code, int) or not (200 <= status_code < 300):
            return True
        # Strip query string before matching. Successful chat completions are
        # represented in trace/rollout audit records and dominate logs at a
        # 672-rollout frontier; retain every non-2xx access line.
        bare = path.split("?", 1)[0]
        session_path = bare.removeprefix("/sessions/")
        successful_session_read = (
            method.upper() == "GET"
            and session_path != bare
            and bool(session_path)
            and (
                "/" not in session_path
                or session_path.endswith("/traces")
            )
        )
        return (
            bare not in _NOISY_ACCESS_PATHS
            and not bare.endswith("/v1/chat/completions")
            and not successful_session_read
        )


_access_filter_installed = False


def _install_access_log_filter() -> None:
    """Idempotently attach the path filter to the uvicorn.access logger."""
    global _access_filter_installed
    if _access_filter_installed:
        return
    logging.getLogger("uvicorn.access").addFilter(_AccessLogPathFilter())
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    _access_filter_installed = True


# ------------------------------------------------------------------
# Store factory
# ------------------------------------------------------------------


def create_store(config: GatewayConfig) -> TraceStore:
    worker = config.store_worker
    if worker == "sqlite":
        from rllm_model_gateway.store.sqlite_store import SqliteTraceStore

        return SqliteTraceStore(db_path=config.db_path)
    elif worker == "memory":
        from rllm_model_gateway.store.memory_store import MemoryTraceStore

        return MemoryTraceStore()
    else:
        raise ValueError(f"Unknown store worker: {worker}")


# ------------------------------------------------------------------
# Routing policy loader
# ------------------------------------------------------------------


def _load_policy(dotted_path: str):
    """Import a class by dotted path, e.g. ``my_pkg.policies.CacheAwarePolicy``."""
    module_path, _, cls_name = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(f"Invalid policy path: {dotted_path}")
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)()


# ------------------------------------------------------------------
# App factory
# ------------------------------------------------------------------


def create_app(
    config: GatewayConfig | None = None,
    store: TraceStore | None = None,
    local_handler: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] | None = None,
) -> FastAPI:
    """Create and return a fully configured FastAPI application."""
    if config is None:
        config = GatewayConfig()

    _install_access_log_filter()

    if store is None:
        store = create_store(config)

    # Routing policy
    policy = None
    if config.routing_policy:
        policy = _load_policy(config.routing_policy)

    router = SessionRouter(
        policy=policy,
        routing_config=config.routing,
        health_check_interval=config.health_check_interval,
        health_check_timeout=config.worker_health_check_timeout,
        failure_threshold=config.worker_health_check_failure_threshold,
        maintenance_health_grace_seconds=(
            config.maintenance_health_grace_seconds
        ),
        recovery_admission_rate_per_second=(
            config.recovery_admission_rate_per_second
        ),
        recovery_admission_burst=config.recovery_admission_burst,
        recovery_admission_jitter_seconds=(
            config.recovery_admission_jitter_seconds
        ),
    )

    # Register initial workers
    for i, wc in enumerate(config.workers):
        router.add_worker(
            WorkerInfo(
                worker_id=wc.worker_id or str(i),
                url=wc.url,
                api_path=wc.api_path,
                model_name=wc.model_name,
                weight=wc.weight,
            )
        )

    # Build the renderer for cumulative token mode. The renderer owns
    # message↔token conversion and the cross-turn bridge (see
    # token_accumulator.TokenAccumulator). The tokenizer is loaded from the
    # served model path (``config.model``), which we assume is a complete,
    # unmodified HuggingFace checkpoint.
    renderer = None
    tokenizer = None
    if config.cumulative_token_mode:
        if not config.model:
            raise ValueError("cumulative_token_mode=True requires 'model' to be set in GatewayConfig (path to the served HuggingFace checkpoint).")
        try:
            from renderers import config_from_name, create_renderer
            from transformers import AutoTokenizer
        except ImportError as err:
            raise ImportError("cumulative_token_mode requires the 'renderers' and 'transformers' packages. Install them with: pip install renderers transformers") from err

        tokenizer = AutoTokenizer.from_pretrained(config.model)

        # renderer_family="auto" lets renderers resolve the family by matching the
        # tokenizer's name_or_path against its MODEL_RENDERER_MAP. This succeeds
        # when ``model`` is a canonical HF id (e.g. "Qwen/Qwen3-8B") but misses for
        # a local/custom checkpoint path, which falls back to DefaultRenderer (whose
        # bridge_to_next_turn always returns None, disabling drift protection). When
        # serving from a path, set renderer_family explicitly. Supported families /
        # MODEL_RENDERER_MAP:
        #   https://github.com/PrimeIntellect-ai/renderers/blob/main/renderers/base.py
        renderer = create_renderer(tokenizer, config_from_name(config.renderer_family))
        logger.info(
            "Built %s (family=%r) from %s for cumulative token mode",
            type(renderer).__name__,
            config.renderer_family,
            config.model,
        )
        if type(renderer).__name__ == "DefaultRenderer":
            raise ValueError(
                f"Cumulative token mode resolved to DefaultRenderer for renderer_family="
                f"{config.renderer_family!r} (model={config.model!r}). DefaultRenderer "
                "provides no cross-turn bridge, so drift-free token forwarding is disabled. "
                "renderer_family='auto' only resolves when 'model' is a canonical HuggingFace "
                "id present in renderers' MODEL_RENDERER_MAP; a local/custom checkpoint path "
                "will not match. Either pass a recognized HF id as 'model', or set "
                "renderer_family explicitly to match your model (e.g. 'qwen3', 'qwen3.5', "
                "'qwen3.6', 'glm-5', 'deepseek-v3', 'gpt-oss'). Check supported families in "
                "MODEL_RENDERER_MAP of: https://github.com/PrimeIntellect-ai/renderers/blob/"
                "main/renderers/base.py"
            )

    proxy = ReverseProxy(
        router=router,
        store=store,
        strip_vllm=config.strip_vllm_fields,
        capture_raw_payloads=config.capture_raw_payloads,
        sync_traces=config.sync_traces,
        local_handler=local_handler,
        cumulative_token_mode=config.cumulative_token_mode,
        dynamic_sequence_budget=config.dynamic_sequence_budget,
        max_context_tokens=config.max_context_tokens,
        worker_recovery_timeout=config.worker_recovery_timeout,
        renderer=renderer,
        tokenizer=tokenizer,
    )
    sessions = SessionManager(store)
    # DELETE is split into a synchronous logical close and bounded asynchronous
    # physical reclamation. This keeps speculative-wave cancellation bursts
    # from blocking the gateway control plane on hundreds of model drains.
    session_delete_tasks: dict[str, asyncio.Task[int]] = {}
    session_delete_failures: dict[str, BaseException] = {}
    session_cleanup_failure_counts: Counter[str] = Counter()
    session_cleanup_retry_tasks: set[asyncio.Task[None]] = set()
    session_cleanup_shutting_down = False
    session_cleanup_semaphore = asyncio.Semaphore(
        config.session_cleanup_concurrency
    )
    session_cleanup_metrics: Counter[str] = Counter()
    session_cleanup_stages: dict[str, tuple[str, float]] = {}
    session_cleanup_enqueued_at: dict[str, float] = {}
    detached_request_tasks: dict[str, tuple[asyncio.Task[Any], ...]] = {}
    late_cleanup_tasks: set[asyncio.Future[Any]] = set()
    late_cleanup_tasks_by_session: dict[
        str, set[asyncio.Future[Any]]
    ] = {}
    runtime_state: dict[str, float] = {
        "event_loop_lag_seconds": 0.0,
        "event_loop_heartbeat_monotonic": time.monotonic(),
        "cleanup_last_progress_monotonic": time.monotonic(),
    }
    runtime_provenance = os.environ.get("RLLM_RUNTIME_CODE_PROVENANCE")
    try:
        parsed_runtime_provenance = (
            json.loads(runtime_provenance) if runtime_provenance else None
        )
    except json.JSONDecodeError:
        parsed_runtime_provenance = {"invalid": runtime_provenance}
    supervision_state = (
        SupervisionState(
            generation=config.supervision_generation,
            gateway_source_digest=_GATEWAY_SOURCE_DIGEST,
            runtime_code_provenance=parsed_runtime_provenance,
        )
        if config.supervision_port is not None
        and config.supervision_generation is not None
        else None
    )
    supervision_server = (
        SupervisionServer(port=config.supervision_port, state=supervision_state)
        if config.supervision_port is not None
        and supervision_state is not None
        else None
    )
    if supervision_state is not None:
        proxy.on_proxy_success = supervision_state.note_proxy_success

    def _observe_late_cleanup_task(
        session_id: str,
        phase: str,
        task: asyncio.Future[Any],
    ) -> None:
        if task in late_cleanup_tasks:
            return
        late_cleanup_tasks.add(task)
        late_cleanup_tasks_by_session.setdefault(session_id, set()).add(task)
        session_cleanup_metrics["late_phase_tasks_started"] += 1

        def _finished(done: asyncio.Future[Any]) -> None:
            late_cleanup_tasks.discard(done)
            session_tasks = late_cleanup_tasks_by_session.get(session_id)
            if session_tasks is not None:
                session_tasks.discard(done)
                if not session_tasks:
                    late_cleanup_tasks_by_session.pop(session_id, None)
            session_cleanup_metrics["late_phase_tasks_completed"] += 1
            if done.cancelled():
                session_cleanup_metrics["late_phase_tasks_cancelled"] += 1
                return
            try:
                error = done.exception()
            except BaseException:
                session_cleanup_metrics["late_phase_task_failures"] += 1
                return
            if error is not None:
                session_cleanup_metrics["late_phase_task_failures"] += 1
                logger.warning(
                    "Detached session cleanup operation failed "
                    "session=%s phase=%s: %r",
                    session_id,
                    phase,
                    error,
                )

        task.add_done_callback(_finished)

    async def _cleanup_phase(
        session_id: str,
        phase: str,
        operation: Awaitable[Any],
    ) -> Any:
        started = time.monotonic()
        session_cleanup_stages[session_id] = (phase, started)
        operation_task = asyncio.ensure_future(operation)
        try:
            done, _ = await asyncio.wait(
                (operation_task,),
                timeout=config.session_cleanup_phase_timeout,
            )
            if operation_task not in done:
                # ``asyncio.wait_for`` waits for cancellation acknowledgement
                # after its deadline. A cancellation-resistant HTTP stream can
                # therefore hold a reaper slot forever. Cancel without awaiting
                # acknowledgement and retain a strong reference so its eventual
                # result/exception is still observed.
                operation_task.cancel()
                _observe_late_cleanup_task(
                    session_id,
                    phase,
                    operation_task,
                )
                raise asyncio.TimeoutError
            return operation_task.result()
        except asyncio.TimeoutError:
            session_cleanup_metrics[f"{phase}_timeouts"] += 1
            raise
        except asyncio.CancelledError:
            if not operation_task.done():
                operation_task.cancel()
                _observe_late_cleanup_task(
                    session_id,
                    phase,
                    operation_task,
                )
            raise
        finally:
            duration = time.monotonic() - started
            session_cleanup_metrics[f"{phase}_seconds_total"] += duration
            session_cleanup_metrics[f"{phase}_seconds_max"] = max(
                session_cleanup_metrics[f"{phase}_seconds_max"],
                duration,
            )
            if duration >= 30.0:
                logger.warning(
                    "Slow session cleanup phase session=%s phase=%s "
                    "duration=%.1fs pending=%d",
                    session_id,
                    phase,
                    duration,
                    len(session_delete_tasks),
                )

    async def _delete_session_once(session_id: str) -> int:
        async with session_cleanup_semaphore:
            session_cleanup_metrics["active"] += 1
            session_cleanup_metrics["max_active"] = max(
                session_cleanup_metrics["max_active"],
                session_cleanup_metrics["active"],
            )
            session_cleanup_metrics["started"] += 1
            try:
                last_error: Exception | None = None
                for attempt in range(1, _SESSION_CLEANUP_MAX_ATTEMPTS + 1):
                    try:
                        try:
                            detached = detached_request_tasks.get(session_id, ())
                            request_drain = (
                                asyncio.gather(*detached, return_exceptions=True)
                                if detached
                                else proxy.cancel_session_requests(session_id)
                            )
                            await _cleanup_phase(
                                session_id,
                                "request_cancel",
                                request_drain,
                            )
                        except asyncio.TimeoutError:
                            # The tombstone already prevents all late trace
                            # writes and implicit session recreation. Do not
                            # let an uncooperative upstream generation occupy a
                            # reaper slot forever.
                            logger.warning(
                                "Timed out cancelling requests for closed "
                                "session %s; continuing safe reclamation "
                                "behind its tombstone",
                                session_id,
                            )
                        await _cleanup_phase(
                            session_id,
                            "trace_drain",
                            proxy.wait_for_pending_traces({session_id}),
                        )
                        proxy.discard_session_state(session_id)
                        deleted = await _cleanup_phase(
                            session_id,
                            "store_delete",
                            sessions.delete_session(session_id),
                        )
                        session_cleanup_metrics["completed"] += 1
                        runtime_state["cleanup_last_progress_monotonic"] = (
                            time.monotonic()
                        )
                        return int(deleted)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        last_error = exc
                        session_cleanup_metrics["attempt_failures"] += 1
                        if attempt == _SESSION_CLEANUP_MAX_ATTEMPTS:
                            break
                        session_cleanup_metrics["retry_attempts"] += 1
                        logger.debug(
                            "Session cleanup attempt %d/%d failed for %s: %r",
                            attempt,
                            _SESSION_CLEANUP_MAX_ATTEMPTS,
                            session_id,
                            exc,
                        )
                        await asyncio.sleep(
                            _SESSION_CLEANUP_RETRY_DELAY_SECONDS * attempt
                        )
                assert last_error is not None
                raise last_error
            except asyncio.CancelledError:
                raise
            except Exception:
                session_cleanup_metrics["failed"] += 1
                raise
            finally:
                session_cleanup_metrics["active"] -= 1
                session_cleanup_stages.pop(session_id, None)
                detached_request_tasks.pop(session_id, None)

    def _get_session_delete_task(session_id: str) -> asyncio.Task[int]:
        existing = session_delete_tasks.get(session_id)
        if existing is not None:
            return existing
        prior_failure = session_delete_failures.pop(session_id, None)
        if sessions.is_closed(session_id) and prior_failure is None:
            # Preserve DELETE idempotency after the physical reaper has left
            # the active task map.  The tombstone intentionally outlives the
            # reaper so a late sync-agent thread cannot implicitly resurrect
            # the session.
            completed = asyncio.get_running_loop().create_future()
            completed.set_result(0)
            return completed
        if prior_failure is not None:
            logger.warning(
                "Retrying previously failed session cleanup for %s: %r",
                session_id,
                prior_failure,
            )

        # Tombstone first. GET immediately reports the session absent and late
        # implicit model requests receive 410 instead of reopening routing,
        # accumulators, or trace state while reclamation is pending.
        proxy.seal_session(session_id)
        sessions.close_session(session_id)
        router.close_session(session_id)
        # Release all large, non-durable proxy state at logical-close time.
        # Only the returned active upstream tasks remain for the bounded reaper.
        detached_request_tasks[session_id] = proxy.discard_session_state(
            session_id
        )
        session_cleanup_metrics["enqueued"] += 1
        enqueued_at = session_cleanup_enqueued_at.setdefault(
            session_id,
            time.monotonic(),
        )
        session_cleanup_stages[session_id] = ("queued", enqueued_at)

        task = asyncio.create_task(_delete_session_once(session_id))
        session_delete_tasks[session_id] = task

        def _forget(done: asyncio.Task[int]) -> None:
            if session_delete_tasks.get(session_id) is done:
                session_delete_tasks.pop(session_id, None)
            if done.cancelled():
                session_cleanup_enqueued_at.pop(session_id, None)
                session_cleanup_stages.pop(session_id, None)
                session_cleanup_metrics["cancelled"] += 1
                return
            error = done.exception()
            if error is not None:
                session_delete_failures[session_id] = error
                session_cleanup_failure_counts[session_id] += 1
                session_cleanup_stages[session_id] = (
                    "retry_wait",
                    session_cleanup_enqueued_at[session_id],
                )
                logger.error(
                    "Session deletion failed for %s: %r",
                    session_id,
                    error,
                )
                if not session_cleanup_shutting_down:
                    delay = min(
                        60.0,
                        0.5
                        * (2 ** min(session_cleanup_failure_counts[session_id] - 1, 7)),
                    )

                    async def retry_later() -> None:
                        await asyncio.sleep(delay)
                        if not session_cleanup_shutting_down:
                            _get_session_delete_task(session_id)

                    retry_task = asyncio.create_task(retry_later())
                    session_cleanup_retry_tasks.add(retry_task)
                    retry_task.add_done_callback(
                        session_cleanup_retry_tasks.discard
                    )
            else:
                session_cleanup_enqueued_at.pop(session_id, None)
                session_cleanup_stages.pop(session_id, None)
                session_cleanup_failure_counts.pop(session_id, None)

        task.add_done_callback(_forget)
        return task

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal session_cleanup_shutting_down
        async def monitor_event_loop() -> None:
            interval = 1.0
            expected = time.monotonic() + interval
            while True:
                await asyncio.sleep(interval)
                now = time.monotonic()
                runtime_state["event_loop_lag_seconds"] = max(
                    0.0, now - expected
                )
                runtime_state["event_loop_heartbeat_monotonic"] = now
                if supervision_state is not None:
                    supervision_state.note_event_loop_heartbeat(
                        heartbeat_monotonic=now,
                        last_proxy_success_monotonic=(
                            proxy.last_proxy_success_monotonic
                        ),
                    )
                expected = now + interval

        await proxy.start()
        if supervision_server is not None:
            supervision_server.start()
        event_loop_monitor = asyncio.create_task(monitor_event_loop())
        if router.workers:
            await router.start_health_checks()
        try:
            yield
        finally:
            event_loop_monitor.cancel()
            await asyncio.gather(event_loop_monitor, return_exceptions=True)
            if supervision_server is not None:
                await asyncio.to_thread(supervision_server.stop)
            session_cleanup_shutting_down = True
            await router.stop_health_checks()
        for retry_task in tuple(session_cleanup_retry_tasks):
            retry_task.cancel()
        if session_cleanup_retry_tasks:
            await asyncio.gather(
                *tuple(session_cleanup_retry_tasks),
                return_exceptions=True,
            )
        if session_delete_tasks:
            pending = tuple(session_delete_tasks.values())
            _, stalled = await asyncio.wait(
                pending,
                timeout=config.session_cleanup_shutdown_timeout,
            )
            if stalled:
                logger.warning(
                    "Gateway shutdown cancelling %d stalled session reapers",
                    len(stalled),
                )
                for task in stalled:
                    task.cancel()
        if late_cleanup_tasks:
            late = tuple(late_cleanup_tasks)
            for task in late:
                if not task.done():
                    task.cancel()
            _, still_pending = await asyncio.wait(
                late,
                timeout=config.session_cleanup_shutdown_timeout,
            )
            if still_pending:
                logger.warning(
                    "Gateway shutdown detached %d cancellation-resistant "
                    "cleanup operations",
                    len(still_pending),
                )
        await proxy.stop()
        await store.close()

    app = FastAPI(title="rllm-model-gateway", version="0.1.0", lifespan=lifespan)

    # -- Middleware ---------------------------------------------------------

    # TODO: Add API key auth middleware here for securing gateway access
    # from cloud containers. Validate an API key from the Authorization
    # header before allowing access to admin and proxy endpoints.
    # Add corresponding `api_key` field to GatewayConfig and client classes.

    app.add_middleware(
        SessionRoutingMiddleware,
        add_logprobs=config.add_logprobs,
        add_return_token_ids=config.add_return_token_ids,
        sessions=sessions,
        model=config.model,
    )

    def _health_payload() -> dict[str, Any]:
        return {
            "status": "ok",
            "pid": os.getpid(),
            "generation": config.supervision_generation,
            "gateway_source_digest": _GATEWAY_SOURCE_DIGEST,
            "runtime_code_provenance": parsed_runtime_provenance,
        }

    def _runtime_payload() -> dict[str, Any]:
        now = time.monotonic()
        values = proxy.runtime_stats()
        values.update(
            {
                "sessions": sessions.active_session_count(),
                "event_loop_lag_seconds": runtime_state[
                    "event_loop_lag_seconds"
                ],
                "event_loop_heartbeat_age_seconds": max(
                    0.0,
                    now - runtime_state["event_loop_heartbeat_monotonic"],
                ),
                "cleanup_progress_age_seconds": max(
                    0.0,
                    now - runtime_state["cleanup_last_progress_monotonic"],
                ),
                "cleanup_pending": len(session_delete_tasks),
                "cleanup_late_tasks": len(late_cleanup_tasks),
            }
        )
        store_runtime_stats = getattr(store, "runtime_stats", None)
        if callable(store_runtime_stats):
            values.update(
                {
                    f"trace_store_{key}": value
                    for key, value in store_runtime_stats().items()
                }
            )
        return values

    # -- Health endpoints --------------------------------------------------

    @app.get("/health")
    async def health():
        return _health_payload()

    @app.get("/health/workers")
    async def health_workers():
        workers = router.get_workers()
        return {
            "workers": [w.model_dump() for w in workers],
            "healthy": sum(1 for w in workers if w.healthy),
            "total": len(workers),
        }

    # -- Session endpoints -------------------------------------------------

    @app.post("/sessions")
    async def create_session(request: Request):
        now = time.monotonic()
        cleanup_backlog = len(session_delete_tasks) + len(
            session_cleanup_retry_tasks
        )
        oldest_cleanup = max(
            (
                now - started
                for started in session_cleanup_enqueued_at.values()
            ),
            default=0.0,
        )
        if (
            cleanup_backlog >= config.session_cleanup_max_pending
            or oldest_cleanup >= config.session_cleanup_stall_timeout
        ):
            session_cleanup_metrics["admission_rejections"] += 1
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Gateway session cleanup backlog exceeded its safety threshold",
                    "pending": cleanup_backlog,
                    "oldest_pending_seconds": oldest_cleanup,
                },
            )
        body = await _safe_json(request)
        sid = body.get("session_id") or str(uuid.uuid4())
        if late_cleanup_tasks_by_session.get(sid):
            return JSONResponse(
                status_code=503,
                content={
                    "error": (
                        "Session still has detached cleanup operations; "
                        "retry creation after they finish"
                    ),
                    "session_id": sid,
                },
            )
        pending_delete = session_delete_tasks.get(sid)
        if pending_delete is not None:
            try:
                await asyncio.shield(pending_delete)
            except Exception as exc:
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": f"Prior cleanup for session {sid} failed: {exc}"
                    },
                )
        # Do not consume the failure here.  The background retry owns it; a
        # create racing with retry_wait must receive 503 without accidentally
        # disabling the only remaining physical-reclamation attempt.
        prior_failure = session_delete_failures.get(sid)
        if prior_failure is not None:
            return JSONResponse(
                status_code=503,
                content={
                    "error": f"Prior cleanup for session {sid} failed: {prior_failure}"
                },
            )
        try:
            router.register_session(sid, body.get("metadata"))
        except ValueError as exc:
            return JSONResponse(status_code=409, content={"error": str(exc)})
        sid = sessions.create_session(
            session_id=sid,
            metadata=body.get("metadata"),
            sampling_params=body.get("sampling_params"),
        )
        proxy.open_session(sid)
        return {"session_id": sid, "url": f"/sessions/{sid}/v1"}

    @app.get("/sessions")
    async def list_sessions(
        since: float | None = Query(None),
        limit: int | None = Query(None),
    ):
        result = await sessions.list_sessions(since=since, limit=limit)
        return [s.model_dump() for s in result]

    # NOTE: ``{session_id:path}`` allows multi-segment IDs (e.g.
    # ``harbor/hello-world:0`` from namespaced Harbor tasks). FastAPI/
    # Starlette match routes in declaration order, so the more-specific
    # ``/sessions/{session_id:path}/traces`` MUST be declared before the
    # bare ``/sessions/{session_id:path}`` — otherwise the bare route's
    # greedy capture swallows ``/sessions/foo/traces`` as
    # ``session_id="foo/traces"`` and the traces endpoint is unreachable.

    @app.get("/sessions/{session_id:path}/traces")
    async def get_session_traces(
        session_id: str,
        since: float | None = Query(None),
        limit: int | None = Query(None),
        consume: bool = Query(False),
    ):
        if sessions.is_closed(session_id) or proxy.is_session_sealed(session_id):
            # Logical deletion is immediate even while physical store
            # reclamation is still in the background.
            return []
        # A trace GET is a barrier only for this session.  Using the global
        # /admin/flush here caused every completed fully-async rollout to wait
        # for unrelated sessions and created a control-plane thundering herd.
        barrier_started = time.perf_counter()
        await proxy.wait_for_pending_traces({session_id})
        barrier_seconds = time.perf_counter() - barrier_started
        logger.info("trace_barrier session=%s wait_seconds=%.6f", session_id, barrier_seconds)
        stream_method = getattr(store, "stream_session_traces_json", None)
        if callable(stream_method):
            stream = await stream_method(
                session_id,
                since=since,
                limit=limit,
            )
            if consume:
                # The stream owns a stable snapshot (including duplicated
                # spill-file descriptors), so logical close and physical
                # reclamation can begin before a large response finishes
                # crossing the loopback socket.
                _get_session_delete_task(session_id)
            async def response_body():
                try:
                    async for chunk in stream:
                        yield chunk
                finally:
                    close = getattr(stream, "aclose", None)
                    if close is not None:
                        await close()
            return StreamingResponse(response_body(), media_type="application/json", headers={"X-RLLM-Trace-Barrier-Seconds": str(barrier_seconds)})
        traces = await store.get_session_traces(
            session_id,
            since=since,
            limit=limit,
        )
        if consume:
            _get_session_delete_task(session_id)
        return traces

    @app.get("/sessions/{session_id:path}")
    async def get_session(session_id: str):
        info = await sessions.get_session_info(session_id)
        if info is None:
            return JSONResponse(
                status_code=404,
                content={"error": f"Session {session_id} not found"},
            )
        return info.model_dump()

    @app.delete("/sessions/{session_id:path}")
    async def delete_session(session_id: str):
        task = _get_session_delete_task(session_id)
        return {
            "deleted": 0,
            "status": "complete" if task.done() else "closing",
        }

    @app.post("/sessions/batch_delete")
    async def batch_delete_sessions(request: Request):
        body = await _safe_json(request)
        session_ids = body.get("session_ids", [])
        unique_session_ids = list(dict.fromkeys(session_ids))
        tasks = [_get_session_delete_task(sid) for sid in unique_session_ids]
        return {
            "deleted": 0,
            "closing": sum(not task.done() for task in tasks),
        }

    # -- Trace endpoints ---------------------------------------------------

    @app.get("/traces/{trace_id}")
    async def get_trace(trace_id: str):
        json_method = getattr(store, "get_trace_json", None)
        if callable(json_method):
            trace = await json_method(trace_id)
        else:
            trace = await store.get_trace(trace_id)
        if trace is None:
            return JSONResponse(
                status_code=404,
                content={"error": f"Trace {trace_id} not found"},
            )
        if isinstance(trace, bytes):
            return Response(content=trace, media_type="application/json")
        return trace

    @app.post("/traces/query")
    async def query_traces(request: Request):
        body = await _safe_json(request)
        session_ids = body.get("session_ids", [])
        await proxy.wait_for_pending_traces(set(session_ids))
        stream_method = getattr(store, "stream_trace_query_json", None)
        if callable(stream_method):
            stream = await stream_method(
                session_ids,
                since=body.get("since"),
                limit=body.get("limit"),
            )
            return StreamingResponse(stream, media_type="application/json")
        results: list[dict[str, Any]] = []
        for session_id in session_ids:
            results.extend(
                await store.get_session_traces(
                    session_id,
                    since=body.get("since"),
                    limit=body.get("limit"),
                )
            )
        return results

    # -- Admin endpoints ---------------------------------------------------

    @app.post("/admin/workers")
    async def add_worker(request: Request):
        body = await _safe_json(request)
        url = body.get("url")
        if not url:
            return JSONResponse(status_code=400, content={"error": "url is required"})
        wid = body.get("worker_id", str(uuid.uuid4()))
        # Build kwargs — omit api_path if not provided so the validator auto-splits
        worker_kwargs: dict[str, Any] = {
            "worker_id": wid,
            "url": url,
            "model_name": body.get("model_name"),
            "weight": body.get("weight", 1),
        }
        if "api_path" in body:
            worker_kwargs["api_path"] = body["api_path"]
        worker = WorkerInfo(**worker_kwargs)
        router.add_worker(worker)
        # Start health checks if this is the first worker
        if len(router.workers) == 1:
            await router.start_health_checks()
        return {"worker_id": wid, "url": worker.url, "api_path": worker.api_path}

    @app.delete("/admin/workers/{worker_id}")
    async def remove_worker(worker_id: str):
        worker = next((w for w in router.workers if w.worker_id == worker_id), None)
        if worker is None:
            return JSONResponse(
                status_code=404,
                content={"error": f"Worker {worker_id} not found"},
            )
        router.remove_worker(worker.url)
        return {"removed": worker_id}

    @app.get("/admin/workers")
    async def list_workers():
        workers = router.get_workers()
        return [w.model_dump() for w in workers]

    @app.get("/admin/routing/stats")
    async def routing_stats():
        return router.routing_stats()

    @app.get("/admin/session_cleanup/stats")
    async def session_cleanup_stats():
        now = time.monotonic()
        queued = sum(
            stage == "queued"
            for stage, _ in session_cleanup_stages.values()
        )
        counter_keys = (
            "enqueued",
            "started",
            "completed",
            "failed",
            "cancelled",
            "attempt_failures",
            "retry_attempts",
            "request_cancel_timeouts",
            "trace_drain_timeouts",
            "store_delete_timeouts",
            "max_active",
            "admission_rejections",
            "late_phase_tasks_started",
            "late_phase_tasks_completed",
            "late_phase_tasks_cancelled",
            "late_phase_task_failures",
        )
        phase_keys = tuple(
            f"{phase}_seconds_{suffix}"
            for phase in ("request_cancel", "trace_drain", "store_delete")
            for suffix in ("total", "max")
        )
        values: dict[str, int | float] = {
            key: session_cleanup_metrics[key]
            for key in (*counter_keys, *phase_keys)
        }
        values.update(
            {
                "active": int(session_cleanup_metrics["active"]),
                "queued": int(queued),
                "pending": len(session_delete_tasks),
                "retry_pending": len(session_cleanup_retry_tasks),
                "failed_session_ids": len(session_delete_failures),
                "late_phase_tasks": len(late_cleanup_tasks),
                "late_phase_sessions": len(
                    late_cleanup_tasks_by_session
                ),
                "oldest_pending_seconds": max(
                    (
                        now - started
                        for started in session_cleanup_enqueued_at.values()
                    ),
                    default=0.0,
                ),
                "progress_age_seconds": max(
                    0.0,
                    now
                    - runtime_state["cleanup_last_progress_monotonic"],
                ),
            }
        )
        for phase in (
            "queued",
            "retry_wait",
            "request_cancel",
            "trace_drain",
            "store_delete",
        ):
            values[f"stage_{phase}"] = sum(
                stage == phase
                for stage, _ in session_cleanup_stages.values()
            )
        return values

    @app.get("/admin/runtime/stats")
    async def runtime_stats():
        return _runtime_payload()

    @app.get("/admin/supervision/snapshot")
    async def supervision_snapshot():
        """Return liveness identity and counters in one control request."""
        return {
            "health": _health_payload(),
            "runtime": _runtime_payload(),
        }

    @app.get("/admin/maintenance")
    async def get_maintenance():
        return router.maintenance_status()

    @app.post("/admin/maintenance")
    async def set_maintenance(request: Request):
        body = await _safe_json(request)
        active = body.get("active")
        transition_id = body.get("transition_id")
        if not isinstance(active, bool):
            return JSONResponse(
                status_code=400,
                content={"error": "active must be a boolean"},
            )
        if not isinstance(transition_id, str) or not transition_id.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "transition_id must be a non-empty string"},
            )
        if len(transition_id) > 256:
            return JSONResponse(
                status_code=400,
                content={"error": "transition_id is too long"},
            )
        grace = body.get("health_grace_seconds")
        if grace is not None and (
            isinstance(grace, bool) or not isinstance(grace, int | float)
        ):
            return JSONResponse(
                status_code=400,
                content={"error": "health_grace_seconds must be numeric"},
            )
        try:
            return router.set_maintenance(
                active=active,
                transition_id=transition_id.strip(),
                health_grace_seconds=grace,
            )
        except ValueError as exc:
            return JSONResponse(
                status_code=409,
                content={"error": str(exc)},
            )

    @app.post("/admin/flush")
    async def flush():
        # Retain a global administrative barrier for explicit callers. Normal
        # trace reads and session cleanup use the session-scoped barrier above
        # so unrelated fully-async rollouts cannot block one another.
        await proxy.wait_for_pending_traces()
        await store.flush()
        return {"status": "flushed"}

    @app.post("/admin/reload")
    async def reload():
        # Placeholder for hot-reload
        return {"status": "ok"}

    @app.get("/admin/weight_version")
    async def get_weight_version():
        return {"weight_version": proxy.weight_version}

    @app.post("/admin/weight_version")
    async def set_weight_version(request: Request):
        body = await _safe_json(request)
        version = body.get("weight_version")
        if version is None:
            return JSONResponse(status_code=400, content={"error": "weight_version is required"})
        try:
            proxy.weight_version = int(version)
        except (TypeError, ValueError):
            return JSONResponse(status_code=400, content={"error": f"invalid weight_version: {version!r}"})
        return {"weight_version": proxy.weight_version}

    # -- Proxy catch-all (must be last) ------------------------------------

    @app.api_route(
        "/v1/{path:path}",
        methods=["GET", "POST"],
    )
    async def proxy_v1(request: Request, path: str):
        # Ensure session exists in manager (implicit creation)
        sid = getattr(request.state, "session_id", None)
        if sid:
            if sessions.is_closed(sid) or proxy.is_session_sealed(sid):
                return JSONResponse(
                    status_code=410,
                    content={"error": f"Session {sid} is closed"},
                )
            is_new_session = not sessions.has_session(sid)
            sessions.ensure_session(sid)
            if is_new_session:
                router.register_session(sid)
                proxy.open_session(sid)
        return await proxy.handle(request)

    # Also handle bare /v1 (e.g. /v1/models)
    @app.api_route("/v1", methods=["GET", "POST"])
    async def proxy_v1_root(request: Request):
        sid = getattr(request.state, "session_id", None)
        if sid:
            if sessions.is_closed(sid) or proxy.is_session_sealed(sid):
                return JSONResponse(
                    status_code=410,
                    content={"error": f"Session {sid} is closed"},
                )
            is_new_session = not sessions.has_session(sid)
            sessions.ensure_session(sid)
            if is_new_session:
                router.register_session(sid)
                proxy.open_session(sid)
        return await proxy.handle(request)

    # Store references on app for external access
    app.state.config = config  # type: ignore[attr-defined]
    app.state.router = router  # type: ignore[attr-defined]
    app.state.proxy = proxy  # type: ignore[attr-defined]
    app.state.sessions = sessions  # type: ignore[attr-defined]
    app.state.store = store  # type: ignore[attr-defined]
    app.state.session_delete_tasks = session_delete_tasks  # type: ignore[attr-defined]
    app.state.session_cleanup_metrics = session_cleanup_metrics  # type: ignore[attr-defined]
    app.state.session_cleanup_stages = session_cleanup_stages  # type: ignore[attr-defined]
    app.state.session_cleanup_enqueued_at = session_cleanup_enqueued_at  # type: ignore[attr-defined]
    app.state.late_cleanup_tasks = late_cleanup_tasks  # type: ignore[attr-defined]
    app.state.late_cleanup_tasks_by_session = late_cleanup_tasks_by_session  # type: ignore[attr-defined]

    return app


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


async def _safe_json(request: Request) -> dict[str, Any]:
    try:
        return await request.json()
    except Exception:
        return {}


# ------------------------------------------------------------------
# Config loading
# ------------------------------------------------------------------


def _load_config(args: argparse.Namespace) -> GatewayConfig:
    """Build a ``GatewayConfig`` from CLI args, env vars, and optional YAML file."""
    data: dict[str, Any] = {}

    # 1. YAML file (lowest priority)
    config_path = getattr(args, "config", None)
    if config_path:
        with open(config_path) as f:
            data.update(yaml.safe_load(f) or {})

    # 2. Env vars
    env_map = {
        "RLLM_GATEWAY_HOST": "host",
        "RLLM_GATEWAY_PORT": "port",
        "RLLM_GATEWAY_DB_PATH": "db_path",
        "RLLM_GATEWAY_LOG_LEVEL": "log_level",
        "RLLM_GATEWAY_STORE": "store_worker",
        "RLLM_GATEWAY_CAPTURE_RAW_PAYLOADS": "capture_raw_payloads",
        "RLLM_GATEWAY_MAX_CONTEXT_TOKENS": "max_context_tokens",
        "RLLM_GATEWAY_WORKER_RECOVERY_TIMEOUT": "worker_recovery_timeout",
        "RLLM_GATEWAY_WORKER_HEALTH_CHECK_TIMEOUT": (
            "worker_health_check_timeout"
        ),
        "RLLM_GATEWAY_WORKER_HEALTH_FAILURE_THRESHOLD": (
            "worker_health_check_failure_threshold"
        ),
        "RLLM_GATEWAY_MAINTENANCE_HEALTH_GRACE": (
            "maintenance_health_grace_seconds"
        ),
        "RLLM_GATEWAY_RECOVERY_ADMISSION_RATE": (
            "recovery_admission_rate_per_second"
        ),
        "RLLM_GATEWAY_RECOVERY_ADMISSION_BURST": (
            "recovery_admission_burst"
        ),
        "RLLM_GATEWAY_RECOVERY_ADMISSION_JITTER": (
            "recovery_admission_jitter_seconds"
        ),
    }
    for env_key, config_key in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            if config_key in {
                "port",
                "max_context_tokens",
                "worker_health_check_failure_threshold",
                "recovery_admission_burst",
            }:
                data[config_key] = int(val)
            elif config_key in {
                "worker_recovery_timeout",
                "worker_health_check_timeout",
                "maintenance_health_grace_seconds",
                "recovery_admission_rate_per_second",
                "recovery_admission_jitter_seconds",
            }:
                data[config_key] = float(val)
            elif config_key == "capture_raw_payloads":
                normalized = val.strip().lower()
                if normalized not in {
                    "1",
                    "true",
                    "yes",
                    "on",
                    "0",
                    "false",
                    "no",
                    "off",
                }:
                    raise ValueError(
                        "RLLM_GATEWAY_CAPTURE_RAW_PAYLOADS must be a boolean"
                    )
                data[config_key] = normalized in {"1", "true", "yes", "on"}
            else:
                data[config_key] = val

    # 3. CLI args (highest priority)
    if getattr(args, "host", None) is not None:
        data["host"] = args.host
    if getattr(args, "port", None) is not None:
        data["port"] = args.port
    if getattr(args, "db_path", None) is not None:
        data["db_path"] = args.db_path
    if getattr(args, "log_level", None) is not None:
        data["log_level"] = args.log_level
    if getattr(args, "store", None) is not None:
        data["store_worker"] = args.store
    if getattr(args, "model", None) is not None:
        data["model"] = args.model
    if getattr(args, "cumulative_token_mode", False):
        data["cumulative_token_mode"] = True
    if getattr(args, "dynamic_sequence_budget", False):
        data["dynamic_sequence_budget"] = True
    if getattr(args, "no_capture_raw_payloads", False):
        data["capture_raw_payloads"] = False
    if getattr(args, "renderer_family", None) is not None:
        data["renderer_family"] = args.renderer_family
    if getattr(args, "max_context_tokens", None) is not None:
        data["max_context_tokens"] = args.max_context_tokens
    if getattr(args, "worker_recovery_timeout", None) is not None:
        data["worker_recovery_timeout"] = args.worker_recovery_timeout
    if getattr(args, "session_cleanup_concurrency", None) is not None:
        data["session_cleanup_concurrency"] = args.session_cleanup_concurrency
    if getattr(args, "session_cleanup_phase_timeout", None) is not None:
        data["session_cleanup_phase_timeout"] = (
            args.session_cleanup_phase_timeout
        )
    if getattr(args, "session_cleanup_shutdown_timeout", None) is not None:
        data["session_cleanup_shutdown_timeout"] = (
            args.session_cleanup_shutdown_timeout
        )
    if getattr(args, "session_cleanup_max_pending", None) is not None:
        data["session_cleanup_max_pending"] = args.session_cleanup_max_pending
    if getattr(args, "session_cleanup_stall_timeout", None) is not None:
        data["session_cleanup_stall_timeout"] = (
            args.session_cleanup_stall_timeout
        )
    if getattr(args, "worker_health_check_timeout", None) is not None:
        data["worker_health_check_timeout"] = args.worker_health_check_timeout
    if getattr(args, "worker_health_failure_threshold", None) is not None:
        data["worker_health_check_failure_threshold"] = (
            args.worker_health_failure_threshold
        )
    if getattr(args, "maintenance_health_grace", None) is not None:
        data["maintenance_health_grace_seconds"] = (
            args.maintenance_health_grace
        )
    if getattr(args, "recovery_admission_rate", None) is not None:
        data["recovery_admission_rate_per_second"] = (
            args.recovery_admission_rate
        )
    if getattr(args, "recovery_admission_burst", None) is not None:
        data["recovery_admission_burst"] = args.recovery_admission_burst
    if getattr(args, "recovery_admission_jitter", None) is not None:
        data["recovery_admission_jitter_seconds"] = (
            args.recovery_admission_jitter
        )
    if getattr(args, "routing_policy", None) is not None:
        data["routing_policy"] = args.routing_policy
    routing: dict[str, Any] = dict(data.get("routing") or {})
    if getattr(args, "routing_mode", None) is not None:
        routing["mode"] = args.routing_mode
    if getattr(args, "routing_migration_min_active_gap", None) is not None:
        routing["migration_min_active_gap"] = args.routing_migration_min_active_gap
    if getattr(args, "routing_migration_sustain_seconds", None) is not None:
        routing["migration_sustain_seconds"] = args.routing_migration_sustain_seconds
    if getattr(args, "routing_max_migrations_per_session", None) is not None:
        routing["max_migrations_per_session"] = args.routing_max_migrations_per_session
    if getattr(args, "routing_metrics_interval_seconds", None) is not None:
        routing["metrics_interval_seconds"] = args.routing_metrics_interval_seconds
    if routing:
        data["routing"] = routing

    supervision_port = getattr(args, "supervision_port", None)
    supervision_generation = getattr(args, "supervision_generation", None)
    if supervision_port is not None:
        data["supervision_port"] = supervision_port
    if supervision_generation is not None:
        data["supervision_generation"] = supervision_generation

    # Workers from CLI --worker flags (WorkerConfig validator auto-splits URLs)
    worker_urls = getattr(args, "worker", None) or []
    if worker_urls:
        data["workers"] = [{"url": raw_url, "worker_id": str(i)} for i, raw_url in enumerate(worker_urls)]

    return GatewayConfig(**data)


# ------------------------------------------------------------------
# CLI entrypoint
# ------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="rllm-model-gateway: lightweight LLM call proxy for RL training")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--supervision-port",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--supervision-generation",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument(
        "--worker",
        type=str,
        action="append",
        help="Worker URL (can be repeated)",
    )
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--store", type=str, default=None, choices=["sqlite", "memory"])
    parser.add_argument("--log-level", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="If set, the gateway rewrites every request body's 'model' field to this value before forwarding.",
    )
    parser.add_argument(
        "--cumulative-token-mode",
        action="store_true",
        default=False,
        help="Enable cumulative token mode for drift-free multi-turn RL training. Loads the tokenizer from --model (the served HuggingFace checkpoint).",
    )
    parser.add_argument(
        "--renderer-family",
        type=str,
        default=None,
        help="renderers family for the cumulative-mode bridge (e.g. 'qwen3', 'qwen3.5', "
        "'qwen3.6', 'glm-5', 'deepseek-v3', 'gpt-oss'). Renderers can auto infer it if --model "
        "is a huggingface model id, but if --model is a local path, you must explicitly set it. "
        "Check the supported model families in MODEL_RENDERER_MAP of "
        "https://github.com/PrimeIntellect-ai/renderers/blob/main/renderers/base.py",
    )
    parser.add_argument(
        "--dynamic-sequence-budget",
        action="store_true",
        default=False,
        help=(
            "Use max-context-tokens as the sole trajectory budget and cap "
            "the initial chat turn before forwarding."
        ),
    )
    parser.add_argument(
        "--no-capture-raw-payloads",
        action="store_true",
        default=False,
        help=(
            "Do not retain raw_request/raw_response copies in traces; "
            "normalized messages, token IDs, logprobs and metadata remain."
        ),
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        help="Maximum cumulative prompt + completion tokens. Cumulative requests are capped before forwarding.",
    )
    parser.add_argument(
        "--worker-recovery-timeout",
        type=float,
        default=None,
        help="Seconds to wait when all inference workers are temporarily unhealthy.",
    )
    parser.add_argument(
        "--session-cleanup-concurrency",
        type=int,
        default=None,
        help=(
            "Maximum concurrent background session reapers. This does not "
            "limit rollout/model request concurrency."
        ),
    )
    parser.add_argument(
        "--session-cleanup-phase-timeout",
        type=float,
        default=None,
        help="Maximum seconds for each physical session cleanup phase.",
    )
    parser.add_argument(
        "--session-cleanup-shutdown-timeout",
        type=float,
        default=None,
        help="Seconds to drain background session reapers during gateway shutdown.",
    )
    parser.add_argument(
        "--session-cleanup-max-pending",
        type=int,
        default=None,
        help="Reject new sessions when the physical cleanup backlog reaches this size.",
    )
    parser.add_argument(
        "--session-cleanup-stall-timeout",
        type=float,
        default=None,
        help="Reject new sessions when the oldest cleanup has made no progress for this many seconds.",
    )
    parser.add_argument("--worker-health-check-timeout", type=float, default=None)
    parser.add_argument("--worker-health-failure-threshold", type=int, default=None)
    parser.add_argument("--maintenance-health-grace", type=float, default=None)
    parser.add_argument("--recovery-admission-rate", type=float, default=None)
    parser.add_argument("--recovery-admission-burst", type=int, default=None)
    parser.add_argument("--recovery-admission-jitter", type=float, default=None)
    parser.add_argument(
        "--routing-mode",
        choices=["sticky_least_loaded", "group_striped_adaptive"],
        default=None,
    )
    parser.add_argument("--routing-policy", type=str, default=None)
    parser.add_argument("--routing-migration-min-active-gap", type=int, default=None)
    parser.add_argument("--routing-migration-sustain-seconds", type=float, default=None)
    parser.add_argument("--routing-max-migrations-per-session", type=int, default=None)
    parser.add_argument("--routing-metrics-interval-seconds", type=float, default=None)

    args = parser.parse_args()
    config = _load_config(args)

    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))

    app = create_app(config)

    import uvicorn

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        timeout_keep_alive=30,
    )


if __name__ == "__main__":
    main()
