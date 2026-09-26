"""Bounded recovery for standalone evaluation, independent of trainer state."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# These describe an unavailable inference runtime, not a benchmark outcome.
RUNTIME_FAILURE_REASONS = frozenset(
    {
        "gateway_unavailable",
        "model_request_retry_exhausted",
    }
)

# A Gateway restart cannot reconcile sandbox leases or restore platform capacity.
# Keep these attempts pending for a later invocation after runtime recovery.
FATAL_RUNTIME_FAILURE_REASONS = frozenset({"minisandbox_cleanup_unconfirmed", "sandbox_capacity_unavailable"})


def runtime_failure_action(reason: str | None, failure: dict | None = None) -> str | None:
    diagnostics = (failure or {}).get("diagnostics") or {}
    if reason in FATAL_RUNTIME_FAILURE_REASONS or diagnostics.get("fatal") is True:
        return "abort"
    if reason in RUNTIME_FAILURE_REASONS:
        return "recover"
    return None


class EvaluationTransportHealth:
    """Stop a systematic transport outage before it consumes the task set."""

    def __init__(self):
        self.failed_tasks: dict[str, str] = {}

    def observe(self, task_id: str, reason: str | None) -> dict | None:
        transport_reasons = {
            "sandbox_exec_not_ready",
            "sandbox_unavailable",
            "setup_storage_probe_failed",
            "agent_exec_unconfirmed",
            "agent_helper_runtime_failed",
        }
        if reason not in transport_reasons:
            self.failed_tasks.clear()
            return None
        self.failed_tasks[task_id] = reason
        if len(self.failed_tasks) < 32:
            return None
        return {"fatal": True, "reason": "evaluation_transport_outage", "distinct_tasks": len(self.failed_tasks), "failures": dict(self.failed_tasks)}


class EvaluationRuntimeError(RuntimeError):
    def __init__(self, message: str, diagnostics: dict | None = None, *, recoverable: bool = True):
        super().__init__(message)
        self.diagnostics = diagnostics or {}
        self.recoverable = recoverable


def minisandbox_node_cleanup_failure(exc: BaseException) -> dict | None:
    """Recognize the typed node-fatal incident even inside RayTaskError."""
    pending, seen = [exc], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if (isinstance(current, EvaluationRuntimeError) and not current.recoverable
                and "minisandbox_cleanup_unconfirmed" in str(current)):
            return {"reason": "minisandbox_cleanup_unconfirmed",
                    "message": str(current)[-2000:], "diagnostics": current.diagnostics}
        for nested in (current.__context__, current.__cause__, getattr(current, "cause", None)):
            if isinstance(nested, BaseException):
                pending.append(nested)
    return None


def run_evaluation_coroutine(coroutine: Any, *, cleanup_timeout: float) -> Any:
    """Close the owner loop without asyncio.run re-waiting forever on cleanup.

    After a supervisor deadline, the owner must return control to its Ray/node
    resource cleanup. Python cannot forcibly stop executor threads; closing the
    loop with wait=False leaves process reclamation to that existing boundary.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    failed = False
    try:
        return loop.run_until_complete(coroutine)
    except BaseException:
        failed = True
        raise
    finally:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        pending.add(loop.create_task(loop.shutdown_asyncgens()))
        try:
            _done, unfinished = loop.run_until_complete(asyncio.wait(pending, timeout=cleanup_timeout))
            if unfinished:
                logger.error("Evaluation loop cleanup exceeded deadline; reclaiming owner resources: %s", [task.get_name() for task in unfinished])
                if not failed:
                    raise EvaluationRuntimeError("evaluation event-loop cleanup deadline exceeded")
        finally:
            loop.close()
            asyncio.set_event_loop(None)


def stalled_runtime(probe: dict, timeout: float) -> bool:
    runtime = probe.get("runtime", {})
    heartbeat = runtime.get("event_loop_heartbeat_age_seconds")
    success = runtime.get("seconds_since_last_proxy_success")
    # Idle/long inference alone is not a dead Gateway: the main loop must also
    # have stopped advancing. In particular long CPU verifiers remain valid.
    return all(isinstance(value, int | float) and not isinstance(value, bool) and value >= timeout for value in (heartbeat, success)) or (
        runtime.get("cleanup_pending", 0) > 0 and runtime.get("cleanup_progress_age_seconds", 0) >= timeout
    )


async def cancel_bounded(task: asyncio.Task, timeout: float) -> None:
    """Do not let wait_for's cancellation handshake extend the deadline."""
    task.cancel()
    done, _ = await asyncio.wait({task}, timeout=timeout)
    if not done:
        raise EvaluationRuntimeError("evaluation cancellation/cleanup deadline exceeded")
    # Observe the task exception without replacing the authoritative incident.
    try:
        task.result()
    except (asyncio.CancelledError, Exception):
        pass


class EvaluationGatewaySupervisor:
    """One lifetime recovery budget per model, with fenced queue generations.

    ``run_generation`` must rescan durable attempts on each invocation. The
    runner must retain its node reservation until cleanup or recovery finishes.
    """

    def __init__(self, gateway: Any, engine: Any, *, record_event: Any):
        self.gateway = gateway
        self.engine = engine
        self.config = gateway.supervision_config
        self.record_event = record_event
        self.generation = 0
        self.accepting_results = True
        self.incident: asyncio.Future | None = None
        self.last_probe: dict = {}
        self.recoveries = 0
        self.last_event: str | None = None

    def report_failure(self, reason: str, diagnostics: dict | None = None, *, recoverable: bool = True) -> None:
        self.accepting_results = False
        self.gateway.mark_unavailable()
        if self.incident is not None and not self.incident.done():
            self.incident.set_result(EvaluationRuntimeError(reason, diagnostics, recoverable=recoverable))

    async def _record(self, event: str, **extra: Any) -> None:
        payload = {
            "event": event,
            "generation": self.generation,
            "recoveries": self.recoveries,
            "probe": self.last_probe,
            "process": self.gateway.process_stats(),
            **extra,
        }
        logger.info("Evaluation Gateway supervision: %s", payload)
        await asyncio.wait_for(self.record_event(payload), timeout=self.config.recovery_stall_timeout_seconds)
        self.last_event = event

    async def _monitor(self) -> None:
        failures = 0
        started: float | None = None
        last_snapshot = 0.0
        workers: list[dict] = []
        while True:
            await asyncio.sleep(self.config.health_interval_seconds)
            cluster = getattr(getattr(self.engine, "hooks", None), "_minisandbox_cluster", None)
            if cluster is not None and cluster.fatal_error:
                self.report_failure(
                    "minisandbox_cleanup_unconfirmed",
                    {
                        **getattr(cluster, "_fatal_diagnostics", {}),
                        "error": cluster.fatal_error,
                        "failure_scope": "node",
                    },
                    recoverable=False,
                )
                return
            if not self.config.enable:
                continue
            hard = False
            stale = False
            error: Exception | None = None
            try:
                self.last_probe = await self.gateway.aprobe_liveness()
                now = time.monotonic()
                if now - last_snapshot >= 30:
                    last_snapshot = now
                    snapshot = await self.gateway.aget_supervision_snapshot_best_effort()
                    if snapshot is not None:
                        self.last_probe = {
                            **self.last_probe,
                            "runtime": {**snapshot.get("runtime", {}), **self.last_probe.get("runtime", {})},
                        }
                    workers = await self.gateway.aprobe_workers(timeout=self.config.health_timeout_seconds)
                    await self._record("heartbeat", workers=workers)
                stale = stalled_runtime(self.last_probe, self.config.recovery_stall_timeout_seconds)
                if stale:
                    error = TimeoutError("Gateway runtime progress stalled")
                elif workers and not all(row["healthy"] for row in workers):
                    error = TimeoutError("vLLM worker health degraded")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                error = exc
                hard = getattr(exc, "gateway_failure_kind", None) == "hard" or not self.gateway.process_stats().get("alive", 0)
            if error is None:
                failures, started = 0, None
                continue
            now = time.monotonic()
            failures += 1
            if started is None:
                started = now - self.config.recovery_stall_timeout_seconds if stale else now
            if not hard and (failures < self.config.failure_threshold or now - started < self.config.recovery_stall_timeout_seconds):
                continue
            if not hard:
                # A fresh, longer out-of-band probe confirms soft failures.
                try:
                    confirmed = await self.gateway.aprobe_liveness(
                        timeout=max(15.0, self.config.health_timeout_seconds * 5),
                    )
                    snapshot = await self.gateway.aget_supervision_snapshot_best_effort()
                    if snapshot is not None:
                        confirmed["runtime"] = {**snapshot.get("runtime", {}), **confirmed.get("runtime", {})}
                    workers = await self.gateway.aprobe_workers(timeout=max(15.0, self.config.health_timeout_seconds * 5))
                    if not stalled_runtime(confirmed, self.config.recovery_stall_timeout_seconds) and workers and all(row["healthy"] for row in workers):
                        failures, started = 0, None
                        continue
                    self.last_probe = confirmed
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    error = exc
            self.report_failure(str(error), {"probe": self.last_probe, "workers": workers})
            return

    async def run(self, run_generation: Any) -> Any:
        try:
            return await self._run(run_generation)
        except (Exception, asyncio.CancelledError) as exc:
            self.accepting_results = False
            self.gateway.mark_unavailable()
            if self.last_event != "failed":
                try:
                    await self._record("failed", error=repr(exc), diagnostics=getattr(exc, "diagnostics", {}))
                except Exception:
                    logger.exception("Failed to persist evaluation runtime failure")
            raise

    async def _run(self, run_generation: Any) -> Any:
        while True:
            self.accepting_results = True
            self.incident = asyncio.get_running_loop().create_future()
            work = asyncio.create_task(run_generation(self.generation), name="evaluation-queue")
            monitor = (
                asyncio.create_task(self._monitor(), name="evaluation-gateway-supervisor")
                if (self.config.enable or getattr(getattr(self.engine, "hooks", None), "_minisandbox_cluster", None) is not None)
                else None
            )
            failure: EvaluationRuntimeError | None = None
            try:
                watched = {work, self.incident}
                if monitor is not None:
                    watched.add(monitor)
                await asyncio.wait(watched, return_when=asyncio.FIRST_COMPLETED)
                if self.incident.done():
                    failure = self.incident.result()
                elif monitor is not None and monitor.done():
                    # An unexpected monitor exception must never silently
                    # remove supervision from a still-running evaluation.
                    monitor.result()
                    raise EvaluationRuntimeError("Gateway supervisor stopped unexpectedly")
                else:
                    result = work.result()
                    cluster = getattr(getattr(self.engine, "hooks", None), "_minisandbox_cluster", None)
                    if cluster is None or not cluster.fatal_error:
                        return result
                    # Final teardown can quarantine the node after the last
                    # periodic probe. Never report a successful run then.
                    failure = EvaluationRuntimeError(
                        "minisandbox_cleanup_unconfirmed",
                        {
                            **getattr(cluster, "_fatal_diagnostics", {}),
                            "error": cluster.fatal_error,
                            "failure_scope": "node",
                        },
                        recoverable=False,
                    )
                self.accepting_results = False
                self.gateway.mark_unavailable()
                await self._record("incident", error=str(failure), diagnostics=failure.diagnostics)
            finally:
                # A cancelled flow closes OpenAI and tears down its sandbox.
                # Gateway cleanup is skipped while unavailable; arestart
                # reinstalls tombstones before any replacement can launch.
                try:
                    if monitor is not None:
                        await cancel_bounded(monitor, self.config.recovery_stall_timeout_seconds)
                finally:
                    if not work.done():
                        self.accepting_results = False
                        self.gateway.mark_unavailable()
                        await cancel_bounded(work, self.config.recovery_stall_timeout_seconds)
                    else:
                        try:
                            work.result()
                        except (asyncio.CancelledError, Exception):
                            pass
                if self.incident is not None and not self.incident.done():
                    self.incident.cancel()
            if failure is None:
                raise EvaluationRuntimeError("evaluation stopped without a result")
            # Configuration may disable recovery, but cannot expand the
            # explicitly bounded one-recovery lifetime policy for evaluation.
            if not failure.recoverable or not self.config.enable or self.recoveries >= min(1, self.config.max_restarts):
                await self._record("failed", error=str(failure))
                raise failure
            self.recoveries += 1
            abandoned = sorted(self.engine.evaluation_session_ids)
            await self._record("recovering", abandoned_sessions=len(abandoned))
            try:
                async with asyncio.timeout(self.config.recovery_stall_timeout_seconds):
                    await self.gateway.arestart(abandoned_session_ids=abandoned)
                    self.last_probe = await self.gateway.aprobe_liveness()
                    workers = await self.gateway.aprobe_workers()
                    if not workers or not all(row["healthy"] for row in workers):
                        raise EvaluationRuntimeError("vLLM workers unhealthy after Gateway restart", {"workers": workers})
                    if stalled_runtime(self.last_probe, self.config.recovery_stall_timeout_seconds):
                        raise EvaluationRuntimeError("Gateway remains stalled after restart")
            except Exception as exc:
                self.gateway.mark_unavailable()
                await self._record("failed", error=repr(exc), diagnostics=getattr(exc, "diagnostics", {}))
                raise EvaluationRuntimeError("Gateway recovery failed") from exc
            self.generation += 1
            await self._record("recovered", workers=workers)
