"""AgentFlowEngine: runs AgentFlows with gateway-mediated trace capture.

Single execution engine for both training and eval. Each rollout:

1. ``hooks.setup(task, agent_flow, uid)`` runs — sandbox-style hooks create
   a per-task sandbox + resolve a per-task verifier; a bare ``evaluator=``
   is wrapped in :class:`rllm.hooks.FixedEvaluatorHooks` so the engine has
   exactly one execution path.
2. The agent flow runs against the gateway session URL.
3. Traces are fetched and the Episode is enriched with token-level Steps
   (strict for training, relaxed for validation).
4. The hook-resolved evaluator scores the enriched Episode.
5. Reward is written back; the hook context is torn down. Sessions are
   batch-deleted from the trace store at the end of the step.

Eval and training differ only in which hooks they install — the per-task
pipeline in :meth:`_run_single` is identical.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import os
import re
import resource
import threading
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import httpx
from tqdm import tqdm

from rllm.data.utils import task_from_row
from rllm.engine.trace_converter import compute_step_metrics, trace_record_to_step
from rllm.eval.types import EvalOutput, Signal
from rllm.gateway.manager import container_reachable_url
from rllm.gateway.session_ids import retry_session_id, split_rollout_session_id
from rllm.harnesses.action_event import summarize_codeflow_bash_behavior
from rllm.rewards.milestone_reward import (
    MILESTONE_REWARD_METADATA_KEY,
    MilestoneRewardConfig,
    annotate_episode_milestone_rewards,
    annotate_episode_milestone_verification_metrics,
    compute_milestone_format_reward,
    get_milestone_reward,
    summarize_milestone_verification,
)
from rllm.types import (
    Action,
    AgentConfig,
    Episode,
    RolloutInfrastructureError,
    ShadowFinalizationError,
    Step,
    Task,
    Trajectory,
    flow_accepts_env,
    flow_accepts_keyword,
    run_agent_flow,
)
from rllm.utils import colorful_print
from rllm.utils.tool_call_reward import annotate_step_tool_call_reward, compute_tool_call_reward
from rllm.workflows.workflow import TerminationReason

if TYPE_CHECKING:
    from rllm_model_gateway.models import TraceRecord

    from rllm.gateway.manager import GatewayManager
    from rllm.types import AgentFlow, Evaluator
    from rllm.utils.episode_logger import EpisodeLogger

logger = logging.getLogger(__name__)



_MIN_FD_LIMIT = 8192
_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_SESSION_CLEANUP_ATTEMPTS = 3
_SESSION_CLEANUP_RETRY_DELAY_S = 0.1
_SESSION_CLEANUP_TIMEOUT_S = 30.0
_SESSION_CLEANUP_CONFIRM_TIMEOUT_S = 30.0
_SESSION_CLEANUP_CONFIRM_POLL_S = 0.5
_SESSION_CLEANUP_STATUS_TIMEOUT_S = 10.0
_CONTEXT_METADATA_KEY = "rllm_context"
_DENOVO_BACKGROUND_METADATA_KEY = "denovo_background_finalize"
_DENOVO_BACKGROUND_ACTIVE_KEY = "denovo_background_finalize_active"
_DISCARD_CONTEXT_TERMINAL = "discard_context_terminal"
_LIMIT_REWARD_AUTO = "auto"
_LIMIT_REWARD_FALLBACK = 0.6
_FULL_VERIFICATION_TOLERANCE = 1e-12
_EXECUTE_TASKS_HEARTBEAT_S = 60.0
_PROGRESS_UNSET = object()
_DYNAMIC_SAMPLING_ROLLOUT_CONTEXT = "_rllm_dynamic_sampling_rollout_context"
_DYNAMIC_SAMPLING_STATUSES = frozenset({"consumed", "filtered"})
_GATEWAY_TRANSPORT_ERROR_NAMES = frozenset(
    {
        "ConnectError",
        "ConnectTimeout",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "PoolTimeout",
        "GatewayUnavailableError",
    }
)


def _denovo_background_contract_active(task_metadata: Any) -> bool:
    """Return whether task metadata opted into detached DeNovo finalization."""

    if not isinstance(task_metadata, dict):
        return False
    rllm_metadata = task_metadata.get("rllm")
    return bool(
        isinstance(rllm_metadata, dict)
        and rllm_metadata.get(_DENOVO_BACKGROUND_ACTIVE_KEY, False)
    )


def _attach_denovo_background_failure_lifecycle(
    metadata: dict[str, Any],
    task_metadata: Any,
    *,
    reason: str,
    stage: str | None,
) -> None:
    """Keep pre-lease failures inside the task's DeNovo lifecycle contract.

    A rollout can fail before ownership of a detached shadow lease reaches the
    batch coordinator.  Persisting that state prevents a recoverable sibling
    cancellation from looking like a mixed DeNovo/legacy group.
    """

    if not _denovo_background_contract_active(task_metadata):
        return
    lifecycle = metadata.get(_DENOVO_BACKGROUND_METADATA_KEY)
    lifecycle = dict(lifecycle) if isinstance(lifecycle, dict) else {}
    lifecycle.update(
        {
            "schema_version": 1,
            "status": "failed_before_lease",
            "shadow_lease_registered": False,
            "background_failure_reason": str(reason),
            "background_failure_stage": str(stage or "rollout"),
        }
    )
    metadata[_DENOVO_BACKGROUND_METADATA_KEY] = lifecycle


def _rollout_reward_log_precision() -> int:
    """Follow the training script's selected task profile."""

    return 3 if os.environ.get("TASK_PROFILE") == "denovoswe" else 1


def _format_rollout_reward_for_log(
    reward: float,
) -> str:
    precision = _rollout_reward_log_precision()
    return f"{reward:.{precision}f}"


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Ignoring invalid integer %s=%r; using %d", name, value, default)
        return default


_TRACE_FETCH_TIMEOUT_SECONDS = 600.0


class EnrichMismatchError(RuntimeError):
    """Raised when gateway traces don't align with the agent's reported steps.

    Indicates a real upstream failure (lost trace, empty token_ids in the vLLM
    response, etc.). The engine classifies it as infrastructure failure
    without re-executing the completed agent.
    """


class GatewaySessionCleanupError(RuntimeError):
    """Raised when a completed attempt cannot release its gateway session."""


def _is_gateway_transport_failure(error: BaseException) -> bool:
    """Recognize failures which make a rollout semantically unclassifiable."""
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ in _GATEWAY_TRANSPORT_ERROR_NAMES:
            return True
        if isinstance(
            current,
            httpx.ConnectError
            | httpx.ConnectTimeout
            | httpx.ReadError
            | httpx.ReadTimeout
            | httpx.RemoteProtocolError
            | httpx.PoolTimeout,
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _is_sandbox_transport_failure(error: BaseException) -> bool:
    """Recognize provider/control-plane failures, not test outcome text."""

    type_names = {
        "AuthError",
        "CreateError",
        "ExecError",
        "RemoteDisconnected",
    }
    evidence = (
        "wait sandbox ready timeout",
        "sandbox already destroyed",
        "job_instance_id=",
    )
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ in type_names:
            return True
        normalized = str(current).lower()
        if any(marker in normalized for marker in evidence):
            return True
        current = current.__cause__ or current.__context__
    return False


def _trace_request_id(trace: TraceRecord) -> str | None:
    metadata = getattr(trace, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    request = metadata.get("rllm_request")
    if not isinstance(request, dict) or not request.get("request_id"):
        return None
    return str(request["request_id"])


def _step_request_id(step: Step) -> str | None:
    metadata = step.metadata if isinstance(step.metadata, dict) else {}
    value = metadata.get("model_request_id")
    return str(value) if value else None


def _trace_semantic_signature(trace: TraceRecord) -> str:
    """Hash response semantics while ignoring trace timing/UUID provenance."""
    payload = {
        "request_id": _trace_request_id(trace),
        "messages": getattr(trace, "messages", []),
        "prompt_token_ids": getattr(trace, "prompt_token_ids", []),
        "response_message": getattr(trace, "response_message", {}),
        "completion_token_ids": getattr(trace, "completion_token_ids", []),
        "logprobs": getattr(trace, "logprobs", None),
        "finish_reason": getattr(trace, "finish_reason", None),
        "weight_version": getattr(trace, "weight_version", None),
        "metadata": getattr(trace, "metadata", {}),
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _raw_episode_with_enrichment_status(
    episode: Episode,
    *,
    status: str,
    trace_count: int,
    step_count: int,
    duplicate_count: int = 0,
    missing_count: int = 0,
    extra_count: int = 0,
    conflict_count: int = 0,
) -> Episode:
    metrics = dict(episode.metrics or {})
    metrics.update(
        {
            "enrichment/status": status,
            "enrichment/trace_count": trace_count,
            "enrichment/step_count": step_count,
            "enrichment/deduplicated_traces": duplicate_count,
            "enrichment/missing_traces": missing_count,
            "enrichment/extra_traces": extra_count,
            "enrichment/conflicting_traces": conflict_count,
        }
    )
    return Episode(
        id=episode.id,
        task=episode.task,
        is_correct=episode.is_correct,
        termination_reason=episode.termination_reason,
        trajectories=[
            trajectory
            if isinstance(trajectory, Trajectory)
            else Trajectory(**trajectory.model_dump())
            for trajectory in episode.trajectories
        ],
        metrics=metrics,
        metadata=episode.metadata,
        artifacts=episode.artifacts,
    )


@dataclass(frozen=True)
class _LimitRewardResolution:
    reward: float
    mode: str
    total_turns: int
    first_full_verification_step_index: int | None = None
    first_full_verification_turn: int | None = None
    first_full_verification_potential: float | None = None
    fallback: bool = False
    fallback_reason: str | None = None

    def to_metadata(self, trajectory: Trajectory | None = None) -> dict[str, Any]:
        return {
            "trajectory_uid": trajectory.uid if trajectory is not None else None,
            "trajectory_name": trajectory.name if trajectory is not None else None,
            "reward": self.reward,
            "mode": self.mode,
            "total_turns": self.total_turns,
            "first_full_verification_step_index": self.first_full_verification_step_index,
            "first_full_verification_turn": self.first_full_verification_turn,
            "first_full_verification_potential": self.first_full_verification_potential,
            "fallback": self.fallback,
            "fallback_reason": self.fallback_reason,
        }


def _normalize_limit_termination_reward(value: Any) -> float | str:
    if isinstance(value, bool):
        raise ValueError(
            "agent_flow.limit_termination_success_reward must be 'auto' or a float in [0.0, 1.0], "
            f"got {value!r}"
        )
    if isinstance(value, str) and value.strip().lower() == _LIMIT_REWARD_AUTO:
        return _LIMIT_REWARD_AUTO
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "agent_flow.limit_termination_success_reward must be 'auto' or a float in [0.0, 1.0], "
            f"got {value!r}"
        ) from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(
            "agent_flow.limit_termination_success_reward must be 'auto' or a float in [0.0, 1.0], "
            f"got {value!r}"
        )
    return parsed


def _resolve_limit_reward(
    configured: float | str,
    trajectory: Trajectory | None,
    milestone_config: MilestoneRewardConfig | None,
    milestone_annotation_error: str | None,
) -> _LimitRewardResolution:
    total_turns = len(trajectory.steps) if trajectory is not None else 0
    if configured != _LIMIT_REWARD_AUTO:
        return _LimitRewardResolution(
            reward=float(configured),
            mode="fixed",
            total_turns=total_turns,
        )

    fallback_reason: str | None = None
    if milestone_config is None or not milestone_config.enable:
        fallback_reason = "milestone_disabled"
    elif not milestone_config.verification_enable:
        fallback_reason = "verification_potential_disabled"
    elif milestone_annotation_error is not None:
        fallback_reason = "milestone_annotation_error"
    elif trajectory is None or not trajectory.steps:
        fallback_reason = "empty_trajectory"
    else:
        records = [get_milestone_reward(step) for step in trajectory.steps]
        if any(record is None for record in records):
            fallback_reason = "milestone_annotation_missing"
        else:
            verification_was_available = False
            for step_index, record in enumerate(records):
                assert record is not None
                verification_was_available = verification_was_available or record.verification_available
                if (
                    record.verification_available
                    and record.verification_updated
                    and record.verification_potential_before < 1.0 - _FULL_VERIFICATION_TOLERANCE
                    and math.isclose(
                        record.verification_potential_after,
                        1.0,
                        rel_tol=0.0,
                        abs_tol=_FULL_VERIFICATION_TOLERANCE,
                    )
                ):
                    turn = step_index + 1
                    return _LimitRewardResolution(
                        reward=turn / total_turns,
                        mode=_LIMIT_REWARD_AUTO,
                        total_turns=total_turns,
                        first_full_verification_step_index=step_index,
                        first_full_verification_turn=turn,
                        first_full_verification_potential=record.verification_potential_after,
                    )
            fallback_reason = "verification_potential_never_reached_one" if verification_was_available else "verification_potential_unavailable"

    return _LimitRewardResolution(
        reward=_LIMIT_REWARD_FALLBACK,
        mode=_LIMIT_REWARD_AUTO,
        total_turns=total_turns,
        fallback=True,
        fallback_reason=fallback_reason,
    )


@dataclass
class TaskContext:
    """Per-task state returned by :meth:`TaskHooks.setup`.

    Encapsulates the per-task evaluator (resolved by the hook from a task's
    [verifier] config, or pre-bound by the caller), the task's live sandbox
    (``env``, ``None`` for host-only rollouts) with the backend that
    provisioned it, and a teardown callback that releases any per-task
    resources (sandboxes, temp dirs, ...).
    """

    evaluator: Evaluator
    env: Any = None  # Sandbox | None — kept loose for the import-cycle reason below
    env_backend: str | None = None  # backend that actually provisioned env
    shadow_runtime: Any = None  # Optional rollout-local shadow worker
    outcome_source: str = "primary_verifier"
    teardown: Any = None  # Callable[[], None] | None — kept loose to avoid Callable import loop
    primary_process_audit: Any = None
    primary_teardown: Any = None
    shadow_teardown: Any = None
    _teardown_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )
    _teardown_started: bool = field(default=False, init=False, repr=False)
    _primary_teardown_started: bool = field(default=False, init=False, repr=False)
    _shadow_teardown_started: bool = field(default=False, init=False, repr=False)
    _shadow_transferred: bool = field(default=False, init=False, repr=False)
    _teardown_futures: dict[str, Future] = field(default_factory=dict, init=False, repr=False)

    def _run_teardown_callback(self, callback: Any, label: str) -> None:
        with self._teardown_lock:
            future = self._teardown_futures.get(label)
            owner = future is None
            if owner:
                future = Future()
                self._teardown_futures[label] = future
        if owner:
            try:
                if callback is not None:
                    callback()
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(None)
        # Repeated calls observe the original completion/failure. "Started"
        # must never be mistaken for confirmed cleanup.
        future.result()

    def run_primary_teardown(self) -> None:
        with self._teardown_lock:
            self._primary_teardown_started = True
            callback = self.primary_teardown
        self._run_teardown_callback(callback, "primary_teardown")

    def run_shadow_teardown(self) -> None:
        with self._teardown_lock:
            self._shadow_teardown_started = True
            callback = self.shadow_teardown
        self._run_teardown_callback(callback, "shadow_teardown")

    def transfer_shadow_teardown(self) -> Any:
        """Transfer shadow ownership to a deferred batch-finalization lease.

        The returned idempotent callback remains the only supported way for
        the new owner to close the shadow.  ``run_teardown`` will continue to
        close the primary, but will no longer abort the transferred worker.
        """

        with self._teardown_lock:
            if self._shadow_teardown_started:
                raise RuntimeError("shadow teardown already started")
            if self._shadow_transferred:
                raise RuntimeError("shadow teardown already transferred")
            if self.shadow_teardown is None:
                raise RuntimeError("task context has no shadow teardown")
            self._shadow_transferred = True
        return self.run_shadow_teardown

    def run_teardown(self) -> None:
        with self._teardown_lock:
            self._teardown_started = True
            teardown = self.teardown
            shadow_transferred = self._shadow_transferred
        # Preserve the historical shadow-before-primary ordering.  Legacy
        # contexts still use the single combined callback below.
        def cleanup_all():
            from rllm.utils.shutdown import shutdown_components
            steps = [] if shadow_transferred else [("shadow", self.run_shadow_teardown)]
            steps.extend([("primary", self.run_primary_teardown), ("legacy", lambda: teardown() if teardown else None)])
            shutdown_components(steps)
        self._run_teardown_callback(cleanup_all, "teardown")


@dataclass
class _ShadowResourceRequest:
    count: int
    cpus: int
    memory_mb: int
    cleanup_unconfirmed: bool = field(default=False, init=False, compare=False)
    release_task: asyncio.Task | None = field(default=None, init=False, compare=False, repr=False)


class _ShadowResourceBudget:
    """Async count/CPU/memory admission for detached DeNovo shadows."""

    def __init__(
        self,
        *,
        max_count: int,
        max_cpus: int,
        max_memory_mb: int,
    ) -> None:
        self.max_count = max_count
        self.max_cpus = max_cpus
        self.max_memory_mb = max_memory_mb
        self._used_count = 0
        self._used_cpus = 0
        self._used_memory_mb = 0
        self._condition = asyncio.Condition()
        self.peak_count = 0
        self.peak_cpus = 0
        self.peak_memory_mb = 0

    async def acquire(self, request: _ShadowResourceRequest) -> None:
        if (
            request.count > self.max_count
            or request.cpus > self.max_cpus
            or request.memory_mb > self.max_memory_mb
        ):
            raise ValueError(
                "one shadow resource request exceeds the configured DeNovo "
                f"background budget: request={request} limits="
                f"({self.max_count}, {self.max_cpus}, {self.max_memory_mb})"
            )
        async with self._condition:
            await self._condition.wait_for(
                lambda: (
                    self._used_count + request.count <= self.max_count
                    and self._used_cpus + request.cpus <= self.max_cpus
                    and self._used_memory_mb + request.memory_mb
                    <= self.max_memory_mb
                )
            )
            self._used_count += request.count
            self._used_cpus += request.cpus
            self._used_memory_mb += request.memory_mb
            self.peak_count = max(self.peak_count, self._used_count)
            self.peak_cpus = max(self.peak_cpus, self._used_cpus)
            self.peak_memory_mb = max(
                self.peak_memory_mb,
                self._used_memory_mb,
            )

    async def release(self, request: _ShadowResourceRequest) -> None:
        async with self._condition:
            self._used_count = max(0, self._used_count - request.count)
            self._used_cpus = max(0, self._used_cpus - request.cpus)
            self._used_memory_mb = max(
                0,
                self._used_memory_mb - request.memory_mb,
            )
            self._condition.notify_all()

    def snapshot(self) -> dict[str, int]:
        return {
            "active_count": self._used_count,
            "active_cpus": self._used_cpus,
            "active_memory_mb": self._used_memory_mb,
            "max_count": self.max_count,
            "max_cpus": self.max_cpus,
            "max_memory_mb": self.max_memory_mb,
            "peak_count": self.peak_count,
            "peak_cpus": self.peak_cpus,
            "peak_memory_mb": self.peak_memory_mb,
        }


@dataclass
class _DeferredShadowLease:
    """Batch-coordinator ownership of one detached DeNovo shadow.

    The lease is deliberately runtime-only: Episodes persist its opaque ID
    and audit fields, never the live sandbox/thread objects.  These methods
    form the narrow lifecycle contract used after primary ownership ends.
    """

    uid: str
    runtime: Any
    raw_episode: Episode
    enriched_episode: Episode
    close_shadow: Callable[[], None]
    resource_request: _ShadowResourceRequest
    primary_verifier_finished_at_monotonic: float
    registered_at_monotonic: float = field(default_factory=time.monotonic)
    cleanup_future: Future | None = field(default=None, init=False, repr=False)
    cleanup_error: BaseException | None = field(default=None, init=False, repr=False)
    budget_released: bool = field(default=False, init=False)

    def seal_input(self) -> None:
        self.runtime.seal_milestone_input(
            primary_verifier_finished_at_monotonic=(
                self.primary_verifier_finished_at_monotonic
            )
        )

    def mark_barrier(
        self,
        *,
        barrier_started_at_monotonic: float,
        deadline_monotonic: float,
    ) -> None:
        record = getattr(
            self.runtime,
            "record_milestone_barrier_start",
            None,
        )
        if callable(record):
            record(
                barrier_started_at_monotonic=barrier_started_at_monotonic,
                deadline_monotonic=deadline_monotonic,
            )

    async def wait_until(self, deadline_monotonic: float) -> bool:
        while (
            not self.runtime.milestone_drained()
            and time.monotonic() < deadline_monotonic
        ):
            await asyncio.sleep(
                min(
                    0.25,
                    max(0.0, deadline_monotonic - time.monotonic()),
                )
            )
        return bool(self.runtime.milestone_drained())

    def seal_partial(
        self,
        *,
        barrier_started_at_monotonic: float,
        deadline_monotonic: float,
    ) -> dict[str, Any]:
        return self.runtime.finalize_milestones_at_barrier(
            self.raw_episode,
            self.enriched_episode,
            barrier_started_at_monotonic=barrier_started_at_monotonic,
            deadline_monotonic=deadline_monotonic,
        )

    def cancel(self, disposition: str) -> dict[str, Any]:
        return self.runtime.cancel_milestones(
            self.raw_episode,
            self.enriched_episode,
            disposition=disposition,
        )


@dataclass
class _ActiveRolloutProgress:
    """Lightweight in-memory state for one active rollout.

    This object is updated from both the asyncio driver and synchronous
    Codeflow workers, so access is always guarded by the engine's threading
    lock.  It deliberately excludes prompts, observations, and tool arguments.
    """

    uid: str
    task_id: str
    rollout_index: int
    started_at: float
    stage_started_at: float
    retry_attempt: int = 0
    stage: str = "setup"
    turn_index: int | None = None


@runtime_checkable
class TaskHooks(Protocol):
    """Per-rollout setup/teardown hook for the engine.

    The engine calls :meth:`setup` before the agent flow runs and
    :meth:`TaskContext.run_teardown` after the evaluator runs (or on failure).
    Sandbox-style hooks create a sandbox and resolve a per-task verifier;
    :class:`rllm.hooks.FixedEvaluatorHooks` binds one evaluator to every task
    and provisions nothing.
    """

    def setup(self, task: Task, agent_flow: AgentFlow, uid: str) -> TaskContext: ...


def enrich_episode_with_traces(
    episode: Episode,
    traces: list[TraceRecord],
    uid: str,
    task: dict,
    *,
    strict: bool = True,
) -> Episode:
    """Merge gateway traces into agent's lightweight Episode.

    Matching strategy (positional):

    - Traces are ordered chronologically.
    - Walk through trajectories in order, match each step to the next trace
      by position.
    - Create training Steps from traces, preserve rewards/done flags from
      agent Steps.

    When ``strict=True`` (default; training path): empty ``prompt_token_ids``
    or ``completion_token_ids`` raise :class:`EnrichMismatchError` so the
    engine can reject the episode as infrastructure failure. Token IDs are required for
    loss math, and missing ones from vLLM indicate an upstream failure.

    When ``strict=False`` (eval path against non-vLLM upstreams like the
    LiteLLM proxy or OpenAI/Anthropic directly): empty token IDs are OK —
    the evaluator reads ``model_response`` / ``chat_completions``, which are
    populated regardless of token-ID availability.
    """
    agent_steps_flat = [
        step
        for trajectory in episode.trajectories
        for step in trajectory.steps
    ]
    cutoff = episode.metadata.get("denovo_deadline") or {}
    discarded = cutoff.get("discarded_request_id")
    if discarded:
        if (episode.termination_reason != TerminationReason.TIMEOUT
                or cutoff.get("worker_stop_confirmed") is not True
                or discarded in {_step_request_id(step) for step in agent_steps_flat}):
            raise EnrichMismatchError(f"[{uid}] invalid deadline request disposition")
        removed = [trace for trace in traces if _trace_request_id(trace) == discarded]
        traces = [trace for trace in traces if _trace_request_id(trace) != discarded]
        cutoff["discarded_trace_count"] = len(removed)
        cutoff["discarded_completion_tokens"] = sum(len(getattr(trace, "completion_token_ids", []) or []) for trace in removed)
    if not traces:
        if strict and agent_steps_flat:
            raise EnrichMismatchError(
                f"[{uid}] enrich mismatch: traces=0 "
                f"agent_steps={len(agent_steps_flat)}"
            )
        logger.warning("[%s] No traces found — returning episode without token data", uid)
        return _raw_episode_with_enrichment_status(
            episode,
            status="degraded" if agent_steps_flat else "empty",
            trace_count=0,
            step_count=len(agent_steps_flat),
            missing_count=len(agent_steps_flat),
        )

    discarded_context_traces = [
        index
        for index, trace in enumerate(traces)
        if isinstance(getattr(trace, "metadata", None), dict)
        and isinstance(trace.metadata.get(_CONTEXT_METADATA_KEY), dict)
        and trace.metadata[_CONTEXT_METADATA_KEY].get("training_disposition")
        == _DISCARD_CONTEXT_TERMINAL
    ]
    if discarded_context_traces:
        termination_value = getattr(
            episode.termination_reason,
            "value",
            episode.termination_reason,
        )
        expected_termination = TerminationReason.MAX_CONTEXT_LENGTH_EXCEEDED.value
        expected_indices = [len(traces) - 1]
        if (
            termination_value != expected_termination
            or discarded_context_traces != expected_indices
        ):
            raise EnrichMismatchError(
                f"[{uid}] context-terminal trace disposition is inconsistent: "
                f"termination={termination_value!r} marked_indices={discarded_context_traces} "
                f"trace_count={len(traces)}"
            )
        traces = traces[:-1]
        episode.metrics["context_limit/discarded_terminal_traces"] = 1

    # New Codeflow turns carry a stable request id on both sides. Match by
    # identity before converting traces so client retries cannot shift every
    # later Step positionally. Legacy agents/traces retain positional behavior.
    agent_request_ids = [_step_request_id(step) for step in agent_steps_flat]
    trace_request_ids = [_trace_request_id(trace) for trace in traces]
    if agent_steps_flat and (
        any(agent_request_ids) or any(trace_request_ids)
    ):
        duplicate_count = 0
        conflict_count = 0
        trace_by_request: dict[str, TraceRecord] = {}
        untagged_count = 0
        for trace, request_id in zip(traces, trace_request_ids, strict=True):
            if request_id is None:
                untagged_count += 1
                continue
            existing = trace_by_request.get(request_id)
            if existing is None:
                trace_by_request[request_id] = trace
                continue
            if _trace_semantic_signature(existing) == _trace_semantic_signature(
                trace
            ):
                duplicate_count += 1
            else:
                conflict_count += 1

        expected_ids = {
            request_id
            for request_id in agent_request_ids
            if request_id is not None
        }
        missing_ids = [
            request_id
            for request_id in agent_request_ids
            if request_id is None or request_id not in trace_by_request
        ]
        extra_ids = set(trace_by_request) - expected_ids
        extra_count = untagged_count + len(extra_ids)
        mixed_agent_ids = any(
            request_id is None for request_id in agent_request_ids
        )
        mismatch = bool(
            mixed_agent_ids
            or missing_ids
            or extra_count
            or conflict_count
        )
        if mismatch:
            message = (
                f"[{uid}] request-id enrich mismatch: "
                f"traces={len(traces)} agent_steps={len(agent_steps_flat)} "
                f"missing={len(missing_ids)} extra={extra_count} "
                f"conflicts={conflict_count} duplicates={duplicate_count}"
            )
            if strict:
                raise EnrichMismatchError(message)
            logger.warning("%s; preserving raw evaluation steps", message)
            return _raw_episode_with_enrichment_status(
                episode,
                status="degraded",
                trace_count=len(traces),
                step_count=len(agent_steps_flat),
                duplicate_count=duplicate_count,
                missing_count=len(missing_ids),
                extra_count=extra_count,
                conflict_count=conflict_count,
            )
        traces = [
            trace_by_request[str(request_id)]
            for request_id in agent_request_ids
        ]
        episode.metrics.update(
            {
                "enrichment/status": "ok",
                "enrichment/trace_count": len(traces),
                "enrichment/step_count": len(agent_steps_flat),
                "enrichment/deduplicated_traces": duplicate_count,
                "enrichment/missing_traces": 0,
                "enrichment/extra_traces": 0,
                "enrichment/conflicting_traces": 0,
            }
        )

    # Convert all retained traces to training steps
    training_steps = [trace_record_to_step(t) for t in traces]

    # Bad traces (missing or empty token_ids) silently corrupt loss math and
    # shrink GRPO groups; reject real mismatches as infrastructure failures.
    n_agent_steps = sum(len(t.steps) for t in episode.trajectories)
    agent_populates_steps = any(len(t.steps) > 0 for t in episode.trajectories)

    empty_prompt = sum(1 for s in training_steps if not s.model_output.prompt_ids)
    empty_compl = sum(1 for s in training_steps if not s.model_output.completion_ids)
    # Only enforce step-count parity when the agent actually populates steps.
    # Trajectories with no agent steps absorb remaining traces wholesale
    # (see branch below), and trajectories with steps consume traces 1:1.
    traces_short = agent_populates_steps and len(training_steps) < n_agent_steps
    traces_long = agent_populates_steps and len(training_steps) > n_agent_steps
    # Empty token IDs are a hard error only in strict (training) mode.
    # Eval against external providers (OpenAI/Anthropic via LiteLLM proxy)
    # legitimately has empty token IDs and that's fine — the evaluator
    # reads `model_response` / `chat_completions`, not token IDs.
    token_ids_missing = strict and (empty_prompt or empty_compl)
    if traces_short or traces_long or token_ids_missing:
        raise EnrichMismatchError(f"[{uid}] enrich mismatch: traces={len(training_steps)} agent_steps={n_agent_steps} empty_prompt_ids={empty_prompt} empty_completion_ids={empty_compl}")

    # Build enriched trajectories
    enriched_trajectories: list[Trajectory] = []
    trace_idx = 0

    for traj in episode.trajectories:
        traj_steps: list[Step] = []

        if traj.steps:
            # Match agent steps to traces positionally. The validation above
            # guarantees trace_idx < len(training_steps) for every agent_step
            # when agent_populates_steps is True.
            for agent_step in traj.steps:
                step = training_steps[trace_idx]
                # Preserve agent-side fields (the trace doesn't carry these — it
                # only holds the raw LLM call) -- action, observation, reward, done.
                step.action = agent_step.action
                step.input = agent_step.input
                step.observation = agent_step.observation
                if step.observation is None and isinstance(agent_step.metadata, dict):
                    step.observation = agent_step.metadata.get("tool_observation")
                if agent_step.metadata:
                    if not isinstance(step.metadata, dict):
                        step.metadata = {"trace_metadata": step.metadata}
                    step.info["agent_step_metadata"] = agent_step.metadata
                    request_id = _step_request_id(agent_step)
                    if request_id:
                        step.metadata["model_request_id"] = request_id
                annotate_step_tool_call_reward(step, overwrite=True)
                step.reward = agent_step.reward
                step.done = agent_step.done
                trace_idx += 1
                traj_steps.append(step)
        else:
            # No agent steps — assign all remaining traces to this trajectory
            # (common for single-trajectory agents that don't populate steps)
            remaining = training_steps[trace_idx:]
            trace_idx += len(remaining)
            for step in remaining:
                annotate_step_tool_call_reward(step, overwrite=True)
            traj_steps = remaining

        enriched_trajectories.append(
            Trajectory(
                uid=traj.uid,
                name=traj.name,
                task=traj.task or task,
                steps=traj_steps,
                reward=traj.reward,
                metadata=traj.metadata,
            )
        )

    # If there are unmatched traces and no trajectories existed, create one
    if not episode.trajectories and traces:
        enriched_trajectories = [
            Trajectory(
                name="default",
                task=task,
                steps=training_steps,
            )
        ]

    # Compute metrics
    metrics = compute_step_metrics(enriched_trajectories)
    metrics["empty"] = int(len(traces) == 0)
    metrics["steps_collected"] = len(traces)
    metrics.update(episode.metrics)
    metrics.setdefault("enrichment/status", "ok")
    metrics.setdefault("enrichment/trace_count", len(traces))
    metrics.setdefault("enrichment/step_count", n_agent_steps)
    metrics.setdefault("enrichment/deduplicated_traces", 0)
    metrics.setdefault("enrichment/missing_traces", 0)
    metrics.setdefault("enrichment/extra_traces", 0)
    metrics.setdefault("enrichment/conflicting_traces", 0)

    return Episode(
        id=uid,
        task=task,
        is_correct=episode.is_correct,
        trajectories=enriched_trajectories,
        metrics=metrics,
        metadata=episode.metadata,
        termination_reason=episode.termination_reason,
        artifacts=episode.artifacts,
    )


def _summarize_llm_latencies(traces: list[Any], agentflow_s: float) -> tuple[float, float]:
    """Return ``(llm_sum_s, llm_wall_s)`` from trace latencies (sum and interval-union)."""
    if not traces:
        return 0.0, 0.0

    llm_sum_s = sum(getattr(tr, "latency_ms", 0.0) or 0.0 for tr in traces) / 1000.0

    intervals: list[tuple[float, float]] = []
    for tr in traces:
        end = float(getattr(tr, "timestamp", 0.0) or 0.0)
        dur = (getattr(tr, "latency_ms", 0.0) or 0.0) / 1000.0
        if end > 0 and dur > 0:
            intervals.append((end - dur, end))
    if not intervals:
        return llm_sum_s, min(llm_sum_s, agentflow_s)

    intervals.sort()
    merged_total = 0.0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            merged_total += cur_end - cur_start
            cur_start, cur_end = start, end
    merged_total += cur_end - cur_start
    return llm_sum_s, min(merged_total, agentflow_s) if agentflow_s > 0 else merged_total


_TIMING_PHASES_DISPLAY: tuple[tuple[str, str], ...] = (
    ("startup_queue", "time/startup_queue_s"),
    ("setup", "time/setup_s"),
    ("session_admission", "time/session_admission_s"),
    ("session_create", "time/session_create_s"),
    ("session_start", "time/session_admission_create_s"),
    ("agentflow", "time/agentflow_s"),
    ("traces", "time/traces_s"),
    ("evaluator", "time/evaluator_s"),
    ("teardown", "time/teardown_s"),
)


def _format_timing_breakdown(metrics: dict[str, float]) -> str:
    """Compact per-rollout timing summary, e.g. ``setup=16s agentflow=1162s [llm=1100s/15 steps ||1.4x] evaluator=9s teardown=0s``.

    The ``agentflow`` phase is annotated with ``[llm=Xs/N steps]`` (wall-clock
    LLM wait, interval-union), plus ``||N.Nx`` when parallel LLM calls push
    the sum past ``agentflow_s``. Empty when no timings present.
    """
    total = metrics.get("time/rollout_s")
    if total is None:
        return ""
    parts: list[str] = []
    for label, key in _TIMING_PHASES_DISPLAY:
        if key not in metrics:
            continue
        if label == "agentflow":
            llm_wall = metrics.get("time/agentflow_llm_wall_s")
            llm_sum = metrics.get("time/agentflow_llm_sum_s")
            n_turns = metrics.get("n_turns")
            agentflow_s = metrics[key]
            if llm_wall is not None and n_turns is not None and n_turns > 0:
                step_label = "step" if int(n_turns) == 1 else "steps"
                pieces = [f"llm={llm_wall:.0f}s/{int(n_turns)} {step_label}"]
                if llm_sum is not None and agentflow_s > 0 and llm_sum > agentflow_s * 1.05:
                    pieces.append(f"||{llm_sum / agentflow_s:.1f}x")
                parts.append(f"agentflow={agentflow_s:.0f}s [{' '.join(pieces)}]")
            else:
                parts.append(f"agentflow={agentflow_s:.0f}s")
        else:
            parts.append(f"{label}={metrics[key]:.0f}s")
    inner = f" ({' '.join(parts)})" if parts else ""
    return f" in {total:.0f}s{inner}"


def _apply_evaluator_infrastructure_failure(
    episode: Episode,
    eval_output: EvalOutput,
) -> bool:
    """Persist an unusable verifier result without manufacturing a reward."""

    metadata = eval_output.metadata if isinstance(eval_output.metadata, dict) else {}
    failure = metadata.get("infrastructure_failure")
    if not isinstance(failure, dict) or not failure.get("reason"):
        return False
    episode.metadata["infrastructure_failure"] = dict(failure)
    episode.metrics.update(metadata)
    episode.metrics["infrastructure_failure/reason"] = str(failure["reason"])
    episode.is_correct = False
    episode.termination_reason = TerminationReason.ERROR
    return True


def _raise_fd_limit(target: int = _MIN_FD_LIMIT) -> None:
    """Best-effort raise of the process soft file-descriptor limit.

    Training with many parallel agent flows (each opening HTTP connections
    through the gateway) can easily exceed the default 1024 FD soft limit.
    """
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < target:
            new_soft = min(target, hard)
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
            logger.info("Raised NOFILE soft limit from %d to %d (hard=%d)", soft, new_soft, hard)
    except (ValueError, OSError) as e:
        logger.warning("Could not raise file descriptor limit: %s", e)


_DENOVO_DEADLINE_STOP_GRACE_SECONDS = 60.0


class AgentFlowEngine:
    """Executes AgentFlows with gateway-mediated trace capture."""

    def __init__(
        self,
        agent_flow: AgentFlow,
        evaluator: Evaluator | None,
        gateway: GatewayManager,
        model: str,
        n_parallel_tasks: int = 128,
        retry_limit: int = 3,
        raise_on_error: bool = True,
        episode_logger: EpisodeLogger | None = None,
        hooks: TaskHooks | None = None,
        train_sampling_params: dict | None = None,
        val_sampling_params: dict | None = None,
        milestone_reward_config: MilestoneRewardConfig | dict | None = None,
        model_window_tokens: int | None = None,
        rollout_group_size: int = 1,
        cancelled_teardown_timeout_seconds: float = 60.0,
        rollout_startup_window: int | None = None,
        teardown_executor_workers: int | None = None,
        trajectory_timeout: float | None = None,
        denovo_shadow_overlap_budget_multiplier: float = 1.0,
        evaluation_admission: Any | None = None,
    ) -> None:
        self.evaluation_admission = evaluation_admission
        # Evaluation recovery changes only physical session identity. Artifact
        # task/attempt indices and the default training identity stay stable.
        self.session_namespace = ""
        self.evaluation_session_ids: set[str] = set()
        if evaluator is None and hooks is None:
            raise ValueError("AgentFlowEngine requires either an `evaluator` (single evaluator for every task) or `hooks` (per-task evaluator + setup/teardown). Both cannot be None.")
        if hooks is None:
            from rllm.hooks import FixedEvaluatorHooks

            hooks = FixedEvaluatorHooks(evaluator)

        if isinstance(milestone_reward_config, MilestoneRewardConfig):
            self.milestone_reward_config = milestone_reward_config
        else:
            raw_milestone = milestone_reward_config if hasattr(milestone_reward_config, "get") else {}
            self.milestone_reward_config = MilestoneRewardConfig.from_config(
                raw_milestone,
                format_weight=float(raw_milestone.get("format_weight", 0.1)),
            )

        # A shadow sandbox only produces the test evidence consumed by the
        # verification-potential component.  Keep that lifecycle derived from
        # the resolved reward configuration so navigation/format-only
        # ablations neither provision an unused sandbox nor advertise two
        # sandbox slots per rollout.
        configure_shadow = getattr(agent_flow, "configure_shadow_sandbox", None)
        if callable(configure_shadow):
            configure_shadow(
                enabled=(
                    self.milestone_reward_config.enable
                    and self.milestone_reward_config.verification_enable
                )
            )

        self._flow_accepts_env = flow_accepts_env(agent_flow)
        self._flow_accepts_shadow_runtime = flow_accepts_keyword(agent_flow, "shadow_runtime")
        if getattr(agent_flow, "needs_env", False) and not self._flow_accepts_env:
            raise TypeError(f"{type(agent_flow).__name__} declares needs_env but its run/arun has no keyword-only 'env' parameter; declare run(self, task, config, *, env).")

        requested_parallel_tasks = n_parallel_tasks
        if getattr(agent_flow, "sandbox_slots_per_rollout", 1) > 1:
            from rllm.harnesses.shadow_sandbox import effective_rollout_concurrency

            n_parallel_tasks = effective_rollout_concurrency(agent_flow, n_parallel_tasks)

        self.agent_flow = agent_flow
        self.denovo_background_finalize_enable = bool(
            getattr(agent_flow, "denovo_background_finalize_enable", False)
        )
        self.denovo_shadow_finalize_enable = bool(
            self.denovo_background_finalize_enable
            and getattr(agent_flow, "shadow_enabled", True)
        )
        try:
            self.denovo_shadow_sandbox_count = max(
                1,
                int(getattr(agent_flow, "denovo_shadow_sandbox_count", 1)),
            )
        except (TypeError, ValueError):
            self.denovo_shadow_sandbox_count = 1
        self.gateway = gateway
        self.model = model
        self.n_parallel_tasks = n_parallel_tasks
        try:
            self.sandbox_slots_per_rollout = max(1, int(getattr(agent_flow, "sandbox_slots_per_rollout", 1)))
        except (TypeError, ValueError):
            self.sandbox_slots_per_rollout = 1
        self.max_active_sandbox_slots = self.n_parallel_tasks * self.sandbox_slots_per_rollout
        if self.n_parallel_tasks != requested_parallel_tasks:
            logger.warning(
                "AgentFlow rollout concurrency explicitly capped: requested=%d effective=%d "
                "sandbox_slots_per_rollout=%d max_active_sandbox_slots=%d",
                requested_parallel_tasks,
                self.n_parallel_tasks,
                self.sandbox_slots_per_rollout,
                self.max_active_sandbox_slots,
            )
        else:
            logger.info(
                "AgentFlow concurrency: active_rollouts=%d sandbox_slots_per_rollout=%d "
                "max_active_sandbox_slots=%d",
                self.n_parallel_tasks,
                self.sandbox_slots_per_rollout,
                self.max_active_sandbox_slots,
            )
        self.retry_limit = retry_limit
        if trajectory_timeout is not None and (
            isinstance(trajectory_timeout, bool)
            or not isinstance(trajectory_timeout, int | float)
            or not math.isfinite(float(trajectory_timeout))
            or float(trajectory_timeout) <= 0
        ):
            raise ValueError("trajectory_timeout must be finite and positive or null")
        self.trajectory_timeout = (
            float(trajectory_timeout) if trajectory_timeout is not None else None
        )
        self.raise_on_error = raise_on_error
        self.episode_logger = episode_logger
        self.hooks = hooks
        self.train_sampling_params = train_sampling_params
        self.val_sampling_params = val_sampling_params
        self.model_window_tokens = (
            int(model_window_tokens)
            if model_window_tokens is not None
            else None
        )
        if self.model_window_tokens is not None and self.model_window_tokens <= 0:
            raise ValueError("model_window_tokens must be a positive integer")
        if (
            isinstance(rollout_group_size, bool)
            or not isinstance(rollout_group_size, int)
            or rollout_group_size <= 0
        ):
            raise ValueError("rollout_group_size must be a positive integer")
        self.rollout_group_size = rollout_group_size
        if (
            isinstance(cancelled_teardown_timeout_seconds, bool)
            or not isinstance(cancelled_teardown_timeout_seconds, int | float)
            or not math.isfinite(float(cancelled_teardown_timeout_seconds))
            or float(cancelled_teardown_timeout_seconds) <= 0
        ):
            raise ValueError(
                "cancelled_teardown_timeout_seconds must be finite and positive"
            )
        self.cancelled_teardown_timeout_seconds = float(
            cancelled_teardown_timeout_seconds
        )
        if rollout_startup_window is None:
            rollout_startup_window = n_parallel_tasks
        if (
            isinstance(rollout_startup_window, bool)
            or not isinstance(rollout_startup_window, int)
            or rollout_startup_window <= 0
        ):
            raise ValueError("rollout_startup_window must be a positive integer or null")
        self.rollout_startup_window = min(
            n_parallel_tasks,
            rollout_startup_window,
        )
        if (
            isinstance(denovo_shadow_overlap_budget_multiplier, bool)
            or not isinstance(
                denovo_shadow_overlap_budget_multiplier,
                int | float,
            )
            or not math.isfinite(float(denovo_shadow_overlap_budget_multiplier))
            or float(denovo_shadow_overlap_budget_multiplier) < 1.0
        ):
            raise ValueError(
                "denovo_shadow_overlap_budget_multiplier must be finite and >= 1"
            )
        self.denovo_shadow_overlap_budget_multiplier = float(
            denovo_shadow_overlap_budget_multiplier
        )
        if teardown_executor_workers is not None and (
            isinstance(teardown_executor_workers, bool)
            or not isinstance(teardown_executor_workers, int)
            or teardown_executor_workers <= 0
        ):
            raise ValueError(
                "teardown_executor_workers must be a positive integer or null"
            )
        self.teardown_executor_workers = (
            teardown_executor_workers if teardown_executor_workers is not None
            else max(1, min(32, n_parallel_tasks))
        )

        self.executor = ThreadPoolExecutor(max_workers=n_parallel_tasks)
        # Lifecycle I/O must remain schedulable even when every synchronous
        # Codeflow worker is stuck and cannot honor asyncio cancellation. Its
        # pool follows the startup window instead of blindly duplicating the
        # full rollout pool (640 threads in the affected experiment).
        self.lifecycle_executor_workers = max(
            4,
            min(n_parallel_tasks, self.rollout_startup_window),
        )
        self.lifecycle_executor = ThreadPoolExecutor(
            max_workers=self.lifecycle_executor_workers,
            thread_name_prefix="rllm-lifecycle",
        )
        # DeNovo final verifiers can each run for an hour.  Isolate them from
        # setup/teardown so releasing a vLLM slot actually lets the next
        # rollout provision its sandboxes.
        self.primary_verifier_executor = (
            ThreadPoolExecutor(
                max_workers=self.rollout_startup_window,
                thread_name_prefix="rllm-primary-verifier",
            )
            if self.denovo_background_finalize_enable
            else None
        )
        # Large cancellation waves can otherwise queue teardown behind slow
        # setup/evaluator calls in the shared lifecycle pool. Reserve cleanup
        # capacity by default; this does not admit additional rollouts/sandboxes.
        self.teardown_executor = (
            ThreadPoolExecutor(
                max_workers=self.teardown_executor_workers,
                thread_name_prefix="rllm-teardown",
            )
            if self.teardown_executor_workers
            else None
        )
        self._semaphore = asyncio.Semaphore(n_parallel_tasks)
        # Bound only setup -> explicit Gateway session creation.  The permit
        # is released before agentflow, so total active rollout concurrency
        # remains n_parallel_tasks while a large wave cannot enqueue hundreds
        # of sandbox setups and session starts at once.
        self._startup_semaphore = asyncio.Semaphore(self.rollout_startup_window)
        self._startup_pending = 0
        self._startup_active = 0
        self._event_loop_lag_seconds = 0.0
        self._event_loop_lag_max_seconds = 0.0
        self._event_loop_lag_samples = 0
        self._event_loop_lag_task: asyncio.Task[None] | None = None
        self._lifecycle_waiters: dict[
            Future[Any],
            set[asyncio.Future[None]],
        ] = {}
        self._lifecycle_poller_task: asyncio.Task[None] | None = None
        self._rollout_progress_lock = threading.RLock()
        self._rollout_progress: dict[str, _ActiveRolloutProgress] = {}
        self._lifecycle_metrics_lock = threading.Lock()
        self._lifecycle_metrics: Counter[str] = Counter()

        # Raise the file descriptor limit to avoid "Too many open files" when
        # running many parallel agent flows with individual HTTP clients.
        _raise_fd_limit()

        # Training step tracking (set by set_training_step)
        self.current_step = 0
        self.current_epoch = 0
        self.current_mode = "train"
        self.rollout_log_path = os.environ.get("RLLM_ROLLOUT_LOG_PATH")
        for deprecated in ("RLLM_ROLLOUT_LOG_PROB", "RLLM_ROLLOUT_LOG_EVERY"):
            if deprecated in os.environ:
                logger.warning(
                    "%s is deprecated and ignored; configured rollout logs are always persisted",
                    deprecated,
                )
        self._rollout_log_completed = max(
            0, _env_int("RLLM_ROLLOUT_LOG_START_INDEX", 0)
        )
        self._defer_dynamic_sampling_rollout_logging = False
        self._gateway_cleanup_deferred = 0
        self._deferred_shadow_leases: dict[str, _DeferredShadowLease] = {}
        self._deferred_shadow_registry_lock = threading.RLock()
        self._denovo_shadow_budget: _ShadowResourceBudget | None = None
        if self.denovo_shadow_finalize_enable:
            shadow_defaults = getattr(hooks, "shadow_sandbox_resources", {})
            if not isinstance(shadow_defaults, dict):
                shadow_defaults = dict(shadow_defaults or {})
            default_cpus = int(shadow_defaults.get("cpus", 0) or 0)
            default_memory_mb = int(
                shadow_defaults.get("memory_mb", 0) or 0
            )
            if default_cpus <= 0 or default_memory_mb <= 0:
                raise ValueError(
                    "DeNovo background finalization requires positive default "
                    "shadow_sandbox_resources cpus and memory_mb"
                )
            multiplier = self.denovo_shadow_overlap_budget_multiplier
            shadow_count = self.denovo_shadow_sandbox_count
            self._denovo_shadow_budget = _ShadowResourceBudget(
                max_count=max(
                    1,
                    math.ceil(
                        requested_parallel_tasks * shadow_count * multiplier
                    ),
                ),
                max_cpus=max(
                    1,
                    math.ceil(
                        requested_parallel_tasks
                        * shadow_count
                        * default_cpus
                        * multiplier
                    ),
                ),
                max_memory_mb=max(
                    1,
                    math.ceil(
                        requested_parallel_tasks
                        * shadow_count
                        * default_memory_mb
                        * multiplier
                    ),
                ),
            )

    def _lifecycle_executor(self) -> ThreadPoolExecutor:
        """Return the lifecycle pool, retaining compatibility with test doubles."""
        executor = getattr(self, "lifecycle_executor", None) or self.executor
        if executor is None:
            raise RuntimeError("AgentFlow lifecycle executor is shut down")
        return executor

    def _denovo_background_task(
        self,
        task: Task,
        *,
        is_validation: bool,
    ) -> bool:
        if is_validation or not getattr(
            self,
            "denovo_background_finalize_enable",
            False,
        ):
            return False
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        rllm_metadata = metadata.get("rllm")
        rllm_metadata = rllm_metadata if isinstance(rllm_metadata, dict) else {}
        profile = str(
            metadata.get("task_profile")
            or rllm_metadata.get("task_profile")
            or ""
        )
        return profile == "repo_generation_denovoswe"

    def _activate_denovo_background_task(self, task: Task) -> Task:
        metadata = dict(task.metadata)
        rllm_metadata = metadata.get("rllm")
        rllm_metadata = (
            dict(rllm_metadata) if isinstance(rllm_metadata, dict) else {}
        )
        rllm_metadata[_DENOVO_BACKGROUND_ACTIVE_KEY] = True
        materialized_outcome_source = str(
            metadata.get("outcome_source")
            or rllm_metadata.get("outcome_source")
            or "shadow_verifier"
        )
        metadata["materialized_outcome_source"] = materialized_outcome_source
        metadata["outcome_source"] = "primary_verifier"
        rllm_metadata["outcome_source"] = "primary_verifier"
        metadata["rllm"] = rllm_metadata
        metadata.setdefault("language", "python")
        return replace(task, metadata=metadata)

    def _shadow_resource_request(self, task: Task) -> _ShadowResourceRequest:
        defaults = getattr(self.hooks, "shadow_sandbox_resources", {})
        defaults = dict(defaults or {})
        overrides = task.metadata.get("shadow_sandbox_resources")
        overrides = dict(overrides or {}) if isinstance(overrides, dict) else {}
        resources = {**defaults, **overrides}
        per_sandbox_cpus = int(resources.get("cpus", 0) or 0)
        per_sandbox_memory_mb = int(resources.get("memory_mb", 0) or 0)
        if per_sandbox_cpus <= 0 or per_sandbox_memory_mb <= 0:
            raise ValueError(
                "DeNovo shadow_sandbox_resources must request positive cpus "
                "and memory_mb, got "
                f"cpus={per_sandbox_cpus} memory_mb={per_sandbox_memory_mb}"
            )
        count = max(1, int(self.denovo_shadow_sandbox_count))
        request = _ShadowResourceRequest(
            count=count,
            cpus=per_sandbox_cpus * count,
            memory_mb=per_sandbox_memory_mb * count,
        )
        return request

    async def _acquire_denovo_shadow_budget(
        self,
        task: Task,
    ) -> _ShadowResourceRequest:
        budget = self._denovo_shadow_budget
        if budget is None:
            raise RuntimeError("DeNovo shadow budget is not configured")
        request = self._shadow_resource_request(task)
        await budget.acquire(request)
        return request

    async def _release_denovo_shadow_budget(
        self,
        request: _ShadowResourceRequest | None,
    ) -> None:
        if request is None or self._denovo_shadow_budget is None or request.cleanup_unconfirmed:
            return
        if request.release_task is None:
            request.release_task = asyncio.create_task(self._denovo_shadow_budget.release(request))
        await asyncio.shield(request.release_task)

    def _submit_primary_verifier_call(
        self,
        function: Callable[..., Any],
        *args: Any,
    ) -> Future[Any]:
        executor = self.primary_verifier_executor
        if executor is None:
            return self._submit_lifecycle_call(
                function,
                *args,
                pending_key="evaluator_pending",
            )
        future = executor.submit(function, *args)
        self._observe_lifecycle_future(
            future,
            pending_key="evaluator_pending",
        )
        return future

    def _update_lifecycle_metric(self, key: str, delta: int = 1) -> None:
        lock = getattr(self, "_lifecycle_metrics_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._lifecycle_metrics_lock = lock
        metrics = getattr(self, "_lifecycle_metrics", None)
        if metrics is None:
            metrics = Counter()
            self._lifecycle_metrics = metrics
        with lock:
            metrics[key] += delta

    def _observe_cancelled_agentflow_worker(
        self,
        error: BaseException | None,
    ) -> None:
        """Account for a late sync-worker result without changing semantics.

        A 410 after cancellation is the expected gateway response to a request
        that reached a tombstoned session. It is observed and counted here;
        an otherwise-live flow still propagates the same 410 normally.
        """

        current = error
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            response = getattr(current, "response", None)
            status_code = getattr(current, "status_code", None)
            if status_code is None and response is not None:
                status_code = getattr(response, "status_code", None)
            if status_code == 410:
                self._update_lifecycle_metric("cancelled_worker_late_410")
                return
            current = current.__cause__ or current.__context__

    def _observe_lifecycle_future(
        self,
        future: Future[Any],
        *,
        pending_key: str,
    ) -> None:
        """Track the underlying executor job and consume an abandoned error."""
        self._update_lifecycle_metric(pending_key, 1)

        def finished(done: Future[Any]) -> None:
            self._update_lifecycle_metric(pending_key, -1)
            if not done.cancelled():
                try:
                    done.exception()
                except BaseException:
                    pass

        future.add_done_callback(finished)

    def _submit_lifecycle_call(
        self,
        function: Callable[..., Any],
        *args: Any,
        pending_key: str,
    ) -> Future[Any]:
        """Submit lifecycle work while retaining its uncancelled native future."""
        future = self._lifecycle_executor().submit(function, *args)
        self._observe_lifecycle_future(future, pending_key=pending_key)
        return future

    def _submit_teardown_call(
        self,
        function: Callable[..., Any],
        *args: Any,
    ) -> Future[Any]:
        """Submit cleanup to its dedicated pool when one is configured."""
        executor = getattr(self, "teardown_executor", None)
        if executor is None:
            return self._submit_lifecycle_call(
                function,
                *args,
                pending_key="teardown_pending",
            )
        future = executor.submit(function, *args)
        self._observe_lifecycle_future(
            future,
            pending_key="teardown_pending",
        )
        return future

    async def _await_lifecycle_future(self, future: Future[Any]) -> Any:
        """Await one native future through a single shared completion poller.

        The production trainer environment cannot reliably wake its asyncio
        selector from ``asyncio.wrap_future`` callbacks.  Polling every Future
        independently at 20 ms, however, creates tens of thousands of ready
        callbacks per second for a 640-rollout wave.  One engine-local poller
        retains cancellation isolation while making timer load constant.
        """
        if future.done():
            return future.result()

        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[None] = loop.create_future()
        waiters = getattr(self, "_lifecycle_waiters", None)
        if waiters is None:
            waiters = {}
            self._lifecycle_waiters = waiters
        waiters.setdefault(future, set()).add(waiter)
        self._ensure_lifecycle_poller()
        try:
            # Cancelling this coroutine must not cancel the native executor
            # future, which may still return a TaskContext requiring teardown.
            await asyncio.shield(waiter)
            return future.result()
        finally:
            registered = waiters.get(future)
            if registered is not None:
                registered.discard(waiter)
                if not registered:
                    waiters.pop(future, None)
            if not waiter.done():
                waiter.cancel()

    def _ensure_lifecycle_poller(self) -> None:
        task = getattr(self, "_lifecycle_poller_task", None)
        if task is not None and not task.done():
            return
        self._lifecycle_poller_task = asyncio.create_task(
            self._poll_lifecycle_futures(),
            name="agentflow-engine-lifecycle-poller",
        )

    async def _poll_lifecycle_futures(self) -> None:
        try:
            while getattr(self, "_lifecycle_waiters", None):
                await asyncio.sleep(0.05)
                for future, waiters in list(self._lifecycle_waiters.items()):
                    if not future.done():
                        continue
                    self._lifecycle_waiters.pop(future, None)
                    for waiter in waiters:
                        if not waiter.done():
                            waiter.set_result(None)
        finally:
            self._lifecycle_poller_task = None

    def _ensure_event_loop_lag_monitor(self) -> None:
        task = getattr(self, "_event_loop_lag_task", None)
        if task is not None and not task.done():
            return
        self._event_loop_lag_task = asyncio.create_task(
            self._monitor_event_loop_lag(),
            name="agentflow-engine-event-loop-lag",
        )

    async def _monitor_event_loop_lag(self) -> None:
        import sys
        import traceback
        from rllm.utils.diagnostic_events import emit_diagnostic

        interval = 0.5
        loop = asyncio.get_running_loop()
        owner = threading.get_ident()
        heartbeat = [time.monotonic()]
        stop = threading.Event()

        def watch():
            last_report = float("-inf")
            while not stop.wait(1.0):
                now = time.monotonic()
                age = now - heartbeat[0]
                if age < 5.0 or now - last_report < 30.0:
                    continue
                last_report = now
                frame = sys._current_frames().get(owner)
                emit_diagnostic(
                    "trainer_event_loop_stall", heartbeat_age_seconds=age,
                    active_threads=threading.active_count(),
                    stack=traceback.format_stack(frame, limit=20) if frame else [],
                )

        watchdog = threading.Thread(target=watch, name="rllm-loop-watchdog", daemon=True)
        watchdog.start()
        try:
            while True:
                expected = loop.time() + interval
                await asyncio.sleep(interval)
                heartbeat[0] = time.monotonic()
                lag = max(0.0, loop.time() - expected)
                self._event_loop_lag_seconds = lag
                self._event_loop_lag_max_seconds = max(
                    getattr(self, "_event_loop_lag_max_seconds", 0.0), lag,
                )
                self._event_loop_lag_samples = getattr(self, "_event_loop_lag_samples", 0) + 1
        finally:
            stop.set()

    def lifecycle_runtime_metrics(self) -> dict[str, int | float]:
        lock = getattr(self, "_lifecycle_metrics_lock", None)
        metrics = getattr(self, "_lifecycle_metrics", Counter())
        if lock is None:
            snapshot = dict(metrics)
        else:
            with lock:
                snapshot = dict(metrics)
        lifecycle_waiters = getattr(self, "_lifecycle_waiters", {})
        result: dict[str, int | float] = {
            "lifecycle/setup_pending": max(0, int(snapshot.get("setup_pending", 0))),
            "lifecycle/evaluator_pending": max(
                0, int(snapshot.get("evaluator_pending", 0))
            ),
            "lifecycle/teardown_pending": max(
                0, int(snapshot.get("teardown_pending", 0))
            ),
            "lifecycle/teardown_timeouts": int(
                snapshot.get("teardown_timeouts", 0)
            ),
            "lifecycle/deferred_cleanups": int(
                snapshot.get("deferred_cleanups", 0)
            ),
            "lifecycle/late_cleanups": int(snapshot.get("late_cleanups", 0)),
            "rollout/full_trajectory_retries": int(
                snapshot.get("full_trajectory_retries", 0)
            ),
            "rollout/trajectory_timeouts": int(
                snapshot.get("trajectory_timeouts", 0)
            ),
            "rollout/agent_timeouts": int(
                snapshot.get("agent_timeouts", 0)
            ),
            "rollout/cancelled_agentflow_children": int(
                snapshot.get("cancelled_agentflow_children", 0)
            ),
            "rollout/cancelled_worker_late_410": int(
                snapshot.get("cancelled_worker_late_410", 0)
            ),
            # Every detached synchronous worker is installed with a done
            # callback that consumes its result. This counter is deliberately
            # explicit so production telemetry can assert the invariant.
            "rollout/unobserved_agentflow_exceptions": int(
                snapshot.get("unobserved_agentflow_exceptions", 0)
            ),
            "shadow/final_recovery_successes": int(
                snapshot.get("shadow_final_recovery_successes", 0)
            ),
            "shadow/final_recovery_failures": int(
                snapshot.get("shadow_final_recovery_failures", 0)
            ),
            "lifecycle/startup_window": int(
                getattr(self, "rollout_startup_window", 0)
            ),
            "lifecycle/executor_workers": int(
                getattr(self, "lifecycle_executor_workers", 0)
            ),
            "lifecycle/teardown_executor_workers": int(
                getattr(self, "teardown_executor_workers", 0)
            ),
            "lifecycle/primary_verifier_executor_workers": int(
                getattr(self, "rollout_startup_window", 0)
                if getattr(self, "primary_verifier_executor", None) is not None
                else 0
            ),
            "lifecycle/polled_futures": len(lifecycle_waiters),
            "lifecycle/polled_waiters": sum(
                len(waiters) for waiters in lifecycle_waiters.values()
            ),
            "lifecycle/startup_pending": max(
                0,
                int(getattr(self, "_startup_pending", 0)),
            ),
            "lifecycle/startup_active": max(
                0,
                int(getattr(self, "_startup_active", 0)),
            ),
            "trainer/event_loop/lag_seconds": float(
                getattr(self, "_event_loop_lag_seconds", 0.0)
            ),
            "trainer/event_loop/lag_max_seconds": float(
                getattr(self, "_event_loop_lag_max_seconds", 0.0)
            ),
            "trainer/event_loop/lag_samples": int(
                getattr(self, "_event_loop_lag_samples", 0)
            ),
        }
        budget = getattr(self, "_denovo_shadow_budget", None)
        if budget is not None:
            for key, value in budget.snapshot().items():
                result[f"denovo_shadow/resources/{key}"] = int(value)
        with getattr(
            self,
            "_deferred_shadow_registry_lock",
            threading.RLock(),
        ):
            result["denovo_shadow/active_leases"] = len(
                getattr(self, "_deferred_shadow_leases", {})
            )
        return result

    async def _run_context_teardown(
        self,
        ctx: TaskContext,
        uid: str,
        *,
        late: bool = False,
    ) -> bool:
        """Run idempotent teardown without allowing it to hold a rollout forever."""
        future = self._submit_teardown_call(ctx.run_teardown)
        timeout = float(
            getattr(self, "cancelled_teardown_timeout_seconds", 60.0)
        )
        try:
            await asyncio.wait_for(
                self._await_lifecycle_future(future),
                timeout=timeout,
            )
            return True
        except TimeoutError:
            self._update_lifecycle_metric("teardown_timeouts")
            self._update_lifecycle_metric("deferred_cleanups")
            logger.warning(
                "[%s] lifecycle teardown exceeded %.1fs; cleanup deferred "
                "late=%s last_stage=teardown",
                uid,
                timeout,
                late,
            )
            return False
        except asyncio.CancelledError:
            self._update_lifecycle_metric("deferred_cleanups")
            logger.info(
                "[%s] lifecycle teardown wait cancelled; underlying cleanup "
                "continues in the lifecycle executor",
                uid,
            )
            raise
        except Exception as exc:
            from rllm.utils.infrastructure_diagnostics import infrastructure_exception_chain
            raise RolloutInfrastructureError(
                "sandbox_cleanup_unconfirmed", f"Task cleanup failed: {exc}",
                retryable=False, stage="teardown", retry_scope="none",
                diagnostics={"cleanup_errors": infrastructure_exception_chain(exc)},
            ) from exc

    def _schedule_late_setup_teardown(
        self,
        setup_future: Future[TaskContext],
        uid: str,
    ) -> None:
        """Compensate when cancellation wins while hook setup is still running."""
        try:
            ctx = setup_future.result()
        except BaseException as exc:
            if not isinstance(exc, asyncio.CancelledError):
                logger.warning(
                    "[%s] cancelled setup later failed; no context to tear down: %r",
                    uid,
                    exc,
                )
            return
        self._update_lifecycle_metric("late_cleanups")
        try:
            self._submit_teardown_call(ctx.run_teardown)
        except RuntimeError:
            # Setup can finish after shutdown has retired both normal pools.
            # Keep reclamation off the callback/event-loop thread, and never
            # reopen a pool that also accepts setup or generation work.
            self._submit_late_cleanup(ctx.run_teardown, uid)

    def _submit_late_cleanup(self, function: Callable[[], Any], uid: str) -> None:
        # The lock is shared with lifecycle accounting and is initialized
        # before a late setup callback can get here.
        lock = self._lifecycle_metrics_lock
        with lock:
            executor = getattr(self, "_late_cleanup_executor", None)
            if executor is None:
                executor = ThreadPoolExecutor(
                    max_workers=8, thread_name_prefix="rllm-late-cleanup",
                )
                self._late_cleanup_executor = executor
            self._late_cleanup_pending = getattr(self, "_late_cleanup_pending", 0) + 1
            future = executor.submit(function)
        self._observe_lifecycle_future(future, pending_key="teardown_pending")

        def finished(done: Future[Any]) -> None:
            try:
                done.result()
            except BaseException:
                logger.exception("[%s] late setup cleanup failed after shutdown", uid)
                self._update_lifecycle_metric("deferred_cleanups")
            finally:
                retire = None
                with lock:
                    self._late_cleanup_pending -= 1
                    if not self._late_cleanup_pending:
                        retire = self._late_cleanup_executor
                        self._late_cleanup_executor = None
                if retire is not None:
                    retire.shutdown(wait=False)

        future.add_done_callback(finished)

    def _ensure_rollout_progress_registry(
        self,
    ) -> tuple[threading.RLock, dict[str, _ActiveRolloutProgress]]:
        """Return lazily initialized progress state.

        A few focused tests construct the engine with ``__new__``.  Lazy
        initialization keeps those callers compatible while production
        instances initialize these fields in ``__init__``.
        """

        lock = getattr(self, "_rollout_progress_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._rollout_progress_lock = lock
        progress = getattr(self, "_rollout_progress", None)
        if progress is None:
            progress = {}
            self._rollout_progress = progress
        return lock, progress

    def _register_rollout_progress(
        self,
        uid: str,
        task_id: str,
        rollout_index: int,
    ) -> None:
        lock, progress = self._ensure_rollout_progress_registry()
        now = time.monotonic()
        with lock:
            progress[uid] = _ActiveRolloutProgress(
                uid=uid,
                task_id=str(task_id),
                rollout_index=int(rollout_index),
                started_at=now,
                stage_started_at=now,
            )

    def _update_rollout_progress(
        self,
        uid: str,
        *,
        stage: str | object = _PROGRESS_UNSET,
        turn_index: int | None | object = _PROGRESS_UNSET,
        retry_attempt: int | object = _PROGRESS_UNSET,
    ) -> None:
        lock, progress = self._ensure_rollout_progress_registry()
        with lock:
            state = progress.get(uid)
            if state is None:
                # Retry attempts use distinct physical Gateway session ids, but
                # scheduler diagnostics remain keyed by the logical rollout id.
                try:
                    group_id, slot = split_rollout_session_id(uid)
                except ValueError:
                    pass
                else:
                    state = progress.get(f"{group_id}:{slot}")
            if state is None:
                return
            changed = False
            if stage is not _PROGRESS_UNSET and str(stage) != state.stage:
                state.stage = str(stage)
                changed = True
            if turn_index is not _PROGRESS_UNSET:
                normalized_turn = (
                    None if turn_index is None else max(0, int(turn_index))
                )
                if normalized_turn != state.turn_index:
                    state.turn_index = normalized_turn
                    changed = True
            if retry_attempt is not _PROGRESS_UNSET:
                normalized_retry = max(0, int(retry_attempt))
                if normalized_retry != state.retry_attempt:
                    state.retry_attempt = normalized_retry
                    changed = True
            if changed:
                state.stage_started_at = time.monotonic()

    def _remove_rollout_progress(self, uid: str) -> None:
        lock, progress = self._ensure_rollout_progress_registry()
        with lock:
            progress.pop(uid, None)

    def describe_rollout_task(self, task: asyncio.Task[Any]) -> str:
        """Return a compact UID/stage diagnostic for cancellation warnings."""
        task_name = task.get_name()
        prefix = "fully-async-rollout:"
        uid = task_name[len(prefix) :] if task_name.startswith(prefix) else task_name
        lock, progress = self._ensure_rollout_progress_registry()
        with lock:
            state = progress.get(uid)
            if state is None:
                return f"{uid} stage=unknown"
            turn = "none" if state.turn_index is None else str(state.turn_index)
            return f"{uid} stage={state.stage} turn={turn}"

    def _rollout_progress_snapshot(
        self,
        uid: str,
        *,
        task_id: str,
        rollout_index: int,
        scheduler_started_at: float,
        now: float,
    ) -> dict[str, Any]:
        lock, progress = self._ensure_rollout_progress_registry()
        with lock:
            state = progress.get(uid)
            if state is None:
                return {
                    "uid": uid,
                    "task_id": str(task_id),
                    "attempt_index": int(rollout_index),
                    "retry_attempt": 0,
                    "turn_index": None,
                    "turn_number": None,
                    "stage": "scheduled",
                    "stage_seconds": max(0.0, now - scheduler_started_at),
                    "active_seconds": max(0.0, now - scheduler_started_at),
                }
            turn_index = state.turn_index
            return {
                "uid": state.uid,
                "task_id": state.task_id,
                "attempt_index": state.rollout_index,
                "retry_attempt": state.retry_attempt,
                "turn_index": turn_index,
                "turn_number": None if turn_index is None else turn_index + 1,
                "stage": state.stage,
                "stage_seconds": max(0.0, now - state.stage_started_at),
                "active_seconds": max(0.0, now - state.started_at),
            }

    def set_training_step(self, step: int, mode: str = "train", epoch: int = 0) -> None:
        self.current_step = step
        self.current_mode = mode
        self.current_epoch = epoch

    def configure_dynamic_sampling_rollout_logging(self, enabled: bool) -> None:
        """Defer training rollout persistence until outcome classification."""

        self._defer_dynamic_sampling_rollout_logging = bool(enabled)

    def _rollout_log_context(
        self,
        task_id: str,
        rollout_idx: int,
        result_idx: int,
    ) -> dict[str, Any]:
        return {
            "task_id": task_id,
            "rollout_idx": rollout_idx,
            "result_idx": result_idx,
            "mode": self.current_mode,
            "epoch": self.current_epoch,
            "global_step": self.current_step,
        }

    def _maybe_log_rollout_sample(self, task_id: str, rollout_idx: int, result_idx: int, episode: Episode) -> None:
        if not self.rollout_log_path:
            return

        context = self._rollout_log_context(task_id, rollout_idx, result_idx)
        dispatch = episode.metadata.get("rollout_lifecycle_dispatch")
        if isinstance(dispatch, dict):
            context.update(dispatch)
        if (
            (getattr(self, "_defer_dynamic_sampling_rollout_logging", False)
             or getattr(self, "denovo_background_finalize_enable", False))
            and context["mode"] == "train"
        ):
            episode.artifacts[_DYNAMIC_SAMPLING_ROLLOUT_CONTEXT] = context
            return

        self._log_rollout_sample(episode, context=context)

    def log_dynamic_sampling_rollout_samples(
        self,
        episodes: list[Episode],
        *,
        filtered_trajectory_ids: set[int],
        lifecycle: dict[str, Any] | None = None,
    ) -> None:
        """Persist deferred samples with their final outcome-filter status."""

        if not self.rollout_log_path:
            return
        for episode in episodes:
            context = episode.artifacts.pop(
                _DYNAMIC_SAMPLING_ROLLOUT_CONTEXT,
                None,
            )
            if context is None:
                logger.warning(
                    "Missing deferred rollout log context for episode %s",
                    episode.id,
                )
                continue
            statuses = {
                id(trajectory): (
                    "filtered"
                    if id(trajectory) in filtered_trajectory_ids
                    else "consumed"
                )
                for trajectory in episode.trajectories
            }
            self._log_rollout_sample(
                episode,
                context=context,
                dynamic_sampling_statuses=(
                    statuses if getattr(self, "_defer_dynamic_sampling_rollout_logging", False) else None
                ),
                lifecycle=lifecycle,
            )

    def discard_dynamic_sampling_rollout_samples(
        self,
        episodes: list[Episode],
    ) -> int:
        """Discard speculative deferred samples without creating JSON files."""
        discarded = 0
        for episode in episodes:
            if episode.artifacts.pop(
                _DYNAMIC_SAMPLING_ROLLOUT_CONTEXT,
                None,
            ) is not None:
                discarded += 1
        return discarded

    @staticmethod
    def _deferred_shadow_lease_id(episode: Episode) -> str | None:
        metadata = episode.metadata if isinstance(episode.metadata, dict) else {}
        lifecycle = metadata.get(_DENOVO_BACKGROUND_METADATA_KEY)
        if not isinstance(lifecycle, dict):
            return None
        value = lifecycle.get("lease_id")
        return str(value) if isinstance(value, str) and value else None

    def has_deferred_shadow(self, episode: Episode) -> bool:
        lease_id = self._deferred_shadow_lease_id(episode)
        if lease_id is None:
            return False
        with self._deferred_shadow_registry_lock:
            return lease_id in self._deferred_shadow_leases

    def _pop_deferred_shadow_lease(
        self,
        lease_id: str,
    ) -> _DeferredShadowLease | None:
        with self._deferred_shadow_registry_lock:
            return self._deferred_shadow_leases.pop(lease_id, None)

    async def _close_deferred_shadow_lease(
        self,
        lease: _DeferredShadowLease,
    ) -> None:
        if lease.cleanup_future is None:
            lease.cleanup_future = self._submit_teardown_call(lease.close_shadow)
        try:
            await self._await_lifecycle_future(lease.cleanup_future)
        except BaseException as exc:
            lease.cleanup_error = exc
            lease.resource_request.cleanup_unconfirmed = True
            lease.enriched_episode.metadata.setdefault(_DENOVO_BACKGROUND_METADATA_KEY, {}).update(
                cleanup_confirmed=False, cleanup_error=str(exc)[:2000],
            )
            raise
        lease.resource_request.cleanup_unconfirmed = False
        if not lease.budget_released:
            await self._release_denovo_shadow_budget(lease.resource_request)
            lease.budget_released = True
        lease.cleanup_error = None
        lease.enriched_episode.metadata.setdefault(_DENOVO_BACKGROUND_METADATA_KEY, {})["cleanup_confirmed"] = True
        with self._deferred_shadow_registry_lock:
            if self._deferred_shadow_leases.get(lease.uid) is lease:
                self._deferred_shadow_leases.pop(lease.uid)

    async def cancel_deferred_shadow_episodes(
        self,
        episodes: list[Episode],
        *,
        disposition: str,
    ) -> int:
        leases: list[_DeferredShadowLease] = []
        for episode in episodes:
            lease_id = self._deferred_shadow_lease_id(episode)
            if lease_id is None:
                continue
            with self._deferred_shadow_registry_lock:
                lease = self._deferred_shadow_leases.get(lease_id)
            if lease is None:
                continue
            shadow_lifecycle = lease.cancel(disposition)
            if episode is not lease.enriched_episode:
                lease.runtime.copy_results_to_episode(
                    lease.raw_episode,
                    episode,
                )
            lifecycle = dict(
                episode.metadata.get(_DENOVO_BACKGROUND_METADATA_KEY) or {}
            )
            cancelled_at = time.monotonic()
            input_sealed_at = shadow_lifecycle.get(
                "input_sealed_at_monotonic"
            )
            primary_finished_at = lifecycle.get(
                "primary_verifier_finished_at_monotonic"
            )
            shadow_lifetime_s = max(
                0.0,
                cancelled_at
                - (
                    float(input_sealed_at)
                    if isinstance(input_sealed_at, int | float)
                    else lease.registered_at_monotonic
                ),
            )
            post_primary_overlap_s = max(
                0.0,
                cancelled_at
                - (
                    float(primary_finished_at)
                    if isinstance(primary_finished_at, int | float)
                    else lease.primary_verifier_finished_at_monotonic
                ),
            )
            lifecycle.update(
                {
                    "status": "shadow_cancelled",
                    "training_disposition": disposition,
                    "shadow_finalize_disposition": shadow_lifecycle.get(
                        "finalize_disposition"
                    )
                    or disposition,
                    "completed_probes_at_disposition": shadow_lifecycle.get(
                        "completed_probes_at_disposition"
                    ),
                    "cancelled_pending_steps": shadow_lifecycle.get(
                        "cancelled_pending_steps"
                    ),
                    "shadow_finalized_at_monotonic": cancelled_at,
                    "shadow_background_lifetime_s": shadow_lifetime_s,
                    "shadow_post_primary_overlap_s": post_primary_overlap_s,
                    "shadow_resource_budget": (
                        self._denovo_shadow_budget.snapshot()
                        if self._denovo_shadow_budget is not None
                        else {}
                    ),
                }
            )
            episode.metadata[_DENOVO_BACKGROUND_METADATA_KEY] = lifecycle
            milestone_config = getattr(
                self,
                "milestone_reward_config",
                None,
            )
            try:
                if milestone_config is not None and milestone_config.enable:
                    annotate_episode_milestone_rewards(
                        episode,
                        milestone_config,
                    )
                annotate_episode_milestone_verification_metrics(
                    episode,
                    milestone_config,
                )
            except Exception as exc:
                # Cancellation is a lifecycle/audit path and must never turn a
                # filtered or requeued group into a trainer-fatal error.
                logger.exception(
                    "[%s] milestone annotation failed after shadow cancellation",
                    episode.id,
                )
                episode.metadata[MILESTONE_REWARD_METADATA_KEY] = {
                    "schema_version": 1,
                    "status": "error",
                    "error": str(exc),
                }
                try:
                    annotate_episode_milestone_verification_metrics(
                        episode,
                        milestone_config,
                    )
                except Exception:
                    logger.exception(
                        "[%s] compact validation annotation also failed "
                        "after shadow cancellation",
                        episode.id,
                    )
            leases.append(lease)
        if leases:
            await asyncio.gather(
                *(self._close_deferred_shadow_lease(lease) for lease in leases)
            )
        return len(leases)

    async def finalize_deferred_shadow_batch(
        self,
        episodes: list[Episode],
        *,
        stall_timeout: float,
    ) -> dict[str, float]:
        """Finalize exactly one selected optimizer batch at one deadline."""

        leases: list[_DeferredShadowLease] = []
        selected_episode_by_lease: dict[str, Episode] = {}
        for episode in episodes:
            lease_id = self._deferred_shadow_lease_id(episode)
            if lease_id is None:
                continue
            with self._deferred_shadow_registry_lock:
                lease = self._deferred_shadow_leases.get(lease_id)
            if lease is None:
                raise RuntimeError(
                    f"deferred shadow lease disappeared before barrier: {lease_id}"
                )
            leases.append(lease)
            selected_episode_by_lease[lease_id] = episode
        if not leases:
            return {}
        if len(leases) != len(episodes):
            raise RuntimeError(
                "selected DeNovo batch mixes deferred and non-deferred rollouts"
            )

        barrier_started_at = max(
            lease.primary_verifier_finished_at_monotonic for lease in leases
        )
        deadline = barrier_started_at + float(stall_timeout)
        barrier_observed_at = time.monotonic()
        for lease in leases:
            lease.mark_barrier(
                barrier_started_at_monotonic=barrier_started_at,
                deadline_monotonic=deadline,
            )
        drained_before_barrier = sum(
            lease.runtime.milestone_drained() for lease in leases
        )
        timed_out = 0
        completed_probes = 0
        pending_probes = 0
        finalized_uids: set[str] = set()
        barrier_wait_s = 0.0
        failure: BaseException | None = None
        try:
            await asyncio.gather(
                *(lease.wait_until(deadline) for lease in leases)
            )
            barrier_wait_s = max(0.0, time.monotonic() - barrier_observed_at)
            for lease in leases:
                lifecycle = lease.seal_partial(
                    barrier_started_at_monotonic=barrier_started_at,
                    deadline_monotonic=deadline,
                )
                finalized_uids.add(lease.uid)
                if lifecycle.get("finalize_disposition") == "timeout":
                    timed_out += 1
                completed_probes += int(
                    lifecycle.get("completed_probes_at_barrier") or 0
                )
                pending_probes += int(
                    lifecycle.get("pending_probes_at_barrier") or 0
                )
                episode = selected_episode_by_lease[lease.uid]
                if episode is not lease.enriched_episode:
                    lease.runtime.copy_results_to_episode(
                        lease.raw_episode,
                        episode,
                    )
                metadata = dict(
                    episode.metadata.get(_DENOVO_BACKGROUND_METADATA_KEY) or {}
                )
                input_sealed_at = lifecycle.get(
                    "input_sealed_at_monotonic"
                )
                metadata.update(
                    {
                        "status": "shadow_finalized",
                        "training_disposition": "selected_for_optimizer",
                        "batch_barrier_started_at_monotonic": barrier_started_at,
                        "batch_barrier_observed_at_monotonic": (
                            barrier_observed_at
                        ),
                        "batch_barrier_deadline_monotonic": deadline,
                        "batch_barrier_budget_remaining_at_reservation_s": max(
                            0.0,
                            deadline - barrier_observed_at,
                        ),
                        "batch_barrier_wait_s": barrier_wait_s,
                        "shadow_background_head_start_s": max(
                            0.0,
                            barrier_started_at
                            - (
                                float(input_sealed_at)
                                if isinstance(input_sealed_at, int | float)
                                else lease.registered_at_monotonic
                            ),
                        ),
                        "shadow_finalize_disposition": lifecycle.get(
                            "finalize_disposition"
                        ),
                        "completed_probes_at_barrier": lifecycle.get(
                            "completed_probes_at_barrier"
                        ),
                        "completed_probes_at_barrier_start": lifecycle.get(
                            "completed_probes_at_barrier_start"
                        ),
                        "pending_probes_at_barrier": lifecycle.get(
                            "pending_probes_at_barrier"
                        ),
                        "shadow_resource_budget": (
                            self._denovo_shadow_budget.snapshot()
                            if self._denovo_shadow_budget is not None
                            else {}
                        ),
                    }
                )
                episode.metadata[_DENOVO_BACKGROUND_METADATA_KEY] = metadata
                milestone_config = self.milestone_reward_config
                if milestone_config.enable:
                    annotate_episode_milestone_rewards(
                        episode,
                        milestone_config,
                    )
                annotate_episode_milestone_verification_metrics(
                    episode,
                    milestone_config,
                )
        except BaseException as exc:
            failure = exc
            disposition = (
                "batch_barrier_cancelled"
                if isinstance(exc, asyncio.CancelledError)
                else "batch_barrier_error"
            )
            for lease in leases:
                if lease.uid in finalized_uids:
                    continue
                try:
                    lease.cancel(disposition)
                    selected = selected_episode_by_lease[lease.uid]
                    if selected is not lease.enriched_episode:
                        lease.runtime.copy_results_to_episode(
                            lease.raw_episode,
                            selected,
                        )
                except Exception:
                    logger.exception(
                        "[%s] failed to seal deferred shadow after barrier error",
                        lease.uid,
                    )
            raise
        finally:
            close_results = await asyncio.gather(
                *(self._close_deferred_shadow_lease(lease) for lease in leases),
                return_exceptions=True,
            )
            for lease, close_result in zip(leases, close_results, strict=True):
                if isinstance(close_result, BaseException):
                    logger.error(
                        "[%s] deferred shadow close failed after barrier: %r",
                        lease.uid,
                        close_result,
                    )
                    if failure is None:
                        raise close_result
                    from rllm.utils.infrastructure_diagnostics import infrastructure_exception_chain
                    failure.cleanup_errors = [*getattr(failure, "cleanup_errors", []), infrastructure_exception_chain(close_result)]
                else:
                    self._pop_deferred_shadow_lease(lease.uid)

        return {
            "denovo_shadow/barrier_rollouts": float(len(leases)),
            "denovo_shadow/drained_before_barrier": float(
                drained_before_barrier
            ),
            "denovo_shadow/timed_out_rollouts": float(timed_out),
            "denovo_shadow/completed_probes": float(completed_probes),
            "denovo_shadow/pending_probes_at_timeout": float(pending_probes),
            "denovo_shadow/barrier_wait_s": float(barrier_wait_s),
            "denovo_shadow/barrier_elapsed_since_latest_primary_s": max(
                0.0,
                min(time.monotonic(), deadline) - barrier_started_at,
            ),
            "denovo_shadow/barrier_budget_remaining_at_reservation_s": max(
                0.0,
                deadline - barrier_observed_at,
            ),
            **(
                {
                    "denovo_shadow/peak_active_sandboxes": float(
                        self._denovo_shadow_budget.peak_count
                    ),
                    "denovo_shadow/peak_active_cpus": float(
                        self._denovo_shadow_budget.peak_cpus
                    ),
                    "denovo_shadow/peak_active_memory_mb": float(
                        self._denovo_shadow_budget.peak_memory_mb
                    ),
                }
                if self._denovo_shadow_budget is not None
                else {}
            ),
        }

    def _log_rollout_sample(
        self,
        episode: Episode,
        *,
        context: dict[str, Any],
        dynamic_sampling_statuses: dict[int, str] | None = None,
        lifecycle: dict[str, Any] | None = None,
    ) -> None:
        task_id = str(context["task_id"])
        rollout_idx = int(context["rollout_idx"])
        result_idx = int(context["result_idx"])

        self._rollout_log_completed += 1

        try:
            os.makedirs(self.rollout_log_path, exist_ok=True)
            if episode.trajectories:
                for traj_idx, trajectory in enumerate(episode.trajectories):
                    status = (
                        dynamic_sampling_statuses.get(id(trajectory))
                        if dynamic_sampling_statuses is not None
                        else None
                    )
                    trajectory_lifecycle = self._rollout_lifecycle(
                        context,
                        status,
                        lifecycle,
                    )
                    payload = self._rollout_sample_payload(
                        task_id,
                        rollout_idx,
                        result_idx,
                        episode,
                        trajectory,
                        traj_idx,
                        log_context=context,
                        dynamic_sampling_status=status,
                        rollout_lifecycle=trajectory_lifecycle,
                    )
                    write_context = dict(context)
                    if trajectory_lifecycle is not None:
                        write_context["rollout_lifecycle"] = trajectory_lifecycle
                    self._write_rollout_sample(
                        payload,
                        task_id,
                        rollout_idx,
                        result_idx,
                        traj_idx,
                        episode,
                        log_context=write_context,
                    )
            else:
                status = (
                    "consumed"
                    if dynamic_sampling_statuses is not None
                    else None
                )
                trajectory_lifecycle = self._rollout_lifecycle(
                    context,
                    status,
                    lifecycle,
                )
                payload = self._rollout_sample_payload(
                    task_id,
                    rollout_idx,
                    result_idx,
                    episode,
                    None,
                    None,
                    log_context=context,
                    dynamic_sampling_status=status,
                    rollout_lifecycle=trajectory_lifecycle,
                )
                write_context = dict(context)
                if trajectory_lifecycle is not None:
                    write_context["rollout_lifecycle"] = trajectory_lifecycle
                self._write_rollout_sample(
                    payload,
                    task_id,
                    rollout_idx,
                    result_idx,
                    None,
                    episode,
                    log_context=write_context,
                )
        except Exception as e:
            logger.warning("Failed to write rollout sample to %s: %s", self.rollout_log_path, e)

    def _rollout_sample_payload(
        self,
        task_id: str,
        rollout_idx: int,
        result_idx: int,
        episode: Episode,
        trajectory: Trajectory | None,
        traj_idx: int | None,
        *,
        log_context: dict[str, Any] | None = None,
        dynamic_sampling_status: str | None = None,
        rollout_lifecycle: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if (
            dynamic_sampling_status is not None
            and dynamic_sampling_status not in _DYNAMIC_SAMPLING_STATUSES
        ):
            raise ValueError(
                "dynamic_sampling_status must be 'consumed' or 'filtered'"
            )
        context = log_context or self._rollout_log_context(
            task_id,
            rollout_idx,
            result_idx,
        )
        steps = trajectory.steps if trajectory is not None else []
        reward = None
        if trajectory is not None:
            reward = trajectory.reward
            if reward is None and steps:
                reward = steps[-1].reward

        effective_global_step: int | None = int(context["global_step"])
        if rollout_lifecycle is not None:
            optimizer_step = rollout_lifecycle.get("optimizer_step")
            effective_global_step = (
                int(optimizer_step)
                if optimizer_step is not None
                and rollout_lifecycle.get("optimizer_committed")
                else None
            )
        payload = {
            "schema_version": 6,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": context["mode"],
            "epoch": context["epoch"],
            "global_step": effective_global_step,
            "dispatch_step": int(
                context.get("dispatch_step", context["global_step"])
            ),
            "sample_index": self._rollout_log_completed,
            "task_id": task_id,
            "rollout_idx": rollout_idx,
            "result_idx": result_idx,
            "episode_id": episode.id,
            "has_trajectory": trajectory is not None,
            "trajectory_idx": traj_idx,
            "reward": reward,
            "termination": self._termination_value(episode.termination_reason),
            "correct": bool(episode.is_correct),
            "n_turns": episode.metrics.get("n_turns", sum(len(t.steps) for t in episode.trajectories)),
            "metrics": episode.metrics,
            "metadata": episode.metadata,
            "task": episode.task,
            "codeflow_bash_behavior": summarize_codeflow_bash_behavior(steps),
            "milestone_verification": summarize_milestone_verification(
                episode,
                trajectory,
                getattr(self, "milestone_reward_config", None),
            ),
            "trajectory": None
            if trajectory is None
            else {
                "uid": trajectory.uid,
                "name": trajectory.name,
                "reward": reward,
                "num_steps": len(steps),
                "metadata": trajectory.metadata,
                "initial_input": steps[0].input if steps else None,
                "steps": [self._rollout_step_payload(step_idx, step) for step_idx, step in enumerate(steps)],
            },
        }
        sequence_metadata = (
            trajectory.metadata
            if trajectory is not None and isinstance(trajectory.metadata, dict)
            else episode.metadata
        )
        if isinstance(sequence_metadata, dict):
            for key in (
                "model_window_tokens",
                "initial_prompt_tokens",
                "initial_response_budget_tokens",
            ):
                if key in sequence_metadata:
                    payload[key] = sequence_metadata[key]
        if dynamic_sampling_status is not None:
            payload["dynamic_sampling_status"] = dynamic_sampling_status
        if rollout_lifecycle is not None:
            payload["rollout_lifecycle"] = rollout_lifecycle
        return payload

    @staticmethod
    def _rollout_lifecycle(
        context: dict[str, Any],
        dynamic_sampling_status: str | None,
        lifecycle: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if dynamic_sampling_status is None and lifecycle is None:
            return None
        value = dict(lifecycle or {})
        value.setdefault("schema_version", 1)
        value["dispatch_step"] = int(
            context.get("dispatch_step", context["global_step"])
        )
        value["dispatch_weight_version"] = int(
            context.get("dispatch_weight_version", 0)
        )
        value["sampling_wave_index"] = context.get(
            "sampling_wave_index"
        )
        if dynamic_sampling_status == "filtered":
            value["optimizer_step"] = None
            value["optimizer_committed"] = False
            value["training_disposition"] = "uniform_filtered"
        else:
            value.setdefault("optimizer_step", None)
            value.setdefault("optimizer_committed", False)
            value.setdefault("training_disposition", "consumed_other")
        return value

    def _annotate_sequence_budget_metadata(self, episode: Episode) -> None:
        """Attach the exact first-turn budget observed in gateway traces."""
        model_window_tokens = getattr(self, "model_window_tokens", None)
        if model_window_tokens is None:
            return

        prompt_lengths: list[int] = []
        for trajectory in episode.trajectories:
            first_step = next(
                (
                    step
                    for step in trajectory.steps
                    if step.model_output is not None
                    and step.model_output.prompt_ids is not None
                ),
                None,
            )
            if first_step is None:
                continue
            initial_prompt_tokens = len(first_step.model_output.prompt_ids)
            initial_response_budget_tokens = max(
                model_window_tokens - initial_prompt_tokens, 0
            )
            trajectory.info.update(
                {
                    "model_window_tokens": model_window_tokens,
                    "initial_prompt_tokens": initial_prompt_tokens,
                    "initial_response_budget_tokens": (
                        initial_response_budget_tokens
                    ),
                }
            )
            prompt_lengths.append(initial_prompt_tokens)

        if not prompt_lengths:
            return
        initial_prompt_tokens = max(prompt_lengths)
        episode.metadata.update(
            {
                "model_window_tokens": model_window_tokens,
                "initial_prompt_tokens": initial_prompt_tokens,
                "initial_response_budget_tokens": max(
                    model_window_tokens - initial_prompt_tokens, 0
                ),
            }
        )

    def _rollout_step_payload(self, step_idx: int, step: Step) -> dict[str, Any]:
        metadata = step.metadata or {}
        observation = step.observation
        if observation is None and isinstance(metadata, dict):
            observation = metadata.get("tool_observation")
            agent_metadata = metadata.get("agent_step_metadata")
            if observation is None and isinstance(agent_metadata, dict):
                observation = agent_metadata.get("tool_observation")
        milestone_config = getattr(self, "milestone_reward_config", None)
        if milestone_config is not None and milestone_config.enable:
            tool_call_reward = compute_milestone_format_reward(step)
        else:
            tool_call_reward = compute_tool_call_reward(step)
        metadata = step.metadata or {}

        model_output = step.model_output
        prompt_length = getattr(model_output, "prompt_length", len(step.prompt_ids or [])) if model_output is not None else len(step.prompt_ids or [])
        completion_length = getattr(model_output, "completion_length", len(step.response_ids or [])) if model_output is not None else len(step.response_ids or [])

        return {
            "index": step_idx,
            "id": step.id,
            "model_response": step.model_response or step.output or "",
            "thought": step.thought,
            "action": step.action.action if isinstance(step.action, Action) else step.action,
            "observation": observation,
            "done": step.done,
            "finish_reason": getattr(model_output, "finish_reason", None),
            "prompt_length": prompt_length,
            "completion_length": completion_length,
            "tool_call_reward": tool_call_reward,
            "metadata": self._rollout_step_metadata(metadata),
        }

    def _rollout_step_metadata(self, metadata: Any) -> Any:
        if not isinstance(metadata, dict):
            return metadata

        omitted_keys = {"tool_observation", "tool_call_reward", "tool_call_reward_missing_metadata"}
        cleaned = {}
        for key, value in metadata.items():
            if key in omitted_keys:
                continue
            if key == "agent_step_metadata" and isinstance(value, dict):
                agent_metadata = {k: v for k, v in value.items() if k not in omitted_keys}
                if agent_metadata:
                    cleaned[key] = agent_metadata
                continue
            cleaned[key] = value
        return cleaned

    def _write_rollout_sample(
        self,
        payload: dict[str, Any],
        task_id: str,
        rollout_idx: int,
        result_idx: int,
        traj_idx: int | None,
        episode: Episode,
        *,
        log_context: dict[str, Any] | None = None,
    ) -> None:
        assert self.rollout_log_path is not None
        filename = self._rollout_sample_filename(
            task_id,
            rollout_idx,
            result_idx,
            traj_idx,
            episode,
            log_context=log_context,
        )
        path = os.path.join(self.rollout_log_path, filename)
        tmp_path = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
                f.write("\n")
                f.flush()
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _rollout_sample_filename(
        self,
        task_id: str,
        rollout_idx: int,
        result_idx: int,
        traj_idx: int | None,
        episode: Episode,
        *,
        log_context: dict[str, Any] | None = None,
    ) -> str:
        context = log_context or self._rollout_log_context(
            task_id,
            rollout_idx,
            result_idx,
        )
        safe_task_id = self._safe_filename(task_id)[:80] or "task"
        safe_mode = self._safe_filename(context["mode"]) or "mode"
        safe_termination = self._safe_filename(self._termination_value(episode.termination_reason) or "unknown")
        traj_part = "none" if traj_idx is None else str(traj_idx)
        lifecycle = context.get("rollout_lifecycle")
        if isinstance(lifecycle, dict):
            optimizer_step = lifecycle.get("optimizer_step")
            if optimizer_step is not None and lifecycle.get(
                "optimizer_committed"
            ):
                step_part = f"_step{int(optimizer_step):06d}"
            elif optimizer_step is not None:
                step_part = f"_attemptedstep{int(optimizer_step):06d}"
            elif lifecycle.get("training_disposition") == "untrained_tail":
                step_part = (
                    f"_untrained_after_step"
                    f"{int(context.get('dispatch_step', context['global_step'])):06d}"
                )
            else:
                step_part = (
                    f"_dispatchstep"
                    f"{int(context.get('dispatch_step', context['global_step'])):06d}"
                )
        else:
            step_part = f"_step{int(context['global_step']):06d}"
        return (
            f"{self._rollout_log_completed:06d}"
            f"{step_part}"
            f"_{safe_mode}"
            f"_task-{safe_task_id}"
            f"_rollout{rollout_idx}"
            f"_result{result_idx}"
            f"_traj{traj_part}"
            f"_{safe_termination}.json"
        )

    @staticmethod
    def _termination_value(termination_reason: Any) -> str | None:
        if termination_reason is None:
            return None
        return getattr(termination_reason, "value", str(termination_reason))

    @staticmethod
    def _safe_filename(value: Any) -> str:
        return _FILENAME_SAFE_RE.sub("_", str(value)).strip("._")

    async def execute_tasks(
        self,
        tasks: list[dict | Task],
        task_ids: list[str] | None = None,
        is_validation: bool = False,
        rollout_indices: list[int] | None = None,
        on_episode_complete: (
            Callable[[str, int, int, Episode], Awaitable[None] | None] | None
        ) = None,
        collect_results: bool = True,
        on_scheduler_update: (
            Callable[[dict[str, Any]], Awaitable[None] | None] | None
        ) = None,
        **kwargs,
    ) -> list[Episode]:
        """Run AgentFlows on a list of tasks; return enriched Episodes.

        ``tasks`` may be raw dicts (training path; the engine wraps each
        in a :class:`Task` internally) or fully-constructed :class:`Task`
        objects (eval path; the engine uses them as-is). When ``task_ids``
        is omitted, fresh UUIDs are assigned.

        Runs per-task pipelines (flow + trace fetch + enrich + evaluate) in a
        bounded rolling window.  A newly available slot is filled immediately;
        long tail tasks therefore occupy only their own slots.  With
        ``collect_results=False`` completed Episodes are released after the
        persistence callback instead of retaining the entire evaluation sweep
        in memory.  The default preserves the historical ordered return value.

        Each attempt deletes its own session; one batch delete at the end is
        an idempotent safety net for older/custom execution paths.
        """
        if not isinstance(collect_results, bool):
            raise TypeError("collect_results must be a boolean")
        if task_ids is None:
            task_ids = [str(uuid.uuid4()) for _ in tasks]
        if len(task_ids) != len(tasks):
            raise ValueError(
                f"task_ids must have the same length as tasks: {len(task_ids)} != {len(tasks)}"
            )
        if rollout_indices is not None:
            if len(rollout_indices) != len(tasks):
                raise ValueError(
                    "rollout_indices must have the same length as tasks: "
                    f"{len(rollout_indices)} != {len(tasks)}"
                )
            invalid = [
                value
                for value in rollout_indices
                if isinstance(value, bool) or not isinstance(value, int) or value < 0
            ]
            if invalid:
                raise ValueError(
                    "rollout_indices must contain non-negative integers; "
                    f"invalid values: {invalid[:5]!r}"
                )

        task_id_counter: dict[str, int] = defaultdict(int)
        resolved_rollout_indices: list[int] = []
        uids: list[str] = []
        seen_uids: set[str] = set()
        for idx, task_id in enumerate(task_ids):
            if rollout_indices is None:
                rollout_idx = task_id_counter[task_id]
                task_id_counter[task_id] += 1
            else:
                rollout_idx = rollout_indices[idx]
            uid = self._session_uid(task_id, rollout_idx)
            if uid in seen_uids:
                raise ValueError(
                    f"duplicate task/rollout identity in one execute_tasks call: {uid}"
                )
            resolved_rollout_indices.append(rollout_idx)
            uids.append(uid)
            seen_uids.add(uid)
        self._ensure_event_loop_lag_monitor()
        max_active = max(
            1,
            min(
                len(tasks) or 1,
                int(getattr(self, "n_parallel_tasks", len(tasks) or 1)),
            ),
        )
        results: list[Episode | None] | None = (
            [None] * len(tasks) if collect_results else None
        )
        active: dict[asyncio.Task, tuple[int, str, float]] = {}
        admission_waiting: set[int] = set()
        launched_uids: list[str] = []
        next_index = 0
        completed = 0

        def launch(index: int) -> None:
            task_id = task_ids[index]
            rollout_idx = resolved_rollout_indices[index]
            admission = getattr(self, "evaluation_admission", None)
            if admission is not None:
                admission_waiting.add(index)

            async def run_admitted():
                token = admission.token() if admission is not None else None
                acquired = False
                try:
                    if admission is not None:
                        await admission.acquire("task", token)
                        acquired = True
                        admission_waiting.discard(index)
                        current = asyncio.current_task()
                        active[current] = (index, uids[index], time.monotonic())
                    return await self.process_task_with_retry(
                        tasks[index], task_id, rollout_idx, index,
                        is_validation=is_validation,
                    )
                finally:
                    admission_waiting.discard(index)
                    if acquired:
                        await asyncio.shield(admission.release("task", token))

            future = asyncio.create_task(run_admitted())
            active[future] = (index, uids[index], time.monotonic())
            launched_uids.extend(
                retry_session_id(uids[index], attempt)
                for attempt in range(
                    1,
                    int(getattr(self, "retry_limit", 1)) + 1,
                )
            )
            if getattr(self, "session_namespace", ""):
                self.evaluation_session_ids.update(launched_uids)

        def scheduler_state(event: str) -> dict[str, Any]:
            now = time.monotonic()
            active_rollouts = [
                self._rollout_progress_snapshot(
                    uid,
                    task_id=task_ids[index],
                    rollout_index=resolved_rollout_indices[index],
                    scheduler_started_at=started_at,
                    now=now,
                )
                for _future, (index, uid, started_at) in active.items()
                if index not in admission_waiting
            ]
            active_rollouts.sort(
                key=lambda item: (-float(item["active_seconds"]), item["uid"])
            )
            active_by_stage = dict(
                sorted(Counter(item["stage"] for item in active_rollouts).items())
            )
            oldest = active_rollouts[0] if active_rollouts else None
            return {
                "event": event,
                "total": len(tasks),
                "launched": next_index,
                "completed": completed,
                "active": len(active) - len(admission_waiting),
                "queued": len(tasks) - next_index + len(admission_waiting),
                "admission_waiting": len(admission_waiting),
                "active_rollouts": active_rollouts,
                "active_by_stage": active_by_stage,
                "oldest_active_uid": (
                    oldest.get("uid") if oldest is not None else None
                ),
                "oldest_active_seconds": (
                    float(oldest.get("active_seconds", 0.0))
                    if oldest is not None
                    else 0.0
                ),
                "oldest_active_turn_index": (
                    oldest.get("turn_index") if oldest is not None else None
                ),
                "oldest_active_turn_number": (
                    oldest.get("turn_number") if oldest is not None else None
                ),
                "oldest_active_stage": (
                    oldest.get("stage") if oldest is not None else None
                ),
            }

        async def emit_scheduler_update(event: str) -> None:
            state = scheduler_state(event)
            if event == "heartbeat":
                logger.info(
                    "Trajectory scheduler heartbeat: completed=%d/%d active=%d "
                    "queued=%d stages=%s oldest=%s turn=%s stage=%s age=%.1fs",
                    state["completed"],
                    state["total"],
                    state["active"],
                    state["queued"],
                    state["active_by_stage"],
                    state["oldest_active_uid"],
                    state["oldest_active_turn_number"],
                    state["oldest_active_stage"],
                    state["oldest_active_seconds"],
                )
            if on_scheduler_update is None:
                return
            try:
                callback_result = on_scheduler_update(state)
                if inspect.isawaitable(callback_result):
                    await callback_result
            except Exception:
                # Progress reporting must never abort or retry a completed
                # rollout.  The durable episode callback remains strict.
                logger.exception("Trajectory scheduler update callback failed")

        while next_index < len(tasks) and len(active) < max_active:
            launch(next_index)
            next_index += 1

        try:
            await emit_scheduler_update("started")
            with tqdm(total=len(tasks), desc="Generating trajectories") as pbar:
                while active:
                    done, _pending = await asyncio.wait(
                        set(active),
                        timeout=_EXECUTE_TASKS_HEARTBEAT_S,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if not done:
                        await emit_scheduler_update("heartbeat")
                        continue

                    completed_items: list[tuple[str, int, int, Episode]] = []
                    first_error: BaseException | None = None
                    for future in done:
                        active.pop(future, None)
                        try:
                            completed_items.append(future.result())
                        except BaseException as exc:
                            if first_error is None:
                                first_error = exc
                    if first_error is not None:
                        raise first_error

                    # Refill before potentially slow shared-filesystem persistence callbacks.
                    while not getattr(self, "session_namespace", "") and next_index < len(tasks) and len(active) < max_active:
                        launch(next_index)
                        next_index += 1

                    for task_id, rollout_idx, result_idx, episode in completed_items:
                        if results is not None:
                            results[result_idx] = episode
                        self._maybe_log_rollout_sample(
                            task_id,
                            rollout_idx,
                            result_idx,
                            episode,
                        )
                        if on_episode_complete is not None:
                            callback_result = on_episode_complete(
                                task_id,
                                rollout_idx,
                                result_idx,
                                episode,
                            )
                            if inspect.isawaitable(callback_result):
                                await callback_result
                        completed += 1
                        pbar.update(1)
                        await emit_scheduler_update("completion")
                    # Supervised evaluation must inspect outcomes before
                    # replacing failed tasks with more work on a dead runtime.
                    while getattr(self, "session_namespace", "") and next_index < len(tasks) and len(active) < max_active:
                        launch(next_index)
                        next_index += 1
            await emit_scheduler_update("finished")
        finally:
            unfinished = list(active)
            for future in unfinished:
                future.cancel()
            if unfinished:
                await asyncio.gather(*unfinished, return_exceptions=True)

            # Each attempt already deletes its session. This idempotent batch
            # delete remains a safety net for older/custom paths and also runs
            # if persistence callbacks fail or the batch is cancelled.
            if launched_uids and getattr(self.gateway, "available", True):
                try:
                    await self.gateway.adelete_sessions(launched_uids)
                except Exception:
                    logger.exception(
                        "Batch session delete failed; sessions may linger in the trace store"
                    )

        if results is None:
            return []
        if any(episode is None for episode in results):
            raise RuntimeError("trajectory scheduler finished with missing results")
        ordered_results: list[Episode] = results  # type: ignore[assignment]

        if self.episode_logger is not None:
            try:
                self.episode_logger.log_episodes_batch(
                    ordered_results,
                    self.current_step,
                    self.current_mode,
                    self.current_epoch,
                )
            except Exception as e:
                logger.error("Failed to log episodes: %s", e)

        return ordered_results

    def _session_uid(self, task_id: str, rollout_idx: int) -> str:
        namespace = getattr(self, "session_namespace", "")
        # Keep the generated namespace delimiter safe in session URLs.
        return f"{namespace}~{task_id}:{rollout_idx}" if namespace else f"{task_id}:{rollout_idx}"

    async def process_task_with_retry(
        self,
        task: dict | Task,
        task_id: str,
        rollout_idx: int,
        result_idx: int,
        is_validation: bool = False,
    ) -> tuple[str, int, int, Episode]:
        """Run the full per-task pipeline with retry.

        Each attempt runs flow + trace fetch + enrich + evaluate, then clears
        its session before returning, retrying, or propagating cancellation.
        """
        self._ensure_event_loop_lag_monitor()
        task_obj = task if isinstance(task, Task) else task_from_row(task, task_id)
        denovo_background = self._denovo_background_task(
            task_obj,
            is_validation=is_validation,
        )
        if denovo_background:
            task_obj = self._activate_denovo_background_task(task_obj)
        task_for_episode = task_obj.metadata
        uid = self._session_uid(task_id, rollout_idx)
        trajectory_timeout = getattr(self, "trajectory_timeout", None)
        rollout_deadline = (
            asyncio.get_running_loop().time() + trajectory_timeout
            if trajectory_timeout is not None and not denovo_background
            else None
        )
        # DeNovo admission is scheduler backpressure, not agent execution.
        # Its deadline starts only after both resource and rollout admission;
        # legacy profiles retain their existing end-to-end timeout contract.
        admission_timings: dict[str, float] = {}
        shadow_resource_request: _ShadowResourceRequest | None = None
        rollout_slot_held = False
        shadow_ownership_transferred = False
        self._register_rollout_progress(uid, task_id, rollout_idx)
        try:
            if denovo_background and getattr(self, "denovo_shadow_finalize_enable", True):
                self._update_rollout_progress(
                    uid,
                    stage="shadow_resource_queue",
                    turn_index=None,
                )
                resource_queue_started = time.perf_counter()
                shadow_resource_request = (
                    await self._acquire_denovo_shadow_budget(task_obj)
                )
                admission_timings["time/shadow_resource_queue_s"] = (
                    time.perf_counter() - resource_queue_started
                )
            rollout_queue_started = time.perf_counter()
            self._update_rollout_progress(
                uid,
                stage="rollout_queue",
                turn_index=None,
            )
            try:
                if rollout_deadline is None:
                    await self._semaphore.acquire()
                    rollout_slot_held = True
                else:
                    queue_remaining = (
                        rollout_deadline - asyncio.get_running_loop().time()
                    )
                    if queue_remaining <= 0:
                        raise TimeoutError
                    await asyncio.wait_for(
                        self._semaphore.acquire(),
                        timeout=queue_remaining,
                    )
                    rollout_slot_held = True
            except TimeoutError as exc:
                self._update_lifecycle_metric("trajectory_timeouts")
                failure = RolloutInfrastructureError(
                    "trajectory_timeout",
                    f"rollout exceeded the total trajectory timeout of {trajectory_timeout:g}s while waiting for a rollout slot",
                    retryable=False,
                    stage="rollout_queue",
                    retry_scope="none",
                )
                if self.raise_on_error:
                    raise failure from exc
                failure_metadata: dict[str, Any] = {
                    "error": {"message": str(failure)},
                    "infrastructure_failure": {
                        "schema_version": 1,
                        "reason": failure.reason,
                        "error_type": type(failure).__name__,
                        "error": str(failure),
                        "attempts": 0,
                        "stage": failure.stage,
                        "retry_scope": failure.retry_scope,
                        "safe_for_final_recovery": False,
                    },
                }
                _attach_denovo_background_failure_lifecycle(
                    failure_metadata,
                    task_for_episode,
                    reason=failure.reason,
                    stage=failure.stage,
                )
                return (
                    task_id,
                    rollout_idx,
                    result_idx,
                    Episode(
                        id=uid,
                        task=task_for_episode,
                        is_correct=False,
                        termination_reason=TerminationReason.ERROR,
                        metadata=failure_metadata,
                    ),
                )
            if denovo_background:
                admission_timings["time/rollout_queue_s"] = (
                    time.perf_counter() - rollout_queue_started
                )
                rollout_deadline = (
                    asyncio.get_running_loop().time() + trajectory_timeout
                    if trajectory_timeout is not None else None
                )
            attempt_failures: list[dict[str, Any]] = []
            try:
                for retry_attempt in range(1, self.retry_limit + 1):
                    attempt_uid = retry_session_id(uid, retry_attempt)
                    next_stage_after_cleanup: str | None = None
                    episode: Episode | None = None
                    gateway_cleaned_early = False
                    self._update_rollout_progress(
                        uid,
                        stage="setup",
                        turn_index=None,
                        retry_attempt=retry_attempt - 1,
                    )
                    try:
                        remaining = (
                            rollout_deadline - asyncio.get_running_loop().time()
                            if rollout_deadline is not None
                            else None
                        )
                        if remaining is not None and remaining <= 0:
                            raise RolloutInfrastructureError(
                                "trajectory_timeout",
                                f"rollout exceeded the total trajectory timeout of {trajectory_timeout:g}s",
                                retryable=False,
                                stage="rollout",
                                retry_scope="none",
                            )
                        async def release_model_stage(
                            session_uid: str = attempt_uid,
                        ) -> None:
                            nonlocal rollout_slot_held, gateway_cleaned_early
                            self._update_rollout_progress(
                                uid,
                                stage="gateway_cleanup_before_verifier",
                            )
                            await self._cleanup_gateway_session(session_uid)
                            gateway_cleaned_early = True
                            if rollout_slot_held:
                                self._semaphore.release()
                                rollout_slot_held = False

                        if denovo_background:
                            episode = await self._run_single_denovo_background(
                                task_obj,
                                attempt_uid,
                                is_validation=is_validation,
                                agent_phase_timeout=remaining,
                                release_model_stage=release_model_stage,
                                shadow_resource_request=shadow_resource_request,
                            )
                            shadow_ownership_transferred = shadow_resource_request is not None
                        else:
                            attempt_coro = self._run_single(
                                task_obj,
                                attempt_uid,
                                is_validation=is_validation,
                            )
                            if remaining is None:
                                episode = await attempt_coro
                            else:
                                attempt_task = asyncio.create_task(attempt_coro)
                                done, _pending = await asyncio.wait(
                                    {attempt_task},
                                    timeout=remaining,
                                )
                                if not done:
                                    attempt_task.cancel()
                                    await asyncio.gather(
                                        attempt_task,
                                        return_exceptions=True,
                                    )
                                    raise RolloutInfrastructureError(
                                        "trajectory_timeout",
                                        f"rollout exceeded the total trajectory timeout of {trajectory_timeout:g}s",
                                        retryable=False,
                                        stage="rollout",
                                        retry_scope="none",
                                    )
                                # Preserve an application-raised TimeoutError. Only
                                # expiry of our own wait is trajectory timeout.
                                episode = await attempt_task
                        # Evaluation also receives infrastructure failures as
                        # episode metadata. Route retryable full-rollout failures
                        # through the same bounded retry/cleanup path as exceptions.
                        failure = (episode.metadata or {}).get("infrastructure_failure")
                        if (
                            is_validation
                            and isinstance(failure, dict)
                            and failure.get("retryable") is True
                            and failure.get("retry_scope") in {"full_rollout", "full"}
                        ):
                            raise RolloutInfrastructureError(
                                str(failure.get("reason") or "rollout_infrastructure_failure"),
                                str(failure.get("message") or failure.get("error") or "episode infrastructure failure"),
                                stage=str(failure.get("stage") or "agentflow"),
                                retryable=True, retry_scope="full_rollout",
                                diagnostics={**dict(failure.get("diagnostics") or {}),
                                             "episode_failure": dict(failure)},
                            )
                        episode.id = uid
                        episode.task = task_for_episode
                        episode.metrics.update(admission_timings)
                        if attempt_failures:
                            episode.metadata = dict(episode.metadata or {})
                            episode.metadata["infrastructure_attempt_history"] = list(attempt_failures)

                        reward_strs = []
                        for traj in episode.trajectories:
                            reward = "N/A"
                            if traj.reward is not None:
                                reward = _format_rollout_reward_for_log(
                                    traj.reward,
                                )
                            elif len(traj.steps) > 0:
                                reward = _format_rollout_reward_for_log(
                                    traj.steps[-1].reward,
                                )
                            reward_strs.append(f"{traj.name}: {reward}")

                        timing_str = _format_timing_breakdown(episode.metrics)
                        colorful_print(
                            f"[{uid}] Rollout completed. Rewards: [{', '.join(reward_strs)}]{timing_str}, Termination: {episode.termination_reason}",
                            fg="green" if episode.is_correct else "yellow",
                        )

                        return task_id, rollout_idx, result_idx, episode

                    except Exception as e:
                        detail = str(e)
                        summary = detail if len(detail) <= 2000 else detail[:900] + " ... " + detail[-1095:]
                        attempt_failures.append({
                            "attempt": retry_attempt,
                            "error_type": type(e).__name__,
                            "error": summary,
                            "reason": getattr(e, "reason", None),
                            "stage": getattr(e, "stage", "rollout"),
                            "diagnostics": getattr(e, "diagnostics", None),
                            "cleanup_errors": getattr(e, "cleanup_errors", None),
                        })
                        logger.error(
                            "[%s] Attempt %d/%d failed: %s (type=%s)",
                            uid,
                            retry_attempt,
                            self.retry_limit,
                            summary,
                            type(e).__name__,
                        )
                        classified = (
                            e
                            if isinstance(e, RolloutInfrastructureError)
                            else None
                        )
                        if (
                            retry_attempt < self.retry_limit
                            and not getattr(e, "cleanup_errors", None)
                            and (classified is None or classified.retryable)
                            and not (
                                denovo_background and not rollout_slot_held
                            )
                        ):
                            self._update_lifecycle_metric(
                                "full_trajectory_retries"
                            )
                            next_stage_after_cleanup = "retry_backoff"
                            continue
                        if classified is not None:
                            if classified.reason == "trajectory_timeout":
                                self._update_lifecycle_metric(
                                    "trajectory_timeouts"
                                )
                            elif classified.reason == "agent_timeout":
                                self._update_lifecycle_metric("agent_timeouts")
                        if self.raise_on_error:
                            raise
                        reason = (
                            classified.reason
                            if classified is not None
                            else (
                                "gateway_unavailable"
                                if _is_gateway_transport_failure(e)
                                else (
                                    "sandbox_unavailable"
                                    if _is_sandbox_transport_failure(e)
                                    else (
                                        "trace_unavailable"
                                        if isinstance(e, EnrichMismatchError)
                                        else "rollout_retry_exhausted"
                                    )
                                )
                            )
                        )
                        failure = {
                            "schema_version": 1,
                            "reason": reason,
                            "error_type": type(e).__name__,
                            "error": str(e)[:2000],
                            "attempts": retry_attempt,
                        }
                        if getattr(e, "cleanup_errors", None):
                            failure["cleanup_errors"] = e.cleanup_errors
                        metadata: dict[str, Any] = {
                            "error": {"message": str(e)},
                            "infrastructure_attempt_history": list(attempt_failures),
                        }
                        if failure is not None:
                            failure["attempt_history"] = list(attempt_failures)
                            if classified is not None:
                                failure.update(
                                    {
                                        "stage": classified.stage,
                                        "retry_scope": classified.retry_scope,
                                        "safe_for_final_recovery": (
                                            classified.safe_for_final_recovery
                                        ),
                                    }
                                )
                                if classified.diagnostics is not None:
                                    failure["diagnostics"] = classified.diagnostics
                            metadata["infrastructure_failure"] = failure
                        _attach_denovo_background_failure_lifecycle(
                            metadata,
                            task_for_episode,
                            reason=reason,
                            stage=(
                                classified.stage
                                if classified is not None
                                else "rollout"
                            ),
                        )
                        return (
                            task_id,
                            rollout_idx,
                            result_idx,
                            Episode(
                                id=uid,
                                task=task_for_episode,
                                is_correct=False,
                                termination_reason=TerminationReason.ERROR,
                                metadata=metadata,
                                metrics=dict(admission_timings),
                            ),
                        )
                    finally:
                        # Fully-async generation calls this method directly,
                        # so it never reaches execute_tasks()' batch cleanup.
                        # Successful traces are already copied into the Episode
                        # before this cleanup starts.
                        self._update_rollout_progress(
                            uid,
                            stage="gateway_cleanup",
                        )
                        try:
                            if not gateway_cleaned_early:
                                await self._cleanup_gateway_session(attempt_uid)
                        except GatewaySessionCleanupError as cleanup_error:
                            # Trace collection and Episode construction are the
                            # semantic rollout result.  Physical gateway
                            # reclamation is handled by the gateway's
                            # tombstone-first background reaper and must not
                            # replace that result with a fake generation EOF.
                            self._gateway_cleanup_deferred += 1
                            logger.error(
                                "[%s] gateway cleanup deferred after rollout "
                                "completion/retry: %s",
                                attempt_uid,
                                cleanup_error,
                            )
                            if episode is not None and isinstance(
                                episode.metadata, dict
                            ):
                                episode.metadata["gateway_session_cleanup"] = {
                                    "schema_version": 1,
                                    "status": "deferred",
                                    "error": str(cleanup_error)[:2000],
                                }
                        if next_stage_after_cleanup is not None:
                            self._update_rollout_progress(
                                uid,
                                stage=next_stage_after_cleanup,
                            )

                raise RuntimeError(f"[{uid}] Exhausted all retries")
            finally:
                if rollout_slot_held:
                    self._semaphore.release()
                    rollout_slot_held = False
        finally:
            if not shadow_ownership_transferred:
                await self._release_denovo_shadow_budget(
                    shadow_resource_request,
                )
            self._remove_rollout_progress(uid)

    async def _cleanup_gateway_session(self, uid: str) -> None:
        """Delete one session reliably without swallowing task cancellation."""

        consume_tombstone = getattr(
            self.gateway,
            "consume_bulk_tombstone",
            None,
        )
        if callable(consume_tombstone) and consume_tombstone(uid):
            return
        if not getattr(self.gateway, "available", True):
            logger.info(
                "[%s] skipping per-session cleanup while gateway supervisor "
                "owns sidecar recovery",
                uid,
            )
            return

        async def confirm_session_absent(*, wait: bool) -> bool:
            checker = getattr(self.gateway, "asession_exists", None)
            if not callable(checker):
                return False

            loop = asyncio.get_running_loop()
            deadline = loop.time() + (
                _SESSION_CLEANUP_CONFIRM_TIMEOUT_S if wait else 0.0
            )
            while True:
                try:
                    exists = await asyncio.wait_for(
                        checker(uid),
                        timeout=_SESSION_CLEANUP_STATUS_TIMEOUT_S,
                    )
                    if not exists:
                        return True
                except Exception as exc:  # noqa: BLE001 - this is a best-effort confirmation
                    logger.debug(
                        "[%s] gateway session cleanup status probe failed: %s",
                        uid,
                        exc,
                    )

                if not wait or loop.time() >= deadline:
                    return False
                await asyncio.sleep(
                    min(
                        _SESSION_CLEANUP_CONFIRM_POLL_S,
                        max(0.0, deadline - loop.time()),
                    )
                )

        async def delete_with_retries() -> None:
            last_error: Exception | None = None
            for attempt in range(1, _SESSION_CLEANUP_ATTEMPTS + 1):
                request_task = asyncio.create_task(self.gateway.adelete_session(uid))
                try:
                    done, _ = await asyncio.wait(
                        {request_task},
                        timeout=_SESSION_CLEANUP_TIMEOUT_S,
                    )
                    if not done:
                        # Cancelling an HTTP client request does not cancel the
                        # server-owned deletion. Confirm the remote state before
                        # deciding this attempt failed. The gateway coalesces a
                        # later idempotent retry with any deletion still active.
                        request_task.cancel()
                        await asyncio.gather(request_task, return_exceptions=True)
                        if await confirm_session_absent(wait=True):
                            logger.info(
                                "[%s] gateway session deletion was confirmed after "
                                "the %gs response deadline",
                                uid,
                                _SESSION_CLEANUP_TIMEOUT_S,
                            )
                            return
                        last_error = TimeoutError(
                            f"gateway session remained present after "
                            f"{_SESSION_CLEANUP_TIMEOUT_S:g}s DELETE deadline and "
                            f"{_SESSION_CLEANUP_CONFIRM_TIMEOUT_S:g}s confirmation grace"
                        )
                        logger.debug(
                            "[%s] gateway session cleanup attempt %d/%d timed out; "
                            "session still exists",
                            uid,
                            attempt,
                            _SESSION_CLEANUP_ATTEMPTS,
                        )
                        if attempt < _SESSION_CLEANUP_ATTEMPTS:
                            await asyncio.sleep(_SESSION_CLEANUP_RETRY_DELAY_S * attempt)
                            continue
                        break
                    await request_task
                    return
                except Exception as exc:  # noqa: BLE001 - transport errors vary by gateway client
                    last_error = exc
                    if await confirm_session_absent(wait=False):
                        logger.info(
                            "[%s] gateway session deletion was confirmed after "
                            "client error: %s",
                            uid,
                            exc,
                        )
                        return
                    logger.debug(
                        "[%s] gateway session cleanup attempt %d/%d failed: %s",
                        uid,
                        attempt,
                        _SESSION_CLEANUP_ATTEMPTS,
                        exc,
                    )
                    if attempt < _SESSION_CLEANUP_ATTEMPTS:
                        await asyncio.sleep(_SESSION_CLEANUP_RETRY_DELAY_S * attempt)
            raise GatewaySessionCleanupError(
                f"[{uid}] gateway session cleanup failed after "
                f"{_SESSION_CLEANUP_ATTEMPTS} attempts"
            ) from last_error

        cleanup_task = asyncio.create_task(delete_with_retries())
        cancellation: asyncio.CancelledError | None = None
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError as exc:
                # A second shutdown signal must not leave a background delete
                # racing with a retry that reuses this session ID.
                cancellation = exc

        cleanup_error = cleanup_task.exception()
        if cleanup_error is not None:
            if cancellation is not None:
                logger.error(
                    "[%s] gateway cleanup also failed while cancellation was pending: %r",
                    uid,
                    cleanup_error,
                )
                raise cancellation
            raise cleanup_error
        if cancellation is not None:
            raise cancellation

    async def _run_denovo_agent_phase(
        self,
        task_obj: Task,
        uid: str,
        *,
        is_validation: bool,
        timings: dict[str, float],
        deadline: float | None = None,
    ) -> tuple[Episode, list[TraceRecord], TaskContext]:
        ctx: TaskContext | None = None
        try:
            flow = self._run_flow_only(
                task_obj=task_obj,
                uid=uid,
                is_validation=is_validation,
                _timings=timings,
                deadline=deadline,
            )
            if deadline is None:
                raw_episode, ctx = await flow
            else:
                # The stop grace belongs to generation only. Downloading and
                # decoding an already completed trajectory has its own HTTP
                # limits and must not discard a safe deadline episode.
                started = time.monotonic()
                worker = asyncio.create_task(flow)
                try:
                    done, _ = await asyncio.wait(
                        {worker}, timeout=max(0.0, deadline - started)
                        + _DENOVO_DEADLINE_STOP_GRACE_SECONDS,
                    )
                    if not done:
                        now = time.monotonic()
                        progress = self._rollout_progress_snapshot(
                            uid, task_id=task_obj.id, rollout_index=0,
                            scheduler_started_at=started, now=now,
                        )
                        diagnostics = {
                            "deadline_stage": progress["stage"], "stop_confirmed": False,
                            "remaining_budget_seconds": max(0.0, deadline - now),
                            "deadline_overrun_seconds": max(0.0, now - deadline),
                            "stop_grace_seconds": _DENOVO_DEADLINE_STOP_GRACE_SECONDS,
                            "rollout_progress": progress,
                        }
                        from rllm.utils.diagnostic_events import emit_diagnostic
                        emit_diagnostic("denovo_deadline_stop_unconfirmed", uid=uid, **diagnostics)
                        worker.cancel()
                        results = await asyncio.gather(worker, return_exceptions=True)
                        failure = RolloutInfrastructureError(
                            "trajectory_timeout", "DeNovo generation did not stop within its deadline grace",
                            retryable=False, stage="agentflow", retry_scope="none",
                            diagnostics=diagnostics,
                        )
                        if getattr(results[0], "cleanup_errors", None):
                            failure.cleanup_errors = results[0].cleanup_errors
                        raise failure
                    # Preserve TimeoutError raised by setup/the flow itself.
                    raw_episode, ctx = worker.result()
                finally:
                    if not worker.done():
                        worker.cancel()
                        await asyncio.gather(worker, return_exceptions=True)
                    # Cancellation can win just as setup/flow returns its
                    # context. Adopt that result for the outer error cleanup.
                    if ctx is None and worker.done() and not worker.cancelled() and worker.exception() is None:
                        _completed_episode, ctx = worker.result()
            self._update_rollout_progress(uid, stage="trace_fetch")
            traces = await self._fetch_traces(uid, timings)
            return raw_episode, traces, ctx
        except BaseException as primary_error:
            # _run_flow_only owns teardown only until it returns.  Trace
            # collection failures after that boundary still own both
            # sandboxes here.
            if ctx is not None:
                try:
                    if not await self._run_context_teardown(ctx, uid):
                        raise RuntimeError("context teardown remains unconfirmed")
                except asyncio.CancelledError:
                    raise primary_error
                except BaseException as cleanup_error:
                    from rllm.utils.infrastructure_diagnostics import infrastructure_exception_chain
                    primary_error.cleanup_errors = [infrastructure_exception_chain(cleanup_error)]
            raise

    async def _run_single_denovo_background(
        self,
        task_obj: Task,
        uid: str,
        *,
        is_validation: bool,
        agent_phase_timeout: float | None,
        release_model_stage: Callable[[], Awaitable[None]],
        shadow_resource_request: _ShadowResourceRequest | None,
    ) -> Episode:
        """Run DeNovo generation, then score on primary without a vLLM slot."""

        timings: dict[str, float] = {}
        rollout_started = time.perf_counter()
        ctx: TaskContext | None = None
        shadow_transferred = False
        lease_registered = False
        try:
            raw_episode, traces, ctx = await self._run_denovo_agent_phase(
                task_obj,
                uid,
                is_validation=is_validation,
                timings=timings,
                deadline=(time.monotonic() + agent_phase_timeout if agent_phase_timeout is not None else None),
            )
            shadow_runtime = getattr(ctx, "shadow_runtime", None)
            use_shadow = getattr(self, "denovo_shadow_finalize_enable", True)
            if use_shadow and (shadow_runtime is None or not getattr(
                shadow_runtime,
                "milestone_only",
                False,
            )):
                raise RolloutInfrastructureError(
                    "denovo_shadow_handoff_unavailable",
                    "DeNovo background rollout did not create a milestone-only shadow",
                    retryable=False,
                    stage="shadow_handoff",
                    retry_scope="none",
                )

            def seal_agent_input():
                if not use_shadow:
                    return None
                try:
                    return shadow_runtime.seal_milestone_input()
                except Exception as exc:
                    raise RolloutInfrastructureError(
                        "denovo_shadow_handoff_snapshot_failed",
                        f"DeNovo shadow handoff could not seal the action journal and final repository snapshot: {exc}",
                        retryable=False, stage="shadow_handoff", retry_scope="none",
                    ) from exc

            # Deadline snapshots must follow process shutdown: daemonized
            # descendants must not race with the final snapshot or verifier.
            handoff_snapshot = None
            await release_model_stage()
            model_released_at = time.monotonic()

            # The agent worker has returned and its Gateway session is already
            # tombstoned. Quiesce daemonized descendants before the evaluator
            # uploads hidden tests into the retained primary sandbox.
            process_audit: dict[str, Any] = {
                "schema_version": 1,
                "status": "not_configured",
            }
            if callable(ctx.primary_process_audit):
                try:
                    audit_future = self._submit_lifecycle_call(
                        ctx.primary_process_audit,
                        pending_key="evaluator_pending",
                    )
                    raw_audit = await self._await_lifecycle_future(audit_future)
                    if isinstance(raw_audit, dict):
                        process_audit = dict(raw_audit)
                except Exception as exc:
                    raise RolloutInfrastructureError(
                        "primary_process_quiescence_failed",
                        f"DeNovo primary process audit failed: {exc}",
                        retryable=False,
                        stage="primary_process_audit",
                        retry_scope="none",
                    ) from exc

            if (
                process_audit.get("status") != "clean"
                or process_audit.get("remaining_count") != 0
            ):
                raise RolloutInfrastructureError(
                    "primary_process_quiescence_failed",
                    "Cannot score DeNovo rollout without confirmed process shutdown",
                    retryable=False, stage="primary_process_audit", retry_scope="none",
                    diagnostics={"process_audit": process_audit},
                )

            cutoff = raw_episode.metadata.get("denovo_deadline")
            if cutoff is not None:
                if (cutoff.get("worker_stop_confirmed") is not True
                        or not any(t.steps for t in raw_episode.trajectories)
                        or process_audit.get("status") != "clean"
                        or process_audit.get("remaining_count") != 0):
                    raise RolloutInfrastructureError("denovo_deadline_stop_unconfirmed", "Cannot score a deadline without completed steps and confirmed process shutdown", retryable=False, stage="deadline_stop", retry_scope="none", diagnostics={"deadline": cutoff, "process_audit": process_audit})
                cutoff.update(model_stop_confirmed=True, tool_stop_confirmed=True, stop_confirmed=True)
            handoff_snapshot = seal_agent_input()
            self._update_rollout_progress(uid, stage="primary_verifier")
            primary_started_at = time.monotonic()
            enriched = await self._finish_episode(
                raw_episode=raw_episode,
                traces=traces,
                uid=uid,
                task_obj=task_obj,
                ctx=ctx,
                is_validation=is_validation,
                _timings=timings,
                defer_shadow_finalize=True,
                defer_milestone_annotation=use_shadow,
            )
            # Outer verifier timeouts can kill the interpreter before its
            # command supervisor publishes a receipt. Confirm quiescence
            # again before accepting either a score or a timeout outcome.
            post_audit = await self._await_lifecycle_future(self._submit_lifecycle_call(
                ctx.primary_process_audit, pending_key="evaluator_pending",
            ))
            if not isinstance(post_audit, dict) or post_audit.get("status") != "clean" or post_audit.get("remaining_count") != 0:
                raise RolloutInfrastructureError(
                    "verifier_stop_unconfirmed", "Primary verifier descendants did not stop",
                    retryable=False, stage="primary_verifier_cleanup", retry_scope="none",
                    diagnostics={"process_audit": post_audit},
                )
            enriched.metadata["primary_verifier_process_audit"] = post_audit
            primary_finished_at = time.monotonic()
            if not use_shadow:
                verifier_failed = isinstance(enriched.metadata.get("infrastructure_failure"), dict)
                if cutoff is not None:
                    if verifier_failed or enriched.metrics.get("verifier_status") == "timeout":
                        raise RolloutInfrastructureError(
                            "denovo_deadline_verifier_failed", "Primary verifier did not complete after deadline",
                            retryable=False, stage="primary_verifier", retry_scope="none",
                        )
                    enriched.metadata["denovo_deadline"] = {**cutoff, "verifier_completed": True}
                enriched.metadata[_DENOVO_BACKGROUND_METADATA_KEY] = {
                    "schema_version": 1, "status": "primary_only_completed",
                    "shadow_lease_registered": False, "shadow_enabled": False,
                    "primary_verifier_status": "failed" if verifier_failed else "completed",
                    "primary_process_audit": process_audit,
                    "model_session_released_at_monotonic": model_released_at,
                    "primary_verifier_started_at_monotonic": primary_started_at,
                    "primary_verifier_finished_at_monotonic": primary_finished_at,
                    "shadow_resources": {"count": 0, "total_cpus": 0, "total_memory_mb": 0},
                }
                self._update_rollout_progress(uid, stage="primary_teardown")
                teardown_started = time.perf_counter()
                if not await self._run_context_teardown(ctx, uid):
                    raise RolloutInfrastructureError("sandbox_cleanup_unconfirmed", "Primary teardown did not complete", retryable=False, stage="teardown", retry_scope="none")
                timings["time/primary_teardown_s"] = time.perf_counter() - teardown_started
                timings["time/rollout_s"] = time.perf_counter() - rollout_started
                enriched.metrics.update(timings)
                return enriched
            shadow_runtime.seal_milestone_input(
                primary_verifier_finished_at_monotonic=primary_finished_at,
            )

            if shadow_resource_request is None:
                raise RuntimeError("DeNovo shadow resource reservation is missing")
            close_shadow = ctx.transfer_shadow_teardown()
            shadow_transferred = True
            lease = _DeferredShadowLease(
                uid=uid,
                runtime=shadow_runtime,
                raw_episode=raw_episode,
                enriched_episode=enriched,
                close_shadow=close_shadow,
                resource_request=shadow_resource_request,
                primary_verifier_finished_at_monotonic=primary_finished_at,
            )
            with self._deferred_shadow_registry_lock:
                if uid in self._deferred_shadow_leases:
                    raise RuntimeError(f"duplicate deferred shadow lease: {uid}")
                self._deferred_shadow_leases[uid] = lease
            lease_registered = True

            lifecycle = dict(
                enriched.metadata.get(_DENOVO_BACKGROUND_METADATA_KEY) or {}
            )
            infrastructure_failure = enriched.metadata.get(
                "infrastructure_failure"
            )
            if isinstance(infrastructure_failure, dict):
                primary_verifier_status = str(
                    infrastructure_failure.get("reason")
                    or "infrastructure_failure"
                )
            elif enriched.metrics.get("verifier_status") == "timeout":
                primary_verifier_status = "timeout"
            else:
                primary_verifier_status = "completed"
            if cutoff is not None:
                if primary_verifier_status != "completed":
                    raise RolloutInfrastructureError("denovo_deadline_verifier_failed", "Primary verifier did not complete after deadline", retryable=False, stage="primary_verifier", retry_scope="none", diagnostics={"deadline": cutoff, "verifier_status": primary_verifier_status})
                enriched.metadata["denovo_deadline"] = {**cutoff, "verifier_completed": True, "verifier_status": primary_verifier_status}
            lifecycle.update(
                {
                    "schema_version": 1,
                    "status": "primary_verifier_completed",
                    "lease_id": uid,
                    "model_session_released_at_monotonic": model_released_at,
                    "primary_verifier_started_at_monotonic": primary_started_at,
                    "primary_verifier_finished_at_monotonic": primary_finished_at,
                    "primary_verifier_duration_s": max(
                        0.0,
                        primary_finished_at - primary_started_at,
                    ),
                    "primary_verifier_status": primary_verifier_status,
                    "primary_process_audit": process_audit,
                    "handoff_snapshot": handoff_snapshot,
                    "shadow_resources": {
                        "count": shadow_resource_request.count,
                        "cpus": (
                            shadow_resource_request.cpus
                            // shadow_resource_request.count
                        ),
                        "memory_mb": (
                            shadow_resource_request.memory_mb
                            // shadow_resource_request.count
                        ),
                        "total_cpus": shadow_resource_request.cpus,
                        "total_memory_mb": shadow_resource_request.memory_mb,
                    },
                    "shadow_resource_budget": (
                        self._denovo_shadow_budget.snapshot()
                        if self._denovo_shadow_budget is not None
                        else {}
                    ),
                }
            )
            enriched.metadata[_DENOVO_BACKGROUND_METADATA_KEY] = lifecycle

            self._update_rollout_progress(uid, stage="primary_teardown")
            teardown_started = time.perf_counter()
            if not await self._run_context_teardown(ctx, uid):
                raise RolloutInfrastructureError("sandbox_cleanup_unconfirmed", "Primary teardown did not complete", retryable=False, stage="teardown", retry_scope="none")
            timings["time/primary_teardown_s"] = (
                time.perf_counter() - teardown_started
            )
            timings["time/rollout_s"] = time.perf_counter() - rollout_started
            enriched.metrics.update(timings)
            return enriched
        except BaseException as primary_error:
            if shadow_resource_request is not None and getattr(primary_error, "cleanup_errors", None):
                shadow_resource_request.cleanup_unconfirmed = True
            if ctx is not None:
                cleanup_errors = []
                if shadow_transferred:
                    try:
                        ctx.run_shadow_teardown()
                    except BaseException as exc:
                        cleanup_errors.append(exc)
                try:
                    if not await self._run_context_teardown(ctx, uid):
                        cleanup_errors.append(RuntimeError("context teardown remains unconfirmed"))
                except BaseException as exc:
                    cleanup_errors.append(exc)
                if cleanup_errors:
                    from rllm.utils.infrastructure_diagnostics import infrastructure_exception_chain
                    primary_error.cleanup_errors = [infrastructure_exception_chain(exc) for exc in cleanup_errors]
                    if shadow_resource_request is not None:
                        shadow_resource_request.cleanup_unconfirmed = True
                elif lease_registered:
                    with self._deferred_shadow_registry_lock:
                        self._deferred_shadow_leases.pop(uid, None)
            raise

    async def _fetch_traces(self, uid: str, timings: dict[str, float]) -> list[TraceRecord]:
        """Retry only the read phase; never regenerate a completed agent here."""
        from rllm.utils.diagnostic_events import emit_diagnostic

        started = time.perf_counter()
        attempts = 0
        errors = []
        recovered = False
        try:
            async with asyncio.timeout(_TRACE_FETCH_TIMEOUT_SECONDS):
                for attempts in range(1, 4):
                    try:
                        traces = await self.gateway.aget_traces(uid)
                        timings.update(getattr(traces, "metrics", {}))
                        recovered = True
                        return traces
                    except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                        retryable = not isinstance(exc, httpx.HTTPStatusError) or exc.response.status_code in {408, 429} or exc.response.status_code >= 500
                        errors.append({"type": type(exc).__name__, "message": str(exc)[:500]})
                        if not retryable or attempts == 3:
                            raise
                        logger.warning("[%s] trace read failed; retrying the same session (%d/3): %s", uid, attempts, type(exc).__name__)
                        await asyncio.sleep(float(attempts))
        except asyncio.CancelledError as exc:
            timings.update(getattr(exc, "trace_fetch_metrics", {}))
            raise
        except Exception as exc:
            current = exc
            seen = set()
            while current is not None and id(current) not in seen:
                seen.add(id(current))
                timings.update(getattr(current, "trace_fetch_metrics", {}))
                current = current.__cause__ or current.__context__
            terminal_error = {"type": type(exc).__name__, "message": str(exc)[:500]}
            if not errors or errors[-1] != terminal_error:
                errors.append(terminal_error)
            raise RolloutInfrastructureError(
                "trace_fetch_failed", f"Trace recovery failed after {attempts} attempt(s): {type(exc).__name__}",
                retryable=False, stage="trace_fetch", retry_scope="none",
                diagnostics={"session_id": uid, "worker_stop_confirmed": True, "attempts": attempts, "errors": errors,
                             "elapsed_seconds": time.perf_counter() - started,
                             "exception_type": type(exc).__name__},
            ) from exc
        finally:
            timings["time/traces_s"] = time.perf_counter() - started
            timings["trace_fetch/attempts"] = attempts
            emit_diagnostic("trace_fetch", uid=uid, recovered=recovered, attempts=attempts,
                            elapsed_seconds=timings["time/traces_s"], errors=errors,
                            metrics={k: v for k, v in timings.items() if k.startswith(("trace_fetch/", "time/trace"))})

    async def _run_single(self, task_obj: Task, uid: str, is_validation: bool = False) -> Episode:
        """Run one full per-task pipeline: flow → fetch traces → enrich → evaluate.

        Records ``time/<phase>_s`` for setup/agentflow/traces/evaluator/
        teardown/rollout into ``episode.metrics``, plus
        ``time/agentflow_llm_wall_s`` (interval-union),
        ``time/agentflow_llm_sum_s`` (naive sum), and ``n_turns``.
        """
        timings: dict[str, float] = {}
        rollout_start = time.perf_counter()
        result_holder: dict[str, Episode] = {}
        primary_error = None

        raw_episode, ctx = await self._run_flow_only(
            task_obj=task_obj,
            uid=uid,
            is_validation=is_validation,
            _timings=timings,
        )
        try:
            self._update_rollout_progress(uid, stage="trace_fetch")
            traces = await self._fetch_traces(uid, timings)

            enriched = await self._finish_episode(
                raw_episode=raw_episode,
                traces=traces,
                uid=uid,
                task_obj=task_obj,
                ctx=ctx,
                is_validation=is_validation,
                _timings=timings,
            )
            enriched.metrics.update(timings)
            result_holder["episode"] = enriched
            return enriched
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            # Offload Modal's blocking terminate()/detach() to the executor.
            self._update_rollout_progress(uid, stage="teardown")
            t = time.perf_counter()
            try:
                if not await self._run_context_teardown(ctx, uid):
                    raise RolloutInfrastructureError(
                        "sandbox_cleanup_unconfirmed", "Task teardown remains unconfirmed",
                        retryable=False, stage="teardown", retry_scope="none",
                    )
            except Exception as cleanup_error:
                if primary_error is None:
                    raise
                from rllm.utils.infrastructure_diagnostics import infrastructure_exception_chain
                primary_error.cleanup_errors = [infrastructure_exception_chain(cleanup_error)]
                logger.error("[%s] task teardown failed; preserving original failure: %s", uid, cleanup_error)
            timings["time/teardown_s"] = time.perf_counter() - t
            timings["time/rollout_s"] = time.perf_counter() - rollout_start
            ep = result_holder.get("episode")
            if ep is not None:
                ep.metrics.update(timings)

    async def _run_flow_only(
        self,
        task_obj: Task,
        uid: str,
        is_validation: bool = False,
        _timings: dict[str, float] | None = None,
        deadline: float | None = None,
    ) -> tuple[Episode, TaskContext]:
        """Run hook setup + the agent flow. Returns ``(raw_episode, ctx)``.

        On flow failure, tears down ``ctx`` and re-raises. On success, the
        caller owns ``ctx.run_teardown()``. Records ``time/setup_s`` and
        ``time/agentflow_s`` when ``_timings`` is provided.
        """
        if _timings is None:
            _timings = {}

        ctx: TaskContext | None = None
        startup_slot_held = False
        startup_active_counted = False
        admission = getattr(self, "evaluation_admission", None)
        startup_token = admission.token() if admission is not None else None
        shared_startup_held = False
        self._update_rollout_progress(
            uid,
            stage="startup_queue",
            turn_index=None,
        )
        queue_started = time.perf_counter()
        self._startup_pending += 1
        try:
            try:
                await self._startup_semaphore.acquire()
                startup_slot_held = True
                if admission is not None:
                    await admission.acquire("startup", startup_token)
                    shared_startup_held = True
            finally:
                self._startup_pending = max(0, self._startup_pending - 1)
            self._startup_active += 1
            startup_active_counted = True
            _timings["time/startup_queue_s"] = (
                time.perf_counter() - queue_started
            )

            # Offload hook setup (blocking Modal/docker I/O) to the lifecycle
            # executor.  The startup permit prevents a full 640-rollout wave
            # from submitting every sandbox setup and session start at once.
            self._update_rollout_progress(
                uid,
                stage="setup",
                turn_index=None,
            )
            t = time.perf_counter()
            setup_future = self._submit_lifecycle_call(
                self.hooks.setup,
                task_obj,
                self.agent_flow,
                uid,
                pending_key="setup_pending",
            )
            try:
                # The native future is not cancelled with this coroutine. If
                # cancellation wins, a late TaskContext still gets one teardown.
                setup_wait = self._await_lifecycle_future(setup_future)
                if deadline is None:
                    ctx = await setup_wait
                else:
                    ctx = await asyncio.wait_for(setup_wait, max(0.0, deadline - time.monotonic()))
            except (asyncio.CancelledError, TimeoutError) as exc:
                setup_future.add_done_callback(
                    lambda done: self._schedule_late_setup_teardown(done, uid)
                )
                if isinstance(exc, TimeoutError) and deadline is not None and time.monotonic() >= deadline:
                    raise RolloutInfrastructureError(
                        "trajectory_timeout", "DeNovo generation deadline expired before setup completed",
                        retryable=False, stage="setup", retry_scope="none",
                        diagnostics={"deadline_stage": "setup", "stop_confirmed": False},
                    ) from exc
                raise
            _timings["time/setup_s"] = time.perf_counter() - t

            if getattr(self.agent_flow, "needs_env", False) and ctx.env is None:
                raise RuntimeError(
                    f"{type(self.agent_flow).__name__} needs a sandbox but hooks {type(self.hooks).__name__} provisioned none — pass hooks=SandboxTaskHooks(...) or run via AgentTrainer / run_dataset."
                )

            # Attach resolved sampling params to the session so the gateway
            # enforces them on every LLM call. Adaptive routing also requires
            # explicit creation so the group/slot hint is available before the
            # first model request.
            session_sampling_params = (self.val_sampling_params if is_validation else self.train_sampling_params) or None
            routing_config = getattr(self.gateway, "routing_config", None)
            adaptive_routing = (
                getattr(routing_config, "mode", "sticky_least_loaded")
                == "group_striped_adaptive"
            )
            if session_sampling_params or adaptive_routing:
                self._update_rollout_progress(
                    uid,
                    stage="session_start",
                    turn_index=None,
                )
                create_session = getattr(
                    self.gateway,
                    "acreate_session_with_timings",
                    None,
                )
                create_kwargs: dict[str, Any] = {
                    "is_validation": is_validation,
                    "sampling_params": session_sampling_params,
                }
                if adaptive_routing:
                    group_id, slot = split_rollout_session_id(uid)
                    create_kwargs["metadata"] = {
                        "routing": {
                            "schema_version": 1,
                            "group_id": group_id,
                            "slot": slot,
                            "group_size": self.rollout_group_size,
                        }
                    }
                if callable(create_session):
                    _created_session, session_timings = await create_session(
                        uid,
                        **create_kwargs,
                    )
                    if isinstance(session_timings, dict):
                        for key in (
                            "time/session_admission_s",
                            "time/session_create_s",
                        ):
                            value = session_timings.get(key)
                            if isinstance(value, int | float) and math.isfinite(
                                float(value)
                            ):
                                _timings[key] = float(value)
                else:
                    session_started = time.perf_counter()
                    await self.gateway.acreate_session(
                        uid,
                        **create_kwargs,
                    )
                    _timings["time/session_admission_create_s"] = (
                        time.perf_counter() - session_started
                    )

            # Only startup is bounded.  Release the permit before the long
            # agentflow so up to n_parallel_tasks rollouts can still run and
            # dynamic sampling can keep collecting waves without a new stop
            # condition.
            self._startup_active = max(0, self._startup_active - 1)
            startup_active_counted = False
            self._startup_semaphore.release()
            startup_slot_held = False
            if shared_startup_held:
                await asyncio.shield(admission.release("startup", startup_token))
                shared_startup_held = False

            # Flows whose LLM client runs *inside* the env (CLI harnesses)
            # need the publicly-reachable URL — rewritten for in-container
            # networking on the backend that actually provisioned this task's
            # sandbox. Host-side flows keep the local gateway URL so they
            # never depend on a tunnel hostname.
            llm_inside_env = getattr(self.agent_flow, "llm_inside_env", False)
            session_url = self.gateway.get_session_url(uid, public=llm_inside_env)
            if llm_inside_env and ctx.env is not None:
                session_url = container_reachable_url(session_url, ctx.env_backend)

            config = AgentConfig(
                base_url=session_url,
                model=self.model,
                session_uid=uid,
                is_validation=is_validation,
                sampling_params=session_sampling_params or {},
            )
            logger.debug("[%s] Starting agent flow at %s", uid, session_url)
            self._update_rollout_progress(
                uid,
                stage="agentflow",
                turn_index=None,
            )
            t = time.perf_counter()
            if deadline is not None and time.monotonic() >= deadline:
                raise RolloutInfrastructureError("trajectory_timeout", "DeNovo deadline expired during setup/session admission", retryable=False, stage="setup", retry_scope="none")
            shadow_runtime = getattr(ctx, "shadow_runtime", None)

            def report_progress(*, stage: str, turn_index: int | None = None) -> None:
                self._update_rollout_progress(
                    uid,
                    stage=stage,
                    turn_index=turn_index,
                )

            try:
                # Only the explicit DeNovo generation budget adds a deadline.
                # Other SWE flows retain their command/verifier limits; legacy
                # task agent.timeout_sec metadata is still ignored.
                episode = await run_agent_flow(
                    self.agent_flow,
                    task_obj,
                    config,
                    executor=self.executor,
                    env=ctx.env if self._flow_accepts_env else None,
                    shadow_runtime=(
                        shadow_runtime
                        if self._flow_accepts_shadow_runtime
                        else None
                    ),
                    progress_reporter=report_progress,
                    deadline=deadline,
                    cancelled_worker_observer=(
                        self._observe_cancelled_agentflow_worker
                    ),
                )
            except asyncio.CancelledError:
                self._update_lifecycle_metric("cancelled_agentflow_children")
                raise
            if (deadline is not None and time.monotonic() >= deadline
                    and not episode.metadata.get("denovo_deadline")
                    and not episode.metadata.get("infrastructure_failure")):
                raise RolloutInfrastructureError("trajectory_timeout", "Agent did not return a safe partial episode at the DeNovo deadline", retryable=False, stage="agentflow", retry_scope="none")
            _timings["time/agentflow_s"] = time.perf_counter() - t
            logger.debug("[%s] Agent flow completed, %d trajectories", uid, len(episode.trajectories))
            return episode, ctx
        except BaseException as primary_error:
            # Tear down on failure; success path defers teardown to the caller.
            if ctx is not None:
                self._update_rollout_progress(uid, stage="teardown")
                try:
                    if not await self._run_context_teardown(ctx, uid):
                        raise RuntimeError("context teardown remains unconfirmed")
                except asyncio.CancelledError:
                    raise primary_error
                except BaseException as cleanup_error:
                    from rllm.utils.infrastructure_diagnostics import infrastructure_exception_chain
                    primary_error.cleanup_errors = [infrastructure_exception_chain(cleanup_error)]
                    logger.error("[%s] hook teardown failed; preserving original failure: %s", uid, cleanup_error)
            raise
        finally:
            if shared_startup_held:
                await asyncio.shield(admission.release("startup", startup_token))
            if startup_slot_held:
                if startup_active_counted:
                    self._startup_active = max(0, self._startup_active - 1)
                self._startup_semaphore.release()

    @staticmethod
    def _denovo_authoritative_test_count(task: Task) -> int | None:
        try:
            instance = json.loads(
                (task.task_dir / "tests" / "instance.json").read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None
        passed_ptp = instance.get("passed_ptp") if isinstance(instance, dict) else None
        if not isinstance(passed_ptp, list) or not passed_ptp:
            return None
        return len([value for value in passed_ptp if isinstance(value, str)])

    def _normalize_denovo_primary_eval_output(
        self,
        task: Task,
        output: EvalOutput,
    ) -> EvalOutput:
        """Validate and normalize the primary DeNovo outcome contract."""

        metadata = dict(output.metadata or {})
        if isinstance(metadata.get("infrastructure_failure"), dict):
            return output
        infrastructure_error = metadata.get("infrastructure_error")
        if infrastructure_error:
            execution = metadata.get("verifier_execution")
            execution = execution if isinstance(execution, dict) else {}
            checked_paths = metadata.get("reward_paths_checked")
            checked_paths = (
                list(checked_paths[:32])
                if isinstance(checked_paths, list)
                else []
            )
            metadata["infrastructure_failure"] = {
                "schema_version": 2,
                "reason": "verifier_execution_failed",
                "stage": "verifier",
                "task_id": task.id,
                "verifier_status": "execution_failed",
                "sandbox_health": (
                    "alive"
                    if metadata.get("sandbox_alive") is True
                    else "lost"
                    if metadata.get("sandbox_alive") is False
                    else "unknown"
                ),
                "sandbox_alive": metadata.get("sandbox_alive"),
                "duration_seconds": execution.get("duration_seconds"),
                "checked_paths": checked_paths,
                "exception_type": "denovoswe_verifier_infrastructure_error",
                "error_summary": str(infrastructure_error)[:2000],
                "retryable": False,
            }
            return EvalOutput(
                reward=0.0,
                is_correct=False,
                signals=list(output.signals),
                metadata=metadata,
            )

        timeout = bool(
            metadata.get("verifier_status") == "timeout"
            or (
                isinstance(metadata.get("verifier_execution"), dict)
                and metadata["verifier_execution"].get("timed_out") is True
            )
        )
        total = self._denovo_authoritative_test_count(task)
        reported_total: Any = None
        count_mismatch = False
        if timeout:
            passed = 0
        else:
            passed = metadata.get("passed_count")
            reported_total = metadata.get("total_count")
            if isinstance(reported_total, int) and not isinstance(
                reported_total, bool
            ):
                if total is not None and reported_total != total:
                    count_mismatch = True
                else:
                    total = reported_total
            if (
                count_mismatch
                or isinstance(reported_total, bool)
                or not isinstance(reported_total, int)
                or isinstance(passed, bool)
                or not isinstance(passed, int)
                or total is None
                or total <= 0
                or not 0 <= passed <= total
            ):
                passed = None

        if passed is None or total is None or total <= 0:
            execution = metadata.get("verifier_execution")
            execution = execution if isinstance(execution, dict) else {}
            checked_paths = metadata.get("reward_paths_checked")
            checked_paths = (
                list(checked_paths[:32])
                if isinstance(checked_paths, list)
                else []
            )
            metadata["infrastructure_failure"] = {
                "schema_version": 2,
                "reason": "verifier_reward_contract_invalid",
                "stage": "verifier",
                "task_id": task.id,
                "verifier_status": "reward_contract_invalid",
                "sandbox_health": (
                    "alive"
                    if metadata.get("sandbox_alive") is True
                    else "lost"
                    if metadata.get("sandbox_alive") is False
                    else "unknown"
                ),
                "sandbox_alive": metadata.get("sandbox_alive"),
                "duration_seconds": execution.get("duration_seconds"),
                "checked_paths": checked_paths,
                "exception_type": "denovoswe_outcome_contract",
                "error_summary": (
                    "DeNovo verifier did not provide a valid passed_count/"
                    "total_count contract"
                    + (
                        f" (materialized_total={total}, "
                        f"reported_total={reported_total})"
                        if count_mismatch
                        else ""
                    )
                ),
                "retryable": False,
            }
            return EvalOutput(
                reward=0.0,
                is_correct=False,
                signals=list(output.signals),
                metadata=metadata,
            )

        pass_rate = float(passed / total)
        outcome = {
            "schema_version": 1,
            "passed_count": int(passed),
            "total_count": int(total),
            "pass_rate": pass_rate,
            "outcome_source": "primary_post_agent_verifier",
            "verifier_profile": "denovoswe_official",
        }
        metadata.update(
            {
                "passed_count": int(passed),
                "total_count": int(total),
                "pass_rate": pass_rate,
                "outcome_source": "primary_post_agent_verifier",
                "verifier_outcome": outcome,
            }
        )
        signals = [
            signal
            for signal in output.signals
            if signal.name != "acceptance_pass_rate"
        ]
        signals.append(Signal(name="acceptance_pass_rate", value=pass_rate))
        return EvalOutput(
            reward=pass_rate,
            is_correct=bool(passed == total and output.is_correct),
            signals=signals,
            metadata=metadata,
        )

    async def _finish_episode(
        self,
        raw_episode: Episode,
        traces: list[TraceRecord],
        uid: str,
        task_obj: Task,
        ctx: TaskContext,
        is_validation: bool = False,
        _timings: dict[str, float] | None = None,
        defer_shadow_finalize: bool = False,
        defer_milestone_annotation: bool = False,
    ) -> Episode:
        """Enrich the raw episode with traces, run the evaluator, apply rewards.

        Training rollouts enrich strictly (empty token IDs are infrastructure
        failures — they are required for loss math); validation relaxes
        (non-vLLM upstreams legitimately return no token IDs and evaluators
        read message text). Records ``time/evaluator_s``,
        ``time/agentflow_llm_{sum,wall}_s``, and ``n_turns`` when ``_timings``
        is provided.
        """
        self._update_rollout_progress(uid, stage="enrichment")
        try:
            enriched = enrich_episode_with_traces(
                raw_episode,
                traces,
                uid,
                task_obj.metadata,
                strict=not is_validation,
            )
        except EnrichMismatchError as exc:
            from rllm.utils.diagnostic_events import emit_diagnostic
            emit_diagnostic("trace_alignment_failed", uid=uid, error=str(exc),
                            trace_count=len(traces),
                            agent_steps=sum(len(t.steps) for t in raw_episode.trajectories))
            raise RolloutInfrastructureError(
                "trace_alignment_failed", str(exc), retryable=False,
                stage="enrichment", retry_scope="none", diagnostics={"session_id": uid},
            ) from exc
        self._annotate_sequence_budget_metadata(enriched)

        infrastructure_failure = raw_episode.metadata.get(
            "infrastructure_failure"
        ) if isinstance(raw_episode.metadata, dict) else None
        if isinstance(infrastructure_failure, dict):
            # The model never returned a usable next turn. Running the
            # verifier would turn a gateway outage into a model score and can
            # consume another ten minutes; preserve the partial audit trail as
            # an infrastructure failure instead.
            enriched.metadata["infrastructure_failure"] = dict(
                infrastructure_failure
            )
            enriched.metrics["infrastructure_failure/reason"] = str(
                infrastructure_failure.get("reason") or "unknown"
            )
            if _timings is not None:
                _timings["time/evaluator_s"] = 0.0
                _agentflow_s = _timings.get("time/agentflow_s", 0.0)
                _llm_sum_s, _llm_wall_s = _summarize_llm_latencies(
                    traces,
                    _agentflow_s,
                )
                _timings["time/agentflow_llm_sum_s"] = _llm_sum_s
                _timings["time/agentflow_llm_wall_s"] = _llm_wall_s
                _timings["n_turns"] = float(
                    sum(
                        len(trajectory.steps)
                        for trajectory in enriched.trajectories
                    )
                )
            return enriched

        # The shadow worker has already been replaying actions and running
        # probes throughout rollout. Bug repair keeps its established primary
        # evaluator path. Legacy repository generation can still use an
        # authoritative final shadow, while the opt-in DeNovo background path
        # runs the final evaluator on the retained primary and defers only
        # milestone annotation to the optimizer-batch barrier.
        t = time.perf_counter()

        def record_evaluator_timing() -> None:
            if _timings is None:
                return
            _timings["time/evaluator_s"] = time.perf_counter() - t
            agentflow_s = _timings.get("time/agentflow_s", 0.0)
            llm_sum_s, llm_wall_s = _summarize_llm_latencies(
                traces,
                agentflow_s,
            )
            _timings["time/agentflow_llm_sum_s"] = llm_sum_s
            _timings["time/agentflow_llm_wall_s"] = llm_wall_s
            _timings["n_turns"] = float(
                sum(
                    len(trajectory.steps)
                    for trajectory in enriched.trajectories
                )
            )

        shadow_runtime = getattr(ctx, "shadow_runtime", None)
        copy_shadow_results = getattr(shadow_runtime, "copy_results_to_episode", None)
        evaluator_error: BaseException | None = None
        eval_output: EvalOutput | None = None
        if getattr(ctx, "outcome_source", "primary_verifier") == "shadow_verifier":
            if shadow_runtime is None:
                raise RolloutInfrastructureError(
                    "shadow_verifier_unavailable",
                    "authoritative shadow verifier runtime is unavailable",
                    retryable=False,
                    stage="shadow_finalize",
                    retry_scope="none",
                )
            self._update_rollout_progress(uid, stage="shadow_finalize")
            shadow_wait_started = time.perf_counter()
            try:
                shadow_future = self._submit_lifecycle_call(
                    shadow_runtime.finalize_episode,
                    raw_episode,
                    None if callable(copy_shadow_results) else enriched,
                    pending_key="evaluator_pending",
                )
                shadow_output = await self._await_lifecycle_future(shadow_future)
                if not isinstance(shadow_output, EvalOutput):
                    raise RuntimeError(
                        "authoritative shadow verifier returned no EvalOutput"
                    )
                eval_output = shadow_output
                if _apply_evaluator_infrastructure_failure(
                    enriched,
                    eval_output,
                ):
                    record_evaluator_timing()
                    return enriched
                if callable(copy_shadow_results):
                    copy_shadow_results(raw_episode, enriched)
                shadow_metadata = raw_episode.metadata.get("shadow_sandbox")
                if isinstance(shadow_metadata, dict):
                    recovery = shadow_metadata.get("final_recovery")
                    if (
                        isinstance(recovery, dict)
                        and recovery.get("status") == "succeeded"
                    ):
                        self._update_lifecycle_metric(
                            "shadow_final_recovery_successes"
                        )
            except Exception as exc:
                classified = exc if isinstance(exc, ShadowFinalizationError) else None
                diagnostics = (
                    classified.diagnostics
                    if classified is not None
                    and isinstance(classified.diagnostics, dict)
                    else {}
                )
                recovery = diagnostics.get("final_recovery")
                if (
                    isinstance(recovery, dict)
                    and recovery.get("status") == "failed"
                ):
                    self._update_lifecycle_metric(
                        "shadow_final_recovery_failures"
                    )
                raise RolloutInfrastructureError(
                    classified.reason if classified is not None else "shadow_verifier_failure",
                    f"authoritative shadow verifier failed for {uid}: {exc}",
                    retryable=(classified.retryable if classified is not None else False),
                    stage=(classified.stage if classified is not None else "shadow_finalize"),
                    retry_scope=(classified.retry_scope if classified is not None else "none"),
                    safe_for_final_recovery=(
                        classified.safe_for_final_recovery
                        if classified is not None
                        else False
                    ),
                    diagnostics=(
                        classified.diagnostics
                        if classified is not None
                        else None
                    ),
                ) from exc
            if _timings is not None:
                _timings["time/shadow_finalize_wait_s"] = time.perf_counter() - shadow_wait_started
        else:
            self._update_rollout_progress(uid, stage="primary_verifier")
            if defer_shadow_finalize:
                evaluator_future = self._submit_primary_verifier_call(
                    ctx.evaluator.evaluate,
                    task_obj,
                    enriched,
                )
            else:
                evaluator_future = self._submit_lifecycle_call(
                    ctx.evaluator.evaluate,
                    task_obj,
                    enriched,
                    pending_key="evaluator_pending",
                )
            try:
                eval_output = await self._await_lifecycle_future(evaluator_future)
                if defer_shadow_finalize and isinstance(eval_output, EvalOutput):
                    eval_output = self._normalize_denovo_primary_eval_output(
                        task_obj,
                        eval_output,
                    )
            except Exception as exc:
                if defer_shadow_finalize:
                    summary = f"{type(exc).__name__}: {exc}"[:2000]
                    eval_output = EvalOutput(
                        reward=0.0,
                        is_correct=False,
                        metadata={
                            "infrastructure_failure": {
                                "schema_version": 2,
                                "reason": "verifier_execution_failed",
                                "stage": "verifier",
                                "task_id": task_obj.id,
                                "verifier_status": "transport_failure",
                                "sandbox_health": "unknown",
                                "sandbox_alive": None,
                                "duration_seconds": max(
                                    0.0,
                                    time.perf_counter() - t,
                                ),
                                "checked_paths": [],
                                "exception_type": type(exc).__name__,
                                "error_summary": summary,
                                "retryable": False,
                            }
                        },
                    )
                else:
                    evaluator_error = exc

            if (
                evaluator_error is None
                and isinstance(eval_output, EvalOutput)
                and _apply_evaluator_infrastructure_failure(
                    enriched,
                    eval_output,
                )
            ):
                # Do not wait for or run a shadow milestone finalizer after the
                # authoritative primary outcome became unclassifiable.  This
                # also prevents a provider outage from holding two sandboxes
                # per rollout for another finalize timeout.
                record_evaluator_timing()
                return enriched

            if shadow_runtime is not None and not defer_shadow_finalize:
                self._update_rollout_progress(uid, stage="shadow_finalize")
                shadow_wait_started = time.perf_counter()
                try:
                    shadow_future = self._submit_lifecycle_call(
                        shadow_runtime.finalize_episode,
                        raw_episode,
                        None if callable(copy_shadow_results) else enriched,
                        pending_key="evaluator_pending",
                    )
                    await self._await_lifecycle_future(shadow_future)
                    if callable(copy_shadow_results):
                        copy_shadow_results(raw_episode, enriched)
                except Exception as exc:
                    logger.exception(
                        "[%s] shadow finalization failed; primary outcome is preserved",
                        uid,
                    )
                    shadow_metadata = dict(
                        enriched.metadata.get("shadow_sandbox") or {}
                    )
                    shadow_metadata.update(
                        {
                            "status": "error",
                            "errors": [
                                *shadow_metadata.get("errors", []),
                                str(exc),
                            ],
                        }
                    )
                    enriched.metadata["shadow_sandbox"] = shadow_metadata
                if _timings is not None:
                    _timings["time/shadow_finalize_wait_s"] = (
                        time.perf_counter() - shadow_wait_started
                    )

        evaluator_elapsed = time.perf_counter() - t

        if evaluator_error is not None:
            raise evaluator_error
        assert eval_output is not None
        if _timings is not None:
            record_evaluator_timing()
            # Preserve the original boundary rather than the few microseconds
            # spent in record_evaluator_timing().
            _timings["time/evaluator_s"] = evaluator_elapsed

        if enriched.termination_reason is None:
            enriched.termination_reason = TerminationReason.ENV_DONE

        self._update_rollout_progress(uid, stage="reward_finalize")
        milestone_reward_config = getattr(self, "milestone_reward_config", None)
        milestone_annotation_error: str | None = None
        if (
            not defer_milestone_annotation
            and milestone_reward_config is not None
            and milestone_reward_config.enable
        ):
            try:
                annotate_episode_milestone_rewards(enriched, milestone_reward_config)
            except Exception as exc:
                # Preserve the expensive rollout. Auto limit rewards fail open
                # to 0.6 and the advantage path retains its existing pre-update
                # validation for systematic annotation failures.
                milestone_annotation_error = str(exc)
                logger.exception("[%s] milestone annotation failed before rollout logging", uid)
                enriched.metadata["milestone_reward"] = {
                    "schema_version": 1,
                    "status": "error",
                    "error": milestone_annotation_error,
                }
        if defer_milestone_annotation:
            enriched.metadata[_DENOVO_BACKGROUND_METADATA_KEY] = {
                "schema_version": 1,
                "status": "awaiting_batch_shadow_finalize",
            }
        else:
            annotate_episode_milestone_verification_metrics(
                enriched,
                milestone_reward_config,
            )

        configured_limit_reward = _normalize_limit_termination_reward(
            getattr(
                self.agent_flow,
                "limit_termination_success_reward",
                _LIMIT_REWARD_FALLBACK,
            )
        )
        limit_outcome_mode = str(
            getattr(
                self.agent_flow,
                "limit_termination_outcome_mode",
                "discount_success",
            )
        )
        if limit_outcome_mode not in {"discount_success", "verifier_outcome"}:
            raise ValueError(
                "agent_flow.limit_termination_outcome_mode must be "
                "'discount_success' or 'verifier_outcome'"
            )

        termination_value = self._termination_value(enriched.termination_reason)
        discounted_limit_terminations = {
            self._termination_value(TerminationReason.MAX_TURNS_EXCEEDED),
            self._termination_value(TerminationReason.MAX_CONTEXT_LENGTH_EXCEEDED),
        }
        limit_termination_success_reward_applied = False
        if (
            limit_outcome_mode == "discount_success"
            and
            eval_output.is_correct
            and termination_value in discounted_limit_terminations
        ):
            trajectories = list(enriched.trajectories)
            resolutions = [
                _resolve_limit_reward(
                    configured_limit_reward,
                    trajectory,
                    milestone_reward_config,
                    milestone_annotation_error,
                )
                for trajectory in trajectories
            ]
            if not resolutions:
                resolutions = [
                    _resolve_limit_reward(
                        configured_limit_reward,
                        None,
                        milestone_reward_config,
                        milestone_annotation_error,
                    )
                ]
            resolved_episode_reward = sum(resolution.reward for resolution in resolutions) / len(resolutions)
            fallback_count = sum(resolution.fallback for resolution in resolutions)
            dynamic_resolutions = [
                resolution
                for resolution in resolutions
                if resolution.mode == _LIMIT_REWARD_AUTO and not resolution.fallback
            ]
            raw_metadata = dict(eval_output.metadata or {})
            raw_metadata.update(
                {
                    "limit_termination_success_reward_applied": True,
                    "limit_termination_success_reward": resolved_episode_reward,
                    "limit_termination_success_reward_configured": configured_limit_reward,
                    "limit_termination_reward_mode": (
                        _LIMIT_REWARD_AUTO
                        if configured_limit_reward == _LIMIT_REWARD_AUTO
                        else "fixed"
                    ),
                    "limit_termination_reason": termination_value,
                    "raw_reward_before_limit_adjustment": eval_output.reward,
                    "raw_is_correct_before_limit_adjustment": eval_output.is_correct,
                }
            )
            if configured_limit_reward == _LIMIT_REWARD_AUTO:
                raw_metadata.update(
                    {
                        "limit_termination_auto_applied": bool(dynamic_resolutions),
                        "limit_termination_auto_dynamic_count": len(dynamic_resolutions),
                        "limit_termination_auto_fallback_count": fallback_count,
                        "limit_termination_auto_fallback_fraction": fallback_count / len(resolutions),
                    }
                )
                if dynamic_resolutions:
                    raw_metadata["limit_termination_first_full_verification_turn_mean"] = sum(
                        resolution.first_full_verification_turn or 0
                        for resolution in dynamic_resolutions
                    ) / len(dynamic_resolutions)
            eval_output = EvalOutput(
                reward=resolved_episode_reward,
                is_correct=eval_output.is_correct,
                signals=eval_output.signals,
                metadata=raw_metadata,
            )
            limit_termination_success_reward_applied = True
            trajectory_payloads = [
                resolution.to_metadata(trajectory)
                for resolution, trajectory in zip(resolutions, trajectories, strict=True)
            ] if trajectories else [resolutions[0].to_metadata()]
            limit_metadata: dict[str, Any] = {
                "schema_version": 2,
                "termination_reason": termination_value,
                "configured_reward": configured_limit_reward,
                "mode": _LIMIT_REWARD_AUTO if configured_limit_reward == _LIMIT_REWARD_AUTO else "fixed",
                "reward": resolved_episode_reward,
                "raw_reward": raw_metadata[
                    "raw_reward_before_limit_adjustment"
                ],
                "fallback": bool(fallback_count),
                "fallback_reasons": sorted(
                    {
                        resolution.fallback_reason
                        for resolution in resolutions
                        if resolution.fallback_reason is not None
                    }
                ),
                "trajectories": trajectory_payloads,
            }
            if len(trajectory_payloads) == 1:
                limit_metadata.update(
                    {
                        key: value
                        for key, value in trajectory_payloads[0].items()
                        if key not in {"trajectory_uid", "trajectory_name", "reward", "mode"}
                    }
                )
            enriched.metadata["limit_termination_reward"] = limit_metadata
            for resolution, trajectory in zip(resolutions, trajectories, strict=True):
                trajectory.reward = resolution.reward
                trajectory.info["limit_termination_reward"] = {
                    "schema_version": 2,
                    "termination_reason": termination_value,
                    "configured_reward": configured_limit_reward,
                    "raw_reward": raw_metadata["raw_reward_before_limit_adjustment"],
                    **resolution.to_metadata(trajectory),
                }

        # Preserve per-trajectory rewards set by multi-trajectory evaluators,
        # except when applying the configured reward for successful limit terminations.
        verifier_outcome = (eval_output.metadata or {}).get("verifier_outcome")
        if isinstance(verifier_outcome, dict):
            enriched.metadata["verifier_outcome"] = dict(verifier_outcome)
        for traj in enriched.trajectories:
            if not limit_termination_success_reward_applied and traj.reward is None:
                traj.reward = eval_output.reward
            if not traj.signals:
                traj.signals = {s.name: s.value for s in eval_output.signals}
            if isinstance(verifier_outcome, dict):
                traj.info["verifier_outcome"] = dict(verifier_outcome)
        enriched.is_correct = eval_output.is_correct

        enriched.metrics.update(eval_output.metadata)
        for signal in eval_output.signals:
            enriched.metrics[signal.name] = signal.value

        return enriched

    def shutdown(self) -> None:
        """Shutdown the engine and cleanup resources."""
        deferred_lock = getattr(self, "_deferred_shadow_registry_lock", None)
        if deferred_lock is None:
            deferred_leases = []
        else:
            with deferred_lock:
                deferred_registry = getattr(
                    self,
                    "_deferred_shadow_leases",
                    {},
                )
                deferred_leases = list(deferred_registry.values())
        deferred_cleanup_errors = []
        for lease in deferred_leases:
            try:
                shadow_lifecycle = lease.cancel("trainer_shutdown")
                cancelled_at = time.monotonic()
                input_sealed_at = shadow_lifecycle.get(
                    "input_sealed_at_monotonic"
                )
                for episode in (lease.raw_episode, lease.enriched_episode):
                    lifecycle = dict(
                        episode.metadata.get(
                            _DENOVO_BACKGROUND_METADATA_KEY
                        )
                        or {}
                    )
                    primary_finished_at = lifecycle.get(
                        "primary_verifier_finished_at_monotonic"
                    )
                    lifecycle.update(
                        {
                            "status": "shadow_cancelled",
                            "training_disposition": "trainer_shutdown",
                            "shadow_finalize_disposition": (
                                shadow_lifecycle.get("finalize_disposition")
                                or "trainer_shutdown"
                            ),
                            "completed_probes_at_disposition": (
                                shadow_lifecycle.get(
                                    "completed_probes_at_disposition"
                                )
                            ),
                            "cancelled_pending_steps": (
                                shadow_lifecycle.get(
                                    "cancelled_pending_steps"
                                )
                            ),
                            "shadow_finalized_at_monotonic": cancelled_at,
                            "shadow_background_lifetime_s": max(
                                0.0,
                                cancelled_at
                                - (
                                    float(input_sealed_at)
                                    if isinstance(
                                        input_sealed_at,
                                        int | float,
                                    )
                                    else lease.registered_at_monotonic
                                ),
                            ),
                            "shadow_post_primary_overlap_s": max(
                                0.0,
                                cancelled_at
                                - (
                                    float(primary_finished_at)
                                    if isinstance(
                                        primary_finished_at,
                                        int | float,
                                    )
                                    else lease.primary_verifier_finished_at_monotonic
                                ),
                            ),
                        }
                    )
                    episode.metadata[
                        _DENOVO_BACKGROUND_METADATA_KEY
                    ] = lifecycle
            except Exception:
                logger.exception(
                    "[%s] deferred shadow audit sealing failed during engine shutdown",
                    lease.uid,
                )
            try:
                cluster = getattr(getattr(self, "hooks", None), "_minisandbox_cluster", None)
                if cluster is not None and cluster.fatal_error:
                    raise RuntimeError("shadow cleanup unconfirmed on quarantined MiniSandbox cluster")
                lease.close_shadow()
            except Exception as exc:
                deferred_cleanup_errors.append(exc)
                lease.cleanup_error = exc
                lease.resource_request.cleanup_unconfirmed = True
                logger.exception(
                    "[%s] deferred shadow close failed during engine shutdown",
                    lease.uid,
                )
            else:
                with deferred_lock:
                    deferred_registry.pop(lease.uid, None)
        lifecycle_poller = getattr(self, "_lifecycle_poller_task", None)
        if lifecycle_poller is not None and not lifecycle_poller.done():
            try:
                if not lifecycle_poller.get_loop().is_closed():
                    lifecycle_poller.cancel()
            except RuntimeError:
                pass
        self._lifecycle_poller_task = None
        lag_task = getattr(self, "_event_loop_lag_task", None)
        if lag_task is not None and not lag_task.done():
            try:
                if not lag_task.get_loop().is_closed():
                    lag_task.cancel()
            except RuntimeError:
                pass
        self._event_loop_lag_task = None
        if self.executor is not None:
            if getattr(self, "session_namespace", "") or getattr(getattr(self, "hooks", None), "sandbox_backend", None) == "minisandbox":
                self.executor.shutdown(wait=False, cancel_futures=True)
            else:
                self.executor.shutdown(wait=True)
            self.executor = None
        teardown_executor = getattr(self, "teardown_executor", None)
        if teardown_executor is not None:
            teardown_executor.shutdown(wait=False, cancel_futures=True)
            self.teardown_executor = None
        primary_verifier_executor = getattr(
            self,
            "primary_verifier_executor",
            None,
        )
        if primary_verifier_executor is not None:
            primary_verifier_executor.shutdown(
                wait=False,
                cancel_futures=True,
            )
            self.primary_verifier_executor = None
        lifecycle_executor = getattr(self, "lifecycle_executor", None)
        if lifecycle_executor is not None:
            # Deferred lifecycle work must not recreate an unbounded shutdown
            # barrier. Backend cleanup has an independent deadline.
            lifecycle_executor.shutdown(wait=False, cancel_futures=True)
            self.lifecycle_executor = None
        if deferred_cleanup_errors:
            from rllm.utils.infrastructure_diagnostics import infrastructure_exception_chain
            raise RolloutInfrastructureError(
                "shadow_cleanup_unconfirmed", "Deferred shadow cleanup failed during shutdown",
                retryable=False, retry_scope="none", stage="shutdown",
                diagnostics={"cleanup_errors": [infrastructure_exception_chain(e) for e in deferred_cleanup_errors]},
            )
