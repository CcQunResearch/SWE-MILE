"""Outcome-reward filtering shared by synchronous and fully-async training."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import Counter, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from rllm.types import TrajectoryGroup

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from rllm.data import DataloaderBatchTicket, DynamicSamplingTaskDataLoader


@dataclass(frozen=True)
class DynamicSamplingPartition:
    """Partition trajectory groups without consulting advantages or shaped rewards."""

    kept: list[TrajectoryGroup]
    filtered: list[TrajectoryGroup]
    uniform: list[TrajectoryGroup]
    easy_uniform: list[TrajectoryGroup]
    hard_uniform: list[TrajectoryGroup]
    easy_saturation: list[TrajectoryGroup]
    hard_collapse: list[TrajectoryGroup]

    @property
    def task_filtered(self) -> bool:
        return bool(self.filtered) and not self.kept

    @property
    def easy(self) -> bool:
        return self.task_filtered and bool(
            self.easy_uniform or self.easy_saturation
        )

    @property
    def hard(self) -> bool:
        return self.task_filtered and bool(
            self.hard_uniform or self.hard_collapse
        )

    @property
    def uniform_pass_count(self) -> bool:
        return self.task_filtered and bool(self.uniform)

    @property
    def filter_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.uniform_pass_count:
            reasons.append("uniform")
        if self.easy:
            reasons.append("easy")
        if self.hard:
            reasons.append("hard")
        return tuple(reasons)


def _verifier_outcome(trajectory: Any) -> tuple[int, float] | None:
    raw = trajectory.info.get("verifier_outcome")
    if not isinstance(raw, dict):
        return None
    passed = raw.get("passed_count")
    total = raw.get("total_count")
    rate = raw.get("pass_rate")
    if (
        isinstance(passed, bool)
        or not isinstance(passed, int)
        or passed < 0
        or isinstance(total, bool)
        or not isinstance(total, int)
        or total <= 0
        or passed > total
        or isinstance(rate, bool)
        or not isinstance(rate, int | float)
        or not math.isfinite(float(rate))
        or not 0.0 <= float(rate) <= 1.0
    ):
        return None
    expected = passed / total
    if not math.isclose(float(rate), expected, rel_tol=0.0, abs_tol=1e-9):
        return None
    return passed, float(rate)


def partition_uniform_outcome_groups(
    groups: list[TrajectoryGroup],
    *,
    outcome_mode: str = "reward_uniform",
    easy_pass_rate_threshold: float = 0.9,
    hard_pass_rate_threshold: float = 0.1,
) -> DynamicSamplingPartition:
    """Classify outcome-filtered groups using one shared sync/async policy.

    ``reward_uniform`` exactly preserves the bug-repair policy. In
    ``verifier_pass_count`` mode, trusted verifier metadata drives independent
    uniform, easy-saturation, and hard-collapse classifications.
    """

    if outcome_mode not in {"reward_uniform", "verifier_pass_count"}:
        raise ValueError(f"unsupported dynamic sampling outcome_mode={outcome_mode!r}")
    if not 0.0 <= hard_pass_rate_threshold <= easy_pass_rate_threshold <= 1.0:
        raise ValueError("dynamic sampling pass-rate thresholds are invalid")

    kept: list[TrajectoryGroup] = []
    filtered: list[TrajectoryGroup] = []
    uniform: list[TrajectoryGroup] = []
    easy_uniform: list[TrajectoryGroup] = []
    hard_uniform: list[TrajectoryGroup] = []
    easy_saturation: list[TrajectoryGroup] = []
    hard_collapse: list[TrajectoryGroup] = []

    for group in groups:
        if outcome_mode == "verifier_pass_count":
            outcomes = [
                _verifier_outcome(trajectory) for trajectory in group.trajectories
            ]
            if not outcomes or any(outcome is None for outcome in outcomes):
                kept.append(group)
                continue
            trusted = [outcome for outcome in outcomes if outcome is not None]
            passed_counts = [outcome[0] for outcome in trusted]
            pass_rates = [outcome[1] for outcome in trusted]
            is_uniform = all(value == passed_counts[0] for value in passed_counts[1:])
            is_easy = all(value >= easy_pass_rate_threshold for value in pass_rates)
            is_hard = all(value <= hard_pass_rate_threshold for value in pass_rates)
            if is_uniform:
                uniform.append(group)
            if is_easy:
                easy_saturation.append(group)
            if is_hard:
                hard_collapse.append(group)
            if is_uniform or is_easy or is_hard:
                filtered.append(group)
            else:
                kept.append(group)
            continue

        rewards = [trajectory.reward for trajectory in group.trajectories]
        if not rewards or any(reward is None for reward in rewards):
            kept.append(group)
            continue
        first = rewards[0]
        if not all(reward == first for reward in rewards[1:]):
            kept.append(group)
            continue
        filtered.append(group)
        uniform.append(group)
        if first == 0:
            hard_uniform.append(group)
        else:
            easy_uniform.append(group)

    return DynamicSamplingPartition(
        kept=kept,
        filtered=filtered,
        uniform=uniform,
        easy_uniform=easy_uniform,
        hard_uniform=hard_uniform,
        easy_saturation=easy_saturation,
        hard_collapse=hard_collapse,
    )


WaveDisposition = Literal["accepted", "failed", "surplus"]


@dataclass
class _SamplingWave:
    index: int
    target: int
    candidate_limit: int
    dispatch_step: int = 0
    dispatch_weight_version: int = 0
    accepted: int = 0
    dispatched: int = 0
    closed: bool = False
    incomplete: bool = False


class DynamicSamplingWaveController:
    """Admission controller for fully-async speculative sampling waves.

    A successful candidate occupies its slot for the lifetime of the wave;
    only filtered candidates release admission capacity. Once ``target``
    successful groups have reserved a place, all still-running candidates are
    returned without classification and the next wave waits for the matching
    optimizer step to commit.
    """

    STATE_SCHEMA_VERSION = 3

    def __init__(
        self,
        *,
        loader: DynamicSamplingTaskDataLoader,
        first_target: int,
        steady_target: int,
        multiplier: float,
        rollout_group_size: int,
        cancel_wait_timeout_seconds: float = 120.0,
        cancel_group: Callable[[str], Awaitable[tuple[int, int]]] | None = None,
        initial_dispatch_step: int = 0,
        initial_weight_version: int = 0,
        denovo_infrastructure_circuit_breaker: bool = False,
        infrastructure_circuit_breaker: bool | None = None,
    ) -> None:
        for name, value in (
            ("first_target", first_target),
            ("steady_target", steady_target),
            ("rollout_group_size", rollout_group_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if (
            isinstance(multiplier, bool)
            or not isinstance(multiplier, int | float)
            or not math.isfinite(float(multiplier))
            or float(multiplier) < 1.0
        ):
            raise ValueError(f"multiplier must be finite and >= 1, got {multiplier!r}")
        if (
            isinstance(cancel_wait_timeout_seconds, bool)
            or not isinstance(cancel_wait_timeout_seconds, int | float)
            or not math.isfinite(float(cancel_wait_timeout_seconds))
            or float(cancel_wait_timeout_seconds) <= 0
        ):
            raise ValueError(
                "cancel_wait_timeout_seconds must be finite and positive, got "
                f"{cancel_wait_timeout_seconds!r}"
            )
        self.loader = loader
        self.first_target = first_target
        self.steady_target = steady_target
        self.multiplier = float(multiplier)
        self.rollout_group_size = rollout_group_size
        self.cancel_wait_timeout_seconds = float(cancel_wait_timeout_seconds)
        self.denovo_infrastructure_circuit_breaker = bool(
            denovo_infrastructure_circuit_breaker
            if infrastructure_circuit_breaker is None else infrastructure_circuit_breaker
        )
        self._cancel_group = cancel_group
        self._recover_group: (
            Callable[[str], Awaitable[tuple[int, int]]] | None
        ) = None
        self._prepare_cancel_groups: (
            Callable[[list[str]], Awaitable[None]] | None
        ) = None
        self._wait_cleanup_ready: Callable[[], Awaitable[None]] | None = None
        self._detach_tasks: Callable[[list[asyncio.Task[Any]]], None] | None = None
        self._describe_task: Callable[[asyncio.Task[Any]], str] | None = None
        self._next_dispatch_step = int(initial_dispatch_step)
        self._next_weight_version = int(initial_weight_version)
        self._wave = self._new_wave(0)
        self._active: dict[str, DataloaderBatchTicket] = {}
        self._active_started_at: dict[str, float] = {}
        self._tasks: dict[str, list[asyncio.Task[Any]]] = {}
        self._early_failure_claims: set[str] = set()
        self._cancellation_tasks: dict[str, asyncio.Task[None]] = {}
        self._cancellation_errors: list[BaseException] = []
        self._wave_cancellation_task: asyncio.Task[None] | None = None
        self._transition_pending = False
        self._transition_started_at: float | None = None
        self._cleanup_barrier_stalled = False
        self._closing_task_ids: set[str] = set()
        self._optimizer_steps_completed = 0
        self._last_committed_step: int | None = None
        self._terminal = False
        # Dataset-aware admission circuit breaker. It is runtime state rather
        # than checkpoint state: a restarted trainer probes provider health
        # afresh instead of restoring an expired monotonic deadline. Legacy
        # controllers leave it disabled unless explicitly enabled by the backend.
        self._infra_failure_window: deque[tuple[float, str]] = deque()
        self._infra_pause_until = 0.0
        self._infra_backoff_seconds = 120.0
        self._infra_failed_canaries = 0
        self._infra_fatal_error: RuntimeError | None = None
        self._infra_circuit_tripped = False
        self._infra_canary_task_id: str | None = None
        self._changed = asyncio.Event()
        self._changed.set()
        self._metrics: Counter[str] = Counter()
        self._metrics["waves_started"] = 1
        self._publish_state()

    def _new_wave(self, index: int) -> _SamplingWave:
        target = self.first_target if index == 0 else self.steady_target
        return _SamplingWave(
            index=index,
            target=target,
            candidate_limit=math.ceil(target * self.multiplier),
            dispatch_step=self._next_dispatch_step,
            dispatch_weight_version=self._next_weight_version,
        )

    def set_cancel_group_callback(
        self,
        callback: Callable[[str], Awaitable[tuple[int, int]]],
    ) -> None:
        self._cancel_group = callback

    def set_wave_cleanup_callbacks(
        self,
        *,
        prepare: Callable[[list[str]], Awaitable[None]] | None,
        wait_ready: Callable[[], Awaitable[None]] | None,
    ) -> None:
        """Install the gateway's two-phase speculative-wave cleanup hooks."""
        self._prepare_cancel_groups = prepare
        self._wait_cleanup_ready = wait_ready

    def set_detach_tasks_callback(
        self,
        callback: Callable[[list[asyncio.Task[Any]]], None],
    ) -> None:
        """Install the coordinator hook used to exclude abandoned rollouts."""
        self._detach_tasks = callback

    def set_task_diagnostic_callback(
        self,
        callback: Callable[[asyncio.Task[Any]], str],
    ) -> None:
        self._describe_task = callback

    def set_recover_group_callback(
        self,
        callback: Callable[[str], Awaitable[tuple[int, int]]],
    ) -> None:
        self._recover_group = callback

    @property
    def wave_index(self) -> int:
        return self._wave.index

    @property
    def target(self) -> int:
        return self._wave.target

    @property
    def candidate_limit(self) -> int:
        return self._wave.candidate_limit

    @property
    def terminal(self) -> bool:
        return self._terminal

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def cleanup_barrier_stalled(self) -> bool:
        """Whether the latest gateway cleanup barrier exceeded its deadline."""
        return self._cleanup_barrier_stalled

    async def wait_for_cleanup_barrier(self) -> bool:
        """Join the current wave transition before a policy weight mutation.

        A barrier timeout remains non-fatal to speculative cancellation itself,
        but is sticky until the trainer verifies cleanup or restarts the
        gateway. Successor-wave admission and weight sync both remain blocked
        while the cleanup state is ambiguous.
        """
        while self._wave_cancellation_task is not None:
            task = self._wave_cancellation_task
            await asyncio.shield(task)
            if self._wave_cancellation_task is task and task.done():
                # Done callbacks normally clear this synchronously. Yield once
                # for alternate event-loop implementations before rechecking.
                await asyncio.sleep(0)
        self._raise_cancellation_error()
        return not self._cleanup_barrier_stalled

    def note_gateway_cleanup_ready(self) -> None:
        """Clear a sticky barrier stall after readiness or process recovery."""
        self._cleanup_barrier_stalled = False
        self._maybe_start_next_wave()
        self._changed.set()
        self._publish_state()

    def can_dispatch(self) -> bool:
        return (
            not self._terminal
            and not self._transition_pending
            and not self._cleanup_barrier_stalled
            and not self._wave.closed
            and self._wave.accepted + len(self._active)
            < self._wave.candidate_limit
            and self._infrastructure_admission_open()
        )

    async def wait_for_dispatch(self) -> bool:
        while not self.can_dispatch():
            self._raise_cancellation_error()
            if self._terminal:
                return False
            self._changed.clear()
            if self.can_dispatch() or self._terminal:
                continue
            pause_remaining = self._infrastructure_pause_remaining()
            if pause_remaining is None:
                await self._changed.wait()
            else:
                try:
                    await asyncio.wait_for(
                        self._changed.wait(),
                        timeout=max(0.001, pause_remaining),
                    )
                except TimeoutError:
                    # The elapsed monotonic deadline itself is the state
                    # change; no producer callback is expected to set Event.
                    pass
        return True

    async def wait_for_change(self) -> None:
        self._raise_cancellation_error()
        self._changed.clear()
        await self._changed.wait()
        self._raise_cancellation_error()

    def register_candidate(
        self,
        task_id: str,
        ticket: DataloaderBatchTicket,
    ) -> int:
        if not self.can_dispatch():
            raise RuntimeError("sampling wave has no candidate admission capacity")
        if task_id in self._active:
            raise ValueError(f"duplicate sampling-wave task id: {task_id}")
        self._active[task_id] = ticket
        self._active_started_at[task_id] = time.monotonic()
        if (
            self.denovo_infrastructure_circuit_breaker
            and self._infra_circuit_tripped
            and time.monotonic() >= self._infra_pause_until
            and self._infra_canary_task_id is None
        ):
            self._infra_canary_task_id = task_id
            self._metrics["infrastructure_circuit_canaries"] += 1
        self._wave.dispatched += 1
        self._metrics["dispatches"] += 1
        self._publish_state()
        return self._wave.index

    def attach_tasks(self, task_id: str, tasks: list[asyncio.Task[Any]]) -> None:
        if task_id not in self._active:
            for task in tasks:
                task.cancel()
            return
        self._tasks[task_id] = tasks

    def complete_candidate(
        self,
        task_id: str,
        *,
        accepted: bool,
        true_uniform: bool = False,
        filter_reasons: tuple[str, ...] = (),
        infrastructure_failure: bool = False,
    ) -> WaveDisposition:
        if task_id not in self._active:
            return "surplus"
        if self._wave.closed:
            self._active.pop(task_id, None)
            self._active_started_at.pop(task_id, None)
            self._tasks.pop(task_id, None)
            self._metrics["completed_surplus"] += 1
            self._metrics["unconsumed_requeued"] += 1
            self._release_infrastructure_canary(task_id)
            self._changed.set()
            self._maybe_start_next_wave()
            self._publish_state()
            return "surplus"

        self._active.pop(task_id, None)
        self._active_started_at.pop(task_id, None)
        self._tasks.pop(task_id, None)
        self._note_infrastructure_result(
            task_id,
            infrastructure_failure=infrastructure_failure,
        )
        if accepted:
            self._wave.accepted += 1
            self._metrics["accepted"] += 1
            disposition: WaveDisposition = "accepted"
            if self._wave.accepted >= self._wave.target:
                self._close_wave(incomplete=False)
        else:
            if infrastructure_failure:
                self._metrics["infrastructure_failed_groups"] += 1
                self._metrics["recovery_requeued_groups"] += 1
            else:
                self._metrics["true_filtered" if true_uniform else "other_filtered"] += 1
                for reason in filter_reasons:
                    if reason not in {"uniform", "easy", "hard"}:
                        raise ValueError(
                            f"unsupported dynamic sampling filter reason: {reason!r}"
                        )
                    self._metrics[f"filtered_{reason}"] += 1
            disposition = "failed"
            self._changed.set()
        self._publish_state()
        return disposition

    def note_pool_unavailable(self) -> bool:
        """Close an incomplete wave once no candidate can still classify."""
        if self._terminal:
            return True
        if self._active:
            return False
        if self.promote_deferred_retries_in_current_wave():
            return False
        if not self._wave.closed and self._wave.accepted < self._wave.target:
            self.loader.finalize_deferred_tail()
            self._close_wave(incomplete=True)
            self._terminal = True
            self._changed.set()
            self._publish_state()
        return self._terminal

    def promote_deferred_retries_in_current_wave(self) -> int:
        """Start another rejection pass when an open wave has gone idle.

        Fully-async sampling initially visits every eligible task before it
        retries a filtered task. If that first pass cannot fill the optimizer
        batch, the same wave must keep making passes until it reaches its target
        or every remaining task reaches its configured rejection threshold.
        """

        if self._terminal or self._wave.closed or self._active:
            return 0
        eligible, deferred = self.loader.sampling_pool_sizes()
        if eligible or not deferred:
            return 0
        promoted = self.loader.advance_sampling_wave()
        if promoted:
            self._metrics["same_wave_retry_promotions"] += promoted
            self._changed.set()
            self._publish_state()
        return promoted

    def on_optimizer_step_committed(
        self,
        completed_step: int,
        *,
        next_dispatch_step: int,
        weight_version: int,
    ) -> None:
        """Unlock a successor wave only after the full step commit barrier."""
        self._optimizer_steps_completed += 1
        self._last_committed_step = int(completed_step)
        self._next_dispatch_step = int(next_dispatch_step)
        self._next_weight_version = int(weight_version)
        self._maybe_start_next_wave()
        self._publish_state()


    def note_discarded_json(self, count: int) -> None:
        self._metrics["discarded_json"] += max(0, int(count))

    async def shutdown(self) -> None:
        """Stop admission and return every still-running candidate."""
        self._terminal = True
        self._wave.closed = True
        self._changed.set()
        if self._wave_cancellation_task is not None:
            await asyncio.shield(self._wave_cancellation_task)
        if self._active:
            await self._cancel_many(
                list(self._active),
                prepare_gateway=True,
                wait_cleanup=True,
            )
        self._publish_state()

    def _close_wave(self, *, incomplete: bool) -> None:
        if self._wave.closed:
            return
        self._wave.closed = True
        self._wave.incomplete = incomplete
        self._metrics["waves_closed"] += 1
        self._metrics["wave_target"] = self._wave.target
        self._metrics["wave_candidate_limit"] = self._wave.candidate_limit
        surplus = list(self._active)
        if surplus:
            self._transition_pending = True
            self._transition_started_at = time.monotonic()
            self._closing_task_ids.update(surplus)
            self._wave_cancellation_task = asyncio.create_task(
                self._cancel_many(
                    surplus,
                    prepare_gateway=True,
                    wait_cleanup=True,
                )
            )

            def finish_transition(done: asyncio.Task[None]) -> None:
                if self._wave_cancellation_task is done:
                    self._wave_cancellation_task = None
                if done.cancelled():
                    return
                error = done.exception()
                if error is not None:
                    self._cancellation_errors.append(error)
                    self._terminal = True
                self._transition_pending = False
                self._transition_started_at = None
                self._closing_task_ids.clear()
                self._changed.set()
                self._maybe_start_next_wave()
                self._publish_state()

            self._wave_cancellation_task.add_done_callback(finish_transition)
        self._changed.set()
        self._maybe_start_next_wave()

    async def abort_active_for_recovery(self) -> list[str]:
        """Return every unclassified candidate after gateway loss.

        The wave stays open and already accepted groups remain reserved.  The
        generation loop can refill exactly these candidate slots after the
        supervisor has restarted the sidecar.
        """
        task_ids = list(dict.fromkeys((*self._active, *self._closing_task_ids)))
        if not task_ids:
            return []
        self._transition_pending = True
        try:
            await self._cancel_many(
                [task_id for task_id in task_ids if task_id in self._active],
                prepare_gateway=False,
                wait_cleanup=False,
                recovery=True,
            )
        finally:
            self._transition_pending = False
            self._changed.set()
            self._publish_state()
        return task_ids

    async def _cancel_many(
        self,
        task_ids: list[str],
        *,
        prepare_gateway: bool,
        wait_cleanup: bool,
        recovery: bool = False,
    ) -> None:
        if prepare_gateway and self._prepare_cancel_groups is not None:
            try:
                await self._prepare_cancel_groups(task_ids)
            except Exception as exc:  # supervisor owns a dead gateway
                self._metrics["bulk_tombstone_failures"] += 1
                logger.error(
                    "Gateway bulk tombstone failed for %d speculative groups; "
                    "continuing cooperative cancellation while supervision "
                    "checks the sidecar: %s",
                    len(task_ids),
                    exc,
                )
        deadline = asyncio.get_running_loop().time() + self.cancel_wait_timeout_seconds
        await asyncio.gather(
            *(
                self._cancel_one(
                    task_id,
                    recovery=recovery,
                    deadline=deadline,
                )
                for task_id in task_ids
            ),
            return_exceptions=False,
        )
        if wait_cleanup and self._wait_cleanup_ready is not None:
            remaining = max(0.0, deadline - asyncio.get_running_loop().time())
            try:
                if remaining <= 0:
                    raise TimeoutError
                await asyncio.wait_for(self._wait_cleanup_ready(), timeout=remaining)
            except TimeoutError:
                self._metrics["cleanup_barrier_timeouts"] += 1
                self._cleanup_barrier_stalled = True
                logger.error(
                    "Speculative gateway cleanup barrier exceeded the %.1fs "
                    "wave cancellation budget; successor-wave admission is "
                    "blocked until supervision verifies or recovers the "
                    "gateway. Tombstoned groups=%s",
                    self.cancel_wait_timeout_seconds,
                    task_ids,
                )

    def _schedule_cancel(self, task_id: str) -> asyncio.Task[None]:
        existing = self._cancellation_tasks.get(task_id)
        if existing is not None:
            return existing
        task = asyncio.create_task(self._cancel_one(task_id))
        self._cancellation_tasks[task_id] = task

        def forget(done: asyncio.Task[None]) -> None:
            if self._cancellation_tasks.get(task_id) is done:
                self._cancellation_tasks.pop(task_id, None)
            if not done.cancelled():
                error = done.exception()
                if error is not None:
                    self._cancellation_errors.append(error)
                    self._terminal = True
                    self._changed.set()

        task.add_done_callback(forget)
        return task

    def _raise_cancellation_error(self) -> None:
        if self._infra_fatal_error is not None:
            raise self._infra_fatal_error
        if not self._cancellation_errors:
            return
        error = self._cancellation_errors.pop(0)
        raise RuntimeError("speculative rollout cancellation failed") from error

    async def _cancel_one(
        self,
        task_id: str,
        *,
        recovery: bool = False,
        deadline: float | None = None,
        exclude_task: asyncio.Task[Any] | None = None,
        infrastructure_failure: bool = False,
    ) -> None:
        if task_id not in self._active:
            return
        completed_rollouts = 0
        discarded_logs = 0
        cancel_callback = (
            self._recover_group
            if recovery and self._recover_group is not None
            else self._cancel_group
        )
        if cancel_callback is not None:
            completed_rollouts, discarded_logs = await cancel_callback(task_id)
        tasks = [
            task
            for task in self._tasks.pop(task_id, [])
            if task is not exclude_task
        ]
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            # A synchronous AgentFlow running in a ThreadPoolExecutor cannot be
            # force-killed. Give its cooperative cleanup a bounded opportunity,
            # then detach the valueless speculative surplus so wave transition
            # and coordinator drain can continue.
            timeout = self.cancel_wait_timeout_seconds
            if deadline is not None:
                timeout = max(0.0, deadline - asyncio.get_running_loop().time())
            done, pending = await asyncio.wait(
                tasks,
                timeout=timeout,
            )
            if done:
                await asyncio.gather(*done, return_exceptions=True)
            if pending:
                detached = list(pending)
                self._metrics["cancellation_wait_timeouts"] += 1
                self._metrics["detached_rollouts"] += len(detached)
                if self._detach_tasks is not None:
                    self._detach_tasks(detached)
                else:
                    for task in detached:
                        task.add_done_callback(self._observe_detached_task)
                logger.warning(
                    "Speculative cancellation timed out after %.1fs; "
                    "detaching group=%s rollout_diagnostics=%s",
                    self.cancel_wait_timeout_seconds,
                    task_id,
                    [
                        self._describe_task(task)
                        if self._describe_task is not None
                        else task.get_name()
                        for task in detached
                    ],
                )
        if task_id not in self._active:
            return
        self._active.pop(task_id, None)
        self._active_started_at.pop(task_id, None)
        self._release_infrastructure_canary(task_id)
        self._metrics[
            "recovery_cancelled_groups" if recovery else "cancelled_groups"
        ] += 1
        if infrastructure_failure:
            self._metrics["infrastructure_failed_groups"] += 1
        self._metrics["cancelled_rollouts"] += max(
            0, self.rollout_group_size - completed_rollouts
        )
        if infrastructure_failure:
            self._metrics["early_stopped_sibling_rollouts"] += max(
                0, self.rollout_group_size - completed_rollouts
            )
        self._metrics["discarded_json"] += discarded_logs
        self._metrics["unconsumed_requeued"] += 1
        if recovery:
            self._metrics["recovery_requeued_groups"] += 1
        self._changed.set()
        self._maybe_start_next_wave()

    async def fail_candidate_early(
        self,
        task_id: str,
        *,
        current_task: asyncio.Task[Any] | None = None,
    ) -> bool:
        """Fail one infrastructure-broken candidate and cancel its siblings.

        The reporting rollout must not cancel or wait on itself. Ticket
        disposition is delegated to the recovery callback, which returns the
        task to the current wave instead of deferring a provably unclassified
        infrastructure failure.
        """

        if task_id not in self._active or task_id in self._early_failure_claims:
            return False
        self._early_failure_claims.add(task_id)
        try:
            self._note_infrastructure_result(
                task_id,
                infrastructure_failure=True,
            )
            await self._cancel_one(
                task_id,
                recovery=True,
                exclude_task=current_task,
                infrastructure_failure=True,
            )
            self._publish_state()
            return True
        finally:
            self._early_failure_claims.discard(task_id)
        self._publish_state()

    def _infrastructure_admission_open(self) -> bool:
        if self._infra_fatal_error is not None:
            return False
        if (
            not self.denovo_infrastructure_circuit_breaker
            or not self._infra_circuit_tripped
        ):
            return True
        if time.monotonic() < self._infra_pause_until:
            return False
        # Exactly one candidate is admitted as the post-pause canary. Other
        # already-running groups may finish, but cannot reset this circuit.
        return self._infra_canary_task_id is None

    def _infrastructure_pause_remaining(self) -> float | None:
        if (
            not self.denovo_infrastructure_circuit_breaker
            or not self._infra_circuit_tripped
        ):
            return None
        remaining = self._infra_pause_until - time.monotonic()
        return max(0.0, remaining) if remaining > 0 else None

    def _trip_infrastructure_circuit(self, *, canary_failure: bool) -> None:
        if canary_failure:
            self._infra_failed_canaries += 1
            if self._infra_failed_canaries >= 3:
                self._infra_fatal_error = RuntimeError("MiniSandbox/verifier infrastructure did not recover after three canaries")
            self._infra_backoff_seconds = min(
                600.0,
                max(120.0, self._infra_backoff_seconds * 2.0),
            )
            self._metrics["infrastructure_circuit_canary_failures"] += 1
        else:
            self._infra_backoff_seconds = 120.0
        self._infra_circuit_tripped = True
        self._infra_canary_task_id = None
        self._infra_pause_until = time.monotonic() + self._infra_backoff_seconds
        self._infra_failure_window.clear()
        self._metrics["infrastructure_circuit_trips"] += 1
        self._changed.set()
        logger.error(
            "Pausing rollout admission for %.0fs after verifier "
            "infrastructure failures%s",
            self._infra_backoff_seconds,
            " (failed canary)" if canary_failure else "",
        )

    def _release_infrastructure_canary(self, task_id: str) -> None:
        if task_id != self._infra_canary_task_id:
            return
        # Surplus/cancellation carries no provider-health evidence. Keep the
        # circuit tripped but permit a replacement canary immediately.
        self._infra_canary_task_id = None
        self._changed.set()

    def _note_infrastructure_result(
        self,
        task_id: str,
        *,
        infrastructure_failure: bool,
    ) -> None:
        if not self.denovo_infrastructure_circuit_breaker:
            return
        if task_id == self._infra_canary_task_id:
            if infrastructure_failure:
                self._trip_infrastructure_circuit(canary_failure=True)
            else:
                self._infra_failed_canaries = 0
                self._infra_circuit_tripped = False
                self._infra_canary_task_id = None
                self._infra_pause_until = 0.0
                self._infra_backoff_seconds = 120.0
                self._infra_failure_window.clear()
                self._metrics["infrastructure_circuit_canary_successes"] += 1
                self._changed.set()
            return
        if not infrastructure_failure or self._infra_circuit_tripped:
            return
        now = time.monotonic()
        cutoff = now - 60.0
        while (
            self._infra_failure_window
            and self._infra_failure_window[0][0] < cutoff
        ):
            self._infra_failure_window.popleft()
        observed_ids = {
            observed_task_id
            for _observed_at, observed_task_id in self._infra_failure_window
        }
        if task_id not in observed_ids:
            self._infra_failure_window.append((now, task_id))
        if len(self._infra_failure_window) >= 3:
            self._trip_infrastructure_circuit(canary_failure=False)

    @staticmethod
    def _observe_detached_task(task: asyncio.Task[Any]) -> None:
        """Consume a detached task's terminal exception without reclassifying it."""
        if not task.cancelled():
            try:
                task.exception()
            except BaseException:
                pass

    def _maybe_start_next_wave(self) -> None:
        if (
            self._terminal
            or not self._wave.closed
            or self._wave.incomplete
            or self._transition_pending
            or self._cleanup_barrier_stalled
            or self._active
            or self._optimizer_steps_completed <= self._wave.index
        ):
            return
        self.loader.advance_sampling_wave()
        self._wave = self._new_wave(self._wave.index + 1)
        self._metrics["waves_started"] += 1
        self._changed.set()

    def drain_metrics(self) -> dict[str, int | float]:
        eligible, deferred = self.loader.sampling_pool_sizes()
        values: dict[str, int | float] = {
            "wave/current_index": self._wave.index,
            "wave/current_dispatch_step": self._wave.dispatch_step,
            "wave/current_dispatch_weight_version": (
                self._wave.dispatch_weight_version
            ),
            "wave/current_target": self._wave.target,
            "wave/current_candidate_limit": self._wave.candidate_limit,
            "wave/current_accepted": self._wave.accepted,
            "wave/current_in_flight": len(self._active),
            "wave/current_cancelling_groups": (
                len(self._active) if self._wave.closed else 0
            ),
            "wave/current_cancelling_rollouts": (
                len(self._active) * self.rollout_group_size
                if self._wave.closed
                else 0
            ),
            "wave/current_pool_eligible": eligible,
            "wave/current_pool_deferred": deferred,
            "wave/current_transition_pending": int(self._transition_pending),
            "wave/current_cleanup_barrier_stalled": int(
                self._cleanup_barrier_stalled
            ),
            "wave/current_cancellation_age_seconds": (
                max(0.0, time.monotonic() - self._transition_started_at)
                if self._transition_started_at is not None
                else 0.0
            ),
        }
        for key in (
            "dispatches",
            "accepted",
            "true_filtered",
            "other_filtered",
            "cancelled_groups",
            "cancelled_rollouts",
            "completed_surplus",
            "unconsumed_requeued",
            "discarded_json",
            "waves_started",
            "waves_closed",
            "infrastructure_failed_groups",
            "recovery_cancelled_groups",
            "recovery_requeued_groups",
            "bulk_tombstone_failures",
            "cleanup_barrier_timeouts",
            "cancellation_wait_timeouts",
            "detached_rollouts",
            "early_stopped_sibling_rollouts",
            "same_wave_retry_promotions",
            "infrastructure_circuit_trips",
            "infrastructure_circuit_canaries",
            "infrastructure_circuit_canary_successes",
            "infrastructure_circuit_canary_failures",
        ):
            values[f"wave/window_{key}"] = int(self._metrics[key])
        self._metrics.clear()
        return {f"dynamic_sampling/{key}": value for key, value in values.items()}

    def runtime_metrics(self) -> dict[str, int | float]:
        """Non-destructive counters for the independent supervisor heartbeat."""
        now = time.monotonic()
        rollout_tasks = [
            task
            for task_id in self._active
            for task in self._tasks.get(task_id, ())
        ]
        completed_by_group = {
            task_id: sum(task.done() for task in self._tasks.get(task_id, ()))
            for task_id in self._active
        }
        eligible, deferred = self.loader.sampling_pool_sizes()
        return {
            "dynamic_sampling/wave/current_index": self._wave.index,
            "dynamic_sampling/wave/current_dispatch_step": (
                self._wave.dispatch_step
            ),
            "dynamic_sampling/wave/current_dispatch_weight_version": (
                self._wave.dispatch_weight_version
            ),
            "dynamic_sampling/wave/current_target": self._wave.target,
            "dynamic_sampling/wave/current_candidate_limit": (
                self._wave.candidate_limit
            ),
            "dynamic_sampling/wave/current_accepted": self._wave.accepted,
            "dynamic_sampling/wave/current_dispatched": self._wave.dispatched,
            "dynamic_sampling/wave/current_pool_eligible": eligible,
            "dynamic_sampling/wave/current_pool_deferred": deferred,
            "dynamic_sampling/wave/current_closed": int(self._wave.closed),
            "dynamic_sampling/wave/current_transition_pending": int(
                self._transition_pending
            ),
            "dynamic_sampling/wave/cancelling_groups": (
                len(self._active) if self._transition_pending else 0
            ),
            "dynamic_sampling/wave/recovery_requeued_groups": int(
                self._metrics["recovery_requeued_groups"]
            ),
            "dynamic_sampling/wave/early_stopped_sibling_rollouts": int(
                self._metrics["early_stopped_sibling_rollouts"]
            ),
            "dynamic_sampling/wave/in_flight_groups": len(self._active),
            "dynamic_sampling/wave/in_flight_rollouts_total": len(rollout_tasks),
            "dynamic_sampling/wave/in_flight_rollouts_done": sum(
                task.done() for task in rollout_tasks
            ),
            "dynamic_sampling/wave/groups_one_rollout_remaining": sum(
                len(self._tasks.get(task_id, ())) == self.rollout_group_size
                and completed == self.rollout_group_size - 1
                for task_id, completed in completed_by_group.items()
            ),
            "dynamic_sampling/wave/oldest_in_flight_age_seconds": (
                max(
                    now - started_at
                    for task_id, started_at in self._active_started_at.items()
                    if task_id in self._active
                )
                if self._active_started_at and self._active
                else 0.0
            ),
            "dynamic_sampling/wave/cancellation_age_seconds": (
                max(0.0, time.monotonic() - self._transition_started_at)
                if self._transition_started_at is not None
                else 0.0
            ),
            "dynamic_sampling/wave/cancellation_wait_timeouts": int(
                self._metrics["cancellation_wait_timeouts"]
            ),
            "dynamic_sampling/wave/cleanup_barrier_timeouts": int(
                self._metrics["cleanup_barrier_timeouts"]
            ),
            "dynamic_sampling/wave/cleanup_barrier_stalled": int(
                self._cleanup_barrier_stalled
            ),
            "dynamic_sampling/wave/detached_rollouts": int(
                self._metrics["detached_rollouts"]
            ),
            "dynamic_sampling/wave/same_wave_retry_promotions": int(
                self._metrics["same_wave_retry_promotions"]
            ),
            "dynamic_sampling/infrastructure_circuit/tripped": int(
                self._infra_circuit_tripped
            ),
            "dynamic_sampling/infrastructure_circuit/pause_remaining_seconds": (
                max(0.0, self._infra_pause_until - now)
                if self._infra_circuit_tripped
                else 0.0
            ),
            "dynamic_sampling/infrastructure_circuit/canary_in_flight": int(
                self._infra_canary_task_id is not None
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.STATE_SCHEMA_VERSION,
            "first_target": self.first_target,
            "steady_target": self.steady_target,
            "multiplier": self.multiplier,
            "rollout_group_size": self.rollout_group_size,
            "cancel_wait_timeout_seconds": self.cancel_wait_timeout_seconds,
            "wave": {
                "index": self._wave.index,
                "target": self._wave.target,
                "candidate_limit": self._wave.candidate_limit,
                "dispatch_step": self._wave.dispatch_step,
                "dispatch_weight_version": (
                    self._wave.dispatch_weight_version
                ),
                "accepted": self._wave.accepted,
                "dispatched": self._wave.dispatched,
                "closed": self._wave.closed,
                "incomplete": self._wave.incomplete,
            },
            "optimizer_steps_completed": self._optimizer_steps_completed,
            "last_committed_step": self._last_committed_step,
            "next_dispatch_step": self._next_dispatch_step,
            "next_weight_version": self._next_weight_version,
            "active": {
                task_id: ticket.to_dict()
                for task_id, ticket in self._active.items()
            },
            "terminal": self._terminal,
        }

    def load_state_dict(
        self,
        state: dict[str, Any],
        *,
        replay_pending: bool = False,
    ) -> None:
        """Restore wave state, optionally rebasing lost in-memory rollouts.

        Trainer checkpoints do not serialize rollout coroutines or queued
        trajectory tensors. On process resume their dataloader tickets are
        replayed, so ``replay_pending`` restarts the current wave while keeping
        its index/config. Unit-level live handoff can restore exact admission
        counters with the default mode.
        """
        if state.get("schema_version") != self.STATE_SCHEMA_VERSION:
            raise ValueError(
                "unsupported dynamic sampling wave checkpoint schema_version="
                f"{state.get('schema_version')!r}"
            )
        expected = {
            "first_target": self.first_target,
            "steady_target": self.steady_target,
            "multiplier": self.multiplier,
            "rollout_group_size": self.rollout_group_size,
            "cancel_wait_timeout_seconds": self.cancel_wait_timeout_seconds,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(
                    f"dynamic sampling wave checkpoint {key} mismatch: "
                    f"saved={state.get(key)!r}, current={value!r}"
                )
        raw_wave = state.get("wave")
        if not isinstance(raw_wave, dict):
            raise ValueError("dynamic sampling wave checkpoint has no wave state")
        index = int(raw_wave["index"])
        expected_target = self.first_target if index == 0 else self.steady_target
        expected_candidate_limit = math.ceil(expected_target * self.multiplier)
        if (
            int(raw_wave["target"]) != expected_target
            or int(raw_wave["candidate_limit"]) != expected_candidate_limit
        ):
            raise ValueError("dynamic sampling wave target configuration mismatch")
        self._optimizer_steps_completed = int(
            state.get("optimizer_steps_completed", 0)
        )
        saved_last_step = state.get("last_committed_step")
        self._last_committed_step = (
            None if saved_last_step is None else int(saved_last_step)
        )
        self._next_dispatch_step = int(
            state.get("next_dispatch_step", raw_wave.get("dispatch_step", 0))
        )
        self._next_weight_version = int(
            state.get(
                "next_weight_version",
                raw_wave.get("dispatch_weight_version", 0),
            )
        )
        self._tasks.clear()
        self._early_failure_claims.clear()
        self._active.clear()
        self._active_started_at.clear()
        self._cancellation_tasks.clear()
        self._cancellation_errors.clear()
        self._wave_cancellation_task = None
        self._transition_pending = False
        self._transition_started_at = None
        self._cleanup_barrier_stalled = False
        self._closing_task_ids.clear()
        self._infra_failure_window.clear()
        self._infra_pause_until = 0.0
        self._infra_backoff_seconds = 120.0
        self._infra_circuit_tripped = False
        self._infra_canary_task_id = None
        if replay_pending:
            self._wave = _SamplingWave(
                index=index,
                target=expected_target,
                candidate_limit=expected_candidate_limit,
                dispatch_step=int(raw_wave.get("dispatch_step", 0)),
                dispatch_weight_version=int(
                    raw_wave.get("dispatch_weight_version", 0)
                ),
            )
            self._terminal = False
        else:
            from rllm.data import DataloaderBatchTicket

            self._wave = _SamplingWave(
                index=index,
                target=expected_target,
                candidate_limit=expected_candidate_limit,
                dispatch_step=int(raw_wave.get("dispatch_step", 0)),
                dispatch_weight_version=int(
                    raw_wave.get("dispatch_weight_version", 0)
                ),
                accepted=int(raw_wave.get("accepted", 0)),
                dispatched=int(raw_wave.get("dispatched", 0)),
                closed=bool(raw_wave.get("closed", False)),
                incomplete=bool(raw_wave.get("incomplete", False)),
            )
            raw_active = state.get("active", {})
            if not isinstance(raw_active, dict):
                raise ValueError("invalid dynamic sampling wave active state")
            self._active = {
                str(task_id): DataloaderBatchTicket.from_dict(ticket)
                for task_id, ticket in raw_active.items()
            }
            now = time.monotonic()
            self._active_started_at = {
                task_id: now for task_id in self._active
            }
            self._terminal = bool(state.get("terminal", False))
        self._changed.set()
        self._publish_state()

    def _publish_state(self) -> None:
        setter = getattr(self.loader, "set_sampling_wave_state", None)
        if callable(setter):
            setter(self.state_dict())


__all__ = [
    "DynamicSamplingPartition",
    "DynamicSamplingWaveController",
    "partition_uniform_outcome_groups",
]
