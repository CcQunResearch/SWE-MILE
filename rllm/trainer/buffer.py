"""TrajectoryGroupBuffer for async training.

Accumulates episodes, processes into ready-to-train trajectory groups,
with optional NVMe offloading for memory management.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import pickle
import re
import tempfile
import time
from collections import Counter, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from tqdm import tqdm

    from rllm.data import DataloaderBatchTicket, DynamicSamplingTaskDataLoader
    from rllm.trainer.dynamic_sampling import DynamicSamplingWaveController

from rllm.trainer.algorithms import (
    AlgorithmConfig,
    CompactFilteringConfig,
    DynamicSamplingConfig,
    RejectionSamplingConfig,
    TransformConfig,
    collect_reward_and_advantage_from_trajectory_groups,
)
from rllm.trainer.algorithms.rejection_sampling import filter_episodes
from rllm.trainer.algorithms.transform import transform_episodes_to_trajectory_groups
from rllm.trainer.dynamic_sampling import partition_uniform_outcome_groups
from rllm.trainer.metrics_aggregator import MetricsAggregator
from rllm.trainer.sync_coordinator import SyncCoordinator
from rllm.types import Episode, TrajectoryGroup
from rllm.workflows.workflow import TerminationReason

logger = logging.getLogger(__name__)
_FATAL_GENERATION_SENTINEL = object()


GenerationTerminalKind = Literal[
    "dataset_exhausted",
    "controlled_stop",
    "fatal_error",
]


class AsyncGenerationError(RuntimeError):
    """Raised when the rollout producer terminates abnormally.

    A fatal producer failure must never be represented by the same ``None``
    sentinel as ordinary dataset exhaustion: doing so makes an incomplete
    optimizer batch look like a legitimate drop-last tail.
    """

    def __init__(self, cause: BaseException):
        super().__init__(f"fully-async generation failed: {cause}")
        self.cause = cause

    def __reduce__(self):
        return type(self), (self.cause,), self.__dict__


class GatewayInfrastructureError(RuntimeError):
    """A rollout group has infrastructure failure (legacy public class name)."""


@dataclass
class TaskBatch:
    """All trajectory groups produced from one task's episodes, plus stripped episodes for UI logging."""

    groups: list[TrajectoryGroup]
    task_id: str | None = None
    episodes: list[Episode] = field(default_factory=list)
    source_ticket: DataloaderBatchTicket | None = None
    # Dynamic-sampling compact rollout logging is intentionally kept separate
    # from the filtered training view.  The original view is needed to log
    # every trajectory after its optimizer disposition is known.
    rollout_log_episodes: list[Episode] = field(default_factory=list)
    filtered_trajectory_positions: set[tuple[int, int]] = field(
        default_factory=set
    )
    deferred_shadow_finalize: bool = False
    preflight_group_signature: tuple[
        tuple[str, tuple[tuple[str, str, str, int], ...]], ...
    ] = ()
    preflight_transform_metrics: dict = field(default_factory=dict)
    preflight_dropped_min_trajs: int = 0


class TrajectoryGroupBuffer:
    """Accumulates episodes, processes into trajectory groups, yields to training.

    When all rollouts for a task arrive:
    1. Record episode-level metrics to aggregator (before any filtering)
    2. Transform episodes -> trajectory groups
    3. Compact filtering + drop groups with < min_trajs_per_group
    4. If dynamic sampling is enabled: classify outcome-uniform groups
    5. Compute advantages
    6. If legacy rejection sampling is enabled: drop all-zero-advantage groups
    7. Queue the task batch for training

    Filtered groups are reported directly to the coordinator (which tracks
    throttle slots and filter counts). Only non-empty task batches are queued.
    All metrics flow through the shared MetricsAggregator.

    Optionally offloads pending episodes and/or queued task batches to
    disk to reduce memory pressure (disabled by default).
    """

    def __init__(
        self,
        group_size: int,
        coordinator: SyncCoordinator,
        aggregator: MetricsAggregator,
        algorithm_config: AlgorithmConfig,
        transform_config: TransformConfig,
        cf_config: CompactFilteringConfig,
        rs_config: RejectionSamplingConfig,
        episode_offload_dir: str | None = None,
        trajectory_group_offload_dir: str | None = None,
        pbar: tqdm | None = None,
        source_resolver: Callable[[DataloaderBatchTicket | None], None] | None = None,
        dynamic_sampling_config: DynamicSamplingConfig | None = None,
        dynamic_sampling_loader: DynamicSamplingTaskDataLoader | None = None,
        dynamic_sampling_rollout_logger: Callable[..., None] | None = None,
        dynamic_sampling_rollout_discarder: (
            Callable[[list[Episode]], int] | None
        ) = None,
        dynamic_sampling_wave_controller: DynamicSamplingWaveController | None = None,
        deferred_shadow_predicate: Callable[[Episode], bool] | None = None,
        deferred_shadow_finalizer: (
            Callable[[list[Episode], float], Awaitable[dict[str, float]]] | None
        ) = None,
        deferred_shadow_canceller: (
            Callable[[list[Episode], str], Awaitable[int]] | None
        ) = None,
        deferred_shadow_stall_timeout: float = 0.0,
    ):
        self._group_size = group_size
        self._coordinator = coordinator
        self._aggregator = aggregator
        self._algorithm_config = algorithm_config
        self._transform_config = transform_config
        self._cf_config = cf_config
        self._rs_config = rs_config
        self._pbar = pbar
        self._source_resolver = source_resolver
        self._dynamic_sampling_config = dynamic_sampling_config or DynamicSamplingConfig()
        self._dynamic_sampling_loader = dynamic_sampling_loader
        self._dynamic_sampling_rollout_logger = dynamic_sampling_rollout_logger
        self._dynamic_sampling_rollout_discarder = dynamic_sampling_rollout_discarder
        self._dynamic_sampling_wave_controller = dynamic_sampling_wave_controller
        self._deferred_shadow_predicate = deferred_shadow_predicate
        self._deferred_shadow_finalizer = deferred_shadow_finalizer
        self._deferred_shadow_canceller = deferred_shadow_canceller
        self._deferred_shadow_stall_timeout = float(
            deferred_shadow_stall_timeout
        )
        if self._deferred_shadow_finalizer is not None and (
            self._deferred_shadow_stall_timeout <= 0
        ):
            raise ValueError(
                "deferred_shadow_stall_timeout must be positive when a "
                "deferred shadow finalizer is configured"
            )
        if self._dynamic_sampling_config.enable and dynamic_sampling_loader is None:
            raise ValueError(
                "dynamic sampling requires a DynamicSamplingTaskDataLoader"
            )

        # Episode offloading: pending episodes serialized to disk
        self._episode_offload_dir = episode_offload_dir
        if episode_offload_dir:
            os.makedirs(episode_offload_dir, exist_ok=True)
        self._pending: dict[str, list[Episode | str]] = {}  # str = offloaded file path
        # Removed from the queue, but not yet committed by the optimizer.
        self._deferred_reservations: dict[int, TaskBatch] = {}
        self._processing_deferred_groups: dict[str, TaskBatch] = {}
        self._source_tickets: dict[str, DataloaderBatchTicket | None] = {}
        self._cancelled_task_ids: set[str] = set()
        # Only DeNovo background-finalize rollouts are persisted after a
        # speculative group is cancelled.  Remember the group disposition so
        # synchronous AgentFlow workers that return after cancellation can be
        # audited with the same final state as already-materialized siblings.
        self._cancelled_task_dispositions: dict[str, str] = {}
        self._cancelled_task_materialized_rollouts: Counter[str] = Counter()
        self._cancelled_task_shadow_leases: Counter[str] = Counter()
        self._detached_file_cleanup_tasks: set[asyncio.Task[None]] = set()
        self._detached_shadow_cleanup_tasks: set[asyncio.Task[Any]] = set()

        # Trajectory group offloading: queued task batches serialized to disk
        self._tg_offload_dir = trajectory_group_offload_dir
        if trajectory_group_offload_dir:
            os.makedirs(trajectory_group_offload_dir, exist_ok=True)
        self._queue: asyncio.Queue[TaskBatch | str | None | object] = (
            asyncio.Queue()
        )
        self._training_queue_size = 0
        self._filtered_count = 0
        self._consumed_count = 0
        self._training_step = 0
        self._queue_update_event = asyncio.Event()
        self._generation_complete = False
        self._generation_terminal_kind: GenerationTerminalKind | None = None
        self._generation_error: BaseException | None = None
        self._infrastructure_failed_groups = 0
        rollout_log_path = os.environ.get("RLLM_ROLLOUT_LOG_PATH")
        explicit_audit_dir = os.environ.get("RLLM_INFRA_FAILURE_AUDIT_DIR")
        self._infrastructure_audit_dir = (
            explicit_audit_dir
            or (
                os.path.join(rollout_log_path, "infrastructure_failures")
                if rollout_log_path
                else None
            )
        )
        self._infrastructure_audit_paths: deque[str] = deque()
        if self._infrastructure_audit_dir:
            os.makedirs(self._infrastructure_audit_dir, exist_ok=True)
            existing = sorted(
                os.path.join(self._infrastructure_audit_dir, name)
                for name in os.listdir(self._infrastructure_audit_dir)
                if name.endswith(".json")
            )
            for stale in existing[:-10_000]:
                try:
                    os.remove(stale)
                except FileNotFoundError:
                    pass
            self._infrastructure_audit_paths.extend(existing[-10_000:])

    def set_training_step(self, step: int) -> None:
        self._training_step = step
        self._refresh_pbar_counters()

    def _refresh_pbar_counters(self) -> None:
        if self._pbar is not None:
            self._pbar.set_postfix(
                step=self._training_step,
                session_queued=self._training_queue_size,
                session_filtered=self._filtered_count,
                session_consumed=self._consumed_count,
                refresh=False,
            )

    def _record_classified_prompt_group(self, *, final: bool = True) -> None:
        self._refresh_pbar_counters()
        if final and self._pbar is not None:
            self._pbar.update(1)

    async def _offload_episode(self, task_id: str, episode: Episode) -> str:
        """Serialize episode to disk, return file path."""
        idx = len(self._pending.get(task_id, []))
        path = os.path.join(self._episode_offload_dir, f"{task_id}_{idx}.pkl")
        await asyncio.to_thread(self._pickle_dump, path, episode)
        return path

    async def _load_pending_episodes(self, task_id: str) -> list[Episode]:
        """Load all pending episodes for a task, deserializing offloaded ones."""
        episodes = []
        for item in self._pending.pop(task_id, []):
            if isinstance(item, str):
                ep = await asyncio.to_thread(self._pickle_load, item)
                episodes.append(ep)
            else:
                episodes.append(item)
        return episodes

    async def _offload_task_batch(self, batch: TaskBatch) -> str:
        """Serialize task batch to disk, return file path."""
        fd, path = tempfile.mkstemp(dir=self._tg_offload_dir, suffix=".pkl")
        os.close(fd)
        await asyncio.to_thread(self._pickle_dump, path, batch)
        return path

    async def _load_task_batch(self, item: TaskBatch | str) -> TaskBatch:
        """Load task batch, deserializing if offloaded."""
        if isinstance(item, str):
            return await asyncio.to_thread(self._pickle_load, item)
        return item

    @staticmethod
    def _pickle_dump(path: str, obj) -> None:
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _pickle_load(path: str):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        os.remove(path)
        return obj

    def _remember_source_ticket(
        self,
        task_id: str,
        source_ticket: DataloaderBatchTicket | None,
    ) -> None:
        if task_id not in self._source_tickets:
            self._source_tickets[task_id] = source_ticket
            return
        if self._source_tickets[task_id] != source_ticket:
            raise ValueError(
                f"task group {task_id} received inconsistent dataloader tickets"
            )

    def register_speculative_group(
        self,
        task_id: str,
        source_ticket: DataloaderBatchTicket,
    ) -> None:
        """Register ticket ownership before any rollout coroutine can finish."""
        self._remember_source_ticket(task_id, source_ticket)

    def _resolve_source_ticket(
        self,
        source_ticket: DataloaderBatchTicket | None,
        *,
        rollout_count: int = 0,
    ) -> None:
        if self._dynamic_sampling_loader is not None:
            self._dynamic_sampling_loader.mark_consumed_other(
                source_ticket,
                rollout_count=rollout_count,
            )
            return
        if self._source_resolver is not None:
            self._source_resolver(source_ticket)

    def _is_deferred_shadow_group(self, episodes: list[Episode]) -> bool:
        predicate = self._deferred_shadow_predicate
        if predicate is None or not episodes:
            return False
        values = [bool(predicate(episode)) for episode in episodes]
        if any(values) and not all(values):
            # A DeNovo sibling may fail before it can register a lease.  The
            # infrastructure branch below owns that group and cancels only the
            # leases that actually exist; it is not a legacy ownership mix.
            if all(
                self._is_denovo_background_episode(episode)
                for episode in episodes
            ) and self._infrastructure_failure(episodes) is not None:
                return True
            raise RuntimeError(
                "one rollout group mixes deferred and legacy shadow ownership"
            )
        return all(values)

    async def _cancel_deferred_shadows(
        self,
        episodes: list[Episode],
        disposition: str,
    ) -> int:
        if (
            self._deferred_shadow_canceller is None
            or self._deferred_shadow_predicate is None
            or not episodes
        ):
            return 0
        if not any(
            self._deferred_shadow_predicate(episode)
            for episode in episodes
        ):
            return 0
        return await self._deferred_shadow_canceller(
            episodes,
            disposition,
        )

    @staticmethod
    def _denovo_background_lifecycle(episode: Episode) -> dict | None:
        metadata = episode.metadata if isinstance(episode.metadata, dict) else {}
        lifecycle = metadata.get("denovo_background_finalize")
        return lifecycle if isinstance(lifecycle, dict) else None

    @staticmethod
    def _denovo_background_task_contract(episode: Episode) -> bool:
        task = episode.task if isinstance(episode.task, dict) else {}
        rllm_metadata = task.get("rllm")
        return bool(
            isinstance(rllm_metadata, dict)
            and rllm_metadata.get(
                "denovo_background_finalize_active",
                False,
            )
        )

    def _is_denovo_background_episode(self, episode: Episode) -> bool:
        """Return whether an episode belongs to the opt-in DeNovo lifecycle.

        This deliberately keys off persisted metadata rather than the live
        lease predicate.  A cancellation pops the lease before rollout JSON is
        written, but the final disposition still has to remain auditable.
        """

        return bool(
            self._denovo_background_lifecycle(episode) is not None
            or self._denovo_background_task_contract(episode)
        )

    def should_persist_cancelled_rollout(
        self,
        task_id: str,
        episode: Episode,
    ) -> bool:
        """Identify a late DeNovo rollout that needs disposition-aware logging."""

        return bool(
            (self._generation_complete or task_id in self._cancelled_task_ids)
            and self._is_denovo_background_episode(episode)
            and self._dynamic_sampling_rollout_logger is not None
        )

    def _annotate_denovo_group_disposition(
        self,
        task_id: str,
        episodes: list[Episode],
        disposition: str,
        *,
        cancelled_shadow_leases: int = 0,
        group_rollouts: int | None = None,
    ) -> bool:
        denovo = [
            episode
            for episode in episodes
            if self._is_denovo_background_episode(episode)
        ]
        if not denovo:
            return False
        if len(denovo) != len(episodes):
            raise RuntimeError(
                "one rollout group mixes DeNovo background and legacy episodes"
            )
        expected_rollouts = int(
            group_rollouts
            if group_rollouts is not None
            else self._group_size
        )
        self._cancelled_task_materialized_rollouts[task_id] = min(
            expected_rollouts,
            self._cancelled_task_materialized_rollouts[task_id]
            + len(episodes),
        )
        self._cancelled_task_shadow_leases[task_id] = min(
            expected_rollouts,
            self._cancelled_task_shadow_leases[task_id]
            + max(0, int(cancelled_shadow_leases)),
        )
        materialized = self._cancelled_task_materialized_rollouts[task_id]
        cancelled_leases = self._cancelled_task_shadow_leases[task_id]
        for episode in episodes:
            lifecycle = dict(
                self._denovo_background_lifecycle(episode) or {}
            )
            if not lifecycle:
                lifecycle.update(
                    {
                        "schema_version": 1,
                        "status": "lifecycle_recovered_from_task_contract",
                        "shadow_lease_registered": False,
                        "lifecycle_recovery_reason": (
                            "missing_episode_background_metadata"
                        ),
                    }
                )
            lifecycle.update(
                {
                    "training_disposition": disposition,
                    "cancelled_task_group_id": task_id,
                    "cancelled_group_rollouts": expected_rollouts,
                    "cancelled_materialized_rollouts": materialized,
                    "cancelled_shadow_leases": cancelled_leases,
                }
            )
            episode.metadata["denovo_background_finalize"] = lifecycle
        return True

    def _persist_denovo_disposition_rollouts(
        self,
        task_id: str,
        episodes: list[Episode],
        disposition: str,
        *,
        cancelled_shadow_leases: int = 0,
        group_rollouts: int | None = None,
    ) -> bool:
        if not self._annotate_denovo_group_disposition(
            task_id,
            episodes,
            disposition,
            cancelled_shadow_leases=cancelled_shadow_leases,
            group_rollouts=group_rollouts,
        ):
            return False
        if self._dynamic_sampling_rollout_logger is None:
            return False
        self._log_dynamic_sampling_rollouts(
            episodes,
            filtered_groups=[],
            lifecycle={
                "schema_version": 1,
                "optimizer_step": None,
                "optimizer_committed": False,
                "training_disposition": disposition,
            },
        )
        return True

    async def _finalize_cancelled_rollouts(
        self,
        task_id: str,
        episodes: list[Episode],
    ) -> bool:
        if not episodes or not all(
            self.should_persist_cancelled_rollout(task_id, episode)
            for episode in episodes
        ):
            return False
        disposition = self._cancelled_task_dispositions.get(
            task_id,
            "trainer_shutdown"
            if self._generation_complete
            else "late_cancelled_rollout",
        )
        cancelled = await self._cancel_deferred_shadows(
            episodes,
            disposition,
        )
        persisted = self._persist_denovo_disposition_rollouts(
            task_id,
            episodes,
            disposition,
            cancelled_shadow_leases=cancelled,
        )
        if not persisted:
            self._discard_dynamic_sampling_rollouts(episodes)
        return True

    async def finalize_cancelled_rollout(
        self,
        task_id: str,
        episode: Episode,
    ) -> bool:
        """Cancel and persist one late DeNovo rollout after its log handoff."""

        return await self._finalize_cancelled_rollouts(task_id, [episode])

    @staticmethod
    def _group_signature(
        groups: list[TrajectoryGroup],
    ) -> tuple[tuple[str, tuple[tuple[str, str, str, int], ...]], ...]:
        return tuple(
            sorted(
                (
                    str(group.group_id),
                    tuple(
                        sorted(
                            (
                                str(
                                    group.metadata[index].get("task_id", "")
                                    if index < len(group.metadata)
                                    else ""
                                ),
                                str(
                                    group.metadata[index].get("rollout_idx", "")
                                    if index < len(group.metadata)
                                    else ""
                                ),
                                str(trajectory.name),
                                len(trajectory.steps),
                            )
                            for index, trajectory in enumerate(
                                group.trajectories
                            )
                        )
                    ),
                )
                for group in groups
            )
        )

    async def _finalize_deferred_batches(
        self,
        batches: list[TaskBatch],
    ) -> None:
        deferred = [batch for batch in batches if batch.deferred_shadow_finalize]
        if not deferred:
            return
        if len(deferred) != len(batches):
            raise RuntimeError(
                "one optimizer reservation mixes deferred DeNovo and legacy batches"
            )
        if self._deferred_shadow_finalizer is None:
            raise RuntimeError("deferred DeNovo batch has no shadow finalizer")
        episodes = [episode for batch in deferred for episode in batch.episodes]
        lifecycle_metrics = await self._deferred_shadow_finalizer(
            episodes,
            self._deferred_shadow_stall_timeout,
        )
        if lifecycle_metrics:
            self._aggregator.record_dict(lifecycle_metrics)

        for batch in deferred:
            groups, transform_metrics = transform_episodes_to_trajectory_groups(
                batch.episodes,
                self._transform_config,
                self._cf_config,
            )
            before_min_traj = len(groups)
            groups = [
                group
                for group in groups
                if len(group.trajectories) >= self._rs_config.min_trajs_per_group
            ]
            signature = self._group_signature(groups)
            if signature != batch.preflight_group_signature:
                raise RuntimeError(
                    "DeNovo trajectory topology changed across the shadow "
                    f"barrier: preflight={batch.preflight_group_signature!r} "
                    f"final={signature!r}"
                )
            advantage_metrics = (
                collect_reward_and_advantage_from_trajectory_groups(
                    groups,
                    self._algorithm_config,
                )
            )
            filtered_zero_adv = 0
            if self._rs_config.filter_uniform_groups:
                before_adv = len(groups)
                groups = [
                    group
                    for group in groups
                    if any(
                        abs(step.advantage) > 1e-8
                        for trajectory in group.trajectories
                        for step in trajectory.steps
                        if step.advantage is not None
                    )
                ]
                filtered_zero_adv = before_adv - len(groups)
            if not groups:
                raise RuntimeError(
                    "selected DeNovo optimizer group became empty after "
                    "batch-barrier advantage computation"
                )
            for group in groups:
                group.weight_version = batch.groups[0].weight_version
            batch.groups = groups
            self._record_processing_metrics(
                batch.rollout_log_episodes or batch.episodes,
                transform_metrics,
                dropped_min_trajs=before_min_traj - len(groups),
                advantage_metrics=advantage_metrics,
                dropped_zero_adv=filtered_zero_adv,
            )

    async def add_episode(
        self,
        task_id: str,
        episode: Episode,
        *,
        source_ticket: DataloaderBatchTicket | None = None,
    ) -> bool:
        """Add episode. When group completes, process and queue task batch."""
        if self._generation_complete or task_id in self._cancelled_task_ids:
            if not await self._finalize_cancelled_rollouts(task_id, [episode]):
                self._discard_dynamic_sampling_rollouts([episode])
            logger.warning("Ignoring episode for task %s after generation was marked complete", task_id)
            return False

        self._remember_source_ticket(task_id, source_ticket)

        # Offload episode to disk if enabled
        if self._episode_offload_dir:
            path = await self._offload_episode(task_id, episode)
            if task_id in self._cancelled_task_ids:
                try:
                    os.remove(path)
                except OSError:
                    pass
                if not await self._finalize_cancelled_rollouts(
                    task_id,
                    [episode],
                ):
                    self._discard_dynamic_sampling_rollouts([episode])
                return False
            self._pending.setdefault(task_id, []).append(path)
        else:
            self._pending.setdefault(task_id, []).append(episode)

        early_infrastructure_failure = self._infrastructure_failure([episode])
        if early_infrastructure_failure:
            self._write_infrastructure_failure_audit(task_id, episode, early_infrastructure_failure)
        failure = episode.metadata.get("infrastructure_failure")
        failure = failure if isinstance(failure, dict) else {}
        diagnostics = failure.get("diagnostics")
        diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
        if early_infrastructure_failure and (
            diagnostics.get("fatal") is True
            or failure.get("reason") in {"minisandbox_cleanup_unconfirmed", "minisandbox_cache_invalid", "verifier_contract_invalid"}
        ):
            from rllm.types import RolloutInfrastructureError

            self._coordinator.pause_generation()
            self.fail_generation(RolloutInfrastructureError(
                str(failure.get("reason") or "infrastructure_failure"),
                f"Fatal infrastructure contract for {task_id}: {early_infrastructure_failure}",
                retryable=False, stage=failure.get("stage"),
                retry_scope=str(failure.get("retry_scope") or "none"), diagnostics=diagnostics,
            ))
            return True
        if (
            early_infrastructure_failure is not None
            and self._dynamic_sampling_loader is not None
            and self._dynamic_sampling_wave_controller is not None
        ):
            self._infrastructure_failed_groups += 1
            claimed = await self._dynamic_sampling_wave_controller.fail_candidate_early(
                task_id,
                current_task=asyncio.current_task(),
            )
            if claimed:
                await self._cancel_deferred_shadows(
                    [episode],
                    "infrastructure_requeue",
                )
                logger.warning(
                    "Requeued task group %s immediately after rollout infrastructure "
                    "failure and cancelled sibling rollouts: event=%s",
                    task_id,
                    json.dumps(
                        self._infrastructure_failure_log_event(
                            task_id,
                            episode,
                        ),
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
                return True
            # A simultaneous final classification or wave-close owns the
            # ticket. Its cancellation path will discard this late episode.
            if task_id in self._cancelled_task_ids:
                return False

        if len(self._pending[task_id]) < self._group_size:
            return False

        # Load all episodes
        if self._episode_offload_dir:
            episodes = await self._load_pending_episodes(task_id)
        else:
            episodes = self._pending.pop(task_id, [])
        if task_id in self._cancelled_task_ids:
            if not await self._finalize_cancelled_rollouts(task_id, episodes):
                self._discard_dynamic_sampling_rollouts(episodes)
            return False
        source_ticket = self._source_tickets.pop(task_id, None)
        if source_ticket is None and self._dynamic_sampling_loader is not None:
            # A speculative cancellation acquired the source ticket while an
            # offloaded group was being deserialized.
            if not await self._finalize_cancelled_rollouts(task_id, episodes):
                self._discard_dynamic_sampling_rollouts(episodes)
            return False
        rollout_count = len(episodes)
        deferred_shadow_group = self._is_deferred_shadow_group(episodes)
        # filter_episodes mutates Episode.trajectories in place. Keep a shallow
        # logging view with the original trajectory list while sharing the
        # artifacts dict that owns the deferred log context.
        rollout_log_episodes = [
            episode.model_copy(
                update={"trajectories": list(episode.trajectories)},
                deep=False,
            )
            for episode in episodes
        ]

        if not self._dynamic_sampling_config.enable and any(
            self._is_denovo_background_episode(ep) for ep in episodes
        ):
            self._processing_deferred_groups[task_id] = TaskBatch(
                groups=[], task_id=task_id, episodes=episodes,
                rollout_log_episodes=rollout_log_episodes,
                source_ticket=source_ticket, deferred_shadow_finalize=True,
            )

        infrastructure_failure = self._infrastructure_failure(episodes)
        if infrastructure_failure is not None:
            if self._dynamic_sampling_loader is None or any(
                ep.metadata.get("infrastructure_failure", {}).get("reason") == "shadow_lifecycle_contract_invalid"
                for ep in episodes
            ):
                if task_id not in self._processing_deferred_groups:
                    self._pending[task_id] = rollout_log_episodes
                    self._source_tickets[task_id] = source_ticket
                error = GatewayInfrastructureError(
                    "infrastructure became unavailable while collecting task group "
                    f"{task_id}: {infrastructure_failure}"
                )
                # Keep the historical outer type, with the failed rollout's
                # structured cause available to the final failure manifest.
                from rllm.types import RolloutInfrastructureError
                failed = next(ep.metadata["infrastructure_failure"] for ep in episodes if ep.metadata.get("infrastructure_failure"))
                error.__cause__ = RolloutInfrastructureError(
                    str(failed.get("reason") or "infrastructure_failure"), infrastructure_failure,
                    retryable=False, stage=str(failed.get("stage") or "rollout"),
                    retry_scope="none", diagnostics=failed.get("diagnostics"),
                )
                self.fail_generation(error)
                return True
            disposition = self._complete_sampling_wave_candidate(
                task_id,
                accepted=False,
                infrastructure_failure=True,
            )
            cancelled_shadow_leases = await self._cancel_deferred_shadows(
                rollout_log_episodes,
                "infrastructure_requeue",
            )
            # Both an admitted failure and a wave-close race have identical
            # dataset semantics. DeNovo alone keeps a lifecycle-only rollout
            # audit after its shadow lease is cancelled; legacy profiles still
            # discard speculative JSON.
            self._return_infrastructure_group(
                task_id,
                source_ticket,
                rollout_log_episodes,
                rollout_count=rollout_count,
                cancelled_shadow_leases=cancelled_shadow_leases,
            )
            self._log_prompt_group_finished(
                task_id=task_id,
                episodes=episodes,
                status="infrastructure_failed",
                reason="infrastructure_unavailable",
                groups_after_transform=0,
                groups_after_min_trajs=0,
                groups_after_reward_filter=0,
            )
            logger.warning(
                "Requeued unclassified task group %s after infrastructure failure "
                "(wave_disposition=%s): event=%s",
                task_id,
                disposition,
                json.dumps(
                    self._infrastructure_failure_log_event(
                        task_id,
                        next(
                            (
                                episode
                                for episode in episodes
                                if isinstance(
                                    episode.metadata.get(
                                        "infrastructure_failure"
                                    ),
                                    dict,
                                )
                            ),
                            episodes[0],
                        ),
                    ),
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            return True

        weight_version = self._min_weight_version(episodes)
        dynamic_filtered_groups: list[TrajectoryGroup] = []
        filtered_trajectory_positions: set[tuple[int, int]] = set()

        # Transform first, but defer all classification metrics until the wave
        # owns this completion. A completion that loses the wave-close race is
        # intentionally invisible to accepted/filtered reward statistics.
        traj_groups, transform_metrics = transform_episodes_to_trajectory_groups(
            episodes,
            self._transform_config,
            self._cf_config,
        )
        # Drop groups with too few trajectories
        before_min_traj = len(traj_groups)
        traj_groups = [g for g in traj_groups if len(g.trajectories) >= self._rs_config.min_trajs_per_group]
        after_min_traj = len(traj_groups)

        if not traj_groups:
            if self._complete_sampling_wave_candidate(
                task_id,
                accepted=False,
            ) == "surplus":
                cancelled_shadow_leases = await self._cancel_deferred_shadows(
                    rollout_log_episodes,
                    "surplus",
                )
                self._return_speculative_group(
                    task_id,
                    source_ticket,
                    rollout_log_episodes,
                    rollout_count=rollout_count,
                    cancelled_shadow_leases=cancelled_shadow_leases,
                )
                return True
            cancelled_shadow_leases = await self._cancel_deferred_shadows(
                rollout_log_episodes,
                "compact_filtered",
            )
            self._annotate_denovo_group_disposition(
                task_id,
                rollout_log_episodes,
                "compact_filtered",
                cancelled_shadow_leases=cancelled_shadow_leases,
            )
            self._record_processing_metrics(
                rollout_log_episodes,
                transform_metrics,
                dropped_min_trajs=before_min_traj - after_min_traj,
            )
            self._log_dynamic_sampling_rollouts(
                rollout_log_episodes,
                filtered_groups=[],
                lifecycle={
                    "schema_version": 1,
                    "optimizer_step": None,
                    "optimizer_committed": False,
                    "training_disposition": "consumed_other",
                },
            )
            if before_min_traj > 0:
                filter_reason = "min_trajs"
            elif self._all_episodes_compact_filtered(episodes):
                filter_reason = "compact_filtering"
            else:
                filter_reason = "no_trajectory_groups"
            self._log_prompt_group_finished(
                task_id=task_id,
                episodes=episodes,
                status="filtered",
                reason=filter_reason,
                groups_after_transform=before_min_traj,
                groups_after_min_trajs=0,
                groups_after_reward_filter=0,
            )
            self._coordinator.on_group_filtered()
            self._resolve_source_ticket(
                source_ticket,
                rollout_count=rollout_count,
            )
            self._filtered_count += 1
            self._record_classified_prompt_group()
            self._processing_deferred_groups.pop(task_id, None)
            return True

        # This happens before advantage computation by design: only verifier
        # outcome rewards decide whether a group is dynamically sampled again.
        if self._dynamic_sampling_config.enable:
            partition = partition_uniform_outcome_groups(
                traj_groups,
                outcome_mode=self._dynamic_sampling_config.outcome_mode,
                easy_pass_rate_threshold=(
                    self._dynamic_sampling_config.easy_pass_rate_threshold
                ),
                hard_pass_rate_threshold=(
                    self._dynamic_sampling_config.hard_pass_rate_threshold
                ),
            )
            dynamic_filtered_groups = partition.filtered
            dynamic_filtered_ids = {
                id(trajectory)
                for group in partition.filtered
                for trajectory in group.trajectories
            }
            filtered_trajectory_positions = {
                (episode_index, trajectory_index)
                for episode_index, episode in enumerate(rollout_log_episodes)
                for trajectory_index, trajectory in enumerate(
                    episode.trajectories
                )
                if id(trajectory) in dynamic_filtered_ids
            }
            if partition.filtered:
                episodes = filter_episodes(episodes, partition.filtered)
                traj_groups = partition.kept
            if partition.task_filtered:
                if self._complete_sampling_wave_candidate(
                    task_id,
                    accepted=False,
                    true_uniform=partition.uniform_pass_count,
                    filter_reasons=partition.filter_reasons,
                ) == "surplus":
                    cancelled_shadow_leases = await self._cancel_deferred_shadows(
                        rollout_log_episodes,
                        "surplus",
                    )
                    self._return_speculative_group(
                        task_id,
                        source_ticket,
                        rollout_log_episodes,
                        rollout_count=rollout_count,
                        cancelled_shadow_leases=cancelled_shadow_leases,
                    )
                    return True
                cancelled_shadow_leases = await self._cancel_deferred_shadows(
                    rollout_log_episodes,
                    "dynamic_filtered",
                )
                self._annotate_denovo_group_disposition(
                    task_id,
                    rollout_log_episodes,
                    "dynamic_filtered",
                    cancelled_shadow_leases=cancelled_shadow_leases,
                )
                self._record_processing_metrics(
                    rollout_log_episodes,
                    transform_metrics,
                    dropped_min_trajs=before_min_traj - after_min_traj,
                    dropped_uniform_outcome=len(partition.filtered),
                )
                self._log_dynamic_sampling_rollouts(
                    rollout_log_episodes,
                    filtered_groups=partition.filtered,
                    lifecycle={
                        "schema_version": 1,
                        "optimizer_step": None,
                        "optimizer_committed": False,
                        "training_disposition": "uniform_filtered",
                        "dynamic_sampling_filter_reasons": list(
                            partition.filter_reasons
                        ),
                    },
                )
                resolution = self._dynamic_sampling_loader.record_filtered_outcome(
                    source_ticket,
                    uniform=partition.uniform_pass_count,
                    easy=partition.easy,
                    hard=partition.hard,
                    rollout_count=rollout_count,
                )
                self._log_prompt_group_finished(
                    task_id=task_id,
                    episodes=episodes,
                    status="filtered",
                    reason=(
                        "dynamic_sampling_"
                        + "_".join(partition.filter_reasons)
                    ),
                    groups_after_transform=before_min_traj,
                    groups_after_min_trajs=len(partition.filtered),
                    groups_after_reward_filter=0,
                )
                self._coordinator.on_group_filtered()
                self._filtered_count += 1
                self._record_classified_prompt_group(
                    final=resolution.status == "dropped"
                )
                return True
        # 4. Compute advantages. DeNovo milestone evidence remains mutable
        # until the selected optimizer-batch barrier, so its advantage pass is
        # deliberately deferred to get_many().
        adv_metrics: dict | None = None
        filtered_zero_adv = 0
        if not deferred_shadow_group:
            adv_metrics = collect_reward_and_advantage_from_trajectory_groups(
                traj_groups,
                self._algorithm_config,
            )
            # 5. Rejection sampling: drop groups with all-zero advantage
            if self._rs_config.filter_uniform_groups:
                before_adv = len(traj_groups)
                traj_groups = [g for g in traj_groups if any(abs(step.advantage) > 1e-8 for traj in g.trajectories for step in traj.steps if step.advantage is not None)]
                filtered_zero_adv = before_adv - len(traj_groups)
        if not traj_groups:
            if self._complete_sampling_wave_candidate(
                task_id,
                accepted=False,
            ) == "surplus":
                cancelled_shadow_leases = await self._cancel_deferred_shadows(
                    rollout_log_episodes,
                    "surplus",
                )
                self._return_speculative_group(
                    task_id,
                    source_ticket,
                    rollout_log_episodes,
                    rollout_count=rollout_count,
                    cancelled_shadow_leases=cancelled_shadow_leases,
                )
                return True
            cancelled_shadow_leases = await self._cancel_deferred_shadows(
                rollout_log_episodes,
                "reward_filtered",
            )
            self._annotate_denovo_group_disposition(
                task_id,
                rollout_log_episodes,
                "reward_filtered",
                cancelled_shadow_leases=cancelled_shadow_leases,
            )
            self._record_processing_metrics(
                rollout_log_episodes,
                transform_metrics,
                dropped_min_trajs=before_min_traj - after_min_traj,
                dropped_uniform_outcome=len(dynamic_filtered_groups),
                advantage_metrics=adv_metrics,
                dropped_zero_adv=filtered_zero_adv,
            )
            self._log_dynamic_sampling_rollouts(
                rollout_log_episodes,
                filtered_groups=dynamic_filtered_groups,
                lifecycle={
                    "schema_version": 1,
                    "optimizer_step": None,
                    "optimizer_committed": False,
                    "training_disposition": "consumed_other",
                },
            )
            self._log_prompt_group_finished(
                task_id=task_id,
                episodes=episodes,
                status="filtered",
                reason="uniform_reward",
                groups_after_transform=before_min_traj,
                groups_after_min_trajs=before_adv,
                groups_after_reward_filter=0,
            )
            self._coordinator.on_group_filtered()
            self._resolve_source_ticket(
                source_ticket,
                rollout_count=rollout_count,
            )
            self._filtered_count += 1
            self._record_classified_prompt_group()
            self._processing_deferred_groups.pop(task_id, None)
            return True

        # 6. Set weight version and queue
        if self._complete_sampling_wave_candidate(
            task_id,
            accepted=True,
        ) == "surplus":
            cancelled_shadow_leases = await self._cancel_deferred_shadows(
                rollout_log_episodes,
                "surplus",
            )
            self._return_speculative_group(
                task_id,
                source_ticket,
                rollout_log_episodes,
                rollout_count=rollout_count,
                cancelled_shadow_leases=cancelled_shadow_leases,
            )
            return True

        if not deferred_shadow_group:
            self._record_processing_metrics(
                rollout_log_episodes,
                transform_metrics,
                dropped_min_trajs=before_min_traj - after_min_traj,
                dropped_uniform_outcome=len(dynamic_filtered_groups),
                advantage_metrics=adv_metrics,
                dropped_zero_adv=filtered_zero_adv,
            )
        for g in traj_groups:
            g.weight_version = weight_version

        if self._dynamic_sampling_loader is not None:
            self._dynamic_sampling_loader.mark_accepted(
                source_ticket,
                rollout_count=rollout_count,
            )

        batch = TaskBatch(
            groups=traj_groups,
            task_id=task_id,
            episodes=episodes,
            source_ticket=source_ticket,
            rollout_log_episodes=rollout_log_episodes,
            filtered_trajectory_positions=filtered_trajectory_positions,
            deferred_shadow_finalize=deferred_shadow_group,
            preflight_group_signature=self._group_signature(traj_groups),
            preflight_transform_metrics=dict(transform_metrics),
            preflight_dropped_min_trajs=before_min_traj - after_min_traj,
        )
        if self._tg_offload_dir:
            await self._queue.put(await self._offload_task_batch(batch))
        else:
            await self._queue.put(batch)
        self._processing_deferred_groups.pop(task_id, None)
        self._training_queue_size += 1
        self._queue_update_event.set()
        self._record_classified_prompt_group()

        self._log_prompt_group_finished(
            task_id=task_id,
            episodes=episodes,
            status="queued",
            reason="accepted",
            groups_after_transform=before_min_traj,
            groups_after_min_trajs=len(traj_groups) + filtered_zero_adv,
            groups_after_reward_filter=len(traj_groups),
        )

        return True

    @staticmethod
    def _compact_shadow_failure_diagnostics(value: object) -> dict:
        if not isinstance(value, dict):
            return {}
        result: dict[str, object] = {
            key: value[key]
            for key in (
                "schema_version",
                "failure_counts",
                "queue_depth_at_finalize",
                "pending_events_at_timeout",
            )
            if key in value
        }
        recovery = value.get("final_recovery")
        if isinstance(recovery, dict):
            result["final_recovery"] = {
                key: recovery[key]
                for key in (
                    "status",
                    "attempts",
                    "final_repo_fingerprint",
                )
                if key in recovery
            }
            recovery_errors = [
                str(recovery.get("original_failure") or ""),
                *(
                    str(item)
                    for item in recovery.get("errors", [])
                    if isinstance(item, str)
                ),
            ]
            recovery_errors = [item for item in recovery_errors if item]
            if recovery_errors:
                result["final_recovery"]["error_digests"] = [
                    {
                        "chars": len(item),
                        "sha256": hashlib.sha256(
                            item.encode("utf-8", errors="surrogatepass")
                        ).hexdigest(),
                    }
                    for item in recovery_errors[:3]
                ]
        final_event = value.get("final_test_event")
        if isinstance(final_event, dict):
            # Never persist verifier names, result vectors, output tails, or
            # hidden test content in the infrastructure audit.
            result["final_test_event"] = {
                key: final_event[key]
                for key in (
                    "status",
                    "trusted",
                    "failure_type",
                    "timed_out",
                    "result_completeness",
                    "restore_confirmation",
                )
                if key in final_event
            }
        replay = value.get("terminal_replay")
        if isinstance(replay, dict):
            result["terminal_replay"] = {
                key: replay[key]
                for key in (
                    "action_id",
                    "turn_id",
                    "tool_name",
                    "failure_type",
                    "expected_repo_state_before",
                    "expected_repo_state_after",
                    "observed_repo_state_before",
                    "observed_repo_state_after",
                    "state_mismatch_paths",
                    "changed_paths",
                )
                if key in replay
            }
        return result

    @staticmethod
    def _compact_helper_failure_diagnostics(value):
        if not isinstance(value, dict):
            return None
        from rllm.utils.infrastructure_diagnostics import compact_runtime_diagnostics

        keys = ("interpreter", "script_sha256", "operation_kind", "phase", "exit_code", "error_kind", "error_type", "sandbox", "operation", "failure_scope", "fatal",
                "node_id", "operation_id", "task_id", "image", "create_error", "cleanup_error", "operation_status",
                "cleanup_budget_seconds", "cleanup_queue_stall_seconds", "close_submissions", "kernel_release",
                "launcher", "namespace_init", "cgroups", "pressure", "blocked_at", "stage", "elapsed_seconds",
                "deadline_stage", "stop_confirmed", "exception_chain", "remote_diagnostics",
                "worker_stop_confirmed", "remaining_budget_seconds", "deadline_overrun_seconds",
                "stop_grace_seconds", "rollout_progress", "trace_fetch_seconds",
                "process_audit", "deadline", "verifier_status")
        result = {key: compact_runtime_diagnostics(value[key]) for key in keys if key in value}
        if value.get("stderr"):
            result["stderr"] = str(value["stderr"])[-1000:]
        probes = value.get("interpreter_probes")
        if isinstance(probes, list):
            result["interpreter_probes"] = [
                {"candidate": str(probe.get("candidate", ""))[:256],
                 "error": str(probe.get("error", ""))[-1000:],
                 "diagnostics": TrajectoryGroupBuffer._compact_helper_failure_diagnostics(probe.get("diagnostics"))}
                for probe in probes[:5] if isinstance(probe, dict)
            ]
        return compact_runtime_diagnostics(result) if result else None

    def _write_infrastructure_failure_audit(
        self,
        task_id: str,
        episode: Episode,
        reason: str,
    ) -> None:
        directory = self._infrastructure_audit_dir
        if not directory:
            return
        metadata = episode.metadata if isinstance(episode.metadata, dict) else {}
        failure = metadata.get("infrastructure_failure")
        failure = failure if isinstance(failure, dict) else {}
        raw_error = str(
            failure.get("error_summary")
            or failure.get("error")
            or failure.get("message")
            or ""
        )
        checked_paths = failure.get("checked_paths")
        checked_paths = (
            [str(path)[:512] for path in checked_paths[:32]]
            if isinstance(checked_paths, list)
            else []
        )
        payload = {
            "schema_version": 2,
            "task_group_id": task_id,
            "rollout_id": episode.id,
            "reason": str(failure.get("reason") or reason.split("/", 1)[0]),
            "stage": failure.get("stage"),
            "task_id": failure.get("task_id") or (
                episode.task.get("id") if isinstance(episode.task, dict)
                else episode.task if isinstance(episode.task, str)
                else getattr(episode.task, "id", None)
            ),
            "attempts": failure.get("attempts"),
            "verifier_status": failure.get("verifier_status"),
            "sandbox_health": failure.get("sandbox_health"),
            "sandbox_alive": failure.get("sandbox_alive"),
            "duration_seconds": failure.get("duration_seconds"),
            "checked_paths": checked_paths,
            "exception_type": (
                failure.get("exception_type")
                or failure.get("error_type")
            ),
            # Verifier transport/contract summaries are already bounded by
            # ScriptEvaluator. Shadow failures may contain hidden test names
            # or output and therefore remain digest-only.
            "error_summary": (
                raw_error[:2000]
                if raw_error and failure.get("stage") in {"verifier", "agentflow"}
                else None
            ),
            "error_digest": (
                {
                    "chars": len(raw_error),
                    "sha256": hashlib.sha256(
                        raw_error.encode("utf-8", errors="surrogatepass")
                    ).hexdigest(),
                }
                if raw_error
                else None
            ),
            "retry_scope": failure.get("retry_scope"),
            "rollout_lifecycle_dispatch": metadata.get(
                "rollout_lifecycle_dispatch"
            ),
            "timings": {
                key: value
                for key, value in (episode.metrics or {}).items()
                if isinstance(key, str)
                and key.startswith("time/")
                and isinstance(value, int | float)
            },
            "shadow_diagnostics": self._compact_shadow_failure_diagnostics(
                failure.get("diagnostics")
            ),
            "helper_diagnostics": self._compact_helper_failure_diagnostics(failure.get("diagnostics")),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > 64 * 1024:
            payload["shadow_diagnostics"] = {
                "truncated": True,
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
            encoded = json.dumps(
                payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "-", task_id)[:80]
        safe_rollout = re.sub(
            r"[^A-Za-z0-9_.-]+", "-", str(episode.id or "unknown")
        )[:80]
        path = os.path.join(
            directory,
            f"{time.time_ns():020d}_{safe_task}_{safe_rollout}.json",
        )
        temporary = path + ".tmp"
        try:
            with open(temporary, "wb") as handle:
                handle.write(encoded)
            os.replace(temporary, path)
            self._infrastructure_audit_paths.append(path)
            while len(self._infrastructure_audit_paths) > 10_000:
                oldest = self._infrastructure_audit_paths.popleft()
                try:
                    os.remove(oldest)
                except FileNotFoundError:
                    pass
        except Exception:
            logger.exception(
                "Failed to write compact infrastructure audit for group %s",
                task_id,
            )
            try:
                os.remove(temporary)
            except FileNotFoundError:
                pass

    def discard_if_cancelled(self, task_id: str, episode: Episode) -> bool:
        """Discard a late detached rollout before any external logging occurs."""
        if self._generation_complete or task_id in self._cancelled_task_ids:
            # The trainer must first install the deferred rollout-log context
            # for an opt-in DeNovo episode, then await
            # finalize_cancelled_rollout(). Legacy profiles preserve the
            # historical no-JSON fast path below.
            if self.should_persist_cancelled_rollout(task_id, episode):
                return False
            if self._deferred_shadow_canceller is not None:
                cleanup = asyncio.create_task(
                    self._cancel_deferred_shadows(
                        [episode],
                        "late_cancelled_rollout",
                    ),
                    name=f"cancel-deferred-shadow:{episode.id}",
                )
                self._detached_shadow_cleanup_tasks.add(cleanup)

                def cleanup_done(done: asyncio.Task[Any]) -> None:
                    self._detached_shadow_cleanup_tasks.discard(done)
                    if not done.cancelled():
                        try:
                            done.exception()
                        except BaseException:
                            pass

                cleanup.add_done_callback(cleanup_done)
            self._discard_dynamic_sampling_rollouts([episode])
            return True
        return False

    def _record_processing_metrics(
        self,
        episodes: list[Episode],
        transform_metrics: dict,
        *,
        dropped_min_trajs: int,
        dropped_uniform_outcome: int = 0,
        advantage_metrics: dict | None = None,
        dropped_zero_adv: int = 0,
    ) -> None:
        self._record_episode_metrics(episodes)
        self._aggregator.record_dict(transform_metrics)
        self._aggregator.record("groups/dropped_min_trajs", dropped_min_trajs)
        if self._dynamic_sampling_config.enable:
            self._aggregator.record(
                "groups/dropped_uniform_outcome",
                dropped_uniform_outcome,
            )
        if advantage_metrics is not None:
            self._aggregator.record_dict(advantage_metrics)
            self._aggregator.record("groups/dropped_zero_adv", dropped_zero_adv)

    def _complete_sampling_wave_candidate(
        self,
        task_id: str,
        *,
        accepted: bool,
        true_uniform: bool = False,
        filter_reasons: tuple[str, ...] = (),
        infrastructure_failure: bool = False,
    ) -> str:
        if self._dynamic_sampling_wave_controller is None:
            return "accepted" if accepted else "failed"
        return self._dynamic_sampling_wave_controller.complete_candidate(
            task_id,
            accepted=accepted,
            true_uniform=true_uniform,
            filter_reasons=filter_reasons,
            infrastructure_failure=infrastructure_failure,
        )

    def _return_infrastructure_group(
        self,
        task_id: str,
        source_ticket: DataloaderBatchTicket | None,
        episodes: list[Episode],
        *,
        rollout_count: int,
        cancelled_shadow_leases: int = 0,
    ) -> None:
        self._persist_denovo_disposition_rollouts(
            task_id,
            episodes,
            "infrastructure_requeue",
            cancelled_shadow_leases=cancelled_shadow_leases,
        )
        discarded = self._discard_dynamic_sampling_rollouts(episodes)
        if self._dynamic_sampling_wave_controller is not None:
            self._dynamic_sampling_wave_controller.note_discarded_json(discarded)
        if self._dynamic_sampling_loader is not None:
            self._dynamic_sampling_loader.return_unconsumed(
                source_ticket,
                rollout_count=rollout_count,
                defer_until_next_wave=False,
            )
        self._coordinator.on_group_filtered()
        self._infrastructure_failed_groups += 1
        self._record_classified_prompt_group(final=False)

    def _return_speculative_group(
        self,
        task_id: str,
        source_ticket: DataloaderBatchTicket | None,
        episodes: list[Episode],
        *,
        rollout_count: int,
        cancelled_shadow_leases: int = 0,
    ) -> None:
        self._persist_denovo_disposition_rollouts(
            task_id,
            episodes,
            "surplus",
            cancelled_shadow_leases=cancelled_shadow_leases,
        )
        discarded = self._discard_dynamic_sampling_rollouts(episodes)
        if self._dynamic_sampling_wave_controller is not None:
            self._dynamic_sampling_wave_controller.note_discarded_json(discarded)
        if self._dynamic_sampling_loader is not None:
            self._dynamic_sampling_loader.return_unconsumed(
                source_ticket,
                rollout_count=rollout_count,
            )
        self._coordinator.on_group_filtered()
        self._record_classified_prompt_group(final=False)

    def _log_dynamic_sampling_rollouts(
        self,
        episodes: list[Episode],
        *,
        filtered_groups: list[TrajectoryGroup],
        lifecycle: dict | None = None,
    ) -> None:
        if self._dynamic_sampling_rollout_logger is None:
            return
        if not self._dynamic_sampling_config.enable:
            episodes = [ep for ep in episodes if self._is_denovo_background_episode(ep)]
            if not episodes:
                return
        self._dynamic_sampling_rollout_logger(
            episodes,
            {
                id(trajectory)
                for group in filtered_groups
                for trajectory in group.trajectories
            },
            lifecycle=lifecycle,
        )

    def _discard_dynamic_sampling_rollouts(
        self,
        episodes: list[Episode],
    ) -> int:
        if self._dynamic_sampling_rollout_discarder is None:
            return 0
        return int(self._dynamic_sampling_rollout_discarder(episodes))

    async def cancel_speculative_group(
        self,
        task_id: str,
        *,
        defer_until_next_wave: bool = True,
    ) -> tuple[int, int]:
        """Cancel one wave candidate and return (completed rollouts, logs)."""
        disposition = (
            "surplus"
            if defer_until_next_wave
            else "infrastructure_requeue"
        )
        self._cancelled_task_ids.add(task_id)
        self._cancelled_task_dispositions[task_id] = disposition
        items = self._pending.pop(task_id, [])
        episodes = [item for item in items if not isinstance(item, str)]
        paths = [item for item in items if isinstance(item, str)]
        source_ticket = self._source_tickets.pop(task_id, None)
        # Ticket ownership and coordinator quota are released before local
        # offload cleanup. A slow filesystem must never become a wave barrier.
        if source_ticket is not None:
            if self._dynamic_sampling_loader is not None:
                self._dynamic_sampling_loader.return_unconsumed(
                    source_ticket,
                    rollout_count=len(items),
                    defer_until_next_wave=defer_until_next_wave,
                )
            self._coordinator.on_group_filtered()
            self._record_classified_prompt_group(final=False)

        if paths and self._deferred_shadow_canceller is not None:
            loaded = await asyncio.gather(
                *(asyncio.to_thread(self._pickle_load, path) for path in paths)
            )
            episodes.extend(loaded)
            paths = []

        cancelled_shadow_leases = await self._cancel_deferred_shadows(
            episodes,
            disposition,
        )

        if paths:
            cleanup_task = asyncio.create_task(
                asyncio.to_thread(self._remove_offloaded_episodes, paths),
                name=f"discard-offloaded-rollouts:{task_id}",
            )
            self._detached_file_cleanup_tasks.add(cleanup_task)

            def cleanup_done(done: asyncio.Task[None]) -> None:
                self._detached_file_cleanup_tasks.discard(done)
                if not done.cancelled():
                    try:
                        done.exception()
                    except BaseException:
                        pass

            cleanup_task.add_done_callback(cleanup_done)

        self._persist_denovo_disposition_rollouts(
            task_id,
            episodes,
            disposition,
            cancelled_shadow_leases=cancelled_shadow_leases,
        )
        discarded = self._discard_dynamic_sampling_rollouts(episodes)
        if self._dynamic_sampling_rollout_discarder is not None:
            # Offloaded dynamic-sampling episodes contain exactly one deferred
            # rollout context apiece; deleting the pickle is the discard.
            discarded += len(paths)
        if source_ticket is None:
            # Classification won the race and owns ticket disposition.
            return len(items), discarded
        return len(items), discarded

    @staticmethod
    def _remove_offloaded_episodes(paths: list[str]) -> None:
        for path in paths:
            try:
                os.remove(path)
            except FileNotFoundError:
                continue

    async def get(self) -> TaskBatch | None:
        """Get next task batch. Returns None when generation is done and buffer is drained."""
        self.raise_if_generation_failed()
        item = await self._queue.get()
        self.raise_if_generation_failed()
        if item is None:
            return None
        self._training_queue_size = max(0, self._training_queue_size - 1)
        self._consumed_count += 1
        self._refresh_pbar_counters()
        batch = await self._load_task_batch(item)
        if batch.deferred_shadow_finalize and not self._dynamic_sampling_config.enable:
            self._deferred_reservations[id(batch)] = batch
        await self._finalize_deferred_batches([batch])
        return batch

    async def get_many(self, count: int) -> list[TaskBatch] | None:
        """Get a full forward/backward chunk, or None if generation ended first."""
        while self._training_queue_size < count:
            self.raise_if_generation_failed()
            if self._generation_complete:
                return None
            self._queue_update_event.clear()
            if self._training_queue_size >= count or self._generation_complete:
                continue
            await self._queue_update_event.wait()

        self.raise_if_generation_failed()

        items = []
        for _ in range(count):
            item = await self._queue.get()
            if item is None:
                return None
            batch = await self._load_task_batch(item)
            items.append(batch)
            self._training_queue_size = max(0, self._training_queue_size - 1)
            if batch.deferred_shadow_finalize and not self._dynamic_sampling_config.enable:
                self._deferred_reservations[id(batch)] = batch

        await self._finalize_deferred_batches(items)

        self._consumed_count += count
        self._refresh_pbar_counters()
        return items

    def release_deferred_reservation(self, batches: list[TaskBatch]) -> None:
        """Release audit ownership only after optimizer/ticket commit."""
        for batch in batches:
            self._deferred_reservations.pop(id(batch), None)

    async def cleanup_deferred_uncommitted(self, *, disposition: str) -> list[TaskBatch]:
        """After producer drain, cancel and audit ordinary-sampling leftovers.

        Source tickets remain unresolved so a checkpoint can replay them.
        """
        self._generation_complete = True
        batches = list(self._deferred_reservations.values())
        batches.extend(self._processing_deferred_groups.values())
        while self._training_queue_size > 0:
            item = self._queue.get_nowait()
            if item is None or item is _FATAL_GENERATION_SENTINEL:
                continue
            batch = await self._load_task_batch(item)
            self._training_queue_size -= 1
            batches.append(batch)
        for task_id in list(self._pending):
            episodes = await self._load_pending_episodes(task_id)
            if not episodes:
                continue
            batches.append(TaskBatch(
                groups=[], task_id=task_id, episodes=episodes,
                source_ticket=self._source_tickets.get(task_id),
                deferred_shadow_finalize=True,
            ))
        first_error: BaseException | None = None
        for batch in batches:
            self._deferred_reservations[id(batch)] = batch
        for batch in batches:
            episodes = batch.rollout_log_episodes or batch.episodes
            task_id = batch.task_id or str(episodes[0].id).rsplit(":", 1)[0]
            try:
                cancelled = await self._cancel_deferred_shadows(episodes, disposition)
            except BaseException as exc:
                cancelled = 0
                if first_error is None:
                    first_error = exc
                for episode in episodes:
                    lifecycle = self._denovo_background_lifecycle(episode)
                    if lifecycle is not None:
                        lifecycle["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            try:
                # An earlier optimizer/weight-sync failure may already own
                # the persisted disposition; do not overwrite that evidence.
                remaining = [
                    ep for ep in episodes
                    if not (self._denovo_background_lifecycle(ep) or {}).get("optimizer_committed")
                    and (self._denovo_background_lifecycle(ep) or {}).get("training_disposition")
                    not in {"optimizer_failed", "weight_sync_failed"}
                ]
                if remaining:
                    self._persist_denovo_disposition_rollouts(
                        task_id, remaining, disposition,
                        cancelled_shadow_leases=cancelled,
                        group_rollouts=len(episodes),
                    )
                self._deferred_reservations.pop(id(batch), None)
                self._processing_deferred_groups.pop(task_id, None)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error
        return batches

    async def drain_untrained_tail(
        self,
        *,
        disposition: str = "untrained_tail",
    ) -> list[TaskBatch]:
        """Remove accepted batches that will not reach another optimizer step."""
        batches: list[TaskBatch] = []
        while self._training_queue_size > 0:
            item = self._queue.get_nowait()
            if item is None or item is _FATAL_GENERATION_SENTINEL:
                continue
            batches.append(await self._load_task_batch(item))
            self._training_queue_size -= 1
        if not self._dynamic_sampling_config.enable:
            for batch in batches:
                if batch.deferred_shadow_finalize:
                    self._deferred_reservations[id(batch)] = batch
        for batch in batches:
            if not batch.deferred_shadow_finalize:
                continue
            cancelled_shadow_leases = await self._cancel_deferred_shadows(
                batch.episodes,
                disposition,
            )
            task_id = getattr(batch, "task_id", None) or (
                str(batch.episodes[0].id).rsplit(":", 1)[0]
                if batch.episodes
                else "<unknown-tail-group>"
            )
            self._annotate_denovo_group_disposition(
                task_id,
                batch.episodes,
                disposition,
                cancelled_shadow_leases=cancelled_shadow_leases,
                group_rollouts=len(batch.episodes),
            )
        self._queue_update_event.set()
        self._refresh_pbar_counters()
        return batches

    def finish_generation(
        self,
        kind: GenerationTerminalKind = "dataset_exhausted",
    ) -> None:
        """Signal a non-fatal producer stop and expose ordinary EOF.

        Only a true dataset exhaustion or an explicit controlled stop may use
        this path.  Fatal errors use :meth:`fail_generation` and preserve all
        pending/accepted ticket ownership for checkpoint replay.
        """
        if kind == "fatal_error":
            raise ValueError("fatal generation must use fail_generation(error)")
        if self._generation_complete:
            return
        self._generation_complete = True
        self._generation_terminal_kind = kind
        for task_id in list(self._pending.keys()):
            items = self._pending.pop(task_id, [])
            for item in items:
                if isinstance(item, str):
                    try:
                        os.remove(item)
                    except OSError:
                        pass
            self._coordinator.on_group_filtered()
            # Do not resolve incomplete/cancelled groups. A checkpoint made
            # before shutdown must replay them after resume.
            self._source_tickets.pop(task_id, None)
            self._filtered_count += 1
            self._record_classified_prompt_group()
        self._queue.put_nowait(None)
        self._queue_update_event.set()

    def fail_generation(self, error: BaseException) -> None:
        """Wake consumers with a fatal producer error without consuming work."""
        if self._generation_error is None:
            self._generation_error = error
        self._generation_terminal_kind = "fatal_error"
        self._generation_complete = True
        # Wake callers blocked directly in queue.get(); get_many is woken by
        # the event below and checks the same fatal state.
        self._queue.put_nowait(_FATAL_GENERATION_SENTINEL)
        self._queue_update_event.set()

    def raise_if_generation_failed(self) -> None:
        if self._generation_error is not None:
            raise AsyncGenerationError(self._generation_error) from self._generation_error

    def mark_generation_complete(self) -> None:
        """Backward-compatible alias for an ordinary dataset exhaustion."""
        self.finish_generation("dataset_exhausted")

    def stats(self) -> dict:
        return {
            "async/buffer_qsize": self._training_queue_size,
            "async/buffer_pending": len(self._pending),
            "async/buffer_filtered": self._filtered_count,
            "async/buffer_consumed": self._consumed_count,
            "async/buffer_generation_failed": int(
                self._generation_terminal_kind == "fatal_error"
            ),
            "rollout/infrastructure_failed_groups": (
                self._infrastructure_failed_groups
            ),
        }

    @staticmethod
    def _infrastructure_failure_log_event(
        task_group_id: str,
        episode: Episode,
    ) -> dict[str, object]:
        raw = episode.metadata.get("infrastructure_failure")
        raw = raw if isinstance(raw, dict) else {}
        summary = str(
            raw.get("error_summary") or raw.get("error") or raw.get("message") or ""
        )
        checked_paths = raw.get("checked_paths")
        checked_paths = (
            [str(path)[:256] for path in checked_paths[:16]]
            if isinstance(checked_paths, list)
            else []
        )
        return {
            "schema_version": 1,
            "task_group_id": task_group_id,
            "rollout_id": episode.id,
            "task_id": raw.get("task_id"),
            "reason": raw.get("reason"),
            "stage": raw.get("stage"),
            "verifier_status": raw.get("verifier_status"),
            "sandbox_health": raw.get("sandbox_health"),
            "duration_seconds": raw.get("duration_seconds"),
            "checked_paths": checked_paths,
            "exception_type": (
                raw.get("exception_type") or raw.get("error_type")
            ),
            "error_summary": (
                summary[:1000]
                if summary and raw.get("stage") in {"verifier", "agentflow"}
                else None
            ),
            "error_sha256": (
                hashlib.sha256(
                    summary.encode("utf-8", errors="surrogatepass")
                ).hexdigest()
                if summary
                else None
            ),
        }

    @staticmethod
    def _infrastructure_failure(episodes: list[Episode]) -> str | None:
        """Return the first explicit unusable-rollout contract in a group.

        The producer is responsible for distinguishing implementation errors
        from infrastructure failures.  Once the structured contract exists,
        limiting this consumer to gateway-only reasons is unsafe: verifier and
        Sandbox outages would otherwise become reward-zero training samples.
        """

        for episode in episodes:
            raw = episode.metadata.get("infrastructure_failure")
            if not isinstance(raw, dict):
                if (getattr(episode, "termination_reason", None) != TerminationReason.ERROR
                        or getattr(episode, "trajectories", None)):
                    continue
                # Older/custom producers can omit the structured contract.
                # A trajectory-less error is unusable, never a compact filter.
                raw = {
                    "schema_version": 1, "reason": "rollout_error",
                    "stage": "rollout", "error_type": "UnclassifiedRolloutError",
                    "error": str(episode.metadata.get("error") or "rollout terminated with error")[:2000],
                }
                episode.metadata["infrastructure_failure"] = raw
            reason = str(raw.get("reason") or "")
            error_type = str(
                raw.get("exception_type")
                or raw.get("error_type")
                or ""
            )
            summary = str(
                raw.get("error_summary")
                or raw.get("error")
                or raw.get("message")
                or ""
            )[:1000]
            if reason:
                return (
                    f"{reason}/{error_type or 'unknown'}: "
                    f"{summary}"
                )
        return None

    def _record_episode_metrics(self, episodes: list[Episode]) -> None:
        """Record episode-level metrics to aggregator (all episodes, including filtered)."""
        for ep in episodes:
            reason = ep.termination_reason or TerminationReason.UNKNOWN
            for r in TerminationReason:
                self._aggregator.record(
                    f"episode/termination_reason/{r.value}",
                    1.0 if reason == r else 0.0,
                )
            for k, v in ep.metrics.items():
                try:
                    self._aggregator.record(f"episode/{k}", float(v))
                except (TypeError, ValueError):
                    continue

            # Episode-level totals across all trajectories
            total_turns = sum(len(traj.steps) for traj in ep.trajectories)
            total_prompt_tokens = sum(len(s.prompt_ids) for traj in ep.trajectories for s in traj.steps)
            total_response_tokens = sum(len(s.response_ids) for traj in ep.trajectories for s in traj.steps)
            self._aggregator.record("episode/num_turns", total_turns)
            self._aggregator.record("episode/prompt_tokens", total_prompt_tokens)
            self._aggregator.record("episode/response_tokens", total_response_tokens)
            self._aggregator.record("episode/correct", 1.0 if ep.is_correct else 0.0)

    def _all_episodes_compact_filtered(self, episodes: list[Episode]) -> bool:
        return all(self._cf_config.should_mask_episode(ep) for ep in episodes)

    @staticmethod
    def _termination_value(reason: TerminationReason | str) -> str:
        return str(getattr(reason, "value", reason))

    def _log_prompt_group_finished(
        self,
        *,
        task_id: str,
        episodes: list[Episode],
        status: str,
        reason: str,
        groups_after_transform: int,
        groups_after_min_trajs: int,
        groups_after_reward_filter: int,
    ) -> None:
        termination_counts = Counter(self._termination_value(ep.termination_reason or TerminationReason.UNKNOWN) for ep in episodes)
        compact_masked = Counter(
            self._termination_value(ep.termination_reason or TerminationReason.UNKNOWN) for ep in episodes if self._cf_config.should_mask_episode(ep)
        )
        rewards = []
        for ep in episodes:
            reward = None
            for traj in ep.trajectories:
                if traj.reward is not None:
                    reward = traj.reward
                elif traj.steps:
                    reward = traj.steps[-1].reward
            rewards.append(reward)

        logger.debug(
            "Prompt group finished task_id=%s status=%s reason=%s episodes=%d rewards=%s "
            "terminations=%s compact_masked=%s groups_after_transform=%d "
            "groups_after_min_trajs=%d groups_after_reward_filter=%d",
            task_id,
            status,
            reason,
            len(episodes),
            rewards,
            dict(termination_counts),
            dict(compact_masked),
            groups_after_transform,
            groups_after_min_trajs,
            groups_after_reward_filter,
        )

    @staticmethod
    def _min_weight_version(episodes: list[Episode]) -> int:
        min_v = float("inf")
        for ep in episodes:
            for traj in ep.trajectories:
                for step in traj.steps:
                    if step.weight_version is not None:
                        min_v = min(min_v, step.weight_version)
        return int(min_v) if min_v != float("inf") else 0
