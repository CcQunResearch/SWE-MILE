import asyncio
import json
import logging
import os
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint
from typing import Any, Literal

import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from rllm.data import (
    DataloaderBatchTicket,
    Dataset,
    DynamicSamplingTaskDataLoader,
    StatefulTaskDataLoader,
)
from rllm.engine.rollout import RolloutEngine
from rllm.gateway.session_ids import task_group_session_ids
from rllm.trainer.algorithms.advantage import (
    AlgorithmConfig,
    collect_reward_and_advantage_from_trajectory_groups,
)
from rllm.trainer.algorithms.config import (
    AsyncTrainingConfig,
    CompactFilteringConfig,
    DynamicSamplingConfig,
    RejectionSamplingConfig,
    TransformConfig,
)
from rllm.trainer.algorithms.metrics import reduce_metrics_lists
from rllm.trainer.algorithms.performance import simple_timer
from rllm.trainer.algorithms.rejection_sampling import (
    RejectionSamplingState,
    apply_rejection_sampling_and_filtering,
    filter_episodes,
)
from rllm.trainer.algorithms.transform import (
    _default_traj_grouping_hook,
    transform_episodes_to_trajectory_groups,
)
from rllm.trainer.algorithms.visualization import print_metrics_table, visualize_trajectory_last_steps
from rllm.trainer.backend_protocol import BackendProtocol
from rllm.trainer.buffer import (
    GenerationTerminalKind,
    TaskBatch,
    TrajectoryGroupBuffer,
)
from rllm.trainer.dynamic_sampling import (
    DynamicSamplingWaveController,
    partition_uniform_outcome_groups,
)
from rllm.trainer.metrics_aggregator import MetricsAggregator
from rllm.trainer.sync_coordinator import SyncCoordinator, SyncCoordinatorConfig
from rllm.types import Episode, RolloutInfrastructureError, TrajectoryGroup
from rllm.utils import EpisodeLogger, Tracking, extract_source_metadata
from rllm.workflows.store import Store
from rllm.workflows.workflow import TerminationReason, Workflow

logger = logging.getLogger(__name__)


@contextmanager
def _detached_warm_queue(hooks):
    """Temporarily detach a training warm queue so the enclosed work (validation,
    whose tasks aren't in the train schedule) falls back to direct sandbox creation."""
    if hooks is None:
        yield
        return
    saved = hooks.warm_queue
    hooks.warm_queue = None
    try:
        yield
    finally:
        hooks.warm_queue = saved


def _fully_async_task_progress(
    train_dataloader: StatefulTaskDataLoader,
    total_epochs: int,
) -> tuple[int, int]:
    """Return total and checkpointed source position for the task progress bar.

    Fully-async training requires a dataloader batch size of one. Version-2
    loaders report the resolved count (dispatched minus pending replay), while
    custom/legacy loaders fall back to their checkpoint cursor.
    """

    tasks_per_epoch = len(train_dataloader)
    total_tasks = tasks_per_epoch * total_epochs
    state_dict = getattr(train_dataloader, "state_dict", None)
    if not callable(state_dict):
        return total_tasks, 0
    state = state_dict()
    stats = getattr(train_dataloader, "stats", None)
    if callable(stats):
        initial_tasks = int(stats().get("resolved", 0))
    else:
        epoch = int(state.get("epoch", 0))
        cursor = int(state.get("cursor", 0))
        initial_tasks = epoch * tasks_per_epoch + cursor
    return total_tasks, min(total_tasks, max(0, initial_tasks))


def _configured_workflow_retry_limit(config: Any) -> int:
    """Read the attempt count from real configs and lightweight test doubles."""

    try:
        workflow = config.workflow
        value = workflow.retry_limit
    except (AttributeError, KeyError):
        return 1
    return max(1, int(value))


@dataclass
class TrainerState:
    """Common trainer state that's backend-agnostic. Reset at each training step."""

    rs_state: RejectionSamplingState = field(default_factory=RejectionSamplingState)
    global_step: int = 0
    # Last optimizer/batch step that completed successfully in this process.
    # ``global_step`` is advanced to the next cursor by some training loops,
    # so it cannot reliably identify the final weights during shutdown.
    last_completed_step: int | None = None
    # The dataloader may change after the last optimizer checkpoint (for
    # example when an async drop-last tail is intentionally consumed).
    force_final_checkpoint: bool = False
    training_completed: bool = False
    # Fully-async transaction phase.  An error checkpoint is safe only while
    # the policy is at a completely committed boundary.
    optimizer_phase: str = "idle"
    safe_to_checkpoint_on_error: bool = False
    fatal_error: dict[str, Any] | None = None
    epoch: int = 0
    total_steps: int = 0
    is_training: bool = True
    weight_version: int = 0
    train_dataloader: StatefulTaskDataLoader | None = None
    # For timing and metrics
    timing_dict: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    extra_info: dict = field(default_factory=dict)
    # For passing the context
    episodes: list[Episode] | None = None
    trajectory_groups: list[TrajectoryGroup] | None = None
    backend_batch: Any | None = None

    def reset_batch(self) -> None:
        """Reset the trainer state for a new batch."""
        self.rs_state.reset()
        self.episodes = None
        self.trajectory_groups = None
        self.backend_batch = None

        self.timing_dict = {}
        self.metrics = {}
        self.extra_info = {}

    @property
    def has_episodes(self) -> bool:
        return self.episodes is not None and len(self.episodes) > 0

    @property
    def has_trajectory_groups(self) -> bool:
        return self.trajectory_groups is not None and len(self.trajectory_groups) > 0

    @property
    def has_backend_batch(self) -> bool:
        return self.backend_batch is not None


class UnifiedTrainer:
    """Unified trainer for backend-agnostic training.

    This trainer uses an async-prioritized design where the core pipeline methods
    are async. This accommodates backends that naturally use async operations
    (like Tinker) while still supporting sync backends.

    The main `fit()` method remains sync for ease of use, but internally runs
    the async training loop in a dedicated event loop thread.
    """

    def __init__(
        self,
        backend_cls: type[BackendProtocol],
        config: DictConfig,
        workflow_class: type[Workflow] | None = None,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        workflow_args: dict | None = None,
        backend_args: dict | None = None,
        *,
        traj_grouping_hook: Callable | None = None,
        traj_group_adv_estimator_map: dict | None = None,
        store: Store | None = None,
        **kwargs,
    ):
        """Initialize the UnifiedTrainer.

        Provide exactly one of ``workflow_class``, ``agent_flow`` (with
        ``evaluator`` or ``hooks``), or ``remote_runtime``.
        """
        has_agent_flow = kwargs.get("agent_flow") is not None and (kwargs.get("evaluator") is not None or kwargs.get("hooks") is not None)
        if not has_agent_flow:
            raise ValueError("SWE-MILE requires an agent_flow with evaluator or sandbox hooks")

        self.workflow_class = workflow_class
        self.workflow_args = workflow_args or {}
        self.store = store
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # initializing and validating common configs
        self.config = config
        self.rllm_config = config.rllm

        # Read user-defined hooks from kwargs
        self.traj_grouping_hook = traj_grouping_hook or _default_traj_grouping_hook
        # Extract the TrajectoryGroup-specific estimator from kwargs
        self.traj_group_adv_estimator_map = traj_group_adv_estimator_map or {}

        # TODO(kylemontgomery1): disaggregate UnitifiedTrainer.__init__ from engine/infra setup

        self.backend = backend_cls(config=config, **(backend_args or {}))

        self.async_config = AsyncTrainingConfig.from_config(self.rllm_config.get("async_training", {}))
        self._validate_and_setup_configs()
        self._setup_logging()

        self._init_dataloaders()

        rollout_engine: RolloutEngine = self.backend.init_rollout_engine(
            cf_config=self.cf_config,
            transform_config=self.transform_config,
            rs_config=self.rs_config,
            algorithm_config=self.algorithm_config,
            total_training_steps=self._total_training_steps,
        )

        # Determine which engine path to use:
        # 1. agent_flow + evaluator → AgentFlowEngine (gateway-based, local)
        # 2. remote_runtime → RemoteAgentFlowEngine (gateway-based, remote)
        # 3. workflow_class → UnifiedWorkflowEngine (direct)
        self._gateway = None
        self._gateway_recovery_lock = asyncio.Lock()

        agent_flow = kwargs.get("agent_flow")
        evaluator = kwargs.get("evaluator")
        hooks = kwargs.get("hooks")


        if agent_flow is not None and (evaluator is not None or hooks is not None):
            initialize_hook_backend = getattr(hooks, "initialize_backend", None)
            if callable(initialize_hook_backend):
                initialize_hook_backend(self.backend, self.config)
            from rllm.engine.agentflow_engine import AgentFlowEngine
            from rllm.gateway.manager import GatewayManager

            gateway_mode = "process" if kwargs.get("backend_name") == "verl" else "thread"
            self._gateway = GatewayManager(self.config, mode=gateway_mode)

            training_sampling_params = OmegaConf.to_container(self.rllm_config.rollout.train, resolve=True)
            val_sampling_params = OmegaConf.to_container(self.rllm_config.rollout.val, resolve=True)
            dynamic_sequence_budget = bool(self.rllm_config.data.get("dynamic_sequence_budget", False))

            try:
                self.agent_workflow_engine = AgentFlowEngine(
                    agent_flow=agent_flow,
                    evaluator=evaluator,
                    gateway=self._gateway,
                    model=self.config.get("model", {}).get("name", "default"),
                    n_parallel_tasks=self.rllm_config.workflow.n_parallel_tasks,
                    retry_limit=self.rllm_config.workflow.retry_limit,
                    raise_on_error=self.rllm_config.workflow.get("raise_on_error", True),
                    episode_logger=self.episode_logger,
                    train_sampling_params=training_sampling_params,
                    val_sampling_params=val_sampling_params,
                    hooks=hooks,
                    milestone_reward_config=self.algorithm_config.milestone_reward,
                    model_window_tokens=(int(self.config.actor_rollout_ref.rollout.max_model_len) if dynamic_sequence_budget else None),
                    rollout_group_size=int(self.rllm_config.rollout.n),
                    cancelled_teardown_timeout_seconds=float(
                        self.rllm_config.workflow.get(
                            "cancelled_teardown_timeout_seconds",
                            60,
                        )
                    ),
                    rollout_startup_window=self.rllm_config.workflow.get(
                        "rollout_startup_window",
                        None,
                    ),
                    teardown_executor_workers=self.rllm_config.workflow.get(
                        "teardown_executor_workers",
                        None,
                    ),
                    trajectory_timeout=self.rllm_config.agent.get(
                        "trajectory_timeout",
                        None,
                    ),
                    denovo_shadow_overlap_budget_multiplier=float(
                        self.rllm_config.workflow.get(
                            "denovo_shadow_overlap_budget_multiplier",
                            1.0,
                        )
                    ),
                )
            except BaseException:
                shutdown_hook_backend = getattr(hooks, "shutdown_backend", None)
                if callable(shutdown_hook_backend):
                    shutdown_hook_backend()
                raise

        configure_dynamic_sampling_logging = getattr(
            self.agent_workflow_engine,
            "configure_dynamic_sampling_rollout_logging",
            None,
        )
        if callable(configure_dynamic_sampling_logging):
            configure_dynamic_sampling_logging(self.dynamic_sampling_config.enable)

        if self._gateway is not None and self.backend.__class__.__name__ == "VerlBackend" and self.rllm_config.algorithm.get("router_replay", "disabled") == "R3":
            raise ValueError("R3 is not supported with the gateway-based rollout (agent_flow / remote_runtime) on the verl backend.")

        self.tokenizer = None
        if hasattr(self.backend, "tokenizer"):
            self.tokenizer = self.backend.tokenizer

    def _init_dataloaders(self) -> None:
        """Build the train and val dataloaders from their datasets."""
        self._train_dataloader: StatefulTaskDataLoader | None = None
        self._val_dataloader: StatefulTaskDataLoader | None = None
        self._total_training_steps: int | None = None

        if self.train_dataset is not None:
            if self.dynamic_sampling_config.enable:
                target_batch_size = self.async_config.mini_batch_size if self.async_config.enable else int(self.rllm_config.data.train_batch_size)
                self._train_dataloader = DynamicSamplingTaskDataLoader(
                    self.train_dataset,
                    shuffle=True,
                    seed=self.rllm_config.data.seed,
                    max_easy_rejections=(self.dynamic_sampling_config.max_easy_rejections),
                    max_hard_rejections=(self.dynamic_sampling_config.max_hard_rejections),
                    outcome_mode=self.dynamic_sampling_config.outcome_mode,
                    easy_pass_rate_threshold=(self.dynamic_sampling_config.easy_pass_rate_threshold),
                    hard_pass_rate_threshold=(self.dynamic_sampling_config.hard_pass_rate_threshold),
                    target_batch_size=target_batch_size,
                    defer_requeues_until_next_wave=self.async_config.enable,
                    checkpoint_config={
                        "training_mode": ("fully_async" if self.async_config.enable else "synchronous"),
                        "data_train_batch_size": int(self.rllm_config.data.train_batch_size),
                        "rollout_group_size": int(self.rllm_config.rollout.n),
                        "async_mini_batch_size": (int(self.async_config.mini_batch_size) if self.async_config.enable else None),
                        "async_fwd_bwd_group_size": (int(self.async_config.fwd_bwd_group_size) if self.async_config.enable else None),
                        "async_step_sampling_multiplier": (self.dynamic_sampling_config.async_step_sampling_multiplier),
                        "cancel_wait_timeout_seconds": (self.dynamic_sampling_config.cancel_wait_timeout_seconds),
                        "outcome_mode": self.dynamic_sampling_config.outcome_mode,
                        "easy_pass_rate_threshold": (self.dynamic_sampling_config.easy_pass_rate_threshold),
                        "hard_pass_rate_threshold": (self.dynamic_sampling_config.hard_pass_rate_threshold),
                    },
                )
            else:
                batch_size = 1 if self.async_config.enable else int(self.rllm_config.data.train_batch_size * self.rllm_config.rejection_sample.multiplier)
                self._train_dataloader = StatefulTaskDataLoader(
                    self.train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    seed=self.rllm_config.data.seed,
                    drop_last=True,
                )
            total_batches = self.rllm_config.trainer.get("total_batches")
            use_total_batches = total_batches is not None and total_batches > 0
            if use_total_batches:
                self._total_training_steps = total_batches
            else:
                total_tasks = len(self.train_dataset) * self.rllm_config.trainer.total_epochs
                if self.dynamic_sampling_config.enable:
                    target_batch_size = self.async_config.mini_batch_size if self.async_config.enable else int(self.rllm_config.data.train_batch_size)
                    self._total_training_steps = total_tasks // target_batch_size
                else:
                    total_task_batches = len(self._train_dataloader) * self.rllm_config.trainer.total_epochs
                    self._total_training_steps = total_task_batches // self.async_config.mini_batch_size if self.async_config.enable else total_task_batches

        if self.val_dataset is not None:
            val_batch_size = self.rllm_config.data.val_batch_size
            if val_batch_size == -1:
                val_batch_size = len(self.val_dataset)
            self._val_dataloader = StatefulTaskDataLoader(self.val_dataset, batch_size=val_batch_size, shuffle=False, drop_last=False)

    def _validate_and_setup_configs(self):
        """Validate and setup common configs."""
        # validate common, backend-agnostic configs
        assert self.rllm_config is not None, "rLLM config is not set"

        self.dynamic_sampling_config = DynamicSamplingConfig.from_config(self.rllm_config.get("dynamic_sampling", {}))
        if self.dynamic_sampling_config.enable and self.rllm_config.rejection_sample.multiplier != 1:
            raise ValueError("dynamic sampling requires rejection_sample.multiplier=1")
        if self.dynamic_sampling_config.enable and self.dynamic_sampling_config.async_step_sampling_multiplier != 1.0 and not self.async_config.enable:
            raise ValueError("dynamic_sampling.async_step_sampling_multiplier != 1 is only supported with Dynamic Sampling and fully-async training enabled")

        if self.rllm_config.rejection_sample.multiplier != 1:
            assert self.rllm_config.rejection_sample.enable is True, "rejection sampling is disabled, but rejection_sample.multiplier is not 1"

        # validate backend-specific configs
        self.backend.validate_config()

        self.cf_config = CompactFilteringConfig.from_config(self.rllm_config.compact_filtering)
        self.transform_config = TransformConfig.from_config(
            self.rllm_config.get("transform", {}),
            broadcast=True,
        )
        self.rs_config = RejectionSamplingConfig.from_config(self.rllm_config.rejection_sample)
        self.algorithm_config = AlgorithmConfig.from_config(
            self.rllm_config.algorithm,
            stepwise_advantage_mode=self.rllm_config.stepwise_advantage.mode,
            stepwise_advantage_config=self.rllm_config.stepwise_advantage,
            estimator_map=self.traj_group_adv_estimator_map,
        )

    def _setup_logging(self):
        """Setup up both the tracking and episode logging."""
        # create episode logger if enabled in config
        self.episode_logger = None
        if self.rllm_config.episode_logging.get("log_episodes", False):
            episode_log_dir = self.rllm_config.episode_logging.get(
                "episode_log_dir",
                f"logs/{self.rllm_config.trainer.project_name}/{self.rllm_config.trainer.experiment_name}",
            )
            self.episode_logger = EpisodeLogger(base_dir=episode_log_dir, subdirectory="episodes")

        source_metadata = extract_source_metadata(
            workflow_class=self.workflow_class,
            workflow_args=self.workflow_args,
        )

        self.logger = Tracking(
            project_name=self.rllm_config.trainer.project_name,
            experiment_name=self.rllm_config.trainer.experiment_name,
            default_backend=self.rllm_config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
            source_metadata=source_metadata,
        )

    # =========================================================================
    # Main training loop methods
    # =========================================================================

    # TODO(kylemontgomery1): better seperation of on policy vs fully async training code

    def fit(self):
        """Main training loop (sync entry point)."""
        if getattr(getattr(self.agent_workflow_engine, "hooks", None), "sandbox_backend", None) == "minisandbox":
            from rllm.utils.bounded_cleanup import run_with_bounded_cleanup
            run_with_bounded_cleanup(self.fit_async(), timeout=120)
        else:
            asyncio.run(self.fit_async())

    async def fit_async(self) -> None:
        """Public async entry point for the full training process."""
        # Initialize remote runtime (if enabled) before the workflow pool

        # initialize the UnifiedWorkflowEngine (init the workflow pool)
        # AgentFlowEngine and RemoteAgentFlowEngine don't need pool initialization
        if hasattr(self.agent_workflow_engine, "initialize_pool"):
            await self.agent_workflow_engine.initialize_pool()

        trainer_state = TrainerState()
        trainer_state.train_dataloader = self._train_dataloader

        await self.backend.on_train_start(trainer_state)
        # Backends restore ``global_step`` to the durable checkpoint step.
        # Preserve that boundary before advancing to the next attempted step;
        # otherwise a failure immediately after resume reports no committed
        # step and cannot write a safe emergency checkpoint.
        if trainer_state.global_step > 0:
            trainer_state.last_completed_step = trainer_state.global_step

        if hasattr(self, "_gateway") and self._gateway is not None:
            self._gateway.start(self.backend.rollout_engine)
            self._gateway.set_weight_version(trainer_state.weight_version)

        if self.rllm_config.trainer.get("val_before_train", True):
            await self._validate_async(trainer_state)
            if self.rllm_config.trainer.get("val_only", False):
                return

        # we start from step (1 + original start batch index)
        trainer_state.global_step += 1

        active_error: BaseException | None = None
        try:
            await self._fit_async(trainer_state)
            trainer_state.training_completed = True
        except BaseException as exc:
            active_error = exc
            from rllm.utils.infrastructure_diagnostics import infrastructure_exception_chain

            trainer_state.fatal_error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                "exception_chain": infrastructure_exception_chain(exc),
            }
            trainer_state.safe_to_checkpoint_on_error = bool(trainer_state.last_completed_step is not None and trainer_state.optimizer_phase in {"idle", "waiting_for_batch"})
            if trainer_state.safe_to_checkpoint_on_error:
                trainer_state.force_final_checkpoint = True
            try:
                self._write_training_failure_manifest(trainer_state)
            except Exception:
                logger.exception("Could not persist failure manifest; preserving training error")
            raise
        finally:
            try:
                await self.backend.on_train_end(trainer_state)
            except BaseException as cleanup_error:
                if active_error is None:
                    raise
                from rllm.utils.infrastructure_diagnostics import infrastructure_exception_chain
                trainer_state.fatal_error.setdefault("shutdown_errors", []).append({
                    "component": "on_train_end", "exception_chain": infrastructure_exception_chain(cleanup_error)})
                try:
                    self._write_training_failure_manifest(trainer_state)
                except Exception:
                    logger.exception("Could not persist on_train_end failure")
                logger.exception(f"{self.backend.__class__.__name__}.on_train_end() failed while preserving the active training error")

    def _write_training_failure_manifest(
        self,
        trainer_state: TrainerState,
    ) -> None:
        """Write an atomic, non-checkpoint diagnostic for interrupted runs."""
        explicit_params = os.environ.get("RLLM_TRAINING_PARAMS_PATH")
        if explicit_params:
            output_dir = Path(explicit_params).expanduser().parent
        else:
            configured = self.rllm_config.trainer.get("wandb_dir", None)
            if not configured:
                logger.warning("Cannot write training_failure_manifest.json: no run log directory")
                return
            output_dir = Path(str(configured)).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "training_failure_manifest.json"
        self._failure_manifest_path = path
        tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
        dataloader = trainer_state.train_dataloader
        stats = getattr(dataloader, "stats", None)
        payload = {
            "schema_version": 1,
            "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_committed_step": trainer_state.last_completed_step,
            "attempted_step": trainer_state.global_step,
            "optimizer_phase": trainer_state.optimizer_phase,
            "safe_to_checkpoint_on_error": (trainer_state.safe_to_checkpoint_on_error),
            "force_final_checkpoint": trainer_state.force_final_checkpoint,
            "weight_version": trainer_state.weight_version,
            "fatal_error": trainer_state.fatal_error,
            "dataloader_stats": stats() if callable(stats) else None,
            "async_failure_state": trainer_state.extra_info.get("async_failure_state"),
        }
        raw_provenance = os.environ.get("RLLM_RUNTIME_CODE_PROVENANCE")
        from rllm.utils.diagnostic_events import diagnostic_reference
        payload["diagnostic_evidence"] = diagnostic_reference()
        if raw_provenance:
            try:
                payload["runtime_code_provenance"] = json.loads(raw_provenance)
            except json.JSONDecodeError:
                payload["runtime_code_provenance"] = {"invalid": raw_provenance}
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except Exception:
            logger.exception("Failed to write %s", path)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    async def _fit_async(self, trainer_state: TrainerState) -> None:
        """Dispatch to sync or concurrent training based on config."""
        # TODO(listar2000): after some benchmarking, maybe we just keep the fully-async and treat on-policy as a special case.
        if self.async_config.enable:
            await self._fit_fully_async(trainer_state)
        else:
            await self._fit_on_policy(trainer_state)

    async def _fit_on_policy(self, trainer_state: TrainerState) -> None:
        """Synchronous training loop (the most vanilla, standalone case that does not support minibatching or off-policy training)."""
        if self.dynamic_sampling_config.enable:
            await self._fit_on_policy_dynamic(trainer_state)
            return
        train_dataloader = self._train_dataloader
        assert train_dataloader is not None, "train_dataset is required for training"
        total_epochs = self.rllm_config.trainer.total_epochs
        use_total_batches = self.rllm_config.trainer.get("total_batches") is not None and self.rllm_config.trainer.total_batches > 0
        trainer_state.total_steps = self._total_training_steps
        break_via_total_batches = False

        hooks = getattr(self.agent_workflow_engine, "hooks", None)
        warm_queue = self._start_train_warm_queue(
            train_dataloader,
            trainer_state,
            total_epochs,
            use_total_batches,
            hooks,
        )
        try:
            for epoch in range(train_dataloader.epoch, total_epochs):
                if break_via_total_batches:
                    break
                trainer_state.epoch = epoch
                pprint(f"epoch {epoch}, step {trainer_state.global_step} started")
                await self.backend.on_epoch_start(trainer_state)

                for batch in train_dataloader:
                    trainer_state.reset_batch()
                    await self.backend.on_batch_start(trainer_state)
                    with simple_timer("step", trainer_state.timing_dict):
                        await self._train_batch_async(batch, trainer_state)
                    await self.backend.on_batch_end(trainer_state)
                    trainer_state.last_completed_step = trainer_state.global_step
                    trainer_state.metrics.update(await self._gateway_routing_metrics())

                    print_metrics_table(trainer_state.metrics, trainer_state.global_step)
                    self.logger.log(
                        data=trainer_state.metrics,
                        step=trainer_state.global_step,
                        episodes=trainer_state.episodes,
                        trajectory_groups=trainer_state.trajectory_groups,
                    )

                    # if the config specifies the `total_batches` parameter, then we check if we should stop
                    if use_total_batches and trainer_state.global_step >= self.rllm_config.trainer.total_batches:
                        break_via_total_batches = True
                        break

                    # periodic validation
                    if self.rllm_config.trainer.test_freq > 0 and trainer_state.global_step % self.rllm_config.trainer.test_freq == 0:
                        with _detached_warm_queue(hooks):
                            await self._validate_async(trainer_state)

                    trainer_state.global_step += 1

                await self.backend.on_epoch_end(trainer_state)
        finally:
            if warm_queue is not None:
                warm_queue.shutdown()
                hooks.warm_queue = None

        # final validation after training (queue already shut down + detached above)
        if self.rllm_config.trainer.test_freq > 0:
            await self._validate_async(trainer_state)

    async def _fit_on_policy_dynamic(self, trainer_state: TrainerState) -> None:
        """Synchronous task-pool training with outcome-driven refill."""
        train_dataloader = self._train_dataloader
        if not isinstance(train_dataloader, DynamicSamplingTaskDataLoader):
            raise TypeError("dynamic sampling requires its stateful task-pool loader")
        total_epochs = int(self.rllm_config.trainer.total_epochs)
        use_total_batches = self.rllm_config.trainer.get("total_batches", -1) > 0
        trainer_state.total_steps = self._total_training_steps
        active_epoch: int | None = None

        hooks = getattr(self.agent_workflow_engine, "hooks", None)
        warm_queue = self._start_train_warm_queue(
            train_dataloader,
            trainer_state,
            total_epochs,
            use_total_batches,
            hooks,
            loader_batches_per_training_step=int(self.rllm_config.data.train_batch_size),
        )
        try:
            while not train_dataloader.generation_exhausted(total_epochs):
                trainer_state.reset_batch()
                with simple_timer("step", trainer_state.timing_dict):
                    trained, active_epoch = await self._train_dynamic_sampling_batch(
                        trainer_state,
                        train_dataloader,
                        total_epochs,
                        active_epoch,
                    )

                trainer_state.metrics.update(self._drain_dynamic_sampling_metrics(train_dataloader))
                if not trained:
                    # No full optimizer batch remains. The loader has already
                    # consumed the accepted drop-last tail, if any.
                    if any(trainer_state.metrics.values()):
                        print_metrics_table(
                            trainer_state.metrics,
                            trainer_state.global_step,
                        )
                        self.logger.log(
                            data=trainer_state.metrics,
                            step=trainer_state.global_step,
                            episodes=trainer_state.episodes,
                            trajectory_groups=trainer_state.trajectory_groups,
                        )
                    if train_dataloader.generation_exhausted(total_epochs):
                        break
                    continue

                await self.backend.on_batch_end(trainer_state)
                trainer_state.last_completed_step = trainer_state.global_step
                trainer_state.metrics.update(await self._gateway_routing_metrics())
                print_metrics_table(trainer_state.metrics, trainer_state.global_step)
                self.logger.log(
                    data=trainer_state.metrics,
                    step=trainer_state.global_step,
                    episodes=trainer_state.episodes,
                    trajectory_groups=trainer_state.trajectory_groups,
                )

                if use_total_batches and trainer_state.global_step >= int(self.rllm_config.trainer.total_batches):
                    break
                if self.rllm_config.trainer.test_freq > 0 and trainer_state.global_step % self.rllm_config.trainer.test_freq == 0:
                    with _detached_warm_queue(hooks):
                        await self._validate_async(trainer_state)
                trainer_state.global_step += 1
        finally:
            if active_epoch is not None:
                trainer_state.epoch = active_epoch
                await self.backend.on_epoch_end(trainer_state)
            if warm_queue is not None:
                warm_queue.shutdown()
                hooks.warm_queue = None

        self._log_remaining_dynamic_sampling_metrics(
            trainer_state,
            train_dataloader,
        )
        if self.rllm_config.trainer.test_freq > 0:
            await self._validate_async(trainer_state)

    async def _train_dynamic_sampling_batch(
        self,
        trainer_state: TrainerState,
        train_dataloader: DynamicSamplingTaskDataLoader,
        total_epochs: int,
        active_epoch: int | None,
    ) -> tuple[bool, int | None]:
        """Refill until one exact synchronous optimizer batch is accepted."""
        target_tasks = int(self.rllm_config.data.train_batch_size)
        accepted_tickets: list[DataloaderBatchTicket] = []
        accepted_episodes: list[Episode] = []
        accepted_groups: list[TrajectoryGroup] = []
        all_rollout_episodes: list[Episode] = []
        transform_aggregator = MetricsAggregator()
        dynamic_sampling_config = getattr(
            self,
            "dynamic_sampling_config",
            DynamicSamplingConfig(enable=True),
        )

        while len(accepted_tickets) < target_tasks:
            item = train_dataloader.next_dispatch(total_epochs)
            if item is None:
                if not train_dataloader.generation_exhausted(total_epochs):
                    raise RuntimeError("synchronous dynamic sampling is waiting for missing rollout feedback")
                break
            batch, source_ticket = item
            if source_ticket.epoch != active_epoch:
                if active_epoch is not None:
                    trainer_state.epoch = active_epoch
                    await self.backend.on_epoch_end(trainer_state)
                active_epoch = source_ticket.epoch
                trainer_state.epoch = active_epoch
                await self.backend.on_epoch_start(trainer_state)

            self.agent_workflow_engine.set_training_step(
                trainer_state.global_step,
                mode="train",
                epoch=source_ticket.epoch,
            )
            episodes = await self.backend.generate_episodes(
                batch,
                agent_workflow_engine=self.agent_workflow_engine,
                is_validation=False,
            )
            rollout_count = len(episodes)
            all_rollout_episodes.extend(episodes)
            if not episodes:
                train_dataloader.mark_consumed_other(
                    source_ticket,
                    rollout_count=0,
                )
                continue

            groups, transform_metrics = transform_episodes_to_trajectory_groups(
                episodes,
                self.transform_config,
                self.cf_config,
                traj_grouping_hook=self.traj_grouping_hook,
            )
            transform_aggregator.record_dict(transform_metrics)
            eligible = [group for group in groups if len(group.trajectories) >= self.rs_config.min_trajs_per_group]
            dropped_min = [group for group in groups if len(group.trajectories) < self.rs_config.min_trajs_per_group]
            transform_aggregator.record(
                "groups/dropped_min_trajs",
                len(dropped_min),
            )
            if not eligible:
                self._log_dynamic_sampling_rollouts(
                    episodes,
                    filtered_groups=[],
                )
                train_dataloader.mark_consumed_other(
                    source_ticket,
                    rollout_count=rollout_count,
                )
                continue

            partition = partition_uniform_outcome_groups(
                eligible,
                outcome_mode=dynamic_sampling_config.outcome_mode,
                easy_pass_rate_threshold=(dynamic_sampling_config.easy_pass_rate_threshold),
                hard_pass_rate_threshold=(dynamic_sampling_config.hard_pass_rate_threshold),
            )
            transform_aggregator.record(
                "groups/dropped_uniform_outcome",
                len(partition.filtered),
            )
            self._log_dynamic_sampling_rollouts(
                episodes,
                filtered_groups=partition.filtered,
            )
            if dropped_min:
                episodes = filter_episodes(episodes, dropped_min)
            if partition.filtered:
                episodes = filter_episodes(episodes, partition.filtered)
            if partition.task_filtered:
                train_dataloader.record_filtered_outcome(
                    source_ticket,
                    uniform=partition.uniform_pass_count,
                    easy=partition.easy,
                    hard=partition.hard,
                    rollout_count=rollout_count,
                )
                continue

            train_dataloader.mark_accepted(
                source_ticket,
                rollout_count=rollout_count,
            )
            accepted_tickets.append(source_ticket)
            accepted_episodes.extend(episodes)
            accepted_groups.extend(partition.kept)

        trainer_state.episodes = accepted_episodes
        trainer_state.trajectory_groups = accepted_groups
        trainer_state.metrics.update(transform_aggregator.flush())
        if all_rollout_episodes:
            workflow_metrics, termination_counts = self._collect_workflow_metrics_from_episodes(all_rollout_episodes)
            for key, value in workflow_metrics.items():
                trainer_state.metrics[f"batch/{key}"] = value
            total_counts = max(sum(termination_counts.values()), 1)
            for reason in TerminationReason:
                trainer_state.metrics[f"batch/termination_reason/{reason.value}"] = termination_counts[reason.value] / total_counts

        if len(accepted_tickets) < target_tasks:
            train_dataloader.mark_untrained_tail_many(accepted_tickets)
            if accepted_tickets:
                trainer_state.force_final_checkpoint = True
            return False, active_epoch

        filtered_groups, filtered_episodes, rs_metrics = apply_rejection_sampling_and_filtering(
            accepted_episodes,
            accepted_groups,
            self.rs_config,
            trainer_state.rs_state,
        )
        trainer_state.metrics.update(rs_metrics)
        trainer_state.trajectory_groups = filtered_groups
        trainer_state.episodes = filtered_episodes
        if not trainer_state.has_trajectory_groups:
            train_dataloader.mark_resolved_many(accepted_tickets)
            return False, active_epoch

        await self.backend.on_batch_start(trainer_state)
        trainer_state.backend_batch = self.backend.transform_to_backend_batch(trainer_state)
        await self.backend.process_backend_batch(trainer_state)
        await self.backend.compute_advantages(trainer_state, self.algorithm_config)
        await self.backend.update_policy(trainer_state)
        # Optimizer success is the commit point. Exceptions above deliberately
        # leave accepted tickets in the checkpoint for exact replay.
        train_dataloader.mark_resolved_many(accepted_tickets)
        return True, active_epoch

    def _log_dynamic_sampling_rollouts(
        self,
        episodes: list[Episode],
        *,
        filtered_groups: list[TrajectoryGroup],
    ) -> None:
        self._write_dynamic_sampling_rollouts(
            episodes,
            {id(trajectory) for group in filtered_groups for trajectory in group.trajectories},
        )

    def _write_dynamic_sampling_rollouts(
        self,
        episodes: list[Episode],
        filtered_trajectory_ids: set[int],
        *,
        lifecycle: dict[str, Any] | None = None,
    ) -> None:
        log_rollouts = getattr(
            self.agent_workflow_engine,
            "log_dynamic_sampling_rollout_samples",
            None,
        )
        if not callable(log_rollouts):
            return
        log_rollouts(
            episodes,
            filtered_trajectory_ids=filtered_trajectory_ids,
            lifecycle=lifecycle,
        )

    def _start_train_warm_queue(
        self,
        train_dataloader,
        trainer_state,
        total_epochs,
        use_total_batches,
        hooks,
        *,
        loader_batches_per_training_step: int = 1,
    ):
        """Prefetch upcoming sandboxes ahead of rollout, mirroring eval's warm pool.

        Returns a started :class:`WarmQueue` attached to ``hooks`` (the caller shuts
        it down), or ``None`` when there is no sandbox backend or the knob is 0.
        Size ``-1`` matches the active-rollout frontier
        (``workflow.n_parallel_tasks``). Warm entries contain primary
        sandboxes only; shadow sandboxes are created when a rollout starts.
        """
        warm_queue_size = self.rllm_config.workflow.get("warm_queue_size", -1)
        if hooks is None or not getattr(hooks, "sandbox_backend", None) or warm_queue_size == 0:
            return None
        if getattr(hooks, "sandbox_backend", None) == "minisandbox":
            raise ValueError(
                "MiniSandbox requires rllm.workflow.warm_queue_size=0 so the "
                "primary and verifier sandboxes can use rollout-UID node affinity"
            )
        if warm_queue_size < -1:
            raise ValueError(f"workflow.warm_queue_size must be -1, 0, or a positive integer, got {warm_queue_size}")
        if loader_batches_per_training_step < 1:
            raise ValueError(f"loader_batches_per_training_step must be positive, got {loader_batches_per_training_step}")

        from rllm.sandbox.snapshot import install_script_for
        from rllm.sandbox.train_schedule import build_train_schedule
        from rllm.sandbox.warm_queue import WarmQueue

        flow = getattr(self.agent_workflow_engine, "agent_flow", None)
        # AgentFlowEngine keeps n_parallel_tasks as the active-rollout
        # frontier. Codeflow's shadow sandbox is additional to each active
        # primary and is intentionally not subtracted from this count.
        frontier = getattr(self.agent_workflow_engine, "n_parallel_tasks", getattr(flow, "max_concurrent", 8))
        size = frontier if warm_queue_size < 0 else warm_queue_size
        consumed = trainer_state.global_step - 1  # global_step is 1-based once fit_async starts
        remaining_training_steps = self._total_training_steps - consumed
        if use_total_batches:
            remaining_training_steps = min(
                remaining_training_steps,
                self.rllm_config.trainer.total_batches - consumed,
            )
        remaining_loader_batches = max(0, remaining_training_steps) * loader_batches_per_training_step
        if isinstance(train_dataloader, DynamicSamplingTaskDataLoader) and not use_total_batches:
            # Preview the complete predictable first pass, including the
            # eventual drop-last tail. Result-driven requeues are intentionally
            # absent and use the sandbox backend's on-demand fallback.
            remaining_loader_batches = -1
        schedule = build_train_schedule(
            train_dataloader,
            group_size=self.rllm_config.rollout.n,
            total_epochs=total_epochs,
            remaining_batches=remaining_loader_batches,
        )
        if not schedule:
            return None
        size = min(size, len(schedule))
        warm_queue = WarmQueue(schedule, hooks.sandbox_backend, size, install_script=install_script_for(flow))
        hooks.warm_queue = warm_queue
        try:
            warm_queue.start()
        except BaseException:
            hooks.warm_queue = None
            warm_queue.shutdown()
            raise
        logger.info("training warm queue started: %d ahead over %d scheduled tasks", size, len(schedule))
        return warm_queue

    async def _train_batch_async(self, batch: Any, trainer_state: TrainerState) -> None:
        """Train a batch (async implementation)."""
        self.agent_workflow_engine.set_training_step(trainer_state.global_step, mode="train", epoch=trainer_state.epoch)

        # TODO(kylemontgomery1): episode generation should be backend-agnostic
        # stage 1: generate episodes (async) and collect metrics (sync)
        trainer_state.episodes = await self.backend.generate_episodes(batch, agent_workflow_engine=self.agent_workflow_engine, is_validation=False)
        if not trainer_state.has_episodes:
            return

        workflow_metrics, termination_counts = self._collect_workflow_metrics_from_episodes(trainer_state.episodes)
        for key, value in workflow_metrics.items():
            trainer_state.metrics[f"batch/{key}"] = np.mean(value)

        total_counts = max(sum(termination_counts.values()), 1)
        for r in TerminationReason:
            trainer_state.metrics[f"batch/termination_reason/{r.value}"] = termination_counts[r.value] / total_counts

        # stage 2: transform episodes to trajectory groups (sync)
        trajectory_groups, transform_metrics = transform_episodes_to_trajectory_groups(trainer_state.episodes, self.transform_config, self.cf_config, traj_grouping_hook=self.traj_grouping_hook)
        trainer_state.trajectory_groups = trajectory_groups
        trainer_state.metrics.update(transform_metrics)

        # stage 3: apply rejection sampling (sync)
        filtered_groups, filtered_episodes, rs_metrics = apply_rejection_sampling_and_filtering(
            trainer_state.episodes,
            trainer_state.trajectory_groups,
            self.rs_config,
            trainer_state.rs_state,
        )
        trainer_state.metrics.update(rs_metrics)
        trainer_state.trajectory_groups = filtered_groups
        trainer_state.episodes = filtered_episodes
        if not trainer_state.has_trajectory_groups:
            return

        # stage 4: transform rllm-native data structures to backend-specific format (sync)
        backend_batch = self.backend.transform_to_backend_batch(trainer_state)
        trainer_state.backend_batch = backend_batch

        # stage 5: process backend batch (async) - compute log probs, critic values, etc.
        await self.backend.process_backend_batch(trainer_state)
        assert trainer_state.has_backend_batch, "Backend batch is not transformed or processed successfully"

        # TODO(kylemontgomery1): compute advantages should be backend-agnostic
        # stage 6: compute advantages (async)
        await self.backend.compute_advantages(trainer_state, self.algorithm_config)

        # stage 7: update policy (async)
        await self.backend.update_policy(trainer_state)

        # stage 8: cleanup, logging, visualization, etc. (sync)
        if self.tokenizer is not None:
            visualize_trajectory_last_steps(
                trainer_state.trajectory_groups,
                tokenizer=self.tokenizer,
                max_steps_to_visualize=2,
                show_workflow_metadata=True,
            )

    # =========================================================================
    # Fully-asynchronous training pipeline
    # =========================================================================

    async def _fit_fully_async(self, trainer_state: TrainerState) -> None:
        """Fully-async generation + training with group-level streaming."""
        assert self.rllm_config.data.train_batch_size == 1, f"Async training requires train_batch_size=1, got {self.rllm_config.data.train_batch_size}"
        assert not getattr(self.agent_workflow_engine, "raise_on_error", False), "Async training requires raise_on_error=False so that process_task_with_retry always returns an episode"
        coord_config = SyncCoordinatorConfig(
            mini_batch_size=self.async_config.mini_batch_size,
            group_size=self.rllm_config.rollout.n,
            staleness_threshold=self.async_config.staleness_threshold,
            trigger_parameter_sync_step=self.async_config.trigger_parameter_sync_step,
        )
        coordinator = SyncCoordinator(coord_config)
        aggregator = MetricsAggregator()
        train_dataloader = self._train_dataloader
        assert train_dataloader is not None, "train_dataset is required for training"
        dynamic_sampling_config = getattr(
            self,
            "dynamic_sampling_config",
            DynamicSamplingConfig(),
        )
        wave_controller: DynamicSamplingWaveController | None = None
        denovo_background_finalize = bool(
            getattr(
                self.agent_workflow_engine,
                "denovo_background_finalize_enable",
                False,
            )
        )
        if isinstance(train_dataloader, DynamicSamplingTaskDataLoader):
            saved_wave_state = train_dataloader.sampling_wave_state()
            wave_controller = DynamicSamplingWaveController(
                loader=train_dataloader,
                first_target=(
                    self.async_config.mini_batch_size
                    if denovo_background_finalize
                    else coord_config.max_rollout_quota
                ),
                steady_target=self.async_config.mini_batch_size,
                multiplier=(dynamic_sampling_config.async_step_sampling_multiplier),
                rollout_group_size=self.rllm_config.rollout.n,
                cancel_wait_timeout_seconds=(dynamic_sampling_config.cancel_wait_timeout_seconds),
                initial_dispatch_step=trainer_state.global_step,
                initial_weight_version=coordinator.weight_version,
                infrastructure_circuit_breaker=(
                    denovo_background_finalize
                    or getattr(getattr(self.agent_workflow_engine, "hooks", None), "sandbox_backend", None) == "minisandbox"
                ),
            )
            if saved_wave_state is not None:
                wave_controller.load_state_dict(
                    saved_wave_state,
                    replay_pending=True,
                )
        discard_dynamic_rollouts = getattr(
            self.agent_workflow_engine,
            "discard_dynamic_sampling_rollout_samples",
            None,
        )
        deferred_shadow_predicate = getattr(
            self.agent_workflow_engine,
            "has_deferred_shadow",
            None,
        )
        finalize_deferred_shadows = getattr(
            self.agent_workflow_engine,
            "finalize_deferred_shadow_batch",
            None,
        )
        cancel_deferred_shadows = getattr(
            self.agent_workflow_engine,
            "cancel_deferred_shadow_episodes",
            None,
        )

        async def deferred_finalizer(
            episodes: list[Episode],
            stall_timeout: float,
        ) -> dict[str, float]:
            return await finalize_deferred_shadows(
                episodes,
                stall_timeout=stall_timeout,
            )

        async def deferred_canceller(
            episodes: list[Episode],
            disposition: str,
        ) -> int:
            return await cancel_deferred_shadows(
                episodes,
                disposition=disposition,
            )

        buffer = TrajectoryGroupBuffer(
            group_size=self.rllm_config.rollout.n,
            coordinator=coordinator,
            aggregator=aggregator,
            algorithm_config=self.algorithm_config,
            transform_config=self.transform_config,
            cf_config=self.cf_config,
            rs_config=self.rs_config,
            episode_offload_dir=self.async_config.episode_offload_dir,
            trajectory_group_offload_dir=self.async_config.trajectory_group_offload_dir,
            source_resolver=getattr(train_dataloader, "mark_resolved", None),
            dynamic_sampling_config=dynamic_sampling_config,
            dynamic_sampling_loader=(train_dataloader if isinstance(train_dataloader, DynamicSamplingTaskDataLoader) else None),
            dynamic_sampling_rollout_logger=(self._write_dynamic_sampling_rollouts if dynamic_sampling_config.enable or denovo_background_finalize else None),
            dynamic_sampling_rollout_discarder=(discard_dynamic_rollouts if callable(discard_dynamic_rollouts) else None),
            dynamic_sampling_wave_controller=wave_controller,
            deferred_shadow_predicate=(
                deferred_shadow_predicate
                if denovo_background_finalize
                and callable(deferred_shadow_predicate)
                else None
            ),
            deferred_shadow_finalizer=(
                deferred_finalizer
                if denovo_background_finalize
                and callable(finalize_deferred_shadows)
                else None
            ),
            deferred_shadow_canceller=(
                deferred_canceller
                if denovo_background_finalize
                and callable(cancel_deferred_shadows)
                else None
            ),
            deferred_shadow_stall_timeout=(
                float(
                    getattr(
                        self.agent_workflow_engine.agent_flow,
                        "shadow_finalize_stall_timeout",
                        0.0,
                    )
                )
                if denovo_background_finalize
                else 0.0
            ),
        )
        if wave_controller is not None:
            wave_controller.set_cancel_group_callback(buffer.cancel_speculative_group)

            async def recover_group(task_id: str) -> tuple[int, int]:
                return await buffer.cancel_speculative_group(
                    task_id,
                    defer_until_next_wave=False,
                )

            wave_controller.set_recover_group_callback(recover_group)
            wave_controller.set_detach_tasks_callback(coordinator.detach_tasks)
            describe_rollout_task = getattr(
                self.agent_workflow_engine,
                "describe_rollout_task",
                None,
            )
            if callable(describe_rollout_task):
                wave_controller.set_task_diagnostic_callback(describe_rollout_task)
            gateway = getattr(self, "_gateway", None)
            prepare_cancel = getattr(gateway, "abulk_tombstone_task_groups", None)
            wait_cleanup = getattr(gateway, "await_cleanup_ready", None)
            if callable(prepare_cancel):

                async def prepare_wave_cancel(task_ids: list[str]) -> None:
                    if not getattr(gateway, "available", True):
                        return
                    await prepare_cancel(
                        task_ids,
                        int(self.rllm_config.rollout.n),
                        max_attempts=_configured_workflow_retry_limit(
                            self.rllm_config
                        ),
                    )
            else:
                prepare_wave_cancel = None
            if callable(wait_cleanup):

                async def wait_wave_cleanup() -> None:
                    if not getattr(gateway, "available", True):
                        return
                    await wait_cleanup()
            else:
                wait_wave_cleanup = None
            wave_controller.set_wave_cleanup_callbacks(
                prepare=prepare_wave_cancel,
                wait_ready=wait_wave_cleanup,
            )
        trainer_state.total_steps = self._total_training_steps

        total_tasks, initial_tasks = _fully_async_task_progress(
            train_dataloader,
            self.rllm_config.trainer.total_epochs,
        )
        pbar = tqdm(
            total=total_tasks,
            initial=initial_tasks,
            desc="Tasks",
            unit="task",
        )
        buffer._pbar = pbar
        buffer.set_training_step(trainer_state.global_step)
        hooks = getattr(self.agent_workflow_engine, "hooks", None)
        warm_queue = None
        gen_task = None
        supervisor_task: asyncio.Task[None] | None = None
        active_error: BaseException | None = None
        try:
            gateway = getattr(self, "_gateway", None)
            supervision = getattr(gateway, "supervision_config", None)
            if gateway is not None and getattr(gateway, "mode", None) == "process" and bool(getattr(supervision, "enable", False)):
                supervisor_task = asyncio.create_task(
                    self._gateway_supervision_loop(
                        trainer_state=trainer_state,
                        buffer=buffer,
                        coordinator=coordinator,
                        wave_controller=wave_controller,
                    ),
                    name="gateway-supervisor",
                )
            warm_queue = self._start_train_warm_queue(
                train_dataloader,
                trainer_state,
                self.rllm_config.trainer.total_epochs,
                self.rllm_config.trainer.get("total_batches", -1) > 0,
                hooks,
                loader_batches_per_training_step=self.async_config.mini_batch_size,
            )
            if wave_controller is None:
                gen_task = asyncio.create_task(
                    self._generation_loop(
                        trainer_state,
                        buffer,
                        coordinator,
                    )
                )
                await self._training_loop(
                    trainer_state,
                    buffer,
                    coordinator,
                    aggregator,
                )
            else:
                gen_task = asyncio.create_task(
                    self._generation_loop(
                        trainer_state,
                        buffer,
                        coordinator,
                        wave_controller,
                    )
                )
                await self._training_loop(
                    trainer_state,
                    buffer,
                    coordinator,
                    aggregator,
                    wave_controller,
                )
            if isinstance(train_dataloader, DynamicSamplingTaskDataLoader):
                trainer_state.force_final_checkpoint = True
                self._log_remaining_dynamic_sampling_metrics(
                    trainer_state,
                    train_dataloader,
                )
        except BaseException as exc:
            active_error = exc
            extra_info = getattr(trainer_state, "extra_info", None)
            if not isinstance(extra_info, dict):
                extra_info = {}
                trainer_state.extra_info = extra_info
            extra_info["async_failure_state"] = {
                "buffer": buffer.stats(),
                "coordinator": coordinator.stats(),
                "wave": (wave_controller.state_dict() if wave_controller is not None else None),
            }
            raise
        finally:
            from rllm.utils.bounded_cleanup import bounded_cleanup

            async def finish_cleanup(awaitable):
                if getattr(hooks, "sandbox_backend", None) == "minisandbox":
                    return await bounded_cleanup(awaitable, timeout=120)
                return await awaitable

            cleanup_error: BaseException | None = None
            if supervisor_task is not None:
                supervisor_task.cancel()
                try:
                    await finish_cleanup(supervisor_task)
                except asyncio.CancelledError:
                    pass
                except BaseException as exc:
                    cleanup_error = exc
            if wave_controller is not None:
                try:
                    await finish_cleanup(wave_controller.shutdown())
                except BaseException as exc:
                    cleanup_error = exc
            if gen_task is not None:
                if not gen_task.done():
                    gen_task.cancel()
                try:
                    await finish_cleanup(gen_task)
                except asyncio.CancelledError:
                    pass
                except BaseException as exc:
                    cleanup_error = exc

            # The generation coroutine only dispatches rollout child tasks;
            # cancelling it does not cancel those children. Drain them here so
            # each child's attempt-finally can delete its gateway session
            # before the gateway/backend begins shutting down.
            coordinator.cancel_tracked_tasks()
            try:
                await finish_cleanup(coordinator.wait_for_drain())
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
            ordinary_deferred = denovo_background_finalize and wave_controller is None
            ordinary_tail_batches = []
            if ordinary_deferred:
                try:
                    ordinary_tail_batches = await finish_cleanup(buffer.cleanup_deferred_uncommitted(
                        disposition="fatal_untrained_tail" if active_error is not None else "untrained_tail",
                    ))
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
            if active_error is not None:
                try:
                    fatal_tail_batches = (
                        ordinary_tail_batches if ordinary_deferred
                        else await buffer.drain_untrained_tail(disposition="fatal_untrained_tail")
                    )
                    if fatal_tail_batches and not ordinary_deferred:
                        self._log_dynamic_sampling_task_batches(
                            fatal_tail_batches,
                            optimizer_step=None,
                            optimizer_committed=False,
                            training_disposition="fatal_untrained_tail",
                        )
                    failure_state = trainer_state.extra_info.get(
                        "async_failure_state"
                    )
                    if isinstance(failure_state, dict):
                        failure_state["fatal_untrained_tail"] = {
                            "groups": len(fatal_tail_batches),
                            "rollouts": sum(
                                len(batch.episodes)
                                for batch in fatal_tail_batches
                            ),
                            "source_tickets_resolved": False,
                        }
                        failure_state["buffer_after_fatal_tail"] = (
                            buffer.stats()
                        )
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
            pbar.close()
            if wave_controller is not None and active_error is None:
                final_wave_metrics = wave_controller.drain_metrics()
                final_wave_metrics.update(await self._gateway_routing_metrics())
                final_wave_metrics["training/non_optimizer_event"] = 1
                final_wave_metrics["training/attempted_step"] = int(trainer_state.global_step)
                self.logger.log(
                    data=final_wave_metrics,
                    step=trainer_state.last_completed_step or 0,
                )
            if warm_queue is not None:
                warm_queue.shutdown()
                hooks.warm_queue = None
            if active_error is None and isinstance(
                train_dataloader,
                DynamicSamplingTaskDataLoader,
            ):
                self._log_remaining_dynamic_sampling_metrics(
                    trainer_state,
                    train_dataloader,
                )
            if cleanup_error is not None:
                if active_error is not None:
                    logger.exception(
                        "fully-async rollout drain also failed during training error",
                        exc_info=cleanup_error,
                    )
                else:
                    raise cleanup_error

    async def _gateway_supervision_loop(self, *, trainer_state, buffer, coordinator, wave_controller) -> None:
        worker_probe_state = {"task": None}
        try:
            await self._gateway_supervision_loop_impl(
                trainer_state=trainer_state, buffer=buffer, coordinator=coordinator,
                wave_controller=wave_controller, worker_probe_state=worker_probe_state,
            )
        finally:
            task = worker_probe_state["task"]
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def _gateway_supervision_loop_impl(
        self,
        *,
        trainer_state: TrainerState,
        buffer: TrajectoryGroupBuffer,
        coordinator: SyncCoordinator,
        wave_controller: DynamicSamplingWaveController | None,
        worker_probe_state: dict,
    ) -> None:
        """Monitor and recover the process gateway without consuming tasks."""
        gateway = self._gateway
        assert gateway is not None
        config = gateway.supervision_config
        consecutive_failures = 0
        soft_failure_started_at: float | None = None
        previous_failure_classification: str | None = None
        workers_unavailable_started_at: float | None = None
        last_metrics_log = 0.0
        last_snapshot_probe = 0.0
        warned_idle = False
        workers = None
        last_worker_probe = 0.0
        last_worker_warning = float("-inf")
        last_worker_signature = None
        while True:
            await asyncio.sleep(config.health_interval_seconds)
            hooks = getattr(getattr(self, "agent_workflow_engine", None), "hooks", None)
            cluster = getattr(hooks, "_minisandbox_cluster", None)
            if cluster is not None and cluster.fatal_error:
                coordinator.pause_generation()
                try:
                    cluster.check_health()
                except Exception as failure:
                    buffer.fail_generation(failure)
                coordinator.cancel_tracked_tasks()
                return
            probe: dict[str, Any] | None = None
            failure: BaseException | None = None
            failure_classification: str | None = None
            try:
                liveness_probe = getattr(gateway, "aprobe_liveness", None)
                if callable(liveness_probe):
                    probe = await liveness_probe()
                    snapshot_probe = getattr(
                        gateway,
                        "aget_supervision_snapshot_best_effort",
                        None,
                    )
                    snapshot_now = time.monotonic()
                    if (
                        callable(snapshot_probe)
                        and snapshot_now - last_snapshot_probe >= 30.0
                    ):
                        last_snapshot_probe = snapshot_now
                        snapshot = await snapshot_probe()
                        if snapshot is not None:
                            liveness_runtime = probe.get("runtime", {})
                            probe = {
                                **probe,
                                "health": snapshot.get(
                                    "health", probe.get("health", {})
                                ),
                                "runtime": {
                                    **snapshot.get("runtime", {}),
                                    **liveness_runtime,
                                },
                            }
                            probe.setdefault("liveness", {})[
                                "snapshot_available"
                            ] = 1
                else:
                    # Compatibility path for test doubles and non-process
                    # managers. Production process gateways always expose the
                    # isolated liveness method.
                    probe = await gateway.aprobe_health()
                worker_probe = getattr(gateway, "aprobe_workers", None)
                worker_task = worker_probe_state["task"]
                if worker_task is not None and worker_task.done():
                    worker_probe_state["task"] = None
                    try:
                        workers = worker_task.result()
                    except Exception as exc:
                        workers = [{"healthy": False, "error": type(exc).__name__}]
                    unhealthy = [row for row in workers if not row.get("healthy")]
                    signature = tuple(sorted((str(row.get("url", "")), str(row.get("error", row.get("status_code", "")))) for row in unhealthy))
                    now = time.monotonic()
                    if (unhealthy or not workers) and (signature != last_worker_signature or now - last_worker_warning >= 60.0):
                        logger.warning("Inference worker health degraded: unhealthy=%d/%d errors=%s", len(unhealthy), len(workers), sorted({str(row.get("error", row.get("status_code"))) for row in unhealthy}))
                        last_worker_warning = now
                    last_worker_signature = signature
                if callable(worker_probe) and worker_probe_state["task"] is None and time.monotonic() - last_worker_probe >= 30.0:
                    last_worker_probe = time.monotonic()
                    worker_probe_state["task"] = asyncio.create_task(worker_probe(timeout=getattr(gateway, "worker_health_check_timeout", 15.0)))
                if workers is not None:
                    probe["workers"] = workers
                runtime = probe.get("runtime", {})
                pending = int(runtime.get("cleanup_pending", 0) or 0)
                progress_age = float(runtime.get("cleanup_progress_age_seconds", 0.0) or 0.0)
                if pending > 0 and progress_age >= config.recovery_stall_timeout_seconds:
                    failure = TimeoutError(
                        "gateway cleanup backlog stalled "
                        f"(pending={pending}, progress_age={progress_age:.1f}s)"
                    )
                    failure_classification = "soft_cleanup_stall"
                    soft_failure_started_at = min(
                        soft_failure_started_at or time.monotonic(),
                        time.monotonic() - progress_age,
                    )
                heartbeat_age = runtime.get(
                    "event_loop_heartbeat_age_seconds"
                )
                data_plane_age = runtime.get(
                    "seconds_since_last_proxy_success"
                )
                if (
                    failure is None
                    and isinstance(heartbeat_age, int | float)
                    and not isinstance(heartbeat_age, bool)
                    and isinstance(data_plane_age, int | float)
                    and not isinstance(data_plane_age, bool)
                    and heartbeat_age
                    >= config.recovery_stall_timeout_seconds
                    and data_plane_age
                    >= config.recovery_stall_timeout_seconds
                ):
                    failure = TimeoutError(
                        "gateway main loop and data plane are both stale "
                        f"(heartbeat_age={heartbeat_age:.1f}s, "
                        f"data_plane_age={data_plane_age:.1f}s)"
                    )
                    failure_classification = "soft_runtime_stall"
                    inferred_start = time.monotonic() - min(
                        float(heartbeat_age),
                        float(data_plane_age),
                    )
                    soft_failure_started_at = min(
                        soft_failure_started_at or inferred_start,
                        inferred_start,
                    )
                # Worker health is an upstream signal, not evidence that the
                # Gateway is stuck. In particular, one busy /health endpoint
                # must not destroy a wave that is still producing responses.
                all_workers_unavailable = (
                    workers is not None
                    and not any(row.get("healthy") for row in workers)
                )
                no_proxy_progress = (
                    isinstance(data_plane_age, int | float)
                    and not isinstance(data_plane_age, bool)
                    and data_plane_age >= config.recovery_stall_timeout_seconds
                )
                if all_workers_unavailable and no_proxy_progress:
                    if workers_unavailable_started_at is None:
                        workers_unavailable_started_at = time.monotonic()
                    if failure is None:
                        failure = RuntimeError("All inference workers unavailable without proxy progress")
                        failure_classification = "soft_workers_unavailable"
                else:
                    workers_unavailable_started_at = None
            except asyncio.CancelledError:
                raise
            except BaseException as exc:  # noqa: BLE001 - supervisor boundary
                failure = exc
                process_alive = bool(
                    gateway.process_stats().get("alive", 0)
                )
                declared_kind = getattr(
                    exc,
                    "gateway_failure_kind",
                    None,
                )
                if declared_kind == "hard" or not process_alive:
                    failure_classification = (
                        "hard_process_exit"
                        if not process_alive
                        else "hard_identity"
                    )
                else:
                    failure_classification = "soft_transport"
                    if soft_failure_started_at is None:
                        soft_failure_started_at = time.monotonic()

            if bool(getattr(wave_controller, "cleanup_barrier_stalled", False)):
                failure = TimeoutError(
                    "speculative wave cleanup barrier is stalled; a successor "
                    "wave cannot safely reuse the Gateway"
                )
                failure_classification = "hard_wave_cleanup_barrier"
                soft_failure_started_at = None

            # Do not borrow an earlier incident's age or failure count when
            # the cause changes (e.g. worker degradation -> transport error).
            if failure_classification != previous_failure_classification:
                consecutive_failures = 0
                if failure_classification == "soft_runtime_stall":
                    soft_failure_started_at = time.monotonic() - min(float(heartbeat_age), float(data_plane_age))
                elif failure_classification == "soft_cleanup_stall":
                    soft_failure_started_at = time.monotonic() - progress_age
                else:
                    soft_failure_started_at = time.monotonic() if failure is not None else None
                previous_failure_classification = failure_classification
            if failure_classification == "soft_workers_unavailable":
                soft_failure_started_at = workers_unavailable_started_at
            else:
                workers_unavailable_started_at = None

            if failure is not None:
                consecutive_failures += 1
                if (
                    consecutive_failures <= config.failure_threshold
                    or consecutive_failures % 12 == 0
                ):
                    logger.warning(
                        "Gateway supervision probe degraded class=%s "
                        "(%d/%d, soft_age=%.1fs): %r",
                        failure_classification,
                        consecutive_failures,
                        config.failure_threshold,
                        (
                            max(
                                0.0,
                                time.monotonic()
                                - soft_failure_started_at,
                            )
                            if soft_failure_started_at is not None
                            else 0.0
                        ),
                        failure,
                    )
            else:
                consecutive_failures = 0
                soft_failure_started_at = None

            now = time.monotonic()
            if now - last_metrics_log >= 30.0:
                metrics = self._gateway_supervision_metrics(
                    probe=probe,
                    consecutive_failures=consecutive_failures,
                    wave_controller=wave_controller,
                    buffer=buffer,
                )
                metrics["training/attempted_step"] = trainer_state.global_step
                metrics["training/last_committed_step"] = (
                    trainer_state.last_completed_step
                    if trainer_state.last_completed_step is not None
                    else -1
                )
                logger.info(
                    "Gateway runtime: alive=%s restarts=%s rss_mb=%.1f "
                    "failures=%s sessions=%s cleanup_pending=%s "
                    "event_loop_lag=%.3fs attempted_step=%s "
                    "last_committed_step=%s phase=%s step_elapsed=%.1fs "
                    "wave=%s accepted=%s/%s dispatched=%s active=%s "
                    "pool_eligible=%s pool_deferred=%s oldest_active=%.1fs "
                    "one_rollout_remaining=%s",
                    metrics.get("gateway/process/alive", 0),
                    metrics.get("gateway/process/restarts", 0),
                    metrics.get("gateway/process/rss_mb", 0.0),
                    consecutive_failures,
                    metrics.get("gateway/state/sessions", 0),
                    metrics.get("gateway/session_cleanup/pending", 0),
                    metrics.get("gateway/event_loop/lag_seconds", 0.0),
                    trainer_state.global_step,
                    trainer_state.last_completed_step,
                    trainer_state.optimizer_phase,
                    metrics.get("training/current_step_elapsed_seconds", 0.0),
                    metrics.get("dynamic_sampling/wave/current_index", 0),
                    metrics.get("dynamic_sampling/wave/current_accepted", 0),
                    metrics.get("dynamic_sampling/wave/current_target", 0),
                    metrics.get("dynamic_sampling/wave/current_dispatched", 0),
                    metrics.get("dynamic_sampling/wave/in_flight_groups", 0),
                    metrics.get(
                        "dynamic_sampling/wave/current_pool_eligible",
                        0,
                    ),
                    metrics.get(
                        "dynamic_sampling/wave/current_pool_deferred",
                        0,
                    ),
                    metrics.get(
                        "dynamic_sampling/wave/oldest_in_flight_age_seconds",
                        0.0,
                    ),
                    metrics.get(
                        "dynamic_sampling/wave/groups_one_rollout_remaining",
                        0,
                    ),
                )
                self.logger.log(
                    data=metrics,
                    step=trainer_state.last_completed_step or 0,
                )
                last_metrics_log = now

            runtime = (probe or {}).get("runtime", {})
            since_success = runtime.get("seconds_since_last_proxy_success")
            in_flight = int(coordinator.stats().get("async/in_flight_groups", 0))
            if isinstance(since_success, int | float) and since_success >= 60 and in_flight > 0:
                if not warned_idle:
                    logger.critical(
                        "Gateway has %d in-flight groups but no successful proxy request for %.1fs",
                        in_flight,
                        since_success,
                    )
                    warned_idle = True
            else:
                warned_idle = False

            if failure is None:
                continue

            is_hard_failure = bool(
                failure_classification
                and failure_classification.startswith("hard_")
            )
            if not is_hard_failure:
                soft_age = (
                    max(0.0, time.monotonic() - soft_failure_started_at)
                    if soft_failure_started_at is not None
                    else 0.0
                )
                if (
                    consecutive_failures < config.failure_threshold
                    or soft_age < config.recovery_stall_timeout_seconds
                ):
                    continue

                # A soft incident becomes destructive only after a fresh,
                # extended liveness confirmation also reports failure/stall.
                confirmation_timeout = max(
                    15.0,
                    5.0 * config.health_timeout_seconds,
                )
                try:
                    liveness_probe = getattr(
                        gateway,
                        "aprobe_liveness",
                        None,
                    )
                    if callable(liveness_probe):
                        confirmation = await liveness_probe(
                            timeout=confirmation_timeout
                        )
                    else:
                        confirmation = await gateway.aprobe_health()
                    worker_probe = getattr(gateway, "aprobe_workers", None)
                    if failure_classification == "soft_workers_unavailable" and callable(worker_probe):
                        workers = await worker_probe(timeout=confirmation_timeout)
                        # /health may itself take the whole extended timeout.
                        # Re-read proxy progress before acting on that result.
                        confirmation = (
                            await liveness_probe(timeout=confirmation_timeout)
                            if callable(liveness_probe) else await gateway.aprobe_health()
                        )
                    if workers is not None:
                        confirmation["workers"] = workers
                    if failure_classification == "soft_cleanup_stall":
                        snapshot_probe = getattr(gateway, "aget_supervision_snapshot_best_effort", None)
                        snapshot = await snapshot_probe() if callable(snapshot_probe) else None
                        if snapshot is not None:
                            confirmation["runtime"] = {
                                **snapshot.get("runtime", {}),
                                **confirmation.get("runtime", {}),
                            }
                    confirmation_runtime = confirmation.get("runtime", {})
                    heartbeat_age = confirmation_runtime.get(
                        "event_loop_heartbeat_age_seconds"
                    )
                    data_plane_age = confirmation_runtime.get(
                        "seconds_since_last_proxy_success"
                    )
                    confirmed_stall = (
                        isinstance(heartbeat_age, int | float)
                        and not isinstance(heartbeat_age, bool)
                        and isinstance(data_plane_age, int | float)
                        and not isinstance(data_plane_age, bool)
                        and heartbeat_age
                        >= config.recovery_stall_timeout_seconds
                        and data_plane_age
                        >= config.recovery_stall_timeout_seconds
                    )
                    confirmed_cleanup_stall = (
                        confirmation_runtime.get("cleanup_pending", 0) > 0
                        and confirmation_runtime.get("cleanup_progress_age_seconds", 0) >= config.recovery_stall_timeout_seconds
                    )
                    confirmed_worker_failure = False
                    if failure_classification == "soft_workers_unavailable" and callable(worker_probe):
                        confirmed_worker_failure = (
                            workers is not None
                            and not any(row.get("healthy") for row in workers)
                            and isinstance(data_plane_age, int | float)
                            and not isinstance(data_plane_age, bool)
                            and data_plane_age >= config.recovery_stall_timeout_seconds
                        )
                    if not (confirmed_stall or confirmed_cleanup_stall or confirmed_worker_failure):
                        logger.info(
                            "Gateway soft degradation cleared during extended "
                            "confirmation; rollout generation remained active"
                        )
                        consecutive_failures = 0
                        soft_failure_started_at = None
                        workers_unavailable_started_at = None
                        previous_failure_classification = None
                        continue
                    probe = confirmation
                    if confirmed_stall or confirmed_cleanup_stall:
                        failure = TimeoutError("gateway soft runtime stall confirmed over isolated channel")
                        failure_classification = (
                            "soft_cleanup_stall_confirmed" if confirmed_cleanup_stall else "soft_runtime_stall_confirmed"
                        )
                    else:
                        failure = RolloutInfrastructureError(
                            "inference_workers_unavailable",
                            "All inference workers remain unavailable without proxy progress after extended confirmation",
                            stage="generation", retryable=False, retry_scope="none",
                            diagnostics={"workers": workers, "runtime": confirmation_runtime},
                        )
                        failure_classification = "soft_workers_unavailable_confirmed"
                except asyncio.CancelledError:
                    raise
                except BaseException as confirmation_error:  # noqa: BLE001
                    failure = confirmation_error
                    if (
                        getattr(
                            confirmation_error,
                            "gateway_failure_kind",
                            None,
                        )
                        == "hard"
                    ):
                        failure_classification = "hard_confirmation"
                    else:
                        failure_classification = "soft_transport_confirmed"

            if failure_classification == "soft_workers_unavailable_confirmed":
                # Restarting a healthy sidecar cannot repair inference workers.
                # Preserve loader tickets for resume and wake the optimizer wait.
                coordinator.pause_generation()
                self._append_gateway_recovery_manifest(
                    trainer_state, status="failed", task_ids=[], error=failure,
                    failure_classification=failure_classification, probe=probe,
                )
                buffer.fail_generation(failure)
                coordinator.cancel_tracked_tasks()
                return

            recovery_error = RuntimeError(f"gateway unavailable after {consecutive_failures} consecutive probes: {failure!r}")
            recovery_error.__cause__ = failure
            if wave_controller is None:
                # Ordinary loaders cannot atomically abandon and requeue active
                # groups. Restarting a memory-store gateway would strand their
                # sessions. Fail the optimizer wait and cancel generation so a
                # checkpoint resume can replay from the committed boundary.
                coordinator.pause_generation()
                gateway.mark_unavailable()
                self._append_gateway_recovery_manifest(
                    trainer_state,
                    status="failed",
                    task_ids=[],
                    error=recovery_error,
                    failure_classification=failure_classification,
                    probe=probe,
                )
                logger.error(
                    "Gateway unavailable in ordinary fully-async training; "
                    "aborting in-flight rollouts for checkpoint resume: %r",
                    recovery_error,
                )
                buffer.fail_generation(recovery_error)
                coordinator.cancel_tracked_tasks()
                return
            detected_phase = trainer_state.optimizer_phase
            if detected_phase == "weight_sync":
                buffer.fail_generation(recovery_error)
                return

            try:
                async with self._gateway_recovery_lock:
                    if trainer_state.optimizer_phase != detected_phase:
                        # A pre-weight-sync recovery may have repaired the
                        # gateway while this supervisor was queued on the same
                        # lock. Discard the stale probe instead of poisoning the
                        # now-healthy weight-sync boundary.
                        logger.info(
                            "Discarding stale Gateway supervision failure "
                            "after optimizer phase changed from %s to %s",
                            detected_phase,
                            trainer_state.optimizer_phase,
                        )
                        consecutive_failures = 0
                        continue
                    if trainer_state.optimizer_phase == "weight_sync":
                        raise RuntimeError(
                            "gateway failed at an ambiguous weight-sync boundary"
                        )
                    coordinator.pause_generation()
                    gateway.mark_unavailable()
                    task_ids = await wave_controller.abort_active_for_recovery()
                    abandoned_session_ids = task_group_session_ids(
                        task_ids,
                        int(self.rllm_config.rollout.n),
                        max_attempts=_configured_workflow_retry_limit(
                            self.rllm_config
                        ),
                    )
                    self._append_gateway_recovery_manifest(
                        trainer_state,
                        status="started",
                        task_ids=task_ids,
                        error=failure,
                        failure_classification=failure_classification,
                        probe=probe,
                    )
                    await gateway.arestart(
                        abandoned_session_ids=abandoned_session_ids,
                    )
                    worker_probe = getattr(gateway, "aprobe_workers", None)
                    if callable(worker_probe):
                        workers = await worker_probe()
                        if not workers or not all(row.get("healthy") for row in workers):
                            raise RuntimeError("Gateway workers unavailable after recovery")
                    note_ready = getattr(
                        wave_controller,
                        "note_gateway_cleanup_ready",
                        None,
                    )
                    if callable(note_ready):
                        note_ready()
                    self._append_gateway_recovery_manifest(
                        trainer_state,
                        status="recovered",
                        task_ids=task_ids,
                        error=failure,
                        failure_classification=failure_classification,
                        probe=probe,
                    )
            except asyncio.CancelledError:
                raise
            except BaseException as exc:  # noqa: BLE001 - fatal recovery boundary
                self._append_gateway_recovery_manifest(
                    trainer_state,
                    status="failed",
                    task_ids=locals().get("task_ids", []),
                    error=exc,
                    failure_classification=failure_classification,
                    probe=probe,
                )
                fatal_error = RuntimeError(f"gateway automatic recovery failed: {type(exc).__name__}: {exc}; triggering probe: {type(failure).__name__}: {failure!r}")
                fatal_error.__cause__ = exc
                buffer.fail_generation(fatal_error)
                return
            consecutive_failures = 0
            soft_failure_started_at = None
            workers_unavailable_started_at = None
            previous_failure_classification = None
            coordinator.resume_generation()

    def _gateway_supervision_metrics(
        self,
        *,
        probe: dict[str, Any] | None,
        consecutive_failures: int,
        wave_controller: DynamicSamplingWaveController | None,
        buffer: TrajectoryGroupBuffer,
    ) -> dict[str, int | float]:
        gateway = self._gateway
        assert gateway is not None
        values = {f"gateway/process/{key}": value for key, value in gateway.process_stats().items()}
        values["gateway/health/consecutive_failures"] = consecutive_failures
        runtime = (probe or {}).get("runtime", {})
        runtime_mapping = {
            "seconds_since_last_proxy_success": "gateway/health/seconds_since_last_proxy_success",
            "event_loop_lag_seconds": "gateway/event_loop/lag_seconds",
            "event_loop_heartbeat_age_seconds": "gateway/event_loop/heartbeat_age_seconds",
            "sessions": "gateway/state/sessions",
            "accumulators": "gateway/state/accumulators",
            "idempotent_requests": "gateway/state/idempotent_requests",
            "idempotent_in_flight": "gateway/state/idempotent_in_flight",
            "idempotent_replayable": "gateway/state/idempotent_replayable",
            "trace_store_live_traces": "gateway/trace_store/live_traces",
            "trace_store_compressed_bytes": "gateway/trace_store/compressed_bytes",
            "trace_store_resident_compressed_bytes": "gateway/trace_store/resident_compressed_bytes",
            "trace_store_spilled_compressed_bytes": "gateway/trace_store/spilled_compressed_bytes",
            "trace_store_spill_files": "gateway/trace_store/spill_files",
            "trace_store_uncompressed_bytes": "gateway/trace_store/uncompressed_bytes",
            "trace_store_read_codec_queue_depth": "gateway/trace_store/read_codec_queue_depth",
            "trace_store_write_codec_queue_depth": "gateway/trace_store/write_codec_queue_depth",
            "trace_store_read_codec_workers": "gateway/trace_store/read_codec_workers",
            "trace_store_write_codec_workers": "gateway/trace_store/write_codec_workers",
            "cleanup_pending": "gateway/session_cleanup/pending",
            "cleanup_late_tasks": "gateway/session_cleanup/late_tasks",
            "cleanup_progress_age_seconds": "gateway/session_cleanup/progress_age_seconds",
        }
        for source, target in runtime_mapping.items():
            value = runtime.get(source)
            if isinstance(value, int | float) and not isinstance(value, bool):
                values[target] = value
        gateway_runtime_metrics = getattr(gateway, "runtime_metrics", None)
        if callable(gateway_runtime_metrics):
            values.update(gateway_runtime_metrics())
        if wave_controller is not None:
            values.update(wave_controller.runtime_metrics())
        lifecycle_metrics = getattr(
            getattr(self, "agent_workflow_engine", None),
            "lifecycle_runtime_metrics",
            None,
        )
        if callable(lifecycle_metrics):
            values.update(lifecycle_metrics())
        values.update(buffer.stats())
        step_started_at = getattr(
            self,
            "_async_training_step_started_at_monotonic",
            None,
        )
        values["training/current_step_elapsed_seconds"] = (
            max(0.0, time.monotonic() - step_started_at)
            if isinstance(step_started_at, int | float)
            else 0.0
        )
        return values

    def _append_gateway_recovery_manifest(
        self,
        trainer_state: TrainerState,
        *,
        status: str,
        task_ids: list[str],
        error: BaseException,
        failure_classification: str | None = None,
        probe: dict[str, Any] | None = None,
    ) -> None:
        explicit_params = os.environ.get("RLLM_TRAINING_PARAMS_PATH")
        if explicit_params:
            output_dir = Path(explicit_params).expanduser().parent
        else:
            configured = self.rllm_config.trainer.get("wandb_dir", None)
            if not configured:
                logger.warning("Cannot write gateway recovery manifest")
                return
            output_dir = Path(str(configured)).expanduser()
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / "gateway_recovery_manifest.jsonl"
            payload = {
                "schema_version": 2,
                "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": status,
                "last_committed_step": trainer_state.last_completed_step,
                "attempted_step": trainer_state.global_step,
                "optimizer_phase": trainer_state.optimizer_phase,
                "task_ids": task_ids,
                "session_count": len(task_ids) * int(self.rllm_config.rollout.n),
                "physical_session_count": (
                    len(task_ids)
                    * int(self.rllm_config.rollout.n)
                    * _configured_workflow_retry_limit(self.rllm_config)
                ),
                "max_attempts_per_rollout": _configured_workflow_retry_limit(
                    self.rllm_config
                ),
                "error_type": type(error).__name__,
                "error": str(error),
                "failure_classification": failure_classification,
                "workers": (probe or {}).get("workers"),
                "trigger_runtime": (probe or {}).get("runtime"),
                "event_loop_heartbeat_age_seconds": (probe or {})
                .get("runtime", {})
                .get("event_loop_heartbeat_age_seconds"),
                "seconds_since_last_proxy_success": (probe or {})
                .get("runtime", {})
                .get("seconds_since_last_proxy_success"),
                "snapshot_available": (probe or {})
                .get("liveness", {})
                .get("snapshot_available"),
                "process": self._gateway.process_stats(),
            }
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, default=str))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            logger.exception("Failed to append gateway recovery manifest")

    async def _generation_loop(
        self,
        trainer_state: TrainerState,
        buffer: TrajectoryGroupBuffer,
        coordinator: SyncCoordinator,
        wave_controller: DynamicSamplingWaveController | None = None,
    ) -> None:
        """Generate episodes and stream to TrajectoryGroupBuffer."""
        group_size = self.rllm_config.rollout.n
        train_dataloader = self._train_dataloader
        assert train_dataloader is not None, "train_dataset is required for training"

        active_epoch: int | None = None
        pending_dispatch = None
        terminal_kind: GenerationTerminalKind = "dataset_exhausted"
        try:
            while True:
                # Wait before advancing the dataloader. This prevents a batch
                # from falling between the saved cursor and the dispatch set.
                await coordinator.wait_interruptibly(coordinator.wait_for_generation_allowed())
                if wave_controller is not None:
                    if not await coordinator.wait_interruptibly(wave_controller.wait_for_dispatch()):
                        terminal_kind = "controlled_stop"
                        break
                    # wait_interruptibly runs the admission coroutine as a
                    # separate task and yields again while cleaning it up.
                    # A rollout can close the wave before this task resumes.
                    if not wave_controller.can_dispatch():
                        continue
                elif not coordinator.has_quota():
                    await coordinator.wait_interruptibly(coordinator.wait_for_throttle())

                item = pending_dispatch
                if item is None:
                    item = train_dataloader.next_dispatch(self.rllm_config.trainer.total_epochs)
                if item is None:
                    if wave_controller is not None:
                        if (
                            wave_controller.promote_deferred_retries_in_current_wave()
                            > 0
                        ):
                            continue
                        if wave_controller.note_pool_unavailable():
                            break
                        await coordinator.wait_interruptibly(wave_controller.wait_for_change())
                        continue
                    if isinstance(
                        train_dataloader,
                        DynamicSamplingTaskDataLoader,
                    ) and not train_dataloader.generation_exhausted(self.rllm_config.trainer.total_epochs):
                        # The task pool may be temporarily empty while every
                        # dispatched task is waiting for outcome classification.
                        # Feedback either requeues or resolves one and wakes us.
                        await coordinator.wait_interruptibly(train_dataloader.wait_for_feedback())
                        continue
                    break
                pending_dispatch = item
                batch, source_ticket = item
                epoch = source_ticket.epoch
                if epoch != active_epoch:
                    if active_epoch is not None:
                        trainer_state.epoch = active_epoch
                        await self.backend.on_epoch_end(trainer_state)
                    active_epoch = epoch
                    trainer_state.epoch = epoch
                    await self.backend.on_epoch_start(trainer_state)
                    self.agent_workflow_engine.set_training_step(
                        trainer_state.global_step,
                        mode="train",
                        epoch=epoch,
                    )

                # Epoch hooks may also yield. Retain an already-issued ticket
                # until admission reopens; the loader keeps it checkpointed
                # for replay if training stops while we are waiting. No await
                # may separate this final check from candidate registration.
                if wave_controller is not None and not wave_controller.can_dispatch():
                    continue

                task = batch[0]
                dispatch_step = trainer_state.global_step

                task_id = str(uuid.uuid4())
                sampling_wave_index: int | None = None
                if wave_controller is not None:
                    sampling_wave_index = wave_controller.register_candidate(
                        task_id,
                        source_ticket,
                    )
                    buffer.register_speculative_group(task_id, source_ticket)
                pending_dispatch = None
                coordinator.on_group_dispatched()
                rollout_tasks: list[asyncio.Task[Any]] = []
                for rollout_idx in range(group_size):
                    t = asyncio.create_task(
                        self._run_fully_async_rollout(
                            task=task,
                            task_id=task_id,
                            rollout_idx=rollout_idx,
                            result_idx=0,
                            dispatch_step=dispatch_step,
                            dispatch_weight_version=coordinator.weight_version,
                            sampling_wave_index=sampling_wave_index,
                            epoch=epoch,
                            buffer=buffer,
                            source_ticket=source_ticket,
                        ),
                        name=(f"fully-async-rollout:{task_id}:{rollout_idx}"),
                    )
                    coordinator.track_task(t)
                    rollout_tasks.append(t)
                if wave_controller is not None:
                    wave_controller.attach_tasks(task_id, rollout_tasks)

            if active_epoch is not None:
                trainer_state.epoch = active_epoch
                await self.backend.on_epoch_end(trainer_state)
            await coordinator.wait_for_drain()
        except asyncio.CancelledError:
            # Parent-driven shutdown owns the terminal disposition.  In
            # particular, reaching total_batches is a controlled stop rather
            # than a producer failure.
            raise
        except BaseException as exc:
            buffer.fail_generation(exc)
            raise
        else:
            buffer.finish_generation(terminal_kind)

    async def _run_fully_async_rollout(
        self,
        *,
        task: Any,
        task_id: str,
        rollout_idx: int,
        result_idx: int,
        dispatch_step: int,
        epoch: int,
        buffer: TrajectoryGroupBuffer,
        dispatch_weight_version: int = 0,
        sampling_wave_index: int | None = None,
        source_ticket: DataloaderBatchTicket | None = None,
    ) -> None:
        """Run one fully-async rollout and stream it to the training buffer."""
        task_id, rollout_idx, result_idx, episode = await self.agent_workflow_engine.process_task_with_retry(
            task=task,
            task_id=task_id,
            rollout_idx=rollout_idx,
            result_idx=result_idx,
        )
        should_persist_cancelled = getattr(
            buffer,
            "should_persist_cancelled_rollout",
            None,
        )
        persist_cancelled = bool(
            callable(should_persist_cancelled)
            and should_persist_cancelled(task_id, episode)
        )
        discard_if_cancelled = getattr(buffer, "discard_if_cancelled", None)
        if (
            not persist_cancelled
            and callable(discard_if_cancelled)
            and discard_if_cancelled(
                task_id,
                episode,
            )
        ):
            logger.info(
                "Discarded late detached rollout task=%s rollout_uid=%s:%s last_stage=agentflow_complete",
                task_id,
                task_id,
                rollout_idx,
            )
            return
        if source_ticket is not None:
            episode.metadata["dataloader_ticket"] = source_ticket.to_dict()
        episode.metadata["rollout_lifecycle_dispatch"] = {
            "schema_version": 1,
            "dispatch_step": int(dispatch_step),
            "dispatch_weight_version": int(dispatch_weight_version),
            "sampling_wave_index": sampling_wave_index,
        }

        maybe_log_rollout_sample = getattr(self.agent_workflow_engine, "_maybe_log_rollout_sample", None)
        if callable(maybe_log_rollout_sample):
            try:
                set_training_step = getattr(self.agent_workflow_engine, "set_training_step", None)
                if callable(set_training_step):
                    set_training_step(dispatch_step, mode="train", epoch=epoch)
                maybe_log_rollout_sample(task_id, rollout_idx, result_idx, episode)
            except Exception:
                logger.exception(
                    "Failed to log fully-async rollout sample %s at step=%s epoch=%s",
                    episode.id,
                    dispatch_step,
                    epoch,
                )

        if persist_cancelled:
            finalize_cancelled = getattr(
                buffer,
                "finalize_cancelled_rollout",
                None,
            )
            if not callable(finalize_cancelled) or not await finalize_cancelled(
                task_id,
                episode,
            ):
                raise RuntimeError(
                    "DeNovo cancelled rollout was selected for lifecycle logging "
                    "but could not be finalized"
                )
            logger.info(
                "Persisted late DeNovo rollout after shadow cancellation "
                "task=%s rollout_uid=%s:%s",
                task_id,
                task_id,
                rollout_idx,
            )
            return

        if self.episode_logger is not None:
            try:
                self.episode_logger.log_episode(episode, dispatch_step, mode="train", epoch=epoch)
            except Exception:
                logger.exception(
                    "Failed to log fully-async episode %s at step=%s epoch=%s",
                    episode.id,
                    dispatch_step,
                    epoch,
                )

        await buffer.add_episode(
            task_id,
            episode,
            source_ticket=source_ticket,
        )

    @staticmethod
    async def _reserve_async_task_batches(
        buffer: TrajectoryGroupBuffer,
        count: int,
    ) -> tuple[list[TaskBatch], bool]:
        """Reserve a complete optimizer batch before any forward/backward work.

        Production buffers expose ``get_many`` and leave a short normal tail in
        the queue until it is explicitly drained.  The sequential fallback is
        retained for lightweight/custom buffers used by downstream callers.
        Fatal generation errors raised by either path deliberately propagate.
        """
        get_many = getattr(buffer, "get_many", None)
        if callable(get_many):
            batches = await get_many(count)
            if batches is not None:
                return list(batches), True
            drain_tail = getattr(buffer, "drain_untrained_tail", None)
            if callable(drain_tail):
                return list(await drain_tail()), False
            return [], False

        batches: list[TaskBatch] = []
        for _ in range(count):
            batch = await buffer.get()
            if batch is None:
                return batches, False
            batches.append(batch)
        return batches, True

    @staticmethod
    def _flush_rollout_window_metrics(
        aggregator: MetricsAggregator,
    ) -> dict[str, float]:
        """Keep classification-window metrics distinct from optimizer data."""
        values = aggregator.flush()
        return {f"rollout_window/{key}": value for key, value in values.items()}

    def _log_dynamic_sampling_task_batches(
        self,
        task_batches: list[TaskBatch],
        *,
        optimizer_step: int | None,
        optimizer_committed: bool,
        training_disposition: str,
    ) -> None:
        dynamic_enabled = bool(getattr(
            getattr(self, "dynamic_sampling_config", None), "enable", False,
        ))
        for task_batch in task_batches:
            if not dynamic_enabled and not task_batch.deferred_shadow_finalize:
                continue
            episodes = task_batch.rollout_log_episodes or task_batch.episodes
            if not episodes:
                continue
            for episode_index, episode in enumerate(episodes):
                source_episode = (
                    task_batch.episodes[episode_index]
                    if episode_index < len(task_batch.episodes)
                    else episode
                )
                denovo_lifecycle = source_episode.metadata.get(
                    "denovo_background_finalize"
                )
                if isinstance(denovo_lifecycle, dict):
                    denovo_lifecycle.update(
                        {
                            "training_disposition": training_disposition,
                            "optimizer_step": optimizer_step,
                            "optimizer_committed": bool(optimizer_committed),
                        }
                    )
                    if episode is not source_episode:
                        episode.metadata["denovo_background_finalize"] = dict(
                            denovo_lifecycle
                        )
            filtered_ids = {
                id(episodes[episode_index].trajectories[trajectory_index])
                for episode_index, trajectory_index in (task_batch.filtered_trajectory_positions)
                if episode_index < len(episodes) and trajectory_index < len(episodes[episode_index].trajectories)
            }
            self._write_dynamic_sampling_rollouts(
                episodes,
                filtered_ids,
                lifecycle={
                    "schema_version": 1,
                    "optimizer_step": optimizer_step,
                    "optimizer_committed": bool(optimizer_committed),
                    "training_disposition": training_disposition,
                },
            )

    async def _training_loop(
        self,
        trainer_state: TrainerState,
        buffer: TrajectoryGroupBuffer,
        coordinator: SyncCoordinator,
        aggregator: MetricsAggregator,
        wave_controller: DynamicSamplingWaveController | None = None,
    ) -> None:
        """Consume task batches from buffer, run forward-backward + optimizer step."""
        mini_batch_size = self.async_config.mini_batch_size
        fwd_bwd_group_size = self.async_config.fwd_bwd_group_size
        num_fwd_bwd_passes = mini_batch_size // fwd_bwd_group_size
        use_total_batches = self.rllm_config.trainer.get("total_batches", -1) > 0
        rollout_engine = getattr(self.agent_workflow_engine, "rollout_engine", None)

        while True:
            trainer_state.reset_batch()
            trainer_state.optimizer_phase = "waiting_for_batch"
            step_start = time.perf_counter()
            self._async_training_step_started_at_monotonic = time.monotonic()
            weight_versions = []
            all_trajectory_groups: list[TrajectoryGroup] = []
            all_episodes: list[Episode] = []
            consumed_source_tickets: list[DataloaderBatchTicket | None] = []
            buffer_wait_time = 0.0

            buffered = buffer._queue.qsize()
            logger.info(
                f"[TrainingLoop] Step {trainer_state.global_step}: waiting for {mini_batch_size} task batches ({num_fwd_bwd_passes} fwd-bwd passes x {fwd_bwd_group_size} groups), {buffered} buffered"
            )

            # 1. Reserve the complete optimizer batch before starting any
            # forward/backward pass.  A fatal producer error can therefore not
            # leave partially accumulated gradients or turn 9/12 groups into a
            # normal drop-last tail.
            t_wait = time.perf_counter()
            task_batches, full_batch = await self._reserve_async_task_batches(
                buffer,
                mini_batch_size,
            )
            buffer_wait_time += time.perf_counter() - t_wait
            groups_consumed = len(task_batches)

            for task_batch in task_batches:
                coordinator.on_group_consumed()
                for group in task_batch.groups:
                    weight_versions.append(group.weight_version)
                all_trajectory_groups.extend(task_batch.groups)
                all_episodes.extend(task_batch.episodes)
                consumed_source_tickets.append(task_batch.source_ticket)

            # Only a true, non-fatal producer EOF reaches this branch.
            if not full_batch:
                logger.info(
                    "[TrainingLoop] Incomplete terminal batch for attempted step %s (%s/%s); applying normal drop-last semantics",
                    trainer_state.global_step,
                    groups_consumed,
                    mini_batch_size,
                )
                train_dataloader = self._train_dataloader
                if not isinstance(train_dataloader, DynamicSamplingTaskDataLoader):
                    self._log_dynamic_sampling_task_batches(
                        task_batches,
                        optimizer_step=None,
                        optimizer_committed=False,
                        training_disposition="untrained_tail",
                    )
                    release_reservation = getattr(buffer, "release_deferred_reservation", None)
                    if callable(release_reservation):
                        release_reservation(task_batches)
                if isinstance(train_dataloader, DynamicSamplingTaskDataLoader):
                    train_dataloader.mark_untrained_tail_many(consumed_source_tickets)
                    self._log_dynamic_sampling_task_batches(
                        task_batches,
                        optimizer_step=None,
                        optimizer_committed=False,
                        training_disposition="untrained_tail",
                    )
                    trainer_state.trajectory_groups = all_trajectory_groups
                    trainer_state.episodes = all_episodes
                    trainer_state.metrics.update(self._flush_rollout_window_metrics(aggregator))
                    trainer_state.metrics.update(buffer.stats())
                    trainer_state.metrics.update(coordinator.stats())
                    trainer_state.metrics.update({f"async/dataloader_{key}": value for key, value in train_dataloader.stats().items()})
                    trainer_state.metrics.update(self._drain_dynamic_sampling_metrics(train_dataloader))
                    if wave_controller is not None:
                        trainer_state.metrics.update(wave_controller.drain_metrics())
                    trainer_state.metrics.update(await self._gateway_routing_metrics())
                    trainer_state.metrics.update(
                        {
                            "training/terminal/normal_tail": 1,
                            "training/terminal/attempted_step": int(trainer_state.global_step),
                            "training/terminal/groups_available": groups_consumed,
                            "training/terminal/groups_required": mini_batch_size,
                        }
                    )
                    trainer_state.force_final_checkpoint = True
                    terminal_step = trainer_state.last_completed_step or 0
                    logger.info(
                        "Fully-Async Terminal Summary: last_committed_step=%s attempted_step=%s accepted_tail=%s/%s kind=dataset_exhausted",
                        trainer_state.last_completed_step,
                        trainer_state.global_step,
                        groups_consumed,
                        mini_batch_size,
                    )
                    self.logger.log(
                        data=trainer_state.metrics,
                        step=terminal_step,
                        episodes=trainer_state.episodes,
                        trajectory_groups=trainer_state.trajectory_groups,
                    )
                break

            # Split the already-reserved batch into forward/backward chunks.
            trainer_state.optimizer_phase = "forward_backward"
            step_aggregator = MetricsAggregator()
            # Report workflow quality for the exact groups assigned to this
            # optimizer transaction.  Classification metrics are a separate
            # rollout-completion window and may include filtered/speculative
            # work, so they cannot stand in for trained-batch metrics.
            workflow_metrics, termination_counts = self._collect_workflow_metrics_from_episodes(all_episodes)
            for key, value in workflow_metrics.items():
                step_aggregator.record(f"batch/{key}", value)
            total_terminations = max(sum(termination_counts.values()), 1)
            for reason in TerminationReason:
                step_aggregator.record(
                    f"batch/termination_reason/{reason.value}",
                    termination_counts[reason.value] / total_terminations,
                )
            for pass_idx in range(num_fwd_bwd_passes):
                chunk_batches = task_batches[pass_idx * fwd_bwd_group_size : (pass_idx + 1) * fwd_bwd_group_size]
                chunk_groups = [group for task_batch in chunk_batches for group in task_batch.groups]

                # Forward-backward on this chunk
                trainer_state.trajectory_groups = chunk_groups

                if trainer_state.has_trajectory_groups:
                    logger.info(f"[TrainingLoop] Step {trainer_state.global_step}: fwd-bwd pass {pass_idx + 1}/{num_fwd_bwd_passes} ({len(chunk_groups)} groups)")
                    await self.backend.on_batch_start(trainer_state)
                    trainer_state.backend_batch = self.backend.transform_to_backend_batch(trainer_state)
                    await self.backend.process_backend_batch(trainer_state)

                    # Drain per-chunk backend metrics into aggregator
                    step_aggregator.record_dict(trainer_state.metrics)
                    trainer_state.metrics = {}

            # 2. Optimizer step.  Ticket and wave state are committed only
            # after the required rollout-weight synchronization succeeds.
            logger.info(f"[TrainingLoop] Step {trainer_state.global_step}: optimizer step")
            trainer_state.optimizer_phase = "optimizer_update"
            try:
                await self.backend.update_policy(trainer_state)
            except BaseException:
                self._log_dynamic_sampling_task_batches(
                    task_batches,
                    optimizer_step=trainer_state.global_step,
                    optimizer_committed=False,
                    training_disposition="optimizer_failed",
                )
                raise
            train_dataloader = self._train_dataloader
            assert train_dataloader is not None

            # 3. Capture pre-sync metrics (before weight sync resets coordinator state)
            staleness_values = [coordinator.weight_version - v for v in weight_versions]
            step_aggregator.record("async/staleness_mean", float(np.mean(staleness_values)))
            step_aggregator.record("async/staleness_min", float(np.min(staleness_values)))
            step_aggregator.record("async/staleness_max", float(np.max(staleness_values)))
            step_aggregator.record("async/groups_consumed", groups_consumed)
            step_aggregator.record("train_step/groups_consumed", groups_consumed)
            step_aggregator.record(
                "train_step/rollouts_consumed",
                groups_consumed * int(coordinator.config.group_size),
            )
            step_aggregator.record("time/buffer_wait", buffer_wait_time)
            pre_sync_coordinator_stats = coordinator.stats()
            pre_sync_buffer_stats = buffer.stats()
            pre_sync_dataloader_stats = {f"async/dataloader_{key}": value for key, value in train_dataloader.stats().items()}

            # 4. Weight sync
            coordinator.on_training_step_complete()
            sync_time = 0.0
            trainer_state.optimizer_phase = "weight_sync_preflight"
            try:
                if coordinator.should_sync():
                    logger.info(f"[TrainingLoop] Step {trainer_state.global_step}: triggering weight sync")
                    t0 = time.perf_counter()
                    await self._perform_weight_sync(
                        trainer_state,
                        coordinator,
                        rollout_engine,
                        wave_controller=wave_controller,
                    )
                    sync_time = time.perf_counter() - t0
                    logger.info(f"[TrainingLoop] Step {trainer_state.global_step}: weight sync complete ({sync_time:.2f}s)")
            except BaseException:
                self._log_dynamic_sampling_task_batches(
                    task_batches,
                    optimizer_step=trainer_state.global_step,
                    optimizer_committed=False,
                    training_disposition="weight_sync_failed",
                )
                raise
            if sync_time > 0:
                step_aggregator.record("time/weight_sync", sync_time)
            step_time = time.perf_counter() - step_start
            step_aggregator.record("time/step", step_time)
            trainer_state.timing_dict["step"] = step_time

            # Set all trajectory groups and stripped episodes for visualization/logging
            trainer_state.trajectory_groups = all_trajectory_groups
            trainer_state.episodes = all_episodes

            if self.tokenizer is not None and trainer_state.has_trajectory_groups:
                visualize_trajectory_last_steps(
                    trainer_state.trajectory_groups,
                    tokenizer=self.tokenizer,
                    max_steps_to_visualize=2,
                    show_workflow_metadata=True,
                )

            # 5. Flush aggregator and merge pre-sync snapshots into trainer_state.metrics
            trainer_state.metrics.update(step_aggregator.flush())
            trainer_state.metrics.update(self._flush_rollout_window_metrics(aggregator))
            trainer_state.metrics.update(pre_sync_buffer_stats)
            trainer_state.metrics.update(pre_sync_coordinator_stats)
            trainer_state.metrics.update(pre_sync_dataloader_stats)
            if isinstance(train_dataloader, DynamicSamplingTaskDataLoader):
                trainer_state.metrics.update(self._drain_dynamic_sampling_metrics(train_dataloader))
                if wave_controller is not None:
                    trainer_state.metrics.update(wave_controller.drain_metrics())

            # 6. Compute derived metrics
            step_time = trainer_state.metrics.get("time/step", 1.0)
            trainer_state.metrics["async/trainer_idle_ratio"] = buffer_wait_time / max(step_time, 1e-9)

            # 7. on_batch_end writes backend metrics (progress, optim, timing)
            # Persist compact rollouts with their true optimizer assignment
            # before the dataloader ticket is made durable.
            self._log_dynamic_sampling_task_batches(
                task_batches,
                optimizer_step=trainer_state.global_step,
                optimizer_committed=True,
                training_disposition="trained",
            )
            train_dataloader.mark_resolved_many(consumed_source_tickets)
            release_reservation = getattr(buffer, "release_deferred_reservation", None)
            if callable(release_reservation):
                release_reservation(task_batches)

            await self.backend.on_batch_end(trainer_state)
            committed_step = trainer_state.global_step
            trainer_state.last_completed_step = committed_step
            trainer_state.optimizer_phase = "idle"
            trainer_state.metrics["training/step_committed"] = 1
            trainer_state.metrics["training/optimizer_step"] = committed_step
            trainer_state.metrics["training/last_committed_step"] = committed_step
            trainer_state.metrics.update(await self._gateway_routing_metrics())

            # 7. Print and log
            print_metrics_table(trainer_state.metrics, trainer_state.global_step)
            self.logger.log(
                data=trainer_state.metrics,
                step=trainer_state.global_step,
                episodes=trainer_state.episodes,
                trajectory_groups=trainer_state.trajectory_groups,
            )

            # Periodic validation
            if self.rllm_config.trainer.test_freq > 0 and trainer_state.global_step % self.rllm_config.trainer.test_freq == 0:
                await self._validate_async_with_pause(trainer_state, coordinator)

            trainer_state.global_step += 1
            buffer.set_training_step(trainer_state.global_step)
            if wave_controller is not None:
                wave_controller.on_optimizer_step_committed(
                    committed_step,
                    next_dispatch_step=trainer_state.global_step,
                    weight_version=coordinator.weight_version,
                )

            if use_total_batches and trainer_state.global_step >= self.rllm_config.trainer.total_batches:
                if wave_controller is not None:
                    # Freeze admission before draining the accepted cushion so
                    # no group can race into the buffer after the tail snapshot.
                    await wave_controller.shutdown()
                    tail_batches = await buffer.drain_untrained_tail()
                    if isinstance(
                        train_dataloader,
                        DynamicSamplingTaskDataLoader,
                    ):
                        train_dataloader.mark_untrained_tail_many([batch.source_ticket for batch in tail_batches])
                        denovo_tail_batches = [
                            batch
                            for batch in tail_batches
                            if batch.deferred_shadow_finalize
                        ]
                        if denovo_tail_batches:
                            self._log_dynamic_sampling_task_batches(
                                denovo_tail_batches,
                                optimizer_step=None,
                                optimizer_committed=False,
                                training_disposition="untrained_tail",
                            )
                        for _ in tail_batches:
                            coordinator.on_group_consumed()
                        if tail_batches:
                            trainer_state.force_final_checkpoint = True
                break

    async def _prepare_gateway_for_weight_sync(
        self,
        trainer_state: TrainerState,
        coordinator: SyncCoordinator,
        wave_controller: DynamicSamplingWaveController | None,
    ) -> bool:
        """Verify cleanup or recover the process gateway before weight mutation.

        Returns whether generation was paused for an automatic restart. The
        caller resumes it only after the complete policy/version sync commits.
        """
        gateway = self._gateway
        if gateway is None:
            return False

        barrier_ready = True
        wait_barrier = getattr(
            wave_controller,
            "wait_for_cleanup_barrier",
            None,
        )
        if callable(wait_barrier):
            barrier_ready = bool(await wait_barrier())

        wait_cleanup = getattr(gateway, "await_cleanup_ready", None)
        try:
            if not barrier_ready:
                raise TimeoutError(
                    "speculative Gateway cleanup barrier exceeded its deadline"
                )
            if callable(wait_cleanup):
                await wait_cleanup()
        except asyncio.CancelledError:
            raise
        except BaseException as trigger:  # noqa: BLE001 - recovery boundary
            if wave_controller is None:
                raise RuntimeError(
                    "Gateway is not cleanup-ready before weight sync and "
                    "there is no sampling-wave controller to requeue active "
                    "rollouts safely"
                ) from trigger
            restart = getattr(gateway, "arestart", None)
            if not callable(restart):
                raise
            logger.error(
                "Gateway is not cleanup-ready before weight sync; restarting "
                "at the safe pre-mutation boundary: %r",
                trigger,
            )
            coordinator.pause_generation()
            gateway.mark_unavailable()
            task_ids = (
                await wave_controller.abort_active_for_recovery()
                if wave_controller is not None
                else []
            )
            abandoned_session_ids = task_group_session_ids(
                task_ids,
                int(self.rllm_config.rollout.n),
                max_attempts=_configured_workflow_retry_limit(
                    self.rllm_config
                ),
            )
            self._append_gateway_recovery_manifest(
                trainer_state,
                status="pre_weight_sync_started",
                task_ids=task_ids,
                error=trigger,
            )
            try:
                await restart(
                    abandoned_session_ids=abandoned_session_ids,
                )
            except BaseException as exc:  # noqa: BLE001 - fatal boundary
                self._append_gateway_recovery_manifest(
                    trainer_state,
                    status="pre_weight_sync_failed",
                    task_ids=task_ids,
                    error=exc,
                )
                error = RuntimeError(
                    "Gateway pre-weight-sync recovery failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                raise error from exc
            self._append_gateway_recovery_manifest(
                trainer_state,
                status="pre_weight_sync_recovered",
                task_ids=task_ids,
                error=trigger,
            )
            note_ready = getattr(
                wave_controller,
                "note_gateway_cleanup_ready",
                None,
            )
            if callable(note_ready):
                note_ready()
            return True

        note_ready = getattr(
            wave_controller,
            "note_gateway_cleanup_ready",
            None,
        )
        if callable(note_ready):
            note_ready()
        return False

    async def _perform_weight_sync(
        self,
        trainer_state: TrainerState,
        coordinator: SyncCoordinator,
        rollout_engine: RolloutEngine | None,
        *,
        wave_controller: DynamicSamplingWaveController | None = None,
    ) -> None:
        """Serialize weight synchronization against gateway recovery."""
        lock = getattr(self, "_gateway_recovery_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            self._gateway_recovery_lock = lock
        recovered_before_sync = False
        async with lock:
            trainer_state.optimizer_phase = "weight_sync_preflight"
            recovered_before_sync = await self._prepare_gateway_for_weight_sync(
                trainer_state,
                coordinator,
                wave_controller,
            )
            # From this assignment until the version mutation completes, a
            # restart would make the policy/version outcome ambiguous.
            trainer_state.optimizer_phase = "weight_sync"
            await self._perform_weight_sync_unlocked(
                trainer_state,
                coordinator,
                rollout_engine,
            )
        if recovered_before_sync and self.async_config.partial_rollout:
            coordinator.resume_generation()

    async def _perform_weight_sync_unlocked(self, trainer_state: TrainerState, coordinator: SyncCoordinator, rollout_engine: RolloutEngine | None) -> None:
        """Synchronize weights between training and rollout engines."""
        if not self.async_config.partial_rollout:
            coordinator.pause_generation()
            await coordinator.wait_for_drain()

        trainer_state.weight_version = coordinator.weight_version + 1
        gateway = self._gateway
        maintenance_sync_id: str | None = None
        enter_maintenance = getattr(gateway, "aenter_weight_sync_maintenance", None)
        exit_maintenance = getattr(gateway, "aexit_weight_sync_maintenance", None)
        try:
            if callable(enter_maintenance):
                maintenance_sync_id = await enter_maintenance(
                    trainer_state.weight_version,
                    sync_id=(f"step-{trainer_state.global_step}-weight-{trainer_state.weight_version}-{uuid.uuid4().hex}"),
                )

            await self.backend.on_policy_updated(trainer_state)
            if rollout_engine is not None:
                rollout_engine.weight_version = trainer_state.weight_version
            if gateway is not None:
                await gateway.aset_weight_version(trainer_state.weight_version)
            if maintenance_sync_id is not None and callable(exit_maintenance):
                await exit_maintenance(maintenance_sync_id)
                maintenance_sync_id = None
            coordinator.on_sync_complete()
        except BaseException:
            # The trainer is about to unwind, but reopening admission lets
            # already-running workflows receive a terminal response and clean
            # up their sessions instead of hanging at a closed barrier.
            if maintenance_sync_id is not None and callable(exit_maintenance):
                try:
                    await exit_maintenance(maintenance_sync_id)
                except Exception:
                    logger.exception("Failed to leave gateway maintenance after weight-sync error")
            raise

        if not self.async_config.partial_rollout:
            coordinator.resume_generation()

    async def _validate_async_with_pause(self, trainer_state: TrainerState, coordinator: SyncCoordinator) -> dict:
        """Validation with dispatch-level pause. Waits for workflows to drain, then runs validation."""
        coordinator.pause_generation()
        await coordinator.wait_for_drain()
        try:
            hooks = getattr(self.agent_workflow_engine, "hooks", None)
            with _detached_warm_queue(hooks):
                return await self._validate_async(trainer_state)
        finally:
            coordinator.resume_generation()

    async def _validate_async(self, trainer_state: TrainerState) -> dict:
        """Validate the model (async implementation)."""
        if self._val_dataloader is None:
            return {}
        n_val_samples = self.rllm_config.rollout.n_val
        val_metrics = defaultdict(list)

        if not await self.backend.on_validation_start(trainer_state):
            return {}
        # manually manage the testing time
        test_begin = time.perf_counter()
        self.agent_workflow_engine.set_training_step(trainer_state.global_step, mode="val", epoch=trainer_state.epoch)

        is_correct_lst, uid_lst, data_source_lst = [], [], []
        workflow_metrics_by_source = defaultdict(lambda: defaultdict(list))

        for batch in self._val_dataloader:
            # Generate episodes and transform to trajectory groups
            val_episodes = await self.backend.generate_episodes(batch, agent_workflow_engine=self.agent_workflow_engine, is_validation=True)
            val_trajectory_groups, _ = transform_episodes_to_trajectory_groups(val_episodes, self.transform_config, self.cf_config, traj_grouping_hook=self.traj_grouping_hook)
            reward_metrics = collect_reward_and_advantage_from_trajectory_groups(val_trajectory_groups, self.algorithm_config, collect_advantage=False)

            is_correct_lst.extend([episode.is_correct for episode in val_episodes])
            uid_lst.extend([episode.task_id for episode in val_episodes])

            data_sources = [episode.info.get("data_source", "unknown") for episode in val_episodes]
            data_source_lst.extend(data_sources)

            for episode, data_source in zip(val_episodes, data_sources, strict=True):
                for key, value in episode.metrics.items():
                    # episode.metrics can contain non-numeric values -- skip in the workflow metrics.
                    try:
                        workflow_metrics_by_source[data_source][key].append(float(value))
                    except (TypeError, ValueError):
                        continue

            for key, value in reward_metrics.items():
                val_metrics[f"val/{key}"].append(value)

        test_end = time.perf_counter()
        val_metrics["time/testing"] = test_end - test_begin
        is_correct_array = np.array(is_correct_lst)
        uid_array = np.array(uid_lst)
        data_source_array = np.array(data_source_lst)

        for data_source in np.unique(data_source_array):
            pass_rates = defaultdict(list)

            data_source_mask = data_source_array == data_source
            is_correct_data_source = is_correct_array[data_source_mask]
            uids_data_source = uid_array[data_source_mask]

            for is_correct, uid in zip(is_correct_data_source, uids_data_source, strict=False):
                pass_rates[uid].append(is_correct)

            val_metrics[f"val/{data_source}/pass@1"] = np.mean(is_correct_data_source)
            val_metrics[f"val/{data_source}/pass@{n_val_samples}"] = np.mean([1 if any(pass_rate) else 0 for pass_rate in pass_rates.values()])

            # Add workflow metrics for this data source
            if data_source in workflow_metrics_by_source:
                for key, values in workflow_metrics_by_source[data_source].items():
                    if values:
                        val_metrics[f"val/{data_source}/{key}"] = np.mean(values)

        # post-process the val metrics to reduce any "list values" into scalars
        reduce_metrics_lists(val_metrics)
        print_metrics_table(val_metrics, trainer_state.global_step, title="Validation")
        self.logger.log(data=val_metrics, step=trainer_state.global_step)
        await self.backend.on_validation_end(trainer_state)
        return val_metrics

    def shutdown(self):
        """Every component gets a cleanup attempt even after another fails."""
        from rllm.utils.shutdown import shutdown_components
        components = []
        for attribute, method in (("_gateway", "stop"),
                                  ("agent_workflow_engine", "shutdown")):
            component = getattr(self, attribute, None)
            if component is not None:
                components.append((attribute, getattr(component, method)))
        engine = getattr(self, "agent_workflow_engine", None)
        shutdown_hook_backend = getattr(getattr(engine, "hooks", None), "shutdown_backend", None)
        if callable(shutdown_hook_backend):
            components.append(("sandbox_backend", shutdown_hook_backend))
        for attribute, method in (("backend", "shutdown"), ("logger", "finish")):
            component = getattr(self, attribute, None)
            if component is not None:
                components.append((attribute, getattr(component, method)))
        shutdown_components(components)
        self._gateway = None

    def record_shutdown_failure(self, primary_error, cleanup_error):
        """Append cleanup evidence without replacing this run's first failure."""
        from rllm.utils.infrastructure_diagnostics import infrastructure_exception_chain
        path = getattr(self, "_failure_manifest_path", None)
        payload = json.loads(path.read_text()) if path is not None and path.is_file() else {}
        if path is None:
            params = os.environ.get("RLLM_TRAINING_PARAMS_PATH")
            directory = Path(params).parent if params else Path(str(self.rllm_config.trainer.wandb_dir))
            path = directory / "training_failure_manifest.json"
        if not payload:
            error = primary_error if primary_error is not None else cleanup_error
            payload = {"schema_version": 1, "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "fatal_error": {"type": type(error).__name__, "message": str(error),
                                       "exception_chain": infrastructure_exception_chain(error)}}
        payload["shutdown_errors"] = infrastructure_exception_chain(cleanup_error)
        from rllm.utils.diagnostic_events import diagnostic_reference
        payload["diagnostic_evidence"] = diagnostic_reference()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.shutdown.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2, default=str)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    # =========================================================================
    # Helper functions
    # =========================================================================
    async def _gateway_routing_metrics(self) -> dict[str, int | float]:
        """Snapshot gateway routing and session-reaper counters."""

        gateway = getattr(self, "_gateway", None)
        if gateway is None:
            return {}
        result: dict[str, int | float] = {}
        routing_config = getattr(gateway, "routing_config", None)
        if getattr(routing_config, "mode", "sticky_least_loaded") == "group_striped_adaptive":
            try:
                stats = await gateway.aget_routing_stats()
            except Exception as exc:
                logger.warning(
                    "Could not collect adaptive gateway routing stats: %s",
                    exc,
                )
            else:
                keys = (
                    "initial_bindings",
                    "adaptive_migrations",
                    "failover_rebindings",
                    "prefix_rebuild_tokens",
                    "pinned_sessions_max",
                    "pinned_sessions_min",
                    "pinned_context_tokens_max",
                    "pinned_context_tokens_min",
                    "active_requests_max",
                    "active_requests_min",
                )
                result.update({f"gateway/routing/{key}": stats[key] for key in keys if isinstance(stats.get(key), int | float) and not isinstance(stats.get(key), bool)})

        try:
            cleanup_stats = await gateway.aget_session_cleanup_stats()
        except Exception as exc:
            logger.warning(
                "Could not collect gateway session cleanup stats: %s",
                exc,
            )
        else:
            result.update({f"gateway/session_cleanup/{key}": value for key, value in cleanup_stats.items() if isinstance(value, int | float) and not isinstance(value, bool)})
        return result

    @staticmethod
    def _drain_dynamic_sampling_metrics(
        train_dataloader: DynamicSamplingTaskDataLoader,
    ) -> dict[str, int]:
        metrics = train_dataloader.drain_step_metrics()
        for summary in train_dataloader.drain_epoch_summaries():
            logger.info(
                "Dynamic sampling epoch %d complete: tasks=%d attempts=%d accepted=%d filtered=%d dropped=%d (easy=%d hard=%d) other=%d untrained_tail=%d rollouts=%d",
                summary["index"],
                summary["tasks_total"],
                summary["task_rollout_attempts"],
                summary["tasks_accepted"],
                summary["total_filtered_attempts"],
                summary["tasks_dropped"],
                summary["tasks_dropped_easy"],
                summary["tasks_dropped_hard"],
                summary["tasks_consumed_other"],
                summary["tasks_untrained_tail"],
                summary["rollouts_generated"],
            )
            metrics.update({f"dynamic_sampling/epoch/{key}": value for key, value in summary.items()})
        return metrics

    def _log_remaining_dynamic_sampling_metrics(
        self,
        trainer_state: TrainerState,
        train_dataloader: DynamicSamplingTaskDataLoader,
    ) -> None:
        train_dataloader.generation_exhausted(int(self.rllm_config.trainer.total_epochs))
        metrics = self._drain_dynamic_sampling_metrics(train_dataloader)
        has_epoch_summary = any(key.startswith("dynamic_sampling/epoch/") for key in metrics)
        has_step_activity = any(
            value
            for key, value in metrics.items()
            if key.startswith("dynamic_sampling/step/")
            and key
            not in {
                "dynamic_sampling/step/pool_remaining",
                "dynamic_sampling/step/pool_eligible",
                "dynamic_sampling/step/pool_deferred",
                "dynamic_sampling/step/in_flight",
            }
        )
        if not has_epoch_summary and not has_step_activity:
            return
        # This is a rollout/dataloader flush, not an optimizer step.  Printing
        # it with the next global_step created convincing but false "Step N"
        # tables without actor or optimizer metrics.
        metrics["training/non_optimizer_event"] = 1
        metrics["training/attempted_step"] = int(trainer_state.global_step)
        log_step = trainer_state.last_completed_step or 0
        logger.info(
            "Dynamic-sampling terminal/window metrics at last_committed_step=%s attempted_step=%s: %s",
            trainer_state.last_completed_step,
            trainer_state.global_step,
            metrics,
        )
        self.logger.log(data=metrics, step=log_step)

    @staticmethod
    def _collect_workflow_metrics_from_episodes(episodes: list[Episode]) -> tuple[dict, Counter]:
        workflow_metrics = defaultdict(list)
        termination_counts = Counter()
        for episode in episodes:
            for k, v in episode.metrics.items():
                workflow_metrics[k].append(v)
            reason = episode.termination_reason or TerminationReason.UNKNOWN
            termination_counts[getattr(reason, "value", reason)] += 1
        # reduce the metrics to a scalar value, with error handling
        reduced_workflow_metrics = {}
        for k, v in workflow_metrics.items():
            try:
                if k.startswith("milestone/verification_unavailable_reason/"):
                    reduced_workflow_metrics[k] = float(np.sum(v))
                else:
                    reduced_workflow_metrics[k] = float(np.mean(v))
            except Exception:
                continue
        return reduced_workflow_metrics, termination_counts


class TrainerLauncher(ABC):
    """
    A unified agent trainer launcher that directly interfaces with the user script to launch training jobs.

    It handles the necessary environment setup (e.g. ray init for `verl`) for different backends. This is an abstract
    class that each backend must implement.
    """

    def __init__(
        self,
        config: DictConfig,
        workflow_class: type[Workflow] | None = None,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        workflow_args: dict | None = None,
        store: Store | None = None,
        **kwargs,
    ):
        """Initialize the TrainerLauncher."""
        self.config = config
        self.workflow_class = workflow_class
        self.workflow_args = workflow_args or {}
        self.store = store
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.kwargs = kwargs

    @abstractmethod
    def train(self):
        raise NotImplementedError("Train method of the trainer launcher is not implemented")


class AgentTrainer:
    """
    A unified agent trainer launcher that directly interfaces with the user script to launch training jobs.
    Adapted directly from `rllm.trainer.agent_trainer.AgentTrainer`.

    This trainer will simply delegate the task to the corresponding launcher class.

    Provide exactly one of ``workflow_class`` or ``agent_flow``. When the run
    needs sandboxes (the flow declares ``needs_env``, or any dataset row
    carries an environment — see :func:`rllm.hooks.scan_env_requirements`),
    :class:`rllm.hooks.SandboxTaskHooks` and gateway loopback/tunnel are
    auto-wired. A passed ``evaluator`` becomes the hooks' FixedEvaluation
    policy (it does not disable the sandbox lifecycle); pass ``hooks=``
    explicitly to take over per-task setup entirely.

    Args:
        sandbox_backend: Backend for the auto-wired sandbox hooks
            (``"docker"`` / ``"local"`` / ``"modal"`` / …). Remote backends
            auto-spawn a cloudflared tunnel.
        sandbox_concurrency: Override ``max_concurrent`` on a
            :class:`SandboxedAgentFlow` agent.
    """

    def __init__(
        self,
        config: DictConfig,
        workflow_class: type[Workflow] | None = None,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        workflow_args: dict | None = None,
        backend: Literal["verl"] = "verl",
        agent_flow: Any = None,
        evaluator: Any = None,
        hooks: Any = None,
        sandbox_backend: str | None = None,
        sandbox_concurrency: int | None = None,
        store: Store | None = None,
        **kwargs,
    ):
        # Loopback/tunnel pinning must happen here, before the launcher
        # constructs GatewayManager, and applies to explicitly-passed hooks too.
        if agent_flow is not None:
            from rllm.gateway.tunnel import is_local_sandbox_backend
            from rllm.hooks import (
                FixedEvaluation,
                SandboxTaskHooks,
                enable_gateway_tunnel,
                pin_gateway_host_loopback,
                scan_env_requirements,
            )

            scan = scan_env_requirements(agent_flow, train_dataset, val_dataset, sandbox_backend=sandbox_backend)
            if hooks is None and scan.needs_env:
                hooks = SandboxTaskHooks(
                    evaluation=FixedEvaluation(evaluator) if evaluator is not None else None,
                    sandbox_backend=sandbox_backend,
                )
                evaluator = None
            if hooks is not None and scan.needs_env:
                config = pin_gateway_host_loopback(config)
                # The hooks-backend clause matters only for explicitly-passed
                # hooks; auto-wired hooks share `sandbox_backend`, already
                # folded into `scan.any_remote`.
                hooks_backend = getattr(hooks, "sandbox_backend", None)
                if getattr(agent_flow, "llm_inside_env", False) and (scan.any_remote or not is_local_sandbox_backend(hooks_backend)):
                    config = enable_gateway_tunnel(config)

        # Forward CLI overrides through the flow's configure(); the wiring
        # warns about anything it returns unconsumed.
        overrides = {k: v for k, v in {"sandbox_backend": sandbox_backend, "sandbox_concurrency": sandbox_concurrency}.items() if v is not None}
        if agent_flow is not None and overrides:
            configure = getattr(agent_flow, "configure", None)
            leftovers = dict(configure(overrides) if callable(configure) else overrides)
            if hooks is not None:
                # The sandbox hooks consume the backend regardless of the flow.
                leftovers.pop("sandbox_backend", None)
            for flag in leftovers:
                logger.warning("--%s has no effect for agent %s", flag.replace("_", "-"), type(agent_flow).__name__)

        has_agent_flow = agent_flow is not None and (evaluator is not None or hooks is not None)
        if not has_agent_flow:
            raise ValueError("SWE-MILE requires an agent_flow with evaluator or sandbox hooks")

        if agent_flow is not None:
            kwargs["agent_flow"] = agent_flow
        if evaluator is not None:
            kwargs["evaluator"] = evaluator
        if hooks is not None:
            kwargs["hooks"] = hooks
        kwargs["backend_name"] = backend

        if backend == "verl":
            from rllm.trainer.verl.verl_launcher import VerlTrainerLauncher

            self.launcher = VerlTrainerLauncher(
                config=config,
                workflow_class=workflow_class,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                workflow_args=workflow_args,
                store=store,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def train(self):
        self.launcher.train()
