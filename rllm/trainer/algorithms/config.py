import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf

from rllm.rewards.milestone_reward import MilestoneRewardConfig
from rllm.types import _DEFAULT_TRAJ_NAME
from rllm.workflows.workflow import TerminationReason


def _explicit_override_keys(hydra_overrides: list[str] | None = None) -> set[str]:
    """Dotted paths the user explicitly set on the Hydra CLI.

    Falls back to ``HydraConfig.get().overrides.task`` when ``hydra_overrides``
    is None (only works inside the Hydra entry-point process).
    """
    if hydra_overrides is None:
        try:
            from hydra.core.hydra_config import HydraConfig

            hydra_overrides = list(HydraConfig.get().overrides.task)
        except (ValueError, AttributeError, ImportError):
            return set()
    keys: set[str] = set()
    for o in hydra_overrides:
        if "=" not in o:
            continue
        keys.add(o.split("=", 1)[0].lstrip("+~"))
    return keys


def _plain(value: Any) -> Any:
    """Resolve-free plain-Python view of a config value (for equality checks)."""
    return OmegaConf.to_container(value, resolve=False) if OmegaConf.is_config(value) else value


def sync_shared_keys(
    config: DictConfig,
    shared_keys: list[tuple[str, str]],
    *,
    hydra_overrides: list[str] | None = None,
    explicit: set[str] | None = None,
    on_native_override: Callable[[str, str, bool], None] | None = None,
) -> None:
    """Keep a backend-native namespace in parity with ``rllm.*`` over a shared-keys table.

    Each entry is ``(native_path, rllm_path)``. Precedence per key:
    rllm CLI > native CLI > rllm yaml (non-None) > native yaml. Backend-agnostic —
    the backend supplies its own table (cf. verl's ``_SHARED_KEYS``).

    ``on_native_override(native_path, rllm_path, conflict)`` is invoked when the user set the
    native path on the CLI — e.g. to emit a deprecation warning. ``explicit`` may be passed to
    reuse a precomputed override set.
    """
    if explicit is None:
        explicit = _explicit_override_keys(hydra_overrides)
    for native_path, rllm_path in shared_keys:
        if rllm_path in explicit:
            rllm_value = OmegaConf.select(config, rllm_path)
            if native_path in explicit and on_native_override is not None:
                on_native_override(native_path, rllm_path, _plain(OmegaConf.select(config, native_path)) != _plain(rllm_value))
            OmegaConf.update(config, native_path, rllm_value, merge=False)
        elif native_path in explicit:
            if on_native_override is not None:
                on_native_override(native_path, rllm_path, False)
            OmegaConf.update(config, rllm_path, OmegaConf.select(config, native_path), merge=False)
        else:
            value = OmegaConf.select(config, rllm_path)
            if value is not None:
                OmegaConf.update(config, native_path, value, merge=False)


@dataclass
class AsyncTrainingConfig:
    """Controls the async training behavior spectrum.

    When `enable` is False, the trainer uses the current synchronous pipeline.
    When `enable` is True, the trainer runs concurrent generation + training
    with group-level streaming and dispatch-time throttle.

    Behavior spectrum:
        - staleness_threshold=0, trigger_parameter_sync_step=1: On-policy
        - staleness_threshold=0, trigger_parameter_sync_step=K: Stream off-policy
        - staleness_threshold>0, partial_rollout=False: Async with staleness
        - staleness_threshold>0, partial_rollout=True: Async with partial rollout
    """

    enable: bool = False
    mini_batch_size: int = 1  # episode groups per optimizer step
    fwd_bwd_group_size: int | None = None  # task batches per forward-backward pass (default: mini_batch_size)
    staleness_threshold: float = 0.0  # 0.0 = on-policy. Controls dispatch throttle quota.
    trigger_parameter_sync_step: int = 1  # optimizer steps between weight sync + version bump
    partial_rollout: bool = True  # enable turn-level gating during weight sync
    server_init_max_attempts: int = 3  # maximum vLLM server-manager initialization attempts
    server_init_retry_quiescence_seconds: float = 15.0  # delay before retrying server initialization
    server_init_gpu_release_timeout_seconds: float = 300.0  # wait for failed workers to release GPUs
    episode_offload_dir: str | None = None  # NVMe offload dir for pending episodes (None = disabled)
    trajectory_group_offload_dir: str | None = None  # NVMe offload dir for queued task batches (None = disabled)
    # The rollout startup policy may skip disk loading only when a full
    # checkpoint sync will initialize the servers before sampling.
    skip_initial_rollout_weights: bool = True

    def __post_init__(self):
        if (
            isinstance(self.server_init_max_attempts, bool)
            or not isinstance(self.server_init_max_attempts, int)
            or self.server_init_max_attempts <= 0
        ):
            raise ValueError("server_init_max_attempts must be a positive integer")
        for name, allow_zero in (
            ("server_init_retry_quiescence_seconds", True),
            ("server_init_gpu_release_timeout_seconds", False),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or (float(value) < 0 if allow_zero else float(value) <= 0)
            ):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be a finite {qualifier} number")
            setattr(self, name, float(value))
        if self.fwd_bwd_group_size is None:
            self.fwd_bwd_group_size = self.mini_batch_size
        if self.enable:
            assert self.fwd_bwd_group_size >= 1
            assert self.mini_batch_size % self.fwd_bwd_group_size == 0, f"mini_batch_size ({self.mini_batch_size}) must be divisible by fwd_bwd_group_size ({self.fwd_bwd_group_size})"

    @classmethod
    def from_config(cls, config: DictConfig) -> "AsyncTrainingConfig":
        return cls(**OmegaConf.to_container(config))  # type: ignore


@dataclass
class CompactFilteringConfig:
    """Configuration for compact filtering of episodes based on termination reasons.

    Compatible with OmegaConf/Hydra config system.
    All fields default to False for backwards compatibility.

    Usage with OmegaConf:
        config = OmegaConf.structured(CompactFilteringConfig)
        # or from YAML
        config = OmegaConf.load("config.yaml").rllm.compact_filtering
    """

    enable: bool = False
    mask_max_prompt_length_exceeded: bool = False
    mask_max_context_length_exceeded: bool = False
    mask_max_response_length_exceeded: bool = False
    mask_env_done: bool = False
    mask_max_turns_exceeded: bool = False
    mask_timeout: bool = False
    mask_unknown: bool = False
    mask_error: bool = False

    @classmethod
    def from_config(cls, config: DictConfig) -> "CompactFilteringConfig":
        """Create a CompactFilteringConfig from a dictionary configuration.

        Args:
            config: Dictionary configuration.
        Returns:
            CompactFilteringConfig: The CompactFilteringConfig built from the configuration.
        """
        return cls(**OmegaConf.to_container(config))  # type: ignore

    def should_mask_episode(self, episode) -> bool:
        deadline = episode.metadata.get("denovo_deadline") or {}
        if (episode.termination_reason == TerminationReason.TIMEOUT
                and deadline.get("stop_confirmed") is True
                and deadline.get("verifier_completed") is True
                and not episode.metadata.get("infrastructure_failure")):
            return False
        return self.should_mask(episode.termination_reason or TerminationReason.UNKNOWN)

    def should_mask(self, termination_reason: TerminationReason) -> bool:
        """Check if a specific termination reason should be masked/filtered out.

        Args:
            termination_reason: The termination reason to check.
        Returns:
            True if this termination reason should be filtered out, False otherwise.
        """
        if not self.enable:
            return False
        return (
            (self.mask_max_prompt_length_exceeded and termination_reason == TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)
            or (
                self.mask_max_context_length_exceeded
                and termination_reason == TerminationReason.MAX_CONTEXT_LENGTH_EXCEEDED
            )
            or (self.mask_max_response_length_exceeded and termination_reason == TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED)
            or (self.mask_env_done and termination_reason == TerminationReason.ENV_DONE)
            or (self.mask_max_turns_exceeded and termination_reason == TerminationReason.MAX_TURNS_EXCEEDED)
            or (self.mask_timeout and termination_reason == TerminationReason.TIMEOUT)
            or (self.mask_unknown and termination_reason == TerminationReason.UNKNOWN)
            or (self.mask_error and termination_reason == TerminationReason.ERROR)
        )


@dataclass
class TransformConfig:
    """Configuration for the episode-to-group transformation pipeline."""

    # Name imputation
    impute_missing_names: bool = True
    # Default trajectory name (if user does not provide a name); treated as unnamed
    default_traj_name: str = _DEFAULT_TRAJ_NAME
    # Whether to drop unnamed trajectories on failure
    drop_unnamed_traj: bool = False

    # Reward configuration
    broadcast: bool = True  # If True, use trajectory-level rewards; if False, use per-step rewards

    @classmethod
    def from_config(cls, transform_config: DictConfig, *, broadcast: bool = True) -> "TransformConfig":
        return cls(
            impute_missing_names=transform_config.get("impute_missing_names", True),
            default_traj_name=transform_config.get("default_traj_name", _DEFAULT_TRAJ_NAME),
            drop_unnamed_traj=transform_config.get("drop_unnamed_traj", False),
            broadcast=broadcast,
        )


@dataclass
class RejectionSamplingConfig:
    """Configuration for rejection sampling."""

    # Rejection sampling mode
    # - "none": No rejection, just track metrics
    # - "episode": Skip whole batch if criteria not met, accumulate until enough partial solves
    # - "group": Filter groups with insufficient trajectories, pass remaining to trainer
    mode: Literal["none", "episode", "group"] = "none"

    # Minimum trajectories required per trajectory group (for "group" mode)
    min_trajs_per_group: int = 2

    # For "episode" mode (verl compatibility): minimum number of tasks with partial solves before proceeding
    min_partial_solve_tasks: int = 1

    # Filter out episode groups where all rollouts have the same is_correct (no gradient signal).
    # Applied at the accumulator level in async training, before groups enter the buffer.
    filter_uniform_groups: bool = False

    @classmethod
    def from_config(cls, config: DictConfig) -> "RejectionSamplingConfig":
        mode = config.get("mode", None)
        if mode is None:
            mode = "episode" if config.get("enable", False) else "none"
        return cls(
            mode=mode,
            min_trajs_per_group=config.get("min_trajs_per_group", 2),
            min_partial_solve_tasks=config.get("min_partial_solve_tasks", 1),
            filter_uniform_groups=config.get("filter_uniform_groups", False),
        )


@dataclass
class DynamicSamplingConfig:
    """Outcome-reward dynamic sampling and task-pool configuration."""

    enable: bool = False
    outcome_mode: str = "reward_uniform"
    max_easy_rejections: int = 3
    max_hard_rejections: int = 5
    easy_pass_rate_threshold: float = 0.9
    hard_pass_rate_threshold: float = 0.1
    async_step_sampling_multiplier: float = 1.0
    cancel_wait_timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        if self.outcome_mode not in {"reward_uniform", "verifier_pass_count"}:
            raise ValueError(
                "dynamic_sampling.outcome_mode must be 'reward_uniform' or "
                f"'verifier_pass_count', got {self.outcome_mode!r}"
            )
        for name in ("max_easy_rejections", "max_hard_rejections"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"dynamic_sampling.{name} must be a positive integer, got {value!r}"
                )
        for name in ("easy_pass_rate_threshold", "hard_pass_rate_threshold"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(
                    f"dynamic_sampling.{name} must be finite and in [0, 1], "
                    f"got {value!r}"
                )
            setattr(self, name, float(value))
        if self.hard_pass_rate_threshold > self.easy_pass_rate_threshold:
            raise ValueError(
                "dynamic_sampling.hard_pass_rate_threshold must be <= "
                "easy_pass_rate_threshold"
            )
        multiplier = self.async_step_sampling_multiplier
        if (
            isinstance(multiplier, bool)
            or not isinstance(multiplier, int | float)
            or not math.isfinite(float(multiplier))
            or float(multiplier) < 1.0
        ):
            raise ValueError(
                "dynamic_sampling.async_step_sampling_multiplier must be a "
                f"finite number >= 1, got {multiplier!r}"
            )
        self.async_step_sampling_multiplier = float(multiplier)
        timeout = self.cancel_wait_timeout_seconds
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, int | float)
            or not math.isfinite(float(timeout))
            or float(timeout) <= 0
        ):
            raise ValueError(
                "dynamic_sampling.cancel_wait_timeout_seconds must be a "
                f"finite positive number, got {timeout!r}"
            )
        self.cancel_wait_timeout_seconds = float(timeout)

    @classmethod
    def from_config(cls, config: DictConfig | dict | None) -> "DynamicSamplingConfig":
        values = config or {}
        return cls(
            enable=bool(values.get("enable", False)),
            outcome_mode=values.get("outcome_mode", "reward_uniform"),
            max_easy_rejections=values.get("max_easy_rejections", 3),
            max_hard_rejections=values.get("max_hard_rejections", 5),
            easy_pass_rate_threshold=values.get(
                "easy_pass_rate_threshold", 0.9
            ),
            hard_pass_rate_threshold=values.get(
                "hard_pass_rate_threshold", 0.1
            ),
            async_step_sampling_multiplier=values.get(
                "async_step_sampling_multiplier", 1.0
            ),
            cancel_wait_timeout_seconds=values.get(
                "cancel_wait_timeout_seconds", 120.0
            ),
        )


@dataclass
class RolloutCorrectionConfig:
    """Configuration for rollout correction (TIS, proximal forward passes).

    Backend-agnostic — each backend interprets these according to its infrastructure.

    Attributes:
        tis_mode: None = disabled (string loss names, current behavior).
              "token" or "sequence" = enable custom callable loss with TIS at that level.
        bypass_mode: When True, use rollout (inference) logprobs as π_old — no
              proximal forward pass. When False, compute π_old via policy.forward()
              (3-policy / decoupled PPO).
        tis_cap: Upper clamp on the TIS importance weight.
    """

    tis_mode: str | None = None
    bypass_mode: bool | None = None
    tis_cap: float = 5.0


class rLLMAdvantageEstimator(str, Enum):
    """
    A unified advantage estimator for rLLM. Work with both `tinker` and `verl` backends at the expense of
    losing some flexibility.
    TODO(listar2000): add more estimators.
    """

    GRPO = "grpo"
    REINFORCE = "reinforce"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    PRPO = "prpo"
    RLOO = "rloo"
    OTHER = "other"

    @classmethod
    def _missing_(cls, value: object) -> "rLLMAdvantageEstimator":
        return cls.OTHER


@dataclass
class AlgorithmConfig:
    """Configuration for algorithm parameters.

    ``estimator_map`` values may be:

    * A bare estimator name or enum — only the advantage estimator is overridden.
    * A ``(estimator, policy_loss)`` tuple — both the advantage estimator *and*
      the backend policy loss function are overridden for that role.

    During ``__post_init__``, tuples are split: the estimator goes into
    ``estimator_map`` and the loss function goes into ``loss_fn_map``.
    """

    use_rllm: bool | None = None  # Deprecated
    estimator: rLLMAdvantageEstimator = rLLMAdvantageEstimator.GRPO
    estimator_map: dict[str, rLLMAdvantageEstimator | str | tuple] = field(default_factory=dict)
    # Per-role policy loss overrides (populated from tuples in estimator_map during __post_init__)
    loss_fn_map: dict[str, str] = field(default_factory=dict)
    # TODO(listar2000): eventually we will remove the `per_step` mode all-together. Now we keep it for backward compatibility.
    stepwise_advantage_enabled: bool = False
    stepwise_advantage_mode: Literal["broadcast", "per_step"] = "broadcast"
    stepwise_outcome_weight: float = 1.0
    stepwise_tool_call_weight: float = 0.0
    stepwise_tool_call_correct_reward: float = 1.0
    stepwise_tool_call_incorrect_reward: float = 0.0
    milestone_reward: MilestoneRewardConfig = field(default_factory=MilestoneRewardConfig)
    norm_adv_by_std_in_grpo: bool = True
    # When True, always use pre-computed step.advantage from the workflow and skip
    # advantage computation (GRPO/REINFORCE). Steps missing advantages default to 0.0.
    # When False (default), always compute advantages normally.
    use_precomputed_advantage: bool = False
    # Global loss function (backend-specific values; null = backend default)
    loss_fn: str | None = None
    lr_schedule: Literal["linear", "cosine", "constant"] = "constant"
    warmup_steps: int = -1
    warmup_steps_ratio: float = 0.0

    # Custom loss / rollout correction fields (used by Fireworks backend with cookbook losses)
    kl_beta: float = 0.0
    eps_clip: float = 0.2
    eps_clip_high: float | None = None
    loss_agg_mode: Literal["token-mean", "seq-mean-token-sum", "seq-mean-token-mean", None] = None
    rollout_correction: RolloutCorrectionConfig = field(default_factory=RolloutCorrectionConfig)
    router_replay: Literal["disabled", "R2", "R3"] = "disabled"

    @classmethod
    def from_config(
        cls,
        algorithm_config: DictConfig,
        *,
        stepwise_advantage_mode: str = "broadcast",
        stepwise_advantage_config: DictConfig | dict | None = None,
        estimator_map: dict | None = None,
    ) -> "AlgorithmConfig":
        """Create an AlgorithmConfig from a dictionary configuration.

        Args:
            algorithm_config: Dictionary configuration.
        Returns:
            AlgorithmConfig: The AlgorithmConfig built from the configuration.
        """
        if algorithm_config.get("use_rllm", None) is not None:
            from warnings import warn

            warn(
                "`algorithm.use_rllm` is deprecated and ignored — advantages are always computed via the rLLM-native path. Remove the field from your config.",
                DeprecationWarning,
                stacklevel=2,
            )

        rc_section = algorithm_config.get("rollout_correction", {})
        rollout_correction = RolloutCorrectionConfig(
            tis_mode=rc_section.get("tis_mode", None),
            bypass_mode=rc_section.get("bypass_mode", None),
            tis_cap=rc_section.get("tis_cap", 2.0),
        )
        stepwise_cfg = stepwise_advantage_config or {}
        milestone_reward = MilestoneRewardConfig.from_config(
            stepwise_cfg.get("milestone", {}),
            format_weight=float(stepwise_cfg.get("tool_call_weight", 0.0)),
        )
        return cls(
            estimator=rLLMAdvantageEstimator(algorithm_config.adv_estimator),
            estimator_map=estimator_map or {},
            stepwise_advantage_enabled=bool(stepwise_cfg.get("enable", False)),
            stepwise_advantage_mode=stepwise_advantage_mode,
            stepwise_outcome_weight=stepwise_cfg.get("outcome_weight", 1.0),
            stepwise_tool_call_weight=stepwise_cfg.get("tool_call_weight", 0.0),
            stepwise_tool_call_correct_reward=stepwise_cfg.get("tool_call_correct_reward", 1.0),
            stepwise_tool_call_incorrect_reward=stepwise_cfg.get("tool_call_incorrect_reward", 0.0),
            milestone_reward=milestone_reward,
            norm_adv_by_std_in_grpo=algorithm_config.get("norm_adv_by_std_in_grpo", True),
            use_precomputed_advantage=algorithm_config.get("use_precomputed_advantage", False),
            loss_fn=algorithm_config.get("loss_fn", None),
            lr_schedule=algorithm_config.get("lr_schedule", "constant"),
            warmup_steps=algorithm_config.get("warmup_steps", -1),
            warmup_steps_ratio=algorithm_config.get("warmup_steps_ratio", 0.0),
            kl_beta=algorithm_config.get("kl_beta", 0.0),
            eps_clip=algorithm_config.get("eps_clip", 0.2),
            eps_clip_high=algorithm_config.get("eps_clip_high", None),
            loss_agg_mode=algorithm_config.get("loss_agg_mode", None),
            rollout_correction=rollout_correction,
            router_replay=algorithm_config.get("router_replay", "disabled"),
        )

    def __post_init__(self):
        if self.use_rllm is not None:
            from warnings import warn

            warn(
                "`algorithm.use_rllm` is deprecated and ignored — advantages are always computed via the rLLM-native path. Remove the field from your config.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Normalize estimator_map: split (estimator, loss_fn) tuples.
        normalized_map: dict[str, rLLMAdvantageEstimator | str] = {}
        for role, value in self.estimator_map.items():
            if isinstance(value, tuple):
                if len(value) != 2:
                    raise ValueError(f"estimator_map tuple for role '{role}' must have exactly 2 elements (estimator, loss_fn), got {len(value)}")
                estimator, loss_fn = value
                normalized_map[role] = estimator
                self.loss_fn_map[role] = str(loss_fn)
            else:
                normalized_map[role] = value
        self.estimator_map = normalized_map

        if self.stepwise_advantage_mode not in {"broadcast", "per_step"}:
            raise ValueError(f"Unknown stepwise_advantage_mode: {self.stepwise_advantage_mode}")

        if self.milestone_reward.enable:
            if not self.stepwise_advantage_enabled:
                raise ValueError("milestone reward requires stepwise_advantage.enable=true")
            if self.stepwise_advantage_mode != "per_step":
                raise ValueError("milestone reward requires stepwise_advantage.mode=per_step")
            if self.estimator != rLLMAdvantageEstimator.RLOO:
                raise ValueError("milestone reward requires algorithm.adv_estimator=rloo")
            if self.use_precomputed_advantage:
                raise ValueError("milestone reward is incompatible with use_precomputed_advantage=true")
            if self.loss_fn not in {None, "vanilla"}:
                raise ValueError("milestone reward requires the vanilla PPO policy loss")
            non_vanilla_roles = [role for role, loss_fn in self.loss_fn_map.items() if loss_fn != "vanilla"]
            if non_vanilla_roles:
                raise ValueError(f"milestone reward requires vanilla PPO for every role, got overrides for {non_vanilla_roles}")
            non_rloo_roles = [
                role
                for role, estimator in self.estimator_map.items()
                if rLLMAdvantageEstimator(estimator) != rLLMAdvantageEstimator.RLOO
            ]
            if non_rloo_roles:
                raise ValueError(f"milestone reward requires RLOO for every role, got overrides for {non_rloo_roles}")
            if self.milestone_reward.format_enable:
                if float(self.stepwise_tool_call_weight) != float(self.milestone_reward.format_weight):
                    raise ValueError("milestone format_weight must match stepwise_advantage.tool_call_weight")
                if float(self.stepwise_tool_call_correct_reward) != 0.0 or float(self.stepwise_tool_call_incorrect_reward) != -1.0:
                    raise ValueError(
                        "milestone format reward requires tool_call_correct_reward=0.0 "
                        "and tool_call_incorrect_reward=-1.0"
                    )
