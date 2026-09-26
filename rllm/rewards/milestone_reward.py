"""R2E-Gym milestone potential and process-advantage computation.

The calculator consumes the model-visible provenance stored in codeflow
``ActionEvent`` objects.  It deliberately recomputes verification potentials
from per-test or calibrated partition-count results, the rollout-local runtime
baseline and the materialized target map so training hyperparameters cannot
drift from values cached by the shadow runtime.
"""

from __future__ import annotations

import json
import math
import posixpath
from dataclasses import asdict, dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rllm.harnesses.action_event import (
    ActionCategory,
    ExecutionStatus,
    ParseStatus,
    TestCountObservation,
    TestEventContinuityStatus,
    TestEventStatus,
    TestPartitionResult,
    TestResultSource,
    ValidationStatus,
    get_action_event,
    resolve_test_event_continuity,
)
from rllm.types import Episode, Step, Trajectory
from rllm.utils.tool_call_reward import compute_tool_call_reward

MILESTONE_REWARD_METADATA_KEY = "milestone_reward"
MILESTONE_REWARD_SCHEMA_VERSION = 5
MILESTONE_CONTEXT_METADATA_KEY = "milestone_context"
MILESTONE_CONTEXT_SCHEMA_VERSION = 6
MILESTONE_VERIFICATION_SUMMARY_SCHEMA_VERSION = 6
_REPO_GENERATION_VERIFICATION_MODES = frozenset({"pass_count"})
BUG_REPAIR_PASS_COUNT_MODE = "bug_repair_pass_count"
BUG_REPAIR_POLICY_NEUTRAL_UNRUN = "bug_repair_v2_neutral_unrun"
BUG_REPAIR_POLICY_PASS_COUNT = "bug_repair_pass_count_v1"
REPO_GENERATION_POLICY_PASS_COUNT = "repo_generation_pass_count_v1"
_VERIFICATION_POLICIES = frozenset(
    {
        BUG_REPAIR_POLICY_NEUTRAL_UNRUN,
        BUG_REPAIR_POLICY_PASS_COUNT,
        REPO_GENERATION_POLICY_PASS_COUNT,
    }
)


def _canonical_verification_potential_mode(value: Any) -> str:
    """Normalize the current verification-potential mode."""
    mode = str(value or "").strip()
    if mode in _REPO_GENERATION_VERIFICATION_MODES:
        return "pass_count"
    if mode in {BUG_REPAIR_PASS_COUNT_MODE}:
        return BUG_REPAIR_PASS_COUNT_MODE
    if mode == "normalized":
        return "bug_repair"
    return mode or "bug_repair"


class VerificationCountContract(BaseModel):
    """Resolved scalar baseline for bug-repair pass-count shaping."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    expected: int = Field(ge=1)
    materialized_baseline_passed: int = Field(ge=0)
    runtime_baseline_passed: int | None = Field(default=None, ge=0)
    baseline_passed: int = Field(ge=0)
    max_additional_passed: int = Field(ge=0)
    baseline_source: Literal["runtime_probe", "materialized_fallback"]
    baseline_probe_succeeded: bool
    collector: str | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> VerificationCountContract:
        for name, value in (
            ("materialized_baseline_passed", self.materialized_baseline_passed),
            ("runtime_baseline_passed", self.runtime_baseline_passed),
            ("baseline_passed", self.baseline_passed),
        ):
            if value is not None and value > self.expected:
                raise ValueError(f"{name} cannot exceed expected")
        if self.max_additional_passed != self.expected - self.baseline_passed:
            raise ValueError(
                "max_additional_passed must equal expected - baseline_passed"
            )
        if self.baseline_source == "runtime_probe":
            if (
                not self.baseline_probe_succeeded
                or self.runtime_baseline_passed is None
                or self.baseline_passed != self.runtime_baseline_passed
            ):
                raise ValueError("runtime baseline source is inconsistent")
        elif (
            self.baseline_probe_succeeded
            or self.baseline_passed != self.materialized_baseline_passed
        ):
            raise ValueError("materialized fallback baseline is inconsistent")
        return self


class MilestoneContext(BaseModel):
    """Rollout-local inputs required to recompute missing reward annotations.

    ``TrajectoryGroup`` deliberately contains trajectories rather than their
    parent episodes.  Persisting the small reward contract here keeps the
    compatibility recomputation path lossless without changing the historical
    ``Trajectory.task`` value (many harnesses store only the task id there).
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[6] = MILESTONE_CONTEXT_SCHEMA_VERSION
    task_metadata: dict[str, Any] = Field(default_factory=dict)
    shadow_baseline_test_event: dict[str, Any] | None = None
    shadow_schema_version: int | None = Field(default=None, ge=1)
    verification_potential_mode: str = "bug_repair"
    verification_potential_policy: Literal[
        "bug_repair_v2_neutral_unrun",
        "bug_repair_pass_count_v1",
        "repo_generation_pass_count_v1",
    ] = BUG_REPAIR_POLICY_NEUTRAL_UNRUN
    verification_result_mode: Literal[
        "per_test",
        "partition_counts",
        "aggregate_counts",
    ] = "per_test"
    verification_count_contract: VerificationCountContract | None = None
    partition_adapter: str | None = None
    partition_calibration_status: str | None = None
    shadow_verification_unavailable_reason: str | None = None



def _json_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return dict(value) if isinstance(value, dict) else None


def attach_episode_milestone_context(episode: Episode) -> None:
    """Persist the minimal episode context needed after trajectory grouping."""
    task = episode.task if isinstance(episode.task, dict) else {}
    task_metadata = {
        key: task[key]
        for key in (
            "workdir",
            "baseline_output_json",
            "target_output_json",
            "relevant_files",
            "verification_potential_mode",
        )
        if key in task
    }
    if "verification_potential_mode" in task_metadata:
        task_metadata["verification_potential_mode"] = _canonical_verification_potential_mode(task_metadata["verification_potential_mode"])
    shadow = episode.metadata.get("shadow_sandbox") if isinstance(episode.metadata, dict) else None
    baseline_event = shadow.get("baseline_test_event") if isinstance(shadow, dict) else None
    shadow_mode = (
        _canonical_verification_potential_mode(
            shadow.get("verification_potential_mode")
        )
        if isinstance(shadow, dict) and shadow.get("verification_potential_mode")
        else None
    )
    verification_mode = shadow_mode or _canonical_verification_potential_mode(
        task.get("verification_potential_mode")
        or (
            "pass_count"
            if isinstance(shadow, dict)
            and shadow.get("result_parser") == "denovoswe_official_v1"
            else "bug_repair"
        )
    )
    raw_policy = shadow.get("verification_potential_policy") if isinstance(shadow, dict) else None
    verification_policy = (
        str(raw_policy)
        if raw_policy in _VERIFICATION_POLICIES
        else (
            REPO_GENERATION_POLICY_PASS_COUNT
            if verification_mode == "pass_count"
            else BUG_REPAIR_POLICY_PASS_COUNT
            if verification_mode == BUG_REPAIR_PASS_COUNT_MODE
            else BUG_REPAIR_POLICY_NEUTRAL_UNRUN
        )
    )
    raw_count_contract = (
        shadow.get("verification_count_contract")
        if isinstance(shadow, dict)
        else None
    )
    context = MilestoneContext(
        task_metadata=task_metadata,
        shadow_baseline_test_event=_json_mapping(baseline_event),
        shadow_schema_version=(int(shadow["schema_version"]) if isinstance(shadow, dict) and isinstance(shadow.get("schema_version"), int) else None),
        verification_potential_mode=verification_mode,
        verification_potential_policy=verification_policy,
        verification_result_mode=(
            str(shadow.get("verification_result_mode"))
            if isinstance(shadow, dict)
            and shadow.get("verification_result_mode")
            in {"per_test", "partition_counts", "aggregate_counts"}
            else "per_test"
        ),
        verification_count_contract=(
            VerificationCountContract.model_validate(raw_count_contract)
            if isinstance(raw_count_contract, dict)
            else None
        ),
        partition_adapter=(str(shadow.get("partition_adapter")) if isinstance(shadow, dict) and shadow.get("partition_adapter") else None),
        partition_calibration_status=(str(shadow.get("partition_calibration_status")) if isinstance(shadow, dict) and shadow.get("partition_calibration_status") else None),
        shadow_verification_unavailable_reason=(
            str(shadow.get("verification_potential_unavailable_reason"))
            if isinstance(shadow, dict) and shadow.get("verification_potential_unavailable_reason")
            else None
        ),
    ).model_dump(mode="json")
    for trajectory in episode.trajectories:
        trajectory.info[MILESTONE_CONTEXT_METADATA_KEY] = context


def get_milestone_context(trajectory: Trajectory) -> MilestoneContext | None:
    """Read persisted context, rejecting corrupt current-schema metadata."""
    metadata = trajectory.metadata if isinstance(trajectory.metadata, dict) else {}
    raw = metadata.get(MILESTONE_CONTEXT_METADATA_KEY)
    if raw is None:
        return None
    if isinstance(raw, MilestoneContext):
        return raw
    if not isinstance(raw, dict):
        raise ValueError(f"invalid {MILESTONE_CONTEXT_METADATA_KEY}: expected an object")
    try:
        return MilestoneContext.model_validate(raw)
    except Exception as exc:
        raise ValueError(f"invalid {MILESTONE_CONTEXT_METADATA_KEY}: {exc}") from exc


def _finite_nonnegative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")


@dataclass(frozen=True)
class MilestoneRewardConfig:
    """Resolved milestone shaping parameters.

    ``format_weight`` is the design's eta coefficient.  Correct and invalid
    tool-call values remain fixed at 0 and -1 for milestone mode.
    """

    enable: bool = False
    verification_enable: bool = True
    navigation_enable: bool = True
    format_enable: bool = True
    beta_any: float = 0.5
    beta_frac: float = 0.5
    verification_weight: float = 0.25
    navigation_weight: float = 0.05
    navigation_search_score: float = 0.2
    navigation_read_score: float = 1.0
    backward_credit_lambda: float = 0.2
    backward_credit_gamma: float = 0.9
    verification_reward_clip_lower: float = -6.0
    verification_reward_clip_upper: float = 3.0
    format_weight: float = 0.1

    def __post_init__(self) -> None:
        for name in (
            "beta_any",
            "beta_frac",
            "verification_weight",
            "navigation_weight",
            "backward_credit_lambda",
            "format_weight",
        ):
            _finite_nonnegative(float(getattr(self, name)), name)
        for name in (
            "navigation_search_score",
            "navigation_read_score",
            "backward_credit_gamma",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0.0, 1.0], got {value!r}")
        clip_lower = float(self.verification_reward_clip_lower)
        clip_upper = float(self.verification_reward_clip_upper)
        if not math.isfinite(clip_lower) or not math.isfinite(clip_upper):
            raise ValueError("verification reward clip bounds must be finite")
        if clip_lower > 0.0 or clip_upper < 0.0 or clip_lower > clip_upper:
            raise ValueError(f"verification reward clip bounds must satisfy lower <= 0 <= upper, got [{clip_lower!r}, {clip_upper!r}]")
        if self.navigation_read_score < self.navigation_search_score:
            raise ValueError("navigation_read_score must be greater than or equal to navigation_search_score")

    @classmethod
    def from_config(
        cls,
        config: Any | None,
        *,
        format_weight: float = 0.1,
    ) -> MilestoneRewardConfig:
        cfg = config if hasattr(config, "get") else {}
        return cls(
            enable=bool(cfg.get("enable", False)),
            verification_enable=bool(cfg.get("verification_enable", True)),
            navigation_enable=bool(cfg.get("navigation_enable", True)),
            format_enable=bool(cfg.get("format_enable", True)),
            beta_any=float(cfg.get("beta_any", 0.5)),
            beta_frac=float(cfg.get("beta_frac", 0.5)),
            verification_weight=float(cfg.get("verification_weight", 0.25)),
            navigation_weight=float(cfg.get("navigation_weight", 0.05)),
            navigation_search_score=float(cfg.get("navigation_search_score", 0.2)),
            navigation_read_score=float(cfg.get("navigation_read_score", 1.0)),
            backward_credit_lambda=float(cfg.get("backward_credit_lambda", 0.2)),
            backward_credit_gamma=float(cfg.get("backward_credit_gamma", 0.9)),
            verification_reward_clip_lower=float(cfg.get("verification_reward_clip_lower", -6.0)),
            verification_reward_clip_upper=float(cfg.get("verification_reward_clip_upper", 3.0)),
            format_weight=float(format_weight),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MilestoneStepReward(BaseModel):
    """Auditable process-reward record attached to one model step."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[5] = MILESTONE_REWARD_SCHEMA_VERSION
    verification_potential_policy: Literal[
        "bug_repair_v2_neutral_unrun",
        "bug_repair_pass_count_v1",
        "repo_generation_pass_count_v1",
    ] = BUG_REPAIR_POLICY_NEUTRAL_UNRUN
    verification_available: bool
    verification_updated: bool = False
    verification_unavailable_reason: str | None = None
    verification_potential_before: float = 0.0
    verification_potential_after: float = 0.0
    verification_passed_before: int | None = Field(default=None, ge=0)
    verification_passed_after: int | None = Field(default=None, ge=0)
    verification_raw_delta: float | None = None
    verification_clipped_delta: float | None = None
    verification_reanchored: bool = False
    probe_merge_group_id: str | None = None
    probe_merge_group_size: int | None = Field(default=None, ge=1)
    probe_merge_group_index: int | None = Field(default=None, ge=0)
    probe_merge_anchor_action_id: str | None = None
    verification_probe_potential_after: float | None = None
    navigation_available: bool
    navigation_potential_before: float = 0.0
    navigation_potential_after: float = 0.0
    navigation_updates: dict[str, float] = Field(default_factory=dict)
    total_potential_before: float = 0.0
    total_potential_after: float = 0.0
    potential_reward: float = 0.0
    backward_credit: float = 0.0
    format_reward: float = 0.0
    process_advantage: float = 0.0



def get_milestone_reward(step: Step) -> MilestoneStepReward | None:
    """Read milestone metadata before or after gateway trace enrichment."""
    metadata = step.metadata if isinstance(step.metadata, dict) else {}
    raw = metadata.get(MILESTONE_REWARD_METADATA_KEY)
    if raw is None:
        agent_metadata = metadata.get("agent_step_metadata")
        if isinstance(agent_metadata, dict):
            raw = agent_metadata.get(MILESTONE_REWARD_METADATA_KEY)
    if isinstance(raw, MilestoneStepReward):
        return raw
    if not isinstance(raw, dict):
        return None
    try:
        return MilestoneStepReward.model_validate(raw)
    except Exception:
        return None


def summarize_milestone_verification(
    episode: Episode,
    trajectory: Trajectory | None,
    config: MilestoneRewardConfig | None = None,
) -> dict[str, Any]:
    """Build one compact, auditable verification-availability summary."""

    if config is not None:
        verification_enabled = bool(config.enable and config.verification_enable)
    else:
        trajectory_metadata = trajectory.metadata if trajectory is not None and isinstance(trajectory.metadata, dict) else {}
        reward_summary = trajectory_metadata.get(MILESTONE_REWARD_METADATA_KEY)
        reward_config = reward_summary.get("config") if isinstance(reward_summary, dict) else None
        verification_enabled = bool(isinstance(reward_config, dict) and reward_config.get("enable") and reward_config.get("verification_enable"))

    shadow = episode.metadata.get("shadow_sandbox") if isinstance(episode.metadata, dict) else None
    shadow = shadow if isinstance(shadow, dict) else {}
    quality_status = str(shadow.get("quality_status") or "unavailable")
    baseline_event = shadow.get("baseline_test_event")
    verification_mode = _canonical_verification_potential_mode(
        shadow.get("verification_potential_mode")
    )
    raw_count_contract = shadow.get("verification_count_contract")
    count_contract: VerificationCountContract | None = None
    if verification_mode in {
        "pass_count",
        BUG_REPAIR_PASS_COUNT_MODE,
    } and isinstance(
        raw_count_contract, dict
    ):
        try:
            count_contract = VerificationCountContract.model_validate(
                raw_count_contract
            )
        except Exception:
            count_contract = None
    baseline_probe_succeeded = bool(
        shadow.get("baseline_probe_succeeded")
        if verification_mode in {"pass_count", BUG_REPAIR_PASS_COUNT_MODE}
        else isinstance(baseline_event, dict)
        and baseline_event.get("status") == TestEventStatus.COMPLETED.value
        and baseline_event.get("trusted") is True
    )
    baseline_available = bool(
        count_contract is not None
        if verification_mode in {"pass_count", BUG_REPAIR_PASS_COUNT_MODE}
        else baseline_probe_succeeded
    )
    steps = trajectory.steps if trajectory is not None else []
    records = [record for step in steps if (record := get_milestone_reward(step)) is not None]
    verification_updates = sum(record.verification_updated for record in records)
    verification_reanchors = sum(
        record.verification_reanchored
        and (
            record.probe_merge_group_id is None
            or record.probe_merge_group_index == 0
        )
        for record in records
    )
    available_count = sum(record.verification_available for record in records)
    available_fraction = available_count / len(steps) if steps else float(baseline_available and verification_enabled)
    final_available = bool(verification_enabled and baseline_available and (len(records) == len(steps) and records[-1].verification_available if steps else True))

    unavailable_reason: str | None = None
    if not verification_enabled:
        status = "disabled"
    elif not baseline_available:
        status = "unavailable"
        concrete_count_reason = (
            shadow.get("verification_potential_unavailable_reason")
            if verification_mode == BUG_REPAIR_PASS_COUNT_MODE
            else None
        )
        if isinstance(concrete_count_reason, str) and concrete_count_reason:
            unavailable_reason = concrete_count_reason
        elif isinstance(baseline_event, dict):
            unavailable_reason = str(baseline_event.get("failure_type") or baseline_event.get("error") or "shadow_baseline_untrusted")
        else:
            unavailable_reason = str(next(iter(shadow.get("failure_counts") or {}), None) or ("shadow_setup" if shadow.get("status") == "unavailable" else "shadow_baseline_unavailable"))
    else:
        gaps = [record.verification_unavailable_reason for record in records if not record.verification_available]
        annotation_error = episode.metadata.get(MILESTONE_REWARD_METADATA_KEY) if isinstance(episode.metadata, dict) else None
        annotation_error = annotation_error.get("error") if isinstance(annotation_error, dict) and annotation_error.get("status") == "error" else None
        if gaps or len(records) != len(steps) or quality_status != "complete":
            status = "partial"
            unavailable_reason = str(
                next((reason for reason in gaps if reason), None)
                or annotation_error
                or ("milestone_annotation_missing" if len(records) != len(steps) else None)
                or next(iter(shadow.get("recoverable_failure_counts") or {}), None)
                or next(iter(shadow.get("failure_counts") or {}), None)
                or quality_status
            )
        else:
            status = "complete"

    count_full = count_partial = count_unavailable = 0
    if verification_mode == BUG_REPAIR_PASS_COUNT_MODE:
        for step in steps:
            action = get_action_event(step)
            event = action.test_event if action is not None else None
            if event is None:
                continue
            if event.result_completeness == "count_full" and event.trusted:
                count_full += 1
            elif event.result_completeness == "count_partial" and event.trusted:
                count_partial += 1
            elif event.status not in {
                TestEventStatus.PENDING,
                TestEventStatus.RUNNING,
                TestEventStatus.SKIPPED,
            }:
                count_unavailable += 1

    return {
        "schema_version": MILESTONE_VERIFICATION_SUMMARY_SCHEMA_VERSION,
        "status": status,
        "baseline_available": baseline_available,
        "baseline_probe_succeeded": baseline_probe_succeeded,
        "materialized_fallback": bool(
            count_contract is not None
            and count_contract.baseline_source == "materialized_fallback"
        ),
        "baseline_source": (
            count_contract.baseline_source
            if count_contract is not None
            else shadow.get("baseline_source")
        ),
        "baseline_collector": (
            count_contract.collector
            if count_contract is not None
            else shadow.get("baseline_collector")
        ),
        "expected_test_count": (
            count_contract.expected if count_contract is not None else None
        ),
        "materialized_baseline_passed": (
            count_contract.materialized_baseline_passed
            if count_contract is not None
            else None
        ),
        "runtime_baseline_passed": (
            count_contract.runtime_baseline_passed
            if count_contract is not None
            else None
        ),
        "baseline_passed": (
            count_contract.baseline_passed
            if count_contract is not None
            else None
        ),
        "max_additional_passed": (
            count_contract.max_additional_passed
            if count_contract is not None
            else None
        ),
        "final_available": final_available,
        "signal_participated": bool(verification_enabled and available_count > 0),
        "verification_updates": int(verification_updates),
        "verification_reanchors": int(verification_reanchors),
        "verification_reanchored": bool(verification_reanchors),
        "available_step_fraction": float(available_fraction),
        "unavailable_reason": unavailable_reason,
        "shadow_quality_status": quality_status,
        "verification_result_mode": str(shadow.get("verification_result_mode") or "per_test"),
        "verification_potential_mode": verification_mode,
        "count_full_steps": count_full,
        "count_partial_steps": count_partial,
        "count_unavailable_steps": count_unavailable,
        "partition_adapter": shadow.get("partition_adapter"),
        "partition_calibration_status": shadow.get("partition_calibration_status"),
    }


def annotate_episode_milestone_verification_metrics(
    episode: Episode,
    config: MilestoneRewardConfig | None,
) -> list[dict[str, Any]]:
    """Expose rollout-level verification availability to trainer tracking."""

    trajectories: list[Trajectory | None] = list(episode.trajectories) or [None]
    summaries = [summarize_milestone_verification(episode, trajectory, config) for trajectory in trajectories]
    total = len(summaries)
    for status in ("complete", "partial", "unavailable", "disabled"):
        episode.metrics[f"milestone/verification_status/{status}_fraction"] = sum(summary["status"] == status for summary in summaries) / total
    episode.metrics["milestone/verification_signal_participated_fraction"] = sum(bool(summary["signal_participated"]) for summary in summaries) / total
    episode.metrics["milestone/baseline_probe_succeeded_fraction"] = sum(
        bool(summary.get("baseline_probe_succeeded")) for summary in summaries
    ) / total
    episode.metrics["milestone/materialized_baseline_fallback_fraction"] = sum(
        bool(summary.get("materialized_fallback")) for summary in summaries
    ) / total
    episode.metrics["milestone/count_full_steps"] = float(
        sum(int(summary.get("count_full_steps") or 0) for summary in summaries)
    )
    episode.metrics["milestone/count_partial_steps"] = float(
        sum(int(summary.get("count_partial_steps") or 0) for summary in summaries)
    )
    episode.metrics["milestone/count_unavailable_steps"] = float(
        sum(int(summary.get("count_unavailable_steps") or 0) for summary in summaries)
    )
    episode.metrics["milestone/verification_reanchors"] = float(
        sum(int(summary.get("verification_reanchors") or 0) for summary in summaries)
    )
    for summary in summaries:
        reason = summary.get("unavailable_reason")
        if reason:
            safe_reason = "".join(character if character.isalnum() or character in "_.-" else "_" for character in str(reason))[:120]
            episode.metrics[f"milestone/verification_unavailable_reason/{safe_reason}"] = (
                episode.metrics.get(
                    f"milestone/verification_unavailable_reason/{safe_reason}",
                    0.0,
                )
                + 1.0
            )
    return summaries


def compute_milestone_format_reward(step: Step) -> float:
    """Return 0 for a valid codeflow call and -1 for protocol/schema errors."""
    event = get_action_event(step)
    if event is not None:
        invalid_policy_violations = {
            "bash_repository_mutation_attempt",
            "bash_modified_protected_tests",
            "edit_modified_protected_tests",
            "repository_head_change_attempt",
            "repository_head_changed",
        }
        valid = (
            event.parse_status == ParseStatus.OK
            and event.validation_status == ValidationStatus.OK
            and event.action_category != ActionCategory.INVALID
            and not (invalid_policy_violations & set(event.policy_violations))
        )
        return 0.0 if valid else -1.0
    return compute_tool_call_reward(step, correct_reward=0.0, incorrect_reward=-1.0)


def _status_map(value: Any) -> dict[str, str]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict) or not value:
        raise ValueError("test state must be a non-empty object")
    result: dict[str, str] = {}
    for name, status in value.items():
        if not isinstance(name, str) or not name or not isinstance(status, str) or not status:
            raise ValueError("test state keys and values must be non-empty strings")
        result[name] = status
    return result


def _verification_contract(task_metadata: dict[str, Any]) -> tuple[dict[str, str], dict[str, str], str | None]:
    try:
        baseline = _status_map(task_metadata.get("baseline_output_json"))
        target = _status_map(task_metadata.get("target_output_json"))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return {}, {}, f"invalid_verification_metadata:{exc}"
    if set(baseline) != set(target):
        return baseline, target, "baseline_target_test_set_mismatch"
    return baseline, target, None


def _verification_potential(
    results: dict[str, str],
    baseline: dict[str, str],
    target: dict[str, str],
    config: MilestoneRewardConfig,
    *,
    mode: str = "bug_repair",
) -> float:
    if _canonical_verification_potential_mode(mode) == "pass_count":
        current_count = sum(results.get(name) == target[name] for name in target)
        baseline_count = sum(baseline.get(name) == target[name] for name in target)
        return float(current_count - baseline_count)
    failing_partition = [name for name in target if baseline[name] != target[name]]
    stable_partition = [name for name in target if baseline[name] == target[name]]
    fixed = sum(results[name] == target[name] for name in failing_partition)
    regressed = sum(results[name] != target[name] for name in stable_partition)
    p_value = fixed / len(failing_partition) if failing_partition else 0.0
    b_value = regressed / len(stable_partition) if stable_partition else 0.0
    return p_value - config.beta_any * float(b_value > 0) - config.beta_frac * b_value


@dataclass
class _NeutralUnrunState:
    """Rollout-local evidence that partial probes are not allowed to erase."""

    confirmed_p2p_by_test: dict[str, bool]
    confirmed_p2p_count: int = 0


@dataclass
class _PassCountState:
    """Conservative P2P regression state for partial count observations."""

    p2p_expected: int = 0
    confirmed_p2p_regression_count: int = 0


def _new_neutral_unrun_state(baseline: dict[str, str], target: dict[str, str]) -> _NeutralUnrunState:
    return _NeutralUnrunState(confirmed_p2p_by_test={name: False for name in target if name in baseline and baseline[name] == target[name]})


def _count_observation_passed(
    observation: TestCountObservation,
    state: _PassCountState,
) -> int:
    """Return effective ``C_t`` while keeping P2P ``NOT_RUN`` neutral.

    Complete observations use their raw pass count.  For a partial
    observation without an authenticated partition breakdown every unreported
    test is non-passing.  With a partition breakdown, explicit P2P
    failures/errors/skips establish regressions and ``NOT_RUN`` retains the
    previous regression count instead of creating or clearing evidence.
    """

    partitions = observation.partition_counts
    if observation.complete or partitions is None:
        if observation.complete and partitions is not None:
            state.confirmed_p2p_regression_count = (
                partitions.p2p.failed
                + partitions.p2p.errored
                + partitions.p2p.skipped
            )
        elif observation.complete:
            state.confirmed_p2p_regression_count = max(
                state.confirmed_p2p_regression_count,
                state.p2p_expected - observation.passed,
            )
        return observation.passed

    explicit_regressions = (
        partitions.p2p.failed
        + partitions.p2p.errored
        + partitions.p2p.skipped
    )
    if partitions.p2p.not_run == 0:
        state.confirmed_p2p_regression_count = explicit_regressions
    else:
        state.confirmed_p2p_regression_count = max(
            state.confirmed_p2p_regression_count,
            explicit_regressions,
        )
    return partitions.f2p.passed + max(
        0,
        partitions.p2p.expected - state.confirmed_p2p_regression_count,
    )


def _potential_from_components(
    fixed: int,
    f2p_expected: int,
    regressed: int,
    p2p_expected: int,
    config: MilestoneRewardConfig,
) -> float:
    fixed_fraction = fixed / f2p_expected if f2p_expected else 0.0
    regression_fraction = regressed / p2p_expected if p2p_expected else 0.0
    return fixed_fraction - config.beta_any * float(regression_fraction > 0) - config.beta_frac * regression_fraction


def _neutral_unrun_per_test_potential(
    results: dict[str, str],
    baseline: dict[str, str],
    target: dict[str, str],
    config: MilestoneRewardConfig,
    state: _NeutralUnrunState,
) -> float:
    failing_partition = [name for name in target if baseline[name] != target[name]]
    stable_partition = [name for name in target if baseline[name] == target[name]]
    fixed = sum(str(results.get(name, "NOT_RUN")).upper() == "PASSED" for name in failing_partition)
    for name in stable_partition:
        status = str(results.get(name, "NOT_RUN")).upper()
        if status == "PASSED":
            state.confirmed_p2p_by_test[name] = False
        elif status in {"FAILED", "ERROR", "ERRORS", "ERRORED"}:
            state.confirmed_p2p_by_test[name] = True
        # SKIPPED and NOT_RUN are deliberately neutral and retain prior evidence.
    regressed = sum(state.confirmed_p2p_by_test.values())
    state.confirmed_p2p_count = regressed
    return _potential_from_components(
        fixed,
        len(failing_partition),
        regressed,
        len(stable_partition),
        config,
    )


def _neutral_unrun_partition_potential(
    partitions: TestPartitionResult,
    baseline: dict[str, str],
    target: dict[str, str],
    config: MilestoneRewardConfig,
    state: _NeutralUnrunState,
) -> float:
    f2p_expected = sum(baseline[name] != target[name] for name in target)
    p2p_expected = len(target) - f2p_expected
    if partitions.f2p.expected != f2p_expected or partitions.p2p.expected != p2p_expected:
        raise ValueError("partition_expected_count_mismatch")
    explicit_regressions = partitions.p2p.failed + partitions.p2p.errored
    if partitions.p2p.is_complete:
        state.confirmed_p2p_count = explicit_regressions
    else:
        state.confirmed_p2p_count = max(state.confirmed_p2p_count, explicit_regressions)
    return _potential_from_components(
        partitions.f2p.passed,
        f2p_expected,
        state.confirmed_p2p_count,
        p2p_expected,
        config,
    )


def _partition_matches_contract(
    partitions: TestPartitionResult | None,
    baseline: dict[str, str],
    target: dict[str, str],
) -> bool:
    if partitions is None or set(baseline) != set(target):
        return False
    f2p_expected = sum(baseline[name] != target[name] for name in target)
    return partitions.f2p.expected == f2p_expected and partitions.p2p.expected == len(target) - f2p_expected


def _trusted_verification_event(
    test_event: Any,
    baseline: dict[str, str],
    target: dict[str, str],
) -> bool:
    if test_event is None or test_event.status != TestEventStatus.COMPLETED or not test_event.trusted or test_event.missing_tests or test_event.extra_tests:
        return False
    if test_event.result_source == TestResultSource.PARTITION_COUNTS:
        return _partition_matches_contract(test_event.partition_counts, baseline, target)
    return set(test_event.test_results) == set(target)


def _trusted_count_event(test_event: Any, expected: int) -> bool:
    """Whether an event contains the current acceptance-count observation."""

    return bool(
        test_event is not None
        and test_event.status == TestEventStatus.COMPLETED
        and test_event.trusted
        and test_event.result_source == TestResultSource.COUNT_COLLECTOR
        and test_event.count_observation is not None
        and test_event.count_observation.schema_version == 4
        and test_event.count_observation.expected == expected
    )


def _materialized_count_inputs(
    task_metadata: dict[str, Any],
) -> tuple[dict[str, str], dict[str, str], int, int, str | None]:
    """Resolve fixed ``N`` and the materialized fallback ``C0``."""

    try:
        baseline = _status_map(task_metadata.get("baseline_output_json"))
        target = _status_map(task_metadata.get("target_output_json"))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return {}, {}, 0, 0, f"invalid_verification_metadata:{exc}"
    expected = len(target)
    baseline_passed = sum(
        str(baseline.get(name, "NOT_RUN")).upper() == "PASSED"
        for name in target
    )
    return baseline, target, expected, baseline_passed, None


def _resolved_verification_count_contract(
    context: MilestoneContext | None,
    task_metadata: dict[str, Any],
) -> tuple[
    VerificationCountContract | None,
    dict[str, str],
    dict[str, str],
    str | None,
]:
    baseline, target, expected, materialized_passed, reason = (
        _materialized_count_inputs(task_metadata)
    )
    if reason is not None:
        return None, baseline, target, reason
    contract = context.verification_count_contract if context is not None else None
    if contract is None:
        return None, baseline, target, "verification_count_contract_unavailable"
    if (
        contract.expected != expected
        or contract.materialized_baseline_passed != materialized_passed
    ):
        return None, baseline, target, "verification_count_contract_mismatch"
    return contract, baseline, target, None


def _relative_repo_path(path: Any, task_metadata: dict[str, Any]) -> str | None:
    if not isinstance(path, str) or not path.strip():
        return None
    raw_root = str(task_metadata.get("workdir") or "/testbed").strip() or "/testbed"
    root = posixpath.normpath(raw_root if raw_root.startswith("/") else f"/{raw_root}")
    raw = path.strip().replace("\\", "/")
    if posixpath.isabs(raw):
        normalized = posixpath.normpath(raw)
        root_prefix = root.rstrip("/") + "/"
        if normalized == root:
            relative = "."
        elif normalized.startswith(root_prefix):
            relative = normalized[len(root_prefix) :]
        else:
            return None
    else:
        relative = posixpath.normpath(raw)
    if relative in {"", "."} or relative == ".." or relative.startswith("../"):
        return None
    return relative.removeprefix("./")


def _navigation_contract(task_metadata: dict[str, Any]) -> tuple[list[str], str | None]:
    raw = task_metadata.get("relevant_files")
    if not isinstance(raw, list) or not raw:
        return [], "missing_relevant_files"
    normalized: list[str] = []
    for path in raw:
        relative = _relative_repo_path(path, task_metadata)
        if relative is None:
            return [], f"invalid_relevant_file:{path!r}"
        if relative not in normalized:
            normalized.append(relative)
    return normalized, None


def _runtime_baseline_from_event(
    raw: dict[str, Any] | None,
    target: dict[str, str],
) -> tuple[dict[str, str], str | None]:
    if not isinstance(raw, dict):
        return {}, "shadow_baseline_unavailable"
    if raw.get("status") != TestEventStatus.COMPLETED.value or raw.get("trusted") is not True:
        return {}, str(raw.get("failure_type") or raw.get("error") or "shadow_baseline_untrusted")
    try:
        observed = _status_map(raw.get("test_results"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}, "shadow_baseline_results_invalid"
    if set(observed) != set(target):
        return {}, "shadow_baseline_test_set_mismatch"
    return observed, None


def _task_metadata(trajectory: Trajectory, episode: Episode | None) -> dict[str, Any]:
    if episode is not None and isinstance(episode.task, dict):
        return episode.task
    if isinstance(trajectory.task, dict):
        return trajectory.task
    context = get_milestone_context(trajectory)
    if context is not None:
        return context.task_metadata
    return {}


def _persisted_baseline_event(trajectory: Trajectory) -> tuple[dict[str, Any] | None, bool]:
    context = get_milestone_context(trajectory)
    if context is None:
        return None, False
    runtime_required = bool(context.shadow_schema_version is not None and context.shadow_schema_version >= 4)
    return context.shadow_baseline_test_event, runtime_required


def _episode_baseline_event(episode: Episode | None) -> tuple[dict[str, Any] | None, bool]:
    if episode is None or not isinstance(episode.metadata, dict):
        return None, False
    shadow = episode.metadata.get("shadow_sandbox")
    if not isinstance(shadow, dict):
        return None, False
    raw = shadow.get("baseline_test_event")
    schema_version = shadow.get("schema_version")
    return raw if isinstance(raw, dict) else None, isinstance(schema_version, int) and schema_version >= 4


def annotate_trajectory_milestone_rewards(
    trajectory: Trajectory,
    config: MilestoneRewardConfig,
    *,
    episode: Episode | None = None,
) -> list[MilestoneStepReward]:
    """Compute and attach milestone process rewards for one trajectory."""
    if not config.enable:
        return []

    task_metadata = _task_metadata(trajectory, episode)
    persisted_baseline_event, persisted_runtime_required = _persisted_baseline_event(trajectory)
    episode_baseline_event, episode_runtime_required = _episode_baseline_event(episode)
    if episode_baseline_event is not None or episode_runtime_required:
        raw_baseline_event = episode_baseline_event
        runtime_required = episode_runtime_required
    else:
        raw_baseline_event = persisted_baseline_event
        runtime_required = persisted_runtime_required
    context = get_milestone_context(trajectory)
    shadow_unavailable_reason = (
        context.shadow_verification_unavailable_reason if context is not None else None
    )
    if episode is not None and isinstance(episode.metadata, dict):
        shadow_summary = episode.metadata.get("shadow_sandbox")
        if isinstance(shadow_summary, dict) and shadow_summary.get("verification_potential_unavailable_reason"):
            shadow_unavailable_reason = str(shadow_summary["verification_potential_unavailable_reason"])
    verification_mode = _canonical_verification_potential_mode(
        (context.verification_potential_mode if context is not None else "")
        or task_metadata.get("verification_potential_mode")
        or (
            "pass_count"
            if isinstance(episode.metadata.get("shadow_sandbox") if episode else None, dict) and episode.metadata["shadow_sandbox"].get("result_parser") == "denovoswe_official_v1"
            else "bug_repair"
        )
    )
    verification_policy = (
        context.verification_potential_policy
        if context is not None
        else (
            REPO_GENERATION_POLICY_PASS_COUNT
            if verification_mode == "pass_count"
            else BUG_REPAIR_POLICY_PASS_COUNT
            if verification_mode == BUG_REPAIR_PASS_COUNT_MODE
            else BUG_REPAIR_POLICY_NEUTRAL_UNRUN
        )
    )
    count_contract: VerificationCountContract | None = None
    expected_count = 0
    baseline_passed_count: int | None = None
    current_passed_count: int | None = None
    if verification_mode == BUG_REPAIR_PASS_COUNT_MODE:
        (
            count_contract,
            baseline,
            target,
            verification_reason,
        ) = _resolved_verification_count_contract(context, task_metadata)
        if (
            verification_reason == "verification_count_contract_unavailable"
            and shadow_unavailable_reason
        ):
            verification_reason = shadow_unavailable_reason
        if count_contract is not None:
            expected_count = count_contract.expected
            baseline_passed_count = count_contract.baseline_passed
            current_passed_count = baseline_passed_count
            verification_policy = BUG_REPAIR_POLICY_PASS_COUNT
    elif verification_mode == "pass_count":
        if not isinstance(raw_baseline_event, dict) or raw_baseline_event.get("status") != TestEventStatus.COMPLETED.value or raw_baseline_event.get("trusted") is not True:
            baseline, target = {}, {}
            verification_reason = "shadow_baseline_unavailable"
        else:
            try:
                baseline = _status_map(raw_baseline_event.get("test_results"))
            except (TypeError, ValueError, json.JSONDecodeError):
                baseline = {}
            target = {name: "PASSED" for name in baseline}
            verification_reason = None if target else "shadow_baseline_results_invalid"
            if verification_reason is None:
                expected_count = len(target)
                baseline_passed_count = sum(
                    baseline.get(name) == target.get(name)
                    for name in target
                )
                current_passed_count = baseline_passed_count
                repo_contract = (
                    context.verification_count_contract
                    if context is not None
                    else None
                )
                if (
                    repo_contract is not None
                    and repo_contract.expected == expected_count
                    and repo_contract.baseline_passed
                    == baseline_passed_count
                ):
                    count_contract = repo_contract
                verification_policy = REPO_GENERATION_POLICY_PASS_COUNT
    else:
        materialized_baseline, target, materialized_verification_reason = _verification_contract(task_metadata)
        if raw_baseline_event is not None:
            baseline, verification_reason = _runtime_baseline_from_event(raw_baseline_event, target)
        elif runtime_required:
            baseline, verification_reason = {}, shadow_unavailable_reason or "shadow_baseline_unavailable"
        else:
            baseline, verification_reason = (
                materialized_baseline,
                materialized_verification_reason,
            )
    has_trusted_step_test_event = False
    for step in trajectory.steps:
        event = get_action_event(step)
        test_event = event.test_event if event is not None else None
        trusted_step = (
            _trusted_count_event(test_event, expected_count)
            if verification_mode == BUG_REPAIR_PASS_COUNT_MODE
            else _trusted_verification_event(test_event, baseline, target)
        )
        if trusted_step:
            has_trusted_step_test_event = True
            break
    has_shadow_baseline = raw_baseline_event is not None
    if (
        verification_reason is None
        and not has_trusted_step_test_event
        and not has_shadow_baseline
        and verification_mode != BUG_REPAIR_PASS_COUNT_MODE
    ):
        verification_reason = "trusted_test_results_unavailable"

    relevant_files, navigation_reason = _navigation_contract(task_metadata)
    relevant = set(relevant_files)
    navigation_scores = dict.fromkeys(relevant_files, 0.0)
    neutral_unrun_state = _new_neutral_unrun_state(baseline, target)
    pass_count_p2p_expected = sum(
        baseline.get(name) == target.get(name) for name in target
    )
    pass_count_state = _PassCountState(
        p2p_expected=(
            pass_count_p2p_expected
            if verification_mode == BUG_REPAIR_PASS_COUNT_MODE
            else 0
        ),
        confirmed_p2p_regression_count=(
            max(
                0,
                pass_count_p2p_expected - baseline_passed_count,
            )
            if verification_mode == BUG_REPAIR_PASS_COUNT_MODE
            and baseline_passed_count is not None
            else 0
        )
    )
    if verification_reason is None:
        if verification_mode == BUG_REPAIR_PASS_COUNT_MODE:
            # The baseline defines Phi_0 rather than becoming a rewarded
            # trajectory step, regardless of whether C0 came from the runtime
            # probe or the materialized fallback.
            verification_potential = 0.0
        elif verification_mode == "bug_repair" and verification_policy == BUG_REPAIR_POLICY_NEUTRAL_UNRUN:
            verification_potential = _neutral_unrun_per_test_potential(
                baseline,
                baseline,
                target,
                config,
                neutral_unrun_state,
            )
        else:
            verification_potential = _verification_potential(
                baseline,
                baseline,
                target,
                config,
                mode=verification_mode,
            )
    else:
        verification_potential = 0.0
    verification_frozen = verification_reason is not None
    navigation_potential = 0.0
    records: list[MilestoneStepReward] = []
    active_merge_group_id: str | None = None
    active_merge_final_potential: float | None = None
    active_merge_step_delta: float | None = None
    active_merge_final_passed: int | None = None
    active_merge_reanchored = False

    for step in trajectory.steps:
        event = get_action_event(step)
        verification_before = verification_potential
        passed_before = current_passed_count
        verification_updated = False
        step_verification_reason = verification_reason
        test_event = event.test_event if event is not None else None
        merge_group_id = (
            test_event.probe_merge_group_id
            if test_event is not None
            else None
        )
        merge_group_size = (
            test_event.probe_merge_group_size
            if test_event is not None
            else None
        )
        merge_group_index = (
            test_event.probe_merge_group_index
            if test_event is not None
            else None
        )
        merge_anchor_action_id = (
            test_event.probe_merge_anchor_action_id
            if test_event is not None
            else None
        )
        reanchor_requested = bool(
            verification_mode == "pass_count"
            and test_event is not None
            and test_event.potential_reanchored
        )
        if reanchor_requested and verification_frozen:
            # A completed parallel probe can predate an earlier lane timeout in
            # wall-clock order.  The runtime marks it only after ordered
            # reconciliation, proving that a trusted absolute state exists on
            # the far side of the gap.  Resume from that state without paying
            # the unobserved cross-gap delta.
            verification_frozen = False
            verification_reason = None
            step_verification_reason = None
        verification_reanchored = False
        probe_potential_after: float | None = None
        if config.verification_enable and not verification_frozen and test_event is not None:
            trusted = (
                _trusted_count_event(test_event, expected_count)
                if verification_mode == BUG_REPAIR_PASS_COUNT_MODE
                else _trusted_verification_event(test_event, baseline, target)
            )
            if trusted:
                if verification_mode == BUG_REPAIR_PASS_COUNT_MODE:
                    assert test_event.count_observation is not None
                    assert baseline_passed_count is not None
                    current_passed_count = _count_observation_passed(
                        test_event.count_observation,
                        pass_count_state,
                    )
                    if not 0 <= current_passed_count <= expected_count:
                        raise ValueError(
                            "effective verification passed count is outside the "
                            "materialized acceptance contract"
                        )
                    verification_potential = float(
                        current_passed_count - baseline_passed_count
                    )
                    lower_bound = float(-baseline_passed_count)
                    upper_bound = float(expected_count - baseline_passed_count)
                    if not lower_bound <= verification_potential <= upper_bound:
                        raise ValueError(
                            "verification count potential is outside the materialized "
                            "acceptance-contract bounds"
                        )
                elif verification_mode == "bug_repair" and verification_policy == BUG_REPAIR_POLICY_NEUTRAL_UNRUN:
                    if test_event.result_source == TestResultSource.PARTITION_COUNTS:
                        assert test_event.partition_counts is not None
                        verification_potential = _neutral_unrun_partition_potential(
                            test_event.partition_counts,
                            baseline,
                            target,
                            config,
                            neutral_unrun_state,
                        )
                    else:
                        verification_potential = _neutral_unrun_per_test_potential(
                            test_event.test_results,
                            baseline,
                            target,
                            config,
                            neutral_unrun_state,
                        )
                else:
                    physical_verification_potential = _verification_potential(
                        test_event.test_results,
                        baseline,
                        target,
                        config,
                        mode=verification_mode,
                    )
                    if verification_mode == "pass_count":
                        physical_passed_count = sum(
                            test_event.test_results.get(name)
                            == target.get(name)
                            for name in target
                        )
                        if merge_group_id is not None:
                            if (
                                merge_group_size is None
                                or merge_group_index is None
                            ):
                                raise ValueError(
                                    "incomplete probe merge metadata"
                                )
                            if merge_group_index == 0:
                                active_merge_group_id = merge_group_id
                                active_merge_final_potential = (
                                    physical_verification_potential
                                )
                                active_merge_final_passed = physical_passed_count
                                active_merge_reanchored = reanchor_requested
                                if active_merge_reanchored:
                                    verification_before = (
                                        physical_verification_potential
                                    )
                                    passed_before = physical_passed_count
                                    verification_potential = (
                                        physical_verification_potential
                                    )
                                    active_merge_step_delta = 0.0
                                else:
                                    active_merge_step_delta = (
                                        physical_verification_potential
                                        - verification_before
                                    ) / merge_group_size
                            elif active_merge_group_id != merge_group_id:
                                raise ValueError(
                                    "probe merge group is not contiguous in trajectory"
                                )
                            assert active_merge_final_potential is not None
                            assert active_merge_step_delta is not None
                            if active_merge_reanchored:
                                verification_before = active_merge_final_potential
                                passed_before = active_merge_final_passed
                                verification_potential = active_merge_final_potential
                            verification_reanchored = active_merge_reanchored
                            probe_potential_after = active_merge_final_potential
                            if merge_group_index == merge_group_size - 1:
                                verification_potential = active_merge_final_potential
                                current_passed_count = active_merge_final_passed
                                active_merge_group_id = None
                                active_merge_final_potential = None
                                active_merge_step_delta = None
                                active_merge_final_passed = None
                                active_merge_reanchored = False
                            else:
                                if not active_merge_reanchored:
                                    verification_potential = (
                                        verification_before + active_merge_step_delta
                                    )
                        else:
                            if reanchor_requested:
                                verification_before = physical_verification_potential
                                passed_before = physical_passed_count
                                verification_reanchored = True
                            verification_potential = physical_verification_potential
                            current_passed_count = physical_passed_count
                    else:
                        verification_potential = physical_verification_potential
                verification_updated = True
            else:
                step_verification_reason = str(test_event.failure_type or test_event.error or "untrusted_test_event")
                if resolve_test_event_continuity(test_event) == TestEventContinuityStatus.LOST:
                    verification_frozen = True
                    verification_reason = step_verification_reason

        navigation_before = navigation_potential
        navigation_updates: dict[str, float] = {}
        if config.navigation_enable and navigation_reason is None and event is not None and event.execution_status == ExecutionStatus.SUCCESS:
            observed_scores: dict[str, float] = {}
            if event.file_exposures:
                for exposure in event.file_exposures:
                    if exposure.confidence != "high":
                        continue
                    path = _relative_repo_path(exposure.path, task_metadata)
                    if path is None or path not in relevant:
                        continue
                    observed_score = config.navigation_read_score if exposure.behavior == "read" and exposure.exposure_kind == "content" else config.navigation_search_score
                    observed_scores[path] = max(observed_scores.get(path, 0.0), observed_score)
            else:
                exposed_ranges = {relative for item in event.exposed_line_ranges if (relative := _relative_repo_path(item.path, task_metadata)) is not None}
                exposed_files = {relative for path in event.exposed_files if (relative := _relative_repo_path(path, task_metadata)) is not None}
                for path in sorted(exposed_files & relevant):
                    observed_score = 0.0
                    shell_behaviors = set(event.shell_behaviors)
                    if event.action_category == ActionCategory.VIEW or "read" in shell_behaviors:
                        if "read" in shell_behaviors or path in exposed_ranges:
                            observed_score = config.navigation_read_score
                        else:
                            observed_score = config.navigation_search_score
                    elif event.action_category == ActionCategory.SEARCH or "search" in shell_behaviors:
                        observed_score = config.navigation_search_score
                    observed_scores[path] = observed_score
            for path, observed_score in sorted(observed_scores.items()):
                if observed_score > navigation_scores[path]:
                    navigation_scores[path] = observed_score
                    navigation_updates[path] = observed_score
            navigation_potential = sum(navigation_scores.values()) / len(relevant_files)

        verification_component_before = config.verification_weight * verification_before if config.verification_enable else 0.0
        verification_component_after = config.verification_weight * verification_potential if config.verification_enable else 0.0
        navigation_component_before = config.navigation_weight * navigation_before if config.navigation_enable else 0.0
        navigation_component_after = config.navigation_weight * navigation_potential if config.navigation_enable else 0.0
        total_before = verification_component_before + navigation_component_before
        total_after = verification_component_after + navigation_component_after
        verification_reward = verification_component_after - verification_component_before
        verification_delta: float | None = None
        clipped_verification_delta: float | None = None
        if config.verification_enable and verification_mode in {
            "pass_count",
            BUG_REPAIR_PASS_COUNT_MODE,
        }:
            verification_delta = verification_potential - verification_before
            clipped_verification_delta = min(
                config.verification_reward_clip_upper,
                max(config.verification_reward_clip_lower, verification_delta),
            )
            verification_reward = config.verification_weight * clipped_verification_delta
        navigation_reward = navigation_component_after - navigation_component_before
        format_reward = compute_milestone_format_reward(step) if config.format_enable else 0.0
        records.append(
            MilestoneStepReward(
                verification_potential_policy=verification_policy,
                verification_available=config.verification_enable and step_verification_reason is None,
                verification_updated=verification_updated,
                verification_unavailable_reason=step_verification_reason,
                verification_potential_before=verification_before,
                verification_potential_after=verification_potential,
                verification_passed_before=(
                    passed_before
                    if verification_mode
                    in {"pass_count", BUG_REPAIR_PASS_COUNT_MODE}
                    else None
                ),
                verification_passed_after=(
                    current_passed_count
                    if verification_mode
                    in {"pass_count", BUG_REPAIR_PASS_COUNT_MODE}
                    else None
                ),
                verification_raw_delta=verification_delta,
                verification_clipped_delta=clipped_verification_delta,
                verification_reanchored=verification_reanchored,
                probe_merge_group_id=merge_group_id,
                probe_merge_group_size=merge_group_size,
                probe_merge_group_index=merge_group_index,
                probe_merge_anchor_action_id=merge_anchor_action_id,
                verification_probe_potential_after=probe_potential_after,
                navigation_available=config.navigation_enable and navigation_reason is None,
                navigation_potential_before=navigation_before,
                navigation_potential_after=navigation_potential,
                navigation_updates=navigation_updates,
                total_potential_before=total_before,
                total_potential_after=total_after,
                potential_reward=verification_reward + navigation_reward,
                format_reward=format_reward,
            )
        )

    if (
        verification_mode == BUG_REPAIR_PASS_COUNT_MODE
        and config.verification_enable
        and records
    ):
        raw_deltas = []
        for record in records:
            expected_delta = (
                record.verification_potential_after
                - record.verification_potential_before
            )
            if record.verification_raw_delta != expected_delta:
                raise ValueError(
                    "verification count step delta does not match its potential change"
                )
            raw_deltas.append(expected_delta)
        if abs(
            sum(raw_deltas)
            - (records[-1].verification_potential_after - records[0].verification_potential_before)
        ) > 1e-9:
            raise ValueError("verification count potential does not telescope")

    discounted_future = 0.0
    for record in reversed(records):
        record.backward_credit = config.backward_credit_lambda * discounted_future
        record.process_advantage = record.potential_reward + record.backward_credit + config.format_weight * record.format_reward
        discounted_future = config.backward_credit_gamma * (record.potential_reward + discounted_future)

    for step, record in zip(trajectory.steps, records, strict=True):
        step.info[MILESTONE_REWARD_METADATA_KEY] = record.model_dump(mode="json")

    summary = {
        "schema_version": MILESTONE_REWARD_SCHEMA_VERSION,
        "config": config.to_dict(),
        "verification_available": bool(records and records[-1].verification_available),
        "verification_unavailable_reason": records[-1].verification_unavailable_reason if records else verification_reason,
        "navigation_available": navigation_reason is None and config.navigation_enable,
        "relevant_files": relevant_files,
        "verification_potential_final": records[-1].verification_potential_after if records else 0.0,
        "verification_potential_mode": verification_mode,
        "verification_potential_policy": verification_policy,
        "verification_result_mode": (context.verification_result_mode if context is not None else "per_test"),
        "verification_count_contract": (
            count_contract.model_dump(mode="json")
            if count_contract is not None
            else None
        ),
        "verification_passed_final": (
            current_passed_count
            if verification_mode in {"pass_count", BUG_REPAIR_PASS_COUNT_MODE}
            else None
        ),
        "verification_reanchors": sum(
            record.verification_reanchored
            and (
                record.probe_merge_group_id is None
                or record.probe_merge_group_index == 0
            )
            for record in records
        ),
        "verification_reanchored": any(
            record.verification_reanchored for record in records
        ),
        "partition_adapter": context.partition_adapter if context is not None else None,
        "partition_calibration_status": (context.partition_calibration_status if context is not None else None),
        "navigation_potential_final": records[-1].navigation_potential_after if records else 0.0,
        "potential_reward_sum": sum(record.potential_reward for record in records),
        "process_advantage_sum": sum(record.process_advantage for record in records),
        "invalid_tool_calls": sum(record.format_reward < 0 for record in records),
    }
    trajectory.info[MILESTONE_REWARD_METADATA_KEY] = summary
    return records


def annotate_episode_milestone_rewards(
    episode: Episode,
    config: MilestoneRewardConfig,
) -> list[list[MilestoneStepReward]]:
    """Attach process rewards before rollout logging and return all records."""
    if not config.enable:
        return []
    attach_episode_milestone_context(episode)
    all_records = [annotate_trajectory_milestone_rewards(trajectory, config, episode=episode) for trajectory in episode.trajectories]
    flat = [record for records in all_records for record in records]
    if flat:
        episode.metrics.update(
            {
                "milestone/verification_available": sum(record.verification_available for record in flat) / len(flat),
                "milestone/navigation_available": sum(record.navigation_available for record in flat) / len(flat),
                "milestone/potential_reward_mean": sum(record.potential_reward for record in flat) / len(flat),
                "milestone/process_advantage_mean": sum(record.process_advantage for record in flat) / len(flat),
                "milestone/invalid_tool_call_fraction": sum(record.format_reward < 0 for record in flat) / len(flat),
            }
        )
    return all_records
