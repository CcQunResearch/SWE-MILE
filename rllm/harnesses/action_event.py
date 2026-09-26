"""Structured action telemetry for the native ``codeflow`` scaffold.

The event is stored as JSON-compatible data in ``Step.metadata`` so existing
Step/Trajectory serialization remains unchanged.  Reward code should use
``get_action_event`` because gateway trace enrichment nests agent-side
metadata under ``agent_step_metadata``.

The local materialization audit covered 4,578 tasks: all milestone fields are
present in the training Parquet, and VERL preserves them under ``extra_info``.
Only 2,361 tasks have ``FAIL_TO_PASS`` / ``PASS_TO_PASS`` lists that form a
complete state partition.  R2E-Gym milestone rewards therefore derive their
full F/G partitions from the rollout-local runtime baseline and
``target_output_json``; ``baseline_output_json`` is retained as an audit
reference rather than the training-time authority.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rllm.types import Step

ACTION_EVENT_METADATA_KEY = "action_event"
ACTION_EVENT_SCHEMA_VERSION = 11


class _StringEnum(str, Enum):
    """String-valued enum that serializes cleanly through Pydantic."""


class ActionCategory(_StringEnum):
    SEARCH = "SEARCH"
    VIEW = "VIEW"
    EDIT = "EDIT"
    EXECUTE = "EXECUTE"
    SUBMIT = "SUBMIT"
    INVALID = "INVALID"


class ParseStatus(_StringEnum):
    OK = "ok"
    ERROR = "error"


class ValidationStatus(_StringEnum):
    OK = "ok"
    ERROR = "error"
    SKIPPED = "skipped"


class ExecutionStatus(_StringEnum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    NOT_RUN = "not_run"


class RepoStateStatus(_StringEnum):
    OK = "ok"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class ChangeType(_StringEnum):
    MODIFIED = "modified"
    CREATED = "created"
    DELETED = "deleted"
    RENAMED = "renamed"


class TestEventStatus(_StringEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"
    STATE_MISMATCH = "state_mismatch"
    SKIPPED = "skipped"


class TestEventContinuityStatus(_StringEnum):
    """Whether shadow state remains usable after a probe result."""

    INTACT = "intact"
    RECOVERABLE_GAP = "recoverable_gap"
    LOST = "lost"


class TestResultSource(_StringEnum):
    PYTEST_SUMMARY = "pytest_summary"
    PYTEST_PLUGIN = "pytest_plugin"
    OFFICIAL_LOG_PARSER = "official_log_parser"
    COLLECTION_ABORT = "collection_abort"
    CONTROLLED_TIMEOUT = "controlled_timeout"
    PARTITION_COUNTS = "partition_counts"
    COUNT_COLLECTOR = "count_collector"


class RestoreConfirmation(_StringEnum):
    CONFIRMED = "confirmed"
    FINGERPRINT_FALLBACK = "fingerprint_fallback"
    FAILED = "failed"


class ChangedFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    change_type: ChangeType
    old_path: str | None = None


class ExposedLineRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_order(self) -> ExposedLineRange:
        if self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        return self


class FileExposure(BaseModel):
    """File-level evidence that was actually visible to the model.

    Low/medium-confidence records are retained for audit, while navigation
    rewards consume only high-confidence records.  Line-level attribution is
    deliberately outside this contract.
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    behavior: Literal["search", "read"]
    exposure_kind: Literal["path", "content"]
    source: Literal[
        "dedicated_tool",
        "bash_rendered_path",
        "bash_single_operation",
    ]
    confidence: Literal["high", "medium", "low"] = "high"


class TestPartitionCounts(BaseModel):
    """Aggregate states observed while running one authoritative test partition."""

    model_config = ConfigDict(extra="forbid")

    expected: int = Field(ge=0)
    passed: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    errored: int = Field(default=0, ge=0)
    skipped: int = Field(default=0, ge=0)
    not_run: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_total(self) -> TestPartitionCounts:
        observed = self.passed + self.failed + self.errored + self.skipped + self.not_run
        if observed != self.expected:
            raise ValueError(f"partition status counts must sum to expected: observed={observed} expected={self.expected}")
        return self

    @property
    def is_complete(self) -> bool:
        return self.skipped == 0 and self.not_run == 0


class TestPartitionResult(BaseModel):
    """F2P/P2P aggregate observation for a calibrated partition probe."""

    model_config = ConfigDict(extra="forbid")

    f2p: TestPartitionCounts
    p2p: TestPartitionCounts


class TestCountObservation(BaseModel):
    """Authenticated aggregate counts for one acceptance-test probe.

    ``reported`` counts tests for which the collector observed a terminal
    state.  The remaining authoritative tests are represented explicitly as
    ``not_run``.  ``unclassified`` is a terminal non-passing result whose
    runner summary did not distinguish failure, error, or skip; it is never
    counted as passed.

    An optional F2P/P2P breakdown authenticates the fixed ``NOT_RUN`` rule:
    missing F2P remains unpassed and missing P2P does not create a new
    regression.  The public reward remains the scalar acceptance-test count.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[4] = 4
    scope: Literal["materialized_acceptance_contract"] = (
        "materialized_acceptance_contract"
    )
    expected: int = Field(ge=1)
    passed: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    errored: int = Field(default=0, ge=0)
    skipped: int = Field(default=0, ge=0)
    unclassified: int = Field(default=0, ge=0)
    reported: int = Field(ge=0)
    not_run: int = Field(ge=0)
    complete: bool
    collector: str = Field(min_length=1)
    collector_version: int = Field(default=1, ge=1)
    command_count: int = Field(default=1, ge=1)
    command_evidence: list[dict[str, Any]] = Field(default_factory=list)
    partition_counts: TestPartitionResult | None = None
    plan_hash: str
    termination: Literal[
        "complete",
        "interrupted",
        "deterministic_abort",
    ]
    partial_reason: str | None = None
    ignored_tests: list[str] = Field(default_factory=list)
    ownership_evidence: list[dict[str, Any]] = Field(default_factory=list)
    ownership_observed: int = Field(default=0, ge=0)
    ownership_sha256: str
    evidence_notes: list[str] = Field(default_factory=list)
    official_extra_count: int = Field(default=0, ge=0)
    official_extra_sample: list[str] = Field(default_factory=list)
    official_native_disagreement_count: int = Field(default=0, ge=0)
    official_native_disagreement_sample: list[str] = Field(default_factory=list)
    native_state_sha256: str

    @model_validator(mode="after")
    def validate_counts(self) -> TestCountObservation:
        categorized = (
            self.passed
            + self.failed
            + self.errored
            + self.skipped
            + self.unclassified
        )
        if categorized != self.reported:
            raise ValueError(
                "count observation terminal states must sum to reported: "
                f"states={categorized} reported={self.reported}"
            )
        if self.reported + self.not_run != self.expected:
            raise ValueError(
                "count observation reported and not_run must sum to expected: "
                f"reported={self.reported} not_run={self.not_run} "
                f"expected={self.expected}"
            )
        if self.complete != (self.not_run == 0):
            raise ValueError(
                "count observation complete must be true exactly when not_run=0"
            )
        if (
            not self.plan_hash
            or len(self.plan_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.plan_hash)
        ):
            raise ValueError("count observation requires a SHA-256 plan_hash")
        if self.complete and self.termination != "complete":
            raise ValueError("complete count observation requires complete termination")
        if not self.complete and self.termination == "complete":
            raise ValueError("partial count observation cannot have complete termination")
        if not self.complete and not self.partial_reason:
            raise ValueError("partial count observation requires partial_reason")
        if (
            not self.ownership_sha256
            or len(self.ownership_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.ownership_sha256
            )
        ):
            raise ValueError("count observation requires an ownership SHA-256")
        if self.ownership_observed != self.reported:
            raise ValueError("count observation ownership count must equal reported")
        if len(self.ownership_evidence) > 50:
            raise ValueError("count observation ownership sample exceeds 50 rows")
        if len(self.evidence_notes) > 20:
            raise ValueError("count observation evidence note limit exceeded")
        if len(self.official_extra_sample) > 50 or len(
            self.official_native_disagreement_sample
        ) > 50:
            raise ValueError("count observation audit sample exceeds 50 rows")
        if (
            len(self.native_state_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.native_state_sha256
            )
        ):
            raise ValueError("native_state_sha256 must be a SHA-256 digest")
        if self.partition_counts is not None:
            groups = (
                self.partition_counts.f2p,
                self.partition_counts.p2p,
            )
            partition_expected = sum(group.expected for group in groups)
            partition_passed = sum(group.passed for group in groups)
            partition_failed = sum(group.failed for group in groups)
            partition_errored = sum(group.errored for group in groups)
            partition_skipped = sum(group.skipped for group in groups)
            partition_not_run = sum(group.not_run for group in groups)
            if (
                self.unclassified != 0
                or partition_expected != self.expected
                or partition_passed != self.passed
                or partition_failed != self.failed
                or partition_errored != self.errored
                or partition_skipped != self.skipped
                or partition_not_run != self.not_run
            ):
                raise ValueError(
                    "count observation partition counts do not match aggregate counts"
                )
        return self


class TestEvent(BaseModel):
    """Future shadow-probe payload.

    Per-test mode stores ``test_results`` because each state must be compared
    with that test's authoritative target.  A separately calibrated
    partition-count mode stores aggregate F2P/P2P states instead; its expected
    sizes are checked again against the rollout baseline/target contract by
    reward and audit consumers. Dataset-authoritative subset contracts project
    named results before storage and retain out-of-contract observations in
    ``ignored_extra_tests``.
    """

    model_config = ConfigDict(extra="forbid")

    probe_id: str
    repo_state: str
    status: TestEventStatus
    action_id: str | None = None
    turn_id: int | None = Field(default=None, ge=0)
    probe_merge_group_id: str | None = None
    probe_merge_group_size: int | None = Field(default=None, ge=1)
    probe_merge_group_index: int | None = Field(default=None, ge=0)
    probe_merge_anchor_action_id: str | None = None
    probe_merge_decision_wait_s: float | None = Field(default=None, ge=0)
    probe_sequence: int | None = Field(default=None, ge=0)
    shadow_lane_id: int | None = Field(default=None, ge=0)
    shadow_lane_attempts: int = Field(default=0, ge=0)
    potential_reanchored: bool = False
    shadow_repo_state: str | None = None
    trusted: bool = False
    continuity_status: TestEventContinuityStatus | None = None
    result_source: TestResultSource | None = None
    result_completeness: Literal[
        "full",
        "baseline_contract_filled",
        "contract_assisted",
        "controlled_partial",
        "partition_counts_full",
        "partition_counts_partial",
        "count_full",
        "count_partial",
        "none",
    ] = "none"
    partition_counts: TestPartitionResult | None = None
    count_observation: TestCountObservation | None = None
    imputed_tests: list[str] = Field(default_factory=list)
    contract_fill_policy: str | None = None
    contract_fill_source: str | None = None
    failure_type: str | None = None
    failure_stage: str | None = None
    failure_origin: str | None = None
    failure_evidence: dict[str, Any] = Field(default_factory=dict)
    exit_code: int | None = None
    timed_out: bool = False
    test_results: dict[str, str] = Field(default_factory=dict)
    missing_tests: list[str] = Field(default_factory=list)
    extra_tests: list[str] = Field(default_factory=list)
    ignored_extra_tests: list[str] = Field(default_factory=list)
    passed: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    errored: int = Field(default=0, ge=0)
    not_run: int = Field(default=0, ge=0)
    probe_attempts: int = Field(default=0, ge=0)
    state_mismatch_paths: list[str] = Field(default_factory=list)
    restore_confirmation: RestoreConfirmation | None = None
    duration: float | None = Field(default=None, ge=0)
    test_potential_before: float | None = None
    test_potential_after: float | None = None
    error: str | None = None
    log_tail: str = ""

    @model_validator(mode="after")
    def validate_partition_result(self) -> TestEvent:
        merge_fields = (
            self.probe_merge_group_id,
            self.probe_merge_group_size,
            self.probe_merge_group_index,
            self.probe_merge_anchor_action_id,
            self.probe_merge_decision_wait_s,
        )
        if any(value is not None for value in merge_fields):
            if any(value is None for value in merge_fields):
                raise ValueError("probe merge metadata must be present together")
            assert self.probe_merge_group_size is not None
            assert self.probe_merge_group_index is not None
            if self.probe_merge_group_index >= self.probe_merge_group_size:
                raise ValueError(
                    "probe_merge_group_index must be smaller than group size"
                )
        uses_partition = self.result_source == TestResultSource.PARTITION_COUNTS
        partition_completeness = self.result_completeness in {
            "partition_counts_full",
            "partition_counts_partial",
        }
        if uses_partition != (self.partition_counts is not None):
            raise ValueError("partition_counts must be present exactly when result_source=partition_counts")
        if uses_partition != partition_completeness:
            raise ValueError("partition-count results require partition_counts_full or partition_counts_partial completeness")
        if self.result_completeness == "partition_counts_full" and self.partition_counts:
            if not self.partition_counts.f2p.is_complete or not self.partition_counts.p2p.is_complete:
                raise ValueError("partition_counts_full cannot contain skipped or not-run tests")
        uses_count = self.result_source == TestResultSource.COUNT_COLLECTOR
        count_completeness = self.result_completeness in {
            "count_full",
            "count_partial",
        }
        if uses_count != (self.count_observation is not None):
            raise ValueError(
                "count_observation must be present exactly when "
                "result_source=count_collector"
            )
        if uses_count != count_completeness:
            raise ValueError(
                "count collector results require count_full or count_partial "
                "completeness"
            )
        if self.count_observation is not None:
            if (
                self.result_completeness == "count_full"
                and not self.count_observation.complete
            ):
                raise ValueError("count_full requires a complete count observation")
            if (
                self.result_completeness == "count_partial"
                and self.count_observation.complete
            ):
                raise ValueError("count_partial requires an incomplete count observation")
        contract_assisted = self.result_completeness == "contract_assisted"
        if contract_assisted:
            if self.contract_fill_policy != "f2p_missing_neutral_v1":
                raise ValueError(
                    "contract_assisted requires f2p_missing_neutral_v1"
                )
            if self.contract_fill_source not in {
                "materialized_baseline",
                "neutral_not_run",
            }:
                raise ValueError("contract_assisted has an invalid fill source")
            if (
                not self.imputed_tests
                or len(self.imputed_tests) != len(set(self.imputed_tests))
                or any(name not in self.test_results for name in self.imputed_tests)
            ):
                raise ValueError(
                    "contract_assisted requires unique imputed tests in test_results"
                )
            imputed_statuses = {
                self.test_results[name] for name in self.imputed_tests
            }
            if (
                self.contract_fill_source == "neutral_not_run"
                and imputed_statuses != {"NOT_RUN"}
            ):
                raise ValueError("neutral contract fill must use NOT_RUN")
            if (
                self.contract_fill_source == "materialized_baseline"
                and "NOT_RUN" in imputed_statuses
            ):
                raise ValueError(
                    "materialized baseline contract fill cannot use NOT_RUN"
                )
        elif self.contract_fill_policy is not None or self.contract_fill_source is not None:
            raise ValueError(
                "contract fill metadata requires contract_assisted completeness"
            )
        return self


class ActionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[11] = ACTION_EVENT_SCHEMA_VERSION
    action_id: str
    turn_id: int = Field(ge=0)
    tool_name: str
    action_category: ActionCategory
    action_subtype: str = ""
    normalized_arguments: dict[str, Any] = Field(default_factory=dict)
    shell_behaviors: list[Literal["search", "read", "edit"]] = Field(default_factory=list)
    shell_behavior_evidence: list[str] = Field(default_factory=list)

    parse_status: ParseStatus
    parse_error: str | None = None
    validation_status: ValidationStatus
    validation_error: str | None = None
    execution_status: ExecutionStatus
    execution_error: str | None = None
    exit_code: int | None = None
    duration: float = Field(default=0.0, ge=0)

    repo_state_status: RepoStateStatus
    repo_state_before: str | None = None
    repo_state_after: str | None = None
    repo_changed: bool | None = None
    changed_files: list[ChangedFile] = Field(default_factory=list)
    attempted_changed_files: list[ChangedFile] = Field(default_factory=list)

    exposed_files: list[str] = Field(default_factory=list)
    exposed_line_ranges: list[ExposedLineRange] = Field(default_factory=list)
    file_exposures: list[FileExposure] = Field(default_factory=list)
    visible_characters: int = Field(default=0, ge=0)
    output_truncated: bool = False

    test_event: TestEvent | None = None
    policy_violations: list[str] = Field(default_factory=list)
    instrumentation_errors: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class RepoFileState:
    path: str
    base_exists: bool
    base_digest: str | None
    base_mode: str | None
    current_exists: bool
    current_digest: str | None
    current_mode: str | None


@dataclass
class RepoSnapshot:
    status: RepoStateStatus
    fingerprint: str | None = None
    baseline_ref: str | None = None
    current_head: str | None = None
    files: dict[str, RepoFileState] = field(default_factory=dict)
    repository_files: tuple[str, ...] | None = None
    inventory_error: str | None = None
    error: str | None = None


@dataclass
class Exposure:
    files: list[str] = field(default_factory=list)
    line_ranges: list[ExposedLineRange] = field(default_factory=list)
    file_exposures: list[FileExposure] = field(default_factory=list)
    visible_characters: int = 0
    output_truncated: bool = False


def action_category_for(tool_name: str, *, parsed: bool = True) -> ActionCategory:
    if not parsed:
        return ActionCategory.INVALID
    return {
        "search": ActionCategory.SEARCH,
        "read_file": ActionCategory.VIEW,
        "edit_file": ActionCategory.EDIT,
        "execute_bash": ActionCategory.EXECUTE,
        "submit": ActionCategory.SUBMIT,
    }.get(tool_name, ActionCategory.INVALID)


def get_action_event(step: Step) -> ActionEvent | None:
    """Read an ActionEvent before or after gateway trace enrichment."""
    metadata = step.metadata if isinstance(step.metadata, dict) else {}
    raw = metadata.get(ACTION_EVENT_METADATA_KEY)
    if raw is None:
        agent_metadata = metadata.get("agent_step_metadata")
        if isinstance(agent_metadata, dict):
            raw = agent_metadata.get(ACTION_EVENT_METADATA_KEY)
    if raw is None:
        return None
    if isinstance(raw, ActionEvent):
        return raw
    if not isinstance(raw, dict):
        return None
    try:
        return ActionEvent.model_validate(raw)
    except Exception:
        return None


def summarize_codeflow_bash_behavior(
    steps: Iterable[Step],
) -> dict[str, Any]:
    """Aggregate best-effort repository behaviors for ``execute_bash`` turns.

    Counts are intentionally multi-label: one shell command may search, read,
    and edit.  Unknown commands remain visible as unclassified attempts rather
    than being guessed into one of those behavior buckets.
    """

    behaviors = ("search", "read", "edit")
    policy_codes = {
        "search": "bash_repository_search_attempt",
        "read": "bash_repository_read_attempt",
        "edit": "bash_repository_mutation_attempt",
    }
    behavior_attempts = {behavior: 0 for behavior in behaviors}
    blocked_attempts = {behavior: 0 for behavior in behaviors}
    executed_attempts = {behavior: 0 for behavior in behaviors}
    total = 0
    classified = 0
    mixed = 0

    for step in steps:
        event = get_action_event(step)
        if event is None or event.tool_name != "execute_bash":
            continue
        total += 1
        detected = [behavior for behavior in behaviors if behavior in event.shell_behaviors]
        if detected:
            classified += 1
        if len(detected) > 1:
            mixed += 1
        violations = set(event.policy_violations)
        executed = event.validation_status == ValidationStatus.OK and event.execution_status != ExecutionStatus.NOT_RUN
        for behavior in detected:
            behavior_attempts[behavior] += 1
            if policy_codes[behavior] in violations:
                blocked_attempts[behavior] += 1
            if executed:
                executed_attempts[behavior] += 1

    return {
        "schema_version": 1,
        "execute_bash_attempts": total,
        "classified_attempts": classified,
        "unclassified_attempts": total - classified,
        "mixed_behavior_attempts": mixed,
        "behavior_attempts": behavior_attempts,
        "blocked_behavior_attempts": blocked_attempts,
        "executed_behavior_attempts": executed_attempts,
    }


def resolve_test_event_continuity(event: TestEvent) -> TestEventContinuityStatus:
    """Resolve explicit continuity evidence or infer it from trusted completion."""
    if event.continuity_status is not None:
        return event.continuity_status
    if event.status == TestEventStatus.COMPLETED and event.trusted:
        return TestEventContinuityStatus.INTACT
    return TestEventContinuityStatus.LOST


def changed_files_between(before: RepoSnapshot, after: RepoSnapshot) -> list[ChangedFile]:
    """Return action-local net file changes between two valid snapshots."""
    if before.status != RepoStateStatus.OK or after.status != RepoStateStatus.OK:
        return []

    paths = set(before.files) | set(after.files)

    def current(snapshot: RepoSnapshot, peer: RepoSnapshot, path: str) -> tuple[bool, str | None, str | None]:
        entry = snapshot.files.get(path)
        if entry is not None:
            return entry.current_exists, entry.current_digest, entry.current_mode
        peer_entry = peer.files.get(path)
        if peer_entry is not None:
            return peer_entry.base_exists, peer_entry.base_digest, peer_entry.base_mode
        return False, None, None

    created: list[tuple[str, str | None]] = []
    deleted: list[tuple[str, str | None]] = []
    changes: list[ChangedFile] = []
    for path in sorted(paths):
        before_exists, before_digest, before_mode = current(before, after, path)
        after_exists, after_digest, after_mode = current(after, before, path)
        if (before_exists, before_digest, before_mode) == (after_exists, after_digest, after_mode):
            continue
        if not before_exists and after_exists:
            created.append((path, after_digest))
        elif before_exists and not after_exists:
            deleted.append((path, before_digest))
        else:
            changes.append(ChangedFile(path=path, change_type=ChangeType.MODIFIED))

    created_by_digest: dict[str, list[str]] = {}
    deleted_by_digest: dict[str, list[str]] = {}
    for path, digest in created:
        if digest:
            created_by_digest.setdefault(digest, []).append(path)
    for path, digest in deleted:
        if digest:
            deleted_by_digest.setdefault(digest, []).append(path)

    renamed_created: set[str] = set()
    renamed_deleted: set[str] = set()
    for digest in sorted(set(created_by_digest) & set(deleted_by_digest)):
        new_paths = created_by_digest[digest]
        old_paths = deleted_by_digest[digest]
        if len(new_paths) == 1 and len(old_paths) == 1:
            renamed_created.add(new_paths[0])
            renamed_deleted.add(old_paths[0])
            changes.append(ChangedFile(path=new_paths[0], old_path=old_paths[0], change_type=ChangeType.RENAMED))

    changes.extend(ChangedFile(path=path, change_type=ChangeType.CREATED) for path, _ in created if path not in renamed_created)
    changes.extend(ChangedFile(path=path, change_type=ChangeType.DELETED) for path, _ in deleted if path not in renamed_deleted)
    return sorted(changes, key=lambda item: (item.path, item.change_type.value, item.old_path or ""))


def _append_unique(values: list[str], value: Any) -> None:
    if isinstance(value, str) and value and value not in values:
        values.append(value)


def _append_range(values: list[ExposedLineRange], path: Any, start: Any, end: Any) -> None:
    try:
        candidate = ExposedLineRange(path=str(path), start_line=int(start), end_line=int(end))
    except (TypeError, ValueError):
        return
    if candidate not in values:
        values.append(candidate)


def _append_file_exposure(
    exposure: Exposure,
    path: Any,
    *,
    behavior: Literal["search", "read"],
    exposure_kind: Literal["path", "content"],
    source: Literal["dedicated_tool", "bash_rendered_path", "bash_single_operation"],
    confidence: Literal["high", "medium", "low"] = "high",
) -> None:
    if not isinstance(path, str) or not path:
        return
    candidate = FileExposure(
        path=path,
        behavior=behavior,
        exposure_kind=exposure_kind,
        source=source,
        confidence=confidence,
    )
    for index, current in enumerate(exposure.file_exposures):
        if current.path != candidate.path or current.behavior != candidate.behavior:
            continue
        if current.exposure_kind == "content" or candidate.exposure_kind == "path":
            return
        exposure.file_exposures[index] = candidate
        return
    exposure.file_exposures.append(candidate)


def _parse_json_observation(observation: str) -> dict[str, Any]:
    try:
        value = json.loads(observation)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _complete_visible_content_lines(payload: dict[str, Any]) -> list[str]:
    """Return complete head/tail lines without treating markers as content."""
    content = str(payload.get("content") or "")
    lines = content.splitlines()
    marker_indexes = {index for index, line in enumerate(lines) if line.startswith("... [truncated")}
    if not marker_indexes:
        return lines
    excluded = set(marker_indexes)
    for index in marker_indexes:
        # Head/tail clipping may cut the physical lines adjacent to the marker.
        excluded.update({index - 1, index + 1})
    return [line for index, line in enumerate(lines) if index not in excluded]


def extract_exposure(
    tool_name: str,
    action_subtype: str,
    observation: str,
    execution_status: ExecutionStatus,
) -> Exposure:
    """Derive navigation provenance from the final model-facing observation."""
    observation = str(observation or "")
    payload = _parse_json_observation(observation)
    truncated = bool(payload.get("truncated")) or "... [truncated" in observation or "<response clipped>" in observation
    exposure = Exposure(visible_characters=len(observation), output_truncated=truncated)
    if execution_status != ExecutionStatus.SUCCESS:
        return exposure

    if tool_name == "search":
        for path in payload.get("matched_files", []):
            _append_unique(exposure.files, path)
            _append_file_exposure(
                exposure,
                path,
                behavior="search",
                exposure_kind="path",
                source="dedicated_tool",
            )
        for snippet in payload.get("displayed_snippets", []):
            if not isinstance(snippet, dict):
                continue
            _append_unique(exposure.files, snippet.get("path"))
            _append_file_exposure(
                exposure,
                snippet.get("path"),
                behavior="search",
                exposure_kind="path",
                source="dedicated_tool",
            )
            _append_range(exposure.line_ranges, snippet.get("path"), snippet.get("start_line"), snippet.get("end_line"))
        return exposure

    if tool_name == "read_file" and action_subtype == "file":
        path = payload.get("path")
        _append_unique(exposure.files, path)
        visible_lines = [
            int(match.group(1))
            for match in re.finditer(
                r"(?m)^\s*(\d+)(?:\t|:)",
                str(payload.get("content") or ""),
            )
        ]
        if visible_lines:
            _append_range(exposure.line_ranges, path, min(visible_lines), max(visible_lines))
        _append_file_exposure(
            exposure,
            path,
            behavior="read",
            exposure_kind="content" if visible_lines else "path",
            source="dedicated_tool",
        )
        return exposure

    if tool_name == "read_file" and action_subtype == "status":
        for line in _complete_visible_content_lines(payload):
            if len(line) < 4:
                continue
            path = line[3:].strip().strip('"')
            if " -> " in path:
                old_path, new_path = path.split(" -> ", 1)
                _append_unique(exposure.files, old_path.strip('"'))
                _append_unique(exposure.files, new_path.strip('"'))
                for candidate in (old_path.strip('"'), new_path.strip('"')):
                    _append_file_exposure(
                        exposure,
                        candidate,
                        behavior="read",
                        exposure_kind="path",
                        source="dedicated_tool",
                    )
            else:
                _append_unique(exposure.files, path)
                _append_file_exposure(
                    exposure,
                    path,
                    behavior="read",
                    exposure_kind="path",
                    source="dedicated_tool",
                )
        return exposure

    if tool_name == "read_file" and action_subtype == "diff":
        current_path = ""
        old_path = ""
        for line in _complete_visible_content_lines(payload):
            if line.startswith("diff --git "):
                current_path = ""
                old_path = ""
                continue
            if line.startswith("--- a/"):
                old_path = line[6:].strip().strip('"')
                continue
            if line.startswith("+++ b/"):
                current_path = line[6:].strip().strip('"')
                _append_unique(exposure.files, current_path)
                _append_file_exposure(
                    exposure,
                    current_path,
                    behavior="read",
                    exposure_kind="path",
                    source="dedicated_tool",
                )
                continue
            if line.startswith("+++ /dev/null") and old_path:
                current_path = old_path
                _append_unique(exposure.files, current_path)
                _append_file_exposure(
                    exposure,
                    current_path,
                    behavior="read",
                    exposure_kind="path",
                    source="dedicated_tool",
                )
                continue
            match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if match and current_path:
                old_start, old_count, new_start, new_count = (int(value) if value is not None else None for value in match.groups())
                old_count = 1 if old_count is None else old_count
                new_count = 1 if new_count is None else new_count
                if new_count > 0:
                    _append_range(exposure.line_ranges, current_path, new_start, new_start + new_count - 1)
                elif old_count > 0:
                    _append_range(exposure.line_ranges, current_path, old_start, old_start + old_count - 1)
                _append_file_exposure(
                    exposure,
                    current_path,
                    behavior="read",
                    exposure_kind="content",
                    source="dedicated_tool",
                )
        return exposure

    if tool_name == "edit_file":
        for path in payload.get("touched_paths", []):
            _append_unique(exposure.files, path)
    return exposure


def is_probable_test_path(path: str) -> bool:
    normalized = str(path or "").replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    filename = parts[-1] if parts else ""
    return filename.startswith("test_") or filename.endswith("_test.py") or any(part in {"test", "Test", "tests", "Tests"} for part in parts)


__all__ = [
    "ACTION_EVENT_METADATA_KEY",
    "ACTION_EVENT_SCHEMA_VERSION",
    "ActionCategory",
    "ActionEvent",
    "ChangeType",
    "ChangedFile",
    "ExecutionStatus",
    "ExposedLineRange",
    "Exposure",
    "FileExposure",
    "ParseStatus",
    "RepoFileState",
    "RepoSnapshot",
    "RepoStateStatus",
    "RestoreConfirmation",
    "TestEvent",
    "TestEventContinuityStatus",
    "TestEventStatus",
    "TestPartitionCounts",
    "TestPartitionResult",
    "TestResultSource",
    "ValidationStatus",
    "action_category_for",
    "changed_files_between",
    "extract_exposure",
    "get_action_event",
    "is_probable_test_path",
    "resolve_test_event_continuity",
    "summarize_codeflow_bash_behavior",
]
