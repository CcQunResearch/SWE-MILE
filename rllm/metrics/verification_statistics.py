#!/usr/bin/env python3
"""Audit milestone verification potentials in saved Codeflow rollouts.

The audit has two deliberately separate views:

1. Strict rollout evidence quality (complete, partial, unavailable, or
   disabled), reported separately from step-level verification availability.
2. Arithmetic integrity. For every auditable trajectory, verification potential
   is recomputed from the rollout-local runtime baseline and each trusted
   TestEvent. Bug-repair tasks additionally use the authoritative target map;
   DeNovoSWE ``pass_count`` tasks use the all-passing runtime-baseline test set.
   Recomputed values are compared with every persisted MilestoneStepReward,
   the trajectory summary, and the compact top-level
   ``milestone_verification`` summary.

An ``unavailable`` rollout is therefore not called correct merely because its
metadata consistently says that verification was unavailable.  It is reported
as ``consistent_unavailable`` and kept separate from fully audited rollouts.

Usage::

    python -m rllm.metrics.verification_statistics ROLLOUT_DIR \
        --workers 32 [--mode train|val|all] [--output-file PATH] \
        [--cache-file PATH] [--no-cache] [--rebuild-cache]

The text report is saved next to ``ROLLOUT_DIR`` by default as
``<ROLLOUT_DIR>.codeflow_verification_potential.statistics``, matching the
location convention used by ``action_statistics.py``.  JSON files
are independent and are read concurrently; ``--workers 1`` selects a serial
scan. Per-file audit results are cached by default. RLLM rollout JSON files are
immutable after their atomic publication, so later invocations only parse new
files; ``--rebuild-cache`` is available for manually edited files. When at
least one selected rollout contains a non-empty ``language``
marker, a second report is written as
``<ROLLOUT_DIR>.codeflow_verification_potential.language.statistics``.  It
breaks verification availability down by language for every global step, for
all global steps combined, and by unique dataset task so repeated rollouts do
not amplify one task-level failure.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import io
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from glob import glob
from typing import Any

DEFAULT_WORKERS = 12
CACHE_SCHEMA_VERSION = 2
CACHE_STATS_VERSION = 20
CACHE_SUFFIX = ".codeflow_verification_potential.cache.json"
STATISTICS_SUFFIX = ".codeflow_verification_potential.statistics"
LANGUAGE_STATISTICS_SUFFIX = ".codeflow_verification_potential.language.statistics"
FLOAT_ABS_TOLERANCE = 1e-9
FLOAT_REL_TOLERANCE = 1e-9
EXAMPLE_LIMIT = 20

VERIFICATION_STATUSES = ("complete", "partial", "unavailable", "disabled", "<missing>")
AUDIT_VERDICTS = (
    "verified",
    "consistent_unavailable",
    "disabled",
    "inconsistent",
    "unauditable",
)

DENOVO_FILTERED_DISPOSITIONS = frozenset(
    {
        "compact_filtered",
        "dynamic_filtered",
        "reward_filtered",
        "uniform_filtered",
    }
)
DENOVO_REQUEUED_DISPOSITIONS = frozenset(
    {
        "infrastructure_requeue",
        "speculative_requeue",
        "late_cancelled_rollout",
    }
)
DENOVO_OPTIMIZER_DISPOSITIONS = frozenset(
    {
        "selected_for_optimizer",
        "trained",
        "optimizer_failed",
        "weight_sync_failed",
    }
)
DENOVO_SURPLUS_DISPOSITIONS = frozenset(
    {
        "surplus",
        "untrained_tail",
        "fatal_untrained_tail",
        "trainer_shutdown",
    }
)
DENOVO_CANCELLED_DISPOSITIONS = (
    DENOVO_FILTERED_DISPOSITIONS
    | DENOVO_REQUEUED_DISPOSITIONS
    | DENOVO_SURPLUS_DISPOSITIONS
)


@dataclass
class FileAudit:
    path: str
    mode: str
    step_idx: int
    task_id: str
    language: str | None
    verification_status: str
    verdict: str
    signal_participated: bool
    baseline_available: bool
    final_available: bool
    steps: int
    reward_records: int
    trusted_updates: int
    available_steps: int
    unavailable_reason: str
    root_unavailable_reason: str = "<none>"
    dynamic_sampling_status: str = "not_applicable"
    optimizer_committed: bool = False
    optimizer_step: int | None = None
    optimizer_consumed: bool = False
    rollout_idx: int = -1
    result_idx: int = -1
    repo_unchanged_steps: int = 0
    repo_unchanged_available_steps: int = 0
    repo_changed_steps: int = 0
    repo_changed_available_steps: int = 0
    repo_changed_unavailable_reasons: Counter[str] = field(default_factory=Counter)
    baseline_unavailable_reasons: Counter[str] = field(default_factory=Counter)
    repo_change_unknown_steps: int = 0
    repo_change_unknown_available_steps: int = 0
    dataset_task_id: str = "<missing>"
    baseline_probe_succeeded: bool = False
    materialized_fallback: bool = False
    safe_baseline_available: bool = False
    benign_baseline_partial: bool = False
    actionable_baseline_failure: bool = False
    baseline_source: str = "<none>"
    baseline_collector: str = "<none>"
    count_plan_hash: str = "<none>"
    count_collectors: Counter[str] = field(default_factory=Counter)
    count_runners: Counter[str] = field(default_factory=Counter)
    count_partial_reasons: Counter[str] = field(default_factory=Counter)
    expected_test_count: int = 0
    materialized_baseline_passed: int = 0
    runtime_baseline_passed: int | None = None
    baseline_passed: int | None = None
    baseline_count_mismatch: int = 0
    verification_potential_mode: str = "bug_repair"
    replays_skipped_no_repo_change: int = 0
    partition_adapter: str = "<none>"
    partition_calibration_reason: str = "<none>"
    partition_calibration_stage: str = "<none>"
    result_completeness: Counter[str] = field(default_factory=Counter)
    observed_test_results: int = 0
    imputed_test_results: int = 0
    not_run_test_results: int = 0
    skipped_test_results: int = 0
    f2p_observed: int = 0
    f2p_skipped: int = 0
    f2p_not_run: int = 0
    p2p_observed: int = 0
    p2p_skipped: int = 0
    p2p_not_run: int = 0
    verification_result_modes: Counter[str] = field(default_factory=Counter)
    partition_calibration_status: str = "not_attempted"
    failure_stages: Counter[str] = field(default_factory=Counter)
    failure_origins: Counter[str] = field(default_factory=Counter)
    failure_contexts: Counter[str] = field(default_factory=Counter)
    contract_assisted_events: int = 0
    contract_assisted_tests: int = 0
    partition_group_calls: int = 0
    partition_group_successes: int = 0
    partition_calibration_attempts: int = 0
    partition_calibration_successes: int = 0
    partition_timeout_recovery_attempts: int = 0
    partition_timeout_recovery_successes: int = 0
    partition_timeout_recovery_partial: int = 0
    partition_timeout_recovery_failures: int = 0
    denovo_background: bool = False
    primary_verifier_status: str = "not_applicable"
    training_disposition: str = "not_applicable"
    shadow_finalize_disposition: str = "not_applicable"
    batch_barrier_started: bool = False
    batch_shadow_timed_out: bool = False
    batch_shadow_timeout_steps: int = 0
    shadow_completed_before_primary: bool = False
    shadow_completed_before_barrier: bool = False
    completed_probes_at_barrier: int = 0
    pending_probes_at_barrier: int = 0
    completed_probes_before_primary: int = 0
    completed_probes_before_barrier: int = 0
    completed_probes_during_barrier: int = 0
    completed_probes_at_disposition: int = 0
    completed_probes_after_primary_before_disposition: int = 0
    cancelled_pending_steps: int = 0
    primary_verifier_duration_s: float = 0.0
    shadow_background_head_start_s: float = 0.0
    shadow_background_lifetime_s: float = 0.0
    shadow_post_primary_overlap_s: float = 0.0
    batch_barrier_wait_s: float = 0.0
    batch_barrier_budget_remaining_s: float = 0.0
    shadow_resource_cpus: int = 0
    shadow_resource_memory_mb: int = 0
    shadow_resource_difficulty_bucket: str = "not_applicable"
    shadow_queue_depth_at_handoff: int = 0
    shadow_queue_depth_at_barrier_start: int = 0
    shadow_max_queue_depth: int = 0
    shadow_peak_active_sandboxes: int = 0
    shadow_peak_active_cpus: int = 0
    shadow_peak_active_memory_mb: int = 0
    shadow_limit_sandboxes: int = 0
    shadow_limit_cpus: int = 0
    shadow_limit_memory_mb: int = 0
    probe_merge_enabled: bool = False
    probe_merge_max_steps: int = 0
    probe_merge_candidate_steps: int = 0
    probe_merge_physical_probes: int = 0
    probe_merge_groups: int = 0
    probe_merge_steps: int = 0
    probe_merge_saved_probes: int = 0
    probe_merge_group_size_histogram: Counter[str] = field(
        default_factory=Counter
    )
    probe_merge_same_changed_files_run_length_histogram: Counter[str] = field(
        default_factory=Counter
    )
    probe_merge_same_changed_files_runs: int = 0
    probe_merge_same_changed_files_steps: int = 0
    probe_merge_decision_wait_samples: int = 0
    probe_merge_decision_wait_s_min: float | None = None
    probe_merge_decision_wait_s_mean: float | None = None
    probe_merge_decision_wait_s_max: float | None = None
    cancelled_group_rollouts: int = 0
    cancelled_materialized_rollouts: int = 0
    cancelled_shadow_leases: int = 0
    issues: Counter[str] = field(default_factory=Counter)
    blockers: Counter[str] = field(default_factory=Counter)
    diagnostics: Counter[str] = field(default_factory=Counter)


@dataclass
class ScanResult:
    audits: list[FileAudit]
    files_found: int
    malformed: int
    malformed_examples: list[str]
    mode_counts: Counter[str]
    cache: CacheDiagnostics | None = None


@dataclass
class CacheDiagnostics:
    enabled: bool
    path: str | None = None
    fast_path: bool = False
    hits: int = 0
    misses: int = 0
    removed: int = 0
    directory_changed_during_scan: bool = False
    load_error: str | None = None
    write_error: str | None = None


def _normalize_reason(value: Any, default: str = "<none>") -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return default
    return text if len(text) <= 180 else text[:177] + "..."


def _root_unavailable_reason(
    shadow_summary: dict[str, Any],
    baseline_event: dict[str, Any] | None,
    fallback: str,
) -> str:
    """Return one causal incident per rollout, not every affected step."""

    terminal_replay = shadow_summary.get("terminal_replay")
    if isinstance(terminal_replay, dict):
        reason = _normalize_reason(terminal_replay.get("failure_type"))
        if reason != "<none>":
            return reason
    if isinstance(baseline_event, dict) and baseline_event.get("trusted") is not True:
        reason = _normalize_reason(
            baseline_event.get("failure_type") or baseline_event.get("error")
        )
        if reason != "<none>":
            return reason
    failure_counts = shadow_summary.get("failure_counts")
    if isinstance(failure_counts, dict):
        for reason, count in failure_counts.items():
            if (
                isinstance(reason, str)
                and reason
                and isinstance(count, int)
                and not isinstance(count, bool)
                and count > 0
                and reason != "suppressed_after_terminal"
            ):
                return reason
    return _normalize_reason(fallback)


def _mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        return dict(parsed) if isinstance(parsed, dict) else None
    return None


def _status_map(value: Any) -> dict[str, str] | None:
    parsed = _mapping(value)
    if not parsed:
        return None
    if not all(isinstance(name, str) and bool(name) and isinstance(status, str) and bool(status) for name, status in parsed.items()):
        return None
    return {str(name): str(status) for name, status in parsed.items()}


def _bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _first_finite(*values: Any) -> float | None:
    for value in values:
        parsed = _finite_float(value)
        if parsed is not None:
            return parsed
    return None


def _float_equal(left: Any, right: float) -> bool:
    parsed = _finite_float(left)
    return parsed is not None and math.isclose(
        parsed,
        right,
        rel_tol=FLOAT_REL_TOLERANCE,
        abs_tol=FLOAT_ABS_TOLERANCE,
    )


def _step_metadata(step: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = step.get("metadata")
    if not isinstance(metadata, dict):
        return {}, {}
    nested = metadata.get("agent_step_metadata")
    agent_metadata = nested if isinstance(nested, dict) else metadata
    return metadata, agent_metadata


def _action_event(step: dict[str, Any]) -> dict[str, Any] | None:
    _, metadata = _step_metadata(step)
    event = metadata.get("action_event")
    return event if isinstance(event, dict) else None


def _reward_record(step: dict[str, Any]) -> dict[str, Any] | None:
    outer, nested = _step_metadata(step)
    raw = outer.get("milestone_reward")
    if raw is None:
        raw = nested.get("milestone_reward")
    return raw if isinstance(raw, dict) else None


def _trajectory_metadata(document: dict[str, Any]) -> dict[str, Any]:
    trajectory = document.get("trajectory")
    if not isinstance(trajectory, dict):
        return {}
    metadata = trajectory.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _language_marker(document: dict[str, Any]) -> str | None:
    """Return the normalized language from known rollout schema locations."""

    task = document.get("task")
    task = task if isinstance(task, dict) else {}
    task_metadata = task.get("metadata")
    task_metadata = task_metadata if isinstance(task_metadata, dict) else {}
    metrics = document.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    metadata = document.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    shadow = metadata.get("shadow_sandbox")
    shadow = shadow if isinstance(shadow, dict) else {}
    trajectory_metadata = _trajectory_metadata(document)
    milestone_context = trajectory_metadata.get("milestone_context")
    milestone_context = milestone_context if isinstance(milestone_context, dict) else {}
    milestone_task_metadata = milestone_context.get("task_metadata")
    milestone_task_metadata = milestone_task_metadata if isinstance(milestone_task_metadata, dict) else {}

    # task.language is the canonical field in current rollout schema.  The
    # remaining locations retain compatibility with older and compact records.
    candidates = (
        task.get("language"),
        task_metadata.get("language"),
        document.get("language"),
        metrics.get("language"),
        shadow.get("language"),
        milestone_task_metadata.get("language"),
    )
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return " ".join(value.split()).casefold()
    task_rllm = task.get("rllm")
    task_rllm = task_rllm if isinstance(task_rllm, dict) else {}
    profile = str(
        task.get("task_profile")
        or task_metadata.get("task_profile")
        or task_rllm.get("task_profile")
        or milestone_task_metadata.get("task_profile")
        or ""
    )
    if (
        profile == "repo_generation_denovoswe"
        or shadow.get("result_parser") == "denovoswe_official_v1"
    ):
        return "python"
    return None


def _test_event(step: dict[str, Any]) -> dict[str, Any] | None:
    event = _action_event(step)
    if event is None:
        return None
    test_event = event.get("test_event")
    return test_event if isinstance(test_event, dict) else None


def _partition_counts(event: dict[str, Any]) -> dict[str, dict[str, int]] | None:
    raw = event.get("partition_counts")
    if not isinstance(raw, dict):
        return None
    parsed: dict[str, dict[str, int]] = {}
    for partition in ("f2p", "p2p"):
        group = raw.get(partition)
        if not isinstance(group, dict):
            return None
        values: dict[str, int] = {}
        for key in ("expected", "passed", "failed", "errored", "skipped", "not_run"):
            value = group.get(key, 0)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                return None
            values[key] = value
        if sum(values[key] for key in ("passed", "failed", "errored", "skipped", "not_run")) != values["expected"]:
            return None
        parsed[partition] = values
    return parsed


def _count_observation(
    event: dict[str, Any],
    expected: int | None = None,
) -> dict[str, Any] | None:
    if (
        event.get("status") != "completed"
        or event.get("trusted") is not True
        or event.get("result_source") != "count_collector"
    ):
        return None
    raw = event.get("count_observation")
    if (
        not isinstance(raw, dict)
        or raw.get("schema_version") != 4
        or raw.get("scope") != "materialized_acceptance_contract"
        or not isinstance(raw.get("collector"), str)
        or not raw.get("collector")
        or isinstance(raw.get("collector_version"), bool)
        or not isinstance(raw.get("collector_version"), int)
        or raw.get("collector_version") < 1
        or isinstance(raw.get("command_count"), bool)
        or not isinstance(raw.get("command_count"), int)
        or raw.get("command_count") < 1
        or not isinstance(raw.get("command_evidence"), list)
    ):
        return None
    ownership_hash = raw.get("ownership_sha256")
    native_digest = raw.get("native_state_sha256")
    if (
        not isinstance(raw.get("plan_hash"), str)
        or len(raw.get("plan_hash")) != 64
        or any(
            character not in "0123456789abcdef"
            for character in raw.get("plan_hash")
        )
        or raw.get("termination")
        not in {"complete", "interrupted", "deterministic_abort"}
        or not isinstance(raw.get("ignored_tests"), list)
        or not isinstance(raw.get("ownership_evidence"), list)
        or not isinstance(raw.get("ownership_observed"), int)
        or isinstance(raw.get("ownership_observed"), bool)
        or raw.get("ownership_observed") != raw.get("reported")
        or not isinstance(ownership_hash, str)
        or len(ownership_hash) != 64
        or any(character not in "0123456789abcdef" for character in ownership_hash)
        or len(raw.get("ownership_evidence")) > 50
        or not isinstance(native_digest, str)
        or len(native_digest) != 64
        or any(character not in "0123456789abcdef" for character in native_digest)
    ):
        return None
    complete = bool(raw.get("complete"))
    if (
        (complete and raw.get("termination") != "complete")
        or (not complete and raw.get("termination") == "complete")
        or (not complete and not isinstance(raw.get("partial_reason"), str))
    ):
            return None
    integer_fields = (
        "expected",
        "passed",
        "failed",
        "errored",
        "skipped",
        "unclassified",
        "reported",
        "not_run",
    )
    values: dict[str, int] = {}
    for key in integer_fields:
        value = raw.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        values[key] = value
    if expected is not None and values["expected"] != expected:
        return None
    if (
        sum(
            values[key]
            for key in ("passed", "failed", "errored", "skipped", "unclassified")
        )
        != values["reported"]
        or values["reported"] + values["not_run"] != values["expected"]
        or bool(raw.get("complete")) != (values["not_run"] == 0)
        or event.get("result_completeness")
        != ("count_full" if values["not_run"] == 0 else "count_partial")
    ):
        return None
    raw_partitions = raw.get("partition_counts")
    if raw_partitions is not None:
        partitions = _partition_counts(
            {"partition_counts": raw_partitions}
        )
        if partitions is None or values["unclassified"] != 0:
            return None
        groups = (partitions["f2p"], partitions["p2p"])
        if (
            sum(group["expected"] for group in groups)
            != values["expected"]
            or sum(group["passed"] for group in groups)
            != values["passed"]
            or sum(group["failed"] for group in groups)
            != values["failed"]
            or sum(group["errored"] for group in groups)
            != values["errored"]
            or sum(group["skipped"] for group in groups)
            != values["skipped"]
            or sum(group["not_run"] for group in groups)
            != values["not_run"]
        ):
            return None
    return {**raw, **values}


def _effective_count_passed(
    observation: dict[str, Any],
    state: dict[str, Any],
) -> int:
    raw_partitions = observation.get("partition_counts")
    partitions = (
        _partition_counts({"partition_counts": raw_partitions})
        if isinstance(raw_partitions, dict)
        else None
    )
    if observation.get("complete") is True or partitions is None:
        if observation.get("complete") is True and partitions is not None:
            state["count_confirmed_p2p"] = (
                partitions["p2p"]["failed"]
                + partitions["p2p"]["errored"]
                + partitions["p2p"]["skipped"]
            )
        elif observation.get("complete") is True:
            state["count_confirmed_p2p"] = max(
                int(state.get("count_confirmed_p2p", 0)),
                int(state.get("count_p2p_expected", 0))
                - int(observation["passed"]),
            )
        return int(observation["passed"])
    explicit = (
        partitions["p2p"]["failed"]
        + partitions["p2p"]["errored"]
        + partitions["p2p"]["skipped"]
    )
    if partitions["p2p"]["not_run"] == 0:
        state["count_confirmed_p2p"] = explicit
    else:
        state["count_confirmed_p2p"] = max(
            int(state.get("count_confirmed_p2p", 0)),
            explicit,
        )
    return partitions["f2p"]["passed"] + max(
        0,
        partitions["p2p"]["expected"]
        - int(state["count_confirmed_p2p"]),
    )


def _trusted_test_event(
    event: dict[str, Any],
    target: dict[str, str],
    baseline: dict[str, str],
) -> bool:
    common = bool(event.get("status") == "completed" and event.get("trusted") is True and not event.get("missing_tests") and not event.get("extra_tests"))
    if not common:
        return False
    if event.get("result_source") == "partition_counts":
        partitions = _partition_counts(event)
        if partitions is None:
            return False
        f2p = sum(baseline[name] != target[name] for name in target)
        return partitions["f2p"]["expected"] == f2p and partitions["p2p"]["expected"] == len(target) - f2p
    if event.get("result_source") == "count_collector":
        return _count_observation(event, len(target)) is not None
    results = _status_map(event.get("test_results"))
    return results is not None and set(results) == set(target)


def _result_event_diagnostics(
    event: dict[str, Any] | None,
) -> tuple[str | None, int, int, int, int, dict[str, int], str | None]:
    if not isinstance(event, dict):
        return None, 0, 0, 0, 0, {}, None
    results = _status_map(event.get("test_results")) or {}
    raw_imputed = event.get("imputed_tests")
    imputed = {str(name) for name in raw_imputed if isinstance(raw_imputed, list) and isinstance(name, str) and name} if isinstance(raw_imputed, list) else set()
    completeness = event.get("result_completeness")
    if completeness not in {
        "full",
        "baseline_contract_filled",
        "controlled_partial",
        "contract_assisted",
        "partition_counts_full",
        "partition_counts_partial",
        "count_full",
        "count_partial",
        "none",
    }:
        completeness = "full" if event.get("status") == "completed" and event.get("trusted") is True and not event.get("missing_tests") and results else "none"
    partition_metrics: dict[str, int] = {}
    partitions = _partition_counts(event)
    count = _count_observation(event)
    if count is not None:
        return (
            str(completeness),
            count["reported"],
            0,
            count["not_run"],
            count["skipped"],
            {},
            "aggregate_counts",
        )
    if partitions is not None:
        for partition, group in partitions.items():
            partition_metrics[f"{partition}_observed"] = group["passed"] + group["failed"] + group["errored"]
            partition_metrics[f"{partition}_skipped"] = group["skipped"]
            partition_metrics[f"{partition}_not_run"] = group["not_run"]
    return (
        str(completeness),
        max(0, len(results) - len(imputed & set(results))),
        len(imputed & set(results)),
        sum(status == "NOT_RUN" for status in results.values()),
        sum(status == "SKIPPED" for status in results.values()),
        partition_metrics,
        "partition_counts"
        if partitions is not None
        else (
            "contract_assisted"
            if completeness == "contract_assisted"
            else "contract_filled"
            if completeness == "baseline_contract_filled"
            else "per_test"
        ),
    )


def _continuity_lost(event: dict[str, Any]) -> bool:
    explicit = event.get("continuity_status")
    if explicit is not None:
        return explicit == "lost"
    return not (event.get("status") == "completed" and event.get("trusted") is True)


def _verification_potential(
    results: dict[str, str],
    baseline: dict[str, str],
    target: dict[str, str],
    *,
    beta_any: float,
    beta_frac: float,
    mode: str = "bug_repair",
    policy: str = "bug_repair_v2_neutral_unrun",
    state: dict[str, Any] | None = None,
    partitions: dict[str, dict[str, int]] | None = None,
) -> float:
    if mode == "pass_count":
        current_count = sum(results.get(name) == target[name] for name in target)
        baseline_count = sum(baseline.get(name) == target[name] for name in target)
        return float(current_count - baseline_count)
    failing = [name for name in target if baseline[name] != target[name]]
    stable = [name for name in target if baseline[name] == target[name]]
    if policy == "bug_repair_v2_neutral_unrun":
        assert state is not None
        if partitions is not None:
            f2p = partitions["f2p"]
            p2p = partitions["p2p"]
            explicit = p2p["failed"] + p2p["errored"]
            if p2p["skipped"] == 0 and p2p["not_run"] == 0:
                state["confirmed_count"] = explicit
            else:
                state["confirmed_count"] = max(int(state.get("confirmed_count", 0)), explicit)
            fixed = f2p["passed"]
            regressed = int(state["confirmed_count"])
        else:
            confirmed = state.setdefault("confirmed_by_test", {name: False for name in stable})
            fixed = sum(str(results.get(name, "NOT_RUN")).upper() == "PASSED" for name in failing)
            for name in stable:
                status = str(results.get(name, "NOT_RUN")).upper()
                if status == "PASSED":
                    confirmed[name] = False
                elif status in {"FAILED", "ERROR", "ERRORS", "ERRORED"}:
                    confirmed[name] = True
            regressed = sum(bool(confirmed.get(name)) for name in stable)
            state["confirmed_count"] = regressed
        fixed_fraction = fixed / len(failing) if failing else 0.0
        regressed_fraction = regressed / len(stable) if stable else 0.0
        return fixed_fraction - beta_any * float(regressed_fraction > 0) - beta_frac * regressed_fraction
    raise ValueError(f"Unsupported verification potential policy: {policy}")


def _compare_bool(issues: Counter[str], record: dict[str, Any], key: str, expected: bool) -> None:
    if _bool(record.get(key)) is not expected:
        issues[f"step_{key}_mismatch"] += 1


def _compare_float(issues: Counter[str], record: dict[str, Any], key: str, expected: float) -> None:
    if not _float_equal(record.get(key), expected):
        issues[f"step_{key}_mismatch"] += 1


def _audit_document(path: str, document: dict[str, Any]) -> FileAudit:
    mode = str(document.get("mode") or "unknown")
    raw_step = document.get("global_step")
    lifecycle = document.get("rollout_lifecycle")
    lifecycle = lifecycle if isinstance(lifecycle, dict) else {}
    if raw_step is None and isinstance(lifecycle, dict):
        raw_step = lifecycle.get("dispatch_step")
    try:
        step_idx = int(raw_step)
    except (TypeError, ValueError):
        step_idx = -1
    try:
        rollout_idx = int(document.get("rollout_idx"))
    except (TypeError, ValueError):
        rollout_idx = -1
    try:
        result_idx = int(document.get("result_idx"))
    except (TypeError, ValueError):
        result_idx = -1

    dynamic_sampling_status = str(
        document.get("dynamic_sampling_status") or "not_applicable"
    )
    optimizer_committed = lifecycle.get("optimizer_committed") is True
    raw_optimizer_step = lifecycle.get("optimizer_step")
    try:
        optimizer_step = (
            int(raw_optimizer_step) if raw_optimizer_step is not None else None
        )
    except (TypeError, ValueError):
        optimizer_step = None
    lifecycle_training_disposition = str(
        lifecycle.get("training_disposition") or "not_applicable"
    )
    optimizer_consumed = bool(
        optimizer_committed
        and lifecycle_training_disposition == "trained"
        and dynamic_sampling_status == "consumed"
    )

    task_id = _normalize_reason(document.get("task_id"), "<missing>")
    task = document.get("task")
    task = task if isinstance(task, dict) else {}
    task_metadata_root = task.get("metadata")
    task_metadata_root = task_metadata_root if isinstance(task_metadata_root, dict) else {}
    dataset_task_id = _normalize_reason(
        task.get("id") or task_metadata_root.get("task_id"),
        "<missing>",
    )
    language = _language_marker(document)
    compact = document.get("milestone_verification")
    compact = compact if isinstance(compact, dict) else {}
    verification_status = str(compact.get("status") or "<missing>")
    unavailable_reason = _normalize_reason(compact.get("unavailable_reason"))
    issues: Counter[str] = Counter()
    blockers: Counter[str] = Counter()
    diagnostics: Counter[str] = Counter()

    trajectory = document.get("trajectory")
    if not isinstance(trajectory, dict):
        blockers["missing_trajectory"] += 1
        return FileAudit(
            path=path,
            mode=mode,
            step_idx=step_idx,
            task_id=task_id,
            language=language,
            verification_status=verification_status,
            verdict="unauditable",
            signal_participated=False,
            baseline_available=False,
            final_available=False,
            steps=0,
            reward_records=0,
            trusted_updates=0,
            available_steps=0,
            unavailable_reason=unavailable_reason,
            dynamic_sampling_status=dynamic_sampling_status,
            optimizer_committed=optimizer_committed,
            optimizer_step=optimizer_step,
            optimizer_consumed=optimizer_consumed,
            rollout_idx=rollout_idx,
            result_idx=result_idx,
            dataset_task_id=dataset_task_id,
            issues=issues,
            blockers=blockers,
            diagnostics=diagnostics,
        )
    steps = trajectory.get("steps")
    if not isinstance(steps, list) or not all(isinstance(step, dict) for step in steps):
        blockers["invalid_trajectory_steps"] += 1
        steps = []

    trajectory_metadata = _trajectory_metadata(document)
    reward_summary = trajectory_metadata.get("milestone_reward")
    reward_summary = reward_summary if isinstance(reward_summary, dict) else {}
    rollout_metadata = document.get("metadata")
    rollout_metadata = rollout_metadata if isinstance(rollout_metadata, dict) else {}
    shadow_summary = rollout_metadata.get("shadow_sandbox")
    shadow_summary = shadow_summary if isinstance(shadow_summary, dict) else {}
    config = reward_summary.get("config")
    config = config if isinstance(config, dict) else None
    if config is None:
        blockers["missing_milestone_reward_config"] += 1
        enabled = False
        verification_enabled = False
        beta_any = 0.0
        beta_frac = 0.0
        verification_weight = 0.0
        navigation_weight = 0.0
        clip_lower = -6.0
        clip_upper = 3.0
        backward_lambda = 0.0
        backward_gamma = 0.0
        format_weight = 0.0
    else:
        enabled = config.get("enable") is True
        verification_enabled = enabled and config.get("verification_enable") is True
        parsed_beta_any = _finite_float(config.get("beta_any"))
        parsed_beta_frac = _finite_float(config.get("beta_frac"))
        if parsed_beta_any is None or parsed_beta_any < 0:
            blockers["invalid_beta_any"] += 1
            beta_any = 0.0
        else:
            beta_any = parsed_beta_any
        if parsed_beta_frac is None or parsed_beta_frac < 0:
            blockers["invalid_beta_frac"] += 1
            beta_frac = 0.0
        else:
            beta_frac = parsed_beta_frac
        scalar_config = {
            "verification_weight": 0.25,
            "navigation_weight": 0.05,
            "verification_reward_clip_lower": -6.0,
            "verification_reward_clip_upper": 3.0,
            "backward_credit_lambda": 0.2,
            "backward_credit_gamma": 0.9,
            "format_weight": 0.1,
        }
        parsed_scalars: dict[str, float] = {}
        for key, default in scalar_config.items():
            parsed = _finite_float(config.get(key, default))
            if parsed is None:
                blockers[f"invalid_{key}"] += 1
                parsed = default
            parsed_scalars[key] = parsed
        verification_weight = parsed_scalars["verification_weight"]
        navigation_weight = parsed_scalars["navigation_weight"]
        clip_lower = parsed_scalars["verification_reward_clip_lower"]
        clip_upper = parsed_scalars["verification_reward_clip_upper"]
        backward_lambda = parsed_scalars["backward_credit_lambda"]
        backward_gamma = parsed_scalars["backward_credit_gamma"]
        format_weight = parsed_scalars["format_weight"]
        if clip_lower > 0 or clip_upper < 0 or clip_lower > clip_upper:
            blockers["invalid_verification_reward_clip"] += 1

    context = trajectory_metadata.get("milestone_context")
    context = context if isinstance(context, dict) else None
    task_metadata: dict[str, Any] = {}
    baseline_event: dict[str, Any] | None = None
    target: dict[str, str] | None = None
    baseline: dict[str, str] | None = None
    count_contract: dict[str, Any] | None = None
    expected_test_count = 0
    materialized_baseline_passed = 0
    runtime_baseline_passed: int | None = None
    baseline_passed: int | None = None
    baseline_probe_succeeded = False
    materialized_fallback = False
    baseline_source = "<none>"
    baseline_collector = "<none>"
    verification_reason: str | None = None
    runtime_required = False
    verification_mode = (
        "pass_count"
        if shadow_summary.get("result_parser") == "denovoswe_official_v1"
        else "bug_repair"
    )
    verification_policy = "bug_repair_v2_neutral_unrun"
    if context is None:
        if verification_enabled:
            blockers["missing_milestone_context"] += 1
    else:
        raw_policy = context.get("verification_potential_policy")
        if raw_policy in {
            "bug_repair_v2_neutral_unrun",
            "bug_repair_pass_count_v1",
            "repo_generation_pass_count_v1",
        }:
            verification_policy = str(raw_policy)
        elif isinstance(context.get("schema_version"), int) and int(context["schema_version"]) >= 3:
            verification_policy = "bug_repair_v2_neutral_unrun"
        raw_task_metadata = context.get("task_metadata")
        if isinstance(raw_task_metadata, dict):
            task_metadata = raw_task_metadata
        elif verification_enabled:
            blockers["missing_task_metadata"] += 1
        raw_verification_mode = (
            task_metadata.get("verification_potential_mode")
            or context.get("verification_potential_mode")
            or reward_summary.get("verification_potential_mode")
            or (
                "pass_count"
                if shadow_summary.get("result_parser") == "denovoswe_official_v1"
                else "bug_repair"
            )
        )
        if str(raw_verification_mode).strip() in {"pass_count"}:
            verification_mode = "pass_count"
            blockers.pop("invalid_beta_any", None)
            blockers.pop("invalid_beta_frac", None)
        elif str(raw_verification_mode).strip() in {"", "bug_repair"}:
            verification_mode = "bug_repair"
        elif str(raw_verification_mode).strip() in {
            "bug_repair_pass_count",
        }:
            verification_mode = "bug_repair_pass_count"
            verification_policy = "bug_repair_pass_count_v1"
            blockers.pop("invalid_beta_any", None)
            blockers.pop("invalid_beta_frac", None)
        elif verification_enabled:
            blockers["unsupported_verification_potential_mode"] += 1
        shadow_schema_version = context.get("shadow_schema_version")
        runtime_required = bool(isinstance(context.get("schema_version"), int) and int(context["schema_version"]) >= 2 and isinstance(shadow_schema_version, int) and shadow_schema_version >= 4)
        raw_baseline = context.get("shadow_baseline_test_event")
        baseline_event = raw_baseline if isinstance(raw_baseline, dict) else None
        if verification_mode == "pass_count":
            if baseline_event is None:
                verification_reason = _normalize_reason(
                    context.get("shadow_verification_unavailable_reason"),
                    "shadow_baseline_unavailable",
                )
            elif baseline_event.get("status") != "completed" or baseline_event.get("trusted") is not True:
                verification_reason = _normalize_reason(
                    baseline_event.get("failure_type") or baseline_event.get("error"),
                    "shadow_baseline_untrusted",
                )
            else:
                observed = _status_map(baseline_event.get("test_results"))
                if observed is None:
                    verification_reason = "shadow_baseline_results_invalid"
                else:
                    baseline = observed
                    target = dict.fromkeys(baseline, "PASSED")
        else:
            target = _status_map(task_metadata.get("target_output_json"))
            materialized_baseline = _status_map(task_metadata.get("baseline_output_json"))
            if verification_enabled and target is None:
                blockers["invalid_target_output_json"] += 1
        if verification_mode == "bug_repair_pass_count":
            raw_contract = context.get("verification_count_contract")
            if not isinstance(raw_contract, dict):
                raw_contract = shadow_summary.get("verification_count_contract")
            if (
                target is None
                or materialized_baseline is None
                or set(materialized_baseline) != set(target)
            ):
                verification_reason = "invalid_verification_metadata"
            elif not isinstance(raw_contract, dict) or raw_contract.get("schema_version") != 1:
                verification_reason = "verification_count_contract_unavailable"
            else:
                expected_test_count = len(target)
                materialized_baseline_passed = sum(
                    str(materialized_baseline.get(name, "NOT_RUN")).upper()
                    == "PASSED"
                    for name in target
                )
                raw_expected = raw_contract.get("expected")
                raw_materialized = raw_contract.get(
                    "materialized_baseline_passed"
                )
                raw_runtime = raw_contract.get("runtime_baseline_passed")
                raw_baseline_passed = raw_contract.get("baseline_passed")
                source = str(raw_contract.get("baseline_source") or "")
                probe_succeeded = raw_contract.get("baseline_probe_succeeded")
                valid_runtime = raw_runtime is None or (
                    isinstance(raw_runtime, int)
                    and not isinstance(raw_runtime, bool)
                    and 0 <= raw_runtime <= expected_test_count
                )
                contract_valid = bool(
                    raw_expected == expected_test_count
                    and raw_materialized == materialized_baseline_passed
                    and valid_runtime
                    and isinstance(raw_baseline_passed, int)
                    and not isinstance(raw_baseline_passed, bool)
                    and 0 <= raw_baseline_passed <= expected_test_count
                    and raw_contract.get("max_additional_passed")
                    == expected_test_count - raw_baseline_passed
                    and source in {"runtime_probe", "materialized_fallback"}
                    and isinstance(probe_succeeded, bool)
                    and (
                        source == "runtime_probe"
                        and probe_succeeded
                        and raw_runtime == raw_baseline_passed
                        or source == "materialized_fallback"
                        and not probe_succeeded
                        and raw_baseline_passed == materialized_baseline_passed
                    )
                )
                if not contract_valid:
                    verification_reason = "verification_count_contract_mismatch"
                else:
                    count_contract = dict(raw_contract)
                    baseline = materialized_baseline
                    baseline_passed = int(raw_baseline_passed)
                    runtime_baseline_passed = (
                        int(raw_runtime) if isinstance(raw_runtime, int) else None
                    )
                    baseline_probe_succeeded = bool(probe_succeeded)
                    materialized_fallback = source == "materialized_fallback"
                    baseline_source = source
                    baseline_collector = str(
                        raw_contract.get("collector") or "<none>"
                    )
        if verification_mode == "bug_repair" and target is not None and baseline_event is not None:
            observed = _status_map(baseline_event.get("test_results"))
            if baseline_event.get("status") != "completed" or baseline_event.get("trusted") is not True:
                verification_reason = _normalize_reason(
                    baseline_event.get("failure_type") or baseline_event.get("error"),
                    "shadow_baseline_untrusted",
                )
            elif observed is None:
                verification_reason = "shadow_baseline_results_invalid"
            elif set(observed) != set(target):
                verification_reason = "shadow_baseline_test_set_mismatch"
            else:
                baseline = observed
        elif verification_mode == "bug_repair" and target is not None and runtime_required:
            verification_reason = _normalize_reason(
                context.get("shadow_verification_unavailable_reason"),
                "shadow_baseline_unavailable",
            )
        elif verification_mode == "bug_repair" and target is not None:
            if materialized_baseline is None:
                verification_reason = "invalid_verification_metadata"
            elif set(materialized_baseline) != set(target):
                verification_reason = "baseline_target_test_set_mismatch"
            else:
                baseline = materialized_baseline

    if verification_mode == "pass_count" and baseline is not None:
        expected_test_count = len(target or baseline)
        raw_materialized_baseline = shadow_summary.get(
            "materialized_baseline_passed"
        )
        if not isinstance(raw_materialized_baseline, int) or isinstance(
            raw_materialized_baseline,
            bool,
        ):
            raw_materialized_baseline = compact.get(
                "materialized_baseline_passed"
            )
        materialized_baseline_passed = (
            int(raw_materialized_baseline)
            if isinstance(raw_materialized_baseline, int)
            and not isinstance(raw_materialized_baseline, bool)
            else 0
        )
        raw_runtime_baseline = shadow_summary.get("runtime_baseline_passed")
        runtime_baseline_passed = (
            int(raw_runtime_baseline)
            if isinstance(raw_runtime_baseline, int)
            and not isinstance(raw_runtime_baseline, bool)
            else sum(value == "PASSED" for value in baseline.values())
        )
        raw_baseline_passed = shadow_summary.get("baseline_passed")
        baseline_passed = (
            int(raw_baseline_passed)
            if isinstance(raw_baseline_passed, int)
            and not isinstance(raw_baseline_passed, bool)
            else runtime_baseline_passed
        )
        baseline_probe_succeeded = bool(
            baseline_event is not None
            and baseline_event.get("status") == "completed"
            and baseline_event.get("trusted") is True
        )
        baseline_source = (
            str(shadow_summary.get("baseline_source") or "runtime_probe")
            if baseline_probe_succeeded
            else "<none>"
        )
        baseline_collector = str(
            shadow_summary.get("baseline_collector")
            or "official_exact_vector"
        )

    baseline_summary_available = bool(
        count_contract is not None
        if verification_mode == "bug_repair_pass_count"
        else baseline_event is not None
        and baseline_event.get("status") == "completed"
        and baseline_event.get("trusted") is True
    )
    result_completeness: Counter[str] = Counter()
    observed_test_results = 0
    imputed_test_results = 0
    not_run_test_results = 0
    skipped_test_results = 0
    partition_totals: Counter[str] = Counter()
    verification_result_modes: Counter[str] = Counter()
    failure_stages: Counter[str] = Counter()
    failure_origins: Counter[str] = Counter()
    failure_contexts: Counter[str] = Counter()
    count_collectors: Counter[str] = Counter()
    count_runners: Counter[str] = Counter()
    count_partial_reasons: Counter[str] = Counter()
    contract_assisted_events = 0
    contract_assisted_tests = 0
    result_events: list[dict[str, Any] | None] = [baseline_event]
    seen_step_probe_ids: set[str] = set()
    for step in steps:
        step_event = _test_event(step)
        probe_id = (
            step_event.get("probe_id")
            if isinstance(step_event, dict)
            else None
        )
        if isinstance(probe_id, str) and probe_id:
            if probe_id in seen_step_probe_ids:
                continue
            seen_step_probe_ids.add(probe_id)
        result_events.append(step_event)
    for result_event in result_events:
        if isinstance(result_event, dict):
            evidence = result_event.get("failure_evidence") or {}
            if isinstance(evidence, dict) and evidence.get("adapter_version") == 1:
                scope = "baseline" if result_event is baseline_event else "probe"
                prefix = f"non_python/{scope}/"
                if scope == "baseline" and evidence.get("baseline_singleflight_cache_hit"):
                    diagnostics[prefix + "reused_events"] += 1
                else:
                    diagnostics[prefix + "events"] += 1
                    for key in ("missing_f2p_count", "missing_p2p_count", "recovered_tests", "known_non_target_count"):
                        value = evidence.get(key)
                        if type(value) is int and value >= 0:
                            diagnostics[prefix + key] += value
                    if evidence.get("accepted_partial"):
                        diagnostics[prefix + "controlled_partial_events"] += 1
                    abort = evidence.get("source_abort") or {}
                    if isinstance(abort, dict) and abort.get("phase") in {"source_compile", "source_load"}:
                        diagnostics[prefix + abort["phase"] + "_events"] += 1
            raw_count = result_event.get("count_observation")
            if isinstance(raw_count, dict):
                collector = raw_count.get("collector")
                if isinstance(collector, str) and collector:
                    count_collectors[collector] += 1
                partial_reason = raw_count.get("partial_reason")
                if isinstance(partial_reason, str) and partial_reason:
                    count_partial_reasons[partial_reason] += 1
                for evidence in raw_count.get("command_evidence") or []:
                    if not isinstance(evidence, dict):
                        continue
                    runner = evidence.get("runner_family")
                    if isinstance(runner, str) and runner:
                        count_runners[runner] += 1
        (
            completeness,
            observed_count,
            imputed_count,
            not_run_count,
            skipped_count,
            partition_metrics,
            result_mode,
        ) = _result_event_diagnostics(result_event)
        if completeness is None:
            continue
        result_completeness[completeness] += 1
        if completeness == "contract_assisted":
            contract_assisted_events += 1
            contract_assisted_tests += imputed_count
        observed_test_results += observed_count
        imputed_test_results += imputed_count
        not_run_test_results += not_run_count
        skipped_test_results += skipped_count
        partition_totals.update(partition_metrics)
        if result_mode is not None:
            verification_result_modes[result_mode] += 1
        failure_stage = _normalize_reason(
            result_event.get("failure_stage"),
            "<none>",
        )
        failure_origin = _normalize_reason(
            result_event.get("failure_origin"),
            "<none>",
        )
        if failure_stage != "<none>":
            failure_stages[failure_stage] += 1
        if failure_origin != "<none>":
            failure_origins[failure_origin] += 1
        if failure_stage != "<none>" or failure_origin != "<none>":
            failure_contexts[f"{failure_stage}\t{failure_origin}"] += 1

    if verification_enabled and target is not None and baseline is not None:
        has_trusted_event = any(
            event is not None
            and (
                _count_observation(event, expected_test_count) is not None
                if verification_mode == "bug_repair_pass_count"
                else _trusted_test_event(event, target, baseline)
            )
            for step in steps
            for event in [_test_event(step)]
        )
        if not has_trusted_event and baseline_event is None:
            verification_reason = "trusted_test_results_unavailable"

    current_potential = 0.0
    frozen = verification_reason is not None
    reward_records = 0
    available_steps = 0
    trusted_updates = 0
    repo_unchanged_steps = 0
    repo_unchanged_available_steps = 0
    repo_changed_steps = 0
    repo_changed_available_steps = 0
    repo_changed_unavailable_reasons: Counter[str] = Counter()
    baseline_unavailable_reasons: Counter[str] = Counter()
    repo_change_unknown_steps = 0
    repo_change_unknown_available_steps = 0
    expected_available: list[bool] = []
    expected_reasons: list[str | None] = []
    expected_updates: list[bool] = []
    batch_shadow_timeout_steps = 0
    potential_state: dict[str, Any] = {
        "confirmed_by_test": {name: False for name in (target or {}) if baseline is not None and baseline.get(name) == target.get(name)},
        "confirmed_count": 0,
        "count_p2p_expected": (
            sum(
                baseline.get(name) == target.get(name)
                for name in (target or {})
            )
            if verification_mode == "bug_repair_pass_count"
            and baseline is not None
            and target is not None
            else 0
        ),
        "count_confirmed_p2p": (
            max(
                0,
                sum(
                    baseline.get(name) == target.get(name)
                    for name in (target or {})
                )
                - baseline_passed,
            )
            if verification_mode == "bug_repair_pass_count"
            and baseline_passed is not None
            and baseline is not None
            and target is not None
            else 0
        ),
    }

    can_recompute = bool(
        verification_enabled
        and target is not None
        and baseline is not None
        and (
            verification_mode == "pass_count"
            or verification_mode == "bug_repair_pass_count"
            or not any(key.startswith("invalid_beta_") for key in blockers)
        )
    )

    expected_count_rewards: list[float | None] = []
    reward_record_objects: list[dict[str, Any] | None] = []
    active_merge_group_id: str | None = None
    active_merge_final_potential: float | None = None
    active_merge_step_delta: float | None = None

    if verification_enabled and not baseline_summary_available:
        baseline_reason = _normalize_reason(
            (
                baseline_event.get("failure_type") or baseline_event.get("error")
                if isinstance(baseline_event, dict)
                else verification_reason
            ),
            "baseline_unavailable",
        )
        baseline_unavailable_reasons[baseline_reason] += 1

    for step in steps:
        before = current_potential
        updated = False
        expected_probe_potential_after: float | None = None
        step_reason = verification_reason
        event = _test_event(step)
        if isinstance(event, dict) and (
            event.get("failure_type") == "batch_shadow_finalize_timeout"
            or event.get("status") == "timeout"
            and event.get("timed_out") is True
        ):
            batch_shadow_timeout_steps += 1
        record = _reward_record(step)
        reward_record_objects.append(record)
        step_policy = (
            str(record.get("verification_potential_policy"))
            if isinstance(record, dict) and record.get("verification_potential_policy") in {"bug_repair_v2_neutral_unrun", "bug_repair_pass_count_v1", "repo_generation_pass_count_v1"}
            else verification_policy
        )
        if verification_enabled and not frozen and event is not None and target is not None and baseline is not None:
            count_observation = (
                _count_observation(event, expected_test_count)
                if verification_mode == "bug_repair_pass_count"
                else None
            )
            if count_observation is not None:
                assert baseline_passed is not None
                current_potential = float(
                    _effective_count_passed(count_observation, potential_state)
                    - baseline_passed
                )
                updated = True
                trusted_updates += 1
            elif _trusted_test_event(event, target, baseline):
                results = _status_map(event.get("test_results")) or {}
                physical_potential = _verification_potential(
                    results,
                    baseline,
                    target,
                    beta_any=beta_any,
                    beta_frac=beta_frac,
                    mode=verification_mode,
                    policy=step_policy,
                    state=potential_state,
                    partitions=_partition_counts(event),
                )
                merge_group_id = event.get("probe_merge_group_id")
                merge_group_size = event.get("probe_merge_group_size")
                merge_group_index = event.get("probe_merge_group_index")
                if (
                    verification_mode == "pass_count"
                    and isinstance(merge_group_id, str)
                    and isinstance(merge_group_size, int)
                    and not isinstance(merge_group_size, bool)
                    and merge_group_size >= 1
                    and isinstance(merge_group_index, int)
                    and not isinstance(merge_group_index, bool)
                    and 0 <= merge_group_index < merge_group_size
                ):
                    expected_probe_potential_after = physical_potential
                    if merge_group_index == 0:
                        active_merge_group_id = merge_group_id
                        active_merge_final_potential = physical_potential
                        active_merge_step_delta = (
                            physical_potential - before
                        ) / merge_group_size
                    elif active_merge_group_id != merge_group_id:
                        issues["noncontiguous_probe_merge_group"] += 1
                    if (
                        active_merge_group_id == merge_group_id
                        and active_merge_final_potential is not None
                        and active_merge_step_delta is not None
                    ):
                        current_potential = (
                            active_merge_final_potential
                            if merge_group_index == merge_group_size - 1
                            else before + active_merge_step_delta
                        )
                        if merge_group_index == merge_group_size - 1:
                            active_merge_group_id = None
                            active_merge_final_potential = None
                            active_merge_step_delta = None
                    else:
                        current_potential = physical_potential
                else:
                    current_potential = physical_potential
                updated = True
                trusted_updates += 1
            else:
                step_reason = _normalize_reason(
                    event.get("failure_type") or event.get("error"),
                    "untrusted_test_event",
                )
                diagnostics[f"untrusted_test_event:{step_reason}"] += 1
                if _continuity_lost(event):
                    frozen = True
                    verification_reason = step_reason

        available = verification_enabled and step_reason is None
        expected_available.append(available)
        expected_reasons.append(step_reason)
        expected_updates.append(updated)
        if available:
            available_steps += 1

        action_event = _action_event(step)
        repo_changed = (
            action_event.get("repo_changed")
            if isinstance(action_event, dict)
            else None
        )
        if repo_changed is True:
            repo_changed_steps += 1
            persisted_available = (
                record.get("verification_available")
                if isinstance(record, dict)
                and isinstance(record.get("verification_available"), bool)
                else available
            )
            if persisted_available:
                repo_changed_available_steps += 1
            else:
                persisted_reason = (
                    record.get("verification_unavailable_reason")
                    if isinstance(record, dict)
                    else None
                )
                event_reason = (
                    event.get("failure_type") or event.get("error")
                    if isinstance(event, dict)
                    else None
                )
                reason = _normalize_reason(
                    persisted_reason or event_reason or step_reason,
                    "verification_unavailable_reason_missing",
                )
                repo_changed_unavailable_reasons[reason] += 1
        elif repo_changed is False:
            repo_unchanged_steps += 1
            if available:
                repo_unchanged_available_steps += 1
        else:
            # Missing/unknown repository fingerprints must not be silently
            # grouped with unchanged steps. They stay outside both public
            # cohort denominators and remain available for reconciliation.
            repo_change_unknown_steps += 1
            if available:
                repo_change_unknown_available_steps += 1

        if record is None:
            issues["missing_step_reward_record"] += 1
            expected_count_rewards.append(None)
            continue
        reward_records += 1
        for merge_key in (
            "probe_merge_group_id",
            "probe_merge_group_size",
            "probe_merge_group_index",
            "probe_merge_anchor_action_id",
        ):
            expected_merge_value = (
                event.get(merge_key) if isinstance(event, dict) else None
            )
            if record.get(merge_key) != expected_merge_value:
                issues[f"step_{merge_key}_mismatch"] += 1
        raw_probe_potential_after = record.get(
            "verification_probe_potential_after"
        )
        if (
            expected_probe_potential_after is None
            and raw_probe_potential_after is not None
        ) or (
            expected_probe_potential_after is not None
            and not _float_equal(
                raw_probe_potential_after,
                expected_probe_potential_after,
            )
        ):
            issues["step_verification_probe_potential_after_mismatch"] += 1
        if not verification_enabled:
            _compare_bool(issues, record, "verification_available", False)
            _compare_bool(issues, record, "verification_updated", False)
            _compare_float(issues, record, "verification_potential_before", 0.0)
            _compare_float(issues, record, "verification_potential_after", 0.0)
            expected_count_rewards.append(None)
            continue
        if can_recompute or verification_reason is not None:
            _compare_bool(issues, record, "verification_available", available)
            _compare_bool(issues, record, "verification_updated", updated)
            _compare_float(issues, record, "verification_potential_before", before)
            _compare_float(issues, record, "verification_potential_after", current_potential)
            actual_reason = record.get("verification_unavailable_reason")
            if (None if actual_reason in (None, "") else str(actual_reason)) != step_reason:
                issues["step_verification_unavailable_reason_mismatch"] += 1
        audit_count_reward = bool(
            verification_mode == "bug_repair_pass_count"
            or verification_mode == "pass_count"
            and isinstance(record.get("schema_version"), int)
            and int(record["schema_version"]) >= 4
        )
        if audit_count_reward and can_recompute:
            raw_delta = current_potential - before
            clipped_delta = min(clip_upper, max(clip_lower, raw_delta))
            _compare_float(issues, record, "verification_raw_delta", raw_delta)
            _compare_float(
                issues,
                record,
                "verification_clipped_delta",
                clipped_delta,
            )
            nav_before = _finite_float(record.get("navigation_potential_before"))
            nav_after = _finite_float(record.get("navigation_potential_after"))
            if nav_before is None or nav_after is None:
                expected_count_rewards.append(None)
            else:
                expected_reward = verification_weight * clipped_delta + navigation_weight * (nav_after - nav_before)
                expected_count_rewards.append(expected_reward)
                _compare_float(
                    issues,
                    record,
                    "potential_reward",
                    expected_reward,
                )
        else:
            expected_count_rewards.append(None)

    if (
        verification_mode in {"pass_count", "bug_repair_pass_count"}
        and can_recompute
        and len(expected_count_rewards) == len(steps)
        and all(value is not None for value in expected_count_rewards)
        and all(record is not None for record in reward_record_objects)
    ):
        discounted_future = 0.0
        for expected_reward, record in reversed(
            list(zip(expected_count_rewards, reward_record_objects, strict=True))
        ):
            assert expected_reward is not None and record is not None
            expected_backward = backward_lambda * discounted_future
            _compare_float(
                issues,
                record,
                "backward_credit",
                expected_backward,
            )
            raw_format = _finite_float(record.get("format_reward"))
            if raw_format is None:
                issues["step_format_reward_invalid"] += 1
            else:
                _compare_float(
                    issues,
                    record,
                    "process_advantage",
                    expected_reward + expected_backward + format_weight * raw_format,
                )
            discounted_future = backward_gamma * (
                expected_reward + discounted_future
            )

    if not isinstance(document.get("milestone_verification"), dict):
        blockers["missing_compact_verification_summary"] += 1

    compact_signal = verification_enabled and available_steps > 0
    compact_final_available = bool(verification_enabled and baseline_summary_available and len(steps) == reward_records and (expected_available[-1] if expected_available else True))
    expected_fraction = available_steps / len(steps) if steps else float(baseline_summary_available and verification_enabled)
    quality_status = str(compact.get("shadow_quality_status") or "unavailable")
    if not verification_enabled:
        expected_status = "disabled"
    elif not baseline_summary_available:
        expected_status = "unavailable"
    elif any(not value for value in expected_available) or reward_records != len(steps) or quality_status != "complete":
        expected_status = "partial"
    else:
        expected_status = "complete"

    if compact:
        if verification_status != expected_status:
            issues["compact_status_mismatch"] += 1
        if _bool(compact.get("baseline_available")) is not baseline_summary_available:
            issues["compact_baseline_available_mismatch"] += 1
        if _bool(compact.get("final_available")) is not compact_final_available:
            issues["compact_final_available_mismatch"] += 1
        if _bool(compact.get("signal_participated")) is not compact_signal:
            issues["compact_signal_participated_mismatch"] += 1
        if compact.get("verification_updates") != sum(expected_updates):
            issues["compact_verification_updates_mismatch"] += 1
        if not _float_equal(compact.get("available_step_fraction"), expected_fraction):
            issues["compact_available_step_fraction_mismatch"] += 1
        first_gap_reason = next((reason for reason in expected_reasons if reason is not None), None)
        if expected_status in {"complete", "disabled"}:
            expected_compact_reason = None
            compact_reason_auditable = True
        elif first_gap_reason is not None:
            expected_compact_reason = first_gap_reason
            compact_reason_auditable = True
        elif expected_status == "unavailable":
            expected_compact_reason = verification_reason
            compact_reason_auditable = verification_reason is not None
        elif reward_records != len(steps):
            expected_compact_reason = "milestone_annotation_missing"
            compact_reason_auditable = True
        else:
            # Full shadow failure counters are intentionally absent from the
            # compact trajectory context, so a quality-only partial reason
            # cannot always be reconstructed from one rollout file.
            expected_compact_reason = None
            compact_reason_auditable = False
        raw_compact_reason = compact.get("unavailable_reason")
        actual_compact_reason = None if raw_compact_reason in (None, "") else str(raw_compact_reason)
        if compact_reason_auditable and actual_compact_reason != expected_compact_reason:
            issues["compact_unavailable_reason_mismatch"] += 1

    if reward_summary:
        expected_final_potential = current_potential if steps else 0.0
        if verification_enabled and (can_recompute or verification_reason is not None):
            if not _float_equal(reward_summary.get("verification_potential_final"), expected_final_potential):
                issues["trajectory_verification_potential_final_mismatch"] += 1
            expected_summary_available = bool(expected_available and expected_available[-1])
            if _bool(reward_summary.get("verification_available")) is not expected_summary_available:
                issues["trajectory_verification_available_mismatch"] += 1
            expected_summary_reason = expected_reasons[-1] if expected_reasons else verification_reason
            raw_summary_reason = reward_summary.get("verification_unavailable_reason")
            actual_summary_reason = None if raw_summary_reason in (None, "") else str(raw_summary_reason)
            if actual_summary_reason != expected_summary_reason:
                issues["trajectory_verification_unavailable_reason_mismatch"] += 1
    elif enabled:
        blockers["missing_trajectory_reward_summary"] += 1

    if issues:
        verdict = "inconsistent"
    elif blockers:
        verdict = "unauditable"
    elif not verification_enabled:
        verdict = "disabled"
    elif baseline is None or verification_reason is not None and available_steps == 0:
        verdict = "consistent_unavailable"
    else:
        verdict = "verified"

    rollout_metadata = document.get("metadata")
    rollout_metadata = rollout_metadata if isinstance(rollout_metadata, dict) else {}
    shadow_summary = rollout_metadata.get("shadow_sandbox")
    shadow_summary = shadow_summary if isinstance(shadow_summary, dict) else {}
    background_lifecycle = rollout_metadata.get(
        "denovo_background_finalize"
    )
    background_lifecycle = (
        background_lifecycle
        if isinstance(background_lifecycle, dict)
        else {}
    )
    shadow_background = shadow_summary.get("background_lifecycle")
    shadow_background = (
        shadow_background if isinstance(shadow_background, dict) else {}
    )
    shadow_resources = background_lifecycle.get("shadow_resources")
    shadow_resources = (
        shadow_resources if isinstance(shadow_resources, dict) else {}
    )
    shadow_resource_budget = background_lifecycle.get(
        "shadow_resource_budget"
    )
    shadow_resource_budget = (
        shadow_resource_budget
        if isinstance(shadow_resource_budget, dict)
        else {}
    )
    training_disposition = (
        lifecycle_training_disposition
        if lifecycle_training_disposition != "not_applicable"
        else str(
            background_lifecycle.get("training_disposition")
            or "pending"
            if background_lifecycle
            else "not_applicable"
        )
    )
    completed_before_primary = int(
        shadow_background.get("completed_probes_at_primary_finish") or 0
    )
    barrier_started = _finite_float(
        background_lifecycle.get("batch_barrier_started_at_monotonic")
    )
    if barrier_started is None:
        barrier_started = _finite_float(
            shadow_background.get("batch_barrier_started_at_monotonic")
        )
    has_barrier = barrier_started is not None
    observed_completed_at_barrier = int(
        _first_finite(
            background_lifecycle.get("completed_probes_at_barrier"),
            shadow_background.get("completed_probes_at_barrier"),
        )
        or 0
    )
    observed_pending_at_barrier = int(
        _first_finite(
            background_lifecycle.get("pending_probes_at_barrier"),
            shadow_background.get("pending_probes_at_barrier"),
        )
        or 0
    )
    if has_barrier:
        completed_before_barrier = int(
            _first_finite(
                background_lifecycle.get("completed_probes_at_barrier_start"),
                shadow_background.get("completed_probes_at_barrier_start"),
                completed_before_primary,
            )
            or 0
        )
        completed_at_barrier = observed_completed_at_barrier
        pending_at_barrier = observed_pending_at_barrier
        completed_at_disposition = 0
        cancelled_pending_steps = 0
    else:
        completed_before_barrier = completed_before_primary
        completed_at_barrier = 0
        pending_at_barrier = 0
        explicit_completed_at_disposition = _first_finite(
            background_lifecycle.get("completed_probes_at_disposition"),
            shadow_background.get("completed_probes_at_disposition"),
        )
        explicit_cancelled_pending = _first_finite(
            background_lifecycle.get("cancelled_pending_steps"),
            shadow_background.get("cancelled_pending_steps"),
        )
        completed_at_disposition = int(explicit_completed_at_disposition or 0)
        cancelled_pending_steps = int(explicit_cancelled_pending or 0)
    barrier_deadline = _finite_float(
        background_lifecycle.get("batch_barrier_deadline_monotonic")
    )
    shadow_finalized = _first_finite(
        background_lifecycle.get("shadow_finalized_at_monotonic"),
        shadow_background.get("finalized_at_monotonic"),
    )
    input_sealed_at = _first_finite(
        shadow_background.get("input_sealed_at_monotonic"),
        background_lifecycle.get("model_session_released_at_monotonic"),
    )
    primary_finished_at = _first_finite(
        background_lifecycle.get("primary_verifier_finished_at_monotonic"),
        shadow_background.get("primary_verifier_finished_at_monotonic"),
    )
    explicit_background_lifetime = _finite_float(
        background_lifecycle.get("shadow_background_lifetime_s")
    )
    shadow_background_lifetime = (
        max(0.0, explicit_background_lifetime)
        if explicit_background_lifetime is not None
        else max(0.0, shadow_finalized - input_sealed_at)
        if shadow_finalized is not None and input_sealed_at is not None
        else 0.0
    )
    explicit_post_primary_overlap = _finite_float(
        background_lifecycle.get("shadow_post_primary_overlap_s")
    )
    shadow_post_primary_overlap = (
        max(0.0, explicit_post_primary_overlap)
        if explicit_post_primary_overlap is not None
        else max(0.0, shadow_finalized - primary_finished_at)
        if shadow_finalized is not None and primary_finished_at is not None
        else 0.0
    )
    computed_barrier_wait = (
        max(
            0.0,
            min(
                shadow_finalized,
                barrier_deadline
                if barrier_deadline is not None
                else shadow_finalized,
            )
            - barrier_started,
        )
        if barrier_started is not None and shadow_finalized is not None
        else 0.0
    )
    explicit_barrier_wait = _finite_float(
        background_lifecycle.get("batch_barrier_wait_s")
    )
    barrier_wait = (
        max(0.0, explicit_barrier_wait)
        if explicit_barrier_wait is not None
        else computed_barrier_wait
    )
    barrier_budget_remaining = _finite_float(
        background_lifecycle.get(
            "batch_barrier_budget_remaining_at_reservation_s"
        )
    )
    if barrier_budget_remaining is None:
        barrier_observed = _finite_float(
            shadow_background.get("batch_barrier_observed_at_monotonic")
        )
        barrier_budget_remaining = (
            max(0.0, barrier_deadline - barrier_observed)
            if barrier_deadline is not None and barrier_observed is not None
            else 0.0
        )
    calibration_diagnostics = shadow_summary.get("partition_calibration_diagnostics")
    calibration_diagnostics = calibration_diagnostics if isinstance(calibration_diagnostics, dict) else {}
    raw_probe_merge_histogram = shadow_summary.get(
        "probe_merge_group_size_histogram"
    )
    probe_merge_histogram: Counter[str] = Counter()
    if isinstance(raw_probe_merge_histogram, dict):
        for raw_size, raw_count in raw_probe_merge_histogram.items():
            if (
                str(raw_size).isdigit()
                and isinstance(raw_count, int)
                and not isinstance(raw_count, bool)
                and raw_count >= 0
            ):
                probe_merge_histogram[str(int(raw_size))] += raw_count
    raw_same_changed_files_run_histogram = shadow_summary.get(
        "probe_merge_same_changed_files_run_length_histogram"
    )
    same_changed_files_run_histogram: Counter[str] = Counter()
    if isinstance(raw_same_changed_files_run_histogram, dict):
        for raw_length, raw_count in (
            raw_same_changed_files_run_histogram.items()
        ):
            if (
                str(raw_length).isdigit()
                and int(raw_length) >= 1
                and isinstance(raw_count, int)
                and not isinstance(raw_count, bool)
                and raw_count >= 0
            ):
                same_changed_files_run_histogram[str(int(raw_length))] += (
                    raw_count
                )
    probe_merge_enabled = shadow_summary.get("probe_merge_enabled") is True
    probe_merge_candidate_steps = int(
        shadow_summary.get("probe_merge_candidate_steps") or 0
    )
    probe_merge_wait_mean = _finite_float(
        shadow_summary.get("probe_merge_decision_wait_s_mean")
    )
    return FileAudit(
        path=path,
        mode=mode,
        step_idx=step_idx,
        task_id=task_id,
        language=language,
        verification_status=verification_status,
        verdict=verdict,
        signal_participated=compact_signal,
        baseline_available=baseline_summary_available,
        baseline_probe_succeeded=(
            baseline_probe_succeeded
            if verification_mode == "bug_repair_pass_count"
            else baseline_summary_available
        ),
        materialized_fallback=materialized_fallback,
        safe_baseline_available=bool(
            baseline_summary_available
            if verification_mode == "pass_count"
            else shadow_summary.get(
                "safe_baseline_available",
                baseline_summary_available,
            )
        ),
        benign_baseline_partial=bool(
            shadow_summary.get("benign_baseline_partial")
            or isinstance(baseline_event, dict)
            and baseline_event.get("failure_type")
            == "baseline_count_incomplete"
        ),
        actionable_baseline_failure=bool(
            shadow_summary.get("actionable_baseline_failure")
            or verification_mode == "bug_repair_pass_count"
            and isinstance(baseline_event, dict)
            and baseline_event.get("trusted") is not True
            and baseline_event.get("failure_type")
            not in {None, "baseline_count_incomplete"}
        ),
        baseline_source=(
            baseline_source
            if verification_mode == "bug_repair_pass_count"
            else "runtime_probe"
            if baseline_summary_available
            else "<none>"
        ),
        baseline_collector=baseline_collector,
        count_plan_hash=str(
            shadow_summary.get("count_plan_hash") or "<none>"
        ),
        count_collectors=count_collectors,
        count_runners=count_runners,
        count_partial_reasons=count_partial_reasons,
        expected_test_count=expected_test_count,
        materialized_baseline_passed=materialized_baseline_passed,
        runtime_baseline_passed=runtime_baseline_passed,
        baseline_passed=baseline_passed,
        baseline_count_mismatch=(
            abs(runtime_baseline_passed - materialized_baseline_passed)
            if runtime_baseline_passed is not None
            else 0
        ),
        final_available=compact_final_available,
        steps=len(steps),
        reward_records=reward_records,
        trusted_updates=trusted_updates,
        available_steps=available_steps,
        unavailable_reason=unavailable_reason,
        root_unavailable_reason=_root_unavailable_reason(
            shadow_summary,
            baseline_event,
            unavailable_reason,
        ),
        dynamic_sampling_status=dynamic_sampling_status,
        optimizer_committed=optimizer_committed,
        optimizer_step=optimizer_step,
        optimizer_consumed=optimizer_consumed,
        rollout_idx=rollout_idx,
        result_idx=result_idx,
        repo_unchanged_steps=repo_unchanged_steps,
        repo_unchanged_available_steps=repo_unchanged_available_steps,
        repo_changed_steps=repo_changed_steps,
        repo_changed_available_steps=repo_changed_available_steps,
        repo_changed_unavailable_reasons=repo_changed_unavailable_reasons,
        baseline_unavailable_reasons=baseline_unavailable_reasons,
        repo_change_unknown_steps=repo_change_unknown_steps,
        repo_change_unknown_available_steps=repo_change_unknown_available_steps,
        dataset_task_id=dataset_task_id,
        verification_potential_mode=verification_mode,
        replays_skipped_no_repo_change=int(shadow_summary.get("replays_skipped_no_repo_change") or 0),
        partition_adapter=str(shadow_summary.get("partition_adapter") or calibration_diagnostics.get("runner") or "<none>"),
        partition_calibration_reason=_normalize_reason(shadow_summary.get("partition_calibration_reason")),
        partition_calibration_stage=_normalize_reason(calibration_diagnostics.get("failed_stage")),
        result_completeness=result_completeness,
        observed_test_results=observed_test_results,
        imputed_test_results=imputed_test_results,
        not_run_test_results=not_run_test_results,
        skipped_test_results=skipped_test_results,
        f2p_observed=partition_totals["f2p_observed"],
        f2p_skipped=partition_totals["f2p_skipped"],
        f2p_not_run=partition_totals["f2p_not_run"],
        p2p_observed=partition_totals["p2p_observed"],
        p2p_skipped=partition_totals["p2p_skipped"],
        p2p_not_run=partition_totals["p2p_not_run"],
        verification_result_modes=verification_result_modes,
        partition_calibration_status=str(
            shadow_summary.get("partition_calibration_status") or "not_attempted"
        ),
        failure_stages=failure_stages,
        failure_origins=failure_origins,
        failure_contexts=failure_contexts,
        contract_assisted_events=contract_assisted_events,
        contract_assisted_tests=contract_assisted_tests,
        partition_group_calls=int(shadow_summary.get("partition_group_calls") or 0),
        partition_group_successes=int(
            shadow_summary.get("partition_group_successes") or 0
        ),
        partition_calibration_attempts=int(
            shadow_summary.get("partition_calibration_attempts") or 0
        ),
        partition_calibration_successes=int(
            shadow_summary.get("partition_calibration_successes") or 0
        ),
        partition_timeout_recovery_attempts=int(
            shadow_summary.get("partition_timeout_recovery_attempts") or 0
        ),
        partition_timeout_recovery_successes=int(
            shadow_summary.get("partition_timeout_recovery_successes") or 0
        ),
        partition_timeout_recovery_partial=int(
            shadow_summary.get("partition_timeout_recovery_partial") or 0
        ),
        partition_timeout_recovery_failures=int(
            shadow_summary.get("partition_timeout_recovery_failures") or 0
        ),
        denovo_background=bool(background_lifecycle),
        primary_verifier_status=str(
            background_lifecycle.get("primary_verifier_status")
            or "not_applicable"
        ),
        training_disposition=training_disposition,
        shadow_finalize_disposition=str(
            background_lifecycle.get("shadow_finalize_disposition")
            or shadow_background.get("finalize_disposition")
            or "pending"
            if background_lifecycle
            else "not_applicable"
        ),
        batch_barrier_started=has_barrier,
        batch_shadow_timed_out=bool(
            background_lifecycle.get("shadow_finalize_disposition")
            == "timeout"
            or shadow_background.get("finalize_disposition") == "timeout"
        ),
        batch_shadow_timeout_steps=batch_shadow_timeout_steps,
        shadow_completed_before_primary=bool(
            shadow_background.get("shadow_drained_at_primary_finish")
        ),
        shadow_completed_before_barrier=bool(
            shadow_background.get("shadow_drained_at_barrier_start")
        ),
        completed_probes_at_barrier=completed_at_barrier,
        pending_probes_at_barrier=pending_at_barrier,
        completed_probes_before_primary=completed_before_primary,
        completed_probes_before_barrier=max(
            0,
            completed_before_barrier - completed_before_primary,
        ),
        completed_probes_during_barrier=max(
            0,
            completed_at_barrier - completed_before_barrier,
        ),
        completed_probes_at_disposition=completed_at_disposition,
        completed_probes_after_primary_before_disposition=max(
            0,
            completed_at_disposition - completed_before_primary,
        ),
        cancelled_pending_steps=cancelled_pending_steps,
        primary_verifier_duration_s=float(
            _finite_float(
                background_lifecycle.get("primary_verifier_duration_s")
            )
            or 0.0
        ),
        shadow_background_head_start_s=float(
            _finite_float(
                background_lifecycle.get("shadow_background_head_start_s")
            )
            or 0.0
        )
        if has_barrier
        else 0.0,
        shadow_background_lifetime_s=shadow_background_lifetime,
        shadow_post_primary_overlap_s=shadow_post_primary_overlap,
        batch_barrier_wait_s=barrier_wait,
        batch_barrier_budget_remaining_s=max(
            0.0,
            barrier_budget_remaining,
        ),
        shadow_resource_cpus=int(shadow_resources.get("cpus") or 0),
        shadow_resource_memory_mb=int(
            shadow_resources.get("memory_mb") or 0
        ),
        shadow_resource_difficulty_bucket=str(
            task.get("shadow_resource_difficulty_bucket")
            or task_metadata_root.get("shadow_resource_difficulty_bucket")
            or "not_applicable"
        ),
        shadow_queue_depth_at_handoff=int(
            shadow_background.get("queue_depth_at_handoff") or 0
        ),
        shadow_queue_depth_at_barrier_start=int(
            shadow_background.get("queue_depth_at_barrier_start") or 0
        ),
        shadow_max_queue_depth=int(
            shadow_summary.get("max_queue_depth") or 0
        ),
        shadow_peak_active_sandboxes=int(
            shadow_resource_budget.get("peak_count") or 0
        ),
        shadow_peak_active_cpus=int(
            shadow_resource_budget.get("peak_cpus") or 0
        ),
        shadow_peak_active_memory_mb=int(
            shadow_resource_budget.get("peak_memory_mb") or 0
        ),
        shadow_limit_sandboxes=int(
            shadow_resource_budget.get("max_count") or 0
        ),
        shadow_limit_cpus=int(
            shadow_resource_budget.get("max_cpus") or 0
        ),
        shadow_limit_memory_mb=int(
            shadow_resource_budget.get("max_memory_mb") or 0
        ),
        probe_merge_enabled=probe_merge_enabled,
        probe_merge_max_steps=int(
            shadow_summary.get("probe_merge_max_steps") or 0
        ),
        probe_merge_candidate_steps=probe_merge_candidate_steps,
        probe_merge_physical_probes=int(
            shadow_summary.get("probes_scheduled") or 0
        )
        if probe_merge_enabled
        else 0,
        probe_merge_groups=int(
            shadow_summary.get("probe_merge_groups") or 0
        ),
        probe_merge_steps=int(
            shadow_summary.get("probe_merge_steps") or 0
        ),
        probe_merge_saved_probes=int(
            shadow_summary.get("probe_merge_saved_probes") or 0
        ),
        probe_merge_group_size_histogram=probe_merge_histogram,
        probe_merge_same_changed_files_run_length_histogram=(
            same_changed_files_run_histogram
        ),
        probe_merge_same_changed_files_runs=int(
            shadow_summary.get("probe_merge_same_changed_files_runs") or 0
        ),
        probe_merge_same_changed_files_steps=int(
            shadow_summary.get("probe_merge_same_changed_files_steps") or 0
        ),
        probe_merge_decision_wait_samples=(
            int(
                shadow_summary.get("probe_merge_decision_wait_samples")
                or 0
            )
            or probe_merge_candidate_steps
            if probe_merge_wait_mean is not None
            else 0
        ),
        probe_merge_decision_wait_s_min=_finite_float(
            shadow_summary.get("probe_merge_decision_wait_s_min")
        ),
        probe_merge_decision_wait_s_mean=probe_merge_wait_mean,
        probe_merge_decision_wait_s_max=_finite_float(
            shadow_summary.get("probe_merge_decision_wait_s_max")
        ),
        cancelled_group_rollouts=int(
            background_lifecycle.get("cancelled_group_rollouts") or 0
        ),
        cancelled_materialized_rollouts=int(
            background_lifecycle.get("cancelled_materialized_rollouts") or 0
        ),
        cancelled_shadow_leases=int(
            background_lifecycle.get("cancelled_shadow_leases") or 0
        ),
        issues=issues,
        blockers=blockers,
        diagnostics=diagnostics,
    )


def _process_file(path: str, mode_filter: str) -> FileAudit | None:
    try:
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
    except Exception:
        return None
    if not isinstance(document, dict):
        return None
    mode = str(document.get("mode") or "unknown")
    if mode_filter != "all" and mode != mode_filter:
        return FileAudit(
            path=path,
            mode=mode,
            step_idx=-1,
            task_id="<filtered>",
            language=None,
            verification_status="<filtered>",
            verdict="<filtered>",
            signal_participated=False,
            baseline_available=False,
            final_available=False,
            steps=0,
            reward_records=0,
            trusted_updates=0,
            available_steps=0,
            unavailable_reason="<none>",
            root_unavailable_reason="<none>",
        )
    return _audit_document(path, document)


def scan_rollout_dir(
    rollout_dir: str,
    mode_filter: str = "train",
    workers: int = DEFAULT_WORKERS,
) -> ScanResult:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    files = sorted(path for path in glob(os.path.join(os.path.abspath(rollout_dir), "*.json")) if ".tmp" not in os.path.basename(path))
    process = partial(_process_file, mode_filter=mode_filter)
    if workers == 1:
        parsed = list(map(process, files))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            parsed = list(executor.map(process, files))
    malformed_examples = [path for path, result in zip(files, parsed, strict=True) if result is None]
    mode_counts = Counter(result.mode for result in parsed if result is not None)
    audits = [result for result in parsed if result is not None and result.verdict != "<filtered>"]
    return ScanResult(
        audits=audits,
        files_found=len(files),
        malformed=len(malformed_examples),
        malformed_examples=malformed_examples[:EXAMPLE_LIMIT],
        mode_counts=mode_counts,
    )


_AUDIT_COUNTER_FIELDS = frozenset(
    {
        "result_completeness",
        "verification_result_modes",
        "failure_stages",
        "failure_origins",
        "failure_contexts",
        "count_collectors",
        "count_runners",
        "count_partial_reasons",
        "repo_changed_unavailable_reasons",
        "baseline_unavailable_reasons",
        "probe_merge_group_size_histogram",
        "probe_merge_same_changed_files_run_length_histogram",
        "issues",
        "blockers",
        "diagnostics",
    }
)


def _serialize_file_audit(audit: FileAudit | None) -> dict[str, Any] | None:
    if audit is None:
        return None
    payload: dict[str, Any] = {}
    for name in FileAudit.__dataclass_fields__:
        value = getattr(audit, name)
        payload[name] = dict(value) if name in _AUDIT_COUNTER_FIELDS else value
    return payload


def _deserialize_file_audit(payload: Any) -> FileAudit | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("cached file audit must be an object or null")
    expected = set(FileAudit.__dataclass_fields__)
    if set(payload) != expected:
        raise ValueError("cached file audit fields do not match this implementation")
    values = dict(payload)
    for name in _AUDIT_COUNTER_FIELDS:
        raw = values[name]
        if not isinstance(raw, dict):
            raise ValueError(f"cached {name} must be an object")
        values[name] = Counter(raw)
    return FileAudit(**values)


def _default_cache_path(rollout_dir: str) -> str:
    return os.path.abspath(os.path.normpath(rollout_dir)) + CACHE_SUFFIX


def _load_cache(cache_path: str, rollout_dir: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with open(cache_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return None, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return None, "cache root is not an object"
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        return None, "cache schema version changed"
    if payload.get("stats_version") != CACHE_STATS_VERSION:
        return None, "statistics implementation version changed"
    if payload.get("rollout_dir") != os.path.realpath(rollout_dir):
        return None, "cache belongs to a different rollout directory"
    if not isinstance(payload.get("entries"), dict):
        return None, "cache entries are missing"
    return payload, None


def _write_json_atomic(path: str, payload: dict[str, Any]) -> str | None:
    parent = os.path.dirname(os.path.abspath(path)) or "."
    try:
        os.makedirs(parent, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _scan_result_from_cached_rows(
    rows: list[tuple[str, FileAudit | None]],
    mode_filter: str,
    diagnostics: CacheDiagnostics,
) -> ScanResult:
    parsed = [audit for _, audit in rows]
    malformed_paths = [path for path, audit in rows if audit is None]
    mode_counts = Counter(audit.mode for audit in parsed if audit is not None)
    audits = [
        audit
        for audit in parsed
        if audit is not None and (mode_filter == "all" or audit.mode == mode_filter)
    ]
    return ScanResult(
        audits=audits,
        files_found=len(rows),
        malformed=len(malformed_paths),
        malformed_examples=malformed_paths[:EXAMPLE_LIMIT],
        mode_counts=mode_counts,
        cache=diagnostics,
    )


def scan_rollout_dir_incremental(
    rollout_dir: str,
    mode_filter: str = "train",
    workers: int = DEFAULT_WORKERS,
    *,
    cache_path: str | None = None,
    rebuild_cache: bool = False,
) -> ScanResult:
    """Scan immutable rollout files while parsing only previously unseen names."""
    if workers < 1:
        raise ValueError("workers must be >= 1")
    rollout_dir = os.path.abspath(rollout_dir)
    cache_path = os.path.abspath(cache_path or _default_cache_path(rollout_dir))
    diagnostics = CacheDiagnostics(enabled=True, path=cache_path)
    try:
        directory_mtime_ns = os.stat(rollout_dir).st_mtime_ns
    except OSError as exc:
        raise ValueError(f"rollout dir is not readable: {rollout_dir}: {exc}") from exc

    cache: dict[str, Any] | None = None
    if not rebuild_cache:
        cache, diagnostics.load_error = _load_cache(cache_path, rollout_dir)

    if cache is not None and cache.get("directory_mtime_ns") == directory_mtime_ns:
        try:
            rows = [
                (
                    os.path.join(rollout_dir, key),
                    _deserialize_file_audit(entry.get("audit")),
                )
                for key, entry in sorted(cache["entries"].items())
                if isinstance(entry, dict)
            ]
        except Exception as exc:
            diagnostics.load_error = f"{type(exc).__name__}: {exc}"
        else:
            diagnostics.fast_path = True
            diagnostics.hits = len(rows)
            return _scan_result_from_cached_rows(rows, mode_filter, diagnostics)

    files = sorted(
        path
        for path in glob(os.path.join(rollout_dir, "*.json"))
        if ".tmp" not in os.path.basename(path) and os.path.abspath(path) != cache_path
    )
    old_entries = cache.get("entries", {}) if cache is not None else {}
    entries: dict[str, Any] = {}
    audits_by_key: dict[str, FileAudit | None] = {}
    new_paths: list[str] = []
    new_keys: list[str] = []
    for path in files:
        key = os.path.relpath(path, rollout_dir)
        old = old_entries.get(key)
        if not rebuild_cache and isinstance(old, dict) and "audit" in old:
            try:
                audit = _deserialize_file_audit(old["audit"])
            except Exception:
                pass
            else:
                diagnostics.hits += 1
                audits_by_key[key] = audit
                entries[key] = old
                continue
        new_paths.append(path)
        new_keys.append(key)

    process = partial(_process_file, mode_filter="all")
    if workers == 1:
        parsed = list(map(process, new_paths))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            parsed = list(executor.map(process, new_paths))
    diagnostics.misses = len(parsed)
    for key, audit in zip(new_keys, parsed, strict=True):
        audits_by_key[key] = audit
        entries[key] = {"audit": _serialize_file_audit(audit)}

    diagnostics.removed = len(set(old_entries) - set(entries))
    rows = [
        (path, audits_by_key[os.path.relpath(path, rollout_dir)])
        for path in files
    ]
    final_directory_mtime_ns = os.stat(rollout_dir).st_mtime_ns
    diagnostics.directory_changed_during_scan = final_directory_mtime_ns != directory_mtime_ns
    diagnostics.write_error = _write_json_atomic(
        cache_path,
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "stats_version": CACHE_STATS_VERSION,
            "rollout_dir": os.path.realpath(rollout_dir),
            "directory_mtime_ns": (
                final_directory_mtime_ns
                if not diagnostics.directory_changed_during_scan
                else -1
            ),
            "entries": entries,
        },
    )
    return _scan_result_from_cached_rows(rows, mode_filter, diagnostics)


def _pct(number: int, denominator: int) -> str:
    return f"{100.0 * number / denominator:5.1f}%" if denominator else "  -  "


def _print_row(values: list[str], widths: list[int]) -> None:
    print("  " + " | ".join(value.rjust(width) for value, width in zip(values, widths, strict=True)))


def _group_by_step(audits: Iterable[FileAudit]) -> dict[int, list[FileAudit]]:
    grouped: dict[int, list[FileAudit]] = defaultdict(list)
    for audit in audits:
        grouped[audit.step_idx].append(audit)
    return grouped


def _print_availability_table(audits: list[FileAudit], mode: str) -> None:
    print(f"\n=== TABLE 1 — Verification availability per global_step  [mode={mode}] ===")
    header = ["step", "rollouts", "complete", "partial", "unavailable", "disabled", "signal"]
    widths = [7, 9, 18, 18, 18, 18, 18]
    _print_row(header, widths)
    print("  " + "-+-".join("-" * width for width in widths))
    for step, rows in sorted(_group_by_step(audits).items()):
        counts = Counter(row.verification_status for row in rows)
        signal = sum(row.signal_participated for row in rows)
        total = len(rows)
        _print_row(
            [str(step), str(total)] + [f"{_pct(counts[status], total)} ({counts[status]})" for status in VERIFICATION_STATUSES[:4]] + [f"{_pct(signal, total)} ({signal})"],
            widths,
        )


def _print_integrity_table(audits: list[FileAudit], mode: str) -> None:
    print(f"\n=== TABLE 2 — Recomputed potential integrity per global_step  [mode={mode}] ===")
    header = ["step", "rollouts", *AUDIT_VERDICTS]
    widths = [7, 9, 22, 25, 18, 22, 20]
    _print_row(header, widths)
    print("  " + "-+-".join("-" * width for width in widths))
    for step, rows in sorted(_group_by_step(audits).items()):
        counts = Counter(row.verdict for row in rows)
        total = len(rows)
        _print_row(
            [str(step), str(total)] + [f"{_pct(counts[verdict], total)} ({counts[verdict]})" for verdict in AUDIT_VERDICTS],
            widths,
        )


def _print_overall(audits: list[FileAudit], mode: str) -> None:
    total = len(audits)
    statuses = Counter(row.verification_status for row in audits)
    verdicts = Counter(row.verdict for row in audits)
    potential_modes = Counter(row.verification_potential_mode for row in audits)
    total_steps = sum(row.steps for row in audits)
    available_steps = sum(row.available_steps for row in audits)
    updates = sum(row.trusted_updates for row in audits)
    all_steps_available = sum(
        _all_verification_steps_available(row) for row in audits
    )
    usable_partial = sum(
        row.verification_status == "partial"
        and _all_verification_steps_available(row)
        for row in audits
    )
    print(f"\n=== TABLE 3 — Overall verification audit  [mode={mode}] ===")
    print(f"  rollouts                         : {total}")
    print(
        "  strict quality complete rollouts : "
        f"{statuses['complete']}/{total} "
        f"({_pct(statuses['complete'], total).strip()})"
    )
    print(
        "  all-step-available rollouts      : "
        f"{all_steps_available}/{total} "
        f"({_pct(all_steps_available, total).strip()})"
    )
    print(
        "  usable partial, no step gaps     : "
        f"{usable_partial}/{total} "
        f"({_pct(usable_partial, total).strip()})"
    )
    print(
        "  rollouts with availability gap   : "
        f"{total - all_steps_available}/{total} "
        f"({_pct(total - all_steps_available, total).strip()})"
    )
    print(
        "  verification potential modes       : "
        + ", ".join(f"{name}={count}" for name, count in potential_modes.most_common())
    )
    print(f"  verification-bearing steps         : {available_steps}/{total_steps} ({_pct(available_steps, total_steps).strip()})")
    print(
        "  repo-unchanged verification steps  : "
        f"{sum(row.repo_unchanged_available_steps for row in audits)}/"
        f"{sum(row.repo_unchanged_steps for row in audits)} "
        f"({_pct(sum(row.repo_unchanged_available_steps for row in audits), sum(row.repo_unchanged_steps for row in audits)).strip()})"
    )
    print(
        "  repo-changed verification steps    : "
        f"{sum(row.repo_changed_available_steps for row in audits)}/"
        f"{sum(row.repo_changed_steps for row in audits)} "
        f"({_pct(sum(row.repo_changed_available_steps for row in audits), sum(row.repo_changed_steps for row in audits)).strip()})"
    )
    print(
        "  repo-change-unknown steps         : "
        f"{sum(row.repo_change_unknown_available_steps for row in audits)}/"
        f"{sum(row.repo_change_unknown_steps for row in audits)}"
    )
    print(f"  trusted verification updates       : {updates}")
    count_rows = [
        row
        for row in audits
        if row.verification_potential_mode == "bug_repair_pass_count"
    ]
    if count_rows:
        baseline_successes = sum(row.baseline_probe_succeeded for row in count_rows)
        fallbacks = sum(row.materialized_fallback for row in count_rows)
        count_completeness: Counter[str] = Counter()
        collectors: Counter[str] = Counter()
        for row in count_rows:
            count_completeness.update(
                {
                    key: value
                    for key, value in row.result_completeness.items()
                    if key in {"count_full", "count_partial"}
                }
            )
            collectors[row.baseline_collector] += 1
        print(
            "  count baseline probe succeeded   : "
            f"{baseline_successes}/{len(count_rows)} "
            f"({_pct(baseline_successes, len(count_rows)).strip()})"
        )
        print(
            "  materialized C0 fallbacks         : "
            f"{fallbacks}/{len(count_rows)} "
            f"({_pct(fallbacks, len(count_rows)).strip()})"
        )
        print(
            "  count full/partial observations   : "
            f"{count_completeness['count_full']}/"
            f"{count_completeness['count_partial']}"
        )
        print(
            "  runtime/materialized C0 abs diff  : "
            f"{sum(row.baseline_count_mismatch for row in count_rows)}"
        )
        print(
            "  baseline count collectors         : "
            + ", ".join(
                f"{name}={count}" for name, count in collectors.most_common()
            )
        )
    print(
        "  shadow replays skipped (no repo change): "
        f"{sum(row.replays_skipped_no_repo_change for row in audits)}"
    )
    print("  strict rollout evidence quality:")
    for status in VERIFICATION_STATUSES:
        if statuses[status]:
            print(f"    {status:<28} {_pct(statuses[status], total)}  ({statuses[status]})")
    print("  arithmetic audit:")
    for verdict in AUDIT_VERDICTS:
        if verdicts[verdict]:
            print(f"    {verdict:<28} {_pct(verdicts[verdict], total)}  ({verdicts[verdict]})")
    result_modes: Counter[str] = Counter()
    calibration = Counter(row.partition_calibration_status for row in audits)
    for row in audits:
        result_modes.update(row.verification_result_modes)
    print("  verification result events:")
    for result_mode, count in result_modes.most_common():
        print(f"    {result_mode:<28} {count}")
    print("  partition calibration rollouts:")
    for status, count in calibration.most_common():
        print(f"    {status:<28} {_pct(count, total)}  ({count})")
    contract_rollouts = sum(row.contract_assisted_events > 0 for row in audits)
    print(
        "  contract-assisted rollouts/events/tests: "
        f"{contract_rollouts}/{sum(row.contract_assisted_events for row in audits)}/"
        f"{sum(row.contract_assisted_tests for row in audits)}"
    )
    partition_calls = sum(row.partition_group_calls for row in audits)
    partition_successes = sum(row.partition_group_successes for row in audits)
    print(
        "  partition group successes        : "
        f"{partition_successes}/{partition_calls} "
        f"({_pct(partition_successes, partition_calls).strip()})"
    )
    recovery_attempts = sum(
        row.partition_timeout_recovery_attempts for row in audits
    )
    recovery_successes = sum(
        row.partition_timeout_recovery_successes for row in audits
    )
    print(
        "  timeout partition recoveries     : "
        f"success/partial/failure/attempt="
        f"{recovery_successes}/"
        f"{sum(row.partition_timeout_recovery_partial for row in audits)}/"
        f"{sum(row.partition_timeout_recovery_failures for row in audits)}/"
        f"{recovery_attempts} "
        f"({_pct(recovery_successes, recovery_attempts).strip()})"
    )
    print(
        "  partition observations           : "
        f"F2P observed/skipped/not-run="
        f"{sum(row.f2p_observed for row in audits)}/"
        f"{sum(row.f2p_skipped for row in audits)}/"
        f"{sum(row.f2p_not_run for row in audits)}, "
        f"P2P={sum(row.p2p_observed for row in audits)}/"
        f"{sum(row.p2p_skipped for row in audits)}/"
        f"{sum(row.p2p_not_run for row in audits)}"
    )
    if verdicts["inconsistent"]:
        result = "FAIL — persisted verification potentials disagree with recomputation"
    elif verdicts["unauditable"]:
        result = "INCOMPLETE — no arithmetic mismatch found, but some files lack audit inputs"
    elif verdicts["consistent_unavailable"]:
        result = "DEGRADED — arithmetic is consistent where available, but some rollouts have no verification signal"
    else:
        result = "PASS — all enabled verification potentials are reproducible"
    print(f"  RESULT                           : {result}")


def _print_denovo_background_lifecycle(
    audits: list[FileAudit],
    mode: str,
    *,
    by_language: bool = False,
) -> None:
    rows = [row for row in audits if row.denovo_background]
    title = (
        "TABLE 14 — DeNovo background lifecycle by language"
        if by_language
        else "TABLE 3B — DeNovo background verifier/shadow lifecycle"
    )
    print(f"\n=== {title}  [mode={mode}] ===")
    if not rows:
        print("  (not applicable)")
        return
    grouped: dict[str, list[FileAudit]] = defaultdict(list)
    for row in rows:
        grouped[(row.language or "<none>") if by_language else "all"].append(row)
    for label, cohort in sorted(grouped.items()):
        primary = Counter(row.primary_verifier_status for row in cohort)
        disposition = Counter(row.training_disposition for row in cohort)
        shadow = Counter(row.shadow_finalize_disposition for row in cohort)
        cancellation_groups: dict[
            tuple[str, str],
            dict[str, Any],
        ] = {}
        for row in cohort:
            raw_disposition = row.training_disposition
            if raw_disposition in DENOVO_FILTERED_DISPOSITIONS:
                cancellation_category = "filtered"
            elif raw_disposition in DENOVO_REQUEUED_DISPOSITIONS:
                cancellation_category = "requeued"
            elif raw_disposition in DENOVO_SURPLUS_DISPOSITIONS:
                cancellation_category = "surplus"
            else:
                continue
            group_identity = (
                row.task_id
                if row.task_id and row.task_id != "<missing>"
                else f"{row.dataset_task_id}:{row.step_idx}"
            )
            group = cancellation_groups.setdefault(
                (cancellation_category, group_identity),
                {
                    "rollouts": set(),
                    "slots": 0,
                    "materialized": 0,
                    "leases": 0,
                },
            )
            rollout_identity: object = (
                (row.rollout_idx, row.result_idx)
                if row.rollout_idx >= 0
                else row.path
            )
            group["rollouts"].add(rollout_identity)
            group["slots"] = max(
                int(group["slots"]),
                row.cancelled_group_rollouts,
            )
            group["materialized"] = max(
                int(group["materialized"]),
                row.cancelled_materialized_rollouts,
            )
            group["leases"] = max(
                int(group["leases"]),
                row.cancelled_shadow_leases,
            )
        cancellation_totals: dict[str, tuple[int, int, int, int]] = {}
        for category in ("filtered", "requeued", "surplus"):
            category_groups = [
                value
                for (group_category, _), value in cancellation_groups.items()
                if group_category == category
            ]
            cancellation_totals[category] = (
                len(category_groups),
                sum(
                    max(int(value["slots"]), len(value["rollouts"]))
                    for value in category_groups
                ),
                sum(
                    max(
                        int(value["materialized"]),
                        len(value["rollouts"]),
                    )
                    for value in category_groups
                ),
                sum(int(value["leases"]) for value in category_groups),
            )
        resources = Counter(
            (
                row.shadow_resource_difficulty_bucket,
                row.shadow_resource_cpus,
                row.shadow_resource_memory_mb,
            )
            for row in cohort
        )
        prefix = f"language={label} " if by_language else ""
        print(f"  {prefix}rollouts={len(cohort)}")
        print(
            "    primary verifier status        : "
            + ", ".join(f"{key}={value}" for key, value in primary.most_common())
        )
        print(
            "    training disposition           : "
            + ", ".join(f"{key}={value}" for key, value in disposition.most_common())
        )
        print(
            "    cancelled groups/slots/persisted/leases: "
            + ", ".join(
                f"{category}={groups}/{slots}/{materialized}/{leases}"
                for category, (
                    groups,
                    slots,
                    materialized,
                    leases,
                ) in cancellation_totals.items()
            )
        )
        print(
            "    shadow finalize disposition    : "
            + ", ".join(f"{key}={value}" for key, value in shadow.most_common())
        )
        timed_out_rows = [row for row in cohort if row.batch_shadow_timed_out]
        barrier_rows = [row for row in cohort if row.batch_barrier_started]
        cancelled_rows = [
            row
            for row in cohort
            if row.training_disposition in DENOVO_CANCELLED_DISPOSITIONS
        ]
        print(
            "    batch timeout rollouts/steps/done/pending: "
            f"{len(timed_out_rows)}/"
            f"{sum(row.batch_shadow_timeout_steps for row in timed_out_rows)}/"
            f"{sum(row.completed_probes_at_barrier for row in timed_out_rows)}/"
            f"{sum(row.pending_probes_at_barrier for row in timed_out_rows)}"
        )
        completed_before_primary_rollouts = sum(
            row.shadow_completed_before_primary for row in cohort
        )
        completed_before_barrier_rollouts = sum(
            not row.shadow_completed_before_primary
            and row.shadow_completed_before_barrier
            for row in cohort
        )
        completed_during_barrier_rollouts = sum(
            row.batch_barrier_started
            and
            not row.shadow_completed_before_barrier
            and not row.batch_shadow_timed_out
            and row.shadow_finalize_disposition == "completed"
            for row in cohort
        )
        print(
            "    shadow drained checkpoints     : "
            f"before-primary={completed_before_primary_rollouts}, "
            f"before-barrier={completed_before_barrier_rollouts}, "
            f"during-barrier={completed_during_barrier_rollouts}"
        )
        print(
            "    terminal rollout dispositions  : "
            f"cancelled={len(cancelled_rows)}, "
            f"batch-timeout={sum(row.batch_shadow_timed_out for row in cohort)}, "
            f"completed={sum(row.shadow_finalize_disposition == 'completed' for row in cohort)}"
        )
        before_primary_probes = sum(
            row.completed_probes_before_primary for row in cohort
        )
        before_barrier_probes = sum(
            row.completed_probes_before_barrier for row in cohort
        )
        during_barrier_probes = sum(
            row.completed_probes_during_barrier for row in cohort
        )
        before_disposition_probes = sum(
            row.completed_probes_after_primary_before_disposition
            for row in cohort
        )
        print(
            "    probe completions by phase     : "
            f"before-primary={before_primary_probes}, "
            f"after-primary/pre-barrier={before_barrier_probes}, "
            f"during-barrier={during_barrier_probes}, "
            f"after-primary/pre-disposition={before_disposition_probes}, "
            "total="
            f"{before_primary_probes + before_barrier_probes + during_barrier_probes + before_disposition_probes}"
        )
        print(
            "    pending steps by disposition   : "
            f"batch-barrier={sum(row.pending_probes_at_barrier for row in cohort)}, "
            f"cancelled={sum(row.cancelled_pending_steps for row in cohort)}"
        )
        primary_mean = (
            sum(row.primary_verifier_duration_s for row in cohort) / len(cohort)
        )
        print(
            "    mean primary verifier duration : "
            f"{primary_mean:.1f}s"
        )
        print(
            "    selected barrier timing        : "
            + (
                "head-start="
                f"{sum(row.shadow_background_head_start_s for row in barrier_rows) / len(barrier_rows):.1f}s, "
                "wait="
                f"{sum(row.batch_barrier_wait_s for row in barrier_rows) / len(barrier_rows):.1f}s, "
                "budget-remaining="
                f"{sum(row.batch_barrier_budget_remaining_s for row in barrier_rows) / len(barrier_rows):.1f}s"
                if barrier_rows
                else "n/a (no optimizer barrier in this cohort)"
            )
        )
        print(
            "    cancelled shadow timing        : "
            + (
                "lifetime="
                f"{sum(row.shadow_background_lifetime_s for row in cancelled_rows) / len(cancelled_rows):.1f}s, "
                "post-primary="
                f"{sum(row.shadow_post_primary_overlap_s for row in cancelled_rows) / len(cancelled_rows):.1f}s"
                if cancelled_rows
                else "n/a"
            )
        )
        resource_availability: list[str] = []
        for (bucket, cpus, memory_mb), count in resources.most_common():
            matching = [
                row
                for row in cohort
                if (
                    row.shadow_resource_difficulty_bucket,
                    row.shadow_resource_cpus,
                    row.shadow_resource_memory_mb,
                )
                == (bucket, cpus, memory_mb)
            ]
            resource_availability.append(
                f"{bucket}:{cpus}C/{memory_mb}MB={count} "
                "repo-changed-verification-observed="
                f"{sum(row.repo_changed_available_steps for row in matching)}/"
                f"{sum(row.repo_changed_steps for row in matching)}"
            )
        print(
            "    difficulty/resources           : "
            + ", ".join(resource_availability)
        )
        print(
            "    per-rollout milestone queue max : "
            "handoff="
            f"{max(row.shadow_queue_depth_at_handoff for row in cohort)}/"
            "barrier="
            + (
                str(max(row.shadow_queue_depth_at_barrier_start for row in barrier_rows))
                if barrier_rows
                else "n/a"
            )
            + "/observed="
            + str(max(row.shadow_max_queue_depth for row in cohort))
        )
        peak_sandboxes = max(row.shadow_peak_active_sandboxes for row in cohort)
        peak_cpus = max(row.shadow_peak_active_cpus for row in cohort)
        peak_memory_mb = max(row.shadow_peak_active_memory_mb for row in cohort)
        limit_sandboxes = max(row.shadow_limit_sandboxes for row in cohort)
        limit_cpus = max(row.shadow_limit_cpus for row in cohort)
        limit_memory_mb = max(row.shadow_limit_memory_mb for row in cohort)

        def peak_limit(peak: int, limit: int, *, suffix: str = "") -> str:
            if limit <= 0:
                return f"{peak}/unknown{suffix}"
            return f"{peak}/{limit}{suffix} ({_pct(peak, limit).strip()})"

        print(
            "    global resource peak/limit     : "
            f"sandboxes={peak_limit(peak_sandboxes, limit_sandboxes)}, "
            f"cpu={peak_limit(peak_cpus, limit_cpus)}, "
            f"memory={peak_limit(peak_memory_mb, limit_memory_mb, suffix='MB')}"
        )


def _print_partition_calibration_details(audits: list[FileAudit], mode: str) -> None:
    counts: Counter[tuple[str, str, str, str]] = Counter()
    for audit in audits:
        counts[
            (
                audit.partition_adapter,
                audit.partition_calibration_status,
                audit.partition_calibration_stage,
                audit.partition_calibration_reason,
            )
        ] += 1
    print(f"\n=== TABLE 7 — Partition calibration by runner and failure stage  [mode={mode}] ===")
    if not counts:
        print("  (none)")
        return
    for (runner, status, stage, reason), count in counts.most_common():
        print(
            f"  runner={runner:<12} status={status:<28} stage={stage:<14} "
            f"count={count:<6} reason={reason}"
        )


def _print_task_loss_sources(audits: list[FileAudit], mode: str) -> None:
    grouped: dict[str, list[FileAudit]] = defaultdict(list)
    for audit in audits:
        grouped[audit.dataset_task_id].append(audit)
    ranked: list[tuple[int, int, str, str, str]] = []
    for task_id, rows in grouped.items():
        complete = sum(row.verification_status == "complete" for row in rows)
        signal = sum(row.signal_participated for row in rows)
        losses = len(rows) - complete
        reasons = Counter(
            row.unavailable_reason
            for row in rows
            if row.verification_status != "complete" and row.unavailable_reason != "<none>"
        )
        top_reason = reasons.most_common(1)[0][0] if reasons else "<none>"
        ranked.append((losses, len(rows) - signal, task_id, rows[0].language or "<none>", top_reason))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    print(f"\n=== TABLE 8 — Task-level largest verification losses  [mode={mode}] ===")
    if not ranked or ranked[0][0] == 0:
        print("  (none)")
        return
    for losses, no_signal, task_id, language, reason in ranked[:EXAMPLE_LIMIT]:
        if losses == 0:
            break
        print(
            f"  task={task_id:<48} language={language:<12} "
            f"non-complete={losses:<4} no-signal={no_signal:<4} top_reason={reason}"
        )


def _print_reason_table(
    audits: list[FileAudit],
    *,
    title: str,
    source: str,
) -> None:
    counts: Counter[str] = Counter()
    for audit in audits:
        value = getattr(audit, source)
        if isinstance(value, Counter):
            counts.update(value)
        elif value and value != "<none>":
            counts[str(value)] += 1
    print(f"\n=== {title} ===")
    if not counts:
        print("  (none)")
        return
    total = sum(counts.values())
    for reason, count in counts.most_common():
        print(f"  {reason:<64} {_pct(count, total)}  ({count})")


def _print_examples(audits: list[FileAudit]) -> None:
    affected = [audit for audit in audits if audit.verdict in {"inconsistent", "unauditable", "consistent_unavailable"}]
    print("\n=== TABLE 9 — Affected rollout examples ===")
    if not affected:
        print("  (none)")
        return
    for audit in affected[:EXAMPLE_LIMIT]:
        reasons = list(audit.issues) or list(audit.blockers) or [audit.unavailable_reason]
        relative = os.path.basename(audit.path)
        print(f"  [{audit.verdict}] step={audit.step_idx} task={audit.task_id} reason={','.join(reasons[:4])}\n    {relative}")
    if len(affected) > EXAMPLE_LIMIT:
        print(f"  ... {len(affected) - EXAMPLE_LIMIT} more affected rollouts omitted")


def _availability_cell(number: int, total: int) -> str:
    return f"{number}/{total} ({_pct(number, total).strip()})"


def _all_verification_steps_available(audit: FileAudit) -> bool:
    """Return whether a rollout has a usable baseline and no step-level gap."""

    return bool(
        audit.baseline_available
        and audit.reward_records == audit.steps
        and audit.available_steps == audit.steps
    )


def _print_language_availability_rows(audits: list[FileAudit], *, include_step: bool) -> None:
    grouped: dict[tuple[int, str] | tuple[str], list[FileAudit]] = defaultdict(list)
    for audit in audits:
        assert audit.language is not None
        key = (audit.step_idx, audit.language) if include_step else (audit.language,)
        grouped[key].append(audit)

    header = (["step"] if include_step else []) + [
        "language",
        "rollouts",
        "strict complete",
        "strict partial",
        "unavailable",
        "disabled",
        "missing",
        "baseline usable",
        "baseline complete",
        "baseline partial",
        "C0 fallback",
        "baseline failure",
        "final",
        "signal",
        "all steps available",
        "usable partial no gaps",
        "any availability gap",
        "verification steps",
        "repo unchanged verification steps",
        "repo changed verification steps",
    ]
    widths = ([7] if include_step else []) + [
        18, 9, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
        24, 26, 24, 24, 32, 30,
    ]
    _print_row(header, widths)
    print("  " + "-+-".join("-" * width for width in widths))
    for key, rows in sorted(grouped.items()):
        counts = Counter(row.verification_status for row in rows)
        total = len(rows)
        total_steps = sum(row.steps for row in rows)
        available_steps = sum(row.available_steps for row in rows)
        all_steps_available = sum(
            _all_verification_steps_available(row) for row in rows
        )
        usable_partial = sum(
            row.verification_status == "partial"
            and _all_verification_steps_available(row)
            for row in rows
        )
        values = ([str(key[0])] if include_step else []) + [
            str(key[-1]),
            str(total),
            *[_availability_cell(counts[status], total) for status in VERIFICATION_STATUSES],
            _availability_cell(sum(row.safe_baseline_available for row in rows), total),
            _availability_cell(sum(row.baseline_probe_succeeded for row in rows), total),
            _availability_cell(sum(row.benign_baseline_partial for row in rows), total),
            _availability_cell(sum(row.materialized_fallback for row in rows), total),
            _availability_cell(sum(row.actionable_baseline_failure for row in rows), total),
            _availability_cell(sum(row.final_available for row in rows), total),
            _availability_cell(sum(row.signal_participated for row in rows), total),
            _availability_cell(all_steps_available, total),
            _availability_cell(usable_partial, total),
            _availability_cell(total - all_steps_available, total),
            _availability_cell(available_steps, total_steps),
            _availability_cell(
                sum(row.repo_unchanged_available_steps for row in rows),
                sum(row.repo_unchanged_steps for row in rows),
            ),
            _availability_cell(
                sum(row.repo_changed_available_steps for row in rows),
                sum(row.repo_changed_steps for row in rows),
            ),
        ]
        _print_row(values, widths)


def _print_language_result_completeness_rows(audits: list[FileAudit], *, include_step: bool) -> None:
    grouped: dict[tuple[int, str] | tuple[str], list[FileAudit]] = defaultdict(list)
    for audit in audits:
        assert audit.language is not None
        key = (audit.step_idx, audit.language) if include_step else (audit.language,)
        grouped[key].append(audit)
    header = (["step"] if include_step else []) + [
        "language",
        "events",
        "full",
        "baseline filled",
        "contract assisted",
        "controlled partial",
        "partition full",
        "partition partial",
        "count full",
        "count partial",
        "none",
        "observed tests",
        "imputed tests",
        "NOT_RUN tests",
        "imputed share",
        "NOT_RUN share",
    ]
    widths = ([7] if include_step else []) + [18, 9, 20, 20, 22, 22, 20, 22, 18, 18, 16, 16, 16, 16, 22, 22]
    _print_row(header, widths)
    print("  " + "-+-".join("-" * width for width in widths))
    for key, rows in sorted(grouped.items()):
        counts: Counter[str] = Counter()
        for row in rows:
            counts.update(row.result_completeness)
        total = sum(counts.values())
        observed_tests = sum(row.observed_test_results for row in rows)
        imputed_tests = sum(row.imputed_test_results for row in rows)
        not_run_tests = sum(row.not_run_test_results for row in rows)
        represented_tests = sum(
            row.observed_test_results
            + (
                row.not_run_test_results
                if row.verification_potential_mode == "bug_repair_pass_count"
                else row.imputed_test_results
            )
            for row in rows
        )
        values = ([str(key[0])] if include_step else []) + [
            str(key[-1]),
            str(total),
            _availability_cell(counts["full"], total),
            _availability_cell(counts["baseline_contract_filled"], total),
            _availability_cell(counts["contract_assisted"], total),
            _availability_cell(counts["controlled_partial"], total),
            _availability_cell(counts["partition_counts_full"], total),
            _availability_cell(counts["partition_counts_partial"], total),
            _availability_cell(counts["count_full"], total),
            _availability_cell(counts["count_partial"], total),
            _availability_cell(counts["none"], total),
            str(observed_tests),
            str(imputed_tests),
            str(not_run_tests),
            _availability_cell(imputed_tests, represented_tests),
            _availability_cell(not_run_tests, represented_tests),
        ]
        _print_row(values, widths)


def _print_language_partition_rows(audits: list[FileAudit], *, include_step: bool) -> None:
    grouped: dict[tuple[int, str] | tuple[str], list[FileAudit]] = defaultdict(list)
    for audit in audits:
        assert audit.language is not None
        key = (audit.step_idx, audit.language) if include_step else (audit.language,)
        grouped[key].append(audit)
    header = (["step"] if include_step else []) + [
        "language",
        "per-test events",
        "partition events",
        "aggregate events",
        "contract-assisted",
        "unavailable rollouts",
        "calibrated rollouts",
        "replay skips",
        "F2P obs/skip/notrun",
        "P2P obs/skip/notrun",
    ]
    widths = ([7] if include_step else []) + [18, 18, 18, 18, 18, 22, 22, 14, 24, 24]
    _print_row(header, widths)
    print("  " + "-+-".join("-" * width for width in widths))
    for key, rows in sorted(grouped.items()):
        modes: Counter[str] = Counter()
        for row in rows:
            modes.update(row.verification_result_modes)
        values = ([str(key[0])] if include_step else []) + [
            str(key[-1]),
            str(modes["per_test"]),
            str(modes["partition_counts"]),
            str(modes["aggregate_counts"]),
            str(modes["contract_assisted"]),
            str(sum(row.verification_status == "unavailable" for row in rows)),
            str(
                sum(
                    row.partition_calibration_status
                    in {"calibrated", "calibrated_shared"}
                    for row in rows
                )
            ),
            str(sum(row.replays_skipped_no_repo_change for row in rows)),
            f"{sum(row.f2p_observed for row in rows)}/{sum(row.f2p_skipped for row in rows)}/{sum(row.f2p_not_run for row in rows)}",
            f"{sum(row.p2p_observed for row in rows)}/{sum(row.p2p_skipped for row in rows)}/{sum(row.p2p_not_run for row in rows)}",
        ]
        _print_row(values, widths)


def _print_language_reason_rows(audits: list[FileAudit], *, include_step: bool) -> None:
    counts: Counter[tuple[int, str, str] | tuple[str, str]] = Counter()
    for audit in audits:
        assert audit.language is not None
        for reason, reason_count in audit.repo_changed_unavailable_reasons.items():
            key = (
                (audit.step_idx, audit.language, reason)
                if include_step
                else (audit.language, reason)
            )
            counts[key] += reason_count
    if not counts:
        print("  (none)")
        return
    for key, count in sorted(counts.items()):
        if include_step:
            step, language, reason = key
            print(f"  step={step:<6} language={language:<14} count={count:<5} reason={reason}")
        else:
            language, reason = key
            print(f"  language={language:<14} count={count:<5} reason={reason}")


def _print_language_root_reason_rows(audits: list[FileAudit]) -> None:
    counts: Counter[tuple[str, str]] = Counter()
    for audit in audits:
        reason = audit.root_unavailable_reason
        if reason and reason != "<none>":
            counts[(audit.language or "<none>", reason)] += 1
    if not counts:
        print("  (none)")
        return
    total = sum(counts.values())
    for (language, reason), count in sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    ):
        print(
            f"  language={language:<14} incidents={count:<5} "
            f"share={_pct(count, total).strip():>9} reason={reason}"
        )


def _print_language_unique_task_rows(audits: list[FileAudit]) -> None:
    grouped: dict[tuple[str, str], list[FileAudit]] = defaultdict(list)
    for audit in audits:
        assert audit.language is not None
        task_id = audit.dataset_task_id
        if not task_id or task_id == "<missing>":
            task_id = audit.task_id
        if not task_id or task_id == "<missing>":
            task_id = audit.path
        grouped[(audit.language, task_id)].append(audit)

    by_language: dict[str, list[list[FileAudit]]] = defaultdict(list)
    for (language, _), rows in grouped.items():
        by_language[language].append(rows)

    header = [
        "language",
        "unique tasks",
        "all rollouts complete",
        "any signal",
        "zero signal",
        "flaky",
    ]
    widths = [18, 14, 24, 20, 20, 20]
    _print_row(header, widths)
    print("  " + "-+-".join("-" * width for width in widths))
    for language, task_groups in sorted(by_language.items()):
        total = len(task_groups)
        all_complete = sum(
            all(row.verification_status == "complete" for row in rows)
            for rows in task_groups
        )
        any_signal = sum(
            any(row.signal_participated for row in rows)
            for rows in task_groups
        )
        zero_signal = sum(
            not any(row.signal_participated for row in rows)
            for rows in task_groups
        )
        flaky = sum(
            len(
                {
                    (row.verification_status, row.signal_participated)
                    for row in rows
                }
            )
            > 1
            for rows in task_groups
        )
        _print_row(
            [
                language,
                str(total),
                _availability_cell(all_complete, total),
                _availability_cell(any_signal, total),
                _availability_cell(zero_signal, total),
                _availability_cell(flaky, total),
            ],
            widths,
        )


def _print_language_reason_task_ranking(audits: list[FileAudit]) -> None:
    """Rank the task/reason pairs that consume repo-changed availability."""

    ranked: Counter[tuple[str, str, str]] = Counter()
    for audit in audits:
        language = audit.language or "<none>"
        task_id = _task_identity(audit)
        for reason, count in audit.repo_changed_unavailable_reasons.items():
            ranked[(language, reason, task_id)] += count
    if not ranked:
        print("  (none)")
        return
    for (language, reason, task_id), count in sorted(
        ranked.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1], item[0][2]),
    )[:100]:
        print(
            f"  language={language:<14} reason={reason:<40} "
            f"steps={count:<6} task={task_id}"
        )


def _task_identity(audit: FileAudit) -> str:
    for value in (audit.dataset_task_id, audit.task_id, audit.path):
        if value and value != "<missing>":
            return value
    return audit.path


def _zero_signal_task_manifest(audits: list[FileAudit]) -> dict[str, Any]:
    grouped: dict[str, list[FileAudit]] = defaultdict(list)
    for audit in audits:
        grouped[_task_identity(audit)].append(audit)
    zero_signal = {
        task_id: rows
        for task_id, rows in grouped.items()
        if not any(row.signal_participated for row in rows)
    }
    return {
        "schema_version": 2,
        "kind": "codeflow_verification_zero_signal_tasks",
        "task_ids": sorted(zero_signal),
        "zero_signal_task_ids": sorted(zero_signal),
        "tasks": {
            task_id: {
                "languages": sorted(
                    {
                        row.language
                        for row in rows
                        if isinstance(row.language, str) and row.language
                    }
                ),
                "rollouts": len(rows),
                "unavailable_reasons": dict(
                    sorted(Counter(row.unavailable_reason for row in rows).items())
                ),
            }
            for task_id, rows in sorted(zero_signal.items())
        },
    }


def _print_language_recovery_rows(audits: list[FileAudit]) -> None:
    grouped: dict[str, list[FileAudit]] = defaultdict(list)
    for audit in audits:
        assert audit.language is not None
        grouped[audit.language].append(audit)

    header = [
        "language",
        "rollouts",
        "unique tasks",
        "contract rollouts",
        "contract tasks",
        "partition rollouts",
        "partition tasks",
        "verification steps",
        "repo unchanged verification steps",
        "repo changed verification steps",
    ]
    widths = [18, 10, 14, 21, 19, 22, 20, 24, 32, 30]
    _print_row(header, widths)
    print("  " + "-+-".join("-" * width for width in widths))
    for language, rows in sorted(grouped.items()):
        task_ids = {_task_identity(row) for row in rows}
        contract_rows = [row for row in rows if row.contract_assisted_events]
        partition_rows = [
            row
            for row in rows
            if row.partition_calibration_successes
            or row.partition_calibration_status == "calibrated"
        ]
        total_steps = sum(row.steps for row in rows)
        available_steps = sum(row.available_steps for row in rows)
        _print_row(
            [
                language,
                str(len(rows)),
                str(len(task_ids)),
                _availability_cell(len(contract_rows), len(rows)),
                _availability_cell(
                    len({_task_identity(row) for row in contract_rows}),
                    len(task_ids),
                ),
                _availability_cell(len(partition_rows), len(rows)),
                _availability_cell(
                    len({_task_identity(row) for row in partition_rows}),
                    len(task_ids),
                ),
                _availability_cell(available_steps, total_steps),
                _availability_cell(
                    sum(row.repo_unchanged_available_steps for row in rows),
                    sum(row.repo_unchanged_steps for row in rows),
                ),
                _availability_cell(
                    sum(row.repo_changed_available_steps for row in rows),
                    sum(row.repo_changed_steps for row in rows),
                ),
            ],
            widths,
        )


def _print_language_failure_context_rows(audits: list[FileAudit]) -> None:
    grouped: dict[tuple[str, str, str], list[FileAudit]] = defaultdict(list)
    for audit in audits:
        assert audit.language is not None
        for encoded in audit.failure_contexts:
            stage, _, origin = encoded.partition("\t")
            grouped[(audit.language, stage, origin)].append(audit)
    if not grouped:
        print("  (none)")
        return
    for (language, stage, origin), rows in sorted(grouped.items()):
        print(
            f"  language={language:<14} stage={stage:<18} origin={origin:<22} "
            f"rollouts={len(rows):<5} unique_tasks="
            f"{len({_task_identity(row) for row in rows})}"
        )


def _print_language_count_collector_rows(audits: list[FileAudit]) -> None:
    grouped: Counter[tuple[str, str, str, str]] = Counter()
    event_collectors: Counter[tuple[str, str]] = Counter()
    event_runners: Counter[tuple[str, str]] = Counter()
    partial_reasons: Counter[tuple[str, str]] = Counter()
    for audit in audits:
        if audit.verification_potential_mode != "bug_repair_pass_count":
            continue
        grouped[
            (
                audit.language or "<none>",
                audit.count_plan_hash,
                audit.baseline_collector,
                audit.baseline_source,
            )
        ] += 1
        for collector, count in audit.count_collectors.items():
            event_collectors[
                (
                    audit.language or "<none>",
                    collector,
                )
            ] += count
        for runner, count in audit.count_runners.items():
            event_runners[
                (
                    audit.language or "<none>",
                    runner,
                )
            ] += count
        for reason, count in audit.count_partial_reasons.items():
            partial_reasons[
                (
                    audit.language or "<none>",
                    reason,
                )
            ] += count
    if not grouped:
        print("  (none)")
        return
    for (language, plan_hash, collector, source), count in sorted(grouped.items()):
        print(
            f"  scope=baseline language={language:<14} "
            f"plan={plan_hash[:12]:<12} "
            f"collector={collector:<32} "
            f"baseline_source={source:<22} rollouts={count}"
        )
    for (language, collector), count in sorted(event_collectors.items()):
        print(
            f"  scope=events   language={language:<14} "
            f"collector={collector:<48} events={count}"
        )
    for (language, runner), count in sorted(event_runners.items()):
        print(
            f"  scope=runner   language={language:<14} "
            f"runner={runner:<48} commands={count}"
        )
    for (language, reason), count in sorted(partial_reasons.items()):
        print(
            f"  scope=partial  language={language:<14} "
            f"reason={reason:<36} events={count}"
        )


def _is_denovo_lifecycle_only(audit: FileAudit) -> bool:
    """Exclude DeNovo rows that never entered an optimizer reservation."""

    return bool(
        audit.denovo_background
        and audit.training_disposition not in DENOVO_OPTIMIZER_DISPOSITIONS
    )


def _report_cohorts(
    audits: list[FileAudit],
    mode: str,
) -> list[tuple[str, list[FileAudit]]]:
    """Return explicit materialized and optimizer-consumed report cohorts."""

    cohorts = [("all-materialized", audits)]
    if mode == "train":
        cohorts.append(
            (
                "optimizer-consumed",
                [audit for audit in audits if audit.optimizer_consumed],
            )
        )
    return cohorts


def _print_language_verification_tables(
    audits: list[FileAudit],
    *,
    mode: str,
    cohort: str,
) -> None:
    label = f"mode={mode}; cohort={cohort}"
    print(
        f"\n=== VERIFICATION COHORT — {cohort} "
        f"[mode={mode}; rollouts={len(audits)}] ==="
    )
    if not audits:
        print("  (no rollouts in this cohort)")
        return
    print(f"\n=== TABLE 1 — Availability by global_step and language  [{label}] ===")
    _print_language_availability_rows(audits, include_step=True)
    print(f"\n=== TABLE 2 — Availability across all global_steps by language  [{label}] ===")
    _print_language_availability_rows(audits, include_step=False)
    print(f"\n=== TABLE 3 — Result completeness by global_step and language  [{label}] ===")
    _print_language_result_completeness_rows(audits, include_step=True)
    print(f"\n=== TABLE 4 — Result completeness across all global_steps by language  [{label}] ===")
    _print_language_result_completeness_rows(audits, include_step=False)
    print(f"\n=== TABLE 5 — Verification mode and partition observations by global_step and language  [{label}] ===")
    _print_language_partition_rows(audits, include_step=True)
    print(f"\n=== TABLE 6 — Verification mode and partition observations across all global_steps by language  [{label}] ===")
    _print_language_partition_rows(audits, include_step=False)
    print(f"\n=== TABLE 7 — Repo-changed unavailable step reasons by global_step and language  [{label}] ===")
    _print_language_reason_rows(audits, include_step=True)
    print(f"\n=== TABLE 8 — Repo-changed unavailable step reasons across all global_steps by language  [{label}] ===")
    _print_language_reason_rows(audits, include_step=False)
    print(f"\n=== TABLE 8B — Top repo-changed unavailable task/reason pairs  [{label}] ===")
    _print_language_reason_task_ranking(audits)
    print(
        f"\n=== TABLE 8C — Root unavailable incidents by language  "
        f"[{label}; grain=one root per rollout] ==="
    )
    _print_language_root_reason_rows(audits)
    print(f"\n=== TABLE 9 — Unique-task availability across all global_steps by language  [{label}] ===")
    _print_language_unique_task_rows(audits)
    print(f"\n=== TABLE 10 — Contract/partition recovery by rollout and unique task  [{label}] ===")
    _print_language_recovery_rows(audits)
    print(f"\n=== TABLE 11 — Failure stage/origin by rollout and unique task  [{label}] ===")
    _print_language_failure_context_rows(audits)
    non_python_diagnostics: dict[str, Counter[str]] = defaultdict(Counter)
    for audit in audits:
        for key, count in audit.diagnostics.items():
            if key.startswith("non_python/") and count:
                non_python_diagnostics[audit.language or "<none>"][key] += count
    if non_python_diagnostics:
        print(f"\n=== TABLE 11B — Non-Python evidence [grain=physical event; baseline reuse separate; missing counts=test nodes; {label}] ===")
        for language, counters in sorted(non_python_diagnostics.items()):
            for key, count in sorted(counters.items()):
                print(f"  language={language:<14} count={count:<8} metric={key}")
    print(f"\n=== TABLE 12 — Count baseline collector/source by language  [{label}] ===")
    _print_language_count_collector_rows(audits)
    print(f"\n=== TABLE 13 — DeNovo shadow probe merging  [{label}] ===")
    _print_language_probe_merge_rows(audits)


def _print_language_probe_merge_rows(audits: list[FileAudit]) -> None:
    def print_histogram(
        histogram: Counter[str],
        *,
        singular_label: str,
        plural_label: str,
    ) -> None:
        if not histogram:
            print("        (none)")
            return
        for size, count in sorted(
            histogram.items(), key=lambda item: int(item[0])
        ):
            step_label = "step" if int(size) == 1 else "steps"
            item_label = singular_label if count == 1 else plural_label
            print(f"        {size} {step_label:<5} -> {count} {item_label}")

    def render(rows: list[FileAudit], scope: str) -> None:
        enabled = [row for row in rows if row.probe_merge_enabled]
        max_steps = Counter(
            row.probe_merge_max_steps for row in enabled
        )
        histogram: Counter[str] = Counter()
        same_changed_files_run_histogram: Counter[str] = Counter()
        for row in enabled:
            histogram.update(row.probe_merge_group_size_histogram)
            same_changed_files_run_histogram.update(
                row.probe_merge_same_changed_files_run_length_histogram
            )
        group_count = sum(histogram.values())
        grouped_steps = sum(
            int(size) * count
            for size, count in histogram.items()
            if str(size).isdigit()
        )
        group_min = min((int(size) for size in histogram), default=0)
        group_max = max((int(size) for size in histogram), default=0)
        group_mean = grouped_steps / group_count if group_count else 0.0
        same_changed_files_run_count = sum(
            same_changed_files_run_histogram.values()
        )
        same_changed_files_run_steps = sum(
            int(length) * count
            for length, count in same_changed_files_run_histogram.items()
            if str(length).isdigit()
        )
        same_changed_files_run_min = min(
            (int(length) for length in same_changed_files_run_histogram),
            default=0,
        )
        same_changed_files_run_max = max(
            (int(length) for length in same_changed_files_run_histogram),
            default=0,
        )
        same_changed_files_run_mean = (
            same_changed_files_run_steps / same_changed_files_run_count
            if same_changed_files_run_count
            else 0.0
        )
        wait_samples = sum(
            row.probe_merge_decision_wait_samples for row in enabled
        )
        wait_mean = (
            sum(
                (row.probe_merge_decision_wait_s_mean or 0.0)
                * row.probe_merge_decision_wait_samples
                for row in enabled
            )
            / wait_samples
            if wait_samples
            else 0.0
        )
        wait_mins = [
            row.probe_merge_decision_wait_s_min
            for row in enabled
            if row.probe_merge_decision_wait_s_min is not None
        ]
        wait_maxs = [
            row.probe_merge_decision_wait_s_max
            for row in enabled
            if row.probe_merge_decision_wait_s_max is not None
        ]
        candidates = sum(row.probe_merge_candidate_steps for row in enabled)
        physical = sum(row.probe_merge_physical_probes for row in enabled)
        groups = sum(row.probe_merge_groups for row in enabled)
        merged_steps = sum(row.probe_merge_steps for row in enabled)
        saved = sum(row.probe_merge_saved_probes for row in enabled)
        original_probe_count = physical + saved
        reduction = (
            100.0 * saved / original_probe_count
            if original_probe_count
            else 0.0
        )
        enabled_share = 100.0 * len(enabled) / len(rows) if rows else 0.0
        max_steps_summary = ", ".join(
            f"{value} steps: {count} rollouts"
            for value, count in sorted(max_steps.items())
        ) or "(none)"
        wait_min = min(wait_mins) if wait_mins else 0.0
        wait_max = max(wait_maxs) if wait_maxs else 0.0
        print(
            f"\n  [{scope}]"
        )
        print(
            "    Rollout coverage                 : "
            f"{len(enabled)}/{len(rows)} enabled ({enabled_share:.1f}%)"
        )
        print(
            "    Configured maximum merge size    : "
            f"{max_steps_summary}"
        )
        print(
            "    Eligible repo-changing steps     : "
            f"{candidates}"
        )
        print(
            "    Physical probes executed         : "
            f"{physical}"
        )
        print(
            "    Estimated probes without merge   : "
            f"{original_probe_count}"
        )
        print(
            "    Probes saved by merging          : "
            f"{saved} ({reduction:.1f}%)"
        )
        print(
            "    Merge groups created             : "
            f"{groups}"
        )
        print(
            "    Logical steps inside merge groups: "
            f"{merged_steps}"
        )
        print(
            "    Merge group size (min/avg/max)   : "
            f"{group_min}/{group_mean:.2f}/{group_max}"
        )
        print(
            "      distribution:"
        )
        print_histogram(
            histogram,
            singular_label="group",
            plural_label="groups",
        )
        print(
            "    Consecutive same changed_files    : "
            f"{same_changed_files_run_count} runs / "
            f"{same_changed_files_run_steps} steps"
        )
        print(
            "      run length (min/avg/max)       : "
            f"{same_changed_files_run_min}/"
            f"{same_changed_files_run_mean:.2f}/"
            f"{same_changed_files_run_max}"
        )
        print(
            "      distribution:"
        )
        print_histogram(
            same_changed_files_run_histogram,
            singular_label="run",
            plural_label="runs",
        )
        print(
            "    Merge-decision wait seconds (min/avg/max): "
            f"{wait_min:.3f}/{wait_mean:.3f}/{wait_max:.3f} "
            f"({wait_samples} samples)"
        )

    grouped: dict[tuple[int, str], list[FileAudit]] = defaultdict(list)
    by_step: dict[int, list[FileAudit]] = defaultdict(list)
    by_language: dict[str, list[FileAudit]] = defaultdict(list)
    for audit in audits:
        if audit.language is None:
            continue
        grouped[(audit.step_idx, audit.language)].append(audit)
        by_step[audit.step_idx].append(audit)
        by_language[audit.language].append(audit)
    print(
        "  Reading guide: saved probes are physical test runs avoided by merge; "
        "the reduction denominator is the number of probes that would have run "
        "without merge."
    )
    render(audits, "OVERALL")

    # A single-step, single-language run previously printed the same cohort four
    # times under different labels.  Preserve useful breakdowns for mixed runs,
    # but suppress any cohort whose exact rollout membership was already shown.
    seen_cohorts: set[frozenset[int]] = {
        frozenset(id(row) for row in audits)
    }
    breakdowns: list[tuple[str, list[FileAudit]]] = []
    breakdowns.extend(
        (f"LANGUAGE: {language}", rows)
        for language, rows in sorted(by_language.items())
    )
    breakdowns.extend(
        (f"GLOBAL STEP: {step}", rows)
        for step, rows in sorted(by_step.items())
    )
    breakdowns.extend(
        (f"GLOBAL STEP {step} / LANGUAGE: {language}", rows)
        for (step, language), rows in sorted(grouped.items())
    )
    for scope, rows in breakdowns:
        cohort_key = frozenset(id(row) for row in rows)
        if cohort_key in seen_cohorts:
            continue
        seen_cohorts.add(cohort_key)
        render(rows, scope)


def _render_language_report(scan: ScanResult, rollout_dir: str, workers: int, mode_filter: str) -> str:
    language_audits = [audit for audit in scan.audits if audit.language is not None]
    has_denovo_background = any(
        audit.denovo_background for audit in language_audits
    )
    # Availability is an observation-quality report, not an optimizer-consumption
    # report.  Filtered/requeued DeNovo rollouts can contain many completed shadow
    # probes before their leases are cancelled, so excluding them hid the majority
    # of the data and made the top-level tables effectively empty before the first
    # optimizer reservation.  Keep the optimizer cohort as an explicit diagnostic,
    # while reporting all persisted rollouts in the common bug-repair-aligned tables.
    background_audits = [audit for audit in language_audits if audit.denovo_background]
    optimizer_audits = [
        audit
        for audit in background_audits
        if audit.training_disposition in DENOVO_OPTIMIZER_DISPOSITIONS
    ]
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        print("=" * 100)
        print("CODEFLOW VERIFICATION-POTENTIAL AVAILABILITY BY LANGUAGE")
        print("=" * 100)
        print(f"rollout dir : {rollout_dir}")
        print(f"scan workers: {workers}")
        print(f"reporting mode: {mode_filter}")
        print(f"language-marked rollouts: {len(language_audits)}/{len(scan.audits)}")
        if has_denovo_background:
            print(
                "optimizer-reserved rollouts: "
                f"{len(optimizer_audits)}/{len(background_audits)} "
                "(availability tables include all persisted DeNovo rollouts)"
            )
        print("Only rollouts with a non-empty language marker participate in this report.")
        print(
            "Repo-change cohorts use the persisted action_event.repo_changed boolean; "
            "unknown values are excluded from both cohort denominators."
        )

        modes = [mode_filter] if mode_filter != "all" else sorted({audit.mode for audit in language_audits})
        for mode in modes:
            lifecycle_audits = [
                audit for audit in language_audits if audit.mode == mode
            ]
            if not lifecycle_audits:
                continue
            for cohort, audits in _report_cohorts(lifecycle_audits, mode):
                _print_language_verification_tables(
                    audits,
                    mode=mode,
                    cohort=cohort,
                )
            if any(audit.denovo_background for audit in lifecycle_audits):
                _print_denovo_background_lifecycle(
                    lifecycle_audits,
                    mode,
                    by_language=True,
                )
    return output.getvalue()


def _print_verification_audit_tables(
    audits: list[FileAudit],
    *,
    mode: str,
    cohort: str,
) -> None:
    label = f"{mode}; cohort={cohort}"
    print(
        f"\n=== VERIFICATION COHORT — {cohort} "
        f"[mode={mode}; rollouts={len(audits)}] ==="
    )
    if not audits:
        print("  (no rollouts in this cohort)")
        return
    _print_availability_table(audits, label)
    _print_integrity_table(audits, label)
    _print_overall(audits, label)
    _print_reason_table(
        audits,
        title=f"TABLE 4 — Arithmetic/metadata mismatch reasons  [mode={label}]",
        source="issues",
    )
    _print_reason_table(
        audits,
        title=f"TABLE 5 — Unauditable reasons  [mode={label}]",
        source="blockers",
    )
    _print_reason_table(
        audits,
        title=f"TABLE 6 — Repo-changed unavailable step reasons  [mode={label}]",
        source="repo_changed_unavailable_reasons",
    )
    _print_reason_table(
        audits,
        title=(
            "TABLE 6B — Baseline unavailable reasons (separate cohort)  "
            f"[mode={label}]"
        ),
        source="baseline_unavailable_reasons",
    )
    _print_reason_table(
        audits,
        title=(
            "TABLE 6C — Root unavailable incidents by rollout  "
            f"[mode={label}; grain=one root per rollout]"
        ),
        source="root_unavailable_reason",
    )
    _print_partition_calibration_details(audits, label)
    _print_task_loss_sources(audits, label)
    _print_examples(audits)


def _render_report(scan: ScanResult, rollout_dir: str, workers: int, mode_filter: str) -> str:
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        mode_counts = scan.mode_counts
        print("=" * 100)
        print("CODEFLOW ROLLOUT VERIFICATION-POTENTIAL AUDIT")
        print("=" * 100)
        print(f"rollout dir : {rollout_dir}")
        print(f"files found : {scan.files_found}   parsed OK: {scan.files_found - scan.malformed}   skipped(malformed): {scan.malformed}")
        print(f"scan workers: {workers}")
        if scan.cache is not None:
            cache = scan.cache
            print(
                "cache       : "
                f"hits={cache.hits} new={cache.misses} removed={cache.removed} "
                f"fast_path={str(cache.fast_path).lower()} path={cache.path}"
            )
            if cache.load_error:
                print(f"cache load warning: {cache.load_error}")
            if cache.write_error:
                print(f"cache write warning: {cache.write_error}")
        print("modes present: " + ", ".join(f"{key}={value}" for key, value in sorted(mode_counts.items())))
        print(f"reporting mode: {mode_filter}")
        has_denovo_background = any(
            audit.denovo_background for audit in scan.audits
        )
        if has_denovo_background:
            background_count = sum(audit.denovo_background for audit in scan.audits)
            optimizer_count = sum(
                audit.denovo_background
                and audit.training_disposition in DENOVO_OPTIMIZER_DISPOSITIONS
                for audit in scan.audits
            )
            print(
                "optimizer-reserved parsed rollouts: "
                f"{optimizer_count}/{background_count} "
                "(availability audit includes all persisted DeNovo rollouts)"
            )
        print(
            "verified means every persisted verification potential was reproduced "
            "from the runtime baseline and trusted TestEvents, using either the "
            "bug-repair target map or DeNovoSWE baseline-relative pass counts. "
            "unavailable and disabled are reported separately."
        )

        modes = [mode_filter] if mode_filter != "all" else sorted(mode_counts)
        for mode in modes:
            lifecycle_audits = [
                audit for audit in scan.audits if audit.mode == mode
            ]
            if not lifecycle_audits:
                print(f"\n(no data for mode={mode})")
                continue
            for cohort, audits in _report_cohorts(lifecycle_audits, mode):
                _print_verification_audit_tables(
                    audits,
                    mode=mode,
                    cohort=cohort,
                )
            if any(audit.denovo_background for audit in lifecycle_audits):
                _print_denovo_background_lifecycle(lifecycle_audits, mode)

        if scan.malformed_examples:
            print("\n=== MALFORMED FILE EXAMPLES ===")
            for path in scan.malformed_examples:
                print(f"  {os.path.basename(path)}")
    return output.getvalue()


def _write_text_atomic(path: str, content: str) -> str | None:
    parent = os.path.dirname(os.path.abspath(path)) or "."
    try:
        os.makedirs(parent, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _default_statistics_path(rollout_dir: str) -> str:
    return os.path.abspath(os.path.normpath(rollout_dir)) + STATISTICS_SUFFIX


def _language_statistics_path(output_path: str) -> str:
    absolute = os.path.abspath(output_path)
    if absolute.endswith(STATISTICS_SUFFIX):
        return absolute[: -len(STATISTICS_SUFFIX)] + LANGUAGE_STATISTICS_SUFFIX
    root, extension = os.path.splitext(absolute)
    return f"{root}.language{extension}" if extension else f"{absolute}.language"


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--workers must be an integer >= 1") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("--workers must be an integer >= 1")
    return parsed


@contextlib.contextmanager
def _exclusive_cache_lock(cache_path: str):
    lock_path = f"{os.path.abspath(cache_path)}.lock"
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rollout_dir", help="Directory containing rollout *.json files.")
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=DEFAULT_WORKERS,
        help=f"Concurrent rollout readers (default: {DEFAULT_WORKERS}; use 1 for serial).",
    )
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "val", "all"],
        help="Rollout mode to report (default: train).",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help=("Statistics report path. Defaults to <ROLLOUT_DIR>.codeflow_verification_potential.statistics. A conditional language report is written beside it with '.language' before the extension."),
    )
    parser.add_argument(
        "--cache-file",
        default=None,
        help="Incremental per-file cache path (default: sibling of ROLLOUT_DIR).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the incremental cache and parse every rollout.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Ignore cached entries and rebuild them from every rollout.",
    )
    parser.add_argument(
        "--zero-signal-task-ids-file",
        default=None,
        help=(
            "Optional JSON manifest for baseline-only canaries containing "
            "the unique tasks with no signal in any selected rollout."
        ),
    )
    args = parser.parse_args()

    if not os.path.isdir(args.rollout_dir):
        print(f"ERROR: rollout dir not found: {args.rollout_dir}", file=sys.stderr)
        return 2
    output_path = os.path.abspath(args.output_file or _default_statistics_path(args.rollout_dir))
    cache_path = os.path.abspath(args.cache_file or _default_cache_path(args.rollout_dir))

    def run() -> int:
        scan = (
            scan_rollout_dir(args.rollout_dir, args.mode, args.workers)
            if args.no_cache
            else scan_rollout_dir_incremental(
                args.rollout_dir,
                args.mode,
                args.workers,
                cache_path=cache_path,
                rebuild_cache=args.rebuild_cache,
            )
        )
        if not scan.files_found:
            print(f"ERROR: no *.json rollout files in {args.rollout_dir}", file=sys.stderr)
            return 2
        report = _render_report(scan, os.path.abspath(args.rollout_dir), args.workers, args.mode)
        write_error = _write_text_atomic(output_path, report)
        if write_error:
            print(f"ERROR: failed to write statistics file {output_path}: {write_error}", file=sys.stderr)
            return 2
        if any(audit.language is not None for audit in scan.audits):
            language_output_path = _language_statistics_path(output_path)
            language_report = _render_language_report(scan, os.path.abspath(args.rollout_dir), args.workers, args.mode)
            write_error = _write_text_atomic(language_output_path, language_report)
            if write_error:
                print(f"ERROR: failed to write language statistics file {language_output_path}: {write_error}", file=sys.stderr)
                return 2
        if args.zero_signal_task_ids_file is not None:
            task_manifest_path = os.path.abspath(args.zero_signal_task_ids_file)
            write_error = _write_json_atomic(
                task_manifest_path,
                _zero_signal_task_manifest(scan.audits),
            )
            if write_error:
                print(
                    "ERROR: failed to write zero-signal task manifest "
                    f"{task_manifest_path}: {write_error}",
                    file=sys.stderr,
                )
                return 2
        return 0

    if args.no_cache:
        return run()
    with _exclusive_cache_lock(cache_path):
        return run()


if __name__ == "__main__":
    raise SystemExit(main())
