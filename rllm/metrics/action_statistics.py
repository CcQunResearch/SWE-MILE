#!/usr/bin/env python3
"""Statistics over saved Codeflow rollout trajectories.

The report summarizes Codeflow's native tools:

    search | read_file | edit_file | execute_bash | submit | other

It reports independent parse, validation, and tool views of every model turn:

1. A lenient distribution that recovers the intended tool from native-call
   metadata or the raw model response when parsing failed.
2. A strict distribution that attributes a named tool only when
   ``parse_status == "ok"`` and the recorded action is a real Codeflow tool.
3. Validation status and validation-error reasons grouped by the selected
   tool.  A strict parsed tool can still fail argument validation; in
   particular, rejected ``submit`` arguments remain visible as ``submit``
   validation errors rather than being mistaken for successful submissions.

Parse status and parse-error reasons are reported separately. This distinction
prevents a native tool-call parsing regression from looking like a change in the
policy's intended tool mix.

Usage::

    python launch/action_statistics.py [ROLLOUT_DIR]
        [--mode train|val|all] [--csv OUT_PREFIX] [--macro] [--workers N]
        [--cache-file PATH] [--output-file PATH] [--no-cache] [--rebuild-cache]

The input is the backward-compatible schema-version-3/4/5 JSON written by
``AgentFlowEngine._rollout_sample_payload``. Files are processed independently,
so threaded and serial aggregation produce identical results.

By default, per-file results are cached in a sibling of ``ROLLOUT_DIR``. An
unchanged rollout directory can be served without enumerating the shared-filesystem
directory again. When the directory changes, only new or replaced JSON files
are reparsed. RLLM publishes rollout files with an atomic ``os.replace``, so
the directory timestamp is a valid fast-path change signal for these logs.
Use ``--rebuild-cache`` after manually editing a rollout file in place.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from glob import glob
from typing import Any

DEFAULT_ROLLOUT_DIR = "<path/to/exp_root>/logs/<run_time>/rollout"
DEFAULT_WORKERS = 12
CACHE_SCHEMA_VERSION = 1
CACHE_STATS_VERSION = 3
CACHE_SUFFIX = ".codeflow_action_distribution.cache.json"
STATISTICS_SUFFIX = ".codeflow_action_distribution.statistics"

BUCKETS = ["search", "read_file", "edit_file", "execute_bash", "submit", "other"]
CODEFLOW_TOOLS = frozenset(BUCKETS[:-1])

OTHER_NO_ACTION = "<no_action_detected>"
OTHER_MULTIPLE_ACTIONS = "<multiple_tool_calls>"
STRICT_PARSE_FAILED = "<parse_failed>"
STRICT_EMPTY_ACTION = "<empty_action>"

# Qwen native rendering may survive in model_response even when the gateway did
# not return a structured tool call. Only the intended name is recovered here;
# arguments are deliberately not reparsed or treated as executable.
_FUNCTION_OPEN_RE = re.compile(r"<function=([^>\s]+)", re.IGNORECASE)
_TOOL_CALL_NAME_RE = re.compile(
    r'<tool_call>\s*\{.*?["\']name["\']\s*:\s*["\']([^"\']+)["\']',
    re.DOTALL,
)


def _normalize_name(name: Any) -> str:
    """Mirror the scaffold's namespace/prefix normalization."""
    normalized = str(name or "").strip()
    if ":" in normalized:
        normalized = normalized.split(":", 1)[0]
    if "." in normalized:
        normalized = normalized.split(".")[-1]
    # Malformed native calls can put a command or parameter body in the name.
    # Keep diagnostics one-line and bounded so one bad call cannot corrupt the
    # terminal table or generate an unreasonably large Counter key.
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if len(normalized) > 120:
        normalized = normalized[:117] + "..."
    return normalized


def _agent_step_metadata(step: dict[str, Any]) -> dict[str, Any]:
    """Read both raw Step metadata and trace-enriched nested metadata."""
    metadata = step.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    nested = metadata.get("agent_step_metadata")
    return nested if isinstance(nested, dict) else metadata


def _action_event(step: dict[str, Any]) -> dict[str, Any]:
    metadata = _agent_step_metadata(step)
    event = metadata.get("action_event")
    return event if isinstance(event, dict) else {}


def parse_status_of(step: dict[str, Any]) -> tuple[str, str]:
    """Return ``(parse_status, parse_reason)`` with ActionEvent fallback."""
    metadata = _agent_step_metadata(step)
    raw_status = metadata.get("parse_status")
    raw_reason = metadata.get("parse_reason")

    if raw_status not in (None, ""):
        return str(raw_status), str(raw_reason or "")

    event = _action_event(step)
    event_status = str(event.get("parse_status") or "")
    if event_status == "ok":
        status = "ok"
    elif event_status in {"error", "parse_error"}:
        status = "parse_error"
    else:
        status = "<missing>"
    return status, str(event.get("parse_error") or raw_reason or "")


def _normalize_diagnostic(value: Any) -> str:
    """Keep free-form diagnostics bounded and safe for text reports/cache keys."""
    normalized = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(normalized) > 240:
        normalized = normalized[:237] + "..."
    return normalized


def validation_status_of(step: dict[str, Any]) -> tuple[str, str]:
    """Return ``(validation_status, validation_error)`` from ActionEvent.

    The direct metadata fallback supports early development traces that wrote
    these fields alongside ``parse_status`` before ActionEvent enrichment was
    stable.  Missing validation telemetry is reported explicitly rather than
    inferred from parse success.
    """
    metadata = _agent_step_metadata(step)
    event = _action_event(step)
    raw_status = event.get("validation_status")
    raw_error = event.get("validation_error")
    if raw_status in (None, ""):
        raw_status = metadata.get("validation_status")
        raw_error = metadata.get("validation_error") or raw_error
    status = str(raw_status) if raw_status not in (None, "") else "<missing>"
    return status, _normalize_diagnostic(raw_error)


def _validation_tool_name(step: dict[str, Any]) -> str:
    """Attribute validation to the tool selected by the model."""
    event_name = _normalize_name(_action_event(step).get("tool_name"))
    if event_name:
        return event_name
    recorded = _recorded_action_name(step)
    if recorded:
        return recorded
    intended, _ = _intended_name(step)
    return intended or "<unknown>"


def _raw_tool_names(step: dict[str, Any]) -> list[str]:
    metadata = _agent_step_metadata(step)
    raw_call = metadata.get("raw_tool_call")
    if isinstance(raw_call, dict):
        name = _normalize_name(raw_call.get("name"))
        return [name] if name else []

    raw_names = metadata.get("raw_tool_names")
    if isinstance(raw_names, list):
        return [name for value in raw_names if (name := _normalize_name(value))]
    return []


def _response_tool_names(model_response: Any) -> list[str]:
    text = str(model_response or "")
    names = [_normalize_name(value) for value in _FUNCTION_OPEN_RE.findall(text)]
    if names:
        return [name for name in names if name]
    return [
        name
        for value in _TOOL_CALL_NAME_RE.findall(text)
        if (name := _normalize_name(value))
    ]


def _intended_name(step: dict[str, Any]) -> tuple[str, str]:
    """Return ``(name, diagnostic_label)`` for a failed/empty action."""
    names = _raw_tool_names(step)
    if not names:
        names = _response_tool_names(step.get("model_response", ""))
    if not names:
        return "", OTHER_NO_ACTION
    if len(names) != 1:
        detail = ",".join(names[:5])
        suffix = ",..." if len(names) > 5 else ""
        return "", f"{OTHER_MULTIPLE_ACTIONS}:{detail}{suffix}"
    return names[0], names[0]


def _recorded_action_name(step: dict[str, Any]) -> str:
    action = step.get("action")
    if isinstance(action, dict):
        return _normalize_name(action.get("name"))
    if isinstance(action, str):
        return _normalize_name(action)
    return ""


def classify_step(step: dict[str, Any]) -> tuple[str, str]:
    """Classify a turn by intended tool, independently of parse success."""
    raw = _recorded_action_name(step)
    diagnostic = raw
    if not raw:
        raw, diagnostic = _intended_name(step)
    if not raw:
        return "other", diagnostic
    return (raw if raw in CODEFLOW_TOOLS else "other"), raw


def classify_step_strict(step: dict[str, Any]) -> tuple[str, str]:
    """Classify only parser-accepted, recorded Codeflow actions."""
    status, _ = parse_status_of(step)
    if status != "ok":
        return "other", STRICT_PARSE_FAILED
    raw = _recorded_action_name(step)
    if not raw:
        return "other", STRICT_EMPTY_ACTION
    return (raw if raw in CODEFLOW_TOOLS else "other"), raw


class StepAcc:
    def __init__(self) -> None:
        self.bucket_counts: dict[int, Counter[str]] = defaultdict(Counter)
        self.strict_bucket_counts: dict[int, Counter[str]] = defaultdict(Counter)
        self.traj_shares: dict[int, list[dict[str, float]]] = defaultdict(list)
        self.parse_counts: dict[int, Counter[str]] = defaultdict(Counter)
        self.parse_reasons: Counter[str] = Counter()
        self.validation_counts: dict[int, Counter[str]] = defaultdict(Counter)
        self.validation_error_tools: Counter[str] = Counter()
        self.validation_errors_by_tool: dict[str, Counter[str]] = defaultdict(Counter)
        self.other_breakdown: Counter[str] = Counter()
        self.strict_other_breakdown: Counter[str] = Counter()
        self.policy_violations: Counter[str] = Counter()
        self.trajectories: Counter[int] = Counter()
        self.total_steps: Counter[int] = Counter()


@dataclass
class FileStats:
    mode: str
    step_idx: int = -1
    bucket_counts: Counter[str] = field(default_factory=Counter)
    strict_bucket_counts: Counter[str] = field(default_factory=Counter)
    parse_counts: Counter[str] = field(default_factory=Counter)
    parse_reasons: Counter[str] = field(default_factory=Counter)
    validation_counts: Counter[str] = field(default_factory=Counter)
    validation_error_tools: Counter[str] = field(default_factory=Counter)
    validation_errors_by_tool: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    other_breakdown: Counter[str] = field(default_factory=Counter)
    strict_other_breakdown: Counter[str] = field(default_factory=Counter)
    policy_violations: Counter[str] = field(default_factory=Counter)
    traj_share: dict[str, float] | None = None


@dataclass
class CacheDiagnostics:
    enabled: bool
    path: str | None = None
    fast_path: bool = False
    hits: int = 0
    misses: int = 0
    invalidated: int = 0
    removed: int = 0
    directory_changed_during_scan: bool = False
    load_error: str | None = None
    write_error: str | None = None


@dataclass
class CachedAggregation:
    accs: dict[str, StepAcc]
    mode_counts: dict[str, int]
    scanned: int
    skipped: int
    files_found: int
    cache: CacheDiagnostics


def _process_file(path: str, mode_filter: str) -> FileStats | None:
    try:
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
    except Exception:
        return None

    mode = str(document.get("mode") or "unknown")
    result = FileStats(mode=mode)
    if mode_filter != "all" and mode != mode_filter:
        return result

    step_value = document.get("global_step")
    lifecycle = document.get("rollout_lifecycle")
    if step_value is None and isinstance(lifecycle, dict):
        step_value = lifecycle.get("dispatch_step")
    try:
        result.step_idx = int(step_value)
    except (TypeError, ValueError):
        result.step_idx = -1

    trajectory = document.get("trajectory") or {}
    steps = trajectory.get("steps") or []
    if not isinstance(steps, list):
        return result

    for step in steps:
        if not isinstance(step, dict):
            continue
        bucket, raw = classify_step(step)
        result.bucket_counts[bucket] += 1
        if bucket == "other":
            result.other_breakdown[raw] += 1

        strict_bucket, strict_raw = classify_step_strict(step)
        result.strict_bucket_counts[strict_bucket] += 1
        if strict_bucket == "other":
            result.strict_other_breakdown[strict_raw] += 1

        status, reason = parse_status_of(step)
        result.parse_counts[status] += 1
        if status == "parse_error":
            result.parse_reasons[reason or "<empty_reason>"] += 1

        validation_status, validation_error = validation_status_of(step)
        result.validation_counts[validation_status] += 1
        if validation_status == "error":
            validation_tool = _validation_tool_name(step)
            result.validation_error_tools[validation_tool] += 1
            result.validation_errors_by_tool[validation_tool][
                validation_error or "<empty_error>"
            ] += 1

        event = _action_event(step)
        violations = event.get("policy_violations")
        if isinstance(violations, list):
            result.policy_violations.update(
                str(violation)
                for violation in violations
                if isinstance(violation, str) and violation
            )

    total = sum(result.bucket_counts.values())
    if total:
        result.traj_share = {bucket: result.bucket_counts[bucket] / total for bucket in BUCKETS}
    return result


def _serialize_file_stats(result: FileStats | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "mode": result.mode,
        "step_idx": result.step_idx,
        "bucket_counts": dict(result.bucket_counts),
        "strict_bucket_counts": dict(result.strict_bucket_counts),
        "parse_counts": dict(result.parse_counts),
        "parse_reasons": dict(result.parse_reasons),
        "validation_counts": dict(result.validation_counts),
        "validation_error_tools": dict(result.validation_error_tools),
        "validation_errors_by_tool": {
            tool: dict(errors)
            for tool, errors in result.validation_errors_by_tool.items()
        },
        "other_breakdown": dict(result.other_breakdown),
        "strict_other_breakdown": dict(result.strict_other_breakdown),
        "policy_violations": dict(result.policy_violations),
        "traj_share": result.traj_share,
    }


def _deserialize_file_stats(payload: Any) -> FileStats | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("cached file stats must be an object or null")
    return FileStats(
        mode=str(payload.get("mode") or "unknown"),
        step_idx=int(payload.get("step_idx", -1)),
        bucket_counts=Counter(payload.get("bucket_counts") or {}),
        strict_bucket_counts=Counter(payload.get("strict_bucket_counts") or {}),
        parse_counts=Counter(payload.get("parse_counts") or {}),
        parse_reasons=Counter(payload.get("parse_reasons") or {}),
        validation_counts=Counter(payload.get("validation_counts") or {}),
        validation_error_tools=Counter(payload.get("validation_error_tools") or {}),
        validation_errors_by_tool=defaultdict(
            Counter,
            {
                str(tool): Counter(errors)
                for tool, errors in (payload.get("validation_errors_by_tool") or {}).items()
                if isinstance(errors, dict)
            },
        ),
        other_breakdown=Counter(payload.get("other_breakdown") or {}),
        strict_other_breakdown=Counter(payload.get("strict_other_breakdown") or {}),
        policy_violations=Counter(payload.get("policy_violations") or {}),
        traj_share=(
            {str(key): float(value) for key, value in payload["traj_share"].items()}
            if isinstance(payload.get("traj_share"), dict)
            else None
        ),
    )


def _default_cache_path(rollout_dir: str) -> str:
    return os.path.abspath(os.path.normpath(rollout_dir)) + CACHE_SUFFIX


def _default_statistics_path(rollout_dir: str) -> str:
    return os.path.abspath(os.path.normpath(rollout_dir)) + STATISTICS_SUFFIX


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


def _write_cache(cache_path: str, payload: dict[str, Any]) -> str | None:
    parent = os.path.dirname(os.path.abspath(cache_path)) or "."
    try:
        os.makedirs(parent, exist_ok=True)
        descriptor, tmp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(cache_path)}.", suffix=".tmp", dir=parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, cache_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _write_text_atomic(path: str, content: str) -> str | None:
    parent = os.path.dirname(os.path.abspath(path)) or "."
    try:
        os.makedirs(parent, exist_ok=True)
        descriptor, tmp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _file_signature(path: str) -> dict[str, int] | None:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _filter_result(result: FileStats | None, mode_filter: str) -> FileStats | None:
    if result is None or mode_filter == "all" or result.mode == mode_filter:
        return result
    # Preserve the file's mode for mode_counts while matching the historical
    # behavior of _process_file(path, mode_filter): non-selected files have no
    # action statistics.
    return FileStats(mode=result.mode)


def _parse_paths(paths: list[str], workers: int) -> list[FileStats | None]:
    process = partial(_process_file, mode_filter="all")
    if workers == 1:
        return list(map(process, paths))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(process, paths))


def aggregate_rollout_dir(
    rollout_dir: str,
    mode_filter: str,
    workers: int = DEFAULT_WORKERS,
    *,
    cache_path: str | None = None,
    use_cache: bool = True,
    rebuild_cache: bool = False,
) -> CachedAggregation:
    """Aggregate a rollout directory with an incremental per-file cache.

    The directory mtime fast path intentionally relies on RLLM's atomic rollout
    writer. A manual in-place content edit does not change directory mtime and
    therefore requires ``rebuild_cache=True``.
    """
    if workers < 1:
        raise ValueError("workers must be >= 1")
    rollout_dir = os.path.abspath(rollout_dir)
    cache_path = os.path.abspath(cache_path or _default_cache_path(rollout_dir))
    diagnostics = CacheDiagnostics(enabled=use_cache, path=cache_path if use_cache else None)

    try:
        directory_mtime_ns = os.stat(rollout_dir).st_mtime_ns
    except OSError as exc:
        raise ValueError(f"rollout dir is not readable: {rollout_dir}: {exc}") from exc

    cache: dict[str, Any] | None = None
    if use_cache and not rebuild_cache:
        cache, diagnostics.load_error = _load_cache(cache_path, rollout_dir)

    if cache is not None and cache.get("directory_mtime_ns") == directory_mtime_ns:
        try:
            cached_results = [
                _deserialize_file_stats(entry.get("stats"))
                for entry in cache["entries"].values()
                if isinstance(entry, dict)
            ]
        except Exception as exc:
            diagnostics.load_error = f"{type(exc).__name__}: {exc}"
        else:
            diagnostics.fast_path = True
            diagnostics.hits = len(cached_results)
            filtered = (_filter_result(result, mode_filter) for result in cached_results)
            accs, mode_counts, scanned, skipped = _merge_results(filtered)
            return CachedAggregation(
                accs=accs,
                mode_counts=mode_counts,
                scanned=scanned,
                skipped=skipped,
                files_found=len(cached_results),
                cache=diagnostics,
            )

    files = sorted(glob(os.path.join(rollout_dir, "*.json")))
    files = [
        path
        for path in files
        if ".tmp" not in os.path.basename(path) and os.path.abspath(path) != cache_path
    ]
    old_entries = cache.get("entries", {}) if cache is not None else {}
    if workers == 1:
        signatures = list(map(_file_signature, files))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            signatures = list(executor.map(_file_signature, files))
    current: list[tuple[str, str, dict[str, int] | None]] = []
    for path, signature in zip(files, signatures, strict=True):
        key = os.path.relpath(path, rollout_dir)
        current.append((key, path, signature))

    results_by_key: dict[str, FileStats | None] = {}
    entries: dict[str, Any] = {}
    paths_to_parse: list[str] = []
    keys_to_parse: list[str] = []
    signatures_to_parse: list[dict[str, int] | None] = []
    for key, path, signature in current:
        old = old_entries.get(key)
        if (
            not rebuild_cache
            and isinstance(old, dict)
            and signature is not None
            and old.get("signature") == signature
        ):
            try:
                results_by_key[key] = _deserialize_file_stats(old.get("stats"))
            except Exception:
                diagnostics.invalidated += 1
            else:
                diagnostics.hits += 1
                entries[key] = old
                continue
        elif isinstance(old, dict):
            diagnostics.invalidated += 1
        paths_to_parse.append(path)
        keys_to_parse.append(key)
        signatures_to_parse.append(signature)

    parsed = _parse_paths(paths_to_parse, workers)
    diagnostics.misses = len(parsed)
    for key, signature, result in zip(keys_to_parse, signatures_to_parse, parsed, strict=True):
        results_by_key[key] = result
        entries[key] = {
            "signature": signature,
            "stats": _serialize_file_stats(result),
        }

    diagnostics.removed = len(set(old_entries) - set(entries))
    ordered_results = [results_by_key[key] for key, _, _ in current]
    filtered = (_filter_result(result, mode_filter) for result in ordered_results)
    accs, mode_counts, scanned, skipped = _merge_results(filtered)

    if use_cache:
        # Capture the timestamp after scanning. The default cache is a sibling
        # of rollout_dir, so its atomic replacement cannot invalidate this
        # timestamp. A custom cache inside rollout_dir remains correct but will
        # miss the no-enumeration fast path on the following run.
        final_directory_mtime_ns = os.stat(rollout_dir).st_mtime_ns
        diagnostics.directory_changed_during_scan = final_directory_mtime_ns != directory_mtime_ns
        diagnostics.write_error = _write_cache(
            cache_path,
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "stats_version": CACHE_STATS_VERSION,
                "rollout_dir": os.path.realpath(rollout_dir),
                # If a live training run published another rollout while this
                # scan was in progress, do not claim the cached file list is a
                # complete directory snapshot. Per-file entries remain useful,
                # and the next invocation performs another incremental scan.
                "directory_mtime_ns": (
                    final_directory_mtime_ns
                    if not diagnostics.directory_changed_during_scan
                    else -1
                ),
                "entries": entries,
            },
        )

    return CachedAggregation(
        accs=accs,
        mode_counts=mode_counts,
        scanned=scanned,
        skipped=skipped,
        files_found=len(files),
        cache=diagnostics,
    )


def _merge_results(
    results: Iterable[FileStats | None],
) -> tuple[dict[str, StepAcc], dict[str, int], int, int]:
    accs: dict[str, StepAcc] = defaultdict(StepAcc)
    mode_counts: Counter[str] = Counter()
    scanned = 0
    skipped = 0

    for result in results:
        if result is None:
            skipped += 1
            continue
        scanned += 1
        mode_counts[result.mode] += 1
        if not result.bucket_counts:
            continue

        acc = accs[result.mode]
        acc.trajectories[result.step_idx] += 1
        acc.total_steps[result.step_idx] += sum(result.bucket_counts.values())
        acc.bucket_counts[result.step_idx].update(result.bucket_counts)
        acc.strict_bucket_counts[result.step_idx].update(result.strict_bucket_counts)
        acc.parse_counts[result.step_idx].update(result.parse_counts)
        acc.parse_reasons.update(result.parse_reasons)
        acc.validation_counts[result.step_idx].update(result.validation_counts)
        acc.validation_error_tools.update(result.validation_error_tools)
        for tool, errors in result.validation_errors_by_tool.items():
            acc.validation_errors_by_tool[tool].update(errors)
        acc.other_breakdown.update(result.other_breakdown)
        acc.strict_other_breakdown.update(result.strict_other_breakdown)
        acc.policy_violations.update(result.policy_violations)
        if result.traj_share is not None:
            acc.traj_shares[result.step_idx].append(result.traj_share)

    return accs, dict(mode_counts), scanned, skipped


def aggregate(
    files: list[str], mode_filter: str, workers: int = DEFAULT_WORKERS
) -> tuple[dict[str, StepAcc], dict[str, int], int, int]:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    process = partial(_process_file, mode_filter=mode_filter)
    if workers == 1:
        return _merge_results(map(process, files))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return _merge_results(executor.map(process, files))


def _pct(num: int, den: int) -> str:
    return f"{100.0 * num / den:5.1f}%" if den else "  -  "


def _action_header() -> list[str]:
    return ["step", "trajs", "turns", *BUCKETS]


def _action_widths() -> list[int]:
    return [6, 6, 7, *[max(14, len(bucket) + 8) for bucket in BUCKETS]]


def _print_row(values: list[str], widths: list[int]) -> None:
    print("  " + " | ".join(value.rjust(width) for value, width in zip(values, widths, strict=True)))


def print_action_tables(acc: StepAcc, mode: str, macro: bool) -> list[list[str]]:
    steps = sorted(step for step in acc.bucket_counts if step >= 0)
    header = _action_header()
    widths = _action_widths()
    print(f"\n=== TABLE 1 — Intended tool distribution per global_step (pooled)  [mode={mode}] ===")
    _print_row(header, widths)
    print("  " + "-+-".join("-" * width for width in widths))

    csv_rows = [[
        "mode", "global_step", "trajectories", "turns",
        *[f"{bucket}_count" for bucket in BUCKETS],
        *[f"{bucket}_pct" for bucket in BUCKETS],
        *[f"strict_{bucket}_count" for bucket in BUCKETS],
        *[f"strict_{bucket}_pct" for bucket in BUCKETS],
    ]]
    for step in steps:
        counts = acc.bucket_counts[step]
        total = sum(counts.values())
        strict = acc.strict_bucket_counts[step]
        strict_total = sum(strict.values())
        _print_row(
            [str(step), str(acc.trajectories[step]), str(total)]
            + [f"{_pct(counts[bucket], total)} ({counts[bucket]})" for bucket in BUCKETS],
            widths,
        )
        csv_rows.append(
            [mode, str(step), str(acc.trajectories[step]), str(total)]
            + [str(counts[bucket]) for bucket in BUCKETS]
            + [f"{counts[bucket] / total if total else 0:.4f}" for bucket in BUCKETS]
            + [str(strict[bucket]) for bucket in BUCKETS]
            + [f"{strict[bucket] / strict_total if strict_total else 0:.4f}" for bucket in BUCKETS]
        )

    if macro:
        print(f"\n--- TABLE 1b — Intended tool distribution (macro trajectory mean)  [mode={mode}] ---")
        _print_row(header, widths)
        print("  " + "-+-".join("-" * width for width in widths))
        for step in steps:
            shares = acc.traj_shares[step]
            count = len(shares)
            means = {
                bucket: (sum(share[bucket] for share in shares) / count if count else 0.0)
                for bucket in BUCKETS
            }
            _print_row(
                [str(step), str(count), str(acc.total_steps[step])]
                + [f"{100 * means[bucket]:5.1f}%" for bucket in BUCKETS],
                widths,
            )

    print(f"\n=== TABLE 2 — Strict parsed tool distribution per global_step  [mode={mode}] ===")
    print(
        "  Named tools require parse_status=ok and a recorded Codeflow action. "
        "This is parse-only: argument validation errors remain under their selected tool; see Tables 5–6."
    )
    _print_row(header, widths)
    print("  " + "-+-".join("-" * width for width in widths))
    for step in steps:
        counts = acc.strict_bucket_counts[step]
        total = sum(counts.values())
        _print_row(
            [str(step), str(acc.trajectories[step]), str(total)]
            + [f"{_pct(counts[bucket], total)} ({counts[bucket]})" for bucket in BUCKETS],
            widths,
        )

    intended = Counter[str]()
    strict = Counter[str]()
    for step in steps:
        intended.update(acc.bucket_counts[step])
        strict.update(acc.strict_bucket_counts[step])
    intended_total = sum(intended.values())
    strict_total = sum(strict.values())
    print(f"\n=== TABLE 3 — Overall intended vs strict distribution  [mode={mode}] ===")
    print(f"  {'bucket':<20} {'intended':>22} | {'strict':>22}")
    print("  " + "-" * 68)
    for bucket in BUCKETS:
        left = f"{_pct(intended[bucket], intended_total)} ({intended[bucket]})"
        right = f"{_pct(strict[bucket], strict_total)} ({strict[bucket]})"
        print(f"  {bucket:<20} {left:>22} | {right:>22}")
    return csv_rows


def print_parse_tables(acc: StepAcc, mode: str) -> list[list[str]]:
    steps = sorted(step for step in acc.parse_counts if step >= 0)
    observed: list[str] = []
    for step in steps:
        for status in acc.parse_counts[step]:
            if status not in observed:
                observed.append(status)
    statuses = [status for status in ("ok", "parse_error", "not_parsed") if status in observed]
    statuses += [status for status in observed if status not in statuses]

    print(f"\n=== TABLE 4 — Parse status per global_step  [mode={mode}] ===")
    widths = [6, 8, *[max(12, len(status) + 2) for status in statuses]]
    _print_row(["step", "turns", *statuses], widths)
    print("  " + "-+-".join("-" * width for width in widths))
    validation_observed: list[str] = []
    for step in steps:
        for status in acc.validation_counts[step]:
            if status not in validation_observed:
                validation_observed.append(status)
    validation_statuses = [
        status
        for status in ("ok", "error", "skipped", "<missing>")
        if status in validation_observed
    ]
    validation_statuses += [
        status for status in validation_observed if status not in validation_statuses
    ]
    csv_rows = [[
        "mode",
        "global_step",
        "turns",
        *statuses,
        *[f"{status}_pct" for status in statuses],
        "validation_turns",
        *[f"validation_{status}_count" for status in validation_statuses],
        *[f"validation_{status}_pct" for status in validation_statuses],
    ]]
    for step in steps:
        counts = acc.parse_counts[step]
        total = sum(counts.values())
        validation_counts = acc.validation_counts[step]
        validation_total = sum(validation_counts.values())
        _print_row([str(step), str(total), *[_pct(counts[status], total) for status in statuses]], widths)
        csv_rows.append(
            [mode, str(step), str(total)]
            + [str(counts[status]) for status in statuses]
            + [f"{counts[status] / total if total else 0:.4f}" for status in statuses]
            + [str(validation_total)]
            + [str(validation_counts[status]) for status in validation_statuses]
            + [
                f"{validation_counts[status] / validation_total if validation_total else 0:.4f}"
                for status in validation_statuses
            ]
        )

    total_errors = sum(acc.parse_reasons.values())
    print(f"\n=== TABLE 4b — Parse-error reasons  [mode={mode}] ===")
    if not total_errors:
        print("  (no parse errors)")
    else:
        for reason, count in acc.parse_reasons.most_common():
            print(f"  {reason:<40} {_pct(count, total_errors)}  ({count})")

    print(f"\n=== TABLE 5 — Validation status per global_step  [mode={mode}] ===")
    validation_widths = [6, 8, *[max(12, len(status) + 2) for status in validation_statuses]]
    _print_row(["step", "turns", *validation_statuses], validation_widths)
    print("  " + "-+-".join("-" * width for width in validation_widths))
    for step in steps:
        counts = acc.validation_counts[step]
        total = sum(counts.values())
        _print_row(
            [str(step), str(total), *[_pct(counts[status], total) for status in validation_statuses]],
            validation_widths,
        )

    validation_error_total = sum(acc.validation_error_tools.values())
    print(f"\n=== TABLE 6 — Validation errors by selected tool  [mode={mode}] ===")
    if not validation_error_total:
        print("  (no validation errors)")
    else:
        for tool, count in acc.validation_error_tools.most_common():
            print(f"  {tool:<40} {_pct(count, validation_error_total)}  ({count})")
            for reason, reason_count in acc.validation_errors_by_tool[tool].most_common(10):
                print(f"    {reason:<38} {_pct(reason_count, count)}  ({reason_count})")

    total_other = sum(acc.other_breakdown.values())
    print(f"\n=== TABLE 7 — Intended 'other' composition  [mode={mode}] ===")
    if not total_other:
        print("  (other bucket is empty)")
    else:
        for raw, count in acc.other_breakdown.most_common(20):
            print(f"  {raw:<40} {_pct(count, total_other)}  ({count})")

    total_strict_other = sum(acc.strict_other_breakdown.values())
    print(f"\n=== TABLE 8 — Strict 'other' composition  [mode={mode}] ===")
    if not total_strict_other:
        print("  (strict other bucket is empty)")
    else:
        for raw, count in acc.strict_other_breakdown.most_common(20):
            print(f"  {raw:<40} {_pct(count, total_strict_other)}  ({count})")

    total_violations = sum(acc.policy_violations.values())
    print(f"\n=== TABLE 9 — Action policy violations  [mode={mode}] ===")
    if not total_violations:
        print("  (no policy violations)")
    else:
        for violation, count in acc.policy_violations.most_common():
            print(f"  {violation:<40} {_pct(count, total_violations)}  ({count})")
    return csv_rows


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--workers must be an integer >= 1") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("--workers must be an integer >= 1")
    return parsed


def _render_report(aggregation: CachedAggregation, args: argparse.Namespace) -> str:
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        accs = aggregation.accs
        mode_counts = aggregation.mode_counts
        scanned = aggregation.scanned
        skipped = aggregation.skipped
        print("=" * 100)
        print("CODEFLOW ROLLOUT TOOL-DISTRIBUTION REPORT")
        print("=" * 100)
        print(f"rollout dir : {args.rollout_dir}")
        print(
            f"files found : {aggregation.files_found}   parsed OK: {scanned}   "
            f"skipped(malformed): {skipped}"
        )
        cache = aggregation.cache
        if cache.enabled:
            cache_mode = "directory-fast-path" if cache.fast_path else "incremental-scan"
            print(
                f"cache       : {cache_mode}   hits: {cache.hits}   misses: {cache.misses}   "
                f"invalidated: {cache.invalidated}   removed: {cache.removed}"
            )
            print(f"cache file  : {cache.path}")
            if cache.load_error:
                print(f"cache read  : ignored ({cache.load_error})")
            if cache.write_error:
                print(f"cache write : failed ({cache.write_error})")
            if cache.directory_changed_during_scan:
                print(
                    "cache note  : rollout directory changed during the scan; "
                    "the next run will rescan incrementally"
                )
        else:
            print("cache       : disabled")
        print("modes present: " + ", ".join(f"{mode}={count}" for mode, count in sorted(mode_counts.items())))
        print(f"reporting mode: {args.mode}")
        print("buckets: search | read_file | edit_file | execute_bash | submit | other")
        print(
            "Table 1 recovers intended native tools on parse failures; "
            "Table 2 requires a parsed recorded action and does not imply validation success."
        )

        modes = [args.mode] if args.mode != "all" else sorted(accs)
        action_csv: list[list[str]] = []
        parse_csv: list[list[str]] = []
        for mode in modes:
            acc = accs.get(mode)
            if acc is None or not acc.bucket_counts:
                print(f"\n(no data for mode={mode})")
                continue
            action_rows = print_action_tables(acc, mode, args.macro)
            parse_rows = print_parse_tables(acc, mode)
            if not action_csv:
                action_csv = action_rows
                parse_csv = parse_rows
            else:
                action_csv.extend(action_rows[1:])
                parse_csv.extend(parse_rows[1:])

        if args.csv:
            import csv

            with open(f"{args.csv}.actions.csv", "w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(action_csv)
            with open(f"{args.csv}.parse.csv", "w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(parse_csv)
            print(f"\nCSV written: {args.csv}.actions.csv , {args.csv}.parse.csv")
    return output.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rollout_dir", nargs="?", default=DEFAULT_ROLLOUT_DIR, help="Directory of rollout *.json files.")
    parser.add_argument("--mode", default="train", choices=["train", "val", "all"], help="Rollout mode to report.")
    parser.add_argument("--macro", action="store_true", help="Report mean per-trajectory shares in addition to pooled shares.")
    parser.add_argument("--csv", default=None, help="Write <PREFIX>.actions.csv and <PREFIX>.parse.csv.")
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=DEFAULT_WORKERS,
        help=f"Concurrent rollout readers (default: {DEFAULT_WORKERS}; use 1 for serial).",
    )
    parser.add_argument(
        "--cache-file",
        default=None,
        help=(
            "Incremental cache path. Defaults to a sibling named "
            "<ROLLOUT_DIR>.codeflow_action_distribution.cache.json."
        ),
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help=(
            "Statistics report path. Defaults to "
            "<ROLLOUT_DIR>.codeflow_action_distribution.statistics."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable reading and writing the incremental rollout cache.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Ignore cached entries, reparse every rollout, and replace the cache.",
    )
    args = parser.parse_args()

    if args.no_cache and args.rebuild_cache:
        parser.error("--no-cache and --rebuild-cache cannot be used together")

    if not os.path.isdir(args.rollout_dir):
        print(f"ERROR: rollout dir not found: {args.rollout_dir}", file=sys.stderr)
        return 2

    aggregation = aggregate_rollout_dir(
        args.rollout_dir,
        args.mode,
        args.workers,
        cache_path=args.cache_file,
        use_cache=not args.no_cache,
        rebuild_cache=args.rebuild_cache,
    )
    if not aggregation.files_found:
        print(f"ERROR: no *.json rollout files in {args.rollout_dir}", file=sys.stderr)
        return 2

    output_path = os.path.abspath(args.output_file or _default_statistics_path(args.rollout_dir))
    write_error = _write_text_atomic(output_path, _render_report(aggregation, args))
    if write_error:
        print(f"ERROR: failed to write statistics file {output_path}: {write_error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
