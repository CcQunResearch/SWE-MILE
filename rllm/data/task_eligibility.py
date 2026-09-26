"""Revision-bound task eligibility manifests for sandbox datasets.

The canonical materialized dataset remains immutable. Expensive runtime audits
write a sidecar manifest, and training filters tasks only after proving that
the manifest covers the exact dataset contract it is about to schedule.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

ELIGIBILITY_MANIFEST_SCHEMA_VERSION = 1
IMAGE_AVAILABILITY_MANIFEST_SCHEMA_VERSION = 2
BASELINE_ELIGIBILITY_MANIFEST_SCHEMA_VERSION = 2


def _task_id(task: Any) -> str:
    if isinstance(task, Mapping):
        value = task.get("id")
    else:
        value = getattr(task, "id", None)
    task_id = str(value or "")
    if not task_id:
        raise ValueError("eligibility dataset contains a task without an id")
    return task_id


def _task_metadata(task: Any) -> Mapping[str, Any]:
    if isinstance(task, Mapping):
        return task
    metadata = getattr(task, "metadata", None)
    return metadata if isinstance(metadata, Mapping) else {}


def task_contract(task: Any) -> dict[str, Any]:
    """Return the stable fields that determine sandbox/verifier eligibility."""

    metadata = _task_metadata(task)
    environment = (
        metadata.get("environment")
        if isinstance(metadata.get("environment"), Mapping)
        else {}
    )
    env_vars = (
        metadata.get("env_vars")
        if isinstance(metadata.get("env_vars"), Mapping)
        else environment.get("env", {})
    )
    if not isinstance(env_vars, Mapping):
        env_vars = {}
    return {
        "id": _task_id(task),
        "source_revision": str(metadata.get("source_revision") or ""),
        "docker_image": str(
            metadata.get("docker_image")
            or environment.get("docker_image")
            or ""
        ),
        "language": str(metadata.get("language") or ""),
        "result_parser": str(metadata.get("result_parser") or ""),
        "log_parser": str(metadata.get("log_parser") or ""),
        "parser_vendor_revision": str(
            metadata.get("parser_vendor_revision") or ""
        ),
        "parser_vendor_sha256": str(
            metadata.get("parser_vendor_sha256") or ""
        ),
        "result_adapter_version": metadata.get("result_adapter_version"),
        "sandbox_proxy_url": str(
            env_vars.get("https_proxy") or env_vars.get("HTTPS_PROXY") or ""
        ),
    }


def baseline_runtime_contract_sha256(
    tasks: Iterable[Any],
    *,
    backend: str,
    sandbox_cpus: int,
    sandbox_memory_mb: int,
    image_resolver: Callable[[str], str],
) -> str:
    """Fingerprint the exact baseline environment required by training."""

    payload = {
        "backend": str(backend),
        "sandbox_cpus": int(sandbox_cpus),
        "sandbox_memory_mb": int(sandbox_memory_mb),
        "dataset_contract_sha256": dataset_contract_sha256(tasks),
        "image_resolution_contract_sha256": image_resolution_contract_sha256(
            tasks, image_resolver
        ),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def dataset_contract_sha256(tasks: Iterable[Any]) -> str:
    contracts = [task_contract(task) for task in tasks]
    payload = json.dumps(
        contracts,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def image_resolution_contract_sha256(
    tasks: Iterable[Any],
    resolver: Callable[[str], str],
) -> str:
    """Fingerprint the task-to-runtime-image mapping used by a backend."""

    mappings = []
    for task in tasks:
        metadata = _task_metadata(task)
        environment = (
            metadata.get("environment")
            if isinstance(metadata.get("environment"), Mapping)
            else {}
        )
        source_image = str(
            environment.get("docker_image")
            or metadata.get("docker_image")
            or ""
        )
        mappings.append(
            {
                "id": _task_id(task),
                "resolved_image": resolver(source_image),
            }
        )
    payload = json.dumps(
        mappings,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_eligibility_manifest(
    path: str | Path,
    *,
    dataset: str,
    split: str,
    tasks: list[Any],
) -> dict[str, Any]:
    manifest_path = Path(path).expanduser()
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"could not read task eligibility manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError("task eligibility manifest must contain an object")
    if value.get("schema_version") != ELIGIBILITY_MANIFEST_SCHEMA_VERSION:
        raise ValueError("task eligibility manifest schema mismatch")
    if value.get("status") != "complete":
        raise ValueError("task eligibility manifest is not complete")
    if value.get("dataset") != dataset or value.get("split") != split:
        raise ValueError(
            "task eligibility manifest dataset/split mismatch: "
            f"expected {dataset}/{split}, got "
            f"{value.get('dataset')}/{value.get('split')}"
        )
    expected_fingerprint = dataset_contract_sha256(tasks)
    if value.get("dataset_contract_sha256") != expected_fingerprint:
        raise ValueError("task eligibility manifest dataset contract is stale")

    eligible = value.get("eligible_task_ids")
    excluded = value.get("excluded_tasks")
    if (
        not isinstance(eligible, list)
        or any(not isinstance(task_id, str) or not task_id for task_id in eligible)
        or len(eligible) != len(set(eligible))
    ):
        raise ValueError("task eligibility manifest has invalid eligible_task_ids")
    if not isinstance(excluded, dict) or any(
        not isinstance(task_id, str)
        or not task_id
        or not isinstance(reason, str)
        or not reason
        for task_id, reason in excluded.items()
    ):
        raise ValueError("task eligibility manifest has invalid excluded_tasks")

    dataset_ids = [_task_id(task) for task in tasks]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("eligibility dataset contains duplicate task ids")
    eligible_ids = set(eligible)
    excluded_ids = set(excluded)
    if eligible_ids & excluded_ids:
        raise ValueError("task eligibility manifest overlaps eligible and excluded ids")
    if eligible_ids | excluded_ids != set(dataset_ids):
        raise ValueError("task eligibility manifest does not cover the full dataset")
    return value


def filter_tasks_with_eligibility_manifest(
    tasks: list[Any],
    path: str | Path,
    *,
    dataset: str,
    split: str,
) -> tuple[list[Any], dict[str, str], dict[str, Any]]:
    manifest = load_eligibility_manifest(
        path,
        dataset=dataset,
        split=split,
        tasks=tasks,
    )
    eligible_ids = set(manifest["eligible_task_ids"])
    excluded = {str(key): str(value) for key, value in manifest["excluded_tasks"].items()}
    return (
        [task for task in tasks if _task_id(task) in eligible_ids],
        excluded,
        manifest,
    )


def load_baseline_eligibility_manifest(
    path: str | Path,
    *,
    dataset: str,
    split: str,
    tasks: list[Any],
    backend: str,
    sandbox_cpus: int,
    sandbox_memory_mb: int,
    image_resolver: Callable[[str], str],
) -> dict[str, Any]:
    """Load a one-pass baseline audit bound to the shadow resources."""

    manifest_path = Path(path).expanduser()
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"could not read baseline eligibility manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError("baseline eligibility manifest must contain an object")
    if value.get("schema_version") != BASELINE_ELIGIBILITY_MANIFEST_SCHEMA_VERSION:
        raise ValueError("baseline eligibility manifest schema mismatch")
    if value.get("status") != "complete":
        raise ValueError("baseline eligibility manifest is not complete")
    if value.get("dataset") != dataset or value.get("split") != split:
        raise ValueError("baseline eligibility manifest dataset/split mismatch")
    expected_dataset_contract = dataset_contract_sha256(tasks)
    if value.get("dataset_contract_sha256") != expected_dataset_contract:
        raise ValueError("baseline eligibility manifest dataset contract is stale")
    expected_runtime_contract = baseline_runtime_contract_sha256(
        tasks,
        backend=backend,
        sandbox_cpus=sandbox_cpus,
        sandbox_memory_mb=sandbox_memory_mb,
        image_resolver=image_resolver,
    )
    if value.get("runtime_contract_sha256") != expected_runtime_contract:
        raise ValueError(
            "baseline eligibility manifest runtime contract is stale; "
            "rerun the baseline audit with the configured shadow CPU/memory"
        )

    eligible = value.get("eligible_task_ids")
    excluded = value.get("excluded_tasks")
    if (
        not isinstance(eligible, list)
        or any(not isinstance(task_id, str) or not task_id for task_id in eligible)
        or len(eligible) != len(set(eligible))
    ):
        raise ValueError("baseline eligibility manifest has invalid eligible_task_ids")
    if not isinstance(excluded, dict) or any(
        not isinstance(task_id, str)
        or not task_id
        or not isinstance(reason, str)
        or not reason
        for task_id, reason in excluded.items()
    ):
        raise ValueError("baseline eligibility manifest has invalid excluded_tasks")
    dataset_ids = [_task_id(task) for task in tasks]
    eligible_ids = set(eligible)
    excluded_ids = set(excluded)
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("baseline eligibility dataset contains duplicate task ids")
    if eligible_ids & excluded_ids:
        raise ValueError("baseline eligibility manifest overlaps task status sets")
    if eligible_ids | excluded_ids != set(dataset_ids):
        raise ValueError("baseline eligibility manifest does not cover the full dataset")
    return value


def filter_tasks_with_baseline_eligibility_manifest(
    tasks: list[Any],
    path: str | Path,
    *,
    dataset: str,
    split: str,
    backend: str,
    sandbox_cpus: int,
    sandbox_memory_mb: int,
    image_resolver: Callable[[str], str],
    contract_tasks: list[Any] | None = None,
) -> tuple[list[Any], dict[str, str], dict[str, Any]]:
    manifest = load_baseline_eligibility_manifest(
        path,
        dataset=dataset,
        split=split,
        tasks=contract_tasks if contract_tasks is not None else tasks,
        backend=backend,
        sandbox_cpus=sandbox_cpus,
        sandbox_memory_mb=sandbox_memory_mb,
        image_resolver=image_resolver,
    )
    eligible_ids = set(manifest["eligible_task_ids"])
    excluded = {
        str(task_id): str(reason)
        for task_id, reason in manifest["excluded_tasks"].items()
    }
    return (
        [task for task in tasks if _task_id(task) in eligible_ids],
        excluded,
        manifest,
    )


def load_image_availability_manifest(
    path: str | Path,
    *,
    dataset: str,
    split: str,
    tasks: list[Any],
    image_resolver: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Load one complete, revision-bound image availability probe result."""

    manifest_path = Path(path).expanduser()
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"could not read image availability manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError("image availability manifest must contain an object")
    if value.get("schema_version") != IMAGE_AVAILABILITY_MANIFEST_SCHEMA_VERSION:
        raise ValueError("image availability manifest schema mismatch")
    if value.get("status") != "complete":
        raise ValueError("image availability manifest is not complete")
    if value.get("dataset") != dataset or value.get("split") != split:
        raise ValueError(
            "image availability manifest dataset/split mismatch: "
            f"expected {dataset}/{split}, got "
            f"{value.get('dataset')}/{value.get('split')}"
        )
    dataset_contract_matches = (
        value.get("dataset_contract_sha256") == dataset_contract_sha256(tasks)
    )
    resolution_contract = value.get("image_resolution_contract_sha256")
    if not isinstance(resolution_contract, str) or not resolution_contract:
        raise ValueError("image availability manifest has no image resolution contract")
    if (
        image_resolver is not None
        and resolution_contract
        != image_resolution_contract_sha256(tasks, image_resolver)
    ):
        raise ValueError("image availability manifest image mapping is stale")

    available = value.get("available_task_ids")
    missing = value.get("missing_task_ids")
    retryable = value.get("retryable_task_ids")
    for field, task_ids in (
        ("available_task_ids", available),
        ("missing_task_ids", missing),
        ("retryable_task_ids", retryable),
    ):
        if (
            not isinstance(task_ids, list)
            or any(not isinstance(task_id, str) or not task_id for task_id in task_ids)
            or len(task_ids) != len(set(task_ids))
        ):
            raise ValueError(f"image availability manifest has invalid {field}")

    dataset_ids = [_task_id(task) for task in tasks]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("image availability dataset contains duplicate task ids")
    available_ids = set(available)
    missing_ids = set(missing)
    retryable_ids = set(retryable)
    if (
        available_ids & missing_ids
        or available_ids & retryable_ids
        or missing_ids & retryable_ids
    ):
        raise ValueError("image availability manifest task status lists overlap")
    if available_ids | missing_ids | retryable_ids != set(dataset_ids):
        raise ValueError("image availability manifest does not cover the full dataset")
    # Parser/adapter-only materialization upgrades intentionally do not force
    # another 6K-image existence audit.  The manifest remains usable when its
    # exact task coverage and task-to-image mapping are unchanged.
    if not dataset_contract_matches and image_resolver is None:
        raise ValueError("image availability manifest dataset contract is stale")
    return value


def filter_tasks_with_image_availability_manifest(
    tasks: list[Any],
    path: str | Path,
    *,
    dataset: str,
    split: str,
    contract_tasks: list[Any] | None = None,
    image_resolver: Callable[[str], str] | None = None,
) -> tuple[list[Any], dict[str, str], dict[str, Any]]:
    """Remove tasks whose image probe was missing or retryable.

    ``contract_tasks`` may be the original unfiltered dataset when another
    revision-bound eligibility manifest has already narrowed ``tasks``.
    """

    manifest = load_image_availability_manifest(
        path,
        dataset=dataset,
        split=split,
        tasks=contract_tasks if contract_tasks is not None else tasks,
        image_resolver=image_resolver,
    )
    missing_ids = set(manifest["missing_task_ids"])
    retryable_ids = set(manifest["retryable_task_ids"])
    excluded = {
        _task_id(task): (
            "missing_image"
            if _task_id(task) in missing_ids
            else "retryable_error"
        )
        for task in tasks
        if _task_id(task) in missing_ids or _task_id(task) in retryable_ids
    }
    return (
        [
            task
            for task in tasks
            if _task_id(task) not in missing_ids
            and _task_id(task) not in retryable_ids
        ],
        excluded,
        manifest,
    )


__all__ = [
    "BASELINE_ELIGIBILITY_MANIFEST_SCHEMA_VERSION",
    "ELIGIBILITY_MANIFEST_SCHEMA_VERSION",
    "IMAGE_AVAILABILITY_MANIFEST_SCHEMA_VERSION",
    "baseline_runtime_contract_sha256",
    "dataset_contract_sha256",
    "filter_tasks_with_baseline_eligibility_manifest",
    "filter_tasks_with_eligibility_manifest",
    "filter_tasks_with_image_availability_manifest",
    "image_resolution_contract_sha256",
    "load_baseline_eligibility_manifest",
    "load_eligibility_manifest",
    "load_image_availability_manifest",
    "task_contract",
]
