"""DeNovoSWE task-directory adapter for the shared schema-v2 OCI cache."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from rllm.data.minisandbox_cache import (
    DEFAULT_IMAGE_WORKERS,
    OCI_CACHE_SCHEMA_VERSION,
    OCI_PLATFORM,
    materialize_minisandbox_oci_cache,
    validate_minisandbox_oci_cache,
)
from rllm.data.repo_generation_exclusions import DENOVOSWE_EXCLUDED_TASK_IDS


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot read DeNovoSWE materialization manifest: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(
            f"DeNovoSWE materialization manifest is not an object: {path}"
        )
    return value


def _tasks_from_materialization_root(
    tasks_root: str | Path,
) -> tuple[list[Any], dict[str, Any]]:
    from rllm.tasks.loader import _merge_task_toml_metadata
    from rllm.types import Task

    root = Path(tasks_root).expanduser().resolve()
    manifest = _load_object(root / "materialization.json")
    if manifest.get("status") != "complete":
        raise ValueError(f"DeNovoSWE task materialization is incomplete: {root}")
    raw_tasks = manifest.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("DeNovoSWE materialization manifest has no task inventory")
    tasks: list[Task] = []
    raw_ids: set[str] = set()
    for raw in raw_tasks:
        if not isinstance(raw, Mapping):
            raise ValueError(
                "DeNovoSWE materialization task inventory is malformed"
            )
        task_id = str(raw.get("id") or "").strip()
        if not task_id:
            raise ValueError("DeNovoSWE materialization contains an empty task id")
        if task_id in raw_ids:
            raise ValueError(
                f"DeNovoSWE materialization has duplicate task id {task_id!r}"
            )
        raw_ids.add(task_id)
        if task_id in DENOVOSWE_EXCLUDED_TASK_IDS:
            continue
        task_dir = root / task_id
        metadata = _merge_task_toml_metadata(
            task_dir,
            {"source_fingerprint": str(raw.get("row_fingerprint") or "")},
        )
        tasks.append(
            Task(
                id=task_id,
                instruction="",
                metadata=metadata,
                dataset_dir=task_dir,
                sub_dir=None,
            )
        )
    excluded = raw_ids & DENOVOSWE_EXCLUDED_TASK_IDS
    if len(tasks) != len(raw_tasks) - len(excluded):
        raise ValueError(
            "DeNovoSWE MiniSandbox inventory does not match the exclusion contract"
        )
    return tasks, manifest


def materialize_denovoswe_oci_cache(
    *,
    tasks_root: str | Path,
    cache_dir: str | Path,
    image_map_file: str | Path | None = None,
    registry_auth_file: str | Path | None = None,
    max_workers: int = DEFAULT_IMAGE_WORKERS,
    run_command: Any = subprocess.run,
) -> Path:
    """Materialize schedulable DeNovoSWE images using the generic v2 cache."""

    tasks, source_manifest = _tasks_from_materialization_root(tasks_root)
    return materialize_minisandbox_oci_cache(
        dataset="denovoswe",
        cache_dir=cache_dir,
        tasks=tasks,
        source_manifest=source_manifest,
        image_map_file=image_map_file,
        registry_auth_file=registry_auth_file,
        max_workers=max_workers,
        strict_inventory=False,
        run_command=run_command,
    )


def validate_denovoswe_oci_cache(
    cache_dir: str | Path,
    tasks: Iterable[Any],
) -> dict[str, dict[str, Any]]:
    """Validate exact scheduled DeNovoSWE coverage for v1 or v2 caches."""

    return validate_minisandbox_oci_cache(
        cache_dir,
        tasks,
        dataset="denovoswe",
    )


__all__ = [
    "DEFAULT_IMAGE_WORKERS",
    "OCI_CACHE_SCHEMA_VERSION",
    "OCI_PLATFORM",
    "materialize_denovoswe_oci_cache",
    "validate_denovoswe_oci_cache",
]
