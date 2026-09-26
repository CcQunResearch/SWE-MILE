"""Resumable schema-v2 OCI caches for SWE-MiniSandbox.

Workers consume verified, pinned layouts prepared before training/evaluation.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Iterable, Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib

from rllm.data.repo_generation_exclusions import (
    DENOVOSWE_EXCLUDED_TASK_IDS,
    NL2REPO_EXCLUDED_TASK_IDS,
)

OCI_CACHE_SCHEMA_VERSION = 2
OCI_RECORD_SCHEMA_VERSION = 2
OCI_PLATFORM = {"os": "linux", "architecture": "amd64"}
DEFAULT_IMAGE_WORKERS = 8
SKOPEO_COPY_RETRY_TIMES = 5
SKOPEO_COPY_RETRY_DELAY = "2s"
SKOPEO_IMAGE_PARALLEL_COPIES = 1


class MiniSandboxRegistryAuthenticationError(RuntimeError):
    """The physical enterprise image exists behind unavailable credentials."""


@dataclass(frozen=True)
class MiniSandboxDatasetSpec:
    """Immutable materialisation contract for one logical dataset view."""

    name: str
    physical_name: str
    split: str
    cache_dir_name: str
    expected_count: int
    physical_expected_count: int
    languages: frozenset[str] = frozenset()
    excluded_task_ids: frozenset[str] = frozenset()
    required_task_profile: str | None = None
    forbidden_paths: tuple[str, ...] = ("/tests",)
    visible_fixture_paths: tuple[str, ...] = ()


_DATASET_SPECS = (
    MiniSandboxDatasetSpec(
        name="denovoswe",
        physical_name="denovoswe",
        split="train",
        cache_dir_name="denovoswe",
        expected_count=3_637,
        physical_expected_count=3_675,
        excluded_task_ids=frozenset(DENOVOSWE_EXCLUDED_TASK_IDS),
        required_task_profile="repo_generation_denovoswe",
    ),
    MiniSandboxDatasetSpec(
        name="swe-rebench-v2-filtered-verified-python",
        physical_name="swe-rebench-v2-filtered-verified",
        split="train",
        cache_dir_name="swe-rebench-v2-filtered-verified-python",
        expected_count=1_952,
        physical_expected_count=6_272,
        languages=frozenset({"python"}),
    ),
    MiniSandboxDatasetSpec(
        name="swe-rebench-v2-filtered-verified-promatched",
        physical_name="swe-rebench-v2-filtered-verified",
        split="train",
        cache_dir_name="swe-rebench-v2-filtered-verified-promatched",
        expected_count=4_738,
        physical_expected_count=6_272,
        languages=frozenset({"python", "go", "js", "ts"}),
    ),
    MiniSandboxDatasetSpec(
        name="r2egym",
        physical_name="r2egym",
        split="train",
        cache_dir_name="r2egym",
        expected_count=4_578,
        physical_expected_count=4_578,
        visible_fixture_paths=("/r2e_tests", "/testbed/run_tests.sh"),
    ),
    MiniSandboxDatasetSpec(
        name="nl2repo",
        physical_name="nl2repo-bench",
        split="test",
        cache_dir_name="nl2repo-bench",
        expected_count=103,
        physical_expected_count=104,
        excluded_task_ids=frozenset(NL2REPO_EXCLUDED_TASK_IDS),
        required_task_profile="repo_generation_nl2repo",
    ),
    MiniSandboxDatasetSpec(
        name="doc2repo",
        physical_name="beyondswe-doc2repo",
        split="test",
        cache_dir_name="beyondswe-doc2repo",
        expected_count=50,
        physical_expected_count=50,
        required_task_profile="repo_generation_doc2repo",
    ),
    MiniSandboxDatasetSpec(
        name="swebench_verified",
        physical_name="swe_bench_verified",
        split="test",
        cache_dir_name="swe_bench_verified",
        expected_count=500,
        physical_expected_count=500,
    ),
    MiniSandboxDatasetSpec(
        name="swebench_pro_public",
        physical_name="swebench_pro_public",
        split="test",
        cache_dir_name="swebench_pro_public",
        expected_count=731,
        physical_expected_count=731,
    ),
)

MINISANDBOX_DATASET_SPECS: dict[str, MiniSandboxDatasetSpec] = {
    spec.name: spec for spec in _DATASET_SPECS
}
_DATASET_ALIASES = {
    # The frozen training view is covered by the existing Python OCI inventory.
    # Resolve to the same spec so manifests/fingerprints remain reusable.
    "swe-rebench-v2-filtered-verified-python-filternorm": "swe-rebench-v2-filtered-verified-python",
    "nl2repo-bench": "nl2repo",
    "beyondswe-doc2repo": "doc2repo",
    "swe_bench_verified": "swebench_verified",
}


def minisandbox_dataset_spec(
    dataset: str | MiniSandboxDatasetSpec,
) -> MiniSandboxDatasetSpec:
    if isinstance(dataset, MiniSandboxDatasetSpec):
        return dataset
    name = _DATASET_ALIASES.get(str(dataset), str(dataset))
    try:
        return MINISANDBOX_DATASET_SPECS[name]
    except KeyError as exc:
        supported = ", ".join(sorted(MINISANDBOX_DATASET_SPECS))
        raise ValueError(
            f"unsupported MiniSandbox dataset {dataset!r}; expected one of {supported}"
        ) from exc


def default_minisandbox_cache_dir(
    data_root: str | Path,
    dataset: str | MiniSandboxDatasetSpec,
) -> Path:
    spec = minisandbox_dataset_spec(dataset)
    return Path(data_root).expanduser().resolve() / "minisandbox" / spec.cache_dir_name


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _inside(root: Path, relative: str) -> Path | None:
    try:
        candidate = (root / relative).resolve()
        candidate.relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return candidate


def _safe_task_id(value: object) -> str:
    task_id = str(value or "").strip()
    if (
        not task_id
        or task_id in {".", ".."}
        or "/" in task_id
        or "\\" in task_id
        or "\x00" in task_id
        or len(task_id.encode("utf-8")) > 240
    ):
        raise ValueError(f"unsafe MiniSandbox task id: {task_id!r}")
    return task_id


def _load_image_map(path: str | Path | None) -> tuple[dict[str, str], str | None]:
    if path in (None, ""):
        return {}, None
    source = Path(path).expanduser().resolve()
    raw = _load_json_object(source)
    if raw is None:
        raise ValueError(f"invalid MiniSandbox image map: {source}")
    if isinstance(raw.get("images"), dict):
        raw = raw["images"]
    if not isinstance(raw, dict):
        raise ValueError(f"MiniSandbox image map must be a JSON object: {source}")
    result = {
        str(key): str(value)
        for key, value in raw.items()
        if isinstance(value, str) and value.strip()
    }
    return result, _sha256_file(source)




def _candidate_sources(
    image: str,
    image_map: Mapping[str, str],
    spec: MiniSandboxDatasetSpec,
    *,
    task_id: str | None = None,
) -> tuple[str, ...]:
    stripped = image.removeprefix("docker.io/").removeprefix(
        "registry-1.docker.io/"
    )
    mapped = image_map.get(image) or image_map.get(stripped)
    candidates: list[str] = []
    for value in (mapped, image):
        if value and value not in candidates:
            candidates.append(value)
    return tuple(candidates)


def _registry_auth_failure(candidates: tuple[str, ...], errors: Iterable[str]) -> bool:
    combined = "\n".join(errors).casefold()
    return any(marker in combined for marker in (
        "authentication required", "invalid username/password", "unauthorized",
        "access to the resource is denied",
    ))


def _prioritize_incomplete_records(
    inventory: list[CacheTask],
    cache_root: Path,
) -> tuple[list[CacheTask], int]:
    """Dispatch failed/missing records before revalidating completed records."""

    def is_incomplete(task: CacheTask) -> bool:
        record = _load_json_object(
            cache_root / "records" / f"{task.task_id}.json"
        )
        return record is None or record.get("status") != "complete"

    with ThreadPoolExecutor(
        max_workers=min(64, len(inventory)),
        thread_name_prefix="minisandbox-resume-scan",
    ) as executor:
        flags = list(executor.map(is_incomplete, inventory))
    incomplete = [task for task, pending in zip(inventory, flags, strict=True) if pending]
    completed = [task for task, pending in zip(inventory, flags, strict=True) if not pending]
    return incomplete + completed, len(incomplete)


_RESOURCE_IDENTITY_KEYS = frozenset(
    {
        "cpus",
        "cpu",
        "memory_mb",
        "memory_gb",
        "storage_mb",
        "disk_mb",
        "disk_gb",
        "primary_sandbox_cpus",
        "primary_sandbox_memory_mb",
        "shadow_sandbox_cpus",
        "shadow_sandbox_memory_mb",
        "shadow_sandbox_resources",
    }
)


def _without_runtime_resources(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_runtime_resources(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _RESOURCE_IDENTITY_KEYS
            and not str(key).endswith(("_cpus", "_memory_mb", "_storage_mb"))
        }
    if isinstance(value, list | tuple):
        return [_without_runtime_resources(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _task_dir(task: Any) -> Path | None:
    value = getattr(task, "task_dir", None)
    if value is None:
        value = getattr(task, "dataset_dir", None)
    if value is None:
        return None
    return Path(value).expanduser()


def _task_metadata(task: Any) -> Mapping[str, Any]:
    metadata = getattr(task, "metadata", None)
    return metadata if isinstance(metadata, Mapping) else {}


def _task_profile(task: Any) -> str:
    metadata = _task_metadata(task)
    rllm = metadata.get("rllm")
    rllm = rllm if isinstance(rllm, Mapping) else {}
    return str(metadata.get("task_profile") or rllm.get("task_profile") or "")


def _task_image(task: Any) -> str:
    metadata = _task_metadata(task)
    environment = metadata.get("environment")
    environment = environment if isinstance(environment, Mapping) else {}
    return str(
        metadata.get("docker_image") or environment.get("docker_image") or ""
    ).strip()


def _task_workdir(task: Any) -> str | None:
    metadata = _task_metadata(task)
    environment = metadata.get("environment")
    environment = environment if isinstance(environment, Mapping) else {}
    value = metadata.get("workdir") or environment.get("workdir")
    return str(value).strip() if value not in (None, "") else None


_STABLE_SOURCE_KEYS = (
    "source_dataset",
    "source_revision",
    "source_fingerprint",
    "repo_name",
    "repository",
    "repo",
    "commit_hash",
    "base_commit",
    "commit",
    "language",
    "package_name",
    "task_profile",
    "verifier_profile",
    "result_parser",
    "log_parser",
    "parser_vendor_revision",
    "parser_vendor_sha256",
    "result_adapter_version",
)


def _task_file_identity(task: Any) -> dict[str, str]:
    root = _task_dir(task)
    if root is None:
        return {}
    result: dict[str, str] = {}
    task_toml = root / "task.toml"
    if task_toml.is_file():
        try:
            parsed = tomllib.loads(task_toml.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(f"cannot read MiniSandbox task config: {task_toml}") from exc
        result["task.toml"] = _sha256_bytes(
            _canonical_json(_without_runtime_resources(parsed))
        )
    for relative in (
        "environment/Dockerfile",
        "environment/setup.sh",
        "setup.sh",
    ):
        path = root / relative
        if path.is_file():
            result[relative] = _sha256_file(path)
    return result


@dataclass(frozen=True)
class CacheTask:
    task_id: str
    image: str
    task_fingerprint: str
    row_fingerprint: str
    workdir: str | None
    task_profile: str
    has_task_contract_files: bool


def cache_task_from_task(
    task: Any,
    dataset: str | MiniSandboxDatasetSpec,
) -> CacheTask:
    spec = minisandbox_dataset_spec(dataset)
    task_id = _safe_task_id(getattr(task, "id", ""))
    image = _task_image(task)
    if not image:
        raise ValueError(f"{task_id}: docker_image is missing")
    profile = _task_profile(task)
    if spec.required_task_profile and profile != spec.required_task_profile:
        raise ValueError(
            f"{task_id}: expected task profile {spec.required_task_profile!r}, got {profile!r}"
        )
    metadata = _task_metadata(task)
    row_fingerprint = str(metadata.get("source_fingerprint") or "").strip()
    # Repo-generation builders already publish a canonical source fingerprint.
    # Treat it as the complete row identity so a task loaded from the compact
    # on-disk materialization manifest fingerprints identically to the richer
    # registry row used by training/evaluation.
    source_identity = (
        {"source_fingerprint": row_fingerprint}
        if row_fingerprint
        else {
            key: metadata[key]
            for key in _STABLE_SOURCE_KEYS
            if key != "source_fingerprint"
            and key in metadata
            and metadata[key] not in (None, "")
        }
    )
    if not row_fingerprint:
        row_fingerprint = _sha256_bytes(
            _canonical_json(
                {
                    "task_id": task_id,
                    "image": image,
                    "source": source_identity,
                }
            )
        )
    task_files = _task_file_identity(task)
    task_fingerprint = _sha256_bytes(
        _canonical_json(
            {
                "logical_dataset": spec.name,
                "physical_dataset": spec.physical_name,
                "split": spec.split,
                "task_id": task_id,
                "image": image,
                "workdir": _task_workdir(task),
                "task_profile": profile,
                "source": source_identity,
                "files": task_files,
            }
        )
    )
    return CacheTask(
        task_id=task_id,
        image=image,
        task_fingerprint=task_fingerprint,
        row_fingerprint=row_fingerprint,
        workdir=_task_workdir(task),
        task_profile=profile,
        has_task_contract_files=bool(task_files),
    )


def _load_registry_inventory(
    spec: MiniSandboxDatasetSpec,
    *,
    strict_inventory: bool,
    inventory_workers: int,
) -> tuple[list[CacheTask], dict[str, Any]]:
    from rllm.data import DatasetRegistry
    from rllm.data.dataset import _wrap_rows_as_tasks

    dataset = DatasetRegistry.load_dataset(spec.physical_name, spec.split)
    if dataset is None:
        raise ValueError(
            f"MiniSandbox source dataset {spec.physical_name}/{spec.split} is not registered"
        )
    source_rows = list(dataset.data)
    if strict_inventory and len(source_rows) != spec.physical_expected_count:
        raise ValueError(
            f"{spec.physical_name}/{spec.split} expected {spec.physical_expected_count} "
            f"source rows, got {len(source_rows)}"
        )
    rows: list[dict[str, Any]] = []
    for raw in source_rows:
        if not isinstance(raw, dict):
            raise ValueError(
                f"{spec.physical_name}/{spec.split} contains a non-object row"
            )
        task_id = _safe_task_id(raw.get("id"))
        if task_id in spec.excluded_task_ids:
            continue
        if spec.languages:
            language = str(raw.get("language") or "").strip().casefold()
            if language not in spec.languages:
                continue
        rows.append(raw)
    if not rows:
        raise ValueError(f"MiniSandbox inventory for {spec.name} is empty")
    if strict_inventory and len(rows) != spec.expected_count:
        raise ValueError(
            f"MiniSandbox inventory for {spec.name} expected {spec.expected_count} "
            f"tasks, got {len(rows)}"
        )
    workers = max(1, min(int(inventory_workers), len(rows), 64))
    if workers == 1:
        tasks = [_wrap_rows_as_tasks([row])[0] for row in rows]
    else:
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="minisandbox-inventory"
        ) as executor:
            tasks = list(executor.map(lambda row: _wrap_rows_as_tasks([row])[0], rows))
    inventory = [cache_task_from_task(task, spec) for task in tasks]
    ids = [task.task_id for task in inventory]
    if len(ids) != len(set(ids)):
        raise ValueError(f"MiniSandbox inventory for {spec.name} has duplicate task ids")
    info = DatasetRegistry.get_dataset_info(spec.physical_name) or {}
    source_manifest = {
        "logical_dataset": spec.name,
        "physical_dataset": spec.physical_name,
        "split": spec.split,
        "registry_metadata": info.get("metadata"),
        "registry_split": (info.get("splits") or {}).get(spec.split),
        "source_rows": len(source_rows),
        "selected_rows": len(inventory),
    }
    return inventory, source_manifest


def _blob_path(cache_root: Path, layout: Path, digest: str) -> Path | None:
    try:
        algorithm, encoded = digest.split(":", 1)
    except ValueError:
        return None
    candidates = (
        layout / "blobs" / algorithm / encoded,
        cache_root / "blobs" / algorithm / encoded,
        cache_root / "blobs" / encoded,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _oci_image_descriptor(layout: Path) -> dict[str, Any]:
    index = _load_json_object(layout / "index.json")
    if index is None or not isinstance(index.get("manifests"), list):
        raise ValueError(f"OCI layout has no index manifest: {layout}")
    descriptors = [item for item in index["manifests"] if isinstance(item, dict)]
    if len(descriptors) != 1:
        raise ValueError(
            f"OCI layout must contain exactly one platform image: {layout}"
        )
    return dict(descriptors[0])


def _oci_blob_inventory(cache_root: Path, layout: Path) -> list[dict[str, Any]]:
    manifest_descriptor = _oci_image_descriptor(layout)
    manifest_digest = str(manifest_descriptor.get("digest") or "")
    manifest_path = _blob_path(cache_root, layout, manifest_digest)
    if manifest_path is None:
        raise ValueError(f"OCI image manifest blob is missing: {manifest_digest}")
    manifest = _load_json_object(manifest_path)
    if manifest is None:
        raise ValueError(f"OCI image manifest is invalid: {manifest_digest}")
    referenced: list[dict[str, Any]] = [manifest_descriptor]
    config = manifest.get("config")
    if isinstance(config, dict):
        referenced.append(config)
    layers = manifest.get("layers")
    if isinstance(layers, list):
        referenced.extend(item for item in layers if isinstance(item, dict))
    blobs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for descriptor in referenced:
        digest = str(descriptor.get("digest") or "")
        if not digest or digest in seen:
            continue
        seen.add(digest)
        path = _blob_path(cache_root, layout, digest)
        if path is None:
            raise ValueError(f"OCI blob is missing: {digest}")
        try:
            declared_size = int(descriptor.get("size"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"OCI descriptor has an invalid size for {digest}: {layout}"
            ) from exc
        actual_size = int(path.stat().st_size)
        if declared_size != actual_size:
            raise ValueError(
                "OCI descriptor size does not match its blob: "
                f"digest={digest} declared={declared_size} actual={actual_size}"
            )
        relative = path.resolve().relative_to(cache_root.resolve())
        blobs.append(
            {
                "digest": digest,
                "size": actual_size,
                "path": relative.as_posix(),
            }
        )
    return blobs


class _BlobVerificationCache:
    """Avoid re-hashing unchanged multi-GB OCI layers on every resume."""

    def __init__(self, cache_root: Path) -> None:
        self.cache_root = cache_root
        self.path = cache_root / ".blob-verification.json"
        raw = _load_json_object(self.path) or {}
        entries = raw.get("entries")
        self.entries: dict[str, dict[str, int]] = (
            {str(key): dict(value) for key, value in entries.items() if isinstance(value, dict)}
            if isinstance(entries, dict)
            else {}
        )
        self._dirty = False
        self._lock = threading.Lock()
        self._revision = 0
        self._entry_locks: dict[str, threading.Lock] = {}

    def _entry_key(self, digest: str, path: Path) -> str:
        try:
            relative = path.resolve().relative_to(self.cache_root.resolve())
            location = relative.as_posix()
        except (OSError, ValueError):
            location = str(path.resolve())
        return f"{digest}|{location}"

    @staticmethod
    def _stat_value(path: Path) -> dict[str, int]:
        stat = path.stat()
        return {
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "ctime_ns": int(stat.st_ctime_ns),
            "inode": int(stat.st_ino),
        }

    @staticmethod
    def _stat_matches(
        cached: Mapping[str, Any] | None,
        current: Mapping[str, int],
    ) -> bool:
        if not isinstance(cached, Mapping):
            return False
        stable_keys = ("size", "mtime_ns", "inode")
        if any(cached.get(key) != current[key] for key in stable_keys):
            return False
        # Schema-1 stat caches predate ctime_ns. Preserve their trusted result
        # once, then promote it below instead of re-hashing a multi-terabyte
        # shared cache solely because the verifier metadata schema improved.
        return "ctime_ns" not in cached or cached.get("ctime_ns") == current["ctime_ns"]

    def _cached_stat_is_trusted(
        self,
        key: str,
        digest: str,
        current: Mapping[str, int],
    ) -> bool:
        cached = self.entries.get(key)
        if not self._stat_matches(cached, current):
            cached = self.entries.get(digest)
        if not self._stat_matches(cached, current):
            return False
        if self.entries.get(key) != current:
            self.entries[key] = dict(current)
            self._dirty = True
            self._revision += 1
        return True

    def trust(self, digest: str, path: Path) -> None:
        key = self._entry_key(digest, path)
        with self._lock:
            self.entries[key] = self._stat_value(path)
            self._dirty = True
            self._revision += 1

    def validate(self, digest: str, path: Path, expected_size: int) -> bool:
        try:
            current = self._stat_value(path)
        except OSError:
            return False
        if current["size"] != expected_size:
            return False
        key = self._entry_key(digest, path)
        with self._lock:
            # Digest-only entries are accepted for compatibility with the
            # earliest stat cache, while new entries are keyed by physical
            # file so hardlinks/copies cannot validate one another by accident.
            if self._cached_stat_is_trusted(key, digest, current):
                return True
            entry_lock = self._entry_locks.setdefault(key, threading.Lock())
        if not re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
            return False
        # Several task records commonly reference the same multi-GB base
        # layer. Only one materialization worker should hash a given physical
        # blob when its stat entry is cold or stale.
        with entry_lock:
            try:
                current = self._stat_value(path)
            except OSError:
                return False
            if current["size"] != expected_size:
                return False
            with self._lock:
                if self._cached_stat_is_trusted(key, digest, current):
                    return True
            try:
                valid = _sha256_file(path) == digest.split(":", 1)[1]
            except OSError:
                return False
            if not valid:
                return False
            with self._lock:
                self.entries[key] = current
                self._dirty = True
                self._revision += 1
            return True

    def save(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            value = {"schema_version": 1, "entries": dict(self.entries)}
            revision = self._revision
        _atomic_json(self.path, value)
        with self._lock:
            if self._revision == revision:
                self._dirty = False


def _record_valid(
    record: Mapping[str, Any] | None,
    *,
    cache_root: Path,
    task: CacheTask,
    spec: MiniSandboxDatasetSpec,
    candidates: tuple[str, ...] | None,
    verifier: _BlobVerificationCache,
) -> bool:
    if not isinstance(record, Mapping) or record.get("status") != "complete":
        return False
    schema = record.get("schema_version")
    if schema != OCI_RECORD_SCHEMA_VERSION:
        return False
    if (
        record.get("task_id") != task.task_id
        or record.get("original_image") != task.image
        or record.get("platform") != OCI_PLATFORM
    ):
        return False
    if candidates is not None and record.get("source_candidates") != list(candidates):
        return False
    task_fingerprint_matches = (
        record.get("task_fingerprint") == task.task_fingerprint
    )
    # The original DeNovo validation API accepted lightweight scheduled
    # Task objects whose source_fingerprint and image were authoritative
    # even when their task directory was unavailable on the caller. Keep
    # that compatibility surface; materialization itself still compares
    # the full task/setup fingerprint before reusing a record.
    if (
        spec.name == "denovoswe"
        and task.row_fingerprint
        and not task.has_task_contract_files
    ):
        task_fingerprint_matches = True
    if (
        record.get("logical_dataset") != spec.name
        or record.get("physical_dataset") != spec.physical_name
        or record.get("split") != spec.split
        or not task_fingerprint_matches
        or record.get("row_fingerprint") != task.row_fingerprint
        or record.get("workdir") != task.workdir
        or record.get("task_profile") != task.task_profile
    ):
        return False
    layout_rel = record.get("layout")
    blobs = record.get("blobs")
    if not isinstance(layout_rel, str) or not isinstance(blobs, list):
        return False
    layout = _inside(cache_root, layout_rel)
    if layout is None or not (layout / "oci-layout").is_file():
        return False
    index = _load_json_object(layout / "index.json")
    manifests = index.get("manifests") if isinstance(index, dict) else None
    if not isinstance(manifests, list) or len(manifests) != 1:
        return False
    descriptor = manifests[0]
    pinned_digest = str(
        record.get("source_manifest_digest") or record.get("manifest_digest") or ""
    )
    if not isinstance(descriptor, dict) or descriptor.get("digest") != pinned_digest:
        return False
    source_descriptor = record.get("source_descriptor")
    if isinstance(source_descriptor, Mapping) and dict(source_descriptor) != descriptor:
        return False
    try:
        descriptor_size = int(descriptor.get("size"))
    except (TypeError, ValueError):
        return False
    manifest_blob = next(
        (
            item
            for item in blobs
            if isinstance(item, Mapping) and item.get("digest") == pinned_digest
        ),
        None,
    )
    try:
        recorded_manifest_size = int(manifest_blob.get("size"))
    except (AttributeError, TypeError, ValueError):
        return False
    if descriptor_size != recorded_manifest_size:
        return False
    for item in blobs:
        if not isinstance(item, Mapping):
            return False
        path = _inside(cache_root, str(item.get("path") or ""))
        try:
            expected_size = int(item.get("size"))
        except (TypeError, ValueError):
            return False
        if path is None or not path.is_file():
            return False
        digest = str(item.get("digest") or "")
        if not verifier.validate(digest, path, expected_size):
            return False
    return True


def _skopeo_copy_command(
    source: str,
    destination: Path,
    *,
    shared_blobs: Path,
    digest_file: Path,
    auth_file: Path | None,
) -> list[str]:
    command = [
        "skopeo",
        "copy",
        "--retry-times",
        str(SKOPEO_COPY_RETRY_TIMES),
        "--retry-delay",
        SKOPEO_COPY_RETRY_DELAY,
        # Dataset-level workers already provide bounded concurrency. Avoid
        # multiplying that by Skopeo's per-image layer fan-out, which can
        # overload the enterprise registry and produce truncated blob streams.
        "--image-parallel-copies",
        str(SKOPEO_IMAGE_PARALLEL_COPIES),
        "--override-os",
        OCI_PLATFORM["os"],
        "--override-arch",
        OCI_PLATFORM["architecture"],
        "--preserve-digests",
        "--dest-shared-blob-dir",
        str(shared_blobs),
        "--digestfile",
        str(digest_file),
    ]
    if auth_file is not None:
        command.extend(["--src-authfile", str(auth_file)])
    command.extend([f"docker://{source}", f"oci:{destination}:image"])
    return command


def _pull_one(
    task: CacheTask,
    *,
    spec: MiniSandboxDatasetSpec,
    cache_root: Path,
    image_map: Mapping[str, str],
    auth_file: Path | None,
    verifier: _BlobVerificationCache,
    run_command: Any,
) -> tuple[dict[str, Any], bool]:
    candidates = _candidate_sources(task.image, image_map, spec, task_id=task.task_id)
    record_path = cache_root / "records" / f"{task.task_id}.json"
    existing = _load_json_object(record_path)
    if _record_valid(
        existing,
        cache_root=cache_root,
        task=task,
        spec=spec,
        candidates=candidates,
        verifier=verifier,
    ):
        return dict(existing), True

    _atomic_json(
        record_path,
        {
            "schema_version": OCI_RECORD_SCHEMA_VERSION,
            "status": "in_progress",
            "logical_dataset": spec.name,
            "task_id": task.task_id,
            "original_image": task.image,
            "task_fingerprint": task.task_fingerprint,
        },
    )
    images_root = cache_root / "images"
    final_layout = images_root / task.task_id
    staging = images_root / (
        f".{task.task_id}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    digest_file = staging / ".digest"
    errors: list[str] = []
    selected_source: str | None = None
    try:
        for source in candidates:
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir(parents=True)
            command = _skopeo_copy_command(
                source,
                staging,
                shared_blobs=cache_root / "blobs",
                digest_file=digest_file,
                auth_file=auth_file,
            )
            try:
                completed = run_command(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                )
            except OSError as exc:
                raise RuntimeError(
                    "skopeo is required for MiniSandbox cache materialization"
                ) from exc
            if completed.returncode == 0:
                selected_source = source
                break
            errors.append(
                f"{source}: {(completed.stderr or completed.stdout or '').strip()[-1200:]}"
            )
        if selected_source is None:
            detail = (
                f"{task.task_id}: every image source failed: "
                + " | ".join(errors)
            )
            if _registry_auth_failure(candidates, errors):
                raise MiniSandboxRegistryAuthenticationError(
                    detail
                    + "; the image registry requires credentials. Set "
                    "MINISANDBOX_REGISTRY_AUTH_FILE to your registry auth file; "
                    "HF_TOKEN does not authenticate an OCI registry."
                )
            raise RuntimeError(detail)
        digest = digest_file.read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
            raise RuntimeError(
                f"{task.task_id}: skopeo returned invalid digest {digest!r}"
            )
        if final_layout.exists():
            shutil.rmtree(final_layout)
        os.replace(staging, final_layout)
        source_descriptor = _oci_image_descriptor(final_layout)
        if source_descriptor.get("digest") != digest:
            raise RuntimeError(
                f"{task.task_id}: source descriptor drift: "
                f"digestfile={digest} descriptor={source_descriptor.get('digest')!r}"
            )
        blobs = _oci_blob_inventory(cache_root, final_layout)
        for blob in blobs:
            path = _inside(cache_root, str(blob["path"]))
            if path is None:
                raise RuntimeError(f"{task.task_id}: unsafe OCI blob path")
            verifier.trust(str(blob["digest"]), path)
        record = {
            "schema_version": OCI_RECORD_SCHEMA_VERSION,
            "status": "complete",
            "logical_dataset": spec.name,
            "physical_dataset": spec.physical_name,
            "split": spec.split,
            "task_id": task.task_id,
            "task_profile": task.task_profile,
            "task_fingerprint": task.task_fingerprint,
            "row_fingerprint": task.row_fingerprint,
            "original_image": task.image,
            "workdir": task.workdir,
            "source_candidates": list(candidates),
            "selected_source": selected_source,
            "source_manifest_digest": digest,
            "source_descriptor": source_descriptor,
            # Runtime v1 compatibility: this is the source-layout descriptor,
            # not the digest of the node-local OCI conversion.
            "manifest_digest": digest,
            "platform": dict(OCI_PLATFORM),
            "layout": final_layout.relative_to(cache_root).as_posix(),
            "blobs": blobs,
            "security": {
                "forbidden_paths": list(spec.forbidden_paths),
                "visible_fixture_paths": list(spec.visible_fixture_paths),
            },
        }
        _atomic_json(record_path, record)
        return record, False
    except BaseException as exc:
        _atomic_json(
            record_path,
            {
                "schema_version": OCI_RECORD_SCHEMA_VERSION,
                "status": "failed",
                "logical_dataset": spec.name,
                "task_id": task.task_id,
                "original_image": task.image,
                "task_fingerprint": task.task_fingerprint,
                "error_type": type(exc).__name__,
                "error": str(exc)[-2000:],
            },
        )
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def materialize_minisandbox_oci_cache(
    *,
    dataset: str | MiniSandboxDatasetSpec,
    cache_dir: str | Path,
    tasks: Iterable[Any] | None = None,
    source_manifest: Mapping[str, Any] | None = None,
    image_map_file: str | Path | None = None,
    registry_auth_file: str | Path | None = None,
    max_workers: int = DEFAULT_IMAGE_WORKERS,
    inventory_workers: int | None = None,
    strict_inventory: bool = True,
    run_command: Any = subprocess.run,
) -> Path:
    """Materialise one complete logical dataset view into a shared OCI cache.

    ``tasks`` is primarily a compatibility/testing seam.  Normal callers omit
    it and the pinned physical registry dataset is selected by the spec.
    """

    spec = minisandbox_dataset_spec(dataset)
    if isinstance(max_workers, bool) or int(max_workers) <= 0:
        raise ValueError("max_workers must be a positive integer")
    if run_command is subprocess.run and shutil.which("skopeo") is None:
        raise RuntimeError(
            "skopeo is required for MiniSandbox cache materialization"
        )
    if tasks is None:
        inventory, resolved_source_manifest = _load_registry_inventory(
            spec,
            strict_inventory=bool(strict_inventory),
            inventory_workers=int(inventory_workers or max_workers),
        )
    else:
        task_values = list(tasks)
        inventory = [cache_task_from_task(task, spec) for task in task_values]
        if not inventory:
            raise ValueError(f"MiniSandbox inventory for {spec.name} is empty")
        ids = [task.task_id for task in inventory]
        if len(ids) != len(set(ids)):
            raise ValueError(f"MiniSandbox inventory for {spec.name} has duplicate task ids")
        if strict_inventory and len(inventory) != spec.expected_count:
            raise ValueError(
                f"MiniSandbox inventory for {spec.name} expected {spec.expected_count} "
                f"tasks, got {len(inventory)}"
            )
        resolved_source_manifest = dict(source_manifest or {})

    cache_root = Path(cache_dir).expanduser().resolve()
    image_map, image_map_sha256 = _load_image_map(image_map_file)
    auth_file = (
        Path(registry_auth_file).expanduser().resolve()
        if registry_auth_file not in (None, "")
        else None
    )
    if auth_file is not None and not auth_file.is_file():
        raise FileNotFoundError(auth_file)
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "records").mkdir(exist_ok=True)
    (cache_root / "images").mkdir(exist_ok=True)
    (cache_root / "blobs").mkdir(exist_ok=True)
    lock_path = cache_root / ".materialization.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        dispatch_inventory, incomplete_record_count = _prioritize_incomplete_records(
            inventory,
            cache_root,
        )
        print(
            f"  resume scan:    {incomplete_record_count} incomplete task(s) first; "
            f"{len(inventory) - incomplete_record_count} complete record(s) to validate",
            flush=True,
        )
        print(
            "  transfer mode:  "
            f"{SKOPEO_COPY_RETRY_TIMES} retries, "
            f"{SKOPEO_IMAGE_PARALLEL_COPIES} layer copy per image",
            flush=True,
        )
        verifier = _BlobVerificationCache(cache_root)
        selected_contract = [
            {
                "task_id": task.task_id,
                "task_fingerprint": task.task_fingerprint,
                "image": task.image,
            }
            for task in inventory
        ]
        config = {
            "schema_version": OCI_CACHE_SCHEMA_VERSION,
            "kind": "swe-minisandbox-oci",
            "logical_dataset": spec.name,
            "physical_dataset": spec.physical_name,
            "split": spec.split,
            "platform": dict(OCI_PLATFORM),
            "expected_tasks": spec.expected_count,
            "strict_inventory": bool(strict_inventory),
            "selected_task_ids_sha256": _sha256_bytes(
                _canonical_json([task.task_id for task in inventory])
            ),
            "dataset_contract_sha256": _sha256_bytes(
                _canonical_json(selected_contract)
            ),
            "source_manifest_sha256": _sha256_bytes(
                _canonical_json(resolved_source_manifest)
            ),
            "image_map_sha256": image_map_sha256,
            "excluded_task_ids_sha256": _sha256_bytes(
                _canonical_json(sorted(spec.excluded_task_ids))
            ),
        }
        manifest: dict[str, Any] = {
            "schema_version": OCI_CACHE_SCHEMA_VERSION,
            "status": "in_progress",
            "config": config,
            "selected_tasks": len(inventory),
            "excluded_tasks": len(spec.excluded_task_ids),
            "completed_tasks": 0,
            "reused_tasks": 0,
            "failed_tasks": [],
            "records": {},
        }
        _atomic_json(cache_root / "materialization.json", manifest)
        records: dict[str, dict[str, Any]] = {}
        reused = 0
        failures: list[tuple[str, BaseException]] = []
        worker_count = min(int(max_workers), len(inventory))
        task_iterator = iter(dispatch_inventory)
        auth_failure: MiniSandboxRegistryAuthenticationError | None = None
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix=f"{spec.name[:24]}-oci",
        ) as executor:
            futures: dict[Any, CacheTask] = {}

            def submit_next() -> bool:
                try:
                    task = next(task_iterator)
                except StopIteration:
                    return False
                future = executor.submit(
                    _pull_one,
                    task,
                    spec=spec,
                    cache_root=cache_root,
                    image_map=image_map,
                    auth_file=auth_file,
                    verifier=verifier,
                    run_command=run_command,
                )
                futures[future] = task
                return True

            for _ in range(worker_count):
                submit_next()
            completed_count = 0
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                finished_this_round = 0
                for future in done:
                    task = futures.pop(future)
                    if future.cancelled():
                        continue
                    try:
                        record, was_reused = future.result()
                    except MiniSandboxRegistryAuthenticationError as exc:
                        failures.append((task.task_id, exc))
                        auth_failure = auth_failure or exc
                    except BaseException as exc:
                        failures.append((task.task_id, exc))
                    else:
                        records[task.task_id] = record
                        reused += int(was_reused)
                    completed_count += 1
                    finished_this_round += 1
                if auth_failure is not None:
                    for future in futures:
                        future.cancel()
                else:
                    for _ in range(finished_this_round):
                        submit_next()
                if (
                    completed_count % 25 == 0
                    or not futures
                    or auth_failure is not None
                ):
                    manifest["completed_tasks"] = len(records)
                    manifest["reused_tasks"] = reused
                    manifest["failed_tasks"] = [
                        task_id for task_id, _ in failures
                    ]
                    manifest["records"] = {
                        key: f"records/{key}.json" for key in sorted(records)
                    }
                    if auth_failure is not None:
                        manifest["aborted_early"] = "registry_authentication"
                    _atomic_json(cache_root / "materialization.json", manifest)
                    verifier.save()
                    print(
                        f"  progress:       {completed_count}/{len(inventory)} processed; "
                        f"complete={len(records)} reused={reused} "
                        f"failed={len(failures)}",
                        flush=True,
                    )
        if failures:
            manifest["status"] = "incomplete"
            manifest["failed_tasks"] = [task_id for task_id, _ in failures]
            _atomic_json(cache_root / "materialization.json", manifest)
            verifier.save()
            task_id, error = failures[0]
            raise RuntimeError(
                f"{spec.name} MiniSandbox OCI cache is incomplete: "
                f"{len(failures)} task(s) failed; first={task_id}: {error}"
            ) from error
        manifest.update(
            {
                "status": "complete",
                "completed_tasks": len(inventory),
                "reused_tasks": reused,
                "failed_tasks": [],
                "records": {
                    key: f"records/{key}.json" for key in sorted(records)
                },
            }
        )
        _atomic_json(cache_root / "materialization.json", manifest)
        verifier.save()
        return cache_root / "materialization.json"


def validate_minisandbox_oci_cache(
    cache_dir: str | Path,
    tasks: Iterable[Any],
    *,
    dataset: str | MiniSandboxDatasetSpec,
) -> dict[str, dict[str, Any]]:
    """Validate cache coverage for the exact post-filter scheduling set."""

    spec = minisandbox_dataset_spec(dataset)
    cache_root = Path(cache_dir).expanduser().resolve()
    manifest = _load_json_object(cache_root / "materialization.json")
    if manifest is None or manifest.get("status") != "complete":
        raise ValueError(
            f"{spec.name} MiniSandbox OCI cache is incomplete: {cache_root}"
        )
    schema = manifest.get("schema_version")
    if schema == OCI_CACHE_SCHEMA_VERSION:
        config = manifest.get("config")
        config = config if isinstance(config, Mapping) else {}
        expected_config = {
            "logical_dataset": spec.name,
            "physical_dataset": spec.physical_name,
            "split": spec.split,
            "platform": OCI_PLATFORM,
        }
        mismatches = {
            key: {"expected": expected, "actual": config.get(key)}
            for key, expected in expected_config.items()
            if config.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                "MiniSandbox cache dataset mismatch: "
                + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
            )
    else:
        raise ValueError(f"unsupported MiniSandbox cache schema: {schema!r}")
    record_paths = manifest.get("records")
    if not isinstance(record_paths, dict):
        raise ValueError("MiniSandbox OCI cache has no record index")
    verifier = _BlobVerificationCache(cache_root)
    result: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    seen: set[str] = set()
    for task in tasks:
        try:
            cache_task = cache_task_from_task(task, spec)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        task_id = cache_task.task_id
        if task_id in seen:
            errors.append(f"{task_id}: duplicate scheduled task id")
            continue
        seen.add(task_id)
        relative = record_paths.get(task_id)
        if not isinstance(relative, str):
            errors.append(f"{task_id}: cache record is missing")
            continue
        record_path = _inside(cache_root, relative)
        record = _load_json_object(record_path) if record_path is not None else None
        if not _record_valid(
            record,
            cache_root=cache_root,
            task=cache_task,
            spec=spec,
            candidates=None,
            verifier=verifier,
        ):
            errors.append(f"{task_id}: cache record or OCI blobs are invalid")
            continue
        result[task_id] = dict(record)
    verifier.save()
    if errors:
        raise ValueError(
            f"{spec.name} MiniSandbox OCI cache validation failed for "
            f"{len(errors)} condition(s); examples: {'; '.join(errors[:5])}"
        )
    return result


__all__ = [
    "CacheTask",
    "DEFAULT_IMAGE_WORKERS",
    "MINISANDBOX_DATASET_SPECS",
    "MiniSandboxDatasetSpec",
    "OCI_CACHE_SCHEMA_VERSION",
    "OCI_PLATFORM",
    "OCI_RECORD_SCHEMA_VERSION",
    "cache_task_from_task",
    "default_minisandbox_cache_dir",
    "materialize_minisandbox_oci_cache",
    "minisandbox_dataset_spec",
    "validate_minisandbox_oci_cache",
]
