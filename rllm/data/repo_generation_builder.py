"""Resumable materializers for repository-generation training and evaluation.

Test assets are deliberately stored only below each task's host-side ``tests``
directory.  The ordinary task setup path uploads ``environment/files`` but not
``tests``; shadow/final evaluators therefore remain the sole consumers of the
golden acceptance data.
"""

from __future__ import annotations

import base64
import binascii
import fcntl
import hashlib
import json
import logging
import math
import os
import re
import shlex
import shutil
import tempfile
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path, PurePosixPath
from typing import Any

import tomllib

from rllm.data.materialization import (
    bounded_ordered_thread_map,
    resolve_materialization_workers,
)

logger = logging.getLogger(__name__)

MATERIALIZATION_SCHEMA_VERSION = 1
TASK_SCHEMA_VERSION = 1
DENOVOSWE_TASK_SCHEMA_VERSION = 3
OFFICIAL_AWEAGENT_COMMIT = "ed7865c57e821fd35f500e6ba1a078160da6a983"

DEFAULT_DENOVOSWE_HF_REPO = "AweAI-Team/DeNovoSWE"
DEFAULT_DENOVOSWE_REVISION = "287448712ca9669feff3c128631de15a9de9ede9"
DEFAULT_DENOVOSWE_FILENAME = "denovoswe_public.jsonl"
DEFAULT_DENOVOSWE_NAME = "denovoswe"
# The paper describes 4,818 curated instances, but the pinned Hugging Face
# revision explicitly publishes only a partial release. Its 4.57 GB
# ``denovoswe_public.jsonl`` contains 3,675 complete JSONL records.
DENOVOSWE_ROWS_AT_DEFAULT_REVISION = 3675

DEFAULT_NL2REPO_HF_REPO = "AweAI-Team/AweAgent-Meta-NL2Repo"
DEFAULT_NL2REPO_REVISION = "e5cb1553c26e313af606c4600abeb2c887846361"
DEFAULT_NL2REPO_FILENAME = "nl2repo_aweagent.jsonl"
DEFAULT_NL2REPO_NAME = "nl2repo-bench"
NL2REPO_ROWS_AT_DEFAULT_REVISION = 104

DEFAULT_DOC2REPO_HF_REPO = "AweAI-Team/BeyondSWE-harbor"
DEFAULT_DOC2REPO_REVISION = "7d2ced21b4c85f646d5ba8786875d7fbbe08ed49"
DEFAULT_DOC2REPO_NAME = "beyondswe-doc2repo"
DOC2REPO_ROWS_AT_DEFAULT_REVISION = 50
DOC2REPO_IMAGE_REPOSITORY = "aweaiteam/beyondswe"

_ASSET_DIR = Path(__file__).with_name("assets")
_CLEAN_ASSET = _ASSET_DIR / "denovoswe_clean.sh"
_DENOVO_VERIFIER_ASSET = _ASSET_DIR / "denovoswe_verifier.py"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _atomic_write(path: Path, payload: bytes, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            temporary.chmod(mode)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_text(path: Path, value: str, *, mode: int | None = None) -> None:
    _atomic_write(path, value.encode("utf-8"), mode=mode)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


@contextmanager
def _materialization_lock(out: Path) -> Iterator[None]:
    out.parent.mkdir(parents=True, exist_ok=True)
    lock_path = out.parent / f".{out.name}.materialization.lock"
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _safe_task_id(raw: Any) -> str:
    value = str(raw or "").strip()
    if not value or value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
        raise ValueError(f"unsafe or empty instance_id: {value!r}")
    return value


def _safe_workdir(raw: Any, *, default: str = "/workspace") -> str:
    value = str(raw or default).strip()
    pure = PurePosixPath(value)
    if not pure.is_absolute() or ".." in pure.parts or value == "/":
        raise ValueError(f"unsafe repository workdir: {value!r}")
    return str(pure)


def _first(row: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            return value
    return default


def _required_string(row: dict[str, Any], task_id: str, *names: str) -> str:
    value = _first(row, *names)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{task_id}: required field aliases {names!r} are empty")
    return value


def _string_list(row: dict[str, Any], task_id: str, *names: str, required: bool) -> list[str]:
    value = _first(row, *names, default=[])
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = [value]
        value = parsed
    if not isinstance(value, list | tuple):
        raise ValueError(f"{task_id}: field aliases {names!r} must be a list")
    result = [str(item).strip() for item in value if str(item).strip()]
    if required and not result:
        raise ValueError(f"{task_id}: field aliases {names!r} must be non-empty")
    return result


def _safe_patch_paths(patch: str, *, task_id: str) -> list[str]:
    result: list[str] = []
    for line in patch.splitlines():
        if not line.startswith("+++ "):
            continue
        raw = line[4:].strip().split("\t", 1)[0]
        if raw == "/dev/null":
            continue
        if raw.startswith("b/"):
            raw = raw[2:]
        pure = PurePosixPath(raw)
        if pure.is_absolute() or ".." in pure.parts or not raw:
            raise ValueError(f"{task_id}: unsafe test patch path {raw!r}")
        if raw not in result:
            result.append(raw)
    if not result:
        raise ValueError(f"{task_id}: test_patch contains no file paths")
    return result


def _toml(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _dataset_toml(name: str, split: str, default_agent: str) -> str:
    return "\n".join(
        [
            "[dataset]",
            f"name = {_toml(name)}",
            'type = "sandbox"',
            f"split = {_toml(split)}",
            f"default_agent = {_toml(default_agent)}",
            'category = "code"',
            "",
        ]
    )


def _dockerfile(image: str, workdir: str) -> str:
    return f"FROM {image}\nENTRYPOINT []\nWORKDIR {workdir}\n"


def _deterministic_git_script(
    workdir: str,
    *,
    prelude: Iterable[str] = (),
    baseline_message: str,
) -> str:
    workdir_q = shlex.quote(workdir)
    commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {workdir_q}",
        *prelude,
        "rm -rf .git",
        "git init -q",
        "git symbolic-ref HEAD refs/heads/main",
        'git config user.name "rllm-baseline"',
        'git config user.email "rllm-baseline@example.invalid"',
        "git config core.autocrlf false",
        "git config core.filemode true",
        "git add -A",
        (f"GIT_AUTHOR_DATE='2000-01-01T00:00:00Z' GIT_COMMITTER_DATE='2000-01-01T00:00:00Z' git commit -q --allow-empty -m {shlex.quote(baseline_message)}"),
        "git status --porcelain=v1 --untracked-files=all | grep -q . && { echo 'non-clean deterministic baseline' >&2; exit 1; } || true",
        "",
    ]
    return "\n".join(commands)


def _manifest_config(
    *,
    kind: str,
    name: str,
    split: str,
    hf_repo_id: str,
    hf_revision: str,
    source_file: str | None,
    limit: int | None,
) -> dict[str, Any]:
    config = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "kind": kind,
        "name": name,
        "split": split,
        "hf_repo_id": hf_repo_id,
        "hf_revision": hf_revision,
        "source_file": source_file,
        "limit": limit,
        "aweagent_source_commit": OFFICIAL_AWEAGENT_COMMIT,
    }
    if kind == "denovoswe":
        config["vendored_assets"] = {
            "clean.sh": _sha256_file(_CLEAN_ASSET),
            "denovoswe_verifier.py": _sha256_file(_DENOVO_VERIFIER_ASSET),
        }
    return config


def _load_manifest(out: Path) -> dict[str, Any] | None:
    path = out / "materialization.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _assert_compatible(manifest: dict[str, Any], config: dict[str, Any], out: Path) -> None:
    if manifest.get("config") != config:
        raise ValueError(f"materialization configuration drift at {out}; rerun with CLEAN=1")


def _task_complete(
    task_dir: Path,
    fingerprint: str,
    required: tuple[str, ...],
    *,
    schema_version: int = TASK_SCHEMA_VERSION,
) -> bool:
    try:
        marker = json.loads((task_dir / ".materialized.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if marker.get("schema_version") != schema_version:
        return False
    if marker.get("row_fingerprint") != fingerprint:
        return False
    return all((task_dir / relative).is_file() for relative in required)


def _registry_complete(name: str, split: str, selected: int) -> bool:
    try:
        import pyarrow.parquet as pq

        from rllm.data import DatasetRegistry

        info = DatasetRegistry.get_dataset_info(name)
        if not info or split not in info.get("splits", {}):
            return False
        split_info = info["splits"][split]
        path = DatasetRegistry._resolve_path(split_info["path"])
        return pq.read_metadata(path).num_rows == selected
    except Exception:
        return False


def materialization_complete(
    *,
    out_dir: str | Path,
    config: dict[str, Any],
    required_files: tuple[str, ...],
) -> bool:
    out = Path(out_dir).expanduser()
    manifest = _load_manifest(out)
    if manifest is None or manifest.get("status") != "complete" or manifest.get("config") != config or not (out / "dataset.toml").is_file():
        return False
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != manifest.get("selected_tasks"):
        return False
    task_schema_version = DENOVOSWE_TASK_SCHEMA_VERSION if config.get("kind") == "denovoswe" else TASK_SCHEMA_VERSION
    for item in tasks:
        if not isinstance(item, dict):
            return False
        try:
            task_id = _safe_task_id(item.get("id"))
        except ValueError:
            return False
        if not _task_complete(
            out / task_id,
            str(item.get("row_fingerprint") or ""),
            required_files,
            schema_version=task_schema_version,
        ):
            return False
    if manifest.get("registered") and not _registry_complete(config["name"], config["split"], len(tasks)):
        return False
    return True


def _refresh_denovoswe_asset(
    out_dir: str | Path, *, source: Path, relative: str, name: str, receipt_key: str,
    migration_key: str, dry_run: bool = False, max_workers: int = 16,
) -> dict[str, Any]:
    """Migrate only the trusted asset; preserve task/source/registry data.

    Safe to resume after interruption. The in-progress manifest retains the
    original hash until all per-task asset and receipt updates are confirmed.
    Run with training stopped so setup cannot observe a mixed asset generation.
    """
    out = Path(out_dir).expanduser()
    new_hash = _sha256_file(source)
    with nullcontext() if dry_run else _materialization_lock(out):
        manifest = _load_manifest(out)
        if not manifest or manifest.get("config", {}).get("kind") != "denovoswe":
            raise ValueError("Expected an existing DeNovo materialization manifest")
        migration = manifest.get(migration_key)
        if any(key.endswith("_asset_migration") and key != migration_key for key in manifest):
            raise ValueError("Complete the pending asset migration first")
        if manifest.get("status") != "complete" and not migration:
            raise ValueError("Cannot refresh an incomplete materialization")
        config = manifest["config"]
        old_hash = config["vendored_assets"][name]
        if migration and migration.get("target_sha256") != new_hash:
            raise ValueError("An interrupted migration targets a different asset")
        tasks = manifest.get("tasks", [])
        if len(tasks) != manifest.get("selected_tasks") or not tasks:
            raise ValueError("Invalid materialized task inventory")
        ids = [_safe_task_id(item["id"]) for item in tasks]
        if len(set(ids)) != len(ids) or _json_fingerprint(ids) != manifest.get("selected_task_ids_sha256"):
            raise ValueError("Invalid materialized task IDs or inventory fingerprint")
        workers = resolve_materialization_workers(len(tasks), max_workers)
        def inspect_task(item):
            task_id = _safe_task_id(item["id"])
            task_dir = out / task_id
            asset = task_dir / relative
            receipt = task_dir / ".materialized.json"
            marker = json.loads(receipt.read_text())
            if marker.get("row_fingerprint") != item["row_fingerprint"]:
                raise ValueError(f"{task_id}: source fingerprint mismatch")
            if not all((task_dir / relative).is_file() for relative in _DENOVO_REQUIRED):
                raise ValueError(f"{task_id}: incomplete task")
            actual = _sha256_file(asset)
            if actual not in {old_hash, new_hash} or marker.get(receipt_key) not in {old_hash, new_hash}:
                raise ValueError(f"{task_id}: unrecognized asset or receipt")
            return asset, receipt, marker, actual
        records = list(bounded_ordered_thread_map(
            inspect_task, tasks, workers=workers, thread_name_prefix="denovo-clean-check",
        ))
        result = {"tasks": len(records), "updates": sum(actual != new_hash or marker.get(receipt_key) != new_hash
                  for _, _, marker, actual in records), "sha256": new_hash, "dry_run": dry_run}
        if dry_run or (result["updates"] == 0 and old_hash == new_hash and not migration):
            return result
        manifest["status"] = "in_progress"
        manifest[migration_key] = {"target_sha256": new_hash, "previous_sha256": old_hash}
        _atomic_json(out / "materialization.json", manifest)
        clean_text = source.read_text()
        def replace_asset(record):
            asset, receipt, marker, actual = record
            if actual != new_hash:
                _atomic_text(asset, clean_text, mode=0o755)
            if marker.get(receipt_key) != new_hash:
                marker[receipt_key] = new_hash
                _atomic_json(receipt, marker)
            if _sha256_file(asset) != new_hash or json.loads(receipt.read_text()).get(receipt_key) != new_hash:
                raise RuntimeError(f"Asset verification failed: {asset}")
        for _ in bounded_ordered_thread_map(
            replace_asset, records, workers=workers, thread_name_prefix="denovo-clean-update",
        ):
            pass
        config["vendored_assets"][name] = new_hash
        manifest.pop(migration_key, None)
        manifest["status"] = "complete"
        _atomic_json(out / "materialization.json", manifest)
        return result



def refresh_denovoswe_clean_assets(
    out_dir: str | Path, *, dry_run: bool = False, max_workers: int = 16,
) -> dict[str, Any]:
    return _refresh_denovoswe_asset(
        out_dir, source=_CLEAN_ASSET, relative="environment/files/.rllm_assets/clean.sh",
        name="clean.sh", receipt_key="clean_sha256", migration_key="clean_asset_migration",
        dry_run=dry_run, max_workers=max_workers,
    )


def refresh_denovoswe_verifier_assets(
    out_dir: str | Path, *, dry_run: bool = False, max_workers: int = 16,
) -> dict[str, Any]:
    """Refresh only verifier code and its receipts; run with training stopped."""
    return _refresh_denovoswe_asset(
        out_dir, source=_DENOVO_VERIFIER_ASSET, relative="tests/denovoswe_verifier.py",
        name="denovoswe_verifier.py", receipt_key="verifier_sha256", migration_key="verifier_asset_migration",
        dry_run=dry_run, max_workers=max_workers,
    )


def _resolve_hf_file(repo_id: str, revision: str, filename: str) -> Path:
    local = Path(repo_id).expanduser()
    if local.exists():
        path = local / filename if local.is_dir() else local
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename=filename,
        )
    )


def _iter_jsonl_rows(path: Path) -> Iterator[tuple[dict[str, Any], str]]:
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value, _sha256_bytes(raw.rstrip(b"\r\n"))


def _inventory_jsonl(
    path: Path,
    *,
    limit: int | None,
) -> tuple[list[dict[str, str]], int]:
    """Scan a large source without retaining documents or binary fixtures.

    DeNovoSWE's pinned JSONL is several GiB and contains base64 archives.  A
    list of parsed rows would hold both the encoded and decoded fixtures at
    once.  The lightweight inventory is therefore built in a first pass; a
    second pass validates and materializes exactly one row at a time.
    """

    selected: list[dict[str, str]] = []
    total = 0
    for row, fingerprint in _iter_jsonl_rows(path):
        total += 1
        if limit is None or len(selected) < limit:
            selected.append(
                {
                    "id": _safe_task_id(_first(row, "instance_id", "id")),
                    "row_fingerprint": fingerprint,
                }
            )
    return selected, total


def _validate_denovo(row: dict[str, Any]) -> dict[str, Any]:
    task_id = _safe_task_id(_first(row, "instance_id", "id"))
    workdir = _safe_workdir(_first(row, "workdir", "repo_dir"))
    image = _required_string(row, task_id, "image", "image_url", "docker_image")
    parent = _required_string(row, task_id, "parent_commit", "base_commit")
    document = _required_string(row, task_id, "document", "repo_document", "readme")
    passed = _string_list(row, task_id, "passed_ptp", "unit_test", required=True)
    failed = _string_list(row, task_id, "failed_ptp", "failed_tests", required=False)
    raw_patch = _first(row, "test_patch", "tests_patch", default="")
    if not isinstance(raw_patch, str):
        raise ValueError(f"{task_id}: test_patch must be a string")
    patch = raw_patch
    binary_b64 = str(_first(row, "test_binary_archive_b64", "binary_fixture_b64", default="") or "")
    try:
        binary = base64.b64decode(binary_b64, validate=True) if binary_b64 else b""
    except (ValueError, binascii.Error) as exc:
        raise ValueError(f"{task_id}: invalid test binary archive") from exc
    if not patch.strip() and not binary:
        raise ValueError(f"{task_id}: test_patch and binary test archive are both empty")
    binary_files = _string_list(row, task_id, "test_binary_files", required=False)
    test_files = _string_list(row, task_id, "test_files", required=False)
    if not test_files:
        if patch.strip():
            test_files = _safe_patch_paths(patch, task_id=task_id)
        else:
            # The official evaluator permits a binary-only acceptance suite.
            # Keep enough host-side inventory for audit without exposing it in
            # the registry or primary sandbox.
            test_files = list(dict.fromkeys([node.split("::", 1)[0] for node in passed] + binary_files))
    repo = str(_first(row, "repo", "repo_name", default=PurePosixPath(workdir).name))
    pypi_name = str(_first(row, "pypi_name", "package_name", default=repo) or repo)
    raw_difficulty = _first(row, "difficulty")
    if isinstance(raw_difficulty, bool):
        raise ValueError(f"{task_id}: difficulty must be a finite float in [0, 1]")
    try:
        difficulty = float(raw_difficulty)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{task_id}: difficulty must be a finite float in [0, 1]"
        ) from exc
    if not math.isfinite(difficulty) or not 0.0 <= difficulty <= 1.0:
        raise ValueError(f"{task_id}: difficulty must be a finite float in [0, 1]")
    return {
        "id": task_id,
        "workdir": workdir,
        "image": image,
        "parent_commit": parent,
        "document": document,
        "test_patch": patch,
        "passed_ptp": passed,
        "failed_ptp": failed,
        "test_files": test_files,
        "test_binary_archive": binary,
        "test_binary_files": binary_files,
        "repo": repo,
        "difficulty": difficulty,
        "pypi_name": pypi_name,
        "pypi_name_candidates": _string_list(row, task_id, "pypi_name_candidates", required=False),
        "import_names": _string_list(row, task_id, "import_names", required=False),
    }


_DENOVO_REQUIRED = (
    "task.toml",
    "instruction.md",
    "environment/Dockerfile",
    "environment/setup.sh",
    "environment/files/.rllm_assets/README.md",
    "environment/files/.rllm_assets/clean.sh",
    "tests/test.sh",
    "tests/instance.json",
    "tests/test.patch",
    "tests/denovoswe_verifier.py",
    "solution/solve.sh",
    ".materialized.json",
)

_DENOVO_STORAGE_OVERRIDES_MB = {
    # The 30-GiB image exhausted its repository filesystem while creating a
    # CodeFlow checkpoint in the 2026-08-24 canary.
    "plus3it_watchmaker_pr-1": 61_440,
}


def _denovo_task_toml(v: dict[str, Any], revision: str, name: str) -> str:
    task_name = f"{name}/{v['id']}"
    return "\n".join(
        [
            'schema_version = "1.1"',
            "",
            "[task]",
            f"name = {_toml(task_name)}",
            f"description = {_toml('DeNovoSWE repository generation')}",
            f"keywords = [{_toml('denovoswe')}, {_toml('repo-generation')}]",
            "",
            "[metadata]",
            f"task_id = {_toml(v['id'])}",
            f"repo_name = {_toml(v['repo'])}",
            f"commit_hash = {_toml(v['parent_commit'])}",
            f"difficulty = {v['difficulty']}",
            'language = "python"',
            f"source_revision = {_toml(revision)}",
            f"pypi_name = {_toml(v['pypi_name'])}",
            "",
            "[environment]",
            f"docker_image = {_toml(v['image'])}",
            f"workdir = {_toml(v['workdir'])}",
            "cpus = 4",
            "memory_mb = 8192",
            "storage_mb = "
            + str(_DENOVO_STORAGE_OVERRIDES_MB.get(v["id"], 30_720)),
            "build_timeout_sec = 1800.0",
            "allow_internet = true",
            "",
            "[agent]",
            "timeout_sec = 7200.0",
            "",
            "[verifier]",
            "timeout_sec = 3600.0",
            "",
            "[rllm]",
            'task_profile = "repo_generation_denovoswe"',
            'setup_failure_mode = "raise"',
            'outcome_source = "shadow_verifier"',
            'verification_potential_mode = "pass_count"',
            'shadow_verifier_profile = "denovoswe_official"',
            'bash_policy_profile = "repo_generation_package"',
            "setup_timeout_sec = 1800.0",
            'test_results_path = "/tmp/rllm/test_results.json"',
            "",
        ]
    )


def _materialize_denovo_task(
    staging: Path,
    v: dict[str, Any],
    *,
    fingerprint: str,
    revision: str,
    name: str,
) -> None:
    workdir_q = shlex.quote(v["workdir"])
    parent_q = shlex.quote(v["parent_commit"])
    prelude = [
        f"git config --global --add safe.directory {workdir_q}",
        f"git checkout -f {parent_q}",
        "RLLM_README_STAGING=$(mktemp /tmp/rllm-denovoswe-readme.XXXXXX)",
        "trap 'rm -f \"$RLLM_README_STAGING\"' EXIT",
        'cp .rllm_assets/README.md "$RLLM_README_STAGING"',
        f"bash .rllm_assets/clean.sh {workdir_q}",
        'install -m 0644 "$RLLM_README_STAGING" README.md',
        'rm -f "$RLLM_README_STAGING"',
        "trap - EXIT",
        "rm -rf .rllm_assets",
        "cat >> .gitignore <<'RLLM_GITIGNORE'",
        "# rLLM managed runtime files",
        ".pytest_cache/",
        "__pycache__/",
        "*.pyc",
        "RLLM_GITIGNORE",
    ]
    instance = {
        "schema_version": 1,
        "instance_id": v["id"],
        "workdir": v["workdir"],
        "parent_commit": v["parent_commit"],
        "passed_ptp": v["passed_ptp"],
        "failed_ptp": v["failed_ptp"],
        "test_files": v["test_files"],
        "test_binary_files": v["test_binary_files"],
        "pypi_name": v["pypi_name"],
        "pypi_name_candidates": v["pypi_name_candidates"],
        "import_names": v["import_names"],
        "verifier_profile": "denovoswe_official",
    }
    _atomic_text(staging / "task.toml", _denovo_task_toml(v, revision, name))
    _atomic_text(staging / "instruction.md", v["document"].rstrip() + "\n")
    _atomic_text(staging / "environment/Dockerfile", _dockerfile(v["image"], v["workdir"]))
    _atomic_text(
        staging / "environment/setup.sh",
        _deterministic_git_script(v["workdir"], prelude=prelude, baseline_message="rllm deterministic denovoswe baseline"),
        mode=0o755,
    )
    _atomic_text(staging / "environment/files/.rllm_assets/README.md", v["document"].rstrip() + "\n")
    shutil.copy2(_CLEAN_ASSET, staging / "environment/files/.rllm_assets/clean.sh")
    (staging / "environment/files/.rllm_assets/clean.sh").chmod(0o755)
    _atomic_json(staging / "tests/instance.json", instance)
    _atomic_text(staging / "tests/test.patch", v["test_patch"])
    if v["test_binary_archive"]:
        _atomic_write(staging / "tests/test_binaries.tar.gz", v["test_binary_archive"])
    shutil.copy2(_DENOVO_VERIFIER_ASSET, staging / "tests/denovoswe_verifier.py")
    (staging / "tests/denovoswe_verifier.py").chmod(0o755)
    _atomic_text(
        staging / "tests/test.sh",
        "#!/usr/bin/env bash\nset -uo pipefail\npython3 /tests/denovoswe_verifier.py --instance /tests/instance.json --test-patch /tests/test.patch --binary-archive /tests/test_binaries.tar.gz\n",
        mode=0o755,
    )
    _atomic_text(
        staging / "solution/solve.sh",
        "#!/usr/bin/env bash\necho 'DeNovoSWE has no public gold implementation.' >&2\nexit 1\n",
        mode=0o755,
    )
    _atomic_json(
        staging / ".materialized.json",
        {
            "schema_version": DENOVOSWE_TASK_SCHEMA_VERSION,
            "row_fingerprint": fingerprint,
            "source_revision": revision,
            "aweagent_source_commit": OFFICIAL_AWEAGENT_COMMIT,
            "clean_sha256": _sha256_file(_CLEAN_ASSET),
            "verifier_sha256": _sha256_file(_DENOVO_VERIFIER_ASSET),
        },
    )


def _validate_nl2(row: dict[str, Any]) -> dict[str, Any]:
    task_id = _safe_task_id(_first(row, "instance_id", "id"))
    image = _required_string(row, task_id, "evaluation_image", "image", "image_url")
    instruction = _required_string(row, task_id, "start_instruction", "instruction")
    package_name = _required_string(row, task_id, "package_name", "repo")
    verify_cmd = _string_list(row, task_id, "verify_cmd", "verify_commands", required=True)
    verify_files = _string_list(row, task_id, "verify_files", "py_test_file_list", required=True)
    raw_count = _first(row, "test_cases_num", "test_case_count")
    if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count <= 0:
        raise ValueError(f"{task_id}: test_cases_num must be a positive integer")
    return {
        "id": task_id,
        "image": image,
        "instruction": instruction,
        "package_name": package_name,
        "verify_cmd": verify_cmd,
        "verify_files": verify_files,
        "test_cases_num": raw_count,
        "workdir": _safe_workdir(_first(row, "workdir")),
    }


_NL2_REQUIRED = (
    "task.toml",
    "instruction.md",
    "environment/Dockerfile",
    "environment/setup.sh",
    "environment/files/start.md",
    "tests/instance.json",
    "tests/test.sh",
    "solution/solve.sh",
    ".materialized.json",
)


def _nl2_task_toml(v: dict[str, Any], revision: str, name: str) -> str:
    task_name = f"{name}/{v['id']}"
    return "\n".join(
        [
            'schema_version = "1.1"',
            "",
            "[task]",
            f"name = {_toml(task_name)}",
            f"description = {_toml('NL2Repo-Bench repository generation')}",
            "",
            "[metadata]",
            f"task_id = {_toml(v['id'])}",
            f"package_name = {_toml(v['package_name'])}",
            f"source_revision = {_toml(revision)}",
            f"test_cases_num = {v['test_cases_num']}",
            "",
            "[environment]",
            f"docker_image = {_toml(v['image'])}",
            f"workdir = {_toml(v['workdir'])}",
            "cpus = 4",
            "memory_mb = 8192",
            "storage_mb = 30720",
            "build_timeout_sec = 1800.0",
            "allow_internet = true",
            "",
            "[agent]",
            "timeout_sec = 7200.0",
            "",
            "[verifier]",
            "timeout_sec = 3600.0",
            "",
            "[rllm]",
            'task_profile = "repo_generation_nl2repo"',
            'setup_failure_mode = "raise"',
            'verifier_kind = "nl2repo-fresh-sandbox"',
            'bash_policy_profile = "repo_generation_package"',
            "",
        ]
    )


def _materialize_nl2_task(
    staging: Path,
    v: dict[str, Any],
    *,
    fingerprint: str,
    revision: str,
    name: str,
) -> None:
    removal = []
    for path in v["verify_files"]:
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"{v['id']}: unsafe verify_files path {path!r}")
        removal.append(f"rm -rf -- {shlex.quote(path)}")
    setup = _deterministic_git_script(
        v["workdir"],
        prelude=[
            *removal,
            "# rllm nl2repo primary cleanup v2",
            "rm -rf -- tests",
            "find . -type f \\( -name 'test_*.py' -o -name '*_test.py' -o -name 'conftest.py' \\) -delete 2>/dev/null || true",
            "find . -type d -name '.pytest_cache' -prune -exec rm -rf {} + 2>/dev/null || true",
        ],
        baseline_message="rllm deterministic nl2repo baseline",
    )
    _atomic_text(staging / "task.toml", _nl2_task_toml(v, revision, name))
    _atomic_text(staging / "instruction.md", v["instruction"].rstrip() + "\n")
    _atomic_text(staging / "environment/Dockerfile", _dockerfile(v["image"], v["workdir"]))
    _atomic_text(staging / "environment/setup.sh", setup, mode=0o755)
    _atomic_text(staging / "environment/files/start.md", v["instruction"].rstrip() + "\n")
    _atomic_json(
        staging / "tests/instance.json",
        {
            "schema_version": 1,
            "instance_id": v["id"],
            "evaluation_image": v["image"],
            "workdir": v["workdir"],
            "package_name": v["package_name"],
            "verify_cmd": v["verify_cmd"],
            "verify_files": v["verify_files"],
            "test_cases_num": v["test_cases_num"],
            "verifier_profile": "nl2repo-fresh-sandbox",
        },
    )
    _atomic_text(
        staging / "tests/test.sh",
        "#!/usr/bin/env bash\necho 'NL2Repo requires the rLLM fresh-sandbox evaluator.' >&2\nexit 2\n",
        mode=0o755,
    )
    _atomic_text(
        staging / "solution/solve.sh",
        "#!/usr/bin/env bash\necho 'NL2Repo has no public gold implementation.' >&2\nexit 1\n",
        mode=0o755,
    )
    _atomic_json(
        staging / ".materialized.json",
        {
            "schema_version": TASK_SCHEMA_VERSION,
            "row_fingerprint": fingerprint,
            "source_revision": revision,
        },
    )


def _registry_row(
    v: dict[str, Any],
    task_dir: Path,
    *,
    kind: str,
    source: str,
    revision: str,
    fingerprint: str,
) -> dict[str, Any]:
    instruction = v.get("document", v.get("instruction", ""))
    result = {
        "id": v["id"],
        "instruction": str(instruction).rstrip() + "\n",
        "task_path": str(task_dir.resolve()),
        "task_profile": f"repo_generation_{kind}",
        "docker_image": v.get("image", ""),
        "source_dataset": source,
        "source_revision": revision,
        "source_fingerprint": fingerprint,
    }
    if kind == "denovoswe":
        result.update(
            {
                "repo_name": v["repo"],
                "difficulty": v["difficulty"],
                "commit_hash": v["parent_commit"],
                "pypi_name": v["pypi_name"],
                "pypi_name_candidates": v["pypi_name_candidates"],
                "import_names": v["import_names"],
                "outcome_source": "shadow_verifier",
                "verifier_profile": "denovoswe_official",
                "verification_potential_mode": "pass_count",
            }
        )
    else:
        result.update(
            {
                "package_name": v["package_name"],
                "test_cases_num": v["test_cases_num"],
                "verifier_profile": "nl2repo-fresh-sandbox",
            }
        )
    return result


def _build_jsonl_benchmark(
    *,
    kind: str,
    out_dir: str | Path,
    name: str,
    split: str,
    hf_repo_id: str,
    hf_revision: str,
    filename: str,
    expected_rows: int,
    limit: int | None,
    default_agent: str,
    clean: bool,
    register: bool,
    max_workers: int | None,
) -> Path:
    if limit is not None and (isinstance(limit, bool) or limit <= 0):
        raise ValueError("limit must be a positive integer")
    out = Path(out_dir).expanduser()
    config = _manifest_config(
        kind=kind,
        name=name,
        split=split,
        hf_repo_id=hf_repo_id,
        hf_revision=hf_revision,
        source_file=filename,
        limit=limit,
    )
    required = _DENOVO_REQUIRED if kind == "denovoswe" else _NL2_REQUIRED
    task_schema_version = DENOVOSWE_TASK_SCHEMA_VERSION if kind == "denovoswe" else TASK_SCHEMA_VERSION
    with _materialization_lock(out):
        if clean and out.exists():
            shutil.rmtree(out)
        if not clean and materialization_complete(out_dir=out, config=config, required_files=required):
            return out
        out.mkdir(parents=True, exist_ok=True)
        existing = _load_manifest(out)
        if existing is not None:
            _assert_compatible(existing, config, out)
        for stale in out.glob(".tmp-*"):
            if stale.is_dir():
                shutil.rmtree(stale)
            else:
                stale.unlink(missing_ok=True)
        source_path = _resolve_hf_file(hf_repo_id, hf_revision, filename)
        inventory, source_count = _inventory_jsonl(source_path, limit=limit)
        if (
            hf_repo_id == (DEFAULT_DENOVOSWE_HF_REPO if kind == "denovoswe" else DEFAULT_NL2REPO_HF_REPO)
            and hf_revision == (DEFAULT_DENOVOSWE_REVISION if kind == "denovoswe" else DEFAULT_NL2REPO_REVISION)
            and source_count != expected_rows
        ):
            raise ValueError(f"pinned {kind} source expected {expected_rows} rows, got {source_count}")
        if not inventory:
            raise ValueError(f"{kind} selection is empty")
        validator = _validate_denovo if kind == "denovoswe" else _validate_nl2
        ids = [item["id"] for item in inventory]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{kind} selection contains duplicate instance IDs")
        _atomic_text(out / "dataset.toml", _dataset_toml(name, split, default_agent))
        manifest: dict[str, Any] = {
            "schema_version": MATERIALIZATION_SCHEMA_VERSION,
            "status": "in_progress",
            "config": config,
            "source_rows": source_count,
            "selected_tasks": len(inventory),
            "selected_task_ids_sha256": _json_fingerprint(ids),
            "completed_tasks": 0,
            "tasks": inventory,
        }
        _atomic_json(out / "materialization.json", manifest)
        def selected_rows() -> Iterator[tuple[dict[str, Any], str]]:
            source_rows = _iter_jsonl_rows(source_path)
            for index, expected in enumerate(inventory, start=1):
                try:
                    row, fingerprint = next(source_rows)
                except StopIteration as exc:
                    raise RuntimeError(
                        f"{kind} source changed between inventory and materialization"
                    ) from exc
                value = validator(row)
                if (
                    value["id"] != expected["id"]
                    or fingerprint != expected["row_fingerprint"]
                ):
                    raise RuntimeError(
                        f"{kind} source changed between inventory and materialization "
                        f"at selected row {index}"
                    )
                yield value, fingerprint

        def materialize_one(
            item: tuple[dict[str, Any], str],
        ) -> tuple[dict[str, Any], bool]:
            value, fingerprint = item
            task_dir = out / value["id"]
            row_reused = _task_complete(
                task_dir,
                fingerprint,
                required,
                schema_version=task_schema_version,
            )
            if not row_reused:
                if task_dir.exists():
                    shutil.rmtree(task_dir)
                staging = out / f".tmp-{value['id']}-{uuid.uuid4().hex}"
                try:
                    if kind == "denovoswe":
                        _materialize_denovo_task(
                            staging,
                            value,
                            fingerprint=fingerprint,
                            revision=hf_revision,
                            name=name,
                        )
                    else:
                        _materialize_nl2_task(
                            staging,
                            value,
                            fingerprint=fingerprint,
                            revision=hf_revision,
                            name=name,
                        )
                    if not _task_complete(
                        staging,
                        fingerprint,
                        required,
                        schema_version=task_schema_version,
                    ):
                        raise RuntimeError(f"{value['id']}: staged task is incomplete")
                    os.replace(staging, task_dir)
                except BaseException:
                    shutil.rmtree(staging, ignore_errors=True)
                    raise
            return (
                _registry_row(
                    value,
                    task_dir,
                    kind=kind,
                    source=hf_repo_id,
                    revision=hf_revision,
                    fingerprint=fingerprint,
                ),
                row_reused,
            )

        workers = resolve_materialization_workers(len(inventory), max_workers)
        logger.info("[%s] materializing with %d worker(s)", kind, workers)
        registration: list[dict[str, Any]] = []
        reused = 0
        results = bounded_ordered_thread_map(
            materialize_one,
            selected_rows(),
            workers=workers,
            thread_name_prefix=f"{kind}-materialize",
        )
        for index, (registry_row, row_reused) in enumerate(results, start=1):
            registration.append(registry_row)
            reused += int(row_reused)
            if index % 25 == 0 or index == len(inventory):
                manifest["completed_tasks"] = index
                manifest["reused_tasks"] = reused
                _atomic_json(out / "materialization.json", manifest)
        if register:
            from rllm.data import DatasetRegistry

            DatasetRegistry.register_dataset(
                name=name,
                data=registration,
                split=split,
                source=f"{hf_repo_id}@{hf_revision}:{filename}",
                description=f"Pinned {kind} repository-generation tasks.",
                category="code",
            )
        manifest.update(
            {
                "status": "complete",
                "completed_tasks": len(inventory),
                "reused_tasks": reused,
                "registered": bool(register),
            }
        )
        _atomic_json(out / "materialization.json", manifest)
        return out


def build_denovoswe(
    *,
    out_dir: str | Path,
    name: str = DEFAULT_DENOVOSWE_NAME,
    split: str = "train",
    hf_repo_id: str = DEFAULT_DENOVOSWE_HF_REPO,
    hf_revision: str = DEFAULT_DENOVOSWE_REVISION,
    filename: str = DEFAULT_DENOVOSWE_FILENAME,
    limit: int | None = None,
    default_agent: str = "codeflow",
    clean: bool = False,
    register: bool = True,
    max_workers: int | None = None,
) -> Path:
    return _build_jsonl_benchmark(
        kind="denovoswe",
        out_dir=out_dir,
        name=name,
        split=split,
        hf_repo_id=hf_repo_id,
        hf_revision=hf_revision,
        filename=filename,
        expected_rows=DENOVOSWE_ROWS_AT_DEFAULT_REVISION,
        limit=limit,
        default_agent=default_agent,
        clean=clean,
        register=register,
        max_workers=max_workers,
    )


def build_nl2repo(
    *,
    out_dir: str | Path,
    name: str = DEFAULT_NL2REPO_NAME,
    split: str = "test",
    hf_repo_id: str = DEFAULT_NL2REPO_HF_REPO,
    hf_revision: str = DEFAULT_NL2REPO_REVISION,
    filename: str = DEFAULT_NL2REPO_FILENAME,
    limit: int | None = None,
    default_agent: str = "codeflow",
    clean: bool = False,
    register: bool = True,
    max_workers: int | None = None,
) -> Path:
    return _build_jsonl_benchmark(
        kind="nl2repo",
        out_dir=out_dir,
        name=name,
        split=split,
        hf_repo_id=hf_repo_id,
        hf_revision=hf_revision,
        filename=filename,
        expected_rows=NL2REPO_ROWS_AT_DEFAULT_REVISION,
        limit=limit,
        default_agent=default_agent,
        clean=clean,
        register=register,
        max_workers=max_workers,
    )


def _resolve_doc2repo_source(repo_id: str, revision: str) -> Path:
    local = Path(repo_id).expanduser()
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=["beyondswe/**"],
        )
    )


def _doc2repo_candidates(source: Path) -> list[Path]:
    root = source / "beyondswe"
    if not root.is_dir():
        raise ValueError(f"BeyondSWE source has no beyondswe directory: {source}")
    result: list[Path] = []
    for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        task_toml = task_dir / "task.toml"
        document = task_dir / "environment/repo_document.md"
        suite = task_dir / "tests/test_suite.zip"
        if not (task_toml.is_file() and document.is_file() and suite.is_file()):
            continue
        try:
            metadata = tomllib.loads(task_toml.read_text(encoding="utf-8")).get("metadata", {})
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(f"invalid BeyondSWE task metadata: {task_toml}") from exc
        tags = {str(item).casefold() for item in metadata.get("tags", [])}
        if "doc2repo" in tags:
            result.append(task_dir)
    return result


_DOC2_REQUIRED = (
    "task.toml",
    "instruction.md",
    "environment/Dockerfile",
    "environment/setup.sh",
    "environment/files/repo_document.md",
    "tests/test.sh",
    "tests/test_suite.zip",
    "tests/score_pytest.py",
    "solution/solve.sh",
    ".materialized.json",
)


def _doc_task_fingerprint(task_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in task_dir.rglob("*") if item.is_file()):
        relative = path.relative_to(task_dir).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _doc2repo_image(task_id: str) -> str:
    tag = _safe_task_id(task_id).lower()
    if len(tag) > 128 or not re.fullmatch(r"[a-z0-9_][a-z0-9_.-]*", tag):
        raise ValueError(f"Doc2Repo instance_id is not a valid Docker tag: {task_id!r}")
    return f"{DOC2REPO_IMAGE_REPOSITORY}:{tag}"


def _dockerfile_base_image(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for line in lines:
        if line.strip().upper().startswith("FROM "):
            return line.split(None, 1)[1].strip().split()[0]
    return ""


def _rewrite_dockerfile_base(path: Path, image: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.strip().upper().startswith("FROM "):
            newline = "\n" if line.endswith("\n") else ""
            lines[index] = f"FROM {image}{newline}"
            _atomic_text(path, "".join(lines))
            return
    raise ValueError(f"Doc2Repo Dockerfile has no FROM instruction: {path}")


def _doc2_task_complete(task_dir: Path, fingerprint: str, task_id: str) -> bool:
    return _task_complete(task_dir, fingerprint, _DOC2_REQUIRED) and (_dockerfile_base_image(task_dir / "environment/Dockerfile") == _doc2repo_image(task_id))


def _doc2_materialization_complete(out: Path, config: dict[str, Any]) -> bool:
    if not materialization_complete(
        out_dir=out,
        config=config,
        required_files=_DOC2_REQUIRED,
    ):
        return False
    manifest = _load_manifest(out)
    assert manifest is not None
    return all(
        _doc2_task_complete(
            out / _safe_task_id(item.get("id")),
            str(item.get("row_fingerprint") or ""),
            _safe_task_id(item.get("id")),
        )
        for item in manifest["tasks"]
    )


def _materialize_doc_task(
    staging: Path,
    source_task: Path,
    *,
    task_id: str,
    fingerprint: str,
    revision: str,
) -> None:
    shutil.copytree(source_task, staging)
    image = _doc2repo_image(task_id)
    _rewrite_dockerfile_base(staging / "environment/Dockerfile", image)
    document = staging / "environment/repo_document.md"
    files_document = staging / "environment/files/repo_document.md"
    files_document.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(document, files_document)
    _atomic_text(
        staging / "environment/setup.sh",
        _deterministic_git_script(
            "/workspace",
            baseline_message="rllm deterministic doc2repo baseline",
        ),
        mode=0o755,
    )
    task_toml = (staging / "task.toml").read_text(encoding="utf-8")
    if "[rllm]" not in task_toml:
        task_toml = task_toml.rstrip() + "\n\n[rllm]\n"
    task_toml += 'task_profile = "repo_generation_doc2repo"\nsetup_failure_mode = "raise"\nverifier_kind = "script"\n'
    _atomic_text(staging / "task.toml", task_toml)
    _atomic_json(
        staging / ".materialized.json",
        {
            "schema_version": TASK_SCHEMA_VERSION,
            "row_fingerprint": fingerprint,
            "source_revision": revision,
            "docker_image": image,
        },
    )


def build_doc2repo(
    *,
    out_dir: str | Path,
    name: str = DEFAULT_DOC2REPO_NAME,
    split: str = "test",
    hf_repo_id: str = DEFAULT_DOC2REPO_HF_REPO,
    hf_revision: str = DEFAULT_DOC2REPO_REVISION,
    limit: int | None = None,
    default_agent: str = "codeflow",
    clean: bool = False,
    register: bool = True,
    max_workers: int | None = None,
) -> Path:
    out = Path(out_dir).expanduser()
    config = _manifest_config(
        kind="doc2repo",
        name=name,
        split=split,
        hf_repo_id=hf_repo_id,
        hf_revision=hf_revision,
        source_file=None,
        limit=limit,
    )
    with _materialization_lock(out):
        if clean and out.exists():
            shutil.rmtree(out)
        if not clean and _doc2_materialization_complete(out, config):
            return out
        out.mkdir(parents=True, exist_ok=True)
        existing = _load_manifest(out)
        if existing is not None:
            _assert_compatible(existing, config, out)
        for stale in out.glob(".tmp-*"):
            if stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)
            else:
                stale.unlink(missing_ok=True)
        source = _resolve_doc2repo_source(hf_repo_id, hf_revision)
        candidates = _doc2repo_candidates(source)
        if hf_repo_id == DEFAULT_DOC2REPO_HF_REPO and hf_revision == DEFAULT_DOC2REPO_REVISION and len(candidates) != DOC2REPO_ROWS_AT_DEFAULT_REVISION:
            raise ValueError(f"pinned Doc2Repo source expected {DOC2REPO_ROWS_AT_DEFAULT_REVISION} tasks, got {len(candidates)}")
        if limit is not None:
            if isinstance(limit, bool) or limit <= 0:
                raise ValueError("limit must be a positive integer")
            candidates = candidates[:limit]
        if not candidates:
            raise ValueError("Doc2Repo selection is empty")
        workers = resolve_materialization_workers(len(candidates), max_workers)
        logger.info("[doc2repo] fingerprinting with %d worker(s)", workers)

        def fingerprint_one(path: Path) -> tuple[Path, dict[str, str]]:
            return (
                path,
                {
                    "id": _safe_task_id(path.name),
                    "row_fingerprint": _doc_task_fingerprint(path),
                },
            )

        prepared = list(
            bounded_ordered_thread_map(
                fingerprint_one,
                candidates,
                workers=workers,
                thread_name_prefix="doc2repo-fingerprint",
            )
        )
        tasks = [item for _source_task, item in prepared]
        _atomic_text(out / "dataset.toml", _dataset_toml(name, split, default_agent))
        manifest: dict[str, Any] = {
            "schema_version": MATERIALIZATION_SCHEMA_VERSION,
            "status": "in_progress",
            "config": config,
            "source_rows": len(_doc2repo_candidates(source)),
            "selected_tasks": len(tasks),
            "selected_task_ids_sha256": _json_fingerprint([item["id"] for item in tasks]),
            "completed_tasks": 0,
            "tasks": tasks,
        }
        _atomic_json(out / "materialization.json", manifest)
        def materialize_one(
            prepared_task: tuple[Path, dict[str, str]],
        ) -> tuple[dict[str, Any], bool]:
            source_task, item = prepared_task
            task_dir = out / item["id"]
            row_reused = _doc2_task_complete(
                task_dir,
                item["row_fingerprint"],
                item["id"],
            )
            if not row_reused:
                if task_dir.exists():
                    shutil.rmtree(task_dir)
                staging = out / f".tmp-{item['id']}-{uuid.uuid4().hex}"
                try:
                    _materialize_doc_task(
                        staging,
                        source_task,
                        task_id=item["id"],
                        fingerprint=item["row_fingerprint"],
                        revision=hf_revision,
                    )
                    if not _doc2_task_complete(
                        staging,
                        item["row_fingerprint"],
                        item["id"],
                    ):
                        raise RuntimeError(f"{item['id']}: staged Doc2Repo task is incomplete")
                    os.replace(staging, task_dir)
                except BaseException:
                    shutil.rmtree(staging, ignore_errors=True)
                    raise
            instruction = (task_dir / "instruction.md").read_text(encoding="utf-8")
            image = _dockerfile_base_image(task_dir / "environment/Dockerfile")
            return (
                {
                    "id": item["id"],
                    "instruction": instruction,
                    "task_path": str(task_dir.resolve()),
                    "task_profile": "repo_generation_doc2repo",
                    "docker_image": image,
                    "source_dataset": hf_repo_id,
                    "source_revision": hf_revision,
                    "source_fingerprint": item["row_fingerprint"],
                    "verifier_profile": "beyondswe_doc2repo_harbor",
                },
                row_reused,
            )

        logger.info("[doc2repo] materializing with %d worker(s)", workers)
        registration: list[dict[str, Any]] = []
        reused = 0
        results = bounded_ordered_thread_map(
            materialize_one,
            prepared,
            workers=workers,
            thread_name_prefix="doc2repo-materialize",
        )
        for index, (registry_row, row_reused) in enumerate(results, start=1):
            registration.append(registry_row)
            reused += int(row_reused)
            if index % 10 == 0 or index == len(tasks):
                manifest["completed_tasks"] = index
                manifest["reused_tasks"] = reused
                _atomic_json(out / "materialization.json", manifest)
        if register:
            from rllm.data import DatasetRegistry

            DatasetRegistry.register_dataset(
                name=name,
                data=registration,
                split=split,
                source=f"{hf_repo_id}@{hf_revision}",
                description="Pinned BeyondSWE Doc2Repo Harbor tasks.",
                category="code",
            )
        manifest.update(
            {
                "status": "complete",
                "completed_tasks": len(tasks),
                "reused_tasks": reused,
                "registered": bool(register),
            }
        )
        _atomic_json(out / "materialization.json", manifest)
        return out


__all__ = [
    "build_denovoswe",
    "refresh_denovoswe_clean_assets",
    "refresh_denovoswe_verifier_assets",
    "build_nl2repo",
    "build_doc2repo",
    "materialization_complete",
]
