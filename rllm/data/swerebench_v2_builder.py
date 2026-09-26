"""Materialize SWE-rebench V2 Filtered Verified tasks for rLLM.

The builder supports the complete multilingual dataset and keeps generic
language filtering for callers that invoke it directly.  The standard SWE
entrypoint materializes only the complete dataset; named training subsets are
resolved as in-memory views.  The source dataset ships one pre-built image,
gold patch, held-out test patch, test command, and language-specific log parser
per row.
The held-out patch is deliberately kept under ``tests/`` so it is uploaded only
when the sandbox verifier runs; the solving agent never receives it.

Materialization is resumable at task granularity.  Each task is written into a
sibling staging directory, validated, and atomically renamed into place.  A
source-row fingerprint in ``.materialized.json`` lets later runs skip complete
tasks without trusting a partially written directory.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Collection, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from rllm.data.assets.denovoswe_verifier import files_added_by_patch
from rllm.data.materialization import resolve_materialization_workers

logger = logging.getLogger(__name__)

DEFAULT_HF_REPO_ID = "PrimeIntellect/SWE-rebench-V2-Filtered-Verified"
DEFAULT_HF_REVISION = "03cc767ee33126b7fc7890ad57047e9dd6914cca"
DEFAULT_HF_SPLIT = "train"
DEFAULT_DATASET_NAME = "swe-rebench-v2-filtered-verified-python"
DEFAULT_FULL_DATASET_NAME = "swe-rebench-v2-filtered-verified"
DEFAULT_PROMATCHED_DATASET_NAME = "swe-rebench-v2-filtered-verified-promatched"
DEFAULT_LANGUAGE = "python"
PROMATCHED_LANGUAGES = frozenset({"python", "go", "js", "ts"})
SOURCE_ROWS_AT_DEFAULT_REVISION = 6272
SOURCE_REPOS_AT_DEFAULT_REVISION = 1890
SOURCE_LANGUAGES_AT_DEFAULT_REVISION = 16
PYTHON_ROWS_AT_DEFAULT_REVISION = 1952
PYTHON_REPOS_AT_DEFAULT_REVISION = 441
PROMATCHED_ROWS_AT_DEFAULT_REVISION = 4738
PROMATCHED_REPOS_AT_DEFAULT_REVISION = 1425

MATERIALIZATION_SCHEMA_VERSION = 7
TASK_SCHEMA_VERSION = 7
PYTHON_RESULT_PARSER = "swerebench_v2_pytest_v1"
OFFICIAL_RESULT_PARSER = "swerebench_v2_official_v2"
RESULT_PARSER = PYTHON_RESULT_PARSER
TEST_RESULTS_PATH = "/tmp/rllm/test_results.json"
PARSER_VENDOR_REVISION = "a4c70f7444c6d35ee211bea9dfca82a6c45ad24e"
RESULT_ADAPTER_VERSION = 6
SUPPORTED_LOG_PARSERS = frozenset(
    {
        "parse_java_mvn",
        "parse_java_mvn_v2",
        "parse_log_ant",
        "parse_log_cargo",
        "parse_log_cpp_v3",
        "parse_log_dart",
        "parse_log_dart_v3",
        "parse_log_elixir",
        "parse_log_gotest",
        "parse_log_gradle_custom",
        "parse_log_gradlew_v1",
        "parse_log_jq",
        "parse_log_js",
        "parse_log_js_2",
        "parse_log_js_3",
        "parse_log_js_4",
        "parse_log_junit",
        "parse_log_maven",
        "parse_log_ocaml",
        "parse_log_ocaml_v2",
        "parse_log_php_v1",
        "parse_log_phpunit",
        "parse_log_pytest",
        "parse_log_r",
        "parse_log_scala_v2",
        "parse_log_scala_v3",
        "parse_log_swift",
        "parse_logs_r_junit",
        "parse_lue_nvim",
    }
)
_ASSET_DIR = Path(__file__).with_name("assets")
_OFFICIAL_LOG_PARSERS_ASSET = _ASSET_DIR / "swerebench_v2_log_parsers.py"
PARSER_VENDOR_SHA256 = hashlib.sha256(
    _OFFICIAL_LOG_PARSERS_ASSET.read_bytes()
).hexdigest()

_DEFAULT_RESOURCES = {
    "cpus": 4,
    "memory_mb": 16384,
    "storage_mb": 30720,
    "build_timeout_sec": 1800.0,
}
_SHADOW_RESOURCE_OVERRIDE = {"cpus": 4, "memory_mb": 16384}
_DEFAULT_TIMEOUTS = {"agent_timeout_sec": 1800.0, "verifier_timeout_sec": 1800.0}
_REQUIRED_TASK_FILES = (
    "task.toml",
    "instruction.md",
    "environment/Dockerfile",
    "tests/test.sh",
    "tests/grader.py",
    "tests/log_parsers.py",
    "tests/instance.json",
    "tests/test.patch",
    "solution/gold.patch",
    "solution/solve.sh",
    ".materialized.json",
)

_MILESTONE_REGISTRY_FIELDS = (
    "expected_output_json",
    "baseline_output_json",
    "target_output_json",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "modified_files",
    "relevant_files",
    "modified_entity_summaries",
    "test_file_names",
    "protected_test_paths",
    "result_parser",
    "log_parser",
    "parser_vendor_revision",
    "parser_vendor_sha256",
    "result_adapter_version",
    "test_results_path",
    "baseline_source",
    "runtime_baseline_required",
)


def _json_fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_write_text(path: Path, text: str, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            temporary.chmod(mode)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _atomic_replace_text(path: Path, text: str, *, mode: int | None = None) -> None:
    """Atomically replace one task file, leaving its completion marker last.

    Task upgrades intentionally avoid a per-file ``fsync``: a task is trusted
    only after the separately replaced completion marker matches schema and
    source fingerprints, so an interrupted upgrade is safely repeated.
    """

    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(text, encoding="utf-8")
        if mode is not None:
            temporary.chmod(mode)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@contextmanager
def _materialization_lock(out: Path) -> Iterator[None]:
    out.parent.mkdir(parents=True, exist_ok=True)
    lock_path = out.parent / f".{out.name}.materialization.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _materialization_workers(
    task_count: int,
    requested: int | None = None,
) -> int:
    return resolve_materialization_workers(
        task_count,
        requested,
        legacy_env_names=(),
    )


def _as_nonempty_string(row: dict[str, Any], field: str, task_id: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{task_id}: {field} must be a non-empty string")
    return value


def _as_string_list(row: dict[str, Any], field: str, task_id: str, *, nonempty: bool) -> list[str]:
    value = row.get(field)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{task_id}: {field} must be a list of non-empty strings")
    result = list(value)
    if nonempty and not result:
        raise ValueError(f"{task_id}: {field} must not be empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{task_id}: {field} contains duplicate test ids")
    return result


def _install_config(row: dict[str, Any], task_id: str) -> dict[str, Any]:
    raw = row.get("install_config")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{task_id}: install_config is malformed JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{task_id}: install_config must be an object")
    config = dict(raw)
    parser_name = config.get("log_parser")
    if parser_name not in SUPPORTED_LOG_PARSERS:
        raise ValueError(f"{task_id}: unsupported SWE-rebench V2 log parser {parser_name!r}")
    test_cmd = config.get("test_cmd")
    if isinstance(test_cmd, str):
        commands = [test_cmd] if test_cmd.strip() else []
    elif isinstance(test_cmd, list):
        commands = [item for item in test_cmd if isinstance(item, str) and item.strip()]
        if len(commands) != len(test_cmd):
            raise ValueError(f"{task_id}: install_config.test_cmd contains an invalid command")
    else:
        commands = []
    if not commands:
        raise ValueError(f"{task_id}: install_config.test_cmd must not be empty")
    config["test_cmd"] = commands[0] if isinstance(test_cmd, str) else commands
    return config


def _safe_patch_path(raw_path: str, *, task_id: str, field: str) -> str:
    value = raw_path.strip().strip('"')
    if value.startswith(("a/", "b/")):
        value = value[2:]
    value = value.split("\t", 1)[0].strip()
    path = PurePosixPath(value)
    if not value or value == "/dev/null" or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{task_id}: {field} contains unsafe patch path {raw_path!r}")
    return value


def patch_paths(patch: str, *, task_id: str, field: str) -> list[str]:
    """Return every old/new path in a unified diff, preserving first order."""

    paths: list[str] = []
    seen: set[str] = set()
    for block in re.split(r"(?m)(?=^diff --git )", patch):
        block_paths: list[str] = []
        for line in block.splitlines():
            if not line.startswith(("--- ", "+++ ")):
                continue
            raw_path = line[4:].strip()
            if raw_path != "/dev/null":
                block_paths.append(_safe_patch_path(raw_path, task_id=task_id, field=field))
        if not block_paths:
            # Empty additions have only a diff header and new-file mode. They
            # still need restoration and protection alongside nonempty tests.
            try:
                block_paths = [path.as_posix() for path in files_added_by_patch(block)]
            except RuntimeError as exc:
                raise ValueError(f"{task_id}: {field}: {exc}") from exc
        for path in block_paths:
            if path not in seen:
                paths.append(path)
                seen.add(path)
    if not paths:
        raise ValueError(f"{task_id}: {field} contains no parseable file paths")
    return paths


def _repo_workdir(repo: str, task_id: str) -> str:
    if "/" not in repo:
        raise ValueError(f"{task_id}: repo must use owner/name form")
    name = repo.split("/", 1)[1].strip()
    if not name or "/" in name or name in {".", ".."}:
        raise ValueError(f"{task_id}: repo contains an invalid name")
    return f"/{name}"


def _instruction(row: dict[str, Any], task_id: str) -> tuple[str, str]:
    problem_statement = row.get("problem_statement")
    if isinstance(problem_statement, str) and problem_statement.strip():
        return problem_statement, "problem_statement"
    interface = row.get("interface")
    if isinstance(interface, str) and interface.strip():
        return interface, "interface_fallback"
    raise ValueError(f"{task_id}: problem_statement and interface must not both be empty")


def _validate_row(row: dict[str, Any], *, language: str | None = DEFAULT_LANGUAGE) -> dict[str, Any]:
    task_id = str(row.get("instance_id") or "<unknown>")
    instance_id = _as_nonempty_string(row, "instance_id", task_id)
    if (
        instance_id in {".", ".."}
        or "/" in instance_id
        or "\\" in instance_id
        or "\x00" in instance_id
    ):
        raise ValueError(f"{instance_id}: instance_id is not a safe task directory name")
    row_language = _as_nonempty_string(row, "language", instance_id).casefold()
    if language is not None and row_language != language.casefold():
        raise ValueError(f"{instance_id}: expected language={language}, got {row.get('language')!r}")
    repo = _as_nonempty_string(row, "repo", instance_id)
    base_commit = _as_nonempty_string(row, "base_commit", instance_id)
    problem_statement, instruction_source = _instruction(row, instance_id)
    image_name = _as_nonempty_string(row, "image_name", instance_id)
    gold_patch = _as_nonempty_string(row, "patch", instance_id)
    test_patch = _as_nonempty_string(row, "test_patch", instance_id)
    fail_to_pass = _as_string_list(row, "FAIL_TO_PASS", instance_id, nonempty=True)
    pass_to_pass = _as_string_list(row, "PASS_TO_PASS", instance_id, nonempty=False)
    overlap = sorted(set(fail_to_pass).intersection(pass_to_pass))
    if overlap:
        raise ValueError(f"{instance_id}: FAIL_TO_PASS overlaps PASS_TO_PASS: {overlap[:5]}")
    install_config = _install_config(row, instance_id)
    modified_files = patch_paths(gold_patch, task_id=instance_id, field="patch")
    test_files = patch_paths(test_patch, task_id=instance_id, field="test_patch")
    return {
        "instance_id": instance_id,
        "language": row_language,
        "repo": repo,
        "base_commit": base_commit,
        "problem_statement": problem_statement,
        "instruction_source": instruction_source,
        "image_name": image_name,
        "gold_patch": gold_patch,
        "test_patch": test_patch,
        "fail_to_pass": fail_to_pass,
        "pass_to_pass": pass_to_pass,
        "install_config": install_config,
        "result_parser": _result_parser_for(row_language),
        "log_parser": str(install_config["log_parser"]),
        "modified_files": modified_files,
        "test_files": test_files,
        "workdir": _repo_workdir(repo, instance_id),
    }


def _result_parser_for(language: str | None) -> str:
    return (
        PYTHON_RESULT_PARSER
        if str(language or "").casefold() == DEFAULT_LANGUAGE
        else OFFICIAL_RESULT_PARSER
    )


def _milestone_metadata(validated: dict[str, Any]) -> dict[str, Any]:
    # Keep dataset-authoritative ids verbatim. Runtime adapters may use a
    # timing-normalized alias only when it maps one-to-one; colliding aliases
    # remain incomplete and are handled by the baseline eligibility audit.
    f2p = list(validated["fail_to_pass"])
    p2p = list(validated["pass_to_pass"])
    baseline = {name: "FAILED" for name in f2p}
    baseline.update({name: "PASSED" for name in p2p})
    target = {name: "PASSED" for name in [*f2p, *p2p]}
    modified_files = list(validated["modified_files"])
    entities = [
        {
            "type": "file",
            "name": path,
            "file_name": path,
            "source": "gold_patch_path",
        }
        for path in modified_files
    ]
    return {
        "expected_output_json": json.dumps(target, ensure_ascii=False, sort_keys=True),
        "baseline_output_json": json.dumps(baseline, ensure_ascii=False, sort_keys=True),
        "target_output_json": json.dumps(target, ensure_ascii=False, sort_keys=True),
        "FAIL_TO_PASS": json.dumps(f2p, ensure_ascii=False),
        "PASS_TO_PASS": json.dumps(p2p, ensure_ascii=False),
        "modified_files": modified_files,
        "relevant_files": modified_files,
        "modified_entity_summaries": entities,
        "test_file_names": list(validated["test_files"]),
        "protected_test_paths": list(validated["test_files"]),
        "result_parser": validated["result_parser"],
        "log_parser": validated["log_parser"],
        "parser_vendor_revision": PARSER_VENDOR_REVISION,
        "parser_vendor_sha256": PARSER_VENDOR_SHA256,
        "result_adapter_version": RESULT_ADAPTER_VERSION,
        "test_results_path": TEST_RESULTS_PATH,
        "baseline_source": "dataset_contract_diagnostic",
        "runtime_baseline_required": True,
    }


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _build_task_toml(validated: dict[str, Any], source_revision: str, dataset_name: str) -> str:
    task_id = validated["instance_id"]
    repo = validated["repo"]
    workdir = validated["workdir"]
    description = f"SWE-rebench V2: {repo} @ {validated['base_commit'][:12]}"
    language = validated["language"]
    lines = [
        'schema_version = "1.1"',
        "",
        "[task]",
        f"name = {_toml_string(f'{dataset_name}/{task_id}')}",
        f"description = {_toml_string(description)}",
        f"keywords = [{_toml_string('swe-rebench-v2')}, {_toml_string(language)}, {_toml_string(repo)}]",
        "",
        "[metadata]",
        f"task_id = {_toml_string(task_id)}",
        f"repo_name = {_toml_string(repo)}",
        f"commit_hash = {_toml_string(validated['base_commit'])}",
        f"language = {_toml_string(language)}",
        f"source_revision = {_toml_string(source_revision)}",
        f"result_parser = {_toml_string(validated['result_parser'])}",
        f"log_parser = {_toml_string(validated['log_parser'])}",
        f"parser_vendor_revision = {_toml_string(PARSER_VENDOR_REVISION)}",
        f"parser_vendor_sha256 = {_toml_string(PARSER_VENDOR_SHA256)}",
        f"result_adapter_version = {RESULT_ADAPTER_VERSION}",
        "protected_test_paths = ["
        + ", ".join(_toml_string(path) for path in validated["test_files"])
        + "]",
        f"test_results_path = {_toml_string(TEST_RESULTS_PATH)}",
        "runtime_baseline_required = true",
    ]
    shadow_resources = validated.get("shadow_sandbox_resources")
    if isinstance(shadow_resources, dict) and shadow_resources:
        lines.extend(
            [
                "",
                "[rllm.shadow_sandbox_resources]",
                f"cpus = {int(shadow_resources['cpus'])}",
                f"memory_mb = {int(shadow_resources['memory_mb'])}",
            ]
        )
    lines.extend(
        [
        "",
        "[environment]",
        f"docker_image = {_toml_string(validated['image_name'])}",
        f"workdir = {_toml_string(workdir)}",
        f"cpus = {_DEFAULT_RESOURCES['cpus']}",
        f"memory_mb = {_DEFAULT_RESOURCES['memory_mb']}",
        f"storage_mb = {_DEFAULT_RESOURCES['storage_mb']}",
        f"build_timeout_sec = {_DEFAULT_RESOURCES['build_timeout_sec']}",
        "allow_internet = true",
        "",
        "[environment.env]",
        'PAGER = "cat"',
        'MANPAGER = "cat"',
        'LESS = "-R"',
        'PIP_PROGRESS_BAR = "off"',
        'TQDM_DISABLE = "1"',
        'CI = "1"',
        'no_proxy = "localhost,127.0.0.1,::1"',
        'NO_PROXY = "localhost,127.0.0.1,::1"',
        ]
    )
    lines.extend(
        [
            "",
            "[agent]",
            f"timeout_sec = {_DEFAULT_TIMEOUTS['agent_timeout_sec']}",
            "",
            "[verifier]",
            f"timeout_sec = {_DEFAULT_TIMEOUTS['verifier_timeout_sec']}",
            "",
        ]
    )
    return "\n".join(lines)


def _build_dockerfile(image_name: str, workdir: str) -> str:
    return f"FROM {image_name}\nENTRYPOINT []\nWORKDIR {workdir}\n"


def _build_solution_script(workdir: str) -> str:
    return f"""#!/bin/bash
set -euo pipefail
cd {json.dumps(workdir)}
git config --global --add safe.directory {json.dumps(workdir)} 2>/dev/null || true
git apply -v --3way --recount --ignore-space-change --whitespace=nowarn /solution/gold.patch
"""


_PATCH_PATH_DISCOVERY_SCRIPT = r'''    # Old materializations can omit header-only empty additions. Ask Git for
    # actual patch paths before any restore/delete; metadata remains a union.
    inventory = run(["git", "-c", "safe.directory=" + workdir, "apply", "--numstat", "-z", PATCH], workdir)
    if inventory.returncode != 0:
        raise RuntimeError("test patch path discovery failed: " + inventory.stderr[-1000:])
    for record in inventory.stdout.split("\0"):
        if not record:
            continue
        fields = record.split("\t", 2)
        if len(fields) != 3:
            raise RuntimeError("invalid test patch path inventory")
        path = safe_path(fields[2])
        # --numstat also reports paths that git apply would refuse to write.
        components = [part for part in path.split("/") if part not in ("", ".")]
        if not components or ".git" in components:
            raise RuntimeError("unsafe test patch path: %r" % path)
        if path not in paths:
            paths.append(path)
'''


_GRADER_SCRIPT = r'''#!/usr/bin/env python3
from __future__ import print_function

import json
import os
import re
import shutil
import subprocess
import sys
import time

INSTANCE = os.environ.get("RLLM_INSTANCE_JSON", "/tests/instance.json")
PATCH = os.environ.get("RLLM_TEST_PATCH", "/tests/test.patch")
PARSERS = os.environ.get("RLLM_LOG_PARSERS", "/tests/log_parsers.py")
OUTPUT = os.environ.get("RLLM_TEST_OUTPUT", "/tmp/test_output.txt")
REWARD = os.environ.get("RLLM_REWARD_JSON", "/tmp/rllm/reward.json")
RESULTS = os.environ.get("RLLM_TEST_RESULTS_JSON", "/tmp/rllm/test_results.json")
PARTITION_JEST_RESULTS = "/tmp/rllm/partition-jest-results.json"
STATUSES = ("PASSED", "FAILED", "SKIPPED", "ERROR")
RESULT_PARSER = "swerebench_v2_official_v2"
LOG_PARSER = "unknown"
LANGUAGE = "unknown"
PARSER_VENDOR_REVISION = "a4c70f7444c6d35ee211bea9dfca82a6c45ad24e"
PARSER_VENDOR_SHA256 = "__PARSER_VENDOR_SHA256__"
RESULT_ADAPTER_VERSION = 6
ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
TIMING_NORMALIZE_RES = (
    re.compile(r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$", re.IGNORECASE),
    re.compile(r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b", re.IGNORECASE),
    re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$", re.IGNORECASE),
)


def atomic_json(path, value):
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def finish(reward, parsed=None, metadata=None, complete=False, execution=None):
    parsed = {} if parsed is None else parsed
    metadata = metadata or {}
    execution = execution or {
        "timed_out": False,
        "resource_exhausted": False,
        "commands": [],
    }
    atomic_json(
        RESULTS,
        {
            "schema_version": 3,
            "parser": RESULT_PARSER,
            "log_parser": LOG_PARSER,
            "language": LANGUAGE,
            "parser_vendor_revision": PARSER_VENDOR_REVISION,
            "parser_vendor_sha256": PARSER_VENDOR_SHA256,
            "result_adapter_version": RESULT_ADAPTER_VERSION,
            "complete": bool(complete),
            "test_results": parsed,
            "execution": execution,
            "diagnostics": dict(metadata),
        },
    )
    atomic_json(
        REWARD,
        {
            "reward": float(reward),
            "is_correct": float(reward) >= 1.0,
            "signals": metadata.pop("signals", {}),
            "metadata": metadata,
        },
    )


def normalize(name):
    for pattern in TIMING_NORMALIZE_RES:
        name = pattern.sub("", name)
    return name.strip()


def partition_canonical_name(raw_name, expected, runner):
    raw_name = normalize(str(raw_name))
    if raw_name in expected:
        return raw_name
    normalized = re.sub(r"\s*(?:>|›)\s*", " ", raw_name).strip()
    candidates = []
    for expected_name in expected:
        expected_normalized = re.sub(
            r"\s*(?:>|›)\s*", " ", normalize(expected_name)
        ).strip()
        if expected_normalized == normalized or expected_normalized.endswith(" " + normalized):
            candidates.append(expected_name)
            continue
        if runner == "go":
            match = re.search(
                r"((?:Test|Example|Fuzz)[^\s:]*(?:/[^\s]+)*)$",
                expected_normalized,
            )
            if match and match.group(1) == normalized:
                candidates.append(expected_name)
    return candidates[0] if len(candidates) == 1 else raw_name


def merge_partition_result(results, name, status):
    priority = {"PASSED": 0, "SKIPPED": 1, "FAILED": 2, "ERROR": 3}
    previous = results.get(name)
    if previous is None or priority[status] > priority.get(previous, -1):
        results[name] = status


def partition_status_counts(statuses):
    values = list(statuses)
    return {
        "passed": values.count("PASSED"),
        "failed": values.count("FAILED"),
        "errored": values.count("ERROR"),
        "skipped": values.count("SKIPPED"),
    }


def partition_summary_counts(runner, region):
    clean = ANSI_ESCAPE_RE.sub("", region)

    def integer(pattern):
        matches = re.findall(pattern, clean, flags=re.IGNORECASE | re.MULTILINE)
        return int(matches[-1]) if matches else 0

    if runner == "pytest":
        rows = re.findall(r"^=+\s*(.*?)\s*=+\s*$", clean, flags=re.MULTILINE)
        if rows:
            row = rows[-1]
            passed = int((re.findall(r"(\d+)\s+passed", row) or [0])[-1])
            passed += int((re.findall(r"(\d+)\s+xpassed", row) or [0])[-1])
            skipped = int((re.findall(r"(\d+)\s+skipped", row) or [0])[-1])
            skipped += int((re.findall(r"(\d+)\s+xfailed", row) or [0])[-1])
            counts = {
                "passed": passed,
                "failed": int((re.findall(r"(\d+)\s+failed", row) or [0])[-1]),
                "errored": int((re.findall(r"(\d+)\s+errors?", row) or [0])[-1]),
                "skipped": skipped,
            }
            if sum(counts.values()):
                return counts, "pytest_summary"
    if runner == "jest":
        total = integer(r"Tests:.*?(\d+)\s+total")
        if total:
            passed = integer(r"Tests:[^\n]*?(\d+)\s+passed")
            failed = integer(r"Tests:[^\n]*?(\d+)\s+failed")
            skipped = integer(r"Tests:[^\n]*?(\d+)\s+skipped")
            skipped += integer(r"Tests:[^\n]*?(\d+)\s+todo")
            return {
                "passed": passed,
                "failed": failed,
                "errored": max(0, total - passed - failed - skipped),
                "skipped": skipped,
            }, "jest_summary"
    if runner == "mocha":
        passed = integer(r"^\s*(\d+)\s+passing\b")
        failed = integer(r"^\s*(\d+)\s+failing\b")
        skipped = integer(r"^\s*(\d+)\s+pending\b")
        if passed + failed + skipped:
            return {
                "passed": passed,
                "failed": failed,
                "errored": 0,
                "skipped": skipped,
            }, "mocha_summary"
    if runner == "vitest":
        rows = re.findall(r"^\s*Tests\s+(.+)$", clean, flags=re.MULTILINE)
        if rows:
            row = rows[-1]
            counts = {
                "passed": int((re.findall(r"(\d+)\s+passed", row) or [0])[-1]),
                "failed": int((re.findall(r"(\d+)\s+failed", row) or [0])[-1]),
                "errored": int((re.findall(r"(\d+)\s+errors?", row) or [0])[-1]),
                "skipped": (
                    int((re.findall(r"(\d+)\s+skipped", row) or [0])[-1])
                    + int((re.findall(r"(\d+)\s+todo", row) or [0])[-1])
                ),
            }
            if sum(counts.values()):
                return counts, "vitest_summary"
    return None, None


def make_partition_observation(instance, collector, counts):
    probe = instance.get("rllm_partition_probe")
    if (
        not isinstance(probe, dict)
        or probe.get("schema_version") != 3
        or probe.get("partition") not in {"f2p", "p2p"}
        or not isinstance(probe.get("adapter"), str)
        or type(probe.get("expected")) is not int
        or probe.get("expected") < 0
        or not isinstance(collector, str)
        or not isinstance(counts, dict)
    ):
        return None
    reported = sum(counts.get(key, 0) for key in ("passed", "failed", "errored", "skipped"))
    if reported > probe["expected"]:
        return None
    return {
        "schema_version": 1,
        "runner": probe["adapter"],
        "partition": probe["partition"],
        "expected": probe["expected"],
        "shard_index": probe.get("shard_index"),
        "shard_count": probe.get("shard_count"),
        "selector_verified": True,
        "collector": collector,
        "counts": dict(counts),
        "reported": reported,
    }


def collect_partition_results(instance, region):
    probe = instance.get("rllm_partition_probe")
    if not isinstance(probe, dict):
        return {}, None, None
    runner = str(probe.get("adapter") or "")
    expected = list(instance.get("FAIL_TO_PASS") or []) + list(
        instance.get("PASS_TO_PASS") or []
    )
    results = {}
    if runner == "go":
        raw_results = {}
        for line in region.splitlines():
            try:
                event = json.loads(line)
            except Exception:
                continue
            if not isinstance(event, dict) or not isinstance(event.get("Test"), str):
                continue
            action = str(event.get("Action") or "").casefold()
            status = {"pass": "PASSED", "fail": "FAILED", "skip": "SKIPPED"}.get(action)
            if status:
                raw_key = (str(event.get("Package") or ""), event["Test"])
                merge_partition_result(raw_results, raw_key, status)
                name = partition_canonical_name(event["Test"], expected, runner)
                merge_partition_result(results, name, status)
        # go test -json emits both a selected subtest and its structural
        # parent. Count the contract node when either side of that hierarchy
        # is annotated, and ignore the unannotated structural event.
        expected_go = {}
        for name in expected:
            match = re.search(r"((?:Test|Example|Fuzz)[^\s:]*(?:/[^\s]+)*)$", normalize(name))
            expected_go.setdefault(
                match.group(1) if match else normalize(name), []
            ).append(name)
        counted = {}
        for (package, raw_name), status in raw_results.items():
            normalized_name = normalize(raw_name)
            match = re.search(r"((?:Test|Example|Fuzz)[^\s:]*(?:/[^\s]+)*)$", normalized_name)
            go_name = match.group(1) if match else normalized_name
            aliases = expected_go.get(go_name, [])
            if len(aliases) == 1:
                merge_partition_result(counted, aliases[0], status)
                continue
            if any(
                go_name.startswith(expected_name + "/")
                or expected_name.startswith(go_name + "/")
                for expected_name in expected_go
            ):
                continue
            # The selector is exact but display names may still differ by a
            # package prefix. Preserve such terminal events for aggregate
            # counting; a count above the contract is rejected by the runtime.
            merge_partition_result(counted, (package, raw_name), status)
        return results, "go_test_json", partition_status_counts(counted.values())
    if runner == "jest":
        try:
            payload = json.load(open(PARTITION_JEST_RESULTS, encoding="utf-8"))
        except Exception:
            payload = {}
        for suite in payload.get("testResults") or []:
            for assertion in suite.get("assertionResults") or []:
                raw_name = assertion.get("fullName") or assertion.get("title")
                raw_status = str(assertion.get("status") or "").casefold()
                status = {
                    "passed": "PASSED",
                    "failed": "FAILED",
                    "pending": "SKIPPED",
                    "todo": "SKIPPED",
                    "disabled": "SKIPPED",
                }.get(raw_status)
                if raw_name and status:
                    name = partition_canonical_name(raw_name, expected, runner)
                    merge_partition_result(results, name, status)
        if results:
            statuses = []
            for suite in payload.get("testResults") or []:
                for assertion in suite.get("assertionResults") or []:
                    raw_status = str(assertion.get("status") or "").casefold()
                    status = {
                        "passed": "PASSED",
                        "failed": "FAILED",
                        "pending": "SKIPPED",
                        "todo": "SKIPPED",
                        "disabled": "SKIPPED",
                    }.get(raw_status)
                    if status:
                        statuses.append(status)
            return results, "jest_json", partition_status_counts(statuses)
        counts, collector = partition_summary_counts(runner, region)
        return results, collector, counts
    if runner == "mocha":
        decoder = json.JSONDecoder()
        payload = None
        for index, character in enumerate(region):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(region[index:])
            except Exception:
                continue
            if isinstance(candidate, dict) and isinstance(candidate.get("tests"), list):
                payload = candidate
        statuses = []
        for test in (payload or {}).get("tests") or []:
            raw_name = test.get("fullTitle") or test.get("title")
            if not raw_name:
                continue
            if test.get("pending"):
                status = "SKIPPED"
            elif test.get("fail") or test.get("state") == "failed" or test.get("err"):
                status = "FAILED"
            else:
                status = "PASSED"
            statuses.append(status)
            name = partition_canonical_name(raw_name, expected, runner)
            merge_partition_result(results, name, status)
        if payload is not None:
            return results, "mocha_json", partition_status_counts(statuses)
        counts, collector = partition_summary_counts(runner, region)
        return results, collector, counts
    if runner == "ava":
        statuses = []
        for line in region.splitlines():
            match = re.match(
                r"\s*(not ok|ok)\s+\d+\s+-\s+(.+?)(\s+#.*)?$",
                line,
            )
            if not match:
                continue
            directive = str(match.group(3) or "").casefold()
            status = (
                "SKIPPED"
                if "# skip" in directive or "# todo" in directive
                else "FAILED"
                if match.group(1) == "not ok"
                else "PASSED"
            )
            statuses.append(status)
            name = partition_canonical_name(match.group(2), expected, runner)
            merge_partition_result(results, name, status)
        return results, "ava_tap", partition_status_counts(statuses)
    if runner in {"pytest", "vitest"}:
        counts, collector = partition_summary_counts(runner, region)
        return results, collector, counts
    return {}, None, None


def parse_log_pytest(log):
    # Vendored semantics from PrimeIntellect research-environments
    # swerebench_v2_v1/log_parsers.py at a90fbd708de9ab18f85b5ffc3a0bdc60825dcc84.
    result = {}
    for raw_line in log.split("\n"):
        line = ANSI_ESCAPE_RE.sub("", raw_line)
        if not any(line.startswith(status) for status in STATUSES) or line.split()[0] not in STATUSES:
            continue
        if line.startswith("FAILED"):
            line = line.replace(" - ", " ")
        parts = line.split()
        if len(parts) <= 1:
            continue
        result[parts[1]] = parts[0]
    return result


def safe_path(path):
    if not path or os.path.isabs(path) or ".." in path.split("/"):
        raise RuntimeError("unsafe test patch path: %r" % path)
    return path


def run(command, cwd, env=None):
    return subprocess.run(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)


def cgroup_oom_kill_count():
    """Read an explicit cgroup OOM-kill counter when the sandbox exposes one."""
    for path in (
        "/sys/fs/cgroup/memory.events.local",
        "/sys/fs/cgroup/memory.events",
    ):
        try:
            values = {}
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    key, raw_value = line.split(None, 1)
                    values[key] = int(raw_value)
            if "oom_kill" in values:
                return values["oom_kill"]
        except (OSError, TypeError, ValueError):
            continue
    return None


def process_signal(returncode):
    if returncode < 0:
        return -returncode
    if returncode in (134, 137, 139):
        return returncode - 128
    return None


def restore_and_apply(instance, workdir):
    base = instance.get("base_commit") or ""
    if not base:
        raise RuntimeError("base_commit is empty")
    paths = [safe_path(path) for path in instance.get("milestone", {}).get("test_file_names", [])]
__PATCH_PATH_DISCOVERY__
    if not paths:
        raise RuntimeError("test patch contains no modified files")
    run(["git", "config", "--global", "--add", "safe.directory", workdir], workdir)
    base_tree = run(["git", "cat-file", "-e", base + "^{tree}"], workdir)
    if base_tree.returncode != 0:
        raise RuntimeError("test patch base unavailable: " + base_tree.stderr[-1000:])
    for path in paths:
        listed = run(["git", "ls-tree", "-z", "--name-only", base, "--", path], workdir)
        if listed.returncode != 0:
            raise RuntimeError("test patch path lookup failed: " + listed.stderr[-1000:])
        exists = bool(listed.stdout)
        if exists:
            restored = run(["git", "checkout", base, "--", path], workdir)
            if restored.returncode != 0:
                raise RuntimeError("failed to restore %s: %s" % (path, restored.stderr[-500:]))
        else:
            target = os.path.join(workdir, path)
            if os.path.isdir(target) and not os.path.islink(target):
                shutil.rmtree(target)
            else:
                try:
                    os.unlink(target)
                except FileNotFoundError:
                    pass
    checked = run(["git", "apply", "--check", PATCH], workdir)
    if checked.returncode != 0:
        raise RuntimeError("test patch check failed: " + checked.stderr[-1000:])
    applied = run(["git", "apply", PATCH], workdir)
    if applied.returncode != 0:
        raise RuntimeError("test patch apply failed: " + applied.stderr[-1000:])


def main():
    global RESULT_PARSER, LOG_PARSER, LANGUAGE
    for path in (REWARD, RESULTS, OUTPUT):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    for stale in (REWARD, RESULTS, OUTPUT, PARTITION_JEST_RESULTS):
        try:
            os.unlink(stale)
        except FileNotFoundError:
            pass
    try:
        with open(INSTANCE, encoding="utf-8") as handle:
            instance = json.load(handle)
        RESULT_PARSER = instance["result_parser"]
        LOG_PARSER = str(instance["install_config"].get("log_parser") or "unknown")
        LANGUAGE = str(instance.get("language") or "unknown")
        workdir = instance["workdir"]
        restore_and_apply(instance, workdir)
        config = instance["install_config"]
        commands = config.get("test_cmd")
        if isinstance(commands, str):
            commands = [commands]
        if not isinstance(commands, list) or not commands:
            raise RuntimeError("install_config.test_cmd is empty")
        # Keep the enterprise image's native PATH/JAVA_HOME/toolchain.  A
        # benchmark-wide PATH is incorrect for multilingual images (notably
        # Java 17/21 images) and can silently select an older host JDK.
        env = dict(os.environ)
        exit_codes = []
        command_durations = []
        command_executions = []
        with open(OUTPUT, "w", encoding="utf-8", errors="replace") as output:
            output.write("SWEREBENCH_V2_TEST_OUTPUT_START\n")
            output.flush()
            for command in commands:
                oom_kills_before = cgroup_oom_kill_count()
                command_started = time.monotonic()
                completed = subprocess.run(
                    ["bash", "-o", "pipefail", "-c", command],
                    cwd=workdir,
                    env=env,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                )
                exit_codes.append(completed.returncode)
                command_duration = time.monotonic() - command_started
                command_durations.append(command_duration)
                oom_kills_after = cgroup_oom_kill_count()
                oom_kill_delta = (
                    max(0, oom_kills_after - oom_kills_before)
                    if oom_kills_before is not None and oom_kills_after is not None
                    else 0
                )
                command_executions.append(
                    {
                        "index": len(command_executions),
                        "exit_code": completed.returncode,
                        "signal": process_signal(completed.returncode),
                        "timed_out": False,
                        "oom_kill_delta": oom_kill_delta,
                        "duration_s": command_duration,
                    }
                )
            output.write("\nSWEREBENCH_V2_TEST_OUTPUT_END\n")
        with open(OUTPUT, encoding="utf-8", errors="replace") as handle:
            log = handle.read()
        region = log.split("SWEREBENCH_V2_TEST_OUTPUT_START", 1)[-1]
        region = region.rsplit("SWEREBENCH_V2_TEST_OUTPUT_END", 1)[0]
        parser_name = LOG_PARSER
        if RESULT_PARSER == "swerebench_v2_pytest_v1":
            if parser_name != "parse_log_pytest":
                raise RuntimeError("Python verifier requires parse_log_pytest")
            parsed = parse_log_pytest(region)
        else:
            import importlib.util

            spec = importlib.util.spec_from_file_location("rllm_swerebench_log_parsers", PARSERS)
            if spec is None or spec.loader is None:
                raise RuntimeError("unable to load SWE-rebench V2 log parsers")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            parser = module.NAME_TO_PARSER.get(parser_name)
            if parser is None:
                raise RuntimeError("unsupported SWE-rebench V2 log parser: %r" % parser_name)
            parsed = parser(region) or {}
        expected = list(instance["FAIL_TO_PASS"] + instance["PASS_TO_PASS"])
        expected_set = set(expected)
        expected_by_normalized = {}
        for expected_name in expected:
            expected_by_normalized.setdefault(normalize(expected_name), []).append(expected_name)
        normalized = {}
        for raw_name, status in parsed.items():
            raw_name = str(raw_name).strip()
            name = normalize(raw_name)
            if not name:
                raise RuntimeError("parser returned an empty test name")
            aliases = expected_by_normalized.get(name, [])
            canonical = raw_name if raw_name in expected_set else aliases[0] if len(aliases) == 1 else name
            if canonical in normalized:
                raise RuntimeError("parser returned duplicate canonical test name: %s" % canonical)
            value = str(getattr(status, "value", status))
            if value not in STATUSES:
                raise RuntimeError("parser returned unsupported status for %s: %r" % (name, value))
            normalized[canonical] = value
        partition_results, partition_collector, partition_counts = collect_partition_results(instance, region)
        partition_observation = make_partition_observation(
            instance,
            partition_collector,
            partition_counts,
        )
        if partition_results:
            normalized = partition_results
        missing = [name for name in expected if name not in normalized]
        failing = {name: normalized.get(name) for name in expected if normalized.get(name) != "PASSED"}
        complete = bool(expected) and not missing
        resolved = complete and not failing
        finish(
            1.0 if resolved else 0.0,
            normalized,
            {
                "signals": {
                    "expected_tests": len(expected),
                    "parsed_tests": len(parsed),
                    "missing_tests": len(missing),
                    "failing_tests": len(failing),
                    "test_command_count": len(commands),
                },
                "missing_tests": missing[:100],
                "failing_tests": dict(list(failing.items())[:100]),
                "test_exit_codes": exit_codes,
                "test_command_durations_s": command_durations,
                "parser": parser_name,
                "parser_version": 2,
                "partition_collector": partition_collector,
                "partition_observation": partition_observation,
                "language": LANGUAGE,
                "complete": complete,
                "log_tail": log[-4000:],
            },
            complete=complete,
            execution={
                "timed_out": False,
                "resource_exhausted": any(
                    command["oom_kill_delta"] > 0
                    for command in command_executions
                ),
                "commands": command_executions,
            },
        )
    except Exception as exc:
        finish(
            0.0,
            metadata={
                "error": "%s: %s" % (type(exc).__name__, exc),
                "infrastructure_failure": {
                    "reason": "verifier_framework_failed",
                    "stage": "verifier",
                    "exception_type": type(exc).__name__,
                    "error_summary": str(exc)[-2000:],
                    "retryable": False,
                },
                "parser": LOG_PARSER,
                "language": LANGUAGE,
                "complete": False,
            },
            complete=False,
        )


if __name__ == "__main__":
    main()
'''.replace("__PARSER_VENDOR_SHA256__", PARSER_VENDOR_SHA256).replace(
    "__PATCH_PATH_DISCOVERY__\n", _PATCH_PATH_DISCOVERY_SCRIPT,
)


_TEST_SCRIPT = """#!/bin/bash
set -uo pipefail
mkdir -p /tmp/rllm /logs/verifier
python3 /tests/grader.py
exit 0
"""


def _instance_payload(row: dict[str, Any], validated: dict[str, Any], milestone: dict[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in row.items() if key not in {"patch", "test_patch"}}
    payload.update(
        {
            "schema_version": TASK_SCHEMA_VERSION,
            "patch_file": "/solution/gold.patch",
            "test_patch_file": "/tests/test.patch",
            "workdir": validated["workdir"],
            "result_parser": validated["result_parser"],
            "log_parser": validated["log_parser"],
            "parser_vendor_revision": PARSER_VENDOR_REVISION,
            "parser_vendor_sha256": PARSER_VENDOR_SHA256,
            "result_adapter_version": RESULT_ADAPTER_VERSION,
            "test_results_path": TEST_RESULTS_PATH,
            "milestone": {
                "baseline_output_json": json.loads(milestone["baseline_output_json"]),
                "target_output_json": json.loads(milestone["target_output_json"]),
                "baseline_source": milestone["baseline_source"],
                "runtime_baseline_required": True,
                "modified_files": milestone["modified_files"],
                "relevant_files": milestone["relevant_files"],
                "modified_entity_summaries": milestone["modified_entity_summaries"],
                "test_file_names": milestone["test_file_names"],
                "protected_test_paths": milestone["protected_test_paths"],
            },
        }
    )
    if validated.get("shadow_sandbox_resources"):
        payload["shadow_sandbox_resources"] = dict(
            validated["shadow_sandbox_resources"]
        )
    return payload


def _materialize_task(
    staging: Path,
    row: dict[str, Any],
    validated: dict[str, Any],
    source_revision: str,
    dataset_name: str,
) -> dict[str, Any]:
    milestone = _milestone_metadata(validated)
    staging.mkdir(parents=True)
    (staging / "environment").mkdir()
    (staging / "tests").mkdir()
    (staging / "solution").mkdir()

    (staging / "task.toml").write_text(
        _build_task_toml(validated, source_revision, dataset_name), encoding="utf-8"
    )
    (staging / "instruction.md").write_text(validated["problem_statement"].rstrip() + "\n", encoding="utf-8")
    (staging / "environment" / "Dockerfile").write_text(
        _build_dockerfile(validated["image_name"], validated["workdir"]), encoding="utf-8"
    )
    (staging / "tests" / "test.sh").write_text(_TEST_SCRIPT, encoding="utf-8")
    (staging / "tests" / "test.sh").chmod(0o755)
    (staging / "tests" / "grader.py").write_text(_GRADER_SCRIPT, encoding="utf-8")
    (staging / "tests" / "grader.py").chmod(0o755)
    shutil.copy2(_OFFICIAL_LOG_PARSERS_ASSET, staging / "tests" / "log_parsers.py")
    (staging / "tests" / "test.patch").write_text(validated["test_patch"], encoding="utf-8")
    (staging / "tests" / "instance.json").write_text(
        json.dumps(
            _instance_payload(row, validated, milestone),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    (staging / "solution" / "gold.patch").write_text(validated["gold_patch"], encoding="utf-8")
    (staging / "solution" / "solve.sh").write_text(_build_solution_script(validated["workdir"]), encoding="utf-8")
    (staging / "solution" / "solve.sh").chmod(0o755)

    marker = _task_marker(row, validated, source_revision)
    (staging / ".materialized.json").write_text(
        json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return milestone


def _task_marker(
    row: dict[str, Any],
    validated: dict[str, Any],
    source_revision: str,
) -> dict[str, Any]:
    return {
        "schema_version": TASK_SCHEMA_VERSION,
        "task_id": validated["instance_id"],
        "source_revision": source_revision,
        "row_fingerprint": _json_fingerprint(row),
        "result_parser": validated["result_parser"],
        "log_parser": validated["log_parser"],
        "parser_vendor_revision": PARSER_VENDOR_REVISION,
        "parser_vendor_sha256": PARSER_VENDOR_SHA256,
        "result_adapter_version": RESULT_ADAPTER_VERSION,
        "shadow_sandbox_resources": validated.get("shadow_sandbox_resources"),
    }


def _task_matches_source_contract(
    task_dir: Path,
    row: dict[str, Any],
    source_revision: str,
    validated: dict[str, Any],
) -> bool:
    """Return whether an old/partial task can be upgraded without rebuilding."""

    if any(not (task_dir / relative).is_file() for relative in _REQUIRED_TASK_FILES):
        return False
    try:
        marker = json.loads(
            (task_dir / ".materialized.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False
    return (
        marker.get("task_id") == validated["instance_id"]
        and marker.get("source_revision") == source_revision
        and marker.get("row_fingerprint") == _json_fingerprint(row)
    )


def _upgrade_task_in_place(
    task_dir: Path,
    row: dict[str, Any],
    validated: dict[str, Any],
    source_revision: str,
    dataset_name: str,
) -> dict[str, Any]:
    """Upgrade schema/provenance files and replace the task marker last."""

    milestone = _milestone_metadata(validated)
    _atomic_replace_text(
        task_dir / "task.toml",
        _build_task_toml(validated, source_revision, dataset_name),
    )
    _atomic_replace_text(
        task_dir / "tests" / "grader.py",
        _GRADER_SCRIPT,
        mode=0o755,
    )
    _atomic_replace_text(
        task_dir / "tests" / "test.sh",
        _TEST_SCRIPT,
        mode=0o755,
    )
    _atomic_replace_text(
        task_dir / "tests" / "log_parsers.py",
        _OFFICIAL_LOG_PARSERS_ASSET.read_text(encoding="utf-8"),
    )
    _atomic_replace_text(
        task_dir / "tests" / "instance.json",
        json.dumps(
            _instance_payload(row, validated, milestone),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
    )
    _atomic_replace_text(
        task_dir / ".materialized.json",
        json.dumps(
            _task_marker(row, validated, source_revision),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return milestone


def _task_complete(
    task_dir: Path,
    row: dict[str, Any],
    source_revision: str,
    validated: dict[str, Any],
) -> bool:
    if any(not (task_dir / relative).is_file() for relative in _REQUIRED_TASK_FILES):
        return False
    try:
        marker = json.loads((task_dir / ".materialized.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        marker.get("schema_version") == TASK_SCHEMA_VERSION
        and marker.get("source_revision") == source_revision
        and marker.get("row_fingerprint") == _json_fingerprint(row)
        and marker.get("result_parser") == validated["result_parser"]
        and marker.get("log_parser") == validated["log_parser"]
        and marker.get("parser_vendor_revision") == PARSER_VENDOR_REVISION
        and marker.get("parser_vendor_sha256") == PARSER_VENDOR_SHA256
        and marker.get("result_adapter_version") == RESULT_ADAPTER_VERSION
        and marker.get("shadow_sandbox_resources")
        == validated.get("shadow_sandbox_resources")
    )


def _write_dataset_toml(
    out: Path,
    *,
    name: str,
    split: str,
    default_agent: str,
    language: str | None,
    languages: Collection[str] | None = None,
) -> None:
    normalized_languages = _normalize_languages(language=language, languages=languages)
    if normalized_languages == (DEFAULT_LANGUAGE,):
        scope = "Python-only slice"
    elif normalized_languages is None:
        scope = "Complete multilingual dataset"
    else:
        scope = f"{', '.join(normalized_languages)} language slice"
    description = (
        f"{scope} of PrimeIntellect/SWE-rebench-V2-Filtered-Verified "
        "with held-out test-patch grading and future Milestone metadata."
    )
    content = "\n".join(
        [
            "[dataset]",
            f"name = {_toml_string(name)}",
            'type = "sandbox"',
            f"description = {_toml_string(description)}",
            'default_sandbox = "docker"',
            f"default_agent = {_toml_string(default_agent)}",
            f"split = {_toml_string(split)}",
            "",
            "[verifier]",
            'script = "tests/test.sh"',
            "",
        ]
    )
    _atomic_write_text(out / "dataset.toml", content)


def _manifest_config(
    *,
    name: str,
    split: str,
    hf_repo_id: str,
    hf_revision: str,
    hf_split: str,
    limit: int | None,
    language: str | None,
    languages: Collection[str] | None = None,
    shadow_resource_override_task_ids: Collection[str] | None = None,
) -> dict[str, Any]:
    normalized_languages = _normalize_languages(language=language, languages=languages)
    selection = (
        "all"
        if normalized_languages is None
        else ",".join(normalized_languages)
    )
    return {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "name": name,
        "split": split,
        "hf_repo_id": hf_repo_id,
        "hf_revision": hf_revision,
        "hf_split": hf_split,
        "language": selection,
        "limit": limit,
        "result_parser_profile": (
            PYTHON_RESULT_PARSER
            if normalized_languages == (DEFAULT_LANGUAGE,)
            else "per_row_language_v5"
        ),
        "parser_vendor_revision": PARSER_VENDOR_REVISION,
        "parser_vendor_sha256": PARSER_VENDOR_SHA256,
        "result_adapter_version": RESULT_ADAPTER_VERSION,
        "shadow_resource_override_task_ids": list(
            _normalize_task_ids(shadow_resource_override_task_ids)
        ),
    }


def _normalize_task_ids(values: Collection[str] | None) -> tuple[str, ...]:
    """Return stable, non-empty task ids for selective shadow resources."""

    if values is None:
        return ()
    if isinstance(values, str):
        values = values.split(",")
    return tuple(
        sorted(
            {
                str(value).strip()
                for value in values
                if str(value).strip()
            }
        )
    )


def _normalize_languages(
    *,
    language: str | None,
    languages: Collection[str] | None,
) -> tuple[str, ...] | None:
    """Return a canonical language selection; ``None`` means all languages."""

    if languages is None:
        return None if language is None else (language.strip().casefold(),)
    if language is not None:
        raise ValueError("language and languages are mutually exclusive")
    normalized = tuple(
        sorted(
            {
                str(value).strip().casefold()
                for value in languages
                if str(value).strip()
            }
        )
    )
    if not normalized:
        raise ValueError("languages must contain at least one non-empty language")
    return normalized


def _load_manifest(out: Path) -> dict[str, Any] | None:
    path = out / "materialization.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid materialization manifest at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"materialization manifest at {path} must contain an object")
    return value


def _manifest_identity(config: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "name",
        "split",
        "hf_repo_id",
        "hf_revision",
        "hf_split",
        "language",
        "limit",
    )
    return {key: config.get(key) for key in keys}


def _assert_manifest_compatible(manifest: dict[str, Any], config: dict[str, Any], out: Path) -> None:
    existing = manifest.get("config")
    if not isinstance(existing, dict) or _manifest_identity(existing) != _manifest_identity(config):
        raise ValueError(
            f"{out} was materialized with a different source/configuration; "
            "set CLEAN=1 to rebuild it explicitly"
        )


def materialization_complete(
    *,
    name: str,
    split: str,
    out_dir: str | Path,
    hf_repo_id: str,
    hf_revision: str,
    hf_split: str,
    limit: int | None,
    language: str | None = DEFAULT_LANGUAGE,
    languages: Collection[str] | None = None,
    shadow_resource_override_task_ids: Collection[str] | None = None,
) -> bool:
    """Return whether manifest and registered parquet prove completion."""

    # Registry rows are consumed inside cluster containers where workstation
    # site-specific storage symlinks do not exist.
    # Persist the physical shared-filesystem path instead of the caller's
    # lexical alias.
    out = Path(out_dir).expanduser().resolve(strict=False)
    manifest = _load_manifest(out)
    if manifest is None:
        return False
    config = _manifest_config(
        name=name,
        split=split,
        hf_repo_id=hf_repo_id,
        hf_revision=hf_revision,
        hf_split=hf_split,
        limit=limit,
        language=language,
        languages=languages,
        shadow_resource_override_task_ids=shadow_resource_override_task_ids,
    )
    _assert_manifest_compatible(manifest, config, out)
    if (
        manifest.get("schema_version") != MATERIALIZATION_SCHEMA_VERSION
        or manifest.get("config") != config
    ):
        return False
    if manifest.get("status") != "complete":
        return False
    selected = manifest.get("selected_tasks")
    if not isinstance(selected, int) or selected <= 0 or manifest.get("completed_tasks") != selected:
        return False
    if not (out / "dataset.toml").is_file():
        return False

    from rllm.data import DatasetRegistry

    info = DatasetRegistry.get_dataset_info(name)
    if not info or split not in info.get("splits", {}):
        return False
    split_info = info["splits"][split]
    if split_info.get("num_examples") != selected:
        return False
    dataset_path = Path(DatasetRegistry._resolve_path(split_info["path"]))
    if not dataset_path.is_file():
        return False
    verl_dataset_path = Path(DatasetRegistry._verl_path_for(str(dataset_path)))
    if not verl_dataset_path.is_file():
        return False
    try:
        import pyarrow.parquet as pq

        if pq.read_metadata(dataset_path).num_rows != selected:
            return False
        if pq.read_metadata(verl_dataset_path).num_rows != selected:
            return False
        task_paths = pq.read_table(
            dataset_path, columns=["task_path"]
        ).column("task_path").to_pylist()
        if (
            len(task_paths) != selected
            or len(set(task_paths)) != selected
            or any(
                not isinstance(task_path, str)
                or not Path(task_path).is_absolute()
                or Path(task_path).parent != out
                for task_path in task_paths
            )
        ):
            return False
        available_task_ids = {Path(task_path).name for task_path in task_paths}
        requested_overrides = set(
            _normalize_task_ids(shadow_resource_override_task_ids)
        )
        if not requested_overrides <= available_task_ids:
            return False
    except Exception:
        return False
    return True


def _load_rows(hf_repo_id: str, hf_split: str, hf_revision: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(hf_repo_id, split=hf_split, revision=hf_revision)
    return [dict(row) for row in dataset]


def _registry_row(
    row: dict[str, Any],
    validated: dict[str, Any],
    milestone: dict[str, Any],
    task_dir: Path,
    source_dataset: str,
    source_revision: str,
) -> dict[str, Any]:
    result = {
        "id": validated["instance_id"],
        "instruction": validated["problem_statement"].rstrip() + "\n",
        "task_path": str(task_dir.resolve(strict=False)),
        "repo_name": validated["repo"],
        "commit_hash": validated["base_commit"],
        "docker_image": validated["image_name"],
        "language": validated["language"],
        "instruction_source": validated["instruction_source"],
        "source_dataset": source_dataset,
        "source_revision": source_revision,
    }
    result.update({field: milestone[field] for field in _MILESTONE_REGISTRY_FIELDS})
    return result


def build_benchmark(
    *,
    out_dir: str | Path,
    name: str = DEFAULT_DATASET_NAME,
    split: str = "train",
    hf_repo_id: str = DEFAULT_HF_REPO_ID,
    hf_revision: str = DEFAULT_HF_REVISION,
    hf_split: str = DEFAULT_HF_SPLIT,
    limit: int | None = None,
    language: str | None = DEFAULT_LANGUAGE,
    languages: Collection[str] | None = None,
    default_agent: str = "codeflow",
    clean: bool = False,
    register: bool = True,
    shadow_resource_override_task_ids: Collection[str] | None = None,
    max_workers: int | None = None,
) -> Path:
    """Materialize and optionally register a language slice or all tasks."""

    normalized_languages = _normalize_languages(
        language=language,
        languages=languages,
    )
    language = (
        normalized_languages[0]
        if normalized_languages is not None and len(normalized_languages) == 1
        else None
    )
    language_set = (
        None if normalized_languages is None else frozenset(normalized_languages)
    )
    shadow_resource_override_ids = frozenset(
        _normalize_task_ids(shadow_resource_override_task_ids)
    )

    out = Path(out_dir).expanduser().resolve(strict=False)
    config = _manifest_config(
        name=name,
        split=split,
        hf_repo_id=hf_repo_id,
        hf_revision=hf_revision,
        hf_split=hf_split,
        limit=limit,
        language=language,
        languages=(normalized_languages if len(normalized_languages or ()) > 1 else None),
        shadow_resource_override_task_ids=shadow_resource_override_ids,
    )
    with _materialization_lock(out):
        if clean and out.exists():
            logger.info("[swe-rebench-v2] removing existing %s", out)
            shutil.rmtree(out)
        if not clean and materialization_complete(
            name=name,
            split=split,
            out_dir=out,
            hf_repo_id=hf_repo_id,
            hf_revision=hf_revision,
            hf_split=hf_split,
            limit=limit,
            language=language,
            languages=(normalized_languages if len(normalized_languages or ()) > 1 else None),
            shadow_resource_override_task_ids=shadow_resource_override_ids,
        ):
            logger.info("[swe-rebench-v2] already complete; skipping %s", out)
            return out

        out.mkdir(parents=True, exist_ok=True)
        existing_manifest = _load_manifest(out)
        if existing_manifest is not None:
            _assert_manifest_compatible(existing_manifest, config, out)
        for stale in out.glob(".tmp-*"):
            if stale.is_dir():
                shutil.rmtree(stale)
            else:
                stale.unlink(missing_ok=True)

        logger.info(
            "[swe-rebench-v2] loading %s split=%s revision=%s",
            hf_repo_id,
            hf_split,
            hf_revision,
        )
        source_rows = _load_rows(hf_repo_id, hf_split, hf_revision)
        if hf_repo_id == DEFAULT_HF_REPO_ID and hf_revision == DEFAULT_HF_REVISION:
            if len(source_rows) != SOURCE_ROWS_AT_DEFAULT_REVISION:
                raise ValueError(
                    f"pinned SWE-rebench source expected {SOURCE_ROWS_AT_DEFAULT_REVISION} rows, got {len(source_rows)}"
                )
        rows = (
            source_rows
            if language_set is None
            else [
                row
                for row in source_rows
                if str(row.get("language") or "").casefold() in language_set
            ]
        )
        if hf_repo_id == DEFAULT_HF_REPO_ID and hf_revision == DEFAULT_HF_REVISION:
            if language == DEFAULT_LANGUAGE:
                if len(rows) != PYTHON_ROWS_AT_DEFAULT_REVISION:
                    raise ValueError(
                        f"pinned SWE-rebench Python slice expected {PYTHON_ROWS_AT_DEFAULT_REVISION} rows, got {len(rows)}"
                    )
                repo_count = len({str(row.get("repo") or "") for row in rows})
                if repo_count != PYTHON_REPOS_AT_DEFAULT_REVISION:
                    raise ValueError(
                        f"pinned SWE-rebench Python slice expected {PYTHON_REPOS_AT_DEFAULT_REVISION} repos, got {repo_count}"
                    )
            elif normalized_languages == tuple(sorted(PROMATCHED_LANGUAGES)):
                if len(rows) != PROMATCHED_ROWS_AT_DEFAULT_REVISION:
                    raise ValueError(
                        "pinned SWE-rebench Pro-matched slice expected "
                        f"{PROMATCHED_ROWS_AT_DEFAULT_REVISION} rows, got {len(rows)}"
                    )
                repo_count = len({str(row.get("repo") or "") for row in rows})
                if repo_count != PROMATCHED_REPOS_AT_DEFAULT_REVISION:
                    raise ValueError(
                        "pinned SWE-rebench Pro-matched slice expected "
                        f"{PROMATCHED_REPOS_AT_DEFAULT_REVISION} repos, got {repo_count}"
                    )
            elif normalized_languages is None:
                repo_count = len({str(row.get("repo") or "") for row in rows})
                language_count = len({str(row.get("language") or "").casefold() for row in rows})
                if repo_count != SOURCE_REPOS_AT_DEFAULT_REVISION:
                    raise ValueError(
                        f"pinned SWE-rebench source expected {SOURCE_REPOS_AT_DEFAULT_REVISION} repos, got {repo_count}"
                    )
                if language_count != SOURCE_LANGUAGES_AT_DEFAULT_REVISION:
                    raise ValueError(
                        f"pinned SWE-rebench source expected {SOURCE_LANGUAGES_AT_DEFAULT_REVISION} languages, got {language_count}"
                    )
        if limit is not None:
            rows = rows[:limit]
        if not rows:
            scope = ",".join(normalized_languages or ()) or "all languages"
            raise ValueError(f"SWE-rebench selection is empty for {scope}")

        task_ids = [str(row.get("instance_id") or "") for row in rows]
        if any(not task_id for task_id in task_ids) or len(task_ids) != len(set(task_ids)):
            raise ValueError("SWE-rebench selection contains empty or duplicate instance_id values")
        unknown_resource_overrides = sorted(
            shadow_resource_override_ids - set(task_ids)
        )
        if unknown_resource_overrides:
            raise ValueError(
                "shadow resource override contains unknown task ids: "
                f"{unknown_resource_overrides[:10]}"
            )
        task_ids_fingerprint = _json_fingerprint(task_ids)
        resume_completed = 0
        if (
            existing_manifest is not None
            and existing_manifest.get("schema_version")
            == MATERIALIZATION_SCHEMA_VERSION
            and existing_manifest.get("config") == config
            and existing_manifest.get("status") == "in_progress"
            and existing_manifest.get("selected_tasks") == len(rows)
            and existing_manifest.get("selected_task_ids_sha256") == task_ids_fingerprint
        ):
            checkpoint = existing_manifest.get("completed_tasks")
            if isinstance(checkpoint, int) and 0 < checkpoint < len(rows):
                resume_completed = checkpoint

        _write_dataset_toml(
            out,
            name=name,
            split=split,
            default_agent=default_agent,
            language=language,
            languages=(normalized_languages if len(normalized_languages or ()) > 1 else None),
        )
        manifest = {
            "schema_version": MATERIALIZATION_SCHEMA_VERSION,
            "status": "in_progress",
            "config": config,
            "source_rows": len(source_rows),
            "selected_tasks": len(rows),
            "selected_repositories": len({str(row.get("repo") or "") for row in rows}),
            "selected_languages": sorted({str(row.get("language") or "").casefold() for row in rows}),
            "selected_task_ids_sha256": task_ids_fingerprint,
            "completed_tasks": resume_completed,
        }
        _atomic_write_json(out / "materialization.json", manifest)

        prepared_rows = [
            (
                row,
                _validate_row(
                    row,
                    language=(language if len(normalized_languages or ()) == 1 else None),
                ),
            )
            for row in rows
        ]
        for _row_value, validated in prepared_rows:
            validated["shadow_sandbox_resources"] = (
                dict(_SHADOW_RESOURCE_OVERRIDE)
                if validated["instance_id"] in shadow_resource_override_ids
                else None
            )

        def materialize_one(
            prepared: tuple[dict[str, Any], dict[str, Any]],
        ) -> tuple[dict[str, Any], bool, bool]:
            row, validated = prepared
            task_id = validated["instance_id"]
            task_dir = out / task_id
            milestone = _milestone_metadata(validated)
            row_reused = _task_complete(
                task_dir,
                row,
                hf_revision,
                validated,
            )
            row_upgraded = False
            if not row_reused:
                if _task_matches_source_contract(
                    task_dir,
                    row,
                    hf_revision,
                    validated,
                ):
                    milestone = _upgrade_task_in_place(
                        task_dir,
                        row,
                        validated,
                        hf_revision,
                        name,
                    )
                    row_upgraded = True
                else:
                    if task_dir.exists():
                        shutil.rmtree(task_dir)
                    staging = out / f".tmp-{task_id}-{uuid.uuid4().hex}"
                    try:
                        milestone = _materialize_task(
                            staging,
                            row,
                            validated,
                            hf_revision,
                            name,
                        )
                        if not _task_complete(
                            staging,
                            row,
                            hf_revision,
                            validated,
                        ):
                            raise RuntimeError(
                                f"{task_id}: staged task failed completeness validation"
                            )
                        os.replace(staging, task_dir)
                    except BaseException:
                        shutil.rmtree(staging, ignore_errors=True)
                        raise
                if not _task_complete(
                    task_dir,
                    row,
                    hf_revision,
                    validated,
                ):
                    raise RuntimeError(
                        f"{task_id}: materialized task failed completeness validation"
                    )
            return (
                _registry_row(
                    row,
                    validated,
                    milestone,
                    task_dir,
                    hf_repo_id,
                    hf_revision,
                ),
                row_reused,
                row_upgraded,
            )

        workers = _materialization_workers(len(prepared_rows), max_workers)
        logger.info(
            "[swe-rebench-v2] materializing with %d worker(s)",
            workers,
        )
        registration_rows_by_index: dict[int, dict[str, Any]] = {}
        completed = 0
        reused = 0
        upgraded = 0
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="swerebench-materialize",
        ) as pool:
            futures = {
                pool.submit(materialize_one, prepared): row_index
                for row_index, prepared in enumerate(prepared_rows)
            }
            for future in as_completed(futures):
                row_index = futures[future]
                registry_row, row_reused, row_upgraded = future.result()
                registration_rows_by_index[row_index] = registry_row
                reused += int(row_reused)
                upgraded += int(row_upgraded)
                completed += 1
                if (
                    completed % 50 == 0 or completed == len(rows)
                ) and completed >= resume_completed:
                    manifest["completed_tasks"] = completed
                    _atomic_write_json(out / "materialization.json", manifest)
                    logger.info(
                        "[swe-rebench-v2] progress %d/%d (reused=%d upgraded=%d)",
                        completed,
                        len(rows),
                        reused,
                        upgraded,
                    )

        registration_rows = [
            registration_rows_by_index[index]
            for index in range(len(prepared_rows))
        ]

        if register:
            from rllm.data import DatasetRegistry

            if normalized_languages == (DEFAULT_LANGUAGE,):
                scope = "Python-only verified slice"
            elif normalized_languages is None:
                scope = "Complete multilingual verified dataset"
            else:
                scope = f"{', '.join(normalized_languages)} verified language slice"
            description = (
                f"{scope} of PrimeIntellect/SWE-rebench-V2-Filtered-Verified, "
                f"pinned at {hf_revision}."
            )
            DatasetRegistry.register_dataset(
                name=name,
                data=registration_rows,
                split=split,
                source=f"{hf_repo_id}@{hf_revision}",
                description=description,
                category="code",
            )

        manifest["status"] = "complete"
        manifest["completed_tasks"] = completed
        manifest["reused_tasks"] = reused
        manifest["upgraded_tasks"] = upgraded
        manifest["registered"] = bool(register)
        _atomic_write_json(out / "materialization.json", manifest)
        logger.info(
            "[swe-rebench-v2] complete: tasks=%d repos=%d reused=%d upgraded=%d out=%s",
            completed,
            manifest["selected_repositories"],
            reused,
            upgraded,
            out,
        )
        return out


__all__ = [
    "DEFAULT_DATASET_NAME",
    "DEFAULT_FULL_DATASET_NAME",
    "DEFAULT_PROMATCHED_DATASET_NAME",
    "DEFAULT_HF_REPO_ID",
    "DEFAULT_HF_REVISION",
    "MATERIALIZATION_SCHEMA_VERSION",
    "OFFICIAL_RESULT_PARSER",
    "PARSER_VENDOR_REVISION",
    "PARSER_VENDOR_SHA256",
    "PYTHON_RESULT_PARSER",
    "PROMATCHED_LANGUAGES",
    "RESULT_PARSER",
    "SUPPORTED_LOG_PARSERS",
    "TASK_SCHEMA_VERSION",
    "build_benchmark",
    "materialization_complete",
    "patch_paths",
]
