"""Builder for the R2E-Gym sandbox benchmark.

R2E-Gym (``R2E-Gym/R2E-Gym-Subset``, ``R2E-Gym/R2E-Gym-Lite``, etc.)
ships SWE-style tasks where each row carries a pre-built per-instance
Docker image (``namanjain12/<repo>_final:<commit>``) that already
contains the broken repo at ``/testbed``, the test fixtures at
``/r2e_tests/``, and the repo's own ``/testbed/run_tests.sh`` grader.
The agent fixes the bug; the verifier runs that ``run_tests.sh`` (after
exposing the fixtures inside the repo as ``/testbed/r2e_tests``) and
compares the parsed pytest output against ``expected_output_json``.

The builder also accepts R2E-Gym's SWE-Bench datasets, whose rows use the
SWE-Bench style ``instance_id`` / ``FAIL_TO_PASS`` / ``PASS_TO_PASS`` /
``patch`` / ``run_tests`` schema.

Row schema (confirmed via HF API):
    repo_name, docker_image, commit_hash, parsed_commit_content,
    execution_result_content, expected_output_json, modified_files,
    relevant_files, prompt, problem_statement, num_non_test_*

On-disk output (``<out_dir>/``)::

    r2egym/
    ├── dataset.toml                       # type="sandbox"
    ├── <task_id>/                         # task_id = <repo>__<short_commit>
    │   ├── task.toml                      # docker_image=<row.docker_image>, workdir=/testbed
    │   ├── instruction.md                 # problem_statement (preferred) or prompt
    │   ├── environment/Dockerfile         # FROM <row.docker_image> + ENTRYPOINT []
    │   ├── tests/
    │   │   ├── test.sh                    # synthesized verifier
    │   │   └── instance.json              # expected_output_json + repo_name
    │   └── solution/solve.sh              # apply the gold patch
    └── ...

Invoked from ``rllm dataset pull r2egym`` via the ``builder`` field in
``rllm/registry/datasets.json`` → :func:`rllm.cli._pull.pull_dataset`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import uuid
from pathlib import Path

from rllm.data.materialization import (
    bounded_ordered_thread_map,
    resolve_materialization_workers,
)

logger = logging.getLogger(__name__)

DEFAULT_HF_REPO_ID = "R2E-Gym/R2E-Gym-Subset"

# Per-instance resource defaults. R2E-Gym images bundle a uv venv + a
# full Python repo (orange3, numpy, pandas-style); tests can OOM with
# the 1 GiB remote-backend default. Docker ignores these.
_DEFAULT_RESOURCES = {
    "cpus": 4,
    "memory_mb": 16384,
    "storage_mb": 30720,
    "build_timeout_sec": 1800.0,
}

_DEFAULT_TIMEOUTS = {
    "agent_timeout_sec": 1800.0,
    "verifier_timeout_sec": 1800.0,
}


def _is_test_path(path: str) -> bool:
    """Match R2E-Gym's FileDiff.is_test_file heuristic (commit_models/diff_classes.py).

    Treats anything under a ``tests/`` / ``Tests/`` / ``test/`` directory or
    matching the ``test_*.py`` / ``*_test.py`` filename convention as a test
    file. Used to strip test changes from the oracle patch — the test
    fixtures already live under ``/r2e_tests/`` in the docker image, so
    applying test-file diffs from the commit would double-stage them.
    """
    if path.endswith("_test.py"):
        return True
    parts = path.split("/")
    last = parts[-1] if parts else ""
    if last.startswith("test_"):
        return True
    return any(p in {"tests", "Tests", "test", "Test"} for p in parts)


def _emit_hunk(hunk: dict) -> list[str]:
    """Re-emit a single hunk as unified-diff text lines (no trailing newline)."""
    desc = hunk.get("descriptor") or {}
    old_r = desc.get("old_range") or {}
    new_r = desc.get("new_range") or {}
    section = desc.get("section") or ""
    section_suffix = f" {section}" if section else ""
    lines = [f"@@ -{old_r.get('start', 0)},{old_r.get('length', 0)} +{new_r.get('start', 0)},{new_r.get('length', 0)} @@{section_suffix}"]
    for line in (hunk.get("line_group") or {}).get("all_lines", []):
        t = line.get("type", "")
        c = line.get("content", "")
        if t == "context":
            lines.append(f" {c}")
        elif t == "added":
            lines.append(f"+{c}")
        elif t == "deleted":
            lines.append(f"-{c}")
        elif t == "note":
            lines.append(f"\\ {c}")
    return lines


def patch_from_parsed_commit(
    parsed_commit_json: str,
    *,
    include_test_files: bool = False,
    python_only: bool = True,
) -> str:
    """Reconstruct a unified ``git diff`` from R2E-Gym's serialized ParsedCommit.

    Walks ``file_diffs`` and re-emits the header / index_line / minus/plus /
    hunks the same way ``r2egym.commit_models.diff_classes.FileDiff.get_patch``
    does, so we don't take a runtime dependency on the heavyweight
    ``r2e-gym`` package (pulls in litellm, anthropic[vertex], torch,
    matplotlib, ...). Defaults match the oracle's needs: skip test files
    (already baked into the image at ``/r2e_tests/``) and skip non-Python
    files (.rst / .yml / etc. that the runtime never executes).
    """
    try:
        pc = json.loads(parsed_commit_json)
    except (TypeError, json.JSONDecodeError):
        return ""

    out: list[str] = []
    for fd in pc.get("file_diffs", []) or []:
        path = ((fd.get("header") or {}).get("file") or {}).get("path") or ""
        if not path:
            continue
        if not include_test_files and _is_test_path(path):
            continue
        if python_only and not path.endswith(".py"):
            continue

        # diff --git header (mirroring FileDiffHeader.get_patch)
        out.append(f"diff --git a/{path} b/{path}")
        misc = (fd.get("header") or {}).get("misc_line")
        if misc:
            out.append(str(misc))

        idx = fd.get("index_line") or {}
        if idx:
            old_hash = idx.get("old_commit_hash") or ""
            new_hash = idx.get("new_commit_hash") or ""
            mode = idx.get("mode") or ""
            if old_hash and new_hash:
                tail = f" {mode}" if mode else ""
                out.append(f"index {old_hash}..{new_hash}{tail}")

        if fd.get("is_binary_file"):
            bl = fd.get("binary_line")
            if bl:
                out.append(str(bl))

        m = fd.get("minus_file") or {}
        p = fd.get("plus_file") or {}
        if m.get("path") and p.get("path"):
            out.append(f"--- {m['path']}")
            out.append(f"+++ {p['path']}")

        for hunk in fd.get("hunks") or []:
            out.extend(_emit_hunk(hunk))

    return "\n".join(out) + ("\n" if out else "")


_SHORT_HASH_LEN = 12
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
_TEST_STATUSES = frozenset({"PASSED", "FAILED", "ERROR"})
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
)


def _nonempty_str(value: object) -> str:
    return str(value or "").strip()


def _task_id_from_docker_image(docker_image: str) -> str:
    """Best-effort extraction for tags like ``...astropy__astropy-12907``."""
    image = _nonempty_str(docker_image)
    if not image:
        return ""
    tag = image.rsplit(":", 1)[-1]
    match = re.search(r"([A-Za-z0-9_-]+__[A-Za-z0-9_-]+-\d+)", tag)
    return match.group(1) if match else ""


def _task_id_for(row: dict) -> str:
    """Stable task id across R2E-Gym and SWE-Bench-shaped rows."""
    for key in ("task_id", "instance_id"):
        task_id = _nonempty_str(row.get(key))
        if task_id:
            return task_id

    image_task_id = _task_id_from_docker_image(row.get("docker_image") or "")
    if image_task_id:
        return image_task_id

    repo = _nonempty_str(row.get("repo_name") or row.get("repo")).replace("/", "__") or "unknown"
    commit = _nonempty_str(row.get("commit_hash") or row.get("base_commit"))[:_SHORT_HASH_LEN]
    return f"{repo}__{commit}" if commit else repo


def _repo_for(row: dict) -> str:
    return _nonempty_str(row.get("repo_name") or row.get("repo"))


def _commit_for(row: dict) -> str:
    return _nonempty_str(row.get("commit_hash") or row.get("base_commit"))


def _is_swebench_style_row(row: dict) -> bool:
    return bool(row.get("instance_id") or row.get("FAIL_TO_PASS") or row.get("PASS_TO_PASS") or row.get("run_tests"))


def _strip_pytest_reason_suffix(value: str) -> str:
    """Strip pytest's `` - reason`` suffix without truncating parametrization.

    Parameter ids can themselves contain ``" - "`` inside square brackets.
    Only a separator outside the brackets is a pytest failure/error reason.
    """
    bracket_depth = 0
    for index, char in enumerate(value):
        if bracket_depth == 0 and value.startswith(" - ", index):
            return value[:index]
        if char == "[":
            bracket_depth += 1
        elif char == "]" and bracket_depth:
            bracket_depth -= 1
    return value


def _parse_r2egym_test_log(log: object) -> dict[str, str]:
    """Parse R2E-Gym's pytest ``-rA`` summary into canonical test states."""
    if not isinstance(log, str) or "short test summary info" not in log:
        return {}

    status_map: dict[str, str] = {}
    summary = log.split("short test summary info", 1)[1]
    for raw_line in summary.splitlines():
        line = _ANSI_ESCAPE_RE.sub("", raw_line).strip()
        match = re.match(r"^(PASSED|FAILED|ERROR)\s+(.+)$", line)
        if not match:
            continue
        status, test_spec = match.groups()
        if status in {"FAILED", "ERROR"}:
            test_spec = _strip_pytest_reason_suffix(test_spec)

        parts = test_spec.split("::")
        # R2E-Gym's historical parser represents a file-level collection
        # ERROR (no ``::test_name`` suffix) with the empty-string key.
        test_name = ".".join(parts[1:]) if len(parts) > 1 else ("" if status == "ERROR" else parts[0])
        test_name = test_name.strip()
        if test_name or status == "ERROR":
            status_map[test_name] = status
    return status_map


def _parse_json_object(value: object, *, field: str, task_id: str) -> dict:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{task_id}: {field} must be a non-empty JSON object")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{task_id}: {field} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{task_id}: {field} must decode to an object")
    return parsed


def _canonicalize_status_map(value: dict, *, field: str, task_id: str) -> dict[str, str]:
    canonical: dict[str, str] = {}
    for raw_name, raw_status in value.items():
        if not isinstance(raw_name, str):
            raise ValueError(f"{task_id}: {field} contains a non-string test name")
        if raw_status not in _TEST_STATUSES:
            raise ValueError(f"{task_id}: {field}[{raw_name!r}] has unsupported status {raw_status!r}")
        name = _ANSI_ESCAPE_RE.sub("", raw_name).strip()
        if not name and raw_status != "ERROR":
            raise ValueError(f"{task_id}: {field} uses an empty test name for non-ERROR status {raw_status!r}")
        if name in canonical and canonical[name] != raw_status:
            raise ValueError(f"{task_id}: {field} has conflicting states for canonical test {name!r}")
        canonical[name] = raw_status
    return canonical


def _require_string_list(value: object, *, field: str, task_id: str, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{task_id}: {field} must be a list of strings")
    if nonempty and not value:
        raise ValueError(f"{task_id}: {field} must not be empty")
    return list(value)


def _build_milestone_metadata(row: dict) -> dict:
    """Build host/sandbox metadata for test and navigation potentials.

    ``expected_output_json`` is the verifier's authoritative target. The raw
    new-commit log is deliberately not used as the target because the two
    sources drift for a small subset of R2E-Gym rows.
    """
    task_id = _task_id_for(row)
    execution = _parse_json_object(row.get("execution_result_content"), field="execution_result_content", task_id=task_id)
    baseline = _parse_r2egym_test_log(execution.get("old_commit_res_stdout"))
    if not baseline:
        raise ValueError(f"{task_id}: old_commit_res_stdout has no parseable pytest summary")

    target_raw = _parse_json_object(row.get("expected_output_json"), field="expected_output_json", task_id=task_id)
    target = _canonicalize_status_map(target_raw, field="expected_output_json", task_id=task_id)
    if not target:
        raise ValueError(f"{task_id}: expected_output_json must not be empty")

    fail_to_pass = [name for name, status in target.items() if status == "PASSED" and baseline.get(name) in {"FAILED", "ERROR"}]
    pass_to_pass = [name for name, status in target.items() if status == "PASSED" and baseline.get(name) == "PASSED"]

    modified_files = _require_string_list(row.get("modified_files"), field="modified_files", task_id=task_id, nonempty=True)
    relevant_files = _require_string_list(row.get("relevant_files"), field="relevant_files", task_id=task_id, nonempty=True)
    test_file_names = _require_string_list(execution.get("test_file_names"), field="test_file_names", task_id=task_id, nonempty=True)
    entity_summaries = row.get("modified_entity_summaries")
    if not isinstance(entity_summaries, list) or not entity_summaries or any(not isinstance(item, dict) for item in entity_summaries):
        raise ValueError(f"{task_id}: modified_entity_summaries must be a non-empty list of objects")

    return {
        "baseline_output_json": json.dumps(baseline, ensure_ascii=False, sort_keys=True),
        "target_output_json": json.dumps(target, ensure_ascii=False, sort_keys=True),
        "FAIL_TO_PASS": json.dumps(fail_to_pass, ensure_ascii=False),
        "PASS_TO_PASS": json.dumps(pass_to_pass, ensure_ascii=False),
        "modified_files": modified_files,
        "relevant_files": relevant_files,
        "modified_entity_summaries": [dict(item) for item in entity_summaries],
        "test_file_names": test_file_names,
    }


def _build_dockerfile(docker_image: str) -> str:
    """``FROM <docker_image>`` + clear ENTRYPOINT, same hazard as SWE-bench Pro."""
    return f"FROM {docker_image}\nENTRYPOINT []\nWORKDIR /testbed\n"


def _build_task_toml(
    *,
    task_id: str,
    repo: str,
    commit_hash: str,
    docker_image: str,
) -> str:
    """Synthesize a Harbor-format ``task.toml``."""
    lines = [
        'schema_version = "1.1"',
        "",
        "[task]",
        f'name = "r2egym/{task_id}"',
        f'description = "R2E-Gym: {repo} @ {commit_hash[:_SHORT_HASH_LEN]}"',
        f'keywords = ["r2e-gym", "{repo}"]',
        "",
        "[metadata]",
        f'task_id = "{task_id}"',
        f'repo_name = "{repo}"',
        f'commit_hash = "{commit_hash}"',
        "",
        "[environment]",
        f'docker_image = "{docker_image}"',
        'workdir = "/testbed"',
        f"cpus = {_DEFAULT_RESOURCES['cpus']}",
        f"memory_mb = {_DEFAULT_RESOURCES['memory_mb']}",
        f"storage_mb = {_DEFAULT_RESOURCES['storage_mb']}",
        f"build_timeout_sec = {_DEFAULT_RESOURCES['build_timeout_sec']}",
        "allow_internet = true",
        "",
        "[agent]",
        f"timeout_sec = {_DEFAULT_TIMEOUTS['agent_timeout_sec']}",
        "",
        "[verifier]",
        f"timeout_sec = {_DEFAULT_TIMEOUTS['verifier_timeout_sec']}",
        "",
    ]
    return "\n".join(lines)


# The verifier is dead simple compared to swebench_pro: the
# R2E-Gym image ships a self-contained ``/testbed/run_tests.sh`` that runs
# the full test suite via pytest (``-W ignore -m pytest -rA r2e_tests``)
# from inside the repo's bundled ``.venv``. We expose the fixtures inside
# the repo (``/testbed/r2e_tests``), invoke the script (against whatever
# state the agent left /testbed in), parse the per-test PASSED/FAILED/ERROR
# statuses from the ``-rA`` short-summary footer, and compare to the
# expected map. Reward = 1.0 iff EVERY expected test appears with the
# expected status — matching ``_calculate_reward_r2e`` in
# ``r2egym.agenthub.runtime.docker``.
_VERIFIER_TEMPLATE = r"""#!/bin/bash
set -uo pipefail

mkdir -p /tmp/rllm /logs/verifier
REWARD_JSON=/tmp/rllm/reward.json

log() { echo "[verifier] $*"; }

write_failure() {
    python3 - "$1" <<'PY' || echo '{"reward": 0.0, "is_correct": false}' > "$REWARD_JSON"
import json, sys
json.dump({"reward": 0.0, "is_correct": False, "metadata": {"error": sys.argv[1]}}, open("/tmp/rllm/reward.json", "w"))
PY
}

cd /testbed 2>/dev/null || { write_failure "/testbed missing"; exit 0; }

# R2E-Gym's grader runs the repo's own ``run_tests.sh`` from the repo root.
# In the raw image that script lives at ``/testbed/run_tests.sh`` and invokes
# ``.venv/bin/python -W ignore -m pytest -rA r2e_tests`` — warnings are
# suppressed on purpose: the repo's setup.cfg turns warnings into errors, so
# without ``-W ignore`` unrelated tests flip to FAILED/ERROR and break the
# expected-output equality check. The test fixtures ship at ``/r2e_tests`` but
# the script references them as ``r2e_tests`` relative to /testbed, so they
# must be reachable inside the repo. r2egym does ``mv /r2e_tests
# /root/r2e_tests`` + a ``/testbed/r2e_tests`` symlink; we just make
# ``/testbed/r2e_tests`` resolve (symlink is non-destructive and collects
# identically — pytest's rootdir is /testbed either way).
if [ ! -e /testbed/r2e_tests ]; then
    if [ -d /r2e_tests ]; then
        ln -s /r2e_tests /testbed/r2e_tests
    elif [ -d /root/r2e_tests ]; then
        ln -s /root/r2e_tests /testbed/r2e_tests
    fi
fi

# Prefer the image's own run_tests.sh (it carries the correct -W ignore / -rA
# flags + the repo venv); fall back to the canonical command if it's absent.
RUN_TESTS=""
for cand in /testbed/run_tests.sh /root/run_tests.sh /r2e_tests/run_tests.sh; do
    if [ -f "$cand" ]; then
        RUN_TESTS="$cand"
        break
    fi
done
if [ -n "$RUN_TESTS" ]; then
    log "Running $RUN_TESTS"
    bash "$RUN_TESTS" > /tmp/test_output.txt 2>&1 || log "run_tests.sh exited non-zero (parser inspects log)"
elif [ -d /testbed/r2e_tests ]; then
    log "run_tests.sh not found; invoking pytest directly"
    PY=.venv/bin/python
    [ -x "$PY" ] || PY=python
    PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' "$PY" -W ignore -m pytest -rA r2e_tests > /tmp/test_output.txt 2>&1 || log "pytest exited non-zero (parser inspects log)"
else
    write_failure "no r2e_tests fixtures or run_tests.sh found"
    exit 0
fi

python3 <<'PY'
import json, re, sys
REWARD = "/tmp/rllm/reward.json"

def _tail(path, n=2000):
    try:
        return open(path).read()[-n:]
    except Exception:
        return ""

def decolor(s):
    return re.sub(r"\x1b\[\d+m", "", s or "")

# Mirror r2egym.repo_analysis.execution_log_parser.parse_log_pytest:
# tests are reported under the "short test summary info" footer that
# pytest emits with -ra/-rA. Each line starts with PASSED/FAILED/ERROR
# followed by the test id (path::Class::test_name); we collapse to the
# dotted-name form that expected_output_json uses
# (".".join(parts after the first :: split)).
def parse_log_pytest(log):
    status_map = {}
    if "short test summary info" not in log:
        return status_map
    section = log.split("short test summary info", 1)[1].strip()
    for raw in section.split("\n"):
        line = decolor(raw)
        if "PASSED" in line:
            name = ".".join(line.split("::")[1:])
            status_map[name] = "PASSED"
        elif "FAILED" in line:
            name = ".".join(line.split("::")[1:]).split(" - ")[0]
            status_map[name] = "FAILED"
        elif "ERROR" in line:
            try:
                name = ".".join(line.split("::")[1:])
            except IndexError:
                name = line
            name = name.split(" - ")[0]
            status_map[name] = "ERROR"
    return status_map

try:
    inst = json.load(open("/tests/instance.json"))
except Exception as e:
    json.dump({"reward": 0.0, "is_correct": False, "metadata": {"error": f"instance.json missing: {e}"}}, open(REWARD, "w"))
    raise SystemExit(0)

try:
    expected = json.loads(inst.get("expected_output_json") or "{}")
except Exception as e:
    json.dump({"reward": 0.0, "is_correct": False, "metadata": {"error": f"expected_output_json malformed: {e}"}}, open(REWARD, "w"))
    raise SystemExit(0)

# Drop ANSI color and strip pytest "<name> - <reason>" tails — the
# expected map sometimes carries the reason, sometimes doesn't.
expected = {decolor(k).split(" - ")[0]: v for k, v in expected.items()}

try:
    log = open("/tmp/test_output.txt").read()
except Exception as e:
    json.dump({"reward": 0.0, "is_correct": False, "metadata": {"error": f"reading test_output.txt: {e}"}}, open(REWARD, "w"))
    raise SystemExit(0)

parsed = {decolor(k).split(" - ")[0]: v for k, v in parse_log_pytest(log).items()}

# Match r2egym's _calculate_reward_r2e: cardinalities must match AND
# every key must map to the same status. Either mismatch ⇒ reward 0.
if len(parsed) != len(expected):
    reward = 0.0
    mismatch = "size"
else:
    reward = 1.0
    mismatch = None
    for k, v in expected.items():
        if k not in parsed or parsed[k] != v:
            reward = 0.0
            mismatch = f"first_mismatch={k}"
            break

passed = sum(1 for v in parsed.values() if v == "PASSED")
expected_passed = sum(1 for v in expected.values() if v == "PASSED")
json.dump({
    "reward": reward,
    "is_correct": reward >= 1.0,
    "signals": {
        "expected_tests": len(expected),
        "parsed_tests": len(parsed),
        "passed_in_parsed": passed,
        "passed_in_expected": expected_passed,
    },
    "metadata": {
        "mismatch": mismatch,
        "log_tail": _tail("/tmp/test_output.txt", 1500),
    },
}, open(REWARD, "w"))
PY
"""


def _build_verifier_script() -> str:
    return _VERIFIER_TEMPLATE


_SWEBENCH_VERIFIED_VERIFIER_TEMPLATE = r"""#!/bin/bash
# rllm-swebench-verifier-version: 4
set -uo pipefail

mkdir -p /tmp/rllm /logs/verifier
REWARD_JSON=/tmp/rllm/reward.json

log() { echo "[swebench-verifier] $*"; }

write_failure() {
    python3 - "$1" <<'PY' || echo '{"reward": 0.0, "is_correct": false}' > "$REWARD_JSON"
import json, sys
json.dump({"reward": 0.0, "is_correct": False, "metadata": {"error": sys.argv[1]}}, open("/tmp/rllm/reward.json", "w"))
PY
}

cd /testbed 2>/dev/null || { write_failure "/testbed missing"; exit 0; }

if [ ! -f /tests/run_tests.sh ]; then
    write_failure "/tests/run_tests.sh missing"
    exit 0
fi

log "Running SWE-Bench test script"
bash /tests/run_tests.sh > /tmp/test_output.txt 2>&1 || log "run_tests.sh exited non-zero (parser inspects log)"

python3 <<'PY'
import json, os, re

REWARD = os.environ.get("RLLM_REWARD_JSON", "/tmp/rllm/reward.json")
INSTANCE = os.environ.get("RLLM_INSTANCE_JSON", "/tests/instance.json")
TEST_OUTPUT = os.environ.get("RLLM_TEST_OUTPUT", "/tmp/test_output.txt")

def _tail(path, n=2000):
    try:
        return open(path, encoding="utf-8", errors="replace").read()[-n:]
    except Exception:
        return ""

def decolor(s):
    return re.sub(r"\x1b\[[0-9;]*m", "", s or "")

def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
        except Exception:
            return [value]
        return as_list(parsed)
    return [str(value)]

def match_target(candidate, targets):
    candidate = candidate.strip()
    if candidate in targets:
        return candidate
    # Pytest appends `` - reason`` to failures.  Match the complete expected
    # test id first so parameter ids containing the same separator survive.
    for target in sorted(targets, key=len, reverse=True):
        if candidate.startswith(target + " - "):
            return target
    return candidate.split(" - ", 1)[0].strip()

def parse_test_results(log, targets, parameter_aliases=None):
    status_map = {}
    target_set = set(targets)
    previous_unittest = None
    for raw in log.splitlines():
        line = decolor(raw).strip()

        # Pytest short-summary format: ``PASSED path::test``.  Parsing the
        # whole log is intentional: some runners omit the summary heading.
        m = re.match(r"^(PASSED|FAILED|ERROR|SKIPPED|XFAILED|XFAIL|XPASS)\s+(.+)$", line)
        if m:
            status, name = m.groups()
            name = match_target(name, target_set)
            if name:
                status_map[name] = status
            continue

        # Older pytest releases (used by several Verified images, notably
        # Astropy) report progress as ``path::test PASSED`` rather than
        # emitting a short summary for successful tests.
        m = re.match(
            r"^(.+?)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAILED|XFAIL|XPASS)(?:\s+\[[^]]+\])?\s*$",
            line,
        )
        if m:
            name, status = m.groups()
            name = match_target(name, target_set)
            if name:
                status_map[name] = status
            continue

        # SymPy's custom test runner uses ``test_name ok|F|E`` and may print
        # a failure separator instead of a pytest summary.
        m = re.match(r"^(test_\S+)\s+(ok|F|E)\s*$", line)
        if m:
            name, outcome = m.groups()
            status = {"ok": "PASSED", "F": "FAILED", "E": "ERROR"}[outcome]
            name = match_target(name, target_set)
            if name:
                status_map[name] = status
            continue
        m = re.match(r"^_+\s+(test_\S+)\s+_+$", line)
        if m:
            name = match_target(m.group(1), target_set)
            if name:
                status_map[name] = "FAILED"
            continue

        # unittest/Django verbosity=2 format:
        # ``test_name (module.Class) ... ok|FAIL|ERROR|skipped ...``.
        # SWE-Bench's Django FAIL_TO_PASS/PASS_TO_PASS ids use the complete
        # left-hand side, so retain it verbatim.
        m = re.match(
            r"^(.+?)\s+\.\.\.\s+(ok|OK|FAIL|ERROR|skipped\b.*|expected failure|unexpected success)\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if m:
            name, outcome = m.groups()
            normalized = outcome.lower()
            if normalized == "ok":
                status = "PASSED"
            elif normalized == "fail":
                status = "FAILED"
            elif normalized == "error":
                status = "ERROR"
            elif normalized.startswith("skipped"):
                status = "SKIPPED"
            elif normalized == "expected failure":
                status = "XFAILED"
            else:
                status = "XPASS"
            name = match_target(name, target_set)
            if name:
                status_map[name] = status
            previous_unittest = name
            continue

        # Django occasionally prints diagnostic output between ``...`` and
        # the final ``ok`` line.  Remember the test id and accept a standalone
        # success marker, matching the official SWE-Bench Django parser.
        if " ... " in line:
            previous_unittest = match_target(line.split(" ... ", 1)[0], target_set)
        if line in {"ok", "OK"} and previous_unittest:
            status_map[previous_unittest] = "PASSED"
    # Historical benchmark ids were sometimes truncated at whitespace inside
    # a pytest parameter (e.g. test_tofile[home_is_data, pathlib.Path]).
    # Keep full ids above; only alias a missing, unclosed parameter prefix.
    # Several variants may share that prefix: every observed variant must pass.
    full_status_map = dict(status_map)
    for target in target_set:
        if target in status_map or "[" not in target or target.endswith("]"):
            continue
        matches = {name: status for name, status in full_status_map.items()
                   if name.startswith(target + " ") and name.endswith("]")}
        if matches:
            status_map[target] = next((status for status in matches.values() if status != "PASSED"), "PASSED")
            if parameter_aliases is not None:
                parameter_aliases[target] = dict(matches)
            # Count canonical ids, not both a full id and its legacy alias.
            for name in matches:
                if name not in target_set:
                    status_map.pop(name, None)
    return status_map

try:
    inst = json.load(open(INSTANCE, encoding="utf-8"))
except Exception as e:
    # Some older SWE-Bench images expose Python 3.5 as ``python3``.  Keep the
    # control-plane parser compatible with it even when the repository's own
    # test environment is older than the rLLM training image.
    json.dump({"reward": 0.0, "is_correct": False, "metadata": {"error": "instance.json missing: {}".format(e)}}, open(REWARD, "w"))
    raise SystemExit(0)

fail_to_pass = as_list(inst.get("FAIL_TO_PASS"))
pass_to_pass = as_list(inst.get("PASS_TO_PASS"))

try:
    log = open(TEST_OUTPUT, encoding="utf-8", errors="replace").read()
except Exception as e:
    json.dump({"reward": 0.0, "is_correct": False, "metadata": {"error": "reading test_output.txt: {}".format(e)}}, open(REWARD, "w"))
    raise SystemExit(0)

target_tests = fail_to_pass + pass_to_pass
parameter_aliases = {}
parsed = parse_test_results(log, target_tests, parameter_aliases)
missing = [t for t in target_tests if t not in parsed]
failing_f2p = {t: parsed.get(t) for t in fail_to_pass if parsed.get(t) != "PASSED"}
regressed_p2p = {t: parsed.get(t) for t in pass_to_pass if parsed.get(t) != "PASSED"}

reward = 1.0
if not target_tests or missing or failing_f2p or regressed_p2p:
    reward = 0.0

json.dump({
    "reward": reward,
    "is_correct": reward >= 1.0,
    "signals": {
        "fail_to_pass": len(fail_to_pass),
        "pass_to_pass": len(pass_to_pass),
        "parsed_tests": len(parsed),
        "missing_tests": len(missing),
        "failing_fail_to_pass": len(failing_f2p),
        "regressed_pass_to_pass": len(regressed_p2p),
        "parser_version": 4,
    },
    "metadata": {
        "missing": missing[:20],
        "parameter_id_aliases": parameter_aliases,
        "failing_fail_to_pass": failing_f2p,
        "regressed_pass_to_pass": regressed_p2p,
        "log_tail": _tail("/tmp/test_output.txt", 1500),
    },
}, open(REWARD, "w"))
PY
"""


def _build_swebench_verified_verifier_script() -> str:
    return _SWEBENCH_VERIFIED_VERIFIER_TEMPLATE


def swebench_verified_verifier_fingerprint() -> str:
    """Fingerprint the verifier semantics used by resumable evaluation.

    This identifies scoring semantics, not formatting or interpreter-only
    changes. Version 4 also records expected failures, handles nested
    parameter prefixes, and requests per-test output through tox. Old scores
    must not be resumed under it.
    """

    return "03d4ae6702179d7d0c6901efa434d55ce21dbdbd466fd4f868fe85adbcfb3570"


def _swebench_verified_runner_output(script: str) -> str:
    """Request pytest evidence through the existing tox posargs delimiter.

    Tox's own -v does not make pytest verbose. Inject only reporting options
    into known benchmark commands; keep test selectors and patch heredocs
    byte-for-byte. Do not export PYTEST_ADDOPTS into tests' subprocesses.
    """

    pattern = r"(?m)^(tox[ \t]+--current-env[^\r\n]*?[ \t]--)([ \t]+)([^\r\n]*)$"

    def verbose(match: re.Match[str]) -> str:
        arguments = match.group(3)
        if arguments.startswith("-rA -vv ") or arguments == "-rA -vv":
            return match.group(0)
        return match.group(1) + match.group(2) + "-rA -vv " + arguments

    return re.sub(pattern, verbose, script)


def ensure_swebench_verified_verifier(task_dir: Path) -> bool:
    """Upgrade scoring and runner reporting before any model is evaluated.

    Reuse materialized data and images. Runner changes only add reporting
    options to tox's existing pytest arguments; task metadata and test
    selection stay unchanged. Return whether either script was replaced.
    """

    verifier = task_dir / "tests" / "test.sh"
    runner = verifier.with_name("run_tests.sh")
    updates = []
    if runner.is_file():
        original = runner.read_text(encoding="utf-8")
        updates.append((runner, _swebench_verified_runner_output(original)))
    updates.append((verifier, _build_swebench_verified_verifier_script()))
    changed = False
    for path, expected in updates:
        try:
            if path.read_text(encoding="utf-8") == expected:
                continue
        except FileNotFoundError:
            pass
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(".{}.rllm-{}.tmp".format(path.name, uuid.uuid4().hex))
        try:
            temporary.write_text(expected, encoding="utf-8")
            temporary.chmod(0o755)
            os.replace(temporary, path)
            changed = True
        finally:
            temporary.unlink(missing_ok=True)
    return changed


def _build_solution_script(has_patch: bool) -> str:
    """Oracle harness: apply the reconstructed gold patch.

    We don't ``git reset --hard`` first — the image is
    already at the buggy state and the gold patch was reconstructed
    against that exact tree. Plain ``git apply`` of the patch reproduces
    the fix on top of /testbed. If no patch could be reconstructed
    (the ``parsed_commit_content`` JSON was empty or non-Python), fail
    loudly so the oracle eval surfaces the issue instead of silently
    "passing" against the unmodified buggy state.
    """
    if not has_patch:
        return "#!/bin/bash\necho 'oracle solve.sh: no gold patch available' >&2\nexit 1\n"
    return "#!/bin/bash\nset -e\ncd /testbed\ngit config --global --add safe.directory /testbed 2>/dev/null || true\ngit apply -v /solution/gold.patch\n"


def _gold_patch_for(row: dict) -> str:
    patch = _nonempty_str(row.get("patch"))
    if patch:
        return patch + ("\n" if not patch.endswith("\n") else "")
    return patch_from_parsed_commit(row.get("parsed_commit_content") or "")


def _extract_instruction(row: dict) -> str:
    """Pick the cleanest task instruction.

    ``problem_statement`` is the curated GitHub-style issue (the canonical
    agent prompt). Some rows embed it inside ``[ISSUE]...[/ISSUE]`` tags
    (matches ``DockerRuntime.get_task_instruction`` in r2e-gym); strip
    those wrappers if present.
    """
    ps = (row.get("problem_statement") or "").strip()
    if ps:
        m = re.search(r"\[ISSUE\](.*?)\[/ISSUE\]", ps, re.DOTALL)
        if m:
            return m.group(1).strip() + "\n"
        return ps + "\n"
    return (row.get("prompt") or "").strip() + "\n"


def _write_dataset_toml(out: Path, *, name: str, split: str, description: str, default_agent: str) -> None:
    content = "\n".join(
        [
            "[dataset]",
            f'name = "{name}"',
            'type = "sandbox"',
            f'description = "{description}"',
            'default_sandbox = "docker"',
            f'default_agent = "{default_agent}"',
            f'split = "{split}"',
            "",
            "[verifier]",
            'script = "tests/test.sh"',
            "",
        ]
    )
    (out / "dataset.toml").write_text(content, encoding="utf-8")


def _materialize_task(task_dir: Path, row: dict, *, materialize_milestone_metadata: bool = False) -> dict:
    """Expand a single HF row into a Harbor-format task tree. Returns stats."""
    task_dir.mkdir(parents=True, exist_ok=True)

    task_id = _task_id_for(row)
    repo = _repo_for(row)
    commit_hash = _commit_for(row)
    docker_image = row.get("docker_image") or ""
    is_swebench = _is_swebench_style_row(row)

    (task_dir / "task.toml").write_text(
        _build_task_toml(
            task_id=task_id,
            repo=repo,
            commit_hash=commit_hash,
            docker_image=docker_image,
        ),
        encoding="utf-8",
    )

    (task_dir / "instruction.md").write_text(_extract_instruction(row), encoding="utf-8")

    env_dir = task_dir / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "Dockerfile").write_text(_build_dockerfile(docker_image), encoding="utf-8")

    tests_dst = task_dir / "tests"
    tests_dst.mkdir(parents=True, exist_ok=True)
    if is_swebench:
        (tests_dst / "test.sh").write_text(_build_swebench_verified_verifier_script(), encoding="utf-8")
        (tests_dst / "run_tests.sh").write_text(_swebench_verified_runner_output(row.get("run_tests") or ""), encoding="utf-8")
        (tests_dst / "run_tests.sh").chmod(0o755)
    else:
        (tests_dst / "test.sh").write_text(_build_verifier_script(), encoding="utf-8")
    (tests_dst / "test.sh").chmod(0o755)
    instance_data = {
        "task_id": task_id,
        "repo_name": repo,
        "commit_hash": commit_hash,
        # expected_output_json is already a JSON string in the row; keep
        # it as a string so the verifier doesn't double-encode it.
        "expected_output_json": row.get("expected_output_json") or "{}",
        "repo": row.get("repo", ""),
        "base_commit": row.get("base_commit", ""),
        "FAIL_TO_PASS": row.get("FAIL_TO_PASS") or "[]",
        "PASS_TO_PASS": row.get("PASS_TO_PASS") or "[]",
    }
    if materialize_milestone_metadata:
        instance_data.update(_build_milestone_metadata(row))
    (tests_dst / "instance.json").write_text(json.dumps(instance_data, indent=2), encoding="utf-8")

    sol_dst = task_dir / "solution"
    sol_dst.mkdir(parents=True, exist_ok=True)
    gold_patch = _gold_patch_for(row)
    (sol_dst / "gold.patch").write_text(gold_patch, encoding="utf-8")
    (sol_dst / "solve.sh").write_text(_build_solution_script(bool(gold_patch.strip())), encoding="utf-8")
    (sol_dst / "solve.sh").chmod(0o755)

    return {"task_id": task_id, "has_patch": bool(gold_patch.strip())}


def _load_rows(hf_repo_id: str, hf_split: str, *, retries: int = 4, backoff_sec: float = 10.0) -> list[dict]:
    """Load HF rows with retries on transient Hub connection errors.

    ``rllm eval`` auto-pulls before running, so a transient
    ``LocalEntryNotFoundError`` from a DNS/SSL blip would crash the eval.
    Cached partial progress makes the retries cheap.
    """
    import time

    from datasets import load_dataset

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            ds = load_dataset(hf_repo_id, split=hf_split)
            return [dict(r) for r in ds]
        except Exception as e:
            last_exc = e
            if attempt == retries:
                break
            wait = backoff_sec * attempt
            logger.warning("[r2egym] load_dataset(%s) failed (attempt %d/%d): %s — retry in %.0fs", hf_repo_id, attempt, retries, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"load_dataset({hf_repo_id!r}, split={hf_split!r}) failed after {retries} attempts") from last_exc


def build_benchmark(
    *,
    name: str = "r2egym",
    split: str = "train",
    out_dir: str | Path,
    catalog_entry: dict | None = None,
    task_ids: list[str] | None = None,
    limit: int | None = None,
    default_agent: str = "codeflow",
    hf_repo_id: str | None = None,
    hf_split: str = "train",
    clean: bool = False,
    register: bool = True,
    materialize_milestone_metadata: bool = False,
    max_workers: int | None = None,
) -> Path:
    """Materialize R2E-Gym into a sandbox benchmark directory.

    Args:
        name: Dataset/registry name (also the dataset.toml ``name``).
        split: Split label written into dataset.toml and the registry.
        out_dir: Output benchmark directory.
        catalog_entry: Optional catalog entry (datasets.json); ``description``,
            ``default_agent``, ``source`` are read from it when present.
        task_ids: Build only these task ids (``<repo>__<short_commit>``).
        limit: Keep only the first N rows (after the ``task_ids`` filter).
        default_agent: ``default_agent`` written into dataset.toml.
        hf_repo_id: Override the HF dataset. Defaults to the catalog
            ``source`` or ``R2E-Gym/R2E-Gym-Subset`` (4,578 train rows).
            Use ``R2E-Gym/R2E-Gym-Lite`` for the 11K-row version with
            multiple dev splits.
        hf_split: HF split to load (``train`` for Subset; Lite ships
            ``dev_*`` splits too).
        clean: Remove ``out_dir`` before building.
        register: Also register ``task_path`` rows in ``DatasetRegistry``.
        materialize_milestone_metadata: Add R2E-Gym baseline/target test
            states and navigation annotations to instance and registry rows.
            Disabled by default to preserve the existing output schema.
        max_workers: Task-directory materialization threads. ``None`` resolves
            ``RLLM_MATERIALIZATION_WORKERS`` (default 16).

    Returns:
        Path to the built benchmark directory.
    """
    if catalog_entry:
        default_agent = catalog_entry.get("default_agent") or default_agent
        hf_repo_id = hf_repo_id or catalog_entry.get("source")
    hf_repo_id = hf_repo_id or DEFAULT_HF_REPO_ID

    out = Path(out_dir).expanduser()
    if clean and out.exists():
        logger.info("[r2egym] removing existing %s", out)
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("[r2egym] loading HF dataset %s split=%s ...", hf_repo_id, hf_split)
    rows = _load_rows(hf_repo_id, hf_split)

    if task_ids is not None:
        keep = set(task_ids)
        rows = [r for r in rows if _task_id_for(r) in keep]
    if limit is not None:
        rows = rows[:limit]
    logger.info("[r2egym] selected %d rows (task_ids=%s, limit=%s)", len(rows), task_ids and len(task_ids), limit)

    skipped = 0
    seen_task_ids: set[str] = set()
    prepared: list[tuple[dict, str]] = []
    for row in rows:
        task_id = _task_id_for(row)
        if task_id in seen_task_ids:
            logger.warning("[r2egym] duplicate task_id %s, skipping duplicate row", task_id)
            skipped += 1
            continue
        seen_task_ids.add(task_id)
        if not row.get("docker_image"):
            logger.warning("[r2egym] %s: missing docker_image, skipping", task_id)
            skipped += 1
            continue
        prepared.append((row, task_id))

    for stale in out.glob(".tmp-*"):
        if stale.is_dir():
            shutil.rmtree(stale, ignore_errors=True)
        else:
            stale.unlink(missing_ok=True)

    def materialize_one(item: tuple[dict, str]) -> dict:
        row, task_id = item
        task_dst = out / task_id
        staging = out / f".tmp-{task_id}-{uuid.uuid4().hex}"
        try:
            stats = _materialize_task(
                staging,
                row,
                materialize_milestone_metadata=materialize_milestone_metadata,
            )
            if task_dst.exists():
                shutil.rmtree(task_dst)
            os.replace(staging, task_dst)
            return stats
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    workers = resolve_materialization_workers(len(prepared), max_workers)
    logger.info("[r2egym] materializing with %d worker(s)", workers)
    no_patch = 0
    written = 0
    for stats in bounded_ordered_thread_map(
        materialize_one,
        prepared,
        workers=workers,
        thread_name_prefix="r2egym-materialize",
    ):
        written += 1
        no_patch += int(not stats["has_patch"])
        if written % 50 == 0 or written == len(prepared):
            logger.info(
                "[r2egym] progress %d/%d (no oracle patch=%d)",
                written,
                len(prepared),
                no_patch,
            )

    description = (catalog_entry or {}).get("description") or (
        f"R2E-Gym ({hf_repo_id}): real-world Python SWE tasks with per-instance Docker images and pytest-output equality grading against expected outputs."
    )
    _write_dataset_toml(out, name=name, split=split, description=description, default_agent=default_agent)
    logger.info("[r2egym] wrote %d task dirs to %s (skipped %d, no oracle patch %d)", written, out, skipped, no_patch)

    if register:
        try:
            from rllm.data import DatasetRegistry

            reg_rows = []
            reg_seen: set[str] = set()
            for row in rows:
                task_id = _task_id_for(row)
                if task_id in reg_seen:
                    continue
                reg_seen.add(task_id)
                task_dst = out / task_id
                if not (task_dst / "task.toml").exists():
                    continue
                registry_row = {
                    "id": task_id,
                    "instruction": (task_dst / "instruction.md").read_text(encoding="utf-8"),
                    "task_path": str(task_dst),
                    "repo_name": _repo_for(row),
                    "commit_hash": _commit_for(row),
                    "docker_image": row.get("docker_image", ""),
                }
                if materialize_milestone_metadata:
                    instance_data = json.loads((task_dst / "tests" / "instance.json").read_text(encoding="utf-8"))
                    registry_row.update({field: instance_data[field] for field in _MILESTONE_REGISTRY_FIELDS})
                reg_rows.append(registry_row)
            DatasetRegistry.register_dataset(
                name=name,
                data=reg_rows,
                split=split,
                source=hf_repo_id,
                description=description,
                category=(catalog_entry or {}).get("category", "code"),
            )
        except Exception:
            if materialize_milestone_metadata:
                raise
            logger.warning("[r2egym] could not register rows in DatasetRegistry (non-fatal)", exc_info=True)

    return out


def main() -> None:
    """CLI: ``python -m rllm.data.r2egym_builder --out-dir <dir>``."""
    import argparse

    parser = argparse.ArgumentParser(description="Materialize R2E-Gym into an rLLM sandbox benchmark directory.")
    parser.add_argument("--out-dir", required=True, help="Output benchmark directory.")
    parser.add_argument("--name", default="r2egym")
    parser.add_argument("--split", default="train")
    parser.add_argument("--hf-repo-id", default=None, help="Override HF source repo (default: R2E-Gym/R2E-Gym-Subset).")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--default-agent", default="mini-swe-agent")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=os.environ.get("RLLM_LOG_LEVEL", "INFO"))
    build_benchmark(
        name=args.name,
        split=args.split,
        out_dir=args.out_dir,
        task_ids=args.task_ids,
        limit=args.limit,
        default_agent=args.default_agent,
        hf_repo_id=args.hf_repo_id,
        hf_split=args.hf_split,
        clean=args.clean,
        register=False,
    )


if __name__ == "__main__":
    main()
