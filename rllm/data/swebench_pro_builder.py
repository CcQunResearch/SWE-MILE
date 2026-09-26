"""Builder for the SWE-bench Pro sandbox benchmark.

SWE-bench Pro (``ScaleAI/SWE-bench_Pro``) ships as a flat HF row dataset:
each row carries the problem statement plus the per-instance Docker tag
(``dockerhub_tag``), test fixtures (``fail_to_pass`` / ``pass_to_pass``),
the test file allowlist (``selected_test_files_to_run``), and a
``before_repo_set_cmd`` whose last line typically checks out the gold
test files from a later commit.

The per-instance ``run_script.sh`` + ``parser.py`` (one of each per task,
~1500 small files in total) live in the companion repo at
``github.com/scaleapi/SWE-bench_Pro-os`` under ``run_scripts/<instance_id>/``.
This builder downloads both, then expands each row into rLLM's sandbox
(task-per-directory) shape so ``rllm eval`` runs each task through the
standard ``SandboxedAgentFlow`` + ``ShellScriptEvaluator`` path with no
new Python evaluator needed.

On-disk output (``<out_dir>/``)::

    swebench_pro/
    ├── dataset.toml                       # type="sandbox"
    ├── <instance_id>/
    │   ├── task.toml                      # docker_image=jefzda/sweap-images:<tag>, workdir=/app
    │   ├── instruction.md                 # problem_statement [+ requirements / interface]
    │   ├── environment/Dockerfile         # FROM jefzda/sweap-images:<tag>
    │   ├── tests/
    │   │   ├── test.sh                    # synthesized verifier (this module)
    │   │   ├── run_script.sh              # per-instance, copied from SWE-bench_Pro-os
    │   │   ├── parser.py                  # per-instance, copied from SWE-bench_Pro-os
    │   │   └── instance.json              # base_commit, before_repo_set_cmd, selected/F2P/P2P
    │   └── solution/solve.sh              # apply the gold patch
    └── ...

The verifier replicates the upstream ``swe_bench_pro_eval.py`` flow:
capture the agent's ``git diff`` at ``/app``, hard-reset to base_commit,
re-apply the diff, run the last line of ``before_repo_set_cmd`` (which
brings in the hidden test files), invoke ``run_script.sh``, parse
results via ``parser.py``, and reward 1.0 iff every name in
``fail_to_pass ∪ pass_to_pass`` is in the parser's PASSED set.

Invoked from ``rllm dataset pull swebench_pro`` via the ``builder`` field
in ``rllm/registry/datasets.json`` → :func:`rllm.cli._pull.pull_dataset`.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rllm.data.materialization import (
    bounded_ordered_thread_map,
    resolve_materialization_workers,
)

logger = logging.getLogger(__name__)

HF_REPO_ID = "ScaleAI/SWE-bench_Pro"
HF_REVISION = "7ab5114912baf22bb098818e604c02fe7ad2c11f"
SCRIPTS_REPO_URL = "https://github.com/scaleapi/SWE-bench_Pro-os.git"
SCRIPTS_REVISION = "ca10a60a5fcae51e6948ffe1485d4153d421e6c5"
DOCKERHUB_NAMESPACE = "jefzda/sweap-images"
SOURCE_ROWS_AT_DEFAULT_REVISION = 731
MATERIALIZATION_SCHEMA_VERSION = 1
TASK_SCHEMA_VERSION = 1

_TASK_REQUIRED_FILES = (
    "task.toml",
    "instruction.md",
    "environment/Dockerfile",
    "tests/test.sh",
    "tests/run_script.sh",
    "tests/parser.py",
    "tests/instance.json",
    "solution/gold.patch",
    "solution/solve.sh",
    ".materialized.json",
)

# Per-instance resource defaults. SWE-bench Pro instances ship full JS / Go /
# Python test suites — the upstream eval allocates 1–4 CPU and 5–30 GiB. We
# err on the higher end so the in-sandbox test runner (Mocha/pytest/go test)
# doesn't OOM. Docker ignores these; Modal/Daytona honor them.
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


def _json_fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_text(path: Path, value: str, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            temporary.chmod(mode)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


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


def _clone_scripts_repo(repo_url: str, revision: str) -> Path:
    """Shallow-clone the SWE-bench_Pro-os repo into a temp dir.

    Only ``run_scripts/`` is consumed (1000+ tiny shell + python files); a
    full clone would pull traj/, dockerfiles/, etc. that we don't need. Git
    sparse-checkout works but adds complexity; a plain shallow clone is
    ~50 MB and one-shot.
    """
    tmp = Path(tempfile.mkdtemp(prefix="swebench-pro-scripts-"))
    logger.info("[swebench_pro] fetching %s@%s into %s ...", repo_url, revision, tmp)
    try:
        commands = (
            ["git", "init", "-q", str(tmp)],
            ["git", "-C", str(tmp), "remote", "add", "origin", repo_url],
            ["git", "-C", str(tmp), "fetch", "-q", "--depth", "1", "origin", revision],
            ["git", "-C", str(tmp), "checkout", "-q", "--detach", "FETCH_HEAD"],
        )
        for command in commands:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
    except subprocess.CalledProcessError as e:
        shutil.rmtree(tmp, ignore_errors=True)
        out = e.stdout.decode("utf-8", errors="replace") if e.stdout else ""
        raise RuntimeError(f"git fetch of {repo_url}@{revision} failed:\n{out}") from e
    return tmp


def _decode_json_list(value: Any) -> list[str]:
    """Parse a field that's either a JSON-encoded list or already a list.

    SWE-bench Pro stores ``fail_to_pass`` / ``pass_to_pass`` /
    ``selected_test_files_to_run`` as Python-literal-list strings on HF
    (e.g. ``'["a", "b"]'``). The upstream eval ``eval()`` s them; we
    ``json.loads`` after substituting single quotes when needed.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if not isinstance(value, str):
        return [str(value)]
    text = value.strip()
    if not text:
        return []
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        try:
            import ast

            loaded = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            logger.warning("[swebench_pro] could not parse list-valued field: %r", text[:120])
            return []
    return [str(v) for v in (loaded or [])]


def _last_nonblank_line(text: str) -> str:
    """Mirror upstream's ``before_repo_set_cmd.strip().split('\\n')[-1]`` semantics."""
    if not text:
        return ""
    for line in reversed(text.strip().splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _scripts_dir_for(scripts_root: Path, instance_id: str) -> Path | None:
    """Resolve ``<scripts_root>/run_scripts/<instance_id>/`` with one fallback.

    The HF ``instance_id`` already includes the ``instance_`` prefix
    (e.g. ``instance_NodeBB__NodeBB-...-vnan``), matching the on-disk
    directory name. Some older mirrors strip the prefix — try both.
    """
    base = scripts_root / "run_scripts"
    primary = base / instance_id
    if primary.is_dir():
        return primary
    if instance_id.startswith("instance_"):
        alt = base / instance_id[len("instance_") :]
    else:
        alt = base / f"instance_{instance_id}"
    if alt.is_dir():
        return alt
    return None


def _build_instruction(row: dict) -> str:
    """Compose the agent instruction from problem_statement + extras.

    SWE-bench Pro splits the brief into three fields:
    - ``problem_statement``: the issue / bug report
    - ``requirements``: explicit deliverables (sometimes null)
    - ``interface``: API / interface spec the fix must conform to
    Both extras are optional. We surface them as labeled sections so the
    agent gets the same context the upstream eval gives.
    """
    parts: list[str] = []
    problem = (row.get("problem_statement") or "").strip()
    if problem:
        parts.append(problem)
    requirements = (row.get("requirements") or "").strip()
    if requirements:
        parts.append("## Requirements\n\n" + requirements)
    interface = (row.get("interface") or "").strip()
    if interface:
        parts.append("## Interface\n\n" + interface)
    return "\n\n".join(parts).rstrip() + "\n"


def _build_dockerfile(dockerhub_tag: str) -> str:
    """Dockerfile that pulls jefzda/sweap-images:<tag> and clears the base ENTRYPOINT.

    The image already ships with the repo at ``/app`` checked out to
    ``base_commit``, plus the test-runner dependencies (npm/pytest/go).

    The base SWE-bench Pro images declare ``ENTRYPOINT ["/bin/bash"]``.
    rLLM's docker backend (``rllm/sandbox/backends/docker.py``) starts
    containers with ``command="sleep infinity"`` to keep them alive for
    the agent + verifier execs — but the inherited entrypoint turns the
    invocation into ``/bin/bash sleep infinity``, which bash interprets
    as ``read script file 'sleep'`` and exits immediately. The container
    dies before the first exec, then every ``sandbox.exec`` returns
    HTTP 409 ``container is not running``. Reset ENTRYPOINT to ``[]``
    so docker uses our ``sleep infinity`` command directly.
    """
    return f"FROM {DOCKERHUB_NAMESPACE}:{dockerhub_tag}\nENTRYPOINT []\nWORKDIR /app\n"


def _build_task_toml(
    *,
    instance_id: str,
    repo: str,
    repo_language: str,
    base_commit: str,
    dockerhub_tag: str,
    hf_revision: str,
    scripts_revision: str,
    source_fingerprint: str,
) -> str:
    """Synthesize a Harbor-format ``task.toml``.

    The loader lifts ``[environment].docker_image`` / ``workdir`` /
    resources into ``task.metadata`` so non-docker backends pull the image
    rather than rebuilding the Dockerfile.
    """
    lines = [
        'schema_version = "1.1"',
        "",
        "[task]",
        f'name = "swebench_pro/{instance_id}"',
        f'description = "SWE-bench Pro: {repo} ({repo_language})"',
        f'keywords = ["swe-bench-pro", "{repo_language}"]',
        "",
        "[metadata]",
        f'instance_id = "{instance_id}"',
        f'repo = "{repo}"',
        f'repo_language = "{repo_language}"',
        f'base_commit = "{base_commit}"',
        f'dockerhub_tag = "{dockerhub_tag}"',
        f'source_revision = "{hf_revision}"',
        f'scripts_revision = "{scripts_revision}"',
        f'source_fingerprint = "{source_fingerprint}"',
        'verifier_profile = "swebench_pro_public"',
        "",
        "[environment]",
        f'docker_image = "{DOCKERHUB_NAMESPACE}:{dockerhub_tag}"',
        'workdir = "/app"',
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


# Verifier script template. Uses python3 (always present in sweap-images;
# jq is not) for JSON parsing and reward computation. Logic mirrors the
# upstream entryscript in ``swe_bench_pro_eval.py:create_entryscript``:
#
#   1. ``git diff`` at /app captures the agent's edits as a patch (binary
#      hunks included so png/icon edits don't break apply).
#   2. ``git reset --hard {base_commit}`` + ``git checkout {base_commit}``
#      reset the worktree to the pre-fix baseline.
#   3. ``git apply`` re-applies the agent's patch on top of the reset.
#   4. The last non-blank line of ``before_repo_set_cmd`` runs (matches
#      upstream's ``.split('\n')[-1]`` semantics) — typically a
#      ``git checkout <future-commit> -- <test files>`` that brings in
#      the hidden test fixtures.
#   5. ``run_script.sh`` runs with the selected test files; ``parser.py``
#      normalizes its output into ``{"tests": [{"name", "status"}]}``.
#   6. Reward is 1.0 iff every ``fail_to_pass ∪ pass_to_pass`` name is in
#      the parser's PASSED set; else 0.0.
_VERIFIER_TEMPLATE = r"""#!/bin/bash
set -uo pipefail

mkdir -p /tmp/rllm /logs/verifier || exit 1
REWARD_JSON=/tmp/rllm/reward.json
rm -f "$REWARD_JSON" || exit 1

log() { echo "[verifier] $*"; }

write_failure() {
    local reason="$1"
    local detail="${2:-$reason}"
    python3 - "$reason" "$detail" <<'PY'
import json, os, sys
reason, detail = sys.argv[1:3]
details = {"path": os.environ.get("PATH", "")}
for name, path in (("stdout", "/tmp/stdout.log"), ("stderr", "/tmp/stderr.log")):
    try:
        with open(path, "rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 2000))
            details[name + "_tail"] = stream.read().decode("utf-8", "replace")
    except (OSError, UnicodeError):
        pass
json.dump({"reward": 0.0, "is_correct": False, "metadata": {
    "error": detail,
    "verifier_status": "infrastructure_failure",
    "verifier_diagnostics": details,
    "infrastructure_failure": {"reason": reason, "stage": "verifier",
                               "exception_type": reason, "error_summary": detail, "retryable": False}
}}, open("/tmp/rllm/reward.json", "w"))
PY
    # If the helper itself fails, leave no valid reward for the strict host
    # evaluator to mistake for a model failure.
    exit 1
}

# Parser intermediates belong to this invocation, never an earlier probe.
rm -f /tmp/output.json /tmp/stdout.log /tmp/stderr.log "$REWARD_JSON" || write_failure "verifier_cleanup_failed"

cd /app 2>/dev/null || write_failure "verifier_workspace_missing"

git config --global --add safe.directory "$(pwd)" 2>/dev/null || true

INSTANCE_JSON=/tests/instance.json
if [ ! -f "$INSTANCE_JSON" ]; then
    write_failure "verifier_asset_missing"
fi
for asset in /tests/run_script.sh /tests/parser.py; do
    [ -f "$asset" ] || write_failure "verifier_asset_missing"
done

BASE_COMMIT="$(python3 -c "import json; print(json.load(open('$INSTANCE_JSON'))['base_commit'])")" || write_failure "verifier_contract_invalid"
if [ -z "$BASE_COMMIT" ]; then
    write_failure "verifier_contract_invalid"
fi

TOOLCHAIN_MISSING="$(python3 - "$INSTANCE_JSON" <<'PY'
import json, shutil, sys
with open(sys.argv[1]) as stream:
    language = str(json.load(stream).get("repo_language") or "").strip().lower()
required = {
    "go": ("go",), "golang": ("go",),
    "javascript": ("node",), "js": ("node",),
    "typescript": ("node",), "ts": ("node",),
    "python": ("python3",),
}.get(language, ())
print(", ".join(tool for tool in required if shutil.which(tool) is None))
PY
)" || write_failure "verifier_contract_invalid"
if [ -n "$TOOLCHAIN_MISSING" ]; then
    write_failure "verifier_toolchain_missing" "Required toolchain executable(s) unavailable: $TOOLCHAIN_MISSING"
fi

# Step 1: capture the agent's diff vs base_commit.
#
# Don't trust HEAD — the ``jefzda/sweap-images:*`` images ship with /app
# checked out to some "ready-to-test" commit which may or may not equal
# base_commit. ``git diff --cached <base_commit>`` ignores HEAD entirely
# and diffs the index against the recorded base_commit tree, so the
# captured patch is always the right "agent's edits vs base_commit"
# regardless of what the harness or the image left HEAD pointing at.
log "Capturing agent diff vs base_commit ($BASE_COMMIT)"
MODEL_PATCH=/tmp/model_patch.diff
git add -A . || write_failure "verifier_patch_capture_failed"
git diff --cached --binary "$BASE_COMMIT" > "$MODEL_PATCH" || write_failure "verifier_patch_capture_failed"
git reset || write_failure "verifier_patch_capture_failed"

PATCH_BYTES=$(wc -c < "$MODEL_PATCH" 2>/dev/null || echo 0)
log "Captured patch: $PATCH_BYTES bytes"

# Step 2: reset to base_commit so the patch applies on a clean tree.
#
# ``git reset --hard`` only touches tracked files — runtime artifacts the
# image build process left behind (Redis ``appendonlydir/*.aof``,
# ``dump.rdb``, Python ``__pycache__/``, …) stay as untracked files. They
# end up in the captured patch as ``new file`` entries via ``git add -A``,
# and ``git apply`` then aborts atomically with
# ``error: <path>: already exists in working directory`` because those
# files still exist after the reset. Atomic apply means *no* hunk lands —
# so P2P tests pass (they're baseline behavior) but F2P silently fails
# because the gold patch never touches the source tree. ``git clean -fd``
# clears the untracked junk; the patch's own new-file entries then
# recreate anything the test runner legitimately needs.
log "Resetting to base_commit"
git reset --hard "$BASE_COMMIT" || write_failure "verifier_checkout_failed"
git checkout "$BASE_COMMIT" || write_failure "verifier_checkout_failed"
git clean -fd || write_failure "verifier_checkout_failed"

# Step 3: re-apply the agent's patch.
if [ -s "$MODEL_PATCH" ]; then
    if ! git apply -v "$MODEL_PATCH" 2>&1 | tail -20; then
        write_failure "verifier_patch_apply_failed"
    fi
else
    log "No agent changes detected"
fi

# Step 4: run the last non-blank line of before_repo_set_cmd. Upstream
# eval (swe_bench_pro_eval.py:create_entryscript) does the same — this is
# typically a `git checkout <future-commit> -- <test files>` that pulls in
# the hidden test fixtures the run_script.sh will exercise.
BEFORE_CMD="$(python3 -c "
import json
data = json.load(open('$INSTANCE_JSON'))
cmd = (data.get('before_repo_set_cmd') or '').strip().splitlines()
print(cmd[-1].strip() if cmd else '', end='')
")" || write_failure "verifier_contract_invalid"
if [ -n "$BEFORE_CMD" ]; then
    log "before_repo_set_cmd: $BEFORE_CMD"
    bash -c "$BEFORE_CMD" || write_failure "verifier_test_setup_failed"
fi

# Step 5: run the per-instance run_script.sh with the selected test files.
SELECTED="$(python3 -c "
import json
data = json.load(open('$INSTANCE_JSON'))
print(','.join(data.get('selected_test_files_to_run') or []), end='')
")" || write_failure "verifier_contract_invalid"
log "Running tests: $SELECTED"
chmod +x /tests/run_script.sh || write_failure "verifier_test_setup_failed"
bash /tests/run_script.sh "$SELECTED" > /tmp/stdout.log 2> /tmp/stderr.log || log "run_script.sh exited non-zero (parser inspects logs)"

RUNNER_STARTUP_ERROR="$(python3 <<'PY'
import re, shutil
launchers = ("/tests/run_script.sh", "bash", "/bin/bash", "/usr/bin/bash", "sh", "/bin/sh")
pattern = re.compile(
    r"^(?:" + "|".join(re.escape(value) for value in launchers) + r"): "
    r"(?:(?:line )?\d+: )?"
    r"(go|node|npm|npx|yarn|python(?:[23](?:\.\d+)?)?|pytest|make|gcc|cc): "
    r"(?:command not found|not found|Permission denied|cannot execute: required file not found)\s*$"
)
for path in ("/tmp/stdout.log", "/tmp/stderr.log"):
    with open(path, encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = pattern.match(line)
            if match and shutil.which(match.group(1)) is None:
                print(line.strip()[:1000])
                raise SystemExit(0)
PY
)" || write_failure "verifier_diagnostics_failed"
if [ -n "$RUNNER_STARTUP_ERROR" ]; then
    write_failure "verifier_test_startup_failed" "$RUNNER_STARTUP_ERROR"
fi

# Step 6: parse the run_script.sh output into the standard JSON shape.
python3 /tests/parser.py /tmp/stdout.log /tmp/stderr.log /tmp/output.json 2>>/tmp/stderr.log || write_failure "verifier_parser_failed"

# Step 7: compare passed set against fail_to_pass ∪ pass_to_pass.
python3 <<'PY'
import json, os
REWARD = "/tmp/rllm/reward.json"

def fail(reason, detail):
    json.dump({"reward": 0.0, "is_correct": False, "metadata": {
        "error": detail,
        "verifier_status": "infrastructure_failure",
        "infrastructure_failure": {"reason": reason, "stage": "verifier",
                                   "exception_type": reason, "error_summary": detail, "retryable": False}
    }}, open(REWARD, "w"))
    raise SystemExit(1)

try:
    inst = json.load(open("/tests/instance.json"))
except Exception as e:
    fail("verifier_contract_invalid", "instance.json invalid: {}".format(e))

def _tail(path, n=2000):
    try:
        with open(path, "rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - n))
            return stream.read().decode("utf-8", "replace")
    except Exception:
        return ""

try:
    out = json.load(open("/tmp/output.json"))
except Exception as e:
    fail("verifier_parser_output_invalid", "parser output invalid: {}\n{}\n{}".format(e, _tail("/tmp/stdout.log"), _tail("/tmp/stderr.log")))

if not isinstance(out, dict) or not isinstance(out.get("tests"), list):
    fail("verifier_parser_output_invalid", "expected a tests array")
if any(not isinstance(t, dict) or not isinstance(t.get("name"), str) or not isinstance(t.get("status"), str) for t in out["tests"]):
    fail("verifier_parser_output_invalid", "invalid test record")
if any(not isinstance(inst.get(key), list) or any(not isinstance(t, str) for t in inst[key]) for key in ("fail_to_pass", "pass_to_pass")):
    fail("verifier_contract_invalid", "expected F2P/P2P name arrays")

f2p = set(inst.get("fail_to_pass") or [])
p2p = set(inst.get("pass_to_pass") or [])
required = f2p | p2p
if not required:
    fail("verifier_contract_invalid", "empty F2P/P2P contract")

passed = {t.get("name", "") for t in (out.get("tests") or []) if t.get("status") == "PASSED"}
missing = sorted(required - passed)
matched = sorted(required & passed)

reward = 1.0 if required and not missing else 0.0
json.dump({
    "reward": reward,
    "is_correct": reward >= 1.0,
    "signals": {
        "f2p_required": len(f2p),
        "p2p_required": len(p2p),
        "passed_required": len(matched),
        "missing_required": len(missing),
    },
    "metadata": {
        "missing": missing[:50],
        "passed_required_sample": matched[:10],
        "verifier_diagnostics": {
            "parsed_tests": len(out["tests"]),
            "missing_required_sample": missing[:50],
            "stdout_tail": _tail("/tmp/stdout.log"),
            "stderr_tail": _tail("/tmp/stderr.log"),
        },
    },
}, open(REWARD, "w"))
PY
"""


def _build_verifier_script() -> str:
    """Return ``tests/test.sh`` content (identical across all instances).

    The per-task knobs (base_commit, F2P/P2P, before_repo_set_cmd,
    selected files) live in ``tests/instance.json`` so the verifier
    script itself stays static — easier to audit and to patch in a
    single place if the upstream eval flow changes.
    """
    return _VERIFIER_TEMPLATE


def swebench_pro_verifier_fingerprint() -> str:
    """Return the semantic fingerprint of the synthesized Pro verifier."""

    return hashlib.sha256(
        (
            f"swebench-pro-verifier-v{TASK_SCHEMA_VERSION}\n"
            + _build_verifier_script()
        ).encode("utf-8")
    ).hexdigest()


def _build_solution_script(base_commit: str) -> str:
    """``solution/solve.sh`` applies the gold patch — used by the ``oracle`` harness.

    Hard-resets to ``base_commit`` first. The ``jefzda/sweap-images:*`` images
    don't guarantee ``/app``'s HEAD is at base_commit — upstream's
    ``swe_bench_pro_eval.py:create_entryscript`` always does an explicit
    ``git reset --hard {base_commit} && git checkout {base_commit}`` before
    applying the agent's patch for exactly this reason. Without the reset,
    ``git apply`` may succeed against the wrong base, and the verifier's
    subsequent ``git diff --cached HEAD`` captures a diff that doesn't
    represent the gold patch — F2P tests then silently fail.
    """
    return (
        "#!/bin/bash\n"
        "set -e\n"
        "cd /app\n"
        "git config --global --add safe.directory /app 2>/dev/null || true\n"
        f'git reset --hard "{base_commit}"\n'
        f'git checkout "{base_commit}"\n'
        "git apply -v /solution/gold.patch\n"
    )


def _refresh_synthesized_files(out: Path) -> int:
    """Rewrite ``tests/test.sh`` and ``solution/solve.sh`` for existing tasks.

    Both files are pure templates of this module — the per-task knobs
    live in ``tests/instance.json``. When the templates change (verifier
    fix, oracle reset added, etc.), re-pulling would rebuild from
    scratch but is heavy (clones the SWE-bench_Pro-os repo for files we
    already have). This helper walks the task tree, reads each
    ``instance.json`` for ``base_commit``, and overwrites the two
    synthesized scripts in place. Returns the number of tasks refreshed.
    """
    refreshed = 0
    for task_dir in sorted(out.iterdir()):
        if not task_dir.is_dir():
            continue
        inst_json = task_dir / "tests" / "instance.json"
        if not inst_json.is_file():
            continue
        try:
            inst = json.loads(inst_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        base_commit = inst.get("base_commit", "")
        tests_dir = task_dir / "tests"
        if tests_dir.is_dir():
            (tests_dir / "test.sh").write_text(_build_verifier_script(), encoding="utf-8")
            (tests_dir / "test.sh").chmod(0o755)
        sol_dir = task_dir / "solution"
        if sol_dir.is_dir():
            (sol_dir / "solve.sh").write_text(_build_solution_script(base_commit), encoding="utf-8")
            (sol_dir / "solve.sh").chmod(0o755)
        refreshed += 1
    return refreshed


def _patch_existing_dockerfiles(out: Path) -> int:
    """Backfill ``ENTRYPOINT []`` into Dockerfiles written by older revisions.

    The first sandbox-builder revision emitted a 2-line Dockerfile (just
    ``FROM`` + ``WORKDIR``). Without ``ENTRYPOINT []`` the inherited base
    entrypoint (``["/bin/bash"]``) eats rLLM's ``sleep infinity`` start
    command and the container exits immediately. Re-walk existing task
    dirs and inject the missing line so a re-pull isn't required —
    callers that already have ~/.rllm/datasets/swebench_pro/<id>/ from
    a prior pull pick up the fix on the next build. New tasks are
    written with the corrected line directly by :func:`_build_dockerfile`.
    """
    patched = 0
    for task_dir in out.iterdir():
        if not task_dir.is_dir():
            continue
        dockerfile = task_dir / "environment" / "Dockerfile"
        if not dockerfile.is_file():
            continue
        text = dockerfile.read_text(encoding="utf-8")
        if "ENTRYPOINT" in text:
            continue
        # Inject after the first FROM line. Plain string ops — these
        # Dockerfiles are 2 lines today; no need for a parser.
        lines = text.splitlines()
        new_lines: list[str] = []
        injected = False
        for line in lines:
            new_lines.append(line)
            if not injected and line.lstrip().upper().startswith("FROM "):
                new_lines.append("ENTRYPOINT []")
                injected = True
        if injected:
            dockerfile.write_text("\n".join(new_lines) + ("\n" if text.endswith("\n") else ""), encoding="utf-8")
            patched += 1
    return patched


def _purge_row_materialize_artifacts(out: Path) -> None:
    """Remove artifacts written by the HF-row materialize path.

    ``swebench_pro`` shipped briefly as an HF-row dataset (a previous
    revision of this PR). Its first ``rllm dataset pull`` landed:

        ~/.rllm/datasets/swebench_pro/
        ├── dataset.toml          # transform-style: no [verifier]
        ├── data/test.jsonl       # the row dump
        └── instruction.md.tpl    # row→prompt template

    A subsequent ``pull`` with the sandbox builder writes the new
    ``dataset.toml`` + ``<instance_id>/`` task tree alongside them — but
    leaves ``data/`` in place. ``BenchmarkLoader._has_data_file`` then
    sees ``data/`` and routes through ``_load_data_dataset`` (one Task
    per row, ``sub_dir=None``, ``task.id`` numeric) instead of
    ``_load_sandbox_dataset``. Symptom: every task fails with
    ``missing reference solution at <dataset_root>/solution/solve.sh``
    because ``task.task_dir`` == dataset root, not the instance dir.

    Wipe the conflicting paths up front so re-pulls converge regardless
    of the previous shape this dataset was pulled in.
    """
    for stale in ("data", "images", "instruction.md.tpl"):
        target = out / stale
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            try:
                target.unlink()
            except OSError:
                pass


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
    _atomic_text(out / "dataset.toml", content)


def _materialize_task(
    task_dir: Path,
    row: dict,
    scripts_dir: Path,
    *,
    hf_repo_id: str,
    hf_revision: str,
    scripts_repo_url: str,
    scripts_revision: str,
    source_fingerprint: str,
) -> dict:
    """Expand a single HF row into a Harbor-format task tree. Returns stats."""
    task_dir.mkdir(parents=True, exist_ok=True)

    instance_id = row["instance_id"]
    repo = row.get("repo", "")
    repo_language = row.get("repo_language", "")
    base_commit = row.get("base_commit", "")
    dockerhub_tag = row.get("dockerhub_tag", "")

    # task.toml
    (task_dir / "task.toml").write_text(
        _build_task_toml(
            instance_id=instance_id,
            repo=repo,
            repo_language=repo_language,
            base_commit=base_commit,
            dockerhub_tag=dockerhub_tag,
            hf_revision=hf_revision,
            scripts_revision=scripts_revision,
            source_fingerprint=source_fingerprint,
        ),
        encoding="utf-8",
    )

    # instruction.md
    (task_dir / "instruction.md").write_text(_build_instruction(row), encoding="utf-8")

    # environment/Dockerfile
    env_dir = task_dir / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "Dockerfile").write_text(_build_dockerfile(dockerhub_tag), encoding="utf-8")

    # tests/
    tests_dst = task_dir / "tests"
    tests_dst.mkdir(parents=True, exist_ok=True)
    (tests_dst / "test.sh").write_text(_build_verifier_script(), encoding="utf-8")
    (tests_dst / "test.sh").chmod(0o755)

    # Copy per-instance run_script.sh + parser.py.
    src_run_script = scripts_dir / "run_script.sh"
    src_parser = scripts_dir / "parser.py"
    shutil.copy2(src_run_script, tests_dst / "run_script.sh")
    (tests_dst / "run_script.sh").chmod(0o755)
    shutil.copy2(src_parser, tests_dst / "parser.py")

    instance_data = {
        "instance_id": instance_id,
        "base_commit": base_commit,
        "before_repo_set_cmd": row.get("before_repo_set_cmd", "") or "",
        "selected_test_files_to_run": _decode_json_list(row.get("selected_test_files_to_run")),
        "fail_to_pass": _decode_json_list(row.get("fail_to_pass")),
        "pass_to_pass": _decode_json_list(row.get("pass_to_pass")),
        "repo": repo,
        "repo_language": repo_language,
    }
    (tests_dst / "instance.json").write_text(json.dumps(instance_data, indent=2), encoding="utf-8")

    # solution/ (gold patch + apply script for the oracle harness)
    sol_dst = task_dir / "solution"
    sol_dst.mkdir(parents=True, exist_ok=True)
    patch = row.get("patch") or ""
    (sol_dst / "gold.patch").write_text(patch, encoding="utf-8")
    (sol_dst / "solve.sh").write_text(_build_solution_script(base_commit), encoding="utf-8")
    (sol_dst / "solve.sh").chmod(0o755)

    (task_dir / ".materialized.json").write_text(
        json.dumps(
            {
                "schema_version": TASK_SCHEMA_VERSION,
                "instance_id": instance_id,
                "source_dataset": hf_repo_id,
                "source_revision": hf_revision,
                "scripts_repo": scripts_repo_url,
                "scripts_revision": scripts_revision,
                "source_fingerprint": source_fingerprint,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return {"f2p": len(instance_data["fail_to_pass"]), "p2p": len(instance_data["pass_to_pass"])}


def _load_rows(hf_repo_id: str, hf_split: str, hf_revision: str) -> list[dict]:
    """Load the HF dataset rows as a list of plain dicts."""
    from datasets import load_dataset

    ds = load_dataset(hf_repo_id, split=hf_split, revision=hf_revision)
    return [dict(r) for r in ds]


def _manifest_config(
    *,
    name: str,
    split: str,
    hf_repo_id: str,
    hf_revision: str,
    hf_split: str,
    scripts_repo_url: str,
    scripts_revision: str,
    task_ids: list[str] | None,
    limit: int | None,
    default_agent: str,
) -> dict[str, Any]:
    return {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "name": name,
        "split": split,
        "hf_repo_id": hf_repo_id,
        "hf_revision": hf_revision,
        "hf_split": hf_split,
        "scripts_repo_url": scripts_repo_url,
        "scripts_revision": scripts_revision,
        "task_ids_sha256": _json_fingerprint(task_ids) if task_ids is not None else None,
        "limit": limit,
        "default_agent": default_agent,
    }


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
    return {
        key: config.get(key)
        for key in (
            "name",
            "split",
            "hf_repo_id",
            "hf_revision",
            "hf_split",
            "scripts_repo_url",
            "scripts_revision",
            "task_ids_sha256",
            "limit",
            "default_agent",
        )
    }


def _assert_manifest_compatible(
    manifest: dict[str, Any], config: dict[str, Any], out: Path
) -> None:
    existing = manifest.get("config")
    if not isinstance(existing, dict) or _manifest_identity(existing) != _manifest_identity(config):
        raise ValueError(
            f"{out} was materialized with a different source/configuration; "
            "set CLEAN=1 to rebuild it explicitly"
        )


def _task_fingerprint(
    row: dict[str, Any],
    scripts_dir: Path,
    *,
    hf_repo_id: str,
    hf_revision: str,
    scripts_repo_url: str,
    scripts_revision: str,
) -> str:
    scripts = {
        filename: hashlib.sha256((scripts_dir / filename).read_bytes()).hexdigest()
        for filename in ("run_script.sh", "parser.py")
    }
    return _json_fingerprint(
        {
            "task_schema_version": TASK_SCHEMA_VERSION,
            "row": row,
            "hf_repo_id": hf_repo_id,
            "hf_revision": hf_revision,
            "scripts_repo_url": scripts_repo_url,
            "scripts_revision": scripts_revision,
            "scripts": scripts,
            "verifier_sha256": hashlib.sha256(_build_verifier_script().encode("utf-8")).hexdigest(),
        }
    )


def _task_complete(task_dir: Path, source_fingerprint: str) -> bool:
    if any(not (task_dir / relative).is_file() for relative in _TASK_REQUIRED_FILES):
        return False
    try:
        marker = json.loads((task_dir / ".materialized.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(marker, dict)
        and marker.get("schema_version") == TASK_SCHEMA_VERSION
        and marker.get("source_fingerprint") == source_fingerprint
    )


def _registry_complete(name: str, split: str, selected: int) -> bool:
    from rllm.data import DatasetRegistry

    info = DatasetRegistry.get_dataset_info(name)
    if not info or split not in info.get("splits", {}):
        return False
    split_info = info["splits"][split]
    if split_info.get("num_examples") != selected:
        return False
    dataset_path = Path(DatasetRegistry._resolve_path(split_info["path"]))
    verl_path = Path(DatasetRegistry._verl_path_for(str(dataset_path)))
    try:
        import pyarrow.parquet as pq

        # Opening the parquet footer already proves that the file exists and
        # is readable.  Separate is_file() calls are costly metadata round
        # trips on shared-filesystem and provide no additional guarantee here.
        return (
            pq.read_metadata(dataset_path).num_rows == selected
            and pq.read_metadata(verl_path).num_rows == selected
        )
    except Exception:
        return False


def materialization_complete(
    *,
    name: str = "swebench_pro_public",
    split: str = "test",
    out_dir: str | Path,
    hf_repo_id: str = HF_REPO_ID,
    hf_revision: str = HF_REVISION,
    hf_split: str = "test",
    scripts_repo_url: str = SCRIPTS_REPO_URL,
    scripts_revision: str = SCRIPTS_REVISION,
    task_ids: list[str] | None = None,
    limit: int | None = None,
    default_agent: str = "codeflow",
) -> bool:
    out = Path(out_dir).expanduser()
    manifest = _load_manifest(out)
    if manifest is None:
        return False
    config = _manifest_config(
        name=name,
        split=split,
        hf_repo_id=hf_repo_id,
        hf_revision=hf_revision,
        hf_split=hf_split,
        scripts_repo_url=scripts_repo_url,
        scripts_revision=scripts_revision,
        task_ids=task_ids,
        limit=limit,
        default_agent=default_agent,
    )
    _assert_manifest_compatible(manifest, config, out)
    selected = manifest.get("selected_tasks")
    if (
        manifest.get("schema_version") != MATERIALIZATION_SCHEMA_VERSION
        or manifest.get("config") != config
        or manifest.get("status") != "complete"
        or not isinstance(selected, int)
        or selected <= 0
        or manifest.get("completed_tasks") != selected
        or not manifest.get("registered")
        or not (out / "dataset.toml").is_file()
    ):
        return False
    task_inventory = manifest.get("tasks")
    if not isinstance(task_inventory, list) or len(task_inventory) != selected:
        return False
    seen_task_ids: set[str] = set()
    for task in task_inventory:
        if not isinstance(task, dict):
            return False
        task_id = task.get("id")
        source_fingerprint = task.get("source_fingerprint")
        if (
            not isinstance(task_id, str)
            or not task_id
            or task_id in seen_task_ids
            or not isinstance(source_fingerprint, str)
            or not source_fingerprint
        ):
            return False
        seen_task_ids.add(task_id)

    # Do not restat every file in every completed task on every invocation.
    # The builder stages each task, validates it with _task_complete(), then
    # atomically renames it into place; it writes status=complete only after
    # all tasks and registry parquet files are complete.  An interrupted run
    # therefore remains status=in_progress and enters the normal per-task
    # resume path, which still calls _task_complete() for every selected row.
    # Trusting the final manifest here avoids roughly 8,000 serial shared-filesystem stat
    # calls for the 731-task public set.
    return _registry_complete(name, split, selected)


def build_benchmark(
    *,
    name: str = "swebench_pro_public",
    split: str = "test",
    out_dir: str | Path,
    catalog_entry: dict | None = None,
    task_ids: list[str] | None = None,
    limit: int | None = None,
    default_agent: str = "codeflow",
    hf_repo_id: str = HF_REPO_ID,
    hf_revision: str = HF_REVISION,
    hf_split: str = "test",
    scripts_repo_url: str = SCRIPTS_REPO_URL,
    scripts_revision: str = SCRIPTS_REVISION,
    clean: bool = False,
    register: bool = True,
    max_workers: int | None = None,
) -> Path:
    """Materialize the pinned SWE-bench Pro public split with task-level resume."""

    if catalog_entry:
        default_agent = catalog_entry.get("default_agent") or default_agent
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer when set")

    out = Path(out_dir).expanduser()
    config = _manifest_config(
        name=name,
        split=split,
        hf_repo_id=hf_repo_id,
        hf_revision=hf_revision,
        hf_split=hf_split,
        scripts_repo_url=scripts_repo_url,
        scripts_revision=scripts_revision,
        task_ids=task_ids,
        limit=limit,
        default_agent=default_agent,
    )
    with _materialization_lock(out):
        if clean and out.exists():
            logger.info("[swebench_pro] removing existing %s", out)
            shutil.rmtree(out)
        if not clean and materialization_complete(
            name=name,
            split=split,
            out_dir=out,
            hf_repo_id=hf_repo_id,
            hf_revision=hf_revision,
            hf_split=hf_split,
            scripts_repo_url=scripts_repo_url,
            scripts_revision=scripts_revision,
            task_ids=task_ids,
            limit=limit,
            default_agent=default_agent,
        ):
            logger.info("[swebench_pro] already complete; skipping %s", out)
            return out

        out.mkdir(parents=True, exist_ok=True)
        _purge_row_materialize_artifacts(out)
        existing_manifest = _load_manifest(out)
        if existing_manifest is not None:
            _assert_manifest_compatible(existing_manifest, config, out)
        for stale in out.glob(".tmp-*"):
            if stale.is_dir():
                shutil.rmtree(stale)
            else:
                stale.unlink(missing_ok=True)

        logger.info(
            "[swebench_pro] loading %s split=%s revision=%s",
            hf_repo_id,
            hf_split,
            hf_revision,
        )
        source_rows = _load_rows(hf_repo_id, hf_split, hf_revision)
        if (
            hf_repo_id == HF_REPO_ID
            and hf_revision == HF_REVISION
            and hf_split == "test"
            and len(source_rows) != SOURCE_ROWS_AT_DEFAULT_REVISION
        ):
            raise ValueError(
                f"pinned SWE-bench Pro source expected {SOURCE_ROWS_AT_DEFAULT_REVISION} rows, "
                f"got {len(source_rows)}"
            )
        rows = source_rows
        if task_ids is not None:
            requested = set(task_ids)
            if len(requested) != len(task_ids):
                raise ValueError("task_ids contains duplicates")
            rows = [row for row in rows if row.get("instance_id") in requested]
            missing = sorted(requested - {str(row.get("instance_id")) for row in rows})
            if missing:
                raise ValueError(f"unknown SWE-bench Pro task_ids: {missing[:5]}")
        if limit is not None:
            rows = rows[:limit]
        ids = [str(row.get("instance_id") or "") for row in rows]
        if not ids or any(not task_id for task_id in ids) or len(ids) != len(set(ids)):
            raise ValueError("SWE-bench Pro selection is empty or contains empty/duplicate instance_id values")

        description = (catalog_entry or {}).get("description") or (
            "SWE-bench Pro (Public): 731 enterprise-grade SWE tasks across 11 "
            "open-source repositories, with pre-built images and F2P/P2P grading."
        )
        _write_dataset_toml(
            out,
            name=name,
            split=split,
            description=description,
            default_agent=default_agent,
        )
        manifest: dict[str, Any] = {
            "schema_version": MATERIALIZATION_SCHEMA_VERSION,
            "status": "in_progress",
            "config": config,
            "source_rows": len(source_rows),
            "selected_tasks": len(rows),
            "selected_task_ids_sha256": _json_fingerprint(ids),
            "completed_tasks": 0,
            "reused_tasks": 0,
            "registered": False,
            "tasks": [],
        }
        _atomic_json(out / "materialization.json", manifest)

        scripts_root = _clone_scripts_repo(scripts_repo_url, scripts_revision)
        try:
            prepared: list[tuple[dict[str, Any], Path, str]] = []
            for row in rows:
                instance_id = str(row["instance_id"])
                scripts_dir = _scripts_dir_for(scripts_root, instance_id)
                if scripts_dir is None:
                    raise FileNotFoundError(
                        f"{instance_id}: no run_scripts entry in {scripts_repo_url}@{scripts_revision}"
                    )
                missing_scripts = [
                    filename
                    for filename in ("run_script.sh", "parser.py")
                    if not (scripts_dir / filename).is_file()
                ]
                if missing_scripts:
                    raise FileNotFoundError(
                        f"{instance_id}: missing official verifier assets {missing_scripts}"
                    )
                source_fingerprint = _task_fingerprint(
                    row,
                    scripts_dir,
                    hf_repo_id=hf_repo_id,
                    hf_revision=hf_revision,
                    scripts_repo_url=scripts_repo_url,
                    scripts_revision=scripts_revision,
                )
                prepared.append((row, scripts_dir, source_fingerprint))

            def materialize_one(
                item: tuple[dict[str, Any], Path, str],
            ) -> tuple[dict[str, Any], bool, dict[str, str]]:
                row, scripts_dir, source_fingerprint = item
                instance_id = str(row["instance_id"])
                task_dst = out / instance_id
                row_reused = _task_complete(task_dst, source_fingerprint)
                if not row_reused:
                    staging = out / f".tmp-{instance_id}-{uuid.uuid4().hex}"
                    try:
                        _materialize_task(
                            staging,
                            row,
                            scripts_dir,
                            hf_repo_id=hf_repo_id,
                            hf_revision=hf_revision,
                            scripts_repo_url=scripts_repo_url,
                            scripts_revision=scripts_revision,
                            source_fingerprint=source_fingerprint,
                        )
                        if not _task_complete(staging, source_fingerprint):
                            raise RuntimeError(
                                f"{instance_id}: staged task failed completeness validation"
                            )
                        if task_dst.exists():
                            shutil.rmtree(task_dst)
                        os.replace(staging, task_dst)
                    except BaseException:
                        shutil.rmtree(staging, ignore_errors=True)
                        raise
                return (
                    {
                        "id": instance_id,
                        "instruction": (task_dst / "instruction.md").read_text(encoding="utf-8"),
                        "task_path": str(task_dst.resolve()),
                        "repo_name": row.get("repo", ""),
                        "commit_hash": row.get("base_commit", ""),
                        "docker_image": f"{DOCKERHUB_NAMESPACE}:{row.get('dockerhub_tag', '')}",
                        "repo_language": row.get("repo_language", ""),
                        "source_dataset": hf_repo_id,
                        "source_revision": hf_revision,
                        "scripts_revision": scripts_revision,
                        "source_fingerprint": source_fingerprint,
                        "verifier_profile": "swebench_pro_public",
                    },
                    row_reused,
                    {
                        "id": instance_id,
                        "source_fingerprint": source_fingerprint,
                    },
                )

            workers = resolve_materialization_workers(len(prepared), max_workers)
            logger.info(
                "[swebench_pro] materializing with %d worker(s)", workers
            )
            registration_rows: list[dict[str, Any]] = []
            reused = 0
            results = bounded_ordered_thread_map(
                materialize_one,
                prepared,
                workers=workers,
                thread_name_prefix="swebench-pro-materialize",
            )
            for index, (registry_row, row_reused, manifest_task) in enumerate(
                results,
                start=1,
            ):
                reused += int(row_reused)
                registration_rows.append(registry_row)
                manifest["completed_tasks"] = index
                manifest["reused_tasks"] = reused
                manifest["tasks"].append(manifest_task)
                _atomic_json(out / "materialization.json", manifest)
                if index % 25 == 0 or index == len(rows):
                    logger.info(
                        "[swebench_pro] progress %d/%d (reused=%d)",
                        index,
                        len(rows),
                        reused,
                    )

            if len(registration_rows) != len(rows):
                raise RuntimeError(
                    f"SWE-bench Pro selected {len(rows)} tasks but materialized {len(registration_rows)}"
                )
            if register:
                from rllm.data import DatasetRegistry

                DatasetRegistry.register_dataset(
                    name=name,
                    data=registration_rows,
                    split=split,
                    source=f"{hf_repo_id}@{hf_revision}",
                    description=description,
                    category=(catalog_entry or {}).get("category", "code"),
                )
                if not _registry_complete(name, split, len(rows)):
                    raise RuntimeError(
                        f"{name}/{split} registry failed {len(rows)}-row completeness validation"
                    )

            manifest["status"] = "complete"
            manifest["completed_tasks"] = len(rows)
            manifest["reused_tasks"] = reused
            manifest["registered"] = bool(register)
            _atomic_json(out / "materialization.json", manifest)
            logger.info(
                "[swebench_pro] complete: tasks=%d reused=%d out=%s",
                len(rows),
                reused,
                out,
            )
            return out
        finally:
            shutil.rmtree(scripts_root, ignore_errors=True)


def main() -> None:
    """CLI: ``python -m rllm.data.swebench_pro_builder --out-dir <dir>``."""
    import argparse

    parser = argparse.ArgumentParser(description="Materialize SWE-bench Pro into an rLLM sandbox benchmark directory.")
    parser.add_argument("--out-dir", required=True, help="Output benchmark directory.")
    parser.add_argument("--name", default="swebench_pro_public")
    parser.add_argument("--split", default="test")
    parser.add_argument("--hf-repo-id", default=HF_REPO_ID)
    parser.add_argument("--hf-revision", default=HF_REVISION)
    parser.add_argument("--hf-split", default="test")
    parser.add_argument("--scripts-repo-url", default=SCRIPTS_REPO_URL)
    parser.add_argument("--scripts-revision", default=SCRIPTS_REVISION)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--default-agent", default="codeflow")
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
        hf_revision=args.hf_revision,
        hf_split=args.hf_split,
        scripts_repo_url=args.scripts_repo_url,
        scripts_revision=args.scripts_revision,
        clean=args.clean,
        register=False,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "HF_REPO_ID",
    "HF_REVISION",
    "MATERIALIZATION_SCHEMA_VERSION",
    "SCRIPTS_REPO_URL",
    "SCRIPTS_REVISION",
    "SOURCE_ROWS_AT_DEFAULT_REVISION",
    "TASK_SCHEMA_VERSION",
    "build_benchmark",
    "materialization_complete",
    "swebench_pro_verifier_fingerprint",
]
