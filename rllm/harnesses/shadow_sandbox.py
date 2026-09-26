"""Shadow sandbox runtime for native ``codeflow`` rollouts.

The runtime owns no sandbox lifecycle; :class:`rllm.hooks.SandboxTaskHooks`
creates and closes both environments.  It serializes replay and probes on one
worker because a mutable working tree cannot safely serve concurrent tests.
Bug-repair tasks keep their established primary-verifier path.  In the
opt-in DeNovoSWE background path, the primary sandbox remains authoritative
for the final outcome while the detached shadow runs milestone probes only.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import logging
import math
import queue
import re
import shlex
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rllm.eval.types import EvalOutput, Signal
from rllm.harnesses.action_event import (
    ACTION_EVENT_METADATA_KEY,
    ExecutionStatus,
    RepoSnapshot,
    RepoStateStatus,
    RestoreConfirmation,
    TestCountObservation,
    TestEvent,
    TestEventContinuityStatus,
    TestEventStatus,
    TestPartitionCounts,
    TestPartitionResult,
    TestResultSource,
    ValidationStatus,
    changed_files_between,
    resolve_test_event_continuity,
)
from rllm.harnesses.codeflow_shell_policy import (
    detect_preconfigured_environment_violation,
    is_bounded_read_only_head_pipeline,
    is_proven_side_effect_free_shell_command,
)
from rllm.harnesses.partition_probe import (
    COUNT_RUNNER_PLAN_SCHEMA_VERSION,
    PartitionAdapter,
    PartitionAdapterUnsupported,
    PartitionObservation,
    build_count_runner_plan,
    build_non_python_partition_adapter,
    build_partition_adapter,
    encode_instance,
    expand_runner_commands,
    observe_partition,
    wrap_count_command,
)
from rllm.harnesses.repository_state import helper_source as repository_helper_source
from rllm.sandbox.structured_exec import (
    StructuredCommandProtocolError,
    frame_structured_command,
    parse_structured_command_output,
)
from rllm.types import Episode, RolloutInfrastructureError, ShadowFinalizationError, Step, Task

logger = logging.getLogger(__name__)

SHADOW_METADATA_KEY = "shadow_sandbox"
SHADOW_SCHEMA_VERSION = 22
FINAL_RECOVERY_CHUNK_BYTES = 1024 * 1024
FINAL_RECOVERY_MAX_BUNDLE_BYTES = 256 * 1024 * 1024
FINAL_RECOVERY_MAX_ATTEMPTS = 2
DEFAULT_SHADOW_FINALIZE_STALL_TIMEOUT = 600.0
CONTRACT_FILL_POLICY = "f2p_missing_neutral_v1"
DEFAULT_SHADOW_PARTITION_TIMEOUT_RECOVERY = True
DEFAULT_SHADOW_PARTITION_RECOVERY_COMMAND_TIMEOUT = 300.0
DEFAULT_SHADOW_PARTITION_RECOVERY_MAX_COMMANDS = 6
DEFAULT_BUG_REPAIR_VERIFICATION_POTENTIAL_MODE = "normalized"
BUG_REPAIR_PASS_COUNT_MODE = "bug_repair_pass_count"
# Compatibility export for callers that still import the old constant.
DEFAULT_SHADOW_FINALIZE_TIMEOUT = DEFAULT_SHADOW_FINALIZE_STALL_TIMEOUT
_STATUS_VALUES = {"PASSED", "FAILED", "ERROR"}
_RETRIABLE_PROBE_FAILURES = {
    "probe_execution",
    "probe_exit_marker",
    "parse_error",
}
_EXACT_TEST_SET_POLICY = "exact"
_ANNOTATED_SUBSET_TEST_SET_POLICY = "annotated_subset"
_STOP = object()


class _ShadowCancelled(Exception):
    """Internal control flow for an explicitly cancelled shadow runtime."""
_ALL_F2P_CONTRACT_FILL_LANGUAGES = frozenset(
    {"python", "go", "javascript", "js", "typescript", "ts"}
)
_PARTITION_INFRA_FAILURES = frozenset(
    {
        "dependency_network",
        "toolchain_unavailable",
        "verifier_crash",
        "probe_execution",
        "probe_checkpoint",
        "probe_before_mismatch",
        "verifier_state_mismatch",
        "repo_state_unavailable",
        "worker",
    }
)
R2E_RESULT_PARSER = "r2e_pytest_summary_v1"
SWEREBENCH_V2_RESULT_PARSER = "swerebench_v2_pytest_v1"
SWEREBENCH_V2_OFFICIAL_RESULT_PARSER_V1 = "swerebench_v2_official_v1"
SWEREBENCH_V2_OFFICIAL_RESULT_PARSER = "swerebench_v2_official_v2"
SWEREBENCH_V2_RESULT_ADAPTER_VERSION = 6
SWEREBENCH_V2_RESULT_PARSERS = frozenset(
    {
        SWEREBENCH_V2_RESULT_PARSER,
        SWEREBENCH_V2_OFFICIAL_RESULT_PARSER_V1,
        SWEREBENCH_V2_OFFICIAL_RESULT_PARSER,
    }
)
DENOVOSWE_RESULT_PARSER = "denovoswe_official_v1"
SUPPORTED_RESULT_PARSERS = frozenset({R2E_RESULT_PARSER, *SWEREBENCH_V2_RESULT_PARSERS, DENOVOSWE_RESULT_PARSER})
DEFAULT_TEST_RESULTS_PATH = "/tmp/rllm/test_results.json"

_NODE_YARN_WRAPPER_SCRIPT = r'''#!/bin/sh
set -u

original_path=${RLLM_ORIGINAL_PATH:-$PATH}
node_bin=$(PATH="$original_path" command -v node 2>/dev/null || true)

# Yarn Berry repositories vendor the exact CLI under .yarn/releases. Prefer
# that project-local CLI to a host Corepack shim; this avoids Corepack/Node
# version skew such as globalThis.fetch being unavailable on an older image.
if [ -n "$node_bin" ]; then
    search_dir=${RLLM_REPO_DIR:-$PWD}
    while [ -n "$search_dir" ] && [ "$search_dir" != "/" ]; do
        for yarn_js in "$search_dir"/.yarn/releases/yarn-*.cjs "$search_dir"/.yarn/releases/yarn-*.js; do
            if [ -f "$yarn_js" ]; then
                PATH="$original_path" exec "$node_bin" "$yarn_js" "$@"
            fi
        done
        search_dir=${search_dir%/*}
        [ -n "$search_dir" ] || search_dir=/
    done
fi

# Do not stop at the first executable named yarn: Hadoop installs a different
# CLI with that name on many benchmark images. Validate every PATH candidate.
old_ifs=$IFS
IFS=:
for path_dir in $original_path; do
    [ -n "$path_dir" ] || path_dir=.
    native_yarn="$path_dir/yarn"
    if [ -f "$native_yarn" ] && [ -x "$native_yarn" ]; then
        native_version=$(PATH="$original_path" timeout 10s "$native_yarn" --version 2>/dev/null || true)
        case "$native_version" in
            [0-9]*.[0-9]*) IFS=$old_ifs; PATH="$original_path" exec "$native_yarn" "$@" ;;
        esac
    fi
done
IFS=$old_ifs

if [ -n "$node_bin" ]; then
    yarn_js=$(PATH="$original_path" "$node_bin" -p \
        "try { require.resolve('yarn/bin/yarn.js') } catch (_) { '' }" \
        2>/dev/null || true)
    if [ -n "$yarn_js" ]; then
        PATH="$original_path" exec "$node_bin" "$yarn_js" "$@"
    fi
fi

npm_bin=$(PATH="$original_path" command -v npm 2>/dev/null || true)
if [ -n "$node_bin" ] && [ -n "$npm_bin" ]; then
    npm_root=$(PATH="$original_path" "$npm_bin" root -g 2>/dev/null || true)
    for yarn_js in "$npm_root"/yarn/bin/yarn.js "$npm_root"/yarn/bin/yarn.cjs; do
        if [ -f "$yarn_js" ]; then
            PATH="$original_path" exec "$node_bin" "$yarn_js" "$@"
        fi
    done
fi

corepack_bin=$(PATH="$original_path" command -v corepack 2>/dev/null || true)
node_major=0
if [ -n "$node_bin" ]; then
    node_major=$(PATH="$original_path" "$node_bin" -p "Number(process.versions.node.split('.')[0])" 2>/dev/null || printf '0')
fi
if [ -n "$corepack_bin" ] && [ "$node_major" -ge 18 ] 2>/dev/null; then
    corepack_version=$(
        COREPACK_ENABLE_DOWNLOAD_PROMPT=0 COREPACK_ENABLE_NETWORK=0 \
        PATH="$original_path" timeout 10s "$corepack_bin" yarn --version 2>/dev/null || true
    )
    case "$corepack_version" in
        [0-9]*.[0-9]*)
            COREPACK_ENABLE_DOWNLOAD_PROMPT=0 COREPACK_ENABLE_NETWORK=0 \
            PATH="$original_path" exec "$corepack_bin" yarn "$@"
            ;;
    esac
fi

printf '%s\n' \
    'RLLM verifier requires Node Yarn, but PATH resolves yarn to a non-Node executable and no Corepack/Yarn module is available' \
    >&2
exit 127
'''

_PACKAGE_SCRIPT_RESOLVER = r'''
import base64
import glob
import hashlib
import json
import os
import re
import shlex
import sys

root = os.path.realpath(sys.argv[1])
commands = json.loads(base64.b64decode(sys.argv[2]).decode("utf-8"))
package_re = re.compile(
    r"(?<![\w./-])(?P<tool>npm|pnpm|yarn)\s+"
    r"(?:(?:workspace\s+(?P<workspace_scope>[^\s;&|]+)\s+)|"
    r"(?:(?:--workspace|--filter|-w)(?:=|\s+)(?P<flag_scope>[^\s;&|]+)\s+))*"
    r"(?:(?P<verb>run|run-script)\s+)?(?P<script>[\w.:_-]+)\b"
)
runner_patterns = {
    "jest": re.compile(r"(?<![\w./:-])(?:npx\s+)?(?:[\w./-]+/)?jest(?:\.js)?(?=\s|$|[;&|])"),
    "mocha": re.compile(r"(?<![\w./:-])(?:npx\s+)?(?:[\w./-]+/)?(?:mocha|_mocha)(?=\s|$|[;&|])"),
    "vitest": re.compile(r"(?<![\w./:-])(?:npx\s+)?(?:[\w./-]+/)?vitest(?=\s|$|[;&|])"),
    "ava": re.compile(r"(?<![\w./:-])(?:npx\s+)?(?:[\w./-]+/)?ava(?=\s|$|[;&|])"),
    "node": re.compile(r"(?<![\w./:-])(?:[\w./-]+/)?node\s+--test(?=\s|$|[;&|])"),
    "tap": re.compile(r"(?<![\w./:-])(?<!reporter\s)(?:npx\s+)?(?:[\w./-]+/)?tap(?=\s|$|[;&|])"),
    "hardhat": re.compile(r"(?<![\w./:-])(?:npx\s+)?(?:[\w./-]+/)?hardhat\s+test(?=\s|$|[;&|])"),
    "borp": re.compile(r"(?<![\w./:-])(?:npx\s+)?(?:[\w./-]+/)?borp(?=\s|$|[;&|])"),
}

def load_package(path):
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("scripts"), dict):
        raise ValueError("package_scripts_missing")
    return value

def package_for(command, match):
    cwd = root
    prefix = command[:match.start()]
    for cd_match in re.finditer(r"(?:^|&&)\s*cd\s+([^;&|]+?)\s*(?=&&|$)", prefix):
        tokens = shlex.split(cd_match.group(1))
        if len(tokens) != 1:
            raise ValueError("package_script_cwd_ambiguous")
        cwd = os.path.realpath(os.path.join(cwd, tokens[0]))
    scope = match.group("workspace_scope") or match.group("flag_scope")
    if scope:
        root_package_path = os.path.join(cwd, "package.json")
        root_package = load_package(root_package_path)
        workspaces = root_package.get("workspaces") or []
        if isinstance(workspaces, dict):
            workspaces = workspaces.get("packages") or []
        candidates = []
        for pattern in workspaces if isinstance(workspaces, list) else []:
            if not isinstance(pattern, str):
                continue
            for directory in glob.glob(os.path.join(cwd, pattern)):
                package_path = os.path.join(directory, "package.json")
                try:
                    package = load_package(package_path)
                except Exception:
                    continue
                if package.get("name") == scope or os.path.basename(directory) == scope:
                    candidates.append(package_path)
        if len(set(candidates)) != 1:
            raise ValueError("package_workspace_ambiguous")
        return os.path.realpath(candidates[0]), cwd, scope
    return os.path.realpath(os.path.join(cwd, "package.json")), cwd, None

result = {}
for index, command in enumerate(commands):
    match = package_re.search(command)
    if match is None:
        continue
    try:
        package_path, package_cwd, workspace_scope = package_for(command, match)
        package = load_package(package_path)
        name = match.group("script")
        seen = set()
        chain = []
        runner = None
        for _ in range(4):
            if name in seen:
                raise ValueError("package_script_cycle")
            seen.add(name)
            body = package["scripts"].get(name)
            if not isinstance(body, str) or not body.strip():
                raise ValueError("package_script_missing")
            if "$(" in body or "`" in body:
                raise ValueError("package_script_dynamic")
            chain.append({"name": name, "content": body})
            hits = [family for family, pattern in runner_patterns.items() if pattern.search(body)]
            if re.search(r"(?<![\w./-])(?:npx\s+)?tsdx\s+test(?=\s|$|[;&|])", body):
                hits.append("jest")
            if re.search(r"(?<![\w./-])(?:npx\s+)?test-ava(?=\s|$|[;&|])", body):
                hits.append("ava")
            hits = sorted(set(hits))
            if len(hits) > 1:
                raise ValueError("package_script_runner_ambiguous")
            if hits:
                runner = hits[0]
                break
            nested = package_re.search(body)
            if nested is None:
                raise ValueError("package_script_runner_unrecognized")
            name = nested.group("script")
        if runner is None:
            raise ValueError("package_script_depth_exceeded")
        relative_path = os.path.relpath(package_path, root).replace(os.sep, "/")
        digest_payload = {
            "path": relative_path,
            "name": match.group("script"),
            "chain": chain,
            "runner": runner,
        }
        digest = hashlib.sha256(
            json.dumps(digest_payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        result[str(index)] = {
            "status": "ok",
            "path": relative_path,
            "name": match.group("script"),
            "hash": digest,
            "runner": runner,
            "chain": chain,
            "cwd": os.path.relpath(package_cwd, root).replace(os.sep, "/"),
            "workspace_scope": workspace_scope,
        }
    except Exception as exc:
        result[str(index)] = {"status": "error", "reason": str(exc)}
print(json.dumps(result, ensure_ascii=True, separators=(",", ":")))
'''

_MOCHA_COUNT_REPORTER_SCRIPT = r'''"use strict";
const fs = require("fs");
const path = require("path");

module.exports = function RllmCountReporter(runner) {
  const tests = [];
  function record(test, status) {
    if (!test || (test.type && test.type !== "test")) return;
    const fullTitle = typeof test.fullTitle === "function" ? test.fullTitle() : String(test.title || "");
    tests.push({
      fullTitle,
      title: String(test.title || ""),
      file: String(test.file || ""),
      status,
    });
  }
  runner.on("pass", (test) => record(test, "passed"));
  runner.on("fail", (test) => record(test, "failed"));
  runner.on("pending", (test) => record(test, "pending"));
  runner.once("end", () => {
    const output = process.env.RLLM_SHADOW_NODE_RESULT_PATH;
    if (!output) return;
    fs.mkdirSync(path.dirname(output), { recursive: true });
    const temporary = output + "." + process.pid + ".tmp";
    fs.writeFileSync(temporary, JSON.stringify({ schema_version: 1, tests }));
    fs.renameSync(temporary, output);
  });
};
'''

_PARSE_DENOVO_RESULTS_SCRIPT = r"""
import json
import sys

path = sys.argv[1]
try:
    value = json.load(open(path, encoding="utf-8"))
except Exception as exc:
    print(json.dumps({"ok": False, "error": "reading DeNovoSWE results: %s" % exc}))
    raise SystemExit(0)
if not isinstance(value, dict):
    print(json.dumps({"ok": False, "error": "DeNovoSWE results are not an object"}))
elif value.get("verifier_status") == "timeout":
    print(json.dumps({"ok": False, "error": "DeNovo test execution timed out", "timed_out": True, "failure_type": "verifier_timeout", "process_cleanup_confirmed": value.get("process_cleanup_confirmed") is True}))
elif value.get("infrastructure_error"):
    print(json.dumps({"ok": False, "error": str(value["infrastructure_error"]), "infrastructure": True, "process_cleanup_confirmed": value.get("process_cleanup_confirmed") is True}))
else:
    print(json.dumps({"ok": True, "outcome": value}, ensure_ascii=False))
"""

_READ_PARTITION_ARTIFACT_SCRIPT = r"""
import json
import sys

artifact_path, log_path = sys.argv[1:3]
try:
    artifact = json.load(open(artifact_path, encoding="utf-8"))
except Exception:
    artifact = None
try:
    log = open(log_path, encoding="utf-8", errors="replace").read()
except Exception:
    log = ""
print(json.dumps({"artifact": artifact, "log": log}, ensure_ascii=False))
"""


def _looks_like_timeout_error(error: BaseException | str | None) -> bool:
    """Recognize an execution-layer timeout without inspecting command output.

    Sandbox implementations occasionally wrap ``TimeoutError`` while retaining
    the exception class name.  Test output, command text, and arbitrary error
    messages are intentionally excluded: benchmark test ids commonly contain
    words such as ``Timeout`` and must not become infrastructure failures.
    """

    if error is None:
        return False
    if isinstance(error, TimeoutError):
        return True
    return type(error).__name__ in {
        "APITimeoutError",
        "DeadlineExceeded",
        "DeadlineExceededError",
        "SandboxTimeoutError",
    }


def _looks_like_resource_error(error: BaseException | str | None) -> bool:
    """Recognize a typed sandbox resource failure without reading its text."""

    if error is None:
        return False
    if isinstance(error, MemoryError):
        return True
    return type(error).__name__ in {
        "OutOfMemoryError",
        "ResourceExhausted",
        "ResourceExhaustedError",
        "SandboxResourceExhaustedError",
    }


def _partition_infrastructure_failure(artifact: dict[str, Any] | None, log: str) -> str | None:
    if isinstance(artifact, dict) and artifact.get("infrastructure_error"):
        return "partition_infrastructure_error"
    execution = artifact.get("execution") if isinstance(artifact, dict) else None
    if isinstance(execution, dict) and execution.get("resource_exhausted") is True:
        return "partition_oom"
    commands = execution.get("commands") if isinstance(execution, dict) else None
    exit_codes = [
        command.get("exit_code")
        for command in commands
        if isinstance(command, dict) and type(command.get("exit_code")) is int
    ] if isinstance(commands, list) else []
    signals = [
        command.get("signal")
        for command in commands
        if isinstance(command, dict) and type(command.get("signal")) is int
    ] if isinstance(commands, list) else []
    if any(signal in {6, 9, 11} for signal in signals) or any(
        code in {134, 137, 139} for code in exit_codes
    ):
        return "partition_process_signal"
    if not any(code != 0 for code in exit_codes):
        return None
    patterns = {
        "partition_network": (
            r"^(?:.*temporary failure in name resolution.*|.*could not resolve host.*|"
            r"network is unreachable|econnreset|enetunreach|npm err! network|"
            r"sockettimeoutexception|connect timed out|connection timed out|"
            r"failed to download|could not install gradle distribution|"
            r"could not get resource|could not transfer artifact).*$"
        ),
        "partition_storage": r"^.*no space left on device.*$",
    }
    for failure_type, pattern in patterns.items():
        if re.search(pattern, log, flags=re.IGNORECASE | re.MULTILINE):
            return failure_type
    return None


@dataclass(frozen=True)
class _RepositoryDeltaBundle:
    local_path: str
    size: int
    sha256: str
    present: tuple[str, ...]
    deleted: tuple[str, ...]
    expected_fingerprint: str
    state_version: int = 1


@dataclass
class _BaselineSingleflight:
    ready: threading.Event
    result: dict[str, Any] | None = None


_BASELINE_SINGLEFLIGHT_LOCK = threading.Lock()
_BASELINE_SINGLEFLIGHTS: dict[str, _BaselineSingleflight] = {}
_BASELINE_SINGLEFLIGHT_MAX_COMPLETED = 128


def _check_repository_helper(result: dict[str, Any], phase: str, sandbox: Any) -> None:
    """Keep execution evidence distinct from a malformed *successful* response."""
    if result.get("ok"):
        return
    diagnostics = dict(result.get("diagnostics") or {})
    diagnostics.setdefault("phase", result.get("phase") or phase)
    diagnostics.setdefault("error_type", result.get("error_type") or "RuntimeError")
    diagnostics.setdefault("exit_code", result.get("exit_code"))
    diagnostics.setdefault("interpreter", getattr(sandbox, "_rllm_codeflow_helper_runtime", None))
    detail = str(result.get("detail") or result.get("error") or "missing helper response")[-1000:]
    diagnostics.setdefault("stderr", detail)
    error = RuntimeError(
        f"repository {phase} failed ({diagnostics['error_type']}, "
        f"exit_code={diagnostics['exit_code']}): {detail}"
    )
    error.diagnostics = diagnostics
    raise error


def _export_primary_repository_delta_bundle(
    harness: Any,
    task: Task,
    primary: Any,
    baseline_ref: str,
    expected_fingerprint: str,
    *,
    uid: str,
    cancel_event: threading.Event | None = None,
) -> _RepositoryDeltaBundle:
    """Capture an exact, bounded primary Git delta at action handoff."""

    def check_cancelled() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise _ShadowCancelled("repository delta capture cancelled")

    check_cancelled()
    root = harness._repository_root(task)
    token = hashlib.sha256(uid.encode("utf-8")).hexdigest()[:20]
    remote_bundle = f"/tmp/rllm-final-recovery-{token}.tar"
    export_script = repository_helper_source() + r'''
import base64, sys
payload = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
try:
    result = repo_export_delta(payload)
except Exception as exc:
    result = {"ok": False, "error": str(exc), "error_type": type(exc).__name__, "phase": "repo_export_delta", "exit_code": getattr(exc, "exit_code", getattr(exc, "returncode", None))}
print(json.dumps(result, ensure_ascii=True))
'''
    chunk_script = r'''
import base64, json, sys
from pathlib import Path
payload = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
path = Path(payload["path"])
if not str(path).startswith("/tmp/rllm-final-recovery-"):
    raise RuntimeError("invalid recovery bundle path")
offset, length = int(payload["offset"]), int(payload["length"])
if offset < 0 or length <= 0 or length > 1024 * 1024:
    raise RuntimeError("invalid recovery bundle range")
with path.open("rb") as handle:
    handle.seek(offset)
    data = handle.read(length)
print(json.dumps({"ok": True, "data": base64.b64encode(data).decode("ascii")}))
'''
    cleanup_script = repository_helper_source() + r'''
import base64, json, sys
from pathlib import Path
path = Path(json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))["path"])
if not str(path).startswith("/tmp/rllm-final-recovery-"):
    raise RuntimeError("invalid recovery bundle path")
repo_unlink(path)
print(json.dumps({"ok": True}))
'''
    local_path: str | None = None
    try:
        exported = harness._codeflow_run_json_script(
            export_script,
            {
                "root": root,
                "baseline_ref": baseline_ref,
                "bundle": remote_bundle,
                "max_bytes": FINAL_RECOVERY_MAX_BUNDLE_BYTES,
            },
            primary,
            task,
            timeout=harness.command_timeout,
            operation_kind="idempotent",
        )
        _check_repository_helper(exported, "export", primary)
        size = int(exported.get("size") or -1)
        digest = str(exported.get("sha256") or "")
        present = exported.get("present")
        deleted = exported.get("deleted")
        if (
            not exported.get("ok")
            or size < 0
            or size > FINAL_RECOVERY_MAX_BUNDLE_BYTES
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not isinstance(present, list)
            or not isinstance(deleted, list)
            or any(not isinstance(path, str) for path in [*present, *deleted])
        ):
            raise RuntimeError("primary repository delta returned malformed metadata")
        with tempfile.NamedTemporaryFile(
            prefix="rllm-replay-recovery-", suffix=".tar", delete=False
        ) as local_bundle:
            local_path = local_bundle.name
            local_digest = hashlib.sha256()
            offset = 0
            while offset < size:
                check_cancelled()
                requested = min(FINAL_RECOVERY_CHUNK_BYTES, size - offset)
                result = harness._codeflow_run_json_script(
                    chunk_script,
                    {"path": remote_bundle, "offset": offset, "length": requested},
                    primary,
                    task,
                    timeout=harness.command_timeout,
                    operation_kind="read_only",
                )
                _check_repository_helper(result, "read_chunk", primary)
                try:
                    chunk = base64.b64decode(str(result["data"]), validate=True)
                except (KeyError, ValueError) as exc:
                    raise RuntimeError("malformed repository delta chunk") from exc
                if len(chunk) != requested:
                    raise RuntimeError(
                        f"short repository delta chunk at offset {offset}: "
                        f"expected={requested} observed={len(chunk)}"
                    )
                local_bundle.write(chunk)
                local_digest.update(chunk)
                offset += len(chunk)
        if local_digest.hexdigest() != digest:
            raise RuntimeError("repository delta SHA-256 mismatch")
        check_cancelled()
        observed = harness._codeflow_capture_repo_snapshot(primary, task, baseline_ref)
        if (
            observed.status != RepoStateStatus.OK
            or observed.fingerprint != expected_fingerprint
            or observed.baseline_ref != baseline_ref
        ):
            raise RuntimeError(
                "primary repository moved during replay recovery capture: "
                f"expected={expected_fingerprint} observed={observed.fingerprint}"
            )
        return _RepositoryDeltaBundle(
            local_path=local_path,
            size=size,
            sha256=digest,
            present=tuple(present),
            deleted=tuple(deleted),
            expected_fingerprint=expected_fingerprint,
            state_version=int(exported.get("state_version", 1)),
        )
    except Exception:
        if local_path is not None:
            Path(local_path).unlink(missing_ok=True)
        raise
    finally:
        if cancel_event is None or not cancel_event.is_set():
            try:
                harness._codeflow_run_json_script(
                    cleanup_script,
                    {"path": remote_bundle},
                    primary,
                    task,
                    timeout=min(float(harness.command_timeout), 60.0),
                    operation_kind="idempotent",
                )
            except Exception:
                if cancel_event is None or not cancel_event.is_set():
                    logger.warning("failed to remove primary replay recovery bundle %s", remote_bundle)


def _restore_shadow_repository_from_bundle(
    harness: Any,
    task: Task,
    shadow: Any,
    baseline_ref: str,
    bundle: _RepositoryDeltaBundle,
    *,
    uid: str,
) -> RepoSnapshot:
    """Reset a shadow to baseline and apply one authenticated handoff bundle."""

    local_path = Path(bundle.local_path)
    if local_path.stat().st_size != bundle.size:
        raise RuntimeError("local replay recovery bundle size mismatch")
    digest = hashlib.sha256()
    with local_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(FINAL_RECOVERY_CHUNK_BYTES), b""):
            digest.update(chunk)
    if digest.hexdigest() != bundle.sha256:
        raise RuntimeError("local replay recovery bundle SHA-256 mismatch")
    token = hashlib.sha256(uid.encode("utf-8")).hexdigest()[:20]
    remote_bundle = f"/tmp/rllm-final-recovery-import-{token}.tar"
    shadow.upload_file(str(local_path), remote_bundle)
    import_script = repository_helper_source() + r'''
import base64, sys
payload = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
try:
    result = repo_import_delta(payload)
except Exception as exc:
    result = {"ok": False, "error": str(exc), "error_type": type(exc).__name__, "phase": "repo_import_delta", "exit_code": getattr(exc, "exit_code", getattr(exc, "returncode", None))}
print(json.dumps(result, ensure_ascii=True))
'''
    try:
        result = harness._codeflow_run_json_script(
            import_script,
            {
                "root": harness._repository_root(task),
                "baseline_ref": baseline_ref,
                "bundle": remote_bundle,
                "max_bytes": FINAL_RECOVERY_MAX_BUNDLE_BYTES,
                "state_version": bundle.state_version,
                "present": list(bundle.present),
                "deleted": list(bundle.deleted),
            },
            shadow,
            task,
            timeout=harness.command_timeout,
            operation_kind="idempotent",
        )
        _check_repository_helper(result, "import", shadow)
    finally:
        cleanup_script = repository_helper_source() + r'''
import base64, json, sys
from pathlib import Path
path = Path(json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))["path"])
if not str(path).startswith("/tmp/rllm-final-recovery-"):
    raise RuntimeError("invalid recovery bundle path")
repo_unlink(path)
print(json.dumps({"ok": True}))
'''
        try:
            harness._codeflow_run_json_script(
                cleanup_script,
                {"path": remote_bundle},
                shadow,
                task,
                timeout=min(float(harness.command_timeout), 60.0),
                operation_kind="idempotent",
            )
        except Exception:
            logger.warning("failed to remove shadow replay recovery bundle %s", remote_bundle)
    observed = harness._codeflow_capture_repo_snapshot(shadow, task, baseline_ref)
    if (
        observed.status != RepoStateStatus.OK
        or observed.fingerprint != bundle.expected_fingerprint
        or observed.baseline_ref != baseline_ref
    ):
        raise RuntimeError(
            "shadow repository mismatch after replay recovery: "
            f"expected={bundle.expected_fingerprint} observed={observed.fingerprint}"
        )
    return observed


def _synchronize_primary_repository_to_fresh_shadow(
    harness: Any,
    task: Task,
    primary: Any,
    fresh_shadow: Any,
    baseline_ref: str,
    *,
    uid: str,
) -> str:
    """Copy the primary Git-visible delta into a pristine shadow safely.

    The sandbox protocol intentionally has no general download API.  A
    bounded tar is therefore exported from the primary and read through
    authenticated one-MiB JSON chunks before being uploaded to the new
    shadow.  Hidden verifier assets live at ``/tests`` and are never inside
    this repository-root-only bundle.
    """

    before = harness._codeflow_capture_repo_snapshot(primary, task, baseline_ref)
    if before.status != RepoStateStatus.OK or not before.fingerprint:
        raise RuntimeError("primary repository unavailable before delta export: " + str(before.error))
    bundle = _export_primary_repository_delta_bundle(
        harness, task, primary, baseline_ref, before.fingerprint, uid=uid,
    )
    try:
        restored = _restore_shadow_repository_from_bundle(
            harness, task, fresh_shadow, baseline_ref, bundle, uid=uid,
        )
        return restored.fingerprint
    finally:
        Path(bundle.local_path).unlink(missing_ok=True)


def _parameterized_node_parts(name: str) -> tuple[str, str] | None:
    function_at = name.rfind("::")
    bracket_at = name.find("[", max(0, function_at))
    if bracket_at <= function_at or not name.endswith("]"):
        return None
    return name[:bracket_at], name[bracket_at + 1 : -1]


def _is_two_parameter_reordering(expected: str, observed: str) -> bool:
    """Return whether opaque pytest ids differ only by swapping two chunks.

    Parameter values themselves may contain ``-`` (negative numbers and
    free-form ids are common), so try every exact split rather than tokenizing
    heuristically.  Callers additionally require an unambiguous one-to-one
    match within the same test function.
    """

    expected_parts = _parameterized_node_parts(expected)
    observed_parts = _parameterized_node_parts(observed)
    if expected_parts is None or observed_parts is None:
        return False
    expected_base, expected_id = expected_parts
    observed_base, observed_id = observed_parts
    if expected_base != observed_base or expected_id == observed_id:
        return False
    return any(observed_id == expected_id[index + 1 :] + "-" + expected_id[:index] for index, char in enumerate(expected_id) if char == "-" and index not in {0, len(expected_id) - 1})


def _reconcile_annotated_parameter_ids(
    observed_results: dict[str, str],
    authoritative_names: list[str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Add unambiguous aliases for reordered pytest parameter node ids."""

    reconciled = dict(observed_results)
    expected = set(authoritative_names)
    missing = sorted(expected - set(reconciled))
    extras = sorted(set(reconciled) - expected)
    candidates = {name: [extra for extra in extras if _is_two_parameter_reordering(name, extra)] for name in missing}
    reverse: dict[str, list[str]] = {}
    for name, matches in candidates.items():
        for match in matches:
            reverse.setdefault(match, []).append(name)
    aliases: dict[str, str] = {}
    for name, matches in candidates.items():
        if len(matches) != 1 or len(reverse.get(matches[0], [])) != 1:
            continue
        alias = matches[0]
        status = reconciled.get(alias)
        if status not in {"PASSED", "FAILED", "ERROR", "SKIPPED"}:
            continue
        reconciled[name] = status
        aliases[name] = alias
    return reconciled, aliases


def _metadata_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("must be a non-empty JSON object")
    parsed = json.loads(value)
    if not isinstance(parsed, dict) or not parsed:
        raise ValueError("must decode to a non-empty object")
    return parsed


def _status_map(value: Any, field: str) -> dict[str, str]:
    raw = _metadata_object(value)
    result: dict[str, str] = {}
    for name, status in raw.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{field} contains an invalid test name")
        if status not in _STATUS_VALUES:
            raise ValueError(f"{field}[{name!r}] has unsupported status {status!r}")
        result[name] = status
    return result


def _string_list(value: Any, field: str) -> list[str]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValueError(f"{field} contains an invalid test name")
        if item in result:
            raise ValueError(f"{field} contains duplicate test {item!r}")
        result.append(item)
    return result


def _test_set_contract(
    metadata: dict[str, Any],
    result_parser: str,
    target_results: dict[str, str],
) -> tuple[str, list[str]]:
    """Resolve the dataset-specific authoritative milestone test set."""

    if result_parser not in SWEREBENCH_V2_RESULT_PARSERS:
        return _EXACT_TEST_SET_POLICY, list(target_results)

    fail_to_pass = _string_list(metadata.get("FAIL_TO_PASS"), "FAIL_TO_PASS")
    pass_to_pass = _string_list(metadata.get("PASS_TO_PASS"), "PASS_TO_PASS")
    authoritative = [*fail_to_pass, *pass_to_pass]
    if not authoritative:
        raise ValueError("FAIL_TO_PASS and PASS_TO_PASS must not both be empty")
    overlap = sorted(set(fail_to_pass) & set(pass_to_pass))
    if overlap:
        raise ValueError(f"FAIL_TO_PASS overlaps PASS_TO_PASS: {overlap[:10]}")
    target_names = set(target_results)
    authoritative_names = set(authoritative)
    if authoritative_names != target_names:
        missing = sorted(target_names - authoritative_names)
        extra = sorted(authoritative_names - target_names)
        raise ValueError(f"FAIL_TO_PASS/PASS_TO_PASS union does not match target_output_json: missing={missing[:10]}, extra={extra[:10]}")
    # The runner plan and its hash are built from FAIL_TO_PASS + PASS_TO_PASS.
    # Keep every projected/count contract in that same authoritative order.
    return _ANNOTATED_SUBSET_TEST_SET_POLICY, authoritative


def _result_parser_contract(metadata: dict[str, Any]) -> tuple[str, str | None]:
    raw_parser = metadata.get("result_parser")
    if raw_parser is None or not str(raw_parser).strip():
        return R2E_RESULT_PARSER, None
    parser = str(raw_parser).strip()
    if parser not in SUPPORTED_RESULT_PARSERS:
        return parser, f"unsupported_result_parser:{parser}"
    if parser in SWEREBENCH_V2_RESULT_PARSERS:
        raw_path = metadata.get("test_results_path")
        if not isinstance(raw_path, str) or not raw_path.startswith("/tmp/rllm/"):
            return parser, "invalid_test_results_path"
        if raw_path.endswith("/") or ".." in raw_path.split("/"):
            return parser, "invalid_test_results_path"
    if parser == SWEREBENCH_V2_OFFICIAL_RESULT_PARSER:
        for field in (
            "language",
            "log_parser",
            "parser_vendor_revision",
            "parser_vendor_sha256",
        ):
            value = metadata.get(field)
            if not isinstance(value, str) or not value.strip():
                return parser, f"missing_{field}"
        vendor_sha = str(metadata["parser_vendor_sha256"])
        if not re.fullmatch(r"[0-9a-f]{64}", vendor_sha):
            return parser, "invalid_parser_vendor_sha256"
        if metadata.get("result_adapter_version") != SWEREBENCH_V2_RESULT_ADAPTER_VERSION:
            return parser, "invalid_result_adapter_version"
    return parser, None


def shadow_eligibility(task: Task, verifier_kind: str) -> tuple[bool, str]:
    """Return whether a task has a complete supported shadow contract."""
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    rllm_metadata = metadata.get("rllm") if isinstance(metadata.get("rllm"), dict) else {}
    if rllm_metadata.get("shadow_enabled") is False:
        return False, "disabled_by_task"
    if verifier_kind != "sandbox-shell":
        return False, f"unsupported_verifier:{verifier_kind}"
    profile = str(metadata.get("shadow_verifier_profile") or metadata.get("verifier_profile") or rllm_metadata.get("shadow_verifier_profile") or "")
    if profile == "denovoswe_official":
        tests_dir = task.task_dir / "tests"
        required = (
            tests_dir / "test.sh",
            tests_dir / "instance.json",
            tests_dir / "test.patch",
            tests_dir / "denovoswe_verifier.py",
        )
        if any(not path.is_file() for path in required):
            return False, "missing_denovoswe_test_assets"
        try:
            instance = json.loads((tests_dir / "instance.json").read_text())
            names = _string_list(instance.get("passed_ptp"), "passed_ptp")
            if not names:
                raise ValueError("passed_ptp is empty")
            _safe_denovo_workdir(instance.get("workdir"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return False, f"invalid_denovoswe_contract:{exc}"
        return True, "eligible"
    test_file_names = metadata.get("test_file_names")
    if not isinstance(test_file_names, list) or not test_file_names or any(not isinstance(item, str) or not item for item in test_file_names):
        return False, "missing_test_file_names"
    try:
        _status_map(metadata.get("baseline_output_json"), "baseline_output_json")
        target_results = _status_map(metadata.get("target_output_json"), "target_output_json")
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return False, f"invalid_milestone_metadata:{exc}"
    if not (task.task_dir / "tests" / "test.sh").is_file():
        return False, "missing_tests/test.sh"
    result_parser, parser_error = _result_parser_contract(metadata)
    if parser_error is not None:
        return False, parser_error
    try:
        _test_set_contract(metadata, result_parser, target_results)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return False, f"invalid_milestone_metadata:{exc}"
    return True, "eligible"


# Public compatibility alias retained for existing R2E callers and traces.
r2egym_shadow_eligibility = shadow_eligibility


def _safe_denovo_workdir(value: Any) -> str:
    raw = str(value or "")
    parts = [part for part in raw.split("/") if part]
    if not raw.startswith("/") or not parts or ".." in parts or raw == "/":
        raise ValueError("workdir must be a safe absolute repository path")
    return raw


def effective_rollout_concurrency(agent_flow: Any, requested: int) -> int:
    """Apply an optional explicit sandbox budget to rollout concurrency.

    ``requested`` is the authoritative active-rollout count. A flow may set
    ``max_concurrent`` to an integer as a compatibility safety cap expressed
    in *sandbox slots*. ``None`` means there is no additional cap, which is
    the codeflow default: ``n`` milestone rollouts may therefore hold ``2n``
    primary/shadow sandboxes.
    """
    requested = max(1, int(requested))
    slots = getattr(agent_flow, "sandbox_slots_per_rollout", 1)
    max_concurrent = getattr(agent_flow, "max_concurrent", None)
    try:
        slots = max(1, int(slots))
    except (TypeError, ValueError):
        slots = 1
    if max_concurrent is None:
        return requested
    try:
        max_concurrent = max(1, int(max_concurrent))
    except (TypeError, ValueError):
        return requested
    return max(1, min(requested, max_concurrent // slots or 1))


@dataclass
class _ReplayJob:
    # ``event`` remains the result sink in the raw Episode.  ``replay_event``
    # is a handoff-time deep copy and is never mutated by the worker, so a
    # detached shadow cannot observe later Episode enrichment mutations.
    event: dict[str, Any]
    replay_event: dict[str, Any]
    trigger_probe: bool
    probe_events: tuple[dict[str, Any], ...] = ()
    recovery_bundle: _RepositoryDeltaBundle | None = None
    recovery_capture_error: str | None = None
    recovery_capture_diagnostics: dict[str, Any] | None = None


@dataclass
class _ProbeJob:
    event: dict[str, Any]
    probe_event: dict[str, Any]
    probe_events: tuple[dict[str, Any], ...]


@dataclass
class _PendingProbeMergeMember:
    event: dict[str, Any]
    replay_job: _ReplayJob
    decision_started_at: float


@dataclass(frozen=True)
class _ParallelReplayEntry:
    """One immutable action admitted to every parallel replay lane."""

    replay_event: dict[str, Any]


@dataclass
class _ParallelProbeJob:
    """One physical verifier invocation, possibly backing a merge group."""

    sequence: int
    event: dict[str, Any]
    probe_event: dict[str, Any]
    probe_events: tuple[dict[str, Any], ...]
    target_cursor: int
    enqueued_at: float
    attempted_lane_ids: set[int]
    redispatches: int = 0
    result: dict[str, Any] | None = None


@dataclass
class _ParallelShadowLane:
    lane_id: int
    sandbox: Any
    runtime: Any
    cursor: int = 0
    healthy: bool = True
    busy: bool = False
    failure_reason: str | None = None
    current_probe_sequence: int | None = None
    worker: threading.Thread | None = None


class _ParallelProbeQueueAdapter:
    """Translate the serial scheduler's queue writes into replay journals.

    Keeping the existing :meth:`ShadowSandboxRuntime.schedule` implementation
    as the sole admission policy is important for the ``count=1`` regression
    contract and for probe-merge look-ahead.  Only the execution queue changes:
    replay actions become immutable journal entries and physical probes enter
    the shared FIFO lane queue.
    """

    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def put(self, item: Any) -> None:
        if item is _STOP:
            self.owner._parallel_seal_queue()
        elif isinstance(item, _ReplayJob):
            self.owner._parallel_accept_replay_job(item)
        elif isinstance(item, _ProbeJob):
            self.owner._parallel_accept_probe_job(item)
        else:  # pragma: no cover - protects future scheduler extensions.
            raise TypeError(f"unsupported parallel shadow queue item: {type(item)!r}")

    def qsize(self) -> int:
        return self.owner._parallel_queue_size()


@dataclass(frozen=True)
class _PartitionGroupRun:
    observation: PartitionObservation
    duration: float
    exit_code: int | None
    timed_out: bool
    log_tail: str
    restore_confirmation: RestoreConfirmation


def _changed_file_identities(value: Any) -> set[tuple[str, str, str]]:
    """Normalize serialized or modeled ChangedFile entries for replay checks."""
    if not isinstance(value, list):
        return set()
    identities: set[tuple[str, str, str]] = set()
    for item in value:
        if isinstance(item, dict):
            path = item.get("path")
            old_path = item.get("old_path")
            change_type = item.get("change_type")
        else:
            path = getattr(item, "path", None)
            old_path = getattr(item, "old_path", None)
            change_type = getattr(item, "change_type", None)
        if not isinstance(path, str) or not path:
            continue
        if hasattr(change_type, "value"):
            change_type = change_type.value
        identities.add((path, old_path if isinstance(old_path, str) else "", str(change_type or "")))
    return identities


def _changed_file_paths(value: Any) -> list[str]:
    return sorted({path for current, old, _ in _changed_file_identities(value) for path in (current, old) if path})


def _probe_merge_changed_paths(value: Any) -> frozenset[str]:
    """Return the current-path-only identity used by DeNovo probe merging."""
    if not isinstance(value, list):
        return frozenset()
    paths: set[str] = set()
    for item in value:
        path = item.get("path") if isinstance(item, dict) else getattr(item, "path", None)
        if not isinstance(path, str) or not path:
            return frozenset()
        paths.add(path)
    return frozenset(paths)


def _abort_paths_intersect_changes(
    abort_paths: Any,
    changed_paths: list[str] | None,
    repository_root: str,
) -> bool:
    """Require path-level evidence before accepting a pytest abort partial."""

    if not isinstance(abort_paths, list) or not changed_paths:
        return False
    root = repository_root.rstrip("/") + "/"

    def normalize(value: str) -> str:
        normalized = value.strip().replace("\\", "/")
        if normalized.startswith(root):
            normalized = normalized[len(root) :]
        while normalized.startswith("./"):
            normalized = normalized[2:]
        return normalized.strip("/")

    changed = {
        normalize(path)
        for path in changed_paths
        if isinstance(path, str) and normalize(path)
    }
    observed = {
        normalize(path)
        for path in abort_paths
        if isinstance(path, str) and normalize(path)
    }
    return any(
        candidate == path
        or candidate.endswith("/" + path)
        or path.endswith("/" + candidate)
        for candidate in observed
        for path in changed
    )


def _default_failure_context(failure_type: str | None) -> tuple[str | None, str | None]:
    if not failure_type:
        return None, None
    if failure_type in {"verifier_state_mismatch", "partition_state_mismatch"}:
        return "restore", "repository_restore"
    if failure_type.startswith("partition_"):
        return "partition_probe", "runner"
    if failure_type in {
        "parse_error",
        "parse_contract_error",
        "pytest_event_contract_error",
    }:
        return "parser", "parser"
    if failure_type in {"verifier_timeout", "finalize_stall_timeout"}:
        return "full_probe", "wrapper"
    if failure_type in {"verifier_crash", "resource_exhausted"}:
        return "full_probe", "grader_process"
    if failure_type in {"test_set_mismatch", "toolchain_unavailable", "dependency_network"}:
        return "full_probe", "runner"
    if failure_type in {"shadow_setup", "setup", "setup_storage", "setup_transport", "setup_provider_transient"}:
        return "setup", failure_type
    return "runtime", "sandbox"


def _setup_failure_type(error: BaseException | str) -> str:
    message = str(error)
    for explicit in (
        "setup_storage",
        "setup_transport",
        "setup_provider_transient",
    ):
        if f"[{explicit}]" in message:
            return explicit
    if re.search(r"(?:no space left on device|\bENOSPC\b)", message, flags=re.IGNORECASE):
        return "setup_storage"
    if (
        re.search(r"sandbox", message, flags=re.IGNORECASE)
        and re.search(r"(?:create|provision|ready)", message, flags=re.IGNORECASE)
        and re.search(
            r"(?:timed?\s*out|deadline exceeded|HTTP\s*(?:429|5\d\d)|"
            r"too many requests|service unavailable)",
            message,
            flags=re.IGNORECASE,
        )
    ):
        return "setup_provider_transient"
    if re.search(
        r"(?:failed to upload|upload (?:file|dir|failed|failure)|"
        r"transport (?:error|failure)|connection reset|broken pipe)",
        message,
        flags=re.IGNORECASE,
    ):
        return "setup_transport"
    return "setup"


_PARSE_R2E_LOG_SCRIPT = r"""
import base64
import json
import re
import sys

payload = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
path = payload["path"]
tail_chars = int(payload.get("tail_chars", 2000))
expected_tests = set(payload.get("expected_tests") or [])
ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

def strip_reason(value):
    depth = 0
    for index, char in enumerate(value):
        if depth == 0 and value.startswith(" - ", index):
            return value[:index]
        if char == "[":
            depth += 1
        elif char == "]" and depth:
            depth -= 1
    return value

try:
    log = open(path, encoding="utf-8", errors="replace").read()
except Exception as exc:
    print(json.dumps({"ok": False, "error": f"reading test output: {exc}"}))
    raise SystemExit(0)

clean_log = ansi.sub("", log)
has_summary = "short test summary info" in clean_log
collection_aborted = bool(re.search(r"Interrupted:\s+\d+\s+errors?\s+during collection", clean_log, flags=re.IGNORECASE))
unsafe_crash = bool(re.search(r"(?:Segmentation fault|\bAborted\b|\bKilled\b|core dumped)", clean_log, flags=re.IGNORECASE))
pytest_config_error = bool(
    re.search(r"ERROR:\s+usage:.*?(?:pytest|__main__\.py)", clean_log, flags=re.IGNORECASE | re.DOTALL)
    or re.search(r"(?:pytest|__main__\.py):\s+error:", clean_log, flags=re.IGNORECASE)
    or re.search(
        r"ERROR:\s+[^\n]*(?:pyproject\.toml|setup\.cfg|pytest\.ini|tox\.ini):\s*[^\n]+",
        clean_log,
        flags=re.IGNORECASE,
    )
)
module_import_error = bool(
    re.search(
        r"Failed to import test module|ImportError while (?:importing test module|loading conftest)|ERROR collecting|collected 0 items? / \d+ errors?",
        clean_log,
        flags=re.IGNORECASE,
    )
    and re.search(r"(?:SyntaxError|IndentationError|TabError|ImportError|ModuleNotFoundError)", clean_log)
)
results = {}
collection_errors = []
summary = log.split("short test summary info", 1)[1] if has_summary else ""
for raw_line in summary.splitlines():
    line = ansi.sub("", raw_line).strip()
    match = re.match(r"^(PASSED|FAILED|ERROR)\s+(.+)$", line)
    if not match:
        continue
    status, test_spec = match.groups()
    if status in {"FAILED", "ERROR"}:
        test_spec = strip_reason(test_spec)
    parts = test_spec.split("::")
    if status == "ERROR" and len(parts) == 1:
        collection_errors.append(test_spec.strip())
        continue
    test_name = ".".join(parts[1:]) if len(parts) > 1 else parts[0]
    test_name = test_name.strip()
    if test_name or status == "ERROR":
        results[test_name] = status

expected_results = {name: status for name, status in results.items() if name in expected_tests}
pseudo_import_results = bool(results) and all(name.startswith("_FailedTest.") for name in results)
abort_kind = None
if not unsafe_crash and not expected_results:
    if pytest_config_error and not has_summary:
        abort_kind = "pytest_config_error"
    elif module_import_error and (pseudo_import_results or not results):
        abort_kind = "test_module_import_error"
    elif collection_aborted and collection_errors and not results:
        abort_kind = "collection_error"

if abort_kind is not None:
    print(json.dumps({
        "ok": True,
        "collection_aborted": True,
        "abort_kind": abort_kind,
        "collection_errors": collection_errors,
        "test_results": {},
        "log_tail": log[-tail_chars:],
    }, ensure_ascii=False))
elif not has_summary:
    print(json.dumps({"ok": False, "error": "pytest short summary is missing", "log_tail": log[-tail_chars:]}))
elif not results:
    print(json.dumps({"ok": False, "error": "pytest summary contained no test states", "log_tail": log[-tail_chars:]}))
else:
    print(json.dumps({"ok": True, "test_results": results, "log_tail": log[-tail_chars:]}, ensure_ascii=False))
"""


_SWEREBENCH_PYTEST_PLUGIN_SCRIPT = r'''
"""Shadow-only pytest result collector injected through PYTEST_PLUGINS."""

import json
import os
from pathlib import Path

RESULTS_PATH = Path(
    os.environ.get(
        "RLLM_SHADOW_TEST_RESULTS_JSON",
        "/tmp/rllm/shadow_test_results.json",
    )
)
EVENTS_PATH = Path(
    os.environ.get(
        "RLLM_SHADOW_TEST_EVENTS_JSONL",
        "/tmp/rllm/shadow_test_events.jsonl",
    )
)
PARSER = "swerebench_v2_pytest_v1"
PLAN_HASH = os.environ.get("RLLM_SHADOW_COUNT_PLAN_HASH", "legacy")
COMMAND_ID = os.environ.get("RLLM_SHADOW_COUNT_COMMAND_ID", "command-0")
_results = {}
_worker_process = False
_worker_id = "controller"
_owner_pid = int(os.environ.setdefault("RLLM_SHADOW_PYTEST_OWNER_PID", str(os.getpid())))
_owner_process = os.getpid() == _owner_pid


def _is_worker(config):
    return hasattr(config, "workerinput")


def pytest_configure(config):
    global _worker_id, _worker_process
    _worker_process = _is_worker(config)
    worker_input = getattr(config, "workerinput", {})
    _worker_id = str(
        worker_input.get("workerid")
        if isinstance(worker_input, dict) and worker_input.get("workerid")
        else os.environ.get("PYTEST_XDIST_WORKER") or "controller"
    )


def _append_event(value):
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # One O_APPEND write keeps each JSONL record intact when nested pytest
    # subprocesses report concurrently.  The parser already tolerates a final
    # partial record if a process is killed mid-write.
    event = dict(value)
    event.setdefault("pid", os.getpid())
    event.setdefault("worker_id", _worker_id)
    event.setdefault("plan_hash", PLAN_HASH)
    event.setdefault("command_id", COMMAND_ID)
    record = (json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    descriptor = os.open(str(EVENTS_PATH), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, record)
    finally:
        os.close(descriptor)


_append_event({"schema_version": 2, "event": "collector_start"})


def _atomic_results():
    if not _owner_process:
        return
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged = {}
    try:
        previous = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        if isinstance(previous, dict) and isinstance(previous.get("test_results"), dict):
            merged.update(previous["test_results"])
    except Exception:
        pass
    merged.update(_results)
    temporary = RESULTS_PATH.with_suffix(RESULTS_PATH.suffix + ".%s.tmp" % os.getpid())
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "parser": PARSER,
                "test_results": merged,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    os.replace(str(temporary), str(RESULTS_PATH))


def pytest_sessionstart(session):
    _append_event({"schema_version": 2, "event": "session_start"})


def pytest_collectreport(report):
    if not getattr(report, "failed", False):
        return
    _append_event(
        {
            "schema_version": 2,
            "event": "collection_error",
            "nodeid": str(getattr(report, "nodeid", "") or ""),
            "message": str(getattr(report, "longrepr", "") or "")[-4000:],
        }
    )


def pytest_internalerror(excrepr, excinfo):
    del excinfo
    _append_event(
        {
            "schema_version": 2,
            "event": "internal_error",
            "message": str(excrepr)[-4000:],
        }
    )


def pytest_runtest_logreport(report):
    status = None
    if report.when == "call":
        if report.skipped:
            status = "SKIPPED"
        elif report.failed:
            status = "FAILED"
        elif report.passed:
            status = "PASSED"
    elif report.failed:
        status = "ERROR"
    elif report.when == "setup" and report.skipped:
        status = "SKIPPED"
    if status is None:
        return
    existing = _results.get(report.nodeid)
    priority = {"PASSED": 0, "SKIPPED": 1, "FAILED": 2, "ERROR": 3}
    if existing is None or priority[status] >= priority.get(existing, -1):
        _results[report.nodeid] = status
    _append_event(
        {
            "schema_version": 2,
            "event": "test_result",
            "nodeid": report.nodeid,
            "status": _results[report.nodeid],
        }
    )


def pytest_sessionfinish(session, exitstatus):
    _append_event(
        {
            "schema_version": 2,
            "event": "session_finish",
            "exitstatus": int(exitstatus),
        }
    )
    if _owner_process and not _is_worker(session.config):
        _atomic_results()


def pytest_unconfigure(config):
    if _owner_process and not _is_worker(config):
        # A third-party hookwrapper may raise during sessionfinish teardown.
        # Persist once more from the final unconfigure hook when reachable.
        _atomic_results()
'''


_PARSE_SWEREBENCH_V2_RESULTS_SCRIPT = r"""
import base64
import json
import re
import sys

argument = sys.argv[1]
if argument.startswith("/"):
    payload = json.load(open(argument, encoding="utf-8"))
else:
    payload = json.loads(base64.b64decode(argument).decode("utf-8"))
if payload.get("schema_version", 1) != 1:
    print(json.dumps({"ok": False, "error": "unsupported parser contract schema", "deterministic": True}))
    raise SystemExit(0)
results_path = payload["results_path"]
output_path = payload["output_path"]
fallback_results_path = payload.get("fallback_results_path")
events_path = payload.get("events_path")
pytest_event_contract = payload.get("pytest_event_contract")
expected_parser = payload["expected_parser"]
expected_log_parser = payload.get("expected_log_parser")
expected_language = payload.get("expected_language")
expected_result_adapter_version = payload.get("expected_result_adapter_version")
expected_test_order = list(payload.get("expected_tests") or [])
expected_tests = set(expected_test_order)
compact = bool(payload.get("compact"))
tail_chars = int(payload.get("tail_chars", 2000))
ansi = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
timing_patterns = (
    re.compile(r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$", re.IGNORECASE),
    re.compile(r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b", re.IGNORECASE),
    re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$", re.IGNORECASE),
)
allowed_statuses = {"PASSED", "FAILED", "ERROR", "SKIPPED"}
status_priority = {"PASSED": 0, "SKIPPED": 1, "FAILED": 2, "ERROR": 3}

def merge_result(target, name, status):
    existing = target.get(name)
    if existing is None:
        target[name] = status
    elif existing in status_priority and status in status_priority:
        if status_priority[status] > status_priority[existing]:
            target[name] = status

def normalize(name):
    for pattern in timing_patterns:
        name = pattern.sub("", name)
    return name.strip()

expected_test_order = [str(name).strip() for name in expected_test_order]
if any(not name for name in expected_test_order) or len(expected_test_order) != len(set(expected_test_order)):
    print(json.dumps({"ok": False, "error": "authoritative test ids are invalid", "deterministic": True}))
    raise SystemExit(0)
expected_tests = set(expected_test_order)
expected_by_normalized = {}
for expected_name in expected_test_order:
    expected_by_normalized.setdefault(normalize(expected_name), []).append(expected_name)

def canonical_name(raw_name):
    raw_name = raw_name.strip()
    if raw_name in expected_tests:
        return raw_name
    name = normalize(raw_name)
    aliases = expected_by_normalized.get(name, [])
    return aliases[0] if len(aliases) == 1 else name

try:
    log = open(output_path, encoding="utf-8", errors="replace").read()
except Exception:
    log = ""

def pytest_event_contract_error(
    error,
    *,
    line_index=None,
    event_kind=None,
    observed_schema=None,
    observed_plan_hash=None,
    observed_command_id=None,
):
    expected_schema = None
    expected_plan_hash = None
    expected_command_ids = []
    if isinstance(pytest_event_contract, dict):
        expected_schema = pytest_event_contract.get("schema_version")
        expected_plan_hash = pytest_event_contract.get("plan_hash")
        raw_command_ids = pytest_event_contract.get("command_ids")
        if isinstance(raw_command_ids, list):
            expected_command_ids = [
                str(value)[:100]
                for value in raw_command_ids[:20]
            ]
    print(json.dumps({
        "ok": False,
        "error": error,
        "failure_type": "pytest_event_contract_error",
        "failure_stage": "parser",
        "failure_origin": "pytest_event_stream",
        "failure_evidence": {
            "line_index": line_index,
            "event_kind": str(event_kind)[:100] if event_kind is not None else None,
            "expected_schema": expected_schema,
            "observed_schema": observed_schema,
            "expected_plan_hash": expected_plan_hash,
            "observed_plan_hash": str(observed_plan_hash)[:100] if observed_plan_hash is not None else None,
            "expected_command_ids": expected_command_ids,
            "observed_command_id": str(observed_command_id)[:100] if observed_command_id is not None else None,
        },
        "deterministic": True,
        "log_tail": log[-tail_chars:],
    }, ensure_ascii=False))
    raise SystemExit(0)

strict_pytest_events = pytest_event_contract is not None
expected_pytest_event_schema = None
expected_pytest_event_plan_hash = None
expected_pytest_event_command_ids = set()
if strict_pytest_events:
    if not isinstance(pytest_event_contract, dict):
        pytest_event_contract_error("pytest event contract must be an object")
    expected_pytest_event_schema = pytest_event_contract.get("schema_version")
    expected_pytest_event_plan_hash = pytest_event_contract.get("plan_hash")
    raw_command_ids = pytest_event_contract.get("command_ids")
    if (
        expected_pytest_event_schema != 2
        or not isinstance(expected_pytest_event_plan_hash, str)
        or not expected_pytest_event_plan_hash
        or not isinstance(raw_command_ids, list)
        or not raw_command_ids
        or any(not isinstance(value, str) or not value for value in raw_command_ids)
        or len(raw_command_ids) != len(set(raw_command_ids))
    ):
        pytest_event_contract_error("pytest event contract is invalid")
    expected_pytest_event_command_ids = set(raw_command_ids)

primary_artifact_error = None
try:
    artifact = json.load(open(results_path, encoding="utf-8"))
except Exception as exc:
    artifact = None
    primary_artifact_error = exc

raw_results = {}
artifact_complete = None
artifact_log_parser = None
artifact_language = None
artifact_schema = None
artifact_execution = {
    "timed_out": False,
    "resource_exhausted": False,
    "commands": [],
}
if artifact is not None:
    if not isinstance(artifact, dict):
        print(json.dumps({"ok": False, "error": "structured test results must be an object", "log_tail": log[-tail_chars:]}))
        raise SystemExit(0)
    artifact_schema = artifact.get("schema_version")
    if type(artifact_schema) is not int or artifact_schema not in {1, 2, 3}:
        print(json.dumps({"ok": False, "error": "unsupported structured test results schema", "log_tail": log[-tail_chars:]}))
        raise SystemExit(0)
    if expected_parser == "swerebench_v2_official_v2" and artifact_schema != 3:
        print(json.dumps({"ok": False, "error": "official v2 adapter v6 requires structured test results schema 3", "deterministic": True, "log_tail": log[-tail_chars:]}))
        raise SystemExit(0)
    if (
        expected_parser == "swerebench_v2_official_v2"
        and (
            expected_result_adapter_version != 6
            or artifact.get("result_adapter_version") != expected_result_adapter_version
        )
    ):
        print(json.dumps({"ok": False, "error": "structured test results adapter version mismatch", "deterministic": True, "log_tail": log[-tail_chars:]}))
        raise SystemExit(0)
    if artifact.get("parser") != expected_parser:
        print(json.dumps({"ok": False, "error": "structured test results parser mismatch", "log_tail": log[-tail_chars:]}))
        raise SystemExit(0)
    artifact_complete = artifact.get("complete") if artifact_schema in {2, 3} else None
    artifact_log_parser = artifact.get("log_parser")
    artifact_language = artifact.get("language")
    if artifact_schema in {2, 3}:
        if type(artifact_complete) is not bool:
            print(json.dumps({"ok": False, "error": "structured test results complete flag is invalid", "deterministic": True, "log_tail": log[-tail_chars:]}))
            raise SystemExit(0)
        if expected_log_parser and artifact_log_parser != expected_log_parser:
            print(json.dumps({"ok": False, "error": "structured test results log parser mismatch", "deterministic": True, "log_tail": log[-tail_chars:]}))
            raise SystemExit(0)
        if expected_language and artifact_language != expected_language:
            print(json.dumps({"ok": False, "error": "structured test results language mismatch", "deterministic": True, "log_tail": log[-tail_chars:]}))
            raise SystemExit(0)
    if artifact_schema == 3:
        artifact_execution = artifact.get("execution")
        if not isinstance(artifact_execution, dict):
            print(json.dumps({"ok": False, "error": "structured execution evidence must be an object", "deterministic": True, "log_tail": log[-tail_chars:]}))
            raise SystemExit(0)
        commands = artifact_execution.get("commands")
        if (
            type(artifact_execution.get("timed_out")) is not bool
            or type(artifact_execution.get("resource_exhausted", False)) is not bool
            or not isinstance(commands, list)
        ):
            print(json.dumps({"ok": False, "error": "structured execution evidence is invalid", "deterministic": True, "log_tail": log[-tail_chars:]}))
            raise SystemExit(0)
        for command in commands:
            if (
                not isinstance(command, dict)
                or type(command.get("index")) is not int
                or type(command.get("exit_code")) is not int
                or type(command.get("timed_out")) is not bool
                or (command.get("signal") is not None and type(command.get("signal")) is not int)
                or (
                    command.get("oom_kill_delta") is not None
                    and (
                        type(command.get("oom_kill_delta")) is not int
                        or command.get("oom_kill_delta") < 0
                    )
                )
            ):
                print(json.dumps({"ok": False, "error": "structured command execution evidence is invalid", "deterministic": True, "log_tail": log[-tail_chars:]}))
                raise SystemExit(0)
    raw_results = artifact.get("test_results")
    if not isinstance(raw_results, dict):
        print(json.dumps({"ok": False, "error": "structured test_results must be an object", "log_tail": log[-tail_chars:]}))
        raise SystemExit(0)

result_source = (
    "official_log_parser"
    if expected_parser in {"swerebench_v2_official_v1", "swerebench_v2_official_v2"}
    else "pytest_summary"
)
collector_active = False
fallback_results = {}
pytest_collection_error = False
pytest_internal_error = False
if isinstance(fallback_results_path, str):
    try:
        fallback_artifact = json.load(open(fallback_results_path, encoding="utf-8"))
    except FileNotFoundError:
        fallback_artifact = None
    except Exception as exc:
        print(json.dumps({"ok": False, "error": "reading shadow pytest results: %s" % exc, "log_tail": log[-tail_chars:]}))
        raise SystemExit(0)
    if fallback_artifact is not None:
        if (
            not isinstance(fallback_artifact, dict)
            or fallback_artifact.get("schema_version") != 1
            or fallback_artifact.get("parser") != expected_parser
            or not isinstance(fallback_artifact.get("test_results"), dict)
        ):
            print(json.dumps({"ok": False, "error": "invalid shadow pytest results artifact", "deterministic": True, "log_tail": log[-tail_chars:]}))
            raise SystemExit(0)
        for name, status in fallback_artifact["test_results"].items():
            merge_result(fallback_results, name, status)
        collector_active = True

if isinstance(events_path, str):
    try:
        event_lines = open(events_path, encoding="utf-8", errors="replace").read().splitlines()
    except FileNotFoundError:
        event_lines = []
    except Exception as exc:
        print(json.dumps({"ok": False, "error": "reading shadow pytest event stream: %s" % exc, "log_tail": log[-tail_chars:]}))
        raise SystemExit(0)
    for index, line in enumerate(event_lines):
        try:
            event = json.loads(line)
        except Exception:
            if index == len(event_lines) - 1:
                # A verifier killed during append may leave only its final
                # JSONL record incomplete. Earlier complete records remain
                # authoritative.
                continue
            if strict_pytest_events:
                pytest_event_contract_error(
                    "malformed shadow pytest event stream",
                    line_index=index,
                )
            print(json.dumps({"ok": False, "error": "malformed shadow pytest event stream", "deterministic": True, "log_tail": log[-tail_chars:]}))
            raise SystemExit(0)
        if strict_pytest_events:
            if not isinstance(event, dict):
                pytest_event_contract_error(
                    "pytest event must be an object",
                    line_index=index,
                )
            kind = event.get("event")
            observed_schema = event.get("schema_version")
            observed_plan_hash = event.get("plan_hash")
            observed_command_id = event.get("command_id")
            allowed_kinds = {
                "collector_start",
                "session_start",
                "collection_error",
                "internal_error",
                "test_result",
                "session_finish",
            }
            if (
                observed_schema != expected_pytest_event_schema
                or not isinstance(kind, str)
                or kind not in allowed_kinds
                or type(event.get("pid")) is not int
                or event.get("pid") <= 0
                or not isinstance(event.get("worker_id"), str)
                or not event.get("worker_id")
                or observed_plan_hash != expected_pytest_event_plan_hash
                or observed_command_id not in expected_pytest_event_command_ids
            ):
                pytest_event_contract_error(
                    "invalid shadow pytest event contract",
                    line_index=index,
                    event_kind=kind,
                    observed_schema=observed_schema,
                    observed_plan_hash=observed_plan_hash,
                    observed_command_id=observed_command_id,
                )
            if kind == "test_result":
                name = event.get("nodeid")
                status = event.get("status")
                if not isinstance(name, str) or not name or status not in allowed_statuses:
                    pytest_event_contract_error(
                        "invalid shadow pytest test event",
                        line_index=index,
                        event_kind=kind,
                        observed_schema=observed_schema,
                        observed_plan_hash=observed_plan_hash,
                        observed_command_id=observed_command_id,
                    )
                merge_result(fallback_results, name, status)
            elif kind == "session_finish" and type(event.get("exitstatus")) is not int:
                pytest_event_contract_error(
                    "invalid shadow pytest session finish event",
                    line_index=index,
                    event_kind=kind,
                    observed_schema=observed_schema,
                    observed_plan_hash=observed_plan_hash,
                    observed_command_id=observed_command_id,
                )
            elif kind == "collection_error":
                if not isinstance(event.get("nodeid"), str) or not isinstance(event.get("message"), str):
                    pytest_event_contract_error(
                        "invalid shadow pytest collection error event",
                        line_index=index,
                        event_kind=kind,
                        observed_schema=observed_schema,
                        observed_plan_hash=observed_plan_hash,
                        observed_command_id=observed_command_id,
                    )
                pytest_collection_error = True
            elif kind == "internal_error":
                if not isinstance(event.get("message"), str):
                    pytest_event_contract_error(
                        "invalid shadow pytest internal error event",
                        line_index=index,
                        event_kind=kind,
                        observed_schema=observed_schema,
                        observed_plan_hash=observed_plan_hash,
                        observed_command_id=observed_command_id,
                    )
                pytest_internal_error = True
            collector_active = True
            continue
        if not isinstance(event, dict) or event.get("schema_version") != 1:
            print(json.dumps({"ok": False, "error": "invalid shadow pytest event", "deterministic": True, "log_tail": log[-tail_chars:]}))
            raise SystemExit(0)
        if event.get("event") in {"session_start", "session_finish"}:
            collector_active = True
        elif event.get("event") == "test_result":
            name = event.get("nodeid")
            status = event.get("status")
            if not isinstance(name, str) or status not in allowed_statuses:
                print(json.dumps({"ok": False, "error": "invalid shadow pytest test event", "deterministic": True, "log_tail": log[-tail_chars:]}))
                raise SystemExit(0)
            merge_result(fallback_results, name, status)
            collector_active = True

if fallback_results:
    # Hook reports are structured and preserve full node ids, so they take
    # precedence over the grader's lossy text-summary parser on conflicts.
    raw_results = dict(raw_results)
    for name, status in fallback_results.items():
        merge_result(raw_results, name, status)
    result_source = "pytest_plugin"
elif primary_artifact_error is not None and not collector_active:
    print(json.dumps({
        "ok": False,
        "error": "reading structured test results: %s" % primary_artifact_error,
        "collector_active": False,
        "log_tail": log[-tail_chars:],
    }))
    raise SystemExit(0)

results = {}
ignored_invalid_extra_tests = []
for raw_name, raw_status in raw_results.items():
    if not isinstance(raw_name, str) or not raw_name.strip():
        print(json.dumps({"ok": False, "error": "structured test results contain an invalid test name", "log_tail": log[-tail_chars:]}))
        raise SystemExit(0)
    name = canonical_name(raw_name)
    if not name:
        print(json.dumps({"ok": False, "error": "structured test name is empty after normalization", "log_tail": log[-tail_chars:]}))
        raise SystemExit(0)
    if raw_status not in allowed_statuses:
        # The annotated SWE-rebench contract is authoritative. Third-party
        # plugins sometimes add diagnostic pseudo-tests with project-specific
        # status strings; an invalid status outside that contract must not
        # poison an otherwise complete authoritative result vector. Invalid
        # statuses for authoritative nodes remain fail-closed.
        if name not in expected_tests:
            ignored_invalid_extra_tests.append(name)
            continue
        print(json.dumps({
            "ok": False,
            "error": "structured test results contain an invalid status for authoritative test: %s=%r" % (name, raw_status),
            "log_tail": log[-tail_chars:],
        }))
        raise SystemExit(0)
    if name in results:
        print(json.dumps({"ok": False, "error": "duplicate structured test name after normalization: %s" % name, "log_tail": log[-tail_chars:]}))
        raise SystemExit(0)
    results[name] = raw_status

clean_log = ansi.sub("", log)
command_executions = artifact_execution.get("commands") or []
command_exit_codes = [command["exit_code"] for command in command_executions]
command_signals = [command["signal"] for command in command_executions if command.get("signal") is not None]
execution_failed = any(code != 0 for code in command_exit_codes)
execution_timed_out = bool(artifact_execution.get("timed_out")) or any(
    command.get("timed_out") is True for command in command_executions
)
execution_evidence_payload = {
    "artifact_schema": artifact_schema,
    "command_exit_codes": command_exit_codes,
    "command_signals": command_signals,
    "resource_exhausted": bool(artifact_execution.get("resource_exhausted")),
    "oom_kill_deltas": [
        command.get("oom_kill_delta", 0)
        for command in command_executions
    ],
}
observed_expected_tests = {name for name in results if name in expected_tests}
for raw_line in clean_log.splitlines():
    match = re.match(r"^(?:PASSED|FAILED|ERROR|SKIPPED)\s+(\S+)", raw_line.strip())
    if match:
        name = canonical_name(match.group(1))
        if name in expected_tests:
            observed_expected_tests.add(name)
unsafe_crash = bool(
    artifact_complete is False
    and (
        any(signal in {6, 9, 11} for signal in command_signals)
        or any(code in {134, 139} for code in command_exit_codes)
    )
)
dependency_network_error = bool(
    artifact_complete is False
    and execution_failed
    and re.search(
        r"(?:Could not resolve host|Temporary failure in name resolution|UnknownHostException|"
        r"SocketTimeoutException|connect timed out|Connection timed out|"
        r"ECONNRESET|ETIMEDOUT|network is unreachable|TLS handshake timeout|"
        r"Could not transfer artifact|Could not resolve dependencies|"
        r"Could not get resource|Could not install Gradle distribution|"
        r"failed to (?:fetch|download)|npm ERR!.*network)",
        clean_log,
        flags=re.IGNORECASE,
    )
)
resource_exhausted = bool(
    artifact_complete is False
    and artifact_execution.get("resource_exhausted") is True
)
toolchain_unavailable = bool(
    artifact_complete is False
    and execution_failed
    and re.search(
        r"(?:UnsupportedClassVersionError|invalid target release|release version \d+ not supported|"
        r"Detected JDK version[^\n]*not in (?:the )?allowed range|"
        r"Required Java version[^\n]*not met|JAVA_HOME is set to an invalid directory|"
        r"No matching toolchains found for requested specification)",
        clean_log,
        flags=re.IGNORECASE,
    )
)
deterministic_runner_failure = bool(re.search(
    r"(?:compilation failed|compile error|failed to compile|BUILD FAILED|"
    r"cannot find symbol|unresolved reference|syntax error|collection error|"
    r"no tests? (?:found|were found)|test command failed before completion)",
    clean_log,
    flags=re.IGNORECASE,
))
diagnostic_timeout = bool(
    artifact_complete is False
    and execution_timed_out
)
pytest_config_error = bool(
    re.search(r"ERROR:\s+usage:.*?(?:pytest|__main__\.py)", clean_log, flags=re.IGNORECASE | re.DOTALL)
    or re.search(r"(?:pytest|__main__\.py):\s+error:", clean_log, flags=re.IGNORECASE)
    or re.search(
        r"ERROR:\s+[^\n]*(?:pyproject\.toml|setup\.cfg|pytest\.ini|tox\.ini):\s*[^\n]+",
        clean_log,
        flags=re.IGNORECASE,
    )
)
collection_aborted = bool(re.search(r"Interrupted:\s+\d+\s+errors?\s+during collection", clean_log, flags=re.IGNORECASE))
module_import_error = bool(
    re.search(
        r"Failed to import test module|ImportError while (?:importing test module|loading conftest)|ERROR collecting|collected 0 items? / \d+ errors?",
        clean_log,
        flags=re.IGNORECASE,
    )
    and re.search(r"(?:SyntaxError|IndentationError|TabError|ImportError|ModuleNotFoundError)", clean_log)
)
# Repository modules can be imported as pytest entry points. If an edit breaks
# one of those modules, pytest dies while constructing its plugin manager,
# before it emits the usual ``ERROR collecting`` banner or reaches our
# ``pytest_sessionstart`` hook. A Python traceback ending in an import/syntax
# failure, with no observed authoritative node and no unsafe process crash,
# still proves that no authoritative test executed.
pytest_startup_import_error = bool(
    re.search(r"Traceback \(most recent call last\):", clean_log)
    and re.search(
        r"(?:SyntaxError|IndentationError|TabError|ImportError|ModuleNotFoundError):",
        clean_log,
    )
)
abort_kind = None
if not unsafe_crash and not observed_expected_tests:
    if pytest_config_error:
        abort_kind = "pytest_config_error"
    elif module_import_error or pytest_startup_import_error:
        abort_kind = "test_module_import_error"
    elif collection_aborted or pytest_collection_error:
        abort_kind = "collection_error"

def emit_results():
    extra_results = {
        name: status for name, status in results.items() if name not in expected_tests
    }
    if compact:
        print(json.dumps({
            "schema_version": 1,
            "ok": True,
            "statuses": [results.get(name) for name in expected_test_order],
            "extra_tests": sorted(extra_results),
            "extra_results": extra_results,
            "result_source": result_source,
            "artifact_complete": artifact_complete,
            "log_parser": artifact_log_parser,
            "language": artifact_language,
            "collector_active": collector_active,
            "execution_evidence": bool(results) or bool(command_executions),
            "execution": execution_evidence_payload,
            "observed_expected_count": len(observed_expected_tests),
            "partial_failure_type": (
                "dependency_network" if dependency_network_error else
                "resource_exhausted" if resource_exhausted else
                "toolchain_unavailable" if toolchain_unavailable else
                "deterministic_runner_partial" if deterministic_runner_failure or artifact_complete is False else None
            ),
            "ignored_invalid_extra_tests": sorted(ignored_invalid_extra_tests),
            "log_tail": log[-tail_chars:],
        }, ensure_ascii=False, separators=(",", ":")))
    else:
        print(json.dumps({
            "ok": True,
            "test_results": results,
            "result_source": result_source,
            "artifact_complete": artifact_complete,
            "log_parser": artifact_log_parser,
            "language": artifact_language,
            "collector_active": collector_active,
            "execution_evidence": bool(results) or bool(command_executions),
            "execution": execution_evidence_payload,
            "observed_expected_count": len(observed_expected_tests),
            "partial_failure_type": (
                "dependency_network" if dependency_network_error else
                "resource_exhausted" if resource_exhausted else
                "toolchain_unavailable" if toolchain_unavailable else
                "deterministic_runner_partial" if deterministic_runner_failure or artifact_complete is False else None
            ),
            "ignored_invalid_extra_tests": sorted(ignored_invalid_extra_tests),
            "log_tail": log[-tail_chars:],
        }, ensure_ascii=False))

if dependency_network_error:
    print(json.dumps({
        "ok": False,
        "error": "dependency download failed through configured sandbox proxy",
        "failure_type": "dependency_network",
        "failure_stage": "full_probe",
        "failure_origin": "runner",
        "failure_evidence": execution_evidence_payload,
        "collector_active": collector_active,
        "log_tail": log[-tail_chars:],
    }, ensure_ascii=False))
elif resource_exhausted:
    print(json.dumps({
        "ok": False,
        "error": "verifier process exhausted sandbox resources",
        "failure_type": "resource_exhausted",
        "failure_stage": "full_probe",
        "failure_origin": "grader_process",
        "failure_evidence": execution_evidence_payload,
        "collector_active": collector_active,
        "log_tail": log[-tail_chars:],
    }, ensure_ascii=False))
elif toolchain_unavailable:
    print(json.dumps({
        "ok": False,
        "error": "sandbox image toolchain does not satisfy the task contract",
        "failure_type": "toolchain_unavailable",
        "failure_stage": "full_probe",
        "failure_origin": "runner",
        "failure_evidence": execution_evidence_payload,
        "collector_active": collector_active,
        "log_tail": log[-tail_chars:],
    }, ensure_ascii=False))
elif diagnostic_timeout:
    print(json.dumps({
        "ok": False,
        "error": "verifier reported an internal timeout before producing a trustworthy result vector",
        "failure_type": "verifier_timeout",
        "failure_stage": "full_probe",
        "failure_origin": "grader_process",
        "failure_evidence": execution_evidence_payload,
        "timed_out": True,
        "collector_active": collector_active,
        "log_tail": log[-tail_chars:],
    }, ensure_ascii=False))
elif unsafe_crash:
    print(json.dumps({
        "ok": False,
        "error": "verifier process crashed before producing a trustworthy result vector",
        "failure_type": "verifier_crash",
        "failure_stage": "full_probe",
        "failure_origin": "grader_process",
        "failure_evidence": execution_evidence_payload,
        "collector_active": collector_active,
        "log_tail": log[-tail_chars:],
    }, ensure_ascii=False))
elif abort_kind is not None:
    print(json.dumps({
        "schema_version": 1,
        "ok": True,
        "collection_aborted": True,
        "abort_kind": abort_kind,
        "test_results": {},
        "execution_evidence": bool(command_executions),
        "execution": execution_evidence_payload,
        "collector_active": collector_active,
        "log_tail": log[-tail_chars:],
    }, ensure_ascii=False))
elif results:
    emit_results()
elif expected_parser == "swerebench_v2_official_v2" and artifact_complete is not True and deterministic_runner_failure:
    # A deterministic build/compile/runner failure proves that the command
    # ran but never reached authoritative tests.  The host decides whether
    # missing ids may be filled from the baseline contract or as NOT_RUN.
    emit_results()
elif expected_parser == "swerebench_v2_official_v2" and artifact is not None and artifact_complete is False:
    # Preserve an adapter-authenticated incomplete vector. The host may use it
    # only under a stricter dataset contract; it must not be collapsed into a
    # generic parser failure merely because no individual test ran.
    emit_results()
else:
    print(json.dumps({
        "ok": False,
        "error": "structured test results are empty without a proven full collection abort",
        "failure_evidence": {
            "pytest_collection_error": pytest_collection_error,
            "pytest_internal_error": pytest_internal_error,
        },
        "collector_active": collector_active,
        "log_tail": log[-tail_chars:],
    }, ensure_ascii=False))
"""


class ShadowSandboxRuntime:
    """Serial action replay and dataset-aware target-test probes."""

    def __init__(self, harness: Any, task: Task, primary: Any, shadow: Any, uid: str):
        self.harness = harness
        self.task = task
        self.primary = primary
        self.shadow = shadow
        self.uid = uid
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        rllm_metadata = metadata.get("rllm") if isinstance(metadata.get("rllm"), dict) else {}
        self.language = str(metadata.get("language") or "")
        self.log_parser = str(metadata.get("log_parser") or "")
        self.parser_vendor_revision = str(metadata.get("parser_vendor_revision") or "")
        self.parser_vendor_sha256 = str(metadata.get("parser_vendor_sha256") or "")
        self.verifier_profile = str(metadata.get("shadow_verifier_profile") or rllm_metadata.get("shadow_verifier_profile") or "")
        self.pass_rate_mode = self.verifier_profile == "denovoswe_official"
        self.milestone_only = bool(
            self.pass_rate_mode
            and rllm_metadata.get("denovo_background_finalize_active", False)
        )
        merge_requested = getattr(
            harness,
            "denovo_shadow_probe_merge_enable",
            False,
        )
        configured_merge_max = getattr(
            harness,
            "denovo_shadow_probe_merge_max_steps",
            3,
        )
        if isinstance(configured_merge_max, bool):
            raise ValueError("denovo shadow probe merge max steps must be an integer")
        try:
            configured_merge_max = int(configured_merge_max)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "denovo shadow probe merge max steps must be an integer"
            ) from exc
        if merge_requested is True and configured_merge_max < 2:
            raise ValueError(
                "denovo shadow probe merge max steps must be at least 2"
            )
        self.probe_merge_enabled = bool(
            self.milestone_only and merge_requested is True
        )
        self.probe_merge_max_steps = configured_merge_max
        if self.pass_rate_mode and not self.language:
            # Existing materializations predate the language field.  The
            # profile is immutable and Python-only, so this is a safe runtime
            # compatibility fill rather than heuristic language detection.
            self.language = "python"
        self._denovo_instance: dict[str, Any] | None = None
        self._swe_instance: dict[str, Any] | None = None
        self._all_f2p_contract = False
        self._f2p_contract: frozenset[str] = frozenset()
        self._p2p_contract: frozenset[str] = frozenset()
        if self.pass_rate_mode:
            self.result_parser = DENOVOSWE_RESULT_PARSER
            self.test_results_path = DEFAULT_TEST_RESULTS_PATH
            instance_path = task.task_dir / "tests" / "instance.json"
            instance = json.loads(instance_path.read_text(encoding="utf-8"))
            if not isinstance(instance, dict):
                raise ValueError("DeNovoSWE instance.json must contain an object")
            authoritative = _string_list(instance.get("passed_ptp"), "passed_ptp")
            if not authoritative:
                raise ValueError("DeNovoSWE passed_ptp must not be empty")
            _safe_denovo_workdir(instance.get("workdir"))
            self._denovo_instance = instance
            self.materialized_baseline_results = {name: "FAILED" for name in authoritative}
            self.target_results = {name: "PASSED" for name in authoritative}
            self.test_set_policy = _EXACT_TEST_SET_POLICY
            self.authoritative_test_names = authoritative
        else:
            self.result_parser, parser_error = _result_parser_contract(metadata)
            if parser_error is not None:
                raise ValueError(parser_error)
            self.test_results_path = str(metadata.get("test_results_path")) if self.result_parser in SWEREBENCH_V2_RESULT_PARSERS else DEFAULT_TEST_RESULTS_PATH
            self.materialized_baseline_results = _status_map(metadata.get("baseline_output_json"), "baseline_output_json")
            self.target_results = _status_map(metadata.get("target_output_json"), "target_output_json")
            self.test_set_policy, self.authoritative_test_names = _test_set_contract(
                metadata,
                self.result_parser,
                self.target_results,
            )
            if self.result_parser in SWEREBENCH_V2_RESULT_PARSERS:
                self._f2p_contract = frozenset(
                    _string_list(metadata.get("FAIL_TO_PASS"), "FAIL_TO_PASS")
                )
                self._p2p_contract = frozenset(
                    _string_list(metadata.get("PASS_TO_PASS"), "PASS_TO_PASS")
                )
                self._all_f2p_contract = not self._p2p_contract
                instance_path = task.task_dir / "tests" / "instance.json"
                if instance_path.is_file():
                    raw_instance = json.loads(instance_path.read_text(encoding="utf-8"))
                    if not isinstance(raw_instance, dict):
                        raise ValueError("SWE-rebench V2 instance.json must be an object")
                    self._swe_instance = raw_instance
        self.runtime_baseline_results: dict[str, str] | None = None
        self.verification_result_mode = "per_test"
        self.partition_adapter: PartitionAdapter | None = None
        self.partition_calibration_status = "not_attempted"
        self.partition_calibration_reason: str | None = None
        self._partition_instance_paths: dict[str, tuple[str, ...]] = {}
        self.partition_calibration_diagnostics: dict[str, Any] = {}
        self._confirmed_p2p_regressions: dict[str, bool] = {}
        self._confirmed_p2p_regression_count = 0
        harness_values = getattr(harness, "__dict__", {})
        configured_bug_repair_mode = str(
            rllm_metadata.get(
                "bug_repair_verification_potential_mode",
                harness_values.get(
                    "bug_repair_verification_potential_mode",
                    DEFAULT_BUG_REPAIR_VERIFICATION_POTENTIAL_MODE,
                ),
            )
        ).strip().casefold()
        if configured_bug_repair_mode not in {"normalized", "pass_count"}:
            raise ValueError(
                "bug_repair_verification_potential_mode must be pass_count or normalized"
            )
        self.bug_repair_verification_potential_mode = configured_bug_repair_mode
        self.bug_repair_pass_count_mode = bool(
            not self.pass_rate_mode and configured_bug_repair_mode == "pass_count"
        )
        self.verification_potential_mode = (
            "pass_count"
            if self.pass_rate_mode
            else BUG_REPAIR_PASS_COUNT_MODE
            if self.bug_repair_pass_count_mode
            else "bug_repair"
        )
        self.verification_potential_policy = (
            "repo_generation_pass_count_v1"
            if self.pass_rate_mode
            else "bug_repair_pass_count_v1"
            if self.bug_repair_pass_count_mode
            else "bug_repair_v2_neutral_unrun"
        )
        if self.bug_repair_pass_count_mode:
            self.verification_result_mode = "aggregate_counts"
        self.materialized_baseline_passed = sum(
            str(self.materialized_baseline_results.get(name, "NOT_RUN")).upper()
            == "PASSED"
            for name in self.authoritative_test_names
        )
        self.runtime_baseline_passed: int | None = None
        self.baseline_passed: int | None = None
        self.baseline_source: str | None = None
        self.baseline_probe_succeeded = False
        self.baseline_collector: str | None = None
        self._count_confirmed_p2p_regressions = 0
        self.partition_timeout_recovery = _bool_value(
            rllm_metadata.get(
                "shadow_partition_timeout_recovery",
                harness_values.get(
                    "shadow_partition_timeout_recovery",
                    DEFAULT_SHADOW_PARTITION_TIMEOUT_RECOVERY,
                ),
            ),
            DEFAULT_SHADOW_PARTITION_TIMEOUT_RECOVERY,
            label="shadow partition timeout recovery",
        )
        self.partition_recovery_command_timeout = _positive_float(
            rllm_metadata.get(
                "shadow_partition_recovery_command_timeout",
                harness_values.get(
                    "shadow_partition_recovery_command_timeout",
                    DEFAULT_SHADOW_PARTITION_RECOVERY_COMMAND_TIMEOUT,
                ),
            ),
            DEFAULT_SHADOW_PARTITION_RECOVERY_COMMAND_TIMEOUT,
            label="shadow partition recovery command timeout",
        )
        self.partition_recovery_max_commands = _positive_int(
            rllm_metadata.get(
                "shadow_partition_recovery_max_commands",
                harness_values.get(
                    "shadow_partition_recovery_max_commands",
                    DEFAULT_SHADOW_PARTITION_RECOVERY_MAX_COMMANDS,
                ),
            ),
            DEFAULT_SHADOW_PARTITION_RECOVERY_MAX_COMMANDS,
            label="shadow partition recovery max commands",
        )
        configured_stall_timeout = harness_values.get("shadow_finalize_stall_timeout")
        if configured_stall_timeout is None:
            configured_stall_timeout = getattr(
                harness,
                "shadow_finalize_stall_timeout",
                DEFAULT_SHADOW_FINALIZE_STALL_TIMEOUT,
            )

        task_stall_timeout = rllm_metadata.get("shadow_finalize_stall_timeout")
        if task_stall_timeout is not None:
            configured_stall_timeout = task_stall_timeout
        self.finalize_stall_timeout = _positive_float(
            configured_stall_timeout,
            DEFAULT_SHADOW_FINALIZE_STALL_TIMEOUT,
        )
        self.probe_execution_timeout = _positive_float(
            metadata.get("verifier_timeout"),
            1800.0,
            label="verifier timeout",
        )
        self.probe_stall_grace = min(
            60.0,
            max(0.01, self.probe_execution_timeout * 0.05),
        )
        # Runtime compatibility alias. This is a hard wall-clock limit that
        # starts only after the primary verifier has finished.
        self.beta_any = _unit_float(rllm_metadata.get("shadow_beta_any"), 0.5)
        self.beta_frac = _unit_float(rllm_metadata.get("shadow_beta_frac"), 0.5)
        self._queue: queue.Queue[Any] = queue.Queue()
        self._lock = threading.RLock()
        self._progress_condition = threading.Condition(self._lock)
        self._thread: threading.Thread | None = None
        self._cleanup_observer: threading.Thread | None = None
        self._cleanup_status = "not_requested"
        self._cancel_requested = threading.Event()
        self._started_at = time.monotonic()
        self._baseline_ref: str | None = None
        self._history: list[Any] = []
        self._milestone_action_journal: list[dict[str, Any]] = []
        self._sealed_milestone_action_journal: tuple[dict[str, Any], ...] | None = None
        self._milestone_input_sealing = False
        self._milestone_handoff_snapshot: dict[str, Any] | None = None
        self._probe_merge_signature: frozenset[str] | None = None
        self._probe_merge_members: list[_PendingProbeMergeMember] = []
        self._probe_merge_sequence = 0
        self._probe_merge_group_size_histogram: dict[int, int] = {}
        self._probe_merge_decision_wait_samples_s: list[float] = []
        self._probe_merge_run_signature: frozenset[str] | None = None
        self._probe_merge_run_length = 0
        self._probe_merge_same_changed_files_run_length_histogram: dict[
            int, int
        ] = {}
        self._fixture_link_existed = False
        self._disabled = False
        self._sealed = False
        self._stop_enqueued = False
        self._current_potential = 0.0
        self._errors: list[str] = []
        self._baseline_event: dict[str, Any] | None = None
        self._final_event: dict[str, Any] | None = None
        self._final_verifier_outcome: dict[str, Any] | None = None
        self._final_recovery_factory: Any = None
        self._final_recovery = {
            "status": "not_attempted",
            "attempts": 0,
            "original_failure": None,
            "errors": [],
            "final_repo_fingerprint": None,
        }
        self._finalize_timed_out = False
        self._queue_depth_at_finalize = 0
        self._pending_events_at_timeout = 0
        self._forced_teardown_requested = False
        self._progress_sequence = 0
        self._last_progress_at = time.monotonic()
        self._current_job_type: str | None = None
        self._current_job_started_at: float | None = None
        self._finalize_progress_resets = 0
        self._finalize_watchdog_reports = 0
        self._baseline_reference = {
            "status": "unavailable",
            "mismatch_count": 0,
            "mismatch_tests": [],
            "truncated": False,
        }
        self._terminal_replay: dict[str, Any] | None = None
        self._replay_warnings: list[dict[str, Any]] = []
        self._replay_recovery_bundle_paths: set[str] = set()
        self._replay_state_capture_diagnostics: list[dict[str, Any]] = []
        self._restore_warnings: list[dict[str, Any]] = []
        self._parser_script_path: str | None = None
        self._parser_contract_path: str | None = None
        self._pytest_plugin_path: str | None = None
        self._pytest_plugin_results_path: str | None = None
        self._pytest_plugin_events_path: str | None = None
        self._count_collector_path: str | None = None
        self._count_contract_path: str | None = None
        self._count_instance_path: str | None = None
        self._count_plan_hash: str | None = None
        self._mocha_reporter_path: str | None = None
        self._count_probe_started_at = 0.0
        self._node_yarn_wrapper_path: str | None = None
        self._parser_command_length: int | None = None
        self._parser_transport = (
            ("uploaded_compact_framed_with_pytest_hook_v2" if self.result_parser == SWEREBENCH_V2_RESULT_PARSER else "uploaded_compact_framed_official_v2")
            if self.result_parser in SWEREBENCH_V2_RESULT_PARSERS
            else ("denovoswe_structured_json_v1" if self.result_parser == DENOVOSWE_RESULT_PARSER else "legacy_inline_json_v1")
        )
        if self.bug_repair_pass_count_mode:
            self._parser_transport = (
                "uploaded_aggregate_count_collector_v1"
                if self.result_parser in SWEREBENCH_V2_RESULT_PARSERS
                else "exact_vector_count_adapter_v1"
            )
        self._counts = {
            "replays_scheduled": 0,
            "replays_skipped_no_repo_change": 0,
            "replays_completed": 0,
            "probes_scheduled": 0,
            "probes_completed": 0,
            "probes_failed": 0,
            "probes_timed_out": 0,
            "state_mismatches": 0,
            "max_queue_depth": 0,
            "trusted_probes": 0,
            "recoverable_probes": 0,
            "terminal_probes": 0,
            "terminal_failures": 0,
            "suppressed_probes": 0,
            "collection_aborts": 0,
            "probe_retries": 0,
            "verifier_side_effects_restored": 0,
            "trusted_not_run_probes": 0,
            "replay_execution_mismatches": 0,
            "replay_timeout_mismatches": 0,
            "replay_observation_mismatches": 0,
            "replay_state_capture_attempts": 0,
            "replay_state_capture_successes": 0,
            "replay_state_capture_failures": 0,
            "replay_state_recovery_attempts": 0,
            "replay_state_recovery_successes": 0,
            "replay_state_recovery_failures": 0,
            "restored_policy_rejections": 0,
            "restore_protocol_warnings": 0,
            "restore_fingerprint_fallbacks": 0,
            "pytest_plugin_probes": 0,
            "controlled_timeout_probes": 0,
            "contract_assisted_baselines": 0,
            "contract_assisted_probes": 0,
            "parameter_id_aliases": 0,
            "final_verifiers": 0,
            "partition_group_calls": 0,
            "partition_group_successes": 0,
            "partition_group_partial": 0,
            "partition_group_timeouts": 0,
            "partition_calibration_attempts": 0,
            "partition_calibration_successes": 0,
            "partition_timeout_recovery_attempts": 0,
            "partition_timeout_recovery_successes": 0,
            "partition_timeout_recovery_partial": 0,
            "partition_timeout_recovery_failures": 0,
            "count_full_probes": 0,
            "count_partial_probes": 0,
            "count_unavailable_probes": 0,
            "baseline_probe_successes": 0,
            "baseline_singleflight_leaders": 0,
            "baseline_singleflight_waits": 0,
            "baseline_singleflight_hits": 0,
            "baseline_singleflight_install_failures": 0,
            "materialized_baseline_fallbacks": 0,
            "baseline_count_mismatches": 0,
            "probe_merge_candidate_steps": 0,
            "probe_merge_groups": 0,
            "probe_merge_steps": 0,
            "probe_merge_saved_probes": 0,
        }
        self._timings = {
            "setup_s": 0.0,
            "baseline_probe_s": 0.0,
            "replay_s": 0.0,
            "probe_s": 0.0,
            "finalize_wait_s": 0.0,
            "post_primary_wait_s": 0.0,
            "partition_probe_s": 0.0,
            "partition_calibration_s": 0.0,
        }
        self._failure_counts: dict[str, int] = {}
        self._recoverable_failure_counts: dict[str, int] = {}
        self._status = "initializing"
        self._background_lifecycle: dict[str, Any] = {
            "schema_version": 1,
            "mode": "milestone_only" if self.milestone_only else "legacy",
            "input_sealed_at_monotonic": None,
            "primary_verifier_finished_at_monotonic": None,
            "batch_barrier_started_at_monotonic": None,
            "batch_barrier_observed_at_monotonic": None,
            "batch_barrier_deadline_monotonic": None,
            "batch_barrier_budget_remaining_s": None,
            "finalized_at_monotonic": None,
            "finalize_disposition": None,
            "completed_probes_at_primary_finish": None,
            "completed_probes_at_barrier_start": None,
            "completed_probes_at_barrier": None,
            "pending_probes_at_barrier": None,
            "completed_probes_at_disposition": None,
            "cancelled_pending_steps": None,
            "shadow_drained_at_primary_finish": None,
            "shadow_drained_at_barrier_start": None,
            "shadow_drained_at_disposition": None,
            "queue_depth_at_handoff": None,
            "queue_depth_at_barrier_start": None,
            "handoff_snapshot": None,
        }

    @classmethod
    def create_and_start(cls, harness: Any, task: Task, primary: Any, shadow: Any, uid: str) -> ShadowSandboxRuntime:
        runtime = cls(harness, task, primary, shadow, uid)
        runtime.start()
        return runtime

    def start(self) -> None:
        tests_dir = self.task.task_dir / "tests"
        from rllm.sandbox.verifier_assets import upload_verifier_assets
        upload_verifier_assets(self.shadow, self.task, tests_dir)
        self._helper_python()
        if self.result_parser in SWEREBENCH_V2_RESULT_PARSERS:
            self._upload_swerebench_parser_contract()
        primary_state = self.harness._codeflow_capture_repo_snapshot(self.primary, self.task, None)
        shadow_state = self.harness._codeflow_capture_repo_snapshot(self.shadow, self.task, None)
        if primary_state.status != RepoStateStatus.OK or shadow_state.status != RepoStateStatus.OK:
            raise RuntimeError(f"initial repository probe failed: primary={primary_state.status.value}:{primary_state.error}; shadow={shadow_state.status.value}:{shadow_state.error}")
        if primary_state.fingerprint != shadow_state.fingerprint or primary_state.baseline_ref != shadow_state.baseline_ref:
            raise RuntimeError(f"initial primary/shadow repository mismatch: primary={primary_state.fingerprint}, shadow={shadow_state.fingerprint}")
        if self.pass_rate_mode:
            root = self.harness._repository_root(self.task)
            root_q = shlex.quote(root)
            primary_root = str(self.primary.exec(f"cd {root_q} && pwd -P", timeout=10)).strip()
            shadow_root = str(self.shadow.exec(f"cd {root_q} && pwd -P", timeout=10)).strip()
            if primary_root != shadow_root or primary_root != root:
                raise RuntimeError(f"initial primary/shadow workdir mismatch: configured={root}, primary={primary_root}, shadow={shadow_root}")
        self._baseline_ref = shadow_state.baseline_ref
        if self._uses_non_python_evidence():
            # Includes cache hits: compile diagnostics on a file edited by an
            # earlier action still describe the current, verified repository.
            self._non_python_initial_snapshot = shadow_state
        if self.result_parser == R2E_RESULT_PARSER:
            root = self.harness._repository_root(self.task)
            link = f"{root.rstrip('/')}/r2e_tests"
            raw = self.shadow.exec(
                f"if [ -e {shlex.quote(link)} ] || [ -L {shlex.quote(link)} ]; then printf yes; else printf no; fi",
                timeout=10,
                user=self.task.metadata.get("verifier_user"),
            )
            self._fixture_link_existed = str(raw).strip() == "yes"
        self._status = "running"
        self._thread = threading.Thread(target=self._worker, name=f"rllm-shadow-{self.uid}", daemon=True)
        self._thread.start()

    def _helper_python(self) -> str:
        runtime = self.harness._codeflow_helper_runtime(
            self.shadow, self.task, user=self.task.metadata.get("verifier_user"),
            minimum=(3, 7),
        )
        return shlex.quote(runtime["executable"])

    def _resolve_package_script_contracts(
        self,
        commands: list[str],
    ) -> dict[int, dict[str, str]] | None:
        if self.language.strip().casefold() not in {
            "javascript",
            "js",
            "typescript",
            "ts",
        }:
            return None
        package_indexes = [
            index
            for index, command in enumerate(commands)
            if re.search(r"(?<![\w./-])(?:npm|pnpm|yarn)\s+", command)
        ]
        if not package_indexes:
            return None
        repository_root = self.harness._repository_root(self.task)
        encoded = base64.b64encode(
            json.dumps(commands, ensure_ascii=True).encode("utf-8")
        ).decode("ascii")
        command = (
            self._helper_python() + " -c "
            + shlex.quote(_PACKAGE_SCRIPT_RESOLVER)
            + " "
            + shlex.quote(repository_root)
            + " "
            + shlex.quote(encoded)
        )
        try:
            raw = self.shadow.exec(
                command,
                timeout=30,
                user=self.task.metadata.get("verifier_user"),
            )
            parsed = json.loads(str(raw).strip())
            if not isinstance(parsed, dict):
                raise ValueError("package script resolver returned a non-object")
            resolutions = {
                int(index): value
                for index, value in parsed.items()
                if str(index).isdigit() and isinstance(value, dict)
            }
            for index in package_indexes:
                resolutions.setdefault(
                    index,
                    {
                        "status": "error",
                        "reason": "package_script_resolution_missing",
                    },
                )
            return resolutions
        except Exception as exc:
            reason = f"package_script_resolution_failed:{type(exc).__name__}"
            return {
                index: {"status": "error", "reason": reason}
                for index in package_indexes
            }

    def _uses_non_python_evidence(self) -> bool:
        return (
            self.result_parser == SWEREBENCH_V2_OFFICIAL_RESULT_PARSER
            and self.language.strip().casefold() in {"go", "js", "ts"}
            and not self.pass_rate_mode
            and not self.bug_repair_pass_count_mode
        )

    def _upload_swerebench_parser_contract(self) -> None:
        """Upload parser code and task contract once, keeping probe argv bounded."""

        parser_source = _PARSE_SWEREBENCH_V2_RESULTS_SCRIPT
        if self._uses_non_python_evidence():
            from rllm.harnesses.non_python_verification import parser_script

            parser_source = parser_script(parser_source)
        plugin_results_name = "shadow_test_results.json"
        plugin_events_name = "shadow_test_events.jsonl"
        contract = {
            "schema_version": 1,
            "results_path": self.test_results_path,
            "output_path": "/tmp/test_output.txt",
            "tail_chars": 2000,
            "expected_parser": self.result_parser,
            "expected_log_parser": self.log_parser or None,
            "expected_language": self.language or None,
            "expected_result_adapter_version": (
                SWEREBENCH_V2_RESULT_ADAPTER_VERSION
                if self.result_parser == SWEREBENCH_V2_OFFICIAL_RESULT_PARSER
                else None
            ),
            "expected_tests": list(self.authoritative_test_names),
            "compact": True,
        }
        if (
            self.result_parser == SWEREBENCH_V2_RESULT_PARSER
            and self.language.strip().casefold() == "python"
            and not self.pass_rate_mode
            and not self.bug_repair_pass_count_mode
        ):
            # The parser and hook are uploaded as one immutable bundle.  The
            # normalized Python path consumes the hook's per-test JSONL, so
            # bind every accepted event to its exact owner contract instead
            # of treating it as the legacy unowned v1 stream.
            contract["pytest_event_contract"] = {
                "schema_version": 2,
                "plan_hash": "legacy",
                "command_ids": ["command-0"],
            }
        if self._uses_non_python_evidence():
            contract["non_python_evidence"] = {
                "schema_version": 1,
                "probe_binding_path": "/tmp/rllm/non_python_probe_token",
            }
        count_contract: dict[str, Any] | None = None
        count_instance: dict[str, Any] | None = None
        count_collector_source = ""
        count_runner_contract: dict[str, Any] | None = None
        if self.bug_repair_pass_count_mode:
            if not isinstance(self._swe_instance, dict):
                raise ValueError(
                    "pass-count mode requires a SWE-rebench runtime instance"
                )
            config = self._swe_instance.get("install_config")
            if not isinstance(config, dict):
                raise ValueError("SWE-rebench install_config is unavailable")
            raw_commands = config.get("test_cmd")
            original_commands = (
                [raw_commands]
                if isinstance(raw_commands, str)
                else list(raw_commands)
                if isinstance(raw_commands, list)
                else []
            )
            if not original_commands or any(
                not isinstance(command, str) or not command.strip()
                for command in original_commands
            ):
                raise ValueError("SWE-rebench test_cmd is unavailable")
            count_plan = "acceptance_selector"
            count_plan_fallback_reason = None
            try:
                expanded_commands = list(
                    expand_runner_commands(self._swe_instance)
                )
                package_script_resolutions = (
                    self._resolve_package_script_contracts(expanded_commands)
                )
                runner_plan = build_count_runner_plan(
                    self._swe_instance,
                    package_script_resolutions=package_script_resolutions,
                )
                commands = [
                    command.command for command in runner_plan.commands
                ]
                count_runner_contract = runner_plan.as_contract()
                self._count_plan_hash = runner_plan.plan_hash
            except PartitionAdapterUnsupported as exc:
                # Unsupported runners retain the original full-suite command.
                # Its aggregate is accepted only if the collector proves that
                # the terminal total is exactly the materialized contract N.
                count_plan = "full_suite_fallback"
                count_plan_fallback_reason = exc.reason
                try:
                    commands = list(expand_runner_commands(self._swe_instance))
                except PartitionAdapterUnsupported:
                    commands = list(original_commands)
                fallback_commands = []
                for index, command in enumerate(commands):
                    expected_tests = list(self.authoritative_test_names)
                    fallback_commands.append(
                        {
                            "command_id": f"command-{index}",
                            "runner": "summary",
                            "source_command_index": index,
                            "shard_index": 0,
                            "shard_count": 1,
                            "command": command,
                            "selector": "",
                            "expected_tests": expected_tests,
                            "expected_aliases": expected_tests,
                            "selection_strategy": "full_suite_fallback",
                            "output_path": None,
                            "package_scope": None,
                            "workspace_scope": None,
                            "test_files": [],
                            "structural_extra_policy": "none",
                            "result_channel": "native",
                            "runner_resolution": "fallback",
                            "wrapped_command_bytes": len(
                                wrap_count_command(command, index).encode("utf-8")
                            ),
                            "script_path": None,
                            "script_name": None,
                            "script_hash": None,
                        }
                    )
                fallback_payload = {
                    "schema_version": COUNT_RUNNER_PLAN_SCHEMA_VERSION,
                    "expected_tests": list(self.authoritative_test_names),
                    "f2p_tests": list(self._swe_instance.get("FAIL_TO_PASS") or []),
                    "p2p_tests": list(self._swe_instance.get("PASS_TO_PASS") or []),
                    "commands": fallback_commands,
                }
                self._count_plan_hash = hashlib.sha256(
                    json.dumps(
                        fallback_payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                count_runner_contract = {
                    "schema_version": fallback_payload["schema_version"],
                    "plan_hash": self._count_plan_hash,
                    "expected": len(self.authoritative_test_names),
                    "expected_tests": list(self.authoritative_test_names),
                    "f2p_tests": fallback_payload["f2p_tests"],
                    "p2p_tests": fallback_payload["p2p_tests"],
                    "commands": fallback_commands,
                }
            wrapped_commands = []
            for index, command in enumerate(commands):
                wrapped_commands.append(wrap_count_command(command, index))
            count_instance = copy.deepcopy(self._swe_instance)
            count_instance["install_config"] = dict(config)
            count_instance["install_config"]["test_cmd"] = wrapped_commands
            count_instance["rllm_shadow_count_collector"] = {
                "schema_version": 1,
                "original_test_command_count": len(original_commands),
                "collector_command_count": len(commands),
                "primary_commands_unchanged": True,
                "count_plan": count_plan,
                "count_plan_fallback_reason": count_plan_fallback_reason,
            }
            count_contract = {
                "schema_version": 1,
                "expected": len(self.authoritative_test_names),
                "expected_tests": list(self.authoritative_test_names),
                "language": self.language,
                "log_parser": self.log_parser,
                "result_parser": self.result_parser,
                "commands": commands,
                "original_commands": original_commands,
                "count_plan": count_plan,
                "count_plan_fallback_reason": count_plan_fallback_reason,
                "artifact_path": self.test_results_path,
                "log_path": "/tmp/test_output.txt",
                "pytest_events_path": "/tmp/rllm/shadow_test_events.jsonl",
                "workdir": str(self._swe_instance.get("workdir") or ""),
                "started_at_path": "/tmp/rllm/count_probe_started",
            }
            if self._count_plan_hash is None or count_runner_contract is None:
                raise ValueError("count runner plan was not constructed")
            count_contract["plan_hash"] = self._count_plan_hash
            count_contract["runner_plan"] = count_runner_contract
            count_contract["f2p_tests"] = list(
                count_runner_contract.get("f2p_tests") or []
            )
            count_contract["p2p_tests"] = list(
                count_runner_contract.get("p2p_tests") or []
            )
            count_collector_source = Path(__file__).with_name(
                "test_count_collector.py"
            ).read_text(encoding="utf-8")
        identity = hashlib.blake2b(
            (
                parser_source
                + "\n"
                + _SWEREBENCH_PYTEST_PLUGIN_SCRIPT
                + "\n"
                + _MOCHA_COUNT_REPORTER_SCRIPT
                + "\n"
                + count_collector_source
                + "\n"
                + json.dumps(contract, sort_keys=True, separators=(",", ":"))
                + "\n"
                + json.dumps(count_contract, sort_keys=True, separators=(",", ":"))
                + "\n"
                + json.dumps(count_instance, sort_keys=True, separators=(",", ":"))
            ).encode("utf-8"),
            digest_size=10,
        ).hexdigest()
        remote_dir = f"/tmp/rllm-shadow/{identity}"
        self._parser_script_path = f"{remote_dir}/parse_results.py"
        self._parser_contract_path = f"{remote_dir}/contract.json"
        self._pytest_plugin_path = f"{remote_dir}/rllm_shadow_pytest_plugin.py"
        self._pytest_plugin_results_path = f"/tmp/rllm/{plugin_results_name}"
        self._pytest_plugin_events_path = f"/tmp/rllm/{plugin_events_name}"
        if count_contract is not None:
            self._count_collector_path = f"{remote_dir}/test_count_collector.py"
            self._count_contract_path = f"{remote_dir}/count_contract.json"
            self._count_instance_path = f"{remote_dir}/count_instance.json"
            self._mocha_reporter_path = (
                "/tmp/rllm/shadow-count/mocha-reporter.js"
            )
        contract["fallback_results_path"] = self._pytest_plugin_results_path
        contract["events_path"] = self._pytest_plugin_events_path
        with tempfile.TemporaryDirectory(prefix="rllm-shadow-parser-") as local_dir:
            script_path = f"{local_dir}/parse_results.py"
            contract_path = f"{local_dir}/contract.json"
            plugin_path = f"{local_dir}/rllm_shadow_pytest_plugin.py"
            yarn_wrapper_path = f"{local_dir}/yarn"
            count_collector_path = f"{local_dir}/test_count_collector.py"
            count_contract_path = f"{local_dir}/count_contract.json"
            count_instance_path = f"{local_dir}/count_instance.json"
            mocha_reporter_path = f"{local_dir}/mocha-reporter.js"
            with open(script_path, "w", encoding="utf-8") as handle:
                handle.write(parser_source)
            with open(plugin_path, "w", encoding="utf-8") as handle:
                handle.write(_SWEREBENCH_PYTEST_PLUGIN_SCRIPT)
            with open(contract_path, "w", encoding="utf-8") as handle:
                json.dump(contract, handle, ensure_ascii=False, separators=(",", ":"))
            self.shadow.upload_file(script_path, self._parser_script_path)
            self.shadow.upload_file(plugin_path, self._pytest_plugin_path)
            self.shadow.upload_file(contract_path, self._parser_contract_path)
            if (
                count_contract is not None
                and count_instance is not None
                and self._count_collector_path is not None
                and self._count_contract_path is not None
                and self._count_instance_path is not None
            ):
                with open(count_collector_path, "w", encoding="utf-8") as handle:
                    handle.write(count_collector_source)
                with open(count_contract_path, "w", encoding="utf-8") as handle:
                    json.dump(
                        count_contract,
                        handle,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                with open(count_instance_path, "w", encoding="utf-8") as handle:
                    json.dump(
                        count_instance,
                        handle,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                if self._mocha_reporter_path is not None:
                    with open(
                        mocha_reporter_path,
                        "w",
                        encoding="utf-8",
                    ) as handle:
                        handle.write(_MOCHA_COUNT_REPORTER_SCRIPT)
                    self.shadow.upload_file(
                        mocha_reporter_path,
                        self._mocha_reporter_path,
                    )
                self.shadow.upload_file(
                    count_collector_path,
                    self._count_collector_path,
                )
                self.shadow.upload_file(
                    count_contract_path,
                    self._count_contract_path,
                )
                self.shadow.upload_file(
                    count_instance_path,
                    self._count_instance_path,
                )
            if self._requires_node_yarn_guard():
                with open(yarn_wrapper_path, "w", encoding="utf-8") as handle:
                    handle.write(_NODE_YARN_WRAPPER_SCRIPT)
                self._node_yarn_wrapper_path = f"{remote_dir}/node-bin/yarn"
                self.shadow.upload_file(
                    yarn_wrapper_path,
                    self._node_yarn_wrapper_path,
                )

    def _requires_node_yarn_guard(self) -> bool:
        if self.language.strip().casefold() not in {
            "javascript",
            "js",
            "typescript",
            "ts",
        }:
            return False
        # npm/pnpm/custom scripts can invoke Yarn below the visible verifier
        # command. Install the PATH-scoped guard for every Node-family task.
        return isinstance(self._swe_instance, dict)

    def _node_yarn_guard_shell(self) -> str:
        if self._node_yarn_wrapper_path is None:
            return ""
        wrapper = shlex.quote(self._node_yarn_wrapper_path)
        node_bin = shlex.quote(self._node_yarn_wrapper_path.rsplit("/", 1)[0])
        return (
            'export RLLM_ORIGINAL_PATH="$PATH"; '
            f"chmod +x {wrapper}; "
            f'export PATH={node_bin}:$PATH; '
        )

    def record_setup_duration(self, duration: float) -> None:
        with self._lock:
            self._timings["setup_s"] = max(0.0, float(duration))

    @staticmethod
    def _canonical_sha256(value: Any) -> str:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _milestone_action_journal_entry(
        raw: dict[str, Any],
        *,
        sequence: int,
    ) -> dict[str, Any]:
        """Return the non-secret replay identity retained at handoff.

        Normalized command arguments and observations intentionally remain in
        the rollout-local replay job/Episode and are not duplicated into the
        lifecycle audit.  Fingerprints and changed paths are sufficient to
        prove ordering and repository continuity without expanding logs with
        source or shell contents.
        """

        return {
            "sequence": int(sequence),
            "action_id": str(raw.get("action_id") or ""),
            "turn_id": (
                int(raw["turn_id"])
                if isinstance(raw.get("turn_id"), int)
                else None
            ),
            "tool_name": str(raw.get("tool_name") or ""),
            "validation_status": str(raw.get("validation_status") or ""),
            "execution_status": str(raw.get("execution_status") or ""),
            "repo_state_status": str(raw.get("repo_state_status") or ""),
            "repo_state_before": (
                str(raw["repo_state_before"])
                if isinstance(raw.get("repo_state_before"), str)
                else None
            ),
            "repo_state_after": (
                str(raw["repo_state_after"])
                if isinstance(raw.get("repo_state_after"), str)
                else None
            ),
            "repo_changed": (
                bool(raw["repo_changed"])
                if isinstance(raw.get("repo_changed"), bool)
                else None
            ),
            "changed_paths": _changed_file_paths(raw.get("changed_files")),
        }

    def _capture_milestone_handoff_snapshot(
        self,
        journal: tuple[dict[str, Any], ...],
        primary: Any,
    ) -> dict[str, Any]:
        snapshot = self.harness._codeflow_capture_repo_snapshot(
            primary,
            self.task,
            self._baseline_ref,
        )
        if snapshot.status != RepoStateStatus.OK or not snapshot.fingerprint:
            raise RuntimeError(
                "final primary repository snapshot is unavailable at DeNovo "
                f"handoff: status={snapshot.status.value} error={snapshot.error}"
            )

        expected_fingerprint = next(
            (
                entry["repo_state_after"]
                for entry in reversed(journal)
                if isinstance(entry.get("repo_state_after"), str)
                and entry["repo_state_after"]
            ),
            None,
        )
        if (
            expected_fingerprint is not None
            and expected_fingerprint != snapshot.fingerprint
        ):
            raise RuntimeError(
                "final primary repository fingerprint disagrees with the "
                "sealed action journal: "
                f"journal={expected_fingerprint} primary={snapshot.fingerprint}"
            )

        delta = [
            {
                "path": state.path,
                "base_exists": state.base_exists,
                "base_digest": state.base_digest,
                "base_mode": state.base_mode,
                "current_exists": state.current_exists,
                "current_digest": state.current_digest,
                "current_mode": state.current_mode,
            }
            for _, state in sorted(snapshot.files.items())
        ]
        # Lifecycle metadata is intentionally bounded.  The complete immutable
        # journal remains in memory for the lease lifetime and the full action
        # records remain in the rollout JSON after final disposition.
        journal_sample = [
            {
                **entry,
                "changed_paths": list(entry.get("changed_paths") or [])[:32],
                "changed_paths_truncated": len(entry.get("changed_paths") or [])
                > 32,
            }
            for entry in journal[:64]
        ]
        delta_sample = delta[:64]
        return {
            "schema_version": 1,
            "status": "sealed",
            "immutable_replay_inputs": True,
            "action_count": len(journal),
            "action_journal_sha256": self._canonical_sha256(journal),
            "action_journal_sample": journal_sample,
            "action_journal_sample_truncated": len(journal) > len(journal_sample),
            "baseline_ref": snapshot.baseline_ref,
            "final_repo_fingerprint": snapshot.fingerprint,
            "expected_final_repo_fingerprint": expected_fingerprint,
            "fingerprint_matches_action_journal": True,
            "final_repo_delta_count": len(delta),
            "final_repo_delta_sha256": self._canonical_sha256(delta),
            "final_repo_delta_sample": delta_sample,
            "final_repo_delta_sample_truncated": len(delta) > len(delta_sample),
            "repository_file_count": (
                len(snapshot.repository_files)
                if snapshot.repository_files is not None
                else None
            ),
        }

    def _clear_probe_merge_group_locked(self) -> None:
        self._probe_merge_members = []
        self._probe_merge_signature = None

    def _finish_probe_merge_run_locked(self) -> None:
        """Record one uncapped consecutive run of identical changed paths."""
        run_length = self._probe_merge_run_length
        if run_length > 0:
            histogram = (
                self._probe_merge_same_changed_files_run_length_histogram
            )
            histogram[run_length] = histogram.get(run_length, 0) + 1
        self._probe_merge_run_signature = None
        self._probe_merge_run_length = 0

    def _extend_probe_merge_run_locked(
        self,
        signature: frozenset[str],
    ) -> None:
        if self._probe_merge_run_signature == signature:
            self._probe_merge_run_length += 1
            return
        self._finish_probe_merge_run_locked()
        self._probe_merge_run_signature = signature
        self._probe_merge_run_length = 1

    def _flush_probe_merge_group_locked(self) -> None:
        """Resolve one look-ahead group and enqueue exactly one probe."""
        if not self._probe_merge_members:
            return
        members = tuple(self._probe_merge_members)
        self._clear_probe_merge_group_locked()
        self._probe_merge_sequence += 1
        group_id = f"{self.uid}:probe-merge:{self._probe_merge_sequence}"
        group_size = len(members)
        anchor = members[-1].event
        anchor_action_id = str(anchor.get("action_id") or "")
        anchor_state = str(anchor.get("repo_state_after") or "unavailable")
        anchor_test_event = (
            anchor.get("test_event")
            if isinstance(anchor.get("test_event"), dict)
            else {}
        )
        probe_id = str(
            anchor_test_event.get("probe_id")
            or f"{anchor_action_id}:probe"
        )
        now = time.monotonic()
        sinks = tuple(member.event for member in members)
        for index, member in enumerate(members):
            wait_s = max(0.0, now - member.decision_started_at)
            self._probe_merge_decision_wait_samples_s.append(wait_s)
            merge_metadata = {
                "probe_id": probe_id,
                "repo_state": anchor_state,
                "probe_merge_group_id": group_id,
                "probe_merge_group_size": group_size,
                "probe_merge_group_index": index,
                "probe_merge_anchor_action_id": anchor_action_id,
                "probe_merge_decision_wait_s": wait_s,
            }
            event_payload = member.event.get("test_event")
            if isinstance(event_payload, dict):
                event_payload.update(merge_metadata)
            replay_payload = member.replay_job.replay_event.get("test_event")
            if isinstance(replay_payload, dict):
                replay_payload.update(merge_metadata)
            # Replay is admitted immediately when the primary step arrives;
            # enrich the mutable job so a late replay failure fans out to the
            # now-resolved logical group.
            member.replay_job.probe_events = sinks
        self._queue.put(
            _ProbeJob(
                event=anchor,
                probe_event=copy.deepcopy(anchor),
                probe_events=sinks,
            )
        )
        self._counts["probes_scheduled"] += 1
        if group_size >= 2:
            self._counts["probe_merge_groups"] += 1
            self._counts["probe_merge_steps"] += group_size
            self._counts["probe_merge_saved_probes"] += group_size - 1
            self._probe_merge_group_size_histogram[group_size] = (
                self._probe_merge_group_size_histogram.get(group_size, 0) + 1
            )
        self._counts["max_queue_depth"] = max(
            self._counts["max_queue_depth"],
            self._queue.qsize(),
        )

    def _fail_probe_merge_group_locked(
        self,
        status: TestEventStatus,
        failure_type: str,
        error: str,
    ) -> None:
        members = tuple(self._probe_merge_members)
        self._clear_probe_merge_group_locked()
        for member in members:
            self._set_failure_event(
                member.event,
                status,
                failure_type,
                error,
            )

    def schedule(self, step: Step) -> None:
        with self._lock:
            if self._cancel_requested.is_set():
                return
        metadata = step.metadata if isinstance(step.metadata, dict) else {}
        raw = metadata.get(ACTION_EVENT_METADATA_KEY)
        if not isinstance(raw, dict):
            return
        if self.milestone_only:
            with self._lock:
                if (
                    self._milestone_input_sealing
                    or self._background_lifecycle["input_sealed_at_monotonic"]
                    is not None
                ):
                    raise RuntimeError(
                        "cannot schedule an action after DeNovo shadow input handoff"
                    )
                self._milestone_action_journal.append(
                    self._milestone_action_journal_entry(
                        raw,
                        sequence=len(self._milestone_action_journal),
                    )
                )
        tool_name = str(raw.get("tool_name") or "")
        repo_changed = raw.get("repo_changed")
        trigger_probe = repo_changed is True
        if trigger_probe and raw.get("test_event") is None and isinstance(raw.get("repo_state_after"), str):
            raw["test_event"] = self._pending_event(raw).model_dump(mode="json")

        merge_signature = (
            _probe_merge_changed_paths(raw.get("changed_files"))
            if self.probe_merge_enabled and trigger_probe
            else frozenset()
        )
        merge_candidate = bool(
            self.probe_merge_enabled
            and trigger_probe
            and merge_signature
            and tool_name in {"edit_file", "execute_bash"}
            and raw.get("validation_status") == ValidationStatus.OK.value
            and raw.get("execution_status") != ExecutionStatus.NOT_RUN.value
            and raw.get("repo_state_status") == RepoStateStatus.OK.value
            and isinstance(raw.get("repo_state_before"), str)
            and isinstance(raw.get("repo_state_after"), str)
        )
        if self.probe_merge_enabled:
            with self._lock:
                if (
                    not merge_candidate
                    or merge_signature != self._probe_merge_run_signature
                ):
                    # This metric follows the original primary action stream,
                    # so max-size probe chunks do not split a continuous run.
                    self._finish_probe_merge_run_locked()
                if self._probe_merge_members and (
                    not merge_candidate
                    or merge_signature != self._probe_merge_signature
                ):
                    # Queue the prior group's probe before this boundary action
                    # can be replayed.
                    self._flush_probe_merge_group_locked()

        if tool_name in {"search", "read_file"} and trigger_probe:
            self._set_failure_event(
                raw,
                TestEventStatus.SKIPPED,
                "policy_violation",
                "read-only tool changed the repository",
                count_terminal=True,
                state_mismatch_paths=_changed_file_paths(raw.get("changed_files")),
            )
            self._disable("read-only tool changed the primary repository", "policy_violation")
            return
        violations = set(raw.get("policy_violations") or [])
        repository_restored_after_policy_rejection = bool(
            tool_name == "execute_bash"
            and raw.get("repo_state_status") == RepoStateStatus.OK.value
            and raw.get("repo_changed") is False
            and isinstance(raw.get("repo_state_before"), str)
            and raw.get("repo_state_before") == raw.get("repo_state_after")
            and not violations
            & {
                "bash_repository_checkpoint_unavailable",
                "bash_repository_probe_failed",
                "bash_repository_restore_failed",
            }
        )
        terminal_policy = "repository_head_changed" in violations or bool(
            violations
            & {
                "bash_repository_checkpoint_unavailable",
                "bash_repository_probe_failed",
                "bash_repository_restore_failed",
            }
        )
        if (
            "bash_modified_protected_tests" in violations
            and not repository_restored_after_policy_rejection
        ):
            terminal_policy = True
        elif "bash_modified_protected_tests" in violations:
            with self._lock:
                self._counts["restored_policy_rejections"] += 1
        if terminal_policy:
            if trigger_probe:
                self._set_failure_event(
                    raw,
                    TestEventStatus.SKIPPED,
                    "policy_violation",
                    "repository policy violation",
                    count_terminal=True,
                    state_mismatch_paths=_changed_file_paths(raw.get("changed_files")),
                )
            self._disable("repository policy violation", "policy_violation")
            return
        if tool_name not in {"edit_file", "execute_bash"}:
            return
        if raw.get("validation_status") != ValidationStatus.OK.value or raw.get("execution_status") == ExecutionStatus.NOT_RUN.value:
            return
        # Bug repair intentionally retains its stable Git-delta-only replay
        # policy.  DeNovoSWE must additionally preserve non-Git state between
        # commands (mkdir, generated environments, package setup, ...), or a
        # later repository-changing command replays against the wrong state.
        # Proven observation-only Bash remains skippable and never triggers a
        # verifier probe.
        replay_without_probe = False
        if (
            self.milestone_only
            and not trigger_probe
            and raw.get("execution_status") == ExecutionStatus.SUCCESS.value
        ):
            if tool_name == "edit_file":
                replay_without_probe = True
            else:
                arguments = raw.get("normalized_arguments")
                command = (
                    arguments.get("command")
                    if isinstance(arguments, dict)
                    else None
                )
                repository_root = self.harness._repository_root(self.task)
                replay_without_probe = not (
                    isinstance(command, str)
                    and is_proven_side_effect_free_shell_command(
                        command,
                        repository_root,
                    )
                )
        if not trigger_probe and not replay_without_probe:
            with self._lock:
                self._counts["replays_skipped_no_repo_change"] += 1
            return
        if raw.get("repo_state_status") != RepoStateStatus.OK.value or not isinstance(raw.get("repo_state_before"), str) or not isinstance(raw.get("repo_state_after"), str):
            if trigger_probe:
                self._set_failure_event(
                    raw,
                    TestEventStatus.STATE_MISMATCH,
                    "repo_state_unavailable",
                    "primary repository state is unavailable",
                    count_terminal=True,
                )
            self._disable("primary repository state is unavailable for replay", "repo_state_unavailable")
            return

        recovery_bundle: _RepositoryDeltaBundle | None = None
        recovery_capture_error: str | None = None
        recovery_capture_diagnostics: dict[str, Any] | None = None
        if (
            not self.pass_rate_mode
            and trigger_probe
            and tool_name == "execute_bash"
            and self._baseline_ref is not None
            and hasattr(self.harness, "_codeflow_run_json_script")
        ):
            with self._lock:
                self._counts["replay_state_capture_attempts"] += 1
            try:
                recovery_bundle = _export_primary_repository_delta_bundle(
                    self.harness,
                    self.task,
                    self.primary,
                    self._baseline_ref,
                    str(raw["repo_state_after"]),
                    uid=f"{self.uid}:{raw.get('action_id')}:handoff",
                    cancel_event=self._cancel_requested,
                )
            except Exception as exc:
                recovery_capture_error = str(exc)[:1000]
                recovery_capture_diagnostics = copy.deepcopy(getattr(exc, "diagnostics", None))
                with self._lock:
                    if self._cancel_requested.is_set():
                        return
                    self._counts["replay_state_capture_failures"] += 1
                    self._replay_state_capture_diagnostics.append({
                        "action_id": raw.get("action_id"),
                        "error": recovery_capture_error,
                        "diagnostics": recovery_capture_diagnostics,
                    })
                    del self._replay_state_capture_diagnostics[:-8]
                logger.warning(
                    "[%s] exact replay-state capture failed for %s: %s",
                    self.uid,
                    raw.get("action_id"),
                    exc,
                )
            else:
                with self._lock:
                    if self._cancel_requested.is_set():
                        Path(recovery_bundle.local_path).unlink(missing_ok=True)
                        return
                    self._counts["replay_state_capture_successes"] += 1
                    self._replay_recovery_bundle_paths.add(
                        recovery_bundle.local_path
                    )

        with self._lock:
            if self._cancel_requested.is_set():
                if recovery_bundle is not None:
                    Path(recovery_bundle.local_path).unlink(missing_ok=True)
                    self._replay_recovery_bundle_paths.discard(recovery_bundle.local_path)
                return
            if self._disabled:
                if recovery_bundle is not None:
                    Path(recovery_bundle.local_path).unlink(missing_ok=True)
                    self._replay_recovery_bundle_paths.discard(
                        recovery_bundle.local_path
                    )
                if trigger_probe:
                    self._set_failure_event(
                        raw,
                        TestEventStatus.SKIPPED,
                        "suppressed_after_terminal",
                        "shadow runtime is disabled after a terminal failure",
                    )
                    self._counts["suppressed_probes"] += 1
                return
            self._counts["replays_scheduled"] += 1
            replay_event = copy.deepcopy(raw)
            if merge_candidate:
                self._extend_probe_merge_run_locked(merge_signature)
                replay_job = _ReplayJob(
                    event=raw,
                    replay_event=replay_event,
                    trigger_probe=False,
                    probe_events=(raw,),
                    recovery_bundle=recovery_bundle,
                    recovery_capture_error=recovery_capture_error,
                    recovery_capture_diagnostics=recovery_capture_diagnostics,
                )
                self._queue.put(replay_job)
                if not self._probe_merge_members:
                    self._probe_merge_signature = merge_signature
                self._probe_merge_members.append(
                    _PendingProbeMergeMember(
                        event=raw,
                        replay_job=replay_job,
                        decision_started_at=time.monotonic(),
                    )
                )
                self._counts["probe_merge_candidate_steps"] += 1
                if (
                    len(self._probe_merge_members)
                    >= self.probe_merge_max_steps
                ):
                    self._flush_probe_merge_group_locked()
                self._counts["max_queue_depth"] = max(
                    self._counts["max_queue_depth"],
                    self._queue.qsize(),
                )
                return
            if trigger_probe:
                self._counts["probes_scheduled"] += 1
            self._queue.put(
                _ReplayJob(
                    event=raw,
                    replay_event=replay_event,
                    trigger_probe=trigger_probe,
                    probe_events=(raw,) if trigger_probe else (),
                    recovery_bundle=recovery_bundle,
                    recovery_capture_error=recovery_capture_error,
                    recovery_capture_diagnostics=recovery_capture_diagnostics,
                )
            )
            self._counts["max_queue_depth"] = max(self._counts["max_queue_depth"], self._queue.qsize())

    def attach_episode(self, episode: Episode) -> None:
        episode.metadata[SHADOW_METADATA_KEY] = self.summary()

    def configure_final_recovery(self, factory: Any) -> None:
        """Install the hook-owned fresh-shadow provisioning callback."""

        if factory is not None and not callable(factory):
            raise TypeError("final recovery factory must be callable or None")
        self._final_recovery_factory = factory

    def _final_recovery_allowed(self) -> bool:
        if (
            not self.pass_rate_mode
            or not callable(self._final_recovery_factory)
            or self.runtime_baseline_results is None
        ):
            return False
        unsafe_failures = {
            "policy_violation",
            "repository_head_changed",
            "bash_repository_checkpoint_unavailable",
            "bash_repository_restore_failed",
            "setup_storage",
        }
        return not bool(unsafe_failures & set(self._failure_counts))

    def _final_failure_diagnostics(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "failure_counts": dict(self._failure_counts),
            "terminal_replay": copy.deepcopy(self._terminal_replay),
            "final_recovery": copy.deepcopy(self._final_recovery),
            "final_test_event": copy.deepcopy(self._final_event),
            "queue_depth_at_finalize": self._queue_depth_at_finalize,
            "pending_events_at_timeout": self._pending_events_at_timeout,
        }

    def _recover_authoritative_final(self, original_failure: str) -> EvalOutput:
        self._final_recovery["status"] = "running"
        self._final_recovery["original_failure"] = original_failure[:2000]
        tests_dir = self.task.task_dir / "tests"
        errors: list[str] = []
        for attempt in range(1, FINAL_RECOVERY_MAX_ATTEMPTS + 1):
            self._final_recovery["attempts"] = attempt
            try:
                fresh_shadow = self._final_recovery_factory()
                from rllm.sandbox.verifier_assets import upload_verifier_assets
                upload_verifier_assets(fresh_shadow, self.task, tests_dir)
                fingerprint = _synchronize_primary_repository_to_fresh_shadow(
                    self.harness,
                    self.task,
                    self.primary,
                    fresh_shadow,
                    str(self._baseline_ref or ""),
                    uid=f"{self.uid}-attempt-{attempt}",
                )
                self.shadow = fresh_shadow
                output = self._run_authoritative_final(recovery=True)
                self._counts["final_verifiers"] += 1
                self._counts["final_recovery_successes"] = (
                    self._counts.get("final_recovery_successes", 0) + 1
                )
                self._final_recovery.update(
                    {
                        "status": "succeeded",
                        "final_repo_fingerprint": fingerprint,
                        "errors": errors,
                    }
                )
                self._status = "recovered"
                return output
            except Exception as exc:
                error = str(exc)[:2000]
                errors.append(error)
                logger.warning(
                    "[%s] authoritative final-only shadow recovery attempt %d/%d failed: %s",
                    self.uid,
                    attempt,
                    FINAL_RECOVERY_MAX_ATTEMPTS,
                    error,
                )
        self._counts["final_recovery_failures"] = (
            self._counts.get("final_recovery_failures", 0) + 1
        )
        self._final_recovery.update(
            {"status": "failed", "errors": errors}
        )
        raise ShadowFinalizationError(
            "shadow_final_recovery_failed",
            "authoritative final-only shadow recovery failed: "
            + (errors[-1] if errors else original_failure),
            retryable=False,
            retry_scope="none",
            safe_for_final_recovery=False,
            diagnostics=self._final_failure_diagnostics(),
        )

    def finalize_episode(
        self,
        raw_episode: Episode,
        enriched_episode: Episode | None = None,
    ) -> EvalOutput | None:
        # Bug-repair invokes this after the primary verifier. DeNovoSWE skips
        # the primary verifier and makes the final shadow probe authoritative.
        with self._lock:
            self._queue_depth_at_finalize = self._queue.qsize()
        self._enqueue_stop()
        thread = self._thread
        wait_started = time.perf_counter()
        stalled = False
        if thread is not None:
            observed_progress = self._progress_sequence
            wait_deadline = time.monotonic() + self.finalize_stall_timeout
            next_report_at = time.monotonic() + 60.0
            while thread.is_alive():
                now = time.monotonic()
                with self._lock:
                    current_job = self._current_job_type
                    current_started = self._current_job_started_at
                remaining = wait_deadline - now
                if remaining <= 0:
                    stalled = True
                    break
                wait_for = min(remaining, max(0.0, next_report_at - now))
                with self._progress_condition:
                    self._progress_condition.wait(timeout=max(0.01, wait_for))
                    if self._progress_sequence != observed_progress:
                        observed_progress = self._progress_sequence
                        self._finalize_progress_resets += 1
                now = time.monotonic()
                if thread.is_alive() and now >= next_report_at:
                    with self._lock:
                        current_job = self._current_job_type
                        current_started = self._current_job_started_at
                        queue_depth = self._queue.qsize()
                        completed = {
                            "replays": self._counts["replays_completed"],
                            "probes": self._counts["probes_completed"],
                        }
                        self._finalize_watchdog_reports += 1
                    logger.info(
                        "[%s] waiting for shadow drain: job=%s job_runtime=%.1fs queue_depth=%d completed=%s remaining=%.1fs",
                        self.uid,
                        current_job or "idle",
                        max(0.0, now - current_started) if current_started is not None else 0.0,
                        queue_depth,
                        completed,
                        max(0.0, wait_deadline - now),
                    )
                    next_report_at = now + 60.0
            if not thread.is_alive():
                thread.join(timeout=0)
        waited = time.perf_counter() - wait_started
        with self._lock:
            self._timings["finalize_wait_s"] += waited
            self._timings["post_primary_wait_s"] += waited
        final_output: EvalOutput | None = None
        final_error: str | None = None
        if self.pass_rate_mode and (thread is None or not thread.is_alive()) and not self._disabled:
            try:
                self._set_current_job("final_verifier_wait")
                self._set_current_job("final_verifier")
                final_output = self._run_authoritative_final()
                self._counts["final_verifiers"] += 1
                self._note_progress("final_verifier")
            except Exception as exc:
                final_error = f"authoritative shadow verifier failed: {exc}"
                self._disable(final_error, "final_verifier")
        if thread is not None and thread.is_alive() and stalled:
            error = f"shadow finalization exceeded the post-primary wait limit of {self.finalize_stall_timeout:g}s"
            self._disable(error, "finalize_stall_timeout")
            with self._lock:
                self._finalize_timed_out = True
                self._forced_teardown_requested = True
                baseline_timed_out = self._mark_incomplete_baseline_timeout(error)
                self._pending_events_at_timeout = self._mark_pending_events(
                    raw_episode,
                    TestEventStatus.TIMEOUT,
                    "finalize_stall_timeout",
                    error,
                    timed_out=True,
                )
                self._counts["terminal_failures"] += self._pending_events_at_timeout + int(baseline_timed_out)
                self._counts["terminal_probes"] += self._pending_events_at_timeout + int(baseline_timed_out)
                self._status = "timeout"
                # Seal while holding the same lock used by worker result
                # replacement.  A probe that returns after the grace deadline
                # must not overwrite the deterministic timeout event.
                self._sealed = True
        elif self._disabled:
            suppressed = self._mark_pending_events(
                raw_episode,
                TestEventStatus.SKIPPED,
                "suppressed_after_terminal",
                "shadow runtime stopped after a terminal failure before the probe completed",
            )
            self._counts["suppressed_probes"] += suppressed
            self._status = "degraded"
        else:
            self._status = "completed"
        if self.pass_rate_mode and final_output is None and self._final_recovery_allowed():
            recovery_error = final_error
            if recovery_error is None:
                with self._lock:
                    recovery_error = (
                        self._errors[-1]
                        if self._errors
                        else "authoritative shadow verifier did not produce an outcome"
                    )
            final_output = self._recover_authoritative_final(recovery_error)
        with self._lock:
            self._sealed = True
        self._write_episode_summary(raw_episode)
        if self._final_verifier_outcome is not None:
            raw_episode.metadata["verifier_outcome"] = copy.deepcopy(self._final_verifier_outcome)
        if enriched_episode is not None:
            self.copy_results_to_episode(raw_episode, enriched_episode)
            if self._final_verifier_outcome is not None:
                enriched_episode.metadata["verifier_outcome"] = copy.deepcopy(self._final_verifier_outcome)
        if self.pass_rate_mode and final_output is None:
            if final_error is None:
                with self._lock:
                    prior_errors = list(self._errors)
                if prior_errors:
                    final_error = (
                        "authoritative shadow verifier unavailable after a prior "
                        f"shadow failure: {prior_errors[-1]}"
                    )
            raise ShadowFinalizationError(
                "shadow_verifier_failure",
                final_error or "authoritative shadow verifier did not produce an outcome",
                retryable=False,
                retry_scope="none",
                safe_for_final_recovery=False,
                diagnostics=self._final_failure_diagnostics(),
            )
        return final_output

    def seal_milestone_input(
        self,
        *,
        primary_verifier_finished_at_monotonic: float | None = None,
    ) -> dict[str, Any]:
        """Declare the DeNovo action stream complete without finalizing it.

        The worker keeps draining in its own thread.  Dropping the primary
        reference is deliberate: milestone-only execution must be able to
        outlive the primary sandbox and may not invoke final-only recovery.
        """

        if not self.milestone_only:
            raise RuntimeError("milestone-only handoff requested for legacy shadow")
        capture_required = False
        primary = None
        with self._lock:
            if self._milestone_handoff_snapshot is None:
                if self._milestone_input_sealing:
                    raise RuntimeError("concurrent DeNovo shadow handoff is not allowed")
                primary = self.primary
                if primary is None:
                    raise RuntimeError(
                        "primary sandbox is unavailable before DeNovo handoff snapshot"
                    )
                # A tail candidate has no next primary action.  Normal seal is
                # therefore the decision boundary that releases its probe.
                self._finish_probe_merge_run_locked()
                self._flush_probe_merge_group_locked()
                self._milestone_input_sealing = True
                capture_required = True
                journal = tuple(
                    copy.deepcopy(self._milestone_action_journal)
                )
            else:
                journal = self._sealed_milestone_action_journal or ()

        if capture_required:
            try:
                handoff_snapshot = self._capture_milestone_handoff_snapshot(
                    journal,
                    primary,
                )
            except BaseException:
                with self._lock:
                    self._milestone_input_sealing = False
                raise
            with self._lock:
                self._sealed_milestone_action_journal = journal
                self._milestone_handoff_snapshot = handoff_snapshot
                self._background_lifecycle["input_sealed_at_monotonic"] = (
                    time.monotonic()
                )
                self._background_lifecycle["queue_depth_at_handoff"] = int(
                    self._queue.qsize()
                )
                self._background_lifecycle["handoff_snapshot"] = copy.deepcopy(
                    handoff_snapshot
                )
                self._milestone_input_sealing = False

        with self._lock:
            if primary_verifier_finished_at_monotonic is not None:
                self._background_lifecycle[
                    "primary_verifier_finished_at_monotonic"
                ] = float(primary_verifier_finished_at_monotonic)
                self._background_lifecycle[
                    "completed_probes_at_primary_finish"
                ] = int(self._counts["probes_completed"])
                self._background_lifecycle[
                    "shadow_drained_at_primary_finish"
                ] = self.milestone_drained()
            self._queue_depth_at_finalize = max(
                self._queue_depth_at_finalize,
                self._queue.qsize(),
            )
            result = copy.deepcopy(self._milestone_handoff_snapshot or {})
        self._enqueue_stop()
        self.primary = None
        return result

    def milestone_drained(self) -> bool:
        thread = self._thread
        return thread is None or not thread.is_alive()

    def record_milestone_barrier_start(
        self,
        *,
        barrier_started_at_monotonic: float,
        deadline_monotonic: float,
    ) -> dict[str, Any]:
        """Snapshot progress when the optimizer batch reserves this lease.

        ``barrier_started_at_monotonic`` is the semantic start required by
        the training contract (the latest selected primary-verifier finish).
        The coordinator may observe that point later, so retain both times
        and the remaining shared budget.  Capturing probe counters here is
        essential: ``finalize_milestones_at_barrier`` may block until the
        deadline and must not mislabel work completed during that wait as
        pre-barrier work.
        """

        if not self.milestone_only:
            raise RuntimeError("batch milestone barrier requested for legacy shadow")
        observed_at = time.monotonic()
        with self._lock:
            self._background_lifecycle.update(
                {
                    "batch_barrier_started_at_monotonic": float(
                        barrier_started_at_monotonic
                    ),
                    "batch_barrier_observed_at_monotonic": observed_at,
                    "batch_barrier_deadline_monotonic": float(
                        deadline_monotonic
                    ),
                    "batch_barrier_budget_remaining_s": max(
                        0.0,
                        float(deadline_monotonic) - observed_at,
                    ),
                    "completed_probes_at_barrier_start": int(
                        self._counts["probes_completed"]
                    ),
                    "shadow_drained_at_barrier_start": self.milestone_drained(),
                    "queue_depth_at_barrier_start": int(self._queue.qsize()),
                }
            )
            return copy.deepcopy(self._background_lifecycle)

    def finalize_milestones_at_barrier(
        self,
        raw_episode: Episode,
        enriched_episode: Episode,
        *,
        barrier_started_at_monotonic: float,
        deadline_monotonic: float,
        timeout_reason: str = "batch_shadow_finalize_timeout",
    ) -> dict[str, Any]:
        """Seal one selected rollout at a shared optimizer-batch deadline."""

        if not self.milestone_only:
            raise RuntimeError("batch milestone finalization requested for legacy shadow")
        self.seal_milestone_input()
        with self._lock:
            barrier_was_recorded = (
                self._background_lifecycle.get(
                    "batch_barrier_started_at_monotonic"
                )
                == float(barrier_started_at_monotonic)
                and self._background_lifecycle.get(
                    "batch_barrier_deadline_monotonic"
                )
                == float(deadline_monotonic)
                and self._background_lifecycle.get(
                    "completed_probes_at_barrier_start"
                )
                is not None
            )
        if not barrier_was_recorded:
            self.record_milestone_barrier_start(
                barrier_started_at_monotonic=barrier_started_at_monotonic,
                deadline_monotonic=deadline_monotonic,
            )
        wait_started = time.perf_counter()
        thread = self._thread
        if thread is not None:
            remaining = max(0.0, float(deadline_monotonic) - time.monotonic())
            if remaining:
                thread.join(timeout=remaining)
        waited = time.perf_counter() - wait_started
        timed_out = bool(thread is not None and thread.is_alive())
        with self._lock:
            self._timings["finalize_wait_s"] += waited
            self._timings["post_primary_wait_s"] += waited
            self._background_lifecycle.update(
                {
                    "batch_barrier_started_at_monotonic": float(
                        barrier_started_at_monotonic
                    ),
                    "batch_barrier_deadline_monotonic": float(
                        deadline_monotonic
                    ),
                    "completed_probes_at_barrier": int(
                        self._counts["probes_completed"]
                    ),
                }
            )

        if timed_out:
            error = (
                "shadow milestone queue did not drain before the selected "
                "optimizer batch deadline"
            )
            self._disable(error, timeout_reason)
            with self._lock:
                self._finalize_timed_out = True
                self._forced_teardown_requested = True
                baseline_timed_out = self._mark_incomplete_baseline_timeout(
                    error,
                    failure_type=timeout_reason,
                )
                self._pending_events_at_timeout = self._mark_pending_events(
                    raw_episode,
                    TestEventStatus.TIMEOUT,
                    timeout_reason,
                    error,
                    timed_out=True,
                )
                self._counts["terminal_failures"] += (
                    self._pending_events_at_timeout + int(baseline_timed_out)
                )
                self._counts["terminal_probes"] += (
                    self._pending_events_at_timeout + int(baseline_timed_out)
                )
                self._status = "timeout"
                self._sealed = True
                self._background_lifecycle["finalize_disposition"] = "timeout"
        elif self._disabled:
            suppressed = self._mark_pending_events(
                raw_episode,
                TestEventStatus.SKIPPED,
                "suppressed_after_terminal",
                "shadow runtime stopped after a terminal failure",
            )
            with self._lock:
                self._counts["suppressed_probes"] += suppressed
                self._status = "degraded"
                self._sealed = True
                self._background_lifecycle["finalize_disposition"] = "degraded"
        else:
            with self._lock:
                self._status = "completed"
                self._sealed = True
                self._background_lifecycle["finalize_disposition"] = "completed"

        with self._lock:
            self._background_lifecycle["pending_probes_at_barrier"] = int(
                self._pending_events_at_timeout
            )
            self._background_lifecycle["finalized_at_monotonic"] = time.monotonic()
        self._write_episode_summary(raw_episode)
        self.copy_results_to_episode(raw_episode, enriched_episode)
        return copy.deepcopy(self._background_lifecycle)

    def cancel_milestones(
        self,
        raw_episode: Episode,
        enriched_episode: Episode,
        *,
        disposition: str,
    ) -> dict[str, Any]:
        """Cancel a filtered/requeued DeNovo shadow and preserve its audit."""

        if not self.milestone_only:
            raise RuntimeError("milestone cancellation requested for legacy shadow")
        reason = f"shadow_cancelled_{disposition}"
        error = f"shadow milestone work cancelled after group disposition={disposition}"
        with self._lock:
            unresolved_merge_steps = len(self._probe_merge_members)
        self._disable(error, reason)
        self._enqueue_stop()
        pending = unresolved_merge_steps + self._mark_pending_events(
            raw_episode,
            TestEventStatus.SKIPPED,
            reason,
            error,
        )
        with self._lock:
            self._counts["suppressed_probes"] += pending
            self._status = "cancelled"
            self._cancel_requested.set()
            self._sealed = True
            self._forced_teardown_requested = True
            self._background_lifecycle.update(
                {
                    "finalize_disposition": disposition,
                    "completed_probes_at_disposition": int(
                        self._counts["probes_completed"]
                    ),
                    "cancelled_pending_steps": int(pending),
                    "shadow_drained_at_disposition": self.milestone_drained(),
                    "finalized_at_monotonic": time.monotonic(),
                }
            )
        cancel_exec = getattr(self.shadow, "cancel_pending_exec", None)
        if callable(cancel_exec):
            cancel_exec()
        self._write_episode_summary(raw_episode)
        self.copy_results_to_episode(raw_episode, enriched_episode)
        return copy.deepcopy(self._background_lifecycle)

    def _run_authoritative_final(self, *, recovery: bool = False) -> EvalOutput:
        primary_state = self.harness._codeflow_capture_repo_snapshot(self.primary, self.task, self._baseline_ref)
        shadow_state = self.harness._codeflow_capture_repo_snapshot(self.shadow, self.task, self._baseline_ref)
        if (
            primary_state.status != RepoStateStatus.OK
            or shadow_state.status != RepoStateStatus.OK
            or not primary_state.fingerprint
            or primary_state.fingerprint != shadow_state.fingerprint
            or primary_state.baseline_ref != shadow_state.baseline_ref
        ):
            raise RuntimeError(f"primary/shadow final repository mismatch: primary={primary_state.fingerprint}:{primary_state.error}, shadow={shadow_state.fingerprint}:{shadow_state.error}")
        event = self._run_probe(
            f"{self.uid}:final",
            primary_state.fingerprint,
            None,
            None,
            baseline=False,
            allow_after_seal=recovery,
        )
        self._final_event = event.model_dump(mode="json")
        if event.status != TestEventStatus.COMPLETED or not event.trusted:
            raise RuntimeError(event.error or f"untrusted final verifier event: {event.failure_type or event.status.value}")
        restored = self.harness._codeflow_capture_repo_snapshot(self.shadow, self.task, self._baseline_ref)
        if restored.status != RepoStateStatus.OK or restored.fingerprint != primary_state.fingerprint:
            raise RuntimeError(f"final verifier did not restore the shadow repository: expected={primary_state.fingerprint}, observed={restored.fingerprint}")
        results = event.test_results
        passed = sum(results.get(name) == "PASSED" for name in self.authoritative_test_names)
        total = len(self.authoritative_test_names)
        failed = sum(results.get(name) in {"FAILED", "SKIPPED", "XFAIL", "XPASS"} for name in self.authoritative_test_names)
        errors = sum(results.get(name) in {"ERROR", "NOT_RUN", None} for name in self.authoritative_test_names)
        rate = passed / total
        outcome = {
            "schema_version": 1,
            "passed_count": passed,
            "total_count": total,
            "pass_rate": rate,
            "outcome_source": "shadow_verifier",
            "verifier_profile": "denovoswe_official",
        }
        self._final_verifier_outcome = outcome
        return EvalOutput(
            reward=rate,
            is_correct=passed == total and failed == 0 and errors == 0,
            signals=[Signal(name="acceptance_pass_rate", value=rate)],
            metadata={
                "verifier_outcome": dict(outcome),
                "test_results": dict(results),
                "failed_count": failed,
                "error_count": errors,
                "final_repo_fingerprint": primary_state.fingerprint,
            },
        )

    def copy_results_to_episode(self, raw_episode: Episode, enriched_episode: Episode) -> None:
        """Copy sealed results after the primary evaluator stops reading the episode."""
        self._copy_results(raw_episode, enriched_episode)
        self._write_episode_summary(enriched_episode)

    def abort(self) -> None:
        with self._lock:
            self._cancel_requested.set()
            if not self._sealed:
                self._disable("shadow runtime aborted", "aborted")
                self._status = "cancelled"
                self._sealed = True
            self._enqueue_stop()
            self._progress_condition.notify_all()
        cancel_exec = getattr(self.shadow, "cancel_pending_exec", None)
        if callable(cancel_exec):
            cancel_exec()

    def _check_cancelled(self) -> None:
        if self._cancel_requested.is_set():
            raise _ShadowCancelled("shadow runtime cancelled")

    def _observe_pending_cleanup(self) -> None:
        """Keep unresolved local workers/HTTP requests visible after stop grace."""
        from rllm.utils.diagnostic_events import emit_diagnostic

        if self._thread is not None:
            self._thread.join()
        while getattr(self.shadow, "pending_exec_count", 0):
            time.sleep(0.1)
        with self._lock:
            # This confirms local drain only; sandbox deletion has its own audit.
            self._cleanup_status = "worker_requests_drained"
        emit_diagnostic("shadow_cleanup", uid=self.uid, status=self._cleanup_status,
                        terminal_error=str(getattr(self, "_terminal_cleanup_error", None)))

    def join_after_close(self, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + max(0.0, timeout)
        if self._thread is not None:
            self._thread.join(timeout=max(0.0, deadline - time.monotonic()))
        # The caller worker can stop before its cancelled HTTP daemon unwinds.
        # Give both the SAME grace budget; do not immediately fail that race.
        while getattr(self.shadow, "pending_exec_count", 0):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(.02, remaining))
        worker_alive = self._thread is not None and self._thread.is_alive()
        pending_exec = getattr(self.shadow, "pending_exec_count", 0)
        if worker_alive or pending_exec:
            from rllm.utils.diagnostic_events import emit_diagnostic

            with self._lock:
                self._cleanup_status = "pending"
                if self._cleanup_observer is None:
                    emit_diagnostic("shadow_cleanup", uid=self.uid, status="pending",
                                    worker_alive=worker_alive, pending_exec_requests=pending_exec)
                    self._cleanup_observer = threading.Thread(
                        target=self._observe_pending_cleanup,
                        name=f"rllm-shadow-cleanup-{self.uid}", daemon=True,
                    )
                    self._cleanup_observer.start()
            raise RolloutInfrastructureError(
                "shadow_cleanup_unconfirmed", "shadow worker or HTTP requests remain after sandbox close",
                retryable=False, stage="teardown", retry_scope="none",
                diagnostics={"uid": self.uid, "cancel_requested": self._cancel_requested.is_set(),
                             "worker_alive": worker_alive, "pending_exec_requests": pending_exec},
            )
        with self._lock:
            self._cleanup_status = "worker_requests_drained"
        failure = getattr(self, "_terminal_cleanup_error", None)
        if failure is not None:
            raise failure

    def summary(self) -> dict[str, Any]:
        with self._lock:
            probe_merge_run_histogram = (
                self._probe_merge_same_changed_files_run_length_histogram
            )
            probe_merge_run_count = sum(probe_merge_run_histogram.values())
            probe_merge_run_steps = sum(
                length * count
                for length, count in probe_merge_run_histogram.items()
            )
            if self._final_recovery.get("status") == "succeeded":
                quality_status = "recovered_final"
            elif self._status in {"degraded", "timeout"} or self._disabled:
                quality_status = "terminal_degraded"
            elif self._counts["recoverable_probes"]:
                quality_status = "recoverable_gaps"
            else:
                quality_status = "complete"
            return {
                "schema_version": SHADOW_SCHEMA_VERSION,
                "enabled": True,
                "status": self._status,
                "cleanup_status": self._cleanup_status,
                "quality_status": quality_status,
                "verifier_asset_provenance": getattr(self.shadow, "_rllm_verifier_asset_provenance", {}),
                "helper_runtimes": list(getattr(self.shadow, "_rllm_helper_runtimes", {}).values()),
                "repository_helper_provenance": list(getattr(self.shadow, "_rllm_repository_helper_provenance", {}).values()),
                "replay_state_capture_diagnostics": copy.deepcopy(self._replay_state_capture_diagnostics),
                "sandbox_diagnostics": getattr(self.shadow, "runtime_diagnostics", {}),
                "baseline_source": (
                    self.baseline_source
                    if self.bug_repair_pass_count_mode or self.pass_rate_mode
                    else "runtime_probe"
                ),
                "baseline_probe_succeeded": (
                    self.baseline_probe_succeeded
                    if self.bug_repair_pass_count_mode or self.pass_rate_mode
                    else bool(
                        isinstance(self._baseline_event, dict)
                        and self._baseline_event.get("trusted") is True
                    )
                ),
                "materialized_baseline_fallback": bool(
                    self.bug_repair_pass_count_mode
                    and self.baseline_source == "materialized_fallback"
                ),
                "safe_baseline_available": bool(
                    (self.bug_repair_pass_count_mode or self.pass_rate_mode)
                    and self.baseline_passed is not None
                ),
                "benign_baseline_partial": bool(
                    self.bug_repair_pass_count_mode
                    and isinstance(self._baseline_event, dict)
                    and self._baseline_event.get("failure_type")
                    == "baseline_count_incomplete"
                ),
                "actionable_baseline_failure": bool(
                    self.bug_repair_pass_count_mode
                    and isinstance(self._baseline_event, dict)
                    and self._baseline_event.get("trusted") is not True
                    and self._baseline_event.get("failure_type")
                    not in {None, "baseline_count_incomplete"}
                ),
                "result_parser": self.result_parser,
                "language": self.language or None,
                "log_parser": self.log_parser or None,
                "parser_vendor_revision": self.parser_vendor_revision or None,
                "parser_vendor_sha256": self.parser_vendor_sha256 or None,
                "test_set_policy": self.test_set_policy,
                "verification_potential_mode": self.verification_potential_mode,
                "verification_potential_policy": self.verification_potential_policy,
                "count_plan_hash": self._count_plan_hash,
                "contract_fill_policy": CONTRACT_FILL_POLICY,
                "verification_result_mode": self.verification_result_mode,
                "verification_count_contract": self._verification_count_contract(),
                "materialized_baseline_passed": (
                    self.materialized_baseline_passed
                    if self.bug_repair_pass_count_mode or self.pass_rate_mode
                    else None
                ),
                "runtime_baseline_passed": self.runtime_baseline_passed,
                "baseline_passed": self.baseline_passed,
                "max_additional_passed": (
                    len(self.authoritative_test_names) - self.baseline_passed
                    if self.baseline_passed is not None
                    else None
                ),
                "baseline_collector": self.baseline_collector,
                "partition_timeout_recovery": self.partition_timeout_recovery,
                "partition_recovery_command_timeout": self.partition_recovery_command_timeout,
                "partition_recovery_max_commands": self.partition_recovery_max_commands,
                "partition_adapter": (self.partition_adapter.name if self.partition_adapter is not None else None),
                "partition_calibration_status": self.partition_calibration_status,
                "partition_calibration_reason": self.partition_calibration_reason,
                "partition_calibration_diagnostics": copy.deepcopy(self.partition_calibration_diagnostics),
                "authoritative_test_count": len(self.authoritative_test_names),
                "test_results_path": (self.test_results_path if self.result_parser in SWEREBENCH_V2_RESULT_PARSERS else None),
                "parser_transport": self._parser_transport,
                "parser_command_length": self._parser_command_length,
                "baseline_test_event": copy.deepcopy(self._baseline_event),
                "verification_potential_unavailable_reason": self._verification_unavailable_reason(),
                "final_test_event": copy.deepcopy(self._final_event),
                "final_verifier_outcome": copy.deepcopy(self._final_verifier_outcome),
                "final_recovery": copy.deepcopy(self._final_recovery),
                "baseline_reference": copy.deepcopy(self._baseline_reference),
                "probe_merge_enabled": self.probe_merge_enabled,
                "probe_merge_max_steps": (
                    self.probe_merge_max_steps
                    if self.probe_merge_enabled
                    else None
                ),
                "probe_merge_group_size_histogram": {
                    str(size): count
                    for size, count in sorted(
                        self._probe_merge_group_size_histogram.items()
                    )
                },
                "probe_merge_same_changed_files_run_length_histogram": {
                    str(length): count
                    for length, count in sorted(
                        probe_merge_run_histogram.items()
                    )
                },
                "probe_merge_same_changed_files_runs": probe_merge_run_count,
                "probe_merge_same_changed_files_steps": probe_merge_run_steps,
                "probe_merge_same_changed_files_run_length_min": (
                    min(probe_merge_run_histogram)
                    if probe_merge_run_histogram
                    else None
                ),
                "probe_merge_same_changed_files_run_length_mean": (
                    probe_merge_run_steps / probe_merge_run_count
                    if probe_merge_run_count
                    else None
                ),
                "probe_merge_same_changed_files_run_length_max": (
                    max(probe_merge_run_histogram)
                    if probe_merge_run_histogram
                    else None
                ),
                "probe_merge_decision_wait_samples": len(
                    self._probe_merge_decision_wait_samples_s
                ),
                "probe_merge_decision_wait_s_min": (
                    min(self._probe_merge_decision_wait_samples_s)
                    if self._probe_merge_decision_wait_samples_s
                    else None
                ),
                "probe_merge_decision_wait_s_mean": (
                    sum(self._probe_merge_decision_wait_samples_s)
                    / len(self._probe_merge_decision_wait_samples_s)
                    if self._probe_merge_decision_wait_samples_s
                    else None
                ),
                "probe_merge_decision_wait_s_max": (
                    max(self._probe_merge_decision_wait_samples_s)
                    if self._probe_merge_decision_wait_samples_s
                    else None
                ),
                "probe_merge_pending_steps": len(self._probe_merge_members),
                **self._counts,
                "terminal_replay": copy.deepcopy(self._terminal_replay),
                "replay_warnings": copy.deepcopy(self._replay_warnings),
                "restore_warnings": copy.deepcopy(self._restore_warnings),
                "timings": dict(self._timings),
                "failure_counts": dict(self._failure_counts),
                "recoverable_failure_counts": dict(self._recoverable_failure_counts),
                "timeout_policy": (
                    "background_until_optimizer_batch_barrier"
                    if self.milestone_only
                    else "per_probe_timeout_plus_bounded_finalize_wait"
                ),
                "background_lifecycle": copy.deepcopy(
                    self._background_lifecycle
                ),
                "finalize_stall_timeout": self.finalize_stall_timeout,
                "finalize_timed_out": self._finalize_timed_out,
                "progress_sequence": self._progress_sequence,
                "finalize_progress_resets": self._finalize_progress_resets,
                "finalize_watchdog_reports": self._finalize_watchdog_reports,
                "current_job_type": self._current_job_type,
                "last_progress_age_s": max(
                    0.0,
                    time.monotonic() - self._last_progress_at,
                ),
                "queue_depth_at_finalize": self._queue_depth_at_finalize,
                "pending_events_at_timeout": self._pending_events_at_timeout,
                "forced_teardown_requested": self._forced_teardown_requested,
                # Retain the old keys as explicit non-enforced values for
                # consumers that predate schema v2.
                "probe_timeout": self.probe_execution_timeout,
                "total_timeout": None,
                "duration": max(0.0, time.monotonic() - self._started_at),
                "errors": list(self._errors),
            }

    def _verification_unavailable_reason(self) -> str | None:
        if self.bug_repair_pass_count_mode and self.baseline_passed is not None:
            return None
        if self.runtime_baseline_results is not None:
            return None
        if isinstance(self._baseline_event, dict):
            reason = self._baseline_event.get("failure_type") or self._baseline_event.get("error")
            if reason:
                return str(reason)
        return str(next(iter(self._failure_counts), "shadow_baseline_unavailable"))

    def _accept_runtime_baseline(self, results: dict[str, str]) -> None:
        """Install one complete runtime probe as this rollout's reward baseline."""
        self.runtime_baseline_results = dict(results)
        self._confirmed_p2p_regressions = {name: False for name, target in self.target_results.items() if results.get(name) == target}
        self._confirmed_p2p_regression_count = 0
        if self.pass_rate_mode:
            runtime_passed = sum(
                str(results.get(name, "NOT_RUN")).upper() == "PASSED"
                for name in self.authoritative_test_names
            )
            self.runtime_baseline_passed = runtime_passed
            self.baseline_passed = runtime_passed
            self.baseline_source = "runtime_probe"
            self.baseline_probe_succeeded = True
            self.baseline_collector = "official_exact_vector"
            self._current_potential = self._potential(results)
        names = sorted(set(self.materialized_baseline_results) | set(results))
        mismatches = [name for name in names if self.materialized_baseline_results.get(name) != results.get(name)]
        if set(self.materialized_baseline_results) != set(results):
            status = "incomplete"
        elif mismatches:
            status = "mismatch"
        else:
            status = "match"
        limit = 50
        self._baseline_reference = {
            "status": status,
            "mismatch_count": len(mismatches),
            "mismatch_tests": mismatches[:limit],
            "truncated": len(mismatches) > limit,
        }

    def _accept_runtime_count_baseline(
        self,
        observation: TestCountObservation,
    ) -> None:
        """Install a complete aggregate observation as ``C0``."""

        if not observation.complete or observation.reported != len(
            self.authoritative_test_names
        ):
            raise ValueError("runtime count baseline must be complete")
        effective_passed = observation.passed
        self.runtime_baseline_passed = effective_passed
        self.baseline_passed = effective_passed
        self.baseline_source = "runtime_probe"
        self.baseline_probe_succeeded = True
        self.baseline_collector = observation.collector
        # This compatibility marker is intentionally not used to compute the
        # scalar reward. It keeps older availability consumers from treating a
        # valid count baseline as absent.
        self.runtime_baseline_results = dict(self.materialized_baseline_results)
        self._current_potential = 0.0
        stable_count = sum(
            self.materialized_baseline_results.get(name)
            == self.target_results.get(name)
            for name in self.authoritative_test_names
        )
        self._count_confirmed_p2p_regressions = max(
            0,
            stable_count - observation.passed,
        )
        mismatch = effective_passed != self.materialized_baseline_passed
        self._counts["baseline_probe_successes"] += 1
        if mismatch:
            self._counts["baseline_count_mismatches"] += 1
        self._baseline_reference = {
            "status": "count_mismatch" if mismatch else "count_match",
            "mismatch_count": abs(
                effective_passed - self.materialized_baseline_passed
            ),
            "mismatch_tests": [],
            "truncated": False,
            "materialized_passed": self.materialized_baseline_passed,
            "runtime_passed": effective_passed,
        }

    def _accept_materialized_count_fallback(self, event: TestEvent) -> None:
        """Install materialized ``C0`` after a restored baseline probe gap."""

        if resolve_test_event_continuity(event) == TestEventContinuityStatus.LOST:
            # The probe result is unavailable, but the exact post-probe
            # fingerprint proves repository continuity. Reclassify only that
            # continuity dimension; the failed status/evidence is retained.
            event.continuity_status = TestEventContinuityStatus.RECOVERABLE_GAP
            self._counts["terminal_probes"] = max(
                0,
                self._counts["terminal_probes"] - 1,
            )
            self._counts["terminal_failures"] = max(
                0,
                self._counts["terminal_failures"] - 1,
            )
            self._counts["recoverable_probes"] += 1
            failure_type = event.failure_type or "baseline_probe"
            self._recoverable_failure_counts[failure_type] = (
                self._recoverable_failure_counts.get(failure_type, 0) + 1
            )
        self.runtime_baseline_passed = None
        self.baseline_passed = self.materialized_baseline_passed
        self.baseline_source = "materialized_fallback"
        self.baseline_probe_succeeded = False
        self.baseline_collector = None
        self.runtime_baseline_results = dict(self.materialized_baseline_results)
        self._current_potential = 0.0
        stable_count = sum(
            self.materialized_baseline_results.get(name)
            == self.target_results.get(name)
            for name in self.authoritative_test_names
        )
        self._count_confirmed_p2p_regressions = max(
            0,
            stable_count - self.materialized_baseline_passed,
        )
        self._counts["materialized_baseline_fallbacks"] += 1
        self._baseline_reference = {
            "status": "materialized_fallback",
            "mismatch_count": 0,
            "mismatch_tests": [],
            "truncated": False,
            "materialized_passed": self.materialized_baseline_passed,
            "runtime_passed": None,
            "probe_failure_type": event.failure_type,
        }

    def _can_use_materialized_count_fallback(
        self,
        event: TestEvent,
        repo_state: str,
    ) -> bool:
        """Require exact post-probe restore before preserving a failed C0 probe."""

        terminal_state_failures = {
            "repo_state_unavailable",
            "probe_checkpoint",
            "probe_before_mismatch",
            "verifier_state_mismatch",
            "replay_state_mismatch",
            "shadow_setup",
            "count_collector_setup",
        }
        return bool(
            self.bug_repair_pass_count_mode
            and event.failure_type not in terminal_state_failures
            and event.status != TestEventStatus.STATE_MISMATCH
            and event.restore_confirmation
            in {
                RestoreConfirmation.CONFIRMED,
                RestoreConfirmation.FINGERPRINT_FALLBACK,
            }
            and event.shadow_repo_state == repo_state
        )

    def _verification_count_contract(self) -> dict[str, Any] | None:
        if not (
            self.bug_repair_pass_count_mode or self.pass_rate_mode
        ) or self.baseline_passed is None:
            return None
        return {
            "schema_version": 1,
            "expected": len(self.authoritative_test_names),
            "materialized_baseline_passed": self.materialized_baseline_passed,
            "runtime_baseline_passed": self.runtime_baseline_passed,
            "baseline_passed": self.baseline_passed,
            "max_additional_passed": (
                len(self.authoritative_test_names) - self.baseline_passed
            ),
            "baseline_source": self.baseline_source,
            "baseline_probe_succeeded": self.baseline_probe_succeeded,
            "collector": self.baseline_collector,
        }

    def _effective_count_passed(
        self,
        observation: TestCountObservation,
    ) -> int:
        """Apply the fixed F2P/P2P NOT_RUN policy to one count step."""

        partitions = observation.partition_counts
        if observation.complete or partitions is None:
            if observation.complete and partitions is not None:
                self._count_confirmed_p2p_regressions = (
                    partitions.p2p.failed
                    + partitions.p2p.errored
                    + partitions.p2p.skipped
                )
            elif observation.complete:
                stable_count = sum(
                    self.materialized_baseline_results.get(name)
                    == self.target_results.get(name)
                    for name in self.authoritative_test_names
                )
                self._count_confirmed_p2p_regressions = max(
                    self._count_confirmed_p2p_regressions,
                    stable_count - observation.passed,
                )
            return observation.passed
        explicit_regressions = (
            partitions.p2p.failed
            + partitions.p2p.errored
            + partitions.p2p.skipped
        )
        if partitions.p2p.not_run == 0:
            self._count_confirmed_p2p_regressions = explicit_regressions
        else:
            self._count_confirmed_p2p_regressions = max(
                self._count_confirmed_p2p_regressions,
                explicit_regressions,
            )
        return partitions.f2p.passed + max(
            0,
            partitions.p2p.expected
            - self._count_confirmed_p2p_regressions,
        )

    def _can_fill_all_f2p_baseline_contract(self) -> bool:
        """Whether every authoritative test is known to fail at baseline."""

        authoritative = set(self.authoritative_test_names)
        return bool(
            not self.pass_rate_mode
            and self._all_f2p_contract
            and self.language.strip().casefold()
            in _ALL_F2P_CONTRACT_FILL_LANGUAGES
            and authoritative
            and authoritative == set(self.materialized_baseline_results)
            and authoritative == set(self.target_results)
            and all(
                self.materialized_baseline_results[name] != self.target_results[name]
                for name in authoritative
            )
        )

    @staticmethod
    def _go_structural_test_id(name: str) -> str:
        match = re.search(
            r"((?:Test|Example|Fuzz)[^\s:]*(?:/[^\s]+)*)$",
            name.strip(),
        )
        return match.group(1) if match else name.strip()

    def _independent_extra_tests(self, names: list[str]) -> list[str]:
        """Return extras that are not Go parent/child execution structure."""

        if not names or self.language.strip().casefold() != "go":
            return list(names)
        authoritative = {
            self._go_structural_test_id(name)
            for name in self.authoritative_test_names
        }
        independent: list[str] = []
        for name in names:
            candidate = self._go_structural_test_id(name)
            if any(
                candidate.startswith(expected + "/")
                or expected.startswith(candidate + "/")
                for expected in authoritative
            ):
                continue
            independent.append(name)
        return independent

    def _can_contract_assist_partial_vector(
        self,
        observed_results: dict[str, str],
        missing: list[str],
        observed_extra: list[str],
        ignored_invalid_extra: list[str],
        *,
        baseline: bool,
        parsed: dict[str, Any],
    ) -> bool:
        """Conservatively fill only missing F2P nodes from a proven run.

        At baseline, every observed F2P must agree with the materialized
        diagnostic contract and every P2P must have run and passed.  After an
        edit, all P2P must still be observed, while missing F2P nodes are
        represented as reward-neutral ``NOT_RUN`` rather than inferred states.
        """

        missing_set = set(missing)
        observed_names = set(observed_results)
        return bool(
            not self.pass_rate_mode
            and self.result_parser == SWEREBENCH_V2_OFFICIAL_RESULT_PARSER
            and missing_set
            and missing_set <= self._f2p_contract
            and self._p2p_contract <= observed_names
            and parsed.get("artifact_complete") is False
            and parsed.get("execution_evidence") is True
            and parsed.get("partial_failure_type")
            == "deterministic_runner_partial"
            and not ignored_invalid_extra
            and not self._independent_extra_tests(observed_extra)
            and all(
                self.materialized_baseline_results.get(name)
                != self.target_results.get(name)
                for name in self._f2p_contract
            )
            and all(
                self.materialized_baseline_results.get(name)
                == self.target_results.get(name)
                for name in self._p2p_contract
            )
            and (
                not baseline
                or (
                    all(observed_results.get(name) == "PASSED" for name in self._p2p_contract)
                    and all(
                        observed_results.get(name)
                        == self.materialized_baseline_results.get(name)
                        for name in self._f2p_contract & observed_names
                    )
                )
            )
        )

    def _record_terminal_replay(
        self,
        raw: dict[str, Any],
        failure_type: str,
        *,
        before: RepoSnapshot | None = None,
        after: RepoSnapshot | None = None,
        replay: Any | None = None,
        state_mismatch_paths: list[str] | None = None,
    ) -> None:
        """Keep the first terminal replay failure even when no probe was due."""
        with self._lock:
            if self._terminal_replay is not None or self._sealed:
                return
            primary_status = str(raw.get("execution_status") or "")
            shadow_status = str(getattr(getattr(replay, "status", None), "value", "") or "")
            timeout = None
            arguments = raw.get("normalized_arguments")
            if isinstance(arguments, dict) and isinstance(arguments.get("timeout"), int | float):
                timeout = float(arguments["timeout"])
            expected_changes = _changed_file_paths(raw.get("changed_files"))
            observed_changes = _changed_file_paths(changed_files_between(before, after)) if before is not None and after is not None else []
            self._terminal_replay = {
                "failure_type": failure_type,
                "action_id": str(raw.get("action_id") or "") or None,
                "turn_id": raw.get("turn_id") if isinstance(raw.get("turn_id"), int) else None,
                "tool_name": str(raw.get("tool_name") or ""),
                "timeout": timeout,
                "primary": {
                    "status": primary_status or None,
                    "exit_code": raw.get("exit_code"),
                    "duration": raw.get("duration"),
                },
                "shadow": {
                    "status": shadow_status or None,
                    "exit_code": getattr(replay, "exit_code", None),
                    "duration": getattr(replay, "duration", None),
                    "error": getattr(replay, "error", None),
                },
                "expected_repo_state_before": raw.get("repo_state_before"),
                "observed_repo_state_before": before.fingerprint if before is not None else None,
                "expected_repo_state_after": raw.get("repo_state_after"),
                "observed_repo_state_after": after.fingerprint if after is not None else None,
                "expected_changed_files": expected_changes,
                "observed_changed_files": observed_changes,
                "state_mismatch_paths": sorted(set(state_mismatch_paths or [])),
            }

    def _is_nonterminal_replay_observation_mismatch(
        self,
        raw: dict[str, Any],
        replay: Any,
        before: RepoSnapshot,
        after: RepoSnapshot,
        *,
        bash_attempts_match: bool,
    ) -> bool:
        if raw.get("tool_name") != "execute_bash" or not bash_attempts_match:
            return False
        primary_exit = raw.get("exit_code")
        shadow_exit = getattr(replay, "exit_code", None)
        if (
            raw.get("execution_status") not in {ExecutionStatus.SUCCESS.value, ExecutionStatus.ERROR.value}
            or getattr(replay, "status", None) not in {ExecutionStatus.SUCCESS, ExecutionStatus.ERROR}
            or not isinstance(primary_exit, int)
            or not isinstance(shadow_exit, int)
            or primary_exit == shadow_exit
        ):
            return False
        if raw.get("policy_violations") or getattr(replay, "policy_violations", None):
            return False
        if "edit" in set(raw.get("shell_behaviors") or []):
            return False
        if raw.get("attempted_changed_files") or getattr(replay, "attempted_changed_files", None):
            return False
        if raw.get("changed_files") or raw.get("repo_changed") is True:
            return False
        if (
            before.status != RepoStateStatus.OK
            or after.status != RepoStateStatus.OK
            or before.fingerprint != raw.get("repo_state_before")
            or after.fingerprint != raw.get("repo_state_after")
            or before.fingerprint != after.fingerprint
        ):
            return False
        arguments = raw.get("normalized_arguments")
        command = arguments.get("command") if isinstance(arguments, dict) else None
        if not isinstance(command, str):
            return False
        repository_root = self.harness._repository_root(self.task)
        if (
            detect_preconfigured_environment_violation(
                command,
                repository_root,
            )
            is not None
        ):
            return False
        return is_bounded_read_only_head_pipeline(
            command,
            repository_root,
        ) or is_proven_side_effect_free_shell_command(
            command,
            repository_root,
        )

    def _record_replay_warning(
        self,
        raw: dict[str, Any],
        replay: Any,
        before: RepoSnapshot,
        after: RepoSnapshot,
    ) -> None:
        arguments = raw.get("normalized_arguments")
        command = arguments.get("command") if isinstance(arguments, dict) else None
        warning = {
            "failure_type": "replay_observation_mismatch",
            "action_id": str(raw.get("action_id") or "") or None,
            "turn_id": raw.get("turn_id") if isinstance(raw.get("turn_id"), int) else None,
            "primary_status": raw.get("execution_status"),
            "primary_exit_code": raw.get("exit_code"),
            "shadow_status": getattr(getattr(replay, "status", None), "value", None),
            "shadow_exit_code": getattr(replay, "exit_code", None),
            "command_evidence": str(command or "")[:1000],
            "repo_state_before": before.fingerprint,
            "repo_state_after": after.fingerprint,
        }
        with self._lock:
            self._counts["replay_observation_mismatches"] += 1
            if len(self._replay_warnings) < 20:
                self._replay_warnings.append(warning)
        logger.warning(
            "[%s] tolerated side-effect-free replay observation mismatch: primary=%s shadow=%s",
            self.uid,
            raw.get("exit_code"),
            getattr(replay, "exit_code", None),
        )

    def _cleanup_replay_recovery_bundle(self, job: _ReplayJob) -> None:
        bundle = job.recovery_bundle
        if bundle is None:
            return
        Path(bundle.local_path).unlink(missing_ok=True)
        with self._lock:
            self._replay_recovery_bundle_paths.discard(bundle.local_path)

    def _cleanup_all_replay_recovery_bundles(self) -> None:
        with self._lock:
            paths = tuple(self._replay_recovery_bundle_paths)
            self._replay_recovery_bundle_paths.clear()
        for path in paths:
            Path(path).unlink(missing_ok=True)

    def _try_recover_replay_state(
        self,
        job: _ReplayJob,
    ) -> tuple[RepoSnapshot | None, str | None]:
        self._check_cancelled()
        bundle = job.recovery_bundle
        if bundle is None or self._baseline_ref is None:
            return None, job.recovery_capture_error
        with self._lock:
            self._counts["replay_state_recovery_attempts"] += 1
        try:
            recovered = _restore_shadow_repository_from_bundle(
                self.harness,
                self.task,
                self.shadow,
                self._baseline_ref,
                bundle,
                uid=f"{self.uid}:{job.replay_event.get('action_id')}:restore",
            )
        except Exception as exc:
            with self._lock:
                self._check_cancelled()
                self._counts["replay_state_recovery_failures"] += 1
            logger.warning(
                "[%s] exact replay-state recovery failed for %s: %s",
                self.uid,
                job.replay_event.get("action_id"),
                exc,
            )
            return None, str(exc)[:1000]
        with self._lock:
            self._check_cancelled()
            self._counts["replay_state_recovery_successes"] += 1
        logger.info(
            "[%s] recovered exact primary repository state for replay %s",
            self.uid,
            job.replay_event.get("action_id"),
        )
        return recovered, None

    def _baseline_singleflight_key(self, initial: RepoSnapshot) -> str | None:
        if self.pass_rate_mode or self.bug_repair_pass_count_mode:
            return None
        payload = {
            "task_dir": str(self.task.task_dir.resolve()),
            "task_id": self.task.id,
            "repo_state": initial.fingerprint,
            "baseline_ref": initial.baseline_ref,
            "result_parser": self.result_parser,
            "parser_vendor_revision": self.parser_vendor_revision,
            "parser_vendor_sha256": self.parser_vendor_sha256,
            "verifier_profile": self.verifier_profile,
            "probe_execution_timeout": self.probe_execution_timeout,
            "partition_timeout_recovery": self.partition_timeout_recovery,
            "partition_recovery_command_timeout": self.partition_recovery_command_timeout,
            "partition_recovery_max_commands": self.partition_recovery_max_commands,
            "materialized_baseline_results": self.materialized_baseline_results,
            "target_results": self.target_results,
            "sandbox_type": (
                f"{type(self.shadow).__module__}.{type(self.shadow).__qualname__}"
            ),
        }
        if self._uses_non_python_evidence():
            payload["non_python_evidence_digest"] = hashlib.sha256(
                Path(__file__).with_name("non_python_verification.py").read_bytes()
            ).hexdigest()
        return hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def _claim_baseline_singleflight(
        self,
        key: str,
    ) -> tuple[_BaselineSingleflight, bool]:
        waited = False
        while True:
            with _BASELINE_SINGLEFLIGHT_LOCK:
                flight = _BASELINE_SINGLEFLIGHTS.get(key)
                if flight is None:
                    flight = _BaselineSingleflight(ready=threading.Event())
                    _BASELINE_SINGLEFLIGHTS[key] = flight
                    self._counts["baseline_singleflight_leaders"] += 1
                    return flight, True
                if flight.result is not None:
                    self._counts["baseline_singleflight_hits"] += 1
                    return flight, False
            if not waited:
                with self._lock:
                    self._counts["baseline_singleflight_waits"] += 1
                waited = True
            flight.ready.wait()

    @staticmethod
    def _finish_baseline_singleflight(
        key: str,
        flight: _BaselineSingleflight,
        result: dict[str, Any] | None,
    ) -> None:
        with _BASELINE_SINGLEFLIGHT_LOCK:
            current = _BASELINE_SINGLEFLIGHTS.get(key)
            if current is not flight:
                return
            if result is None:
                _BASELINE_SINGLEFLIGHTS.pop(key, None)
            else:
                flight.result = copy.deepcopy(result)
                completed = [
                    name
                    for name, entry in _BASELINE_SINGLEFLIGHTS.items()
                    if entry.result is not None and name != key
                ]
                while len(completed) >= _BASELINE_SINGLEFLIGHT_MAX_COMPLETED:
                    _BASELINE_SINGLEFLIGHTS.pop(completed.pop(0), None)
            flight.ready.set()

    def _install_shared_baseline(
        self,
        initial: RepoSnapshot,
        payload: dict[str, Any],
    ) -> TestEvent:
        event_payload = copy.deepcopy(payload["event"])
        event_payload.update(
            probe_id=f"{self.uid}:baseline:shared",
            repo_state=initial.fingerprint,
            shadow_repo_state=initial.fingerprint,
            duration=0.0,
            probe_attempts=0,
        )
        evidence = dict(event_payload.get("failure_evidence") or {})
        evidence["baseline_singleflight_cache_hit"] = True
        event_payload["failure_evidence"] = evidence
        event = TestEvent.model_validate(event_payload)
        if self._uses_non_python_evidence():
            self._non_python_test_owners = dict(evidence.get("test_owners") or {})
        if payload.get("partition_calibrated"):
            if not isinstance(self._swe_instance, dict):
                raise RuntimeError("shared partition baseline is missing instance.json")
            adapter = self._build_shadow_partition_adapter(
                recovery_command_budget=self.partition_recovery_max_commands,
            )
            self._upload_partition_instances(adapter)
            self.partition_adapter = adapter
            self.verification_result_mode = "partition_counts"
            self.partition_calibration_status = "calibrated_shared"
            self.partition_calibration_reason = None
            self.partition_calibration_diagnostics = copy.deepcopy(
                payload.get("partition_calibration_diagnostics") or {}
            )
            self.partition_calibration_diagnostics[
                "baseline_singleflight_cache_hit"
            ] = True
        self._accept_runtime_baseline(event.test_results)
        self._current_potential = float(event.test_potential_after or 0.0)
        with self._lock:
            self._counts["trusted_probes"] += 1
        self._note_progress("baseline_probe")
        return event

    def _initialize_baseline(self) -> bool:
        self._check_cancelled()
        self._set_current_job("baseline")
        initial = self.harness._codeflow_capture_repo_snapshot(
            self.shadow,
            self.task,
            self._baseline_ref,
        )
        self._check_cancelled()
        if initial.status != RepoStateStatus.OK or not initial.fingerprint:
            self._replace_baseline_event(
                self._failure_test_event(
                    probe_id=f"{self.uid}:baseline",
                    repo_state=initial.fingerprint or "unavailable",
                    status=TestEventStatus.ERROR,
                    failure_type="repo_state_unavailable",
                    error=(
                        initial.error
                        or "initial shadow repository state is unavailable"
                    ),
                )
            )
            self._counts["terminal_probes"] += 1
            self._disable(
                "initial shadow repository state is unavailable",
                "shadow_baseline_unavailable",
            )
            return False

        flight_key = self._baseline_singleflight_key(initial)
        flight: _BaselineSingleflight | None = None
        flight_leader = False
        if flight_key is not None:
            flight, flight_leader = self._claim_baseline_singleflight(flight_key)
            if not flight_leader:
                assert flight.result is not None
                try:
                    baseline_event = self._install_shared_baseline(
                        initial,
                        flight.result,
                    )
                except Exception as exc:
                    with self._lock:
                        self._counts[
                            "baseline_singleflight_install_failures"
                        ] += 1
                    logger.warning(
                        "[%s] shared baseline installation failed; running an "
                        "independent baseline: %s",
                        self.uid,
                        exc,
                    )
                    flight_key = None
                    flight = None
                else:
                    self._replace_baseline_event(baseline_event)
                    return True

        flight_finished = False
        try:
            baseline_event = self._run_probe(
                f"{self.uid}:baseline",
                initial.fingerprint,
                None,
                None,
                baseline=True,
            )
            self._check_cancelled()
            if (
                not self.bug_repair_pass_count_mode
                and self._should_try_partition_calibration(baseline_event)
            ):
                calibrated = self._calibrate_partition(
                    initial.fingerprint,
                    baseline_event,
                )
                if calibrated is not None:
                    baseline_event = calibrated
            self._replace_baseline_event(baseline_event)
            if (
                baseline_event.status != TestEventStatus.COMPLETED
                or not baseline_event.trusted
            ):
                if self._can_use_materialized_count_fallback(
                    baseline_event,
                    initial.fingerprint,
                ):
                    self._accept_materialized_count_fallback(baseline_event)
                    self._replace_baseline_event(baseline_event)
                    logger.warning(
                        "[%s] pass-count baseline probe failed after an exact "
                        "repository restore; continuing with materialized C0=%d: "
                        "%s; error=%s; evidence=%s",
                        self.uid,
                        self.materialized_baseline_passed,
                        baseline_event.failure_type or baseline_event.error,
                        str(baseline_event.error or "")[:500],
                        json.dumps(
                            baseline_event.failure_evidence or {},
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )[:1000],
                    )
                    return True
                self._disable(
                    baseline_event.error or "baseline probe failed",
                    "shadow_baseline_unavailable",
                )
                return False
            if flight_key is not None and flight is not None and flight_leader:
                self._finish_baseline_singleflight(
                    flight_key,
                    flight,
                    {
                        "event": baseline_event.model_dump(mode="json"),
                        "partition_calibrated": bool(
                            self.partition_adapter is not None
                            and self.verification_result_mode == "partition_counts"
                        ),
                        "partition_calibration_diagnostics": copy.deepcopy(
                            self.partition_calibration_diagnostics
                        ),
                    },
                )
                flight_finished = True
            return True
        finally:
            if (
                flight_key is not None
                and flight is not None
                and flight_leader
                and not flight_finished
            ):
                self._finish_baseline_singleflight(flight_key, flight, None)

    def _worker(self) -> None:
        try:
            self._check_cancelled()
            if not self._initialize_baseline():
                return

            while True:
                self._check_cancelled()
                item = self._queue.get()
                if item is _STOP:
                    break
                with self._lock:
                    if self._sealed:
                        break
                assert isinstance(item, _ReplayJob | _ProbeJob)
                if self._disabled:
                    if isinstance(item, _ProbeJob) or item.trigger_probe:
                        marked = self._set_job_failure_events(
                            item,
                            TestEventStatus.SKIPPED,
                            "suppressed_after_terminal",
                            "shadow runtime is disabled after a terminal failure",
                        )
                        if marked:
                            self._counts["suppressed_probes"] += 1
                    if isinstance(item, _ReplayJob):
                        self._cleanup_replay_recovery_bundle(item)
                    continue
                if isinstance(item, _ProbeJob):
                    self._set_current_job("probe")
                    self._process_probe_job(item)
                    continue
                self._set_current_job("replay")
                try:
                    self._process_job(item)
                finally:
                    self._cleanup_replay_recovery_bundle(item)
        except _ShadowCancelled:
            pass
        except Exception as exc:
            if self._cancel_requested.is_set():
                logger.debug("[%s] cancelled shadow worker stopped: %s", self.uid, exc)
            else:
                logger.exception("[%s] shadow worker failed", self.uid)
                self._disable(f"shadow worker failed: {exc}", "worker")
        finally:
            self._cleanup_all_replay_recovery_bundles()
            with self._progress_condition:
                self._current_job_type = None
                self._current_job_started_at = None
                self._progress_condition.notify_all()

    def _process_job(self, job: _ReplayJob) -> None:
        self._check_cancelled()
        # Replay exclusively from the handoff-time copy.  Test results are
        # written to the original Episode event only after the replay/probe
        # reaches a terminal state.
        raw = job.replay_event
        expected_before = str(raw["repo_state_before"])
        expected_after = str(raw["repo_state_after"])
        before = self.harness._codeflow_capture_repo_snapshot(self.shadow, self.task, self._baseline_ref)
        self._check_cancelled()
        if before.status != RepoStateStatus.OK or before.fingerprint != expected_before:
            self._counts["state_mismatches"] += 1
            mismatch_paths = sorted(before.files)
            self._record_terminal_replay(
                raw,
                "replay_before_mismatch",
                before=before,
                state_mismatch_paths=mismatch_paths,
            )
            if job.probe_events:
                self._set_job_failure_events(
                    job,
                    TestEventStatus.STATE_MISMATCH,
                    "replay_before_mismatch",
                    f"expected {expected_before}, observed {before.fingerprint}",
                    state_mismatch_paths=mismatch_paths,
                    count_terminal=True,
                )
            self._disable("shadow state does not match action repo_state_before", "replay_before_mismatch")
            return

        replay_started = time.perf_counter()
        replay = self.harness._codeflow_execute_validated(
            str(raw["tool_name"]),
            copy.deepcopy(raw.get("normalized_arguments") or {}),
            self.shadow,
            self.task,
            self._history,
        )
        with self._lock:
            if self._sealed:
                return
            self._timings["replay_s"] += time.perf_counter() - replay_started
        after = self.harness._codeflow_capture_repo_snapshot(self.shadow, self.task, self._baseline_ref)
        expected_status = str(raw.get("execution_status") or "")
        bash_attempts_match = True
        if raw.get("tool_name") == "execute_bash":
            primary_attempts = _changed_file_identities(raw.get("attempted_changed_files"))
            shadow_attempts = _changed_file_identities(replay.attempted_changed_files)
            bash_attempts_match = primary_attempts == shadow_attempts
        execution_mismatch = replay.status.value != expected_status or (raw.get("tool_name") == "execute_bash" and replay.exit_code != raw.get("exit_code")) or not bash_attempts_match

        with self._lock:
            if self._sealed:
                return
        after_matches = after.status == RepoStateStatus.OK and after.fingerprint == expected_after
        if not after_matches:
            self._counts["state_mismatches"] += 1
        nonterminal_observation_mismatch = bool(
            after_matches
            and execution_mismatch
            and self._is_nonterminal_replay_observation_mismatch(
                raw,
                replay,
                before,
                after,
                bash_attempts_match=bash_attempts_match,
            )
        )
        recovery_error: str | None = None
        if not after_matches or (
            execution_mismatch and not nonterminal_observation_mismatch
        ):
            recovered_after, recovery_error = self._try_recover_replay_state(job)
            if recovered_after is not None:
                after = recovered_after
                after_matches = True
                execution_mismatch = False
                nonterminal_observation_mismatch = False
        if not after_matches:
            expected_paths = _changed_file_identities(raw.get("changed_files"))
            observed_paths = _changed_file_identities(changed_files_between(before, after))
            mismatch_paths = sorted(path for path, _, _ in expected_paths | observed_paths)
            self._record_terminal_replay(
                raw,
                "replay_after_mismatch",
                before=before,
                after=after,
                replay=replay,
                state_mismatch_paths=mismatch_paths,
            )
            if job.probe_events:
                self._set_job_failure_events(
                    job,
                    TestEventStatus.STATE_MISMATCH,
                    "replay_after_mismatch",
                    f"expected {expected_after}, observed {after.fingerprint}",
                    state_mismatch_paths=mismatch_paths,
                    count_terminal=True,
                    failure_evidence={
                        "replay_state_capture_error": job.recovery_capture_error,
                        "replay_state_capture_diagnostics": job.recovery_capture_diagnostics,
                        "replay_state_recovery_error": recovery_error,
                    },
                )
            self._disable("shadow state does not match action repo_state_after", "replay_after_mismatch")
            return

        if execution_mismatch and not nonterminal_observation_mismatch:
            self._counts["replay_execution_mismatches"] += 1
            if replay.status == ExecutionStatus.TIMEOUT or expected_status == ExecutionStatus.TIMEOUT.value:
                self._counts["replay_timeout_mismatches"] += 1
            observed_paths = _changed_file_paths(changed_files_between(before, after))
            self._record_terminal_replay(
                raw,
                "replay_execution_mismatch",
                before=before,
                after=after,
                replay=replay,
                state_mismatch_paths=observed_paths,
            )
            if job.probe_events:
                self._set_job_failure_events(
                    job,
                    TestEventStatus.ERROR,
                    "replay_execution_mismatch",
                    "primary and shadow execution differ: "
                    f"status/exit={expected_status}/{raw.get('exit_code')} vs {replay.status.value}/{replay.exit_code}; "
                    f"attempted_changes_match={bash_attempts_match}",
                    count_terminal=True,
                    failure_evidence={
                        "replay_state_capture_error": job.recovery_capture_error,
                        "replay_state_capture_diagnostics": job.recovery_capture_diagnostics,
                        "replay_state_recovery_error": recovery_error,
                    },
                )
            self._disable("shadow action execution differs from primary", "replay_execution_mismatch")
            return
        if nonterminal_observation_mismatch:
            self._record_replay_warning(raw, replay, before, after)
        self._counts["replays_completed"] += 1
        self._note_progress("replay")
        if job.trigger_probe:
            self._set_current_job("probe")
            self._run_job_probe(job, raw)

    def _process_probe_job(self, job: _ProbeJob) -> None:
        self._check_cancelled()
        self._run_job_probe(job, job.probe_event)

    def _run_job_probe(
        self,
        job: _ReplayJob | _ProbeJob,
        raw: dict[str, Any],
    ) -> None:
        event = self._run_probe(
            str(
                (raw.get("test_event") or {}).get("probe_id")
                or f"{raw['action_id']}:probe"
            ),
            str(raw["repo_state_after"]),
            str(raw.get("action_id") or "") or None,
            int(raw.get("turn_id"))
            if isinstance(raw.get("turn_id"), int)
            else None,
            baseline=False,
            agent_changed_paths=_changed_file_paths(raw.get("changed_files")),
        )
        self._replace_job_test_events(job, event)
        if (
            resolve_test_event_continuity(event)
            == TestEventContinuityStatus.LOST
        ):
            self._disable(
                event.error or "shadow probe failed",
                event.failure_type or "probe",
            )

    def _should_try_partition_calibration(self, event: TestEvent) -> bool:
        if self.pass_rate_mode:
            return False
        if event.result_completeness == "full" and set(event.test_results) == set(self.target_results):
            self.partition_calibration_status = "not_needed_full_vector"
            return False
        if self._swe_instance is None:
            self.partition_calibration_status = "unsupported"
            self.partition_calibration_reason = "missing_instance_json"
            return False
        if (
            event.failure_type == "verifier_timeout"
            and not self.partition_timeout_recovery
        ):
            self.partition_calibration_status = "not_attempted_policy"
            self.partition_calibration_reason = event.failure_type
            return False
        if event.failure_type in _PARTITION_INFRA_FAILURES:
            self.partition_calibration_status = "not_attempted_infrastructure"
            self.partition_calibration_reason = event.failure_type
            return False
        return True

    def _build_shadow_partition_adapter(self, **kwargs: Any) -> PartitionAdapter:
        assert self._swe_instance is not None
        if not self._uses_non_python_evidence():
            return build_partition_adapter(self._swe_instance, **kwargs)
        paths = ()
        config = self._swe_instance.get("install_config") or {}
        if "ava" in str(config.get("test_cmd", "")):
            # Resolve real paths once from the private shadow checkout. Never
            # guess AVA's title selector from its path-prefixed display ID.
            script = (
                "import os,json,sys; root=sys.argv[1]; paths=[]\n"
                "for base,dirs,files in os.walk(root):\n"
                " dirs[:]=[d for d in dirs if d not in {'.git','node_modules','.yarn'}]\n"
                " for name in files:\n"
                "  if name.endswith(('.js','.cjs','.mjs','.ts','.tsx')): paths.append(os.path.relpath(os.path.join(base,name),root))\n"
                "print(json.dumps(paths))"
            )
            try:
                raw = self.shadow.exec(
                    self._helper_python() + " -c " + shlex.quote(script) + " " + shlex.quote(self.harness._repository_root(self.task)),
                    timeout=30, user=self.task.metadata.get("verifier_user"),
                )
                paths = tuple(json.loads(str(raw)))
            except Exception as exc:
                raise PartitionAdapterUnsupported("ava_test_file_inventory_unavailable") from exc
        return build_non_python_partition_adapter(self._swe_instance, ava_test_files=paths)

    def _upload_partition_instances(self, adapter: PartitionAdapter) -> None:
        encoded = [
            encode_instance(shard.instance)
            for partition in ("f2p", "p2p")
            for shard in adapter.shards(partition)
        ]
        identity = hashlib.blake2b(
            "\n".join(encoded).encode("utf-8"),
            digest_size=10,
        ).hexdigest()
        remote_dir = f"/tmp/rllm-shadow/partition-{identity}"
        with tempfile.TemporaryDirectory(prefix="rllm-partition-") as local_dir:
            for partition in ("f2p", "p2p"):
                paths: list[str] = []
                for index, shard in enumerate(adapter.shards(partition)):
                    local_path = f"{local_dir}/{partition}-{index}.instance.json"
                    with open(local_path, "w", encoding="utf-8") as handle:
                        handle.write(encode_instance(shard.instance))
                    remote_path = f"{remote_dir}/{partition}-{index}.instance.json"
                    self.shadow.upload_file(local_path, remote_path)
                    paths.append(remote_path)
                self._partition_instance_paths[partition] = tuple(paths)

    def _calibrate_partition(
        self,
        repo_state: str,
        original_baseline: TestEvent,
    ) -> TestEvent | None:
        started = time.perf_counter()
        self._counts["partition_calibration_attempts"] += 1
        self.partition_calibration_status = "attempting"
        try:
            assert self._swe_instance is not None
            adapter = self._build_shadow_partition_adapter(
                baseline_log=original_baseline.log_tail,
                recovery_command_budget=(
                    self.partition_recovery_max_commands
                    if original_baseline.failure_type == "verifier_timeout"
                    else None
                ),
            )
            expected_f2p = {name for name, target in self.target_results.items() if self.materialized_baseline_results.get(name) != target}
            expected_p2p = set(self.target_results) - expected_f2p
            if set(adapter.f2p) != expected_f2p or set(adapter.p2p) != expected_p2p:
                raise ValueError("partition_dataset_contract_mismatch")
            self.partition_calibration_diagnostics = {
                "runner": adapter.name,
                "runner_plan": [
                    {
                        "runner": invocation.runner,
                        "command_index": invocation.command_index,
                    }
                    for invocation in adapter.plan.invocations
                ],
                "f2p_shards": len(adapter.f2p_shards),
                "p2p_shards": len(adapter.p2p_shards),
                "combined_shards": len(adapter.combined_shards),
                "f2p_expected": len(adapter.f2p),
                "p2p_expected": len(adapter.p2p),
            }
            command_count = len(adapter.f2p_shards) + len(adapter.p2p_shards)
            self.partition_calibration_diagnostics.update(
                execution_mode="f2p_p2p_isolated_counts_v1",
                command_count=command_count,
                command_budget=self.partition_recovery_max_commands,
            )
            if command_count > self.partition_recovery_max_commands:
                raise PartitionAdapterUnsupported(
                    "partition_command_budget_exceeded:"
                    f"{command_count}>{self.partition_recovery_max_commands}"
                )
            self._upload_partition_instances(adapter)
            try:
                f2p_run = self._run_partition_group(
                    adapter,
                    "f2p",
                    repo_state,
                    f"{self.uid}:baseline:f2p",
                )
            except Exception:
                self.partition_calibration_diagnostics["failed_group"] = "f2p"
                raise
            try:
                p2p_run = self._run_partition_group(
                    adapter,
                    "p2p",
                    repo_state,
                    f"{self.uid}:baseline:p2p",
                )
            except Exception:
                self.partition_calibration_diagnostics["failed_group"] = "p2p"
                raise
            f2p_observation = f2p_run.observation
            p2p_observation = p2p_run.observation
            self.partition_calibration_diagnostics.update(
                f2p_observed=f2p_observation.reported,
                p2p_observed=p2p_observation.reported,
                f2p_evidence_source=f2p_observation.evidence_source,
                p2p_evidence_source=p2p_observation.evidence_source,
            )
            if (
                f2p_observation.reported != len(adapter.f2p)
                or f2p_observation.counts.passed != 0
                or f2p_observation.counts.skipped != 0
            ):
                self.partition_calibration_diagnostics.update(
                    failed_group="f2p",
                    f2p_observed=f2p_observation.reported,
                )
                raise ValueError("f2p_baseline_calibration_mismatch")
            if p2p_observation.reported != len(adapter.p2p) or p2p_observation.counts.passed != len(adapter.p2p):
                self.partition_calibration_diagnostics.update(
                    failed_group="p2p",
                    p2p_observed=p2p_observation.reported,
                )
                raise ValueError("p2p_baseline_calibration_mismatch")
        except PartitionAdapterUnsupported as exc:
            self.partition_calibration_status = "unsupported"
            self.partition_calibration_reason = exc.reason
            self.partition_calibration_diagnostics.setdefault("failed_stage", "adapter")
            logger.debug("[%s] partition selector unsupported: %s", self.uid, exc)
            return None
        except Exception as exc:
            self.partition_calibration_status = "failed"
            self.partition_calibration_reason = str(exc)[:500]
            self.partition_calibration_diagnostics.setdefault("failed_stage", "calibration")
            logger.debug("[%s] partition calibration failed: %s", self.uid, exc)
            return None
        finally:
            self._timings["partition_calibration_s"] += time.perf_counter() - started

        self.partition_adapter = adapter
        self.verification_result_mode = "partition_counts"
        self.partition_calibration_status = "calibrated"
        self.partition_calibration_reason = None
        self._counts["partition_calibration_successes"] += 1
        self._accept_runtime_baseline(self.materialized_baseline_results)
        partitions = TestPartitionResult(
            f2p=f2p_observation.counts,
            p2p=p2p_observation.counts,
        )
        partition_runs = (f2p_run, p2p_run)
        partition_exit_code = next(
            (
                run.exit_code
                for run in reversed(partition_runs)
                if run.exit_code not in {0, None}
            ),
            partition_runs[-1].exit_code,
        )
        completeness = "partition_counts_full" if partitions.f2p.is_complete and partitions.p2p.is_complete else "partition_counts_partial"
        return self._result_event(
            f"{self.uid}:baseline:partition",
            repo_state,
            TestEventStatus.COMPLETED,
            dict(self.materialized_baseline_results),
            None,
            None,
            sum(run.duration for run in partition_runs),
            partition_exit_code,
            repo_state,
            [],
            [],
            trusted=True,
            potential_before=0.0,
            potential_after=0.0,
            log_tail="\n".join(run.log_tail for run in partition_runs)[-2000:],
            continuity_status=TestEventContinuityStatus.INTACT,
            result_source=TestResultSource.PARTITION_COUNTS,
            result_completeness=completeness,
            partition_counts=partitions,
        )

    def _read_partition_artifacts(self, result_path: str, log_path: str) -> tuple[dict[str, Any] | None, str]:
        command = f"{self._helper_python()} -c {shlex.quote(_READ_PARTITION_ARTIFACT_SCRIPT)} {shlex.quote(result_path)} {shlex.quote(log_path)}"
        raw = self.shadow.exec(
            command,
            timeout=30,
            user=self.task.metadata.get("verifier_user"),
        )
        payload = json.loads(str(raw).strip())
        if not isinstance(payload, dict):
            raise ValueError("partition_artifact_reader_returned_non_object")
        artifact = payload.get("artifact")
        return artifact if isinstance(artifact, dict) else None, str(payload.get("log") or "")

    def _run_partition_group(
        self,
        adapter: PartitionAdapter,
        partition: str,
        repo_state: str,
        probe_id: str,
    ) -> _PartitionGroupRun:
        self._check_cancelled()
        shards = adapter.shards(partition)
        expected_names = (
            adapter.f2p
            if partition == "f2p"
            else adapter.p2p
            if partition == "p2p"
            else (*adapter.f2p, *adapter.p2p)
        )
        paths = self._partition_instance_paths.get(partition)
        if not shards:
            if expected_names:
                raise RuntimeError(f"partition_shards_missing:{partition}")
            return _PartitionGroupRun(
                observation=PartitionObservation(
                    counts=TestPartitionCounts(expected=0),
                    named_results={},
                    extra_named_tests=(),
                    reported=0,
                    evidence_source="empty_partition",
                    selector_verified=True,
                ),
                duration=0.0,
                exit_code=0,
                timed_out=False,
                log_tail="",
                restore_confirmation=RestoreConfirmation.CONFIRMED,
            )
        if paths is None or len(paths) != len(shards):
            raise RuntimeError(f"partition_instance_missing:{partition}")
        runs = [
            self._run_partition_shard(
                adapter,
                partition,
                shard.runner,
                shard.test_ids,
                paths[index],
                repo_state,
                f"{probe_id}:shard-{index + 1}-of-{len(shards)}",
            )
            for index, shard in enumerate(shards)
        ]
        named_results: dict[str, str] = {}
        duplicate_named_results: set[str] = set()
        for run in runs:
            for name, status in run.observation.named_results.items():
                if name in named_results:
                    duplicate_named_results.add(name)
                    continue
                named_results[name] = status
        assigned_names = [name for shard in shards for name in shard.test_ids]
        assignments_overlap = len(set(assigned_names)) != len(assigned_names)
        if assignments_overlap:
            # Workspace scripts sometimes receive a broadcast selector because
            # benchmark labels do not encode package ownership. Authenticated
            # isolated counts may still be summed across those invocations;
            # baseline calibration requires the partition-wide total and state
            # to match exactly, and any total above the contract fails closed.
            # Without aggregate evidence, retain the older exact-name
            # requirement.
            if all(run.observation.selector_verified for run in runs):
                reported = sum(run.observation.reported for run in runs)
                if reported > len(expected_names):
                    raise ValueError(
                        "partition_overlapping_runner_count_exceeds_expected:"
                        f"{reported}>{len(expected_names)}"
                    )
                counts = TestPartitionCounts(
                    expected=len(expected_names),
                    passed=sum(run.observation.counts.passed for run in runs),
                    failed=sum(run.observation.counts.failed for run in runs),
                    errored=sum(run.observation.counts.errored for run in runs),
                    skipped=sum(run.observation.counts.skipped for run in runs),
                    not_run=len(expected_names) - reported,
                )
                evidence_source = "overlap_aggregate:" + "+".join(
                    dict.fromkeys(
                        run.observation.evidence_source for run in runs
                    )
                )
                selector_verified = True
            else:
                if duplicate_named_results:
                    raise ValueError(
                        "partition_duplicate_named_test:"
                        + sorted(duplicate_named_results)[0]
                    )
                statuses = list(named_results.values())
                counts = TestPartitionCounts(
                    expected=len(expected_names),
                    passed=statuses.count("PASSED"),
                    failed=statuses.count("FAILED"),
                    errored=statuses.count("ERROR"),
                    skipped=statuses.count("SKIPPED"),
                    not_run=len(expected_names) - len(named_results),
                )
                reported = len(named_results)
                evidence_source = "named_results"
                selector_verified = False
        else:
            if duplicate_named_results:
                raise ValueError(
                    "partition_duplicate_named_test:"
                    + sorted(duplicate_named_results)[0]
                )
            expected_total = sum(run.observation.counts.expected for run in runs)
            if expected_total != len(expected_names):
                raise ValueError(
                    "partition_shard_expected_mismatch:"
                    f"{expected_total}!={len(expected_names)}"
                )
            counts = TestPartitionCounts(
                expected=expected_total,
                passed=sum(run.observation.counts.passed for run in runs),
                failed=sum(run.observation.counts.failed for run in runs),
                errored=sum(run.observation.counts.errored for run in runs),
                skipped=sum(run.observation.counts.skipped for run in runs),
                not_run=sum(run.observation.counts.not_run for run in runs),
            )
            reported = sum(run.observation.reported for run in runs)
            evidence_source = "+".join(
                dict.fromkeys(run.observation.evidence_source for run in runs)
            )
            selector_verified = all(
                run.observation.selector_verified for run in runs
            )
        return _PartitionGroupRun(
            observation=PartitionObservation(
                counts=counts,
                named_results=named_results,
                extra_named_tests=tuple(
                    sorted({name for run in runs for name in run.observation.extra_named_tests})
                ),
                reported=reported,
                evidence_source=evidence_source,
                selector_verified=selector_verified,
            ),
            duration=sum(run.duration for run in runs),
            exit_code=next((run.exit_code for run in reversed(runs) if run.exit_code not in {0, None}), runs[-1].exit_code),
            timed_out=any(run.timed_out for run in runs),
            log_tail="\n".join(run.log_tail for run in runs)[-1000:],
            restore_confirmation=(
                RestoreConfirmation.FINGERPRINT_FALLBACK
                if any(run.restore_confirmation == RestoreConfirmation.FINGERPRINT_FALLBACK for run in runs)
                else RestoreConfirmation.CONFIRMED
            ),
        )

    @staticmethod
    def _partition_view(
        run: _PartitionGroupRun,
        expected_names: tuple[str, ...],
    ) -> PartitionObservation:
        expected = set(expected_names)
        named = {
            name: status
            for name, status in run.observation.named_results.items()
            if name in expected
        }
        statuses = list(named.values())
        return PartitionObservation(
            counts=TestPartitionCounts(
                expected=len(expected_names),
                passed=statuses.count("PASSED"),
                failed=statuses.count("FAILED"),
                errored=statuses.count("ERROR"),
                skipped=statuses.count("SKIPPED"),
                not_run=len(expected_names) - len(named),
            ),
            named_results=named,
            extra_named_tests=(),
            reported=len(named),
        )

    def _run_partition_shard(
        self,
        adapter: PartitionAdapter,
        partition: str,
        runner: str,
        expected_names: tuple[str, ...],
        instance_path: str,
        repo_state: str,
        probe_id: str,
    ) -> _PartitionGroupRun:
        self._check_cancelled()
        started = time.perf_counter()
        self._counts["partition_group_calls"] += 1
        safe_probe = re.sub(r"[^A-Za-z0-9_.-]", "_", probe_id)[-100:]
        result_path = f"/tmp/rllm/{safe_probe}.results.json"
        log_path = f"/tmp/rllm/{safe_probe}.output.txt"
        reward_path = f"/tmp/rllm/{safe_probe}.reward.json"
        marker = "__RLLM_PARTITION_RC__"
        command_timeout = min(
            self.probe_execution_timeout,
            self.partition_recovery_command_timeout,
        )
        native_cleanup = (
            "rm -f /tmp/rllm/partition-jest-results.json; "
            if self._uses_non_python_evidence() else ""
        )
        command = (
            native_cleanup
            + f"rm -f {shlex.quote(result_path)} {shlex.quote(log_path)} "
            f"{shlex.quote(reward_path)}; mkdir -p /tmp/rllm; "
            f"{self._node_yarn_guard_shell()}"
            f"export RLLM_INSTANCE_JSON={shlex.quote(instance_path)}; "
            f"export RLLM_TEST_RESULTS_JSON={shlex.quote(result_path)}; "
            f"export RLLM_TEST_OUTPUT={shlex.quote(log_path)}; "
            f"export RLLM_REWARD_JSON={shlex.quote(reward_path)}; "
            "timeout -k 30s "
            f"{command_timeout:g}s bash /tests/test.sh; rc=$?; "
            f"printf '\n{marker}%s\n' \"$rc\""
        )
        before = self.harness._codeflow_capture_repo_snapshot(self.shadow, self.task, self._baseline_ref)
        self._check_cancelled()
        if before.status != RepoStateStatus.OK or before.fingerprint != repo_state:
            raise RuntimeError("partition_state_mismatch_before")
        checkpoint_user = self.task.metadata.get("verifier_user") or self.task.metadata.get("agent_user")
        checkpoint, checkpoint_error = self.harness._codeflow_create_repo_checkpoint(
            self.shadow,
            self.task,
            user=checkpoint_user,
        )
        self._check_cancelled()
        if checkpoint is None:
            raise RuntimeError("partition_checkpoint_unavailable:" + str(checkpoint_error or "unknown"))
        raw_output = ""
        execution_error: BaseException | None = None
        artifact: dict[str, Any] | None = None
        log = ""
        try:
            raw_output = self.shadow.exec(
                command,
                timeout=command_timeout + min(30.0, self.probe_stall_grace),
                user=self.task.metadata.get("verifier_user"),
            )
        except Exception as exc:
            execution_error = exc
        self._check_cancelled()
        try:
            artifact, log = self._read_partition_artifacts(result_path, log_path)
        except Exception as exc:
            logger.debug("[%s] unable to read %s partition artifacts: %s", self.uid, partition, exc)
        self._check_cancelled()
        after = self.harness._codeflow_capture_repo_snapshot(self.shadow, self.task, self._baseline_ref)
        self._check_cancelled()
        restore_error = self.harness._codeflow_restore_repo_checkpoint(
            checkpoint,
            self.shadow,
            self.task,
            user=checkpoint_user,
        )
        self._check_cancelled()
        self.harness._codeflow_discard_repo_checkpoint(
            checkpoint,
            self.shadow,
            self.task,
            user=checkpoint_user,
        )
        self._check_cancelled()
        restored = self.harness._codeflow_capture_repo_snapshot(self.shadow, self.task, self._baseline_ref)
        self._check_cancelled()
        fingerprint_matches = restored.status == RepoStateStatus.OK and restored.fingerprint == repo_state
        confirmation = RestoreConfirmation.CONFIRMED
        if restore_error is not None and str(getattr(restore_error, "kind", "")) == "protocol" and fingerprint_matches:
            confirmation = RestoreConfirmation.FINGERPRINT_FALLBACK
        elif restore_error is not None or not fingerprint_matches:
            raise RuntimeError(f"partition_state_mismatch_after:expected={repo_state} observed={restored.fingerprint} restore={restore_error}")
        if changed_files_between(before, after):
            self._counts["verifier_side_effects_restored"] += 1
        match = re.search(rf"{re.escape(marker)}(\d+)\s*$", str(raw_output))
        exit_code = int(match.group(1)) if match else None
        timed_out = exit_code == 124 or _looks_like_timeout_error(execution_error)
        if execution_error is not None and not timed_out:
            raise RuntimeError(f"partition_probe_execution:{execution_error}")
        if match is None and not timed_out:
            raise RuntimeError("partition_probe_exit_marker_missing")
        infrastructure_failure = _partition_infrastructure_failure(artifact, log)
        if infrastructure_failure is not None:
            raise RuntimeError(infrastructure_failure)
        if self._uses_non_python_evidence() and runner in {"jest", "mocha", "ava", "vitest"}:
            from rllm.harnesses.non_python_verification import canonicalize_partition_artifact

            artifact = canonicalize_partition_artifact(
                artifact, set(expected_names), getattr(self, "_non_python_test_owners", {}),
            )
        observation = observe_partition(
            runner,
            expected_names,
            artifact,
            log,
            partition=partition,
        )
        if observation.extra_named_tests:
            raise ValueError("partition_extra_named_tests:" + ",".join(observation.extra_named_tests[:10]))
        duration = time.perf_counter() - started
        self._counts["partition_group_successes"] += 1
        if not observation.counts.is_complete:
            self._counts["partition_group_partial"] += 1
        if timed_out:
            self._counts["partition_group_timeouts"] += 1
        self._timings["partition_probe_s"] += duration
        self._note_progress("partition_probe")
        return _PartitionGroupRun(
            observation=observation,
            duration=duration,
            exit_code=exit_code,
            timed_out=timed_out,
            log_tail=log[-1000:],
            restore_confirmation=confirmation,
        )

    def _run_partition_probe(
        self,
        probe_id: str,
        repo_state: str,
        action_id: str | None,
        turn_id: int | None,
        *,
        probe_attempts: int,
    ) -> TestEvent:
        adapter = self.partition_adapter
        if adapter is None:
            return self._failure_test_event(
                probe_id,
                repo_state,
                TestEventStatus.ERROR,
                "partition_adapter_missing",
                "partition-count mode has no calibrated adapter",
                action_id,
                turn_id,
                continuity_status=TestEventContinuityStatus.LOST,
            )
        try:
            f2p_run = self._run_partition_group(
                adapter,
                "f2p",
                repo_state,
                f"{probe_id}:f2p",
            )
            p2p_run = self._run_partition_group(
                adapter,
                "p2p",
                repo_state,
                f"{probe_id}:p2p",
            )
            f2p_observation = f2p_run.observation
            p2p_observation = p2p_run.observation
        except _ShadowCancelled:
            raise
        except Exception as exc:
            lost = "state_mismatch" in str(exc)
            return self._failure_test_event(
                probe_id,
                repo_state,
                TestEventStatus.STATE_MISMATCH if lost else TestEventStatus.ERROR,
                "partition_state_mismatch" if lost else "partition_probe_unavailable",
                str(exc),
                action_id,
                turn_id,
                continuity_status=(TestEventContinuityStatus.LOST if lost else TestEventContinuityStatus.RECOVERABLE_GAP),
                probe_attempts=probe_attempts,
            )
        partitions = TestPartitionResult(
            f2p=f2p_observation.counts,
            p2p=p2p_observation.counts,
        )
        potential_before = self._current_potential
        potential_after = self._partition_potential(partitions)
        self._current_potential = potential_after
        full = partitions.f2p.is_complete and partitions.p2p.is_complete
        partition_runs = (f2p_run, p2p_run)
        partition_exit_code = next(
            (
                run.exit_code
                for run in reversed(partition_runs)
                if run.exit_code not in {0, None}
            ),
            partition_runs[-1].exit_code,
        )
        partition_timed_out = any(run.timed_out for run in partition_runs)
        restore_confirmation = (
            RestoreConfirmation.FINGERPRINT_FALLBACK
            if any(
                run.restore_confirmation == RestoreConfirmation.FINGERPRINT_FALLBACK
                for run in partition_runs
            )
            else RestoreConfirmation.CONFIRMED
        )
        return self._result_event(
            probe_id,
            repo_state,
            TestEventStatus.COMPLETED,
            {},
            action_id,
            turn_id,
            sum(run.duration for run in partition_runs),
            partition_exit_code,
            repo_state,
            [],
            [],
            trusted=True,
            failure_type=("verifier_timeout" if partition_timed_out else None),
            potential_before=potential_before,
            potential_after=potential_after,
            log_tail="\n".join(run.log_tail for run in partition_runs)[-2000:],
            continuity_status=TestEventContinuityStatus.INTACT,
            result_source=TestResultSource.PARTITION_COUNTS,
            probe_attempts=probe_attempts,
            restore_confirmation=restore_confirmation,
            timed_out=partition_timed_out,
            result_completeness=("partition_counts_full" if full else "partition_counts_partial"),
            partition_counts=partitions,
        )

    @staticmethod
    def _count_partition_observation_is_auditable(
        observation: PartitionObservation,
        expected_names: tuple[str, ...],
    ) -> bool:
        """Require authenticated aggregate evidence or an exact named vector.

        A partial named vector cannot prove that a selector excluded unrelated
        tests. It is accepted only when the private grader authenticated the
        selector. A complete exact named vector remains a safe fallback.
        """

        return bool(
            observation.selector_verified
            or (
                observation.reported == len(expected_names)
                and set(observation.named_results) == set(expected_names)
                and not observation.extra_named_tests
            )
        )

    def _partition_timeout_adapter(
        self,
        *,
        expected_f2p: set[str],
        expected_p2p: set[str],
        execution_mode: str,
    ) -> PartitionAdapter:
        """Build and upload one bounded adapter without changing normal mode."""

        if self.partition_adapter is not None:
            if (
                set(self.partition_adapter.f2p) != expected_f2p
                or set(self.partition_adapter.p2p) != expected_p2p
            ):
                raise PartitionAdapterUnsupported(
                    "partition_runtime_contract_mismatch"
                )
            return self.partition_adapter
        if not isinstance(self._swe_instance, dict):
            raise PartitionAdapterUnsupported("missing_instance_json")
        adapter = build_partition_adapter(
            self._swe_instance,
            recovery_command_budget=self.partition_recovery_max_commands,
        )
        authoritative = set(self.authoritative_test_names)
        if (
            set(adapter.f2p) != expected_f2p
            or set(adapter.p2p) != expected_p2p
            or expected_f2p & expected_p2p
            or expected_f2p | expected_p2p != authoritative
        ):
            raise PartitionAdapterUnsupported(
                "partition_dataset_contract_mismatch"
            )
        command_count = len(adapter.f2p_shards) + len(adapter.p2p_shards)
        if command_count > self.partition_recovery_max_commands:
            raise PartitionAdapterUnsupported(
                "partition_command_budget_exceeded:"
                f"{command_count}>{self.partition_recovery_max_commands}"
            )
        self._upload_partition_instances(adapter)
        self.partition_adapter = adapter
        self.partition_calibration_status = "timeout_recovery_ready"
        self.partition_calibration_reason = None
        self.partition_calibration_diagnostics = {
            "execution_mode": execution_mode,
            "runner": adapter.name,
            "command_count": command_count,
            "command_budget": self.partition_recovery_max_commands,
            "f2p_expected": len(adapter.f2p),
            "p2p_expected": len(adapter.p2p),
        }
        return adapter

    def _count_partition_timeout_adapter(self) -> PartitionAdapter:
        return self._partition_timeout_adapter(
            expected_f2p=set(self._f2p_contract),
            expected_p2p=set(self._p2p_contract),
            execution_mode="count_timeout_partition_recovery_v1",
        )

    def _normalized_partition_timeout_adapter(self) -> PartitionAdapter:
        if self.runtime_baseline_results is None:
            raise PartitionAdapterUnsupported("runtime_baseline_unavailable")
        expected_f2p = {
            name
            for name, target in self.target_results.items()
            if self.runtime_baseline_results.get(name) != target
        }
        expected_p2p = set(self.target_results) - expected_f2p
        return self._partition_timeout_adapter(
            expected_f2p=expected_f2p,
            expected_p2p=expected_p2p,
            execution_mode="normalized_timeout_partition_recovery_v1",
        )

    def _run_count_partition_timeout_recovery(
        self,
        original: TestEvent,
        probe_id: str,
        repo_state: str,
        action_id: str | None,
        turn_id: int | None,
        *,
        probe_attempts: int,
    ) -> TestEvent:
        """Recover an unavailable full-suite timeout with one bounded split."""

        self._counts["partition_timeout_recovery_attempts"] += 1
        try:
            adapter = self._count_partition_timeout_adapter()
            f2p_run = self._run_partition_group(
                adapter,
                "f2p",
                repo_state,
                f"{probe_id}:timeout-recovery:f2p",
            )
            p2p_run = self._run_partition_group(
                adapter,
                "p2p",
                repo_state,
                f"{probe_id}:timeout-recovery:p2p",
            )
            if not self._count_partition_observation_is_auditable(
                f2p_run.observation,
                adapter.f2p,
            ) or not self._count_partition_observation_is_auditable(
                p2p_run.observation,
                adapter.p2p,
            ):
                raise ValueError("partition_selector_evidence_unverified")

            partitions = TestPartitionResult(
                f2p=f2p_run.observation.counts,
                p2p=p2p_run.observation.counts,
            )
            groups = (partitions.f2p, partitions.p2p)
            recovered_states = {
                str(name): str(status).upper()
                for run in (f2p_run, p2p_run)
                for name, status in run.observation.named_results.items()
            }
            ownership_rows = [
                {
                    "test_id": name,
                    "command_id": "timeout-recovery",
                    "status": status,
                }
                for name, status in sorted(recovered_states.items())
            ]
            ownership_sha256 = hashlib.sha256(
                json.dumps(
                    ownership_rows,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            native_state_sha256 = hashlib.sha256(
                json.dumps(
                    [
                        {"test_id": name, "status": status}
                        for name, status in sorted(recovered_states.items())
                    ],
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            recovery_plan_hash = hashlib.sha256(
                json.dumps(
                    {
                        "collector": "partition_timeout_recovery",
                        "f2p_tests": list(adapter.f2p),
                        "p2p_tests": list(adapter.p2p),
                        "runner": adapter.name,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            complete = all(group.not_run == 0 for group in groups)
            observation = TestCountObservation(
                schema_version=4,
                expected=len(self.authoritative_test_names),
                passed=sum(group.passed for group in groups),
                failed=sum(group.failed for group in groups),
                errored=sum(group.errored for group in groups),
                skipped=sum(group.skipped for group in groups),
                unclassified=0,
                reported=sum(
                    group.passed
                    + group.failed
                    + group.errored
                    + group.skipped
                    for group in groups
                ),
                not_run=sum(group.not_run for group in groups),
                complete=complete,
                collector="partition_timeout_recovery",
                command_count=(
                    len(adapter.f2p_shards) + len(adapter.p2p_shards)
                ),
                command_evidence=[
                    {
                        "partition": partition,
                        "runner": adapter.name,
                        "reported": run.observation.reported,
                        "expected": len(names),
                        "collector": run.observation.evidence_source,
                        "selector_verified": run.observation.selector_verified,
                        "timed_out": run.timed_out,
                    }
                    for partition, names, run in (
                        ("f2p", adapter.f2p, f2p_run),
                        ("p2p", adapter.p2p, p2p_run),
                    )
                ],
                partition_counts=partitions,
                plan_hash=recovery_plan_hash,
                termination="complete" if complete else "interrupted",
                partial_reason=(None if complete else "timeout_recovery_partial"),
                ownership_observed=len(ownership_rows),
                ownership_sha256=ownership_sha256,
                ownership_evidence=ownership_rows[:50],
                native_state_sha256=native_state_sha256,
            )
            if self.baseline_passed is None:
                raise RuntimeError("count_baseline_unavailable")
            current_passed = self._effective_count_passed(observation)
            potential_before = self._current_potential
            potential_after = float(current_passed - self.baseline_passed)
            self._current_potential = potential_after
            self._counts["partition_timeout_recovery_successes"] += 1
            if not observation.complete:
                self._counts["partition_timeout_recovery_partial"] += 1
            runs = (f2p_run, p2p_run)
            return self._result_event(
                probe_id,
                repo_state,
                TestEventStatus.COMPLETED,
                {},
                action_id,
                turn_id,
                sum(run.duration for run in runs),
                next(
                    (
                        run.exit_code
                        for run in reversed(runs)
                        if run.exit_code not in {0, None}
                    ),
                    runs[-1].exit_code,
                ),
                repo_state,
                [],
                [],
                trusted=True,
                failure_type="verifier_timeout_recovered",
                potential_before=potential_before,
                potential_after=potential_after,
                log_tail="\n".join(run.log_tail for run in runs)[-2000:],
                continuity_status=TestEventContinuityStatus.INTACT,
                result_source=TestResultSource.COUNT_COLLECTOR,
                probe_attempts=probe_attempts,
                restore_confirmation=(
                    RestoreConfirmation.FINGERPRINT_FALLBACK
                    if any(
                        run.restore_confirmation
                        == RestoreConfirmation.FINGERPRINT_FALLBACK
                        for run in runs
                    )
                    else RestoreConfirmation.CONFIRMED
                ),
                timed_out=True,
                result_completeness=(
                    "count_full" if observation.complete else "count_partial"
                ),
                count_observation=observation,
                failure_stage="partition_recovery",
                failure_origin="runner",
                failure_evidence={
                    "original_failure_type": original.failure_type,
                    "original_failure_origin": original.failure_origin,
                    "full_suite_timed_out": True,
                },
            )
        except _ShadowCancelled:
            raise
        except Exception as exc:
            self._counts["partition_timeout_recovery_failures"] += 1
            self.partition_calibration_reason = str(exc)[:500]
            if self.partition_calibration_status != "timeout_recovery_ready":
                self.partition_calibration_status = (
                    "timeout_recovery_unavailable"
                )
            original.failure_evidence = {
                **original.failure_evidence,
                "partition_recovery_attempted": True,
                "partition_recovery_error": str(exc)[:1000],
            }
            if "state_mismatch" in str(exc):
                return self._failure_test_event(
                    probe_id,
                    repo_state,
                    TestEventStatus.STATE_MISMATCH,
                    "partition_state_mismatch",
                    str(exc),
                    action_id,
                    turn_id,
                    continuity_status=TestEventContinuityStatus.LOST,
                    probe_attempts=probe_attempts,
                    failure_stage="partition_recovery",
                    failure_origin="restore",
                    failure_evidence={
                        "original_failure_type": original.failure_type,
                    },
                )
            return original

    def _run_normalized_partition_timeout_recovery(
        self,
        original: TestEvent,
        probe_id: str,
        repo_state: str,
        action_id: str | None,
        turn_id: int | None,
        *,
        probe_attempts: int,
    ) -> TestEvent:
        """Recover a normalized full-suite timeout with one bounded split."""

        self._counts["partition_timeout_recovery_attempts"] += 1
        try:
            adapter = self._normalized_partition_timeout_adapter()
            f2p_run = self._run_partition_group(
                adapter,
                "f2p",
                repo_state,
                f"{probe_id}:timeout-recovery:f2p",
            )
            p2p_run = self._run_partition_group(
                adapter,
                "p2p",
                repo_state,
                f"{probe_id}:timeout-recovery:p2p",
            )
            if not self._count_partition_observation_is_auditable(
                f2p_run.observation,
                adapter.f2p,
            ) or not self._count_partition_observation_is_auditable(
                p2p_run.observation,
                adapter.p2p,
            ):
                raise ValueError("partition_selector_evidence_unverified")

            partitions = TestPartitionResult(
                f2p=f2p_run.observation.counts,
                p2p=p2p_run.observation.counts,
            )
            potential_before = self._current_potential
            potential_after = self._partition_potential(partitions)
            self._current_potential = potential_after
            full = partitions.f2p.is_complete and partitions.p2p.is_complete
            self._counts["partition_timeout_recovery_successes"] += 1
            if not full:
                self._counts["partition_timeout_recovery_partial"] += 1
            runs = (f2p_run, p2p_run)
            return self._result_event(
                probe_id,
                repo_state,
                TestEventStatus.COMPLETED,
                {},
                action_id,
                turn_id,
                sum(run.duration for run in runs),
                next(
                    (
                        run.exit_code
                        for run in reversed(runs)
                        if run.exit_code not in {0, None}
                    ),
                    runs[-1].exit_code,
                ),
                repo_state,
                [],
                [],
                trusted=True,
                failure_type="verifier_timeout_recovered",
                potential_before=potential_before,
                potential_after=potential_after,
                log_tail="\n".join(run.log_tail for run in runs)[-2000:],
                continuity_status=TestEventContinuityStatus.INTACT,
                result_source=TestResultSource.PARTITION_COUNTS,
                probe_attempts=probe_attempts,
                restore_confirmation=(
                    RestoreConfirmation.FINGERPRINT_FALLBACK
                    if any(
                        run.restore_confirmation
                        == RestoreConfirmation.FINGERPRINT_FALLBACK
                        for run in runs
                    )
                    else RestoreConfirmation.CONFIRMED
                ),
                timed_out=True,
                result_completeness=(
                    "partition_counts_full"
                    if full
                    else "partition_counts_partial"
                ),
                partition_counts=partitions,
                failure_stage="partition_recovery",
                failure_origin="runner",
                failure_evidence={
                    "original_failure_type": original.failure_type,
                    "original_failure_origin": original.failure_origin,
                    "full_suite_timed_out": True,
                },
            )
        except _ShadowCancelled:
            raise
        except Exception as exc:
            self._counts["partition_timeout_recovery_failures"] += 1
            self.partition_calibration_reason = str(exc)[:500]
            if self.partition_calibration_status != "timeout_recovery_ready":
                self.partition_calibration_status = (
                    "timeout_recovery_unavailable"
                )
            original.failure_evidence = {
                **original.failure_evidence,
                "partition_recovery_attempted": True,
                "partition_recovery_error": str(exc)[:1000],
            }
            if "state_mismatch" in str(exc):
                return self._failure_test_event(
                    probe_id,
                    repo_state,
                    TestEventStatus.STATE_MISMATCH,
                    "partition_state_mismatch",
                    str(exc),
                    action_id,
                    turn_id,
                    continuity_status=TestEventContinuityStatus.LOST,
                    probe_attempts=probe_attempts,
                    failure_stage="partition_recovery",
                    failure_origin="restore",
                    failure_evidence={
                        "original_failure_type": original.failure_type,
                    },
                )
            return original

    def _run_probe(
        self,
        probe_id: str,
        repo_state: str,
        action_id: str | None,
        turn_id: int | None,
        *,
        baseline: bool,
        agent_changed_paths: list[str] | None = None,
        allow_after_seal: bool = False,
    ) -> TestEvent:
        self._check_cancelled()
        with self._lock:
            if self._sealed and not allow_after_seal:
                return self._failure_test_event(
                    probe_id,
                    repo_state,
                    TestEventStatus.SKIPPED,
                    "suppressed_after_terminal",
                    "probe was suppressed after shadow results were sealed",
                    action_id,
                    turn_id,
                    continuity_status=TestEventContinuityStatus.LOST,
                )
        started = time.perf_counter()
        attempts = 0
        while True:
            self._check_cancelled()
            attempts += 1
            if not baseline and self.verification_result_mode == "partition_counts":
                event = self._run_partition_probe(
                    probe_id,
                    repo_state,
                    action_id,
                    turn_id,
                    probe_attempts=attempts,
                )
            else:
                event = self._run_probe_impl(
                    probe_id,
                    repo_state,
                    action_id,
                    turn_id,
                    baseline=baseline,
                    probe_attempts=attempts,
                    agent_changed_paths=agent_changed_paths,
                )
            self._check_cancelled()
            if (
                self.bug_repair_pass_count_mode
                and not baseline
                and self.partition_timeout_recovery
                and not event.trusted
                and event.timed_out
                and resolve_test_event_continuity(event)
                == TestEventContinuityStatus.RECOVERABLE_GAP
                and event.restore_confirmation
                in {
                    RestoreConfirmation.CONFIRMED,
                    RestoreConfirmation.FINGERPRINT_FALLBACK,
                }
            ):
                event = self._run_count_partition_timeout_recovery(
                    event,
                    probe_id,
                    repo_state,
                    action_id,
                    turn_id,
                    probe_attempts=attempts,
                )
            if (
                not self.pass_rate_mode
                and not self.bug_repair_pass_count_mode
                and not baseline
                and self.partition_timeout_recovery
                and event.trusted
                and event.timed_out
                and event.failure_type == "verifier_timeout"
                and event.result_completeness == "controlled_partial"
                and event.result_source == TestResultSource.CONTROLLED_TIMEOUT
                and event.restore_confirmation
                in {
                    RestoreConfirmation.CONFIRMED,
                    RestoreConfirmation.FINGERPRINT_FALLBACK,
                }
            ):
                event = self._run_normalized_partition_timeout_recovery(
                    event,
                    probe_id,
                    repo_state,
                    action_id,
                    turn_id,
                    probe_attempts=attempts,
                )
            self._note_progress("baseline_probe" if baseline else "probe")
            continuity = resolve_test_event_continuity(event)
            retriable_failure = self.verification_result_mode != "partition_counts" and (
                event.failure_type in _RETRIABLE_PROBE_FAILURES or (self.result_parser == R2E_RESULT_PARSER and event.failure_type == "test_set_mismatch")
            )
            if event.trusted or continuity == TestEventContinuityStatus.LOST or not retriable_failure or attempts >= 2:
                break
            self._counts["probe_retries"] += 1

        if (
            baseline
            and not self.bug_repair_pass_count_mode
            and not event.trusted
            and resolve_test_event_continuity(event)
            == TestEventContinuityStatus.RECOVERABLE_GAP
        ):
            event.continuity_status = TestEventContinuityStatus.LOST
        elapsed = time.perf_counter() - started
        event.duration = elapsed
        event.probe_attempts = attempts
        with self._lock:
            if self._sealed:
                return event
            self._timings["baseline_probe_s" if baseline else "probe_s"] += elapsed
            continuity = resolve_test_event_continuity(event)
            if event.timed_out:
                self._counts["probes_timed_out"] += 1
            if event.trusted:
                self._counts["trusted_probes"] += 1
                if event.result_completeness == "count_full":
                    self._counts["count_full_probes"] += 1
                elif event.result_completeness == "count_partial":
                    self._counts["count_partial_probes"] += 1
                if event.result_completeness == "contract_assisted":
                    self._counts[
                        "contract_assisted_baselines"
                        if baseline
                        else "contract_assisted_probes"
                    ] += 1
                if event.not_run:
                    self._counts["trusted_not_run_probes"] += 1
                if not baseline:
                    self._counts["probes_completed"] += 1
                if event.result_source == TestResultSource.COLLECTION_ABORT:
                    self._counts["collection_aborts"] += 1
            else:
                self._counts["probes_failed"] += 1
                if self.bug_repair_pass_count_mode:
                    self._counts["count_unavailable_probes"] += 1
                if continuity == TestEventContinuityStatus.RECOVERABLE_GAP:
                    self._counts["recoverable_probes"] += 1
                    failure_type = event.failure_type or "probe"
                    self._recoverable_failure_counts[failure_type] = self._recoverable_failure_counts.get(failure_type, 0) + 1
                else:
                    self._counts["terminal_probes"] += 1
                    self._counts["terminal_failures"] += 1
        return event

    def _run_probe_impl(
        self,
        probe_id: str,
        repo_state: str,
        action_id: str | None,
        turn_id: int | None,
        *,
        baseline: bool,
        probe_attempts: int,
        agent_changed_paths: list[str] | None = None,
    ) -> TestEvent:
        started = time.perf_counter()
        marker = "__RLLM_SHADOW_RC__"
        result_artifact = shlex.quote(self.test_results_path)
        plugin_setup = ""
        count_setup = ""
        language_setup = self._node_yarn_guard_shell()
        plugin_artifacts = ""
        verifier_command = "bash /tests/test.sh"
        timeout_flag = "0"
        if self.result_parser in SWEREBENCH_V2_RESULT_PARSERS:
            verifier_command = f"timeout -k 30s {self.probe_execution_timeout:g}s bash /tests/test.sh"
            timeout_flag = '$(if [ "$rc" -eq 124 ]; then printf 1; else printf 0; fi)'
        if (
            (
                self.result_parser == SWEREBENCH_V2_RESULT_PARSER
                or (
                    self.bug_repair_pass_count_mode
                    and self.language.strip().casefold() == "python"
                )
            )
            and self._pytest_plugin_path is not None
            and self._pytest_plugin_results_path is not None
            and self._pytest_plugin_events_path is not None
        ):
            plugin_dir = shlex.quote(self._pytest_plugin_path.rsplit("/", 1)[0])
            plugin_results = shlex.quote(self._pytest_plugin_results_path)
            plugin_events = shlex.quote(self._pytest_plugin_events_path)
            plugin_artifacts = f" {plugin_results} {plugin_events}"
            plugin_setup = (
                f"export PYTHONPATH={plugin_dir}${{PYTHONPATH:+:$PYTHONPATH}}; "
                "export PYTEST_PLUGINS=rllm_shadow_pytest_plugin"
                "${PYTEST_PLUGINS:+,$PYTEST_PLUGINS}; "
                f"export RLLM_SHADOW_COUNT_PLAN_HASH={shlex.quote(self._count_plan_hash or 'legacy')}; "
                f"export RLLM_SHADOW_TEST_RESULTS_JSON={plugin_results}; "
                f"export RLLM_SHADOW_TEST_EVENTS_JSONL={plugin_events}; "
            )
        if self.bug_repair_pass_count_mode:
            if (
                self.result_parser in SWEREBENCH_V2_RESULT_PARSERS
                and self._count_instance_path is None
            ):
                return self._failure_test_event(
                    probe_id,
                    repo_state,
                    TestEventStatus.ERROR,
                    "count_collector_setup",
                    "derived pass-count instance was not uploaded",
                    action_id,
                    turn_id,
                    continuity_status=TestEventContinuityStatus.LOST,
                    probe_attempts=probe_attempts,
                    failure_stage="setup",
                    failure_origin="count_collector",
                )
            if self._count_instance_path is not None:
                count_setup = (
                    f"export RLLM_INSTANCE_JSON={shlex.quote(self._count_instance_path)}; "
                    ": > /tmp/rllm/count_probe_started; "
                )
        non_python_setup = ""
        if self._uses_non_python_evidence():
            self._non_python_probe_token = uuid.uuid4().hex
            non_python_setup = (
                f"printf %s {shlex.quote(self._non_python_probe_token)} "
                "> /tmp/rllm/non_python_probe_token; "
            )
        command = (
            f"rm -f /tmp/rllm/reward.json /tmp/test_output.txt /tmp/rllm/shadow-count/*.json {result_artifact}{plugin_artifacts}; "
            "mkdir -p /tmp/rllm /tmp/rllm/shadow-count /logs/verifier; "
            f"{language_setup}"
            f"{plugin_setup}"
            f"{count_setup}"
            f"{non_python_setup}"
            "chmod +x /tests/test.sh; "
            f"{verifier_command}; rc=$?; "
            f'printf \'\\n{marker}%s:%s\\n\' "$rc" "{timeout_flag}"'
        )
        probe_before = self.harness._codeflow_capture_repo_snapshot(self.shadow, self.task, self._baseline_ref)
        self._check_cancelled()
        if probe_before.status != RepoStateStatus.OK or probe_before.fingerprint != repo_state:
            self._counts["state_mismatches"] += 1
            return self._failure_test_event(
                probe_id,
                repo_state,
                TestEventStatus.STATE_MISMATCH,
                "probe_before_mismatch",
                f"expected {repo_state}, observed {probe_before.fingerprint or probe_before.error}",
                action_id,
                turn_id,
                continuity_status=TestEventContinuityStatus.LOST,
                probe_attempts=probe_attempts,
            )

        checkpoint_user = self.task.metadata.get("verifier_user") or self.task.metadata.get("agent_user")
        checkpoint, checkpoint_error = self.harness._codeflow_create_repo_checkpoint(
            self.shadow,
            self.task,
            user=checkpoint_user,
        )
        self._check_cancelled()
        if checkpoint is None:
            return self._failure_test_event(
                probe_id,
                repo_state,
                TestEventStatus.ERROR,
                "probe_checkpoint",
                checkpoint_error or "failed to create probe repository checkpoint",
                action_id,
                turn_id,
                continuity_status=TestEventContinuityStatus.LOST,
                probe_attempts=probe_attempts,
            )

        execution_error: BaseException | None = None
        raw_output = ""
        try:
            raw_output = self.shadow.exec(
                command,
                timeout=(self.probe_execution_timeout + self.probe_stall_grace if self.result_parser in SWEREBENCH_V2_RESULT_PARSERS else self.probe_execution_timeout),
                user=self.task.metadata.get("verifier_user"),
            )
        except Exception as exc:
            execution_error = exc
        self._check_cancelled()
        duration = time.perf_counter() - started
        match = re.search(
            rf"{re.escape(marker)}(\d+)(?::([01]))?\s*$",
            str(raw_output),
        )
        exit_code = int(match.group(1)) if match else None
        verifier_timed_out = bool(match and match.group(2) == "1")
        if self.pass_rate_mode and (execution_error is not None or match is None):
            # An RPC timeout is not proof that the verifier stopped. Retire
            # this lane; never restore its repository under a live verifier.
            return self._failure_test_event(
                probe_id, repo_state, TestEventStatus.ERROR,
                "verifier_stop_unconfirmed", str(execution_error or "verifier completion marker missing"),
                action_id, turn_id, duration=duration,
                continuity_status=TestEventContinuityStatus.LOST,
                probe_attempts=probe_attempts, failure_stage="full_probe",
                failure_origin="sandbox",
            )
        parsed = self._parse_probe_output()
        self._check_cancelled()
        if self.pass_rate_mode and parsed.get("process_cleanup_confirmed") is not True:
            return self._failure_test_event(
                probe_id, repo_state, TestEventStatus.ERROR,
                "verifier_stop_unconfirmed", "DeNovo verifier did not confirm descendant cleanup",
                action_id, turn_id, duration=duration,
                continuity_status=TestEventContinuityStatus.LOST,
                probe_attempts=probe_attempts, failure_stage="full_probe",
                failure_origin="verifier",
            )
        after_probe = self.harness._codeflow_capture_repo_snapshot(self.shadow, self.task, self._baseline_ref)
        self._check_cancelled()
        side_effects = changed_files_between(probe_before, after_probe)
        side_effect_paths = sorted({path for item in side_effects for path in (item.path, item.old_path) if isinstance(path, str) and path})
        restore_error = self.harness._codeflow_restore_repo_checkpoint(
            checkpoint,
            self.shadow,
            self.task,
            user=checkpoint_user,
        )
        self._check_cancelled()
        self.harness._codeflow_discard_repo_checkpoint(
            checkpoint,
            self.shadow,
            self.task,
            user=checkpoint_user,
        )
        self._check_cancelled()
        observed = self.harness._codeflow_capture_repo_snapshot(self.shadow, self.task, self._baseline_ref)
        self._check_cancelled()
        observed_fingerprint = observed.fingerprint if observed.status == RepoStateStatus.OK else None

        restore_error_kind = str(getattr(restore_error, "kind", "operation")) if restore_error is not None else None
        fingerprint_matches = observed.status == RepoStateStatus.OK and observed_fingerprint == repo_state
        restore_confirmation = RestoreConfirmation.CONFIRMED
        if (
            restore_error is not None
            and restore_error_kind in {"protocol", "transport"}
            and fingerprint_matches
        ):
            restore_confirmation = RestoreConfirmation.FINGERPRINT_FALLBACK
            with self._lock:
                if not self._sealed:
                    self._counts["restore_protocol_warnings"] += 1
                    self._counts["restore_fingerprint_fallbacks"] += 1
                    if len(self._restore_warnings) < 20:
                        self._restore_warnings.append(
                            {
                                "confirmation": "fingerprint_fallback",
                                "probe_id": probe_id,
                                "expected_fingerprint": repo_state,
                                "observed_fingerprint": observed_fingerprint,
                                "error": str(restore_error)[:1000],
                                "detail": str(getattr(restore_error, "detail", "") or "")[:2000],
                            }
                        )
            logger.warning(
                "[%s] repository restore helper response was malformed, but the Git-visible fingerprint exactly matched %s; preserving continuity: %s",
                self.uid,
                repo_state,
                restore_error,
            )
        elif restore_error is not None or not fingerprint_matches:
            with self._lock:
                if not self._sealed:
                    self._counts["state_mismatches"] += 1
            return self._failure_test_event(
                probe_id,
                repo_state,
                TestEventStatus.STATE_MISMATCH,
                "verifier_state_mismatch",
                f"expected {repo_state}, observed {observed_fingerprint or observed.error}; restore={restore_error or 'completed'}",
                action_id,
                turn_id,
                duration=duration,
                exit_code=exit_code,
                shadow_repo_state=observed_fingerprint,
                log_tail=str(parsed.get("log_tail") or ""),
                continuity_status=TestEventContinuityStatus.LOST,
                probe_attempts=probe_attempts,
                state_mismatch_paths=side_effect_paths,
                restore_confirmation=RestoreConfirmation.FAILED,
                failure_evidence={
                    "expected_fingerprint": repo_state,
                    "observed_fingerprint": observed_fingerprint,
                    "restore_error_kind": restore_error_kind,
                },
            )
        if side_effect_paths:
            with self._lock:
                if not self._sealed:
                    self._counts["verifier_side_effects_restored"] += 1
        controlled_timeout = verifier_timed_out or bool(parsed.get("timed_out")) or _looks_like_timeout_error(execution_error)
        resource_error = _looks_like_resource_error(execution_error)
        if resource_error:
            return self._failure_test_event(
                probe_id,
                repo_state,
                TestEventStatus.ERROR,
                "resource_exhausted",
                "sandbox reported a typed resource exhaustion failure",
                action_id,
                turn_id,
                duration=duration,
                exit_code=exit_code,
                shadow_repo_state=observed_fingerprint,
                log_tail=str(parsed.get("log_tail") or ""),
                continuity_status=TestEventContinuityStatus.RECOVERABLE_GAP,
                probe_attempts=probe_attempts,
                state_mismatch_paths=side_effect_paths,
                restore_confirmation=restore_confirmation,
                failure_stage="full_probe",
                failure_origin="sandbox",
                failure_evidence={
                    "exception_type": type(execution_error).__name__,
                },
            )
        if execution_error is not None and not controlled_timeout:
            return self._failure_test_event(
                probe_id,
                repo_state,
                TestEventStatus.ERROR,
                "probe_execution",
                str(execution_error),
                action_id,
                turn_id,
                duration=duration,
                shadow_repo_state=observed_fingerprint,
                log_tail=str(parsed.get("log_tail") or ""),
                continuity_status=TestEventContinuityStatus.RECOVERABLE_GAP,
                probe_attempts=probe_attempts,
                state_mismatch_paths=side_effect_paths,
                restore_confirmation=restore_confirmation,
            )
        if match is None and not controlled_timeout:
            return self._failure_test_event(
                probe_id,
                repo_state,
                TestEventStatus.ERROR,
                "probe_exit_marker",
                "shadow verifier exit marker is missing",
                action_id,
                turn_id,
                duration=duration,
                shadow_repo_state=observed_fingerprint,
                log_tail=str(parsed.get("log_tail") or ""),
                continuity_status=TestEventContinuityStatus.RECOVERABLE_GAP,
                probe_attempts=probe_attempts,
                state_mismatch_paths=side_effect_paths,
                restore_confirmation=restore_confirmation,
            )
        if controlled_timeout and baseline:
            return self._failure_test_event(
                probe_id,
                repo_state,
                TestEventStatus.TIMEOUT,
                "verifier_timeout",
                str(execution_error or "baseline verifier exceeded its declared timeout"),
                action_id,
                turn_id,
                duration=duration,
                exit_code=exit_code,
                timed_out=True,
                shadow_repo_state=observed_fingerprint,
                log_tail=str(parsed.get("log_tail") or ""),
                continuity_status=TestEventContinuityStatus.LOST,
                result_source=TestResultSource.CONTROLLED_TIMEOUT,
                probe_attempts=probe_attempts,
                state_mismatch_paths=side_effect_paths,
                restore_confirmation=restore_confirmation,
                failure_stage="full_probe",
                failure_origin=(
                    "wrapper" if verifier_timed_out else "sandbox"
                ),
                failure_evidence={
                    "wrapper_marker_seen": match is not None,
                    "wrapper_timed_out": verifier_timed_out,
                    "exception_type": (
                        type(execution_error).__name__
                        if execution_error is not None
                        else None
                    ),
                },
            )
        if (
            controlled_timeout
            and not baseline
            and not self.bug_repair_pass_count_mode
            and not parsed.get("ok")
            and (
                parsed.get("collector_active")
                or self.result_parser == SWEREBENCH_V2_OFFICIAL_RESULT_PARSER
            )
        ):
            parsed = {
                **parsed,
                "ok": True,
                "test_results": {},
                "result_source": ("official_log_parser" if self.result_parser == SWEREBENCH_V2_OFFICIAL_RESULT_PARSER else "pytest_plugin"),
            }
        if (
            self.bug_repair_pass_count_mode
            and not parsed.get("ok")
            and parsed.get("failure_type") == "deterministic_abort_candidate"
        ):
            abort_candidate = parsed.get("abort_candidate")
            abort_paths = (
                abort_candidate.get("paths")
                if isinstance(abort_candidate, dict)
                else None
            )
            authenticated_code_abort = bool(
                isinstance(abort_candidate, dict)
                and abort_candidate.get("bootstrap_authenticated") is True
                and abort_candidate.get("repo_failure_authenticated") is True
            )
            path_intersection = _abort_paths_intersect_changes(
                abort_paths,
                agent_changed_paths,
                self.harness._repository_root(self.task),
            )
            agent_caused = bool(
                not baseline
                and agent_changed_paths
                and (path_intersection or authenticated_code_abort)
            )
            candidate_observation = (
                abort_candidate.get("observation")
                if isinstance(abort_candidate, dict)
                else None
            )
            if agent_caused and isinstance(candidate_observation, dict):
                candidate_observation = dict(candidate_observation)
                candidate_observation.pop("observed_test_states", None)
                parsed = {
                    "ok": True,
                    "count_observation": candidate_observation,
                    "log_tail": str(parsed.get("log_tail") or ""),
                    "abort_promoted": True,
                }
            else:
                parsed = {
                    **parsed,
                    "failure_type": "count_unavailable",
                    "failure_evidence": {
                        **(
                            dict(parsed.get("failure_evidence"))
                            if isinstance(parsed.get("failure_evidence"), dict)
                            else {}
                        ),
                        "deterministic_abort_candidate": True,
                        "agent_changed_paths": list(agent_changed_paths or [])[:50],
                        "path_intersection": path_intersection,
                        "authenticated_code_abort": authenticated_code_abort,
                    },
                }
        if not parsed.get("ok"):
            deterministic_parse_failure = bool(parsed.get("deterministic"))
            parsed_failure_type = str(parsed.get("failure_type") or "")
            parsed_timed_out = bool(
                parsed.get("timed_out")
                or (self.bug_repair_pass_count_mode and controlled_timeout)
            )
            return self._failure_test_event(
                probe_id,
                repo_state,
                (
                    TestEventStatus.TIMEOUT
                    if parsed_timed_out
                    else TestEventStatus.ERROR
                ),
                (
                    "verifier_timeout"
                    if self.bug_repair_pass_count_mode and controlled_timeout
                    else "baseline_count_incomplete"
                    if baseline
                    and parsed_failure_type == "selector_contract_incomplete"
                    else "pytest_event_contract_error"
                    if parsed_failure_type == "pytest_event_contract_error"
                    else "parse_contract_error"
                    if deterministic_parse_failure
                    else parsed_failure_type or "parse_error"
                ),
                str(parsed.get("error") or "failed to parse probe output"),
                action_id,
                turn_id,
                duration=duration,
                exit_code=exit_code,
                timed_out=parsed_timed_out,
                shadow_repo_state=observed_fingerprint,
                log_tail=str(parsed.get("log_tail") or ""),
                continuity_status=TestEventContinuityStatus.RECOVERABLE_GAP,
                probe_attempts=probe_attempts,
                state_mismatch_paths=side_effect_paths,
                restore_confirmation=restore_confirmation,
                result_source=(
                    TestResultSource.CONTROLLED_TIMEOUT
                    if parsed_timed_out
                    else None
                ),
                failure_stage=(
                    "full_probe"
                    if self.bug_repair_pass_count_mode and controlled_timeout
                    else "baseline"
                    if baseline
                    and parsed_failure_type == "selector_contract_incomplete"
                    else str(parsed.get("failure_stage") or "parser")
                ),
                failure_origin=(
                    "wrapper"
                    if self.bug_repair_pass_count_mode and verifier_timed_out
                    else "sandbox"
                    if self.bug_repair_pass_count_mode and controlled_timeout
                    else str(parsed.get("failure_origin") or "parser")
                ),
                failure_evidence=(
                    {
                        **(
                            dict(parsed.get("failure_evidence"))
                            if isinstance(parsed.get("failure_evidence"), dict)
                            else {}
                        ),
                        **(
                            {
                                "wrapper_timed_out": verifier_timed_out,
                                "collector_failure_type": parsed_failure_type,
                            }
                            if self.bug_repair_pass_count_mode
                            and controlled_timeout
                            else {}
                        ),
                    }
                ),
            )

        if self.bug_repair_pass_count_mode:
            try:
                count_observation = TestCountObservation.model_validate(
                    parsed.get("count_observation")
                )
            except Exception as exc:
                return self._failure_test_event(
                    probe_id,
                    repo_state,
                    TestEventStatus.ERROR,
                    "count_contract_error",
                    f"invalid TestCountObservation: {exc}",
                    action_id,
                    turn_id,
                    duration=duration,
                    exit_code=exit_code,
                    shadow_repo_state=observed_fingerprint,
                    log_tail=str(parsed.get("log_tail") or ""),
                    continuity_status=TestEventContinuityStatus.RECOVERABLE_GAP,
                    probe_attempts=probe_attempts,
                    state_mismatch_paths=side_effect_paths,
                    restore_confirmation=restore_confirmation,
                    failure_stage="count_collector",
                    failure_origin="collector",
                )
            if baseline and not count_observation.complete:
                return self._failure_test_event(
                    probe_id,
                    repo_state,
                    TestEventStatus.ERROR,
                    "baseline_count_incomplete",
                    (
                        "baseline count collector reported only "
                        f"{count_observation.reported}/"
                        f"{count_observation.expected} tests"
                    ),
                    action_id,
                    turn_id,
                    duration=duration,
                    exit_code=exit_code,
                    shadow_repo_state=observed_fingerprint,
                    log_tail=str(parsed.get("log_tail") or ""),
                    continuity_status=TestEventContinuityStatus.RECOVERABLE_GAP,
                    probe_attempts=probe_attempts,
                    state_mismatch_paths=side_effect_paths,
                    restore_confirmation=restore_confirmation,
                    failure_stage="baseline",
                    failure_origin="count_collector",
                    failure_evidence={
                        "expected": count_observation.expected,
                        "reported": count_observation.reported,
                        "collector": count_observation.collector,
                    },
                )
            try:
                if baseline:
                    self._accept_runtime_count_baseline(count_observation)
                    potential_before = potential_after = 0.0
                else:
                    if self.baseline_passed is None:
                        return self._failure_test_event(
                            probe_id,
                            repo_state,
                            TestEventStatus.ERROR,
                            "count_baseline_unavailable",
                            "resolved count baseline is unavailable",
                            action_id,
                            turn_id,
                            duration=duration,
                            exit_code=exit_code,
                            shadow_repo_state=observed_fingerprint,
                            log_tail=str(parsed.get("log_tail") or ""),
                            continuity_status=TestEventContinuityStatus.LOST,
                            probe_attempts=probe_attempts,
                            state_mismatch_paths=side_effect_paths,
                            restore_confirmation=restore_confirmation,
                        )
                    current_passed = self._effective_count_passed(count_observation)
                    potential_before = self._current_potential
                    potential_after = float(
                        current_passed - self.baseline_passed
                    )
                    self._current_potential = potential_after
            except ValueError as exc:
                # A count-policy/observation incompatibility is a recoverable
                # verification gap. It must never terminate the serial shadow
                # worker and suppress all later probes for the rollout.
                return self._failure_test_event(
                    probe_id,
                    repo_state,
                    TestEventStatus.ERROR,
                    "count_contract_error",
                    str(exc),
                    action_id,
                    turn_id,
                    duration=duration,
                    exit_code=exit_code,
                    shadow_repo_state=observed_fingerprint,
                    log_tail=str(parsed.get("log_tail") or ""),
                    continuity_status=TestEventContinuityStatus.RECOVERABLE_GAP,
                    probe_attempts=probe_attempts,
                    state_mismatch_paths=side_effect_paths,
                    restore_confirmation=restore_confirmation,
                    failure_stage="count_policy",
                    failure_origin="host",
                    failure_evidence={
                        "policy": CONTRACT_FILL_POLICY,
                        "observation_schema_version": (
                            count_observation.schema_version
                        ),
                        "collector": count_observation.collector,
                        "expected": count_observation.expected,
                        "reported": count_observation.reported,
                    },
                )
            return self._result_event(
                probe_id,
                repo_state,
                TestEventStatus.COMPLETED,
                {},
                action_id,
                turn_id,
                duration,
                exit_code,
                observed_fingerprint,
                [],
                [],
                trusted=True,
                failure_type=("verifier_timeout" if controlled_timeout else None),
                potential_before=potential_before,
                potential_after=potential_after,
                error=(str(execution_error) if execution_error else None),
                log_tail=str(parsed.get("log_tail") or ""),
                continuity_status=TestEventContinuityStatus.INTACT,
                result_source=TestResultSource.COUNT_COLLECTOR,
                probe_attempts=probe_attempts,
                state_mismatch_paths=side_effect_paths,
                restore_confirmation=restore_confirmation,
                timed_out=controlled_timeout,
                result_completeness=(
                    "count_full" if count_observation.complete else "count_partial"
                ),
                count_observation=count_observation,
                failure_stage=("full_probe" if controlled_timeout else None),
                failure_origin=("wrapper" if controlled_timeout else None),
                failure_evidence=(
                    {"timed_out": True, "wrapper_timed_out": verifier_timed_out}
                    if controlled_timeout
                    else None
                ),
            )

        if parsed.get("collection_aborted"):
            if baseline:
                abort_kind = str(parsed.get("abort_kind") or "collection_error")
                if self._can_fill_all_f2p_baseline_contract() and abort_kind in {
                    "collection_error",
                    "test_module_import_error",
                    "pytest_config_error",
                }:
                    results = dict(self.materialized_baseline_results)
                    self._accept_runtime_baseline(results)
                    potential_before = potential_after = self._accepted_baseline_potential()
                    return self._result_event(
                        probe_id,
                        repo_state,
                        TestEventStatus.COMPLETED,
                        results,
                        action_id,
                        turn_id,
                        duration,
                        exit_code,
                        observed_fingerprint,
                        [],
                        [],
                        trusted=True,
                        failure_type=abort_kind,
                        potential_before=potential_before,
                        potential_after=potential_after,
                        log_tail=str(parsed.get("log_tail") or ""),
                        continuity_status=TestEventContinuityStatus.INTACT,
                        result_source=TestResultSource.COLLECTION_ABORT,
                        probe_attempts=probe_attempts,
                        state_mismatch_paths=side_effect_paths,
                        restore_confirmation=restore_confirmation,
                        result_completeness="baseline_contract_filled",
                        imputed_tests=list(self.authoritative_test_names),
                    )
                return self._failure_test_event(
                    probe_id,
                    repo_state,
                    TestEventStatus.ERROR,
                    str(parsed.get("abort_kind") or "collection_error"),
                    "authoritative tests did not start; baseline contract cannot be filled",
                    action_id,
                    turn_id,
                    duration=duration,
                    exit_code=exit_code,
                    shadow_repo_state=observed_fingerprint,
                    log_tail=str(parsed.get("log_tail") or ""),
                    continuity_status=TestEventContinuityStatus.RECOVERABLE_GAP,
                    probe_attempts=probe_attempts,
                    state_mismatch_paths=side_effect_paths,
                    restore_confirmation=restore_confirmation,
                )
            results = dict.fromkeys(self.target_results, "NOT_RUN")
            completeness = "controlled_partial"
            failure_type = str(parsed.get("abort_kind") or "collection_error")
            potential_after = self._potential(results)
            potential_before = self._current_potential
            self._current_potential = potential_after
            return self._result_event(
                probe_id,
                repo_state,
                TestEventStatus.COMPLETED,
                results,
                action_id,
                turn_id,
                duration,
                exit_code,
                observed_fingerprint,
                [],
                [],
                trusted=True,
                failure_type=failure_type,
                potential_before=potential_before,
                potential_after=potential_after,
                log_tail=str(parsed.get("log_tail") or ""),
                continuity_status=TestEventContinuityStatus.INTACT,
                result_source=TestResultSource.COLLECTION_ABORT,
                probe_attempts=probe_attempts,
                state_mismatch_paths=side_effect_paths,
                restore_confirmation=restore_confirmation,
                result_completeness=completeness,
                imputed_tests=list(self.authoritative_test_names),
            )

        observed_results = {str(key): str(value) for key, value in dict(parsed.get("test_results") or {}).items()}
        parameter_aliases: dict[str, str] = {}
        if self.test_set_policy == _ANNOTATED_SUBSET_TEST_SET_POLICY:
            observed_results, parameter_aliases = _reconcile_annotated_parameter_ids(
                observed_results,
                self.authoritative_test_names,
            )
            if parameter_aliases:
                with self._lock:
                    if not self._sealed:
                        self._counts["parameter_id_aliases"] += len(parameter_aliases)
        raw_result_source = str(parsed.get("result_source") or "")
        if raw_result_source == "pytest_plugin":
            result_source = TestResultSource.PYTEST_PLUGIN
        elif raw_result_source == "official_log_parser":
            result_source = TestResultSource.OFFICIAL_LOG_PARSER
        else:
            result_source = TestResultSource.PYTEST_SUMMARY
        if result_source == TestResultSource.PYTEST_PLUGIN:
            with self._lock:
                if not self._sealed:
                    self._counts["pytest_plugin_probes"] += 1
        expected_names = set(self.target_results)
        missing = sorted(expected_names - set(observed_results))
        observed_extra = sorted(set(observed_results) - expected_names)
        ignored_invalid_extra = sorted(
            set(parsed.get("ignored_invalid_extra_tests") or [])
        )
        partial_failure_type = str(parsed.get("partial_failure_type") or "")
        if self._uses_non_python_evidence() and baseline:
            conflicting_baseline = sorted(
                name for name, status in observed_results.items()
                if (name in self._p2p_contract and status != "PASSED")
                or (name in self._f2p_contract and status != self.materialized_baseline_results.get(name))
            )
            if conflicting_baseline:
                return self._result_event(
                    probe_id, repo_state, TestEventStatus.ERROR, observed_results,
                    action_id, turn_id, duration, exit_code, observed_fingerprint,
                    missing, observed_extra,
                    failure_type="non_python_baseline_contract_mismatch",
                    error="observed baseline states disagree with the F2P/P2P contract",
                    log_tail=str(parsed.get("log_tail") or ""),
                    continuity_status=TestEventContinuityStatus.LOST,
                    result_source=result_source, probe_attempts=probe_attempts,
                    state_mismatch_paths=side_effect_paths,
                    restore_confirmation=restore_confirmation,
                    failure_stage="baseline_calibration", failure_origin="runner",
                    failure_evidence={
                        **(parsed.get("non_python_evidence") or {}),
                        "baseline": True,
                        "conflicting_tests": conflicting_baseline[:20],
                        "conflicting_test_count": len(conflicting_baseline),
                        "missing_f2p_count": len(set(missing) & self._f2p_contract),
                        "missing_p2p_count": len(set(missing) & self._p2p_contract),
                    },
                )
        contract_assisted = self._can_contract_assist_partial_vector(
            observed_results,
            missing,
            observed_extra,
            ignored_invalid_extra,
            baseline=baseline,
            parsed=parsed,
        )
        non_python_evidence = parsed.get("non_python_evidence") or {}
        if self._uses_non_python_evidence() and baseline:
            self._non_python_test_owners = dict(non_python_evidence.get("test_owners") or {})
        source_abort = non_python_evidence.get("source_abort") or {}
        non_python_changed_paths = list(agent_changed_paths or [])
        if self._uses_non_python_evidence():
            initial_snapshot = getattr(self, "_non_python_initial_snapshot", None)
            if initial_snapshot is not None:
                non_python_changed_paths = sorted(set(non_python_changed_paths) | set(
                    _changed_file_paths(changed_files_between(initial_snapshot, probe_before))
                ))
        non_python_partial = bool(
            self._uses_non_python_evidence()
            and not baseline
            and self.runtime_baseline_results is not None
            and missing
            and not observed_extra
            and not ignored_invalid_extra
            and execution_error is None
            and not verifier_timed_out
            and source_abort.get("phase") in {"source_compile", "source_load"}
            and _abort_paths_intersect_changes(
                source_abort.get("paths") or [],
                non_python_changed_paths,
                self.harness._repository_root(self.task),
            )
        )
        non_python_diagnostics = None
        if self._uses_non_python_evidence():
            non_python_diagnostics = {
                **{key: value for key, value in non_python_evidence.items() if baseline or key != "test_owners"},
                "missing_f2p_count": len(set(missing) & self._f2p_contract),
                "missing_p2p_count": len(set(missing) & self._p2p_contract),
                "baseline": baseline,
                "accepted_partial": non_python_partial,
                "source_changed_paths": non_python_changed_paths[:50],
            }
        controlled_partial = bool(
            not baseline
            and missing
            and not contract_assisted
            and self.result_parser != SWEREBENCH_V2_OFFICIAL_RESULT_PARSER
            and (
                controlled_timeout
            )
        )
        controlled_partial = controlled_partial or non_python_partial
        imputed_tests = (
            list(missing)
            if contract_assisted or controlled_partial
            else []
        )
        if contract_assisted and baseline:
            for name in missing:
                observed_results[name] = self.materialized_baseline_results[name]
            missing = []
        elif contract_assisted:
            for name in missing:
                observed_results[name] = "NOT_RUN"
            missing = []
        elif controlled_partial:
            for name in missing:
                observed_results[name] = "NOT_RUN"
            missing = []
            if controlled_timeout:
                with self._lock:
                    if not self._sealed:
                        self._counts["controlled_timeout_probes"] += 1
        projected_results = {name: observed_results[name] for name in self.authoritative_test_names if name in observed_results}
        ignored_extra = sorted(set(observed_extra) | set(ignored_invalid_extra)) if self.test_set_policy == _ANNOTATED_SUBSET_TEST_SET_POLICY else []
        extra = observed_extra if self.test_set_policy == _EXACT_TEST_SET_POLICY else []
        if (controlled_partial or contract_assisted) and not extra:
            if baseline:
                self._accept_runtime_baseline(projected_results)
                potential_before = potential_after = self._accepted_baseline_potential()
            else:
                potential_after = self._potential(projected_results)
                potential_before = self._current_potential
                self._current_potential = potential_after
            return self._result_event(
                probe_id,
                repo_state,
                TestEventStatus.COMPLETED,
                projected_results,
                action_id,
                turn_id,
                duration,
                exit_code,
                observed_fingerprint,
                [],
                [],
                trusted=True,
                failure_type=(
                    "verifier_timeout"
                    if controlled_timeout
                    else None
                    if contract_assisted
                    else source_abort["phase"] if non_python_partial
                    else partial_failure_type or "deterministic_runner_partial"
                ),
                failure_evidence=non_python_diagnostics,
                failure_stage=(source_abort.get("phase") if non_python_partial else None),
                failure_origin=("runner" if non_python_partial else None),
                ignored_extra_tests=ignored_extra,
                potential_before=potential_before,
                potential_after=potential_after,
                error=str(execution_error),
                log_tail=str(parsed.get("log_tail") or ""),
                continuity_status=TestEventContinuityStatus.INTACT,
                result_source=(TestResultSource.CONTROLLED_TIMEOUT if controlled_timeout else result_source),
                probe_attempts=probe_attempts,
                state_mismatch_paths=side_effect_paths,
                restore_confirmation=restore_confirmation,
                timed_out=controlled_timeout,
                result_completeness=(
                    "contract_assisted"
                    if contract_assisted
                    else "controlled_partial"
                ),
                imputed_tests=imputed_tests,
                contract_fill_policy=(
                    CONTRACT_FILL_POLICY if contract_assisted else None
                ),
                contract_fill_source=(
                    "materialized_baseline"
                    if contract_assisted and baseline
                    else "neutral_not_run"
                    if contract_assisted
                    else None
                ),
            )
        if missing or extra:
            return self._result_event(
                probe_id,
                repo_state,
                TestEventStatus.ERROR,
                (observed_results if self.test_set_policy == _EXACT_TEST_SET_POLICY else projected_results),
                action_id,
                turn_id,
                duration,
                exit_code,
                observed_fingerprint,
                missing,
                extra,
                failure_type="test_set_mismatch",
                error=(f"missing={missing[:10]}, extra={extra[:10]}, ignored_extra={ignored_extra[:10]}"),
                ignored_extra_tests=ignored_extra,
                log_tail=str(parsed.get("log_tail") or ""),
                continuity_status=TestEventContinuityStatus.LOST if baseline else TestEventContinuityStatus.RECOVERABLE_GAP,
                result_source=result_source,
                probe_attempts=probe_attempts,
                state_mismatch_paths=side_effect_paths,
                restore_confirmation=restore_confirmation,
                failure_evidence={
                    **(non_python_diagnostics or {}),
                    "missing_count": len(missing),
                    "extra_count": len(extra),
                    "ignored_extra_count": len(ignored_extra),
                    "observed_count": len(projected_results),
                },
            )
        if baseline:
            self._accept_runtime_baseline(projected_results)
            potential_before = potential_after = self._accepted_baseline_potential()
        else:
            potential_after = self._potential(projected_results)
            potential_before = self._current_potential
            self._current_potential = potential_after
        return self._result_event(
            probe_id,
            repo_state,
            TestEventStatus.COMPLETED,
            projected_results,
            action_id,
            turn_id,
            duration,
            exit_code,
            observed_fingerprint,
            [],
            [],
            trusted=True,
            failure_evidence=non_python_diagnostics,
            ignored_extra_tests=ignored_extra,
            potential_before=potential_before,
            potential_after=potential_after,
            log_tail=str(parsed.get("log_tail") or ""),
            continuity_status=TestEventContinuityStatus.INTACT,
            result_source=result_source,
            probe_attempts=probe_attempts,
            state_mismatch_paths=side_effect_paths,
            restore_confirmation=restore_confirmation,
            result_completeness=(
                "contract_assisted"
                if contract_assisted
                else "full"
            ),
            imputed_tests=imputed_tests,
            contract_fill_policy=(
                CONTRACT_FILL_POLICY
                if contract_assisted
                else None
            ),
            contract_fill_source=(
                "materialized_baseline"
                if contract_assisted
                else None
            ),
        )

    def _parse_probe_output(self) -> dict[str, Any]:
        if self.result_parser == DENOVOSWE_RESULT_PARSER:
            return self._parse_denovoswe_probe_output()
        if self.bug_repair_pass_count_mode:
            return self._parse_count_probe_output()
        if self.result_parser in SWEREBENCH_V2_RESULT_PARSERS:
            return self._parse_swerebench_v2_probe_output()
        return self._parse_r2e_probe_output()

    def _parse_count_probe_output(self) -> dict[str, Any]:
        """Read one authenticated scalar observation from the collector."""

        if (
            self._count_collector_path is None
            or self._count_contract_path is None
        ):
            # Non-SWE-rebench bug-repair tasks can still use an exact official
            # vector, but they do not have a language aggregate collector.
            parsed = (
                self._parse_swerebench_v2_probe_output()
                if self.result_parser in SWEREBENCH_V2_RESULT_PARSERS
                else self._parse_r2e_probe_output()
            )
            if not parsed.get("ok"):
                return parsed
            results = parsed.get("test_results")
            if not isinstance(results, dict) or set(results) != set(
                self.authoritative_test_names
            ):
                return {
                    "ok": False,
                    "failure_type": "count_unavailable",
                    "error": "exact result vector is unavailable for pass-count mode",
                    "log_tail": str(parsed.get("log_tail") or ""),
                }
            statuses = [str(value).upper() for value in results.values()]
            state_rows = [
                {"test_id": str(name), "status": str(status).upper()}
                for name, status in sorted(results.items())
            ]
            native_state_sha256 = hashlib.sha256(
                json.dumps(
                    state_rows,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            ownership_rows = [
                {
                    "test_id": row["test_id"],
                    "command_id": "official-vector",
                    "status": row["status"],
                }
                for row in state_rows
            ]
            ownership_sha256 = hashlib.sha256(
                json.dumps(
                    ownership_rows,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            plan_hash = hashlib.sha256(
                json.dumps(
                    {
                        "collector": "official_exact_vector",
                        "expected_tests": sorted(self.authoritative_test_names),
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            try:
                observation = TestCountObservation(
                    schema_version=4,
                    expected=len(self.authoritative_test_names),
                    passed=sum(value == "PASSED" for value in statuses),
                    failed=sum(value == "FAILED" for value in statuses),
                    errored=sum(value == "ERROR" for value in statuses),
                    skipped=sum(value == "SKIPPED" for value in statuses),
                    unclassified=0,
                    reported=len(statuses),
                    not_run=0,
                    complete=True,
                    collector="official_exact_vector",
                    plan_hash=plan_hash,
                    termination="complete",
                    ownership_observed=len(statuses),
                    ownership_sha256=ownership_sha256,
                    ownership_evidence=ownership_rows[:50],
                    native_state_sha256=native_state_sha256,
                    command_evidence=[
                        {
                            "command_index": None,
                            "collector": "official_exact_vector",
                            "reported": len(statuses),
                            "terminal": True,
                        }
                    ],
                )
            except Exception as exc:
                return {
                    "ok": False,
                    "failure_type": "count_contract_error",
                    "error": f"invalid exact result vector: {exc}",
                    "log_tail": str(parsed.get("log_tail") or ""),
                }
            return {
                "ok": True,
                "count_observation": observation.model_dump(mode="json"),
                "log_tail": str(parsed.get("log_tail") or ""),
            }

        helper_command = (
            f"{self._helper_python()} {shlex.quote(self._count_collector_path)} "
            f"{shlex.quote(self._count_contract_path)}"
        )
        framed = frame_structured_command(helper_command)
        self._parser_command_length = len(framed.command.encode("utf-8"))
        if self._parser_command_length >= 4096:
            return {
                "ok": False,
                "failure_type": "count_collector_transport",
                "error": (
                    "count collector command exceeds 4 KiB: "
                    f"{self._parser_command_length}"
                ),
            }
        try:
            raw = self.shadow.exec(
                framed.command,
                timeout=60,
                user=self.task.metadata.get("verifier_user"),
            )
            response = parse_structured_command_output(raw, framed.nonce)
        except StructuredCommandProtocolError as exc:
            return {
                "ok": False,
                "failure_type": "count_collector_transport",
                "error": str(exc),
                "transport_error": True,
            }
        except Exception as exc:
            return {
                "ok": False,
                "failure_type": "count_collector_transport",
                "error": str(exc),
                "transport_error": True,
            }
        if response.exit_code != 0:
            return {
                "ok": False,
                "failure_type": "count_collector_crash",
                "error": (
                    f"count collector exited {response.exit_code}: "
                    f"{response.stderr[-1000:]}"
                ),
                "failure_evidence": {
                    "exit_code": response.exit_code,
                    "stderr_tail": response.stderr[-500:],
                },
            }
        try:
            parsed = json.loads(response.stdout.strip())
        except Exception as exc:
            return {
                "ok": False,
                "failure_type": "count_collector_parse",
                "error": f"count collector returned invalid JSON: {exc}",
            }
        if not isinstance(parsed, dict) or not parsed.get("ok"):
            return (
                parsed
                if isinstance(parsed, dict)
                else {
                    "ok": False,
                    "failure_type": "count_collector_parse",
                    "error": "count collector returned a non-object",
                }
            )
        raw_observation = parsed.get("observation")
        observed_test_states: dict[str, str] = {}
        if isinstance(raw_observation, dict):
            raw_observation = dict(raw_observation)
            raw_states = raw_observation.pop("observed_test_states", {})
            if isinstance(raw_states, dict):
                observed_test_states = {
                    str(name): str(status).upper()
                    for name, status in raw_states.items()
                }
        try:
            observation = TestCountObservation.model_validate(raw_observation)
        except Exception as exc:
            return {
                "ok": False,
                "failure_type": "count_contract_error",
                "error": f"invalid TestCountObservation: {exc}",
                "log_tail": str(parsed.get("log_tail") or ""),
            }
        if observation.schema_version != 4 or observation.plan_hash != self._count_plan_hash:
            return {
                "ok": False,
                "failure_type": "count_contract_error",
                "error": "count collector returned the wrong observation plan",
                "failure_evidence": {
                    "expected_plan_hash": self._count_plan_hash,
                    "observed_plan_hash": observation.plan_hash,
                    "observation_schema_version": observation.schema_version,
                },
                "log_tail": str(parsed.get("log_tail") or ""),
            }
        if observation.expected != len(self.authoritative_test_names):
            return {
                "ok": False,
                "failure_type": "count_total_mismatch",
                "error": "count observation expected total changed at runtime",
                "log_tail": str(parsed.get("log_tail") or ""),
            }
        authoritative = set(self.authoritative_test_names)
        native_state_sha256 = hashlib.sha256(
            json.dumps(
                [
                    {"test_id": name, "status": status}
                    for name, status in sorted(observed_test_states.items())
                ],
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if (
            observation.partition_counts is None
            or not set(observed_test_states).issubset(authoritative)
            or any(
                status not in {"PASSED", "FAILED", "ERROR", "SKIPPED"}
                for status in observed_test_states.values()
            )
            or len(observed_test_states) != observation.reported
            or native_state_sha256 != observation.native_state_sha256
        ):
            return {
                "ok": False,
                "failure_type": "count_contract_error",
                "error": "count collector returned an invalid native state map",
                "failure_evidence": {
                    "expected": observation.expected,
                    "reported": observation.reported,
                    "state_count": len(observed_test_states),
                    "observed_state_sha256": native_state_sha256,
                    "declared_state_sha256": observation.native_state_sha256,
                    "plan_hash": observation.plan_hash,
                },
                "log_tail": str(parsed.get("log_tail") or ""),
            }
        return {
            "ok": True,
            "count_observation": observation.model_dump(mode="json"),
            "log_tail": str(parsed.get("log_tail") or ""),
        }

    def _parse_denovoswe_probe_output(self) -> dict[str, Any]:
        command = f"{self._helper_python()} -c {shlex.quote(_PARSE_DENOVO_RESULTS_SCRIPT)} {shlex.quote(self.test_results_path)}"
        try:
            raw = self.shadow.exec(
                command,
                timeout=30,
                user=self.task.metadata.get("verifier_user"),
            )
            parsed = json.loads(str(raw).strip())
        except Exception as exc:
            return {"ok": False, "error": str(exc), "transport_error": True}
        if not isinstance(parsed, dict) or not parsed.get("ok"):
            return (
                parsed
                if isinstance(parsed, dict)
                else {
                    "ok": False,
                    "error": "DeNovoSWE parser returned a non-object",
                }
            )
        outcome = parsed.get("outcome")
        if not isinstance(outcome, dict):
            return {"ok": False, "error": "DeNovoSWE outcome is missing", "deterministic": True}
        results = outcome.get("test_results")
        allowed = {"PASSED", "FAILED", "ERROR", "SKIPPED", "NOT_RUN", "XFAIL", "XPASS"}
        if (
            outcome.get("schema_version") != 1
            or outcome.get("verifier_profile") != "denovoswe_official"
            or not isinstance(results, dict)
            or set(results) != set(self.authoritative_test_names)
            or any(status not in allowed for status in results.values())
        ):
            return {
                "ok": False,
                "error": "DeNovoSWE outcome contract mismatch",
                "deterministic": True,
                "process_cleanup_confirmed": outcome.get("process_cleanup_confirmed") is True,
            }
        passed = sum(status == "PASSED" for status in results.values())
        failed = sum(status in {"FAILED", "SKIPPED", "XFAIL", "XPASS"} for status in results.values())
        errored = sum(status in {"ERROR", "NOT_RUN"} for status in results.values())
        total = len(self.authoritative_test_names)
        rate = passed / total
        raw_rate = outcome.get("pass_rate")
        if (
            outcome.get("passed_count") != passed
            or outcome.get("failed_count") != failed
            or outcome.get("error_count") != errored
            or outcome.get("total_count") != total
            or isinstance(raw_rate, bool)
            or not isinstance(raw_rate, int | float)
            or not math.isclose(float(raw_rate), rate, rel_tol=0.0, abs_tol=1e-9)
        ):
            return {
                "ok": False,
                "error": "DeNovoSWE outcome counts are inconsistent",
                "deterministic": True,
                "process_cleanup_confirmed": outcome.get("process_cleanup_confirmed") is True,
            }
        return {
            "ok": True,
            "test_results": {str(key): str(value) for key, value in results.items()},
            "outcome": dict(outcome),
            "process_cleanup_confirmed": outcome.get("process_cleanup_confirmed") is True,
            "result_source": "denovoswe_official",
            "log_tail": str(outcome.get("log_tail") or ""),
        }

    def _parse_r2e_probe_output(self) -> dict[str, Any]:
        encoded = base64.b64encode(
            json.dumps(
                {
                    "path": "/tmp/test_output.txt",
                    "tail_chars": 2000,
                    "expected_tests": sorted(self.target_results),
                }
            ).encode()
        ).decode("ascii")
        command = f"{self._helper_python()} -c {shlex.quote(_PARSE_R2E_LOG_SCRIPT)} {shlex.quote(encoded)}"
        try:
            raw = self.shadow.exec(command, timeout=30, user=self.task.metadata.get("verifier_user"))
            parsed = json.loads(str(raw).strip())
            return parsed if isinstance(parsed, dict) else {"ok": False, "error": "parser returned a non-object"}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _parse_swerebench_v2_probe_output(self) -> dict[str, Any]:
        if self._parser_script_path is None or self._parser_contract_path is None:
            return {
                "ok": False,
                "error": "SWE-rebench parser transport was not initialized",
                "deterministic": True,
            }
        helper_command = f"{self._helper_python()} {shlex.quote(self._parser_script_path)} {shlex.quote(self._parser_contract_path)}"
        if self._uses_non_python_evidence():
            helper_command += " " + shlex.quote(getattr(self, "_non_python_probe_token", ""))
        framed = frame_structured_command(helper_command)
        self._parser_command_length = len(framed.command.encode("utf-8"))
        if self._parser_command_length >= 4096:
            return {
                "ok": False,
                "error": f"parser command exceeds 4 KiB: {self._parser_command_length}",
                "deterministic": True,
            }
        try:
            raw = self.shadow.exec(
                framed.command,
                timeout=30,
                user=self.task.metadata.get("verifier_user"),
            )
            response = parse_structured_command_output(raw, framed.nonce)
        except StructuredCommandProtocolError as exc:
            return {"ok": False, "error": str(exc), "transport_error": True}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "transport_error": True}
        if response.exit_code != 0:
            return {
                "ok": False,
                "error": f"parser helper exited {response.exit_code}: {response.stderr[-1000:]}",
                "deterministic": True,
            }
        try:
            parsed = json.loads(response.stdout.strip())
        except Exception as exc:
            return {
                "ok": False,
                "error": f"parser returned invalid JSON: {exc}",
                "deterministic": True,
            }
        if not isinstance(parsed, dict):
            return {"ok": False, "error": "parser returned a non-object", "deterministic": True}
        if not parsed.get("ok") or parsed.get("collection_aborted"):
            return parsed
        if parsed.get("schema_version") != 1:
            return {"ok": False, "error": "compact parser schema mismatch", "deterministic": True}
        statuses = parsed.get("statuses")
        if not isinstance(statuses, list) or len(statuses) != len(self.authoritative_test_names):
            return {
                "ok": False,
                "error": "compact parser status vector length mismatch",
                "deterministic": True,
            }
        allowed_statuses = {"PASSED", "FAILED", "ERROR", "SKIPPED", None}
        if any(status not in allowed_statuses for status in statuses):
            return {"ok": False, "error": "compact parser contains invalid status", "deterministic": True}
        extra_tests = parsed.get("extra_tests") or []
        if not isinstance(extra_tests, list) or any(not isinstance(name, str) for name in extra_tests):
            return {"ok": False, "error": "compact parser contains invalid extra tests", "deterministic": True}
        ignored_invalid_extra_tests = parsed.get("ignored_invalid_extra_tests") or []
        if not isinstance(ignored_invalid_extra_tests, list) or any(not isinstance(name, str) or not name for name in ignored_invalid_extra_tests):
            return {
                "ok": False,
                "error": "compact parser contains invalid ignored-extra diagnostics",
                "deterministic": True,
            }
        result_source = str(parsed.get("result_source") or "pytest_summary")
        if result_source not in {
            "pytest_summary",
            "pytest_plugin",
            "official_log_parser",
        }:
            return {
                "ok": False,
                "error": "compact parser contains invalid result source",
                "deterministic": True,
            }
        if self.result_parser == SWEREBENCH_V2_OFFICIAL_RESULT_PARSER and result_source != "official_log_parser":
            return {
                "ok": False,
                "error": "official v2 result source mismatch",
                "deterministic": True,
            }
        extra_results = parsed.get("extra_results")
        if extra_results is None:
            extra_results = {name: "NOT_RUN" for name in extra_tests}
        if (
            not isinstance(extra_results, dict)
            or set(extra_results) != set(extra_tests)
            or any(not isinstance(name, str) or status not in {"PASSED", "FAILED", "ERROR", "SKIPPED", "NOT_RUN"} for name, status in extra_results.items())
        ):
            return {"ok": False, "error": "compact parser contains invalid extra test results", "deterministic": True}
        test_results = {name: status for name, status in zip(self.authoritative_test_names, statuses, strict=True) if status is not None}
        # Extras are diagnostic-only under annotated-subset. Their concrete
        # status cannot affect verification potential, so retain names compactly.
        test_results.update(extra_results)
        return {
            "ok": True,
            "test_results": test_results,
            "result_source": result_source,
            "collector_active": bool(parsed.get("collector_active")),
            "execution_evidence": bool(parsed.get("execution_evidence")),
            **({"non_python_evidence": parsed.get("non_python_evidence", {})}
               if self._uses_non_python_evidence() else {}),
            "partial_failure_type": parsed.get("partial_failure_type"),
            "artifact_complete": parsed.get("artifact_complete"),
            "log_parser": parsed.get("log_parser"),
            "language": parsed.get("language"),
            "ignored_invalid_extra_tests": sorted(set(ignored_invalid_extra_tests)),
            "log_tail": str(parsed.get("log_tail") or ""),
        }

    def _restore_fixture_link(self) -> str | None:
        if self.result_parser != R2E_RESULT_PARSER:
            return None
        if self._fixture_link_existed:
            return None
        root = self.harness._repository_root(self.task)
        link = f"{root.rstrip('/')}/r2e_tests"
        try:
            self.shadow.exec(f"if [ -L {shlex.quote(link)} ]; then rm -- {shlex.quote(link)}; fi", timeout=10, user=self.task.metadata.get("verifier_user"))
        except Exception as exc:
            error = f"failed to restore r2e_tests link: {exc}"
            self._disable(error, "fixture_cleanup")
            return error
        return None

    def _potential(self, results: dict[str, str]) -> float:
        if self.runtime_baseline_results is None:
            raise RuntimeError("runtime baseline is unavailable")
        if self.pass_rate_mode:
            return sum(results.get(name) == "PASSED" for name in self.authoritative_test_names) / len(self.authoritative_test_names)
        failing_partition = [name for name, target in self.target_results.items() if self.runtime_baseline_results[name] != target]
        stable_partition = [name for name, target in self.target_results.items() if self.runtime_baseline_results[name] == target]
        fixed = sum(str(results.get(name, "NOT_RUN")).upper() == "PASSED" for name in failing_partition)
        for name in stable_partition:
            status = str(results.get(name, "NOT_RUN")).upper()
            if status == "PASSED":
                self._confirmed_p2p_regressions[name] = False
            elif status in {"FAILED", "ERROR", "ERRORS", "ERRORED"}:
                self._confirmed_p2p_regressions[name] = True
        regressed = sum(self._confirmed_p2p_regressions.values())
        self._confirmed_p2p_regression_count = regressed
        p_value = fixed / len(failing_partition) if failing_partition else 0.0
        b_value = regressed / len(stable_partition) if stable_partition else 0.0
        return p_value - self.beta_any * float(b_value > 0) - self.beta_frac * b_value

    def _partition_potential(self, partitions: TestPartitionResult) -> float:
        if self.runtime_baseline_results is None:
            raise RuntimeError("runtime baseline is unavailable")
        failing_count = sum(self.runtime_baseline_results[name] != target for name, target in self.target_results.items())
        stable_count = len(self.target_results) - failing_count
        if partitions.f2p.expected != failing_count or partitions.p2p.expected != stable_count:
            raise ValueError("partition expected counts do not match baseline contract")
        explicit_regressions = partitions.p2p.failed + partitions.p2p.errored
        if partitions.p2p.is_complete:
            self._confirmed_p2p_regression_count = explicit_regressions
        else:
            self._confirmed_p2p_regression_count = max(
                self._confirmed_p2p_regression_count,
                explicit_regressions,
            )
        fixed_fraction = partitions.f2p.passed / failing_count if failing_count else 0.0
        regression_fraction = self._confirmed_p2p_regression_count / stable_count if stable_count else 0.0
        return fixed_fraction - self.beta_any * float(regression_fraction > 0) - self.beta_frac * regression_fraction

    def _accepted_baseline_potential(self) -> float:
        return self._current_potential if self.pass_rate_mode else 0.0

    def _pending_event(self, raw: dict[str, Any]) -> TestEvent:
        return TestEvent(
            probe_id=f"{raw['action_id']}:probe",
            action_id=str(raw["action_id"]),
            turn_id=int(raw["turn_id"]),
            repo_state=str(raw["repo_state_after"]),
            status=TestEventStatus.PENDING,
        )

    def _result_event(
        self,
        probe_id: str,
        repo_state: str,
        status: TestEventStatus,
        results: dict[str, str],
        action_id: str | None,
        turn_id: int | None,
        duration: float,
        exit_code: int | None,
        shadow_repo_state: str | None,
        missing: list[str],
        extra: list[str],
        *,
        trusted: bool = False,
        failure_type: str | None = None,
        error: str | None = None,
        potential_before: float | None = None,
        potential_after: float | None = None,
        log_tail: str = "",
        continuity_status: TestEventContinuityStatus | None = None,
        result_source: TestResultSource | None = None,
        probe_attempts: int = 0,
        state_mismatch_paths: list[str] | None = None,
        ignored_extra_tests: list[str] | None = None,
        restore_confirmation: RestoreConfirmation | None = None,
        timed_out: bool = False,
        result_completeness: str = "none",
        imputed_tests: list[str] | None = None,
        contract_fill_policy: str | None = None,
        contract_fill_source: str | None = None,
        failure_stage: str | None = None,
        failure_origin: str | None = None,
        failure_evidence: dict[str, Any] | None = None,
        partition_counts: TestPartitionResult | None = None,
        count_observation: TestCountObservation | None = None,
    ) -> TestEvent:
        default_stage, default_origin = _default_failure_context(failure_type)
        failure_stage = failure_stage or default_stage
        failure_origin = failure_origin or default_origin
        evidence = dict(failure_evidence or {})
        if exit_code is not None:
            evidence.setdefault("exit_code", exit_code)
        if timed_out:
            evidence.setdefault("timed_out", True)
        if count_observation is not None:
            passed = count_observation.passed
            failed = count_observation.failed + count_observation.skipped
            errored = (
                count_observation.errored
                + count_observation.unclassified
            )
            not_run = count_observation.not_run
        elif partition_counts is not None:
            groups = (partition_counts.f2p, partition_counts.p2p)
            passed = sum(group.passed for group in groups)
            failed = sum(group.failed for group in groups)
            errored = sum(group.errored for group in groups)
            not_run = sum(group.not_run for group in groups)
        else:
            passed = sum(value == "PASSED" for value in results.values())
            failed = sum(value == "FAILED" for value in results.values())
            errored = sum(value == "ERROR" for value in results.values())
            not_run = sum(value == "NOT_RUN" for value in results.values())
        return TestEvent(
            probe_id=probe_id,
            action_id=action_id,
            turn_id=turn_id,
            repo_state=repo_state,
            shadow_repo_state=shadow_repo_state,
            status=status,
            trusted=trusted,
            continuity_status=continuity_status or (TestEventContinuityStatus.INTACT if trusted else TestEventContinuityStatus.LOST),
            result_source=result_source or TestResultSource.PYTEST_SUMMARY,
            result_completeness=result_completeness,
            partition_counts=partition_counts,
            count_observation=count_observation,
            imputed_tests=list(imputed_tests or []),
            contract_fill_policy=contract_fill_policy,
            contract_fill_source=contract_fill_source,
            failure_type=failure_type,
            failure_stage=failure_stage,
            failure_origin=failure_origin,
            failure_evidence=evidence,
            exit_code=exit_code,
            timed_out=timed_out,
            test_results=results,
            missing_tests=missing,
            extra_tests=extra,
            ignored_extra_tests=list(ignored_extra_tests or []),
            passed=passed,
            failed=failed,
            errored=errored,
            not_run=not_run,
            probe_attempts=probe_attempts,
            state_mismatch_paths=list(state_mismatch_paths or []),
            restore_confirmation=restore_confirmation,
            duration=duration,
            test_potential_before=potential_before,
            test_potential_after=potential_after,
            error=error,
            log_tail=log_tail,
        )

    def _failure_test_event(
        self,
        probe_id: str,
        repo_state: str,
        status: TestEventStatus,
        failure_type: str,
        error: str,
        action_id: str | None = None,
        turn_id: int | None = None,
        *,
        duration: float | None = None,
        exit_code: int | None = None,
        timed_out: bool = False,
        shadow_repo_state: str | None = None,
        log_tail: str = "",
        continuity_status: TestEventContinuityStatus | None = None,
        result_source: TestResultSource | None = None,
        probe_attempts: int = 0,
        state_mismatch_paths: list[str] | None = None,
        restore_confirmation: RestoreConfirmation | None = None,
        failure_stage: str | None = None,
        failure_origin: str | None = None,
        failure_evidence: dict[str, Any] | None = None,
    ) -> TestEvent:
        default_stage, default_origin = _default_failure_context(failure_type)
        failure_stage = failure_stage or default_stage
        failure_origin = failure_origin or default_origin
        evidence = dict(failure_evidence or {})
        if exit_code is not None:
            evidence.setdefault("exit_code", exit_code)
        if timed_out:
            evidence.setdefault("timed_out", True)
        if state_mismatch_paths:
            evidence.setdefault("state_mismatch_paths", list(state_mismatch_paths)[:50])
        return TestEvent(
            probe_id=probe_id,
            action_id=action_id,
            turn_id=turn_id,
            repo_state=repo_state,
            shadow_repo_state=shadow_repo_state,
            status=status,
            continuity_status=continuity_status or TestEventContinuityStatus.LOST,
            result_source=result_source,
            failure_type=failure_type,
            failure_stage=failure_stage,
            failure_origin=failure_origin,
            failure_evidence=evidence,
            exit_code=exit_code,
            timed_out=timed_out,
            duration=duration,
            probe_attempts=probe_attempts,
            state_mismatch_paths=list(state_mismatch_paths or []),
            restore_confirmation=restore_confirmation,
            error=error,
            log_tail=log_tail,
        )

    def _set_failure_event(
        self,
        raw: dict[str, Any],
        status: TestEventStatus,
        failure_type: str,
        error: str,
        *,
        timed_out: bool = False,
        state_mismatch_paths: list[str] | None = None,
        count_terminal: bool = False,
        failure_evidence: dict[str, Any] | None = None,
    ) -> None:
        existing = raw.get("test_event") if isinstance(raw.get("test_event"), dict) else {}
        event = self._failure_test_event(
            str(existing.get("probe_id") or f"{raw.get('action_id', self.uid)}:probe"),
            str(raw.get("repo_state_after") or "unavailable"),
            status,
            failure_type,
            error,
            str(raw.get("action_id") or "") or None,
            int(raw["turn_id"]) if isinstance(raw.get("turn_id"), int) else None,
            timed_out=timed_out,
            state_mismatch_paths=state_mismatch_paths,
            failure_evidence=failure_evidence,
        )
        merge_updates = {
            key: existing.get(key)
            for key in (
                "probe_merge_group_id",
                "probe_merge_group_size",
                "probe_merge_group_index",
                "probe_merge_anchor_action_id",
                "probe_merge_decision_wait_s",
                "probe_sequence",
                "shadow_lane_id",
                "shadow_lane_attempts",
                "potential_reanchored",
            )
            if existing.get(key) is not None
        }
        if merge_updates:
            event = event.model_copy(update=merge_updates)
        self._replace_test_event(raw, event)
        if count_terminal:
            self._counts["terminal_probes"] += 1

    def _set_job_failure_events(
        self,
        job: _ReplayJob | _ProbeJob,
        status: TestEventStatus,
        failure_type: str,
        error: str,
        *,
        state_mismatch_paths: list[str] | None = None,
        count_terminal: bool = False,
        failure_evidence: dict[str, Any] | None = None,
    ) -> int:
        marked = 0
        targets = job.probe_events or (
            (job.event,)
            if isinstance(job, _ProbeJob) or job.trigger_probe
            else ()
        )
        for raw in targets:
            existing = raw.get("test_event")
            if not isinstance(existing, dict) or existing.get("status") not in {
                "pending",
                "running",
            }:
                continue
            self._set_failure_event(
                raw,
                status,
                failure_type,
                error,
                state_mismatch_paths=state_mismatch_paths,
                failure_evidence=failure_evidence,
            )
            marked += 1
        if count_terminal and marked:
            self._counts["terminal_probes"] += 1
        return marked

    def _replace_job_test_events(
        self,
        job: _ReplayJob | _ProbeJob,
        event: TestEvent,
    ) -> None:
        targets = job.probe_events or (job.event,)
        for raw in targets:
            existing = (
                raw.get("test_event")
                if isinstance(raw.get("test_event"), dict)
                else {}
            )
            payload = event.model_dump(mode="json")
            payload["action_id"] = str(raw.get("action_id") or "") or None
            payload["turn_id"] = (
                int(raw["turn_id"])
                if isinstance(raw.get("turn_id"), int)
                else None
            )
            for key in (
                "probe_merge_group_id",
                "probe_merge_group_size",
                "probe_merge_group_index",
                "probe_merge_anchor_action_id",
                "probe_merge_decision_wait_s",
                "probe_sequence",
                "shadow_lane_id",
                "shadow_lane_attempts",
                "potential_reanchored",
            ):
                if existing.get(key) is not None:
                    payload[key] = existing[key]
            self._replace_test_event(raw, TestEvent.model_validate(payload))

    def _replace_test_event(self, raw: dict[str, Any], event: TestEvent) -> None:
        with self._lock:
            if not self._sealed:
                raw["test_event"] = event.model_dump(mode="json")

    def _replace_baseline_event(self, event: TestEvent) -> None:
        with self._lock:
            if not self._sealed:
                self._baseline_event = event.model_dump(mode="json")

    def _mark_incomplete_baseline_timeout(
        self,
        error: str,
        *,
        failure_type: str = "finalize_stall_timeout",
    ) -> bool:
        if self._baseline_event is not None and self._baseline_event.get("status") not in {"pending", "running"}:
            return False
        self._baseline_event = self._failure_test_event(
            probe_id=f"{self.uid}:baseline",
            repo_state="unavailable",
            status=TestEventStatus.TIMEOUT,
            failure_type=failure_type,
            error=error,
            timed_out=True,
        ).model_dump(mode="json")
        self._counts["probes_timed_out"] += 1
        return True

    def _disable(self, error: str, failure_type: str = "runtime") -> None:
        with self._progress_condition:
            if self._sealed:
                return
            first_failure = not self._disabled
            self._disabled = True
            self._finish_probe_merge_run_locked()
            if first_failure and self._probe_merge_members:
                self._fail_probe_merge_group_locked(
                    TestEventStatus.SKIPPED,
                    failure_type,
                    error or "shadow runtime became unavailable",
                )
            if error and error not in self._errors:
                self._errors.append(error)
            if first_failure:
                self._failure_counts[failure_type] = self._failure_counts.get(failure_type, 0) + 1
            self._progress_condition.notify_all()

    def _set_current_job(self, job_type: str) -> None:
        with self._progress_condition:
            if self._sealed:
                return
            self._current_job_type = job_type
            self._current_job_started_at = time.monotonic()
            self._progress_condition.notify_all()

    def _note_progress(self, job_type: str) -> None:
        with self._progress_condition:
            if self._sealed:
                return
            self._progress_sequence += 1
            self._last_progress_at = time.monotonic()
            self._current_job_type = job_type
            self._current_job_started_at = self._last_progress_at
            self._progress_condition.notify_all()

    def _enqueue_stop(self) -> None:
        with self._lock:
            if not self._stop_enqueued:
                self._stop_enqueued = True
                self._queue.put(_STOP)

    def _mark_pending_events(
        self,
        episode: Episode,
        status: TestEventStatus,
        failure_type: str,
        error: str,
        *,
        timed_out: bool = False,
    ) -> int:
        marked = 0
        timed_out_probe_ids: set[str] = set()
        for trajectory in episode.trajectories:
            for step in trajectory.steps:
                metadata = step.metadata if isinstance(step.metadata, dict) else {}
                raw = metadata.get(ACTION_EVENT_METADATA_KEY)
                if isinstance(raw, dict) and isinstance(raw.get("test_event"), dict) and raw["test_event"].get("status") in {"pending", "running"}:
                    probe_id = raw["test_event"].get("probe_id")
                    if isinstance(probe_id, str) and probe_id:
                        timed_out_probe_ids.add(probe_id)
                    self._set_failure_event(raw, status, failure_type, error, timed_out=timed_out)
                    marked += 1
        if timed_out:
            self._counts["probes_timed_out"] += (
                len(timed_out_probe_ids) or marked
            )
        return marked

    def _write_episode_summary(self, episode: Episode) -> None:
        summary = self.summary()
        episode.metadata[SHADOW_METADATA_KEY] = summary
        for key in (
            "replays_scheduled",
            "replays_skipped_no_repo_change",
            "replays_completed",
            "probes_scheduled",
            "probes_completed",
            "probes_failed",
            "probes_timed_out",
            "state_mismatches",
            "max_queue_depth",
            "trusted_probes",
            "recoverable_probes",
            "terminal_probes",
            "terminal_failures",
            "suppressed_probes",
            "collection_aborts",
            "probe_retries",
            "verifier_side_effects_restored",
            "trusted_not_run_probes",
            "replay_execution_mismatches",
            "replay_timeout_mismatches",
            "replay_observation_mismatches",
            "restore_protocol_warnings",
            "restore_fingerprint_fallbacks",
            "pytest_plugin_probes",
            "controlled_timeout_probes",
            "contract_assisted_baselines",
            "contract_assisted_probes",
            "parameter_id_aliases",
            "final_verifiers",
            "probe_merge_candidate_steps",
            "probe_merge_groups",
            "probe_merge_steps",
            "probe_merge_saved_probes",
        ):
            episode.metrics[f"shadow/{key}"] = summary[key]
        episode.metrics["shadow/duration"] = summary["duration"]

    @staticmethod
    def _copy_results(raw_episode: Episode, enriched_episode: Episode) -> None:
        by_id: dict[str, dict[str, Any]] = {}
        for trajectory in raw_episode.trajectories:
            for step in trajectory.steps:
                metadata = step.metadata if isinstance(step.metadata, dict) else {}
                event = metadata.get(ACTION_EVENT_METADATA_KEY)
                if isinstance(event, dict) and isinstance(event.get("action_id"), str):
                    by_id[event["action_id"]] = copy.deepcopy(event)
        for trajectory in enriched_episode.trajectories:
            for step in trajectory.steps:
                metadata = step.metadata if isinstance(step.metadata, dict) else {}
                owner = metadata
                event = metadata.get(ACTION_EVENT_METADATA_KEY)
                if not isinstance(event, dict):
                    agent_metadata = metadata.get("agent_step_metadata")
                    if isinstance(agent_metadata, dict):
                        owner = agent_metadata
                        event = agent_metadata.get(ACTION_EVENT_METADATA_KEY)
                if isinstance(event, dict) and event.get("action_id") in by_id:
                    owner[ACTION_EVENT_METADATA_KEY] = copy.deepcopy(by_id[event["action_id"]])


class ParallelDenovoShadowRuntime(ShadowSandboxRuntime):
    """DeNovo-only pool of independent replay/probe lanes.

    The inherited scheduler remains authoritative for action eligibility and
    probe merging.  Its queue writes are translated into an immutable replay
    journal plus a FIFO of physical probes.  Each lane advances only its own
    replay cursor and is exclusively owned by one worker until verifier
    restore has completed.
    """

    def __init__(
        self,
        harness: Any,
        task: Task,
        primary: Any,
        shadows: tuple[Any, ...],
        uid: str,
    ) -> None:
        if len(shadows) <= 1:
            raise ValueError("parallel DeNovo shadow runtime requires at least two lanes")
        super().__init__(harness, task, primary, shadows[0], uid)
        if not self.pass_rate_mode or not self.milestone_only:
            raise ValueError(
                "parallel shadow sandboxes are supported only for background DeNovoSWE verification"
            )

        self.shadows = tuple(shadows)
        self._parallel_journal: list[_ParallelReplayEntry] = []
        self._parallel_sealed_journal: tuple[_ParallelReplayEntry, ...] | None = None
        self._parallel_cursor_by_event: dict[int, int] = {}
        self._parallel_cursor_by_action: dict[str, int] = {}
        self._parallel_probe_queue: deque[_ParallelProbeJob] = deque()
        self._parallel_probe_jobs: list[_ParallelProbeJob] = []
        self._parallel_replayed_entries: set[int] = set()
        self._parallel_input_sealed = False
        self._parallel_results_frozen = False
        self._parallel_baseline_ready = False
        self._parallel_baseline_failed = False
        self._parallel_inflight = 0
        self._parallel_peak_inflight = 0
        self._parallel_started_count = 0
        self._parallel_queue_wait_samples_s: list[float] = []
        self._parallel_reconciled = False
        self._parallel_lanes = [
            _ParallelShadowLane(
                lane_id=index,
                sandbox=sandbox,
                runtime=ShadowSandboxRuntime(
                    harness,
                    task,
                    primary,
                    sandbox,
                    f"{uid}:lane-{index}",
                ),
            )
            for index, sandbox in enumerate(shadows)
        ]
        self._queue = _ParallelProbeQueueAdapter(self)
        self._counts.update(
            {
                "probe_lane_attempts": 0,
                "lane_replay_actions": 0,
                "probe_redispatches": 0,
                "lane_failures": 0,
                "potential_reanchors": 0,
            }
        )
        self._background_lifecycle.update(
            {
                "schema_version": 2,
                "shadow_sandboxes_requested": len(shadows),
                "shadow_sandboxes_started": 0,
                "shadow_sandboxes_healthy": 0,
                "peak_parallel_probes": 0,
            }
        )

    @classmethod
    def create_and_start(
        cls,
        harness: Any,
        task: Task,
        primary: Any,
        shadows: tuple[Any, ...],
        uid: str,
    ) -> ParallelDenovoShadowRuntime:
        runtime = cls(harness, task, primary, tuple(shadows), uid)
        runtime.start()
        return runtime

    def start(self) -> None:
        """Atomically validate every lane before starting any worker."""

        tests_dir = self.task.task_dir / "tests"
        primary_state = self.harness._codeflow_capture_repo_snapshot(
            self.primary,
            self.task,
            None,
        )
        if primary_state.status != RepoStateStatus.OK or not primary_state.fingerprint:
            raise RuntimeError(
                "initial primary repository probe failed: "
                f"{primary_state.status.value}:{primary_state.error}"
            )
        root = self.harness._repository_root(self.task)
        root_q = shlex.quote(root)
        primary_root = str(
            self.primary.exec(f"cd {root_q} && pwd -P", timeout=10)
        ).strip()
        if primary_root != root:
            raise RuntimeError(
                "initial primary workdir mismatch: "
                f"configured={root}, primary={primary_root}"
            )

        for lane in self._parallel_lanes:
            from rllm.sandbox.verifier_assets import upload_verifier_assets
            upload_verifier_assets(lane.sandbox, self.task, tests_dir)
            lane_state = self.harness._codeflow_capture_repo_snapshot(
                lane.sandbox,
                self.task,
                None,
            )
            if lane_state.status != RepoStateStatus.OK:
                raise RuntimeError(
                    f"initial lane {lane.lane_id} repository probe failed: "
                    f"{lane_state.status.value}:{lane_state.error}"
                )
            if (
                lane_state.fingerprint != primary_state.fingerprint
                or lane_state.baseline_ref != primary_state.baseline_ref
            ):
                raise RuntimeError(
                    f"initial primary/lane-{lane.lane_id} repository mismatch: "
                    f"primary={primary_state.fingerprint}, "
                    f"shadow={lane_state.fingerprint}"
                )
            lane_root = str(
                lane.sandbox.exec(f"cd {root_q} && pwd -P", timeout=10)
            ).strip()
            if lane_root != root:
                raise RuntimeError(
                    f"initial lane {lane.lane_id} workdir mismatch: "
                    f"configured={root}, shadow={lane_root}"
                )
            lane.runtime._baseline_ref = lane_state.baseline_ref
            lane.runtime._status = "running"
            self._parallel_started_count += 1

        self._baseline_ref = primary_state.baseline_ref
        self._status = "running"
        self._background_lifecycle["shadow_sandboxes_started"] = (
            self._parallel_started_count
        )
        self._background_lifecycle["shadow_sandboxes_healthy"] = len(
            self._parallel_lanes
        )
        self._thread = threading.Thread(
            target=self._parallel_supervisor,
            name=f"rllm-shadow-pool-{self.uid}",
            daemon=True,
        )
        self._thread.start()

    def _parallel_accept_replay_job(self, job: _ReplayJob) -> None:
        with self._progress_condition:
            if self._parallel_input_sealed:
                raise RuntimeError("cannot append replay work after pool input seal")
            immutable = _ParallelReplayEntry(
                replay_event=copy.deepcopy(job.replay_event),
            )
            self._parallel_journal.append(immutable)
            cursor = len(self._parallel_journal)
            self._parallel_cursor_by_event[id(job.event)] = cursor
            action_id = str(job.event.get("action_id") or "")
            if action_id:
                self._parallel_cursor_by_action[action_id] = cursor
            if job.trigger_probe:
                self._parallel_enqueue_probe_locked(
                    event=job.event,
                    probe_event=job.replay_event,
                    probe_events=job.probe_events or (job.event,),
                    target_cursor=cursor,
                )
            self._progress_condition.notify_all()

    def _parallel_accept_probe_job(self, job: _ProbeJob) -> None:
        with self._progress_condition:
            if self._parallel_input_sealed:
                raise RuntimeError("cannot append probe work after pool input seal")
            target_cursor = self._parallel_cursor_by_event.get(id(job.event))
            if target_cursor is None:
                action_id = str(job.event.get("action_id") or "")
                target_cursor = self._parallel_cursor_by_action.get(action_id)
            if target_cursor is None:
                raise RuntimeError("probe merge anchor is absent from replay journal")
            self._parallel_enqueue_probe_locked(
                event=job.event,
                probe_event=job.probe_event,
                probe_events=job.probe_events or (job.event,),
                target_cursor=target_cursor,
            )
            self._progress_condition.notify_all()

    def _parallel_enqueue_probe_locked(
        self,
        *,
        event: dict[str, Any],
        probe_event: dict[str, Any],
        probe_events: tuple[dict[str, Any], ...],
        target_cursor: int,
    ) -> None:
        sequence = len(self._parallel_probe_jobs)
        physical = _ParallelProbeJob(
            sequence=sequence,
            event=event,
            probe_event=copy.deepcopy(probe_event),
            probe_events=tuple(probe_events),
            target_cursor=target_cursor,
            enqueued_at=time.monotonic(),
            attempted_lane_ids=set(),
        )
        self._parallel_probe_jobs.append(physical)
        self._parallel_probe_queue.append(physical)
        for raw in physical.probe_events:
            payload = raw.get("test_event")
            if isinstance(payload, dict):
                payload["probe_sequence"] = sequence

    def _parallel_seal_queue(self) -> None:
        with self._progress_condition:
            self._parallel_input_sealed = True
            if self._parallel_sealed_journal is None:
                self._parallel_sealed_journal = tuple(
                    _ParallelReplayEntry(
                        replay_event=copy.deepcopy(entry.replay_event)
                    )
                    for entry in self._parallel_journal
                )
            self._progress_condition.notify_all()

    def _parallel_queue_size(self) -> int:
        with self._lock:
            return len(self._parallel_probe_queue)

    def _parallel_supervisor(self) -> None:
        try:
            if not self._parallel_initialize_baseline():
                return
            for lane in self._parallel_lanes:
                if not lane.healthy:
                    continue
                lane.worker = threading.Thread(
                    target=self._parallel_lane_worker,
                    args=(lane,),
                    name=f"rllm-shadow-{self.uid}-lane-{lane.lane_id}",
                    daemon=True,
                )
                lane.worker.start()
            for lane in self._parallel_lanes:
                if lane.worker is not None:
                    lane.worker.join()
        except _ShadowCancelled:
            if not self._parallel_results_frozen:
                raise
        except Exception as exc:
            logger.exception("[%s] parallel shadow supervisor failed", self.uid)
            self._disable(f"parallel shadow supervisor failed: {exc}", "worker")
        finally:
            with self._progress_condition:
                self._current_job_type = None
                self._current_job_started_at = None
                self._progress_condition.notify_all()

    def _disable(self, error: str, failure_type: str = "runtime") -> None:
        """Atomically freeze pool outputs whenever the runtime terminates."""

        with self._progress_condition:
            self._parallel_results_frozen = True
            # Lane runtimes have their own cancellation events. Freezing the
            # pool alone prevents publication, but does not stop their RPCs.
            for lane in getattr(self, "_parallel_lanes", ()):
                lane.runtime._cancel_requested.set()
            super()._disable(error, failure_type)

    def _parallel_initialize_baseline(self) -> bool:
        last_event: TestEvent | None = None
        attempt_limit = min(2, len(self._parallel_lanes))
        for attempt, lane in enumerate(
            self._parallel_lanes[:attempt_limit],
            start=1,
        ):
            with self._lock:
                if self._sealed or self._parallel_results_frozen:
                    return False
            try:
                succeeded = lane.runtime._initialize_baseline()
                payload = lane.runtime._baseline_event
                event = (
                    TestEvent.model_validate(payload)
                    if isinstance(payload, dict)
                    else lane.runtime._failure_test_event(
                        f"{self.uid}:baseline",
                        "unavailable",
                        TestEventStatus.ERROR,
                        "shadow_baseline_unavailable",
                        "baseline worker returned no event",
                    )
                )
            except Exception as exc:
                succeeded = False
                event = lane.runtime._failure_test_event(
                    f"{self.uid}:baseline",
                    "unavailable",
                    TestEventStatus.ERROR,
                    "shadow_baseline_unavailable",
                    str(exc),
                )
            self._retire_unconfirmed_lane(lane, event)
            event = event.model_copy(
                update={
                    "shadow_lane_id": lane.lane_id,
                    "shadow_lane_attempts": attempt,
                }
            )
            last_event = event
            if succeeded and event.trusted:
                event = event.model_copy(
                    update={
                        "test_potential_before": 0.0,
                        "test_potential_after": 0.0,
                    }
                )
                baseline_results = dict(event.test_results)
                with self._progress_condition:
                    if self._sealed or self._parallel_results_frozen:
                        return False
                    self._accept_runtime_baseline(baseline_results)
                    self._current_potential = 0.0
                    self._replace_baseline_event(event)
                    self._parallel_record_event_counts_locked(
                        event,
                        baseline=True,
                    )
                    for other in self._parallel_lanes:
                        if other.healthy:
                            other.runtime._accept_runtime_baseline(
                                baseline_results
                            )
                            other.runtime._current_potential = 0.0
                    self._parallel_baseline_ready = True
                    self._background_lifecycle[
                        "shadow_sandboxes_healthy"
                    ] = self._parallel_healthy_count_locked()
                    self._note_progress("baseline_probe")
                    self._progress_condition.notify_all()
                return True

            with self._progress_condition:
                self._parallel_quarantine_lane_locked(
                    lane,
                    event.error or event.failure_type or "baseline probe failed",
                )
                self._progress_condition.notify_all()

        with self._progress_condition:
            self._parallel_baseline_failed = True
            for lane in self._parallel_lanes:
                if lane.healthy:
                    lane.healthy = False
                    lane.failure_reason = "baseline_pool_unavailable"
            if last_event is not None:
                self._replace_baseline_event(last_event)
            self._background_lifecycle["shadow_sandboxes_healthy"] = 0
            self._disable(
                (last_event.error if last_event is not None else None)
                or "baseline probe failed on both lanes",
                "shadow_baseline_unavailable",
            )
            self._progress_condition.notify_all()
        return False

    def _parallel_lane_worker(self, lane: _ParallelShadowLane) -> None:
        while True:
            job = self._parallel_take_job(lane)
            if job is None:
                return
            event = self._parallel_execute_job(lane, job)
            self._parallel_finish_job(lane, job, event)

    def _retire_unconfirmed_lane(self, lane: _ParallelShadowLane, event: TestEvent) -> None:
        if event.failure_type != "verifier_stop_unconfirmed":
            return
        try:
            lane.sandbox.close()
        except Exception as exc:
            self._terminal_cleanup_error = exc
            self._disable(str(exc), "shadow_cleanup_unconfirmed")

    def _parallel_take_job(
        self,
        lane: _ParallelShadowLane,
    ) -> _ParallelProbeJob | None:
        with self._progress_condition:
            while True:
                if (
                    self._sealed
                    or self._parallel_results_frozen
                    or not lane.healthy
                ):
                    return None
                if self._disabled and self._stop_enqueued:
                    return None
                if self._parallel_probe_queue:
                    head = self._parallel_probe_queue[0]
                    if (
                        lane.lane_id not in head.attempted_lane_ids
                        and lane.cursor <= head.target_cursor
                    ):
                        self._parallel_probe_queue.popleft()
                        if head.attempted_lane_ids:
                            self._counts["probe_redispatches"] += 1
                        head.attempted_lane_ids.add(lane.lane_id)
                        lane.busy = True
                        lane.current_probe_sequence = head.sequence
                        self._parallel_inflight += 1
                        self._parallel_peak_inflight = max(
                            self._parallel_peak_inflight,
                            self._parallel_inflight,
                        )
                        self._background_lifecycle["peak_parallel_probes"] = (
                            self._parallel_peak_inflight
                        )
                        self._parallel_queue_wait_samples_s.append(
                            max(0.0, time.monotonic() - head.enqueued_at)
                        )
                        self._counts["probe_lane_attempts"] += 1
                        self._current_job_type = "parallel_probe"
                        self._current_job_started_at = time.monotonic()
                        return head

                    if not self._parallel_has_future_lane_locked(head):
                        self._parallel_probe_queue.popleft()
                        self._parallel_mark_unrunnable_locked(head)
                        continue

                if (
                    self._parallel_input_sealed
                    and not self._parallel_probe_queue
                    and self._parallel_inflight == 0
                ):
                    return None
                self._progress_condition.wait()

    def _parallel_has_future_lane_locked(self, job: _ParallelProbeJob) -> bool:
        return any(
            lane.healthy
            and lane.lane_id not in job.attempted_lane_ids
            and lane.cursor <= job.target_cursor
            for lane in self._parallel_lanes
        )

    def _parallel_execute_job(
        self,
        lane: _ParallelShadowLane,
        job: _ParallelProbeJob,
    ) -> TestEvent:
        try:
            for cursor in range(lane.cursor, job.target_cursor):
                entry = self._parallel_journal[cursor]
                lane.runtime._process_job(
                    _ReplayJob(
                        event={},
                        replay_event=copy.deepcopy(entry.replay_event),
                        trigger_probe=False,
                    )
                )
                if lane.runtime._disabled:
                    terminal = lane.runtime._terminal_replay or {}
                    failure_type = str(
                        terminal.get("failure_type") or "parallel_replay_failure"
                    )
                    raise RuntimeError(
                        f"{failure_type}: "
                        + str(
                            lane.runtime._errors[-1]
                            if lane.runtime._errors
                            else "lane replay continuity was lost"
                        )
                    )
                lane.cursor = cursor + 1
                with self._progress_condition:
                    if self._sealed or self._parallel_results_frozen:
                        break
                    self._counts["lane_replay_actions"] += 1
                    if cursor not in self._parallel_replayed_entries:
                        self._parallel_replayed_entries.add(cursor)
                        self._counts["replays_completed"] += 1
                    self._note_progress("replay")

            raw = job.probe_event
            with self._lock:
                if self._sealed or self._parallel_results_frozen:
                    return lane.runtime._failure_test_event(
                        str(
                            (raw.get("test_event") or {}).get("probe_id")
                            or f"{raw.get('action_id', self.uid)}:probe"
                        ),
                        str(raw.get("repo_state_after") or "unavailable"),
                        TestEventStatus.SKIPPED,
                        "suppressed_after_terminal",
                        "parallel probe was cancelled before verifier start",
                        continuity_status=TestEventContinuityStatus.LOST,
                    )
            event = lane.runtime._run_probe(
                str(
                    (raw.get("test_event") or {}).get("probe_id")
                    or f"{raw.get('action_id', self.uid)}:probe"
                ),
                str(raw["repo_state_after"]),
                str(raw.get("action_id") or "") or None,
                (
                    int(raw["turn_id"])
                    if isinstance(raw.get("turn_id"), int)
                    else None
                ),
                baseline=False,
                agent_changed_paths=_changed_file_paths(raw.get("changed_files")),
            )
        except Exception as exc:
            raw = job.probe_event
            terminal = lane.runtime._terminal_replay or {}
            failure_type = str(
                terminal.get("failure_type") or "parallel_lane_failure"
            )
            event = lane.runtime._failure_test_event(
                str(
                    (raw.get("test_event") or {}).get("probe_id")
                    or f"{raw.get('action_id', self.uid)}:probe"
                ),
                str(raw.get("repo_state_after") or "unavailable"),
                TestEventStatus.STATE_MISMATCH,
                failure_type,
                str(exc),
                str(raw.get("action_id") or "") or None,
                (
                    int(raw["turn_id"])
                    if isinstance(raw.get("turn_id"), int)
                    else None
                ),
                continuity_status=TestEventContinuityStatus.LOST,
                failure_stage="replay",
                failure_origin="shadow_lane",
            )
        self._retire_unconfirmed_lane(lane, event)
        return event.model_copy(
            update={
                "probe_sequence": job.sequence,
                "shadow_lane_id": lane.lane_id,
                "shadow_lane_attempts": len(job.attempted_lane_ids),
            }
        )

    def _parallel_finish_job(
        self,
        lane: _ParallelShadowLane,
        job: _ParallelProbeJob,
        event: TestEvent,
    ) -> None:
        with self._progress_condition:
            self._parallel_inflight = max(0, self._parallel_inflight - 1)
            lane.busy = False
            lane.current_probe_sequence = None
            if self._sealed or self._parallel_results_frozen:
                self._progress_condition.notify_all()
                return
            continuity = resolve_test_event_continuity(event)
            if continuity == TestEventContinuityStatus.LOST:
                self._parallel_quarantine_lane_locked(
                    lane,
                    event.error or event.failure_type or "lane continuity lost",
                )
                if job.redispatches < 1 and self._parallel_healthy_count_locked():
                    job.redispatches += 1
                    job.enqueued_at = time.monotonic()
                    self._parallel_probe_queue.appendleft(job)
                    self._progress_condition.notify_all()
                    return
                if self._parallel_healthy_count_locked():
                    event = self._parallel_recoverable_gap_event(
                        job,
                        event,
                        "parallel probe failed after its cross-lane redispatch",
                    )

            if not self._sealed:
                self._parallel_publish_job_locked(job, event)
            if (
                continuity == TestEventContinuityStatus.LOST
                and not self._parallel_healthy_count_locked()
            ):
                self._disable(
                    event.error or "all parallel shadow lanes lost continuity",
                    "all_shadow_lanes_lost",
                )
            self._note_progress("probe")
            self._progress_condition.notify_all()

    def _parallel_mark_unrunnable_locked(self, job: _ParallelProbeJob) -> None:
        raw = job.probe_event
        healthy = self._parallel_healthy_count_locked()
        status = (
            TestEventContinuityStatus.RECOVERABLE_GAP
            if healthy
            else TestEventContinuityStatus.LOST
        )
        event = self._failure_test_event(
            str(
                (raw.get("test_event") or {}).get("probe_id")
                or f"{raw.get('action_id', self.uid)}:probe"
            ),
            str(raw.get("repo_state_after") or "unavailable"),
            TestEventStatus.ERROR,
            "parallel_probe_redispatch_unavailable",
            "no healthy lane can replay this probe without moving backwards",
            str(raw.get("action_id") or "") or None,
            int(raw["turn_id"]) if isinstance(raw.get("turn_id"), int) else None,
            continuity_status=status,
            failure_stage="dispatch",
            failure_origin="shadow_pool",
            failure_evidence={
                "target_cursor": job.target_cursor,
                "attempted_lane_ids": sorted(job.attempted_lane_ids),
            },
        ).model_copy(
            update={
                "probe_sequence": job.sequence,
                "shadow_lane_attempts": len(job.attempted_lane_ids),
            }
        )
        if not self._sealed:
            self._parallel_publish_job_locked(job, event)
        if not healthy:
            self._disable(
                event.error or "all parallel shadow lanes lost continuity",
                "all_shadow_lanes_lost",
            )

    def _parallel_recoverable_gap_event(
        self,
        job: _ParallelProbeJob,
        event: TestEvent,
        message: str,
    ) -> TestEvent:
        evidence = dict(event.failure_evidence)
        evidence.update(
            {
                "lane_failure_recovered": True,
                "attempted_lane_ids": sorted(job.attempted_lane_ids),
            }
        )
        return event.model_copy(
            update={
                "continuity_status": TestEventContinuityStatus.RECOVERABLE_GAP,
                "failure_evidence": evidence,
                "error": event.error or message,
                "shadow_lane_attempts": len(job.attempted_lane_ids),
            }
        )

    def _parallel_publish_job_locked(
        self,
        job: _ParallelProbeJob,
        event: TestEvent,
    ) -> None:
        if self._sealed or self._parallel_results_frozen:
            return
        payload = event.model_dump(mode="json")
        job.result = copy.deepcopy(payload)
        for raw in job.probe_events:
            existing = (
                raw.get("test_event")
                if isinstance(raw.get("test_event"), dict)
                else {}
            )
            current = copy.deepcopy(payload)
            current["action_id"] = str(raw.get("action_id") or "") or None
            current["turn_id"] = (
                int(raw["turn_id"])
                if isinstance(raw.get("turn_id"), int)
                else None
            )
            for key in (
                "probe_merge_group_id",
                "probe_merge_group_size",
                "probe_merge_group_index",
                "probe_merge_anchor_action_id",
                "probe_merge_decision_wait_s",
            ):
                if existing.get(key) is not None:
                    current[key] = existing[key]
            raw["test_event"] = TestEvent.model_validate(current).model_dump(
                mode="json"
            )
        self._parallel_record_event_counts_locked(event, baseline=False)

    def _parallel_record_event_counts_locked(
        self,
        event: TestEvent,
        *,
        baseline: bool,
    ) -> None:
        continuity = resolve_test_event_continuity(event)
        if event.timed_out:
            self._counts["probes_timed_out"] += 1
        if event.trusted:
            self._counts["trusted_probes"] += 1
            if not baseline:
                self._counts["probes_completed"] += 1
            if event.result_source == TestResultSource.COLLECTION_ABORT:
                self._counts["collection_aborts"] += 1
        else:
            self._counts["probes_failed"] += 1
            if continuity == TestEventContinuityStatus.RECOVERABLE_GAP:
                self._counts["recoverable_probes"] += 1
                failure_type = event.failure_type or "probe"
                self._recoverable_failure_counts[failure_type] = (
                    self._recoverable_failure_counts.get(failure_type, 0) + 1
                )
            else:
                self._counts["terminal_probes"] += 1
                self._counts["terminal_failures"] += 1

    def _parallel_quarantine_lane_locked(
        self,
        lane: _ParallelShadowLane,
        reason: str,
    ) -> None:
        if not lane.healthy:
            return
        lane.healthy = False
        lane.busy = False
        lane.failure_reason = str(reason)[:1000]
        self._counts["lane_failures"] += 1
        self._failure_counts["shadow_lane_failure"] = (
            self._failure_counts.get("shadow_lane_failure", 0) + 1
        )
        self._background_lifecycle["shadow_sandboxes_healthy"] = (
            self._parallel_healthy_count_locked()
        )

    def _parallel_healthy_count_locked(self) -> int:
        return sum(lane.healthy for lane in self._parallel_lanes)

    def seal_milestone_input(
        self,
        *,
        primary_verifier_finished_at_monotonic: float | None = None,
    ) -> dict[str, Any]:
        result = super().seal_milestone_input(
            primary_verifier_finished_at_monotonic=(
                primary_verifier_finished_at_monotonic
            )
        )
        for lane in self._parallel_lanes:
            lane.runtime.primary = None
        return result

    def _mark_pending_events(
        self,
        episode: Episode,
        status: TestEventStatus,
        failure_type: str,
        error: str,
        *,
        timed_out: bool = False,
    ) -> int:
        marked = super()._mark_pending_events(
            episode,
            status,
            failure_type,
            error,
            timed_out=timed_out,
        )
        with self._lock:
            if not self._parallel_healthy_count_locked():
                return marked
            # The lane pool is being sealed, not proven irrecoverable.  Earlier
            # missing physical probes are verification gaps; already-completed
            # later probes remain usable as conservative absolute anchors.
            for trajectory in episode.trajectories:
                for step in trajectory.steps:
                    metadata = (
                        step.metadata if isinstance(step.metadata, dict) else {}
                    )
                    raw = metadata.get(ACTION_EVENT_METADATA_KEY)
                    payload = raw.get("test_event") if isinstance(raw, dict) else None
                    if (
                        isinstance(payload, dict)
                        and payload.get("failure_type") == failure_type
                        and payload.get("status") == status.value
                    ):
                        payload["continuity_status"] = (
                            TestEventContinuityStatus.RECOVERABLE_GAP.value
                        )
                        raw["test_event"] = TestEvent.model_validate(
                            payload
                        ).model_dump(mode="json")
        return marked

    def finalize_milestones_at_barrier(
        self,
        raw_episode: Episode,
        enriched_episode: Episode,
        *,
        barrier_started_at_monotonic: float,
        deadline_monotonic: float,
        timeout_reason: str = "batch_shadow_finalize_timeout",
    ) -> dict[str, Any]:
        super().finalize_milestones_at_barrier(
            raw_episode,
            enriched_episode,
            barrier_started_at_monotonic=barrier_started_at_monotonic,
            deadline_monotonic=deadline_monotonic,
            timeout_reason=timeout_reason,
        )
        self._parallel_reconcile_potentials()
        self._write_episode_summary(raw_episode)
        self.copy_results_to_episode(raw_episode, enriched_episode)
        with self._lock:
            return copy.deepcopy(self._background_lifecycle)

    def cancel_milestones(
        self,
        raw_episode: Episode,
        enriched_episode: Episode,
        *,
        disposition: str,
    ) -> dict[str, Any]:
        super().cancel_milestones(
            raw_episode,
            enriched_episode,
            disposition=disposition,
        )
        self._parallel_reconcile_potentials()
        self._write_episode_summary(raw_episode)
        self.copy_results_to_episode(raw_episode, enriched_episode)
        with self._lock:
            return copy.deepcopy(self._background_lifecycle)

    def _parallel_reconcile_potentials(self) -> None:
        """Apply physical probe results once, in admission order."""

        with self._lock:
            if self._parallel_reconciled:
                return
            self._parallel_reconciled = True
            if self.baseline_passed is None:
                return
            current = 0.0
            gap_open = False
            reanchors = 0
            for job in sorted(
                self._parallel_probe_jobs,
                key=lambda current_job: current_job.sequence,
            ):
                payload = next(
                    (
                        raw.get("test_event")
                        for raw in job.probe_events
                        if isinstance(raw.get("test_event"), dict)
                    ),
                    None,
                )
                if not isinstance(payload, dict):
                    gap_open = True
                    continue
                event = TestEvent.model_validate(payload)
                if (
                    event.status == TestEventStatus.COMPLETED
                    and event.trusted
                    and event.test_results
                ):
                    passed = sum(
                        event.test_results.get(name) == "PASSED"
                        for name in self.authoritative_test_names
                    )
                    potential = float(passed - self.baseline_passed)
                    reanchored = gap_open
                    before = potential if reanchored else current
                    if reanchored:
                        reanchors += 1
                    current = potential
                    gap_open = False
                    update = {
                        "test_potential_before": before,
                        "test_potential_after": potential,
                        "potential_reanchored": reanchored,
                    }
                    for raw in job.probe_events:
                        current_payload = raw.get("test_event")
                        if not isinstance(current_payload, dict):
                            continue
                        current_payload.update(update)
                        raw["test_event"] = TestEvent.model_validate(
                            current_payload
                        ).model_dump(mode="json")
                    if job.result is not None:
                        job.result.update(update)
                else:
                    gap_open = True
            self._current_potential = current
            self._counts["potential_reanchors"] = reanchors
            self._background_lifecycle["potential_reanchors"] = reanchors

    def summary(self) -> dict[str, Any]:
        base = super().summary()
        with self._lock:
            healthy = self._parallel_healthy_count_locked()
            waits = list(self._parallel_queue_wait_samples_s)
            lane_audit = [
                {
                    "lane_id": lane.lane_id,
                    "healthy": lane.healthy,
                    "busy": lane.busy,
                    "replay_cursor": lane.cursor,
                    "current_probe_sequence": lane.current_probe_sequence,
                    "failure_reason": lane.failure_reason,
                }
                for lane in self._parallel_lanes
            ]
            pool = {
                "schema_version": 1,
                "requested": len(self._parallel_lanes),
                "started": self._parallel_started_count,
                "healthy": healthy,
                "peak_parallel_probes": self._parallel_peak_inflight,
                "queued_probes": len(self._parallel_probe_queue),
                "inflight_probes": self._parallel_inflight,
                "probe_lane_attempts": self._counts["probe_lane_attempts"],
                "lane_replay_actions": self._counts["lane_replay_actions"],
                "redispatches": self._counts["probe_redispatches"],
                "lane_failures": self._counts["lane_failures"],
                "potential_reanchors": self._counts["potential_reanchors"],
                "queue_wait_samples": len(waits),
                "queue_wait_s_min": min(waits) if waits else None,
                "queue_wait_s_mean": (
                    sum(waits) / len(waits) if waits else None
                ),
                "queue_wait_s_max": max(waits) if waits else None,
                "lanes": lane_audit,
            }
            base.update(
                {
                    "runtime_mode": "parallel_denovo",
                    "shadow_pool": pool,
                    "shadow_sandboxes_requested": pool["requested"],
                    "shadow_sandboxes_started": pool["started"],
                    "shadow_sandboxes_healthy": pool["healthy"],
                    "peak_parallel_probes": pool["peak_parallel_probes"],
                }
            )
            return base

    def _write_episode_summary(self, episode: Episode) -> None:
        super()._write_episode_summary(episode)
        summary = episode.metadata.get(SHADOW_METADATA_KEY, {})
        if isinstance(summary, dict):
            for key in (
                "shadow_sandboxes_requested",
                "shadow_sandboxes_started",
                "shadow_sandboxes_healthy",
                "peak_parallel_probes",
                "probe_lane_attempts",
                "lane_replay_actions",
                "probe_redispatches",
                "lane_failures",
                "potential_reanchors",
            ):
                if key in summary:
                    episode.metrics[f"shadow/{key}"] = summary[key]


# Compatibility name retained for callers that imported the original class.


class UnavailableShadowRuntime:
    """Fail-open runtime used when an eligible task's shadow setup fails."""

    def __init__(
        self,
        uid: str,
        error: str,
        finalize_stall_timeout: float = DEFAULT_SHADOW_FINALIZE_STALL_TIMEOUT,
    ):
        self.uid = uid
        self.error = error
        self.failure_type = _setup_failure_type(error)
        self.finalize_stall_timeout = _positive_float(
            finalize_stall_timeout,
            DEFAULT_SHADOW_FINALIZE_STALL_TIMEOUT,
        )
        self.started_at = time.monotonic()
        self.setup_duration = 0.0

    def record_setup_duration(self, duration: float) -> None:
        self.setup_duration = max(0.0, float(duration))

    def schedule(self, step: Step) -> None:
        metadata = step.metadata if isinstance(step.metadata, dict) else {}
        raw = metadata.get(ACTION_EVENT_METADATA_KEY)
        if not isinstance(raw, dict) or raw.get("repo_changed") is not True or not isinstance(raw.get("repo_state_after"), str):
            return
        raw["test_event"] = TestEvent(
            probe_id=f"{raw['action_id']}:probe",
            action_id=str(raw["action_id"]),
            turn_id=int(raw["turn_id"]),
            repo_state=str(raw["repo_state_after"]),
            status=TestEventStatus.SKIPPED,
            failure_type=self.failure_type,
            failure_stage="setup",
            failure_origin=self.failure_type,
            failure_evidence={"exception": self.error[:1000]},
            error=self.error,
        ).model_dump(mode="json")

    def attach_episode(self, episode: Episode) -> None:
        episode.metadata[SHADOW_METADATA_KEY] = self.summary()

    def finalize_episode(self, raw_episode: Episode, enriched_episode: Episode | None = None) -> None:
        raw_episode.metadata[SHADOW_METADATA_KEY] = self.summary()
        if enriched_episode is not None:
            self.copy_results_to_episode(raw_episode, enriched_episode)

    def copy_results_to_episode(self, raw_episode: Episode, enriched_episode: Episode) -> None:
        ShadowSandboxRuntime._copy_results(raw_episode, enriched_episode)
        enriched_episode.metadata[SHADOW_METADATA_KEY] = self.summary()

    def abort(self) -> None:
        return

    def join_after_close(self, timeout: float = 5.0) -> None:  # noqa: ARG002
        return

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": SHADOW_SCHEMA_VERSION,
            "enabled": False,
            "status": "unavailable",
            "quality_status": "terminal_degraded",
            "baseline_source": "runtime_probe",
            "baseline_probe_succeeded": False,
            "materialized_baseline_fallback": False,
            "safe_baseline_available": False,
            "benign_baseline_partial": False,
            "actionable_baseline_failure": True,
            "verification_potential_mode": "bug_repair",
            "verification_potential_policy": "bug_repair_v2_neutral_unrun",
            "count_plan_hash": None,
            "verification_result_mode": "per_test",
            "verification_count_contract": None,
            "materialized_baseline_passed": None,
            "runtime_baseline_passed": None,
            "baseline_passed": None,
            "max_additional_passed": None,
            "baseline_collector": None,
            "contract_fill_policy": None,
            "partition_timeout_recovery": None,
            "partition_recovery_command_timeout": None,
            "partition_recovery_max_commands": None,
            "test_set_policy": None,
            "authoritative_test_count": 0,
            "parser_transport": None,
            "parser_command_length": None,
            "baseline_test_event": None,
            "verification_potential_unavailable_reason": self.failure_type,
            "baseline_reference": {
                "status": "unavailable",
                "mismatch_count": 0,
                "mismatch_tests": [],
                "truncated": False,
            },
            "probe_merge_enabled": False,
            "probe_merge_max_steps": None,
            "probe_merge_group_size_histogram": {},
            "probe_merge_same_changed_files_run_length_histogram": {},
            "probe_merge_same_changed_files_runs": 0,
            "probe_merge_same_changed_files_steps": 0,
            "probe_merge_same_changed_files_run_length_min": None,
            "probe_merge_same_changed_files_run_length_mean": None,
            "probe_merge_same_changed_files_run_length_max": None,
            "probe_merge_decision_wait_samples": 0,
            "probe_merge_decision_wait_s_min": None,
            "probe_merge_decision_wait_s_mean": None,
            "probe_merge_decision_wait_s_max": None,
            "probe_merge_pending_steps": 0,
            "probe_merge_candidate_steps": 0,
            "probe_merge_groups": 0,
            "probe_merge_steps": 0,
            "probe_merge_saved_probes": 0,
            "replays_scheduled": 0,
            "replays_skipped_no_repo_change": 0,
            "replays_completed": 0,
            "probes_scheduled": 0,
            "probes_completed": 0,
            "probes_failed": 0,
            "probes_timed_out": 0,
            "state_mismatches": 0,
            "max_queue_depth": 0,
            "trusted_probes": 0,
            "recoverable_probes": 0,
            "terminal_probes": 0,
            "terminal_failures": 1,
            "suppressed_probes": 0,
            "collection_aborts": 0,
            "probe_retries": 0,
            "verifier_side_effects_restored": 0,
            "trusted_not_run_probes": 0,
            "replay_execution_mismatches": 0,
            "replay_timeout_mismatches": 0,
            "replay_observation_mismatches": 0,
            "replay_state_capture_attempts": 0,
            "replay_state_capture_successes": 0,
            "replay_state_capture_failures": 0,
            "replay_state_recovery_attempts": 0,
            "replay_state_recovery_successes": 0,
            "replay_state_recovery_failures": 0,
            "restored_policy_rejections": 0,
            "restore_protocol_warnings": 0,
            "restore_fingerprint_fallbacks": 0,
            "pytest_plugin_probes": 0,
            "controlled_timeout_probes": 0,
            "contract_assisted_baselines": 0,
            "contract_assisted_probes": 0,
            "parameter_id_aliases": 0,
            "partition_timeout_recovery_attempts": 0,
            "partition_timeout_recovery_successes": 0,
            "partition_timeout_recovery_partial": 0,
            "partition_timeout_recovery_failures": 0,
            "count_full_probes": 0,
            "count_partial_probes": 0,
            "count_unavailable_probes": 0,
            "baseline_probe_successes": 0,
            "materialized_baseline_fallbacks": 0,
            "baseline_count_mismatches": 0,
            "terminal_replay": None,
            "replay_warnings": [],
            "restore_warnings": [],
            "timings": {
                "setup_s": self.setup_duration,
                "baseline_probe_s": 0.0,
                "replay_s": 0.0,
                "probe_s": 0.0,
                "finalize_wait_s": 0.0,
                "post_primary_wait_s": 0.0,
            },
            "failure_counts": {self.failure_type: 1},
            "recoverable_failure_counts": {},
            "timeout_policy": "bounded_post_primary_wait",
            "finalize_stall_timeout": self.finalize_stall_timeout,
            "finalize_timed_out": False,
            "progress_sequence": 0,
            "finalize_progress_resets": 0,
            "finalize_watchdog_reports": 0,
            "current_job_type": None,
            "last_progress_age_s": 0.0,
            "queue_depth_at_finalize": 0,
            "pending_events_at_timeout": 0,
            "forced_teardown_requested": False,
            "probe_timeout": None,
            "total_timeout": None,
            "duration": max(0.0, time.monotonic() - self.started_at),
            "errors": [self.error],
        }


def _positive_float(
    value: Any,
    default: float,
    *,
    label: str = "shadow finalize stall timeout",
) -> float:
    try:
        parsed = float(default if value is None else value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be positive, got {value!r}") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{label} must be positive, got {value!r}")
    return parsed


def _positive_int(value: Any, default: int, *, label: str) -> int:
    raw = default if value is None else value
    if isinstance(raw, bool):
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    try:
        parsed = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer, got {value!r}") from exc
    if isinstance(raw, float) and not raw.is_integer():
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    if isinstance(raw, str) and str(parsed) != raw.strip():
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    if parsed <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    return parsed


def _bool_value(value: Any, default: bool, *, label: str) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().casefold() in {"true", "false"}:
        return value.strip().casefold() == "true"
    raise ValueError(f"{label} must be a boolean, got {value!r}")


def _unit_float(value: Any, default: float) -> float:
    try:
        parsed = float(default if value is None else value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if 0.0 <= parsed <= 1.0 else float(default)


__all__ = [
    "DEFAULT_TEST_RESULTS_PATH",
    "R2E_RESULT_PARSER",
    "ParallelDenovoShadowRuntime",
    "SUPPORTED_RESULT_PARSERS",
    "SWEREBENCH_V2_RESULT_PARSER",
    "ShadowSandboxRuntime",
    "DEFAULT_SHADOW_FINALIZE_TIMEOUT",
    "SHADOW_METADATA_KEY",
    "UnavailableShadowRuntime",
    "effective_rollout_concurrency",
    "r2egym_shadow_eligibility",
    "shadow_eligibility",
]
