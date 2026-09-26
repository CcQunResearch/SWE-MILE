"""ShellScriptEvaluator: run a shell verifier inside a sandbox.

Used when ``dataset.toml`` / ``task.toml`` declares ``[verifier].script``
or when ``tests/test.sh`` is auto-detected. Implements the rLLM
:class:`~rllm.types.Evaluator` protocol.

Reward contract (Harbor-compatible): the script writes to one of
``/tmp/rllm/reward.json``, ``/logs/verifier/reward.json``, or
``/logs/verifier/reward.txt``. The first existing file wins.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import re
import shlex
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from rllm.eval.types import EvalOutput, Signal
from rllm.eval.verifier_artifacts import capture_script, resolve_bound_files
from rllm.sandbox.protocol import Sandbox
from rllm.types import Episode, RolloutInfrastructureError, Task


def _retryable_verifier_transport(exc: BaseException) -> bool:
    if isinstance(exc, RolloutInfrastructureError):
        return exc.retryable
    if isinstance(exc, (PermissionError, ValueError, FileNotFoundError)):
        return False
    message = str(exc).lower()
    if any(token in message for token in ("unauthorized", "forbidden", "authentication", "401", "403")):
        return False
    return isinstance(exc, (TimeoutError, ConnectionError)) or any(
        token in message for token in ("deadline exceeded", "readtimeout", "timed out", "response uncertain", "connection reset")
    )

logger = logging.getLogger(__name__)


# Reward file search order (first existing file wins)
_REWARD_PATHS = [
    "/tmp/rllm/reward.json",
    "/logs/verifier/reward.json",
    "/logs/verifier/reward.txt",
]

_VERIFIER_STATUS_PATH = "/tmp/rllm/verifier_status.json"
_VERIFIER_TIMEOUT_GRACE_SECONDS = 30
_VERIFIER_NETWORK_GUARD_DIR = "/tmp/rllm-verifier-network-guard"
_VERIFIER_NETWORK_TEST_OUTPUT_PATH = "/tmp/test_output.txt"
_VERIFIER_NETWORK_LOG_BYTES = 128 * 1024
_VERIFIER_NETWORK_PROBE_MARKER = "__RLLM_NETWORK_PROBE_RESULT__="
_VERIFIER_NETWORK_ERROR_RE = re.compile(
    r"(?:ConnectionError|ConnectTimeout|ReadTimeout|ProxyError|SSLError|"
    r"NameResolutionError|Temporary failure in name resolution|"
    r"Name or service not known|Network is unreachable|No route to host|"
    r"Connection (?:timed out|refused|reset)|Max retries exceeded|"
    r"RemoteDisconnected)",
    re.IGNORECASE,
)

_VERIFIER_NETWORK_GUARD = """\
\"\"\"Bound otherwise-unbounded public-network calls in selected verifiers.\"\"\"
import os
import socket

try:
    _rllm_timeout = float(os.environ.get("RLLM_NETWORK_DEFAULT_TIMEOUT_SECONDS", "20"))
except (TypeError, ValueError):
    _rllm_timeout = 20.0
if _rllm_timeout > 0:
    socket.setdefaulttimeout(_rllm_timeout)
    _rllm_original_settimeout = socket.socket.settimeout
    _rllm_original_create_connection = socket.create_connection

    def _rllm_bounded_settimeout(self, value):
        return _rllm_original_settimeout(
            self,
            _rllm_timeout if value is None else value,
        )

    def _rllm_bounded_create_connection(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, *args, **kwargs):
        if timeout is None or timeout is socket._GLOBAL_DEFAULT_TIMEOUT:
            timeout = _rllm_timeout
        return _rllm_original_create_connection(address, timeout, *args, **kwargs)

    socket.socket.settimeout = _rllm_bounded_settimeout
    socket.create_connection = _rllm_bounded_create_connection
"""

_VERIFIER_NETWORK_PROBE_SCRIPT = r"""
import base64
import concurrent.futures
import json
import os
import time
import urllib.error
import urllib.request

policy = json.loads(base64.b64decode(os.environ["RLLM_NETWORK_PROBE_B64"]))
urls = list(policy["probe_urls"])
timeout = float(policy["probe_timeout_seconds"])
attempts = int(policy["probe_attempts"])
delay = float(policy["probe_retry_delay_seconds"])
history = {url: [] for url in urls}

def probe(url, attempt):
    started = time.monotonic()
    status = None
    error = None
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "rllm-verifier-network-probe/1"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(response.getcode() or 0)
            response.read(1)
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
    except Exception as exc:
        error = "%s: %s" % (type(exc).__name__, exc)
    return {
        "attempt": attempt,
        "status": status,
        "ok": status is not None and 100 <= status < 400,
        "error": error,
        "duration_seconds": round(time.monotonic() - started, 3),
    }

remaining = set(urls)
for attempt in range(1, attempts + 1):
    if not remaining:
        break
    current = [url for url in urls if url in remaining]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(current))) as pool:
        future_by_url = {pool.submit(probe, url, attempt): url for url in current}
        for future in concurrent.futures.as_completed(future_by_url):
            url = future_by_url[future]
            result = future.result()
            history[url].append(result)
            if result["ok"]:
                remaining.discard(url)
    if remaining and attempt < attempts and delay:
        time.sleep(delay)

result = {
    "schema_version": 1,
    "ok": not remaining,
    "urls": [
        {
            "url": url,
            "ok": bool(history[url] and history[url][-1]["ok"]),
            "attempts": history[url],
        }
        for url in urls
    ],
}
print("__RLLM_NETWORK_PROBE_RESULT__=" + json.dumps(result, sort_keys=True))
"""


class ShellScriptEvaluator:
    """Run a verifier script inside the sandbox, parse the reward file.

    Constructed by :func:`rllm.eval._resolution._resolve_evaluator` once
    the sandbox is alive — the evaluator carries its sandbox reference
    internally instead of fishing it out of episode artifacts.
    """

    def __init__(
        self,
        sandbox: Sandbox,
        script_path: str = "tests/test.sh",
        verifier_user: str | None = None,
        verifier_timeout: float = 600.0,
        reward_file_override: str | None = None,
        timeout_is_failure: bool = False,
    ):
        self.sandbox = sandbox
        self.script_path = script_path  # relative to the task's directory
        self.verifier_user = verifier_user
        self.verifier_timeout = verifier_timeout
        self.reward_file_override = reward_file_override
        self.timeout_is_failure = timeout_is_failure

    def _command(
        self,
        script_name: str,
        workdir: str | None,
        network_policy: dict | None = None,
        operation_id: str | None = None,
    ) -> tuple[str, float]:
        """Build a verifier command with an observable timeout boundary.

        The sandbox transport gets a short grace period beyond the benchmark
        timeout.  That grace is used only to persist a zero reward and status
        record after the BusyBox/coreutils-compatible ``timeout`` invocation
        terminates the verifier; it does not give the tests additional time.
        """

        cd_prefix = f"cd {shlex.quote(workdir)} && " if workdir else ""
        network_prefix = ""
        if network_policy is not None:
            socket_timeout = network_policy["default_socket_timeout_seconds"]
            network_prefix = (
                f"export RLLM_NETWORK_DEFAULT_TIMEOUT_SECONDS={shlex.quote(str(socket_timeout))}; "
                f"export PYTHONPATH={shlex.quote(_VERIFIER_NETWORK_GUARD_DIR)}"
                "${PYTHONPATH:+:$PYTHONPATH}; "
            )
        runner = (
            f"chmod +x /tests/{shlex.quote(script_name)} && "
            f"{cd_prefix}{network_prefix}/tests/{shlex.quote(script_name)}"
        )
        if not self.timeout_is_failure:
            return runner, self.verifier_timeout

        timeout_seconds = max(1, int(math.ceil(self.verifier_timeout)))
        reward_path = shlex.quote(self.reward_file_override or _REWARD_PATHS[0])
        operation_id = operation_id or uuid.uuid4().hex
        capture_paths = list(dict.fromkeys([self.reward_file_override, *_REWARD_PATHS]))
        capture_paths = [path for path in capture_paths if path]
        cleanup_paths = " ".join(
            shlex.quote(path)
            for path in dict.fromkeys(
                [*(_REWARD_PATHS), self.reward_file_override, _VERIFIER_STATUS_PATH]
            )
            if path
        )
        proxy_exports = _proxy_exports(network_policy) if network_policy else ""
        command = f"""
{proxy_exports}mkdir -p /tmp/rllm /logs/verifier
rm -f {cleanup_paths}
_rllm_verifier_started=$(date +%s)
set +e
timeout -s TERM -k 10 {timeout_seconds} bash -c {shlex.quote(runner)}
_rllm_verifier_rc=$?
_rllm_verifier_finished=$(date +%s)
_rllm_verifier_elapsed=$((_rllm_verifier_finished-_rllm_verifier_started))
_rllm_verifier_timed_out=false
if [ "$_rllm_verifier_rc" -eq 124 ] || {{ [ "$_rllm_verifier_rc" -eq 137 ] && [ "$_rllm_verifier_elapsed" -ge {timeout_seconds} ]; }}; then
  _rllm_verifier_timed_out=true
  RLLM_REWARD_PATH={reward_path} python3 - "$_rllm_verifier_rc" "$_rllm_verifier_elapsed" "{timeout_seconds}" <<'PY'
import json, os, sys
path = os.environ["RLLM_REWARD_PATH"]
payload = {{
    "reward": 0.0,
    "is_correct": False,
    "metadata": {{
        "verifier_status": "timeout",
        "verifier_exit_code": int(sys.argv[1]),
        "verifier_duration_seconds": int(sys.argv[2]),
        "verifier_timeout_seconds": int(sys.argv[3]),
    }},
}}
temporary = path + ".tmp"
with open(temporary, "w") as output:
    json.dump(payload, output)
    output.flush()
    os.fsync(output.fileno())
os.replace(temporary, path)
PY
  if [ -f /tmp/test_output.txt ]; then
    tail -c 4000 /tmp/test_output.txt || true
  fi
fi
RLLM_STATUS_PATH={shlex.quote(_VERIFIER_STATUS_PATH)} python3 - "$_rllm_verifier_rc" "$_rllm_verifier_elapsed" "$_rllm_verifier_timed_out" <<'PY'
{capture_script(_VERIFIER_STATUS_PATH, operation_id, capture_paths)}
PY
exit 0
""".strip()
        return command, self.verifier_timeout + _VERIFIER_TIMEOUT_GRACE_SECONDS

    def evaluate(self, task: Task, episode: Episode) -> EvalOutput:
        policy_failure = episode.metadata.get("repository_policy_failure")
        if (
            isinstance(policy_failure, dict)
            and policy_failure.get("reason") == "agent_destroyed_git_repository"
            and policy_failure.get("repository_restored") is False
        ):
            # This is a confirmed terminal model action, not a broken trusted
            # verifier. There is no valid repository on which to apply tests.
            return EvalOutput(
                reward=0.0,
                is_correct=False,
                metadata={
                    "repository_policy_failure": dict(policy_failure),
                    "verifier_status": "skipped_repository_policy_failure",
                    "complete": False,
                },
            )
        tests_dir = task.task_dir / Path(self.script_path).parent
        script_name = Path(self.script_path).name

        def preparation_failure(reason: str, detail: str) -> EvalOutput:
            metadata = {"error": detail}
            if self.timeout_is_failure:
                metadata["infrastructure_failure"] = {
                    "reason": reason,
                    "stage": "verifier_setup",
                    "exception_type": reason,
                    "error_summary": detail[-2000:],
                    "retryable": False,
                }
            return EvalOutput(reward=0.0, is_correct=False, metadata=metadata)

        profile = (task.metadata.get("rllm") or {}).get("eval_verifier_profile")
        required = [script_name]
        if profile == "swebench_pro_public":
            required.extend(["instance.json", "run_script.sh", "parser.py"])
        elif profile == "doc2repo":
            required.extend(["test_suite.zip", "score_pytest.py"])
        for name in required:
            if not (tests_dir / name).is_file():
                return preparation_failure("verifier_asset_missing", f"verifier asset {tests_dir / name} not found")

        v_user = self.verifier_user

        # Prepare reward directories
        try:
            self.sandbox.exec("mkdir -p /tmp/rllm /logs/verifier", timeout=10, user=v_user)
        except RolloutInfrastructureError:
            raise
        except Exception:
            pass

        # Upload to /tests/ (Harbor convention — scripts may reference /tests/*.py)
        try:
            from rllm.sandbox.verifier_assets import upload_verifier_assets

            upload_verifier_assets(self.sandbox, task, tests_dir)
            script = None
            if profile == "swebench_pro_public":
                from rllm.data.swebench_pro_builder import _build_verifier_script

                script = _build_verifier_script()
            elif profile == "doc2repo":
                script = (Path(__file__).parents[1] / "data/assets/doc2repo_verifier.sh").read_text(encoding="utf-8")
            if script is not None:
                with tempfile.TemporaryDirectory(prefix="rllm-eval-verifier-") as tmp:
                    current = Path(tmp) / script_name
                    current.write_text(script, encoding="utf-8")
                    self.sandbox.upload_file(str(current), f"/tests/{script_name}")
        except Exception as exc:
            if isinstance(exc, RolloutInfrastructureError):
                return EvalOutput(reward=0.0, is_correct=False, metadata={
                    "infrastructure_failure": {"reason": exc.reason, "stage": exc.stage,
                        "retryable": exc.retryable, "retry_scope": exc.retry_scope,
                        "exception_type": type(exc).__name__, "diagnostics": exc.diagnostics,
                        "error_summary": str(exc)[-2000:]},
                })
            if not self.timeout_is_failure:
                raise
            output = preparation_failure("verifier_upload_failed", f"{type(exc).__name__}: {exc}")
            output.metadata["infrastructure_failure"].update(
                retryable=_retryable_verifier_transport(exc), retry_scope="full_rollout",
                exception_type=type(exc).__name__, diagnostics=getattr(exc, "diagnostics", None),
            )
            return output

        # Only ``cd`` when the task explicitly declared a workdir.
        # Otherwise the Dockerfile's WORKDIR wins — required for swesmith
        # and similar harbor task families whose verifier scripts run
        # ``git`` and ``pytest`` against ``/testbed`` (the image WORKDIR)
        # and silently collect zero tests when forced into ``/workspace``.
        network_policy = _verifier_network_policy(task)
        network_preflight: dict | None = None
        network_guard_error: str | None = None
        if network_policy is not None:
            network_preflight = _probe_verifier_network(
                self.sandbox,
                network_policy,
                user=v_user,
            )
            if not network_preflight.get("ok"):
                return _network_infrastructure_output(
                    task,
                    network_policy,
                    network_preflight,
                    stage="preflight",
                )
            network_guard_error = _install_verifier_network_guard(
                self.sandbox,
                network_policy,
                user=v_user,
            )

        workdir = task.metadata.get("workdir")
        operation_id = uuid.uuid4().hex
        command, transport_timeout = self._command(
            script_name,
            workdir,
            network_policy if network_policy is not None and network_guard_error is None else None,
            operation_id=operation_id,
        )
        execution_error: BaseException | None = None
        execution_output = ""
        try:
            background_exec = getattr(self.sandbox, "exec_background", None)
            if self.timeout_is_failure and callable(background_exec):
                execution_output = background_exec(
                    command,
                    timeout=transport_timeout,
                    user=v_user,
                )
            else:
                execution_output = self.sandbox.exec(
                    command,
                    timeout=transport_timeout,
                    user=v_user,
                )
        except RolloutInfrastructureError as exc:
            if not exc.retryable:
                raise
            execution_error = exc
        except Exception as e:
            execution_error = e
            # Verifier exit != 0 is the *expected* outcome when an agent
            # didn't solve the task; the reward (read from reward.txt
            # below) carries the signal. Log at debug so a benchmark of
            # 100 unsolved tasks doesn't spam 100 multi-KB stack traces.
            logger.debug("Verifier transport failed for %s: %s", task.id, e)

        # Read reward (as verifier — agent may not have read access)
        reward_paths = list(_REWARD_PATHS)
        if self.reward_file_override:
            reward_paths.insert(0, self.reward_file_override)
        artifact_reader = getattr(self.sandbox, "read_files", None)
        read_error = None
        artifacts = {}
        artifact_exception = None
        bound_result = False
        files = {}
        if callable(artifact_reader):
            try:
                artifacts = artifact_reader([_VERIFIER_STATUS_PATH, *reward_paths], timeout=getattr(self.sandbox, "control_read_timeout", 30), user=v_user)
                if not isinstance(artifacts, dict) or any(not isinstance(record, dict) for record in artifacts.values()):
                    raise ValueError("malformed artifact snapshot")
            except RolloutInfrastructureError:
                # Preserve node-loss/cleanup/authentication policy from the backend.
                raise
            except Exception as exc:
                artifact_exception = exc
                artifacts = {path: {"state": "transport_error", "error": type(exc).__name__,
                                    "error_summary": str(exc)[-2000:]}
                             for path in [_VERIFIER_STATUS_PATH, *reward_paths]}
            status_record = artifacts.get(_VERIFIER_STATUS_PATH, {})
            status = None
            if status_record.get("state") == "present":
                try:
                    status = json.loads(status_record["content"])
                    if not isinstance(status, dict):
                        raise ValueError("status must be an object")
                except (ValueError, KeyError, TypeError):
                    read_error = "verifier_result_invalid"
                    status = None
            elif status_record.get("state") not in {"missing"}:
                read_error = "verifier_result_read_failed"
            bound_result = _bound_verifier_status(status, operation_id)
            if self.timeout_is_failure and not bound_result:
                read_error = read_error or "verifier_result_unconfirmed"
            files = (resolve_bound_files(self.sandbox, status, _VERIFIER_STATUS_PATH, reward_paths, user=v_user)
                     if bound_result else artifacts)
            output = _reward_from_file_records(files, reward_paths)
            read_error = read_error or output.metadata.get("artifact_error")
            output.metadata["verifier_artifact_states"] = {path: record.get("state") for path, record in artifacts.items()}
        else:
            output = _read_reward_from_sandbox(self.sandbox, reward_paths, user=v_user)
            status = _read_json_file(self.sandbox, _VERIFIER_STATUS_PATH, user=v_user)
            bound_result = _bound_verifier_status(status, operation_id)
            if bound_result:
                files = resolve_bound_files(self.sandbox, status, _VERIFIER_STATUS_PATH, reward_paths, user=v_user)
                output = _reward_from_file_records(files, reward_paths)
                read_error = output.metadata.get("artifact_error")
        output.metadata["verifier_asset_provenance"] = getattr(self.sandbox, "_rllm_verifier_asset_provenance", {})
        output.metadata["verifier_operation_id"] = operation_id
        output.metadata["reward_paths_checked"] = reward_paths
        if status and status.get("operation_id") not in {None, operation_id}:
            read_error = "verifier_result_unconfirmed"
            status = None
        if status is not None:
            output.metadata["verifier_execution"] = status
            output.metadata.setdefault(
                "verifier_status",
                "timeout" if status.get("timed_out") else "completed",
            )
        if execution_output:
            output.metadata["verifier_log_tail"] = execution_output[-4000:]
        if status is not None and status.get("timed_out") and read_error is None:
            alive = _sandbox_alive(self.sandbox)
            output.metadata["sandbox_alive"] = alive
            if alive is True:
                # The shell wrapper normally writes this zero reward itself.
                # Synthesize it here as a second line of defense if the reward
                # file write raced with a filesystem flush, but only after the
                # status record and sandbox health independently confirm a
                # declared verifier timeout.
                output.reward = 0.0
                output.is_correct = False
                output.metadata.pop("error", None)
                return _finalize_network_outcome(
                    self.sandbox,
                    task,
                    output,
                    network_policy,
                    network_preflight,
                    network_guard_error,
                    user=v_user,
                )
            output.metadata["error"] = (
                "no reward file found: verifier timed out but sandbox health "
                "is unavailable"
            )
            read_error = "verifier_sandbox_lost" if alive is False else "verifier_result_unconfirmed"
        # With the timeout wrapper enabled the command always exits zero after
        # atomically writing its status.  Any other sandbox.exec exception is
        # therefore a transport failure, even if an old reward artifact was
        # visible because the command never reached its in-shell cleanup.
        strict_transport_failure = bool(
            self.timeout_is_failure and execution_error is not None and not bound_result
        )
        if (
            output.metadata.get("error") != "no reward file found"
            and not strict_transport_failure
            and read_error is None
        ):
            if execution_error is not None and bound_result:
                output.metadata["verifier_transport_reconciled"] = True
            return _finalize_network_outcome(
                self.sandbox,
                task,
                output,
                network_policy,
                network_preflight,
                network_guard_error,
                user=v_user,
            )

        alive = _sandbox_alive(self.sandbox)
        reason = (
            "verifier_sandbox_lost"
            if alive is False
            else (
                read_error or "verifier_execution_failed"
                if execution_error is not None
                else read_error or "verifier_reward_missing"
            )
        )
        error_summary = (
            _error_summary(execution_error)
            if execution_error is not None
            else read_error or "verifier completed without a reward artifact"
        )
        sandbox_health = (
            "alive" if alive is True else "lost" if alive is False else "unknown"
        )
        failure = {
            "schema_version": 2,
            "operation_id": operation_id,
            "reason": reason,
            "stage": "verifier",
            "task_id": task.id,
            "verifier_status": (
                "timeout"
                if isinstance(status, dict) and status.get("timed_out")
                else "execution_failed"
                if execution_error is not None
                else read_error or "reward_missing"
            ),
            "sandbox_health": sandbox_health,
            "sandbox_alive": alive,
            "duration_seconds": (
                status.get("duration_seconds") if isinstance(status, dict) else None
            ),
            "checked_paths": reward_paths,
            "exception_type": (
                type(execution_error).__name__
                if execution_error is not None
                else read_error or "reward_contract_missing"
            ),
            "error_summary": error_summary,
            "error_sha256": hashlib.sha256(
                error_summary.encode("utf-8", "replace")
            ).hexdigest(),
            # Only the evaluation engine consumes this full-rollout retry hint;
            # training retains its existing group requeue policy. Never rerun
            # the verifier in this sandbox after an uncertain response.
            "retryable": bool(self.timeout_is_failure and (
                (any(record.get("state") == "transport_error" and record.get("retryable", True) for record in [*artifacts.values(), *files.values()] if isinstance(record, dict))
                 and (artifact_exception is None or _retryable_verifier_transport(artifact_exception)))
                or alive is False
                or (execution_error is not None and not bound_result
                    and _retryable_verifier_transport(execution_error))
            )),
            "retry_scope": "full_rollout",
            "diagnostics": {
                "operation_id": operation_id,
                "artifact_reads": {path: {key: value for key, value in record.items() if key != "content"}
                                   for path, record in artifacts.items()},
                "bound_artifacts": {path: {key: value for key, value in record.items() if key != "content"}
                                    for path, record in files.items() if isinstance(record, dict)},
                "artifact_error_detail": output.metadata.get("artifact_error_detail"),
                "read_exception_type": type(artifact_exception).__name__ if artifact_exception else None,
                "read_error_summary": str(artifact_exception)[-2000:] if artifact_exception else None,
            },
        }
        if isinstance(execution_error, RolloutInfrastructureError):
            failure.update(reason=execution_error.reason, retryable=execution_error.retryable,
                           retry_scope=execution_error.retry_scope)
            failure["diagnostics"]["execution"] = execution_error.diagnostics
        if artifact_exception is not None and not _retryable_verifier_transport(artifact_exception):
            failure["retryable"] = False
        output.metadata["verifier_diagnostics"] = failure["diagnostics"]
        output.metadata.update(
            {
                "verifier_status": failure["verifier_status"],
                "sandbox_alive": alive,
                "infrastructure_failure": failure,
            }
        )
        if strict_transport_failure or read_error:
            output.reward = 0.0
            output.is_correct = False
            output.metadata["error"] = read_error or "verifier sandbox transport failed"
        if execution_error is not None:
            output.metadata["verifier_execution_error"] = error_summary
        return _finalize_network_outcome(
            self.sandbox,
            task,
            output,
            network_policy,
            network_preflight,
            network_guard_error,
            user=v_user,
        )


# ---------------------------------------------------------------------------
# Optional public-network verifier policy
# ---------------------------------------------------------------------------


def _verifier_network_policy(task: Task) -> dict | None:
    """Return a validated runtime-only network policy declared by the driver."""

    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    rllm_metadata = metadata.get("rllm")
    if not isinstance(rllm_metadata, dict):
        return None
    raw = rllm_metadata.get("verifier_network_policy")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{task.id}: verifier_network_policy must be a mapping")

    raw_urls = raw.get("probe_urls")
    if not isinstance(raw_urls, list | tuple) or not raw_urls:
        raise ValueError(f"{task.id}: verifier_network_policy.probe_urls must be non-empty")
    probe_urls: list[str] = []
    for value in raw_urls:
        url = str(value).strip()
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"{task.id}: invalid verifier network probe URL: {url!r}")
        if parsed.hostname.lower() in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError(f"{task.id}: verifier network probe must target public egress: {url!r}")
        probe_urls.append(url)

    def bounded_float(name: str, default: float, *, low: float, high: float) -> float:
        value = float(raw.get(name, default))
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(
                f"{task.id}: verifier_network_policy.{name} must be between {low} and {high}"
            )
        return value

    attempts = int(raw.get("probe_attempts", 3))
    if not 1 <= attempts <= 5:
        raise ValueError(f"{task.id}: verifier_network_policy.probe_attempts must be between 1 and 5")
    python_executable = str(
        raw.get("python_executable", "/opt/miniconda3/envs/testbed/bin/python")
    ).strip()
    if not python_executable.startswith("/"):
        raise ValueError(f"{task.id}: verifier network probe Python must be an absolute path")

    raw_proxy_env = raw.get("proxy_env", {})
    if not isinstance(raw_proxy_env, dict):
        raise ValueError(f"{task.id}: verifier_network_policy.proxy_env must be a mapping")
    proxy_env: dict[str, str] = {}
    for name in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        value = str(raw_proxy_env.get(name, "")).strip()
        if not value:
            continue
        if name != "no_proxy":
            parsed = urlsplit(value)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError(f"{task.id}: invalid verifier proxy URL for {name}: {value!r}")
        proxy_env[name] = value
    raw_verifier_env = raw.get("verifier_env", {})
    if not isinstance(raw_verifier_env, dict):
        raise ValueError(f"{task.id}: verifier_network_policy.verifier_env must be a mapping")
    verifier_env: dict[str, str] = {}
    for name, raw_value in raw_verifier_env.items():
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", str(name)):
            raise ValueError(f"{task.id}: invalid verifier environment variable: {name!r}")
        verifier_env[str(name)] = str(raw_value)

    return {
        "schema_version": int(raw.get("schema_version", 1)),
        "probe_urls": probe_urls,
        "probe_attempts": attempts,
        "probe_timeout_seconds": bounded_float(
            "probe_timeout_seconds",
            10.0,
            low=1.0,
            high=60.0,
        ),
        "probe_retry_delay_seconds": bounded_float(
            "probe_retry_delay_seconds",
            1.0,
            low=0.0,
            high=30.0,
        ),
        "default_socket_timeout_seconds": bounded_float(
            "default_socket_timeout_seconds",
            20.0,
            low=1.0,
            high=120.0,
        ),
        "python_executable": python_executable,
        "proxy_env": proxy_env,
        "verifier_env": verifier_env,
    }


def _proxy_exports(policy: dict) -> str:
    """Build a shell-safe verifier-only proxy environment prefix."""

    values = policy.get("proxy_env") or {}
    assignments: list[str] = []
    for name in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        value = values.get(name)
        if value:
            assignments.extend(
                [
                    f"{name}={shlex.quote(value)}",
                    f"{name.upper()}={shlex.quote(value)}",
                ]
            )
    assignments.extend(
        f"{name}={shlex.quote(value)}"
        for name, value in sorted((policy.get("verifier_env") or {}).items())
    )
    return ("export " + " ".join(assignments) + "\n") if assignments else ""


def _probe_verifier_network(
    sandbox: Sandbox,
    policy: dict,
    *,
    user: str | None = None,
) -> dict:
    """Probe every required public endpoint with bounded retries in-sandbox."""

    probe_payload = {
        key: policy[key]
        for key in (
            "probe_urls",
            "probe_attempts",
            "probe_timeout_seconds",
            "probe_retry_delay_seconds",
        )
    }
    encoded = base64.b64encode(
        json.dumps(probe_payload, sort_keys=True).encode("utf-8")
    ).decode("ascii")
    command = (
        f"{_proxy_exports(policy)}"
        f"RLLM_NETWORK_PROBE_B64={shlex.quote(encoded)} "
        f"{shlex.quote(policy['python_executable'])} - <<'PY'\n"
        f"{_VERIFIER_NETWORK_PROBE_SCRIPT}\n"
        "PY"
    )
    transport_timeout = (
        policy["probe_attempts"]
        * (policy["probe_timeout_seconds"] + policy["probe_retry_delay_seconds"])
        + 15.0
    )
    try:
        raw = sandbox.exec(command, timeout=transport_timeout, user=user)
    except Exception as exc:
        return {
            "schema_version": 1,
            "ok": False,
            "probe_error": f"{type(exc).__name__}: {exc}"[-2000:],
            "urls": [{"url": url, "ok": False, "attempts": []} for url in policy["probe_urls"]],
        }

    marker_index = str(raw).rfind(_VERIFIER_NETWORK_PROBE_MARKER)
    if marker_index < 0:
        return {
            "schema_version": 1,
            "ok": False,
            "probe_error": "network probe returned no structured result",
            "probe_output_tail": str(raw)[-2000:],
            "urls": [{"url": url, "ok": False, "attempts": []} for url in policy["probe_urls"]],
        }
    payload = str(raw)[marker_index + len(_VERIFIER_NETWORK_PROBE_MARKER) :].splitlines()[0]
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        return {
            "schema_version": 1,
            "ok": False,
            "probe_error": f"invalid network probe result: {exc}",
            "probe_output_tail": str(raw)[-2000:],
            "urls": [{"url": url, "ok": False, "attempts": []} for url in policy["probe_urls"]],
        }
    if not isinstance(parsed, dict):
        return {
            "schema_version": 1,
            "ok": False,
            "probe_error": "network probe result is not an object",
            "urls": [{"url": url, "ok": False, "attempts": []} for url in policy["probe_urls"]],
        }
    return parsed


def _install_verifier_network_guard(
    sandbox: Sandbox,
    policy: dict,
    *,
    user: str | None = None,
) -> str | None:
    """Install a task-local ``sitecustomize`` that bounds un-timed sockets."""

    encoded = base64.b64encode(_VERIFIER_NETWORK_GUARD.encode("utf-8")).decode("ascii")
    destination = f"{_VERIFIER_NETWORK_GUARD_DIR}/sitecustomize.py"
    command = (
        f"mkdir -p {shlex.quote(_VERIFIER_NETWORK_GUARD_DIR)} && "
        f"printf %s {shlex.quote(encoded)} | base64 -d > {shlex.quote(destination)}"
    )
    try:
        sandbox.exec(command, timeout=10, user=user)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"[-2000:]
    return None


def _network_infrastructure_output(
    task: Task,
    policy: dict,
    probe: dict,
    *,
    stage: str,
) -> EvalOutput:
    return EvalOutput(
        reward=0.0,
        is_correct=False,
        metadata={
            "verifier_status": "network_unavailable",
            "verifier_network": {
                "schema_version": policy["schema_version"],
                "stage": stage,
                "preflight" if stage == "preflight" else "postflight": probe,
            },
            "infrastructure_failure": {
                "reason": "verifier_network_unavailable",
                "stage": stage,
                "task_id": task.id,
            },
        },
    )


def _read_file_tail(
    sandbox: Sandbox,
    path: str,
    *,
    max_bytes: int,
    user: str | None = None,
) -> str:
    try:
        return str(
            sandbox.exec(
                f"tail -c {int(max_bytes)} {shlex.quote(path)} 2>/dev/null || true",
                timeout=10,
                user=user,
            )
        )
    except Exception:
        return ""


def _has_public_network_error(output: str, policy: dict) -> bool:
    normalized = output.lower()
    public_hosts = {
        str(urlsplit(url).hostname).lower()
        for url in policy["probe_urls"]
        if urlsplit(url).hostname
    }
    return bool(
        _VERIFIER_NETWORK_ERROR_RE.search(output)
        and any(host in normalized for host in public_hosts)
    )


def _finalize_network_outcome(
    sandbox: Sandbox,
    task: Task,
    output: EvalOutput,
    policy: dict | None,
    preflight: dict | None,
    guard_error: str | None,
    *,
    user: str | None = None,
) -> EvalOutput:
    if policy is None:
        return output

    network_metadata: dict = {
        "schema_version": policy["schema_version"],
        "preflight": preflight,
        "socket_timeout_seconds": policy["default_socket_timeout_seconds"],
        "socket_guard_active": guard_error is None,
    }
    if guard_error is not None:
        network_metadata["socket_guard_error"] = guard_error
    output.metadata["verifier_network"] = network_metadata
    if output.is_correct:
        return output

    execution = output.metadata.get("verifier_execution")
    timed_out = isinstance(execution, dict) and bool(execution.get("timed_out"))
    test_output = _read_file_tail(
        sandbox,
        _VERIFIER_NETWORK_TEST_OUTPUT_PATH,
        max_bytes=_VERIFIER_NETWORK_LOG_BYTES,
        user=user,
    )
    log_evidence = _has_public_network_error(test_output, policy)
    if not timed_out and not log_evidence:
        return output

    postflight = _probe_verifier_network(sandbox, policy, user=user)
    network_metadata["postflight"] = postflight
    network_metadata["public_network_error_in_verifier_log"] = log_evidence
    if postflight.get("ok"):
        return output

    output.metadata["infrastructure_failure"] = {
        "reason": "verifier_network_unavailable",
        "stage": "verifier",
        "task_id": task.id,
        "verifier_timed_out": timed_out,
        "public_network_error_in_verifier_log": log_evidence,
    }
    return output


# ---------------------------------------------------------------------------
# Reward parsing helpers (extracted from rllm/tasks/task.py)
# ---------------------------------------------------------------------------


def _read_reward_from_sandbox(sandbox: Sandbox, paths: list[str], user: str | None = None) -> EvalOutput:
    """Try reading reward from the sandbox at each path in order."""
    for path in paths:
        try:
            check = sandbox.exec(f"test -f {path} && echo yes || echo no", timeout=10, user=user).strip()
            if check != "yes":
                continue
            raw = sandbox.exec(f"cat {path}", timeout=10, user=user).strip()
            if not raw:
                continue
            if path.endswith(".txt"):
                reward = float(raw)
                return EvalOutput(reward=reward, is_correct=reward >= 1.0)
            return _parse_reward_json(raw)
        except Exception as e:
            logger.debug("Could not read reward from %s: %s", path, e)
            continue

    # This is provisional: evaluate() adds process/sandbox evidence and marks
    # the result as infrastructure failure.  A missing verifier contract must
    # never silently become a genuine model zero.
    return EvalOutput(
        reward=0.0,
        is_correct=False,
        metadata={"error": "no reward file found"},
    )


def _error_summary(exc: BaseException) -> str:
    message = f"{type(exc).__name__}: {exc}"
    return message if len(message) <= 2000 else message[:900] + " ... " + message[-1095:]


def _bound_verifier_status(status: dict | None, operation_id: str) -> bool:
    return bool(
        isinstance(status, dict)
        and status.get("operation_id") == operation_id
        and type(status.get("exit_code")) is int
        and type(status.get("timed_out")) is bool
        and type(status.get("duration_seconds")) in {int, float}
        and math.isfinite(status["duration_seconds"])
        and status["duration_seconds"] >= 0
        and isinstance(status.get("reward_files"), dict)
    )


def _reward_from_file_records(records: dict, paths: list[str]) -> EvalOutput:
    """Preserve the difference between absence and an unreadable artifact."""
    for path in paths:
        record = records.get(path, {})
        if not isinstance(record, dict):
            return EvalOutput(reward=0.0, is_correct=False, metadata={"error": "verifier_result_invalid", "artifact_error": "verifier_result_invalid"})
        state = record.get("state")
        if state == "missing":
            continue
        error = "verifier_result_read_failed"
        if state == "present":
            try:
                raw = record["content"]
                if path.endswith(".json"):
                    data = json.loads(raw)
                    if not isinstance(data, dict) or not ("reward" in data or data.get("rewards")):
                        raise ValueError("missing reward contract")
                output = (_parse_reward_json(raw) if path.endswith(".json") else
                          EvalOutput(reward=float(raw), is_correct=float(raw) >= 1.0))
                if not math.isfinite(output.reward):
                    raise ValueError("non-finite reward")
                return output
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                error = "verifier_result_invalid"
                record = {**record, "error": "invalid_reward_json_or_contract", "parse_error": str(exc)[:300]}
        elif state == "invalid":
            error = "verifier_result_invalid"
        return EvalOutput(reward=0.0, is_correct=False, metadata={"error": error, "artifact_error": error,
            "artifact_error_detail": {"path": path, **{k: v for k, v in record.items() if k != "content"}}})
    return EvalOutput(reward=0.0, is_correct=False, metadata={"error": "no reward file found"})


def _read_json_file(sandbox: Sandbox, path: str, user: str | None = None) -> dict | None:
    try:
        exists = sandbox.exec(
            f"test -f {shlex.quote(path)} && echo yes || echo no",
            timeout=10,
            user=user,
        ).strip()
        if exists != "yes":
            return None
        raw = sandbox.exec(f"cat {shlex.quote(path)}", timeout=10, user=user).strip()
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _sandbox_alive(sandbox: Sandbox) -> bool | None:
    probe = getattr(sandbox, "is_alive", None)
    if not callable(probe):
        return None
    try:
        value = probe()
        return None if value is None else bool(value)
    except Exception:
        return None


def _parse_reward_json(raw: str) -> EvalOutput:
    """Parse a JSON reward file into an EvalOutput.

    Supports both ``{"reward": 0.5}`` and Harbor-style ``{"rewards": {...}}``.
    """
    data = json.loads(raw)

    if "reward" in data:
        reward = float(data["reward"])
    elif "rewards" in data and data["rewards"]:
        reward = sum(float(v) for v in data["rewards"].values()) / len(data["rewards"])
    else:
        reward = 0.0

    is_correct = data.get("is_correct", reward >= 1.0)

    signals: list[Signal] = []
    for key, val in data.get("signals", {}).items():
        signals.append(Signal(name=key, value=float(val)))
    for key, val in data.get("rewards", {}).items():
        if key != "reward":
            signals.append(Signal(name=key, value=float(val)))

    return EvalOutput(
        reward=reward,
        is_correct=is_correct,
        signals=signals,
        metadata=data.get("metadata", {}),
    )
