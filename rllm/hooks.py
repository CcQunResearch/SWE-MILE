"""Public per-task hooks for :class:`rllm.engine.agentflow_engine.AgentFlowEngine`.

:class:`SandboxTaskHooks` is the canonical implementation, used by
``rllm eval`` and by :class:`rllm.trainer.unified_trainer.AgentTrainer`
for sandbox-style flows and env-carrying datasets.

The central question — *does this rollout need a sandbox?* — is answered in
exactly one place, :func:`resolve_rollout_plan`, as the join of three
declared needs:

* the **flow** declares it (``needs_env`` attr — ``True`` on
  :class:`~rllm.sandbox.sandboxed_flow.SandboxedAgentFlow`, ``False`` on
  ``@rllm.rollout`` flows),
* the **evaluation policy** needs it (:class:`FromTaskEvaluation` resolving a
  ``sandbox-shell`` / ``python-hybrid`` verifier; :class:`FixedEvaluation` never does),
* the **task** declares it (``environment/`` dir or ``task_path`` metadata).

Wiring-time call sites (trainer/CLI) use :func:`scan_env_requirements` over
the datasets to decide *whether to install these hooks at all* and whether
the gateway needs a public tunnel — the same predicate, evaluated before
any per-task bind.
"""

from __future__ import annotations

import base64
import errno
import json
import logging
import re
import shlex
import threading
import time
import uuid
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig, OmegaConf

from rllm.gateway.tunnel import is_local_sandbox_backend
from rllm.types import Evaluator, RolloutInfrastructureError, Task

if TYPE_CHECKING:
    from rllm.engine.agentflow_engine import TaskContext
    from rllm.sandbox.protocol import Sandbox
    from rllm.sandbox.warm_queue import WarmQueue
    from rllm.types import AgentFlow

logger = logging.getLogger(__name__)


_PROCESS_SNAPSHOT_SCRIPT = r"""
import json, os

ignored = {os.getpid(), os.getppid()}
items = {}
for name in os.listdir('/proc'):
    if not name.isdigit():
        continue
    pid = int(name)
    if pid in ignored:
        continue
    try:
        stat = open(f'/proc/{pid}/stat', encoding='utf-8').read()
        close = stat.rfind(')')
        tail = stat[close + 2:].split()
        starttime = tail[19]
        comm = stat[stat.find('(') + 1:close]
        raw = open(f'/proc/{pid}/cmdline', 'rb').read()
        command = raw.replace(b'\0', b' ').decode('utf-8', 'replace').strip()
    except (FileNotFoundError, PermissionError, IndexError, OSError):
        continue
    identity = f'{pid}:{starttime}'
    items[identity] = {
        'pid': pid,
        'starttime': starttime,
        'comm': comm[:128],
        'command': command[:512],
    }
print(json.dumps(items, sort_keys=True, separators=(',', ':')))
""".strip()


_PROCESS_QUIESCE_SCRIPT = r"""
import base64, json, os, signal, sys, time

baseline = set(json.loads(base64.b64decode(sys.argv[1]).decode('utf-8')))
self_pids = {os.getpid(), os.getppid()}

def snapshot():
    items = {}
    for name in os.listdir('/proc'):
        if not name.isdigit():
            continue
        pid = int(name)
        if pid in self_pids:
            continue
        try:
            stat = open(f'/proc/{pid}/stat', encoding='utf-8').read()
            close = stat.rfind(')')
            tail = stat[close + 2:].split()
            starttime = tail[19]
            comm = stat[stat.find('(') + 1:close]
            raw = open(f'/proc/{pid}/cmdline', 'rb').read()
            command = raw.replace(b'\0', b' ').decode('utf-8', 'replace').strip()
        except (FileNotFoundError, PermissionError, IndexError, OSError):
            continue
        identity = f'{pid}:{starttime}'
        items[identity] = {
            'pid': pid,
            'starttime': starttime,
            'comm': comm[:128],
            'command': command[:512],
        }
    return items

def signal_items(items, sig):
    for item in items.values():
        try:
            os.kill(int(item['pid']), sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass

current = snapshot()
detected = {key: value for key, value in current.items() if key not in baseline}
signal_items(detected, signal.SIGTERM)
if detected:
    time.sleep(0.5)
deadline = time.monotonic() + 5.0
while True:
    final_remaining = {key: value for key, value in snapshot().items() if key not in baseline}
    detected.update(final_remaining)
    if not final_remaining or time.monotonic() >= deadline:
        break
    signal_items(final_remaining, signal.SIGKILL)
    time.sleep(0.1)
print(json.dumps({
    'schema_version': 1,
    'status': 'clean' if not final_remaining else 'residual_processes',
    'detected_count': len(detected),
    'terminated_count': len(detected) - len(final_remaining),
    'remaining_count': len(final_remaining),
    'detected': list(detected.values())[:32],
    'remaining': list(final_remaining.values())[:32],
}, sort_keys=True, separators=(',', ':')))
""".strip()


def _sandbox_json_command(
    sandbox: Any,
    script: str,
    *arguments: str,
) -> dict[str, Any]:
    command = " ".join(
        [
            "python3",
            "-c",
            shlex.quote(script),
            *(shlex.quote(argument) for argument in arguments),
        ]
    )
    raw = sandbox.exec(command, timeout=30, user="root")
    lines = [line.strip() for line in str(raw).splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("primary process audit returned no JSON")
    parsed = json.loads(lines[-1])
    if not isinstance(parsed, dict):
        raise RuntimeError("primary process audit returned a non-object")
    return parsed


def _capture_primary_process_baseline(sandbox: Any) -> tuple[str, ...]:
    snapshot = _sandbox_json_command(sandbox, _PROCESS_SNAPSHOT_SCRIPT)
    return tuple(sorted(str(identity) for identity in snapshot))


def _quiesce_primary_processes(
    sandbox: Any,
    baseline: tuple[str, ...],
) -> dict[str, Any]:
    encoded = base64.b64encode(
        json.dumps(list(baseline), separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    result = _sandbox_json_command(
        sandbox,
        _PROCESS_QUIESCE_SCRIPT,
        encoded,
    )
    if result.get("status") != "clean" or result.get("remaining_count") != 0:
        raise RuntimeError(
            "primary sandbox retained processes after agent completion: "
            + json.dumps(result, ensure_ascii=True, sort_keys=True)[:4000]
        )
    return result


def _sandbox_setup_retry_kind(error: BaseException) -> str | None:
    """Classify setup failures safe for a fresh-sandbox retry."""

    messages: list[str] = []
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(f"{type(current).__name__}: {current}")
        if isinstance(current, OSError) and current.errno == errno.ENOSPC:
            return "setup_storage"
        current = current.__cause__ or current.__context__
    message = "\n".join(messages)
    if re.search(
        r"(?:no space left on device|\bENOSPC\b)",
        message,
        flags=re.IGNORECASE,
    ):
        return "setup_storage"
    if re.search(
        r"(?:failed to upload|upload (?:file|dir|failed|failure)|"
        r"transport (?:error|failure)|connection (?:reset|refused|aborted)|"
        r"remote disconnected|read timed out|exit marker missing|broken pipe|"
        r"exec response uncertain|exec admission deadline exceeded|exec HTTP failure)",
        message,
        flags=re.IGNORECASE,
    ):
        return "setup_transport"
    provider_setup = bool(
        re.search(r"sandbox", message, flags=re.IGNORECASE)
        and re.search(r"(?:create|provision|ready)", message, flags=re.IGNORECASE)
    )
    if provider_setup and re.search(
        r"(?:timed?\s*out|deadline exceeded|HTTP\s*429|status(?:_code)?[=: ]+429|"
        r"HTTP\s*5\d\d|status(?:_code)?[=: ]+5\d\d|too many requests|"
        r"temporarily unavailable|service unavailable)",
        message,
        flags=re.IGNORECASE,
    ):
        return "setup_provider_transient"
    return None


def _preflight_agent_shell(task: Task, sandbox: Sandbox) -> None:
    """Validate an evaluation-only agent shell before the model can use it."""
    rllm_metadata = task.metadata.get("rllm") or {}
    if rllm_metadata.get("agent_environment_policy") != "preconfigured_eval_v1":
        return
    init_command = str(rllm_metadata.get("agent_shell_init") or "").strip()
    if not init_command:
        from rllm.types import RolloutInfrastructureError

        raise RolloutInfrastructureError(
            "agent_environment_unavailable",
            "preconfigured evaluation environment has no agent_shell_init",
            retryable=False,
        )

    probe = (
        "import json, os, sys; "
        "print(json.dumps({'conda_prefix': os.environ.get('CONDA_PREFIX'), "
        "'conda_default_env': os.environ.get('CONDA_DEFAULT_ENV'), "
        "'python_executable': sys.executable}))"
    )
    workdir = str(task.metadata.get("workdir") or "/testbed")
    inner_command = (
        f"cd {shlex.quote(workdir)} && {init_command} && "
        f"python -c {shlex.quote(probe)}"
    )
    command = f"/bin/bash -o pipefail -c {shlex.quote(inner_command)}"
    try:
        output = sandbox.exec(
            command,
            timeout=30,
            user=task.metadata.get("agent_user"),
        )
        payload = json.loads(str(output).strip().splitlines()[-1])
        prefix = str(payload.get("conda_prefix") or "")
        executable = str(payload.get("python_executable") or "")
        environment = str(payload.get("conda_default_env") or "")
        if (
            environment != "testbed"
            or not prefix
            or not executable.startswith(prefix.rstrip("/") + "/")
        ):
            raise ValueError(
                "expected conda environment 'testbed' with Python under CONDA_PREFIX; "
                f"got env={environment!r} prefix={prefix!r} python={executable!r}"
            )
    except Exception as exc:
        from rllm.types import RolloutInfrastructureError

        if isinstance(exc, RolloutInfrastructureError):
            raise
        retry_kind = _sandbox_setup_retry_kind(exc)
        raise RolloutInfrastructureError(
            "agent_environment_unavailable",
            f"failed to activate the preconfigured testbed environment: {exc}",
            retryable=retry_kind is not None,
            stage="setup",
            retry_scope="full" if retry_kind is not None else "none",
        ) from exc


def _preflight_sandbox_storage(
    task: Task,
    sandbox: Sandbox,
    *,
    sandbox_backend: str | None = None,
) -> None:
    """Fail closed before rollout when the repository filesystem is nearly full."""

    from rllm.types import RolloutInfrastructureError

    environment = task.metadata.get("environment")
    environment = environment if isinstance(environment, Mapping) else {}
    raw_quota_mb = environment.get("storage_mb")
    # Storage probing is an opt-in contract.  Generic/local sandbox fakes and
    # legacy tasks do not necessarily expose ``df``; DeNovoSWE materialization
    # always declares the quota that needs this guard.
    if raw_quota_mb is None:
        return
    try:
        quota_mb = int(raw_quota_mb)
    except (TypeError, ValueError):
        quota_mb = 0
    required_kib = max(2 * 1024 * 1024, int(max(0, quota_mb) * 1024 * 0.10))
    workdir = str(task.metadata.get("workdir") or "/testbed")
    command = (
        f"set -- $(df -Pk {shlex.quote(workdir)} | tail -n 1); "
        "printf '%s %s' \"$2\" \"$4\""
    )
    try:
        execute = getattr(sandbox, "exec_readonly", sandbox.exec)
        raw = str(execute(command, timeout=getattr(sandbox, "control_read_timeout", 30), user="root")).strip()
        total_kib, available_kib = (int(value) for value in raw.split()[-2:])
    except RolloutInfrastructureError:
        raise
    except Exception as exc:
        retry_kind = _sandbox_setup_retry_kind(exc)
        raise RolloutInfrastructureError(
            "setup_storage_probe_failed",
            f"sandbox storage preflight failed for {workdir}: {exc}",
            retryable=retry_kind is not None,
            stage="setup",
            retry_scope="full" if retry_kind is not None else "none",
        ) from exc
    if total_kib <= 0 or available_kib < required_kib:
        # A remote task can land on a node whose writable layer
        # has substantially less free space than the declared task quota.
        # Other tasks using the same image routinely succeed, so reject this
        # instance before rollout and let the outer retry obtain a fresh one.
        retryable = sandbox_backend == "k8s"
        raise RolloutInfrastructureError(
            "setup_storage_unavailable",
            "sandbox repository filesystem has insufficient free space: "
            f"total_kib={total_kib} available_kib={available_kib} "
            f"required_kib={required_kib} "
            f"declared_storage_mb={quota_mb or 'unknown'}",
            retryable=retryable,
            stage="setup",
            retry_scope="full" if retryable else "none",
            diagnostics={
                "workdir": workdir,
                "total_kib": total_kib,
                "available_kib": available_kib,
                "required_kib": required_kib,
                "declared_storage_mb": quota_mb,
                "storage_quota_forwarded": sandbox_backend in {"minisandbox", "k8s"},
                "job_instance_id": getattr(sandbox, "_job_instance_id", None),
            },
        )

# ---------------------------------------------------------------------------
# Evaluation policies
# ---------------------------------------------------------------------------


class FixedEvaluation:
    """Evaluation policy: one host-side evaluator scores every task.

    Host-side by definition — it never contributes an env requirement.
    """

    def __init__(self, evaluator: Evaluator):
        from rllm.eval._resolution import _adapt_legacy_evaluator

        self.evaluator = _adapt_legacy_evaluator(evaluator)

    def detect(self, task: Task) -> tuple[str, dict]:  # noqa: ARG002
        return "fixed", {}

    def resolve(self, task: Task, sandbox: Sandbox | None, kind: str, config: dict) -> Evaluator:  # noqa: ARG002
        return self.evaluator


class FromTaskEvaluation:
    """Evaluation policy: resolve a per-task verifier from the task's ``[verifier]`` config."""

    def detect(self, task: Task) -> tuple[str, dict]:
        from rllm.eval._resolution import _detect_verifier

        return _detect_verifier(task)

    def resolve(self, task: Task, sandbox: Sandbox | None, kind: str, config: dict) -> Evaluator:
        from rllm.eval._resolution import _resolve_evaluator

        return _resolve_evaluator(task, sandbox, kind, config)


EvaluationPolicy = FixedEvaluation | FromTaskEvaluation

# Verifier kinds that must run against a live sandbox.
_ENV_VERIFIER_KINDS = frozenset(
    {"sandbox-shell", "python-hybrid", "nl2repo-fresh-sandbox"}
)


# ---------------------------------------------------------------------------
# The env-requirement join (one place, pure, per task)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RolloutPlan:
    """Pure bind-time resolution for one task: env requirement + verifier choice."""

    needs_env: bool
    verifier_kind: str
    verifier_config: dict = field(default_factory=dict)


def flow_needs_env(agent_flow: Any) -> bool:
    """The flow's declared env requirement (``False`` when undeclared)."""
    return bool(getattr(agent_flow, "needs_env", False))


def task_declares_env(task_or_row: Any) -> bool:
    """True when a task (or raw dataset row) declares a sandbox environment.

    Carriers: ``task_path`` metadata (harbor-sourced rows) or an
    ``environment/`` dir next to the task/dataset (local benchmarks).
    """
    meta = getattr(task_or_row, "metadata", None)
    if meta is None and isinstance(task_or_row, dict):
        meta = task_or_row
    if isinstance(meta, dict) and meta.get("task_path"):
        return True
    if isinstance(task_or_row, Task):
        return (task_or_row.task_dir / "environment").is_dir() or (task_or_row.dataset_dir / "environment").is_dir()
    return False


# Warn once per task id, not once per rollout (the same task repeats across
# group rollouts and epochs).
_warned_no_consumer: set[str] = set()


def resolve_rollout_plan(task: Task, agent_flow: Any, evaluation: EvaluationPolicy) -> RolloutPlan:
    """Decide once whether this rollout needs a sandbox, and which verifier scores it.

    ``needs_env = flow ∨ evaluator ∨ task`` — with the no-consumer rule: a
    task-declared env that neither the flow nor the evaluator can reach is
    downgraded (with a warning) instead of provisioned for nobody.
    """
    kind, config = evaluation.detect(task)
    flow_env = flow_needs_env(agent_flow)
    eval_env = kind in _ENV_VERIFIER_KINDS
    task_env = task_declares_env(task)

    needs_env = flow_env or eval_env or task_env
    if needs_env and not flow_env and not eval_env:
        if task.id not in _warned_no_consumer:
            _warned_no_consumer.add(task.id)
            logger.warning(
                "task '%s' declares a sandbox environment, but neither the agent flow (%s) nor the verifier (kind=%s) can use one — skipping sandbox provisioning",
                task.id,
                type(agent_flow).__name__,
                kind,
            )
        needs_env = False

    return RolloutPlan(needs_env=needs_env, verifier_kind=kind, verifier_config=config)


# ---------------------------------------------------------------------------
# Wiring-time scan (trainer/CLI: install hooks? tunnel?)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvRequirements:
    """Run-level env requirements: whether any rollout may need a sandbox, and remoteness."""

    needs_env: bool
    any_remote: bool


def scan_env_requirements(agent_flow: Any, *datasets: Any, sandbox_backend: str | None = None) -> EnvRequirements:
    """Scan flow + all dataset rows for env requirements before wiring.

    ``any_remote`` drives the run-global gateway tunnel decision: when
    ``sandbox_backend`` is explicitly set it wins (it overrides per-task
    metadata at provision time); otherwise any row's ``sandbox_backend``
    metadata counts.
    """
    needs_env = flow_needs_env(agent_flow)
    any_remote = sandbox_backend is not None and not is_local_sandbox_backend(sandbox_backend)

    for ds in datasets:
        if ds is None:
            continue
        for row in ds:
            if not needs_env and task_declares_env(row):
                needs_env = True
            if sandbox_backend is None and not any_remote:
                meta = getattr(row, "metadata", None) or (row if isinstance(row, dict) else {})
                row_backend = meta.get("sandbox_backend") if isinstance(meta, dict) else None
                if row_backend and not is_local_sandbox_backend(row_backend):
                    any_remote = True
            if needs_env and (any_remote or sandbox_backend is not None):
                return EnvRequirements(needs_env=needs_env, any_remote=any_remote)

    return EnvRequirements(needs_env=needs_env, any_remote=any_remote)


# ---------------------------------------------------------------------------
# SandboxTaskHooks
# ---------------------------------------------------------------------------


class SandboxTaskHooks:
    """Per-task setup/teardown hook with sandbox lifecycle + verifier resolution.

    Args:
        evaluation: :class:`FixedEvaluation` (one host evaluator for every task) or
            :class:`FromTaskEvaluation` (per-task ``[verifier]`` resolution, the default).
        sandbox_backend: Override for the sandbox backend
            (``"docker"``, ``"local"``, ``"modal"``, ...). Falls back to
            per-task ``metadata['sandbox_backend']`` then ``"docker"``.
        shadow_sandbox_resources: Optional per-run ``cpus`` / ``memory_mb``
            defaults applied only while provisioning a shadow sandbox. A task's
            ``shadow_sandbox_resources`` metadata may selectively override them.
    """

    def __init__(
        self,
        evaluation: EvaluationPolicy | None = None,
        sandbox_backend: str | None = None,
        shadow_sandbox_resources: Mapping[str, int] | None = None,
    ) -> None:
    
        self.evaluation: EvaluationPolicy = evaluation if evaluation is not None else FromTaskEvaluation()
        self.sandbox_backend = sandbox_backend
        self.shadow_sandbox_resources = self._validate_shadow_resources(
            shadow_sandbox_resources
        )
        # Optional per-run warm queue (set by run_dataset / the trainer); when
        # present, setup pops a prefetched sandbox instead of creating one inline.
        self.warm_queue: WarmQueue | None = None
        self._minisandbox_cluster = None
    def initialize_backend(self, backend: Any, config: DictConfig) -> None:
        """Start job-scoped services after VERL has placed its GPU actors."""
        if self.sandbox_backend != "minisandbox":
            return
        if self._minisandbox_cluster is not None:
            return
        from rllm.sandbox.backends.minisandbox import (
            MiniSandboxCluster,
            set_active_minisandbox_cluster,
        )

        discover = getattr(backend, "get_execution_node_ids", None)
        if not callable(discover):
            raise RuntimeError(
                "swe.sandbox_backend=minisandbox requires a backend that can "
                "report the current job's GPU worker nodes"
            )
        node_ids = list(discover())
        shared_cache_dir = OmegaConf.select(
            config, "swe.minisandbox.shared_cache_dir", default=None
        )
        local_root = OmegaConf.select(
            config,
            "swe.minisandbox.local_root",
            default="/tmp/swe-minisandbox",
        )
        if not shared_cache_dir:
            raise ValueError("swe.minisandbox.shared_cache_dir must be configured")
        experiment = str(
            OmegaConf.select(
                config, "rllm.trainer.experiment_name", default="rllm"
            )
        )
        safe_experiment = re.sub(r"[^A-Za-z0-9_.-]", "-", experiment)[-32:]
        run_id = f"{safe_experiment}-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        canary_task_id = OmegaConf.select(
            config,
            "swe.minisandbox.canary_task_id",
            default=None,
        )
        try:
            cluster = MiniSandboxCluster.start(
                node_ids=node_ids,
                shared_cache_dir=str(shared_cache_dir),
                local_root=str(local_root),
                run_id=run_id,
                canary_task_id=(
                    str(canary_task_id)
                    if canary_task_id not in (None, "")
                    else None
                ),
            )
        except RolloutInfrastructureError:
            raise
        except Exception as exc:
            # Cache drift, local OCI conversion, umoci, namespace, cgroup,
            # network, and real-image canary failures are all job
            # infrastructure faults. They must fail startup before a rollout
            # can enter reward, replay, or dynamic-sampling accounting.
            raise RolloutInfrastructureError(
                "minisandbox_preflight_failed",
                f"MiniSandbox job preflight failed: {exc}",
                retryable=False,
                stage="preflight",
                retry_scope="full_rollout",
            ) from exc
        self._minisandbox_cluster = cluster
        set_active_minisandbox_cluster(cluster)
        logger.info(
            "started MiniSandbox services on %d VERL GPU node(s)", len(node_ids)
        )

    def shutdown_backend(self) -> None:
        cluster = self._minisandbox_cluster
        if cluster is None:
            return
        from rllm.sandbox.backends.minisandbox import (
            set_active_minisandbox_cluster,
        )

        self._minisandbox_cluster = None
        try:
            cluster.shutdown()
        finally:
            set_active_minisandbox_cluster(None)

    def _backend_kwargs(self, task: Task, uid: str) -> dict[str, Any]:
        from rllm.eval._resolution import _resolve_backend

        if _resolve_backend(task, self.sandbox_backend) != "minisandbox":
            return {}
        return {"route_key": uid}

    def _get_sandbox(self, task: Task, install: str, uid: str) -> Any:
        from rllm.sandbox.snapshot import get_sandbox
        from rllm.types import RolloutInfrastructureError

        backend_kwargs = self._backend_kwargs(task, uid)
        if backend_kwargs:
            try:
                return get_sandbox(
                    task,
                    self.sandbox_backend,
                    backend_kwargs=backend_kwargs,
                )
            except RolloutInfrastructureError:
                raise
            except Exception as exc:
                # MiniSandbox creation is entirely infrastructure-side (node
                # routing, OCI caching/unpack, cgroup/network namespaces). It
                # occurs before the agent runs, so it must never become a
                # model-failure episode or exhaust the task pool as a normal
                # dynamic-sampling rejection.
                raise RolloutInfrastructureError(
                    "minisandbox_unavailable",
                    f"MiniSandbox provisioning failed: {exc}",
                    retryable=True,
                    stage="setup",
                    retry_scope="full_rollout",
                ) from exc
        return get_sandbox(
            task,
            self.sandbox_backend,
        )

    def _shadow_task(self, task: Task) -> Task:
        """Return a provisioning-only task with isolated shadow resources."""
        task_resources = self._validate_shadow_resources(
            task.metadata.get("shadow_sandbox_resources")
        )
        resources = {**self.shadow_sandbox_resources, **task_resources}
        if not resources:
            return task
        metadata = deepcopy(task.metadata)
        environment = dict(metadata.get("environment") or {})
        environment.update(resources)
        metadata["environment"] = environment
        return replace(task, metadata=metadata)

    @staticmethod
    def _validate_shadow_resources(
        resources: Mapping[str, int] | None,
    ) -> dict[str, int]:
        if resources is None:
            return {}
        if not isinstance(resources, Mapping):
            raise ValueError("shadow_sandbox_resources must be a mapping")
        unknown = set(resources) - {"cpus", "memory_mb"}
        if unknown:
            raise ValueError(
                "shadow_sandbox_resources contains unsupported keys: "
                + ", ".join(sorted(unknown))
            )
        validated: dict[str, int] = {}
        for key, raw_value in resources.items():
            if isinstance(raw_value, bool) or not isinstance(raw_value, int):
                raise ValueError(
                    f"shadow_sandbox_resources.{key} must be a positive integer"
                )
            value = int(raw_value)
            if value <= 0:
                raise ValueError(
                    f"shadow_sandbox_resources.{key} must be a positive integer"
                )
            validated[key] = value
        return validated

    def setup(self, task: Task, agent_flow: AgentFlow, uid: str) -> TaskContext:
        from rllm.engine.agentflow_engine import TaskContext
        from rllm.eval._resolution import _resolve_backend, _setup_task_environment

        plan = resolve_rollout_plan(task, agent_flow, self.evaluation)
        rllm_metadata = task.metadata.get("rllm")
        rllm_metadata = (
            rllm_metadata if isinstance(rllm_metadata, Mapping) else {}
        )
        denovo_background_finalize = bool(
            rllm_metadata.get("denovo_background_finalize_active", False)
        )
        authoritative_shadow = bool(
            task.metadata.get("outcome_source") == "shadow_verifier"
            and not denovo_background_finalize
        )

        should_use_shadow = getattr(agent_flow, "should_use_shadow_sandbox", None)
        use_shadow = bool(plan.needs_env and callable(should_use_shadow)
                          and should_use_shadow(task, plan.verifier_kind))
        shadow_required = authoritative_shadow or (
            denovo_background_finalize and getattr(agent_flow, "shadow_enabled", True)
        )
        if shadow_required and not use_shadow:
            raise RolloutInfrastructureError(
                "shadow_lifecycle_contract_invalid",
                "task requiring shadow verification did not provision a shadow sandbox",
                retryable=False, stage="setup_contract", retry_scope="none",
                diagnostics={"task_id": task.id, "verifier_kind": plan.verifier_kind},
            )

        sandbox = None
        shadow_sandbox = None
        shadow_sandboxes: list[Any] = []
        shadow_runtime = None
        primary_process_baseline: tuple[str, ...] | None = None
        replacement_shadow_sandboxes: list[Any] = []
        replacement_shadow_lock = threading.Lock()
        env_backend = None
        try:
            if plan.needs_env:
                from rllm.env import env_int
                from rllm.sandbox.snapshot import install_script_for

                install = install_script_for(agent_flow)
                sandbox = (
                    self.warm_queue.pop(task)
                    if self.warm_queue is not None
                    else self._get_sandbox(task, install, uid)
                )
                env_backend = _resolve_backend(task, self.sandbox_backend)
                _setup_task_environment(task, sandbox)
                # CLI install, unless the image already contains exactly this
                # script (baked_install, recorded at snapshot boot).
                if install and getattr(sandbox, "baked_install", "") != install:
                    try:
                        sandbox.exec(install, timeout=getattr(agent_flow, "install_timeout", env_int("RLLM_HARNESS_INSTALL_TIMEOUT_S", 600)), user="root")
                    except RolloutInfrastructureError:
                        raise
                    except Exception as e:
                        raise RuntimeError(f"Failed to install {getattr(agent_flow, 'name', type(agent_flow).__name__)} in sandbox: {e}") from e
                _preflight_agent_shell(task, sandbox)
                _preflight_sandbox_storage(
                    task,
                    sandbox,
                    sandbox_backend=env_backend,
                )
                if denovo_background_finalize:
                    primary_process_baseline = _capture_primary_process_baseline(
                        sandbox
                    )

            evaluator = self.evaluation.resolve(task, sandbox, plan.verifier_kind, plan.verifier_config)
            configure_fresh_sandbox = getattr(
                evaluator, "configure_fresh_sandbox", None
            )
            if callable(configure_fresh_sandbox):
                configure_fresh_sandbox(
                    lambda: self._get_sandbox(task, "", uid)
                )

            if use_shadow:
                shadow_setup_started = time.perf_counter()
                shadow_error: Exception | None = None
                shadow_failure_kind: str | None = None
                requested_shadow_count = 1
                resolve_shadow_count = getattr(
                    agent_flow,
                    "shadow_sandbox_count",
                    None,
                )
                if callable(resolve_shadow_count):
                    raw_shadow_count = resolve_shadow_count(task)
                    if (
                        isinstance(raw_shadow_count, bool)
                        or not isinstance(raw_shadow_count, int)
                        or raw_shadow_count <= 0
                    ):
                        raise ValueError(
                            "shadow_sandbox_count(task) must return a positive integer"
                        )
                    requested_shadow_count = raw_shadow_count

                def provision_shadow_sandbox(*, replacement: bool) -> Any:
                    shadow_task = self._shadow_task(task)
                    candidate = None
                    try:
                        candidate = self._get_sandbox(
                            shadow_task, install, uid
                        )
                        _setup_task_environment(shadow_task, candidate)
                        if install and getattr(candidate, "baked_install", "") != install:
                            candidate.exec(
                                install,
                                timeout=getattr(
                                    agent_flow,
                                    "install_timeout",
                                    env_int("RLLM_HARNESS_INSTALL_TIMEOUT_S", 600),
                                ),
                                user="root",
                            )
                        _preflight_agent_shell(shadow_task, candidate)
                        _preflight_sandbox_storage(
                            shadow_task,
                            candidate,
                            sandbox_backend=_resolve_backend(
                                shadow_task,
                                self.sandbox_backend,
                            ),
                        )
                    except BaseException:
                        if candidate is not None:
                            try:
                                candidate.close()
                            except Exception:
                                logger.exception(
                                    "shadow sandbox close failed after provisioning error"
                                )
                        raise
                    if replacement:
                        with replacement_shadow_lock:
                            replacement_shadow_sandboxes.append(candidate)
                    return candidate

                for shadow_attempt in range(2):
                    try:
                        shadow_sandboxes = []
                        for _lane_index in range(requested_shadow_count):
                            shadow_sandboxes.append(
                                provision_shadow_sandbox(replacement=False)
                            )
                        shadow_sandbox = shadow_sandboxes[0]
                        if requested_shadow_count == 1:
                            shadow_runtime = agent_flow.create_shadow_runtime(
                                task,
                                sandbox,
                                shadow_sandbox,
                                uid,
                            )
                        else:
                            create_pool = getattr(
                                agent_flow,
                                "create_shadow_runtime_pool",
                                None,
                            )
                            if not callable(create_pool):
                                raise RuntimeError(
                                    "agent flow does not support a multi-shadow runtime"
                                )
                            shadow_runtime = create_pool(
                                task,
                                sandbox,
                                tuple(shadow_sandboxes),
                                uid,
                            )
                        record_setup_duration = getattr(
                            shadow_runtime,
                            "record_setup_duration",
                            None,
                        )
                        if callable(record_setup_duration):
                            record_setup_duration(
                                time.perf_counter() - shadow_setup_started
                            )
                        configure_final_recovery = getattr(
                            shadow_runtime,
                            "configure_final_recovery",
                            None,
                        )
                        if callable(configure_final_recovery) and not denovo_background_finalize:
                            configure_final_recovery(
                                lambda: provision_shadow_sandbox(
                                    replacement=True
                                )
                            )
                        shadow_error = None
                        break
                    except Exception as exc:
                        shadow_error = exc
                        retry_kind = _sandbox_setup_retry_kind(exc)
                        shadow_failure_kind = retry_kind
                        logger.warning(
                            "[%s] shadow sandbox setup failed%s: %s",
                            uid,
                            (
                                f" ({retry_kind}; rebuilding once)"
                                if shadow_attempt == 0 and retry_kind is not None
                                else ""
                            ),
                            exc,
                        )
                        from rllm.utils.shutdown import shutdown_components
                        cleanup_steps = []
                        if shadow_runtime is not None:
                            cleanup_steps.append(("shadow_abort", shadow_runtime.abort))
                        cleanup_steps.extend((f"shadow_{i}", candidate.close) for i, candidate in enumerate(shadow_sandboxes))
                        if shadow_runtime is not None:
                            cleanup_steps.append(("shadow_join", shadow_runtime.join_after_close))
                        try:
                            shutdown_components(cleanup_steps)
                        except Exception as cleanup_error:
                            from rllm.utils.infrastructure_diagnostics import infrastructure_exception_chain
                            failure = RolloutInfrastructureError(
                                "shadow_cleanup_unconfirmed", str(exc), retryable=False,
                                stage="setup_cleanup", retry_scope="none",
                                diagnostics={"task_id": task.id, "cleanup_errors": infrastructure_exception_chain(cleanup_error)},
                            )
                            failure.cleanup_errors = infrastructure_exception_chain(cleanup_error)
                            raise failure from exc
                        shadow_runtime = None
                        shadow_sandbox = None
                        shadow_sandboxes = []
                        if isinstance(exc, RolloutInfrastructureError):
                            raise
                        if shadow_attempt == 0 and retry_kind is not None:
                            continue
                        break

                if shadow_error is not None:
                    exc = shadow_error
                    if authoritative_shadow:
                        raise RuntimeError(
                            "authoritative shadow sandbox setup failed"
                        ) from exc
                    create_unavailable = getattr(agent_flow, "create_unavailable_shadow_runtime", None)
                    if callable(create_unavailable):
                        shadow_runtime = create_unavailable(
                            uid,
                            (
                                f"[{shadow_failure_kind}] {exc}"
                                if shadow_failure_kind is not None
                                else str(exc)
                            ),
                        )
                        record_setup_duration = getattr(shadow_runtime, "record_setup_duration", None)
                        if callable(record_setup_duration):
                            record_setup_duration(time.perf_counter() - shadow_setup_started)
        except BaseException as setup_error:
            # Nothing has registered a teardown yet — close the sandbox here
            # or it leaks (and the retry path provisions another).
            cleanup_errors = []
            if sandbox is not None:
                try:
                    sandbox.close()
                except Exception as exc:
                    cleanup_errors.append(exc)
                    logger.exception("sandbox.close failed after setup error")
            for candidate in shadow_sandboxes:
                try:
                    candidate.close()
                except Exception as exc:
                    cleanup_errors.append(exc)
                    logger.exception("shadow sandbox close failed after setup error")
            if cleanup_errors:
                from rllm.utils.infrastructure_diagnostics import infrastructure_exception_chain
                setup_error.cleanup_errors = [infrastructure_exception_chain(exc) for exc in cleanup_errors]
            if isinstance(setup_error, Exception) and not isinstance(setup_error, RolloutInfrastructureError):
                failure = RolloutInfrastructureError(
                    "sandbox_setup_failed", str(setup_error), retryable=not cleanup_errors, stage="setup",
                    diagnostics={"task_id": task.id, "exception_type": type(setup_error).__name__,
                                 "error": str(setup_error)[:4000]},
                )
                failure.cleanup_errors = getattr(setup_error, "cleanup_errors", [])
                raise failure from setup_error
            if cleanup_errors and isinstance(setup_error, RolloutInfrastructureError):
                setup_error.retryable = False
                setup_error.retry_scope = "none"
            raise

        def shadow_teardown() -> None:
            # Shadow ownership may be transferred to the DeNovoSWE batch
            # coordinator.  Keep this callback self-contained and idempotent
            # through TaskContext's split teardown guards.
            from rllm.utils.shutdown import shutdown_components
            steps = []
            if shadow_runtime is not None:
                steps.append(("shadow_abort", shadow_runtime.abort))
            steps.extend((f"shadow_{i}", candidate.close) for i, candidate in enumerate(shadow_sandboxes))
            with replacement_shadow_lock:
                replacements = list(replacement_shadow_sandboxes)
                replacement_shadow_sandboxes.clear()
            steps.extend((f"replacement_shadow_{i}", candidate.close) for i, candidate in enumerate(replacements))
            if shadow_runtime is not None:
                steps.append(("shadow_join", shadow_runtime.join_after_close))
            shutdown_components(steps)

        def primary_teardown() -> None:
            # Sandboxes are ephemeral — the hook owns this one's lifecycle and
            # closes it directly (the #616 fix); flows never hold one.
            if sandbox is not None:
                sandbox.close()

        return TaskContext(
            evaluator=evaluator,
            env=sandbox,
            env_backend=env_backend,
            shadow_runtime=shadow_runtime,
            outcome_source=("shadow_verifier" if authoritative_shadow else "primary_verifier"),
            primary_process_audit=(
                (
                    lambda: _quiesce_primary_processes(
                        sandbox,
                        primary_process_baseline,
                    )
                )
                if denovo_background_finalize
                and sandbox is not None
                and primary_process_baseline is not None
                else None
            ),
            primary_teardown=primary_teardown,
            shadow_teardown=shadow_teardown,
        )


class FixedEvaluatorHooks:
    """Hooks that bind one evaluator to every task and provision nothing.

    What :class:`~rllm.engine.agentflow_engine.AgentFlowEngine` wraps a bare
    ``evaluator=`` in, so the engine has exactly one execution path
    (always-hooks). Routes through ``_adapt_legacy_evaluator`` so dict-style
    ``evaluate(task: dict, episode)`` evaluators keep working.
    """

    def __init__(self, evaluator: Evaluator):
        from rllm.eval._resolution import _adapt_legacy_evaluator

        self.evaluator = _adapt_legacy_evaluator(evaluator)

    def setup(self, task: Task, agent_flow: AgentFlow, uid: str) -> TaskContext:  # noqa: ARG002
        from rllm.engine.agentflow_engine import TaskContext

        return TaskContext(evaluator=self.evaluator)


# ---------------------------------------------------------------------------
# Gateway wiring helpers
# ---------------------------------------------------------------------------


def pin_gateway_host_loopback(config: DictConfig) -> DictConfig:
    """Pin ``rllm.gateway.host=127.0.0.1`` if not explicitly set, so docker containers can reach it via ``host.docker.internal``."""
    if config.rllm.get("gateway", {}).get("host"):
        return config
    return OmegaConf.merge(
        config,
        OmegaConf.create({"rllm": {"gateway": {"host": "127.0.0.1"}}}),
    )


def enable_gateway_tunnel(config: DictConfig) -> DictConfig:
    """Auto-wire ``rllm.gateway.tunnel="cloudflared"`` when no tunnel is already set.

    Callers decide *when* (sandboxes run off-host — see
    :func:`scan_env_requirements`); this helper only applies the setting.
    """
    gw = config.rllm.get("gateway", {}) or {}
    if gw.get("tunnel"):
        return config
    return OmegaConf.merge(
        config,
        OmegaConf.create({"rllm": {"gateway": {"tunnel": "cloudflared"}}}),
    )


__all__ = [
    "EnvRequirements",
    "EvaluationPolicy",
    "FixedEvaluation",
    "FixedEvaluatorHooks",
    "FromTaskEvaluation",
    "RolloutPlan",
    "SandboxTaskHooks",
    "enable_gateway_tunnel",
    "pin_gateway_host_loopback",
    "resolve_rollout_plan",
    "scan_env_requirements",
]
