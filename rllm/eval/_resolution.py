"""Per-task verifier resolution + sandbox lifecycle helpers.

These were originally part of the ``rllm.runner.Runner`` per-task driver
that drove ``rllm eval`` before eval was unified onto
:class:`rllm.engine.agentflow_engine.AgentFlowEngine`.
``Runner`` is gone; the helpers live here because:

* :class:`rllm.hooks.SandboxTaskHooks` calls them on every rollout to set
  up the sandbox and resolve the per-task evaluator.
* :func:`build_dataset_evaluator` is the train CLI's entry point for
  resolving a single dataset-wide evaluator from a ``[verifier]`` block.

Module is private (``_resolution``) — external callers should go through
:class:`rllm.hooks.SandboxTaskHooks` or :func:`build_dataset_evaluator`.
"""

from __future__ import annotations

import base64
import importlib
import inspect
import json
import logging
import os
import re
import shlex
import tempfile
import uuid
from collections.abc import Callable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import tomllib

from rllm.eval.module_evaluator import PythonModuleEvaluator, _coerce_eval_result
from rllm.eval.script_evaluator import ShellScriptEvaluator
from rllm.eval.types import EvalOutput
from rllm.sandbox.protocol import Sandbox
from rllm.types import Episode, Evaluator, Task

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Verifier resolution
# ---------------------------------------------------------------------------


def _detect_verifier(task: Task) -> tuple[str, dict]:
    """Inspect task.toml/dataset.toml + filesystem; return (kind, config).

    Kinds: ``"sandbox-shell"``, ``"python-host"``, ``"python-hybrid"``,
    ``"registered"``, ``"import"``.
    """
    rllm_metadata = task.metadata.get("rllm") or {}
    declared_kind = str(task.metadata.get("verifier_kind") or rllm_metadata.get("verifier_kind") or "")
    if declared_kind == "nl2repo-fresh-sandbox":
        return declared_kind, {"contract": "tests/instance.json"}

    config = _read_verifier_config(task)
    task_dir = task.task_dir
    has_dockerfile = (task_dir / "environment" / "Dockerfile").exists() or (task.dataset_dir / "environment" / "Dockerfile").exists()

    if "script" in config:
        return "sandbox-shell", config
    if "module" in config:
        return ("python-hybrid" if has_dockerfile else "python-host"), config
    if "name" in config:
        return "registered", config
    if "import_path" in config:
        return "import", config

    # Auto-detect by file presence
    if (task_dir / "tests" / "test.sh").exists():
        return "sandbox-shell", {"script": "tests/test.sh"}
    if (task_dir / "tests" / "evaluate.py").exists():
        return ("python-hybrid" if has_dockerfile else "python-host"), {"module": "tests.evaluate"}
    # Shared verifier at benchmark level (rows-with-shared-verifier shape)
    if (task.dataset_dir / "tests" / "evaluate.py").exists():
        return ("python-hybrid" if has_dockerfile else "python-host"), {"module": "tests.evaluate"}
    if (task.dataset_dir / "tests" / "test.sh").exists():
        return "sandbox-shell", {"script": "tests/test.sh"}

    return "missing", {}


def _read_verifier_config(task: Task) -> dict:
    """Read ``[verifier]`` from task.toml (per-task) or dataset.toml (shared)."""
    candidates = []
    if task.sub_dir is not None:
        candidates.append(task.dataset_dir / task.sub_dir / "task.toml")
    candidates.append(task.dataset_dir / "dataset.toml")
    for cfg_path in candidates:
        if cfg_path.exists():
            try:
                raw = tomllib.loads(cfg_path.read_text())
            except Exception:
                continue
            verifier = raw.get("verifier", {})
            if verifier:
                return verifier
    return {}


def _resolve_evaluator(
    task: Task,
    sandbox: Sandbox | None,
    kind: str,
    verifier_config: dict,
) -> Evaluator:
    """Construct an Evaluator instance for this task."""
    if kind == "sandbox-shell":
        if sandbox is None:
            raise RuntimeError("sandbox-shell verifier requires an active sandbox")
        rllm_metadata = task.metadata.get("rllm") or {}
        return ShellScriptEvaluator(
            sandbox=sandbox,
            script_path=verifier_config.get("script", "tests/test.sh"),
            verifier_user=task.metadata.get("verifier_user"),
            verifier_timeout=float(
                rllm_metadata.get(
                    "resolved_verifier_timeout",
                    task.metadata.get("verifier_timeout", 600.0),
                )
            ),
            reward_file_override=verifier_config.get("reward_file"),
            timeout_is_failure=bool(
                rllm_metadata.get(
                    "verifier_timeout_is_failure",
                    False,
                )
            ),
        )

    if kind == "nl2repo-fresh-sandbox":
        if sandbox is None:
            raise RuntimeError("nl2repo-fresh-sandbox verifier requires an active primary sandbox")
        from rllm.eval.repo_generation import NL2RepoFreshSandboxEvaluator

        rllm_metadata = task.metadata.get("rllm") or {}
        return NL2RepoFreshSandboxEvaluator(
            sandbox,
            timeout=float(
                rllm_metadata.get(
                    "resolved_verifier_timeout",
                    task.metadata.get("verifier_timeout", 1800.0),
                )
            ),
        )

    if kind in ("python-host", "python-hybrid"):
        # Look in the task's own dir first, then the shared benchmark dir
        module = verifier_config.get("module", "tests.evaluate")
        function = verifier_config.get("function", "evaluate")
        for base in (task.task_dir, task.dataset_dir):
            try:
                ev = PythonModuleEvaluator.from_module(base, module, function)
                ev.sandbox = sandbox
                return ev
            except FileNotFoundError:
                continue
        raise FileNotFoundError(f"Verifier module '{module}' not found in {task.task_dir} or {task.dataset_dir}")

    if kind == "registered":
        from rllm.eval.evaluator_loader import load_evaluator

        return _adapt_legacy_evaluator(load_evaluator(verifier_config["name"]))

    if kind == "import":
        ev = _load_callable(verifier_config["import_path"])
        if isinstance(ev, type):
            ev = ev()
        if hasattr(ev, "evaluate"):
            return _adapt_legacy_evaluator(ev)
        # Bare function — wrap as a thin Evaluator
        return _FunctionEvaluator(ev)

    raise RuntimeError(f"No verifier configured for task '{task.id}' (dataset_dir={task.dataset_dir})")


def dataset_verifier_kind(dataset_dir: Path, sub_dir: Path | None = None) -> str:
    """The dataset-level verifier kind (``"missing"`` when none is configured).

    Used by the train CLI to distinguish env-style verifiers (resolved per
    task inside the sandbox — leave the trainer's ``evaluator`` unset) from a
    genuinely missing verifier (fail fast).
    """
    probe = Task(id="", instruction="", metadata={}, dataset_dir=dataset_dir, sub_dir=sub_dir)
    kind, _ = _detect_verifier(probe)
    return kind


def build_dataset_evaluator(dataset_dir: Path, sub_dir: Path | None = None) -> Evaluator | None:
    """Build a single :class:`Evaluator` from a dataset's ``[verifier]`` config.

    Supports the host-only verifier kinds (``module``, ``name``,
    ``import_path``, plus auto-detected ``tests/evaluate.py``) so the
    trainer — which expects one Evaluator for the whole dataset — can
    reuse the same per-task resolution that :class:`Runner` performs for
    eval. Sandbox-shell verifiers return ``None`` because they need a
    per-task sandbox lifecycle that lives inside :class:`Runner`.
    """
    probe = Task(id="", instruction="", metadata={}, dataset_dir=dataset_dir, sub_dir=sub_dir)
    kind, config = _detect_verifier(probe)
    if kind in (
        "sandbox-shell",
        "python-hybrid",
        "nl2repo-fresh-sandbox",
        "missing",
    ):
        return None
    return _resolve_evaluator(probe, sandbox=None, kind=kind, verifier_config=config)


# ---------------------------------------------------------------------------
# Sandbox setup (extracted from rllm/tasks/runner.py)
# ---------------------------------------------------------------------------


def _resolve_backend(task: Task, sandbox_backend: str | None) -> str:
    """Resolve the effective sandbox backend for a task."""
    return sandbox_backend or task.metadata.get("sandbox_backend") or "minisandbox"


def _create_base_sandbox(task: Task, backend: str, *, image: str | None = None, name: str | None = None, **backend_kwargs) -> Sandbox:
    """Create a sandbox from a base ``image`` — no Dockerfile RUN replay.

    ``image`` defaults to the task's resolved base image. ``backend_kwargs``
    pass through to the selected backend constructor.
    """
    from rllm.sandbox.sandboxed_flow import create_sandbox

    image = image if image is not None else _resolve_image(task, backend)
    if name is None:
        safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "-", task.id)
        name = f"rllm-{safe_id}-{uuid.uuid4().hex[:6]}"
    if backend == "minisandbox":
        backend_kwargs.setdefault("task_id", task.id)
    return create_sandbox(backend, name=name, image=image, **_sandbox_resource_kwargs(task, backend), **backend_kwargs)


def _replay_dockerfile(task: Task, sandbox: Sandbox, backend: str) -> None:
    """Replay the Dockerfile RUN steps on a live sandbox (stage C).

    Task images provide the base filesystem. Replay additional RUN steps
    before task setup; setup and verifier contracts check required tools.
    """
    for cmd in _dockerfile_run_commands(task):
        _safe_exec(sandbox, cmd, timeout=900)


def _create_sandbox_for_task(task: Task, sandbox_backend: str | None, **backend_kwargs) -> Sandbox:
    """Cold-path sandbox creation: base image + RUN replay (today's behavior)."""
    backend = _resolve_backend(task, sandbox_backend)
    sandbox = _create_base_sandbox(task, backend, **backend_kwargs)
    _replay_dockerfile(task, sandbox, backend)
    return sandbox


def _sandbox_resource_kwargs(task: Task, backend: str) -> dict:
    """Forward task.toml resource and isolation settings to the backend."""
    env = task.metadata.get("environment", {}) or {}
    cpus, mem_mb, disk_mb = env.get("cpus"), env.get("memory_mb"), env.get("storage_mb")
    kw: dict = {}
    if backend == "k8s":
        if not cpus or not mem_mb:
            raise ValueError("K8s tasks must declare positive environment.cpus and environment.memory_mb")
        kw.update(cpus=int(cpus), memory_mb=int(mem_mb),
                  create_timeout=float(env.get("build_timeout_sec") or 1800.0),
                  working_dir=str(env.get("workdir") or "/"),
                  allow_internet=bool(env.get("allow_internet", False)))
        if disk_mb:
            kw["storage_mb"] = int(disk_mb)
        if env.get("env"):
            kw["env"] = dict(env["env"])
    elif backend == "minisandbox":
        kw["create_timeout"] = float(env.get("build_timeout_sec") or 1800.0)
        if not cpus or not mem_mb:
            raise ValueError("MiniSandbox tasks must declare positive environment.cpus and environment.memory_mb")
        kw["cpus"] = int(cpus)
        kw["memory_mb"] = int(mem_mb)
        if disk_mb:
            kw["storage_mb"] = int(disk_mb)
        if env.get("env"):
            runtime_env, removed_proxy_keys = _minisandbox_runtime_environment(
                env.get("env") or {}
            )
            kw["env"] = runtime_env
            if removed_proxy_keys:
                logger.debug(
                    "Removed MiniSandbox proxy variables for %s: %s",
                    task.id,
                    ", ".join(removed_proxy_keys),
                )
        kw["working_dir"] = str(env.get("workdir") or "/")
        kw["allow_internet"] = bool(env.get("allow_internet", False))
    return kw


def _minisandbox_runtime_environment(
    environment: Mapping[str, Any],
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Return a MiniSandbox-only direct-egress environment copy.

    Host proxy endpoints can be unreachable from isolated network namespaces.
    Remote adapters receive the task environment unchanged.
    """

    removed: list[str] = []
    result: dict[str, str] = {}
    for raw_key, raw_value in environment.items():
        key = str(raw_key)
        if key.casefold() in {"http_proxy", "https_proxy", "all_proxy"}:
            removed.append(key)
            continue
        result[key] = str(raw_value)
    return result, tuple(sorted(removed))


def _dockerfile_run_commands(task: Task) -> list[str]:
    """Return a task's ``environment/Dockerfile`` ``RUN`` shell steps (joining
    ``\\``-continuations). Non-``RUN`` directives — ``COPY``/``ADD`` etc. — are
    skipped; only ``RUN`` is replayable on a live sandbox.
    """
    dockerfile = task.task_dir / "environment" / "Dockerfile"
    if not dockerfile.exists():
        dockerfile = task.dataset_dir / "environment" / "Dockerfile"
    if not dockerfile.exists():
        return []
    try:
        lines = dockerfile.read_text().splitlines()
    except OSError:
        return []

    commands: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.upper().startswith("RUN "):
            parts = [stripped[4:]]
            while parts[-1].rstrip().endswith("\\"):
                parts[-1] = parts[-1].rstrip()[:-1]
                i += 1
                if i >= len(lines):
                    break
                parts.append(lines[i])
            cmd = "\n".join(parts).strip()
            if cmd:
                commands.append(cmd)
        i += 1
    return commands










def _task_profile(task: Task) -> str:
    rllm_metadata = task.metadata.get("rllm") or {}
    return str(task.metadata.get("task_profile") or rllm_metadata.get("task_profile") or "")


def _doc2repo_image(task: Task) -> str:
    """Return the official public BeyondSWE image for one Doc2Repo task."""
    tag = task.id.strip().lower()
    if len(tag) > 128 or not re.fullmatch(r"[a-z0-9_][a-z0-9_.-]*", tag):
        raise ValueError(f"invalid Doc2Repo task id for Docker tag: {task.id!r}")
    return f"aweaiteam/beyondswe:{tag}"


def _resolve_image(task: Task, backend: str) -> str:
    """Resolve the public task image and optional user-operated mirror."""
    env_config = task.metadata.get("environment", {}) or {}
    configured = env_config.get("docker_image", "python:3.11-slim")

    # The public Doc2Repo image tag is the lower-cased task identifier.
    if _task_profile(task) == "repo_generation_doc2repo":
        configured = _doc2repo_image(task)

    return _resolve_public_image(configured, task_id=task.id)




def _setup_task_environment(task: Task, sandbox: Sandbox) -> None:
    """Upload environment/files/, run setup.sh and [rllm].setup_commands.

    Honors the agent/verifier user split when configured in task.toml.
    """
    # ``workdir`` is unset for tasks (e.g. swesmith) whose Dockerfile
    # already declares a meaningful WORKDIR (``/testbed``) — forcing
    # ``/workspace`` would override it and break verifiers that
    # ``cd``-into-cwd or ``git checkout``. The ``mkdir`` / ``chown``
    # / ``upload_dir(files)`` steps below only fire when a workdir is
    # explicitly declared.
    workdir = task.metadata.get("workdir")
    strict_setup = task.metadata.get("setup_failure_mode") == "raise"
    setup_timeout = float(task.metadata.get("setup_timeout_sec", 300.0))
    trusted_setup_exec = getattr(sandbox, "exec_setup", None)
    has_trusted_setup_exec = callable(trusted_setup_exec)

    def setup_exec(
        command: str,
        *,
        timeout: float | None,
        user: str | None = None,
        idempotent: bool = False,
    ) -> str:
        executor = trusted_setup_exec if has_trusted_setup_exec else sandbox.exec
        if idempotent and not has_trusted_setup_exec and callable(getattr(sandbox, "exec_readonly", None)):
            executor = sandbox.exec_readonly
        if strict_setup:
            return executor(command, timeout=timeout, user=user)
        try:
            return executor(command, timeout=timeout, user=user)
        except Exception as exc:
            logger.debug("setup exec failed (suppressed): %s — %s", command[:200], exc)
            return ""

    env_root = task.task_dir / "environment"
    if not env_root.is_dir():
        env_root = task.dataset_dir / "environment"

    if workdir:
        workdir_q = shlex.quote(str(workdir))
        setup_exec(f"mkdir -p -- {workdir_q}", timeout=getattr(sandbox, "control_read_timeout", 30), idempotent=True)
        if has_trusted_setup_exec:
            setup_exec(
                f"chown 0:0 -- {workdir_q} && chmod u+rwx -- {workdir_q}",
                timeout=30,
            )

    files_dir = env_root / "files"
    if files_dir.is_dir():
        # Falls back to ``/workspace`` when files/ ships but no workdir
        # is declared — preserves the historical default for tasks that
        # actually rely on it.
        sandbox.upload_dir(str(files_dir), workdir or "/workspace")

    if _task_profile(task) == "repo_generation_nl2repo":
        from rllm.eval.repo_generation import NL2RepoFreshSandboxEvaluator
        from rllm.sandbox.nl2repo_environment import prepare_nl2repo_image
        prepare_nl2repo_image(sandbox, NL2RepoFreshSandboxEvaluator._load_contract(task))

    setup_script = env_root / "setup.sh"
    if setup_script.exists():
        if _task_profile(task) in {"repo_generation_nl2repo", "repo_generation_doc2repo"}:
            from rllm.sandbox.git_bundle import ensure_nl2repo_git
            from rllm.sandbox.nl2repo_setup import configure_repo_generation_package_manager

            configure_repo_generation_package_manager(sandbox, timeout=setup_timeout)
            ensure_nl2repo_git(sandbox)
        sandbox.upload_file(str(setup_script), "/tmp/rllm_setup.sh")
        setup_exec(
            "chmod +x /tmp/rllm_setup.sh && /tmp/rllm_setup.sh",
            timeout=setup_timeout,
        )

    for cmd in task.metadata.get("setup_commands", []) or []:
        setup_exec(cmd, timeout=setup_timeout)

    # Image build steps can leave repository files owned by arbitrary UIDs.
    # Trusted MiniSandbox setup retains only filesystem ownership capabilities;
    # normalize the much smaller post-clean tree before capability-free agent
    # commands begin.
    if workdir and has_trusted_setup_exec:
        setup_exec(
            f"chown -R 0:0 -- {workdir_q} && chmod -R u+rwX -- {workdir_q}",
            timeout=setup_timeout,
        )

    agent_user = task.metadata.get("agent_user")
    if agent_user:
        setup_exec("mkdir -p /logs/verifier /tmp/rllm /tests", timeout=10)
        setup_exec("chmod 700 /logs/verifier /tmp/rllm /tests", timeout=10)
        setup_exec("chown root:root /logs/verifier /tmp/rllm /tests", timeout=10)
        if workdir:
            setup_exec(
                f"chown -R {shlex.quote(str(agent_user))} -- {workdir_q}",
                timeout=30,
            )

    env_vars = task.metadata.get("env_vars", {}) or task.metadata.get("environment", {}).get("env", {})
    if env_vars:
        exports = " && ".join(f"export {k}='{v}'" for k, v in env_vars.items())
        setup_exec(exports, timeout=10)

    # Security assertions are backend-specific and run before the model sees
    # the filesystem. NL2Repo's pristine verifier image intentionally ships
    # tests, but its primary setup contract must remove every verifier-shaped
    # Python file. The fresh final-verifier sandbox bypasses this setup path.
    from rllm.sandbox.backends.minisandbox import MiniSandbox

    if isinstance(sandbox, MiniSandbox):
        security_checks = ["{ test ! -e /tests || { echo 'MiniSandbox setup: /tests remains visible' >&2; exit 1; }; }"]
        if _task_profile(task) == "repo_generation_nl2repo":
            security_checks.extend(
                [
                    "{ test ! -e /workspace/tests || { echo 'NL2Repo setup: /workspace/tests remains visible' >&2; exit 1; }; }",
                    "test -z \"$(find /workspace -type f "
                    "\\( -name 'test_*.py' -o -name '*_test.py' -o "
                    "-name 'conftest.py' \\) -print -quit 2>/dev/null)\" || "
                    "{ echo 'NL2Repo setup: verifier-shaped Python files remain visible' >&2; exit 1; }",
                ]
            )
        trusted_setup_exec(
            " && ".join(security_checks),
            timeout=max(60.0, setup_timeout),
            user="root",
        )


def _safe_exec(sandbox: Sandbox, command: str, timeout: float | None = None, user: str | None = None) -> str:
    try:
        return sandbox.exec(command, timeout=timeout, user=user)
    except Exception as e:
        logger.debug("exec failed (suppressed): %s — %s", command[:200], e)
        return ""


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class _FunctionEvaluator:
    """Wrap a bare ``evaluate(task, episode)`` callable as an Evaluator."""

    def __init__(self, fn: Callable):
        self.fn = fn

    def evaluate(self, task: Task, episode: Episode) -> EvalOutput:
        result = self.fn(task, episode)
        return _coerce_eval_result(result) if not isinstance(result, EvalOutput) else result


def _adapt_legacy_evaluator(ev: Any) -> Evaluator:
    """Adapt evaluators with ``evaluate(task: dict, episode)`` to ``evaluate(task: Task, episode)``.

    Only an explicit ``dict`` annotation (string form included — with
    ``from __future__ import annotations`` annotations are strings) or a
    legacy parameter name opts into the dict calling convention; an
    unannotated evaluator gets the ``Task``.
    """
    sig = inspect.signature(ev.evaluate)
    params = list(sig.parameters.values())
    if not params:
        return ev
    first = params[0]
    annotation = first.annotation if first.annotation is not inspect.Parameter.empty else None

    is_dict_annotation = annotation is dict or annotation == "dict" or (isinstance(annotation, str) and annotation.startswith("dict"))
    if is_dict_annotation or first.name in ("task_data", "task_info"):
        return _LegacyDictAdapter(ev)
    return ev


class _LegacyDictAdapter:
    """Pass ``task.metadata`` (dict) to an old-style Evaluator."""

    def __init__(self, inner: Any):
        self.inner = inner

    def evaluate(self, task: Task, episode: Episode) -> EvalOutput:
        return self.inner.evaluate(task.metadata, episode)


def _load_callable(import_path: str) -> Callable:
    """Resolve ``module.path:attr`` to a Python object."""
    if ":" not in import_path:
        raise ValueError(f"import_path must be 'module:attr', got {import_path!r}")
    module_path, attr_name = import_path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


def _resolve_public_image(image: str, *, task_id: str | None = None) -> str:
    """Apply an explicit user image mapping; otherwise retain the public image."""
    mapping_path = os.environ.get("SWE_MILE_IMAGE_MAP_FILE", "")
    if not mapping_path:
        return image
    mapping = json.loads(Path(mapping_path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or not all(isinstance(k, str) and isinstance(v, str) and v for k, v in mapping.items()):
        raise ValueError("SWE_MILE_IMAGE_MAP_FILE must contain an image-to-image JSON object")
    return mapping.get(image, image)
