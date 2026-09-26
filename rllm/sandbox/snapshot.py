"""Task environment fingerprints and cold sandbox provisioning."""

from __future__ import annotations

import hashlib

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rllm.sandbox.protocol import Sandbox
    from rllm.types import Task

def env_key(backend: str, base_image: str, run_commands: list[str], install_script: str = "") -> str:
    """Content-hash fingerprint of an environment: ``rllm-env-<hash12>``.

    Hashes ``(backend, base_image, RUN block, install script)`` — never
    ``task.id`` — so GRPO group copies share one key and any image/RUN/install
    change yields a new key (a clean miss, never a stale hit). An empty
    ``install_script`` contributes nothing, keeping task-only keys stable.
    """
    parts = [backend, base_image, *run_commands]
    if install_script:
        parts += ["install:", install_script]
    payload = "\n".join(parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"rllm-env-{digest}"

def env_key_for(task: Task, backend: str, install_script: str = "") -> str:
    """Fingerprint a task's environment via the shared image/RUN resolution."""
    from rllm.eval._resolution import _dockerfile_run_commands, _resolve_image

    return env_key(backend, _resolve_image(task, backend), _dockerfile_run_commands(task), install_script)

def install_script_for(agent_flow: object) -> str:
    """The flow's CLI install script, ``""`` when it has none (host flows, bash loops)."""
    fn = getattr(agent_flow, "install_script", None)
    return fn() if callable(fn) else ""

def get_sandbox(task: Task, backend: str | None, *, backend_kwargs: dict | None = None) -> Sandbox:
    """Create the task image, replay setup, and return its live sandbox."""
    from rllm.eval._resolution import _create_sandbox_for_task
    return _create_sandbox_for_task(task, backend, **(backend_kwargs or {}))
