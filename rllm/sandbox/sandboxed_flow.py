"""SandboxedAgentFlow: base class for agents that run against a sandbox.

The flow holds no sandbox state. :class:`rllm.hooks.SandboxTaskHooks` owns
the lifecycle: it creates the sandbox, runs the harness install (when not
already baked into the image), and closes it after evaluation. The engine
passes the live sandbox into ``run(task, config, *, env)`` as a call
argument, so parallel rollouts can share one flow instance safely.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from rllm.sandbox.protocol import Sandbox
from rllm.types import AgentConfig, Episode, Task

logger = logging.getLogger(__name__)


class SandboxedAgentFlow(ABC):
    """Base class for agents that run against a sandboxed execution environment.

    The sandbox backend is pluggable via ``sandbox_backend``:
    ``"minisandbox"`` or ``"k8s"``.

    Subclasses implement :meth:`run` and may override :meth:`get_image`
    for per-task images.
    """

    # Declared env requirement — read by rllm.hooks.resolve_rollout_plan.
    needs_env: bool = True
    # Where the flow's LLM client runs. Host-side loops (bash/oracle) keep the
    # local gateway URL; CLI harnesses (BaseCliHarness) override to True so
    # the in-sandbox process gets the publicly-reachable (tunneled) URL.
    llm_inside_env: bool = False
    sandbox_backend: str = "minisandbox"
    image: str = "python:3.11-slim"
    # Default cap on concurrent sandboxes. The eval/train runner clamps
    # effective concurrency to this value, so it is the single source of
    # truth for sandboxed flows. Subclasses override only to deviate;
    # ``--sandbox-concurrency`` overrides it per-run.
    max_concurrent: int = 64
    # Active sandbox slots consumed by one rollout.  Most flows use one;
    # codeflow R2E-Gym shadow evaluation uses primary + shadow.
    sandbox_slots_per_rollout: int = 1

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def configure(self, overrides: dict) -> dict:
        """Apply caller/CLI overrides this flow understands; return the rest.

        The wiring layer warns about anything returned, so a flag that a
        given agent can't honor is visible instead of a silent no-op.
        """
        leftovers = dict(overrides)
        backend = leftovers.pop("sandbox_backend", None)
        if backend is not None:
            self.sandbox_backend = backend
        concurrency = leftovers.pop("sandbox_concurrency", None)
        if concurrency is not None:
            self.max_concurrent = concurrency
        return leftovers

    def get_image(self, task: dict) -> str:
        """Return container image for this task. Override for per-task images."""
        return self.image

    @abstractmethod
    def run(self, task: Task, config: AgentConfig, *, env: Sandbox) -> Episode: ...


def create_sandbox(backend: str, name: str, image: str, **kwargs) -> Sandbox:
    """Create one of the two supported SWE-MILE sandbox backends."""
    if backend == "minisandbox":
        from rllm.sandbox.backends.minisandbox import MiniSandbox
        return MiniSandbox(name=name, image=image, **kwargs)
    if backend == "k8s":
        from rllm.sandbox.backends.k8s import KubernetesSandbox
        return KubernetesSandbox(name=name, image=image, **kwargs)
    raise ValueError(f"Unsupported sandbox backend: {backend!r}; choose minisandbox or k8s")






