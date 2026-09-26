"""Adapter for a user-operated Kubernetes sandbox service.

Set SWE_MILE_K8S_ADAPTER=your_package.sandbox:create_sandbox on every Ray node.
The factory receives name, image, cpus, memory_mb, create_timeout, working_dir,
allow_internet, and optional storage_mb/env. Credentials stay in the adapter's
environment; task metadata and configuration logs must not contain secrets.
"""
from __future__ import annotations

import importlib
import inspect
import os
from functools import lru_cache

_REQUIRED = ("exec", "exec_readonly", "read_files", "upload_file", "upload_dir",
             "is_alive", "cancel_pending_exec", "pending_exec_count", "close")


@lru_cache(maxsize=16)
def _load_factory(spec: str):
    module, separator, name = spec.partition(":")
    if not separator or not module or not name or ":" in name:
        raise ValueError("SWE_MILE_K8S_ADAPTER must be package.module:create_sandbox")
    factory = getattr(importlib.import_module(module), name)
    if not callable(factory):
        raise TypeError("SWE_MILE_K8S_ADAPTER must name a callable factory")
    return factory


def check_k8s_adapter():
    """Check configuration without creating a sandbox or contacting a cluster."""
    spec = os.environ.get("SWE_MILE_K8S_ADAPTER", "").strip()
    if not spec:
        raise ValueError("K8s backend requires SWE_MILE_K8S_ADAPTER=package.module:create_sandbox")
    return _load_factory(spec)


class KubernetesSandbox:
    """Delegate to an independently implemented service client.

    The adapter must honor resources, exec users and timeouts; isolate primary
    and shadow filesystems; stop commands on cancellation; and confirm resource
    deletion in close(). The factory cleans up partially created resources on
    failure. This wrapper never retries a mutating operation. If its completion
    is uncertain, the adapter raises RolloutInfrastructureError instead of
    resubmitting it or fabricating an exit status.
    """
    supports_exec_cancellation = True

    def __init__(self, name: str, image: str, **kwargs):
        sandbox = check_k8s_adapter()(name=name, image=image, **kwargs)
        try:
            missing = [m for m in _REQUIRED if not callable(getattr(sandbox, m, None))]
            if missing:
                raise TypeError("K8s adapter is missing methods: " + ", ".join(missing))
            for method in ("exec", "exec_readonly"):
                inspect.signature(getattr(sandbox, method)).bind("true", timeout=1, user="0")
            inspect.signature(sandbox.read_files).bind(["/tmp/file"], timeout=1, user="0", max_bytes=1024)
            if not sandbox.is_alive():
                raise RuntimeError("K8s adapter returned a sandbox that is not ready")
        except BaseException:
            close = getattr(sandbox, "close", None)
            if callable(close):
                close()
            raise
        self._sandbox = sandbox
        self._closed = False

    def exec(self, command: str, timeout: float | None = None, user: str | None = None) -> str:
        return self._sandbox.exec(command, timeout=timeout, user=user)

    def exec_readonly(self, command: str, timeout: float = 30, user: str | None = None) -> str:
        return self._sandbox.exec_readonly(command, timeout=timeout, user=user)

    def exec_setup(self, command: str, timeout: float | None = None, user: str | None = None) -> str:
        return self.exec(command, timeout=timeout, user=user)

    def read_files(self, paths: list[str], timeout: float = 30, user: str | None = None,
                   max_bytes: int = 1024 * 1024) -> dict:
        """Return the bounded file snapshot contract in sandbox/file_snapshot.py."""
        return self._sandbox.read_files(paths, timeout=timeout, user=user, max_bytes=max_bytes)

    def upload_file(self, local_path: str, remote_path: str) -> None:
        self._sandbox.upload_file(local_path, remote_path)

    def upload_dir(self, local_path: str, remote_path: str) -> None:
        self._sandbox.upload_dir(local_path, remote_path)

    def is_alive(self) -> bool:
        if self._closed:
            return False
        try:
            return bool(self._sandbox.is_alive())
        except Exception:
            return False

    def cancel_pending_exec(self) -> None:
        self._sandbox.cancel_pending_exec()

    def pending_exec_count(self) -> int:
        return int(self._sandbox.pending_exec_count())

    def close(self) -> None:
        if not self._closed:
            try:
                self._sandbox.cancel_pending_exec()
            finally:
                self._sandbox.close()
                self._closed = True
