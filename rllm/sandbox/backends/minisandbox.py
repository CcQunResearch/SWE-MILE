"""Ray-managed node-local SWE-MiniSandbox backend for SWE datasets.

The driver-side :class:`MiniSandboxCluster` starts exactly one privileged
service on every GPU node selected by the current VERL job.  Lightweight
``MiniSandbox`` handles then route lifecycle and command RPCs to those
services.  OCI images live in a shared, immutable cache; unpacked root files
and writable overlays stay on each node's local disk.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import io
import json
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any

from rllm.sandbox.backends.session_gc import STARTUP_RECLAMATION_TIMEOUT
from rllm.utils import rpc_diagnostics as rpc_diag

logger = logging.getLogger(__name__)

MINISANDBOX_BACKEND_REVISION = 25
MINISANDBOX_UNPACK_REVISION = 2
CLOSE_RECONCILE_SECONDS = 120.0

_ACTIVE_CLUSTER: MiniSandboxCluster | None = None
_ACTIVE_CLUSTER_LOCK = threading.Lock()


class _ReplyWaitTimeout(TimeoutError):
    """A local reply wait expired; the remote operation is still owned by Ray."""


class _RayReply:
    """Subscribe once, then wait locally without repeated blocking CoreWorker gets.

    Hundreds of rollout/shadow threads share one Ray worker. Synchronous
    ray.get calls can overrun their timeout inside CoreWorker, delaying both
    command replies and lifecycle cleanup. ObjectRef.future delivers the same
    reply to a Python condition variable without that per-wait native path.
    A timeout never cancels the subscription or replays the remote operation.
    """

    def __init__(self, ray, reference):
        self._ray, self._reference = ray, reference
        subscribe = getattr(reference, "future", None)
        self._future = subscribe() if callable(subscribe) else None

    def future(self):
        if self._future is None:
            raise TypeError("RPC reference does not support future subscriptions")
        return self._future

    def get(self, timeout):
        if self._future is None:
            return self._ray.get(self._reference, timeout=timeout)
        try:
            return self._future.result(timeout=timeout)
        except TimeoutError:
            # A remote command may itself raise TimeoutError. Do not confuse
            # a completed exception with a local, retryable reply wait.
            if self._future.done():
                return self._future.result()
            raise _ReplyWaitTimeout("MiniSandbox reply wait expired") from None


class _CreateControl:
    """Cooperative cancellation of cold-cache work, with an outer RPC budget."""

    def __init__(self, timeout: float, *, queued: bool = False):
        self.cancelled = threading.Event()
        self.timeout = float(timeout)
        self.queued_at = time.monotonic()
        self.started_at = None
        self.finished_at = None
        self.deadline = None if queued else self.queued_at + self.timeout
        self.started = False
        self.stage = "queued"

    def check(self) -> None:
        if self.cancelled.is_set() or (self.deadline is not None and time.monotonic() >= self.deadline):
            error = RuntimeError(f"MiniSandbox create cancelled or timed out at {self.stage}")
            error.minisandbox_cleanup_confirmed = True
            error.diagnostics = {"stage": self.stage, "started": self.started,
                                 "queue_seconds": (self.started_at if self.started_at is not None else time.monotonic()) - self.queued_at,
                                 "operation_timeout": self.timeout}
            raise error

    def begin(self) -> None:
        self.check()
        self.started_at = time.monotonic()
        self.deadline = self.started_at + self.timeout
        self.started = True
        self.stage = "creating"


def _wait_for_create(ray, actor, reference, operation_id: str, timeout: float):
    """One RPC submission; queued waits advance only on actual create completions.

    A live but stalled actor cannot keep the caller waiting with heartbeats.
    Once admitted, the worker's remaining execution budget bounds the RPC.
    """
    timeout_error = getattr(getattr(ray, "exceptions", None), "GetTimeoutError", TimeoutError)
    deadline = time.monotonic() + timeout + 60.0
    progress = None
    admitted = False
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("MiniSandbox create execution stalled" if admitted else "MiniSandbox create admission stalled")
        try:
            return ray.get(reference, timeout=min(30.0, remaining))
        except timeout_error:
            try:
                status = ray.get(actor.operation_status.remote(operation_id), timeout=min(5.0, max(0.1, deadline - time.monotonic())))
            except timeout_error:
                # A delayed control reply says nothing about the original
                # create. Keep its reference and deadline, without cancelling.
                continue
            if not isinstance(status, dict) or not isinstance(status.get("queue_progress"), int):
                raise RuntimeError("MiniSandbox create status requires backend revision 14")
            if status.get("started") is True:
                if not admitted:
                    budget = status.get("remaining_seconds")
                    if not isinstance(budget, (int, float)) or not 0 <= budget <= timeout:
                        raise RuntimeError("invalid MiniSandbox create execution budget")
                    deadline = time.monotonic() + budget + 60.0
                    admitted = True
            elif not admitted:
                if progress is not None and status["queue_progress"] > progress:
                    deadline = time.monotonic() + timeout + 60.0
                progress = status["queue_progress"]


def _operation_error(status: dict) -> RuntimeError:
    detail = status.get("error") or {}
    error = RuntimeError(str(detail.get("message") or "MiniSandbox operation failed"))
    error.diagnostics = {**(detail.get("diagnostics") or {}), "operation_status": status}
    error.minisandbox_cleanup_confirmed = detail.get("cleanup_confirmed") is True
    return error


def _wait_for_lifecycle(ray, actor, reference, operation_id: str, *, kind: str, timeout: float,
                        node_monitor=None, reconcile_timeout: float = 0.0, on_reconcile=None):
    """Poll one server-owned operation; RPC timeouts never replay mutations.

    Control methods return immediately. Only completed FIFO work extends a
    queued wait. Once running, a server-reported remaining budget fixes the
    execution deadline, independently of other operations' progress.
    """
    timeout_error = (getattr(getattr(ray, "exceptions", None), "GetTimeoutError", TimeoutError), _ReplyWaitTimeout)
    queue_budget = timeout + 60.0 if kind == "create" else 60.0
    grace = 60.0 if kind == "create" else 5.0
    deadline = time.monotonic() + queue_budget
    close_wait_deadline = deadline + timeout + grace
    admitted = False
    progress = None
    status = {}
    acknowledged = False
    status_ref = None
    status_received = False
    status_timeouts = 0
    last_status_at = None
    reconcile_deadline = None
    reply = _RayReply(ray, reference)
    while True:
        if reconcile_deadline is not None:
            deadline = min(deadline, reconcile_deadline)
        elif kind == "close":
            deadline = min(deadline, close_wait_deadline)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            # Reconcile the original operation over an independent Ray quota
            # and event loop. A lost control reply does not prove a stuck FIFO
            # or an unsuccessful close. Never resubmit the mutation here.
            diagnostic = {}
            probe = getattr(actor, "lifecycle_diagnostics", None)
            if probe is not None:
                try:
                    diagnostic = _RayReply(ray, probe.remote(operation_id, kind)).get(timeout=5.0)
                    recovered = diagnostic.get("operation_status")
                    if isinstance(recovered, dict) and isinstance(recovered.get("queue_progress"), int):
                        status = recovered
                        status_received = True
                except Exception as exc:
                    diagnostic = {"error": str(exc)[-2000:]}
            if node_monitor:
                from rllm.utils.node_diagnostics import query_monitor
                delivery = {"ensure_close": True} if kind == "close" and node_monitor.get("close_submission") else {}
                independent = query_monitor(node_monitor, operation_id, **delivery)
                diagnostic["node_control"] = independent
                receipt = independent.get("receipt")
                if (kind == "close" and status.get("state") not in {"succeeded", "failed", "cancelled"}
                        and isinstance(receipt, dict) and receipt.get("operation_id") == operation_id):
                    # Only a completed future emits this receipt. A generic
                    # finally/finish event is never a cleanup confirmation.
                    if receipt.get("state") == "succeeded" and receipt.get("cleanup_confirmed") is True:
                        logger.warning("MiniSandbox close reconciled through independent monitor: %s", operation_id)
                        return True
                    if receipt.get("state") == "failed":
                        error = RuntimeError(receipt.get("error") or "MiniSandbox close failed")
                        error.diagnostics = {"operation_id": operation_id, "independent_diagnostics": diagnostic}
                        raise error
            if kind == "close" and status.get("state") == "succeeded" and status.get("cleanup_confirmed") is True:
                return True
            if kind == "create" and status.get("state") == "succeeded":
                return status.get("result")
            if kind == "cancel" and status.get("cancel_state") == "succeeded":
                return status.get("cleanup_confirmed") is True
            terminal = status.get("cancel_state") if kind == "cancel" else status.get("state")
            if terminal in {"failed", "cancelled"}:
                raise _operation_error(status)
            budget_status = (status.get("cleanup_status") or status) if kind == "cancel" else status
            # The first received state may already be executing. Honor its
            # existing server deadline instead of spending execution time in
            # the client's previously unacknowledged queue wait.
            if not admitted and budget_status.get("started") is True:
                budget = budget_status.get("remaining_seconds")
                if isinstance(budget, (int, float)) and 0 < budget <= timeout:
                    deadline = time.monotonic() + budget + grace
                    admitted = True
                    continue
            phase = "execution" if admitted else "queue"
            if not status_received or budget_status.get("state") == "unknown":
                phase = "control"
            if kind == "close" and reconcile_timeout > 0:
                if reconcile_deadline is None:
                    reconcile_deadline = deadline + reconcile_timeout
                    if on_reconcile is not None:
                        on_reconcile()
                    rpc_diag.event("close_reconciling", operation_id, deadline_phase=phase,
                                   reconcile_timeout_seconds=reconcile_timeout)
                if time.monotonic() < reconcile_deadline:
                    deadline = min(time.monotonic() + 10.0, reconcile_deadline)
                    continue
            error = TimeoutError(f"MiniSandbox {kind} {phase} deadline exceeded")
            error.diagnostics = {"operation_id": operation_id, "operation_status": status,
                                 "operation": kind, "budget_seconds": timeout,
                                 "deadline_phase": phase, "submit_acknowledged": acknowledged,
                                 "status_received": status_received, "status_timeouts": status_timeouts,
                                 "last_status_age_seconds": None if last_status_at is None else time.monotonic() - last_status_at,
                                 "independent_diagnostics": diagnostic}
            raise error
        try:
            if not acknowledged:
                try:
                    allowance = min(5.0, remaining)
                    rpc_diag.call("submit_ack_wait", operation_id, lambda: reply.get(timeout=allowance),
                                  trace=kind == "close", wait_budget_seconds=allowance)
                    acknowledged = True
                except timeout_error:
                    pass  # Reconcile a lost submit reply by its operation ID.
            if status_ref is None:
                method = actor.operation_status if kind != "close" else actor.cleanup_status
                status_ref = _RayReply(ray, rpc_diag.call("status_submit", operation_id, lambda: method.remote(operation_id), trace=kind == "close"))
            allowance = min(5.0, max(0.001, deadline - time.monotonic()))
            status = rpc_diag.call("status_wait", operation_id, lambda: status_ref.get(timeout=allowance),
                                   trace=kind == "close", wait_budget_seconds=allowance)
            status_ref = None
        except timeout_error:
            status_timeouts += 1
            continue
        if not isinstance(status, dict) or not isinstance(status.get("queue_progress"), int):
            raise RuntimeError("MiniSandbox lifecycle status requires backend revision 15")
        status_received = True
        last_status_at = time.monotonic()
        if kind == "cancel":
            if status.get("cancel_state") == "succeeded":
                return status.get("cleanup_confirmed") is True
            if status.get("cancel_state") == "failed":
                raise _operation_error(status)
            # Cancellation may be reclaiming a late create in the close FIFO.
            budget_status = status.get("cleanup_status") or status
        else:
            if status.get("state") == "succeeded":
                return status.get("result") if kind == "create" else True
            if status.get("state") in {"failed", "cancelled"}:
                raise _operation_error(status)
            budget_status = status
        if budget_status.get("started") is True:
            if not admitted:
                budget = budget_status.get("remaining_seconds")
                if not isinstance(budget, (int, float)) or not 0 <= budget <= timeout:
                    raise RuntimeError("invalid MiniSandbox lifecycle execution budget")
                deadline = time.monotonic() + budget + grace
                admitted = True
        elif not admitted:
            current_progress = budget_status.get("queue_progress", 0)
            if progress is not None and current_progress > progress:
                deadline = time.monotonic() + queue_budget
            progress = current_progress
        time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))


def _build_command(command: list[str], control: _CreateControl | None):
    if control is None:
        return subprocess.run(command, capture_output=True, text=True, timeout=1800)
    control.check()
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True) as process:
        try:
            while True:
                control.check()
                try:
                    stdout, stderr = process.communicate(timeout=0.5)
                    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
                except subprocess.TimeoutExpired:
                    continue
        except BaseException:
            # Kill the entire local copy/unpack process group before releasing
            # the cache build lock. No sandbox has been created at this stage.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate(timeout=5)
            raise


def _exception_chain(exc: BaseException):
    """RayTaskError stores the remote exception in .cause, not __cause__."""
    pending, seen = [exc], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for nested in (current.__context__, current.__cause__, getattr(current, "cause", None)):
            if isinstance(nested, BaseException):
                pending.append(nested)


def _exception_diagnostics(exc: BaseException) -> dict:
    chain = list(_exception_chain(exc))
    return {
        "exception_chain": [{"type": type(item).__name__, "message": str(item)[-2000:]}
                            for item in chain[:8]],
        "remote_diagnostics": [item.diagnostics for item in chain
                               if isinstance(getattr(item, "diagnostics", None), dict)
                               and item.diagnostics][:8],
    }


def _node_service_failure(exc: BaseException) -> bool:
    """Distinguish Ray/service loss from an ordinary sandbox command error."""

    service_error_names = {
        "ActorDiedError",
        "NodeDiedError",
        "ObjectLostError",
        "OwnerDiedError",
        "RayActorError",
        "RaySystemError",
        "WorkerCrashedError",
    }
    service_markers = (
        "actor died",
        "actor is dead",
        "is not alive",
        "node service is closed",
        "namespace exited",
        "unknown minisandbox:",
        "worker crashed",
    )
    for current in _exception_chain(exc):
        if type(current).__name__ in service_error_names:
            return True
        if any(marker in str(current).casefold() for marker in service_markers):
            return True
    return False


def _raise_node_service_failure(
    exc: BaseException,
    *,
    operation: str,
) -> None:
    for current in _exception_chain(exc):
        if type(current).__name__ == "MiniSandboxCommandInterrupted":
            from rllm.types import RolloutInfrastructureError
            raise RolloutInfrastructureError(
                "sandbox_closed", "MiniSandbox command interrupted by sandbox close",
                stage="sandbox_rpc", retryable=True, retry_scope="full_rollout",
                diagnostics={**getattr(current, "diagnostics", {}), "operation": operation},
            ) from exc
        detail = getattr(current, "diagnostics", {})
        if (type(current).__name__ == "MiniSandboxRuntimeError"
                and isinstance(detail, dict)
                and isinstance(detail.get("returncode"), int) and detail["returncode"] < 0):
            from rllm.types import RolloutInfrastructureError
            raise RolloutInfrastructureError(
                "minisandbox_command_signalled", "MiniSandbox helper terminated by signal",
                stage="sandbox_rpc", retryable=True, retry_scope="full_rollout",
                diagnostics={**detail, "operation": operation},
            ) from exc
    if type(exc).__name__ == "GetTimeoutError":
        from rllm.types import RolloutInfrastructureError
        raise RolloutInfrastructureError(
            "minisandbox_rpc_timeout", f"MiniSandbox RPC deadline exceeded during {operation}",
            retryable=True, stage="sandbox_rpc", retry_scope="full_rollout",
            diagnostics={"operation": operation, "exception_type": type(exc).__name__},
        ) from exc
    if not _node_service_failure(exc):
        return
    from rllm.types import RolloutInfrastructureError

    raise RolloutInfrastructureError(
        "minisandbox_service_lost",
        f"MiniSandbox node service failed during {operation}: {exc}",
        retryable=True,
        stage="sandbox_rpc",
        retry_scope="full_rollout",
    ) from exc


def _observe_late_rpc(reference, request_id: str, *, operation: str, sandbox_id: str) -> None:
    """Consume one abandoned reply without waiting, cancelling or resubmitting.

    A timed-out write can remain queued behind actor work while close proceeds.
    Its later exception must be observed, rather than emitted by Ray's object
    destructor as an unhandled error. The original caller still fails.
    """
    try:
        future = reference.future()
        def completed(done):
            fields = {"operation": operation, "sandbox_id": sandbox_id}
            try:
                done.result()
                fields["outcome"] = "returned"
            except BaseException as exc:
                fields.update(outcome="failed", exception_type=type(exc).__name__,
                              reason=getattr(exc, "reason", None))
            rpc_diag.event("late_rpc_result", request_id, **fields)
        future.add_done_callback(completed)
    except Exception as exc:
        rpc_diag.event("late_rpc_observation_failed", request_id, operation=operation,
                       sandbox_id=sandbox_id, exception_type=type(exc).__name__)


def _safe_component(value: str, *, label: str) -> str:
    value = str(value).strip()
    if not value or value in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"unsafe MiniSandbox {label}: {value!r}")
    return value


def _safe_remote_path(value: str, *, allow_root: bool = False) -> str:
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe MiniSandbox path: {value!r}")
    if not allow_root and str(path) == "/":
        raise ValueError("MiniSandbox destination may not be root")
    return str(path)


def _ray_node_identity(ray_module: Any) -> tuple[str, str]:
    """Return the current Ray node identity across supported Ray releases.

    ``RuntimeContext.get_node_ip_address`` is not present in the Ray version
    shipped by VERL 0.8.  Newer/other releases have exposed it there, while
    the compatible public API used by VERL is ``ray.util.get_node_ip_address``.
    Keep both paths so the node service does not depend on one Ray minor
    version's RuntimeContext surface.
    """

    context = ray_module.get_runtime_context()
    node_id = str(context.get_node_id()).strip()
    if not node_id:
        raise RuntimeError("Ray runtime context returned an empty node id")

    context_ip_getter = getattr(context, "get_node_ip_address", None)
    util_ip_getter = getattr(getattr(ray_module, "util", None), "get_node_ip_address", None)
    for getter in (context_ip_getter, util_ip_getter):
        if not callable(getter):
            continue
        node_ip = str(getter()).strip()
        if node_ip:
            return node_id, node_ip
    raise RuntimeError("Ray does not expose a usable node IP through RuntimeContext or ray.util")


def _network_namespace_inode(path: str | Path) -> int:
    return int(os.stat(path).st_ino)


def _wait_for_isolated_network_namespace(
    process: subprocess.Popen[Any],
    *,
    timeout: float = 5.0,
) -> None:
    """Wait until an ``unshare --net`` child has entered its new namespace.

    ``Popen`` returns after fork, before the child necessarily executes
    ``unshare(2)``. Moving a veth to the PID during that window leaves it in
    the caller's namespace; the later ``nsenter`` then sees an empty namespace
    and reports ``Cannot find device``. Compare namespace inodes to close that
    race before attaching the veth.
    """

    parent_inode = _network_namespace_inode("/proc/self/ns/net")
    deadline = time.monotonic() + float(timeout)
    target_path = f"/proc/{process.pid}/ns/net"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            stderr = ""
            if process.stderr is not None:
                stderr = process.stderr.read().strip()
            detail = f": {stderr}" if stderr else ""
            raise RuntimeError(f"MiniSandbox network namespace preflight exited with code {return_code}{detail}")
        try:
            if _network_namespace_inode(target_path) != parent_inode:
                return
        except (FileNotFoundError, ProcessLookupError):
            pass
        time.sleep(0.01)
    raise TimeoutError(f"MiniSandbox network namespace preflight did not isolate PID {process.pid} within {timeout:g}s")


def _oci_layout_manifest_digest(layout: Path) -> str:
    index = _load_object(layout / "index.json")
    manifests = index.get("manifests")
    if not isinstance(manifests, list) or len(manifests) != 1:
        raise RuntimeError(f"local OCI layout must contain one image: {layout}")
    descriptor = manifests[0]
    if not isinstance(descriptor, dict):
        raise RuntimeError(f"local OCI layout has an invalid image descriptor: {layout}")
    digest = str(descriptor.get("digest") or "")
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
        raise RuntimeError(f"local OCI layout has an invalid manifest digest: {layout}")
    return digest


def _local_oci_copy_command(
    *,
    source_layout: Path,
    destination_layout: Path,
    shared_blob_dir: Path,
) -> list[str]:
    """Convert a pinned Docker-schema OCI layout for umoci consumption.

    The source layout contains exactly one descriptor and is validated against
    the materialization record before this command runs. Omitting its optional
    ref-name supports the older Skopeo in the training image. Digest
    preservation is intentionally not requested: OCI output conversion may
    rewrite a Docker schema-2 manifest while preserving its config and layers,
    and umoci needs the converted OCI media types.
    """

    return [
        "skopeo",
        "copy",
        "--src-shared-blob-dir",
        str(shared_blob_dir),
        f"oci:{source_layout}",
        f"oci:{destination_layout}:image",
    ]


def set_active_minisandbox_cluster(cluster: MiniSandboxCluster | None) -> None:
    global _ACTIVE_CLUSTER
    with _ACTIVE_CLUSTER_LOCK:
        if cluster is not None and _ACTIVE_CLUSTER not in (None, cluster):
            raise RuntimeError("a MiniSandbox cluster is already active in this process")
        _ACTIVE_CLUSTER = cluster


def get_active_minisandbox_cluster() -> MiniSandboxCluster:
    with _ACTIVE_CLUSTER_LOCK:
        cluster = _ACTIVE_CLUSTER
    if cluster is None:
        raise RuntimeError("MiniSandbox node services are not initialized; training must start them after VERL worker placement")
    return cluster


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read MiniSandbox JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"MiniSandbox JSON file is not an object: {path}")
    return value


class MiniSandboxNodeService:
    """Privileged implementation hosted by one Ray actor on one GPU node."""

    def __init__(
        self,
        *,
        run_id: str,
        shared_cache_dir: str,
        local_root: str,
        blocked_addresses: list[str] | None = None,
        diagnostics_dir: str | None = None,
    ):
        import ray
        from rllm.sandbox.minisandbox_runtime.oci_runtime import (
            BridgeNetworkManager,
            CgroupV1Manager,
            OciRootfsRuntime,
            check_runtime_requirements,
        )

        # Resolve actor placement before creating any host network/cgroup
        # resources, so an incompatible Ray API fails without leaving node
        # state behind.
        self.node_id, self.node_ip = _ray_node_identity(ray)

        self.run_id = _safe_component(run_id, label="run id")
        self.shared_cache = Path(shared_cache_dir).expanduser().resolve()
        self.local_root = Path(local_root).expanduser().resolve()
        check_runtime_requirements(self.local_root)
        free_mb = shutil.disk_usage(self.local_root).free // 1024 // 1024
        if free_mb < 32 * 1024:
            raise RuntimeError(f"MiniSandbox local_root preflight requires at least 32 GiB free, found {free_mb} MiB at {self.local_root}")
        self.cache_root = self.local_root / "cache"
        self.run_root = self.local_root / "runs" / self.run_id
        self.sessions_root = self.run_root / "sessions"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        self._clean_stale_sessions()

        manifest = _load_object(self.shared_cache / "materialization.json")
        if manifest.get("status") != "complete":
            raise RuntimeError(f"MiniSandbox cache is incomplete: {self.shared_cache}")
        raw_records = manifest.get("records")
        if not isinstance(raw_records, dict):
            raise RuntimeError("MiniSandbox cache manifest has no record index")
        self.records = {str(key): str(value) for key, value in raw_records.items()}
        self._cgroups = CgroupV1Manager(self.run_id)
        blocked = list(blocked_addresses or [])
        kubernetes_host = os.environ.get("KUBERNETES_SERVICE_HOST")
        if kubernetes_host:
            blocked.append(kubernetes_host)
        self._network = BridgeNetworkManager(self.run_id, tuple(dict.fromkeys(blocked)))
        probe_process = None
        probe_network = None
        try:
            probe_cgroup = self._cgroups.create(f"preflight-{uuid.uuid4().hex[:8]}", cpus=1, memory_mb=256)
            probe_cgroup.close()
            self._network.start()
            probe_process = subprocess.Popen(
                ["unshare", "--net", "--", "sleep", "30"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            _wait_for_isolated_network_namespace(probe_process)
            probe_network = self._network.attach(f"preflight-{uuid.uuid4().hex[:8]}", probe_process.pid)
            route = subprocess.run(
                [
                    "nsenter",
                    "--target",
                    str(probe_process.pid),
                    "--net",
                    "ip",
                    "route",
                    "get",
                    "1.1.1.1",
                ],
                capture_output=True,
                text=True,
            )
            if route.returncode != 0:
                raise RuntimeError("MiniSandbox veth/NAT preflight failed: " + (route.stderr or route.stdout).strip())
        except BaseException:
            self._network.release(probe_network)
            if probe_process is not None:
                probe_process.terminate()
                try:
                    probe_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    probe_process.kill()
            try:
                self._network.close()
            except Exception:
                logger.exception("failed to clean MiniSandbox preflight network")
            self._cgroups.close()
            raise
        else:
            self._network.release(probe_network)
            probe_process.terminate()
            try:
                probe_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                probe_process.kill()
        self._runtime_type = OciRootfsRuntime
        self._sandboxes: dict[str, Any] = {}
        self._resources: dict[str, tuple[int, int]] = {}
        self._sandbox_digests: dict[str, str] = {}
        self._sandbox_cache_locks: dict[str, Any] = {}
        self._active_digests: dict[str, int] = {}
        self._cache_lock = threading.RLock()
        self._unpack_semaphore = threading.Semaphore(4)
        self._lock = threading.RLock()
        self._closed = False
        self._executor = ThreadPoolExecutor(
            max_workers=min(512, max(128, os.cpu_count() or 128)),
            thread_name_prefix="minisandbox-node",
        )
        asyncio.get_event_loop().set_default_executor(self._executor)
        # Cold OCI builds must not occupy the command/cleanup thread pool
        # while waiting for the four existing unpack slots.
        self._create_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="minisandbox-create")
        self._cleanup_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="minisandbox-cleanup")
        self._monitor = None
        self._monitor_error = None
        try:
            from rllm.utils.node_diagnostics import NodeMonitor, monitor_instance_paths
            monitor_local, monitor_archive = monitor_instance_paths(
                self.local_root / "diagnostics" / self.run_id,
                Path(diagnostics_dir) / self.node_id if diagnostics_dir else None,
            )
            self._monitor = NodeMonitor(
                node_id=self.node_id, local_path=monitor_local,
                host=self.node_ip,
                archive_path=monitor_archive,
                cgroups={key: value for key, value in self._cgroups.parents.items() if key in {"memory", "cpu", "pids"}},
            )
        except Exception as exc:
            self._monitor_error = f"{type(exc).__name__}: {exc}"
            logger.exception("MiniSandbox independent monitor could not start")
        from rllm.sandbox.backends.session_gc import SessionGarbageCollector
        self._session_gc = SessionGarbageCollector(self.local_root / "garbage")

    def _monitor_event(self, event, **details):
        monitor = getattr(self, "_monitor", None)
        if monitor is not None:
            monitor.emit(event, emitted_at=time.time(), emitted_monotonic=time.monotonic(),
                         emitter_pid=os.getpid(), emitter_thread=threading.get_ident(), **details)

    def _monitor_reference(self):
        monitor = getattr(self, "_monitor", None)
        return monitor.describe() if monitor is not None else {"collection_error": getattr(self, "_monitor_error", "not configured")}

    def _start_monitor_heartbeat(self):
        if getattr(self, "_monitor", None) is None or getattr(self, "_monitor_heartbeat", None) is not None:
            return
        loop = asyncio.get_running_loop()
        register = getattr(self._monitor, "set_close_handler", None)
        if callable(register):
            register(lambda sandbox_id: loop.call_soon_threadsafe(self._deliver_independent_close, sandbox_id))
        async def heartbeat():
            expected = time.monotonic()
            while True:
                now = time.monotonic()
                self._monitor_event("heartbeat", loop_lag_seconds=max(0.0, now - expected),
                                    create_pending=sum(not f.done() for f in getattr(self, "_create_operations", {}).values()),
                                    cleanup_pending=sum(not f.done() for f in getattr(self, "_cleanup_operations", {}).values()),
                                    command_pending=sum(not record[2].done() for record in getattr(self, "_command_operations", {}).values()),
                                    command_results=len(getattr(self, "_command_operations", {})))
                expected = now + 5
                await asyncio.sleep(5)
        self._monitor_heartbeat = asyncio.create_task(heartbeat())

    def _deliver_independent_close(self, sandbox_id: str) -> None:
        if sandbox_id not in getattr(self, "_sandboxes", {}) and sandbox_id not in getattr(self, "_cleanup_operations", {}):
            self._monitor_event("close_delivery_rejected", operation_id=sandbox_id)
            return
        requests = self.__dict__.setdefault("_independent_close_requests", {})
        if sandbox_id in requests:
            return

        async def deliver():
            try:
                await self.submit_close(sandbox_id)
                original = self._cleanup_operations[sandbox_id]
                if original.done():
                    succeeded = not original.cancelled() and original.exception() is None
                    self._monitor_event("close_terminal", operation_id=sandbox_id,
                                        state="succeeded" if succeeded else "failed", cleanup_confirmed=succeeded,
                                        error=None if succeeded else ("cancelled" if original.cancelled() else str(original.exception())[:2000]))
            finally:
                requests.pop(sandbox_id, None)

        task = asyncio.create_task(deliver())
        requests[sandbox_id] = task
        task.add_done_callback(self._observe_operation)

    def _clean_stale_sessions(self) -> None:
        for session in self.sessions_root.iterdir():
            if not session.is_dir():
                continue
            marker = str(session / "runtime.json").encode("utf-8")
            stale_pids: set[int] = set()
            for process_dir in Path("/proc").iterdir():
                if not process_dir.name.isdigit():
                    continue
                try:
                    command = (process_dir / "cmdline").read_bytes()
                except OSError:
                    continue
                if marker in command and (b"rllm.sandbox.minisandbox_runtime.oci_helper" in command or b"unshare" in command or b"nsenter" in command):
                    stale_pids.add(int(process_dir.name))
            state_path = session / "state.json"
            try:
                state = _load_object(state_path)
                pid = int(state.get("resolved_host_pid") or state.get("host_pid"))
                command = Path(f"/proc/{pid}/cmdline").read_bytes()
            except (OSError, TypeError, ValueError, RuntimeError):
                pid = 0
                command = b""
            if pid > 1 and b"rllm.sandbox.minisandbox_runtime.oci_helper" in command:
                stale_pids.add(pid)
            for stale_pid in sorted(stale_pids, reverse=True):
                try:
                    os.kill(stale_pid, 15)
                except ProcessLookupError:
                    pass
            for stale_pid in sorted(stale_pids, reverse=True):
                for _ in range(100):
                    if not Path(f"/proc/{stale_pid}").exists():
                        break
                    time.sleep(0.05)
                if Path(f"/proc/{stale_pid}").exists():
                    try:
                        os.kill(stale_pid, 9)
                    except ProcessLookupError:
                        pass
            shutil.rmtree(session, ignore_errors=True)

    def _record(self, task_id: str, image: str) -> dict[str, Any]:
        relative = self.records.get(task_id)
        if relative is None:
            raise RuntimeError(f"{task_id}: no MiniSandbox OCI cache record")
        record_path = (self.shared_cache / relative).resolve()
        try:
            record_path.relative_to(self.shared_cache)
        except ValueError as exc:
            raise RuntimeError(f"{task_id}: unsafe MiniSandbox cache record") from exc
        record = _load_object(record_path)
        if record.get("status") != "complete" or record.get("task_id") != task_id or record.get("original_image") != image or record.get("platform") != {"os": "linux", "architecture": "amd64"}:
            raise RuntimeError(f"{task_id}: MiniSandbox OCI record drift detected")
        source_layout = (self.shared_cache / str(record.get("layout") or "")).resolve()
        try:
            source_layout.relative_to(self.shared_cache)
        except ValueError as exc:
            raise RuntimeError(f"{task_id}: unsafe MiniSandbox OCI layout") from exc
        return record

    def _unpacked_image(
        self,
        record: Mapping[str, Any],
        active_lock: Any,
        required_mb: int,
        control: _CreateControl | None = None,
    ) -> tuple[Path, dict[str, Any]]:
        digest = str(record.get("manifest_digest") or "")
        if not re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
            raise RuntimeError(f"invalid MiniSandbox OCI digest: {digest!r}")
        key = digest.split(":", 1)[1]
        destination = self.cache_root / key
        complete = destination / ".complete.json"

        def valid() -> bool:
            metadata = _load_object(complete) if complete.is_file() else None
            return bool(
                isinstance(metadata, dict)
                and metadata.get("manifest_digest") == digest
                and metadata.get("unpack_revision") == MINISANDBOX_UNPACK_REVISION
                and (destination / "rootfs").is_dir()
                and (destination / "config.json").is_file()
            )

        build_lock_path = self.cache_root / f".{key}.build.lock"
        def acquire(handle, mode):
            if control is None:
                fcntl.flock(handle.fileno(), mode)
                return
            control.stage = "cache_lock"
            while True:
                control.check()
                try:
                    fcntl.flock(handle.fileno(), mode | fcntl.LOCK_NB)
                    return
                except BlockingIOError:
                    control.cancelled.wait(0.1)

        with build_lock_path.open("a+b") as build_lock:
            acquire(build_lock, fcntl.LOCK_EX)
            if valid():
                fcntl.flock(active_lock.fileno(), fcntl.LOCK_SH)
                with self._cache_lock:
                    self._ensure_cache_space(required_mb, key)
                os.utime(complete, None)
                return destination / "rootfs", _load_object(destination / "config.json")
            acquire(active_lock, fcntl.LOCK_EX)
            with self._cache_lock:
                self._ensure_cache_space(required_mb, key)
            if valid():
                fcntl.flock(active_lock.fileno(), fcntl.LOCK_SH)
                os.utime(complete, None)
                return destination / "rootfs", _load_object(destination / "config.json")
            source_layout = self.shared_cache / str(record.get("layout") or "")
            source_digest = _oci_layout_manifest_digest(source_layout)
            if source_digest != digest:
                raise RuntimeError(f"shared OCI layout does not contain the pinned manifest digest: expected={digest} actual={source_digest} layout={source_layout}")
            staging = self.cache_root / f".{key}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            shutil.rmtree(staging, ignore_errors=True)
            # Skopeo creates the OCI layout itself, but requires its parent to
            # exist.  The per-build staging directory was previously removed
            # above and never recreated, so every cold-cache conversion failed
            # before any image data was copied.
            staging.mkdir()
            layout = staging / "layout"
            bundle = staging / "bundle"
            command = _local_oci_copy_command(
                source_layout=source_layout,
                destination_layout=layout,
                shared_blob_dir=self.shared_cache / "blobs",
            )
            try:
                if control is not None:
                    control.stage = "oci_copy"
                copied = _build_command(command, control)
                if copied.returncode != 0:
                    raise RuntimeError("local OCI conversion failed: " + (copied.stderr or copied.stdout).strip()[-2000:])
                local_digest = _oci_layout_manifest_digest(layout)
                if control is not None:
                    control.stage = "oci_unpack"
                unpacked = _build_command(
                    [
                        "umoci",
                        "unpack",
                        "--keep-dirlinks",
                        "--image",
                        f"{layout}:image",
                        str(bundle),
                    ],
                    control,
                )
                if unpacked.returncode != 0:
                    raise RuntimeError("local OCI unpack failed: " + (unpacked.stderr or unpacked.stdout).strip()[-2000:])
                if not (bundle / "rootfs").is_dir() or not (bundle / "config.json").is_file():
                    raise RuntimeError("umoci produced an incomplete OCI bundle")
                if destination.exists():
                    shutil.rmtree(destination)
                os.replace(bundle, destination)
                complete_value = {
                    "schema_version": 1,
                    "unpack_revision": MINISANDBOX_UNPACK_REVISION,
                    "manifest_digest": digest,
                    "local_manifest_digest": local_digest,
                    "platform": {"os": "linux", "architecture": "amd64"},
                }
                temporary = destination / f".complete.{os.getpid()}.tmp"
                temporary.write_text(
                    json.dumps(complete_value, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary, complete)
            finally:
                shutil.rmtree(staging, ignore_errors=True)
            fcntl.flock(active_lock.fileno(), fcntl.LOCK_SH)
            return destination / "rootfs", _load_object(destination / "config.json")

    def _ensure_cache_space(self, required_mb: int, protected_digest: str) -> None:
        required = max(32 * 1024, int(required_mb) + 8 * 1024) * 1024 * 1024
        candidates: list[tuple[float, Path]] = []
        for path in self.cache_root.iterdir():
            marker = path / ".complete.json"
            if not path.is_dir() or not re.fullmatch(r"[a-f0-9]{64}", path.name) or not marker.is_file() or path.name == protected_digest or self._active_digests.get(path.name, 0) > 0:
                continue
            candidates.append((marker.stat().st_mtime, path))
        for _last_used, path in sorted(candidates):
            if shutil.disk_usage(self.local_root).free >= required:
                break
            active_lock = self.cache_root / f".{path.name}.active.lock"
            build_lock = self.cache_root / f".{path.name}.build.lock"
            build_handle = build_lock.open("a+b")
            active_handle = active_lock.open("a+b")
            try:
                try:
                    fcntl.flock(
                        build_handle.fileno(),
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                    fcntl.flock(
                        active_handle.fileno(),
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                except BlockingIOError:
                    continue
                shutil.rmtree(path, ignore_errors=True)
            finally:
                active_handle.close()
                build_handle.close()
        free = shutil.disk_usage(self.local_root).free
        if free < required:
            raise RuntimeError(f"insufficient local MiniSandbox disk after evicting inactive OCI roots: required_free_mb={required // 1024 // 1024} actual_free_mb={free // 1024 // 1024}")

    @staticmethod
    def _oci_process(bundle_config: Mapping[str, Any]) -> tuple[dict[str, str], str]:
        process = bundle_config.get("process")
        process = process if isinstance(process, Mapping) else {}
        environment: dict[str, str] = {}
        raw_environment = process.get("env")
        if isinstance(raw_environment, list):
            for raw in raw_environment:
                if isinstance(raw, str) and "=" in raw:
                    key, value = raw.split("=", 1)
                    environment[key] = value
        cwd = str(process.get("cwd") or "/")
        return environment, cwd

    def _create_sync(
        self,
        *,
        name: str,
        task_id: str,
        image: str,
        cpus: int,
        memory_mb: int,
        storage_mb: int | None,
        working_dir: str | None,
        environment: Mapping[str, str] | None,
        allow_internet: bool,
        _control: _CreateControl | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("MiniSandbox node service is closed")
        collector = getattr(self, "_session_gc", None)
        if collector is not None:
            collector.check_admission()
        record = self._record(task_id, image)
        digest_key = str(record["manifest_digest"]).split(":", 1)[1]
        cache_lock = (self.cache_root / f".{digest_key}.active.lock").open("a+b")
        try:
            with self._unpack_semaphore:
                if _control is not None:
                    _control.check()
                rootfs, config = self._unpacked_image(
                    record,
                    cache_lock,
                    int(storage_mb or 0),
                    **({"control": _control} if _control is not None else {}),
                )
            if _control is not None:
                _control.check()
            with self._cache_lock:
                self._active_digests[digest_key] = self._active_digests.get(digest_key, 0) + 1
        except BaseException as exc:
            cache_lock.close()
            exc.minisandbox_cleanup_confirmed = True
            raise
        image_environment, image_cwd = self._oci_process(config)
        image_environment.update({str(key): str(value) for key, value in (environment or {}).items()})
        # MiniSandbox uses direct, firewall-filtered public egress. Remove
        # private corporate proxy endpoints even if an image baked them into
        # its OCI process environment; NO_PROXY remains harmless and useful
        # for loopback traffic.
        for key in tuple(image_environment):
            if key.casefold() in {"http_proxy", "https_proxy", "all_proxy"}:
                image_environment.pop(key, None)
        sandbox_id = f"{_safe_component(name, label='name')}-{uuid.uuid4().hex[:10]}"
        runtime = None
        try:
            if _control is not None:
                _control.stage = "runtime_start"
                _control.check()
            runtime = self._runtime_type(
                sandbox_id=sandbox_id,
                base_rootfs=rootfs,
                session_root=self.sessions_root / sandbox_id,
                cpus=int(cpus),
                memory_mb=int(memory_mb),
                working_dir=str(working_dir or image_cwd),
                environment=image_environment,
                allow_internet=bool(allow_internet),
                cgroups=self._cgroups,
                network=self._network,
            )
            runtime.start()
            # Some benchmark images contain build-time test artifacts.  Both
            # primary and verifier sandboxes begin without /tests; the existing
            # verifier path uploads the hidden suite only to its own sandbox.
            runtime.exec("rm -rf -- /tests", timeout=60, user="root")
        except BaseException as exc:
            if runtime is not None:
                with self._lock:
                    self._sandboxes[sandbox_id] = runtime
                    self._resources[sandbox_id] = (int(cpus), int(memory_mb))
                    self._sandbox_digests[sandbox_id] = digest_key
                    self._sandbox_cache_locks[sandbox_id] = cache_lock
                try:
                    self._close_sync(sandbox_id)
                except BaseException as cleanup_error:
                    # Retain the object and cache pin for shutdown reconciliation.
                    cleanup_error.minisandbox_cleanup_confirmed = False
                    raise cleanup_error from exc
            else:
                with self._cache_lock:
                    remaining = self._active_digests.get(digest_key, 1) - 1
                    if remaining > 0:
                        self._active_digests[digest_key] = remaining
                    else:
                        self._active_digests.pop(digest_key, None)
                cache_lock.close()
                self._assert_preflight_cleanup(sandbox_id)
            exc.minisandbox_cleanup_confirmed = True
            raise
        with self._lock:
            self._sandboxes[sandbox_id] = runtime
            self._resources[sandbox_id] = (int(cpus), int(memory_mb))
            self._sandbox_digests[sandbox_id] = digest_key
            self._sandbox_cache_locks[sandbox_id] = cache_lock
        return {
            "sandbox_id": sandbox_id,
            "node_id": self.node_id,
            "node_ip": self.node_ip,
            "manifest_digest": record["manifest_digest"],
        }

    def _init_create_operations(self) -> None:
        if not hasattr(self, "_create_operations"):
            self._create_operations = {}
            self._create_controls = {}
            self._cancelled_creates = set()
            self._create_slots = asyncio.Semaphore(4)
            self._create_progress = 0
            self._cancel_operations = {}

    def _init_cleanup_operations(self) -> None:
        if not hasattr(self, "_cleanup_operations"):
            self._cleanup_operations = {}
            self._cleanup_controls = {}
            self._cleanup_slots = asyncio.Semaphore(8)
            self._cleanup_progress = 0
            self._cleanup_evidence = {}
            self._cleanup_probes = {}
            self._diagnostic_executor = ThreadPoolExecutor(2, thread_name_prefix="minisandbox-diagnostics")

    @staticmethod
    def _observe_operation(future) -> None:
        if not future.cancelled():
            future.exception()  # Retain the error in the operation, without orphan warnings.

    @staticmethod
    def _future_status(future) -> dict:
        if future is None:
            return {"state": "unknown"}
        if future.cancelled():
            return {"state": "cancelled", "error": {"message": "queued operation cancelled", "cleanup_confirmed": True}}
        if not future.done():
            return {"state": "running"}
        error = future.exception()
        if error is not None:
            return {"state": "failed", "error": {
                "type": type(error).__name__, "message": str(error)[:4000],
                "diagnostics": _exception_diagnostics(error),
                "cleanup_confirmed": getattr(error, "minisandbox_cleanup_confirmed", False),
            }}
        return {"state": "succeeded", "result": future.result()}

    async def submit_close(self, sandbox_id: str) -> dict:
        """Coalesce closes before they enter the FIFO or consume a worker."""
        self._monitor_event("close_rpc_enter", operation_id=sandbox_id)
        self._init_cleanup_operations()
        self._start_monitor_heartbeat()
        if sandbox_id not in self._cleanup_operations:
            control = _CreateControl(55.0, queued=True)
            self._cleanup_controls[sandbox_id] = control
            self._monitor_event("start", operation_id=sandbox_id, operation="close")

            async def run():
                async with self._cleanup_slots:
                    control.begin()
                    control.stage = "sandbox_cleanup"
                    self._monitor_event("close_slot_acquired", operation_id=sandbox_id,
                                        queue_seconds=control.started_at - control.queued_at)
                    try:
                        for attempt in range(2):
                            try:
                                await asyncio.get_running_loop().run_in_executor(
                                    getattr(self, "_cleanup_executor", None), self._close_sync, sandbox_id,
                                )
                                break
                            except Exception:
                                if attempt or time.monotonic() >= control.deadline:
                                    raise
                        for operation, task in list(getattr(self, "_create_operations", {}).items()):
                            if task.done() and not task.cancelled() and task.exception() is None:
                                if task.result().get("sandbox_id") == sandbox_id:
                                    self._create_operations.pop(operation, None)
                                    self._create_controls.pop(operation, None)
                        commands = getattr(self, "_command_operations", {})
                        for request_id, (owner, _method, command) in list(commands.items()):
                            if owner == sandbox_id and command.done():
                                commands.pop(request_id, None)
                    finally:
                        control.finished_at = time.monotonic()
                        self._cleanup_progress += 1
                        self._monitor_event("finish", operation_id=sandbox_id, operation="close")
                # Completed tombstones contain no runtime or task contents.
                # Never evict a running operation or its resource lease.
                if len(self._cleanup_operations) > 10000:
                    for key, task in list(self._cleanup_operations.items()):
                        if key != sandbox_id and task.done() and not task.cancelled() and task.exception() is None:
                            self._cleanup_operations.pop(key, None)
                            self._cleanup_controls.pop(key, None)
                            self._cleanup_evidence.pop(key, None)
                            self._cleanup_probes.pop(key, None)
                            if len(self._cleanup_operations) <= 10000:
                                break

            future = asyncio.create_task(run())
            future.add_done_callback(self._observe_operation)
            def record_failure(done):
                succeeded = not done.cancelled() and done.exception() is None
                self._monitor_event("close_terminal", operation_id=sandbox_id,
                                    state="succeeded" if succeeded else "failed",
                                    cleanup_confirmed=succeeded,
                                    error=None if succeeded else ("cancelled" if done.cancelled() else str(done.exception())[-2000:]))
                if not done.cancelled() and done.exception() is not None:
                    self._monitor_event("failure", operation_id=sandbox_id, operation="close", error=str(done.exception())[-2000:])
            future.add_done_callback(record_failure)
            self._cleanup_operations[sandbox_id] = future
        self._monitor_event("close_rpc_return", operation_id=sandbox_id)
        return {"operation_id": sandbox_id}

    async def cleanup_status(self, sandbox_id: str) -> dict:
        self._monitor_event("close_status_enter", operation_id=sandbox_id)
        self._init_cleanup_operations()
        control = self._cleanup_controls.get(sandbox_id)
        status = self._future_status(self._cleanup_operations.get(sandbox_id))
        # Kernel/cgroup reads run outside the control event loop and the busy
        # command/cleanup pools. A stuck close remains diagnosable by polling.
        previous = self._cleanup_probes.get(sandbox_id)
        evidence = self._cleanup_evidence.get(sandbox_id, {})
        if (control is not None and control.started and status["state"] == "running"
                and time.monotonic() - control.started_at >= 5
                and (previous is None or previous.done())
                and time.monotonic() - evidence.get("sampled_at_monotonic", 0) >= 5):
            probe = asyncio.get_running_loop().run_in_executor(
                self._diagnostic_executor, self._cleanup_runtime_evidence, sandbox_id,
            )
            self._cleanup_probes[sandbox_id] = probe
            def record(done):
                try:
                    self._cleanup_evidence[sandbox_id] = done.result()
                except BaseException as exc:
                    self._cleanup_evidence[sandbox_id] = {"sampled_at_monotonic": time.monotonic(), "error": str(exc)[-1000:]}
            probe.add_done_callback(record)
        self._monitor_event("close_status_return", operation_id=sandbox_id, state=status.get("state"))
        return {**self._control_status(sandbox_id, control, self._cleanup_progress), **status,
                "runtime_diagnostics": evidence,
                "cleanup_confirmed": status["state"] == "succeeded"}

    def _cleanup_runtime_evidence(self, sandbox_id: str) -> dict:
        runtime = getattr(self, "_sandboxes", {}).get(sandbox_id)
        collect = getattr(runtime, "_cleanup_diagnostics", None)
        evidence = collect("cleanup_wait") if callable(collect) else {}
        return {**evidence, "sampled_at_monotonic": time.monotonic()}

    async def _cleanup_sandbox(self, sandbox_id: str) -> None:
        await self.submit_close(sandbox_id)
        await asyncio.shield(self._cleanup_operations[sandbox_id])

    async def submit_create(self, operation_id: str | None = None, operation_timeout: float = 1800.0, **kwargs) -> dict[str, Any]:
        self._start_monitor_heartbeat()
        if getattr(self, "_closing", False) or getattr(self, "_closed", False):
            error = RuntimeError("MiniSandbox node service is closed")
            error.minisandbox_cleanup_confirmed = True
            raise error
        self._init_create_operations()
        operation_id = operation_id or uuid.uuid4().hex
        if operation_id in self._cancelled_creates:
            error = RuntimeError("MiniSandbox create operation was cancelled")
            error.minisandbox_cleanup_confirmed = True
            raise error
        task = self._create_operations.get(operation_id)
        if task is None:
            control = _CreateControl(operation_timeout, queued=True)
            self._create_controls[operation_id] = control
            self._monitor_event("start", operation_id=operation_id, operation="create")

            async def run():
                # Waiting for a creation slot consumes no worker thread, and
                # cancellation can confirm that no runtime was ever started.
                async with self._create_slots:
                    control.begin()
                    try:
                        return await asyncio.get_running_loop().run_in_executor(
                            getattr(self, "_create_executor", None),
                            partial(self._create_sync, _control=control, **kwargs),
                        )
                    finally:
                        control.finished_at = time.monotonic()
                        self._create_progress += 1
                        self._monitor_event("finish", operation_id=operation_id, operation="create")

            task = asyncio.create_task(run())
            task.add_done_callback(self._observe_operation)
            self._create_operations[operation_id] = task
        return {"operation_id": operation_id}

    async def create(self, operation_id: str | None = None, operation_timeout: float = 1800.0, **kwargs) -> dict[str, Any]:
        # Compatibility API for local callers. Production Ray clients use
        # submit_create and operation_status, leaving control RPCs available.
        operation_id = operation_id or uuid.uuid4().hex
        await self.submit_create(operation_id, operation_timeout, **kwargs)
        task = self._create_operations[operation_id]
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            if operation_id in self._cancelled_creates:
                error = RuntimeError("MiniSandbox queued create was cancelled")
                error.minisandbox_cleanup_confirmed = True
                raise error from None
            raise
        if operation_id in self._cancelled_creates:
            await self._cleanup_sandbox(result["sandbox_id"])
            raise RuntimeError("MiniSandbox late create was reclaimed")
        return result

    async def request_cancel_create(self, operation_id: str) -> dict:
        self._init_create_operations()
        if operation_id not in self._cancel_operations:
            task = asyncio.create_task(self.cancel_create(operation_id))
            task.add_done_callback(self._observe_operation)
            self._cancel_operations[operation_id] = task
        return {"operation_id": operation_id}

    async def cancel_create(self, operation_id: str) -> bool:
        self._init_create_operations()
        self._cancelled_creates.add(operation_id)
        task = self._create_operations.get(operation_id)
        control = self._create_controls.get(operation_id)
        if control is not None:
            control.cancelled.set()
        if task is not None:
            if control is not None and not control.started:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                self._create_operations.pop(operation_id, None)
                self._create_controls.pop(operation_id, None)
                return True
            try:
                result = await asyncio.shield(task)
            except Exception as exc:
                self._create_operations.pop(operation_id, None)
                self._create_controls.pop(operation_id, None)
                return bool(getattr(exc, "minisandbox_cleanup_confirmed", False))
            await self._cleanup_sandbox(result["sandbox_id"])
            self._create_operations.pop(operation_id, None)
            self._create_controls.pop(operation_id, None)
        return True

    async def operation_status(self, operation_id: str) -> dict[str, Any]:
        control = getattr(self, "_create_controls", {}).get(operation_id)
        status = self._future_status(getattr(self, "_create_operations", {}).get(operation_id))
        cancel = self._future_status(getattr(self, "_cancel_operations", {}).get(operation_id))
        result = {**self._control_status(operation_id, control, getattr(self, "_create_progress", 0)), **status,
                  "cancel_state": cancel["state"], "cleanup_confirmed": cancel.get("result") is True}
        if cancel["state"] == "failed":
            result["error"] = cancel["error"]
        if status["state"] == "succeeded" and cancel["state"] == "running":
            result["cleanup_status"] = await self.cleanup_status(status["result"]["sandbox_id"])
        # While cancelling a create that has not returned, bound confirmation
        # by the caller's 60s no-progress deadline, not its expired create budget.
        if cancel["state"] == "running":
            result["started"] = False
        return result

    async def lifecycle_diagnostics(self, operation_id: str, kind: str) -> dict:
        """Read-only fallback on a separate event loop; no pool, lock or I/O.

        Do not call cleanup_status/operation_status here: they own mutable
        asyncio state on the control loop. Completed futures and previously
        sampled evidence can be inspected without scheduling more work there.
        """
        def snapshot(identifier, *, cleanup=False):
            prefix = "_cleanup" if cleanup else "_create"
            control = getattr(self, prefix + "_controls", {}).get(identifier)
            state = self._future_status(getattr(self, prefix + "_operations", {}).get(identifier))
            result = {**self._control_status(identifier, control, getattr(self, prefix + "_progress", 0)), **state}
            if cleanup:
                result.update(cleanup_confirmed=state["state"] == "succeeded",
                              runtime_diagnostics=getattr(self, "_cleanup_evidence", {}).get(identifier, {}))
            return result

        status = snapshot(operation_id, cleanup=kind == "close")
        if kind == "cancel":
            cancel = self._future_status(getattr(self, "_cancel_operations", {}).get(operation_id))
            status.update(cancel_state=cancel["state"], cleanup_confirmed=cancel.get("result") is True)
            if cancel["state"] == "failed":
                status["error"] = cancel["error"]
            if cancel["state"] == "running":
                if status["state"] == "succeeded":
                    status["cleanup_status"] = snapshot(status["result"]["sandbox_id"], cleanup=True)
                status["started"] = False
        # Frame names/locations only: never include locals, commands or tokens.
        threads = []
        names = {thread.ident: thread.name for thread in threading.enumerate()}
        for ident, frame in sys._current_frames().items():
            stack = []
            relevant = False
            while frame is not None:
                filename = frame.f_code.co_filename
                relevant |= filename.endswith(("minisandbox.py", "oci_runtime.py"))
                if len(stack) < 12:
                    stack.append({"file": filename, "line": frame.f_lineno, "function": frame.f_code.co_name})
                frame = frame.f_back
            if relevant and len(threads) < 32:
                threads.append({"thread_id": ident, "name": names.get(ident), "stack": stack})
        return {"node_id": self.node_id, "operation_id": operation_id,
                "operation_status": status, "threads": threads, "node_monitor": self._monitor_reference()}

    def _control_status(self, operation_id, control, progress) -> dict:
        return {
            "node_id": self.node_id, "operation_id": operation_id,
            "stage": control.stage if control is not None else "unknown",
            "started": control.started if control is not None else None,
            "cancelled": control.cancelled.is_set() if control is not None else None,
            "queue_progress": progress,
            "queue_seconds": ((control.started_at if control.started_at is not None else time.monotonic()) - control.queued_at) if control is not None else None,
            "execution_seconds": (control.finished_at if control.finished_at is not None else time.monotonic()) - control.started_at if control is not None and control.started_at is not None else None,
            "finished_at_monotonic": control.finished_at if control is not None else None,
            "remaining_seconds": max(0.0, control.deadline - time.monotonic()) if control is not None and control.deadline is not None else None,
        }

    def _assert_preflight_runtime_isolation(
        self,
        runtime: Any,
    ) -> tuple[str, tuple[Path, ...]]:
        """Validate namespaces, cgroup membership, veth, and egress guards."""

        try:
            host_pid = int(runtime._host_pid)
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "MiniSandbox real-image preflight has no namespace init PID"
            ) from exc
        if host_pid <= 1 or not Path(f"/proc/{host_pid}").is_dir():
            raise RuntimeError(
                "MiniSandbox real-image preflight namespace process is absent"
            )
        for namespace in ("mnt", "pid", "net", "uts", "ipc", "cgroup"):
            host_namespace = Path(f"/proc/self/ns/{namespace}")
            child_namespace = Path(f"/proc/{host_pid}/ns/{namespace}")
            try:
                isolated = host_namespace.stat().st_ino != child_namespace.stat().st_ino
            except OSError as exc:
                raise RuntimeError(
                    f"MiniSandbox cannot inspect {namespace} namespace isolation"
                ) from exc
            if not isolated:
                raise RuntimeError(
                    f"MiniSandbox real-image preflight did not isolate {namespace} namespace"
                )

        lease = getattr(runtime, "_network_lease", None)
        host_veth = str(getattr(lease, "host_veth", "") or "")
        if not host_veth or not (Path("/sys/class/net") / host_veth).exists():
            raise RuntimeError(
                "MiniSandbox real-image preflight did not create its veth"
            )
        namespace_prefix = [
            "nsenter",
            "--target",
            str(host_pid),
            "--net",
            "iptables",
            "-C",
            "OUTPUT",
        ]
        blocked_destinations = [
            "10.0.0.0/8",
            "100.64.0.0/10",
            "169.254.0.0/16",
            "172.16.0.0/12",
            "192.168.0.0/16",
        ]
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+){3}", self.node_ip):
            blocked_destinations.append(self.node_ip)
        for destination in blocked_destinations:
            checked = subprocess.run(
                [*namespace_prefix, "-d", destination, "-j", "REJECT"],
                capture_output=True,
                text=True,
            )
            if checked.returncode != 0:
                raise RuntimeError(
                    "MiniSandbox real-image preflight is missing a private-egress "
                    f"guard for {destination}: "
                    + (checked.stderr or checked.stdout).strip()[-1000:]
                )

        cgroup = getattr(runtime, "_cgroup", None)
        raw_paths = getattr(cgroup, "paths", None)
        if not isinstance(raw_paths, Mapping) or not raw_paths:
            raise RuntimeError(
                "MiniSandbox real-image preflight has no cgroup lease"
            )
        cgroup_paths = tuple(
            sorted(
                {Path(value) for value in raw_paths.values()},
                key=str,
            )
        )
        missing = [str(path) for path in cgroup_paths if not path.is_dir()]
        if missing:
            raise RuntimeError(
                "MiniSandbox real-image preflight cgroup is missing: "
                + ", ".join(missing)
            )
        return host_veth, cgroup_paths

    def _preflight_task_sync(self, task_id: str, image: str) -> dict[str, Any]:
        """Exercise the complete image-to-runtime path before rollout fan-out."""

        collector = getattr(self, "_session_gc", None)
        if collector is not None:
            collector.wait_for_admission()
        record = self._record(task_id, image)
        security = record.get("security")
        security = security if isinstance(security, Mapping) else {}
        raw_forbidden = security.get("forbidden_paths")
        forbidden_paths = (
            [str(value) for value in raw_forbidden]
            if isinstance(raw_forbidden, list)
            else ["/tests"]
        )
        raw_visible = security.get("visible_fixture_paths")
        visible_paths = (
            [str(value) for value in raw_visible]
            if isinstance(raw_visible, list)
            else []
        )
        workdir = record.get("workdir")
        workdir = str(workdir) if isinstance(workdir, str) and workdir else None
        result = self._create_sync(
            name="preflight",
            task_id=task_id,
            image=image,
            cpus=1,
            memory_mb=512,
            storage_mb=None,
            working_dir=workdir,
            environment=None,
            # Exercise the veth and public-egress firewall even when the
            # sampled task itself is offline. No model command is run here.
            allow_internet=True,
        )
        sandbox_id = str(result["sandbox_id"])
        host_veth: str | None = None
        cgroup_paths: tuple[Path, ...] = ()
        try:
            checks = [
                "test -d /proc/self",
                "test -r /proc/self/status",
                "command -v sh >/dev/null",
                "(command -v python >/dev/null && "
                "python -c 'import os; assert os.path.isdir(\"/proc/self\")' "
                "|| python3 -c 'import os; assert os.path.isdir(\"/proc/self\")')",
            ]
            if workdir is not None:
                checked_workdir = _safe_remote_path(workdir, allow_root=True)
                checks.append(f"test \"$PWD\" = {shlex.quote(checked_workdir)}")
            for path in forbidden_paths:
                checked = _safe_remote_path(path, allow_root=False)
                checks.append(f"test ! -e {shlex.quote(checked)}")
            for path in visible_paths:
                checked = _safe_remote_path(path, allow_root=False)
                checks.append(f"test -e {shlex.quote(checked)}")
            checks.append("echo MINISANDBOX_TASK_PREFLIGHT_OK")
            output = self._runtime(sandbox_id).exec(
                " && ".join(checks),
                timeout=60,
                user="root",
            )
            if "MINISANDBOX_TASK_PREFLIGHT_OK" not in str(output):
                raise RuntimeError(f"MiniSandbox task preflight returned unexpected output: {output!r}")
            host_veth, cgroup_paths = self._assert_preflight_runtime_isolation(
                self._runtime(sandbox_id)
            )
            return {
                "node_id": self.node_id,
                "node_ip": self.node_ip,
                "task_id": task_id,
                "manifest_digest": result["manifest_digest"],
            }
        finally:
            self._close_sync(sandbox_id, reclaim_filesystem=True)
            self._assert_preflight_cleanup(
                sandbox_id,
                host_veth=host_veth,
                cgroup_paths=cgroup_paths,
            )

    def _assert_preflight_cleanup(
        self,
        sandbox_id: str,
        *,
        host_veth: str | None = None,
        cgroup_paths: tuple[Path, ...] = (),
        allow_session: bool = False,
    ) -> None:
        """Fail startup if a canary leaves process/mount/veth/cgroup state."""

        session = self.sessions_root / sandbox_id
        marker = str(session).encode("utf-8")
        residue: list[str] = []
        if session.exists() and not allow_session:
            residue.append(f"session={session}")
        try:
            mountinfo = Path("/proc/self/mountinfo").read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError as exc:
            raise RuntimeError("MiniSandbox cleanup cannot verify remaining mounts") from exc
        if str(session) in mountinfo:
            residue.append("mount")
        if host_veth and (Path("/sys/class/net") / host_veth).exists():
            residue.append(f"veth={host_veth}")
        for cgroup_path in cgroup_paths:
            if cgroup_path.exists():
                residue.append(f"cgroup={cgroup_path}")
        for process_dir in Path("/proc").iterdir():
            if not process_dir.name.isdigit():
                continue
            try:
                command = (process_dir / "cmdline").read_bytes()
            except OSError:
                continue
            if marker in command:
                residue.append(f"pid={process_dir.name}")
                break
        if residue:
            raise RuntimeError(
                "MiniSandbox real-image preflight cleanup left residue: "
                + ", ".join(residue)
            )

    async def preflight_task(self, task_id: str, image: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._preflight_task_sync, task_id, image)

    def _runtime(self, sandbox_id: str):
        with self._lock:
            if sandbox_id in getattr(self, "_cleanup_operations", {}):
                from rllm.types import RolloutInfrastructureError
                raise RolloutInfrastructureError(
                    "sandbox_closed", f"MiniSandbox is closing or closed: {sandbox_id}",
                    stage="sandbox_rpc", retryable=True, retry_scope="full_rollout",
                    diagnostics={"sandbox_id": sandbox_id, "execution_started": False},
                )
            runtime = self._sandboxes.get(sandbox_id)
        if runtime is None:
            raise RuntimeError(f"unknown MiniSandbox: {sandbox_id}")
        return runtime

    async def submit_command(self, request_id: str, sandbox_id: str, method: str, args: tuple) -> dict:
        """Acknowledge promptly; command execution must not hold a Ray RPC open."""
        from rllm.types import RolloutInfrastructureError

        if method not in {"exec", "read_files", "write_file", "extract_archive"}:
            raise ValueError(f"unsupported MiniSandbox command method: {method}")
        self._runtime(sandbox_id)
        if not hasattr(self, "_command_operations"):
            self._command_operations = {}
        operations = self._command_operations
        previous = operations.get(request_id)
        if previous is not None:
            if previous[:2] != (sandbox_id, method):
                raise ValueError("MiniSandbox command request identity mismatch")
            return {"request_id": request_id}
        if len(operations) >= 4096:
            raise RolloutInfrastructureError(
                "minisandbox_command_backlog", "MiniSandbox command result backlog is full",
                retryable=True, stage="sandbox_rpc", retry_scope="full_rollout",
                diagnostics={"execution_started": False, "pending_results": len(operations)},
            )

        async def run():
            self._monitor_event("command_worker_enter", operation_id=request_id, sandbox_id=sandbox_id, method=method)
            try:
                return await getattr(self, method)(sandbox_id, *args)
            finally:
                self._monitor_event("command_worker_return", operation_id=request_id, sandbox_id=sandbox_id, method=method)

        task = asyncio.create_task(run())
        operations[request_id] = (sandbox_id, method, task)
        task.add_done_callback(self._observe_operation)

        def finished(done):
            if sandbox_id in getattr(self, "_cleanup_operations", {}):
                operations.pop(request_id, None)

        task.add_done_callback(finished)
        self._monitor_event("command_submitted", operation_id=request_id, sandbox_id=sandbox_id, method=method)
        return {"request_id": request_id}

    async def command_result(self, request_id: str) -> dict:
        record = getattr(self, "_command_operations", {}).get(request_id)
        if record is None:
            from rllm.types import RolloutInfrastructureError
            raise RolloutInfrastructureError(
                "minisandbox_command_result_unavailable", f"unknown or reclaimed MiniSandbox command: {request_id}",
                retryable=True, stage="sandbox_rpc", retry_scope="full_rollout",
                diagnostics={"request_id": request_id, "execution_started": None},
            )
        task = record[2]
        if not task.done():
            return {"request_id": request_id, "ready": False}
        return {"request_id": request_id, "ready": True, "result": task.result()}

    async def acknowledge_command(self, request_id: str) -> None:
        operations = getattr(self, "_command_operations", {})
        record = operations.get(request_id)
        if record is not None and record[2].done():
            operations.pop(request_id, None)

    async def exec(
        self,
        sandbox_id: str,
        command: str,
        timeout: float | None,
        user: str | None,
        trusted_setup: bool = False,
    ) -> str:
        runtime = self._runtime(sandbox_id)
        return await self._run_command_io(runtime.exec, command, timeout=timeout, user=user, trusted_setup=bool(trusted_setup))

    async def _run_command_io(self, function, *args, **kwargs):
        return await asyncio.get_running_loop().run_in_executor(
            getattr(self, "_executor", None), partial(function, *args, **kwargs),
        )

    async def read_files(self, sandbox_id: str, paths: list[str], timeout: float, user: str | None, max_bytes: int = 1024 * 1024) -> dict:
        runtime = self._runtime(sandbox_id)
        return await self._run_command_io(runtime.read_files, paths, timeout=timeout, user=user, max_bytes=max_bytes)

    async def write_file(
        self, sandbox_id: str, destination: str, payload: bytes, mode: int,
        expected_sha256: str | None = None, expected_size: int | None = None,
    ) -> None:
        runtime = self._runtime(sandbox_id)
        integrity = {}
        if expected_sha256 is not None:
            integrity["expected_sha256"] = expected_sha256
        if expected_size is not None:
            integrity["expected_size"] = expected_size
        await self._run_command_io(runtime.write_file, destination, payload, mode=int(mode), **integrity)

    async def extract_archive(self, sandbox_id: str, destination_parent: str, payload: bytes) -> None:
        runtime = self._runtime(sandbox_id)
        await self._run_command_io(runtime.extract_archive, destination_parent, payload)

    async def is_alive(self, sandbox_id: str) -> bool:
        try:
            return bool(self._runtime(sandbox_id).is_alive())
        except Exception:
            return False

    def _close_sync(self, sandbox_id: str, *, reclaim_filesystem: bool = False) -> None:
        self._monitor_event("close_worker_enter", operation_id=sandbox_id)
        # Keep the runtime, cache pin and resource ledger until close succeeds.
        with self._lock:
            runtime = self._sandboxes.get(sandbox_id)
            if not hasattr(self, "_close_locks"):
                self._close_locks = {}
            close_lock = self._close_locks.setdefault(sandbox_id, threading.Lock())
        self._monitor_event("close_lock_wait", operation_id=sandbox_id)
        with close_lock:
            self._monitor_event("close_lock_acquired", operation_id=sandbox_id)
            if runtime is None or sandbox_id not in self._sandboxes:
                return
            # The underlying runtime sets its own closed flag before cleanup;
            # a second close can be a no-op after failure. Verify physical state.
            lease = getattr(runtime, "_network_lease", None)
            host_veth = getattr(lease, "host_veth", None)
            cgroup_paths = tuple(Path(path) for path in (getattr(getattr(runtime, "_cgroup", None), "paths", {}) or {}).values())
            if not hasattr(self, "_cleanup_targets"):
                self._cleanup_targets = {}
            targets = self._cleanup_targets.setdefault(sandbox_id, (host_veth, cgroup_paths))
            control = getattr(self, "_cleanup_controls", {}).get(sandbox_id)
            if control is not None:
                previous_deadline = getattr(runtime, "_cleanup_deadline", None)
                runtime._cleanup_deadline = min(previous_deadline, control.deadline) if previous_deadline is not None else control.deadline
            collector = None if reclaim_filesystem else getattr(self, "_session_gc", None)
            self._monitor_event("runtime_close_begin", operation_id=sandbox_id)
            try:
                if collector is None:
                    runtime.close()
                else:
                    runtime.close(reclaim_filesystem=False)
                    if not getattr(runtime, "_isolation_closed", False):
                        raise RuntimeError("MiniSandbox runtime did not confirm isolation cleanup")
            finally:
                self._monitor_event("runtime_close_return", operation_id=sandbox_id)
            if hasattr(self, "sessions_root"):
                settle_deadline = time.monotonic() + 2.0
                while True:
                    try:
                        self._assert_preflight_cleanup(
                            sandbox_id, host_veth=targets[0], cgroup_paths=targets[1],
                            **({"allow_session": True} if collector is not None else {}),
                        )
                        break
                    except RuntimeError:
                        if time.monotonic() >= settle_deadline:
                            raise
                        time.sleep(0.05)
            if collector is not None:
                collector.submit(runtime.session_root)
                self._monitor_event("filesystem_gc_handoff", operation_id=sandbox_id)

            with self._lock:
                self._sandboxes.pop(sandbox_id, None)
                self._resources.pop(sandbox_id, None)
                digest_key = self._sandbox_digests.pop(sandbox_id, None)
                cache_lock = self._sandbox_cache_locks.pop(sandbox_id, None)
            if digest_key is not None:
                with self._cache_lock:
                    remaining = self._active_digests.get(digest_key, 1) - 1
                    if remaining > 0:
                        self._active_digests[digest_key] = remaining
                    else:
                        self._active_digests.pop(digest_key, None)
            if cache_lock is not None:
                cache_lock.close()
            self._cleanup_targets.pop(sandbox_id, None)
            self._close_locks.pop(sandbox_id, None)

    async def close(self, sandbox_id: str) -> None:
        await self._cleanup_sandbox(sandbox_id)

    def _stats_sync(self) -> dict[str, Any]:
        with self._lock:
            resources = tuple(self._resources.values())
        from rllm.sandbox.minisandbox_runtime.oci_runtime import _cgroup_mounts, _nearest_nonempty, _parse_cpu_set

        cgroup_mounts = _cgroup_mounts()
        cpuset_mount = cgroup_mounts["cpuset"]
        available_cpus = len(_parse_cpu_set(_nearest_nonempty(cpuset_mount, "cpuset.cpus")))
        memory_kib = 0
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    memory_kib = int(line.split()[1])
                    break
        except (OSError, ValueError, IndexError):
            pass
        memory_mb = memory_kib // 1024
        try:
            cgroup_memory_mb = int((cgroup_mounts["memory"] / "memory.limit_in_bytes").read_text(encoding="ascii").strip()) // 1024 // 1024
        except (OSError, ValueError):
            cgroup_memory_mb = 0
        if 0 < cgroup_memory_mb < memory_mb:
            memory_mb = cgroup_memory_mb
        cached = [path.name for path in self.cache_root.iterdir() if path.is_dir() and (path / ".complete.json").is_file()]
        return {
            "node_id": self.node_id,
            "node_ip": self.node_ip,
            "active": len(resources),
            "declared_cpus": sum(item[0] for item in resources),
            "declared_memory_mb": sum(item[1] for item in resources),
            "capacity_cpus": available_cpus,
            "capacity_memory_mb": memory_mb,
            "cached_digests": cached,
            "node_monitor": self._monitor_reference(),
            "filesystem_gc": self._session_gc.stats() if getattr(self, "_session_gc", None) else {},
        }

    async def stats(self) -> dict[str, Any]:
        self._start_monitor_heartbeat()
        return await asyncio.to_thread(self._stats_sync)

    def _shutdown_sync(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._lock:
            sandbox_ids = tuple(self._sandboxes)
        failures = []
        for sandbox_id in sandbox_ids:
            try:
                self._close_sync(sandbox_id)
            except Exception as exc:
                failures.append(exc)
                logger.exception("failed to close MiniSandbox %s", sandbox_id)
        if failures:
            raise RuntimeError("MiniSandbox shutdown retained unconfirmed sandbox resources") from failures[0]
        self._network.close()
        self._cgroups.close()
        shutil.rmtree(self.run_root, ignore_errors=True)

    async def shutdown(self) -> None:
        # Signal active cold-cache work before reclaiming node resources.
        self._closing = True
        for control in getattr(self, "_create_controls", {}).values():
            control.cancelled.set()
        operations = list(getattr(self, "_create_operations", {}))
        failures: list[BaseException] = []
        try:
            if operations:
                results = await asyncio.gather(*(self.cancel_create(op) for op in operations), return_exceptions=True)
                failures.extend(result for result in results if isinstance(result, BaseException))
            try:
                await asyncio.get_running_loop().run_in_executor(self._cleanup_executor, self._shutdown_sync)
            except Exception as exc:
                failures.append(exc)
            if failures:
                raise RuntimeError("MiniSandbox shutdown retained unconfirmed sandbox resources") from failures[0]
        finally:
            heartbeat = getattr(self, "_monitor_heartbeat", None)
            if heartbeat is not None:
                heartbeat.cancel()
            monitor = getattr(self, "_monitor", None)
            if monitor is not None:
                monitor.close()
            collector = getattr(self, "_session_gc", None)
            if collector is not None:
                collector.close()
            for executor in (self._create_executor, self._cleanup_executor, self._executor, getattr(self, "_diagnostic_executor", None)):
                if executor is not None:
                    executor.shutdown(wait=False, cancel_futures=True)


@dataclass
class _NodeState:
    actor: Any
    node_id: str
    capacity_cpus: int
    capacity_memory_mb: int
    cached_digests: set[str]
    active: int = 0
    declared_cpus: int = 0
    declared_memory_mb: int = 0
    healthy: bool = True
    node_monitor: dict | None = None


class MiniSandboxCluster:
    """Driver-side sticky weighted least-loaded router."""

    def __init__(
        self,
        nodes: list[_NodeState],
        task_digests: Mapping[str, str] | None = None,
    ) -> None:
        if not nodes:
            raise ValueError("MiniSandbox requires at least one selected GPU node")
        self._nodes = nodes
        self._by_id = {node.node_id: node for node in nodes}
        self._affinity: dict[str, str] = {}
        self._leases: dict[str, tuple[str, int, int]] = {}
        self._operation_ids: dict[str, str] = {}
        self._task_digests = dict(task_digests or {})
        self._lock = threading.RLock()
        self._closed = False
        self.fatal_error: str | None = None

    @classmethod
    def start(
        cls,
        *,
        node_ids: list[str],
        shared_cache_dir: str,
        local_root: str,
        run_id: str,
        canary_task_id: str | None = None,
    ) -> MiniSandboxCluster:
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        unique_nodes = list(dict.fromkeys(str(value) for value in node_ids))
        if not unique_nodes:
            raise ValueError("VERL reported no trainer/rollout GPU nodes")
        cache_root = Path(shared_cache_dir).expanduser().resolve()
        manifest = _load_object(cache_root / "materialization.json")
        raw_records = manifest.get("records")
        if not isinstance(raw_records, dict) or not raw_records:
            raise RuntimeError("MiniSandbox cache manifest has no records")
        if canary_task_id is None:
            canary_task_id = sorted(str(task_id) for task_id in raw_records)[0]
        else:
            canary_task_id = _safe_component(
                canary_task_id,
                label="canary task id",
            )
            if canary_task_id not in raw_records:
                raise RuntimeError(
                    f"MiniSandbox canary task is absent from cache: {canary_task_id}"
                )
        canary_record_path = (cache_root / str(raw_records[canary_task_id])).resolve()
        try:
            canary_record_path.relative_to(cache_root)
        except ValueError as exc:
            raise RuntimeError(f"unsafe MiniSandbox cache record for {canary_task_id}") from exc
        canary_record = _load_object(canary_record_path)
        canary_image = str(canary_record.get("original_image") or "")
        if not canary_image:
            raise RuntimeError(f"MiniSandbox canary record has no image: {canary_task_id}")
        # Keep request registries on one loop. Long command work runs outside
        # the RPC lifetime; concurrency groups alone do not isolate transport.
        for method in ("submit_create", "submit_close", "request_cancel_create",
                       "operation_status", "cleanup_status", "create", "close",
                       "cancel_create", "stats", "shutdown", "submit_command",
                       "command_result", "acknowledge_command"):
            ray.method(concurrency_group="control")(getattr(MiniSandboxNodeService, method))
        ray.method(concurrency_group="diagnostics")(MiniSandboxNodeService.lifecycle_diagnostics)
        remote_type = ray.remote(
            max_concurrency=512,
            concurrency_groups={"control": 32, "diagnostics": 2},
            num_cpus=0,
            max_restarts=1,
            max_task_retries=0,
        )(MiniSandboxNodeService)
        actors = []
        blocked_addresses = [str(node.get("NodeManagerAddress")) for node in ray.nodes() if node.get("Alive") and node.get("NodeManagerAddress")]
        from rllm.utils.diagnostic_events import diagnostic_run_directory
        diagnostic_root = diagnostic_run_directory()
        diagnostics_dir = str(diagnostic_root / "nodes" / run_id) if diagnostic_root else None
        for index, node_id in enumerate(unique_nodes):
            actor = remote_type.options(
                name=f"rllm-minisandbox-{run_id}-{index}",
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
            ).remote(
                run_id=f"{run_id}-{index}",
                shared_cache_dir=shared_cache_dir,
                local_root=local_root,
                blocked_addresses=blocked_addresses,
                diagnostics_dir=diagnostics_dir,
            )
            actors.append((node_id, actor))

        def cleanup_startup_actors() -> None:
            try:
                ray.get(
                    [actor.shutdown.remote() for _, actor in actors],
                    timeout=30,
                )
            except Exception:
                pass
            for _, actor in actors:
                try:
                    ray.kill(actor, no_restart=True)
                except Exception:
                    logger.exception(
                        "failed to stop MiniSandbox actor after startup error"
                    )

        try:
            preflight_results = ray.get(
                [actor.preflight_task.remote(canary_task_id, canary_image) for _, actor in actors],
                timeout=1860 + STARTUP_RECLAMATION_TIMEOUT,
            )
            stats = ray.get([actor.stats.remote() for _, actor in actors], timeout=30)
            logger.info(
                "MiniSandbox real-image preflight passed on %d node(s): task=%s digest=%s",
                len(preflight_results),
                canary_task_id,
                str(preflight_results[0].get("manifest_digest") or ""),
            )
            nodes: list[_NodeState] = []
            for expected_node, actor, value in zip(
                unique_nodes,
                (item[1] for item in actors),
                stats,
                strict=True,
            ):
                actual_node = str(value.get("node_id") or "")
                if actual_node != expected_node:
                    raise RuntimeError(
                        "MiniSandbox node affinity failed: "
                        f"expected={expected_node} actual={actual_node}"
                    )
                monitor_reference = value.get("node_monitor")
                if (monitor_reference or {}).get("control_endpoint"):
                    from rllm.utils.node_diagnostics import query_monitor
                    control_probe = query_monitor(monitor_reference, "startup-preflight")
                    if control_probe.get("collection_error"):
                        raise RuntimeError(
                            f"MiniSandbox independent monitor preflight failed on {actual_node}: "
                            f"{control_probe['collection_error']}; node UDP endpoint must be reachable"
                        )
                    logger.info("MiniSandbox independent monitor preflight passed: node=%s", actual_node)
                nodes.append(
                    _NodeState(
                        actor=actor,
                        node_id=actual_node,
                        capacity_cpus=max(1, int(value["capacity_cpus"])),
                        capacity_memory_mb=max(
                            1, int(value["capacity_memory_mb"])
                        ),
                        cached_digests=set(value.get("cached_digests") or []),
                        node_monitor=value.get("node_monitor"),
                    )
                )
            task_digests: dict[str, str] = {}
            for task_id, relative in raw_records.items():
                record_path = (cache_root / str(relative)).resolve()
                try:
                    record_path.relative_to(cache_root)
                except ValueError as exc:
                    raise RuntimeError(
                        f"unsafe MiniSandbox cache record for {task_id}"
                    ) from exc
                record = _load_object(record_path)
                digest = str(record.get("manifest_digest") or "")
                if digest.startswith("sha256:"):
                    task_digests[str(task_id)] = digest.removeprefix("sha256:")
            return cls(nodes, task_digests=task_digests)
        except BaseException:
            cleanup_startup_actors()
            raise

    def begin_cleanup_reconciliation(self, node_id: str, sandbox_id: str) -> None:
        with self._lock:
            pending = self.__dict__.setdefault("_reconciling_closes", {})
            pending.setdefault(node_id, set()).add(sandbox_id)

    def finish_cleanup_reconciliation(self, node_id: str, sandbox_id: str) -> None:
        with self._lock:
            pending = getattr(self, "_reconciling_closes", {})
            operations = pending.get(node_id)
            if operations is not None:
                operations.discard(sandbox_id)
                if not operations:
                    pending.pop(node_id, None)

    def quarantine(self, node_id: str, detail: str, diagnostics: dict | None = None) -> None:
        collect = False
        with self._lock:
            node = self._by_id[node_id]
            node.healthy = False
            if self.fatal_error is None:
                self.fatal_error = detail
                self._fatal_diagnostics = {"node_id": node_id, "node_monitor": node.node_monitor, **(diagnostics or {})}
                collect = bool(node.node_monitor)
        if collect:
            from rllm.utils.node_diagnostics import archived_evidence
            evidence = archived_evidence(node.node_monitor)
            with self._lock:
                self._fatal_diagnostics["node_monitor"] = evidence

    def check_health(self) -> None:
        if self.fatal_error:
            from rllm.types import RolloutInfrastructureError
            raise RolloutInfrastructureError(
                "minisandbox_cleanup_unconfirmed", self.fatal_error,
                retryable=False, stage="sandbox_cleanup",
                diagnostics={**getattr(self, "_fatal_diagnostics", {}), "failure_scope": "node", "fatal": True},
            )

    def _choose(
        self,
        route_key: str,
        cpus: int,
        memory_mb: int,
        task_id: str | None = None,
    ) -> _NodeState:
        pending = getattr(self, "_reconciling_closes", {})
        sticky = self._affinity.get(route_key)
        if sticky is not None:
            node = self._by_id.get(sticky)
            if node is not None and node.healthy and not pending.get(node.node_id):
                return node
        candidates = [node for node in self._nodes if node.healthy and not pending.get(node.node_id)]
        if not candidates:
            if pending:
                from rllm.types import RolloutInfrastructureError
                raise RolloutInfrastructureError(
                    "minisandbox_cleanup_reconciling", "MiniSandbox nodes are reconciling cleanup; admission paused",
                    retryable=True, stage="sandbox_admission", retry_scope="full_rollout",
                    diagnostics={"failure_scope": "rollout", "nodes": sorted(pending), "execution_started": False},
                )
            raise RuntimeError("every MiniSandbox node service is unhealthy")

        digest = self._task_digests.get(str(task_id or ""))

        def score(node: _NodeState) -> tuple[float, int, int, str]:
            resource_load = max(
                (node.declared_cpus + cpus) / node.capacity_cpus,
                (node.declared_memory_mb + memory_mb) / node.capacity_memory_mb,
            )
            cache_miss = int(not digest or digest not in node.cached_digests)
            return resource_load, cache_miss, node.active, node.node_id

        node = min(candidates, key=score)
        self._affinity[route_key] = node.node_id
        return node

    def create(
        self,
        *,
        name: str,
        task_id: str,
        image: str,
        route_key: str,
        cpus: int,
        memory_mb: int,
        storage_mb: int | None,
        working_dir: str | None,
        environment: Mapping[str, str] | None,
        allow_internet: bool,
        create_timeout: float = 1800.0,
    ) -> tuple[Any, str, str]:
        import ray

        self.check_health()
        operation_id = uuid.uuid4().hex
        with self._lock:
            if self._closed:
                raise RuntimeError("MiniSandbox cluster is closed")
            node = self._choose(route_key, cpus, memory_mb, task_id)
            node.active += 1
            node.declared_cpus += cpus
            node.declared_memory_mb += memory_mb
        try:
            submit = getattr(node.actor, "submit_create", None)
            result_ref = (submit if submit is not None else node.actor.create).remote(
                    operation_id=operation_id,
                    operation_timeout=float(create_timeout),
                    name=name,
                    task_id=task_id,
                    image=image,
                    cpus=cpus,
                    memory_mb=memory_mb,
                    storage_mb=storage_mb,
                    working_dir=working_dir,
                    environment=dict(environment or {}),
                    allow_internet=allow_internet,
            )
            if submit is not None:
                result = _wait_for_lifecycle(ray, node.actor, result_ref, operation_id, kind="create", timeout=float(create_timeout), node_monitor=node.node_monitor)
            else:
                result = _wait_for_create(ray, node.actor, result_ref, operation_id, float(create_timeout))
        except BaseException as create_error:
            confirmed = False
            cleanup_error = None
            try:
                cancel = getattr(node.actor, "request_cancel_create", None)
                if cancel is not None:
                    confirmed = _wait_for_lifecycle(ray, node.actor, cancel.remote(operation_id), operation_id, kind="cancel", timeout=55.0, node_monitor=node.node_monitor)
                else:
                    confirmed = ray.get(node.actor.cancel_create.remote(operation_id), timeout=60) is True
            except Exception as exc:
                cleanup_error = repr(exc)[-2000:]
            if not confirmed:
                # The reservation includes a possible late create. Keep it
                # charged and stop admission until the job boundary cleans up.
                status = {}
                try:
                    status = ray.get(node.actor.operation_status.remote(operation_id), timeout=5)
                except Exception:
                    pass
                self.quarantine(node.node_id, f"create cleanup unconfirmed: {operation_id}: {create_error}", {
                    "operation_id": operation_id, "task_id": task_id,
                    "image": image, "create_error": repr(create_error)[-2000:],
                    "cleanup_error": cleanup_error, "operation_status": status,
                })
                self.check_health()
            service_healthy = False
            try:
                ray.get(node.actor.stats.remote(), timeout=30)
                service_healthy = True
            except Exception:
                pass
            with self._lock:
                node.active -= 1
                node.declared_cpus -= cpus
                node.declared_memory_mb -= memory_mb
                node.healthy = service_healthy
                if not service_healthy and self._affinity.get(route_key) == node.node_id:
                    self._affinity.pop(route_key, None)
            raise
        sandbox_id = str(result["sandbox_id"])
        digest = str(result.get("manifest_digest") or "").removeprefix("sha256:")
        with self._lock:
            node.cached_digests.add(digest)
            self._leases[sandbox_id] = (node.node_id, cpus, memory_mb)
            self._operation_ids[sandbox_id] = operation_id
        return node.actor, sandbox_id, node.node_id

    def release(self, sandbox_id: str) -> None:
        with self._lock:
            lease = self._leases.pop(sandbox_id, None)
            self._operation_ids.pop(sandbox_id, None)
            if lease is None:
                return
            node_id, cpus, memory_mb = lease
            node = self._by_id[node_id]
            node.active = max(0, node.active - 1)
            node.declared_cpus = max(0, node.declared_cpus - cpus)
            node.declared_memory_mb = max(0, node.declared_memory_mb - memory_mb)

    def shutdown(self) -> None:
        import ray

        with self._lock:
            if self._closed:
                return
            self._closed = True
            actors = [node.actor for node in self._nodes]
        errors: list[BaseException] = []
        try:
            ray.get([actor.shutdown.remote() for actor in actors], timeout=120)
        except BaseException as exc:
            errors.append(exc)
        finally:
            for actor in actors:
                try:
                    ray.kill(actor, no_restart=True)
                except Exception:
                    logger.exception("failed to stop MiniSandbox node service")
        if errors:
            raise RuntimeError("MiniSandbox node service shutdown failed") from errors[0]


class MiniSandbox:
    """Synchronous Sandbox protocol adapter backed by one node service."""

    def __init__(
        self,
        *,
        name: str,
        image: str,
        task_id: str,
        route_key: str | None = None,
        cpus: int = 1,
        memory_mb: int = 4096,
        storage_mb: int | None = None,
        env: Mapping[str, str] | None = None,
        working_dir: str | None = None,
        allow_internet: bool = True,
        create_timeout: float = 1800.0,
        **_ignored: Any,
    ) -> None:
        self.name = name
        self.image = image
        self.task_id = task_id
        self._cluster = get_active_minisandbox_cluster()
        self._closed = False
        self._actor, self._sandbox_id, self.node_id = self._cluster.create(
            name=name,
            task_id=task_id,
            image=image,
            route_key=str(route_key or task_id),
            cpus=int(cpus),
            memory_mb=int(memory_mb),
            storage_mb=(int(storage_mb) if storage_mb is not None else None),
            working_dir=working_dir,
            environment=env,
            allow_internet=bool(allow_internet),
            create_timeout=create_timeout,
        )

    @property
    def runtime_diagnostics(self) -> dict[str, Any]:
        task_id = getattr(self, "task_id", None)
        sandbox_id = getattr(self, "_sandbox_id", None)
        cluster = getattr(self, "_cluster", None)
        return {
            "node_id": getattr(self, "node_id", None), "sandbox_id": sandbox_id,
            "operation_id": getattr(cluster, "_operation_ids", {}).get(sandbox_id),
            "task_id": task_id, "image": getattr(self, "image", None),
            "manifest_digest": getattr(cluster, "_task_digests", {}).get(task_id),
            "close_state": getattr(self, "_close_state", "open"),
        }

    def _raise_rpc_failure(self, exc: BaseException, operation: str) -> None:
        from rllm.types import RolloutInfrastructureError
        try:
            _raise_node_service_failure(exc, operation=operation)
        except RolloutInfrastructureError as failure:
            failure.diagnostics = {**(failure.diagnostics or {}), "sandbox": self.runtime_diagnostics}
            raise

    def _assert_rpc_open(self, operation: str, *, submitted: bool = False) -> None:
        if self._closed or getattr(self, "_close_state", "open") != "open":
            from rllm.types import RolloutInfrastructureError
            raise RolloutInfrastructureError(
                "sandbox_closed", f"MiniSandbox {self.name} is closing or closed",
                stage="sandbox_rpc", retryable=True, retry_scope="full_rollout",
                diagnostics={"operation": operation, "sandbox": self.runtime_diagnostics,
                             "request_submitted": submitted,
                             "execution_started": None if submitted else False},
            )

    def _command_rpc(self, method: str, args: tuple, *, operation: str, budget: float, details: dict | None = None):
        protocol = all(callable(getattr(getattr(self._actor, name, None), "remote", None))
                       for name in ("submit_command", "command_result", "acknowledge_command"))
        return self._rpc_once(
            lambda: getattr(self._actor, method).remote(self._sandbox_id, *args),
            operation=operation, budget=budget, details=details,
            command_method=method if protocol else None, command_args=args,
        )

    def _rpc_once(self, submit, *, operation: str, budget: float, details: dict | None = None,
                  command_method: str | None = None, command_args: tuple = ()):
        """Submit once; each reply wait retains its reference and total deadline."""
        import ray

        from rllm.types import RolloutInfrastructureError

        self._assert_rpc_open(operation)
        started = time.monotonic()
        deadline = started + max(0.0, float(budget))
        request_id = uuid.uuid4().hex
        waits = 0
        reference = None
        observed = False
        command_acknowledged = False
        try:
            if deadline <= started:
                raise ValueError("MiniSandbox RPC budget must be positive")
            if command_method is not None:
                submit = lambda: self._actor.submit_command.remote(request_id, self._sandbox_id, command_method, command_args)
            reference = _RayReply(ray, rpc_diag.call("rpc_submit", request_id, submit, sandbox_id=self._sandbox_id, operation=operation))
            while True:
                remaining = deadline - time.monotonic()
                self._assert_rpc_open(operation, submitted=True)
                if remaining <= 0:
                    raise RolloutInfrastructureError(
                        "minisandbox_rpc_timeout", f"MiniSandbox RPC deadline exceeded during {operation}",
                        retryable=True, stage="sandbox_rpc", retry_scope="full_rollout",
                    )
                if reference is None and command_method is not None:
                    reference = _RayReply(ray, self._actor.command_result.remote(request_id))
                    observed = False
                try:
                    allowance = min(10.0, remaining)
                    result = rpc_diag.call("rpc_result_wait", request_id, lambda: reference.get(timeout=allowance),
                                           wait_budget_seconds=allowance, sandbox_id=self._sandbox_id, operation=operation)
                    observed = True
                    if command_method is None:
                        return result
                    if not isinstance(result, dict) or result.get("request_id") != request_id:
                        raise RuntimeError("invalid MiniSandbox command response identity")
                    if not command_acknowledged:
                        command_acknowledged = True
                        poll_delay = 0.01
                        reference = None
                        continue
                    if result.get("ready") is True and "result" in result:
                        return result["result"]
                    if result.get("ready") is not False:
                        raise RuntimeError("invalid MiniSandbox command result state")
                    reference = None
                    time.sleep(min(poll_delay, max(0.0, deadline - time.monotonic())))
                    poll_delay = min(0.25, poll_delay * 2)
                except Exception as exc:
                    if not isinstance(exc, _ReplyWaitTimeout) and type(exc).__name__ != "GetTimeoutError":
                        observed = True
                        raise
                    # A local wait timeout does not cancel the remote request.
                    # Never issue submit() again, including for an upload.
                    waits += 1
                    if waits == 1:
                        rpc_diag.event("rpc_first_wait_timeout", request_id, sandbox_id=self._sandbox_id,
                                       operation=operation, elapsed_seconds=time.monotonic() - started)
                        rpc_diag.capture_threads(request_id, "rpc_first_wait_timeout")
        except BaseException as exc:
            failure = exc
            try:
                self._raise_rpc_failure(exc, operation=operation)
            except BaseException as classified:
                failure = classified
            failure.diagnostics = {
                **(getattr(failure, "diagnostics", None) or {}),
                **_exception_diagnostics(exc),
                "operation": operation, "request_id": request_id,
                "exception_type": type(exc).__name__, "recovery_waits": waits,
                "budget_seconds": budget, "elapsed_seconds": time.monotonic() - started,
                "command_protocol": "short_rpc" if command_method is not None else "legacy",
                "command_acknowledged": command_acknowledged,
                "sandbox": self.runtime_diagnostics, **(details or {}),
            }
            rpc_diag.event("rpc_failure", request_id, operation=operation,
                           sandbox_id=self._sandbox_id, diagnostics=failure.diagnostics)
            if getattr(exc, "reason", None) == "minisandbox_rpc_timeout":
                node = getattr(self._cluster, "_by_id", {}).get(self.node_id)
                monitor_reference = getattr(node, "node_monitor", None)
                if monitor_reference:
                    from rllm.utils.node_diagnostics import query_monitor
                    # Read-only evidence request, never a replay of the timed
                    # out command. Capture while the symptom is still active.
                    failure.diagnostics["node_control"] = query_monitor(monitor_reference, request_id)
            if failure is not exc:
                raise failure from exc
            raise
        finally:
            if reference is not None and not observed:
                _observe_late_rpc(reference, request_id, operation=operation, sandbox_id=self._sandbox_id)
            if command_acknowledged and observed:
                try:
                    ack = self._actor.acknowledge_command.remote(request_id)
                    _observe_late_rpc(ack, request_id, operation="command acknowledgement", sandbox_id=self._sandbox_id)
                except Exception:
                    logger.debug("command result retained until sandbox close: %s", request_id, exc_info=True)

    def exec(self, command: str, timeout: float | None = None, user: str | None = None) -> str:
        run_timeout = float(timeout if timeout is not None else 1200)
        return str(self._command_rpc(
            "exec", (command, run_timeout, user),
            operation="exec", budget=run_timeout + 60,
        ))

    def exec_setup(
        self, command: str, timeout: float | None = None, user: str | None = None,
    ) -> str:
        """Host-supplied setup retains its existing narrow filesystem caps."""
        run_timeout = float(timeout if timeout is not None else 1200)
        return str(self._command_rpc(
            "exec", (command, run_timeout, user, True),
            operation="trusted setup", budget=run_timeout + 60,
        ))

    def _readonly(self, submit, *, operation: str, timeout: float, command_method: str | None = None, command_args=None):
        from rllm.types import RolloutInfrastructureError

        self._assert_rpc_open(operation)
        deadline = time.monotonic() + max(0.0, float(timeout))
        attempt = 0
        last_error = None
        while not self._closed:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            allowance = min(10.0 * 2 ** min(attempt, 2), remaining)
            try:
                # A short command limit is not a deadline for its Ray reply.
                # Keep waiting on that same request until the original read
                # deadline; only a confirmed command failure permits resubmit.
                if command_method is not None:
                    return self._command_rpc(command_method, command_args(allowance), operation=operation, budget=remaining)
                return self._rpc_once(lambda allowance=allowance: submit(allowance), operation=operation, budget=remaining)
            except (RolloutInfrastructureError, TimeoutError) as exc:
                if isinstance(exc, RolloutInfrastructureError) and exc.reason != "minisandbox_rpc_timeout":
                    raise
                last_error = exc
                attempt += 1
                if time.monotonic() < deadline:
                    time.sleep(min(0.25 * 2 ** min(attempt - 1, 3), max(0.0, deadline - time.monotonic())))
        if last_error is None:
            last_error = RolloutInfrastructureError(
                "minisandbox_rpc_timeout", f"MiniSandbox read deadline exceeded during {operation}",
                retryable=True, stage="sandbox_rpc", retry_scope="full_rollout",
            )
        last_error.diagnostics = {
            **(getattr(last_error, "diagnostics", None) or {}),
            "read_attempts": attempt, "read_budget_seconds": timeout, "sandbox": self.runtime_diagnostics,
        }
        raise last_error

    def exec_readonly(self, command: str, timeout: float = 30, user: str | None = None) -> str:
        """Retry only caller-declared read-only/idempotent operations."""
        return str(self._readonly(
            lambda allowance: self._actor.exec.remote(self._sandbox_id, command, allowance, user),
            operation="readonly exec", timeout=timeout,
            command_method="exec", command_args=lambda allowance: (command, allowance, user),
        ))

    def read_files(self, paths: list[str], timeout: float = 30, user: str | None = None, max_bytes: int = 1024 * 1024) -> dict:
        from rllm.types import RolloutInfrastructureError

        from rllm.sandbox.file_snapshot import validate_max_bytes
        validate_max_bytes(max_bytes)
        paths = [_safe_remote_path(path) for path in paths]
        try:
            records = self._readonly(
                lambda allowance: self._actor.read_files.remote(self._sandbox_id, paths, allowance, user, max_bytes),
                operation="file snapshot", timeout=timeout,
                command_method="read_files", command_args=lambda allowance: (paths, allowance, user, max_bytes),
            )
        except (RolloutInfrastructureError, TimeoutError) as exc:
            if isinstance(exc, RolloutInfrastructureError) and exc.reason != "minisandbox_rpc_timeout":
                raise
            return {path: {"state": "transport_error", "error": type(exc).__name__,
                           "diagnostics": getattr(exc, "diagnostics", None)} for path in paths}
        if not isinstance(records, dict) or any(
            not isinstance(records.get(path), dict)
            or records[path].get("state") not in {"present", "missing", "unreadable", "invalid"}
            or (records[path].get("state") == "present" and not isinstance(records[path].get("content"), str))
            for path in paths
        ):
            return {path: {"state": "invalid", "error": "malformed_file_snapshot"} for path in paths}
        return records

    def upload_file(self, local_path: str, remote_path: str) -> None:
        source = Path(local_path)
        if not source.is_file():
            raise FileNotFoundError(local_path)
        destination = _safe_remote_path(remote_path)
        payload = source.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        self._command_rpc(
            "write_file", (destination, payload, source.stat().st_mode & 0o777, digest, len(payload)),
            operation="file upload", budget=300 + 60,
            details={"destination": destination, "expected_sha256": digest, "expected_size": len(payload)},
        )

    def upload_dir(self, local_path: str, remote_path: str) -> None:
        source = Path(local_path)
        if not source.is_dir():
            raise FileNotFoundError(local_path)
        destination = PurePosixPath(_safe_remote_path(remote_path))
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as archive:
            archive.add(source, arcname=destination.name)
        self._command_rpc(
            "extract_archive", (str(destination.parent), stream.getvalue()),
            operation="directory upload", budget=600 + 60,
            details={"destination": str(destination)},
        )

    def is_alive(self) -> bool:
        if self._closed:
            return False
        try:
            import ray

            return bool(ray.get(self._actor.is_alive.remote(self._sandbox_id), timeout=5))
        except Exception:
            return False

    def close(self) -> None:
        if self._closed:
            return
        import ray

        self._close_state = "closing"
        submit = getattr(self._actor, "submit_close", None)
        if submit is not None:
            try:
                # The server coalesces all handles for this sandbox. Even a
                # cancelled/lost client wait cannot start another cleanup.
                node = getattr(self._cluster, "_by_id", {}).get(self.node_id)
                reference = rpc_diag.call("close_submit", self._sandbox_id,
                                          lambda: submit.remote(self._sandbox_id), trace=True, node_id=self.node_id)
                begin_reconciliation = getattr(self._cluster, "begin_cleanup_reconciliation", None)
                _wait_for_lifecycle(
                    ray, self._actor, reference, self._sandbox_id, kind="close", timeout=55.0,
                    node_monitor=getattr(node, "node_monitor", None), reconcile_timeout=CLOSE_RECONCILE_SECONDS,
                    on_reconcile=(lambda: begin_reconciliation(self.node_id, self._sandbox_id))
                    if begin_reconciliation is not None else None,
                )
            except BaseException as exc:
                self._close_state = "failed"
                self._cluster.quarantine(self.node_id, f"close unconfirmed: {self._sandbox_id}: {exc}", {
                    **_exception_diagnostics(exc), "sandbox": self.runtime_diagnostics,
                    "close_submissions": 1, "cleanup_budget_seconds": 55.0,
                    "cleanup_queue_stall_seconds": 60.0,
                })
                self._cluster.check_health()
                raise
            finally:
                finish_reconciliation = getattr(self._cluster, "finish_cleanup_reconciliation", None)
                if finish_reconciliation is not None:
                    finish_reconciliation(self.node_id, self._sandbox_id)
            self._closed = True
            self._close_state = "closed"
            self._cluster.release(self._sandbox_id)
            return
        deadline = time.monotonic() + 60.0
        last_error: BaseException | None = None
        reference = None
        submissions = 0
        for _attempt in range(2):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                reference = self._actor.close.remote(self._sandbox_id)
                submissions += 1
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("MiniSandbox cleanup deadline exceeded")
                    try:
                        ray.get(reference, timeout=min(10.0, remaining))
                        break
                    except Exception as exc:
                        if type(exc).__name__ != "GetTimeoutError":
                            raise
                        # A lost/late response leaves the same close running.
                        # Only a confirmed exception may start another close.
                        last_error = exc
            except BaseException as exc:
                last_error = exc
                if not isinstance(exc, Exception):
                    break
            else:
                self._closed = True
                self._close_state = "closed"
                self._cluster.release(self._sandbox_id)
                return
        self._close_state = "failed"
        self._cluster.quarantine(self.node_id, f"close unconfirmed: {self._sandbox_id}: {last_error}", {
            **(_exception_diagnostics(last_error) if last_error is not None else {}),
            "sandbox": self.runtime_diagnostics, "close_submissions": submissions,
            "cleanup_budget_seconds": 60.0,
        })
        self._cluster.check_health()


__all__ = [
    "MiniSandbox",
    "MiniSandboxCluster",
    "MiniSandboxNodeService",
    "get_active_minisandbox_cluster",
    "set_active_minisandbox_cluster",
]
