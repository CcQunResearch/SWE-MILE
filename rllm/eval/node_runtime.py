"""Process-local, evaluation-only adapters for VERL standalone node placement."""

from __future__ import annotations

import inspect
import os
import signal
import time
from pathlib import Path
from typing import Any


def available_per_node(ray: Any) -> dict:
    # Ray exposes no public per-node available-resource API. Keep the pinned
    # runtime dependency in one place, and fail closed if it changes.
    return ray._private.state.available_resources_per_node()


class NodeRayResources:
    """Ray facade used by the existing initialization retry/reclamation code."""

    def __init__(self, ray: Any, node_id: str):
        self.ray = ray
        self.node_id = node_id

    def available_resources(self) -> dict:
        alive = any(row.get("Alive") and row.get("NodeID") == self.node_id for row in self.ray.nodes())
        if not alive:
            raise RuntimeError(f"evaluation GPU node disappeared: {self.node_id}")
        return available_per_node(self.ray).get(self.node_id, {})

    def __getattr__(self, name: str) -> Any:
        return getattr(self.ray, name)


class NodeRuntimeIsolation:
    """Install only in a fresh, single-model Runner process.

    Pin GPU bundles (not merely their owner), and register every created actor
    and PG before initialization can block. The driver can therefore clean a
    partially constructed manager even if its owner process dies.
    """

    def __init__(self, ray: Any, coordinator: Any, owner: str, node: dict):
        self.ray, self.coordinator, self.owner, self.node = ray, coordinator, owner, node
        self.actors: list = []
        self.placement_groups: list = []
        self.processes: list = []

    def __enter__(self):
        from ray.actor import ActorClass
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy, PlacementGroupSchedulingStrategy
        from verl.single_controller.ray import base

        self.actor_class = ActorClass
        self.base = base
        self.original_actor_remote = ActorClass._remote
        self.original_pg = base.placement_group
        if "actor_options" not in inspect.signature(self.original_actor_remote).parameters:
            raise RuntimeError("unsupported Ray ActorClass._remote interface for node_parallel")

        def placement_group(*args, **kwargs):
            if args:
                raise RuntimeError("unsupported VERL positional placement_group call")
            bundles = kwargs.get("bundles")
            if not bundles:
                raise RuntimeError("VERL placement group has no resource bundles")
            kwargs["bundles"] = [dict(bundle, **{self.node["node_resource"]: 0.001}) for bundle in bundles]
            kwargs["strategy"] = "STRICT_PACK"
            kwargs["name"] = str(kwargs.get("name", "rollout")) + "_" + self.owner
            pg = self.original_pg(**kwargs)
            self.placement_groups.append(pg)
            self.ray.get(self.coordinator.register_resource.remote(self.owner, "placement_groups", pg))
            return pg

        def actor_remote(actor_class, args=None, kwargs=None, **options):
            strategy = options.get("scheduling_strategy")
            if isinstance(strategy, NodeAffinitySchedulingStrategy):
                if strategy.node_id != self.node["node_id"]:
                    raise RuntimeError("evaluation actor attempted to escape its assigned node")
                options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(self.node["node_id"], soft=False)
            elif isinstance(strategy, PlacementGroupSchedulingStrategy):
                if strategy.placement_group.id not in {pg.id for pg in self.placement_groups}:
                    raise RuntimeError("evaluation actor requested an unowned placement group")
            elif options.get("placement_group") not in (None, "default"):
                if options["placement_group"].id not in {pg.id for pg in self.placement_groups}:
                    raise RuntimeError("evaluation actor requested an unowned placement group")
            else:
                options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(self.node["node_id"], soft=False)
            handle = self.original_actor_remote(actor_class, args=args, kwargs=kwargs, **options)
            self.actors.append(handle)
            self.ray.get(self.coordinator.register_resource.remote(self.owner, "actors", handle))
            return handle

        base.placement_group = placement_group
        ActorClass._remote = actor_remote
        return self

    def manager_class(self, base_manager):
        owner = self.owner

        class IsolatedManager(base_manager):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                base_replica = self.rollout_replica_class

                class IsolatedReplica(base_replica):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        self.name_suffix += "_" + owner

                self.rollout_replica_class = IsolatedReplica

        return IsolatedManager

    def cleanup(self) -> None:
        cleanup_resources(self.ray, {"actors": self.actors, "placement_groups": self.placement_groups, "processes": self.processes})

    def gateway_class(self, base_gateway):
        isolation = self

        class RegisteredGateway(base_gateway):
            def _on_process_started(self):
                record = process_identity(self._process.pid)
                if record is not None:
                    isolation.processes.append(record)
                    isolation.ray.get(isolation.coordinator.register_resource.remote(isolation.owner, "processes", record))

        return RegisteredGateway

    def __exit__(self, *exc):
        self.base.placement_group = self.original_pg
        self.actor_class._remote = self.original_actor_remote


def process_identity(pid: int) -> dict | None:
    """Linux process start time protects against PID reuse during cleanup."""
    if pid < 2:
        raise ValueError("refusing to register a system process")
    try:
        # comm (field 2) can contain spaces and parentheses; fields after it
        # start with state (field 3), and starttime is field 22.
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "start_ticks": fields[19]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def cleanup_registered_processes(records: list[dict]) -> None:
    for record in records:
        if process_identity(record["pid"]) != record:
            continue
        pidfd = None
        try:
            # Pin the kernel process identity across check/signal, not just
            # across separate cleanup calls (Linux >= 5.3).
            pidfd = os.pidfd_open(record["pid"])
            if process_identity(record["pid"]) != record:
                continue
            signal.pidfd_send_signal(pidfd, signal.SIGTERM)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and process_identity(record["pid"]) == record:
                time.sleep(0.05)
            if process_identity(record["pid"]) == record:
                signal.pidfd_send_signal(pidfd, signal.SIGKILL)
        except ProcessLookupError:
            pass
        finally:
            if pidfd is not None:
                os.close(pidfd)


def cleanup_resources(ray: Any, resources: dict, *, node_id: str | None = None, graceful: bool = False) -> None:
    """Idempotent cleanup restricted to handles registered by this runtime."""
    import logging

    errors = []
    processes = resources.get("processes", [])
    if processes:
        try:
            if node_id is None:
                cleanup_registered_processes(processes)
            else:
                from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
                ref = ray.remote(num_cpus=0)(cleanup_registered_processes).options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False),
                ).remote(processes)
                ray.get(ref, timeout=5 * len(processes) + 10)
        except Exception as exc:
            errors.append(exc)
            logging.getLogger(__name__).warning("Could not clean registered Gateway processes", exc_info=True)
    if graceful:
        from rllm.trainer.verl.server_lifecycle import request_graceful_server_shutdown
        request_graceful_server_shutdown(resources["actors"], ray)
    for actor in reversed(resources["actors"]):
        try:
            ray.kill(actor, no_restart=True)
        except Exception as exc:
            errors.append(exc)
            logging.getLogger(__name__).warning("Could not kill registered evaluation actor", exc_info=True)
    for pg in resources["placement_groups"]:
        try:
            ray.util.remove_placement_group(pg)
        except Exception as exc:
            errors.append(exc)
            logging.getLogger(__name__).warning("Could not remove registered evaluation placement group", exc_info=True)
    if errors:
        raise RuntimeError(f"failed releasing {len(errors)} registered evaluation resource(s)") from errors[0]


def wait_for_owned_resources(ray: Any, resources: dict, node: dict, *, timeout: float, stable: float) -> None:
    """Check exact PGs plus the selected node, never cluster-wide free GPUs."""
    proxy = NodeRayResources(ray, node["node_id"])
    deadline = time.monotonic() + timeout
    stable_since = None
    while time.monotonic() < deadline:
        removed = all(
            ray.util.placement_group_table(pg).get("state") == "REMOVED"
            for pg in resources["placement_groups"]
        )
        free = proxy.available_resources().get("GPU", 0) >= node["baseline_gpu"]
        if removed and free:
            stable_since = time.monotonic() if stable_since is None else stable_since
            if time.monotonic() - stable_since >= stable:
                return
        else:
            stable_since = None
        time.sleep(1)
    raise TimeoutError(f"evaluation resources were not reclaimed on node {node['node_id']}")
