"""Shared admission and node selection for concurrent checkpoint evaluation.

The coordinator is instantiated as one async Ray actor by the evaluation driver.
It deliberately has no Ray dependency, so scheduling invariants can be tested
without GPUs. Permits belong to a model runtime, never to a task ID alone.
"""

from __future__ import annotations

import asyncio
from collections import Counter, OrderedDict
from typing import Any
from uuid import uuid4


class EvaluationCoordinator:
    def __init__(self, task_limit: int, startup_limit: int, owners: list[str] | None = None):
        if task_limit < 1 or startup_limit < 1:
            raise ValueError("evaluation admission limits must be positive")
        self.limits = {"task": task_limit, "startup": min(task_limit, startup_limit)}
        # The driver reserves the entire wave before starting any runtime.
        # None preserves the unreserved admission API for standalone clients.
        if owners is not None and len(set(owners)) != len(owners):
            raise ValueError("evaluation owners must be unique")
        self.owners = None if owners is None else list(owners)
        self.waiters = {kind: OrderedDict() for kind in self.limits}
        self.held = {kind: {} for kind in self.limits}
        self.closed: set[str] = set()
        self.quarantined: set[str] = set()
        self.released: set[tuple[str, str, str]] = set()
        self.last_owner = {kind: None for kind in self.limits}
        self.progress: dict[str, dict] = {}
        self.resources: dict[str, dict[str, list]] = {}

    async def acquire(self, kind: str, owner: str, token: str) -> bool:
        key = (owner, token)
        if (owner in self.closed or (kind, owner, token) in self.released
                or (self.owners is not None and owner not in self.owners)):
            return False
        if key in self.held[kind]:
            return True
        future = self.waiters[kind].get(key)
        if future is None:
            future = asyncio.get_running_loop().create_future()
            self.waiters[kind][key] = future
        # Let concurrently submitted requests join the fair selection first.
        asyncio.get_running_loop().call_soon(self._drain, kind)
        return await asyncio.shield(future)

    def _drain(self, kind: str) -> None:
        waiting, held = self.waiters[kind], self.held[kind]
        while waiting and len(held) < self.limits[kind]:
            counts = Counter(owner for owner, _token in held)
            owners = list(dict.fromkeys(owner for owner, _token in waiting))
            # Round-robin ties in the order owners joined the queue.
            last = self.last_owner[kind]
            if last in owners:
                pivot = owners.index(last) + 1
                owners = owners[pivot:] + owners[:pivot]
            if self.owners is not None:
                cap = self._owner_limit(kind)
                owners = [owner for owner in owners if counts[owner] < cap]
                if not owners:
                    return
            owner = min(owners, key=lambda item: counts[item])
            key = next(key for key in waiting if key[0] == owner)
            future = waiting.pop(key)
            held[key] = True
            self.last_owner[kind] = owner
            if not future.done():
                future.set_result(True)

    def release(self, kind: str, owner: str, token: str) -> None:
        if owner in self.quarantined:
            # Cancelling an engine task does not confirm namespace cleanup.
            return
        # Tombstones also cover cancellation racing a not-yet-executed acquire.
        self.released.add((kind, owner, token))
        key = (owner, token)
        self.held[kind].pop(key, None)
        future = self.waiters[kind].pop(key, None)
        if future is not None and not future.done():
            future.set_result(False)
        self._drain(kind)

    def _owner_limit(self, kind: str) -> int:
        count = len(self.owners or ())
        # Ceiling plus global least-occupied selection shares indivisible
        # permits, including limits smaller than the number of models.
        return (self.limits[kind] + count - 1) // count if count else 0

    def close_owner(self, owner: str, successor: str | None = None) -> None:
        """After runtime shutdown, atomically hand its reservation to a successor."""
        if owner in self.quarantined:
            raise RuntimeError("cannot release an evaluation owner with unconfirmed node cleanup")
        if successor is not None:
            if self.owners is None or owner not in self.owners:
                raise ValueError("replacement requires a reserved evaluation owner")
            if successor in self.owners or successor in self.closed:
                raise ValueError("successor must be a new evaluation owner")
        if self.owners is not None and owner in self.owners:
            index = self.owners.index(owner)
            if successor is None:
                self.owners.pop(index)
            else:
                self.owners[index] = successor
        self.closed.add(owner)
        for kind in self.limits:
            for key in list(self.waiters[kind]):
                if key[0] == owner:
                    future = self.waiters[kind].pop(key)
                    if not future.done():
                        future.set_result(False)
            for key in list(self.held[kind]):
                if key[0] == owner:
                    del self.held[kind][key]
            self._drain(kind)
        # The owner tombstone now rejects every late request, so per-attempt
        # tombstones need not accumulate across a long checkpoint sweep.
        self.released = {key for key in self.released if key[1] != owner}

    def quarantine_owner(self, owner: str) -> bool:
        """Fence a failed node without lending its unconfirmed capacity."""
        self.closed.add(owner)
        self.quarantined.add(owner)
        for kind in self.limits:
            for key in list(self.waiters[kind]):
                if key[0] == owner:
                    future = self.waiters[kind].pop(key)
                    if not future.done():
                        future.set_result(False)
        # Keep both the owner's reservation and any held tokens. A failed
        # namespace cleanup is not evidence that its resources are available.
        # If quarantine consumes an entire admission budget, other owners
        # cannot make progress. Let the driver abort instead of waiting forever.
        return all(sum(key[0] in self.quarantined for key in self.held[kind]) < limit
                   for kind, limit in self.limits.items())

    def register_resource(self, owner: str, kind: str, handle: Any) -> None:
        self.resources.setdefault(owner, {"actors": [], "placement_groups": [], "processes": []})[kind].append(handle)

    def get_resources(self, owner: str) -> dict:
        return self.resources.get(owner, {"actors": [], "placement_groups": [], "processes": []})

    def update_progress(self, owner: str, payload: dict) -> None:
        self.progress[owner] = payload

    def snapshot(self) -> dict:
        return {
            "limits": dict(self.limits),
            "reserved_models": list(self.owners or ()),
            "per_model_limits": {
                kind: self._owner_limit(kind) for kind in self.limits
            } if self.owners is not None else None,
            "active": {kind: len(held) for kind, held in self.held.items()},
            "waiting": {kind: len(waiting) for kind, waiting in self.waiters.items()},
            "active_by_model": {
                kind: dict(Counter(owner for owner, _token in held))
                for kind, held in self.held.items()
            },
            "models": dict(self.progress),
        }


class RayEvaluationAdmission:
    """Async client; remote releases are acknowledged, including cancellation."""

    def __init__(self, coordinator: Any, owner: str):
        self.coordinator = coordinator
        self.owner = owner

    async def acquire(self, kind: str, token: str) -> None:
        try:
            granted = await self.coordinator.acquire.remote(kind, self.owner, token)
            if not granted:
                raise asyncio.CancelledError("evaluation runtime closed")
        except BaseException:
            await asyncio.shield(self.release(kind, token))
            raise

    async def release(self, kind: str, token: str) -> None:
        await self.coordinator.release.remote(kind, self.owner, token)

    @staticmethod
    def token() -> str:
        return uuid4().hex


def select_evaluation_nodes(nodes: list[dict], available: dict, count: int, gpus: int, replica_gpus: int) -> list[dict]:
    if min(count, gpus, replica_gpus) < 1:
        raise ValueError("node and GPU counts must be positive")
    if replica_gpus > gpus or gpus % replica_gpus:
        raise ValueError("node_parallel requires the TP/DP/PP replica footprint to divide rollout.n_gpus_per_node")
    candidates = []
    for node in sorted(nodes, key=lambda row: (row.get("NodeManagerAddress", ""), row.get("NodeID", ""))):
        node_id = node.get("NodeID")
        resources = available.get(node_id, {})
        # VERL standalone pools reserve two CPUs per GPU; the owner needs one.
        if node.get("Alive") and resources.get("GPU", 0) >= gpus and resources.get("CPU", 0) >= 2 * gpus + 1:
            node_key = "node:" + node["NodeManagerAddress"]
            if node_key not in node.get("Resources", {}):
                raise ValueError(f"Ray node {node_id} has no hard-affinity resource {node_key}")
            candidates.append({"node_id": node_id, "node_resource": node_key, "baseline_gpu": float(resources["GPU"])})
    if len(candidates) < count:
        raise ValueError(f"node_parallel needs {count} available GPU nodes with {gpus} GPUs and {2 * gpus + 1} CPUs each; found {len(candidates)}")
    return candidates[:count]
