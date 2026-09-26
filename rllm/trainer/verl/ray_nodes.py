"""Current-job Ray placement discovery shared by training and evaluation."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def discover_current_job_gpu_node_ids(ray_module: Any | None = None) -> list[str]:
    """Return alive GPU actor nodes owned by the caller's Ray job only."""

    if ray_module is None:
        import ray as ray_module
    from ray.util.state import list_actors

    runtime_context = ray_module.get_runtime_context()
    raw_job_id = runtime_context.get_job_id()
    job_id = raw_job_id.hex() if hasattr(raw_job_id, "hex") else str(raw_job_id)
    actors = list_actors(
        filters=[("job_id", "=", job_id), ("state", "=", "ALIVE")],
        detail=True,
        limit=10_000,
    )

    def field(item: Any, name: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(name, default)
        return getattr(item, name, default)

    node_ids: list[str] = []
    for actor in actors:
        resources = field(actor, "required_resources", {}) or {}
        if not isinstance(resources, dict):
            try:
                resources = dict(resources)
            except (TypeError, ValueError):
                resources = {}
        requires_gpu = any(
            "GPU" in str(key).upper() and float(value) > 0
            for key, value in resources.items()
        )
        if not requires_gpu:
            continue
        node_id = str(field(actor, "node_id", "") or "")
        if node_id and node_id not in node_ids:
            node_ids.append(node_id)
    if not node_ids:
        raise RuntimeError(
            "Ray has no alive GPU actors in the current job; refusing to deploy "
            "MiniSandbox services to unowned cluster nodes"
        )
    alive_nodes = {
        str(item.get("NodeID"))
        for item in ray_module.nodes()
        if item.get("Alive") and item.get("NodeID")
    }
    missing = [node_id for node_id in node_ids if node_id not in alive_nodes]
    if missing:
        raise RuntimeError(
            "current-job GPU actor nodes disappeared before MiniSandbox startup: "
            + ", ".join(missing)
        )
    logger.info(
        "MiniSandbox node scope resolved from current Ray job %s: %s",
        job_id,
        ", ".join(node_ids),
    )
    return node_ids


__all__ = ["discover_current_job_gpu_node_ids"]
