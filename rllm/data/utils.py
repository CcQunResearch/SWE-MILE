from __future__ import annotations
import uuid
from pathlib import Path
from typing import Any
from rllm.types import Task

def task_from_row(row: dict[str, Any], task_id: str) -> Task:
    """Build a :class:`Task` from a dataset row.

    Shared by the engine (per-task at rollout) and the sandbox warm-pool
    schedule so both derive the same ``env_key`` for a given row.
    """
    return Task(
        id=str(task_id),
        instruction=str(row.get("question", row.get("instruction", ""))),
        metadata=row,
        dataset_dir=Path("."),
    )

def interleave_tasks(batch: list[dict | Task], group_size: int) -> tuple[list[dict | Task], list[str]]:
    """Interleave each task ``group_size`` times; return ``(tasks, task_ids)`` with one shared
    id per group for GRPO grouping — the task's own ``id`` when truthy, else a uuid."""
    tasks: list[dict | Task] = []
    task_ids: list[str] = []
    for item in batch:
        item_id = item.id if isinstance(item, Task) else item.get("id")
        uid = str(item_id) if item_id else str(uuid.uuid4())
        for _ in range(group_size):
            tasks.append(item)
            task_ids.append(uid)
    return tasks, task_ids
