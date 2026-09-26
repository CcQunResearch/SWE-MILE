"""Precompute the remaining training task order as an ordered list of Tasks.

The on-policy trainer hands this list to a :class:`rllm.sandbox.warm_queue.WarmQueue`
exactly as eval does, so sandbox creation overlaps with rollout. The order is
reproduced from the live dataloader's checkpointed position: because
:class:`rllm.data.dataloader.StatefulTaskDataLoader`'s order is a pure function of
``(seed, epoch, dataset)``, a clone walked through pending replay tickets and
then the live cursor yields the same batches the live loop will train on,
without touching the live loader.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rllm.data.utils import interleave_tasks, task_from_row
from rllm.types import Task

if TYPE_CHECKING:
    from rllm.data.dataloader import StatefulTaskDataLoader


def _as_task(item: dict | Task) -> Task:
    """A schedule entry as a Task. Harbor rows are already Tasks; dict rows use the
    same conversion the engine applies at rollout, so the ``env_key`` matches."""
    return item if isinstance(item, Task) else task_from_row(item, str(item.get("id", "")))


def build_train_schedule(
    live_loader: StatefulTaskDataLoader,
    *,
    group_size: int,
    total_epochs: int,
    remaining_batches: int = -1,
) -> list[Task]:
    """The remaining training tasks in consumption order, GRPO copies included.

    Clones ``live_loader`` so the live iteration is untouched, then walks the
    remaining epochs the way the trainer does, expanding each batch through the
    same :func:`interleave_tasks` the backend uses. ``remaining_batches`` caps the
    walk in **loader-batch** units (``<= 0`` walks to the end of training).
    """
    preview_batches = getattr(live_loader, "preview_batches", None)
    if callable(preview_batches):
        batches = preview_batches(
            total_epochs=total_epochs,
            remaining_batches=remaining_batches,
        )
        schedule: list[Task] = []
        for batch in batches:
            interleaved, _ids = interleave_tasks(batch, group_size)
            schedule.extend(_as_task(item) for item in interleaved)
        return schedule

    clone = live_loader.clone()
    schedule: list[Task] = []
    emitted = 0
    while True:
        item = clone.next_dispatch(total_epochs)
        if item is None:
            break
        batch, ticket = item
        interleaved, _ids = interleave_tasks(batch, group_size)
        schedule.extend(_as_task(item) for item in interleaved)
        clone.mark_resolved(ticket)
        emitted += 1
        if 0 < remaining_batches <= emitted:
            return schedule
    return schedule


__all__ = ["build_train_schedule"]
