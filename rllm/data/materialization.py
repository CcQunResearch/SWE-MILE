"""Shared bounded thread-pool helpers for task-directory materializers."""

from __future__ import annotations

import os
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TypeVar

_Input = TypeVar("_Input")
_Output = TypeVar("_Output")

DEFAULT_MATERIALIZATION_WORKERS = 16
MAX_MATERIALIZATION_WORKERS = 64
MATERIALIZATION_WORKERS_ENV = "RLLM_MATERIALIZATION_WORKERS"


def resolve_materialization_workers(
    task_count: int,
    requested: int | None = None,
    *,
    legacy_env_names: Sequence[str] = (),
) -> int:
    """Resolve one validated worker count shared by every dataset builder.

    ``requested`` has highest priority.  Direct builder callers can otherwise
    use the unified environment variable.  Dataset-specific legacy variables
    remain optional fallbacks so existing launch commands keep working.
    """

    if isinstance(task_count, bool) or task_count < 0:
        raise ValueError("task_count must be a non-negative integer")
    if task_count == 0:
        return 1

    raw: object = requested
    source = "max_workers"
    if raw is None:
        source = MATERIALIZATION_WORKERS_ENV
        raw = os.environ.get(MATERIALIZATION_WORKERS_ENV)
    if raw in {None, ""}:
        for name in legacy_env_names:
            value = os.environ.get(name)
            if value not in {None, ""}:
                source = name
                raw = value
                break
    if raw in {None, ""}:
        raw = DEFAULT_MATERIALIZATION_WORKERS

    if isinstance(raw, bool):
        raise ValueError(
            f"{source} must be an integer in [1, {MAX_MATERIALIZATION_WORKERS}]"
        )
    try:
        workers = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source} must be an integer in [1, {MAX_MATERIALIZATION_WORKERS}]"
        ) from exc
    if not 1 <= workers <= MAX_MATERIALIZATION_WORKERS:
        raise ValueError(
            f"{source} must be an integer in [1, {MAX_MATERIALIZATION_WORKERS}]"
        )
    return min(workers, task_count)


def bounded_ordered_thread_map(
    function: Callable[[_Input], _Output],
    items: Iterable[_Input],
    *,
    workers: int,
    thread_name_prefix: str,
) -> Iterator[_Output]:
    """Apply ``function`` concurrently with at most ``workers`` queued items.

    Results are yielded in input order so registry rows and manifests remain
    deterministic.  The bounded queue is important for DeNovoSWE: individual
    JSONL rows may contain large base64 fixtures and must not all be retained
    by thousands of pending futures.
    """

    if workers <= 0:
        raise ValueError("workers must be positive")
    if workers == 1:
        for item in items:
            yield function(item)
        return

    iterator = iter(items)
    pending: deque[Future[_Output]] = deque()
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix=thread_name_prefix,
    ) as pool:
        for _ in range(workers):
            try:
                item = next(iterator)
            except StopIteration:
                break
            pending.append(pool.submit(function, item))

        while pending:
            future = pending.popleft()
            try:
                yield future.result()
            except BaseException:
                for queued in pending:
                    queued.cancel()
                raise
            try:
                item = next(iterator)
            except StopIteration:
                continue
            pending.append(pool.submit(function, item))


__all__ = [
    "DEFAULT_MATERIALIZATION_WORKERS",
    "MATERIALIZATION_WORKERS_ENV",
    "MAX_MATERIALIZATION_WORKERS",
    "bounded_ordered_thread_map",
    "resolve_materialization_workers",
]
