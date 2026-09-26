"""Frozen training-history view over the materialized SWE-rebench Python rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rllm.data.assets.swerebench_v2_filternorm_index import TASK_GROUP_COUNTS

FILTERNORM_DATASET_NAME = "swe-rebench-v2-filtered-verified-python-filternorm"


def excluded_task_ids(
    counts: Mapping[str, Sequence[Sequence[int]]],
) -> frozenset[str]:
    """Each run contributes (easy, hard, passed_uniform, other_complete).

    Unfinished/infrastructure attempts are absent. A task without a completed
    group stays in the dataset, as does any task with a non-easy/hard group.
    """
    excluded = set()
    for task_id, runs in counts.items():
        easy_hard = keep = 0
        for run in runs:
            if len(run) != 4 or any(type(n) is not int or n < 0 for n in run):
                raise ValueError(f"Invalid completed-group counts for {task_id}")
            easy_hard += run[0] + run[1]
            keep += run[2] + run[3]
        if easy_hard and not keep:
            excluded.add(task_id)
    return frozenset(excluded)


FILTERNORM_EXCLUDED_TASK_IDS = excluded_task_ids(TASK_GROUP_COUNTS)
FILTERNORM_ROWS = len(TASK_GROUP_COUNTS) - len(FILTERNORM_EXCLUDED_TASK_IDS)


def filter_filternorm_rows(rows: list[dict]) -> list[dict]:
    """Validate the frozen source membership, retaining row order and paths."""
    ids = [str(row.get("id") or "") for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(TASK_GROUP_COUNTS):
        raise RuntimeError(
            f"{FILTERNORM_DATASET_NAME} source task IDs differ from its frozen "
            "Python index; use the complete pinned SWE-rebench V2 source"
        )
    return [row for row in rows if row["id"] not in FILTERNORM_EXCLUDED_TASK_IDS]
