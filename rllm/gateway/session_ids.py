"""Logical and physical Gateway session identity helpers."""

from __future__ import annotations

import re

_RETRY_SESSION_SUFFIX_RE = re.compile(
    r"^(?P<logical>.+)~retry-(?P<attempt>(?:[2-9]|[1-9][0-9]+))$"
)


def retry_session_id(logical_session_id: str, attempt: int) -> str:
    """Return a stable physical Gateway session id for one rollout attempt.

    The first attempt keeps the historical ``task_id:slot`` id. Retries use a
    distinct trace-store key so a late request from an abandoned attempt cannot
    collide with the next attempt's turn sequence.
    """

    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt <= 0:
        raise ValueError(f"attempt must be a positive integer, got {attempt!r}")
    if attempt == 1:
        return logical_session_id
    return f"{logical_session_id}~retry-{attempt}"


def split_rollout_session_id(session_id: str) -> tuple[str, int]:
    """Resolve physical session ids back to their logical group and slot."""

    match = _RETRY_SESSION_SUFFIX_RE.fullmatch(session_id)
    logical_session_id = match.group("logical") if match is not None else session_id
    group_id, separator, slot_text = logical_session_id.rpartition(":")
    if not separator:
        raise ValueError(
            "rollout session id must use task_id:rollout_idx, got "
            f"{session_id!r}"
        )
    try:
        slot = int(slot_text)
    except ValueError as exc:
        raise ValueError(
            "rollout session id must end in an integer rollout_idx, got "
            f"{session_id!r}"
        ) from exc
    if slot < 0:
        raise ValueError(f"rollout_idx must be non-negative, got {session_id!r}")
    return group_id, slot


def task_group_session_ids(
    task_ids: list[str],
    group_size: int,
    *,
    max_attempts: int = 1,
) -> list[str]:
    """Enumerate every physical session a set of rollout groups can create."""

    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 0:
        raise ValueError(f"group_size must be a positive integer, got {group_size!r}")
    if (
        isinstance(max_attempts, bool)
        or not isinstance(max_attempts, int)
        or max_attempts <= 0
    ):
        raise ValueError(
            f"max_attempts must be a positive integer, got {max_attempts!r}"
        )
    return [
        retry_session_id(f"{task_id}:{slot}", attempt)
        for task_id in task_ids
        for slot in range(group_size)
        for attempt in range(1, max_attempts + 1)
    ]


__all__ = [
    "retry_session_id",
    "split_rollout_session_id",
    "task_group_session_ids",
]
