from __future__ import annotations

from typing import Any

from rllm.types import Action, Step

TOOL_CALL_REWARD_KEY = "tool_call_reward"

_TOOL_ALIASES = {
    "bash": "execute_bash",
    "terminal": "execute_bash",
    "file_editor": "str_replace_editor",
    "finish": "submit",
}

_VALID_TOOL_NAMES = {
    "execute_bash",
    "str_replace_editor",
    "submit",
    "search",
    "read_file",
    "edit_file",
    "task_tracker",
}


def normalize_tool_name(tool_name: Any) -> str:
    normalized = str(tool_name or "").strip()
    if ":" in normalized:
        normalized = normalized.split(":", 1)[0]
    if "." in normalized:
        normalized = normalized.split(".")[-1]
    return _TOOL_ALIASES.get(normalized, normalized)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _metadata_candidates(metadata: Any) -> list[dict[str, Any]]:
    meta = _as_dict(metadata)
    candidates = [meta]
    agent_meta = _as_dict(meta.get("agent_step_metadata"))
    if agent_meta:
        candidates.append(agent_meta)
    return candidates


def _first_metadata_value(metadata: Any, key: str) -> Any:
    for candidate in _metadata_candidates(metadata):
        if key in candidate:
            return candidate[key]
    return None


def _action_name(action: Any, metadata: Any) -> str:
    if isinstance(action, Action):
        action = action.action
    if isinstance(action, dict):
        name = action.get("name") or action.get("tool") or action.get("function")
        if name:
            return normalize_tool_name(name)

    raw_tool_call = _as_dict(_first_metadata_value(metadata, "raw_tool_call"))
    if raw_tool_call.get("name"):
        return normalize_tool_name(raw_tool_call.get("name"))

    raw_tool_names = _first_metadata_value(metadata, "raw_tool_names")
    if isinstance(raw_tool_names, list) and len(raw_tool_names) == 1:
        return normalize_tool_name(raw_tool_names[0])
    return ""


def existing_tool_call_reward(step: Step) -> float | None:
    value = _first_metadata_value(step.metadata, TOOL_CALL_REWARD_KEY)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def tool_call_parse_metadata_missing(step: Step) -> bool:
    return _first_metadata_value(step.metadata, "parse_status") is None


def compute_tool_call_reward(
    step: Step,
    *,
    correct_reward: float = 1.0,
    incorrect_reward: float = 0.0,
) -> float:
    parse_status = _first_metadata_value(step.metadata, "parse_status")
    if parse_status != "ok":
        return float(incorrect_reward)

    # Native Codeflow records schema/policy validation separately from the
    # OpenAI tool-call parse status.  Treat a well-formed call that violates
    # the advertised tool contract as the same existing format error rather
    # than rewarding it merely for naming a real tool.
    try:
        from rllm.harnesses.action_event import (
            ActionCategory,
            ValidationStatus,
            get_action_event,
        )

        event = get_action_event(step)
    except Exception:
        event = None
    if event is not None and (
        event.validation_status != ValidationStatus.OK
        or event.action_category == ActionCategory.INVALID
        or bool(event.policy_violations)
    ):
        return float(incorrect_reward)

    name = _action_name(step.action, step.metadata)
    if name in _VALID_TOOL_NAMES:
        return float(correct_reward)
    return float(incorrect_reward)


def get_or_compute_tool_call_reward(
    step: Step,
    *,
    correct_reward: float = 1.0,
    incorrect_reward: float = 0.0,
    prefer_existing: bool = True,
) -> float:
    if prefer_existing:
        existing = existing_tool_call_reward(step)
        if existing is not None:
            return existing
    return compute_tool_call_reward(step, correct_reward=correct_reward, incorrect_reward=incorrect_reward)


def annotate_step_tool_call_reward(
    step: Step,
    *,
    correct_reward: float = 1.0,
    incorrect_reward: float = 0.0,
    overwrite: bool = False,
) -> float:
    if not overwrite:
        existing = existing_tool_call_reward(step)
        if existing is not None:
            step.info.setdefault("tool_call_reward_missing_metadata", tool_call_parse_metadata_missing(step))
            return existing

    reward = compute_tool_call_reward(step, correct_reward=correct_reward, incorrect_reward=incorrect_reward)
    step.info[TOOL_CALL_REWARD_KEY] = reward
    step.info["tool_call_reward_missing_metadata"] = tool_call_parse_metadata_missing(step)
    return reward
