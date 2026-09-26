"""Codeflow native tool execution for SWE-MILE."""

from __future__ import annotations

import base64
import copy
import hashlib
import inspect
import json
import logging
import math
import posixpath
import re
import secrets
import shlex
import tempfile
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from rllm.agents.system_prompts import (
    CODEFLOW_BASH_ONLY_SYSTEM_PROMPT_NATIVE_TOOL_CALL,
    CODEFLOW_DENOVOSWE_BASH_ONLY_SYSTEM_PROMPT_NATIVE_TOOL_CALL,
    CODEFLOW_DENOVOSWE_SYSTEM_PROMPT_NATIVE_TOOL_CALL,
    CODEFLOW_DENOVOSWE_USER_PROMPT,
    CODEFLOW_SYSTEM_PROMPT_NATIVE_TOOL_CALL,
    CODEFLOW_USER_PROMPT,
)
from rllm.harnesses.action_event import (
    ACTION_EVENT_METADATA_KEY,
    ActionEvent,
    ChangedFile,
    ExecutionStatus,
    FileExposure,
    ParseStatus,
    RepoFileState,
    RepoSnapshot,
    RepoStateStatus,
    ValidationStatus,
    action_category_for,
    changed_files_between,
    extract_exposure,
    is_probable_test_path,
)
from rllm.harnesses.codeflow_shell_policy import (
    CODEFLOW_TOOL_BEHAVIOR_ORDER,
    DEFAULT_CODEFLOW_TOOL_MODE,
    DEFAULT_CODEFLOW_TOOL_RESTRICTED_MODE,
    ShellBehaviorAnalysis,
    analyze_repository_shell_behaviors,
    canonicalize_shell_command,
    detect_denovo_detached_process_violation,
    detect_preconfigured_environment_violation,
    detect_repo_generation_package_violation,
    normalize_codeflow_tool_mode,
    normalize_codeflow_tool_restricted_mode,
    shell_file_attributions,
)
from rllm.harnesses.repository_state import helper_source as repository_helper_source
from rllm.sandbox.protocol import Sandbox
from rllm.sandbox.sandboxed_flow import SandboxedAgentFlow
from rllm.sandbox.structured_exec import (
    StructuredCommandProtocolError,
    frame_structured_command,
    parse_structured_command_output,
)
from rllm.types import (
    AgentConfig,
    AgentFlowCancellation,
    AgentFlowDeadline,
    Episode,
    RolloutInfrastructureError,
    Step,
    Task,
    Trajectory,
    agent_flow_cancellation,
)
from rllm.utils.tool_call_reward import annotate_step_tool_call_reward
from rllm.workflows.workflow import TerminationReason

logger = logging.getLogger(__name__)

SUPPORTED_SCAFFOLDS = {"codeflow"}
SUPPORTED_PROTOCOLS = {"native_tool_call"}
OUTPUT_LIMIT = 16_000
CODEFLOW_SEARCH_OUTPUT_LIMIT = 12_000
CODEFLOW_READ_OUTPUT_LIMIT = 12_000
CODEFLOW_BASH_OUTPUT_LIMIT = 16_000
CODEFLOW_SHORT_OUTPUT_LIMIT = 4_000
CODEFLOW_MAX_RENDERED_LINE_CHARS = 2_000
CODEFLOW_DEFAULT_SEARCH_RESULTS = 20
_CODEFLOW_INLINE_HELPER_MAX_BYTES = 3072
_CODEFLOW_HELPER_CACHE_ATTRIBUTE = "_rllm_codeflow_helper_hashes"
# Sandbox adapters both guarantee a writable /tmp.
# /var/tmp is read-only in some hardened workers, which used to make every
# repository checkpoint fail before the action was even executed.
_CODEFLOW_CONTROL_TMPDIR = "/tmp"
_CODEFLOW_HELPER_DIR = f"{_CODEFLOW_CONTROL_TMPDIR}/rllm-codeflow-helpers"
_CODEFLOW_CHECKPOINT_PREFIX = (
    f"{_CODEFLOW_CONTROL_TMPDIR}/rllm-codeflow-checkpoint-"
)
_CODEFLOW_PROTECTED_TEST_GUIDANCE = (
    "Protected verifier tests cannot be modified. Fix production code instead. "
    "You may add your own tests outside protected paths."
)
_CODEFLOW_HELPER_RETRYABLE_TRANSPORT_MARKERS = (
    "exit marker missing",
    "missing exit marker",
    "connection reset",
    "connection aborted",
    "remote disconnected",
    "read timed out",
    "readerror",
    "transport error",
    "response uncertain",
    "deadline exceeded",
    "temporarily unavailable",
)
NATIVE_TOOL_CALL_PROTOCOL = "native_tool_call"
_CONTEXT_METADATA_KEY = "rllm_context"
_DISCARD_CONTEXT_TERMINAL = "discard_context_terminal"
CODEFLOW_EVAL_NO_PROGRESS_POLICY_VERSION = "codeflow_eval_no_progress_v1"
CODEFLOW_EVAL_NO_PROGRESS_THRESHOLD = 3


def _codeflow_eval_guard_enabled(task: Task) -> bool:
    metadata = task.metadata.get("rllm") or {}
    return (
        metadata.get("agent_environment_policy") == "preconfigured_eval_v1"
        or metadata.get("eval_no_progress") is True
    )



CODEFLOW_NATIVE_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search repository text, symbols, file names, or directory names without modifying files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["text", "symbol", "file", "directory"],
                        "description": "Search mode. text is a case-sensitive literal search; symbol matches identifier boundaries.",
                    },
                    "query": {"type": "string", "description": "Literal text, symbol, file-name fragment, or directory-name fragment to find."},
                    "path": {"type": "string", "description": "Repository-relative path or absolute path inside the repository. Defaults to the repository root."},
                    "include_globs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional glob patterns applied to repository-relative file paths.",
                    },
                    "context_lines": {"type": "integer", "minimum": 0, "maximum": 20, "description": "Context lines around text or symbol matches. Defaults to 2."},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 200, "default": 20, "description": "Maximum matches to return. Defaults to 20."},
                    "offset": {"type": "integer", "minimum": 0, "default": 0, "description": "Zero-based match offset for pagination. Defaults to 0."},
                },
                "required": ["mode", "query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file content, the current git diff, or git status without modifying the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["file", "diff", "status"], "description": "Read operation."},
                    "path": {"type": "string", "description": "Repository-relative path or absolute path inside the repository. Required for file mode and optional for diff/status."},
                    "start_line": {"type": "integer", "minimum": 1, "description": "First 1-based line for file mode. Defaults to 1."},
                    "end_line": {"type": "integer", "minimum": 1, "description": "Last 1-based line for file mode, inclusive. Defaults to 200 lines from start_line."},
                },
                "required": ["mode"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Create, replace, insert, delete, atomically apply a patch, or undo the most recent successful edit_file action.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["replace", "insert", "create", "delete", "apply_patch", "undo"], "description": "Edit operation."},
                    "path": {"type": "string", "description": "Repository-relative path or absolute path inside the repository. Optional for apply_patch and undo."},
                    "old_text": {"type": "string", "description": "Exact text to replace."},
                    "new_text": {"type": "string", "description": "Replacement, insertion, or new-file content."},
                    "expected_count": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Exact number of old_text occurrences to replace. Defaults to 1.",
                    },
                    "line": {"type": "integer", "minimum": 0, "description": "Insert after this 1-based line; 0 inserts at the beginning."},
                    "patch": {"type": "string", "description": "Git-compatible patch for apply_patch; may modify multiple files."},
                },
                "required": ["mode"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": (
                "Execute a bash command for tests, dependency installation, Python, or operations not supported by "
                "search/read_file/edit_file. Git-visible repository changes are rejected and rolled back; put temporary "
                "scripts and outputs under /tmp. Do not use it for supported search, read, or edit operations. Repository "
                "searches via grep, rg, find, fd, git grep, and similar commands are rejected; use search instead."
            ),
            "parameters": {'type': 'object', 'properties': {'command': {'type': 'string', 'description': 'The bash command to execute.'}, 'timeout': {'type': 'number', 'description': 'Optional command timeout in seconds.'}}, 'required': ['command']},
        },
    },
    {'type': 'function', 'function': {'name': 'submit', 'description': 'Submit the current repository state when the issue is resolved.', 'parameters': {'type': 'object', 'properties': {}, 'required': []}}},
]
CODEFLOW_NATIVE_TOOL_SCHEMAS[-1]["function"]["description"] = (
    "Submit the current repository state when the issue is resolved. Call submit({}) with no arguments."
)
CODEFLOW_NATIVE_TOOL_SCHEMAS[-1]["function"]["parameters"]["additionalProperties"] = False
CODEFLOW_NATIVE_TOOL_SCHEMAS[3]["function"]["parameters"]["properties"]["timeout"].update(
    {
        "description": "Optional command timeout in seconds.",
    }
)
CODEFLOW_NATIVE_TOOL_NAMES = {
    tool["function"]["name"]
    for tool in CODEFLOW_NATIVE_TOOL_SCHEMAS
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict)
}


@dataclass
class ParsedAction:
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    thought: str = ""
    status: str = "not_parsed"
    reason: str = ""
    protocol: str = ""


@dataclass
class CodeFlowUndoEntry:
    kind: str
    paths: tuple[str, ...]
    path: str = ""
    before_exists: bool = False
    before_content: str = ""
    after_digest: str = ""
    patch: str = ""


@dataclass
class CodeFlowValidationResult:
    status: ValidationStatus
    normalized_arguments: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    subtype: str = ""
    policy_violations: list[str] = field(default_factory=list)
    shell_analysis: ShellBehaviorAnalysis | None = None


@dataclass
class CodeFlowExecutionResult:
    observation: str
    done: bool = False
    status: ExecutionStatus = ExecutionStatus.NOT_RUN
    error: str | None = None
    exit_code: int | None = None
    duration: float = 0.0
    attempted_changed_files: list[ChangedFile] = field(default_factory=list)
    policy_violations: list[str] = field(default_factory=list)
    instrumentation_errors: list[str] = field(default_factory=list)
    fatal: bool = False
    observation_stats: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CodeFlowRepoCheckpoint:
    path: str
    baseline_ref: str


@dataclass(frozen=True)
class CodeFlowRestoreError:
    message: str
    kind: str = "operation"
    detail: str | None = None

    def __str__(self) -> str:
        return self.message


def _normalize_scaffold(scaffold: str, default: str = "codeflow") -> str:
    value = str(scaffold or "").strip().lower()
    if not value:
        return default
    if value not in SUPPORTED_SCAFFOLDS:
        raise ValueError(f"Unsupported SWE scaffold: {scaffold}. Expected one of {sorted(SUPPORTED_SCAFFOLDS)}.")
    return value


def _default_protocol_for_scaffold(scaffold: str) -> str:
    _normalize_scaffold(scaffold)
    return NATIVE_TOOL_CALL_PROTOCOL


def _normalize_protocol(protocol: str, scaffold: str) -> str:
    value = str(protocol or "").strip()
    if not value:
        value = _default_protocol_for_scaffold(scaffold)
    if value not in SUPPORTED_PROTOCOLS:
        raise ValueError(f"Unsupported SWE protocol: {protocol}. Expected one of {sorted(SUPPORTED_PROTOCOLS)}.")
    normalized_scaffold = _normalize_scaffold(scaffold)
    if normalized_scaffold == "codeflow" and value != NATIVE_TOOL_CALL_PROTOCOL:
        raise ValueError("The codeflow scaffold currently supports only the native_tool_call protocol.")
    if value == NATIVE_TOOL_CALL_PROTOCOL and normalized_scaffold not in {"sweagent", "codeflow"}:
        raise ValueError("native_tool_call protocol is currently supported only for the sweagent and codeflow scaffolds.")
    return value


def _normalize_tool_name(tool_name: str) -> str:
    normalized = str(tool_name or "").strip()
    if ":" in normalized:
        normalized = normalized.split(":", 1)[0]
    if "." in normalized:
        normalized = normalized.split(".")[-1]
    return normalized


def _canonical_native_tool_name(tool_name: str, scaffold: str = "codeflow") -> str:
    _normalize_scaffold(scaffold)
    return _normalize_tool_name(tool_name)


def _copy_native_tool_schemas(scaffold: str) -> list[dict[str, Any]]:
    _normalize_scaffold(scaffold)
    return copy.deepcopy(CODEFLOW_NATIVE_TOOL_SCHEMAS)


def _codeflow_file_tool_policy_prompt(
    restricted_mode: Sequence[str],
) -> str:
    restricted = set(restricted_mode)
    lines = [
        "- Prefer the dedicated search, read_file, and edit_file tools for repository exploration, reading, and editing.",
    ]
    policies = {
        "search": (
            "- Repository searching through execute_bash is restricted. Do not use grep, rg, find, fd, git grep, ls/tree, or equivalent shell/Python searches on repository files; use search instead.",
            "- Use search as the default for repository searches and directory exploration. Avoid grep, rg, find, fd, "
            "git grep, ls/tree, or equivalent shell/Python searches when search can perform the operation. "
            "Bash search is a fallback only when the dedicated tool cannot express the required operation.",
        ),
        "read": (
            "- Repository reading through execute_bash is restricted. Do not use cat, head/tail, sed/awk, Python file "
            "reads, git diff/status/show, or equivalent commands on repository files; use read_file instead.",
            "- Use read_file as the default for repository file content, diffs, and status. Avoid cat, head/tail, sed/awk, "
            "Python file reads, git diff/status/show, or equivalent commands when read_file can perform the operation. "
            "Bash reading is a fallback only when the dedicated tool cannot express the required operation.",
        ),
        "edit": (
            "- Repository editing through execute_bash is restricted. Every Git-visible repository change must use edit_file; Bash mutation attempts are rejected and rolled back.",
            "- Use edit_file as the default for creating, modifying, and deleting repository files. Avoid shell redirection, "
            "sed -i, Python file writes, or equivalent commands when edit_file can perform the operation. "
            "Bash editing is a fallback only when the dedicated tool cannot express the required operation.",
        ),
    }
    for behavior in CODEFLOW_TOOL_BEHAVIOR_ORDER:
        lines.append(policies[behavior][0 if behavior in restricted else 1])
    lines.append(
        "- Use execute_bash primarily for tests, builds, and commands not covered by the dedicated tools. Put temporary scripts "
        "and outputs under /tmp. Searches and reads outside the repository and filtering command output remain allowed. "
        "Follow the protected-test, Git HEAD, environment, download, timeout, and sandbox boundary rules."
    )
    return "\n".join(lines)


def _configure_codeflow_tool_schemas(
    schemas: list[dict[str, Any]],
    restricted_mode: Sequence[str],
    command_timeout: float,
) -> None:
    restricted = set(restricted_mode)
    labels = {
        "search": "repository search",
        "read": "repository file reading",
        "edit": "Git-visible repository editing",
    }
    restricted_text = ", ".join(
        labels[value] for value in CODEFLOW_TOOL_BEHAVIOR_ORDER if value in restricted
    ) or "none of repository search, reading, or editing"
    allowed_text = ", ".join(
        labels[value] for value in CODEFLOW_TOOL_BEHAVIOR_ORDER if value not in restricted
    ) or "none of repository search, reading, or editing"
    for tool in schemas:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict) and function.get("name") == "execute_bash":
            function["description"] = (
                "Execute Bash primarily for tests, builds, and commands not covered by dedicated tools. "
                "Prefer search, read_file, and edit_file for repository operations. "
                f"Restricted through Bash: {restricted_text}; use the corresponding dedicated tools. "
                f"Fallback only when the dedicated tool cannot express the operation: {allowed_text}. "
                "Temporary files under /tmp and external-output filtering remain allowed. "
                "Follow the timeout, protected-test, Git HEAD, environment, download, and sandbox safety rules."
            )

            function["parameters"]["properties"]["timeout"].update({
                "description": f"Optional command timeout in seconds. Defaults to {command_timeout:g} seconds.",
                "default": command_timeout,
            })


def _configure_codeflow_tool_mode_schemas(
    schemas: list[dict[str, Any]],
    tool_mode: str,
) -> None:
    if tool_mode != "bash_only":
        return
    schemas[:] = [
        tool
        for tool in schemas
        if isinstance(tool, dict)
        and isinstance(tool.get("function"), dict)
        and tool["function"].get("name") in {"execute_bash", "submit"}
    ]
    for tool in schemas:
        function = tool.get("function")
        if isinstance(function, dict) and function.get("name") == "execute_bash":
            function["description"] = (
                "Execute Bash inside the repository sandbox for all repository "
                "exploration, search, reading, working-tree editing, diff inspection, "
                "and tests. Git-visible working-tree edits are allowed. Protected-test, "
                "Git HEAD/refs, environment, download, timeout, and sandbox safety rules "
                "remain in force. Put unrelated temporary files under /tmp."
            )


def _model_dump(obj: Any) -> dict[str, Any] | None:
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        return dump(exclude_none=True)
    return None


def _tool_call_to_dict(tool_call: Any) -> dict[str, Any]:
    if isinstance(tool_call, dict):
        return dict(tool_call)

    dumped = _model_dump(tool_call)
    if dumped is not None:
        return dumped

    function = getattr(tool_call, "function", None)
    if isinstance(function, dict):
        function_dict = dict(function)
    else:
        function_dict = {
            "name": getattr(function, "name", ""),
            "arguments": getattr(function, "arguments", "{}"),
        }

    return {
        "id": getattr(tool_call, "id", ""),
        "type": getattr(tool_call, "type", "function"),
        "function": function_dict,
    }


def _assistant_message_to_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        data = dict(message)
    else:
        dumped = _model_dump(message)
        if dumped is not None:
            data = dumped
        else:
            data = {
                "role": getattr(message, "role", "assistant"),
                "content": getattr(message, "content", None),
            }
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                data["tool_calls"] = [_tool_call_to_dict(tool_call) for tool_call in tool_calls]

    data.setdefault("role", "assistant")
    if data.get("tool_calls"):
        data["tool_calls"] = [_tool_call_to_dict(tool_call) for tool_call in data["tool_calls"]]
    return {key: value for key, value in data.items() if value is not None}


def _response_context_metadata(response: Any) -> dict[str, Any]:
    """Read the gateway's OpenAI-compatible response extension."""
    if isinstance(response, dict):
        value = response.get(_CONTEXT_METADATA_KEY)
    else:
        value = getattr(response, _CONTEXT_METADATA_KEY, None)
        if value is None:
            extras = getattr(response, "model_extra", None)
            value = extras.get(_CONTEXT_METADATA_KEY) if isinstance(extras, dict) else None
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    return dict(value) if isinstance(value, dict) else {}


def _mark_last_observation_unconsumed(steps: list[Step], context: dict[str, Any]) -> None:
    """Keep the terminal observation for audit without claiming model exposure."""
    if not steps:
        return
    step = steps[-1]
    step.info["tool_observation_consumed_by_model"] = False
    step.info["context_limit"] = dict(context)
    event = step.info.get(ACTION_EVENT_METADATA_KEY)
    if isinstance(event, dict):
        event["exposed_files"] = []
        event["exposed_line_ranges"] = []
        event["file_exposures"] = []
        event["visible_characters"] = 0
        event["output_truncated"] = False


def _message_content(message: dict[str, Any]) -> str:
    content = message.get("content")
    if content is None:
        return ""
    return str(content)


def _codeflow_native_correction_message(observation: str) -> dict[str, str]:
    """Represent a format correction as part of the current tool cycle.

    Qwen3.5 intentionally re-renders when a new user query arrives so it can
    apply thinking-retention rules. A missing native tool call is not a new
    task query, but a plain user correction looks like one and breaks the
    cumulative-token prefix. The renderer treats this wrapper as a non-query
    tool response even though no real tool-call id exists.
    """
    content = str(observation or "").strip()
    return {
        "role": "user",
        "content": f"<tool_response>\n{content}\n</tool_response>",
    }


def _tool_call_id(tool_call: Any, turn: int, index: int) -> str:
    data = _tool_call_to_dict(tool_call)
    return str(data.get("id") or f"call_{turn}_{index}")


def _summarize_raw_tool_call(tool_call: Any) -> dict[str, Any]:
    data = _tool_call_to_dict(tool_call)
    function = data.get("function") if isinstance(data.get("function"), dict) else {}
    arguments = function.get("arguments", "")
    arguments_text = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
    if len(arguments_text) > 1000:
        arguments_text = arguments_text[:1000] + f"... [truncated, {len(arguments_text)} chars total]"
    return {
        "id": data.get("id", ""),
        "type": data.get("type", "function"),
        "name": function.get("name", ""),
        "arguments": arguments_text,
    }


def _native_tool_call_to_action(tool_call: Any, scaffold: str = "codeflow") -> ParsedAction:
    data = _tool_call_to_dict(tool_call)
    function = data.get("function")
    if not isinstance(function, dict):
        return ParsedAction(status="parse_error", reason="missing_tool_call_function", protocol=NATIVE_TOOL_CALL_PROTOCOL)

    normalized_scaffold = _normalize_scaffold(scaffold)
    name = _canonical_native_tool_name(function.get("name", ""), normalized_scaffold)
    if not name:
        return ParsedAction(status="parse_error", reason="empty_action", protocol=NATIVE_TOOL_CALL_PROTOCOL)
    allowed_names = CODEFLOW_NATIVE_TOOL_NAMES
    if name not in allowed_names:
        return ParsedAction(name=name, status="parse_error", reason="unknown_tool", protocol=NATIVE_TOOL_CALL_PROTOCOL)

    raw_arguments = function.get("arguments", "{}")
    if raw_arguments in (None, ""):
        arguments: Any = {}
    elif isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return ParsedAction(name=name, status="parse_error", reason="malformed_json_tool_arguments", protocol=NATIVE_TOOL_CALL_PROTOCOL)
    else:
        arguments = raw_arguments

    if not isinstance(arguments, dict):
        return ParsedAction(name=name, status="parse_error", reason="tool_arguments_not_object", protocol=NATIVE_TOOL_CALL_PROTOCOL)

    return ParsedAction(name=name, arguments=arguments, status="ok", protocol=NATIVE_TOOL_CALL_PROTOCOL)


def _truncate(output: str, limit: int = OUTPUT_LIMIT) -> str:
    output = str(output or "")
    if len(output) <= limit:
        return output
    return output[:limit] + f"\n... [truncated, {len(output)} chars total]"


def _compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _head_tail_text(value: str, limit: int) -> tuple[str, bool]:
    """Fit text exactly, preserving both ends and an explicit omission marker."""
    value = str(value or "")
    if len(value) <= limit:
        return value, False
    if limit <= 0:
        return "", True
    omitted = len(value) - limit
    marker = f"\n... [truncated, {omitted} chars omitted]\n"
    available = max(0, limit - len(marker))
    while True:
        head = (available + 1) // 2
        tail = available - head
        omitted = len(value) - head - tail
        marker = f"\n... [truncated, {omitted} chars omitted]\n"
        adjusted = max(0, limit - len(marker))
        if adjusted == available:
            break
        available = adjusted
    rendered = value[:head] + marker + (value[-tail:] if tail else "")
    return (rendered if len(rendered) <= limit else value[:limit]), True


def _clip_rendered_line(
    value: str,
    *,
    limit: int = CODEFLOW_MAX_RENDERED_LINE_CHARS,
    match_start: int | None = None,
) -> tuple[str, bool]:
    """Clip one physical line without introducing synthetic line breaks."""
    value = str(value or "")
    if len(value) <= limit:
        return value, False
    if limit <= 0:
        return "", True
    if match_start is None:
        available = max(0, limit - 64)
        while True:
            head = (available + 1) // 2
            tail = available - head
            omitted = len(value) - head - tail
            marker = f"...[{omitted} chars omitted]..."
            rendered = value[:head] + marker + (value[-tail:] if tail else "")
            if len(rendered) <= limit:
                return rendered, True
            available = max(0, available - (len(rendered) - limit))

    center = max(0, min(len(value), int(match_start)))
    available = max(0, limit - 64)
    while True:
        start = max(0, min(len(value) - available, center - available // 2))
        end = min(len(value), start + available)
        prefix = f"...[{start} chars omitted]..." if start else ""
        suffix_count = len(value) - end
        suffix = f"...[{suffix_count} chars omitted]" if suffix_count else ""
        rendered = prefix + value[start:end] + suffix
        if len(rendered) <= limit:
            return rendered, True
        available = max(0, available - (len(rendered) - limit))


def _render_json_content(
    payload: dict[str, Any],
    *,
    limit: int,
    content_key: str = "content",
) -> tuple[str, bool]:
    """Render a JSON object with an exact budget and head+tail content shaping."""
    fitted = copy.deepcopy(payload)
    output = _compact_json(fitted)
    if len(output) <= limit:
        return output, False
    content = str(fitted.get(content_key) or "")
    fitted["truncated"] = True
    fitted[content_key] = ""
    base_output = _compact_json(fitted)
    for key, value in tuple(fitted.items()):
        if len(base_output) <= limit:
            break
        if key == content_key or not isinstance(value, list):
            continue
        while value and len(base_output) > limit:
            value.pop()
            base_output = _compact_json(fitted)
    if len(base_output) > limit:
        fallback = {"error": str(fitted.get("error") or "structured observation exceeded output limit"), "truncated": True}
        output = _compact_json(fallback)
        while len(output) > limit and fallback["error"]:
            overflow = len(output) - limit
            fallback["error"] = fallback["error"][: max(0, len(fallback["error"]) - overflow - 1)]
            output = _compact_json(fallback)
        return (output if len(output) <= limit else "{}"), True
    available = max(0, limit - len(base_output))
    while True:
        clipped, _ = _head_tail_text(content, available)
        fitted[content_key] = clipped
        output = _compact_json(fitted)
        if len(output) <= limit:
            return output, True
        available = max(0, available - (len(output) - limit))


def _render_bash_observation(
    stdout: str,
    stderr: str,
    *,
    exit_code: int | None,
    timed_out: bool,
    limit: int = CODEFLOW_BASH_OUTPUT_LIMIT,
) -> tuple[str, bool]:
    """Keep execution status and both stream boundaries under all truncation."""
    stdout = str(stdout or "")
    stderr = str(stderr or "")
    header = f"exit_code={exit_code if exit_code is not None else 'null'} timed_out={'true' if timed_out else 'false'}"

    def build(stdout_value: str, stderr_value: str) -> str:
        return f"{header}\n[stdout]\n{stdout_value}\n[stderr]\n{stderr_value}"

    full = build(stdout, stderr)
    if len(full) <= limit:
        return full, False
    available = max(0, limit - len(build("", "")))
    if stdout and stderr:
        stdout_budget = available // 2
        stderr_budget = available - stdout_budget
    elif stdout:
        stdout_budget, stderr_budget = available, 0
    else:
        stdout_budget, stderr_budget = 0, available
    while True:
        rendered_stdout, stdout_truncated = _head_tail_text(stdout, stdout_budget)
        rendered_stderr, stderr_truncated = _head_tail_text(stderr, stderr_budget)
        observation = build(rendered_stdout, rendered_stderr)
        if len(observation) <= limit:
            return observation, stdout_truncated or stderr_truncated
        overflow = len(observation) - limit
        if stdout_budget >= stderr_budget and stdout_budget:
            stdout_budget = max(0, stdout_budget - overflow)
        else:
            stderr_budget = max(0, stderr_budget - overflow)


def _observation_stats(
    observation: str,
    *,
    source_characters: int,
    output_limit: int,
    compaction: Sequence[str] = (),
    truncation_reasons: Sequence[str] = (),
    returned_matches: int | None = None,
    next_offset: int | None = None,
    reference_turn: int | None = None,
) -> dict[str, Any]:
    model_characters = len(observation)
    return {
        "schema_version": 1,
        "source_characters": int(source_characters),
        "model_characters": model_characters,
        "saved_characters": max(0, int(source_characters) - model_characters),
        "output_limit": int(output_limit),
        "compaction": list(compaction),
        "truncation_reasons": list(truncation_reasons),
        "returned_matches": returned_matches,
        "next_offset": next_offset,
        "reference_turn": reference_turn,
    }


def _json_observation(payload: dict[str, Any], limit: int = OUTPUT_LIMIT) -> str:
    """Serialize a tool observation without ever truncating it into invalid JSON."""
    if isinstance(payload.get("content"), str):
        observation, _ = _render_json_content(payload, limit=limit)
        return observation

    fitted = copy.deepcopy(payload)

    def dump() -> str:
        return _compact_json(fitted)

    output = dump()
    if len(output) <= limit:
        return output

    fitted["truncated"] = True
    snippets = fitted.get("displayed_snippets")
    while isinstance(snippets, list) and len(snippets) > 1 and len(output) > limit:
        snippets.pop()
        output = dump()

    if isinstance(snippets, list) and snippets and len(output) > limit:
        content = str(snippets[0].get("content", ""))
        snippets[0]["content"] = ""
        output = dump()
        available = max(0, limit - len(output))
        while True:
            clipped, _ = _head_tail_text(content, available)
            snippets[0]["content"] = clipped
            output = dump()
            if len(output) <= limit:
                break
            if available == 0:
                break
            available = max(0, available - (len(output) - limit))

    for key in ("matched_line_ranges", "matched_files", "touched_paths"):
        values = fitted.get(key)
        while isinstance(values, list) and values and len(output) > limit:
            values.pop()
            output = dump()

    if len(output) > limit:
        fallback, _ = _render_json_content(
            {"error": "structured observation exceeded output limit", "content": "", "truncated": True},
            limit=limit,
        )
        return fallback
    return output












SCAFFOLD_PROMPTS = {"codeflow": {"system": {"native_tool_call": CODEFLOW_SYSTEM_PROMPT_NATIVE_TOOL_CALL}, "user": {"native_tool_call": CODEFLOW_USER_PROMPT}}}


class SWEScaffoldHarness(SandboxedAgentFlow):
    """Execute the Codeflow native tool protocol inside a sandbox."""

    name = "swe-scaffold"
    scaffold = "codeflow"
    protocol = ""
    sandbox_backend = "minisandbox"
    max_turns = 50
    command_timeout = 120
    max_concurrent = 64
    limit_termination_success_reward: float | str = 0.6
    limit_termination_outcome_mode: str = "discount_success"
    codeflow_tool_mode = DEFAULT_CODEFLOW_TOOL_MODE
    codeflow_tool_restricted_mode = DEFAULT_CODEFLOW_TOOL_RESTRICTED_MODE

    def __init__(
        self,
        *,
        scaffold: str | None = None,
        protocol: str | None = None,
        max_turns: int | None = None,
        command_timeout: int | None = None,
        codeflow_tool_mode: str | None = None,
        codeflow_tool_restricted_mode: Sequence[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scaffold = _normalize_scaffold(scaffold or self.scaffold)
        self.protocol = _normalize_protocol(protocol or self.protocol, self.scaffold)
        if max_turns is not None:
            self.max_turns = max_turns
        if command_timeout is not None:
            self.command_timeout = command_timeout
        self.configure_codeflow_tool_mode(codeflow_tool_mode)
        self.configure_codeflow_tool_restricted_mode(
            []
            if self.codeflow_tool_mode == "bash_only"
            and codeflow_tool_restricted_mode is None
            else codeflow_tool_restricted_mode
        )
        self.name = self.scaffold

    def set_protocol(self, protocol: str | None) -> None:
        self.protocol = _normalize_protocol(protocol or "", self.scaffold)

    def configure_codeflow_tool_restricted_mode(
        self,
        restricted_mode: Sequence[str] | None,
    ) -> None:
        normalized = normalize_codeflow_tool_restricted_mode(restricted_mode)
        if self.codeflow_tool_mode == "bash_only" and normalized:
            raise ValueError(
                "swe.codeflow_tool_mode=bash_only requires "
                "swe.codeflow_tool_restricted_mode=[]"
            )
        self.codeflow_tool_restricted_mode = normalized

    def configure_codeflow_tool_mode(self, tool_mode: str | None) -> None:
        self.codeflow_tool_mode = normalize_codeflow_tool_mode(tool_mode)
        if self.codeflow_tool_mode == "bash_only":
            self.codeflow_tool_restricted_mode = ()

    def run(
        self,
        task: Task,
        config: AgentConfig,
        *,
        env: Sandbox,
        shadow_runtime: Any = None,
        progress_reporter: Any = None,
        cancellation_token: AgentFlowCancellation | None = None,
    ) -> Episode:
        from openai import OpenAI

        client_kwargs: dict[str, Any] = {}
        client_kwargs.update(
            timeout=float(getattr(self, "model_request_timeout", 3600.0)),
            max_retries=0,
        )
        client_parameters = inspect.signature(OpenAI).parameters
        accepts_client_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in client_parameters.values()
        )
        supported_client_kwargs = {
            key: value
            for key, value in client_kwargs.items()
            if accepts_client_kwargs or key in client_parameters
        }
        request_abort = None
        if accepts_client_kwargs or "http_client" in client_parameters:
            import httpx
            from rllm_model_gateway.http_client import shared_ssl_context
            if cancellation_token is not None:
                from rllm.utils.model_request_abort import ModelRequestAbort
                request_abort = ModelRequestAbort(config.session_uid)

            supported_client_kwargs["http_client"] = httpx.Client(
                verify=shared_ssl_context(),
                timeout=client_kwargs.get("timeout", 600.0),
                follow_redirects=True,
                # Each client belongs to one sequential rollout. Tool calls
                # leave its socket idle; don't reuse a server-expired socket.
                limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
                event_hooks={"request": [request_abort.on_request]} if request_abort else None,
            )
        # The real OpenAI client accepts both controls. Lightweight test
        # doubles with the historical two-argument constructor simply omit
        # them; signature-based filtering avoids masking a real constructor
        # TypeError from production.
        client = OpenAI(
            base_url=config.base_url,
            api_key="EMPTY",
            **supported_client_kwargs,
        )
        close_lock = threading.Lock()
        client_closed = False

        def close_client_once() -> None:
            nonlocal client_closed
            with close_lock:
                if client_closed:
                    return
                client_closed = True
            if request_abort is not None:
                request_abort.abort(reason=(
                    "external_cancel" if cancellation_token.cancelled else
                    "deadline" if cancellation_token.deadline_expired else "flow_finished"
                ))
            close_client = getattr(client, "close", None)
            if callable(close_client):
                try:
                    close_client()
                except Exception:
                    logger.warning(
                        "[%s] failed to close OpenAI client",
                        config.session_uid,
                        exc_info=True,
                    )

        token_context = agent_flow_cancellation.set(cancellation_token)
        try:
            if cancellation_token is not None:
                cancellation_token.register_abort(close_client_once)
                cancellation_token.raise_if_cancelled()
            return self._run_with_client(
                client,
                task,
                config,
                env=env,
                shadow_runtime=shadow_runtime,
                progress_reporter=progress_reporter,
                cancellation_token=cancellation_token,
            )
        finally:
            agent_flow_cancellation.reset(token_context)
            close_client_once()

    def _run_with_client(
        self,
        client: Any,
        task: Task,
        config: AgentConfig,
        *,
        env: Sandbox,
        shadow_runtime: Any = None,
        progress_reporter: Any = None,
        cancellation_token: AgentFlowCancellation | None = None,
    ) -> Episode:
        prompt_spec = SCAFFOLD_PROMPTS[self.scaffold]
        prompt_key = "tool_calling" if self.protocol == "tool_calling" else self.protocol
        system_prompt = prompt_spec["system"].get(prompt_key, prompt_spec["system"].get("xml", ""))
        if (
            self.scaffold == "codeflow"
            and self.codeflow_tool_mode == "bash_only"
            and self.protocol == NATIVE_TOOL_CALL_PROTOCOL
        ):
            system_prompt = CODEFLOW_BASH_ONLY_SYSTEM_PROMPT_NATIVE_TOOL_CALL
        rllm_metadata = task.metadata.get("rllm") or {}
        task_profile = task.metadata.get("task_profile") or rllm_metadata.get("task_profile")
        user_template = prompt_spec["user"].get(prompt_key, prompt_spec["user"].get("xml", "{problem_statement}"))
        if self.scaffold == "codeflow" and task_profile in {"repo_generation_denovoswe", "repo_generation_nl2repo"}:
            system_prompt = (
                CODEFLOW_DENOVOSWE_BASH_ONLY_SYSTEM_PROMPT_NATIVE_TOOL_CALL
                if self.codeflow_tool_mode == "bash_only"
                else CODEFLOW_DENOVOSWE_SYSTEM_PROMPT_NATIVE_TOOL_CALL
            )
            user_template = CODEFLOW_DENOVOSWE_USER_PROMPT
        repository_root = self._repository_root(task)
        system_prompt = system_prompt.replace("{repository_root}", repository_root)
        system_prompt = system_prompt.replace(
            "{codeflow_file_tool_policy}",
            _codeflow_file_tool_policy_prompt(
                self.codeflow_tool_restricted_mode
            ),
        )
        system_prompt += f"\n\nThe default execute_bash command timeout is {self.command_timeout:g} seconds."
        if (
            self.scaffold == "codeflow"
            and rllm_metadata.get("agent_environment_policy")
            == "preconfigured_eval_v1"
        ):
            system_prompt += (
                "\n\nEvaluation environment policy:\n"
                "- Every execute_bash command automatically runs in the preconfigured "
                "testbed conda environment used by the verifier.\n"
                "- Do not install, update, remove, or download dependencies. External "
                "pip/conda/package-manager and curl/wget commands are rejected.\n"
                "- A missing dependency indicates an evaluation environment failure; do "
                "not spend turns attempting to repair the environment.\n"
                "- Repeating the same Bash command without an intervening successful edit "
                "is limited to two executions.\n"
            )
        if (
            self.scaffold == "codeflow"
            and rllm_metadata.get("bash_policy_profile")
            == "repo_generation_package"
        ):
            system_prompt += (
                "\n\nRepository-generation source policy:\n"
                "- Implement the target package only from the supplied specification.\n"
                "- Do not install, download, clone, inspect, or extract the target "
                "package's upstream source. Third-party dependencies and a local "
                "editable install of your own implementation are allowed.\n"
            )
        user_prompt = user_template.format(
            problem_statement=str(task.instruction),
            repository_root=repository_root,
        )
        snapshot = None
        if task_profile == "repo_generation_nl2repo":
            from rllm.sandbox.repo_generation_environment import inspect_repo_generation_environment, generation_environment_prompt
            proxy_url = rllm_metadata.get("repo_generation_proxy_url", "")
            snapshot = inspect_repo_generation_environment(env, proxy_url=proxy_url, check_network=bool(proxy_url))
            if proxy_url and snapshot.get("package_index", {}).get("exit_code") != 0:
                from rllm.types import RolloutInfrastructureError
                raise RolloutInfrastructureError(
                    "repo_generation_package_network_unavailable",
                    "Configured proxy cannot download a build dependency: " + snapshot.get("package_index", {}).get("output", "missing probe result")[-2000:],
                    retryable=True, stage="agent_environment",
                )
            user_prompt += generation_environment_prompt(snapshot, task_profile)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if self.protocol == NATIVE_TOOL_CALL_PROTOCOL:
            episode = self._run_native_tool_call(
                client,
                task,
                config,
                env,
                messages,
                shadow_runtime=shadow_runtime,
                progress_reporter=progress_reporter,
                cancellation_token=cancellation_token,
            )

            if snapshot is not None:
                episode.metadata["repo_generation_environment"] = snapshot
                episode.metadata["repo_generation_prompt"] = "generation_environment_v2"
            return episode

    @staticmethod
    def _model_request_id(session_uid: str, turn: int) -> str:
        session_digest = hashlib.sha256(str(session_uid).encode("utf-8")).hexdigest()[:24]
        return f"rllm-{session_digest}-turn-{int(turn)}"

    @staticmethod
    def _report_rollout_progress(
        reporter: Any,
        *,
        stage: str,
        turn_index: int | None,
    ) -> None:
        """Publish best-effort in-memory progress without affecting rollout."""

        if not callable(reporter):
            return
        try:
            reporter(stage=stage, turn_index=turn_index)
        except Exception:
            logger.debug("Codeflow progress reporter failed", exc_info=True)

    @staticmethod
    def _codeflow_invalid_action_fingerprint(
        action: ParsedAction,
        event: ActionEvent | None,
        task: Task,
    ) -> tuple[str, str] | None:
        """Fingerprint one model-attributable invalid evaluation action."""

        if (
            not _codeflow_eval_guard_enabled(task)
            or action.status != "ok"
            or event is None
            or event.validation_status != ValidationStatus.ERROR
        ):
            return None

        if action.name == "execute_bash":
            command = action.arguments.get("command")
            if not isinstance(command, str) or not command.strip():
                return None
            canonical_arguments: Any = {
                "command": canonicalize_shell_command(command),
            }
            summary = _truncate(command.strip(), 300)
        else:
            canonical_arguments = action.arguments
            summary = _truncate(
                json.dumps(
                    action.arguments,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ),
                300,
            )

        payload = json.dumps(
            {"tool_name": action.name, "arguments": canonical_arguments},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        fingerprint = "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return fingerprint, summary

    def _native_model_request(
        self,
        client: Any,
        config: AgentConfig,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        *,
        turn: int,
        request_id: str,
        cancellation_token: AgentFlowCancellation | None = None,
    ) -> tuple[Any | None, dict[str, Any] | None]:
        """Generate one native-tool turn without replaying prior tool actions."""
        attempts = max(1, int(getattr(self, "model_request_retries", 3)))
        timeout = float(getattr(self, "model_request_timeout", 3600.0))
        last_error: BaseException | None = None
        for attempt in range(1, attempts + 1):
            if cancellation_token is not None:
                cancellation_token.raise_if_deadline()
                if cancellation_token.remaining is not None:
                    timeout = min(timeout, max(0.001, cancellation_token.remaining))
            try:
                return (
                    client.chat.completions.create(
                        model=config.model,
                        messages=messages,
                        tools=tool_schemas,
                        tool_choice="auto",
                        timeout=timeout,
                        extra_headers={
                            "X-RLLM-Request-ID": request_id,
                            "X-RLLM-Turn-Index": str(turn),
                        },
                    ),
                    None,
                )
            except Exception as exc:
                if cancellation_token is not None:
                    cancellation_token.raise_if_deadline()
                # Resolve exception types only after a failure. This keeps the
                # native harness compatible with small OpenAI test doubles,
                # while production still retries exactly timeout, connection,
                # 429 and 5xx failures.
                import openai

                retryable_types = tuple(
                    error_type
                    for name in ("APITimeoutError", "APIConnectionError")
                    if isinstance(
                        (error_type := getattr(openai, name, None)),
                        type,
                    )
                )
                status_type = getattr(openai, "APIStatusError", None)
                if retryable_types and isinstance(exc, retryable_types):
                    last_error = exc
                elif isinstance(status_type, type) and isinstance(exc, status_type):
                    status_code = int(getattr(exc, "status_code", 0) or 0)
                    if status_code != 429 and status_code < 500:
                        raise
                    last_error = exc
                else:
                    raise
            if attempt < attempts:
                base_delay = min(
                    float(
                        getattr(
                            self,
                            "model_request_retry_max_delay",
                            12.0,
                        )
                    ),
                    float(
                        getattr(
                            self,
                            "model_request_retry_base_delay",
                            1.0,
                        )
                    )
                    * (2 ** (attempt - 1)),
                )
                jitter_max = max(
                    0.0,
                    float(
                        getattr(
                            self,
                            "model_request_retry_jitter",
                            3.0,
                        )
                    ),
                )
                digest = hashlib.blake2b(
                    f"{request_id}:{attempt}".encode(),
                    digest_size=8,
                ).digest()
                jitter = (
                    int.from_bytes(digest, "big")
                    / float((1 << 64) - 1)
                    * jitter_max
                )
                delay = base_delay + jitter
                logger.warning(
                    "[%s] model request turn %d attempt %d/%d failed (%s); "
                    "retrying the same request id in %.2fs",
                    config.session_uid,
                    turn,
                    attempt,
                    attempts,
                    type(last_error).__name__,
                    delay,
                )
                if cancellation_token is None:
                    time.sleep(delay)
                elif cancellation_token.wait(delay):
                    cancellation_token.raise_if_cancelled()

        assert last_error is not None
        return None, {
            "schema_version": 1,
            "reason": "model_request_retry_exhausted",
            "request_id": request_id,
            "turn": int(turn),
            "attempts": attempts,
            "timeout_seconds": timeout,
            "error_type": type(last_error).__name__,
            "error": str(last_error)[:2000],
        }

    def _run_native_tool_call(
        self,
        client: Any,
        task: Task,
        config: AgentConfig,
        env: Sandbox,
        messages: list[dict[str, Any]],
        *,
        shadow_runtime: Any = None,
        progress_reporter: Any = None,
        cancellation_token: AgentFlowCancellation | None = None,
    ) -> Episode:
        tool_schemas = _copy_native_tool_schemas(self.scaffold)
        _configure_codeflow_tool_schemas(
            tool_schemas,
            self.codeflow_tool_restricted_mode,
            self.command_timeout,
        )
        _configure_codeflow_tool_mode_schemas(
            tool_schemas,
            self.codeflow_tool_mode,
        )
        max_turns = int(task.metadata.get("rllm", {}).get("max_turns") or task.metadata.get("max_turns") or self.max_turns)
        editor_history: dict[str, list[str | None]] = {}
        codeflow_history: list[CodeFlowUndoEntry] = []
        bash_command_counts: dict[str, int] = {}
        # Per-rollout by construction: never share observation references
        # across concurrent tasks or Harness instances.
        codeflow_observation_cache: dict[tuple[str, str, str, str], int] = {}
        steps: list[Step] = []
        submitted = False
        codeflow_baseline_ref: str | None = None
        context_termination: dict[str, Any] | None = None
        model_request_failure: dict[str, Any] | None = None
        deadline_audit: dict[str, Any] | None = None
        repository_infrastructure_failure: dict[str, Any] | None = None
        repository_policy_failure: dict[str, Any] | None = None
        no_progress_termination: dict[str, Any] | None = None
        no_progress_fingerprint: str | None = None
        no_progress_first_turn = 0
        no_progress_count = 0
        timeout_state: str | None = None
        timeout_count = 0
        timeout_first_turn = 0

        self._codeflow_helper_runtime(env, task)

        if (
            self.scaffold == "codeflow"
            and (task.metadata.get("rllm") or {}).get("agent_environment_policy")
            == "preconfigured_eval_v1"
        ):
            for tool in tool_schemas:
                function = tool.get("function") if isinstance(tool, dict) else None
                if isinstance(function, dict) and function.get("name") == "execute_bash":
                    function["description"] = (
                        str(function.get("description") or "")
                        + " The testbed conda environment is activated automatically. "
                        "Dependency installation and external downloads are rejected."
                    )

        for turn in range(max_turns):
            if cancellation_token is not None:
                cancellation_token.raise_if_cancelled()
            if cancellation_token is not None and cancellation_token.deadline_expired:
                deadline_audit = deadline_audit or {"deadline_stage": "between_steps", "turn": turn}
                break
            input_content = _message_content(messages[-1]) if messages else ""
            request_id: str | None = None
            self._report_rollout_progress(
                progress_reporter,
                stage="model_request",
                turn_index=turn,
            )
            request_id = self._model_request_id(config.session_uid, turn)
            try:
                response, request_failure = self._native_model_request(
                    client, config, messages, tool_schemas, turn=turn,
                    request_id=request_id, cancellation_token=cancellation_token,
                )
                if cancellation_token is not None:
                    cancellation_token.raise_if_deadline()
            except AgentFlowDeadline:
                deadline_audit = {"deadline_stage": "model_request", "turn": turn,
                                  "discarded_request_id": request_id}
                break
            if request_failure is not None:
                model_request_failure = request_failure
                break
            assert response is not None
            response_context = _response_context_metadata(response)
            if (
                self.scaffold == "codeflow"
                and response_context.get("input_exhausted") is True
            ):
                context_termination = {
                    **response_context,
                    "termination_kind": "input_exhausted",
                    "last_observation_consumed_by_model": False,
                    "discarded_terminal_completion_tokens": 0,
                }
                _mark_last_observation_unconsumed(steps, context_termination)
                break

            choice = response.choices[0]
            assistant_message = _assistant_message_to_dict(choice.message)
            assistant_text = str(assistant_message.get("content") or "")
            raw_tool_calls = assistant_message.get("tool_calls") or []
            raw_tool_calls = raw_tool_calls if isinstance(raw_tool_calls, list) else []

            if (
                self.scaffold == "codeflow"
                and response_context.get("training_disposition")
                == _DISCARD_CONTEXT_TERMINAL
            ):
                context_termination = {
                    **response_context,
                    "termination_kind": "output_exhausted",
                    "last_observation_consumed_by_model": True,
                    "discarded_terminal_completion_tokens": int(
                        response_context.get("completion_tokens") or 0
                    ),
                }
                break

            if len(raw_tool_calls) == 1:
                action = _native_tool_call_to_action(raw_tool_calls[0], self.scaffold)
                action_event = None
                self._report_rollout_progress(
                    progress_reporter,
                    stage="action_processing",
                    turn_index=turn,
                )
                execution, action_event, codeflow_baseline_ref = self._codeflow_process_action(
                    action,
                    env,
                    task,
                    codeflow_history,
                    session_uid=config.session_uid,
                    turn=turn,
                    baseline_ref=codeflow_baseline_ref,
                    bash_command_counts=bash_command_counts,
                    observation_cache=codeflow_observation_cache,
                )
                model_observation = execution.observation
                done = execution.done
                fatal = execution.fatal
                messages.append(assistant_message)
                step = self._native_step(
                    turn,
                    input_content,
                    assistant_text,
                    action,
                    done,
                    model_observation,
                    raw_tool_calls=raw_tool_calls,
                    action_event=action_event,
                    request_id=request_id,
                    observation_stats=(execution.observation_stats if self.scaffold == "codeflow" else None),
                )
                if self.scaffold == "codeflow" and shadow_runtime is not None:
                    shadow_runtime.schedule(step)
                steps.append(step)
                if cancellation_token is not None and cancellation_token.deadline_expired:
                    deadline_audit = {"deadline_stage": "action_processing", "turn": turn}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": _tool_call_id(raw_tool_calls[0], turn, 0),
                        "content": model_observation,
                    }
                )
                if done or fatal:
                    if fatal and "bash_destroyed_git_repository" in execution.policy_violations:
                        repository_policy_failure = {
                            "reason": "agent_destroyed_git_repository",
                            "message": str(execution.error),
                            "turn": turn,
                            "repository_restored": False,
                        }
                    if fatal and self.scaffold == "codeflow" and str(execution.error or "").startswith("repository checkpoint unavailable"):
                        repository_infrastructure_failure = {
                            "reason": "repository_checkpoint_unavailable",
                            "task_id": task.id,
                            "message": str(execution.error),
                            "stage": "agentflow",
                            "retryable": True,
                            "retry_scope": "full_rollout",
                            "turn": turn,
                            "diagnostics": getattr(env, "_rllm_codeflow_last_helper_error", None),
                        }
                    submitted = done
                    break
                if (
                    self.scaffold == "codeflow"
                    and (task.metadata.get("rllm") or {}).get("eval_no_progress") is True
                ):
                    # Diagnostics between failed probes do not constitute a
                    # repair. Allow another timeout budget after the repository
                    # actually changes, including edits by a failed command.
                    state = action_event.repo_state_after
                    if state is None or state != timeout_state or action_event.repo_changed:
                        timeout_count = 0
                        timeout_state = state
                    if (
                        state is not None
                        and action_event.repo_changed is False
                        and action.name == "execute_bash"
                        and execution.status == ExecutionStatus.TIMEOUT
                    ):
                        if timeout_count == 0:
                            timeout_first_turn = turn
                        timeout_count += 1
                        logger.warning(
                            "[%s] turn %d command timed out with unchanged repository (%d/%d)",
                            config.session_uid, turn + 1, timeout_count,
                            CODEFLOW_EVAL_NO_PROGRESS_THRESHOLD,
                        )
                        if timeout_count >= CODEFLOW_EVAL_NO_PROGRESS_THRESHOLD:
                            no_progress_count = timeout_count
                            no_progress_termination = {
                                "schema_version": 1,
                                "policy_version": "bounded_bash_no_progress_v1",
                                "reason": "repeated_timeout_without_repository_change",
                                "counting_scope": "timeouts_since_last_repository_change",
                                "threshold": CODEFLOW_EVAL_NO_PROGRESS_THRESHOLD,
                                "consecutive_count": timeout_count,
                                "first_turn_index": timeout_first_turn,
                                "last_turn_index": turn,
                                "tool_name": action.name,
                                "repository_state": state,
                            }
                            break
                invalid_fingerprint = self._codeflow_invalid_action_fingerprint(
                    action,
                    action_event,
                    task,
                )
                if invalid_fingerprint is None:
                    no_progress_fingerprint = None
                    no_progress_count = 0
                else:
                    fingerprint, action_summary = invalid_fingerprint
                    if fingerprint == no_progress_fingerprint:
                        no_progress_count += 1
                    else:
                        no_progress_fingerprint = fingerprint
                        no_progress_first_turn = turn
                        no_progress_count = 1
                    if no_progress_count >= CODEFLOW_EVAL_NO_PROGRESS_THRESHOLD:
                        no_progress_termination = {
                            "schema_version": 1,
                            "policy_version": CODEFLOW_EVAL_NO_PROGRESS_POLICY_VERSION,
                            "threshold": CODEFLOW_EVAL_NO_PROGRESS_THRESHOLD,
                            "consecutive_count": no_progress_count,
                            "first_turn_index": no_progress_first_turn,
                            "last_turn_index": turn,
                            "tool_name": action.name,
                            "action_summary": action_summary,
                            "action_fingerprint": fingerprint,
                            "policy_violations": sorted(
                                set(action_event.policy_violations)
                            ),
                        }
                        break
                continue

            reason = "missing_tool_call" if not raw_tool_calls else "multiple_tool_calls"
            # A missing/multiple tool call is a different model action, so it
            # breaks a run of otherwise identical validation failures.  It
            # retains its existing format-error semantics but does not count
            # toward the evaluation-only no-progress loop threshold.
            no_progress_fingerprint = None
            no_progress_count = 0
            action = ParsedAction(thought=assistant_text, status="parse_error", reason=reason, protocol=NATIVE_TOOL_CALL_PROTOCOL)
            action_event = None
            execution, action_event, codeflow_baseline_ref = self._codeflow_process_action(
                action,
                env,
                task,
                codeflow_history,
                session_uid=config.session_uid,
                turn=turn,
                baseline_ref=codeflow_baseline_ref,
                bash_command_counts=bash_command_counts,
                observation_cache=codeflow_observation_cache,
            )
            model_observation = execution.observation
            messages.append(assistant_message)
            step = self._native_step(
                turn,
                input_content,
                assistant_text,
                action,
                False,
                model_observation,
                raw_tool_calls=raw_tool_calls,
                action_event=action_event,
                request_id=request_id,
                observation_stats=(execution.observation_stats if self.scaffold == "codeflow" else None),
            )
            if self.scaffold == "codeflow" and shadow_runtime is not None:
                shadow_runtime.schedule(step)
            steps.append(step)

            if not raw_tool_calls:
                messages.append(_codeflow_native_correction_message(model_observation))
            else:
                for idx, tool_call in enumerate(raw_tool_calls):
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": _tool_call_id(tool_call, turn, idx),
                            "content": model_observation,
                        }
                    )

        trajectory_metadata: dict[str, Any] = {
            "scaffold": self.scaffold,
            "protocol": self.protocol,
        }
        trajectory_metadata["codeflow_tool_mode"] = self.codeflow_tool_mode
        trajectory_metadata["codeflow_tool_restricted_mode"] = list(
            self.codeflow_tool_restricted_mode
        )
        if context_termination is not None:
            trajectory_metadata["context_limit"] = dict(context_termination)
        if model_request_failure is not None:
            trajectory_metadata["model_request_failure"] = dict(
                model_request_failure
            )
        if no_progress_termination is not None:
            trajectory_metadata["no_progress_loop"] = dict(
                no_progress_termination
            )
        trajectory = Trajectory(
            uid=config.session_uid,
            name=self.name,
            task=task.id,
            steps=steps,
            output=steps[-1].output if steps else "",
            metadata=trajectory_metadata,
        )
        termination_reason = (
            TerminationReason.ERROR
            if model_request_failure is not None or repository_infrastructure_failure is not None or repository_policy_failure is not None
            else (
                TerminationReason.ENV_DONE
                if submitted
                else (
                    TerminationReason.NO_PROGRESS_LOOP
                    if no_progress_termination is not None
                    else (
                        TerminationReason.MAX_CONTEXT_LENGTH_EXCEEDED
                        if context_termination is not None
                        else TerminationReason.MAX_TURNS_EXCEEDED
                    )
                )
            )
        )
        episode = Episode(
            id=config.session_uid,
            task=task.id,
            trajectories=[trajectory],
            termination_reason=termination_reason,
        )
        if cancellation_token is not None and cancellation_token.deadline_expired:
            deadline_audit = deadline_audit or {"deadline_stage": "between_steps", "turn": len(steps)}
        if deadline_audit is not None and not (model_request_failure or repository_infrastructure_failure or repository_policy_failure):
            episode.termination_reason = TerminationReason.TIMEOUT
            episode.metadata["denovo_deadline"] = {
                "schema_version": 1, **deadline_audit,
                "worker_stop_confirmed": True, "verifier_completed": False,
                "completed_steps": len(steps),
            }
        episode.metadata["repository_helper_provenance"] = list(getattr(env, "_rllm_repository_helper_provenance", {}).values())
        episode.metadata["codeflow_tool_mode"] = self.codeflow_tool_mode
        episode.metadata["codeflow_tool_restricted_mode"] = list(
            self.codeflow_tool_restricted_mode
        )
        if context_termination is not None:
            episode.metadata["context_limit"] = dict(context_termination)
            episode.metrics.update(
                {
                    "context_limit/terminated": 1,
                    "context_limit/input_tokens": int(
                        context_termination.get("input_tokens") or 0
                    ),
                    "context_limit/requested_output_tokens": int(
                        context_termination.get("requested_output_tokens") or 0
                    ),
                    "context_limit/effective_output_tokens": int(
                        context_termination.get("effective_output_tokens") or 0
                    ),
                    "context_limit/discarded_completion_tokens": int(
                        context_termination.get("discarded_terminal_completion_tokens") or 0
                    ),
                }
            )
        if no_progress_termination is not None:
            episode.metadata["no_progress_loop"] = dict(no_progress_termination)
            episode.metrics.update(
                {
                    "no_progress_loop/terminated": 1,
                    "no_progress_loop/consecutive_count": no_progress_count,
                    "no_progress_loop/last_turn_index": int(
                        no_progress_termination["last_turn_index"]
                    ),
                }
            )
        if model_request_failure is not None:
            episode.metadata["infrastructure_failure"] = dict(
                model_request_failure
            )
            episode.metrics.update(
                {
                    "model_request/failure": 1,
                    "model_request/turn": int(model_request_failure["turn"]),
                    "model_request/attempts": int(
                        model_request_failure["attempts"]
                    ),
                }
            )
        if repository_infrastructure_failure is not None:
            episode.metadata["infrastructure_failure"] = repository_infrastructure_failure
            episode.metrics["infrastructure_failure/reason"] = repository_infrastructure_failure["reason"]
        if repository_policy_failure is not None:
            episode.metadata["repository_policy_failure"] = repository_policy_failure
        if self.scaffold == "codeflow" and shadow_runtime is not None:
            shadow_runtime.attach_episode(episode)
        return episode

    def _native_step(
        self,
        turn: int,
        input_content: str,
        assistant_text: str,
        action: ParsedAction,
        done: bool,
        model_observation: str,
        *,
        raw_tool_calls: list[Any],
        action_event: ActionEvent | None = None,
        request_id: str | None = None,
        observation_stats: dict[str, Any] | None = None,
    ) -> Step:
        metadata: dict[str, Any] = {
            "scaffold": self.scaffold,
            "protocol": action.protocol or NATIVE_TOOL_CALL_PROTOCOL,
            "parse_status": action.status,
            "parse_reason": action.reason,
            "tool_observation": model_observation,
            "raw_tool_call_count": len(raw_tool_calls),
        }
        metadata["codeflow_tool_mode"] = self.codeflow_tool_mode
        if request_id:
            metadata["model_request_id"] = request_id
        if observation_stats:
            metadata["tool_observation_stats"] = copy.deepcopy(observation_stats)
        if action_event is not None:
            metadata[ACTION_EVENT_METADATA_KEY] = action_event.model_dump(mode="json")
        if len(raw_tool_calls) == 1:
            metadata["raw_tool_call"] = _summarize_raw_tool_call(raw_tool_calls[0])
        elif len(raw_tool_calls) > 1:
            metadata["raw_tool_names"] = [_summarize_raw_tool_call(tool_call).get("name", "") for tool_call in raw_tool_calls]

        step = Step(
            id=f"step-{turn}",
            input=input_content,
            output=assistant_text,
            action={"name": action.name, "arguments": action.arguments},
            done=done,
            metadata=metadata,
        )
        annotate_step_tool_call_reward(step)
        return step

    @staticmethod
    def _format_native_tool_observation(observation: str, action: ParsedAction) -> str:
        if action.status != "ok":
            return f"Action parse error ({action.reason}). Please call exactly one available tool, or call submit when the task is complete."
        return observation



    def _exec(self, command: str, sandbox: Sandbox, task: Task, *, timeout: float | None = None) -> str:
        if not command.strip():
            return "No command provided."
        full_command = self._with_task_shell_context(command, task)
        try:
            return _truncate(sandbox.exec(full_command, timeout=timeout, user=task.metadata.get("agent_user")))
        except Exception as exc:
            return _truncate(f"Error: {exc}")

    @staticmethod
    def _repository_root(task: Task) -> str:
        root = str(task.metadata.get("workdir") or "/testbed").strip() or "/testbed"
        return posixpath.normpath(root if root.startswith("/") else f"/{root}")

    def _codeflow_path(self, path: Any, task: Task, *, allow_empty: bool = False) -> tuple[str, str]:
        root = self._repository_root(task)
        raw_path = str(path or "").strip()
        if not raw_path:
            if allow_empty:
                return root, "."
            raise ValueError("path is required")
        normalized = posixpath.normpath(raw_path if raw_path.startswith("/") else posixpath.join(root, raw_path))
        root_prefix = root.rstrip("/") + "/"
        if normalized != root and not normalized.startswith(root_prefix):
            raise ValueError(f"path escapes repository root {root}")
        relative = posixpath.relpath(normalized, root)
        return normalized, relative

    def _codeflow_helper_runtime(self, sandbox: Sandbox, task: Task, *, user: str | None = None, minimum: tuple[int, int] = (3, 6)) -> dict[str, Any]:
        """Select a framework interpreter without changing the task's PATH."""
        key = (user, minimum, 1)
        runtimes = getattr(sandbox, "_rllm_helper_runtimes", {})
        if key in runtimes:
            return runtimes[key]
        cached = getattr(sandbox, "_rllm_codeflow_helper_runtime", None)
        if minimum == (3, 6) and user is None and isinstance(cached, dict) and cached.get("executable"):
            return cached
        # Parseable on Python 3.5; check both syntax and the pathlib API used
        # by checkpoint symlink handling before running any helper mutation.
        script = '''import json, sys
from pathlib import Path
assert sys.version_info >= (3, 6), "Codeflow helpers require Python >= 3.6"
compile('f"{1}"', '<helper-capability>', 'exec')
Path('.').resolve(strict=False)
print(json.dumps({"executable": sys.executable, "version": list(sys.version_info[:3])}))
'''
        script += "\nassert sys.version_info >= " + repr(minimum) + ", 'helper capability version mismatch'\n"
        if minimum >= (3, 7):
            script += "import dataclasses\ncompile('from __future__ import annotations\\nx: list[str]', '<collector-capability>', 'exec')\n"
        probes = []
        for executable in ("python3", "/opt/miniconda3/bin/python3", "/usr/bin/python3", "/opt/conda/bin/python3", "/usr/local/bin/python3"):
            result = self._codeflow_run_json_script(
                script, {}, sandbox, task, timeout=getattr(sandbox, "control_read_timeout", 30), user=user,
                operation_kind="read_only", _runtime={"executable": executable},
            )
            if isinstance(result.get("executable"), str) and isinstance(result.get("version"), list):
                runtime = {"executable": result["executable"], "version": result["version"]}
                if minimum == (3, 6) and user is None:
                    sandbox._rllm_codeflow_helper_runtime = runtime
                runtimes[key] = runtime
                sandbox._rllm_helper_runtimes = runtimes
                return runtime
            probes.append({"candidate": executable, "error": result.get("error"), "diagnostics": result.get("diagnostics")})
            if result.get("_error_kind") in {"transport", "protocol"}:
                raise RolloutInfrastructureError(
                    "agent_helper_runtime_failed", "Helper interpreter probe unavailable: " + str(result.get("error")),
                    stage="setup", retryable=True, diagnostics={"interpreter_probes": probes},
                )
        raise RolloutInfrastructureError(
            "agent_helper_runtime_failed", "No compatible Codeflow helper interpreter: " + json.dumps(probes)[:3500],
            stage="setup", retryable=False, diagnostics={"interpreter_probes": probes, "fatal": True, "failure_scope": "task"},
        )

    def _codeflow_run_json_script(
        self,
        script: str,
        payload: dict[str, Any],
        sandbox: Sandbox,
        task: Task,
        *,
        timeout: float | None = None,
        user: str | None = None,
        operation_kind: str = "mutating",
        _runtime: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if operation_kind not in {"mutating", "read_only", "idempotent"}:
            raise ValueError(f"unsupported helper operation kind: {operation_kind!r}")
        runtime = _runtime if _runtime is not None else self._codeflow_helper_runtime(sandbox, task, user=user)
        repository_helper = "REPOSITORY_STATE_VERSION =" in script
        sandbox._rllm_codeflow_last_helper_error = None
        interpreter = shlex.quote(runtime["executable"])
        # Initialize Python's filesystem/stdio codecs as UTF-8, then restore
        # the original environment before this helper starts child commands.
        # In particular execute_bash must retain the task's own locale.
        script = '''import os as _rllm_locale_os
for _rllm_locale_key in ("LC_ALL", "PYTHONIOENCODING"):
    _rllm_locale_prefix = "_RLLM_HELPER_ORIGINAL_" + _rllm_locale_key
    _rllm_locale_set = _rllm_locale_os.environ.pop(_rllm_locale_prefix + "_SET", "")
    _rllm_locale_value = _rllm_locale_os.environ.pop(_rllm_locale_prefix, "")
    if _rllm_locale_set:
        _rllm_locale_os.environ[_rllm_locale_key] = _rllm_locale_value
    else:
        _rllm_locale_os.environ.pop(_rllm_locale_key, None)
''' + script
        encoded = base64.b64encode(json.dumps(payload, ensure_ascii=False).encode("utf-8")).decode("ascii")
        # Uploading only the program leaves large repository snapshots in argv:
        # both the host OCI helper and the task interpreter can then hit E2BIG.
        # Bind a large payload into the same content-verified uploaded program.
        # This preserves argv[1] for existing helpers without an extra mutation
        # or replay, and the ordinary integrity guard covers BOTH code and data.
        if len(encoded) > _CODEFLOW_INLINE_HELPER_MAX_BYTES and callable(getattr(sandbox, "upload_file", None)):
            script = "import sys as _rllm_payload_sys\n_rllm_payload_sys.argv[1:] = [" + repr(encoded) + "]\n" + script
            encoded = ""
        helper_digest = hashlib.sha256(script.encode("utf-8")).hexdigest()
        command = f"{interpreter} -c {shlex.quote(script)} {shlex.quote(encoded)}"
        upload_helper = None
        helper_repair_marker = None
        # A remote exec transport can include the complete shell command in its
        # request and error envelope. Large inline checkpoint/fingerprint
        # helpers can therefore be truncated before the outer exit marker is
        # returned, even though their structured stdout is tiny. Upload each
        # content-addressed helper once per sandbox and keep subsequent argv
        # bounded to the small payload only.
        if (
            len(command.encode("utf-8"))
            > _CODEFLOW_INLINE_HELPER_MAX_BYTES
            and callable(getattr(sandbox, "upload_file", None))
        ):
            digest = hashlib.blake2b(
                script.encode("utf-8"), digest_size=12
            ).hexdigest()
            remote_script = f"{_CODEFLOW_HELPER_DIR}/{digest}.py"
            uploaded = getattr(
                sandbox, _CODEFLOW_HELPER_CACHE_ATTRIBUTE, None
            )
            if not isinstance(uploaded, set):
                uploaded = set()
                try:
                    setattr(
                        sandbox,
                        _CODEFLOW_HELPER_CACHE_ATTRIBUTE,
                        uploaded,
                    )
                except Exception:
                    # Custom Sandbox protocol implementations may use slots;
                    # correctness only requires re-uploading in that case.
                    pass
            def upload_helper():
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w",
                        encoding="utf-8",
                        prefix="rllm-codeflow-helper-",
                        suffix=".py",
                    ) as handle:
                        handle.write(script)
                        handle.flush()
                        sandbox.upload_file(handle.name, remote_script)
                    uploaded.add(digest)
                except RolloutInfrastructureError:
                    # Preserve node/RPC failures, including fatal cleanup
                    # evidence, through the checkpoint and helper paths.
                    raise
                except Exception as exc:
                    detail = _truncate(str(exc), 1000)
                    kind = "transport" if any(
                        marker in detail.lower() for marker in _CODEFLOW_HELPER_RETRYABLE_TRANSPORT_MARKERS
                    ) else "execution"
                    diagnostics = {
                        "interpreter": runtime, "script_sha256": helper_digest,
                        "operation_kind": operation_kind, "error_kind": kind,
                        "error_type": type(exc).__name__, "stderr": detail,
                        "upload": dict(getattr(exc, "diagnostics", None) or {}),
                    }
                    sandbox._rllm_codeflow_last_helper_error = diagnostics
                    return {
                        "error": "failed to upload structured helper: " + detail,
                        "detail": detail, "_error_kind": kind, "diagnostics": diagnostics,
                    }
                return None

            if digest not in uploaded:
                failure = upload_helper()
                if failure is not None:
                    return failure
            # Verify and execute the SAME bytes, so deletion/replacement after
            # validation cannot redirect execution. Only this pre-execution
            # guard may authorize repair, including for mutating helpers.
            helper_repair_marker = secrets.token_hex(16)
            loader = """import hashlib,os,stat,sys
p,h,m=sys.argv[1:4]
try:
 f=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
 with os.fdopen(f,'rb') as r:
  if not stat.S_ISREG(os.fstat(r.fileno()).st_mode): raise ValueError('not regular')
  b=r.read(_HELPER_BYTE_LIMIT_)
 if hashlib.sha256(b).hexdigest()!=h: raise ValueError('digest mismatch')
except (OSError,ValueError):
 sys.stderr.write(m)
 sys.exit(86)
sys.argv=[p]+sys.argv[4:]
sys.path[0]=os.path.dirname(p)
exec(compile(b,p,'exec'),{'__name__':'__main__','__file__':p})
""".replace("_HELPER_BYTE_LIMIT_", str(len(script.encode("utf-8")) + 1))
            command = (
                f"{interpreter} -c {shlex.quote(loader)} {shlex.quote(remote_script)} "
                f"{helper_digest} {helper_repair_marker} {shlex.quote(encoded)}"
            )
        # These private helpers exchange UTF-8 JSON and pass decoded Git paths
        # to filesystem/subprocess APIs. Old task Python interpreters otherwise
        # use ASCII in a C locale (newer pathlib can silently omit those paths).
        # Scope the locale to instrumentation, not agent commands or verifier.
        command = (
            '_RLLM_HELPER_ORIGINAL_LC_ALL_SET="${LC_ALL+x}" '
            '_RLLM_HELPER_ORIGINAL_LC_ALL="${LC_ALL-}" '
            '_RLLM_HELPER_ORIGINAL_PYTHONIOENCODING_SET="${PYTHONIOENCODING+x}" '
            '_RLLM_HELPER_ORIGINAL_PYTHONIOENCODING="${PYTHONIOENCODING-}" '
            "LC_ALL=C.UTF-8 PYTHONIOENCODING=utf-8 " + command
        )
        timeout_seconds = float(timeout or self.command_timeout)
        deadline = time.monotonic() + timeout_seconds
        attempts = 3 if operation_kind in {"read_only", "idempotent"} else 1
        result: dict[str, Any] = {}
        helper_repaired = False
        for attempt in range(1, attempts + 2):
            framed = frame_structured_command(command)
            remaining = deadline - time.monotonic()
            if remaining < 1.0:
                diagnostics = {
                    "interpreter": runtime, "script_sha256": helper_digest,
                    "operation_kind": operation_kind, "error_kind": "transport",
                    "last_error": result.get("error"),
                }
                sandbox._rllm_codeflow_last_helper_error = diagnostics
                return {
                    "error": "structured helper deadline exceeded",
                    "_error_kind": "transport", "diagnostics": diagnostics,
                }
            try:
                execute = sandbox.exec
                if operation_kind == "read_only" and callable(getattr(sandbox, "exec_readonly", None)):
                    execute = sandbox.exec_readonly
                if operation_kind != "read_only" and callable(getattr(sandbox, "exec_confirmed", None)):
                    execute = sandbox.exec_confirmed
                raw = execute(
                    self._with_task_shell_context(framed.command, task),
                    timeout=max(1.0, remaining),
                    user=task.metadata.get("agent_user") if user is None else user,
                )
            except RolloutInfrastructureError:
                # Preserve typed node/RPC failures and fatal cleanup evidence.
                raise
            except Exception as exc:
                message = str(exc)
                lowered = message.casefold()
                if (
                    operation_kind != "read_only"
                    and callable(getattr(sandbox, "exec_confirmed", None))
                    and "background command failed (exit " not in lowered
                    and not re.search(r"command failed \(exit \d+\)", lowered)
                ):
                    raise RolloutInfrastructureError(
                        "agent_exec_unconfirmed",
                        "Remote mutating helper completion is unknown: " + message[:1000],
                        stage="agentflow", retryable=True, retry_scope="full_rollout",
                        diagnostics=getattr(exc, "diagnostics", None),
                    ) from exc
                kind = (
                    "transport"
                    if any(
                        marker in lowered
                        for marker in _CODEFLOW_HELPER_RETRYABLE_TRANSPORT_MARKERS
                    )
                    else "execution"
                )
                result = {"error": message, "detail": message[-1000:], "_error_kind": kind, "error_type": type(exc).__name__}
                if isinstance(getattr(exc, "diagnostics", None), dict):
                    result["backend_diagnostics"] = exc.diagnostics
            else:
                try:
                    framed_result = parse_structured_command_output(raw, framed.nonce)
                except StructuredCommandProtocolError as exc:
                    result = {
                        "error": "tool returned malformed structured output",
                        "detail": _truncate(f"{exc}; raw={raw}", 1000),
                        "_error_kind": "protocol",
                    }
                else:
                    if (
                        upload_helper is not None
                        and framed_result.exit_code == 86
                        and framed_result.stderr == helper_repair_marker
                        and not framed_result.stdout
                    ):
                        uploaded.discard(digest)
                        if not helper_repaired:
                            helper_repaired = True
                            failure = upload_helper()
                            if failure is not None:
                                return failure
                            continue
                        result = {
                            "error": "structured helper unavailable after re-upload",
                            "_error_kind": "execution", "phase": "helper_integrity",
                            "error_type": "HelperIntegrityError", "exit_code": 86,
                            "detail": remote_script,
                        }
                    elif framed_result.exit_code != 0:
                        output_tail = str(framed_result.stderr or framed_result.stdout)[-1000:]
                        exception_lines = re.findall(r"^([A-Za-z_][\w.]*(?:Error|Exception)):", output_tail, re.MULTILINE)
                        result = {
                            "error": f"structured helper exited with code {framed_result.exit_code}",
                            "exit_code": framed_result.exit_code,
                            "detail": output_tail,
                            "error_type": exception_lines[-1] if exception_lines else None,
                            "_error_kind": "execution",
                        }
                    else:
                        try:
                            parsed = json.loads(framed_result.stdout.strip())
                        except json.JSONDecodeError:
                            result = {
                                "error": "tool returned malformed structured output",
                                "detail": _truncate(
                                    framed_result.stderr or framed_result.stdout,
                                    1000,
                                ),
                                "_error_kind": "protocol",
                            }
                        else:
                            if isinstance(parsed, dict) and not parsed.get("_error_kind"):
                                if repository_helper:
                                    provenance = getattr(sandbox, "_rllm_repository_helper_provenance", {})
                                    provenance[helper_digest] = {
                                        "script_sha256": helper_digest, "interpreter": runtime,
                                        "operation_kind": operation_kind,
                                        "ok": parsed.get("ok"), "exit_code": framed_result.exit_code,
                                    }
                                    sandbox._rllm_repository_helper_provenance = provenance
                                if parsed.get("ok") is False:
                                    diagnostics = {
                                        "interpreter": runtime,
                                        "script_sha256": helper_digest,
                                        "operation_kind": operation_kind,
                                        "phase": parsed.get("phase"),
                                        "exit_code": parsed["exit_code"] if parsed.get("exit_code") is not None else framed_result.exit_code,
                                        "error_type": parsed.get("error_type"),
                                        "stderr": str(parsed.get("error") or "")[-1000:],
                                    }
                                    parsed["diagnostics"] = diagnostics
                                    sandbox._rllm_codeflow_last_helper_error = diagnostics
                                return parsed
                            result = parsed if isinstance(parsed, dict) else {
                                "error": "tool returned a non-object structured output",
                                "_error_kind": "protocol",
                            }
            retryable = result.get("_error_kind") in {"protocol", "transport"}
            if not retryable or attempt >= attempts + int(helper_repaired):
                diagnostics = {
                    "interpreter": runtime, "script_sha256": helper_digest,
                    "sandbox": getattr(sandbox, "runtime_diagnostics", {}),
                    "operation_kind": operation_kind, "exit_code": result.get("exit_code"),
                    "phase": result.get("phase") or "helper_execution", "error_type": result.get("error_type"),
                    "stderr": result.get("detail"), "error_kind": result.get("_error_kind"),
                }
                if result.get("backend_diagnostics"):
                    diagnostics["backend"] = result["backend_diagnostics"]
                result["diagnostics"] = diagnostics
                if result.get("detail"):
                    result["error"] = str(result.get("error", "helper failed")) + ": " + str(result["detail"])[:1000]
                sandbox._rllm_codeflow_last_helper_error = diagnostics
                return result
            logger.warning(
                "repository helper %s failure; retrying (%d/%d, operation=%s): %s",
                result.get("_error_kind"),
                attempt,
                attempts,
                operation_kind,
                result.get("detail") or result.get("error"),
            )
            time.sleep(min(1.0, 0.25 * (2 ** (attempt - 1))))
        return result

    def _with_task_shell_context(self, command: str, task: Task) -> str:
        env_vars = task.metadata.get("env_vars", {}) or task.metadata.get("environment", {}).get("env", {}) or {}
        exports = "; ".join(f"export {k}={shlex.quote(str(v))}" for k, v in env_vars.items())
        workdir = task.metadata.get("workdir")
        prefix = ""
        if exports:
            prefix += exports + "; "
        if workdir:
            prefix += f"cd {shlex.quote(str(workdir))} && "
        return prefix + command

    def _codeflow_bash_preflight_error(self, command: str, task: Task) -> str | None:
        """Reject obvious repository editors; the postcondition remains authoritative."""
        repository_root = self._repository_root(task).rstrip("/")
        root = re.escape(repository_root)
        checks = (
            (r"\bsed\b[^\n;&|]*(?:-[A-Za-z]*i\b|--in-place\b)", "sed in-place editing"),
            (r"\bperl\b[^\n;&|]*\s-pi(?:\s|$)", "perl in-place editing"),
            (
                r"\bgit(?:\s+(?:-[A-Za-z]+|--[A-Za-z-]+)(?:=\S+|\s+\S+)?)*\s+"
                r"(?:add|am|apply|checkout|cherry-pick|clean|commit|merge|mv|rebase|reset|restore|rm|stash|switch)\b",
                "Git working-tree or ref mutation",
            ),
            (rf"\b(?:cat|tee|cp|mv|rm)\b[^\n;&|]*{root}(?:/|\b)", "direct repository file mutation"),
            (rf"\bopen\s*\(\s*['\"]{root}/[^'\"]*['\"]\s*,\s*['\"][wax+]", "Python repository file writing"),
            (rf"\bPath\s*\(\s*['\"]{root}/[^'\"]*['\"]\s*\)\s*\.write_(?:text|bytes)\b", "Python repository file writing"),
        )
        for pattern, label in checks:
            if re.search(pattern, command, flags=re.IGNORECASE):
                return f"execute_bash cannot modify repository files ({label}); use edit_file and put temporary files under /tmp"

        # Commands run with the repository as their working directory, so a
        # literal relative redirection is necessarily a repository write.
        # Dynamic targets remain the responsibility of the transactional
        # postcondition because guessing shell expansion here would create
        # false positives for legitimate test commands.
        def obvious_repository_target(target: str) -> bool:
            normalized = target.removeprefix("./")
            first = normalized.split("/", 1)[0].lower()
            name = normalized.rsplit("/", 1)[-1].lower()
            source_suffixes = (
                ".c",
                ".cc",
                ".cpp",
                ".go",
                ".h",
                ".hpp",
                ".java",
                ".js",
                ".json",
                ".jsx",
                ".md",
                ".py",
                ".rs",
                ".rst",
                ".sh",
                ".toml",
                ".ts",
                ".tsx",
                ".yaml",
                ".yml",
            )
            return first in {"lib", "src", "test", "tests"} or name in {"makefile", "dockerfile"} or name.startswith("requirements") or name.endswith(source_suffixes)

        for match in re.finditer(
            r"(?<![<>])(?:\d*)>{1,2}(?![>&])\s*(?P<quote>['\"]?)(?P<target>[^\s;&|<>'\"]+)(?P=quote)",
            command,
        ):
            target = match.group("target")
            if target.startswith(("$", "/dev/", "/tmp/")):
                continue
            if target == repository_root or target.startswith(repository_root + "/") or (not target.startswith("/") and obvious_repository_target(target)):
                return "execute_bash cannot modify repository files (shell redirection); use edit_file and put temporary files under /tmp"

        for pattern in (
            r"\bopen\s*\(\s*['\"](?P<path>[^'\"]+)['\"]\s*,\s*['\"][wax+]",
            r"\bPath\s*\(\s*['\"](?P<path>[^'\"]+)['\"]\s*\)\s*\.write_(?:text|bytes)\b",
        ):
            for match in re.finditer(pattern, command):
                target = match.group("path")
                if target.startswith("/tmp/"):
                    continue
                if not target.startswith("/") or target == repository_root or target.startswith(repository_root + "/"):
                    return "execute_bash cannot modify repository files (Python file writing); use edit_file and put temporary files under /tmp"
        return None

    @staticmethod
    def _codeflow_bash_safety_preflight_error(command: str) -> str | None:
        """Reject ref-changing Git operations independently of edit freedom."""

        if re.search(
            r"\bgit(?:\s+(?:-[A-Za-z]+|--[A-Za-z-]+)(?:=\S+|\s+\S+)?)*\s+"
            r"(?:am|checkout|cherry-pick|commit|merge|rebase|reset|stash|switch)\b",
            command,
            flags=re.IGNORECASE,
        ):
            return (
                "execute_bash cannot change Git HEAD or refs; modify working-tree "
                "files only and leave repository history unchanged"
            )
        return None

    def _codeflow_normalized_protected_test_paths(
        self,
        task: Task,
    ) -> tuple[tuple[str, ...], bool]:
        """Return explicit protected paths and whether legacy R2E fallback applies."""

        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        explicit = metadata.get("protected_test_paths")
        if explicit is None:
            rllm_metadata = (
                metadata.get("rllm")
                if isinstance(metadata.get("rllm"), dict)
                else {}
            )
            explicit = rllm_metadata.get("protected_test_paths")
        if isinstance(explicit, str):
            raw_paths = [explicit]
        elif isinstance(explicit, list):
            raw_paths = [value for value in explicit if isinstance(value, str)]
        else:
            raw_paths = []

        normalized: list[str] = []
        for value in raw_paths:
            try:
                _, relative = self._codeflow_path(value, task)
            except ValueError:
                continue
            relative = relative.rstrip("/") or "."
            if relative not in normalized:
                normalized.append(relative)

        task_section = (
            metadata.get("task")
            if isinstance(metadata.get("task"), dict)
            else {}
        )
        task_keywords = (
            task_section.get("keywords")
            if isinstance(task_section.get("keywords"), list)
            else []
        )
        use_r2e_default = not normalized and (
            any(
                key in metadata
                for key in (
                    "baseline_output_json",
                    "expected_output_json",
                    "target_output_json",
                    "test_file_names",
                )
            )
            or "r2e-gym" in task_keywords
            or str(task_section.get("name") or "").startswith("r2egym/")
        )
        return tuple(normalized), use_r2e_default

    def _codeflow_protected_changed_paths(
        self,
        paths: Sequence[str],
        task: Task,
    ) -> list[str]:
        protected, use_r2e_default = (
            self._codeflow_normalized_protected_test_paths(task)
        )
        matched: list[str] = []
        for raw_path in paths:
            path = str(raw_path).rstrip("/") or "."
            explicitly_protected = any(
                root == "." or path == root or path.startswith(root + "/")
                for root in protected
            )
            if explicitly_protected or (
                use_r2e_default and is_probable_test_path(path)
            ):
                if path not in matched:
                    matched.append(path)
        return sorted(matched)

    @staticmethod
    def _codeflow_patch_paths(patch: str) -> list[str]:
        """Extract every repository path named by a unified Git patch."""

        paths: list[str] = []

        def add(candidate: str) -> None:
            candidate = candidate.strip()
            if candidate == "/dev/null":
                return
            if candidate.startswith(("a/", "b/")):
                candidate = candidate[2:]
            if candidate and candidate not in paths:
                paths.append(candidate)

        for line in patch.splitlines():
            if line.startswith("diff --git "):
                try:
                    fields = shlex.split(line)
                except ValueError as exc:
                    raise ValueError(f"invalid diff header: {exc}") from exc
                if len(fields) >= 4:
                    add(fields[2])
                    add(fields[3])
                continue
            if not line.startswith(("--- ", "+++ ")):
                continue
            raw = line[4:].split("\t", 1)[0].strip()
            if raw.startswith(('"', "'")):
                try:
                    fields = shlex.split(raw)
                except ValueError as exc:
                    raise ValueError(f"invalid patch path: {exc}") from exc
                raw = fields[0] if fields else ""
            add(raw)
        return paths

    @staticmethod
    def _codeflow_protected_test_error(paths: Sequence[str]) -> str:
        suffix = f" Attempted protected paths: {', '.join(paths)}." if paths else ""
        return _CODEFLOW_PROTECTED_TEST_GUIDANCE + suffix

    def _codeflow_validate_action(
        self,
        action: ParsedAction,
        task: Task,
        history: list[CodeFlowUndoEntry] | None = None,
    ) -> CodeFlowValidationResult:
        if action.status != "ok":
            return CodeFlowValidationResult(status=ValidationStatus.SKIPPED)

        name = action.name
        args = action.arguments
        if name not in CODEFLOW_NATIVE_TOOL_NAMES:
            return CodeFlowValidationResult(status=ValidationStatus.ERROR, error=f"unknown tool: {name}")
        available_names = (
            {"execute_bash", "submit"}
            if self.codeflow_tool_mode == "bash_only"
            else CODEFLOW_NATIVE_TOOL_NAMES
        )
        if name not in available_names:
            return CodeFlowValidationResult(
                status=ValidationStatus.ERROR,
                error=(
                    f"tool {name!r} is not available when "
                    "swe.codeflow_tool_mode=bash_only"
                ),
                subtype="tool_unavailable",
            )

        def fail(
            message: str,
            subtype: str = "",
            *,
            policy_violations: list[str] | None = None,
            shell_analysis: ShellBehaviorAnalysis | None = None,
        ) -> CodeFlowValidationResult:
            return CodeFlowValidationResult(
                status=ValidationStatus.ERROR,
                error=message,
                subtype=subtype,
                policy_violations=list(policy_violations or []),
                shell_analysis=shell_analysis,
            )

        def require_keys(allowed: set[str], subtype: str = "") -> CodeFlowValidationResult | None:
            extras = sorted(set(args) - allowed)
            if extras:
                return fail(f"unexpected arguments: {', '.join(extras)}", subtype)
            return None

        def integer(value: Any, label: str) -> int:
            if isinstance(value, bool) or not isinstance(value, int | float) or not float(value).is_integer():
                raise ValueError(f"{label} must be an integer")
            return int(value)

        def absolute_path(value: Any, *, allow_empty: bool = False) -> str:
            return self._codeflow_path(value, task, allow_empty=allow_empty)[0]

        try:
            if name == "search":
                invalid = require_keys({"mode", "query", "path", "include_globs", "context_lines", "max_results", "offset"})
                if invalid:
                    return invalid
                mode = args.get("mode")
                if mode not in {"text", "symbol", "file", "directory"}:
                    return fail("mode must be one of: text, symbol, file, directory")
                query = args.get("query")
                if not isinstance(query, str) or not query:
                    return fail("query must be a non-empty string", str(mode or ""))
                raw_globs = args.get("include_globs", [])
                if not isinstance(raw_globs, list) or any(not isinstance(item, str) for item in raw_globs):
                    return fail("include_globs must be an array of strings", mode)
                context_lines = integer(args.get("context_lines", 2), "context_lines")
                max_results = integer(args.get("max_results", CODEFLOW_DEFAULT_SEARCH_RESULTS), "max_results")
                offset = integer(args.get("offset", 0), "offset")
                if not 0 <= context_lines <= 20:
                    return fail("context_lines must be between 0 and 20", mode)
                if not 1 <= max_results <= 200:
                    return fail("max_results must be between 1 and 200", mode)
                if offset < 0:
                    return fail("offset must be greater than or equal to 0", mode)
                return CodeFlowValidationResult(
                    status=ValidationStatus.OK,
                    subtype=mode,
                    normalized_arguments={
                        "mode": mode,
                        "query": query,
                        "path": absolute_path(args.get("path") or "."),
                        "include_globs": list(raw_globs),
                        "context_lines": context_lines,
                        "max_results": max_results,
                        "offset": offset,
                    },
                )

            if name == "read_file":
                invalid = require_keys({"mode", "path", "start_line", "end_line"})
                if invalid:
                    return invalid
                mode = args.get("mode")
                if mode not in {"file", "diff", "status"}:
                    return fail("mode must be one of: file, diff, status")
                if mode != "file" and ({"start_line", "end_line"} & set(args)):
                    return fail("start_line and end_line are only valid for file mode", mode)
                if mode == "file":
                    path = absolute_path(args.get("path"))
                    start_line = integer(args.get("start_line", 1), "start_line")
                    end_line = integer(args.get("end_line", start_line + 199), "end_line")
                    if start_line < 1 or end_line < start_line:
                        return fail("line range must satisfy 1 <= start_line <= end_line", mode)
                    normalized = {"mode": mode, "path": path, "start_line": start_line, "end_line": end_line}
                else:
                    normalized = {"mode": mode, "path": absolute_path(args.get("path"), allow_empty=True)}
                return CodeFlowValidationResult(status=ValidationStatus.OK, subtype=mode, normalized_arguments=normalized)

            if name == "edit_file":
                invalid = require_keys({"mode", "path", "old_text", "new_text", "expected_count", "line", "patch"})
                if invalid:
                    return invalid
                mode = args.get("mode")
                if mode not in {"replace", "insert", "create", "delete", "apply_patch", "undo"}:
                    return fail("mode must be one of: replace, insert, create, delete, apply_patch, undo")
                allowed_by_mode = {
                    "replace": {"mode", "path", "old_text", "new_text", "expected_count"},
                    "insert": {"mode", "path", "new_text", "line"},
                    "create": {"mode", "path", "new_text"},
                    "delete": {"mode", "path"},
                    "apply_patch": {"mode", "path", "patch"},
                    "undo": {"mode", "path"},
                }[mode]
                extras = sorted(set(args) - allowed_by_mode)
                if extras:
                    return fail(f"arguments not valid for {mode}: {', '.join(extras)}", mode)
                normalized: dict[str, Any] = {"mode": mode}
                if mode in {"replace", "insert", "create", "delete"}:
                    normalized["path"] = absolute_path(args.get("path"))
                elif str(args.get("path") or "").strip():
                    normalized["path"] = absolute_path(args.get("path"))
                if mode == "replace":
                    if not isinstance(args.get("old_text"), str) or not args["old_text"]:
                        return fail("old_text must be a non-empty string", mode)
                    if not isinstance(args.get("new_text"), str):
                        return fail("new_text must be a string", mode)
                    expected_count = integer(args.get("expected_count", 1), "expected_count")
                    if expected_count < 1:
                        return fail("expected_count must be at least 1", mode)
                    normalized.update(
                        old_text=args["old_text"],
                        new_text=args["new_text"],
                        expected_count=expected_count,
                    )
                elif mode == "insert":
                    if not isinstance(args.get("new_text"), str) or not args["new_text"]:
                        return fail("new_text must be a non-empty string", mode)
                    line = integer(args.get("line"), "line")
                    if line < 0:
                        return fail("line must be non-negative", mode)
                    normalized.update(new_text=args["new_text"], line=line)
                elif mode == "create":
                    if not isinstance(args.get("new_text"), str):
                        return fail("new_text must be a string", mode)
                    normalized["new_text"] = args["new_text"]
                elif mode == "apply_patch":
                    if not isinstance(args.get("patch"), str) or not args["patch"].strip():
                        return fail("patch must be a non-empty string", mode)
                    normalized["patch"] = args["patch"]
                candidate_paths: list[str] = []
                if mode in {"replace", "insert", "create", "delete"}:
                    _, relative = self._codeflow_path(normalized["path"], task)
                    candidate_paths = [relative]
                elif mode == "apply_patch":
                    for candidate in self._codeflow_patch_paths(normalized["patch"]):
                        _, relative = self._codeflow_path(candidate, task)
                        candidate_paths.append(relative)
                elif mode == "undo":
                    if history:
                        candidate_paths.extend(history[-1].paths)
                    elif "path" in normalized:
                        _, relative = self._codeflow_path(normalized["path"], task)
                        candidate_paths.append(relative)
                protected_paths = self._codeflow_protected_changed_paths(
                    candidate_paths,
                    task,
                )
                if protected_paths:
                    return fail(
                        self._codeflow_protected_test_error(protected_paths),
                        mode,
                        policy_violations=["edit_modified_protected_tests"],
                    )
                return CodeFlowValidationResult(status=ValidationStatus.OK, subtype=mode, normalized_arguments=normalized)

            if name == "execute_bash":
                invalid = require_keys({"command", "timeout"}, "shell")
                if invalid:
                    return invalid
                command = args.get("command")
                if not isinstance(command, str) or not command.strip():
                    return fail("command must be a non-empty string", "shell")
                shell_analysis = analyze_repository_shell_behaviors(
                    command,
                    self._repository_root(task),
                )
                restricted_mode = set(self.codeflow_tool_restricted_mode)
                policy_error: str | None = None
                if "edit" in restricted_mode:
                    policy_error = self._codeflow_bash_preflight_error(
                        command, task
                    )
                    if policy_error and "edit" not in shell_analysis.behaviors:
                        shell_analysis = ShellBehaviorAnalysis(
                            behaviors=tuple(
                                behavior
                                for behavior in CODEFLOW_TOOL_BEHAVIOR_ORDER
                                if behavior
                                in {*shell_analysis.behaviors, "edit"}
                            ),
                            search_paths=shell_analysis.search_paths,
                            read_paths=shell_analysis.read_paths,
                            evidence=(*shell_analysis.evidence, "edit:preflight"),
                        )
                restricted_behaviors = [
                    behavior
                    for behavior in CODEFLOW_TOOL_BEHAVIOR_ORDER
                    if behavior in restricted_mode
                    and behavior in shell_analysis.behaviors
                    and (behavior != "edit" or policy_error is not None)
                ]
                if restricted_behaviors:
                    tool_for = {
                        "search": "search",
                        "read": "read_file",
                        "edit": "edit_file",
                    }
                    code_for = {
                        "search": "bash_repository_search_attempt",
                        "read": "bash_repository_read_attempt",
                        "edit": "bash_repository_mutation_attempt",
                    }
                    descriptions = "; ".join(
                        f"repository {behavior} is restricted in execute_bash; use {tool_for[behavior]} instead"
                        for behavior in restricted_behaviors
                    )
                    subtype = (
                        {
                            "search": "repository_search",
                            "read": "repository_read",
                            "edit": "repository_edit",
                        }[restricted_behaviors[0]]
                        if len(restricted_behaviors) == 1
                        else "repository_file_policy"
                    )
                    return fail(
                        f"execute_bash rejected: {descriptions}",
                        subtype,
                        policy_violations=[
                            code_for[behavior]
                            for behavior in restricted_behaviors
                        ],
                        shell_analysis=shell_analysis,
                    )
                safety_error = self._codeflow_bash_safety_preflight_error(
                    command
                )
                if safety_error:
                    return fail(
                        safety_error,
                        "repository_history",
                        policy_violations=["repository_head_change_attempt"],
                        shell_analysis=shell_analysis,
                    )
                rllm_metadata = task.metadata.get("rllm") or {}
                if rllm_metadata.get("denovo_background_finalize_active") is True:
                    detached_violation = detect_denovo_detached_process_violation(
                        command,
                        self._repository_root(task),
                    )
                    if detached_violation is not None:
                        return fail(
                            detached_violation.message,
                            detached_violation.subtype,
                            policy_violations=[detached_violation.code],
                            shell_analysis=shell_analysis,
                        )
                if (
                    task.metadata.get("bash_policy_profile")
                    or rllm_metadata.get("bash_policy_profile")
                ) == "repo_generation_package":
                    package_violation = detect_repo_generation_package_violation(
                        command,
                        dict(task.metadata),
                    )
                    if package_violation is not None:
                        return fail(
                            package_violation.message,
                            package_violation.subtype,
                            policy_violations=[package_violation.code],
                            shell_analysis=shell_analysis,
                        )
                if (
                    rllm_metadata.get(
                        "agent_environment_policy"
                    )
                    == "preconfigured_eval_v1"
                ):
                    environment_violation = (
                        detect_preconfigured_environment_violation(
                            command,
                            self._repository_root(task),
                        )
                    )
                    if environment_violation is not None:
                        return fail(
                            environment_violation.message,
                            environment_violation.subtype,
                            policy_violations=[environment_violation.code],
                            shell_analysis=shell_analysis,
                        )
                timeout = args.get("timeout", self.command_timeout)
                if isinstance(timeout, bool) or not isinstance(timeout, int | float):
                    return fail(
                        "timeout must be a number",
                        "shell",
                        shell_analysis=shell_analysis,
                    )
                timeout = float(timeout)
                if not math.isfinite(timeout) or timeout <= 0:
                    return fail(
                        "timeout must be a finite positive number",
                        "shell",
                        shell_analysis=shell_analysis,
                    )
                if (
                    _codeflow_eval_guard_enabled(task)
                    and timeout > float(self.command_timeout)
                ):
                    return fail(
                        f"timeout cannot exceed {float(self.command_timeout):g} seconds "
                        "in this evaluation",
                        "timeout_exceeds_limit",
                        policy_violations=["bash_timeout_limit_exceeded"],
                        shell_analysis=shell_analysis,
                    )
                return CodeFlowValidationResult(
                    status=ValidationStatus.OK,
                    subtype="shell",
                    normalized_arguments={"command": command, "timeout": timeout},
                    shell_analysis=shell_analysis,
                )

            invalid = require_keys(set(), "submit")
            if invalid:
                return invalid
            return CodeFlowValidationResult(status=ValidationStatus.OK, subtype="submit", normalized_arguments={})
        except ValueError as exc:
            return fail(str(exc), str(args.get("mode") or ""))

    def _codeflow_execute_validated(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        sandbox: Sandbox,
        task: Task,
        history: list[CodeFlowUndoEntry],
    ) -> CodeFlowExecutionResult:
        if tool_name == "submit":
            return CodeFlowExecutionResult(observation="Task submitted.", done=True, status=ExecutionStatus.SUCCESS)
        if tool_name == "execute_bash":
            return self._codeflow_execute_bash(arguments, sandbox, task)
        if tool_name == "edit_file":
            return self._codeflow_execute_edit_validated(
                arguments,
                sandbox,
                task,
                history,
            )

        if tool_name == "search" and "_codeflow_search" not in self.__dict__:
            return self._codeflow_search_execution(arguments, sandbox, task)
        if tool_name == "read_file" and "_codeflow_read_file" not in self.__dict__:
            return self._codeflow_read_file_execution(arguments, sandbox, task)

        started = time.perf_counter()
        try:
            if tool_name == "search":
                observation = self._codeflow_search(arguments, sandbox, task)
            elif tool_name == "read_file":
                observation = self._codeflow_read_file(arguments, sandbox, task)
            else:
                observation = _json_observation({"error": f"unknown tool: {tool_name}"}, CODEFLOW_SHORT_OUTPUT_LIMIT)
        except Exception as exc:
            duration = time.perf_counter() - started
            return CodeFlowExecutionResult(
                observation=_json_observation({"error": str(exc)}, CODEFLOW_SHORT_OUTPUT_LIMIT),
                status=ExecutionStatus.ERROR,
                error=str(exc),
                duration=duration,
            )

        duration = time.perf_counter() - started
        try:
            payload = json.loads(observation)
        except json.JSONDecodeError:
            return CodeFlowExecutionResult(
                observation=observation,
                status=ExecutionStatus.ERROR,
                error="tool returned malformed structured output",
                duration=duration,
            )
        if not isinstance(payload, dict):
            return CodeFlowExecutionResult(
                observation=observation,
                status=ExecutionStatus.ERROR,
                error="tool returned a non-object structured output",
                duration=duration,
            )
        observation = _json_observation(payload, CODEFLOW_SHORT_OUTPUT_LIMIT)
        error = payload.get("error")
        if error is not None or payload.get("ok") is False:
            return CodeFlowExecutionResult(
                observation=observation,
                status=ExecutionStatus.ERROR,
                error=str(error or "tool operation failed"),
                duration=duration,
            )
        return CodeFlowExecutionResult(
            observation=observation,
            status=ExecutionStatus.SUCCESS,
            duration=duration,
            observation_stats=_observation_stats(
                observation,
                source_characters=len(_compact_json(payload)),
                output_limit=CODEFLOW_SHORT_OUTPUT_LIMIT,
                compaction=("compact_json",),
            ),
        )

    def _codeflow_execute_edit_validated(
        self,
        arguments: dict[str, Any],
        sandbox: Sandbox,
        task: Task,
        history: list[CodeFlowUndoEntry],
    ) -> CodeFlowExecutionResult:
        """Execute one edit behind a fail-closed checkpoint and postcondition."""

        started = time.perf_counter()
        before = self._codeflow_capture_repo_snapshot(sandbox, task, None)
        if before.status != RepoStateStatus.OK or not before.fingerprint:
            error = (
                "repository checkpoint unavailable before edit_file: "
                f"{before.error or before.status.value}"
            )
            return CodeFlowExecutionResult(
                observation=_json_observation(
                    {"error": error, "fatal": True},
                    CODEFLOW_SHORT_OUTPUT_LIMIT,
                ),
                status=ExecutionStatus.ERROR,
                error=error,
                duration=time.perf_counter() - started,
                instrumentation_errors=[error],
                fatal=True,
            )

        checkpoint, checkpoint_error = self._codeflow_create_repo_checkpoint(
            sandbox,
            task,
        )
        if checkpoint is None:
            error = (
                "repository checkpoint unavailable before edit_file: "
                f"{checkpoint_error or 'unknown failure'}"
            )
            return CodeFlowExecutionResult(
                observation=_json_observation(
                    {"error": error, "fatal": True},
                    CODEFLOW_SHORT_OUTPUT_LIMIT,
                ),
                status=ExecutionStatus.ERROR,
                error=error,
                duration=time.perf_counter() - started,
                instrumentation_errors=[error],
                fatal=True,
            )

        history_length = len(history)
        try:
            observation = self._codeflow_edit_file(
                arguments,
                sandbox,
                task,
                history,
            )
        except RolloutInfrastructureError:
            raise
        except Exception as exc:
            observation = _json_observation({"error": str(exc)})

        attempted_after = self._codeflow_capture_repo_snapshot(
            sandbox,
            task,
            checkpoint.baseline_ref,
        )
        if (
            attempted_after.status != RepoStateStatus.OK
            or not attempted_after.fingerprint
        ):
            restore_error = self._codeflow_restore_repo_checkpoint(
                checkpoint,
                sandbox,
                task,
            )
            del history[history_length:]
            final = self._codeflow_capture_repo_snapshot(
                sandbox,
                task,
                checkpoint.baseline_ref,
            )
            self._codeflow_discard_repo_checkpoint(checkpoint, sandbox, task)
            restore_protocol_fallback = (
                restore_error is not None
                and restore_error.kind in {"protocol", "transport"}
            )
            restored = (
                (restore_error is None or restore_protocol_fallback)
                and final.status == RepoStateStatus.OK
                and final.fingerprint == before.fingerprint
            )
            error = "could not verify repository state after edit_file"
            if not restored:
                error += (
                    "; rollback failed: "
                    f"{restore_error or final.error or 'fingerprint mismatch'}"
                )
            return CodeFlowExecutionResult(
                observation=_json_observation(
                    {
                        "error": error,
                        "fatal": not restored,
                        "repository_restored": restored,
                    },
                    CODEFLOW_SHORT_OUTPUT_LIMIT,
                ),
                status=ExecutionStatus.ERROR,
                error=error,
                duration=time.perf_counter() - started,
                instrumentation_errors=[error],
                fatal=not restored,
            )

        attempted_changes = changed_files_between(before, attempted_after)
        attempted_paths = {
            path
            for item in attempted_changes
            for path in (item.path, item.old_path)
            if isinstance(path, str)
        }
        protected_paths = self._codeflow_protected_changed_paths(
            sorted(attempted_paths),
            task,
        )
        if protected_paths:
            restore_error = self._codeflow_restore_repo_checkpoint(
                checkpoint,
                sandbox,
                task,
            )
            del history[history_length:]
            final = self._codeflow_capture_repo_snapshot(
                sandbox,
                task,
                checkpoint.baseline_ref,
            )
            self._codeflow_discard_repo_checkpoint(checkpoint, sandbox, task)
            restore_protocol_fallback = (
                restore_error is not None
                and restore_error.kind in {"protocol", "transport"}
            )
            restored = (
                (restore_error is None or restore_protocol_fallback)
                and final.status == RepoStateStatus.OK
                and final.fingerprint == before.fingerprint
            )
            error = self._codeflow_protected_test_error(protected_paths)
            error += (
                " The repository was restored."
                if restored
                else " Repository rollback failed."
            )
            attempted_payload = [
                item.model_dump(mode="json") for item in attempted_changes
            ]
            return CodeFlowExecutionResult(
                observation=_json_observation(
                    {
                        "error": error,
                        "fatal": not restored,
                        "repository_restored": restored,
                        "attempted_changed_files": attempted_payload,
                    },
                    CODEFLOW_SHORT_OUTPUT_LIMIT,
                ),
                status=ExecutionStatus.ERROR,
                error=error,
                duration=time.perf_counter() - started,
                attempted_changed_files=attempted_changes,
                policy_violations=["edit_modified_protected_tests"],
                instrumentation_errors=(
                    []
                    if restored
                    else [
                        "edit_file protected-test rollback failed: "
                        f"{restore_error or final.error or 'fingerprint mismatch'}"
                    ]
                ),
                fatal=not restored,
            )

        self._codeflow_discard_repo_checkpoint(checkpoint, sandbox, task)
        duration = time.perf_counter() - started
        try:
            payload = json.loads(observation)
        except json.JSONDecodeError:
            return CodeFlowExecutionResult(
                observation=observation,
                status=ExecutionStatus.ERROR,
                error="tool returned malformed structured output",
                duration=duration,
            )
        if not isinstance(payload, dict):
            return CodeFlowExecutionResult(
                observation=observation,
                status=ExecutionStatus.ERROR,
                error="tool returned a non-object structured output",
                duration=duration,
            )
        observation = _json_observation(payload, CODEFLOW_SHORT_OUTPUT_LIMIT)
        error = payload.get("error")
        if error is not None or payload.get("ok") is False:
            return CodeFlowExecutionResult(
                observation=observation,
                status=ExecutionStatus.ERROR,
                error=str(error or "tool operation failed"),
                duration=duration,
            )
        return CodeFlowExecutionResult(
            observation=observation,
            status=ExecutionStatus.SUCCESS,
            duration=duration,
            observation_stats=_observation_stats(
                observation,
                source_characters=len(_compact_json(payload)),
                output_limit=CODEFLOW_SHORT_OUTPUT_LIMIT,
                compaction=("compact_json",),
            ),
        )

    def _codeflow_create_repo_checkpoint(
        self,
        sandbox: Sandbox,
        task: Task,
        *,
        user: str | None = None,
    ) -> tuple[CodeFlowRepoCheckpoint | None, str | None]:
        script = repository_helper_source() + r'''
import base64
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

payload = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
root = Path(payload["root"]).resolve()
checkpoint_parent = Path(payload["checkpoint_parent"])
checkpoint = None

def git(*args, check=True):
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={root}", "-C", str(root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and completed.returncode != 0:
        message = (completed.stderr or completed.stdout or b"git command failed").decode("utf-8", errors="replace").strip()
        error = RuntimeError(message)
        error.exit_code = completed.returncode
        raise error
    return completed

def decode_paths(raw):
    return [os.fsdecode(value) for value in raw.split(b"\0") if value]

def path_key(relative):
    return base64.b64encode(os.fsencode(relative)).decode("ascii")

def empty_directory_entries(ignored_directories):
    entries = []
    for current, directories, _files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        # Git control directories are restored by Git itself (or by the
        # dedicated gitlink path below) and must never become repository
        # checkpoint payload.
        retained_directories = []
        for name in directories:
            candidate = current_path / name
            relative = os.fsdecode(os.fsencode(candidate.relative_to(root)))
            if (
                name == ".git"
                or candidate.is_symlink()
                or relative in ignored_directories
                or relative in gitlink_paths
            ):
                continue
            retained_directories.append(name)
        directories[:] = retained_directories
        if current_path == root:
            continue
        try:
            if any(current_path.iterdir()):
                continue
            info = current_path.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISDIR(info.st_mode):
            continue
        relative = os.fsdecode(os.fsencode(current_path.relative_to(root)))
        entries.append(
            {
                "path": relative,
                "path_b64": path_key(relative),
                "mode": stat.S_IMODE(info.st_mode),
            }
        )
    return sorted(entries, key=lambda entry: base64.b64decode(entry["path_b64"]))

def directory_fingerprint(candidate, include_git=False):
    return repo_directory_fingerprint(root, candidate, include_git)


gitlink_paths = set()

def archive_filter(member):
    parts = Path(member.name).parts
    if ".git" in parts:
        return None
    return member

def gitlink_metadata(relative, candidate):
    def nested_git(*args, check=True):
        completed = subprocess.run(
            ["git", "-c", f"safe.directory={candidate}", "-C", str(candidate), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if check and completed.returncode != 0:
            message = (completed.stderr or completed.stdout or b"nested git command failed").decode("utf-8", errors="replace").strip()
            raise RuntimeError("gitlink %s: %s" % (relative, message))
        return completed

    head = nested_git("rev-parse", "HEAD").stdout.decode("ascii").strip()
    symbolic_result = nested_git("symbolic-ref", "-q", "HEAD", check=False)
    symbolic = symbolic_result.stdout.decode("utf-8", errors="replace").strip() if symbolic_result.returncode == 0 else None
    raw_index = nested_git("rev-parse", "--git-path", "index").stdout.decode("utf-8", errors="surrogateescape").strip()
    index_path = Path(raw_index)
    if not index_path.is_absolute():
        index_path = candidate / index_path
    if index_path.is_symlink() or not index_path.is_file():
        raise RuntimeError("gitlink index is unavailable: %s" % relative)
    index_name = hashlib.sha256(os.fsencode(relative)).hexdigest() + ".index"
    index_archive = checkpoint / "gitlink-indexes" / index_name
    index_archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(index_path, index_archive)
    control = candidate / ".git"
    control_kind = "directory" if control.is_dir() else "file" if control.is_file() else "missing"
    control_bytes = control.read_bytes() if control_kind == "file" else b""
    return {
        "git_head": head,
        "git_symbolic_head": symbolic,
        "git_index_archive": "gitlink-indexes/" + index_name,
        "git_index_sha256": hashlib.sha256(index_archive.read_bytes()).hexdigest(),
        "git_control_kind": control_kind,
        "git_control_mode": stat.S_IMODE(control.lstat().st_mode) if control_kind != "missing" else None,
        "git_control_b64": base64.b64encode(control_bytes).decode("ascii") if control_kind == "file" else None,
        "git_control_sha256": hashlib.sha256(control_bytes).hexdigest() if control_kind == "file" else None,
    }

def file_entry(relative):
    candidate = root / relative
    info = repo_lstat(root, relative)
    if info is None:
        return {"path": relative, "path_b64": path_key(relative), "kind": "missing"}
    mode = stat.S_IMODE(info.st_mode)
    if stat.S_ISLNK(info.st_mode):
        target = os.readlink(candidate)
        encoded = os.fsencode(target)
        return {
            "path": relative,
            "path_b64": path_key(relative),
            "kind": "symlink",
            "mode": mode,
            "size": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    if stat.S_ISREG(info.st_mode):
        digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return {
            "path": relative,
            "path_b64": path_key(relative),
            "kind": "file",
            "mode": mode,
            "size": info.st_size,
            "sha256": digest.hexdigest(),
        }
    if stat.S_ISDIR(info.st_mode) and relative in gitlink_paths:
        fingerprint = directory_fingerprint(candidate)
        if fingerprint is None:
            raise RuntimeError("unsupported checkpoint path inside gitlink: %s" % relative)
        digest, size, entries = fingerprint
        return {
            "path": relative,
            "path_b64": path_key(relative),
            "kind": "gitlink_worktree",
            "mode": mode,
            "size": size,
            "entries": entries,
            "sha256": digest,
            **gitlink_metadata(relative, candidate),
        }
    if stat.S_ISDIR(info.st_mode):
        fingerprint = directory_fingerprint(candidate, include_git=True)
        if fingerprint is None:
            raise RuntimeError("unsupported checkpoint path inside untracked directory: %s" % relative)
        digest, size, entries = fingerprint
        return {
            "path": relative, "path_b64": path_key(relative),
            "kind": "untracked_directory" if relative in untracked else "tracked_directory", "mode": mode,
            "size": size, "entries": entries, "sha256": digest,
        }
    raise RuntimeError("unsupported checkpoint path type: %s" % relative)

try:
    if git("rev-parse", "--is-inside-work-tree").stdout.strip() != b"true":
        raise RuntimeError("repository root is not a Git work tree")
    head = git("rev-parse", "HEAD").stdout.decode("ascii").strip()
    symbolic_result = git("symbolic-ref", "-q", "HEAD", check=False)
    symbolic = symbolic_result.stdout.decode("utf-8", errors="replace").strip() if symbolic_result.returncode == 0 else None
    checkpoint_parent.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(tempfile.mkdtemp(prefix="rllm-codeflow-checkpoint-", dir=checkpoint_parent))
    patch = git("diff", "--binary", "--full-index", "HEAD", "--").stdout
    (checkpoint / "working.patch").write_bytes(patch)

    raw_index = git("rev-parse", "--git-path", "index").stdout.decode("utf-8", errors="surrogateescape").strip()
    index_path = Path(raw_index)
    if not index_path.is_absolute():
        index_path = root / index_path
    if index_path.is_symlink():
        raise RuntimeError("repository index is a symlink")
    index_exists = index_path.is_file()
    if index_exists:
        shutil.copyfile(index_path, checkpoint / "index")

    # Disable rename detection so both sides of a staged/unstaged rename are
    # represented. The content archives are authoritative for restore;
    # working.patch remains as an audit/debug artifact.
    tracked = decode_paths(
        git("diff", "--name-only", "--no-renames", "-z", "HEAD", "--").stdout
    )
    untracked = decode_paths(
        git("ls-files", "--others", "--exclude-standard", "-z").stdout
    )
    tracked_delta = list(tracked)
    cached_paths = set(decode_paths(git("ls-files", "--cached", "-z").stdout))
    filesystem = repo_filesystem_metadata(root, cached_paths | set(untracked))
    extra_paths = set(filesystem["file_modes"])
    for linked in filesystem["hardlinks"]:
        extra_paths.update(linked)
    tracked.extend(extra_paths & cached_paths)
    for record in git("ls-files", "--stage", "-z").stdout.split(b"\0"):
        if not record:
            continue
        header, separator, raw_path = record.partition(b"\t")
        if separator and header.split(b" ", 1)[0] == b"160000":
            gitlink_paths.add(os.fsdecode(raw_path))
    tracked = sorted(set(tracked), key=os.fsencode)
    untracked = sorted(set(untracked), key=os.fsencode)
    ignored_directories = {
        os.fsdecode(value[:-1])
        for value in git(
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "--directory",
            "-z",
        ).stdout.split(b"\0")
        if value.endswith(b"/")
    }
    present = [relative for relative in tracked + untracked
               if repo_lstat(root, relative) is not None]
    repo_archive(root, checkpoint / "contents.tar", present, exclude_git_roots=gitlink_paths,
                 parent_directories=filesystem["directories"])

    manifest = {
        "schema_version": 5,
        "tracked_delta": [path_key(relative) for relative in tracked_delta],
        "tracked": [file_entry(relative) for relative in tracked],
        "untracked": [file_entry(relative) for relative in untracked],
        # Git does not represent empty directories, and the content archives
        # only contain files.  Preserve empty leaves explicitly so a probe's
        # ``git clean -ffd`` cannot remove directories created by an earlier
        # replayed command that a later command still depends on.
        "empty_directories": empty_directory_entries(ignored_directories),
        "index_sha256": (
            hashlib.sha256((checkpoint / "index").read_bytes()).hexdigest()
            if index_exists
            else None
        ),
    }
    (checkpoint / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )

    metadata = {
        "root": str(root),
        "head": head,
        "symbolic_head": symbolic,
        "index_path": str(index_path),
        "index_exists": index_exists,
        "manifest_schema_version": 5,
    }
    (checkpoint / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )
    result = {"ok": True, "path": str(checkpoint), "baseline_ref": head}
except Exception as exc:
    if checkpoint is not None:
        shutil.rmtree(checkpoint, ignore_errors=True)
    result = {"ok": False, "error": str(exc), "error_type": type(exc).__name__, "phase": "checkpoint_create", "exit_code": getattr(exc, "exit_code", None)}
print(json.dumps(result, ensure_ascii=False))
'''
        result = self._codeflow_run_json_script(
            script,
            {
                "root": self._repository_root(task),
                "checkpoint_parent": _CODEFLOW_CONTROL_TMPDIR,
            },
            sandbox,
            task,
            timeout=self.command_timeout,
            user=user,
        )
        if not result.get("ok"):
            return None, str(result.get("error") or "failed to create repository checkpoint")
        path = str(result.get("path") or "")
        baseline_ref = str(result.get("baseline_ref") or "")
        if not path.startswith(_CODEFLOW_CHECKPOINT_PREFIX) or not baseline_ref:
            return None, "repository checkpoint returned malformed metadata"
        return CodeFlowRepoCheckpoint(path=path, baseline_ref=baseline_ref), None

    def _codeflow_restore_repo_checkpoint(
        self,
        checkpoint: CodeFlowRepoCheckpoint,
        sandbox: Sandbox,
        task: Task,
        *,
        user: str | None = None,
    ) -> CodeFlowRestoreError | None:
        script = repository_helper_source() + r'''
import base64
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

payload = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
root = Path(payload["root"]).resolve()
checkpoint = Path(payload["checkpoint"])
checkpoint_prefix = str(payload["checkpoint_prefix"])

def git(*args, input_bytes=None, check=True):
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={root}", "-C", str(root), *args],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and completed.returncode != 0:
        message = (completed.stderr or completed.stdout or b"git command failed").decode("utf-8", errors="replace").strip()
        error = RuntimeError(message)
        error.exit_code = completed.returncode
        raise error
    return completed

def decode_path(path_b64):
    return os.fsdecode(base64.b64decode(path_b64))

def path_key(relative):
    return base64.b64encode(os.fsencode(relative)).decode("ascii")

def directory_fingerprint(candidate, include_git=False):
    return repo_directory_fingerprint(root, candidate, include_git)


def gitlink_current_metadata(candidate):
    def nested_git(*args, check=True):
        completed = subprocess.run(
            ["git", "-c", f"safe.directory={candidate}", "-C", str(candidate), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if check and completed.returncode != 0:
            return None
        return completed

    head_result = nested_git("rev-parse", "HEAD")
    if head_result is None:
        return None
    symbolic_result = nested_git("symbolic-ref", "-q", "HEAD", check=False)
    raw_index_result = nested_git("rev-parse", "--git-path", "index")
    if raw_index_result is None:
        return None
    raw_index = raw_index_result.stdout.decode("utf-8", errors="surrogateescape").strip()
    index_path = Path(raw_index)
    if not index_path.is_absolute():
        index_path = candidate / index_path
    if not index_path.is_file():
        return None
    control = candidate / ".git"
    control_kind = "directory" if control.is_dir() else "file" if control.is_file() else "missing"
    control_bytes = control.read_bytes() if control_kind == "file" else b""
    return {
        "git_head": head_result.stdout.decode("ascii").strip(),
        "git_symbolic_head": symbolic_result.stdout.decode("utf-8", errors="replace").strip() if symbolic_result is not None and symbolic_result.returncode == 0 else None,
        "git_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "git_control_kind": control_kind,
        "git_control_mode": stat.S_IMODE(control.lstat().st_mode) if control_kind != "missing" else None,
        "git_control_sha256": hashlib.sha256(control_bytes).hexdigest() if control_kind == "file" else None,
    }

def current_entry(saved):
    relative = decode_path(saved["path_b64"])
    candidate = root / relative
    info = repo_lstat(root, relative)
    if info is None:
        return {"path_b64": saved["path_b64"], "kind": "missing"}
    mode = stat.S_IMODE(info.st_mode)
    if stat.S_ISLNK(info.st_mode):
        target = os.readlink(candidate)
        encoded = os.fsencode(target)
        return {
            "path_b64": saved["path_b64"],
            "kind": "symlink",
            "mode": mode,
            "size": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    if stat.S_ISREG(info.st_mode):
        digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return {
            "path_b64": saved["path_b64"],
            "kind": "file",
            "mode": mode,
            "size": info.st_size,
            "sha256": digest.hexdigest(),
        }
    if stat.S_ISDIR(info.st_mode) and saved.get("kind") in {"untracked_directory", "tracked_directory"}:
        fingerprint = directory_fingerprint(candidate, include_git=True)
        if fingerprint is None:
            return {"path_b64": saved["path_b64"], "kind": "unsupported", "mode": mode}
        digest, size, entries = fingerprint
        return {
            "path_b64": saved["path_b64"], "kind": saved["kind"],
            "mode": mode, "size": size, "entries": entries, "sha256": digest,
        }
    if stat.S_ISDIR(info.st_mode) and saved.get("kind") == "gitlink_worktree":
        fingerprint = directory_fingerprint(candidate)
        if fingerprint is None:
            return {"path_b64": saved["path_b64"], "kind": "unsupported", "mode": mode}
        git_metadata = gitlink_current_metadata(candidate)
        if git_metadata is None:
            return {"path_b64": saved["path_b64"], "kind": "unsupported", "mode": mode}
        digest, size, entries = fingerprint
        return {
            "path_b64": saved["path_b64"],
            "kind": "gitlink_worktree",
            "mode": mode,
            "size": size,
            "entries": entries,
            "sha256": digest,
            **git_metadata,
        }
    return {"path_b64": saved["path_b64"], "kind": "unsupported", "mode": mode}

def comparable(entry):
    return {
        key: entry.get(key)
        for key in (
            "path_b64", "kind", "mode", "size", "entries", "sha256",
            "git_head", "git_symbolic_head", "git_index_sha256",
            "git_control_kind", "git_control_mode", "git_control_sha256",
        )
        if key in entry
    }

def safe_extract(archive_path):
    with tarfile.open(archive_path, "r") as archive:
        repo_extract_archive(archive, root)


def restore_empty_directory(saved):
    relative = decode_path(saved.get("path_b64") or "")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts or ".git" in pure.parts:
        raise RuntimeError("unsafe checkpoint directory: %s" % relative)
    target = root
    for part in pure.parts:
        target = target / part
        if target.is_symlink():
            raise RuntimeError("checkpoint directory traverses symlink: %s" % relative)
        if target.exists() and not target.is_dir():
            raise RuntimeError("checkpoint directory path changed type: %s" % relative)
        target.mkdir(exist_ok=True)
    os.chmod(target, int(saved["mode"]))
    return target

try:
    if not str(checkpoint).startswith(checkpoint_prefix):
        raise RuntimeError("invalid repository checkpoint path")
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    if Path(metadata["root"]).resolve() != root:
        raise RuntimeError("repository checkpoint root mismatch")
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    manifest_schema_version = manifest.get("schema_version")
    if (
        manifest_schema_version not in {3, 4, 5}
        or metadata.get("manifest_schema_version") != manifest_schema_version
    ):
        raise RuntimeError("unsupported repository checkpoint manifest")
    empty_directories = manifest.get("empty_directories", [])
    if not isinstance(empty_directories, list) or (
        manifest_schema_version >= 4
        and any(
            not isinstance(saved, dict)
            or not isinstance(saved.get("path_b64"), str)
            or not isinstance(saved.get("mode"), int)
            for saved in empty_directories
        )
    ):
        raise RuntimeError("malformed repository checkpoint directories")
    archives = [checkpoint / "contents.tar"] if manifest_schema_version >= 5 else [
        checkpoint / "tracked.tar", checkpoint / "untracked.tar"]
    # Preflight every archive and manifest path before changing HEAD or files.
    planned_directories = set()
    for archive_path in archives:
        with tarfile.open(archive_path, "r") as archive:
            members = repo_validate_archive(archive, root)
            planned_directories.update(name for name, member in members.items() if member.isdir())
    for saved in manifest["tracked"] + manifest["untracked"] + empty_directories:
        relative = repo_relative(decode_path(saved["path_b64"]))
        if saved.get("kind") != "missing":
            repo_path(root, relative, planned_directories=planned_directories)
    index_path = Path(str(metadata["index_path"]))
    current_index = Path(os.fsdecode(git("rev-parse", "--git-path", "index").stdout.strip()))
    if not current_index.is_absolute():
        current_index = root / current_index
    if index_path != current_index or index_path.is_symlink():
        raise RuntimeError("repository checkpoint index path mismatch")
    if metadata.get("index_exists") and (
        (checkpoint / "index").is_symlink()
        or hashlib.sha256((checkpoint / "index").read_bytes()).hexdigest() != manifest.get("index_sha256")
    ):
        raise RuntimeError("repository checkpoint index payload mismatch")
    head = str(metadata["head"])
    symbolic = metadata.get("symbolic_head")
    if symbolic:
        git("symbolic-ref", "HEAD", str(symbolic))
        git("update-ref", str(symbolic), head)
    else:
        git("update-ref", "--no-deref", "HEAD", head)
    git("reset", "--hard", head)
    git("clean", "-ffd")

    for saved in manifest["tracked"]:
        if saved.get("kind") != "missing":
            continue
        target = root / decode_path(saved["path_b64"])
        if repo_lstat(root, decode_path(saved["path_b64"])) is None:
            continue
        if not target.is_symlink() and target.is_dir():
            shutil.rmtree(target)
        else:
            try:
                target.unlink()
            except FileNotFoundError:
                pass
    # A tracked symlink/file may have been replaced by a directory in the
    # image. Reset recreates the index type; remove it before extracting the
    # saved directory, otherwise tar can follow the freshly restored symlink.
    for saved in manifest["tracked"]:
        if saved.get("kind") != "tracked_directory":
            continue
        target = root / decode_path(saved["path_b64"])
        if not target.is_symlink() and target.is_dir():
            shutil.rmtree(target)
        elif target.is_symlink() or target.exists():
            target.unlink()

    for saved in manifest["tracked"]:
        if saved.get("kind") != "gitlink_worktree":
            continue
        target = root / decode_path(saved["path_b64"])
        target.mkdir(parents=True, exist_ok=True)
        for child in list(target.iterdir()):
            if child.name == ".git":
                continue
            if not child.is_symlink() and child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    for archive_path in archives:
        safe_extract(archive_path)
    restored_empty_directories = [
        restore_empty_directory(saved) for saved in empty_directories
    ]

    if metadata.get("index_exists"):
        index_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checkpoint / "index", index_path)
        # Verify the byte-for-byte restore before invoking any more Git
        # commands.  Read-only-looking commands such as ``git diff`` are
        # allowed to refresh the index stat cache and optional extensions.
        # That changes the index file's bytes without changing its staged
        # entries, and previously made every otherwise successful restore in
        # a non-trivial repository fail with ``<git-index>``.
        restored_index_hash = hashlib.sha256(index_path.read_bytes()).hexdigest()
        if restored_index_hash != manifest.get("index_sha256"):
            raise RuntimeError("repository checkpoint index copy mismatch")
    elif index_path.exists():
        index_path.unlink()

    for saved in manifest["tracked"]:
        if saved.get("kind") != "gitlink_worktree":
            continue
        target = root / decode_path(saved["path_b64"])
        control = target / ".git"
        control_kind = saved.get("git_control_kind")
        if control_kind == "file":
            if control.is_dir():
                raise RuntimeError("gitlink control path changed type: %s" % target)
            control.write_bytes(base64.b64decode(saved.get("git_control_b64") or ""))
            os.chmod(control, int(saved["git_control_mode"]))
        elif control_kind == "directory":
            if not control.is_dir():
                raise RuntimeError("gitlink control directory is unavailable: %s" % target)
        else:
            raise RuntimeError("unsupported gitlink control metadata: %s" % target)

        def nested_git(*args, check=True):
            completed = subprocess.run(
                ["git", "-c", f"safe.directory={target}", "-C", str(target), *args],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if check and completed.returncode != 0:
                message = (completed.stderr or completed.stdout or b"nested git command failed").decode("utf-8", errors="replace").strip()
                raise RuntimeError("gitlink %s: %s" % (target, message))
            return completed

        nested_symbolic = saved.get("git_symbolic_head")
        nested_head = str(saved.get("git_head") or "")
        if not nested_head:
            raise RuntimeError("gitlink head is unavailable: %s" % target)
        if nested_symbolic:
            nested_git("symbolic-ref", "HEAD", str(nested_symbolic))
            nested_git("update-ref", str(nested_symbolic), nested_head)
        else:
            nested_git("update-ref", "--no-deref", "HEAD", nested_head)
        raw_nested_index = nested_git("rev-parse", "--git-path", "index").stdout.decode("utf-8", errors="surrogateescape").strip()
        nested_index = Path(raw_nested_index)
        if not nested_index.is_absolute():
            nested_index = target / nested_index
        archive_relative = PurePosixPath(str(saved.get("git_index_archive") or ""))
        if (
            archive_relative.is_absolute()
            or ".." in archive_relative.parts
            or not archive_relative.parts
            or archive_relative.parts[0] != "gitlink-indexes"
        ):
            raise RuntimeError("unsafe gitlink index archive: %s" % archive_relative)
        nested_index.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checkpoint / Path(*archive_relative.parts), nested_index)
        if hashlib.sha256(nested_index.read_bytes()).hexdigest() != saved.get("git_index_sha256"):
            raise RuntimeError("gitlink index copy mismatch: %s" % target)

    expected_tracked = set(manifest.get("tracked_delta", [saved["path_b64"] for saved in manifest["tracked"]]))
    expected_untracked = {saved["path_b64"] for saved in manifest["untracked"]}
    actual_tracked = {
        path_key(os.fsdecode(value))
        for value in git(
            "diff", "--name-only", "--no-renames", "-z", "HEAD", "--"
        ).stdout.split(b"\0")
        if value
    }
    actual_untracked = {
        path_key(os.fsdecode(value))
        for value in git(
            "ls-files", "--others", "--exclude-standard", "-z"
        ).stdout.split(b"\0")
        if value
    }
    mismatches = []
    for saved in [*manifest["tracked"], *manifest["untracked"]]:
        observed = current_entry(saved)
        if comparable(saved) != comparable(observed):
            mismatches.append(saved.get("path") or decode_path(saved["path_b64"]))
    if len(empty_directories) != len(restored_empty_directories):
        raise RuntimeError(
            "repository checkpoint empty-directory manifest length mismatch: "
            "%s != %s"
            % (len(empty_directories), len(restored_empty_directories))
        )
    for saved, observed in zip(empty_directories, restored_empty_directories):
        try:
            info = observed.lstat()
        except FileNotFoundError:
            mismatches.append(saved.get("path") or decode_path(saved["path_b64"]))
            continue
        if (
            not stat.S_ISDIR(info.st_mode)
            or observed.is_symlink()
            or stat.S_IMODE(info.st_mode) != int(saved["mode"])
        ):
            mismatches.append(saved.get("path") or decode_path(saved["path_b64"]))
    if actual_tracked != expected_tracked:
        missing = sorted(expected_tracked - actual_tracked)
        extra = sorted(actual_tracked - expected_tracked)
        mismatches.extend(
            "<tracked-path-set:%s:%s>" % (kind, decode_path(value))
            for kind, values in (("missing", missing), ("extra", extra))
            for value in values[:20]
        )
    if actual_untracked != expected_untracked:
        missing = sorted(expected_untracked - actual_untracked)
        extra = sorted(actual_untracked - expected_untracked)
        mismatches.extend(
            "<untracked-path-set:%s:%s>" % (kind, decode_path(value))
            for kind, values in (("missing", missing), ("extra", extra))
            for value in values[:20]
        )
    if git("rev-parse", "HEAD").stdout.decode("ascii").strip() != head:
        mismatches.append("<git-head>")
    if mismatches:
        raise RuntimeError(
            "repository checkpoint manifest mismatch: "
            + ", ".join(str(path) for path in mismatches[:50])
        )
    result = {
        "ok": True,
        "path_summary": {
            "tracked": len(expected_tracked),
            "untracked": len(expected_untracked),
            "empty_directories": len(empty_directories),
            "verified": (
                len(manifest["tracked"])
                + len(manifest["untracked"])
                + len(empty_directories)
            ),
            "index_copy_verified": bool(metadata.get("index_exists")),
            "sample": [
                saved.get("path") or decode_path(saved["path_b64"])
                for saved in [
                    *manifest["tracked"],
                    *manifest["untracked"],
                    *empty_directories,
                ][:20]
            ],
        },
    }
except Exception as exc:
    result = {"ok": False, "error": str(exc), "error_type": type(exc).__name__, "phase": "checkpoint_restore", "exit_code": getattr(exc, "exit_code", None)}
print(json.dumps(result, ensure_ascii=False))
'''
        result = self._codeflow_run_json_script(
            script,
            {
                "root": self._repository_root(task),
                "checkpoint": checkpoint.path,
                "checkpoint_prefix": _CODEFLOW_CHECKPOINT_PREFIX,
            },
            sandbox,
            task,
            timeout=self.command_timeout,
            user=user,
            operation_kind="idempotent",
        )
        if result.get("ok"):
            logger.debug(
                "repository checkpoint restored and verified (task=%s paths=%s)",
                task.id,
                result.get("path_summary"),
            )
            return None
        return CodeFlowRestoreError(
            message=str(
                result.get("error") or "failed to restore repository checkpoint"
            ),
            kind=str(result.get("_error_kind") or "operation"),
            detail=(
                str(result.get("detail"))
                if result.get("detail") is not None
                else None
            ),
        )

    def _codeflow_discard_repo_checkpoint(
        self,
        checkpoint: CodeFlowRepoCheckpoint,
        sandbox: Sandbox,
        task: Task,
        *,
        user: str | None = None,
    ) -> None:
        script = r'''
import base64
import json
import shutil
import sys

payload = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
path = str(payload["checkpoint"])
if path.startswith(str(payload["checkpoint_prefix"])):
    shutil.rmtree(path, ignore_errors=True)
print(json.dumps({"ok": True}))
'''
        # Discard only retires an already-consumed, unique checkpoint. A
        # cleanup RPC failure must not invalidate an otherwise audited action;
        # sandbox teardown will reclaim the leftover checkpoint.
        try:
            result = self._codeflow_run_json_script(
                script,
                {"checkpoint": checkpoint.path, "checkpoint_prefix": _CODEFLOW_CHECKPOINT_PREFIX},
                sandbox, task, timeout=30, user=user,
            )
            if result.get("error"):
                logger.warning("Checkpoint discard deferred to sandbox teardown: task=%s error=%s", task.id, result["error"])
        except Exception as exc:
            if isinstance(exc, RolloutInfrastructureError) and exc.reason == "sandbox_closed":
                # The unique checkpoint is reclaimed with its sandbox. This
                # is expected cancellation, not a failed training command.
                from rllm.utils.diagnostic_events import emit_diagnostic
                emit_diagnostic("checkpoint_discard_cancelled", task_id=task.id,
                                diagnostics=getattr(exc, "diagnostics", None))
                return
            logger.warning("Checkpoint discard deferred to sandbox teardown: task=%s error=%s", task.id, exc)

    def _codeflow_bash_repository_safety_violations(
        self,
        changed_files: list[Any],
        before: RepoSnapshot,
        after: RepoSnapshot,
        task: Task,
    ) -> list[str]:
        """Return repository safety violations that edit freedom cannot waive."""

        violations: list[str] = []
        changed_paths = {
            path
            for item in changed_files
            for path in (
                getattr(item, "path", None),
                getattr(item, "old_path", None),
            )
            if isinstance(path, str)
        }
        if self._codeflow_protected_changed_paths(sorted(changed_paths), task):
            violations.append("bash_modified_protected_tests")

        if (
            before.current_head
            and after.current_head
            and before.current_head != after.current_head
        ):
            violations.append("repository_head_changed")
        return violations

    def _codeflow_execute_bash(
        self,
        arguments: dict[str, Any],
        sandbox: Sandbox,
        task: Task,
    ) -> CodeFlowExecutionResult:
        before = self._codeflow_capture_repo_snapshot(sandbox, task, None)
        if before.status != RepoStateStatus.OK or not before.fingerprint:
            error = f"repository checkpoint unavailable: {before.error or before.status.value}"
            return CodeFlowExecutionResult(
                observation=_json_observation(
                    {"error": error, "fatal": True},
                    CODEFLOW_SHORT_OUTPUT_LIMIT,
                ),
                status=ExecutionStatus.ERROR,
                error=error,
                policy_violations=["bash_repository_checkpoint_unavailable"],
                instrumentation_errors=[error],
                fatal=True,
            )

        checkpoint, checkpoint_error = self._codeflow_create_repo_checkpoint(sandbox, task)
        if checkpoint is None:
            error = f"repository checkpoint unavailable: {checkpoint_error or 'unknown failure'}"
            return CodeFlowExecutionResult(
                observation=_json_observation(
                    {"error": error, "fatal": True},
                    CODEFLOW_SHORT_OUTPUT_LIMIT,
                ),
                status=ExecutionStatus.ERROR,
                error=error,
                policy_violations=["bash_repository_checkpoint_unavailable"],
                instrumentation_errors=[error],
                fatal=True,
            )

        execution = self._codeflow_execute_bash_unchecked(arguments, sandbox, task)
        attempted_after = self._codeflow_capture_repo_snapshot(sandbox, task, checkpoint.baseline_ref)
        if attempted_after.status != RepoStateStatus.OK or not attempted_after.fingerprint:
            restore_error = self._codeflow_restore_repo_checkpoint(checkpoint, sandbox, task)
            final = self._codeflow_capture_repo_snapshot(sandbox, task, checkpoint.baseline_ref)
            self._codeflow_discard_repo_checkpoint(checkpoint, sandbox, task)
            restore_protocol_fallback = (
                restore_error is not None
                and getattr(restore_error, "kind", "operation")
                in {"protocol", "transport"}
            )
            restored = (
                (restore_error is None or restore_protocol_fallback)
                and final.status == RepoStateStatus.OK
                and final.fingerprint == before.fingerprint
            )
            error = "could not verify repository state after execute_bash"
            if not restored:
                error += f"; rollback failed: {restore_error or final.error or 'fingerprint mismatch'}"
            # A valid pre-action checkpoint followed by two Git-reported
            # missing-repository results identifies damage by the action.
            # Transport/probe failures alone must remain infrastructure faults.
            destroyed_git = not restored and all(
                "not a git repository" in str(snapshot.error or "").lower()
                for snapshot in (attempted_after, final)
            )
            return CodeFlowExecutionResult(
                observation=_json_observation(
                    {
                        "error": error,
                        "fatal": not restored,
                        "repository_restored": restored,
                        "restore_confirmation": (
                            "fingerprint_fallback"
                            if restore_protocol_fallback and restored
                            else "confirmed" if restored else "failed"
                        ),
                    },
                    CODEFLOW_SHORT_OUTPUT_LIMIT,
                ),
                status=ExecutionStatus.ERROR,
                error=error,
                exit_code=execution.exit_code,
                duration=execution.duration,
                policy_violations=[
                    "bash_repository_probe_recovered"
                    if restored
                    else "bash_repository_restore_failed"
                ] + (["bash_destroyed_git_repository"] if destroyed_git else []),
                instrumentation_errors=[error],
                fatal=not restored,
            )

        attempted_changes = changed_files_between(before, attempted_after)
        repository_changed = attempted_after.fingerprint != before.fingerprint
        if not repository_changed:
            self._codeflow_discard_repo_checkpoint(checkpoint, sandbox, task)
            return execution

        safety_violations = self._codeflow_bash_repository_safety_violations(
            attempted_changes,
            before,
            attempted_after,
            task,
        )
        edit_restricted = "edit" in set(
            self.codeflow_tool_restricted_mode
        )
        if not edit_restricted and not safety_violations:
            self._codeflow_discard_repo_checkpoint(checkpoint, sandbox, task)
            return execution

        restore_error = self._codeflow_restore_repo_checkpoint(checkpoint, sandbox, task)
        final = self._codeflow_capture_repo_snapshot(sandbox, task, checkpoint.baseline_ref)
        self._codeflow_discard_repo_checkpoint(checkpoint, sandbox, task)
        restore_protocol_fallback = (
            restore_error is not None
            and getattr(restore_error, "kind", "operation")
            in {"protocol", "transport"}
        )
        restored = (
            (restore_error is None or restore_protocol_fallback)
            and final.status == RepoStateStatus.OK
            and final.fingerprint == before.fingerprint
        )
        attempted_payload = [item.model_dump(mode="json") for item in attempted_changes]
        if not restored:
            error = f"execute_bash modified the repository and rollback failed: {restore_error or final.error or 'fingerprint mismatch'}"
            rollback_violations = list(safety_violations)
            if edit_restricted:
                rollback_violations.insert(
                    0, "bash_repository_mutation_attempt"
                )
            rollback_violations.append("bash_repository_restore_failed")
            payload = {
                "error": error,
                "fatal": True,
                "repository_restored": False,
                "attempted_changed_files": attempted_payload,
                "content": execution.observation,
            }
            observation, truncated = _render_json_content(
                payload, limit=CODEFLOW_BASH_OUTPUT_LIMIT
            )
            return CodeFlowExecutionResult(
                observation=observation,
                status=ExecutionStatus.ERROR,
                error=error,
                exit_code=execution.exit_code,
                duration=execution.duration,
                attempted_changed_files=attempted_changes,
                policy_violations=rollback_violations,
                instrumentation_errors=[error],
                fatal=True,
                observation_stats=_observation_stats(
                    observation,
                    source_characters=len(_compact_json(payload)),
                    output_limit=CODEFLOW_BASH_OUTPUT_LIMIT,
                    compaction=("compact_json", "head_tail") if truncated else ("compact_json",),
                    truncation_reasons=("character_budget",) if truncated else (),
                ),
            )

        if safety_violations:
            if "bash_modified_protected_tests" in safety_violations:
                protected_paths = self._codeflow_protected_changed_paths(
                    sorted(
                        {
                            path
                            for item in attempted_changes
                            for path in (item.path, item.old_path)
                            if isinstance(path, str)
                        }
                    ),
                    task,
                )
                error = self._codeflow_protected_test_error(protected_paths)
                error += " The repository was restored."
                if "repository_head_changed" in safety_violations:
                    error += " Git HEAD/refs cannot be modified."
            else:
                error = (
                    "execute_bash violated repository safety policy; the "
                    "repository was restored. Git HEAD/refs cannot be modified."
                )
            violations = list(safety_violations)
            if edit_restricted:
                violations.insert(0, "bash_repository_mutation_attempt")
        else:
            error = "execute_bash attempted to modify Git-visible repository files; the repository was restored. Use edit_file instead."
            violations = ["bash_repository_mutation_attempt"]
        payload = {
            "error": error,
            "repository_restored": True,
            "attempted_changed_files": attempted_payload,
            "original_exit_code": execution.exit_code,
            "content": execution.observation,
        }
        observation, truncated = _render_json_content(
            payload, limit=CODEFLOW_BASH_OUTPUT_LIMIT
        )
        return CodeFlowExecutionResult(
            observation=observation,
            status=ExecutionStatus.ERROR,
            error=error,
            exit_code=execution.exit_code,
            duration=execution.duration,
            attempted_changed_files=attempted_changes,
            policy_violations=violations,
            observation_stats=_observation_stats(
                observation,
                source_characters=len(_compact_json(payload)),
                output_limit=CODEFLOW_BASH_OUTPUT_LIMIT,
                compaction=("compact_json", "head_tail") if truncated else ("compact_json",),
                truncation_reasons=("character_budget",) if truncated else (),
            ),
        )

    def _codeflow_execute_bash_unchecked(
        self,
        arguments: dict[str, Any],
        sandbox: Sandbox,
        task: Task,
    ) -> CodeFlowExecutionResult:
        script = r'''
import base64
import ctypes
import json
import os
import signal
import stat
import subprocess
import sys
import time

payload = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
started = time.monotonic()
process = None
def children(pid):
    try:
        with open("/proc/{0}/task/{0}/children".format(pid)) as stream:
            return [int(value) for value in stream.read().split()]
    except FileNotFoundError:
        return []

def descendants():
    pending = children(os.getpid())
    result = []
    seen = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            with open("/proc/{}/stat".format(pid)) as stream:
                stat_fields = stream.read().rsplit(") ", 1)[1].split()
        except FileNotFoundError:
            continue
        result.append((pid, stat_fields[19], stat_fields[0]))
        pending.extend(children(pid))
    return result

def stop_children(process):
    # GNU timeout and multiprocessing workers may create their own process
    # groups. A group kill alone neither stops them nor closes inherited pipes.
    # Subreaping keeps orphaned descendants attached to THIS helper only.
    started_cleanup = time.monotonic()
    deadline = started_cleanup + 5.5
    output = (b"", b"")
    drained = False
    while True:
        process.poll()  # Let Popen reap its own direct child.
        remaining = descendants()
        for pid, identity, state in remaining:
            if state == "Z" and pid != process.pid:
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass  # Its parent will reap it, or it is adopted next turn.
                continue
            sig = signal.SIGTERM if time.monotonic() - started_cleanup < 0.5 else signal.SIGKILL
            fd = None
            try:
                if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
                    fd = os.pidfd_open(pid)
                with open("/proc/{}/stat".format(pid)) as stream:
                    current = stream.read().rsplit(") ", 1)[1].split()[19]
                if current != identity:
                    continue
                if fd is not None:
                    signal.pidfd_send_signal(fd, sig)
                else:
                    os.kill(pid, sig)  # Legacy task images (Python 3.6+).
            except ProcessLookupError:
                pass
            except FileNotFoundError:
                pass
            finally:
                if fd is not None:
                    os.close(fd)
        left = deadline - time.monotonic()
        if left <= 0:
            return output, {"cleanup_confirmed": False, "phase": "command_descendant_cleanup",
                            "remaining_processes": descendants()[:32]}
        try:
            if not drained:
                output = process.communicate(timeout=min(0.05, left))
                drained = True  # Python 3.6 cannot communicate on closed pipes again.
            if not descendants():
                return output, {"cleanup_confirmed": True, "phase": "command_descendant_cleanup"}
            time.sleep(min(0.01, max(0, deadline - time.monotonic())))
        except subprocess.TimeoutExpired as exc:
            output = (exc.output or b"", exc.stderr or b"")
# Some compilers unlink their output on error, including when -o /dev/null
# was requested. Repair only a device that was valid before this command;
# do it in this still-running helper before the transport needs to exec again.
null_device = None
try:
    candidate = os.lstat("/dev/null")
    if stat.S_ISCHR(candidate.st_mode) and candidate.st_rdev == os.makedev(1, 3):
        null_device = candidate
except OSError:
    pass
try:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER; helper-local.
        raise OSError(ctypes.get_errno(), "cannot establish command child ownership")
    process = subprocess.Popen(
        ["bash", "-o", "pipefail", "-c", payload["command"]],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=payload["timeout"])
        result = {
            "ok": process.returncode == 0,
            "timed_out": False,
            "exit_code": process.returncode,
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
        }
    except subprocess.TimeoutExpired:
        (stdout, stderr), cleanup = stop_children(process)
        result = {
            "ok": False,
            "timed_out": True,
            "exit_code": None,
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
            "diagnostics": cleanup,
        }
        if not cleanup["cleanup_confirmed"]:
            result.update(error="command descendants did not stop within cleanup deadline",
                          _error_kind="helper_runtime")
except Exception as exc:
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            pass
    result = {"ok": False, "timed_out": False, "exit_code": None, "stdout": "", "stderr": "", "error": str(exc), "_error_kind": "helper_runtime"}
if null_device is not None:
    try:
        try:
            os.lstat("/dev/null")
        except FileNotFoundError:
            os.mknod("/dev/null", null_device.st_mode, null_device.st_rdev)
            os.chmod("/dev/null", stat.S_IMODE(null_device.st_mode))
            os.chown("/dev/null", null_device.st_uid, null_device.st_gid)
            result["runtime_repairs"] = ["restored_missing_dev_null"]
    except Exception as exc:
        result.update(ok=False, error="failed to restore /dev/null: " + str(exc),
                      _error_kind="helper_runtime")
result["duration"] = time.monotonic() - started
print(json.dumps(result, ensure_ascii=False))
'''
        timeout = float(arguments["timeout"])
        cancellation = agent_flow_cancellation.get()
        if cancellation is not None and cancellation.remaining is not None:
            # The command helper kills and reaps the process group before
            # returning. A transport/stop failure still raises infrastructure.
            timeout = min(timeout, max(0.001, cancellation.remaining))
        command = str(arguments["command"])
        rllm_metadata = task.metadata.get("rllm") or {}
        if (
            rllm_metadata.get("agent_environment_policy")
            == "preconfigured_eval_v1"
        ):
            shell_init = str(rllm_metadata.get("agent_shell_init") or "").strip()
            command = f"{shell_init} && ( {command} )"
        elif (task.metadata.get("task_profile") or rllm_metadata.get("task_profile")) == "repo_generation_nl2repo":
            from rllm.sandbox.repo_generation_environment import repo_generation_environment_exports
            command = repo_generation_environment_exports(rllm_metadata.get("repo_generation_proxy_url", "")) + f"( {command} )"
        started = time.perf_counter()
        result = self._codeflow_run_json_script(
            script,
            {"command": command, "timeout": timeout},
            sandbox,
            task,
            timeout=timeout + getattr(sandbox, "command_transport_grace", 10),
        )
        duration = time.perf_counter() - started
        if result.get("runtime_repairs"):
            logger.warning("Codeflow runtime device repaired: task=%s repairs=%s", task.id, result["runtime_repairs"])
        # A child command's nonzero exit is a normal tool result. These tags
        # instead mean the trusted wrapper could not execute/report it.
        if result.get("_error_kind") in {"helper_runtime", "execution", "protocol", "transport"}:
            raise RolloutInfrastructureError(
                "agent_helper_runtime_failed",
                "Bash helper failed: " + str(result.get("detail") or result.get("error", ""))[:1000],
                stage="agentflow", retryable=True, retry_scope="full_rollout",
                diagnostics=result.get("diagnostics"),
            )
        if result.get("error"):
            error = str(result["error"])
            observation, truncated = _render_bash_observation(
                "", f"Error: {error}", exit_code=None, timed_out=False
            )
            return CodeFlowExecutionResult(
                observation=observation,
                status=ExecutionStatus.ERROR,
                error=error,
                duration=duration,
                observation_stats=_observation_stats(
                    observation,
                    source_characters=len(observation),
                    output_limit=CODEFLOW_BASH_OUTPUT_LIMIT,
                    compaction=("head_tail",) if truncated else (),
                    truncation_reasons=("character_budget",) if truncated else (),
                ),
            )

        stdout = str(result.get("stdout") or "")
        stderr = str(result.get("stderr") or "")
        exit_code = result.get("exit_code")
        exit_code = int(exit_code) if isinstance(exit_code, int) and not isinstance(exit_code, bool) else None
        timed_out = bool(result.get("timed_out"))
        if timed_out:
            error = f"command timed out after {timeout:g} seconds"
            if not stderr:
                stderr = f"Error: {error}"
            status = ExecutionStatus.TIMEOUT
        elif exit_code != 0:
            error = f"command exited with code {exit_code}" if exit_code is not None else "command execution failed"
            if not stderr:
                stderr = f"Error: {error}"
            status = ExecutionStatus.ERROR
        else:
            error = None
            status = ExecutionStatus.SUCCESS
        source_observation, _ = _render_bash_observation(
            stdout,
            stderr,
            exit_code=exit_code,
            timed_out=timed_out,
            limit=max(CODEFLOW_BASH_OUTPUT_LIMIT, len(stdout) + len(stderr) + 256),
        )
        observation, truncated = _render_bash_observation(
            stdout,
            stderr,
            exit_code=exit_code,
            timed_out=timed_out,
        )
        return CodeFlowExecutionResult(
            observation=observation,
            status=status,
            error=error,
            exit_code=exit_code,
            duration=duration,
            observation_stats=_observation_stats(
                observation,
                source_characters=len(source_observation),
                output_limit=CODEFLOW_BASH_OUTPUT_LIMIT,
                compaction=("head_tail",) if truncated else (),
                truncation_reasons=("character_budget",) if truncated else (),
            ),
        )

    def _codeflow_capture_repo_snapshot(
        self,
        sandbox: Sandbox,
        task: Task,
        baseline_ref: str | None,
    ) -> RepoSnapshot:
        script = repository_helper_source() + r'''
import base64
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

payload = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
root = Path(payload["root"]).resolve()

class RepoUnavailable(RuntimeError):
    pass

def git(*args, check=True):
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={root}", "-c", "core.filemode=true", "-C", str(root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and completed.returncode != 0:
        message = (completed.stderr or completed.stdout or b"git command failed").decode("utf-8", errors="replace").strip()
        raise RuntimeError(message)
    return completed

def decode_path(value):
    return value.decode("utf-8", errors="replace")

def base_state(baseline, path):
    entry = git("ls-tree", "-z", baseline, "--", path).stdout.rstrip(b"\0")
    if not entry:
        return False, None, None
    prefix = entry.split(b"\t", 1)[0].split()
    mode = prefix[0].decode("ascii")
    object_id = prefix[2].decode("ascii")
    if mode == "160000":
        content = object_id.encode("ascii")
    else:
        shown = git("show", f"{baseline}:{path}", check=False)
        if shown.returncode != 0:
            raise RuntimeError(f"failed to read baseline path: {path}")
        content = shown.stdout
    return True, hashlib.sha256(content).hexdigest(), mode

def current_state(path):
    candidate = root / path
    info = repo_lstat(root, path)
    if info is None:
        return False, None, None
    if stat.S_ISLNK(info.st_mode):
        content = os.readlink(candidate).encode("utf-8", errors="surrogateescape")
        mode = "120000"
    elif stat.S_ISREG(info.st_mode):
        content = candidate.read_bytes()
        mode = "100755" if info.st_mode & stat.S_IXUSR else "100644"
    elif stat.S_ISDIR(info.st_mode):
        tree = repo_directory_fingerprint(root, candidate, include_git=False)
        if tree is None:
            raise RuntimeError("unsupported repository directory: %s" % path)
        nested = subprocess.run(
            ["git", "-C", str(candidate), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        content = (nested.stdout.strip() if nested.returncode == 0 else b"directory") + b"\0" + tree[0].encode("ascii")
        mode = "160000"

    else:
        content = f"special:{info.st_mode}".encode("ascii")
        mode = f"special:{info.st_mode}"
    return True, hashlib.sha256(content).hexdigest(), mode

try:
    if not root.is_dir():
        raise RepoUnavailable(f"repository root does not exist: {root}")
    if shutil.which("git") is None:
        raise RepoUnavailable("git executable is unavailable")
    inside_result = git("rev-parse", "--is-inside-work-tree", check=False)
    inside = inside_result.stdout.strip()
    if inside != b"true":
        detail = (inside_result.stderr or inside_result.stdout).decode("utf-8", errors="replace").strip()
        raise RepoUnavailable(detail or "repository root is not a Git work tree")
    baseline = payload.get("baseline_ref") or git("rev-parse", "HEAD").stdout.decode("ascii").strip()
    git("cat-file", "-e", f"{baseline}^{{commit}}")
    current_head = git("rev-parse", "HEAD").stdout.decode("ascii").strip()

    changed = git("diff", "--name-status", "-z", "--find-renames", baseline, "--").stdout.split(b"\0")
    paths = set()
    index = 0
    while index < len(changed) and changed[index]:
        status_value = changed[index].decode("ascii", errors="replace")
        index += 1
        if status_value[:1] in {"R", "C"}:
            if index + 1 >= len(changed):
                raise RuntimeError("malformed git diff --name-status output")
            paths.add(decode_path(changed[index]))
            paths.add(decode_path(changed[index + 1]))
            index += 2
        else:
            if index >= len(changed):
                raise RuntimeError("malformed git diff --name-status output")
            paths.add(decode_path(changed[index]))
            index += 1

    tracked = git("ls-files", "--cached", "-z").stdout.split(b"\0")
    untracked = git("ls-files", "--others", "--exclude-standard", "-z").stdout.split(b"\0")
    paths.update(decode_path(value) for value in untracked if value)
    repository_files = sorted(
        {
            decode_path(value)
            for value in [*tracked, *untracked]
            if value and repo_lstat(root, decode_path(value)) is not None
        }
    )

    files = []
    for path in sorted(paths):
        base_exists, base_digest, base_mode = base_state(baseline, path)
        current_exists, current_digest, current_mode = current_state(path)
        if (base_exists, base_digest, base_mode) == (current_exists, current_digest, current_mode):
            continue
        files.append(
            {
                "path": path,
                "base_exists": base_exists,
                "base_digest": base_digest,
                "base_mode": base_mode,
                "current_exists": current_exists,
                "current_digest": current_digest,
                "current_mode": current_mode,
            }
        )

    ignored_directories = {
        os.fsdecode(value[:-1]) for value in git(
            "ls-files", "--others", "--ignored", "--exclude-standard", "--directory", "-z"
        ).stdout.split(b"\0") if value.endswith(b"/")
    }
    empty_directories = [
        [name, stat.S_IMODE((root / name).lstat().st_mode)]
        for name in sorted(repo_empty_directories(root, ignored_directories), key=os.fsencode)
    ]
    canonical = json.dumps(
        {"baseline_ref": baseline, "current_head": current_head, "files": files,
         "filesystem": repo_filesystem_metadata(root, {os.fsdecode(value) for value in tracked + untracked if value}),
         "state_version": REPOSITORY_STATE_VERSION, "empty_directories": empty_directories},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    result = {
        "ok": True,
        "baseline_ref": baseline,
        "current_head": current_head,
        "fingerprint": "sha256:" + hashlib.sha256(canonical).hexdigest(),
        "files": files,
        "repository_files": repository_files,
    }
except RepoUnavailable as exc:
    result = {"ok": False, "unavailable": True, "error": str(exc)}
except Exception as exc:
    result = {"ok": False, "error": str(exc)}
print(json.dumps(result, ensure_ascii=False))
'''
        result = self._codeflow_run_json_script(
            script,
            {"root": self._repository_root(task), "baseline_ref": baseline_ref},
            sandbox,
            task,
            operation_kind="read_only",
        )
        if not result.get("ok"):
            status = RepoStateStatus.UNAVAILABLE if result.get("unavailable") else RepoStateStatus.ERROR
            return RepoSnapshot(status=status, baseline_ref=baseline_ref, error=str(result.get("error") or "repository probe failed"))
        try:
            repository_files_payload = result.get("repository_files")
            if not isinstance(repository_files_payload, list):
                raise ValueError("repository_files is missing or not a list")
            files = {
                str(item["path"]): RepoFileState(
                    path=str(item["path"]),
                    base_exists=bool(item["base_exists"]),
                    base_digest=item.get("base_digest"),
                    base_mode=item.get("base_mode"),
                    current_exists=bool(item["current_exists"]),
                    current_digest=item.get("current_digest"),
                    current_mode=item.get("current_mode"),
                )
                for item in result.get("files", [])
                if isinstance(item, dict) and isinstance(item.get("path"), str)
            }
            return RepoSnapshot(
                status=RepoStateStatus.OK,
                fingerprint=str(result["fingerprint"]),
                baseline_ref=str(result["baseline_ref"]),
                current_head=str(result["current_head"]),
                files=files,
                repository_files=tuple(
                    str(path)
                    for path in repository_files_payload
                    if isinstance(path, str)
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            return RepoSnapshot(status=RepoStateStatus.ERROR, baseline_ref=baseline_ref, error=f"malformed repository probe output: {exc}")

    def _codeflow_process_action(
        self,
        action: ParsedAction,
        sandbox: Sandbox,
        task: Task,
        history: list[CodeFlowUndoEntry],
        *,
        session_uid: str,
        turn: int,
        baseline_ref: str | None,
        bash_command_counts: dict[str, int] | None = None,
        observation_cache: dict[tuple[str, str, str, str], int] | None = None,
    ) -> tuple[CodeFlowExecutionResult, ActionEvent, str | None]:
        preconfigured_policy = _codeflow_eval_guard_enabled(task)
        bash_occurrence = 0
        if (
            preconfigured_policy
            and action.name == "execute_bash"
            and action.status == "ok"
            and bash_command_counts is not None
        ):
            raw_command = action.arguments.get("command")
            if isinstance(raw_command, str) and raw_command.strip():
                command_key = canonicalize_shell_command(raw_command)
                bash_occurrence = bash_command_counts.get(command_key, 0) + 1
                bash_command_counts[command_key] = bash_occurrence

        # Count before validation so policy-rejected commands cannot avoid the
        # repeat budget.  On the third occurrence retain the original policy
        # diagnosis as provenance and add the repeat violation.
        validation = self._codeflow_validate_action(action, task, history)
        if bash_occurrence > 2:
            violations = list(validation.policy_violations)
            if "bash_repeated_command_attempt" not in violations:
                violations.append("bash_repeated_command_attempt")
            repeated_error = (
                "execute_bash cannot run the same command more than twice "
                "without an intervening successful repository edit"
            )
            if validation.error:
                repeated_error = f"{validation.error}; {repeated_error}"
            validation = CodeFlowValidationResult(
                status=ValidationStatus.ERROR,
                normalized_arguments=dict(validation.normalized_arguments),
                error=repeated_error,
                subtype=(
                    validation.subtype
                    if validation.status == ValidationStatus.ERROR
                    else "repeated_bash_command"
                ),
                policy_violations=violations,
                shell_analysis=validation.shell_analysis,
            )
        before = self._codeflow_capture_repo_snapshot(sandbox, task, baseline_ref)
        resolved_baseline = baseline_ref or before.baseline_ref

        if action.status != "ok":
            execution = CodeFlowExecutionResult(
                observation=self._format_native_tool_observation(f"Parse error: {action.reason}", action),
                status=ExecutionStatus.NOT_RUN,
            )
            after = before
        elif validation.status == ValidationStatus.ERROR:
            execution = CodeFlowExecutionResult(
                observation=_json_observation({"error": validation.error}, CODEFLOW_SHORT_OUTPUT_LIMIT),
                status=ExecutionStatus.NOT_RUN,
            )
            after = before
        else:
            execution = self._codeflow_execute_validated(
                action.name,
                validation.normalized_arguments,
                sandbox,
                task,
                history,
            )
            if action.name == "submit":
                after = before
            else:
                after = self._codeflow_capture_repo_snapshot(sandbox, task, resolved_baseline)
                resolved_baseline = resolved_baseline or after.baseline_ref

        if before.status == RepoStateStatus.OK and after.status == RepoStateStatus.OK:
            repo_state_status = RepoStateStatus.OK
            repo_changed: bool | None = before.fingerprint != after.fingerprint
            changed_files = changed_files_between(before, after)
        else:
            repo_state_status = (
                RepoStateStatus.ERROR
                if RepoStateStatus.ERROR in {before.status, after.status}
                else RepoStateStatus.UNAVAILABLE
            )
            repo_changed = None
            changed_files = []

        if (
            preconfigured_policy
            and action.name in {"edit_file", "execute_bash"}
            and validation.status == ValidationStatus.OK
            and execution.status == ExecutionStatus.SUCCESS
            and repo_changed is True
            and bash_command_counts is not None
        ):
            bash_command_counts.clear()

        instrumentation_errors: list[str] = []
        snapshots = (("before", before),) if after is before else (("before", before), ("after", after))
        for label, snapshot in snapshots:
            if snapshot.status != RepoStateStatus.OK:
                diagnostic = f"repo_state_{label}: {snapshot.status.value}: {snapshot.error or 'unknown probe failure'}"
                if diagnostic not in instrumentation_errors:
                    instrumentation_errors.append(diagnostic)
        for diagnostic in execution.instrumentation_errors:
            if diagnostic not in instrumentation_errors:
                instrumentation_errors.append(diagnostic)
        if (
            action.name == "execute_bash"
            and (
                before.repository_files is None
                or after.repository_files is None
            )
        ):
            instrumentation_errors.append(
                "repository_inventory_unavailable: Bash file attribution disabled"
            )

        if (
            observation_cache is not None
            and action.name in {"search", "read_file"}
            and validation.status == ValidationStatus.OK
            and execution.status == ExecutionStatus.SUCCESS
            and before.status == RepoStateStatus.OK
            and before.fingerprint
        ):
            full_observation = execution.observation
            cache_key = (
                action.name,
                _compact_json(validation.normalized_arguments),
                before.fingerprint,
                hashlib.sha256(full_observation.encode("utf-8")).hexdigest(),
            )
            reference_turn = observation_cache.get(cache_key)
            if reference_turn is None:
                observation_cache[cache_key] = turn
            else:
                reference = _compact_json(
                    {
                        "unchanged": True,
                        "same_as_turn": reference_turn,
                        "message": "Result is identical to the earlier tool response.",
                    }
                )
                previous_stats = dict(execution.observation_stats)
                compaction = list(previous_stats.get("compaction") or ())
                if "repeat_reference" not in compaction:
                    compaction.append("repeat_reference")
                execution.observation = reference
                execution.observation_stats = _observation_stats(
                    reference,
                    source_characters=max(
                        len(full_observation),
                        int(previous_stats.get("source_characters") or 0),
                    ),
                    output_limit=int(previous_stats.get("output_limit") or CODEFLOW_READ_OUTPUT_LIMIT),
                    compaction=compaction,
                    truncation_reasons=previous_stats.get("truncation_reasons") or (),
                    returned_matches=previous_stats.get("returned_matches"),
                    next_offset=previous_stats.get("next_offset"),
                    reference_turn=reference_turn,
                )

        if not execution.observation_stats:
            if (
                validation.status == ValidationStatus.ERROR
                or action.status != "ok"
                or (action.name != "execute_bash" and execution.status != ExecutionStatus.SUCCESS)
            ):
                output_limit = CODEFLOW_SHORT_OUTPUT_LIMIT
            elif action.name == "execute_bash":
                output_limit = CODEFLOW_BASH_OUTPUT_LIMIT
            elif action.name == "read_file":
                output_limit = CODEFLOW_READ_OUTPUT_LIMIT
            elif action.name == "search":
                output_limit = CODEFLOW_SEARCH_OUTPUT_LIMIT
            else:
                output_limit = CODEFLOW_SHORT_OUTPUT_LIMIT
            execution.observation_stats = _observation_stats(
                execution.observation,
                source_characters=len(execution.observation),
                output_limit=output_limit,
                compaction=("compact_json",)
                if execution.observation.lstrip().startswith("{")
                else (),
            )

        category = action_category_for(
            action.name,
            parsed=action.status == "ok" and validation.status != ValidationStatus.ERROR,
        )
        exposure = extract_exposure(
            action.name,
            validation.subtype,
            execution.observation,
            execution.status,
        )
        shell_behaviors = list(
            validation.shell_analysis.behaviors
            if validation.shell_analysis is not None
            else ()
        )
        shell_behavior_evidence = list(
            validation.shell_analysis.evidence
            if validation.shell_analysis is not None
            else ()
        )
        if (
            action.name == "execute_bash"
            and (repo_changed is True or execution.attempted_changed_files)
            and "edit" not in shell_behaviors
        ):
            shell_behaviors.append("edit")
            shell_behaviors.sort(key=CODEFLOW_TOOL_BEHAVIOR_ORDER.index)
            shell_behavior_evidence.append("edit:repository_snapshot")
        if (
            action.name == "execute_bash"
            and validation.shell_analysis is not None
            and execution.status == ExecutionStatus.SUCCESS
            and before.repository_files is not None
            and after.repository_files is not None
        ):
            known_files = sorted(
                set(before.repository_files) | set(after.repository_files)
            )
            for attribution in shell_file_attributions(
                validation.shell_analysis,
                execution.observation,
                self._repository_root(task),
                known_files,
            ):
                file_exposure = FileExposure(
                    path=attribution.path,
                    behavior=attribution.behavior,
                    exposure_kind=attribution.exposure_kind,
                    source=attribution.source,
                    confidence=attribution.confidence,
                )
                if file_exposure not in exposure.file_exposures:
                    exposure.file_exposures.append(file_exposure)
                if (
                    attribution.confidence == "high"
                    and attribution.path not in exposure.files
                ):
                    exposure.files.append(attribution.path)
        policy_changed_files = changed_files
        policy_repo_changed = repo_changed
        if action.name == "execute_bash" and execution.attempted_changed_files:
            policy_changed_files = execution.attempted_changed_files
            policy_repo_changed = True
        policy_violations = self._codeflow_policy_violations(
            action.name,
            policy_changed_files,
            policy_repo_changed,
            before,
            after,
            task,
        )
        for violation in validation.policy_violations:
            if violation not in policy_violations:
                policy_violations.append(violation)
        for violation in execution.policy_violations:
            if violation not in policy_violations:
                policy_violations.append(violation)
        if (
            action.name == "execute_bash"
            and validation.status == ValidationStatus.ERROR
            and str(validation.error or "").startswith("execute_bash cannot modify repository files")
            and "bash_repository_mutation_attempt" not in policy_violations
        ):
            policy_violations.append("bash_repository_mutation_attempt")
        event = ActionEvent(
            action_id=f"{session_uid}:step-{turn}",
            turn_id=turn,
            tool_name=action.name,
            action_category=category,
            action_subtype=validation.subtype,
            normalized_arguments=validation.normalized_arguments,
            shell_behaviors=shell_behaviors,
            shell_behavior_evidence=shell_behavior_evidence,
            parse_status=ParseStatus.OK if action.status == "ok" else ParseStatus.ERROR,
            parse_error=action.reason or None,
            validation_status=validation.status,
            validation_error=validation.error,
            execution_status=execution.status,
            execution_error=execution.error,
            exit_code=execution.exit_code,
            duration=execution.duration,
            repo_state_status=repo_state_status,
            repo_state_before=before.fingerprint if repo_state_status == RepoStateStatus.OK else None,
            repo_state_after=after.fingerprint if repo_state_status == RepoStateStatus.OK else None,
            repo_changed=repo_changed,
            changed_files=changed_files,
            attempted_changed_files=execution.attempted_changed_files,
            exposed_files=exposure.files,
            exposed_line_ranges=exposure.line_ranges,
            file_exposures=exposure.file_exposures,
            visible_characters=exposure.visible_characters,
            output_truncated=exposure.output_truncated,
            test_event=None,
            policy_violations=policy_violations,
            instrumentation_errors=instrumentation_errors,
        )
        return execution, event, resolved_baseline

    def _codeflow_policy_violations(
        self,
        tool_name: str,
        changed_files: list[Any],
        repo_changed: bool | None,
        before: RepoSnapshot,
        after: RepoSnapshot,
        task: Task,
    ) -> list[str]:
        violations: list[str] = []
        if repo_changed:
            if tool_name == "search":
                violations.append("search_tool_modified_repository")
            elif tool_name == "read_file":
                violations.append("view_tool_modified_repository")

            if tool_name == "execute_bash":
                violations.extend(
                    self._codeflow_bash_repository_safety_violations(
                        changed_files,
                        before,
                        after,
                        task,
                    )
                )
            else:
                safety = self._codeflow_bash_repository_safety_violations(
                    changed_files,
                    before,
                    after,
                    task,
                )
                if "bash_modified_protected_tests" in safety:
                    violations.append("edit_modified_protected_tests")
        if tool_name != "execute_bash" and (
            before.current_head
            and after.current_head
            and before.current_head != after.current_head
        ):
            violations.append("repository_head_changed")
        return violations

    def _codeflow_search_execution(
        self,
        args: dict[str, Any],
        sandbox: Sandbox,
        task: Task,
    ) -> CodeFlowExecutionResult:
        started = time.perf_counter()
        mode = str(args.get("mode") or "").strip()
        query = str(args.get("query") or "")
        if mode not in {"text", "symbol", "file", "directory"}:
            observation = _json_observation(
                {"error": "mode must be one of: text, symbol, file, directory"},
                CODEFLOW_SHORT_OUTPUT_LIMIT,
            )
            return CodeFlowExecutionResult(observation=observation, status=ExecutionStatus.ERROR, error="invalid search mode", duration=time.perf_counter() - started)
        if not query:
            observation = _json_observation({"error": "query is required"}, CODEFLOW_SHORT_OUTPUT_LIMIT)
            return CodeFlowExecutionResult(observation=observation, status=ExecutionStatus.ERROR, error="query is required", duration=time.perf_counter() - started)
        try:
            path, _ = self._codeflow_path(args.get("path") or ".", task)
            context_lines = int(args.get("context_lines", 2))
            max_results = int(args.get("max_results", CODEFLOW_DEFAULT_SEARCH_RESULTS))
            offset = int(args.get("offset", 0))
        except (TypeError, ValueError) as exc:
            observation = _json_observation({"error": str(exc)}, CODEFLOW_SHORT_OUTPUT_LIMIT)
            return CodeFlowExecutionResult(
                observation=observation,
                status=ExecutionStatus.ERROR,
                error=str(exc),
                duration=time.perf_counter() - started,
            )
        if not 0 <= context_lines <= 20:
            error = "context_lines must be between 0 and 20"
            return CodeFlowExecutionResult(
                observation=_json_observation({"error": error}, CODEFLOW_SHORT_OUTPUT_LIMIT),
                status=ExecutionStatus.ERROR,
                error=error,
                duration=time.perf_counter() - started,
            )
        if not 1 <= max_results <= 200:
            error = "max_results must be between 1 and 200"
            return CodeFlowExecutionResult(
                observation=_json_observation({"error": error}, CODEFLOW_SHORT_OUTPUT_LIMIT),
                status=ExecutionStatus.ERROR,
                error=error,
                duration=time.perf_counter() - started,
            )
        if offset < 0:
            error = "offset must be greater than or equal to 0"
            return CodeFlowExecutionResult(
                observation=_json_observation({"error": error}, CODEFLOW_SHORT_OUTPUT_LIMIT),
                status=ExecutionStatus.ERROR,
                error=error,
                duration=time.perf_counter() - started,
            )
        include_globs = args.get("include_globs") or []
        if not isinstance(include_globs, list) or any(not isinstance(item, str) or not item for item in include_globs):
            error = "include_globs must be an array of non-empty strings"
            return CodeFlowExecutionResult(
                observation=_json_observation({"error": error}, CODEFLOW_SHORT_OUTPUT_LIMIT),
                status=ExecutionStatus.ERROR,
                error=error,
                duration=time.perf_counter() - started,
            )

        script = r'''
import base64
import fnmatch
import json
import os
import re
import sys
from pathlib import Path

payload = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
result = {"matches": [], "has_more": False, "long_line_clipped": False, "source_characters": 0}
try:
    root = Path(payload["root"]).resolve()
    requested_target = Path(payload["path"])
    if requested_target.is_symlink():
        raise ValueError("symbolic links are not supported by search")
    target = requested_target.resolve()
    target.relative_to(root)
    if not target.exists():
        raise ValueError(f"path not found: {payload['path']}")
    mode = payload["mode"]
    query = payload["query"]
    patterns = payload["include_globs"]
    context = payload["context_lines"]
    maximum = payload["max_results"]
    offset = payload["offset"]
    line_limit = payload["line_limit"]

    def relative(path):
        return path.relative_to(root).as_posix()

    def included(path):
        rel = relative(path)
        return not patterns or any(fnmatch.fnmatch(rel, pattern) for pattern in patterns)

    def files():
        if target.is_file():
            if included(target):
                yield target
            return
        if not target.is_dir():
            return
        for current, directories, names in os.walk(target):
            directories[:] = sorted(name for name in directories if name != ".git" and not (Path(current) / name).is_symlink())
            for name in sorted(names):
                candidate = Path(current) / name
                if not candidate.is_symlink() and included(candidate):
                    yield candidate

    def clip_line(line, match_start=None):
        if len(line) <= line_limit:
            return line, False
        if match_start is None:
            room = max(0, line_limit - 64)
            while True:
                left = (room + 1) // 2
                right = room - left
                omitted = len(line) - left - right
                marker = f"...[{omitted} chars omitted]..."
                rendered = line[:left] + marker + (line[-right:] if right else "")
                if len(rendered) <= line_limit:
                    return rendered, True
                room = max(0, room - (len(rendered) - line_limit))
        room = max(0, line_limit - 64)
        while True:
            start = max(0, min(len(line) - room, match_start - room // 2))
            end = min(len(line), start + room)
            prefix = f"...[{start} chars omitted]..." if start else ""
            suffix_count = len(line) - end
            suffix = f"...[{suffix_count} chars omitted]" if suffix_count else ""
            rendered = prefix + line[start:end] + suffix
            if len(rendered) <= line_limit:
                return rendered, True
            room = max(0, room - (len(rendered) - line_limit))

    skipped = 0
    if mode in {"file", "directory"}:
        if mode == "file":
            candidates = files()
        else:
            candidates = []
            if target.is_dir():
                candidates.append(target)
                for current, directories, _ in os.walk(target):
                    directories[:] = sorted(name for name in directories if name != ".git" and not (Path(current) / name).is_symlink())
                    candidates.extend(Path(current) / name for name in directories)
        for candidate in candidates:
            if query in candidate.name and (mode == "directory" or included(candidate)):
                if skipped < offset:
                    skipped += 1
                    continue
                if len(result["matches"]) >= maximum:
                    result["has_more"] = True
                    break
                result["matches"].append({"path": relative(candidate)})
    else:
        matcher = None
        if mode == "symbol":
            matcher = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(query) + r"(?![A-Za-z0-9_])")
        stop = False
        for candidate in files():
            try:
                data = candidate.read_bytes()
                if b"\0" in data:
                    continue
                text = data.decode("utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            lines = text.splitlines()
            for index, line in enumerate(lines):
                matched = query in line if matcher is None else matcher.search(line) is not None
                if not matched:
                    continue
                if skipped < offset:
                    skipped += 1
                    continue
                if len(result["matches"]) >= maximum:
                    result["has_more"] = True
                    stop = True
                    break
                rel = relative(candidate)
                line_number = index + 1
                start = max(1, line_number - context)
                end = min(len(lines), line_number + context)
                rendered_lines = []
                for number in range(start, end + 1):
                    raw_line = lines[number - 1]
                    match_start = None
                    if number == line_number:
                        found = matcher.search(raw_line) if matcher is not None else None
                        match_start = found.start() if found is not None else raw_line.find(query)
                    clipped_line, clipped = clip_line(raw_line, match_start)
                    result["long_line_clipped"] = result["long_line_clipped"] or clipped
                    result["source_characters"] += len(raw_line)
                    rendered_lines.append({"number": number, "text": clipped_line})
                result["matches"].append({
                    "path": rel,
                    "line_number": line_number,
                    "start_line": start,
                    "end_line": end,
                    "lines": rendered_lines,
                })
            if stop:
                break
except Exception as exc:
    result = {"matches": [], "has_more": False, "long_line_clipped": False, "source_characters": 0, "error": str(exc)}
print(json.dumps(result, ensure_ascii=False))
'''
        result = self._codeflow_run_json_script(
            script,
            {
                "root": self._repository_root(task),
                "path": path,
                "mode": mode,
                "query": query,
                "include_globs": include_globs,
                "context_lines": context_lines,
                "max_results": max_results,
                "offset": offset,
                "line_limit": CODEFLOW_MAX_RENDERED_LINE_CHARS,
            },
            sandbox,
            task,
        )
        if result.get("error"):
            error = str(result["error"])
            observation = _json_observation({"error": error}, CODEFLOW_SHORT_OUTPUT_LIMIT)
            return CodeFlowExecutionResult(observation=observation, status=ExecutionStatus.ERROR, error=error, duration=time.perf_counter() - started)

        page_matches = result.get("matches") if isinstance(result.get("matches"), list) else []
        records = [record for record in page_matches if isinstance(record, dict)]
        source_characters = max(len(_compact_json({"matches": records})), int(result.get("source_characters") or 0))

        def build_payload(selected: list[dict[str, Any]]) -> dict[str, Any]:
            matched_files: list[str] = []
            snippets: list[dict[str, Any]] = []
            if mode in {"file", "directory"}:
                matched_files.extend(str(record.get("path") or "") for record in selected if record.get("path"))
            else:
                merged: list[dict[str, Any]] = []
                for record in selected:
                    record_path = str(record.get("path") or "")
                    record_lines = {
                        int(item["number"]): str(item.get("text") or "")
                        for item in record.get("lines", [])
                        if isinstance(item, dict) and isinstance(item.get("number"), int)
                    }
                    if record_path and record_path not in matched_files:
                        matched_files.append(record_path)
                    if (
                        merged
                        and merged[-1]["path"] == record_path
                        and int(record.get("start_line") or 0) <= merged[-1]["end_line"] + 1
                    ):
                        merged[-1]["end_line"] = max(merged[-1]["end_line"], int(record.get("end_line") or 0))
                        merged[-1]["matched_lines"].add(int(record.get("line_number") or 0))
                        merged[-1]["lines"].update(record_lines)
                    else:
                        merged.append(
                            {
                                "path": record_path,
                                "start_line": int(record.get("start_line") or 0),
                                "end_line": int(record.get("end_line") or 0),
                                "matched_lines": {int(record.get("line_number") or 0)},
                                "lines": record_lines,
                            }
                        )
                for snippet in merged:
                    line_numbers = sorted(snippet["lines"])
                    if not line_numbers:
                        continue
                    snippets.append(
                        {
                            "path": snippet["path"],
                            "start_line": line_numbers[0],
                            "end_line": line_numbers[-1],
                            "matched_lines": sorted(number for number in snippet["matched_lines"] if number > 0),
                            "content": "\n".join(f"{number}:{snippet['lines'][number]}" for number in line_numbers),
                        }
                    )
            more = bool(result.get("has_more")) or len(selected) < len(records)
            return {
                "matched_files": matched_files,
                "displayed_snippets": snippets,
                "returned_matches": len(selected),
                "next_offset": offset + len(selected) if more else None,
                "truncated": more or bool(result.get("long_line_clipped")),
            }

        selected = records
        payload = build_payload(selected)
        observation = _compact_json(payload)
        budget_trimmed = False
        while len(observation) > CODEFLOW_SEARCH_OUTPUT_LIMIT and len(selected) > 1:
            selected = selected[:-1]
            budget_trimmed = True
            payload = build_payload(selected)
            payload["truncated"] = True
            observation = _compact_json(payload)
        if len(observation) > CODEFLOW_SEARCH_OUTPUT_LIMIT and selected and mode in {"text", "symbol"}:
            selected = copy.deepcopy(selected)
            lines = selected[0].get("lines") if isinstance(selected[0].get("lines"), list) else []
            matched_line = int(selected[0].get("line_number") or 0)
            while len(observation) > CODEFLOW_SEARCH_OUTPUT_LIMIT and len(lines) > 1:
                first_distance = abs(int(lines[0].get("number") or 0) - matched_line)
                last_distance = abs(int(lines[-1].get("number") or 0) - matched_line)
                lines.pop(0 if first_distance >= last_distance else -1)
                selected[0]["start_line"] = int(lines[0].get("number") or matched_line)
                selected[0]["end_line"] = int(lines[-1].get("number") or matched_line)
                budget_trimmed = True
                payload = build_payload(selected)
                payload["truncated"] = True
                observation = _compact_json(payload)
            while len(observation) > CODEFLOW_SEARCH_OUTPUT_LIMIT and lines:
                item = lines[0]
                text = str(item.get("text") or "")
                if not text:
                    break
                overflow = len(observation) - CODEFLOW_SEARCH_OUTPUT_LIMIT
                target_limit = max(0, len(text) - overflow - 64)
                match_start = text.find(query)
                clipped, _ = _clip_rendered_line(
                    text,
                    limit=target_limit,
                    match_start=(match_start if match_start >= 0 else len(text) // 2),
                )
                if clipped == text:
                    clipped = text[: max(0, len(text) - overflow - 1)]
                item["text"] = clipped
                budget_trimmed = True
                payload = build_payload(selected)
                payload["truncated"] = True
                observation = _compact_json(payload)
        if len(observation) > CODEFLOW_SEARCH_OUTPUT_LIMIT:
            observation = _json_observation({"error": "structured observation exceeded output limit", "truncated": True}, CODEFLOW_SEARCH_OUTPUT_LIMIT)
            selected = []
            payload = json.loads(observation)
            budget_trimmed = True

        reasons: list[str] = []
        if bool(result.get("has_more")) or len(selected) < len(records):
            reasons.append("result_page")
        if budget_trimmed:
            reasons.append("character_budget")
        if bool(result.get("long_line_clipped")):
            reasons.append("long_line")
        stats = _observation_stats(
            observation,
            source_characters=source_characters,
            output_limit=CODEFLOW_SEARCH_OUTPUT_LIMIT,
            compaction=(
                ("compact_json", "merge_overlapping_snippets")
                if mode in {"text", "symbol"}
                else ("compact_json",)
            ),
            truncation_reasons=reasons,
            returned_matches=int(payload.get("returned_matches") or 0),
            next_offset=payload.get("next_offset") if isinstance(payload.get("next_offset"), int) else None,
        )
        return CodeFlowExecutionResult(
            observation=observation,
            status=ExecutionStatus.SUCCESS,
            duration=time.perf_counter() - started,
            observation_stats=stats,
        )

    def _codeflow_search(self, args: dict[str, Any], sandbox: Sandbox, task: Task) -> str:
        return self._codeflow_search_execution(args, sandbox, task).observation

    def _codeflow_read_file_execution(
        self,
        args: dict[str, Any],
        sandbox: Sandbox,
        task: Task,
    ) -> CodeFlowExecutionResult:
        started = time.perf_counter()
        mode = str(args.get("mode") or "").strip()
        if mode not in {"file", "diff", "status"}:
            error = "mode must be one of: file, diff, status"
            return CodeFlowExecutionResult(
                observation=_json_observation({"error": error}, CODEFLOW_SHORT_OUTPUT_LIMIT),
                status=ExecutionStatus.ERROR,
                error=error,
                duration=time.perf_counter() - started,
            )

        raw_path = args.get("path")
        if mode == "file" and not str(raw_path or "").strip():
            error = "path is required for file mode"
            return CodeFlowExecutionResult(
                observation=_json_observation({"error": error}, CODEFLOW_SHORT_OUTPUT_LIMIT),
                status=ExecutionStatus.ERROR,
                error=error,
                duration=time.perf_counter() - started,
            )
        try:
            path, relative = self._codeflow_path(raw_path, task, allow_empty=mode != "file")
        except ValueError as exc:
            return CodeFlowExecutionResult(
                observation=_json_observation({"error": str(exc)}, CODEFLOW_SHORT_OUTPUT_LIMIT),
                status=ExecutionStatus.ERROR,
                error=str(exc),
                duration=time.perf_counter() - started,
            )

        if mode == "file":
            try:
                start_line = int(args.get("start_line", 1))
                end_line = int(args.get("end_line", start_line + 199))
            except (TypeError, ValueError):
                error = "start_line and end_line must be integers"
                return CodeFlowExecutionResult(
                    observation=_json_observation({"error": error}, CODEFLOW_SHORT_OUTPUT_LIMIT),
                    status=ExecutionStatus.ERROR,
                    error=error,
                    duration=time.perf_counter() - started,
                )
            if start_line < 1 or end_line < start_line:
                error = "line range must satisfy 1 <= start_line <= end_line"
                return CodeFlowExecutionResult(
                    observation=_json_observation({"error": error}, CODEFLOW_SHORT_OUTPUT_LIMIT),
                    status=ExecutionStatus.ERROR,
                    error=error,
                    duration=time.perf_counter() - started,
                )
            script = r'''
import base64
import json
import sys
from pathlib import Path

payload = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
try:
    root = Path(payload["root"]).resolve()
    path = Path(payload["path"])
    resolved = path.resolve()
    resolved.relative_to(root)
    if path.is_symlink():
        raise ValueError("symbolic links are not supported by read_file")
    if not resolved.is_file():
        raise ValueError(f"file not found: {payload['relative']}")
    text = resolved.read_text(encoding="utf-8")
    lines = text.splitlines()
    start = payload["start_line"]
    requested_end = payload["end_line"]
    end = min(requested_end, len(lines))
    line_limit = payload["line_limit"]

    def clip_line(line):
        if len(line) <= line_limit:
            return line, False
        room = max(0, line_limit - 64)
        while True:
            left = (room + 1) // 2
            right = room - left
            omitted = len(line) - left - right
            marker = f"...[{omitted} chars omitted]..."
            rendered = line[:left] + marker + (line[-right:] if right else "")
            if len(rendered) <= line_limit:
                return rendered, True
            room = max(0, room - (len(rendered) - line_limit))

    selected = []
    source_characters = 0
    long_line_clipped = False
    if start <= len(lines):
        for number in range(start, end + 1):
            raw_line = lines[number - 1]
            rendered, clipped = clip_line(raw_line)
            source_characters += len(raw_line)
            long_line_clipped = long_line_clipped or clipped
            selected.append({"number": number, "text": rendered})
    result = {
        "mode": "file",
        "path": payload["relative"],
        "start_line": start,
        "end_line": end,
        "lines": selected,
        "total_lines": len(lines),
        "source_characters": source_characters,
        "long_line_clipped": long_line_clipped,
    }
except Exception as exc:
    result = {"mode": "file", "path": payload["relative"], "error": str(exc)}
print(json.dumps(result, ensure_ascii=False))
'''
            result = self._codeflow_run_json_script(
                script,
                {
                    "root": self._repository_root(task),
                    "path": path,
                    "relative": relative,
                    "start_line": start_line,
                    "end_line": end_line,
                    "line_limit": CODEFLOW_MAX_RENDERED_LINE_CHARS,
                },
                sandbox,
                task,
            )
            if result.get("error"):
                error = str(result["error"])
                return CodeFlowExecutionResult(
                    observation=_json_observation(
                        {"mode": "file", "path": relative, "error": error, "truncated": False},
                        CODEFLOW_SHORT_OUTPUT_LIMIT,
                    ),
                    status=ExecutionStatus.ERROR,
                    error=error,
                    duration=time.perf_counter() - started,
                )
            raw_lines = result.get("lines") if isinstance(result.get("lines"), list) else []
            rendered_lines: list[tuple[int, str]] = []
            long_line_clipped = bool(result.get("long_line_clipped"))
            source_characters = max(
                len(_compact_json(result)),
                int(result.get("source_characters") or 0),
            )
            total_lines = int(result.get("total_lines") or 0)
            for item in raw_lines:
                if not isinstance(item, dict) or not isinstance(item.get("number"), int):
                    continue
                rendered, clipped = _clip_rendered_line(str(item.get("text") or ""))
                long_line_clipped = long_line_clipped or clipped
                candidate_lines = [*rendered_lines, (int(item["number"]), rendered)]
                actual_end = candidate_lines[-1][0]
                more = actual_end < min(end_line, total_lines) or actual_end < total_lines
                candidate = {
                    "mode": "file",
                    "path": relative,
                    "start_line": start_line,
                    "end_line": actual_end,
                    "next_start_line": actual_end + 1 if more else None,
                    "content": "\n".join(f"{number}:{text}" for number, text in candidate_lines),
                    "truncated": more or long_line_clipped,
                }
                while (
                    len(_compact_json(candidate)) > CODEFLOW_READ_OUTPUT_LIMIT
                    and not rendered_lines
                    and rendered
                ):
                    overflow = len(_compact_json(candidate)) - CODEFLOW_READ_OUTPUT_LIMIT
                    target_limit = max(0, len(rendered) - overflow - 64)
                    clipped, _ = _clip_rendered_line(rendered, limit=target_limit)
                    if clipped == rendered:
                        clipped = rendered[: max(0, len(rendered) - overflow - 1)]
                    rendered = clipped
                    candidate_lines = [(int(item["number"]), rendered)]
                    candidate["content"] = f"{int(item['number'])}:{rendered}"
                    candidate["truncated"] = True
                    long_line_clipped = True
                if len(_compact_json(candidate)) > CODEFLOW_READ_OUTPUT_LIMIT:
                    break
                rendered_lines = candidate_lines
            actual_end = rendered_lines[-1][0] if rendered_lines else min(start_line - 1, total_lines)
            more = actual_end < min(end_line, total_lines) or actual_end < total_lines
            payload = {
                "mode": "file",
                "path": relative,
                "start_line": start_line,
                "end_line": actual_end,
                "next_start_line": actual_end + 1 if more else None,
                "content": "\n".join(f"{number}:{text}" for number, text in rendered_lines),
                "truncated": more or long_line_clipped,
            }
            observation = _compact_json(payload)
            reasons: list[str] = []
            if actual_end < min(end_line, total_lines):
                reasons.append("character_budget")
            elif actual_end < total_lines:
                reasons.append("result_page")
            if long_line_clipped:
                reasons.append("long_line")
            return CodeFlowExecutionResult(
                observation=observation,
                status=ExecutionStatus.SUCCESS,
                duration=time.perf_counter() - started,
                observation_stats=_observation_stats(
                    observation,
                    source_characters=source_characters,
                    output_limit=CODEFLOW_READ_OUTPUT_LIMIT,
                    compaction=("compact_json", "compact_line_numbers"),
                    truncation_reasons=reasons,
                ),
            )

        pathspec = "" if relative == "." else f" -- {shlex.quote(relative)}"
        root = shlex.quote(self._repository_root(task))
        if mode == "diff":
            command = f"git -C {root} --no-pager diff --no-ext-diff HEAD{pathspec}"
        else:
            command = f"git -C {root} status --short --untracked-files=all{pathspec}"
        try:
            output = str(
                sandbox.exec(
                    self._with_task_shell_context(command, task),
                    timeout=self.command_timeout,
                    user=task.metadata.get("agent_user"),
                )
                or ""
            )
        except Exception as exc:
            error = str(exc)
            observation = _json_observation({"mode": mode, "path": relative, "error": error, "truncated": False}, CODEFLOW_SHORT_OUTPUT_LIMIT)
            return CodeFlowExecutionResult(observation=observation, status=ExecutionStatus.ERROR, error=error, duration=time.perf_counter() - started)
        source_characters = len(_compact_json({"mode": mode, "path": relative, "content": output, "truncated": False}))
        observation, truncated = _render_json_content(
            {"mode": mode, "path": relative, "content": output, "truncated": False},
            limit=CODEFLOW_READ_OUTPUT_LIMIT,
        )
        return CodeFlowExecutionResult(
            observation=observation,
            status=ExecutionStatus.SUCCESS,
            duration=time.perf_counter() - started,
            observation_stats=_observation_stats(
                observation,
                source_characters=source_characters,
                output_limit=CODEFLOW_READ_OUTPUT_LIMIT,
                compaction=("compact_json", "head_tail") if truncated else ("compact_json",),
                truncation_reasons=("character_budget",) if truncated else (),
            ),
        )

    def _codeflow_read_file(self, args: dict[str, Any], sandbox: Sandbox, task: Task) -> str:
        return self._codeflow_read_file_execution(args, sandbox, task).observation

    def _codeflow_edit_file(
        self,
        args: dict[str, Any],
        sandbox: Sandbox,
        task: Task,
        history: list[CodeFlowUndoEntry],
    ) -> str:
        mode = str(args.get("mode") or "").strip()
        if mode not in {"replace", "insert", "create", "delete", "apply_patch", "undo"}:
            return _json_observation({"error": "mode must be one of: replace, insert, create, delete, apply_patch, undo"})
        if mode == "undo":
            return self._codeflow_undo(args, sandbox, task, history)
        if mode == "apply_patch":
            return self._codeflow_apply_patch(args, sandbox, task, history)

        try:
            path, relative = self._codeflow_path(args.get("path"), task)
        except ValueError as exc:
            return _json_observation({"error": str(exc)})
        if relative == ".":
            return _json_observation({"error": "edit_file requires a file path, not the repository root"})
        if mode in {"replace", "create", "insert"} and "new_text" not in args:
            return _json_observation({"error": f"new_text is required for {mode}"})
        if mode == "replace" and not str(args.get("old_text") or ""):
            return _json_observation({"error": "old_text is required for replace"})
        if mode == "replace":
            raw_expected_count = args.get("expected_count", 1)
            if (
                isinstance(raw_expected_count, bool)
                or not isinstance(raw_expected_count, int | float)
                or not float(raw_expected_count).is_integer()
            ):
                return _json_observation({"error": "expected_count must be an integer"})
            expected_count = int(raw_expected_count)
            if expected_count < 1:
                return _json_observation({"error": "expected_count must be at least 1"})
        else:
            expected_count = 1
        if mode == "insert":
            if "line" not in args:
                return _json_observation({"error": "line is required for insert"})
            try:
                line = int(args["line"])
            except (TypeError, ValueError):
                return _json_observation({"error": "line must be an integer"})
            if line < 0:
                return _json_observation({"error": "line must be non-negative"})
            if not str(args.get("new_text") or ""):
                return _json_observation({"error": "new_text must be non-empty for insert"})
        else:
            line = 0

        script = r'''
import base64
import hashlib
import json
import sys
from pathlib import Path

payload = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
try:
    root = Path(payload["root"]).resolve()
    path = Path(payload["path"])
    resolved = path.resolve()
    resolved.relative_to(root)
    if path.is_symlink():
        raise ValueError("symbolic links are not supported by edit_file")
    mode = payload["mode"]
    existed = resolved.exists()
    if existed and not resolved.is_file():
        raise ValueError("path is not a regular file")
    before = resolved.read_bytes() if existed else b""

    if mode == "create":
        if existed:
            raise ValueError("path already exists")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(payload["new_text"], encoding="utf-8")
        message = "created file"
    elif mode == "delete":
        if not existed:
            raise ValueError("file not found")
        resolved.unlink()
        message = "deleted file"
    else:
        if not existed:
            raise ValueError("file not found")
        text = before.decode("utf-8")
        if mode == "replace":
            old = payload["old_text"]
            count = text.count(old)
            expected_count = payload["expected_count"]
            if count != expected_count:
                if expected_count == 1:
                    raise ValueError(f"old_text must appear exactly once, found {count}")
                raise ValueError(f"old_text must appear exactly {expected_count} times, found {count}")
            resolved.write_text(text.replace(old, payload["new_text"]), encoding="utf-8")
            message = f"replaced {expected_count} occurrence(s)"
        else:
            line = payload["line"]
            lines = text.splitlines(keepends=True)
            if line > len(lines):
                raise ValueError(f"line must be between 0 and {len(lines)}")
            insertion = payload["new_text"]
            if not insertion.endswith(("\n", "\r")):
                insertion += "\n"
            if line > 0 and lines[line - 1] and not lines[line - 1].endswith(("\n", "\r")):
                lines[line - 1] += "\n"
            lines.insert(line, insertion)
            resolved.write_text("".join(lines), encoding="utf-8")
            message = "inserted text"

    after_exists = resolved.exists()
    after_digest = hashlib.sha256(resolved.read_bytes()).hexdigest() if after_exists else "__MISSING__"
    result = {
        "ok": True,
        "mode": mode,
        "message": message,
        "touched_paths": [payload["relative"]],
        "_undo": {
            "before_exists": existed,
            "before_content": base64.b64encode(before).decode("ascii"),
            "after_digest": after_digest,
        },
    }
except Exception as exc:
    result = {"ok": False, "mode": payload["mode"], "error": str(exc), "touched_paths": []}
print(json.dumps(result, ensure_ascii=False))
'''
        result = self._codeflow_run_json_script(
            script,
            {
                "root": self._repository_root(task),
                "path": path,
                "relative": relative,
                "mode": mode,
                "old_text": str(args.get("old_text") or ""),
                "new_text": str(args.get("new_text") or ""),
                "expected_count": expected_count,
                "line": line,
            },
            sandbox,
            task,
        )
        undo = result.pop("_undo", None)
        if result.get("ok") and isinstance(undo, dict):
            history.append(
                CodeFlowUndoEntry(
                    kind="file",
                    paths=(relative,),
                    path=path,
                    before_exists=bool(undo.get("before_exists")),
                    before_content=str(undo.get("before_content") or ""),
                    after_digest=str(undo.get("after_digest") or ""),
                )
            )
        return _json_observation(result)

    def _codeflow_apply_patch(
        self,
        args: dict[str, Any],
        sandbox: Sandbox,
        task: Task,
        history: list[CodeFlowUndoEntry],
    ) -> str:
        patch = str(args.get("patch") or "")
        if not patch.strip():
            return _json_observation({"error": "patch is required for apply_patch", "touched_paths": []})
        if str(args.get("path") or "").strip():
            try:
                self._codeflow_path(args.get("path"), task)
            except ValueError as exc:
                return _json_observation({"error": str(exc), "touched_paths": []})
        script = r'''
import base64
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

payload = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
try:
    root = Path(payload["root"]).resolve()
    patch = payload["patch"]
    touched = []
    for line in patch.splitlines():
        if not line.startswith(("--- ", "+++ ")):
            continue
        candidate = line[4:].split("\t", 1)[0].strip()
        if candidate == "/dev/null":
            continue
        if candidate.startswith(("a/", "b/")):
            candidate = candidate[2:]
        pure = PurePosixPath(candidate)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"patch path escapes repository root: {candidate}")
        if candidate not in touched:
            touched.append(candidate)
    checked = subprocess.run(["git", "apply", "--check", "-"], cwd=root, input=patch, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if checked.returncode != 0:
        message = (checked.stderr or checked.stdout or "git apply --check failed").strip()
        if any(token in message.lower() for token in ("corrupt patch", "patch fragment without header", "unrecognized input")):
            code = "malformed_patch"
        elif "patch does not apply" in message.lower() or "patch failed" in message.lower():
            code = "context_mismatch"
        else:
            code = "git_apply_check_failed"
        result = {
            "ok": False,
            "mode": "apply_patch",
            "error": message,
            "error_code": code,
            "recovery_hint": "Read the current file and use exact multiline replace for ordinary single-file edits.",
            "touched_paths": [],
        }
        print(json.dumps(result, ensure_ascii=False))
        raise SystemExit(0)
    applied = subprocess.run(["git", "apply", "-"], cwd=root, input=patch, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if applied.returncode != 0:
        raise ValueError((applied.stderr or applied.stdout or "git apply failed").strip())
    result = {"ok": True, "mode": "apply_patch", "message": "applied patch", "touched_paths": touched}
except Exception as exc:
    result = {
        "ok": False,
        "mode": "apply_patch",
        "error": str(exc),
        "error_code": "git_apply_failed",
        "recovery_hint": "Read the current file and use exact multiline replace for ordinary single-file edits.",
        "touched_paths": [],
    }
print(json.dumps(result, ensure_ascii=False))
'''
        result = self._codeflow_run_json_script(
            script,
            {"root": self._repository_root(task), "patch": patch},
            sandbox,
            task,
        )
        if result.get("ok"):
            history.append(
                CodeFlowUndoEntry(
                    kind="patch",
                    paths=tuple(str(item) for item in result.get("touched_paths", []) if isinstance(item, str)),
                    patch=patch,
                )
            )
        return _json_observation(result)

    def _codeflow_undo(
        self,
        args: dict[str, Any],
        sandbox: Sandbox,
        task: Task,
        history: list[CodeFlowUndoEntry],
    ) -> str:
        if not history:
            return _json_observation({"ok": False, "mode": "undo", "error": "no successful edit_file action to undo", "touched_paths": []})
        entry = history[-1]
        guard = str(args.get("path") or "").strip()
        if guard:
            try:
                _, relative = self._codeflow_path(guard, task)
            except ValueError as exc:
                return _json_observation({"ok": False, "mode": "undo", "error": str(exc), "touched_paths": []})
            if relative not in entry.paths:
                return _json_observation(
                    {
                        "ok": False,
                        "mode": "undo",
                        "error": "path does not belong to the most recent edit_file action",
                        "touched_paths": [],
                    }
                )

        if entry.kind == "patch":
            script = r'''
import base64
import json
import subprocess
import sys
from pathlib import Path

payload = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
try:
    root = Path(payload["root"]).resolve()
    patch = payload["patch"]
    checked = subprocess.run(["git", "apply", "--check", "--reverse", "-"], cwd=root, input=patch, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if checked.returncode != 0:
        raise ValueError("repository changed since apply_patch: " + (checked.stderr or checked.stdout or "reverse check failed").strip())
    reverted = subprocess.run(["git", "apply", "--reverse", "-"], cwd=root, input=patch, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if reverted.returncode != 0:
        raise ValueError((reverted.stderr or reverted.stdout or "reverse apply failed").strip())
    result = {"ok": True, "mode": "undo", "message": "reverted patch", "touched_paths": payload["paths"]}
except Exception as exc:
    result = {"ok": False, "mode": "undo", "error": str(exc), "touched_paths": []}
print(json.dumps(result, ensure_ascii=False))
'''
            result = self._codeflow_run_json_script(
                script,
                {"root": self._repository_root(task), "patch": entry.patch, "paths": list(entry.paths)},
                sandbox,
                task,
            )
        else:
            script = r'''
import base64
import hashlib
import json
import sys
from pathlib import Path

payload = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
try:
    root = Path(payload["root"]).resolve()
    path = Path(payload["path"])
    resolved = path.resolve()
    resolved.relative_to(root)
    if path.is_symlink():
        raise ValueError("symbolic links are not supported by edit_file")
    if resolved.exists() and not resolved.is_file():
        raise ValueError("path is not a regular file")
    current_digest = hashlib.sha256(resolved.read_bytes()).hexdigest() if resolved.exists() else "__MISSING__"
    if current_digest != payload["after_digest"]:
        raise ValueError("file changed since the edit; refusing to overwrite subsequent changes")
    if payload["before_exists"]:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_bytes(base64.b64decode(payload["before_content"]))
    elif resolved.exists():
        resolved.unlink()
    result = {"ok": True, "mode": "undo", "message": "reverted file edit", "touched_paths": [payload["relative"]]}
except Exception as exc:
    result = {"ok": False, "mode": "undo", "error": str(exc), "touched_paths": []}
print(json.dumps(result, ensure_ascii=False))
'''
            result = self._codeflow_run_json_script(
                script,
                {
                    "root": self._repository_root(task),
                    "path": entry.path,
                    "relative": entry.paths[0],
                    "before_exists": entry.before_exists,
                    "before_content": entry.before_content,
                    "after_digest": entry.after_digest,
                },
                sandbox,
                task,
            )
        if result.get("ok"):
            history.pop()
        return _json_observation(result)






def _coerce_timeout(value: Any, default: int) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return float(default)






class CodeFlowScaffoldHarness(SWEScaffoldHarness):
    scaffold = "codeflow"
    command_timeout = 300
    model_request_timeout = 3600.0
    model_request_retries = 3
    model_request_retry_base_delay = 1.0
    model_request_retry_max_delay = 12.0
    model_request_retry_jitter = 3.0
    # ``rllm.workflow.n_parallel_tasks`` is the rollout concurrency source of
    # truth. Do not inherit SWE's historical 64-sandbox cap: an eligible
    # verification-enabled milestone rollout owns one primary plus its configured
    # shadow pool, so ``n`` active rollouts may consume up to
    # ``n * (1 + shadow_count)`` slots. The engine switches this instance to
    # one slot when verification is disabled.
    max_concurrent = None
    shadow_enabled = True
    sandbox_slots_per_rollout = 2
    shadow_finalize_stall_timeout = 600.0
    bug_repair_verification_potential_mode = "normalized"
    shadow_partition_timeout_recovery = True
    shadow_partition_recovery_command_timeout = 300.0
    shadow_partition_recovery_max_commands = 6
    denovo_shadow_probe_merge_enable = False
    denovo_shadow_probe_merge_max_steps = 3
    denovo_shadow_sandbox_count = 1

    def configure_shadow_sandbox(self, *, enabled: bool) -> None:
        """Derive shadow provisioning and capacity from verification rewards."""
        self.shadow_enabled = bool(enabled)
        shadow_count = max(
            1,
            int(getattr(self, "denovo_shadow_sandbox_count", 1)),
        )
        self.sandbox_slots_per_rollout = (
            1 + shadow_count if self.shadow_enabled else 1
        )

    def shadow_sandbox_count(self, task: Task) -> int:
        """Return the exact shadow-pool size requested for one task."""

        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        rllm_metadata = (
            metadata.get("rllm")
            if isinstance(metadata.get("rllm"), dict)
            else {}
        )
        task_profile = str(
            metadata.get("task_profile")
            or rllm_metadata.get("task_profile")
            or ""
        )
        if (
            task_profile == "repo_generation_denovoswe"
            and rllm_metadata.get("denovo_background_finalize_active") is True
        ):
            return max(
                1,
                int(getattr(self, "denovo_shadow_sandbox_count", 1)),
            )
        return 1

    def should_use_shadow_sandbox(self, task: Task, verifier_kind: str) -> bool:
        from rllm.harnesses.shadow_sandbox import shadow_eligibility

        if not self.shadow_enabled:
            return False
        eligible, _ = shadow_eligibility(task, verifier_kind)
        return eligible

    def create_shadow_runtime(self, task: Task, primary: Sandbox, shadow: Sandbox, uid: str):
        from rllm.harnesses.shadow_sandbox import ShadowSandboxRuntime

        return ShadowSandboxRuntime.create_and_start(self, task, primary, shadow, uid)

    def create_shadow_runtime_pool(
        self,
        task: Task,
        primary: Sandbox,
        shadows: Sequence[Sandbox],
        uid: str,
    ):
        from rllm.harnesses.shadow_sandbox import ParallelDenovoShadowRuntime

        return ParallelDenovoShadowRuntime.create_and_start(
            self,
            task,
            primary,
            tuple(shadows),
            uid,
        )

    def create_unavailable_shadow_runtime(self, uid: str, error: str):
        from rllm.harnesses.shadow_sandbox import UnavailableShadowRuntime

        timeout = self.__dict__.get("shadow_finalize_stall_timeout")
        if timeout is None:
            timeout = self.shadow_finalize_stall_timeout
        return UnavailableShadowRuntime(uid, error, timeout)






