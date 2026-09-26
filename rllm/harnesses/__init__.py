"""Top-level harnesses for rLLM.

Built-in agent flows live here. The CLI ``--agent`` flag picks one
by registered name; entries are listed in ``rllm/registry/agents.json``
and resolved through :func:`rllm.eval.agent_loader.load_agent`.
"""

from rllm.harnesses.action_event import (
    ACTION_EVENT_SCHEMA_VERSION,
    ActionCategory,
    ActionEvent,
    ChangedFile,
    ChangeType,
    ExecutionStatus,
    ExposedLineRange,
    FileExposure,
    ParseStatus,
    RepoStateStatus,
    RestoreConfirmation,
    TestEvent,
    TestEventContinuityStatus,
    TestEventStatus,
    TestCountObservation,
    TestPartitionCounts,
    TestPartitionResult,
    TestResultSource,
    ValidationStatus,
    get_action_event,
    resolve_test_event_continuity,
    summarize_codeflow_bash_behavior,
)

__all__ = [
    "ACTION_EVENT_SCHEMA_VERSION",
    "ActionCategory",
    "ActionEvent",
    "ChangeType",
    "ChangedFile",
    "ExecutionStatus",
    "ExposedLineRange",
    "FileExposure",
    "ParseStatus",
    "RepoStateStatus",
    "RestoreConfirmation",
    "TestEvent",
    "TestEventContinuityStatus",
    "TestEventStatus",
    "TestCountObservation",
    "TestPartitionCounts",
    "TestPartitionResult",
    "TestResultSource",
    "ValidationStatus",
    "get_action_event",
    "resolve_test_event_continuity",
    "summarize_codeflow_bash_behavior",
]
