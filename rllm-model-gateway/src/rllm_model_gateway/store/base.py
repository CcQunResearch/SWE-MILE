"""TraceStore protocol — the abstract interface for trace persistence."""

from collections.abc import AsyncIterator
from typing import Any, Protocol


class TraceStore(Protocol):
    """Abstract storage backend for trace persistence.

    Implementations must be async.  The interface uses plain dicts so that
    backends are free to serialise however they like (JSON columns, DynamoDB
    items, etc.).
    """

    async def store_trace(self, trace_id: str, session_id: str, data: dict[str, Any]) -> None:
        """Store a single trace."""
        ...

    async def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        """Get a trace by ID.  Returns ``None`` when not found."""
        ...

    async def get_trace_json(self, trace_id: str) -> bytes | None:
        """Get one trace as its serialized JSON object."""
        ...

    async def get_session_traces(
        self,
        session_id: str,
        since: float | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get all traces for a session, ordered by timestamp ascending."""
        ...

    async def stream_session_traces_json(
        self,
        session_id: str,
        since: float | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream a JSON array of traces without materializing Python dicts."""
        ...

    async def stream_trace_query_json(
        self,
        session_ids: list[str],
        since: float | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream the combined result of a multi-session trace query."""
        ...

    async def delete_session(self, session_id: str) -> int:
        """Delete all traces for a session.  Returns count deleted."""
        ...

    async def list_sessions(
        self,
        since: float | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """List sessions with trace counts."""
        ...

    async def flush(self) -> None:
        """Flush any buffered writes to durable storage."""
        ...

    async def close(self) -> None:
        """Release any resources held by the store."""
        ...
