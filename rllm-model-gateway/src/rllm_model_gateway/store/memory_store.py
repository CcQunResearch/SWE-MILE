"""Bounded-memory trace store for testing and embedded usage.

The public name remains ``MemoryTraceStore`` for compatibility.  Large,
long-lived rollout waves transparently spill compressed trace bodies to
anonymous local temporary files so cumulative conversations cannot grow the
gateway RSS without bound.
"""

import asyncio
import json
import logging
import os
import tempfile
import threading
import time
import zlib
from collections import defaultdict
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, BinaryIO

# This is deliberately an internal fixed limit rather than another trainer
# knob.  A 672-rollout frontier previously retained 1.7 GiB of compressed
# traces and drove the gateway above 4 GiB RSS.  Keeping 256 MiB resident leaves
# room for request buffers, token accumulators, and transient codec output.
_MAX_RESIDENT_TRACE_BYTES = 256 * 1024 * 1024

# Keep codec work off the HTTP loop.  Reads and writes deliberately use
# separate pools: a large rollout frontier can continuously enqueue trace
# compression, and the old shared four-thread FIFO made completed sessions
# wait 5-15 minutes before their first decompression job ran.
_TRACE_WRITE_CODEC_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(1, int(os.environ.get("RLLM_GATEWAY_TRACE_WRITE_WORKERS", "8"))),
    thread_name_prefix="gateway-trace-write",
)
_TRACE_READ_CODEC_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(1, int(os.environ.get("RLLM_GATEWAY_TRACE_READ_WORKERS", "8"))),
    thread_name_prefix="gateway-trace-read",
)
_TRACE_STREAM_CHUNK_BYTES = min(
    1024 * 1024,
    max(64 * 1024, int(os.environ.get("RLLM_GATEWAY_TRACE_STREAM_CHUNK_BYTES", str(1024 * 1024)))),
)


async def _run_codec(executor: ThreadPoolExecutor, func, *args, metrics=None):
    queued = time.perf_counter()

    def run():
        started = time.perf_counter()
        if metrics is not None:
            metrics["codec_queue_seconds"] += started - queued
        try:
            return func(*args)
        finally:
            if metrics is not None:
                metrics["codec_work_seconds"] += time.perf_counter() - started

    future = executor.submit(run)
    # Polling avoids coupling the codec pool to asyncio's default executor and
    # keeps completion delivery portable to restricted runtimes where the
    # loop's cross-thread wakeup fd may be rate-limited.
    delay = 0.0
    try:
        while not future.done():
            await asyncio.sleep(delay)
            delay = min(0.05, max(0.001, delay * 2))
    except asyncio.CancelledError:
        # If codec or spill I/O has already started, let it finish before the
        # coroutine reports cancellation. Poll asynchronously: calling the
        # concurrent Future's blocking ``result()`` here freezes the gateway
        # event loop precisely during a cancellation burst. This path is only
        # used during request/session teardown and prevents store.close() from
        # closing an anonymous spill fd while a worker thread is still writing
        # it.
        if not future.cancel():
            while not future.done():
                try:
                    await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    # The first cancellation is already being honored. Further
                    # cancellation requests must not turn this into a blocking
                    # wait or let the spill fd close under the worker thread.
                    continue
            try:
                future.result()
            except BaseException:
                pass
        raise
    return future.result()


@dataclass(frozen=True, slots=True)
class _CompressedTrace:
    payload: bytes
    uncompressed_bytes: int


@dataclass(slots=True)
class _SpillFile:
    handle: BinaryIO
    lock: threading.Lock = field(default_factory=threading.Lock)
    live_traces: int = 0


@dataclass(frozen=True, slots=True)
class _SpilledTrace:
    spill_file: _SpillFile
    offset: int
    compressed_bytes: int
    uncompressed_bytes: int


_StoredTrace = _CompressedTrace | _SpilledTrace


def _encode_trace(data: dict[str, Any]) -> _CompressedTrace:
    serialized = json.dumps(
        data,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return _CompressedTrace(
        payload=zlib.compress(serialized, level=1),
        uncompressed_bytes=len(serialized),
    )


def _append_spilled_payload(spill_file: _SpillFile, payload: bytes) -> int:
    """Append one compressed payload and return its byte offset."""
    with spill_file.lock:
        fd = spill_file.handle.fileno()
        offset = os.lseek(fd, 0, os.SEEK_END)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(fd, remaining)
            if written <= 0:
                raise OSError("short write while spilling gateway trace")
            remaining = remaining[written:]
        return offset


def _duplicate_spill_fds(traces: list[_StoredTrace]) -> dict[int, int]:
    """Pin spill files for a read snapshot, independent of session cleanup."""
    fds: dict[int, int] = {}
    try:
        for trace in traces:
            if not isinstance(trace, _SpilledTrace):
                continue
            key = id(trace.spill_file)
            if key not in fds:
                fds[key] = os.dup(trace.spill_file.handle.fileno())
    except BaseException:
        for fd in fds.values():
            os.close(fd)
        raise
    return fds


def _close_fds(fds: dict[int, int]) -> None:
    while fds:
        _, fd = fds.popitem()
        os.close(fd)


def _compressed_payload(
    trace: _StoredTrace,
    spill_fds: dict[int, int] | None = None,
) -> bytes:
    if isinstance(trace, _CompressedTrace):
        return trace.payload
    fd = spill_fds[id(trace.spill_file)] if spill_fds is not None else trace.spill_file.handle.fileno()
    payload = os.pread(fd, trace.compressed_bytes, trace.offset)
    if len(payload) != trace.compressed_bytes:
        raise OSError(f"short read from spilled gateway trace: expected={trace.compressed_bytes} actual={len(payload)}")
    return payload


def _decode_trace(
    trace: _StoredTrace,
    spill_fds: dict[int, int] | None = None,
) -> dict[str, Any]:
    return json.loads(zlib.decompress(_compressed_payload(trace, spill_fds)))


def _decode_trace_json(
    trace: _StoredTrace,
    spill_fds: dict[int, int] | None = None,
) -> bytes:
    return zlib.decompress(_compressed_payload(trace, spill_fds))


def _iter_trace_json(traces, spill_fds):
    """Incremental decompression, including traces larger than a stream chunk."""
    yield b"["
    for index, trace in enumerate(traces):
        if index:
            yield b","
        decoder = zlib.decompressobj()
        size = len(trace.payload) if isinstance(trace, _CompressedTrace) else trace.compressed_bytes
        for offset in range(0, size, _TRACE_STREAM_CHUNK_BYTES):
            count = min(_TRACE_STREAM_CHUNK_BYTES, size - offset)
            if isinstance(trace, _CompressedTrace):
                pending = trace.payload[offset:offset + count]
            else:
                pending = os.pread(spill_fds[id(trace.spill_file)], count, trace.offset + offset)
                if len(pending) != count:
                    raise OSError("short read from spilled gateway trace")
            while pending:
                chunk = decoder.decompress(pending, _TRACE_STREAM_CHUNK_BYTES)
                pending = decoder.unconsumed_tail
                if chunk:
                    yield chunk
        if not decoder.eof:
            raise ValueError("truncated compressed gateway trace")
    yield b"]"


class _TraceJSONStream:
    def __init__(self, traces, label):
        self.fds = _duplicate_spill_fds(traces)
        self.chunks = _iter_trace_json(traces, self.fds)
        self.label = label
        self.metrics = {"codec_queue_seconds": 0.0, "codec_work_seconds": 0.0, "response_bytes": 0}
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.closed:
            raise StopAsyncIteration
        try:
            chunk = await _run_codec(_TRACE_READ_CODEC_EXECUTOR, next, self.chunks, None, metrics=self.metrics)
            if chunk is None:
                await self.aclose()
                raise StopAsyncIteration
            self.metrics["response_bytes"] += len(chunk)
            return chunk
        except BaseException:
            await self.aclose()
            raise

    async def aclose(self):
        if not self.closed:
            self.closed = True
            self.chunks.close()
            _close_fds(self.fds)
            logging.getLogger(__name__).info("trace_stream session=%s metrics=%s", self.label, self.metrics)

    def __del__(self):
        # Also reclaim an unstarted response after an early disconnect.
        _close_fds(getattr(self, "fds", {}))


class MemoryTraceStore:
    """Ephemeral store with a bounded compressed resident working set.

    Trace serialization and compression are deliberately executed outside the
    gateway event loop.  Once the resident byte limit is reached, compressed
    bodies are appended to per-session anonymous temporary files.  Those files
    have no filesystem name, are reclaimed when their session is deleted, and
    are also reclaimed by the OS if the gateway is force-killed.
    """

    def __init__(
        self,
        *,
        max_resident_bytes: int = _MAX_RESIDENT_TRACE_BYTES,
    ) -> None:
        if max_resident_bytes < 0:
            raise ValueError("max_resident_bytes must be non-negative")
        self._max_resident_bytes = max_resident_bytes
        # trace_id -> resident or locally spilled compressed JSON object
        self._traces: dict[str, _StoredTrace] = {}
        # trace_id -> created_at
        self._timestamps: dict[str, float] = {}
        # session_id -> list[trace_id]  (insertion order)
        self._session_index: dict[str, list[str]] = defaultdict(list)
        # trace_id -> sessions referencing it.  Maintaining the reverse index
        # makes per-rollout cleanup proportional to that rollout's traces
        # instead of scanning every other live session.
        self._trace_sessions: dict[str, set[str]] = defaultdict(set)
        self._resident_compressed_bytes = 0
        self._spilled_compressed_bytes = 0
        self._uncompressed_bytes = 0
        self._spill_files: dict[str, _SpillFile] = {}

    def _get_spill_file(self, session_id: str) -> _SpillFile:
        spill_file = self._spill_files.get(session_id)
        if spill_file is None:
            # Anonymous files avoid stale artifacts after SIGKILL/restart.  Use
            # unbuffered access because writes use os.write and reads use pread.
            spill_file = _SpillFile(
                tempfile.TemporaryFile(
                    mode="w+b",
                    buffering=0,
                    prefix="rllm-gateway-traces-",
                )
            )
            self._spill_files[session_id] = spill_file
        return spill_file

    def _release_stored_trace(self, trace: _StoredTrace) -> None:
        if isinstance(trace, _CompressedTrace):
            self._resident_compressed_bytes -= len(trace.payload)
        else:
            self._spilled_compressed_bytes -= trace.compressed_bytes
            spill_file = trace.spill_file
            spill_file.live_traces -= 1
            if spill_file.live_traces == 0:
                for session_id, candidate in tuple(self._spill_files.items()):
                    if candidate is spill_file:
                        self._spill_files.pop(session_id, None)
                spill_file.handle.close()
        self._uncompressed_bytes -= trace.uncompressed_bytes

    def _account_stored_trace(self, trace: _StoredTrace) -> None:
        if isinstance(trace, _CompressedTrace):
            self._resident_compressed_bytes += len(trace.payload)
        else:
            self._spilled_compressed_bytes += trace.compressed_bytes
        self._uncompressed_bytes += trace.uncompressed_bytes

    async def store_trace(self, trace_id: str, session_id: str, data: dict[str, Any]) -> None:
        encoded = await _run_codec(_TRACE_WRITE_CODEC_EXECUTOR, _encode_trace, data)
        now = time.time()
        previous = self._traces.get(trace_id)
        previous_resident_bytes = len(previous.payload) if isinstance(previous, _CompressedTrace) else 0
        projected_resident_bytes = self._resident_compressed_bytes - previous_resident_bytes + len(encoded.payload)
        stored: _StoredTrace
        if projected_resident_bytes <= self._max_resident_bytes:
            stored = encoded
        else:
            spill_file = self._get_spill_file(session_id)
            try:
                offset = await _run_codec(
                    _TRACE_WRITE_CODEC_EXECUTOR,
                    _append_spilled_payload,
                    spill_file,
                    encoded.payload,
                )
            except BaseException:
                if spill_file.live_traces == 0:
                    self._spill_files.pop(session_id, None)
                    spill_file.handle.close()
                raise
            spill_file.live_traces += 1
            stored = _SpilledTrace(
                spill_file=spill_file,
                offset=offset,
                compressed_bytes=len(encoded.payload),
                uncompressed_bytes=encoded.uncompressed_bytes,
            )

        # Account the replacement only after a spill write has succeeded.  If
        # old and new entries share a spill file, incrementing the new entry
        # first prevents the replacement from prematurely closing the file.
        if previous is not None:
            self._release_stored_trace(previous)
        self._traces[trace_id] = stored
        self._account_stored_trace(stored)
        if trace_id not in self._timestamps:
            self._timestamps[trace_id] = now
        idx = self._session_index[session_id]
        if trace_id not in idx:
            idx.append(trace_id)
        self._trace_sessions[trace_id].add(session_id)

    async def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        trace = self._traces.get(trace_id)
        if trace is None:
            return None
        spill_fds = _duplicate_spill_fds([trace])
        try:
            return await _run_codec(
                _TRACE_READ_CODEC_EXECUTOR,
                _decode_trace,
                trace,
                spill_fds,
            )
        finally:
            _close_fds(spill_fds)

    async def get_trace_json(self, trace_id: str) -> bytes | None:
        trace = self._traces.get(trace_id)
        if trace is None:
            return None
        spill_fds = _duplicate_spill_fds([trace])
        try:
            return await _run_codec(
                _TRACE_READ_CODEC_EXECUTOR,
                _decode_trace_json,
                trace,
                spill_fds,
            )
        finally:
            _close_fds(spill_fds)

    async def get_session_traces(
        self,
        session_id: str,
        since: float | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        ids = self._session_index.get(session_id, [])
        encoded: list[_StoredTrace] = []
        for tid in ids:
            ts = self._timestamps.get(tid, 0.0)
            if since is not None and ts < since:
                continue
            data = self._traces.get(tid)
            if data is not None:
                encoded.append(data)
        if limit is not None:
            encoded = encoded[:limit]
        spill_fds = _duplicate_spill_fds(encoded)
        try:
            return await _run_codec(
                _TRACE_READ_CODEC_EXECUTOR,
                lambda: [_decode_trace(item, spill_fds) for item in encoded],
            )
        finally:
            _close_fds(spill_fds)

    async def stream_session_traces_json(
        self,
        session_id: str,
        since: float | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[bytes]:
        ids = self._session_index.get(session_id, [])
        encoded = [trace for tid in ids if (since is None or self._timestamps.get(tid, 0.0) >= since) and (trace := self._traces.get(tid)) is not None]
        if limit is not None:
            encoded = encoded[:limit]
        return _TraceJSONStream(encoded, session_id)

    async def stream_trace_query_json(
        self,
        session_ids: list[str],
        since: float | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[bytes]:
        encoded: list[_StoredTrace] = []
        for session_id in session_ids:
            session_traces = [trace for tid in self._session_index.get(session_id, []) if (since is None or self._timestamps.get(tid, 0.0) >= since) and (trace := self._traces.get(tid)) is not None]
            encoded.extend(session_traces if limit is None else session_traces[:limit])
        return _TraceJSONStream(encoded, "query")

    async def delete_session(self, session_id: str) -> int:
        ids = self._session_index.pop(session_id, [])
        deleted = 0
        for tid in ids:
            sessions = self._trace_sessions.get(tid)
            if sessions is not None:
                sessions.discard(session_id)
            if not sessions:
                self._trace_sessions.pop(tid, None)
                trace = self._traces.pop(tid, None)
                if trace is not None:
                    self._release_stored_trace(trace)
                self._timestamps.pop(tid, None)
                deleted += 1
        return deleted

    async def list_sessions(
        self,
        since: float | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for sid, tids in self._session_index.items():
            if not tids:
                continue
            timestamps = [self._timestamps[t] for t in tids if t in self._timestamps]
            if not timestamps:
                continue
            first_at = min(timestamps)
            if since is not None and first_at < since:
                continue
            results.append(
                {
                    "session_id": sid,
                    "trace_count": len(tids),
                    "first_trace_at": first_at,
                    "last_trace_at": max(timestamps),
                }
            )
        results.sort(key=lambda r: r["first_trace_at"], reverse=True)
        if limit is not None:
            results = results[:limit]
        return results

    async def flush(self) -> None:
        """No-op for in-memory store."""

    async def close(self) -> None:
        """Close all anonymous spill files and release the ephemeral store."""
        for spill_file in self._spill_files.values():
            spill_file.handle.close()
        self._spill_files.clear()
        self._traces.clear()
        self._timestamps.clear()
        self._session_index.clear()
        self._trace_sessions.clear()
        self._resident_compressed_bytes = 0
        self._spilled_compressed_bytes = 0
        self._uncompressed_bytes = 0

    def runtime_stats(self) -> dict[str, int]:
        """Return constant-time live working-set metrics."""
        return {
            "live_traces": len(self._traces),
            "compressed_bytes": (self._resident_compressed_bytes + self._spilled_compressed_bytes),
            "resident_compressed_bytes": self._resident_compressed_bytes,
            "spilled_compressed_bytes": self._spilled_compressed_bytes,
            "spill_files": len(self._spill_files),
            "uncompressed_bytes": self._uncompressed_bytes,
            "read_codec_queue_depth": _TRACE_READ_CODEC_EXECUTOR._work_queue.qsize(),
            "write_codec_queue_depth": _TRACE_WRITE_CODEC_EXECUTOR._work_queue.qsize(),
            "read_codec_workers": _TRACE_READ_CODEC_EXECUTOR._max_workers,
            "write_codec_workers": _TRACE_WRITE_CODEC_EXECUTOR._max_workers,
        }
