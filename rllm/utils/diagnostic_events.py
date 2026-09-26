"""Bounded diagnostic events, isolated from logging handlers and shared storage.

Callers only encode/enqueue. A daemon writes local files; a disposable child
archives them. Neither disk nor shared-filesystem I/O runs on a Ray/control caller thread.
"""
from __future__ import annotations

import atexit
from collections import deque
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
import uuid

MAX_RECORD_BYTES = 64 * 1024
QUEUE_RECORDS = 2048
QUEUE_BYTES = 8 * 1024 * 1024
SEGMENT_BYTES = 8 * 1024 * 1024
SEGMENTS = 7  # 56 MiB events, leaving room inside 64 MiB for status/temp files.
ARCHIVE_SECONDS = 15.0
_sink = None
_sink_lock = threading.Lock()
_unsubmitted_events = 0


def _safe(value):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(value))[:160]


def diagnostic_run_directory():
    """Resolve a run path lexically; never stat or create shared directories."""
    configured = os.environ.get("RLLM_DIAGNOSTICS_DIR")
    params = os.environ.get("RLLM_TRAINING_PARAMS_PATH")
    if not configured and not params:
        return None
    base = Path(params).parent / "diagnostics" if params else Path(configured)
    run = os.environ.get("RLLM_DIAGNOSTICS_RUN_ID")
    if not run:
        run = Path(params).stem if params else "standalone"
    return base / _safe(run)


def configure_diagnostics(run_log_dir=None):
    """Driver-only setup before building Ray's forwarded runtime environment."""
    global _sink, _unsubmitted_events
    close_diagnostics()
    _sink = None
    _unsubmitted_events = 0
    params = os.environ.get("RLLM_TRAINING_PARAMS_PATH")
    if params:
        os.environ["RLLM_DIAGNOSTICS_DIR"] = str(Path(params).parent / "diagnostics")
    elif run_log_dir:
        os.environ["RLLM_DIAGNOSTICS_DIR"] = str(Path(str(run_log_dir)) / "diagnostics")
    os.environ["RLLM_DIAGNOSTICS_RUN_ID"] = f"{time.time_ns()}-{uuid.uuid4().hex[:8]}"
    return diagnostic_run_directory()


def _memory_limits():
    # Called only by the background writer for allocation failures.
    import resource
    result = {"memlock": resource.getrlimit(resource.RLIMIT_MEMLOCK), "cgroup": {}}
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            _, controllers, relative = line.split(":", 2)
            if not controllers:
                root = Path("/sys/fs/cgroup") / relative.lstrip("/")
                names = ("memory.max", "memory.current", "memory.events")
            elif "memory" in controllers.split(","):
                root = Path("/sys/fs/cgroup/memory") / relative.lstrip("/")
                names = ("memory.limit_in_bytes", "memory.usage_in_bytes", "memory.failcnt")
            else:
                continue
            for name in names:
                try:
                    result["cgroup"][name] = (root / name).read_text()[:1024].strip()
                except OSError:
                    pass
    except (OSError, ValueError) as exc:
        result["collection_error"] = type(exc).__name__
    return result


def _encode(record):
    encoded = (json.dumps(record, ensure_ascii=True) + "\n").encode()
    if len(encoded) > MAX_RECORD_BYTES:
        # Preserve identity and auditable truncation without arbitrary payloads.
        record = {key: record[key] for key in (
            "event", "wall_time", "monotonic", "pid", "hostname", "thread_id",
            "operation_id", "phase", "sequence",
        ) if key in record}
        record.update(truncated=True, original_bytes=len(encoded))
        encoded = (json.dumps(record) + "\n").encode()
    return encoded


class DiagnosticSink:
    def __init__(self, *, local_path=None, archive_path=None, start=True):
        self.pid = os.getpid()
        self.hostname = socket.gethostname()
        self.identity = f"{_safe(self.hostname)}.{self.pid}.{time.time_ns()}"
        run = diagnostic_run_directory()
        self.local_path = Path(local_path) if local_path else (
            Path("/tmp/rllm-diagnostics") / self.identity
        )
        self.archive_path = Path(archive_path) if archive_path else (
            run / "processes" / self.identity if run else None
        )
        self.queue = deque()
        self.queue_bytes = 0
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.stop = threading.Event()
        self.dropped = 0
        self.written = 0
        self.sequence = 0
        self.io_error = None
        self.archive_error = None
        self.archived_at = None
        self.archive = None
        self.archive_started = 0.0
        self.next_archive = 0.0
        self.segment = 0
        self.segment_size = 0
        self.thread = None
        if start:
            self.thread = threading.Thread(target=self._run, name="rllm-diagnostic-writer", daemon=True)
            self.thread.start()

    def submit(self, event, **fields):
        started = time.monotonic()
        record = {**fields, "event": event, "wall_time": time.time(), "monotonic": started,
                  "pid": self.pid, "hostname": self.hostname, "thread_id": threading.get_ident()}
        encoded = _encode(record)
        # Drop instead of waiting even for the short queue bookkeeping lock.
        if not self.lock.acquire(blocking=False):
            self.dropped += 1
            return False
        try:
            if self.stop.is_set() or len(self.queue) >= QUEUE_RECORDS or self.queue_bytes + len(encoded) > QUEUE_BYTES:
                self.dropped += 1
                return False
            self.queue.append((encoded, time.monotonic() - started))
            self.queue_bytes += len(encoded)
        finally:
            self.lock.release()
        self.wake.set()
        return True

    def reference(self):
        return {"local_path": str(self.local_path),
                "archive_path": str(self.archive_path) if self.archive_path else None,
                "dropped_events": self.dropped, "written_events": self.written,
                "unsubmitted_events": _unsubmitted_events,
                "io_error": self.io_error, "archive_error": self.archive_error,
                "archived_at": self.archived_at}

    def _append(self, encoded, enqueue_seconds):
        row = json.loads(encoded)
        row.update(sequence=self.sequence, diagnostic_submit_seconds=enqueue_seconds)
        if "PIN_MEMORY" in row["event"] or row["event"] in {"OPTIMIZER_TRANSFER_ERROR", "FSDP_PARAMETER_TRANSFER_ERROR"}:
            row["memory_limits"] = _memory_limits()
            row["memory_limits_sampled_at"] = time.time()
        encoded = _encode(row)
        self.local_path.mkdir(parents=True, exist_ok=True)
        if self.segment_size and self.segment_size + len(encoded) > SEGMENT_BYTES:
            self.segment += 1
            self.segment_size = 0
        path = self.local_path / f"events.{self.segment % SEGMENTS}.jsonl"
        with path.open("ab" if self.segment_size else "wb") as stream:
            stream.write(encoded)
        self.segment_size += len(encoded)
        self.sequence += 1
        self.written += 1

    def _status(self):
        self.local_path.mkdir(parents=True, exist_ok=True)
        path = self.local_path / "status.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps({**self.reference(), "sampled_at": time.time(),
                                    "queued_events": len(self.queue), "queue_bytes": self.queue_bytes,
                                    "pid": self.pid, "hostname": self.hostname}) + "\n")
        os.replace(temp, path)

    def _poll_archive(self, now, *, force=False):
        if self.archive is not None:
            code = self.archive.poll()
            if code is not None:
                if code == 0:
                    self.archived_at, self.archive_error = time.time(), None
                else:
                    self.archive_error = self.archive_error or f"archiver_exit_{code}"
                self.archive = None
            elif now - self.archive_started > ARCHIVE_SECONDS:
                self.archive_error = "archive_timeout; local recording continues"
                self.archive.kill()
                # Keep this handle until reaped; do not accumulate stuck children.
        if self.archive_path and self.archive is None and (force or now >= self.next_archive):
            self.archive = subprocess.Popen(
                [sys.executable, str(Path(__file__).with_name("node_diagnostics.py")),
                 "--archive", str(self.local_path), str(self.archive_path)],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            self.archive_started, self.next_archive = now, now + ARCHIVE_SECONDS

    def _run(self):
        next_status = 0.0
        try:
            while True:
                self.wake.wait(.25)
                self.wake.clear()
                with self.lock:
                    batch = list(self.queue)
                    self.queue.clear()
                    self.queue_bytes = 0
                for encoded, enqueue_seconds in batch:
                    try:
                        self._append(encoded, enqueue_seconds)
                    except Exception as exc:
                        self.io_error = f"{type(exc).__name__}: {str(exc)[:300]}"
                        self.dropped += 1
                now = time.monotonic()
                try:
                    if now >= next_status or self.stop.is_set():
                        self._status()
                        next_status = now + 1
                    self._poll_archive(now)
                except Exception as exc:
                    self.archive_error = f"{type(exc).__name__}: {str(exc)[:300]}"
                if self.stop.is_set() and not self.queue:
                    break
            # Best effort final archive; training never waits without a bound.
            if self.archive is not None:
                try:
                    self.archive.wait(timeout=.5)
                except subprocess.TimeoutExpired:
                    self.archive.kill()
            self._poll_archive(time.monotonic(), force=True)
            if self.archive is not None:
                try:
                    self.archive.wait(timeout=.5)
                except subprocess.TimeoutExpired:
                    self.archive.kill()
        except Exception as exc:
            self.io_error = f"{type(exc).__name__}: {str(exc)[:300]}"

    def close(self, timeout=2.0):
        self.stop.set()
        self.wake.set()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=timeout)


def _get_sink():
    global _sink
    if _sink is None or _sink.pid != os.getpid():
        if not _sink_lock.acquire(blocking=False):
            return None
        try:
            if _sink is None or _sink.pid != os.getpid():
                _sink = DiagnosticSink()
        finally:
            _sink_lock.release()
    return _sink


def emit_diagnostic(event, **fields):
    global _unsubmitted_events
    try:
        sink = _get_sink()
        if sink is not None:
            return sink.submit(event, **fields)
    except Exception:
        pass  # Diagnostics never replace a training failure.
    _unsubmitted_events += 1
    return False


def diagnostic_reference():
    directory = diagnostic_run_directory()
    result = {"directory": str(directory) if directory else None, "unsubmitted_events": _unsubmitted_events}
    if _sink is not None and _sink.pid == os.getpid():
        result["process"] = _sink.reference()
    return result


def close_diagnostics(timeout=2.0):
    if _sink is not None and _sink.pid == os.getpid():
        _sink.close(timeout)


def _after_fork():
    global _sink, _sink_lock, _unsubmitted_events
    _sink, _sink_lock = None, threading.Lock()
    _unsubmitted_events = 0


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)
atexit.register(close_diagnostics)
