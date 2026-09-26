"""Observe Ray's existing deserialization RLock without replacing it.

The watchdog never acquires the observed lock. It can report contention while
ray.get is still blocked, including a holder outside our observed context.
"""
from __future__ import annotations

import contextlib
import os
import re
import sys
import threading
import time
from rllm.utils.diagnostic_events import emit_diagnostic

_guard = threading.Lock()
_pending = {}
_watchdog = None
_last_stack = 0.0
_last_summary = 0.0


def _bytes(objects):
    total, unknown = 0, 0
    for data, metadata, *_ in objects:
        for value in (data, metadata):
            if value is None:
                continue
            try:
                total += memoryview(value).nbytes
            except (TypeError, ValueError):
                unknown += 1
    return total, unknown


def _stack(frame):
    result = []
    while frame is not None:
        result.append(f"{frame.f_code.co_filename}:{frame.f_lineno}:{frame.f_code.co_name}")
        frame = frame.f_back
    return result


def _report(now=None):
    global _last_stack, _last_summary
    now = time.monotonic() if now is None else now
    with _guard:
        slow = [dict(item) for item in _pending.values() if now - item["since"] > 1]
        if not slow or now - _last_summary < 1:
            return
        _last_summary = now
        capture = now - _last_stack >= 30
        if capture:
            _last_stack = now
    frames = sys._current_frames() if capture else {}
    samples = []
    # Keep holder payload/timing evidence even when hundreds of older waiters
    # would otherwise occupy every sample slot.
    for item in sorted(slow, key=lambda row: (row["phase"] != "holding", row["since"]))[:4]:
        # CPython RLock exposes its owner in repr. Never acquire it to inspect
        # state; a non-CPython implementation simply has an unknown owner.
        match = re.search(r"owner=(\d+)", repr(item["lock"]))
        owner = int(match[1]) if match else None
        sample = {key: value for key, value in item.items() if key != "lock"}
        sample.update(elapsed_seconds=now - item["since"], owner_thread=owner)
        if capture:
            sample["owner_stack"] = _stack(frames.get(owner))
            sample["waiter_stack"] = _stack(frames.get(item["thread_id"]))
        samples.append(sample)
    emit_diagnostic("RAY_DESERIALIZATION_LOCK", **{
        "pid": os.getpid(), "wall_time": time.time(),
        "waiting": sum(item["phase"] == "waiting" for item in slow),
        "holding": sum(item["phase"] == "holding" for item in slow),
        "samples": samples,
    })


def _watch():
    while True:
        time.sleep(.25)
        try:
            _report()
        except Exception:
            pass  # Logging and inspection cannot affect Ray scheduling.


def _register(lock, objects):
    global _watchdog
    token = object()
    size, unknown = _bytes(objects)
    with _guard:
        _pending[token] = {"lock": lock, "thread_id": threading.get_ident(), "since": time.monotonic(),
                           "phase": "waiting", "object_count": len(objects), "payload_bytes": size,
                           "unknown_buffers": unknown}
        if _watchdog is None or not _watchdog.is_alive():
            _watchdog = threading.Thread(target=_watch, name="rllm-ray-lock-watchdog", daemon=True)
            try:
                _watchdog.start()
            except Exception:
                _pending.pop(token, None)
                raise
    return token


def _timing(token, acquired_at, finished_at, cpu_seconds):
    with _guard:
        item = _pending.pop(token, None)
    if item:
        hold = finished_at - acquired_at if acquired_at is not None else 0.0
        if max(hold, item.get("wait_seconds", 0)) > 1:
            # One completed timing per slow operation; full stacks are only
            # emitted by the process-wide, rate-limited watchdog above.
            emit_diagnostic("RAY_DESERIALIZATION_TIMING", **{
                "pid": os.getpid(), "thread_id": item["thread_id"], "wall_time": time.time(),
                "wait_seconds": item.get("wait_seconds"), "hold_seconds": hold,
                "object_count": item["object_count"], "payload_bytes": item["payload_bytes"],
                "thread_cpu_seconds": cpu_seconds,
                "post_release_seconds": time.monotonic() - finished_at,
            })


@contextlib.contextmanager
def observe_deserialization_lock(lock, serialized_objects):
    token = None
    acquired_at = None
    finished_at = time.monotonic()
    cpu_seconds = 0.0
    try:
        token = _register(lock, serialized_objects)
    except Exception:
        pass
    try:
        # Use the original object's context manager, including its recursion
        # count, exception behaviour and exact acquire/release pairing.
        with lock:
            acquired_at = time.monotonic()
            cpu_started = time.thread_time()
            try:
                with _guard:
                    if token in _pending:
                        item = _pending[token]
                        item.update(phase="holding", wait_seconds=acquired_at - item["since"], since=acquired_at)
            except Exception:
                pass
            try:
                yield
            finally:
                finished_at = time.monotonic()
                cpu_seconds = time.thread_time() - cpu_started
    finally:
        try:
            _timing(token, acquired_at, finished_at, cpu_seconds)
        except Exception:
            pass


def _after_fork():
    global _guard, _pending, _watchdog, _last_stack, _last_summary
    _guard, _pending, _watchdog = threading.Lock(), {}, None
    _last_stack = _last_summary = 0.0


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)
