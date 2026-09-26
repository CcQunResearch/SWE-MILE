"""Bounded RPC timing diagnostics. Never inspect locals or request payloads."""
import os
import sys
import threading
import time
from collections import Counter
from rllm.utils.diagnostic_events import emit_diagnostic

_last_stack = 0.0


def event(phase, operation_id, **fields):
    try:
        emit_diagnostic("MINISANDBOX_RPC", **{
            "phase": phase, "operation_id": operation_id, "wall_time": time.time(),
            "monotonic": time.monotonic(), "pid": os.getpid(),
            "thread_id": threading.get_ident(), **fields,
        })
    except Exception:
        pass  # Observability must not replace the original failure.


def call(phase, operation_id, function, *, trace=False, **fields):
    """Time the actual blocking call, including a timeout that returns late."""
    started = time.monotonic()
    if trace:
        event(phase + "_begin", operation_id, **fields)
    outcome = "returned"
    try:
        return function()
    except BaseException as exc:
        outcome = type(exc).__name__
        raise
    finally:
        elapsed = time.monotonic() - started
        slow = elapsed > float(fields.get("wait_budget_seconds", 1)) + 1
        if trace or slow:
            event(phase + "_end", operation_id, elapsed_seconds=elapsed, outcome=outcome, **fields)
        if slow:
            capture_threads(operation_id, phase)


def capture_threads(operation_id, phase):
    global _last_stack
    now = time.monotonic()
    if now - _last_stack < 30:
        return
    _last_stack = now
    try:
        counts = Counter()
        for frame in sys._current_frames().values():
            stack = []
            for _ in range(8):
                if frame is None:
                    break
                stack.append(f"{frame.f_code.co_filename}:{frame.f_lineno}:{frame.f_code.co_name}")
                frame = frame.f_back
            counts[tuple(stack)] += 1
        event("client_threads", operation_id, trigger=phase,
              stack_groups=[{"count": count, "stack": stack} for stack, count in counts.most_common(32)],
              total_threads=sum(counts.values()), total_stack_groups=len(counts))
    except Exception:
        pass
