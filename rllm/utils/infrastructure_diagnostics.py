"""Bounded evidence for infrastructure failure records."""

from __future__ import annotations


def compact_runtime_diagnostics(value, depth=0, *, _budget=None):
    budget = [32768] if _budget is None else _budget
    if depth > 10 or budget[0] <= 0:
        return "<diagnostic limit>"
    if isinstance(value, dict):
        result = {}
        for key, item in list(value.items())[:40]:
            if budget[0] <= 0:
                result["truncated"] = True
                break
            key = str(key)[:128]
            budget[0] -= len(key) + 8
            result[key] = compact_runtime_diagnostics(item, depth + 1, _budget=budget)
        return result
    if isinstance(value, (tuple, list)):
        result = []
        for item in value[:16]:
            if budget[0] <= 0:
                break
            result.append(compact_runtime_diagnostics(item, depth + 1, _budget=budget))
        return result
    if value is None or isinstance(value, (bool, int, float)):
        budget[0] -= 32
        return value
    text = value if isinstance(value, str) else str(value)
    text = text[-max(1, min(4096, budget[0])):]
    budget[0] -= len(text)
    return text


def infrastructure_exception_chain(error):
    """Follow Python and Ray wrappers, retaining typed diagnostic fields."""
    records, seen = [], set()
    pending = [error]
    while pending and len(records) < 8:
        error = pending.pop(0)
        if id(error) in seen:
            continue
        seen.add(id(error))
        record = {"type": type(error).__name__, "message": str(error)[-4000:]}
        for key in ("reason", "stage", "retry_scope", "retryable", "diagnostics", "cleanup_errors"):
            value = getattr(error, key, None)
            if value is not None:
                record[key] = compact_runtime_diagnostics(value)
        records.append(record)
        for nested in (error.__cause__, getattr(error, "cause", None), error.__context__):
            if isinstance(nested, BaseException):
                pending.append(nested)
    return records
