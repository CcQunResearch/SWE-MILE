"""Publish immutable per-operation reward bytes before the completion receipt."""

from __future__ import annotations

import base64
import hashlib
import json
import shlex

from rllm.sandbox.file_snapshot import FILE_SNAPSHOT_SCRIPT, MAX_FILE_BYTES
from rllm.types import RolloutInfrastructureError


def snapshot_path(status_path: str, operation_id: str, reward_path: str) -> str:
    return f"{status_path}.results/{operation_id}/{hashlib.sha256(reward_path.encode()).hexdigest()}"


def capture_script(status_path: str, operation_id: str, paths: list[str]) -> str:
    """Executed inside the verifier wrapper, once, after the verifier exits."""
    destinations = {path: snapshot_path(status_path, operation_id, path) for path in paths}
    return f'''import hashlib,json,os,stat,sys
path = {status_path!r}
payload = {{"schema_version": 3, "operation_id": {operation_id!r},
           "exit_code": int(sys.argv[1]), "duration_seconds": int(sys.argv[2]),
           "timed_out": sys.argv[3].lower() == "true", "reward_files": {{}}}}
destinations = {destinations!r}
limit = {MAX_FILE_BYTES}
for reward_path, destination in destinations.items():
    record = {{}}
    try:
        fd = os.open(reward_path, os.O_RDONLY | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("not_regular_file")
            raw = source.read(limit + 1)
        if len(raw) > limit:
            record = {{"state": "invalid", "error": "reward_too_large", "size": max(info.st_size, len(raw)), "max_bytes": limit}}
        else:
            raw.decode("utf-8")
            os.makedirs(os.path.dirname(destination), mode=0o700, exist_ok=True)
            temporary = destination + ".tmp"
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
            with os.fdopen(fd, "wb") as output:
                output.write(raw)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
            record = {{"state": "snapshot", "snapshot_path": destination, "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}}
    except FileNotFoundError:
        record = {{"state": "missing"}}
    except UnicodeDecodeError:
        record = {{"state": "invalid", "error": "invalid_utf8"}}
    except ValueError as exc:
        record = {{"state": "invalid", "error": str(exc)}}
    except Exception as exc:
        record = {{"state": "unreadable", "error": type(exc).__name__}}
    payload["reward_files"][reward_path] = record
temporary = path + ".tmp"
with open(temporary, "w") as output:
    json.dump(payload, output)
    output.flush()
    os.fsync(output.fileno())
os.replace(temporary, path)
'''


def resolve_bound_files(sandbox, status: dict, status_path: str, paths: list[str], *, user=None) -> dict:
    """Only read receipt-selected bytes; never recover from mutable reward files."""
    records = dict(status["reward_files"])
    for path in paths:
        record = records.get(path, {})
        if not isinstance(record, dict) or record.get("state") == "missing":
            continue
        if record.get("state") != "snapshot":
            break  # Legacy inline result, or a terminal artifact error.
        expected = snapshot_path(status_path, status["operation_id"], path)
        size = record.get("size")
        if (record.get("snapshot_path") != expected or type(size) is not int
                or not 0 <= size <= MAX_FILE_BYTES or not isinstance(record.get("sha256"), str)):
            records[path] = {"state": "invalid", "error": "invalid_snapshot_receipt"}
            break
        try:
            reader = getattr(sandbox, "read_files", None)
            timeout = getattr(sandbox, "control_read_timeout", 30)
            if callable(reader):
                received = reader([expected], timeout=timeout, user=user, max_bytes=MAX_FILE_BYTES)
            else:
                encoded = base64.b64encode(json.dumps([expected]).encode()).decode()
                command = f"python3 -c {shlex.quote(FILE_SNAPSHOT_SCRIPT)} {shlex.quote(encoded)} {MAX_FILE_BYTES}"
                output = sandbox.exec(command, timeout=timeout, user=user)
                received = json.loads(output.split("__RLLM_FILE_SNAPSHOT__", 1)[1])
            received = received[expected]
            if received.get("state") != "present":
                records[path] = {**received, "snapshot_path": expected}
                if received.get("state") == "missing":
                    records[path].update(state="invalid", error="snapshot_missing")
                break
            raw = received["content"].encode("utf-8")
            if len(raw) != size or hashlib.sha256(raw).hexdigest() != record["sha256"]:
                records[path] = {"state": "invalid", "error": "snapshot_digest_mismatch", "size": len(raw), "expected_size": size}
            else:
                records[path] = {**record, **received, "state": "present"}
        except RolloutInfrastructureError:
            raise
        except PermissionError as exc:
            records[path] = {"state": "unreadable", "error": type(exc).__name__, "retryable": False}
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            records[path] = {"state": "invalid", "error": "snapshot_read_invalid", "error_summary": str(exc)[:500]}
        except Exception as exc:
            authentication = any(token in str(exc).lower() for token in ("unauthorized", "forbidden", "authentication", "401", "403"))
            records[path] = {"state": "transport_error", "error": type(exc).__name__, "error_summary": str(exc)[-2000:], "retryable": not authentication}
        break
    return records
