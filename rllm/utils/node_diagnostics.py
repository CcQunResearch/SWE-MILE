"""Out-of-process MiniSandbox flight recorder; no Ray or GPU imports.

The actor sends nonblocking datagrams. Sampling, local buffering and shared
storage writes run in separate processes, so none can stall actor control.
"""

from __future__ import annotations

import argparse
import hmac
import secrets
import faulthandler
import json
import os
import select
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from collections import OrderedDict

SAMPLE_SECONDS = 5.0
ARCHIVE_SECONDS = 15.0
BUFFER_BYTES = 64 * 1024 * 1024
SEGMENT_BYTES = 8 * 1024 * 1024
SEGMENTS = 7  # Leave 8 MiB for state, bounded trace capture and temporary files.
_TRACE_LOCK = threading.Lock()
_TRACE_OWNER = None


def query_monitor(reference, operation_id, *, timeout=2.0, ensure_close=False):
    """Read a receipt; optionally deliver the same idempotent close off Ray.

    Delivery acknowledgement is not cleanup confirmation. Only the original
    node-owned close future may produce a successful terminal receipt.
    """
    try:
        endpoint = (reference or {}).get("control_endpoint")
        if not endpoint:
            return {"collection_error": "independent control endpoint unavailable"}
        nonce = secrets.token_hex(16)
        request = {"token": endpoint["token"], "nonce": nonce, "operation_id": operation_id}
        if ensure_close:
            request["action"] = "ensure_close"
        deadline = time.monotonic() + timeout
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
            connection.connect((endpoint["host"], endpoint["port"]))
            while time.monotonic() < deadline:
                connection.send(json.dumps(request).encode())
                connection.settimeout(min(.5, max(.001, deadline - time.monotonic())))
                try:
                    response = json.loads(connection.recv(8192))
                except socket.timeout:
                    continue
                if (response.get("nonce") != nonce or response.get("operation_id") != operation_id
                        or response.get("node_id") != reference.get("node_id")
                        or not hmac.compare_digest(str(response.get("token", "")), endpoint["token"])):
                    continue
                response.pop("token", None)
                return response
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return {"collection_error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    return {"collection_error": "independent control deadline exceeded"}


def _read(path, limit=4096):
    try:
        with open(path) as stream:
            return stream.read(limit)
    except OSError as exc:
        return {"error": type(exc).__name__, "message": str(exc)[:300]}


def _identity(pid):
    value = _read(f"/proc/{pid}/stat")
    return value.rsplit(") ", 1)[1].split()[19] if isinstance(value, str) else None


def monitor_instance_paths(local_path, archive_path=None):
    """Never let a restarted actor overwrite its predecessor's flight recorder."""
    instance = f"{os.getpid()}-{time.time_ns()}-{secrets.token_hex(4)}"
    return (Path(local_path) / instance,
            Path(archive_path) / instance if archive_path else None)


def _atomic_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=True) + "\n")
    os.replace(temporary, path)


class RingWriter:
    def __init__(self, root, segment_bytes=SEGMENT_BYTES, segments=SEGMENTS):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.segment_bytes, self.segments = segment_bytes, segments
        self.sequence = 0

    def append(self, payload):
        encoded = json.dumps(payload, ensure_ascii=True).encode() + b"\n"
        if len(encoded) > self.segment_bytes:
            encoded = json.dumps({"event": "record_too_large", "bytes": len(encoded), "sampled_at": time.time()}).encode() + b"\n"
        path = self.root / f"events.{self.sequence % self.segments}.jsonl"
        if path.exists() and path.stat().st_size + len(encoded) > self.segment_bytes:
            self.sequence += 1
            path = self.root / f"events.{self.sequence % self.segments}.jsonl"
            path.write_bytes(b"")
        with path.open("ab") as stream:
            stream.write(encoded)


def _sample(config, anomaly=False):
    pid = config["pid"]
    same = _identity(pid) == config["identity"]
    result = {"sampled_at": time.time(), "node_id": config["node_id"], "pid": pid,
              "process_identity_matches": same, "kernel_release": os.uname().release,
              "pressure": {key: _read(f"/proc/pressure/{key}") for key in ("cpu", "memory", "io")},
              "memory": _read("/proc/meminfo"), "cgroups": {}}
    for controller, root in config.get("cgroups", {}).items():
        paths = [Path(root)]
        if controller == "memory":
            paths += list(Path(root).parents)[:8]
        names = {"memory": ("memory.usage_in_bytes", "memory.limit_in_bytes", "memory.max_usage_in_bytes", "memory.oom_control", "memory.failcnt"),
                 "cpu": ("cpu.stat",), "pids": ("pids.current", "pids.max")}.get(controller, ())
        result["cgroups"][controller] = {str(path): {name: _read(path / name) for name in names} for path in paths}
    if same:
        result["process"] = {name: _read(f"/proc/{pid}/{name}") for name in ("stat", "status", "io", "wchan")}
        if anomaly:
            try:
                threads = list(Path(f"/proc/{pid}/task").iterdir())
                result["thread_count"] = len(threads)
                result["threads"] = {thread.name: {name: _read(thread / name, 2048) for name in ("stack", "wchan", "status")} for thread in threads[:64]}
            except OSError as exc:
                result["thread_error"] = str(exc)
    if anomaly:
        result["ray_logs"] = {}
        logs = Path(config.get("ray_logs", "/tmp/ray/session_latest/logs"))
        candidates = [logs / "raylet.err", logs / "raylet.out"]
        # Worker file names include the PID in Ray's standard log layout.
        candidates += list(logs.glob(f"worker-*-{pid}.*"))[:4]
        candidates += list(logs.glob(f"python-core-worker-*_{pid}.log"))[:2]
        for path in candidates:
            try:
                with path.open("rb") as stream:
                    stream.seek(0, os.SEEK_END)
                    stream.seek(max(0, stream.tell() - 32768))
                    result["ray_logs"][path.name] = stream.read(32768).decode("utf-8", "replace")
            except OSError as exc:
                result["ray_logs"][path.name] = {"error": str(exc)[:300]}
    return result


def _archive(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for path in [*source.glob("events.*.jsonl"), source / "status.json"]:
        if not path.is_file():
            continue
        target = destination / path.name
        temporary = target.with_suffix(".tmp")
        # Only this expendable process performs shared-filesystem operations.
        shutil.copyfile(path, temporary)
        os.replace(temporary, target)


def _monitor(config, channel_fd, trace_fd):
    channel = socket.socket(fileno=channel_fd)
    channel.setblocking(False)
    listener = socket.socket(fileno=config["control_fd"]) if "control_fd" in config else None
    receipts = OrderedDict()
    root = Path(config["local_path"])
    writer = RingWriter(root)
    heartbeat = time.monotonic()
    next_sample = 0.0
    next_archive = 0.0
    last_anomaly = -60.0
    archive = None
    archive_started = 0.0
    archive_error = None
    archived_at = None
    stopping = False
    last_sample = None
    active = {}
    service_heartbeat = {}
    force_anomaly = False
    trace = bytearray()
    identity = config["identity"]
    death_recorded = False
    pidfd = None
    if config.get("trace_signal") is not None and hasattr(os, "pidfd_open"):
        try:
            pidfd = os.pidfd_open(config["pid"])
            if _identity(config["pid"]) != identity:
                os.close(pidfd)
                pidfd = None
        except OSError:
            pass
    while True:
        now = time.monotonic()
        ready, _, _ = select.select([channel, trace_fd] + ([listener] if listener else []), [], [], min(0.25, max(0, next_sample - now)))
        if channel in ready:
            for _ in range(256):
                try:
                    message = json.loads(channel.recv(8192))
                except BlockingIOError:
                    break
                except (ValueError, OSError):
                    break
                heartbeat = now
                kind = message.get("event")
                if kind == "heartbeat":
                    service_heartbeat = message
                if kind == "stop":
                    stopping = True
                elif kind == "start":
                    if len(active) < 4096:
                        active[message["operation_id"]] = message
                elif kind == "finish":
                    active.pop(message["operation_id"], None)
                elif kind == "failure":
                    force_anomaly = True
                    next_sample = 0
                elif kind == "close_terminal":
                    receipts[message["operation_id"]] = message
                    receipts.move_to_end(message["operation_id"])
                    while len(receipts) > 4096:
                        receipts.popitem(last=False)
                if kind != "heartbeat":
                    writer.append({**message, "received_at": time.time()})
        if listener is not None and listener in ready:
            for _ in range(32):
                try:
                    payload, address = listener.recvfrom(8192)
                    request = json.loads(payload)
                    if not hmac.compare_digest(str(request.get("token", "")), config["control_token"]):
                        continue
                    operation_id, nonce = request.get("operation_id"), request.get("nonce")
                    if not isinstance(operation_id, str) or len(operation_id) > 256 or not isinstance(nonce, str) or len(nonce) > 64:
                        continue
                    action = request.get("action")
                    if action not in (None, "ensure_close"):
                        continue
                    if action == "ensure_close" and operation_id not in receipts:
                        try:
                            channel.send(json.dumps({"event": "request_close", "operation_id": operation_id}).encode())
                        except (BlockingIOError, OSError):
                            pass
                    # Trigger a full sample even when the actor heartbeat is
                    # healthy: its Ray response path may still be broken.
                    force_anomaly, next_sample, next_archive = True, 0, 0
                    if now - last_anomaly >= 30:
                        writer.append({"event": "control_capture_requested", "operation_id": operation_id, "received_at": time.time()})
                    response = {"token": config["control_token"], "nonce": nonce,
                                "node_id": config["node_id"], "operation_id": operation_id,
                                "receipt": receipts.get(operation_id), "capture_requested": True,
                                "sampled_at": last_sample, "local_path": str(root),
                                "archive_path": config.get("archive_path")}
                    listener.sendto(json.dumps(response, ensure_ascii=False).encode(), address)
                except BlockingIOError:
                    break
                except (OSError, ValueError, TypeError, AttributeError):
                    continue
        if trace_fd in ready:
            for _ in range(16):
                try:
                    chunk = os.read(trace_fd, 65536)
                    if not chunk:
                        break
                    trace.extend(chunk[:max(0, 512 * 1024 - len(trace))])
                except BlockingIOError:
                    break
        if archive is not None:
            code = archive.poll()
            if code is not None:
                if code == 0:
                    archived_at, archive_error = time.time(), None
                else:
                    detail = archive.stderr.read(4096).decode("utf-8", "replace") if archive.stderr else ""
                    archive_error = archive_error or f"archiver_exit_{code}: {detail[-1000:]}"
                if archive.stderr:
                    archive.stderr.close()
                archive = None
            elif now - archive_started > ARCHIVE_SECONDS:
                archive_error = "archive_timeout; local sampling continues"
                archive.kill()
                # An uninterruptible process is retained, never multiplied.
        if now >= next_sample or stopping:
            dead = _identity(config["pid"]) != identity
            anomaly = force_anomaly or now - heartbeat > 3 * SAMPLE_SECONDS or dead
            force_anomaly = False
            # A process exit must retain the final worker/core-worker logs,
            # even if a slow-heartbeat sample was taken moments earlier.
            collect = (dead and not death_recorded) or (anomaly and now - last_anomaly >= 30)
            try:
                sample = _sample(config, anomaly=collect)
                sample.update(event="sample", heartbeat_age_seconds=now - heartbeat, service_heartbeat=service_heartbeat,
                              active_operations=list(active.values())[:32], active_operation_count=len(active),
                              archive_error=archive_error, archived_at=archived_at)
                if trace:
                    sample["python_threads"] = trace.decode("utf-8", "replace")
                    trace.clear()
                if collect:
                    last_anomaly = now
                    if pidfd is not None and sample["process_identity_matches"]:
                        try:
                            signal.pidfd_send_signal(pidfd, config["trace_signal"])
                        except OSError as exc:
                            sample["python_trace_error"] = str(exc)
                    else:
                        sample["python_trace_error"] = config.get("trace_error") or "safe signal unavailable"
                writer.append(sample)
                death_recorded |= dead
                last_sample = sample["sampled_at"]
                state = {"sampled_at": last_sample, "node_id": config["node_id"], "monitor_pid": os.getpid(),
                         "local_path": str(root), "archive_path": config.get("archive_path"),
                         "archive_error": archive_error, "archived_at": archived_at,
                         "heartbeat_age_seconds": now - heartbeat, "process_identity_matches": sample["process_identity_matches"]}
                _atomic_json(root / "status.json", state)
                try:
                    channel.send(json.dumps(state).encode())
                except (BlockingIOError, OSError):
                    pass
                stopping |= not sample["process_identity_matches"]
            except Exception as exc:
                state = {"sampled_at": last_sample, "collection_error": f"{type(exc).__name__}: {str(exc)[:500]}"}
                try:
                    _atomic_json(root / "status.json", state)
                except OSError:
                    pass
                try:
                    channel.send(json.dumps(state).encode())
                except OSError:
                    pass
            next_sample = now + SAMPLE_SECONDS
        if config.get("archive_path") and archive is None and (now >= next_archive or stopping):
            archive = subprocess.Popen([sys.executable, __file__, "--archive", str(root), config["archive_path"]],
                                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            archive_started = now
            next_archive = now + ARCHIVE_SECONDS
        if stopping:
            # An already running archive may have copied the ring BEFORE the
            # final death sample. Finish it, then publish the final snapshot.
            # Keep the same bounded shutdown and at most one archiver alive.
            final_deadline = time.monotonic() + 2
            if archive is not None:
                try:
                    archive.wait(timeout=max(0, final_deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    archive.kill()
            if config.get("archive_path") and (archive is None or archive.poll() is not None) and time.monotonic() < final_deadline:
                archive = subprocess.Popen([sys.executable, __file__, "--archive", str(root), config["archive_path"]],
                                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                try:
                    archive.wait(timeout=max(0, final_deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    archive.kill()
            break
    if pidfd is not None:
        os.close(pidfd)
    if listener is not None:
        listener.close()


class NodeMonitor:
    def __init__(self, *, node_id, local_path, archive_path=None, cgroups=None, ray_logs=None, host="127.0.0.1"):
        global _TRACE_OWNER
        self.channel, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.channel.setblocking(False)
        trace_read, self.trace_write = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
        self.signal = None
        trace_error = None
        try:
            with _TRACE_LOCK:
                if _TRACE_OWNER is not None or signal.getsignal(signal.SIGUSR2) != signal.SIG_DFL:
                    raise RuntimeError("SIGUSR2 already has a handler")
                faulthandler.register(signal.SIGUSR2, file=self.trace_write, all_threads=True)
                self.signal = signal.SIGUSR2
                _TRACE_OWNER = self
        except (OSError, RuntimeError, ValueError) as exc:
            trace_error = str(exc)
        self.config = {"pid": os.getpid(), "identity": _identity(os.getpid()), "node_id": node_id,
                       "local_path": str(local_path), "archive_path": str(archive_path) if archive_path else None,
                       "cgroups": {key: str(value) for key, value in (cgroups or {}).items()},
                       "ray_logs": str(ray_logs) if ray_logs else "/tmp/ray/session_latest/logs",
                       "trace_signal": self.signal, "trace_error": trace_error}
        self.state = {}
        self.dropped_events = 0
        listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            listener.bind((host, 0))
            listener.setblocking(False)
            self.endpoint = {"host": host, "port": listener.getsockname()[1], "token": secrets.token_hex(32)}
            self.config.update(control_fd=listener.fileno(), control_token=self.endpoint["token"])
            self.process = subprocess.Popen([sys.executable, __file__, "--monitor", json.dumps(self.config),
                                             str(child.fileno()), str(trace_read)],
                                            pass_fds=(child.fileno(), trace_read, listener.fileno()), start_new_session=True,
                                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except BaseException:
            self.channel.close()
            with _TRACE_LOCK:
                if _TRACE_OWNER is self:
                    faulthandler.unregister(self.signal)
                    _TRACE_OWNER = None
            os.close(self.trace_write)
            raise
        finally:
            listener.close()
            child.close()
            os.close(trace_read)

    def emit(self, event, **details):
        try:
            payload = json.dumps({"event": event, **details}, ensure_ascii=False).encode()
            if len(payload) > 8192:
                raise ValueError("monitor event exceeds datagram limit")
            self.channel.send(payload)
        except (OSError, ValueError):
            self.dropped_events += 1
        self._receive_status()

    def set_close_handler(self, handler):
        self._close_handler = handler

    def _receive_status(self):
        for _ in range(4):
            try:
                message = json.loads(self.channel.recv(8192))
                if message.get("event") == "request_close":
                    handler = getattr(self, "_close_handler", None)
                    operation_id = message.get("operation_id")
                    if callable(handler) and isinstance(operation_id, str) and len(operation_id) <= 256:
                        handler(operation_id)
                else:
                    self.state = message
            except (OSError, ValueError, RuntimeError):
                break

    def describe(self):
        self._receive_status()
        return {"local_path": self.config["local_path"], "archive_path": self.config["archive_path"],
                "node_id": self.config["node_id"], "control_endpoint": self.endpoint,
                "monitor_pid": self.process.pid, "monitor_exit_code": self.process.poll(),
                "last_status": self.state, "dropped_events": self.dropped_events,
                "close_submission": callable(getattr(self, "_close_handler", None))}

    def close(self):
        if getattr(self, "closed", False):
            return
        self.closed = True
        self.emit("stop")
        # Keep the socket and trace handler alive until the monitor exits: a
        # queued stop datagram can be lost on close, and an in-flight trace
        # signal must never arrive after its handler was unregistered.
        def reap():
            global _TRACE_OWNER
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                try:
                    self.process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    return
            self.channel.close()
            with _TRACE_LOCK:
                if _TRACE_OWNER is self:
                    faulthandler.unregister(self.signal)
                    _TRACE_OWNER = None
                os.close(self.trace_write)
        threading.Thread(target=reap, name="minisandbox-monitor-reaper", daemon=True).start()


def archived_evidence(reference, timeout=2):
    """A bounded filesystem probe, independent of the node's Ray process."""
    result = dict(reference or {})
    if not result.get("archive_path"):
        return {**result, "collection_error": result.get("collection_error") or "archive path unavailable"}
    process = None
    try:
        process = subprocess.Popen([sys.executable, __file__, "--read-status", result["archive_path"]],
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        output, _ = process.communicate(timeout=timeout)
        result["archived_status"] = json.loads(output)
    except (subprocess.TimeoutExpired, ValueError, OSError) as exc:
        if process is not None and process.poll() is None:
            process.kill()
        result["collection_error"] = type(exc).__name__
        # Do not wait for a process blocked in shared-storage kernel I/O.
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor", nargs=3)
    parser.add_argument("--archive", nargs=2)
    parser.add_argument("--read-status")
    args = parser.parse_args()
    if args.monitor:
        error_channel = socket.socket(fileno=os.dup(int(args.monitor[1])))
        error_channel.setblocking(False)
        try:
            _monitor(json.loads(args.monitor[0]), int(args.monitor[1]), int(args.monitor[2]))
        except Exception as exc:
            try:
                error_channel.send(json.dumps({"sampled_at": None, "collection_error": f"{type(exc).__name__}: {str(exc)[:1000]}"}).encode())
            except OSError:
                pass
            raise
        finally:
            error_channel.close()
    elif args.archive:
        _archive(*args.archive)
    elif args.read_status:
        value = _read(Path(args.read_status) / "status.json", 65536)
        print(json.dumps(json.loads(value) if isinstance(value, str) else value))
