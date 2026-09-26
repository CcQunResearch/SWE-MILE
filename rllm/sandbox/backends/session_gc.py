"""Durable node-local reclamation of already isolated sandbox directories."""
from __future__ import annotations

import fcntl
import logging
import os
from pathlib import Path
import re
import shutil
import threading
import time
import uuid

logger = logging.getLogger(__name__)
_ENTRY = re.compile(r"^[0-9]+-[0-9a-f]{32}$")
STARTUP_RECLAMATION_TIMEOUT = 120.0


class ReclamationBackpressure(RuntimeError):
    """Admission is blocked until the durable reclamation queue recovers."""


class SessionGarbageCollector:
    """Atomic rename is the journal; a node-wide flock bounds deletion to one worker.

    Only directories whose processes, namespaces and mounts have been checked
    by the caller may enter this queue. Interrupted deletions remain discoverable
    across actor/job restarts. Slow filesystem I/O never occupies a close worker.
    """

    def __init__(self, root: Path, *, max_pending=256, max_age=1800.0,
                 min_free_bytes=32 * 1024**3, start=True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.root.is_symlink():
            raise ValueError("MiniSandbox garbage root must not be a symlink")
        self.max_pending = max_pending
        self.max_age = max_age
        self.min_free_bytes = min_free_bytes
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.last_error = None
        self.thread = None
        if start:
            self.thread = threading.Thread(target=self._run, name="minisandbox-filesystem-gc", daemon=True)
            self.thread.start()

    def _entries(self):
        return sorted(p for p in self.root.iterdir() if _ENTRY.fullmatch(p.name))

    def submit(self, session: Path) -> Path | None:
        session = Path(session)
        if session.is_symlink():
            raise ValueError("MiniSandbox session must not be a symlink")
        destination = self.root / f"{time.time_ns()}-{uuid.uuid4().hex}"
        try:
            session.rename(destination)
        except FileNotFoundError:
            if session.exists() or not self.root.is_dir():
                raise
            destination = None
        # Persist both sides of the rename before releasing the sandbox ledger.
        for directory in (self.root, session.parent):
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        self.wake.set()
        return destination

    def stats(self):
        entries = self._entries()
        oldest = min((int(p.name.split("-", 1)[0]) / 1e9 for p in entries), default=time.time())
        filesystem = os.statvfs(self.root)
        return {"pending": len(entries), "oldest_age_seconds": max(0.0, time.time() - oldest),
                "free_bytes": filesystem.f_bavail * filesystem.f_frsize,
                "free_inodes": filesystem.f_favail, "total_inodes": filesystem.f_files,
                "last_error": self.last_error}

    def check_admission(self):
        state = self.stats()
        if (state["pending"] >= self.max_pending
                or (state["pending"] and state["oldest_age_seconds"] >= self.max_age)
                or state["free_bytes"] < self.min_free_bytes
                or (state["total_inodes"] and state["free_inodes"] < max(1024, state["total_inodes"] // 100))):
            error = ReclamationBackpressure(f"MiniSandbox filesystem reclamation backpressure: {state}")
            error.minisandbox_cleanup_confirmed = True
            error.diagnostics = {"stage": "filesystem_gc_admission", **state}
            raise error

    def wait_for_admission(self, *, timeout=STARTUP_RECLAMATION_TIMEOUT):
        """Give the background worker time to reclaim entries inherited on restart.

        Keep the ordinary admission thresholds, durable timestamps and single
        deletion worker intact. Never run potentially blocking rmtree here.
        """
        deadline = time.monotonic() + timeout
        waiting = False
        while True:
            try:
                self.check_admission()
            except ReclamationBackpressure as error:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self.stop.is_set():
                    error.diagnostics["startup_wait_timeout_seconds"] = timeout
                    raise
                if not waiting:
                    logger.warning("MiniSandbox startup waiting up to %.1fs for reclamation: %s", timeout, error)
                    waiting = True
                self.wake.set()
                self.stop.wait(min(0.25, remaining))
            else:
                if waiting:
                    logger.info("MiniSandbox startup reclamation admission recovered")
                return

    def collect_one(self):
        with (self.root / ".worker.lock").open("a+b") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            entries = self._entries()
            if not entries:
                return False
            path = entries[0]
            if path.is_symlink():
                raise RuntimeError(f"Refusing symlink in MiniSandbox reclamation queue: {path.name}")
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                if path.exists():
                    raise
            self.last_error = None
            return True

    def _run(self):
        while not self.stop.is_set():
            try:
                if self.collect_one():
                    continue
            except Exception as error:
                self.last_error = f"{type(error).__name__}: {error}"[:1000]
                logger.warning("MiniSandbox filesystem reclamation will retry: %s", self.last_error)
            self.wake.wait(5.0)
            self.wake.clear()

    def close(self):
        self.stop.set()
        self.wake.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
