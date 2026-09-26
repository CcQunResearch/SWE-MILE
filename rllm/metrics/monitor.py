#!/usr/bin/env python3
"""Run incremental Codeflow verification statistics beside a live trainer.

The monitor polls only file names and delegates audit work to
``verification_statistics.py``. A periodic audit starts after
``--every`` uncached rollout JSON files have appeared. Creating ``--stop-file``
requests one unconditional final audit (when at least one rollout exists) and
then terminates the monitor.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SIGNAL_STOP_REQUESTED = False


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be an integer >= 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def _log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def _rollout_count(rollout_dir: Path) -> int:
    try:
        with os.scandir(rollout_dir) as entries:
            return sum(
                entry.is_file(follow_symlinks=False)
                and entry.name.endswith(".json")
                and ".tmp" not in entry.name
                for entry in entries
            )
    except FileNotFoundError:
        return 0


def _cache_entry_count(cache_file: Path, rollout_dir: Path) -> int:
    try:
        payload: Any = json.loads(cache_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return 0
    if not isinstance(payload, dict):
        return 0
    try:
        same_directory = Path(str(payload.get("rollout_dir"))).resolve() == rollout_dir.resolve()
    except (OSError, TypeError, ValueError):
        return 0
    entries = payload.get("entries")
    return len(entries) if same_directory and isinstance(entries, dict) else 0


def _statistics_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(args.statistics_script),
        str(args.rollout_dir),
        "--workers",
        str(args.workers),
        "--mode",
        args.mode,
        "--cache-file",
        str(args.cache_file),
        "--output-file",
        str(args.output_file),
    ]


def _run_statistics(args: argparse.Namespace, *, final: bool, rollout_count: int) -> bool:
    kind = "final" if final else "periodic"
    cached_before = _cache_entry_count(args.cache_file, args.rollout_dir)
    _log(
        f"starting {kind} audit: rollouts={rollout_count} "
        f"cached={cached_before} workers={args.workers}"
    )
    started = time.monotonic()
    completed = subprocess.run(_statistics_command(args), check=False)
    elapsed = time.monotonic() - started
    if completed.returncode == 0:
        cached_after = _cache_entry_count(args.cache_file, args.rollout_dir)
        rollout_count_after = _rollout_count(args.rollout_dir)
        if rollout_count != cached_before and cached_after == cached_before:
            _log(
                f"{kind} audit produced no cache progress; "
                f"cached remains {cached_after}"
            )
            return False
        if final and cached_after != rollout_count_after:
            _log(
                "final audit cache is incomplete: "
                f"rollouts={rollout_count_after} cached={cached_after}"
            )
            return False
        _log(
            f"completed {kind} audit in {elapsed:.1f}s: "
            f"rollouts={rollout_count_after} cached={cached_after} "
            f"output={args.output_file}"
        )
        return True
    _log(f"{kind} audit failed after {elapsed:.1f}s with exit code {completed.returncode}")
    return False


def monitor(args: argparse.Namespace) -> int:
    global _SIGNAL_STOP_REQUESTED

    def request_stop(signum: int, _frame: Any) -> None:
        global _SIGNAL_STOP_REQUESTED
        _SIGNAL_STOP_REQUESTED = True
        _log(f"received signal {signum}; final audit requested")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    retry_not_before = 0.0
    _log(
        f"monitor started: every={args.every} poll_seconds={args.poll_seconds:g} "
        f"rollout_dir={args.rollout_dir}"
    )
    while True:
        count = _rollout_count(args.rollout_dir)
        if args.stop_file.exists() or _SIGNAL_STOP_REQUESTED:
            if count == 0:
                _log("stop requested with no rollout JSON files; final audit skipped")
                return 0
            return 0 if _run_statistics(args, final=True, rollout_count=count) else 1

        cached = _cache_entry_count(args.cache_file, args.rollout_dir)
        # cached > count indicates that resume preparation removed post-checkpoint
        # rollouts; reconcile the cache immediately instead of waiting for N files.
        threshold_reached = count >= cached + args.every or cached > count
        now = time.monotonic()
        if threshold_reached and now >= retry_not_before:
            if _run_statistics(args, final=False, rollout_count=count):
                retry_not_before = 0.0
            else:
                retry_not_before = now + max(60.0, args.poll_seconds)
        time.sleep(args.poll_seconds)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout_dir", type=Path)
    parser.add_argument("--statistics-script", required=True, type=Path)
    parser.add_argument("--cache-file", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--stop-file", required=True, type=Path)
    parser.add_argument("--every", type=_positive_int, default=300)
    parser.add_argument("--workers", type=_positive_int, default=128)
    parser.add_argument("--poll-seconds", type=_positive_float, default=5.0)
    parser.add_argument("--mode", choices=["train", "val", "all"], default="train")
    return parser


def main() -> int:
    args = _parser().parse_args()
    args.rollout_dir = args.rollout_dir.resolve()
    args.statistics_script = args.statistics_script.resolve()
    args.cache_file = args.cache_file.resolve()
    args.output_file = args.output_file.resolve()
    args.stop_file = args.stop_file.resolve()
    if not args.statistics_script.is_file():
        print(f"ERROR: statistics script not found: {args.statistics_script}", file=sys.stderr)
        return 2
    return monitor(args)


if __name__ == "__main__":
    raise SystemExit(main())
