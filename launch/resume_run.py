"""Prepare durable artifacts for resuming the Codeflow VERL training run.

The shell entrypoint resolves the checkpoint and owns the process lock.  This
module performs the operations that are safer in Python: checkpoint structure
validation, strictly post-step rollout cut-off validation/removal, and
extraction of scalar W&B history into a stable JSONL format that a new offline
run can replay.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

DispatchTicketKey = tuple[int, int, int]


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


_ACTOR_SHARD_RE = re.compile(r"^(model|optim|extra_state)_world_size_(\d+)_rank_(\d+)\.pt$")
_ROLLOUT_SAMPLE_FILENAME_RE = re.compile(r"^(\d+)_.*\.json$")


def _validate_checkpoint(checkpoint_dir: Path, expected_step: int) -> dict[str, int]:
    match_prefix = "global_step_"
    if not checkpoint_dir.name.startswith(match_prefix):
        raise ValueError(f"checkpoint folder must be named global_step_N: {checkpoint_dir}")
    try:
        path_step = int(checkpoint_dir.name[len(match_prefix) :])
    except ValueError as exc:
        raise ValueError(f"checkpoint folder has an invalid step: {checkpoint_dir}") from exc
    if path_step != expected_step:
        raise ValueError(f"checkpoint step mismatch: requested {expected_step}, path contains {path_step}")
    actor_dir = checkpoint_dir / "actor"
    data_path = checkpoint_dir / "data.pt"
    if not actor_dir.is_dir():
        raise FileNotFoundError(f"checkpoint actor directory is missing: {actor_dir}")
    if not data_path.is_file():
        raise FileNotFoundError(f"checkpoint dataloader state is missing: {data_path}")

    kinds = {
        "model": "model_shards",
        "optim": "optimizer_shards",
        "extra_state": "extra_state_shards",
    }
    shards: dict[str, list[tuple[int, int]]] = {kind: [] for kind in kinds}
    for path in actor_dir.iterdir():
        match = _ACTOR_SHARD_RE.fullmatch(path.name)
        if match:
            shards[match.group(1)].append((int(match.group(2)), int(match.group(3))))
    missing = [label for kind, label in kinds.items() if not shards[kind]]
    if missing:
        raise FileNotFoundError(f"checkpoint is missing required actor shard types: {', '.join(missing)}")

    world_sizes: set[int] = set()
    for kind, entries in shards.items():
        kind_world_sizes = {world_size for world_size, _ in entries}
        if len(kind_world_sizes) != 1:
            raise ValueError(f"checkpoint {kind} shards contain inconsistent world sizes: {sorted(kind_world_sizes)}")
        world_size = next(iter(kind_world_sizes))
        ranks = {rank for _, rank in entries}
        expected_ranks = set(range(world_size))
        if ranks != expected_ranks or len(entries) != world_size:
            raise ValueError(f"checkpoint {kind} shards are incomplete for world size {world_size}: found ranks {sorted(ranks)}")
        world_sizes.add(world_size)
    if len(world_sizes) != 1:
        raise ValueError(f"checkpoint model, optimizer, and extra-state shard world sizes differ: {sorted(world_sizes)}")

    return {label: len(shards[kind]) for kind, label in kinds.items()} | {"world_size": next(iter(world_sizes))}


def _ticket_key(value: Any, *, source: str) -> DispatchTicketKey:
    if not isinstance(value, dict):
        raise ValueError(f"dataloader ticket in {source} must be a mapping")
    parts = []
    for key in ("epoch", "start", "end"):
        item = value.get(key)
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"dataloader ticket {key} in {source} must be an integer")
        parts.append(item)
    epoch, start, end = parts
    if epoch < 0 or start < 0 or end <= start:
        raise ValueError(f"invalid dataloader ticket in {source}: {value}")
    return epoch, start, end


def _checkpoint_pending_tickets(
    dataloader_path: Path,
) -> tuple[str, int, set[DispatchTicketKey], tuple[int, int] | None]:
    """Inspect pending tickets across stateful v2 and dynamic v2/v3 states."""
    try:
        import torch
    except ImportError:
        # Lightweight tooling without torch cannot inspect serialized checkpoints.
        # The trainer will still perform the authoritative load before training.
        return "unknown", 1, set(), None
    torch_load = getattr(torch, "load", None)
    if not callable(torch_load):
        # Some lightweight test environments provide an import-only torch stub.
        return "unknown", 1, set(), None
    try:
        state = torch_load(dataloader_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"unable to read dataloader checkpoint {dataloader_path}: {exc}") from exc
    if not isinstance(state, dict):
        return "unknown", 1, set(), None
    loader_type = "dynamic_sampling" if state.get("loader_type") == "dynamic_sampling" else "stateful"
    schema_version = state.get("schema_version", 1)
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError("dataloader checkpoint schema_version must be an integer")
    supported_versions = {3} if loader_type == "dynamic_sampling" else {2}
    if schema_version not in supported_versions:
        raise ValueError(f"unsupported dataloader checkpoint schema_version={schema_version}")
    pending = state.get("pending_dispatches", [])
    if not isinstance(pending, list):
        raise ValueError("dataloader checkpoint pending_dispatches must be a list")
    tickets = {_ticket_key(value, source=str(dataloader_path)) for value in pending}
    if len(tickets) != len(pending):
        raise ValueError("dataloader checkpoint pending_dispatches has duplicates")
    epoch = state.get("epoch")
    cursor = state.get("cursor")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0 or isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
        raise ValueError("dataloader checkpoint has an invalid dispatch frontier")
    # Dynamic sampling chooses positions from checkpointed pools and
    # deliberately serializes cursor=0.  That value is not a sequential
    # dispatch frontier and must never be used to classify rollout artifacts.
    dispatch_frontier = None if loader_type == "dynamic_sampling" else (epoch, cursor)
    return loader_type, schema_version, tickets, dispatch_frontier


def _read_rollout_metadata(
    path: Path,
) -> tuple[int, int | None, DispatchTicketKey | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse rollout JSON {path}: {exc}") from exc
    lifecycle = payload.get("rollout_lifecycle") if isinstance(payload, dict) else None
    step = payload.get("global_step") if isinstance(payload, dict) else None
    if isinstance(lifecycle, dict):
        optimizer_step = lifecycle.get("optimizer_step")
        dispatch_step = lifecycle.get("dispatch_step")
        step = optimizer_step if optimizer_step is not None else dispatch_step
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"rollout JSON has invalid global_step: {path}")
    sample_index = payload.get("sample_index") if isinstance(payload, dict) else None
    if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index < 0:
        sample_index = None
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    ticket_value = metadata.get("dataloader_ticket") if isinstance(metadata, dict) else None
    ticket = _ticket_key(ticket_value, source=str(path)) if ticket_value is not None else None
    return step, sample_index, ticket


def _read_rollout_step(path: Path) -> int:
    return _read_rollout_metadata(path)[0]


def _plan_rollout_cleanup(
    rollout_dir: Path,
    resume_step: int,
) -> tuple[list[Path], list[Path], Counter[int], int]:
    if not rollout_dir.is_dir():
        raise FileNotFoundError(f"resume rollout directory is missing: {rollout_dir}")
    remove: list[Path] = []
    temporary: list[Path] = []
    retained_steps: Counter[int] = Counter()
    retained_max_sample_index = 0
    for path in sorted(rollout_dir.iterdir()):
        if not path.is_file():
            continue
        if ".tmp." in path.name:
            temporary.append(path)
            continue
        if path.suffix != ".json":
            continue
        step, payload_sample_index, _ticket = _read_rollout_metadata(path)
        # Rollout JSON is an append-only audit artifact, not checkpointed
        # trainer state.  Pending tickets are intentionally replayed, and
        # dynamic sampling can reuse the same ticket across several attempts.
        # Retain every checkpoint-or-earlier record and cut only records whose
        # effective optimizer/dispatch step is strictly after the checkpoint.
        if step > resume_step:
            remove.append(path)
        else:
            retained_steps[step] += 1
            filename_match = _ROLLOUT_SAMPLE_FILENAME_RE.fullmatch(path.name)
            filename_sample_index = int(filename_match.group(1)) if filename_match is not None else 0
            retained_max_sample_index = max(
                retained_max_sample_index,
                filename_sample_index,
                payload_sample_index or 0,
            )
    return (
        remove,
        temporary,
        retained_steps,
        retained_max_sample_index,
    )


def _wandb_candidates(log_dir: Path, destination_root: Path) -> list[Path]:
    candidates = list((log_dir / "wandb").glob("offline-run-*/run-*.wandb"))
    for resume_root in log_dir.glob("wandb.resume.*"):
        if resume_root == destination_root:
            continue
        candidates.extend(resume_root.glob("wandb/offline-run-*/run-*.wandb"))
    return sorted({path.resolve() for path in candidates if path.is_file()})


def _scan_wandb_history(path: Path) -> dict[int, dict[str, Any]]:
    """Read history records without allowing W&B's scanner to touch source."""

    try:
        from wandb.proto import wandb_internal_pb2
        from wandb.sdk.internal.datastore import DataStore
    except ImportError as exc:
        raise RuntimeError("the installed W&B package cannot read offline history") from exc

    histories: dict[int, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="rllm-wandb-resume-") as temp_dir:
        copy_path = Path(temp_dir) / path.name
        shutil.copyfile(path, copy_path)
        store = DataStore()
        try:
            store.open_for_scan(str(copy_path))
            while True:
                data = store.scan_data()
                if data is None:
                    break
                record = wandb_internal_pb2.Record()
                record.ParseFromString(data)
                if record.WhichOneof("record_type") != "history":
                    continue
                values: dict[str, Any] = {}
                for item in record.history.item:
                    key = item.key or "/".join(item.nested_key)
                    if not key:
                        continue
                    try:
                        values[key] = json.loads(item.value_json)
                    except json.JSONDecodeError:
                        values[key] = item.value_json
                step_value = values.get("_step")
                if isinstance(step_value, bool) or not isinstance(step_value, int):
                    step_value = int(record.history.step.num)
                if step_value < 0:
                    continue
                metrics = {key: value for key, value in values.items() if not key.startswith("_")}
                histories.setdefault(step_value, {}).update(metrics)
        finally:
            store.close()
    return histories


def _export_wandb_history(
    log_dir: Path,
    destination_root: Path,
    resume_step: int,
) -> tuple[Path, Path, int, int]:
    candidates = _wandb_candidates(log_dir, destination_root)
    if not candidates:
        raise FileNotFoundError(f"no offline W&B run found below {log_dir}")

    selected_path: Path | None = None
    selected_history: dict[int, dict[str, Any]] = {}
    selected_max_step = -1
    selected_record_count = -1
    for candidate in candidates:
        history = _scan_wandb_history(candidate)
        max_step = max(history, default=-1)
        record_count = len(history)
        selected_mtime = selected_path.stat().st_mtime_ns if selected_path else -1
        if (max_step, record_count, candidate.stat().st_mtime_ns) > (
            selected_max_step,
            selected_record_count,
            selected_mtime,
        ):
            selected_path = candidate
            selected_history = history
            selected_max_step = max_step
            selected_record_count = record_count

    if selected_path is None or selected_max_step < 0:
        raise ValueError(f"offline W&B runs below {log_dir} contain no history records")
    if resume_step not in selected_history:
        raise ValueError(f"selected W&B run {selected_path} has no history for checkpoint step {resume_step}")
    replay = [{"step": step, "data": selected_history[step]} for step in sorted(selected_history) if step <= resume_step]
    if not replay:
        raise ValueError(f"selected W&B run {selected_path} has no history at or before step {resume_step}")

    history_path = destination_root / "history.before_resume.jsonl"
    destination_root.mkdir(parents=True, exist_ok=False)
    temporary = history_path.with_name(f".{history_path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            for item in replay:
                file.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary, history_path)
    finally:
        temporary.unlink(missing_ok=True)
    return selected_path, history_path, selected_max_step, len(replay)


def prepare_resume(
    *,
    experiment_root: Path,
    log_dir: Path,
    checkpoint_dir: Path,
    resume_step: int,
    run_timestamp: str,
    wandb_destination: Path,
) -> dict[str, Any]:
    experiment_root = experiment_root.resolve()
    log_dir = log_dir.resolve()
    checkpoint_dir = checkpoint_dir.resolve()
    wandb_destination = wandb_destination.resolve()
    if not experiment_root.is_dir():
        raise FileNotFoundError(f"resume experiment root is missing: {experiment_root}")
    if not log_dir.is_dir():
        raise FileNotFoundError(f"resume log directory is missing: {log_dir}")
    expected_training_root = experiment_root / "checkpoints" / "training"
    if checkpoint_dir.parent != expected_training_root:
        raise ValueError(f"checkpoint must be inside {expected_training_root}, got {checkpoint_dir}")

    shard_counts = _validate_checkpoint(checkpoint_dir, resume_step)
    (
        dataloader_type,
        dataloader_schema,
        pending_tickets,
        dispatch_frontier,
    ) = _checkpoint_pending_tickets(checkpoint_dir / "data.pt")
    rollout_dir = log_dir / "rollout"
    (
        remove,
        temporary,
        retained_steps,
        retained_max_sample_index,
    ) = _plan_rollout_cleanup(
        rollout_dir,
        resume_step,
    )

    # W&B parsing is deliberately completed before destructive rollout cleanup.
    source_wandb, history_path, source_max_step, replay_records = _export_wandb_history(
        log_dir,
        wandb_destination,
        resume_step,
    )

    removed_steps: Counter[int] = Counter()
    for path in remove:
        removed_steps[_read_rollout_step(path)] += 1
        path.unlink()
    for path in temporary:
        path.unlink(missing_ok=True)

    derived_paths = [
        Path(f"{rollout_dir}.codeflow_action_distribution.cache.json"),
        Path(f"{rollout_dir}.codeflow_action_distribution.statistics"),
        Path(f"{rollout_dir}.action_distribution.cache.json"),
        Path(f"{rollout_dir}.action_distribution.statistics"),
        log_dir / "action_distribution.cache.json",
        log_dir / "action_distribution.statistics",
        log_dir / "codeflow_verification_potential.cache.json",
        log_dir / "codeflow_verification_potential.statistics",
        log_dir / "codeflow_verification_potential.language.statistics",
    ]
    removed_derived = []
    for path in derived_paths:
        if path.exists():
            path.unlink()
            removed_derived.append(str(path))

    manifest = {
        "schema_version": 2,
        "run_timestamp": run_timestamp,
        "experiment_root": str(experiment_root),
        "log_dir": str(log_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "resume_step": resume_step,
        "checkpoint_shards": shard_counts,
        "dataloader_state": str(checkpoint_dir / "data.pt"),
        "dataloader": {
            "loader_type": dataloader_type,
            "schema_version": dataloader_schema,
            "pending_replay_groups": len(pending_tickets),
            "dispatch_frontier": (
                {
                    "epoch": dispatch_frontier[0],
                    "cursor": dispatch_frontier[1],
                }
                if dispatch_frontier is not None
                else None
            ),
        },
        "rollout": {
            "directory": str(rollout_dir),
            "cleanup_policy": "effective_step_strictly_after_resume_step",
            "removed_records": len(remove),
            "removed_temporary_files": len(temporary),
            # Kept for manifest compatibility. Pending tickets and dispatch
            # frontiers are no longer deletion criteria for audit JSON.
            "removed_pending_records": 0,
            "removed_post_checkpoint_records": 0,
            "removed_steps": dict(sorted(removed_steps.items())),
            "retained_records": sum(retained_steps.values()),
            "retained_max_step": max(retained_steps, default=None),
            "retained_max_sample_index": retained_max_sample_index,
            "invalidated_derived_files": removed_derived,
        },
        "wandb": {
            "source": str(source_wandb),
            "source_max_step": source_max_step,
            "replay_through_step": resume_step,
            "replay_records": replay_records,
            "history_path": str(history_path),
            "destination": str(wandb_destination),
        },
        "new_logs": {
            "train": str(log_dir / f"train.resume.{run_timestamp}.log"),
            "metrics": str(log_dir / f"metrics.resume.{run_timestamp}.jsonl"),
            "training_params": str(log_dir / f"training_params.resume.{run_timestamp}.json"),
        },
    }
    manifest_path = log_dir / f"resume.{run_timestamp}.manifest.json"
    _atomic_write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--resume-step", type=int, required=True)
    parser.add_argument("--run-timestamp", required=True)
    parser.add_argument("--wandb-destination", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = prepare_resume(
        experiment_root=args.experiment_root,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        resume_step=args.resume_step,
        run_timestamp=args.run_timestamp,
        wandb_destination=args.wandb_destination,
    )
    print(
        "Prepared resume at step "
        f"{manifest['resume_step']}: removed "
        f"{manifest['rollout']['removed_records']} post-step rollout records; "
        f"pending_replay_groups="
        f"{manifest['dataloader']['pending_replay_groups']}; "
        f"W&B history={manifest['wandb']['replay_records']} records",
        flush=True,
    )


if __name__ == "__main__":
    main()
