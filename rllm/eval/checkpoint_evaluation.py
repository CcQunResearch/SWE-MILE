"""Durable storage and aggregation for multi-checkpoint pass@n evaluation.

This module deliberately has no Ray, VERL, vLLM, or sandbox imports.  The
filesystem protocol can therefore be unit-tested on a CPU-only host and used
by the distributed evaluation runner without importing the training stack.
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import html
import json
import math
import os
import re
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from rllm.harnesses.action_event import summarize_codeflow_bash_behavior
from rllm.types import Action, Episode, Step, Task, Trajectory
from rllm.workflows.workflow import TerminationReason

_GLOBAL_STEP_RE = re.compile(r"^global_step_(\d+)$")
_ATTEMPT_RE = re.compile(r"^attempt_(\d+)\.json$")
_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_MODEL_FINGERPRINT_SCHEMA = 1
_ROLLOUT_SCHEMA = 2
_SUPPORTED_ROLLOUT_SCHEMAS = frozenset({_ROLLOUT_SCHEMA})
_MANIFEST_SCHEMA = 2
_SUPPORTED_MANIFEST_SCHEMAS = frozenset({_MANIFEST_SCHEMA})
_FAILURE_SCHEMA = 3
_SMALL_FILE_HASH_LIMIT = 4 * 1024 * 1024
# HF shards live on shared-filesystem in the target deployment.  Size/mtime plus small
# head/tail samples catch replacement without rereading tens of gigabytes on
# every resume preflight.
_LARGE_FILE_SAMPLE = 64 * 1024
_BEIJING_TIMEZONE = ZoneInfo("Asia/Shanghai")


def _beijing_timestamp() -> str:
    """Return an ISO-like timestamp explicitly anchored to Beijing time."""

    return datetime.now(_BEIJING_TIMEZONE).strftime("%Y-%m-%dT%H:%M:%S%z")


@dataclass(frozen=True)
class EvaluationModel:
    """One immutable model revision in an evaluation sweep."""

    key: str
    step: int
    path: Path
    kind: str
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "step": self.step,
            "path": str(self.path),
            "kind": self.kind,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class AttemptSpec:
    """A stable dataset-position/attempt pair that still needs sampling."""

    task_index: int
    task_id: str
    attempt_index: int


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def semantic_config_id(config: dict[str, Any]) -> str:
    """Return a readable, deterministic ID for trajectory-compatible config."""

    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"swebench-verified-codeflow-native-{digest}"


def semantic_configs_compatible(stored: dict[str, Any], current: dict[str, Any]) -> bool:
    """Reuse attempts only for the same current evaluation contract."""
    return stored == current


def _hash_file_sample(path: Path, digest: Any) -> None:
    stat = path.stat()
    digest.update(str(path.name).encode())
    digest.update(str(stat.st_size).encode())
    digest.update(str(stat.st_mtime_ns).encode())
    with path.open("rb") as handle:
        if stat.st_size <= _SMALL_FILE_HASH_LIMIT:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        else:
            digest.update(handle.read(_LARGE_FILE_SAMPLE))
            handle.seek(max(0, stat.st_size - _LARGE_FILE_SAMPLE))
            digest.update(handle.read(_LARGE_FILE_SAMPLE))


def validate_hf_model(path: str | Path) -> Path:
    """Validate a Hugging Face model directory and all indexed shards."""

    model_path = Path(path).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_path}")
    for required in ("config.json", "tokenizer_config.json"):
        candidate = model_path / required
        if not candidate.is_file() or candidate.stat().st_size == 0:
            raise FileNotFoundError(f"model file is missing or empty: {candidate}")

    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid model shard index: {index_path}: {exc}") from exc
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"model shard index has no weight_map: {index_path}")
        shard_names = sorted(set(weight_map.values()))
        if any(not isinstance(name, str) or not name for name in shard_names):
            raise ValueError(f"model shard index contains an invalid filename: {index_path}")
        missing = [name for name in shard_names if not (model_path / name).is_file() or (model_path / name).stat().st_size == 0]
        if missing:
            raise FileNotFoundError(f"model checkpoint is incomplete; missing/empty shards: {missing[:5]}")
    else:
        candidates = [model_path / "model.safetensors", model_path / "pytorch_model.bin"]
        if not any(candidate.is_file() and candidate.stat().st_size > 0 for candidate in candidates):
            raise FileNotFoundError(f"model weights or model.safetensors.index.json are missing: {model_path}")
    return model_path


def fingerprint_hf_model(path: str | Path) -> str:
    """Fingerprint model identity without rereading every byte of large shards."""

    model_path = validate_hf_model(path)
    files: set[Path] = {
        candidate
        for candidate in (
            model_path / "config.json",
            model_path / "generation_config.json",
            model_path / "tokenizer_config.json",
            model_path / "tokenizer.json",
            model_path / "chat_template.jinja",
            model_path / "processor_config.json",
        )
        if candidate.is_file()
    }
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        files.add(index_path)
        index = json.loads(index_path.read_text(encoding="utf-8"))
        files.update(model_path / str(name) for name in set(index["weight_map"].values()))
    else:
        files.update(candidate for candidate in (model_path / "model.safetensors", model_path / "pytorch_model.bin") if candidate.is_file())

    digest = hashlib.sha256()
    digest.update(f"hf-model-fingerprint-v{_MODEL_FINGERPRINT_SCHEMA}\n".encode())
    for candidate in sorted(files, key=lambda item: item.name):
        _hash_file_sample(candidate, digest)
    return digest.hexdigest()


def discover_evaluation_models(
    experiment_root: str | Path,
    *,
    base_model_path: str | Path | None,
    evaluate_base: bool,
    checkpoint_stride: int,
    checkpoint_interval: int = 10,
) -> list[EvaluationModel]:
    """Discover and validate the base model and stride-selected HF exports."""

    if isinstance(checkpoint_stride, bool) or checkpoint_stride <= 0:
        raise ValueError("checkpoint_stride must be a positive integer")
    if isinstance(checkpoint_interval, bool) or checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be a positive integer")

    models: list[EvaluationModel] = []
    if evaluate_base:
        if base_model_path is None or not str(base_model_path).strip():
            raise ValueError("base_model_path is required when evaluate_base=true")
        base_path = validate_hf_model(base_model_path)
        models.append(
            EvaluationModel(
                key="base",
                step=0,
                path=base_path,
                kind="base",
                fingerprint=fingerprint_hf_model(base_path),
            )
        )

    checkpoint_root = Path(experiment_root).expanduser().resolve() / "checkpoints" / "huggingface"
    if checkpoint_root.is_dir():
        candidates: list[tuple[int, Path]] = []
        modulus = checkpoint_interval * checkpoint_stride
        for candidate in checkpoint_root.iterdir():
            match = _GLOBAL_STEP_RE.fullmatch(candidate.name)
            if match is None or not candidate.is_dir():
                continue
            step = int(match.group(1))
            if step > 0:
                candidates.append((step, candidate))
        latest_step = max((step for step, _ in candidates), default=None)
        selected = [(step, candidate) for step, candidate in candidates if step % modulus == 0 or step == latest_step]
        for step, candidate in sorted(selected):
            model_path = validate_hf_model(candidate)
            models.append(
                EvaluationModel(
                    key=f"global_step_{step:06d}",
                    step=step,
                    path=model_path,
                    kind="checkpoint",
                    fingerprint=fingerprint_hf_model(model_path),
                )
            )
    elif not evaluate_base:
        raise FileNotFoundError(f"Hugging Face checkpoint directory is missing: {checkpoint_root}")

    if not models:
        raise ValueError("no models were selected for evaluation")
    return models


def task_directory_name(task_index: int, task_id: str) -> str:
    safe_id = _SAFE_RE.sub("_", task_id).strip("._")[:120] or "task"
    return f"{task_index:06d}_{safe_id}"


def _termination_value(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _action_value(action: Any) -> Any:
    return action.action if isinstance(action, Action) else action


def _step_diagnostics(step: Step) -> dict[str, Any]:
    metadata = step.metadata if isinstance(step.metadata, dict) else {}
    agent_metadata = metadata.get("agent_step_metadata")
    if not isinstance(agent_metadata, dict):
        agent_metadata = metadata
    diagnostics: dict[str, Any] = {}
    action_event = agent_metadata.get("action_event")
    if isinstance(action_event, dict):
        diagnostics["action_event"] = {
            key: action_event.get(key)
            for key in (
                "schema_version",
                "action_id",
                "action_category",
                "action_subtype",
                "tool_name",
                "parse_status",
                "validation_status",
                "execution_status",
                "duration",
                "repo_state_before",
                "repo_state_after",
                "repo_changed",
                "shell_behaviors",
                "shell_behavior_evidence",
                "exposed_files",
                "exposed_line_ranges",
                "file_exposures",
                "changed_files",
                "attempted_changed_files",
                "parse_error",
                "validation_error",
                "execution_error",
                "policy_violations",
                "instrumentation_errors",
            )
            if key in action_event
        }
    context = metadata.get("rllm_context")
    if isinstance(context, dict):
        diagnostics["rllm_context"] = context
    tool_reward = metadata.get("tool_call_reward")
    if tool_reward is not None:
        diagnostics["tool_call_reward"] = tool_reward
    for key in ("codeflow_tool_mode", "codeflow_tool_restricted_mode"):
        if key in agent_metadata:
            diagnostics[key] = agent_metadata[key]
    return diagnostics


def _step_payload(index: int, step: Step) -> dict[str, Any]:
    model_output = step.model_output
    observation = step.observation
    metadata = step.metadata if isinstance(step.metadata, dict) else {}
    if observation is None:
        observation = metadata.get("tool_observation")
        nested = metadata.get("agent_step_metadata")
        if observation is None and isinstance(nested, dict):
            observation = nested.get("tool_observation")
    return {
        "index": index,
        "response": step.model_response or step.output or "",
        "thought": step.thought,
        "action": _action_value(step.action),
        "observation": observation,
        "done": bool(step.done),
        "reward": step.reward,
        "finish_reason": getattr(model_output, "finish_reason", None),
        "prompt_length": getattr(model_output, "prompt_length", len(step.prompt_ids or [])),
        "completion_length": getattr(model_output, "completion_length", len(step.response_ids or [])),
        "diagnostics": _step_diagnostics(step),
    }


def _step_token_ids(step: Step, field: str, model_field: str) -> list[Any]:
    """Return canonical step token IDs without relying on tensor truthiness."""

    value = getattr(step, field, None)
    if value is not None and len(value) > 0:
        return list(value)
    model_output = step.model_output
    value = getattr(model_output, model_field, None) if model_output is not None else None
    return list(value) if value is not None else []


def _trajectory_token_usage(trajectory: Trajectory) -> dict[str, int | str]:
    """Count model-output and merged full-sequence tokens for one trajectory.

    The full length follows the same cumulative-prefix merge used by the VERL
    training transform. The initial prompt is counted once; later prompts add
    only newly appended observations before their model completion. A context
    reset starts a new segment instead of pretending the unrelated prompts are
    one sequence.
    """

    output_tokens = 0
    token_length = 0
    segment_full_sequence: list[Any] | None = None
    for step in trajectory.steps:
        prompt = _step_token_ids(step, "prompt_ids", "prompt_ids")
        completion = _step_token_ids(step, "response_ids", "completion_ids")
        output_tokens += len(completion)

        if (
            segment_full_sequence is not None
            and len(prompt) >= len(segment_full_sequence)
            and prompt[: len(segment_full_sequence)] == segment_full_sequence
        ):
            token_length += len(prompt) - len(segment_full_sequence) + len(completion)
        else:
            token_length += len(prompt) + len(completion)
        segment_full_sequence = prompt + completion

    return {
        "output_tokens": output_tokens,
        "token_length": token_length,
        "counting_mode": "cumulative_prefix_segments",
    }


def _episode_token_usage(episode: Episode) -> dict[str, int | float | None | str]:
    usages = [_trajectory_token_usage(trajectory) for trajectory in episode.trajectories]
    output_tokens = [int(usage["output_tokens"]) for usage in usages]
    token_lengths = [int(usage["token_length"]) for usage in usages]
    return {
        "trajectory_count": len(usages),
        "output_tokens": sum(output_tokens),
        "token_length": sum(token_lengths),
        "average_trajectory_output_tokens": _mean_numeric(output_tokens),
        "average_trajectory_token_length": _mean_numeric(token_lengths),
        "counting_mode": "cumulative_prefix_segments",
    }


def _trajectory_payload(trajectory: Trajectory) -> dict[str, Any]:
    metadata = trajectory.metadata if isinstance(trajectory.metadata, dict) else {}
    kept_metadata = {
        key: metadata[key]
        for key in ("codeflow_tool_mode", "codeflow_tool_restricted_mode")
        if key in metadata
    }
    payload = {
        "name": trajectory.name,
        "reward": trajectory.reward,
        "signals": trajectory.signals,
        "token_usage": _trajectory_token_usage(trajectory),
        "steps": [_step_payload(index, step) for index, step in enumerate(trajectory.steps)],
    }
    if kept_metadata:
        payload["metadata"] = kept_metadata
    return payload


def _scalar_metrics(metrics: Any) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}
    return {str(key): value for key, value in metrics.items() if value is None or isinstance(value, str | int | float | bool)}


def episode_to_rollout_payload(
    episode: Episode,
    *,
    semantic_id: str,
    model: EvaluationModel,
    task: Task,
    task_index: int,
    attempt_index: int,
) -> dict[str, Any]:
    """Serialize an evaluation trajectory without training-only provenance."""

    metadata = episode.metadata if isinstance(episode.metadata, dict) else {}
    kept_metadata = {
        key: metadata[key]
        for key in (
            "error",
            "limit_termination_reward",
            "codeflow_tool_mode",
            "codeflow_tool_restricted_mode",
            "infrastructure_attempt_history",
            "no_progress_loop",
            "repo_generation_prompt",
            "repo_generation_environment",
        )
        if key in metadata
    }
    task_metadata = task.metadata if isinstance(task.metadata, dict) else {}
    environment = task_metadata.get("environment") if isinstance(task_metadata.get("environment"), dict) else {}
    rollout_metrics = _scalar_metrics(episode.metrics)
    if isinstance(episode.metrics, dict):
        for nested_key in ("verifier_execution", "verifier_network", "verifier_diagnostics"):
            if isinstance(episode.metrics.get(nested_key), dict):
                rollout_metrics[nested_key] = episode.metrics[nested_key]
        if episode.metrics.get("verifier_profile") == "nl2repo-fresh-sandbox":
            # Keep bounded final-verifier evidence outside scalar metric
            # aggregation. Without it zero parsed tests cannot be audited.
            commands = episode.metrics.get("command_results")
            if isinstance(commands, list):
                kept_metadata["nl2repo_verifier_commands"] = [
                    {
                        "command": str(item.get("command", "")),
                        "executed_command": str(item.get("executed_command", item.get("command", ""))),
                        **{key: item[key] for key in ("status", "stage", "reason") if key in item},
                        "exit_code": item.get("exit_code"),
                        "output": str(item.get("output", ""))[-4000:],
                    }
                    for item in commands if isinstance(item, dict)
                ]
    all_steps = [
        step
        for trajectory in episode.trajectories
        for step in trajectory.steps
    ]
    return {
        "schema_version": _ROLLOUT_SCHEMA,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "semantic_config_id": semantic_id,
        "model": {
            "key": model.key,
            "step": model.step,
            "fingerprint": model.fingerprint,
        },
        "task": {
            "index": task_index,
            "id": task.id,
            "instruction": task.instruction,
            "repo_name": task_metadata.get("repo_name"),
            "commit_hash": task_metadata.get("commit_hash"),
            "docker_image": task_metadata.get("docker_image") or environment.get("docker_image"),
        },
        "attempt_index": attempt_index,
        "reward": _episode_reward(episode),
        "correct": bool(episode.is_correct),
        "termination": _termination_value(episode.termination_reason),
        "turns": sum(len(trajectory.steps) for trajectory in episode.trajectories),
        "token_usage": _episode_token_usage(episode),
        "metrics": rollout_metrics,
        "metadata": kept_metadata,
        "codeflow_bash_behavior": summarize_codeflow_bash_behavior(all_steps),
        "trajectories": [_trajectory_payload(trajectory) for trajectory in episode.trajectories],
    }


def _episode_reward(episode: Episode) -> float | None:
    rewards = [trajectory.reward for trajectory in episode.trajectories if trajectory.reward is not None]
    if rewards:
        return sum(rewards) / len(rewards)
    return None


def infrastructure_failure_reason(episode: Episode) -> str | None:
    """Return why the verifier did not produce a usable sample, if any."""

    metadata = episode.metadata if isinstance(episode.metadata, dict) else {}
    classified = metadata.get("infrastructure_failure")
    if isinstance(classified, dict) and classified.get("reason"):
        return str(classified["reason"])
    metrics = episode.metrics if isinstance(episode.metrics, dict) else {}
    # Evaluator metadata is flattened into Episode.metrics by AgentFlowEngine.
    # Preserve the same explicit infrastructure contract used by rollout
    # failures so a verifier can safely reject an unusable environment without
    # consuming a pass@k attempt.
    evaluator_classified = metrics.get("infrastructure_failure")
    if isinstance(evaluator_classified, dict) and evaluator_classified.get("reason"):
        return str(evaluator_classified["reason"])
    if _termination_value(episode.termination_reason) == _termination_value(TerminationReason.ERROR):
        return "rollout_retry_exhausted"
    error = metrics.get("error")
    if isinstance(error, str):
        normalized = error.lower()
        if "no reward file found" in normalized:
            if metrics.get("sandbox_alive") is False:
                return "verifier_sandbox_lost"
            if metrics.get("verifier_execution_error"):
                return "verifier_execution_failed"
            return "verifier_reward_missing"
        if normalized.startswith("no ") and " directory" in normalized:
            return "verifier_files_missing"
        if normalized.startswith("verifier script ") and " not found" in normalized:
            return "verifier_script_missing"
    return None


def _failure_retry_class(reason: str | None, failure: dict | None = None) -> str:
    from rllm.eval.gateway_supervision import runtime_failure_action

    if runtime_failure_action(reason, failure) is not None:
        return "runtime_recovery"
    if reason in {
        "agent_environment_unavailable",
        "verifier_files_missing",
        "verifier_script_missing",
    }:
        return "deterministic"
    if reason in {"model_request_retry_exhausted", "rollout_retry_exhausted"}:
        return "engine_retry_exhausted"
    return "transient"


def _invalid_runtime_terminal(payload: dict) -> bool:
    terminal = payload.get("terminal_failure") or {}
    failure = (payload.get("metrics") or {}).get("infrastructure_failure") or payload.get("infrastructure_failure") or {}
    return bool(terminal) and _failure_retry_class(terminal.get("reason"), failure) == "runtime_recovery"


def is_infrastructure_failure(episode: Episode) -> bool:
    """Whether this Episode needs failure handling before becoming terminal."""

    return infrastructure_failure_reason(episode) is not None


class EvaluationStore:
    """Filesystem source of truth for resumable, incremental evaluation."""

    def __init__(
        self,
        root: str | Path,
        *,
        semantic_id: str,
        semantic_config: dict[str, Any],
        runs_root: str | Path | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve() / semantic_id
        self.semantic_id = semantic_id
        self.semantic_config = semantic_config
        self.models_dir = self.root / "models"
        self.summaries_dir = self.root / "summaries"
        self.runs_dir = Path(runs_root).expanduser().resolve() if runs_root is not None else self.root / "runs"

    @contextmanager
    def lock(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / ".evaluation.lock"
        with lock_path.open("a+", encoding="utf-8") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"another evaluator is already writing semantic config {self.semantic_id}") from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def initialize_manifest(self, *, run_id: str, requested_budget: int) -> dict[str, Any]:
        manifest_path = self.root / "manifest.json"
        if manifest_path.exists():
            manifest = self._load_json(manifest_path)
            if manifest.get("schema_version") not in _SUPPORTED_MANIFEST_SCHEMAS:
                raise ValueError(f"unsupported evaluation manifest: {manifest_path}")
            if manifest.get("semantic_config_id") != self.semantic_id:
                raise ValueError(f"semantic config ID mismatch in {manifest_path}")
            stored_semantic_config = manifest.get("semantic_config")
            if not isinstance(stored_semantic_config, dict) or not semantic_configs_compatible(
                stored_semantic_config,
                self.semantic_config,
            ):
                raise ValueError(f"semantic configuration changed without changing its ID: {manifest_path}")
            manifest["semantic_config"] = self.semantic_config
            manifest["schema_version"] = _MANIFEST_SCHEMA
        else:
            manifest = {
                "schema_version": _MANIFEST_SCHEMA,
                "semantic_config_id": self.semantic_id,
                "semantic_config": self.semantic_config,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "runs": [],
                "models": {},
            }
        manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        manifest["requested_budget"] = requested_budget
        manifest.setdefault("runs", []).append(
            {
                "id": run_id,
                "started_at": manifest["updated_at"],
                "requested_budget": requested_budget,
            }
        )
        _atomic_json(manifest_path, manifest)
        return manifest

    def ensure_model(self, model: EvaluationModel) -> Path:
        model_dir = self.models_dir / model.key
        model_path = model_dir / "model.json"
        payload = {
            "schema_version": 1,
            **model.to_dict(),
        }
        if model_path.exists():
            existing = self._load_json(model_path)
            if existing.get("fingerprint") != model.fingerprint:
                raise ValueError(f"model {model.key} changed fingerprint; refusing to mix old and new trajectories")
            if int(existing.get("step", -1)) != model.step:
                raise ValueError(f"model step mismatch in {model_path}")
        else:
            _atomic_json(model_path, payload)
        return model_dir

    def attempt_path(self, model: EvaluationModel, spec: AttemptSpec) -> Path:
        return self.models_dir / model.key / "rollouts" / task_directory_name(spec.task_index, spec.task_id) / f"attempt_{spec.attempt_index:04d}.json"

    def scan_attempts(
        self,
        model: EvaluationModel,
        tasks: Sequence[Task],
    ) -> dict[int, dict[int, dict[str, Any]]]:
        """Read and validate every formal attempt JSON for one model."""

        model_dir = self.ensure_model(model)
        expected_ids = {index: task.id for index, task in enumerate(tasks)}
        attempts: dict[int, dict[int, dict[str, Any]]] = {}
        rollout_root = model_dir / "rollouts"
        if not rollout_root.exists():
            return attempts
        for path in sorted(rollout_root.glob("*/*.json")):
            if ".tmp." in path.name or path.name.startswith("."):
                continue
            match = _ATTEMPT_RE.fullmatch(path.name)
            if match is None:
                raise ValueError(f"unexpected JSON file in rollout tree: {path}")
            payload = self._load_json(path)
            task_info = payload.get("task")
            model_info = payload.get("model")
            if payload.get("schema_version") not in _SUPPORTED_ROLLOUT_SCHEMAS:
                raise ValueError(f"unsupported rollout schema in {path}")
            if payload.get("semantic_config_id") != self.semantic_id:
                raise ValueError(f"semantic config mismatch in {path}")
            if not isinstance(task_info, dict) or not isinstance(model_info, dict):
                raise ValueError(f"missing task/model identity in {path}")
            try:
                task_index = int(task_info["index"])
                attempt_index = int(payload["attempt_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid task/attempt identity in {path}") from exc
            if task_index not in expected_ids or task_info.get("id") != expected_ids[task_index]:
                raise ValueError(f"dataset task identity mismatch in {path}")
            if attempt_index != int(match.group(1)):
                raise ValueError(f"attempt filename/payload mismatch in {path}")
            if model_info.get("key") != model.key or model_info.get("fingerprint") != model.fingerprint:
                raise ValueError(f"model identity mismatch in {path}")
            if not isinstance(payload.get("correct"), bool) or payload.get("termination") is None:
                raise ValueError(f"rollout lacks completed verifier outcome in {path}")
            if _invalid_runtime_terminal(payload):
                # Historical node/runtime outages are audit records, not
                # completed benchmark attempts. Preserve the file until a
                # replacement result is ready to commit.
                continue
            bucket = attempts.setdefault(task_index, {})
            if attempt_index in bucket:
                raise ValueError(f"duplicate attempt {attempt_index} for task index {task_index}")
            bucket[attempt_index] = payload
        return attempts

    def missing_attempts(
        self,
        model: EvaluationModel,
        tasks: Sequence[Task],
        budget: int,
    ) -> list[AttemptSpec]:
        attempts = self.scan_attempts(model, tasks)
        missing: list[AttemptSpec] = []
        for task_index, task in enumerate(tasks):
            existing = attempts.get(task_index, {})
            for attempt_index in range(budget):
                if attempt_index not in existing:
                    missing.append(
                        AttemptSpec(
                            task_index=task_index,
                            task_id=task.id,
                            attempt_index=attempt_index,
                        )
                    )
        return missing

    def failure_records(
        self,
        model: EvaluationModel,
        spec: AttemptSpec,
    ) -> list[dict[str, Any]]:
        failure_dir = self.models_dir / model.key / "failures" / task_directory_name(spec.task_index, spec.task_id) / f"attempt_{spec.attempt_index:04d}"
        if not failure_dir.exists():
            return []
        records: list[dict[str, Any]] = []
        for path in sorted(failure_dir.glob("*.json")):
            record = self._load_json(path)
            records.append(record)
        return records

    def _prepare_attempt_destination(self, model: EvaluationModel, spec: AttemptSpec) -> Path:
        destination = self.attempt_path(model, spec)
        if destination.exists():
            previous = self._load_json(destination)
            if not _invalid_runtime_terminal(previous):
                raise FileExistsError(f"attempt already exists: {destination}")
            archive = self.models_dir / model.key / "failures" / task_directory_name(spec.task_index, spec.task_id) / f"attempt_{spec.attempt_index:04d}" / "00000000-superseded_runtime_terminal.json"
            if not archive.exists():
                _atomic_json(archive, {**previous, "failure_reason": previous["terminal_failure"]["reason"], "retry_class": "runtime_recovery"})
        return destination

    def save_episode(
        self,
        model: EvaluationModel,
        task: Task,
        spec: AttemptSpec,
        episode: Episode,
    ) -> Path:
        if is_infrastructure_failure(episode):
            return self.save_failure(model, task, spec, episode)
        destination = self._prepare_attempt_destination(model, spec)
        payload = episode_to_rollout_payload(
            episode,
            semantic_id=self.semantic_id,
            model=model,
            task=task,
            task_index=spec.task_index,
            attempt_index=spec.attempt_index,
        )
        _atomic_json(destination, payload)
        return destination

    def save_failure(
        self,
        model: EvaluationModel,
        task: Task,
        spec: AttemptSpec,
        episode: Episode,
    ) -> Path:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        safe_task = task_directory_name(spec.task_index, task.id)
        destination = self.models_dir / model.key / "failures" / safe_task / f"attempt_{spec.attempt_index:04d}" / f"{timestamp}-{uuid.uuid4().hex[:8]}.json"
        all_steps = [
            step
            for trajectory in episode.trajectories
            for step in trajectory.steps
        ]
        payload = {
            "schema_version": _FAILURE_SCHEMA,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "semantic_config_id": self.semantic_id,
            "model": {
                "key": model.key,
                "step": model.step,
                "fingerprint": model.fingerprint,
            },
            "task": {"index": spec.task_index, "id": task.id},
            "attempt_index": spec.attempt_index,
            "termination": _termination_value(episode.termination_reason),
            "failure_reason": infrastructure_failure_reason(episode),
            "attempt_history": (episode.metadata or {}).get("infrastructure_attempt_history", []),
            "retry_class": _failure_retry_class(
                infrastructure_failure_reason(episode),
                (episode.metadata or {}).get("infrastructure_failure") or (episode.metrics or {}).get("infrastructure_failure"),
            ),
            "error": (episode.metadata or {}).get("error") or (episode.metrics or {}).get("error"),
            "infrastructure_failure": (
                (episode.metadata or {}).get("infrastructure_failure")
                or (episode.metrics or {}).get("infrastructure_failure")
            ),
            "metrics": _scalar_metrics(episode.metrics),
            "verifier_execution": (episode.metrics or {}).get("verifier_execution") if isinstance(episode.metrics, dict) else None,
            "verifier_network": (episode.metrics or {}).get("verifier_network") if isinstance(episode.metrics, dict) else None,
            "verifier_diagnostics": (episode.metrics or {}).get("verifier_diagnostics") if isinstance(episode.metrics, dict) else None,
            "verifier_execution_error": (episode.metrics or {}).get("verifier_execution_error") if isinstance(episode.metrics, dict) else None,
            "sandbox_alive": (episode.metrics or {}).get("sandbox_alive") if isinstance(episode.metrics, dict) else None,
            "codeflow_bash_behavior": summarize_codeflow_bash_behavior(all_steps),
            "trajectories": [
                _trajectory_payload(trajectory)
                for trajectory in episode.trajectories
            ],
        }
        _atomic_json(destination, payload)
        return destination

    def save_terminal_failure(
        self,
        model: EvaluationModel,
        task: Task,
        spec: AttemptSpec,
        episode: Episode,
        *,
        failure_records_seen: int = 1,
    ) -> Path:
        """Persist an exhausted infrastructure failure as a failed attempt.

        Raw executions remain under ``failures/`` for audit.  This companion
        rollout is the durable terminal marker used by resume and aggregation,
        so a task which already consumed all of its retries is not sampled
        again merely because its verifier never produced a normal outcome.
        """

        reason = infrastructure_failure_reason(episode) or "infrastructure_failure"
        classified = (episode.metadata or {}).get("infrastructure_failure") or (episode.metrics or {}).get("infrastructure_failure")
        if _failure_retry_class(reason, classified) == "runtime_recovery":
            raise ValueError(f"runtime failure cannot be finalized as a zero-score attempt: {reason}")
        destination = self._prepare_attempt_destination(model, spec)
        payload = episode_to_rollout_payload(
            episode,
            semantic_id=self.semantic_id,
            model=model,
            task=task,
            task_index=spec.task_index,
            attempt_index=spec.attempt_index,
        )
        payload["correct"] = False
        payload["reward"] = 0.0
        payload["terminal_failure"] = {
            "kind": "infrastructure_failure",
            "reason": reason,
            "retry_class": _failure_retry_class(reason),
            "retries_exhausted": True,
            "failure_records_seen": int(failure_records_seen),
            "finalized_at": _beijing_timestamp(),
        }
        metrics = payload.setdefault("metrics", {})
        classified = (
            (episode.metadata or {}).get("infrastructure_failure")
            or (episode.metrics or {}).get("infrastructure_failure")
        )
        if isinstance(classified, dict):
            metrics["infrastructure_failure"] = classified
        _atomic_json(destination, payload)
        return destination

    def promote_exhausted_failures(
        self,
        model: EvaluationModel,
        tasks: Sequence[Task],
        budget: int,
    ) -> list[AttemptSpec]:
        """Recover exhausted failure records after an interrupted evaluation.

        Engine/deterministic failures are already exhausted when emitted by
        the rollout engine.  Transient verifier failures receive one explicit
        queue-tail retry, so two persisted executions are required before an
        interrupted attempt can be finalized automatically.
        """

        attempts = self.scan_attempts(model, tasks)
        promoted: list[AttemptSpec] = []
        for task_index, task in enumerate(tasks):
            existing = attempts.get(task_index, {})
            for attempt_index in range(budget):
                if attempt_index in existing:
                    continue
                spec = AttemptSpec(task_index, task.id, attempt_index)
                records = self.failure_records(model, spec)
                if not records:
                    continue
                latest = records[-1]
                retry_class = str(latest.get("retry_class") or "transient")
                if retry_class == "runtime_recovery" or _failure_retry_class(latest.get("failure_reason"), latest.get("infrastructure_failure")) == "runtime_recovery":
                    continue
                if retry_class == "transient" and len(records) < 2:
                    continue
                destination = self.attempt_path(model, spec)
                if destination.exists():
                    continue
                trajectories = latest.get("trajectories")
                if not isinstance(trajectories, list):
                    trajectories = []
                metrics = latest.get("metrics")
                if not isinstance(metrics, dict):
                    metrics = {}
                metrics = dict(metrics)
                for key in (
                    "infrastructure_failure",
                    "verifier_execution",
                    "verifier_network",
                    "verifier_diagnostics",
                    "verifier_execution_error",
                    "sandbox_alive",
                ):
                    value = latest.get(key)
                    if value is not None:
                        metrics[key] = value
                reason = str(latest.get("failure_reason") or "infrastructure_failure")
                task_metadata = task.metadata if isinstance(task.metadata, dict) else {}
                environment = task_metadata.get("environment") if isinstance(task_metadata.get("environment"), dict) else {}
                turns = sum(
                    len(trajectory.get("steps", []))
                    for trajectory in trajectories
                    if isinstance(trajectory, dict)
                    and isinstance(trajectory.get("steps", []), list)
                )
                payload = {
                    "schema_version": _ROLLOUT_SCHEMA,
                    "saved_at": _beijing_timestamp(),
                    "semantic_config_id": self.semantic_id,
                    "model": {
                        "key": model.key,
                        "step": model.step,
                        "fingerprint": model.fingerprint,
                    },
                    "task": {
                        "index": task_index,
                        "id": task.id,
                        "instruction": task.instruction,
                        "repo_name": task_metadata.get("repo_name"),
                        "commit_hash": task_metadata.get("commit_hash"),
                        "docker_image": task_metadata.get("docker_image") or environment.get("docker_image"),
                    },
                    "attempt_index": attempt_index,
                    "reward": 0.0,
                    "correct": False,
                    "termination": latest.get("termination") or "error",
                    "turns": turns,
                    "token_usage": None,
                    "metrics": metrics,
                    "metadata": {"error": latest.get("error")} if latest.get("error") else {},
                    "codeflow_bash_behavior": latest.get("codeflow_bash_behavior") or {},
                    "trajectories": trajectories,
                    "terminal_failure": {
                        "kind": "infrastructure_failure",
                        "reason": reason,
                        "retry_class": retry_class,
                        "retries_exhausted": True,
                        "failure_records_seen": len(records),
                        "finalized_at": _beijing_timestamp(),
                        "recovered_failure": True,
                    },
                }
                _atomic_json(destination, payload)
                existing[attempt_index] = payload
                promoted.append(spec)
        return promoted

    def coverage(
        self,
        model: EvaluationModel,
        tasks: Sequence[Task],
        budget: int,
    ) -> dict[str, Any]:
        attempts = self.scan_attempts(model, tasks)
        valid = sum(1 for task_attempts in attempts.values() for attempt_index in task_attempts if attempt_index < budget)
        complete_tasks = sum(all(attempt_index in attempts.get(task_index, {}) for attempt_index in range(budget)) for task_index in range(len(tasks)))
        expected = len(tasks) * budget
        missing = expected - valid
        verifier_timeouts = sum(
            payload.get("metrics", {}).get("verifier_status") == "timeout" for task_attempts in attempts.values() for attempt_index, payload in task_attempts.items() if attempt_index < budget
        )
        infrastructure_failure_records = self.failure_count(model)
        terminal_infrastructure_failure_tasks = sum(
            any(
                attempt_index < budget
                and _is_terminal_infrastructure_failure(payload)
                for attempt_index, payload in task_attempts.items()
            )
            for task_attempts in attempts.values()
        )
        return {
            "valid_attempts": valid,
            "effective_valid_attempts": sum(
                not _is_terminal_infrastructure_failure(payload)
                for task_attempts in attempts.values()
                for attempt_index, payload in task_attempts.items()
                if attempt_index < budget
            ),
            "capability_complete": complete_tasks == len(tasks) and terminal_infrastructure_failure_tasks == 0,
            "expected_attempts": expected,
            "coverage": valid / expected if expected else 1.0,
            "complete_tasks": complete_tasks,
            "task_count": len(tasks),
            "complete": complete_tasks == len(tasks),
            # Backward-compatible legacy field.  Its grain is failure records,
            # so retries of one task can contribute more than one count.
            "infrastructure_failures": infrastructure_failure_records,
            "infrastructure_failure_records": infrastructure_failure_records,
            "terminal_infrastructure_failure_tasks": (
                terminal_infrastructure_failure_tasks
            ),
            "missing_attempts": missing,
            "verifier_timeouts": verifier_timeouts,
        }

    def failure_count(self, model: EvaluationModel) -> int:
        failure_root = self.models_dir / model.key / "failures"
        return sum(1 for _ in failure_root.glob("*/*/*.json")) if failure_root.exists() else 0

    def write_run_progress(self, run_id: str, payload: dict[str, Any], *, model_key: str | None = None) -> Path:
        """Atomically publish live scheduler progress for one evaluation run."""

        if not run_id or Path(run_id).name != run_id or _SAFE_RE.sub("_", run_id) != run_id:
            raise ValueError(f"invalid evaluation run ID: {run_id!r}")
        destination = self.runs_dir / run_id / "progress.json"
        if model_key is not None:
            if not model_key or _SAFE_RE.sub("_", model_key) != model_key or model_key in {".", ".."}:
                raise ValueError(f"invalid evaluation model key: {model_key!r}")
            destination = self.runs_dir / run_id / "models" / model_key / "progress.json"
        _atomic_json(
            destination,
            {
                "schema_version": 3,
                "semantic_config_id": self.semantic_id,
                "run_id": run_id,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                **payload,
            },
        )
        return destination

    def write_runtime_event(self, run_id: str, model_key: str, payload: dict[str, Any]) -> None:
        """Keep recovery diagnostics in this run, outside resumable outcomes."""
        for value in (run_id, model_key):
            if not value or Path(value).name != value or value in {".", ".."} or _SAFE_RE.sub("_", value) != value:
                raise ValueError(f"invalid runtime event path component: {value!r}")
        directory = self.runs_dir / run_id / "models" / model_key
        directory.mkdir(parents=True, exist_ok=True)
        event = {"schema_version": 1, "updated_at": _beijing_timestamp(), "semantic_config_id": self.semantic_id, **payload}
        with (directory / "gateway_events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            stream.flush()
        _atomic_json(directory / "gateway_status.json", event)

    def update_manifest_models(
        self,
        models: Sequence[EvaluationModel],
        tasks: Sequence[Task],
        budget: int,
    ) -> dict[str, Any]:
        manifest_path = self.root / "manifest.json"
        manifest = self._load_json(manifest_path)
        model_status = manifest.setdefault("models", {})
        for model in models:
            model_status[model.key] = {
                **model.to_dict(),
                **self.coverage(model, tasks, budget),
                "budget": budget,
            }
        manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        _atomic_json(manifest_path, manifest)
        return manifest

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSON file {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"JSON root must be an object: {path}")
        return payload


def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    """Standard unbiased estimator: ``1 - C(n-c,k)/C(n,k)``."""

    if not 1 <= k <= n:
        raise ValueError(f"k must satisfy 1 <= k <= n, got k={k}, n={n}")
    if not 0 <= c <= n:
        raise ValueError(f"c must satisfy 0 <= c <= n, got c={c}, n={n}")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _mean_numeric(values: Sequence[Any]) -> float | None:
    numeric = [float(value) for value in values if not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(float(value))]
    return sum(numeric) / len(numeric) if numeric else None


def _stored_trajectory_token_usage(trajectory: Any) -> dict[str, int] | None:
    """Read new usage telemetry or reconstruct it from legacy step lengths."""

    if not isinstance(trajectory, dict):
        return None
    usage = trajectory.get("token_usage")
    if isinstance(usage, dict):
        output_tokens = usage.get("output_tokens")
        token_length = usage.get("token_length")
        if (
            isinstance(output_tokens, int)
            and not isinstance(output_tokens, bool)
            and output_tokens >= 0
            and isinstance(token_length, int)
            and not isinstance(token_length, bool)
            and token_length >= output_tokens
        ):
            return {
                "output_tokens": output_tokens,
                "token_length": token_length,
            }

    # Rollout schema v1/v2 already retained per-step lengths. Exact token IDs
    # were intentionally omitted, so legacy reconstruction assumes the normal
    # cumulative agent context whenever the next prompt is at least as long as
    # the previous prompt+completion; shorter prompts begin a reset segment.
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        return None
    output_tokens = 0
    token_length = 0
    previous_full_length: int | None = None
    for step in steps:
        if not isinstance(step, dict):
            continue
        prompt_length = step.get("prompt_length")
        completion_length = step.get("completion_length")
        if not isinstance(prompt_length, int) or isinstance(prompt_length, bool) or prompt_length < 0:
            continue
        if not isinstance(completion_length, int) or isinstance(completion_length, bool) or completion_length < 0:
            continue
        output_tokens += completion_length
        if previous_full_length is not None and prompt_length >= previous_full_length:
            token_length += prompt_length - previous_full_length + completion_length
        else:
            token_length += prompt_length + completion_length
        previous_full_length = prompt_length + completion_length
    return {
        "output_tokens": output_tokens,
        "token_length": token_length,
    }


def _payload_trajectory_token_usages(payloads: Sequence[dict[str, Any]]) -> list[dict[str, int]]:
    usages: list[dict[str, int]] = []
    for payload in payloads:
        trajectories = payload.get("trajectories")
        if not isinstance(trajectories, list):
            continue
        for trajectory in trajectories:
            usage = _stored_trajectory_token_usage(trajectory)
            if usage is not None:
                usages.append(usage)
    return usages


def _sample_metrics(payloads: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """Aggregate direct per-rollout metrics for one complete sample set."""

    if not payloads:
        return None
    successful = [payload for payload in payloads if payload["correct"]]
    incorrect = [payload for payload in payloads if not payload["correct"]]
    token_usages = _payload_trajectory_token_usages(payloads)
    return {
        "attempts": len(payloads),
        "successes": len(successful),
        "mean_accuracy": len(successful) / len(payloads),
        "average_reward": _mean_numeric([payload.get("reward") for payload in payloads]),
        "average_steps": _mean_numeric([payload.get("turns") for payload in payloads]),
        "average_steps_by_outcome": {
            "correct": _mean_numeric([payload.get("turns") for payload in successful]),
            "incorrect": _mean_numeric([payload.get("turns") for payload in incorrect]),
        },
        "average_trajectory_output_tokens": _mean_numeric(
            [usage["output_tokens"] for usage in token_usages]
        ),
        "average_trajectory_token_length": _mean_numeric(
            [usage["token_length"] for usage in token_usages]
        ),
    }


def _is_terminal_infrastructure_failure(payload: dict[str, Any]) -> bool:
    terminal = payload.get("terminal_failure")
    return (
        isinstance(terminal, dict)
        and terminal.get("kind") == "infrastructure_failure"
        and terminal.get("retries_exhausted") is True
    )


def _aggregate_task_selection(
    attempts: dict[int, dict[int, dict[str, Any]]],
    task_indices: Sequence[int],
    budget: int,
) -> dict[str, Any]:
    """Aggregate one explicitly selected set of complete pass@k tasks."""

    correct_counts: list[int] = []
    payloads: list[dict[str, Any]] = []
    payloads_by_attempt: dict[int, list[dict[str, Any]]] = {
        attempt_index: [] for attempt_index in range(budget)
    }
    for task_index in task_indices:
        task_payloads = [
            attempts[task_index][attempt_index]
            for attempt_index in range(budget)
        ]
        correct_counts.append(
            sum(bool(payload["correct"]) for payload in task_payloads)
        )
        payloads.extend(task_payloads)
        for attempt_index, payload in enumerate(task_payloads):
            payloads_by_attempt[attempt_index].append(payload)

    sample_metrics = _sample_metrics(payloads)
    per_sample_metrics = (
        {
            str(attempt_index + 1): {
                "sample_number": attempt_index + 1,
                "attempt_index": attempt_index,
                **(_sample_metrics(sample_payloads) or {}),
            }
            for attempt_index, sample_payloads in payloads_by_attempt.items()
        }
        if payloads
        else None
    )
    pass_at_k = (
        {
            str(k): sum(
                unbiased_pass_at_k(budget, count, k)
                for count in correct_counts
            )
            / len(correct_counts)
            for k in range(1, budget + 1)
        }
        if correct_counts
        else None
    )
    score_at_k: dict[str, float] | None = None
    if task_indices:
        prefix_scores: dict[str, list[float]] = {
            str(k): [] for k in range(1, budget + 1)
        }
        for task_index in task_indices:
            rewards: list[float] = []
            for attempt_index in range(budget):
                raw_reward = attempts[task_index][attempt_index].get("reward", 0.0)
                try:
                    reward = float(raw_reward)
                except (TypeError, ValueError):
                    reward = 0.0
                if not math.isfinite(reward):
                    reward = 0.0
                rewards.append(min(1.0, max(0.0, reward)))
            for k in range(1, budget + 1):
                prefix_scores[str(k)].append(max(rewards[:k], default=0.0))
        score_at_k = {
            key: sum(values) / len(values)
            for key, values in prefix_scores.items()
        }
    successes = int(sample_metrics["successes"]) if sample_metrics is not None else 0
    selected_attempts = len(task_indices) * budget
    return {
        "correct_counts": correct_counts,
        "sample_metrics": sample_metrics,
        "per_sample_metrics": per_sample_metrics,
        "pass_at_k": pass_at_k,
        "score_at_k": score_at_k,
        "attempts": selected_attempts,
        "success_attempts": successes,
        "incorrect_attempts": selected_attempts - successes,
        "solved_tasks": sum(count > 0 for count in correct_counts),
    }


def aggregate_model(
    store: EvaluationStore,
    model: EvaluationModel,
    tasks: Sequence[Task],
    budget: int,
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    attempts = store.scan_attempts(model, tasks)
    coverage = store.coverage(model, tasks, budget)
    if not coverage["complete"] and not allow_partial:
        raise ValueError(f"model {model.key} is incomplete for pass@{budget}: {coverage['valid_attempts']}/{coverage['expected_attempts']}")

    observed_task_indices = [task_index for task_index in range(len(tasks)) if all(attempt_index in attempts.get(task_index, {}) for attempt_index in range(budget))]
    correct_counts: list[int] = []
    observed_payloads: list[dict[str, Any]] = []
    payloads_by_attempt: dict[int, list[dict[str, Any]]] = {attempt_index: [] for attempt_index in range(budget)}
    for task_index in observed_task_indices:
        task_payloads = [attempts[task_index][attempt_index] for attempt_index in range(budget)]
        count = sum(bool(payload["correct"]) for payload in task_payloads)
        correct_counts.append(count)
        observed_payloads.extend(task_payloads)
        for attempt_index, payload in enumerate(task_payloads):
            payloads_by_attempt[attempt_index].append(payload)
    observed_sample_metrics = _sample_metrics(observed_payloads)
    success_attempts = int(observed_sample_metrics["successes"]) if observed_sample_metrics is not None else 0
    observed_per_sample_metrics = (
        {
            str(attempt_index + 1): {
                "sample_number": attempt_index + 1,
                "attempt_index": attempt_index,
                **(_sample_metrics(payloads) or {}),
            }
            for attempt_index, payloads in payloads_by_attempt.items()
        }
        if observed_payloads
        else None
    )
    observed_pass_at = {str(k): sum(unbiased_pass_at_k(budget, count, k) for count in correct_counts) / len(correct_counts) for k in range(1, budget + 1)} if correct_counts else None
    official_pass_at = observed_pass_at if coverage["complete"] else None
    observed_score_at: dict[str, float] | None = None
    if observed_task_indices:
        prefix_scores: dict[str, list[float]] = {
            str(k): [] for k in range(1, budget + 1)
        }
        for task_index in observed_task_indices:
            rewards: list[float] = []
            for attempt_index in range(budget):
                raw_reward = attempts[task_index][attempt_index].get("reward", 0.0)
                try:
                    reward = float(raw_reward)
                except (TypeError, ValueError):
                    reward = 0.0
                if not math.isfinite(reward):
                    reward = 0.0
                rewards.append(min(1.0, max(0.0, reward)))
            for k in range(1, budget + 1):
                prefix_scores[str(k)].append(max(rewards[:k], default=0.0))
        observed_score_at = {
            key: sum(values) / len(values)
            for key, values in prefix_scores.items()
        }
    official_score_at = observed_score_at if coverage["complete"] else None
    observed_attempts = len(observed_task_indices) * budget
    infrastructure_task_indices = [
        task_index
        for task_index in range(len(tasks))
        if any(
            _is_terminal_infrastructure_failure(payload)
            for attempt_index, payload in attempts.get(task_index, {}).items()
            if attempt_index < budget
        )
    ]
    infrastructure_task_index_set = set(infrastructure_task_indices)
    capability_task_indices = [
        task_index
        for task_index in range(len(tasks))
        if task_index not in infrastructure_task_index_set
    ]
    observed_capability_task_indices = [
        task_index
        for task_index in observed_task_indices
        if task_index not in infrastructure_task_index_set
    ]
    capability_complete = len(observed_capability_task_indices) == len(
        capability_task_indices
    )
    capability = _aggregate_task_selection(
        attempts,
        observed_capability_task_indices,
        budget,
    )
    capability_sample_metrics = capability["sample_metrics"]
    capability_warning = None
    if infrastructure_task_indices:
        capability_warning = "Tasks with terminal infrastructure failures were excluded; these selected-task metrics may be selection-biased and are not full-dataset capability scores."
    if not capability_complete:
        capability_warning = (
            "Infrastructure-excluded observed metrics still omit tasks with "
            "incomplete non-infrastructure attempts and may be selection-biased."
        )
    elif not capability_task_indices:
        capability_warning = (
            "All dataset tasks were excluded because they contain a terminal "
            "infrastructure failure; capability metrics are unavailable."
        )
    infrastructure_excluded_metrics = {
        "schema_version": 1,
        "selection": "tasks_without_terminal_infrastructure_failures",
        "selection_warning": capability_warning,
        "metrics_status": "complete" if capability_complete else "partial",
        "full_selection_metrics_available": capability_complete,
        "total_dataset_tasks": len(tasks),
        "task_count": len(capability_task_indices),
        "observed_task_count": len(observed_capability_task_indices),
        "excluded_infrastructure_tasks": len(infrastructure_task_indices),
        "excluded_infrastructure_task_ids": [
            tasks[task_index].id for task_index in infrastructure_task_indices
        ],
        "excluded_incomplete_tasks": len(capability_task_indices)
        - len(observed_capability_task_indices),
        "observed_attempts": capability["attempts"],
        "success_attempts": capability["success_attempts"],
        "incorrect_attempts": capability["incorrect_attempts"],
        "solved_tasks": capability["solved_tasks"],
        "observed_mean_sample_accuracy": (
            capability_sample_metrics["mean_accuracy"]
            if capability_sample_metrics is not None
            else None
        ),
        "mean_sample_accuracy": (
            capability_sample_metrics["mean_accuracy"]
            if capability_complete and capability_sample_metrics is not None
            else None
        ),
        "observed_average_reward": (
            capability_sample_metrics["average_reward"]
            if capability_sample_metrics is not None
            else None
        ),
        "average_reward": (
            capability_sample_metrics["average_reward"]
            if capability_complete and capability_sample_metrics is not None
            else None
        ),
        "observed_average_steps": (
            capability_sample_metrics["average_steps"]
            if capability_sample_metrics is not None
            else None
        ),
        "average_steps": (
            capability_sample_metrics["average_steps"]
            if capability_complete and capability_sample_metrics is not None
            else None
        ),
        "observed_average_steps_by_outcome": (
            capability_sample_metrics["average_steps_by_outcome"]
            if capability_sample_metrics is not None
            else None
        ),
        "average_steps_by_outcome": (
            capability_sample_metrics["average_steps_by_outcome"]
            if capability_complete and capability_sample_metrics is not None
            else None
        ),
        "observed_average_trajectory_output_tokens": (
            capability_sample_metrics["average_trajectory_output_tokens"]
            if capability_sample_metrics is not None
            else None
        ),
        "average_trajectory_output_tokens": (
            capability_sample_metrics["average_trajectory_output_tokens"]
            if capability_complete and capability_sample_metrics is not None
            else None
        ),
        "observed_average_trajectory_token_length": (
            capability_sample_metrics["average_trajectory_token_length"]
            if capability_sample_metrics is not None
            else None
        ),
        "average_trajectory_token_length": (
            capability_sample_metrics["average_trajectory_token_length"]
            if capability_complete and capability_sample_metrics is not None
            else None
        ),
        "observed_per_sample_metrics": capability["per_sample_metrics"],
        "per_sample_metrics": (
            capability["per_sample_metrics"] if capability_complete else None
        ),
        "observed_pass_at_k": capability["pass_at_k"],
        "pass_at_k": capability["pass_at_k"] if capability_complete else None,
        "observed_score_at_k": capability["score_at_k"],
        "score_at_k": capability["score_at_k"] if capability_complete else None,
    }
    result = {
        "schema_version": 6,
        "semantic_config_id": store.semantic_id,
        "budget": budget,
        "model": model.to_dict(),
        **coverage,
        "metrics_status": "complete" if coverage["complete"] else "partial",
        "full_dataset_metrics_available": bool(coverage["complete"]),
        "selection": "tasks_with_all_attempts",
        "selection_warning": (
            "Infrastructure failures are counted as zero; scheduling completion does not imply complete capability evaluation."
            if coverage["terminal_infrastructure_failure_tasks"] else None
        ) if coverage["complete"]
        else ("Observed pass@k values use only tasks with all requested attempts and may be selection-biased; they are not official full-dataset scores."),
        "observed_task_count": len(observed_task_indices),
        "total_task_count": len(tasks),
        "excluded_incomplete_tasks": len(tasks) - len(observed_task_indices),
        "observed_attempts": observed_attempts,
        "success_attempts": success_attempts,
        "incorrect_attempts": observed_attempts - success_attempts,
        "solved_tasks": sum(count > 0 for count in correct_counts),
        "observed_mean_sample_accuracy": (observed_sample_metrics["mean_accuracy"] if observed_sample_metrics is not None else None),
        "mean_sample_accuracy": (observed_sample_metrics["mean_accuracy"] if coverage["complete"] and observed_sample_metrics is not None else None),
        "observed_full_success_rate": (observed_sample_metrics["mean_accuracy"] if observed_sample_metrics is not None else None),
        "full_success_rate": (observed_sample_metrics["mean_accuracy"] if coverage["complete"] and observed_sample_metrics is not None else None),
        "observed_average_reward": (observed_sample_metrics["average_reward"] if observed_sample_metrics is not None else None),
        "average_reward": (observed_sample_metrics["average_reward"] if coverage["complete"] and observed_sample_metrics is not None else None),
        "observed_average_pass_rate": (observed_sample_metrics["average_reward"] if observed_sample_metrics is not None else None),
        "average_pass_rate": (observed_sample_metrics["average_reward"] if coverage["complete"] and observed_sample_metrics is not None else None),
        "observed_average_steps": (observed_sample_metrics["average_steps"] if observed_sample_metrics is not None else None),
        "average_steps": (observed_sample_metrics["average_steps"] if coverage["complete"] and observed_sample_metrics is not None else None),
        "observed_average_steps_by_outcome": (observed_sample_metrics["average_steps_by_outcome"] if observed_sample_metrics is not None else None),
        "average_steps_by_outcome": (observed_sample_metrics["average_steps_by_outcome"] if coverage["complete"] and observed_sample_metrics is not None else None),
        "observed_average_trajectory_output_tokens": (
            observed_sample_metrics["average_trajectory_output_tokens"]
            if observed_sample_metrics is not None
            else None
        ),
        "average_trajectory_output_tokens": (
            observed_sample_metrics["average_trajectory_output_tokens"]
            if coverage["complete"] and observed_sample_metrics is not None
            else None
        ),
        "observed_average_trajectory_token_length": (
            observed_sample_metrics["average_trajectory_token_length"]
            if observed_sample_metrics is not None
            else None
        ),
        "average_trajectory_token_length": (
            observed_sample_metrics["average_trajectory_token_length"]
            if coverage["complete"] and observed_sample_metrics is not None
            else None
        ),
        "observed_per_sample_metrics": observed_per_sample_metrics,
        "per_sample_metrics": (observed_per_sample_metrics if coverage["complete"] else None),
        "observed_pass_at_k": observed_pass_at,
        "pass_at_k": official_pass_at,
        "observed_score_at_k": observed_score_at,
        "score_at_k": official_score_at,
        "infrastructure_excluded_metrics": infrastructure_excluded_metrics,
        "generated_at": _beijing_timestamp(),
    }
    results_dir = store.models_dir / model.key / "results"
    # The sampling budget belongs in the payload, not the filename.  A later
    # incremental pass@n run atomically replaces this snapshot with metrics
    # for its new total sample budget.
    _atomic_json(results_dir / "metrics.json", result)
    return result


def _summary_csv(results: Sequence[dict[str, Any]], budget: int) -> str:
    from io import StringIO

    output = StringIO()
    pass_columns = [f"pass@{k}" for k in range(1, budget + 1)]
    score_columns = [f"score@{k}" for k in range(1, budget + 1)]
    infrastructure_excluded_pass_columns = [
        f"infrastructure_excluded_pass@{k}" for k in range(1, budget + 1)
    ]
    infrastructure_excluded_score_columns = [
        f"infrastructure_excluded_score@{k}" for k in range(1, budget + 1)
    ]
    fields = [
        "model",
        "step",
        "valid_attempts",
        "expected_attempts",
        "coverage",
        "complete_tasks",
        "task_count",
        "metrics_status",
        "observed_task_count",
        "excluded_incomplete_tasks",
        "success_attempts",
        "incorrect_attempts",
        "solved_tasks",
        "infrastructure_failures",
        "infrastructure_failure_records",
        "terminal_infrastructure_failure_tasks",
        "effective_valid_attempts",
        "capability_complete",
        "verifier_timeouts",
        "mean_sample_accuracy",
        "full_success_rate",
        "average_reward",
        "average_pass_rate",
        "average_steps",
        "average_steps_correct",
        "average_steps_incorrect",
        "average_trajectory_output_tokens",
        "average_trajectory_token_length",
        "observed_mean_sample_accuracy",
        "observed_full_success_rate",
        "observed_average_reward",
        "observed_average_pass_rate",
        "observed_average_steps",
        "observed_average_steps_correct",
        "observed_average_steps_incorrect",
        "observed_average_trajectory_output_tokens",
        "observed_average_trajectory_token_length",
        "infrastructure_excluded_metrics_status",
        "infrastructure_excluded_task_count",
        "infrastructure_excluded_observed_task_count",
        "excluded_infrastructure_tasks",
        "excluded_infrastructure_task_ids",
        "infrastructure_excluded_success_attempts",
        "infrastructure_excluded_incorrect_attempts",
        "infrastructure_excluded_solved_tasks",
        "infrastructure_excluded_mean_sample_accuracy",
        "infrastructure_excluded_average_reward",
        "infrastructure_excluded_average_steps",
        "infrastructure_excluded_average_trajectory_output_tokens",
        "infrastructure_excluded_average_trajectory_token_length",
        *pass_columns,
        *score_columns,
        *infrastructure_excluded_pass_columns,
        *infrastructure_excluded_score_columns,
        *(f"observed_pass@{k}" for k in range(1, budget + 1)),
        *(f"observed_score@{k}" for k in range(1, budget + 1)),
        *(f"observed_infrastructure_excluded_pass@{k}" for k in range(1, budget + 1)),
        *(f"observed_infrastructure_excluded_score@{k}" for k in range(1, budget + 1)),
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for result in results:
        infrastructure_excluded = result.get("infrastructure_excluded_metrics") or {}
        row = {
            "model": result["model"]["key"],
            "step": result["model"]["step"],
            "valid_attempts": result["valid_attempts"],
            "expected_attempts": result["expected_attempts"],
            "coverage": result["coverage"],
            "complete_tasks": result["complete_tasks"],
            "task_count": result["task_count"],
            "metrics_status": result["metrics_status"],
            "observed_task_count": result["observed_task_count"],
            "excluded_incomplete_tasks": result["excluded_incomplete_tasks"],
            "success_attempts": result["success_attempts"],
            "incorrect_attempts": result["incorrect_attempts"],
            "solved_tasks": result["solved_tasks"],
            "infrastructure_failures": result["infrastructure_failures"],
            "infrastructure_failure_records": result[
                "infrastructure_failure_records"
            ],
            "terminal_infrastructure_failure_tasks": result[
                "terminal_infrastructure_failure_tasks"
            ],
            "verifier_timeouts": result["verifier_timeouts"],
            "effective_valid_attempts": result["effective_valid_attempts"],
            "capability_complete": result["capability_complete"],
            "mean_sample_accuracy": result.get("mean_sample_accuracy"),
            "full_success_rate": result.get("full_success_rate"),
            "average_reward": result.get("average_reward"),
            "average_pass_rate": result.get("average_pass_rate"),
            "average_steps": result.get("average_steps"),
            "average_steps_correct": ((result.get("average_steps_by_outcome") or {}).get("correct")),
            "average_steps_incorrect": ((result.get("average_steps_by_outcome") or {}).get("incorrect")),
            "average_trajectory_output_tokens": result.get("average_trajectory_output_tokens"),
            "average_trajectory_token_length": result.get("average_trajectory_token_length"),
            "observed_mean_sample_accuracy": result.get("observed_mean_sample_accuracy"),
            "observed_full_success_rate": result.get("observed_full_success_rate"),
            "observed_average_reward": result.get("observed_average_reward"),
            "observed_average_pass_rate": result.get("observed_average_pass_rate"),
            "observed_average_steps": result.get("observed_average_steps"),
            "observed_average_steps_correct": ((result.get("observed_average_steps_by_outcome") or {}).get("correct")),
            "observed_average_steps_incorrect": ((result.get("observed_average_steps_by_outcome") or {}).get("incorrect")),
            "observed_average_trajectory_output_tokens": result.get("observed_average_trajectory_output_tokens"),
            "observed_average_trajectory_token_length": result.get("observed_average_trajectory_token_length"),
            "infrastructure_excluded_metrics_status": infrastructure_excluded.get("metrics_status"),
            "infrastructure_excluded_task_count": infrastructure_excluded.get("task_count"),
            "infrastructure_excluded_observed_task_count": infrastructure_excluded.get("observed_task_count"),
            "excluded_infrastructure_tasks": infrastructure_excluded.get("excluded_infrastructure_tasks"),
            "excluded_infrastructure_task_ids": "|".join(
                infrastructure_excluded.get("excluded_infrastructure_task_ids") or []
            ),
            "infrastructure_excluded_success_attempts": infrastructure_excluded.get("success_attempts"),
            "infrastructure_excluded_incorrect_attempts": infrastructure_excluded.get("incorrect_attempts"),
            "infrastructure_excluded_solved_tasks": infrastructure_excluded.get("solved_tasks"),
            "infrastructure_excluded_mean_sample_accuracy": infrastructure_excluded.get("mean_sample_accuracy"),
            "infrastructure_excluded_average_reward": infrastructure_excluded.get("average_reward"),
            "infrastructure_excluded_average_steps": infrastructure_excluded.get("average_steps"),
            "infrastructure_excluded_average_trajectory_output_tokens": infrastructure_excluded.get("average_trajectory_output_tokens"),
            "infrastructure_excluded_average_trajectory_token_length": infrastructure_excluded.get("average_trajectory_token_length"),
        }
        official = result.get("pass_at_k") or {}
        observed = result.get("observed_pass_at_k") or {}
        official_score = result.get("score_at_k") or {}
        observed_score = result.get("observed_score_at_k") or {}
        infrastructure_excluded_pass = infrastructure_excluded.get("pass_at_k") or {}
        infrastructure_excluded_score = infrastructure_excluded.get("score_at_k") or {}
        observed_infrastructure_excluded_pass = infrastructure_excluded.get("observed_pass_at_k") or {}
        observed_infrastructure_excluded_score = infrastructure_excluded.get("observed_score_at_k") or {}
        row.update({f"pass@{k}": official.get(str(k), "") for k in range(1, budget + 1)})
        row.update({f"score@{k}": official_score.get(str(k), "") for k in range(1, budget + 1)})
        row.update({f"infrastructure_excluded_pass@{k}": infrastructure_excluded_pass.get(str(k), "") for k in range(1, budget + 1)})
        row.update({f"infrastructure_excluded_score@{k}": infrastructure_excluded_score.get(str(k), "") for k in range(1, budget + 1)})
        row.update({f"observed_pass@{k}": observed.get(str(k), "") for k in range(1, budget + 1)})
        row.update({f"observed_score@{k}": observed_score.get(str(k), "") for k in range(1, budget + 1)})
        row.update({f"observed_infrastructure_excluded_pass@{k}": observed_infrastructure_excluded_pass.get(str(k), "") for k in range(1, budget + 1)})
        row.update({f"observed_infrastructure_excluded_score@{k}": observed_infrastructure_excluded_score.get(str(k), "") for k in range(1, budget + 1)})
        writer.writerow(row)
    return output.getvalue()


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _nested_metric(result: dict[str, Any], path: Sequence[str]) -> float | None:
    value: Any = result
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return _finite_float(value)


def _nice_step(span: float, target_ticks: int = 5) -> float:
    rough = max(span / max(1, target_ticks), 1e-12)
    exponent = math.floor(math.log10(rough))
    fraction = rough / (10**exponent)
    for candidate in (1.0, 2.0, 2.5, 5.0, 10.0):
        if fraction <= candidate:
            return candidate * (10**exponent)
    return 10.0 * (10**exponent)


def _focused_axis(
    values: Sequence[float], *, bounded_unit: bool
) -> tuple[float, float, list[float]]:
    if not values:
        return (0.0, 1.0, [index / 5 for index in range(6)])
    low = min(values)
    high = max(values)
    if bounded_unit:
        low = min(1.0, max(0.0, low))
        high = min(1.0, max(0.0, high))
        span = high - low
        target_span = max(0.1, span * 1.2)
        center = (low + high) / 2
        raw_low = max(0.0, center - target_span / 2)
        raw_high = min(1.0, center + target_span / 2)
        if raw_high - raw_low < target_span - 1e-12:
            if raw_low <= 1e-12:
                raw_high = min(1.0, target_span)
            elif raw_high >= 1.0 - 1e-12:
                raw_low = max(0.0, 1.0 - target_span)
    else:
        if high > low:
            padding = (high - low) * 0.1
        elif high > 0:
            padding = max(high * 0.05, 1.0)
        else:
            padding = 1.0
        raw_low = max(0.0, low - padding)
        raw_high = high + padding
    if raw_high <= raw_low:
        raw_high = raw_low + 1.0
    step = _nice_step(raw_high - raw_low)
    axis_low = math.floor(raw_low / step) * step
    axis_high = math.ceil(raw_high / step) * step
    if bounded_unit:
        axis_low = max(0.0, axis_low)
        axis_high = min(1.0, axis_high)
    else:
        axis_low = max(0.0, axis_low)
    if axis_high <= axis_low:
        axis_high = min(1.0, axis_low + step) if bounded_unit else axis_low + step
    tick_count = max(1, int(round((axis_high - axis_low) / step)))
    ticks = [axis_low + index * step for index in range(tick_count + 1)]
    if ticks[-1] < axis_high - step * 1e-6:
        ticks.append(axis_high)
    return axis_low, axis_high, ticks


def _format_chart_value(value: float, *, percent: bool) -> str:
    if percent:
        return f"{value * 100:.1f}%"
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:,.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _line_summary_svg(
    results: Sequence[dict[str, Any]],
    *,
    title: str,
    subtitle: str,
    series: Sequence[dict[str, Any]],
    bounded_unit: bool,
) -> str:
    width, height = 1100, 640
    left, right, top, bottom = 105, 40, 82, 110
    plot_w = width - left - right
    plot_h = height - top - bottom
    plotted = [
        result
        for result in results
        if any(_nested_metric(result, spec["path"]) is not None for spec in series)
    ]
    values = [
        value
        for result in plotted
        for spec in series
        if (value := _nested_metric(result, spec["path"])) is not None
    ]
    axis_low, axis_high, ticks = _focused_axis(values, bounded_unit=bounded_unit)

    def x_at(index: int) -> float:
        if len(plotted) == 1:
            return left + plot_w / 2
        return left + (plot_w / max(1, len(plotted) - 1)) * index

    def y_at(value: float) -> float:
        return top + (axis_high - value) / (axis_high - axis_low) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" data-y-min="{axis_low:.12g}" data-y-max="{axis_high:.12g}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            "<style>text{font-family:Arial,sans-serif;fill:#222}"
            ".axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}"
            ".partial{stroke-dasharray:7 5}.partial-dot{fill:white}"
            ".complete-dot{fill:currentColor}.coverage{fill:#666}</style>"
        ),
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="22">{html.escape(title)}</text>',
        f'<text x="{width / 2}" y="53" text-anchor="middle" font-size="13" fill="#666">{html.escape(subtitle)}</text>',
    ]
    for tick in ticks:
        y = y_at(tick)
        label = _format_chart_value(tick, percent=bounded_unit)
        lines.append(
            f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}"/>'
        )
        lines.append(
            f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" font-size="13">{html.escape(label)}</text>'
        )
    lines.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}"/>',
        ]
    )

    for series_index, spec in enumerate(series):
        css_class = f"series-{series_index}"
        dash = f"stroke-dasharray:{spec['dash']};" if spec.get("dash") else ""
        lines.append(
            f'<style>.{css_class}{{stroke:{spec["color"]};stroke-width:3;{dash}}}'
            f'.{css_class}-dot{{color:{spec["color"]};stroke:{spec["color"]};stroke-width:3}}</style>'
        )
        previous: tuple[int, float, str] | None = None
        for index, result in enumerate(plotted):
            value = _nested_metric(result, spec["path"])
            if value is None:
                previous = None
                continue
            status = str(result.get("metrics_status", "partial"))
            if previous is not None:
                previous_index, previous_value, previous_status = previous
                partial_class = (
                    " partial" if "partial" in {previous_status, status} else ""
                )
                lines.append(
                    f'<line class="{css_class}{partial_class}" '
                    f'x1="{x_at(previous_index):.1f}" y1="{y_at(previous_value):.1f}" '
                    f'x2="{x_at(index):.1f}" y2="{y_at(value):.1f}"/>'
                )
            dot_class = "complete-dot" if status == "complete" else "partial-dot"
            value_label = _format_chart_value(value, percent=bounded_unit)
            lines.append(
                f'<circle class="{css_class}-dot {dot_class}" cx="{x_at(index):.1f}" '
                f'cy="{y_at(value):.1f}" r="5"><title>{html.escape(spec["label"])}: '
                f'{html.escape(value_label)}</title></circle>'
            )
            previous = (index, value, status)

    for index, result in enumerate(plotted):
        x = x_at(index)
        model = result.get("model") or {}
        label = "base" if model.get("step") == 0 else f"step {model.get('step')}"
        lines.append(
            f'<text x="{x:.1f}" y="{height - bottom + 25}" text-anchor="end" '
            f'font-size="12" transform="rotate(-35 {x:.1f} {height - bottom + 25})">{html.escape(label)}</text>'
        )
        lines.append(
            f'<text class="coverage" x="{x:.1f}" y="{top + 14}" text-anchor="middle" '
            f'font-size="11">{result.get("observed_task_count", 0)}/{result.get("total_task_count", 0)}</text>'
        )

    legend_x = left
    for series_index, spec in enumerate(series):
        css_class = f"series-{series_index}"
        lines.append(
            f'<line class="{css_class}" x1="{legend_x}" y1="{height - 32}" '
            f'x2="{legend_x + 35}" y2="{height - 32}"/>'
        )
        lines.append(
            f'<text x="{legend_x + 45}" y="{height - 27}" font-size="14">{html.escape(spec["label"])}</text>'
        )
        legend_x += 75 + max(135, len(str(spec["label"])) * 8)
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _summary_svg(
    results: Sequence[dict[str, Any]],
    budget: int,
    *,
    title: str = "SWE-Bench Verified Codeflow",
    metric: str = "pass",
) -> str:
    if metric not in {"pass", "score"}:
        raise ValueError(f"unsupported summary metric {metric!r}")
    metric_key = f"observed_{metric}_at_k"
    return _line_summary_svg(
        results,
        title=f"{title} observed {metric}@{budget}",
        subtitle="Focused y-axis based on displayed values; labels show observed task coverage.",
        series=(
            {
                "label": f"observed {metric}@1",
                "path": (metric_key, "1"),
                "color": "#2563eb",
            },
            {
                "label": f"observed {metric}@{budget}",
                "path": (metric_key, str(budget)),
                "color": "#d97706",
                "dash": "9 4",
            },
        ),
        bounded_unit=True,
    )


def _infrastructure_excluded_chart_results(
    results: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    chart_results: list[dict[str, Any]] = []
    for result in results:
        selected = result.get("infrastructure_excluded_metrics") or {}
        chart_results.append(
            {
                "model": result["model"],
                "metrics_status": selected.get("metrics_status", "partial"),
                "observed_task_count": selected.get("observed_task_count", 0),
                "total_task_count": selected.get("task_count", 0),
                "observed_pass_at_k": selected.get("observed_pass_at_k"),
                "observed_score_at_k": selected.get("observed_score_at_k"),
                "observed_average_steps": selected.get("observed_average_steps"),
                "observed_average_steps_by_outcome": selected.get(
                    "observed_average_steps_by_outcome"
                ),
                "observed_average_trajectory_output_tokens": selected.get(
                    "observed_average_trajectory_output_tokens"
                ),
                "observed_average_trajectory_token_length": selected.get(
                    "observed_average_trajectory_token_length"
                ),
            }
        )
    return chart_results


def _write_resource_charts(
    charts_dir: Path,
    results: Sequence[dict[str, Any]],
    *,
    title_prefix: str,
    suffix: str = "",
    selection_label: str = "all terminal attempts",
) -> None:
    charts = (
        (
            f"average_steps{suffix}.svg",
            "Average steps by checkpoint",
            "Steps per trajectory",
            (
                {
                    "label": "overall",
                    "path": ("observed_average_steps",),
                    "color": "#2563eb",
                },
                {
                    "label": "correct",
                    "path": ("observed_average_steps_by_outcome", "correct"),
                    "color": "#d97706",
                    "dash": "9 4",
                },
                {
                    "label": "incorrect",
                    "path": ("observed_average_steps_by_outcome", "incorrect"),
                    "color": "#6b7f2a",
                    "dash": "3 4",
                },
            ),
        ),
        (
            f"average_trajectory_output_tokens{suffix}.svg",
            "Average trajectory output tokens by checkpoint",
            "Generated model tokens per trajectory",
            (
                {
                    "label": "output tokens",
                    "path": ("observed_average_trajectory_output_tokens",),
                    "color": "#2563eb",
                },
            ),
        ),
        (
            f"average_trajectory_token_length{suffix}.svg",
            "Average trajectory token length by checkpoint",
            "Cumulative-prefix-deduplicated full-sequence tokens per trajectory",
            (
                {
                    "label": "trajectory token length",
                    "path": ("observed_average_trajectory_token_length",),
                    "color": "#2563eb",
                },
            ),
        ),
    )
    for filename, title, unit, series in charts:
        _atomic_text(
            charts_dir / filename,
            _line_summary_svg(
                results,
                title=f"{title_prefix}: {title}",
                subtitle=(
                    f"{unit}; {selection_label}; focused y-axis based on displayed values."
                ),
                series=series,
                bounded_unit=False,
            ),
        )


def write_summary(
    store: EvaluationStore,
    models: Sequence[EvaluationModel],
    tasks: Sequence[Task],
    budget: int,
) -> dict[str, Any]:
    """Write JSON/CSV/SVG for complete and explicitly labelled partial models."""

    results: list[dict[str, Any]] = []
    for model in sorted(models, key=lambda item: item.step):
        results.append(
            aggregate_model(
                store,
                model,
                tasks,
                budget,
                allow_partial=True,
            )
        )
    summary = {
        "schema_version": 5,
        "semantic_config_id": store.semantic_id,
        "budget": budget,
        "dataset_tasks": len(tasks),
        "models": results,
        "generated_at": _beijing_timestamp(),
    }
    _atomic_json(store.summaries_dir / "metrics.json", summary)
    _atomic_text(
        store.summaries_dir / "metrics.csv",
        _summary_csv(results, budget),
    )
    dataset_config = store.semantic_config.get("dataset", {})
    benchmark_profile = (
        str(dataset_config.get("benchmark_profile", ""))
        if isinstance(dataset_config, dict)
        else ""
    )
    repo_generation = benchmark_profile in {"nl2repo", "doc2repo"}
    if benchmark_profile == "nl2repo":
        chart_title = "NL2Repo-Bench Codeflow"
    elif benchmark_profile == "doc2repo":
        chart_title = "BeyondSWE Doc2Repo Codeflow"
    elif benchmark_profile == "swebench_pro_public":
        chart_title = "SWE-bench Pro Public Codeflow"
    else:
        chart_title = "SWE-Bench Verified Codeflow"
    charts_dir = store.summaries_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    for legacy in (
        store.summaries_dir / "performance.svg",
        store.summaries_dir / "performance_infrastructure_excluded.svg",
    ):
        legacy.unlink(missing_ok=True)
    _atomic_text(
        charts_dir / "performance.svg",
        _summary_svg(
            results,
            budget,
            title=chart_title,
            metric="score" if repo_generation else "pass",
        ),
    )
    infrastructure_excluded_chart_results = _infrastructure_excluded_chart_results(
        results
    )
    _atomic_text(
        charts_dir / "performance_infrastructure_excluded.svg",
        _summary_svg(
            infrastructure_excluded_chart_results,
            budget,
            title=f"{chart_title} (infrastructure failures excluded)",
            metric="score" if repo_generation else "pass",
        ),
    )
    _write_resource_charts(
        charts_dir,
        results,
        title_prefix=chart_title,
    )
    _write_resource_charts(
        charts_dir,
        infrastructure_excluded_chart_results,
        title_prefix=chart_title,
        suffix="_infrastructure_excluded",
        selection_label="tasks without terminal infrastructure failures",
    )
    return summary


__all__ = [
    "AttemptSpec",
    "EvaluationModel",
    "EvaluationStore",
    "aggregate_model",
    "discover_evaluation_models",
    "episode_to_rollout_payload",
    "fingerprint_hf_model",
    "infrastructure_failure_reason",
    "is_infrastructure_failure",
    "semantic_config_id",
    "task_directory_name",
    "unbiased_pass_at_k",
    "validate_hf_model",
    "write_summary",
]
