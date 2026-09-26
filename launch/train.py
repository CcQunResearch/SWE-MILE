from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from omegaconf import DictConfig, OmegaConf

from rllm.data.repo_generation_exclusions import (
    filter_inaccessible_repo_generation_tasks,
)
from rllm.data.swerebench_v2_builder import (
    PROMATCHED_LANGUAGES,
    PROMATCHED_ROWS_AT_DEFAULT_REVISION,
    PYTHON_ROWS_AT_DEFAULT_REVISION,
    SOURCE_ROWS_AT_DEFAULT_REVISION,
)
from rllm.data.swerebench_v2_filternorm import (
    FILTERNORM_DATASET_NAME,
    filter_filternorm_rows,
)
from rllm.data.task_eligibility import (
    filter_tasks_with_eligibility_manifest,
    filter_tasks_with_image_availability_manifest,
)
from rllm.harnesses.codeflow_shell_policy import (
    normalize_codeflow_tool_mode,
    normalize_codeflow_tool_restricted_mode,
)
from rllm.sandbox.backends.minisandbox import MINISANDBOX_BACKEND_REVISION

if TYPE_CHECKING:
    from rllm.data import Dataset


logger = logging.getLogger(__name__)

_MODEL_ROOT = os.environ.get("MODEL_ROOT", "<path/to/models>")
_MODEL_PROFILES = {
    "qwen35_9b": f"{_MODEL_ROOT}/Qwen3.5-9B",
    "qwen36_35b_a3b": f"{_MODEL_ROOT}/Qwen3.6-35B-A3B",
    "qwen38_27b": f"{_MODEL_ROOT}/Qwen3.8-27B",
}
_REASONING_EFFORTS = frozenset({"low", "medium", "xhigh"})
_QWEN_DENSE_TYPES = frozenset({"qwen3_5", "qwen3_5_text"})
_QWEN_MOE_TYPES = frozenset({"qwen3_5_moe", "qwen3_5_moe_text"})


def _model_artifacts(model_path: str) -> tuple[dict, str | None]:
    """Read architecture and native template without loading model weights."""
    path = Path(model_path).expanduser()
    config_path = path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    template_path = path / "chat_template.jinja"
    tokenizer_path = path / "tokenizer_config.json"
    template = None
    if template_path.is_file():
        template = template_path.read_text(encoding="utf-8")
    elif tokenizer_path.is_file():
        template = json.loads(tokenizer_path.read_text(encoding="utf-8")).get("chat_template")
    return config, template


def _model_profile_for_path(model_path: str) -> str:
    """Prefer local architecture/template capabilities over cosmetic names.

    Qwen3.6 shares the Qwen3.5 MoE architecture; its preserved-thinking
    template survives HF export and distinguishes renamed checkpoints.
    """
    config, template = _model_artifacts(model_path)
    model_type = config.get("model_type")
    if config and model_type not in _QWEN_DENSE_TYPES | _QWEN_MOE_TYPES:
        return "qwen35_9b"  # Preserve the existing non-Qwen caller fallback.
    if template is not None and config:
        if model_type in _QWEN_DENSE_TYPES:
            if isinstance(template, str) and all(word in template for word in ("reasoning_effort", "xhigh", "medium", "low")):
                return "qwen38_27b"
            return "qwen35_9b"
        text_config = config.get("text_config", config)
        if (isinstance(template, str) and "preserve_thinking" in template and "reasoning_effort" not in template
                and text_config.get("num_attention_heads") == 16
                and text_config.get("num_experts") == 256):
            return "qwen36_35b_a3b"
        return "qwen35_moe"
    # Canonical identifiers also work before the checkpoint is mounted locally.
    name = str(config.get("_name_or_path") or model_path).rstrip("/")
    if model_type in _QWEN_MOE_TYPES or not config:
        if re.search(r"(?:^|/)qwen3[._-]6[-_]35b[-_]a3b(?:$|/)", name, flags=re.IGNORECASE):
            return "qwen36_35b_a3b"
    if model_type in _QWEN_DENSE_TYPES or not config:
        if re.search(r"(?:^|/)qwen3[._-]8[-_]27b(?:$|/)", name, flags=re.IGNORECASE):
            return "qwen38_27b"
    return "qwen35_moe" if model_type in _QWEN_MOE_TYPES else "qwen35_9b"


def _is_qwen38_model(model_path: str) -> bool:
    return _model_profile_for_path(model_path) == "qwen38_27b"


def _qwen_text_config(model_path: str) -> dict | None:
    config, _ = _model_artifacts(model_path)
    if config:
        return config.get("text_config", config) if config.get("model_type") in _QWEN_DENSE_TYPES | _QWEN_MOE_TYPES else None
    if re.search(r"(?:^|/)qwen3[._-][568][-_]", model_path, flags=re.IGNORECASE):
        return {}  # World-size checks still apply when local metadata is absent.
    return None


def _configure_model_profile(
    config: DictConfig,
    *,
    evaluation: bool = False,
    explicit_overrides: Sequence[str] | None = None,
) -> str:
    """Infer model behavior from the effective checkpoint, never set SP.

    Hydra's backend supplies an unrelated default model. Only a task override
    makes that path explicit; plain DictConfigs from library callers already
    contain their intended paths and are respected as-is.
    """
    from hydra.core.hydra_config import HydraConfig

    if explicit_overrides is None and HydraConfig.initialized():
        explicit_overrides = list(HydraConfig.get().overrides.task)
    explicit_keys = (
        {item.split("=", 1)[0].lstrip("+") for item in explicit_overrides}
        if explicit_overrides is not None else None
    )

    def get(key: str, default=None):
        return OmegaConf.select(config, key, default=default)

    def set_value(key: str, value) -> None:
        OmegaConf.update(config, key, value, merge=False, force_add=True)

    def explicit_path(key: str):
        return get(key) if explicit_keys is None or key in explicit_keys else None

    # Preserve the Python evaluation profile fallback for existing callers;
    # an explicit checkpoint path always determines the actual model identity.
    fallback_profile = get("swe.model_profile", "qwen35_9b") if evaluation else "qwen35_9b"
    fallback_path = _MODEL_PROFILES.get(str(fallback_profile), _MODEL_PROFILES["qwen35_9b"])
    model_path = explicit_path("actor_rollout_ref.model.path") or os.environ.get("MODEL_PATH") or fallback_path
    if evaluation:
        model_path = explicit_path("eval.base_model_path") or model_path
    if "<" in str(model_path):
        raise ValueError("Set MODEL_PATH or actor_rollout_ref.model.path to a model directory")
    profile = _model_profile_for_path(str(model_path))
    effort = get("swe.reasoning_effort")
    if profile != "qwen38_27b" and effort is not None:
        raise ValueError("swe.reasoning_effort requires a Qwen3.8-27B checkpoint at the configured model path")
    if profile == "qwen36_35b_a3b":
        # The shell's historical qwen3.5 renderer is a default. Qwen3.6 adds
        # native historical-thinking and JSON tool-argument behavior.
        family = get("rllm.gateway.renderer_family", "auto")
        if family not in {"auto", "qwen3.5", "qwen3.6"}:
            raise ValueError(f"Qwen3.6 requires rllm.gateway.renderer_family=qwen3.6, got {family!r}")
        set_value("rllm.gateway.renderer_family", "qwen3.6")
        for split in ("train", "val"):
            key = f"rllm.rollout.{split}.chat_template_kwargs"
            kwargs = OmegaConf.to_container(get(key), resolve=True) if get(key) is not None else {}
            if not isinstance(kwargs, dict):
                raise ValueError(f"{key} must be a mapping")
            if "reasoning_effort" in kwargs:
                raise ValueError(f"{key}.reasoning_effort is not supported by Qwen3.6; use enable_thinking instead")
            for control in ("enable_thinking", "preserve_thinking"):
                if control in kwargs and not isinstance(kwargs[control], bool):
                    raise ValueError(f"{key}.{control} must be a boolean")
            # Leave native defaults in the tokenizer: thinking on, historical
            # user-message thinking off. Explicit split controls stay intact.
    if profile == "qwen38_27b":
        effort = effort if effort is not None else "medium"
        if not isinstance(effort, str) or effort not in _REASONING_EFFORTS:
            raise ValueError("swe.reasoning_effort must be low, medium, or xhigh")
        # Gateway sessions enforce these fields on every chat request. Keep
        # template controls out of vLLM's token-in SamplingParams API.
        for split in ("train", "val"):
            key = f"rllm.rollout.{split}.chat_template_kwargs"
            kwargs = OmegaConf.to_container(get(key), resolve=True) if get(key) is not None else {}
            if not isinstance(kwargs, dict):
                raise ValueError(f"{key} must be a mapping")
            kwargs.setdefault("reasoning_effort", effort)
            if not isinstance(kwargs["reasoning_effort"], str) or kwargs["reasoning_effort"] not in _REASONING_EFFORTS:
                raise ValueError(f"{key}.reasoning_effort must be low, medium, or xhigh")
            set_value(key, kwargs)
        set_value("swe.reasoning_effort", effort)
    set_value("actor_rollout_ref.model.path", str(model_path))
    if not explicit_path("model.name"):
        set_value("model.name", str(model_path))
    if evaluation and not explicit_path("eval.base_model_path"):
        set_value("eval.base_model_path", str(model_path))
    set_value("swe.model_profile", profile)
    logger.info("SWE model profile=%s path=%s reasoning_effort=%s", profile, model_path, effort)
    return profile


def _validate_model_sequence_parallelism(config: DictConfig) -> None:
    """Catch incompatible head partitioning before allocating training workers."""
    model_path = OmegaConf.select(config, "actor_rollout_ref.model.path")
    text_config = _qwen_text_config(str(model_path))
    if text_config is None:
        return
    nodes = OmegaConf.select(config, "trainer.nnodes")
    gpus_per_node = OmegaConf.select(config, "trainer.n_gpus_per_node")
    world_size = int(nodes) * int(gpus_per_node) if nodes is not None and gpus_per_node is not None else None
    heads = text_config.get("num_attention_heads")
    kv_heads = text_config.get("num_key_value_heads")
    for worker in ("actor", "ref"):
        # VERL FSDPActorConfig.__post_init__ copies a flat SP > 1 into
        # fsdp_config, including ref's inherited flat SP / default engine SP=1.
        flat_size = OmegaConf.select(config, f"actor_rollout_ref.{worker}.ulysses_sequence_parallel_size")
        fields = ("ulysses_sequence_parallel_size", "fsdp_config.ulysses_sequence_parallel_size")
        if isinstance(flat_size, int) and not isinstance(flat_size, bool) and flat_size > 1:
            fields = fields[:1]
        for field in fields:
            key = f"actor_rollout_ref.{worker}.{field}"
            size = OmegaConf.select(config, key, default=None)
            if size is None:
                continue
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise ValueError(f"{key}={size!r} must be a positive integer")
            if heads is not None and kv_heads is not None and (heads % size or (kv_heads % size and size % kv_heads)):
                raise ValueError(
                    f"{key}={size} is incompatible with {heads} attention heads and {kv_heads} KV heads "
                    f"in {model_path}; SP must divide the attention heads and the trainer GPU count, "
                    "and either divide or be a multiple of the KV head count."
                )
            if world_size is not None and (world_size <= 0 or world_size % size):
                raise ValueError(
                    f"{key}={size} must divide trainer.nnodes * trainer.n_gpus_per_node "
                    f"({nodes} * {gpus_per_node} = {world_size} training GPUs). "
                    "Rollout GPUs do not belong to this Ulysses mesh. "
                    "Explicitly choose compatible trainer resources and SP; no settings were changed."
                )


def _validate_model_tensor_parallelism(config: DictConfig, *, model_path: str | None = None) -> None:
    """Check the vLLM attention/GDN/MoE partitions without tuning resources."""
    model_path = model_path or str(OmegaConf.select(config, "actor_rollout_ref.model.path"))
    text_config = _qwen_text_config(model_path)
    if text_config is None:
        return
    key = "actor_rollout_ref.rollout.tensor_model_parallel_size"
    size = OmegaConf.select(config, key, default=1)
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"{key}={size!r} must be a positive integer")
    for field in ("num_attention_heads", "linear_num_key_heads", "linear_num_value_heads",
                  "moe_intermediate_size", "shared_expert_intermediate_size"):
        value = text_config.get(field)
        if value is not None and value % size:
            raise ValueError(f"{key}={size} must divide {field}={value} in {model_path}; no settings were changed")
    kv_heads = text_config.get("num_key_value_heads")
    if kv_heads is not None and kv_heads % size and size % kv_heads:
        raise ValueError(f"{key}={size} must divide or be a multiple of num_key_value_heads={kv_heads} in {model_path}")
    experts = text_config.get("num_experts")
    if experts is not None and size > experts:
        raise ValueError(f"{key}={size} must not exceed num_experts={experts} in {model_path}")


_SWEREBENCH_FULL_DATASET = "swe-rebench-v2-filtered-verified"
_SWEREBENCH_FULL_ROWS = SOURCE_ROWS_AT_DEFAULT_REVISION
_SWEREBENCH_VIRTUAL_VIEWS: dict[str, tuple[frozenset[str], int]] = {
    FILTERNORM_DATASET_NAME: (
        frozenset({"python"}),
        PYTHON_ROWS_AT_DEFAULT_REVISION,
    ),
    "swe-rebench-v2-filtered-verified-python": (
        frozenset({"python"}),
        PYTHON_ROWS_AT_DEFAULT_REVISION,
    ),
    "swe-rebench-v2-filtered-verified-promatched": (
        PROMATCHED_LANGUAGES,
        PROMATCHED_ROWS_AT_DEFAULT_REVISION,
    ),
}

_MINISANDBOX_TRAINING_PROFILES = {
    FILTERNORM_DATASET_NAME: "bug_repair",
    "denovoswe": "denovoswe",
    "swe-rebench-v2-filtered-verified-python": "bug_repair",
    "swe-rebench-v2-filtered-verified-promatched": "bug_repair",
    "r2egym": "bug_repair",
}


def _configure_minisandbox_training(
    config: DictConfig,
    *,
    train_name: object,
    val_name: object,
    train_split: object | None = None,
    val_split: object | None = None,
) -> str:
    backend = str(
        OmegaConf.select(config, "swe.sandbox_backend", default="minisandbox")
    ).strip().lower()
    if backend == "k8s":
        from rllm.sandbox.backends.k8s import check_k8s_adapter
        check_k8s_adapter()
        return backend
    if backend != "minisandbox":
        raise ValueError("swe.sandbox_backend must be minisandbox or k8s")
    train_dataset = str(train_name)
    val_dataset = str(val_name)
    expected_profile = _MINISANDBOX_TRAINING_PROFILES.get(train_dataset)
    errors: list[str] = []
    if expected_profile is None:
        errors.append(
            "swe.train_dataset must be denovoswe, "
            "swe-rebench-v2-filtered-verified-python-filternorm, "
            "swe-rebench-v2-filtered-verified-python, "
            "swe-rebench-v2-filtered-verified-promatched, or r2egym"
        )
    if val_dataset != train_dataset:
        errors.append("swe.val_dataset must match swe.train_dataset")
    if (
        train_split is not None
        and val_split is not None
        and str(train_split) != str(val_split)
    ):
        errors.append("swe.val_split must match swe.train_split")
    task_profile = os.environ.get("TASK_PROFILE")
    if expected_profile is not None and task_profile != expected_profile:
        errors.append(f"TASK_PROFILE must be {expected_profile}")
    if errors:
        raise ValueError(
            "invalid MiniSandbox training configuration: " + "; ".join(errors)
        )
    data_root = os.environ.get("DATA_ROOT")
    shared_cache_dir = OmegaConf.select(
        config, "swe.minisandbox.shared_cache_dir", default=None
    )
    if shared_cache_dir in (None, ""):
        if not data_root:
            raise ValueError(
                "DATA_ROOT is required to derive the default "
                "swe.minisandbox.shared_cache_dir"
            )
        from rllm.data.minisandbox_cache import default_minisandbox_cache_dir

        shared_cache_dir = str(
            default_minisandbox_cache_dir(data_root, train_dataset)
        )
        OmegaConf.update(
            config,
            "swe.minisandbox.shared_cache_dir",
            shared_cache_dir,
            merge=False,
            force_add=True,
        )
    OmegaConf.update(
        config,
        "swe.minisandbox.dataset",
        train_dataset,
        merge=False,
        force_add=True,
    )
    local_root = OmegaConf.select(
        config, "swe.minisandbox.local_root", default=None
    )
    if local_root in (None, ""):
        OmegaConf.update(
            config,
            "swe.minisandbox.local_root",
            "/tmp/swe-minisandbox",
            merge=False,
            force_add=True,
        )
    return backend


def _validate_minisandbox_task_contract(
    cache_dir: str | Path,
    train_dataset: Dataset,
    val_dataset: Dataset,
    *,
    dataset_name: str = "denovoswe",
) -> dict[str, dict[str, object]]:
    from rllm.data.minisandbox_cache import validate_minisandbox_oci_cache
    from rllm.types import RolloutInfrastructureError

    tasks_by_id = {
        task.id: task for task in [*train_dataset.data, *val_dataset.data]
    }
    from rllm.sandbox.verifier_assets import validate_training_verifier_assets

    for task in tasks_by_id.values():
        validate_training_verifier_assets(task, dataset_name)
    try:
        records = validate_minisandbox_oci_cache(
            cache_dir,
            tasks_by_id.values(),
            dataset=dataset_name,
        )
        if len(records) != len(tasks_by_id):
            raise ValueError(
                "MiniSandbox OCI cache does not exactly cover the scheduled tasks"
            )
    except RolloutInfrastructureError:
        raise
    except Exception as exc:
        raise RolloutInfrastructureError(
            "minisandbox_cache_invalid",
            f"MiniSandbox training cache preflight failed: {exc}",
            retryable=False,
            stage="preflight",
            retry_scope="full_rollout",
        ) from exc
    logger.info(
        "Validated %s MiniSandbox OCI cache for %d scheduled task(s): %s",
        dataset_name,
        len(records),
        Path(cache_dir).expanduser().resolve(),
    )
    return records


def _runtime_code_provenance() -> dict[str, object]:
    """Return a reproducible driver/source identity for postmortem audits."""
    repo_root = Path(__file__).resolve().parents[1]

    def git(*args: str) -> bytes:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            return b""

    commit = git("rev-parse", "HEAD").decode("utf-8", "replace").strip()
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    diff = git("diff", "--no-ext-diff", "--binary", "HEAD")
    git_available = bool(commit)
    dirty = bool(status) if git_available else None
    dirty_hasher = hashlib.sha256()
    dirty_hasher.update(status)
    dirty_hasher.update(b"\0")
    dirty_hasher.update(diff)
    # ``git diff HEAD`` excludes untracked source.  Several SWE components may
    # intentionally be developed as new files in a dirty worktree, so hashing
    # only their names would let two materially different runtimes claim the
    # same provenance.  Include source-like untracked contents deterministically.
    raw_untracked = git("ls-files", "--others", "--exclude-standard", "-z")
    source_suffixes = {".py", ".sh", ".toml", ".yaml", ".yml"}
    for raw_path in sorted(filter(None, raw_untracked.split(b"\0"))):
        relative = raw_path.decode("utf-8", "surrogateescape")
        candidate = repo_root / relative
        if not (relative == "Dockerfile" or candidate.suffix in source_suffixes):
            continue
        try:
            contents = candidate.read_bytes()
        except OSError:
            continue
        dirty_hasher.update(b"\0untracked\0")
        dirty_hasher.update(raw_path)
        dirty_hasher.update(b"\0")
        dirty_hasher.update(contents)
    source_hasher = hashlib.sha256()
    source_suffixes = {".py", ".sh", ".toml", ".yaml", ".yml"}
    excluded_parts = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "reports",
    }
    source_files = sorted(
        path
        for path in repo_root.rglob("*")
        if path.is_file()
        and not excluded_parts.intersection(path.relative_to(repo_root).parts)
        and (path.name == "Dockerfile" or path.suffix in source_suffixes)
    )
    for candidate in source_files:
        relative = candidate.relative_to(repo_root).as_posix()
        try:
            contents = candidate.read_bytes()
        except OSError:
            continue
        source_hasher.update(relative.encode("utf-8", "surrogateescape"))
        source_hasher.update(b"\0")
        source_hasher.update(contents)
        source_hasher.update(b"\0")
    return {
        "schema_version": 1,
        "git_commit": commit or None,
        "git_available": git_available,
        "dirty": dirty,
        "dirty_digest": (dirty_hasher.hexdigest() if dirty is True else None),
        "source_tree_digest": source_hasher.hexdigest(),
        "source_file_count": len(source_files),
        "repo_root": str(repo_root),
        "minisandbox_backend_revision": MINISANDBOX_BACKEND_REVISION,
    }


def _resolved_codeflow_tool_restricted_mode(
    config: DictConfig,
    *,
    tool_mode: str | None = None,
) -> tuple[str, ...]:
    configured = OmegaConf.select(
        config,
        "swe.codeflow_tool_restricted_mode",
        default=None,
    )
    mode = normalize_codeflow_tool_restricted_mode(
        [] if configured is None and tool_mode == "bash_only" else configured
    )
    OmegaConf.update(
        config,
        "swe.codeflow_tool_restricted_mode",
        list(mode),
        merge=False,
        force_add=True,
    )
    return mode


def _resolved_codeflow_tool_mode(config: DictConfig) -> str:
    mode = normalize_codeflow_tool_mode(
        OmegaConf.select(config, "swe.codeflow_tool_mode", default=None)
    )
    OmegaConf.update(
        config,
        "swe.codeflow_tool_mode",
        mode,
        merge=False,
        force_add=True,
    )
    return mode


def _validate_codeflow_tool_configuration(
    tool_mode: str,
    restricted_mode: tuple[str, ...],
) -> None:
    if tool_mode == "bash_only" and restricted_mode:
        raise ValueError(
            "swe.codeflow_tool_mode=bash_only requires "
            "swe.codeflow_tool_restricted_mode=[]"
        )


def _reject_removed_rollout_log_probability(config: DictConfig) -> None:
    swe = OmegaConf.select(config, "swe", default=None)
    if swe is not None and "rollout_log_prob" in swe:
        raise ValueError("swe.rollout_log_prob was removed; every rollout selected for persistence is now logged")


def _training_sandbox_resources(config: DictConfig) -> tuple[int, int, int, int]:
    swe = OmegaConf.select(config, "swe", default=None)
    for legacy, replacements in {
        "sandbox_cpus": "primary_sandbox_cpus/shadow_sandbox_cpus",
        "sandbox_memory_mb": "primary_sandbox_memory_mb/shadow_sandbox_memory_mb",
    }.items():
        if swe is not None and legacy in swe:
            raise ValueError(
                f"swe.{legacy} was removed for training; use swe.{replacements}"
            )

    names = (
        "primary_sandbox_cpus",
        "primary_sandbox_memory_mb",
        "shadow_sandbox_cpus",
        "shadow_sandbox_memory_mb",
    )
    values: list[int] = []
    for name in names:
        raw_value = OmegaConf.select(config, f"swe.{name}", default=None)
        if isinstance(raw_value, bool) or not isinstance(raw_value, int) or raw_value <= 0:
            raise ValueError(f"swe.{name} must be a positive integer, got {raw_value!r}")
        values.append(raw_value)
    return values[0], values[1], values[2], values[3]


def _save_launcher_backup(source: Path, destination: Path) -> None:
    """Preserve the first launcher copy without overwriting an existing backup."""
    if destination.exists():
        return
    contents = source.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as file:
            file.write(contents)
    except FileExistsError:
        pass


def _save_training_params(config: DictConfig) -> Path:
    provenance = _runtime_code_provenance()
    logger.info(
        "Runtime code provenance: %s",
        json.dumps(provenance, sort_keys=True),
    )
    os.environ["RLLM_RUNTIME_CODE_PROVENANCE"] = json.dumps(
        provenance,
        sort_keys=True,
    )
    explicit_path = os.environ.get("RLLM_TRAINING_PARAMS_PATH")
    if explicit_path:
        output_path = Path(explicit_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        params = OmegaConf.to_container(config, resolve=True, enum_to_str=True)
        if isinstance(params, dict):
            params["runtime_code_provenance"] = provenance
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(params, file, ensure_ascii=False, indent=2)
            file.write("\n")
        return output_path

    run_log_dir = OmegaConf.select(config, "rllm.trainer.wandb_dir", default=None)
    if not run_log_dir:
        raise ValueError("rllm.trainer.wandb_dir must be set to RUN_LOG_DIR so the training parameters can be saved")

    output_dir = Path(str(run_log_dir)).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "training_params.json"
    params = OmegaConf.to_container(config, resolve=True, enum_to_str=True)
    if isinstance(params, dict):
        params["runtime_code_provenance"] = provenance
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(params, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return output_path


def _to_unit_float(value, *, name: str = "value") -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a float in [0.0, 1.0], got {value!r}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a float in [0.0, 1.0], got {value!r}") from exc
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be a float in [0.0, 1.0], got {value!r}")
    return parsed


def _to_positive_float(value, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive number, got {value!r}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number, got {value!r}") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be a positive number, got {value!r}")
    return parsed


def _shadow_finalize_stall_timeout(config: DictConfig) -> float:
    return _to_positive_float(
        OmegaConf.select(config, "swe.shadow_finalize_stall_timeout", default=600.0),
        name="swe.shadow_finalize_stall_timeout",
    )


def _shadow_verification_recovery_settings(
    config: DictConfig,
) -> tuple[bool, float, int]:
    recovery = OmegaConf.select(
        config,
        "swe.shadow_partition_timeout_recovery",
        default=True,
    )
    if not isinstance(recovery, bool):
        raise ValueError(
            "swe.shadow_partition_timeout_recovery must be a boolean"
        )
    timeout = _to_positive_float(
        OmegaConf.select(
            config,
            "swe.shadow_partition_recovery_command_timeout",
            default=300,
        ),
        name="swe.shadow_partition_recovery_command_timeout",
    )
    raw_max_commands = OmegaConf.select(
        config,
        "swe.shadow_partition_recovery_max_commands",
        default=6,
    )
    if (
        isinstance(raw_max_commands, bool)
        or not isinstance(raw_max_commands, int)
        or raw_max_commands <= 0
    ):
        raise ValueError(
            "swe.shadow_partition_recovery_max_commands must be a positive integer"
        )
    return recovery, timeout, raw_max_commands


def _bug_repair_verification_potential_mode(config: DictConfig) -> str:
    mode = str(
        OmegaConf.select(
            config,
            "swe.bug_repair_verification_potential_mode",
            default="normalized",
        )
    ).strip().casefold()
    if mode not in {"pass_count", "normalized"}:
        raise ValueError(
            "swe.bug_repair_verification_potential_mode must be pass_count or normalized"
        )
    return mode


def _limit_termination_success_reward(config: DictConfig) -> float | str:
    removed_reward = OmegaConf.select(
        config,
        "swe.non_submit_success_reward",
        default=None,
    )
    if removed_reward is not None:
        raise ValueError("swe.non_submit_success_reward was removed; use swe.limit_termination_success_reward instead")
    value = OmegaConf.select(
        config,
        "swe.limit_termination_success_reward",
        default=0.6,
    )
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    return _to_unit_float(value, name="swe.limit_termination_success_reward")


def _limit_termination_outcome_mode(config: DictConfig) -> str:
    value = str(
        OmegaConf.select(
            config,
            "swe.limit_termination_outcome_mode",
            default="discount_success",
        )
    ).strip()
    if value not in {"discount_success", "verifier_outcome"}:
        raise ValueError(f"swe.limit_termination_outcome_mode must be 'discount_success' or 'verifier_outcome', got {value!r}")
    return value


def _load_sandbox_dataset(name: str, split: str) -> Dataset:
    from rllm.data import Dataset, DatasetRegistry
    from rllm.data.dataset import _wrap_rows_as_tasks

    view = _SWEREBENCH_VIRTUAL_VIEWS.get(name)
    physical_name = _SWEREBENCH_FULL_DATASET if view is not None else name
    dataset = DatasetRegistry.load_dataset(physical_name, split)
    if dataset is None:
        if view is not None:
            raise RuntimeError(
                f"Virtual dataset '{name}' requires registered source "
                f"'{_SWEREBENCH_FULL_DATASET}' split '{split}'. Materialize only "
                "the complete SWE-rebench V2 dataset with "
                "bash launch/prepare_dataset.sh."
            )
        raise RuntimeError(f"Dataset '{name}' split '{split}' is not registered. Materialize it first, for example: rllm dataset pull {name}")

    rows = list(dataset.data)
    if view is not None:
        languages, expected_rows = view
        if len(rows) != _SWEREBENCH_FULL_ROWS:
            raise RuntimeError(
                f"Virtual dataset '{name}' expected {_SWEREBENCH_FULL_ROWS} rows "
                f"in source '{_SWEREBENCH_FULL_DATASET}/{split}', got {len(rows)}. "
                "Re-materialize the complete pinned dataset."
            )
        source_ids: list[str] = []
        normalized_languages: list[str] = []
        task_parents: set[Path] = set()
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise RuntimeError(
                    f"Virtual dataset source row {index} is not a mapping"
                )
            task_id = str(row.get("id") or "").strip()
            raw_language = row.get("language")
            if not isinstance(raw_language, str):
                raise RuntimeError(
                    f"Virtual dataset source task '{task_id or index}' has an "
                    f"invalid language: {raw_language!r}"
                )
            language = raw_language.strip().casefold()
            if not task_id:
                raise RuntimeError(
                    f"Virtual dataset source row {index} has an empty id"
                )
            if not language:
                raise RuntimeError(
                    f"Virtual dataset source task '{task_id}' has an empty language"
                )
            raw_task_path = row.get("task_path")
            if not isinstance(raw_task_path, str) or not raw_task_path.strip():
                raise RuntimeError(
                    f"Virtual dataset source task '{task_id}' has an invalid task_path"
                )
            # Normalize lexically: Path.resolve() can issue one shared-filesystem metadata
            # request per source row before the language filter has run.
            task_parent = Path(
                os.path.abspath(os.path.expanduser(raw_task_path))
            ).parent
            if task_parent.name != _SWEREBENCH_FULL_DATASET:
                raise RuntimeError(
                    f"Virtual dataset source task '{task_id}' does not point to the "
                    f"complete task directory '{_SWEREBENCH_FULL_DATASET}': "
                    f"{raw_task_path}"
                )
            source_ids.append(task_id)
            normalized_languages.append(language)
            task_parents.add(task_parent)
        if len(source_ids) != len(set(source_ids)):
            raise RuntimeError(
                f"Virtual dataset source '{_SWEREBENCH_FULL_DATASET}/{split}' "
                "contains duplicate task ids"
            )
        if len(task_parents) != 1:
            raise RuntimeError(
                f"Virtual dataset source '{_SWEREBENCH_FULL_DATASET}/{split}' "
                "contains inconsistent task roots"
            )
        rows = [
            row
            for row, language in zip(rows, normalized_languages, strict=True)
            if language in languages
        ]
        if not rows:
            raise RuntimeError(
                f"Virtual dataset '{name}' selected no rows for languages "
                f"{sorted(languages)}"
            )
        if len(rows) != expected_rows:
            raise RuntimeError(
                f"Virtual dataset '{name}' expected {expected_rows} rows for "
                f"languages {sorted(languages)}, got {len(rows)}. Re-materialize "
                "the complete pinned dataset."
            )
        if name == FILTERNORM_DATASET_NAME:
            rows = filter_filternorm_rows(rows)
        print(
            f"Resolved virtual dataset {name}/{split} -> "
            f"{_SWEREBENCH_FULL_DATASET}/{split}: "
            f"languages={','.join(sorted(languages))} "
            f"rows={len(rows)}/{_SWEREBENCH_FULL_ROWS}",
            flush=True,
        )
    if rows and isinstance(rows[0], dict) and rows[0].get("task_path"):
        try:
            workers = int(os.environ.get("RLLM_DATASET_LOAD_WORKERS", "16"))
        except ValueError as exc:
            raise ValueError("RLLM_DATASET_LOAD_WORKERS must be an integer in [1, 64]") from exc
        if not 1 <= workers <= 64:
            raise ValueError("RLLM_DATASET_LOAD_WORKERS must be an integer in [1, 64]")
        # Each materialized sandbox row merges one small task.toml. Serial
        # metadata reads take several minutes on shared-filesystem and leave the allocated
        # GPU cluster idle; bounded threads preserve row order while hiding
        # remote metadata latency.
        if workers == 1 or len(rows) == 1:
            rows = _wrap_rows_as_tasks(rows)
        else:
            with ThreadPoolExecutor(max_workers=min(workers, len(rows)), thread_name_prefix="swe-dataset") as pool:
                rows = list(pool.map(lambda row: _wrap_rows_as_tasks([row])[0], rows))
    return Dataset(data=rows, name=name, split=split)


def _override_sandbox_resources(dataset: Dataset, cpus: int | None, memory_mb: int | None) -> None:
    if cpus is None and memory_mb is None:
        return

    for item in dataset.data:
        metadata = item.metadata if hasattr(item, "metadata") else item
        if not isinstance(metadata, dict):
            continue
        environment = dict(metadata.get("environment") or {})
        if cpus is not None:
            environment["cpus"] = int(cpus)
        if memory_mb is not None:
            environment["memory_mb"] = int(memory_mb)
        metadata["environment"] = environment


def _configure_verifier_timeout_policy(
    config: DictConfig,
    dataset: Dataset,
    *,
    label: str,
) -> None:
    """Make declared sandbox verifier timeouts valid zero-reward outcomes."""

    enabled = OmegaConf.select(
        config,
        "swe.verifier_timeout_is_failure",
        default=True,
    )
    if not isinstance(enabled, bool):
        raise ValueError("swe.verifier_timeout_is_failure must be true or false")
    configured = 0
    for item in dataset.data:
        metadata = item.metadata if hasattr(item, "metadata") else item
        if not isinstance(metadata, dict):
            continue
        rllm_metadata = metadata.get("rllm")
        rllm_metadata = (
            dict(rllm_metadata) if isinstance(rllm_metadata, Mapping) else {}
        )
        rllm_metadata["verifier_timeout_is_failure"] = enabled
        metadata["rllm"] = rllm_metadata
        configured += 1
    logger.info(
        "Configured verifier timeout-as-failure=%s for %d task(s) in %s",
        enabled,
        configured,
        label,
    )


def _positive_integer_config(config: DictConfig, key: str) -> int:
    value = OmegaConf.select(config, key, default=None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer, got {value!r}")
    return value


def _denovo_shadow_sandbox_count(config: DictConfig) -> int:
    value = OmegaConf.select(
        config,
        "swe.denovo_shadow_sandbox_count",
        default=1,
    )
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            "swe.denovo_shadow_sandbox_count must be a positive integer, "
            f"got {value!r}"
        )
    return value


def _has_swe_setting(config: DictConfig, name: str) -> bool:
    swe = OmegaConf.select(config, "swe", default=None)
    return swe is not None and name in swe


def _configure_repository_difficulty_max_turns(
    config: DictConfig,
    dataset: Dataset,
    *,
    label: str,
) -> dict[str, int | bool]:
    """Write the effective DeNovoSWE turn budget into each task's metadata."""

    denovo_items: list[tuple[int, object, dict]] = []
    for index, item in enumerate(dataset.data):
        metadata = item.metadata if hasattr(item, "metadata") else item
        if not isinstance(metadata, dict):
            continue
        raw_rllm = metadata.get("rllm")
        raw_rllm = raw_rllm if isinstance(raw_rllm, Mapping) else {}
        task_profile = str(
            metadata.get("task_profile") or raw_rllm.get("task_profile") or ""
        )
        if task_profile == "repo_generation_denovoswe":
            denovo_items.append((index, item, metadata))
    if not denovo_items:
        return {
            "enabled": False,
            "tasks": 0,
            "low": 0,
            "medium": 0,
            "high": 0,
            "uniform": 0,
        }

    enabled = OmegaConf.select(
        config,
        "swe.repository_difficulty_max_turns_enable",
        default=False,
    )
    if not isinstance(enabled, bool):
        raise ValueError(
            "swe.repository_difficulty_max_turns_enable must be true or false"
        )

    if enabled:
        low_max_turns = _positive_integer_config(
            config,
            "swe.repository_difficulty_low_max_turns",
        )
        medium_max_turns = _positive_integer_config(
            config,
            "swe.repository_difficulty_medium_max_turns",
        )
        high_max_turns = _positive_integer_config(
            config,
            "swe.repository_difficulty_high_max_turns",
        )
        uniform_max_turns = None
    else:
        low_max_turns = medium_max_turns = high_max_turns = 0
        uniform_max_turns = _positive_integer_config(config, "swe.max_turns")

    counts = {"low": 0, "medium": 0, "high": 0, "uniform": 0}
    configured_tasks = 0
    for index, item, metadata in denovo_items:
        rllm_metadata = metadata.get("rllm")
        rllm_metadata = dict(rllm_metadata) if isinstance(rllm_metadata, Mapping) else {}
        task_id = str(getattr(item, "id", metadata.get("id", index)))
        if enabled:
            task_toml_metadata = metadata.get("metadata")
            task_toml_metadata = (
                task_toml_metadata
                if isinstance(task_toml_metadata, Mapping)
                else {}
            )
            raw_difficulty = metadata.get("difficulty")
            if raw_difficulty is None:
                raw_difficulty = task_toml_metadata.get("difficulty")
            if raw_difficulty is None:
                raise ValueError(
                    f"{label} DeNovoSWE task {task_id!r} is missing difficulty; "
                    "re-materialize the dataset with launch/prepare_dataset.sh"
                )
            difficulty = _to_unit_float(
                raw_difficulty,
                name=f"{label} task {task_id!r} difficulty",
            )
            if difficulty < 0.33:
                max_turns = low_max_turns
                bucket = "low"
            elif difficulty < 0.67:
                max_turns = medium_max_turns
                bucket = "medium"
            else:
                max_turns = high_max_turns
                bucket = "high"
        else:
            assert uniform_max_turns is not None
            max_turns = uniform_max_turns
            bucket = "uniform"

        rllm_metadata["max_turns"] = max_turns
        metadata["rllm"] = rllm_metadata
        counts[bucket] += 1
        configured_tasks += 1

    summary: dict[str, int | bool] = {
        "enabled": enabled,
        "tasks": configured_tasks,
        **counts,
    }
    if configured_tasks:
        logger.info("Configured repository-difficulty max turns for %s: %s", label, summary)
    return summary


def _configure_repository_difficulty_shadow_resources(
    config: DictConfig,
    dataset: Dataset,
    *,
    label: str,
) -> dict[str, int | bool]:
    """Write DeNovoSWE difficulty-adaptive shadow-only resources."""

    denovo_items: list[tuple[int, object, dict]] = []
    for index, item in enumerate(dataset.data):
        metadata = item.metadata if hasattr(item, "metadata") else item
        if not isinstance(metadata, dict):
            continue
        raw_rllm = metadata.get("rllm")
        raw_rllm = raw_rllm if isinstance(raw_rllm, Mapping) else {}
        task_profile = str(
            metadata.get("task_profile") or raw_rllm.get("task_profile") or ""
        )
        if task_profile == "repo_generation_denovoswe":
            denovo_items.append((index, item, metadata))
    if not denovo_items:
        return {
            "enabled": False,
            "tasks": 0,
            "low": 0,
            "medium": 0,
            "high": 0,
            "uniform": 0,
        }

    enabled = OmegaConf.select(
        config,
        "swe.repository_difficulty_shadow_resources_enable",
        default=False,
    )
    if not isinstance(enabled, bool):
        raise ValueError(
            "swe.repository_difficulty_shadow_resources_enable must be true or false"
        )

    resources_by_bucket: dict[str, dict[str, int]] = {}
    if enabled:
        for bucket in ("low", "medium", "high"):
            resources_by_bucket[bucket] = {
                "cpus": _positive_integer_config(
                    config,
                    f"swe.repository_difficulty_{bucket}_shadow_cpus",
                ),
                "memory_mb": _positive_integer_config(
                    config,
                    f"swe.repository_difficulty_{bucket}_shadow_memory_mb",
                ),
            }

    counts = {"low": 0, "medium": 0, "high": 0, "uniform": 0}
    for index, item, metadata in denovo_items:
        task_id = str(getattr(item, "id", metadata.get("id", index)))
        if not enabled:
            metadata.pop("shadow_sandbox_resources", None)
            metadata["shadow_resource_difficulty_bucket"] = "uniform"
            counts["uniform"] += 1
            continue
        task_toml_metadata = metadata.get("metadata")
        task_toml_metadata = (
            task_toml_metadata
            if isinstance(task_toml_metadata, Mapping)
            else {}
        )
        raw_difficulty = metadata.get("difficulty")
        if raw_difficulty is None:
            raw_difficulty = task_toml_metadata.get("difficulty")
        if raw_difficulty is None:
            raise ValueError(
                f"{label} DeNovoSWE task {task_id!r} is missing difficulty; "
                "re-materialize the dataset with launch/prepare_dataset.sh"
            )
        difficulty = _to_unit_float(
            raw_difficulty,
            name=f"{label} task {task_id!r} difficulty",
        )
        bucket = (
            "low"
            if difficulty < 0.33
            else "medium"
            if difficulty < 0.67
            else "high"
        )
        metadata["shadow_sandbox_resources"] = dict(
            resources_by_bucket[bucket]
        )
        metadata["shadow_resource_difficulty_bucket"] = bucket
        counts[bucket] += 1

    summary: dict[str, int | bool] = {
        "enabled": enabled,
        "tasks": len(denovo_items),
        **counts,
    }
    logger.info(
        "Configured repository-difficulty shadow resources for %s: %s mappings=%s",
        label,
        summary,
        resources_by_bucket if enabled else "uniform",
    )
    return summary


def _filter_inaccessible_training_tasks(dataset: Dataset, label: str) -> list[str]:
    filtered, removed = filter_inaccessible_repo_generation_tasks(dataset.data)
    if removed:
        dataset.data = filtered
        logger.warning(
            "Excluded %d inaccessible repository-generation task(s) from %s: %s",
            len(removed),
            label,
            ", ".join(removed),
        )
    return removed


def _filter_training_tasks_by_eligibility_manifest(
    dataset: Dataset,
    *,
    dataset_name: str,
    split: str,
    manifest_path: str | None,
    required: bool,
    label: str,
) -> dict[str, str]:
    """Apply one complete, revision-bound runtime eligibility manifest."""

    if not manifest_path:
        if required:
            raise ValueError(
                f"{label} requires swe.eligibility_manifest, but no path was configured"
            )
        return {}
    filtered, excluded, manifest = filter_tasks_with_eligibility_manifest(
        list(dataset.data),
        manifest_path,
        dataset=dataset_name,
        split=split,
    )
    if not filtered:
        raise ValueError(f"eligibility manifest excludes every task from {label}")
    dataset.data = filtered
    if excluded:
        by_reason: dict[str, int] = {}
        for reason in excluded.values():
            by_reason[reason] = by_reason.get(reason, 0) + 1
        logger.warning(
            "Excluded %d runtime-ineligible task(s) from %s using %s; "
            "reasons=%s eligible=%d contract=%s",
            len(excluded),
            label,
            manifest_path,
            dict(sorted(by_reason.items())),
            len(filtered),
            manifest.get("dataset_contract_sha256"),
        )
    else:
        logger.info(
            "Eligibility manifest accepted every task from %s: %s",
            label,
            manifest_path,
        )
    return excluded


def _default_image_availability_manifest(dataset_name: str) -> Path | None:
    """Return the conventional data-root probe result when it exists."""

    data_root = os.environ.get("DATA_ROOT")
    if data_root:
        root = Path(data_root).expanduser()
    else:
        rllm_home = os.environ.get("RLLM_HOME")
        if not rllm_home:
            return None
        root = Path(rllm_home).expanduser().parent
    candidate = root / "audits" / dataset_name / "image-availability.json"
    return candidate if candidate.is_file() else None


def _resolve_image_availability_manifest(
    config: DictConfig,
    *,
    dataset_name: str,
    validation: bool,
) -> Path | None:
    key = (
        "swe.val_image_availability_manifest"
        if validation
        else "swe.image_availability_manifest"
    )
    configured = OmegaConf.select(config, key, default=None)
    if configured is not None:
        path = Path(str(configured)).expanduser()
        if not path.is_file():
            raise ValueError(f"configured {key} does not exist: {path}")
        return path
    return _default_image_availability_manifest(dataset_name)


def _filter_training_tasks_by_image_availability_manifest(
    dataset: Dataset,
    *,
    dataset_name: str,
    split: str,
    manifest_path: Path | None,
    contract_tasks: list[object],
    label: str,
) -> dict[str, str]:
    """Apply an automatically discovered one-time image probe result."""

    if manifest_path is None:
        logger.info("No image availability manifest found for %s", label)
        return {}
    from rllm.eval._resolution import _resolve_public_image

    filtered, excluded, manifest = filter_tasks_with_image_availability_manifest(
        list(dataset.data),
        manifest_path,
        dataset=dataset_name,
        split=split,
        contract_tasks=contract_tasks,
        image_resolver=_resolve_public_image,
    )
    if not filtered:
        raise ValueError(f"image availability manifest excludes every task from {label}")
    dataset.data = filtered
    if excluded:
        reason_counts: dict[str, int] = {}
        for reason in excluded.values():
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        logger.warning(
            "Excluded %d task(s) based on image probe results from %s using %s; "
            "reasons=%s remaining=%d contract=%s",
            len(excluded),
            label,
            manifest_path,
            dict(sorted(reason_counts.items())),
            len(filtered),
            manifest.get("dataset_contract_sha256"),
        )
    else:
        logger.info(
            "Image availability manifest accepted every task from %s: %s",
            label,
            manifest_path,
        )
    return excluded


def _metadata_sequence(value: object) -> list[str]:
    """Normalize Arrow/Pandas list values without accepting scalar strings."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _status_mapping(value: object, field: str) -> dict[str, str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field} is not valid JSON: {exc}") from exc
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{field} must be a non-empty JSON object")
    result = dict(value)
    invalid = [name for name, status in result.items() if not isinstance(name, str) or status not in {"PASSED", "FAILED", "ERROR"}]
    if invalid:
        raise ValueError(f"{field} contains invalid test states")
    return result


def _validate_milestone_training_contract(config: DictConfig, train_dataset: Dataset) -> dict[str, int]:
    """Fail early when a Milestone run cannot produce trustworthy rewards.

    Materialized baseline/target test-set mismatches are diagnostic only. The
    rollout-local runtime baseline is authoritative for verification rewards.
    """
    enabled = bool(OmegaConf.select(config, "rllm.stepwise_advantage.milestone.enable", default=False))
    verification_enabled = enabled and bool(
        OmegaConf.select(
            config,
            "rllm.stepwise_advantage.milestone.verification_enable",
            default=True,
        )
    )
    if not enabled:
        return {
            "tasks": len(train_dataset),
            "shadow_eligible": 0,
            "verification_disabled": 0,
            "baseline_reference_incomplete": 0,
        }

    required_config = {
        "rllm.agent.name": "codeflow",
        "swe.protocol": "native_tool_call",
        "rllm.gateway.cumulative_token_mode": "true",
        "rllm.gateway.renderer_family": "qwen3.6" if OmegaConf.select(config, "swe.model_profile") == "qwen36_35b_a3b" else "qwen3.5",
        "rllm.algorithm.adv_estimator": "rloo",
        "rllm.algorithm.loss_fn": "vanilla",
        "rllm.stepwise_advantage.mode": "per_step",
    }
    errors: list[str] = []
    for key, expected in required_config.items():
        actual = OmegaConf.select(config, key, default=None)
        if str(actual).lower() != expected:
            errors.append(f"{key} must be {expected!r}, got {actual!r}")
    if not bool(OmegaConf.select(config, "rllm.stepwise_advantage.enable", default=False)):
        errors.append("rllm.stepwise_advantage.enable must be true")
    if errors:
        raise ValueError("Invalid Milestone training configuration: " + "; ".join(errors))

    invalid_tasks: list[str] = []
    shadow_eligible = 0
    verification_disabled = 0
    baseline_reference_incomplete = 0
    denovo_tasks = 0
    for index, item in enumerate(train_dataset.data):
        metadata = item.metadata if hasattr(item, "metadata") else item
        task_id = str(getattr(item, "id", index))
        if not isinstance(metadata, Mapping):
            invalid_tasks.append(f"{task_id}: metadata is not a mapping")
            continue
        try:
            rllm_metadata = metadata.get("rllm") if isinstance(metadata.get("rllm"), Mapping) else {}
            task_profile = str(metadata.get("task_profile") or rllm_metadata.get("task_profile") or "")
            is_denovo = task_profile == "repo_generation_denovoswe"
            if is_denovo:
                denovo_tasks += 1
                baseline = target = None
                outcome_source = str(metadata.get("outcome_source") or rllm_metadata.get("outcome_source") or "")
                potential_mode = str(metadata.get("verification_potential_mode") or rllm_metadata.get("verification_potential_mode") or "")
                verifier_profile = str(metadata.get("shadow_verifier_profile") or metadata.get("verifier_profile") or rllm_metadata.get("shadow_verifier_profile") or "")
                if outcome_source != "shadow_verifier":
                    raise ValueError("outcome_source must be shadow_verifier")
                if potential_mode != "pass_count":
                    raise ValueError(
                        "verification_potential_mode must be pass_count "
                        "for repository-generation training"
                    )
                if verifier_profile != "denovoswe_official":
                    raise ValueError("shadow_verifier_profile must be denovoswe_official")
            else:
                baseline = _status_mapping(metadata.get("baseline_output_json"), "baseline_output_json")
                target = _status_mapping(metadata.get("target_output_json"), "target_output_json")
                if not _metadata_sequence(metadata.get("relevant_files")):
                    raise ValueError("relevant_files must be non-empty")
                if not _metadata_sequence(metadata.get("test_file_names")):
                    raise ValueError("test_file_names must be non-empty")
            environment = metadata.get("environment") if isinstance(metadata.get("environment"), Mapping) else {}
            if not (metadata.get("docker_image") or environment.get("docker_image")):
                raise ValueError("docker_image is missing")
            task_dir = Path(getattr(item, "task_dir", metadata.get("task_path", "")))
            if not task_dir.is_dir():
                raise ValueError(f"task directory does not exist: {task_dir}")
            if not (task_dir / "tests" / "test.sh").is_file():
                raise ValueError("tests/test.sh is missing")
            if verification_enabled:
                from rllm.harnesses.shadow_sandbox import shadow_eligibility

                eligible, reason = shadow_eligibility(item, "sandbox-shell")
                if not eligible:
                    raise ValueError(f"shadow contract is invalid: {reason}")
        except (TypeError, ValueError) as exc:
            invalid_tasks.append(f"{task_id}: {exc}")
            continue

        shadow_eligible += 1
        if baseline is not None and target is not None and set(baseline) != set(target):
            baseline_reference_incomplete += 1

    shadow_sandbox_count = _denovo_shadow_sandbox_count(config)
    shadow_count_configured = _has_swe_setting(
        config,
        "denovo_shadow_sandbox_count",
    )
    if denovo_tasks:
        denovo_errors: list[str] = []
        if denovo_tasks != len(train_dataset):
            denovo_errors.append("repo_generation_denovoswe tasks cannot be mixed with bug-repair tasks")
        if bool(
            OmegaConf.select(
                config,
                "rllm.stepwise_advantage.milestone.navigation_enable",
                default=True,
            )
        ):
            denovo_errors.append("milestone.navigation_enable must be false")
        navigation_weight = float(
            OmegaConf.select(
                config,
                "rllm.stepwise_advantage.milestone.navigation_weight",
                default=0.0,
            )
        )
        if navigation_weight != 0.0:
            denovo_errors.append("milestone.navigation_weight must be 0")
        if _limit_termination_outcome_mode(config) != "verifier_outcome":
            denovo_errors.append("swe.limit_termination_outcome_mode must be verifier_outcome")
        if (
            bool(OmegaConf.select(config, "rllm.dynamic_sampling.enable", default=False))
            and str(
                OmegaConf.select(
                    config,
                    "rllm.dynamic_sampling.outcome_mode",
                    default="reward_uniform",
                )
            )
            != "verifier_pass_count"
        ):
            denovo_errors.append("rllm.dynamic_sampling.outcome_mode must be verifier_pass_count")
        background_finalize = OmegaConf.select(
            config,
            "swe.denovo_background_finalize_enable",
            default=False,
        )
        if background_finalize is True:
            if not bool(
                OmegaConf.select(
                    config,
                    "rllm.async_training.enable",
                    default=False,
                )
            ):
                denovo_errors.append(
                    "swe.denovo_background_finalize_enable requires "
                    "rllm.async_training.enable=true"
                )
        probe_merge_enable = OmegaConf.select(
            config,
            "swe.denovo_shadow_probe_merge_enable",
            default=False,
        )
        if probe_merge_enable is True and background_finalize is not True:
            denovo_errors.append(
                "swe.denovo_shadow_probe_merge_enable requires "
                "swe.denovo_background_finalize_enable=true"
            )
        if shadow_sandbox_count > 1 and background_finalize is not True:
            denovo_errors.append(
                "swe.denovo_shadow_sandbox_count>1 requires "
                "swe.denovo_background_finalize_enable=true"
            )
        if denovo_errors:
            raise ValueError("Invalid DeNovoSWE training configuration: " + "; ".join(denovo_errors))
    elif (
        OmegaConf.select(
            config,
            "swe.denovo_background_finalize_enable",
            default=False,
        )
        is True
        or OmegaConf.select(
            config,
            "swe.denovo_shadow_probe_merge_enable",
            default=False,
        )
        is True
        or shadow_count_configured
    ):
        raise ValueError(
            "DeNovo background/probe-merge/shadow-count settings are only valid for a "
            "repo_generation_denovoswe training dataset"
        )

    if invalid_tasks:
        examples = "; ".join(invalid_tasks[:5])
        raise ValueError(f"Milestone dataset contract failed for {len(invalid_tasks)}/{len(train_dataset)} tasks; examples: {examples}")
    if shadow_eligible == 0:
        raise ValueError("Milestone dataset contains no shadow-eligible tasks")
    if baseline_reference_incomplete:
        logger.warning(
            "Materialized baseline reference is incomplete for %s/%s tasks; runtime baseline probes remain enabled",
            baseline_reference_incomplete,
            len(train_dataset),
        )
    summary = {
        "tasks": len(train_dataset),
        "shadow_eligible": shadow_eligible,
        "verification_disabled": verification_disabled,
        "baseline_reference_incomplete": baseline_reference_incomplete,
    }
    logger.info("Validated SWE Milestone dataset contract: %s", summary)
    return summary


def _run_training(config: DictConfig) -> None:
    _configure_model_profile(config)
    _validate_model_sequence_parallelism(config)
    _validate_model_tensor_parallelism(config)
    _reject_removed_rollout_log_probability(config)
    (
        primary_sandbox_cpus,
        primary_sandbox_memory_mb,
        shadow_sandbox_cpus,
        shadow_sandbox_memory_mb,
    ) = _training_sandbox_resources(config)
    codeflow_tool_mode = _resolved_codeflow_tool_mode(config)
    codeflow_tool_restricted_mode = _resolved_codeflow_tool_restricted_mode(
        config,
        tool_mode=codeflow_tool_mode,
    )
    _validate_codeflow_tool_configuration(
        codeflow_tool_mode,
        codeflow_tool_restricted_mode,
    )
    shadow_finalize_stall_timeout = _shadow_finalize_stall_timeout(config)
    denovo_background_finalize_enable = OmegaConf.select(
        config,
        "swe.denovo_background_finalize_enable",
        default=False,
    )
    if not isinstance(denovo_background_finalize_enable, bool):
        raise ValueError(
            "swe.denovo_background_finalize_enable must be true or false"
        )
    denovo_shadow_sandbox_count = _denovo_shadow_sandbox_count(config)
    if denovo_shadow_sandbox_count > 1:
        if os.environ.get("TASK_PROFILE") != "denovoswe":
            raise ValueError(
                "swe.denovo_shadow_sandbox_count>1 requires "
                "TASK_PROFILE=denovoswe"
            )
        if not denovo_background_finalize_enable:
            raise ValueError(
                "swe.denovo_shadow_sandbox_count>1 requires "
                "swe.denovo_background_finalize_enable=true"
            )
    denovo_shadow_probe_merge_enable = OmegaConf.select(
        config,
        "swe.denovo_shadow_probe_merge_enable",
        default=False,
    )
    if not isinstance(denovo_shadow_probe_merge_enable, bool):
        raise ValueError(
            "swe.denovo_shadow_probe_merge_enable must be true or false"
        )
    denovo_shadow_probe_merge_max_steps = OmegaConf.select(
        config,
        "swe.denovo_shadow_probe_merge_max_steps",
        default=3,
    )
    if (
        isinstance(denovo_shadow_probe_merge_max_steps, bool)
        or not isinstance(denovo_shadow_probe_merge_max_steps, int)
        or denovo_shadow_probe_merge_max_steps < 2
    ):
        raise ValueError(
            "swe.denovo_shadow_probe_merge_max_steps must be an integer "
            "greater than or equal to 2"
        )
    if denovo_shadow_probe_merge_enable:
        if os.environ.get("TASK_PROFILE") != "denovoswe":
            raise ValueError(
                "swe.denovo_shadow_probe_merge_enable requires "
                "TASK_PROFILE=denovoswe"
            )
        if not denovo_background_finalize_enable:
            raise ValueError(
                "swe.denovo_shadow_probe_merge_enable requires "
                "swe.denovo_background_finalize_enable=true"
            )
    (
        shadow_partition_timeout_recovery,
        shadow_partition_recovery_command_timeout,
        shadow_partition_recovery_max_commands,
    ) = _shadow_verification_recovery_settings(config)
    bug_repair_verification_potential_mode = (
        _bug_repair_verification_potential_mode(config)
    )
    OmegaConf.update(
        config,
        "swe.shadow_finalize_stall_timeout",
        shadow_finalize_stall_timeout,
        merge=False,
        force_add=True,
    )
    train_name = OmegaConf.select(config, "swe.train_dataset", default="r2egym")
    train_split = OmegaConf.select(config, "swe.train_split", default="train")
    val_name = OmegaConf.select(config, "swe.val_dataset", default=train_name)
    val_split = OmegaConf.select(config, "swe.val_split", default=train_split)
    sandbox_backend = _configure_minisandbox_training(
        config,
        train_name=train_name,
        val_name=val_name,
        train_split=train_split,
        val_split=val_split,
    )
    if sandbox_backend == "k8s":
        from rllm.sandbox.backends.k8s import check_k8s_adapter
        check_k8s_adapter()
    if sandbox_backend == "minisandbox":
        # The pinned OCI cache is authoritative for MiniSandbox. Remote backend
        # service-side image audit describes a different backend and must not
        # filter an otherwise valid local-cache task.
        train_image_manifest = None
        val_image_manifest = None
    else:
        train_image_manifest = _resolve_image_availability_manifest(
            config,
            dataset_name=str(train_name),
            validation=False,
        )
        explicit_val_image_manifest = OmegaConf.select(
            config,
            "swe.val_image_availability_manifest",
            default=None,
        )
        if (
            explicit_val_image_manifest is None
            and str(val_name) == str(train_name)
            and str(val_split) == str(train_split)
        ):
            val_image_manifest = train_image_manifest
        else:
            val_image_manifest = _resolve_image_availability_manifest(
                config,
                dataset_name=str(val_name),
                validation=True,
            )
    if train_image_manifest is not None:
        OmegaConf.update(
            config,
            "swe.image_availability_manifest",
            str(train_image_manifest),
            merge=False,
            force_add=True,
        )
    if val_image_manifest is not None:
        OmegaConf.update(
            config,
            "swe.val_image_availability_manifest",
            str(val_image_manifest),
            merge=False,
            force_add=True,
        )
    training_params_path = _save_training_params(config)
    print(f"Saved resolved training parameters to {training_params_path}", flush=True)

    from rllm.eval.agent_loader import load_agent
    from rllm.hooks import SandboxTaskHooks
    from rllm.trainer import AgentTrainer

    agent_name = str(config.rllm.agent.get("name", "codeflow"))
    train_dataset = _load_sandbox_dataset(train_name, train_split)
    val_dataset = _load_sandbox_dataset(val_name, val_split)
    _filter_inaccessible_training_tasks(
        train_dataset,
        f"train dataset {train_name}/{train_split}",
    )
    _filter_inaccessible_training_tasks(
        val_dataset,
        f"validation dataset {val_name}/{val_split}",
    )
    train_contract_tasks = list(train_dataset.data)
    val_contract_tasks = list(val_dataset.data)
    eligibility_manifest = OmegaConf.select(
        config, "swe.eligibility_manifest", default=None
    )
    require_eligibility_manifest = bool(
        OmegaConf.select(
            config,
            "swe.require_eligibility_manifest",
            default=False,
        )
    )
    _filter_training_tasks_by_eligibility_manifest(
        train_dataset,
        dataset_name=str(train_name),
        split=str(train_split),
        manifest_path=(
            str(eligibility_manifest) if eligibility_manifest is not None else None
        ),
        required=require_eligibility_manifest,
        label=f"train dataset {train_name}/{train_split}",
    )
    val_eligibility_manifest = OmegaConf.select(
        config, "swe.val_eligibility_manifest", default=None
    )
    if (
        val_eligibility_manifest is None
        and str(val_name) == str(train_name)
        and str(val_split) == str(train_split)
    ):
        val_eligibility_manifest = eligibility_manifest
    _filter_training_tasks_by_eligibility_manifest(
        val_dataset,
        dataset_name=str(val_name),
        split=str(val_split),
        manifest_path=(
            str(val_eligibility_manifest)
            if val_eligibility_manifest is not None
            else None
        ),
        required=(
            require_eligibility_manifest
            and str(val_name) == str(train_name)
            and str(val_split) == str(train_split)
        ),
        label=f"validation dataset {val_name}/{val_split}",
    )
    if sandbox_backend != "minisandbox":
        _filter_training_tasks_by_image_availability_manifest(
            train_dataset,
            dataset_name=str(train_name),
            split=str(train_split),
            manifest_path=train_image_manifest,
            contract_tasks=train_contract_tasks,
            label=f"train dataset {train_name}/{train_split}",
        )
        _filter_training_tasks_by_image_availability_manifest(
            val_dataset,
            dataset_name=str(val_name),
            split=str(val_split),
            manifest_path=val_image_manifest,
            contract_tasks=val_contract_tasks,
            label=f"validation dataset {val_name}/{val_split}",
        )
    _override_sandbox_resources(
        train_dataset,
        primary_sandbox_cpus,
        primary_sandbox_memory_mb,
    )
    _override_sandbox_resources(
        val_dataset,
        primary_sandbox_cpus,
        primary_sandbox_memory_mb,
    )
    _configure_verifier_timeout_policy(
        config,
        train_dataset,
        label=f"train dataset {train_name}/{train_split}",
    )
    _configure_verifier_timeout_policy(
        config,
        val_dataset,
        label=f"validation dataset {val_name}/{val_split}",
    )
    _configure_repository_difficulty_max_turns(
        config,
        train_dataset,
        label=f"train dataset {train_name}/{train_split}",
    )
    _configure_repository_difficulty_max_turns(
        config,
        val_dataset,
        label=f"validation dataset {val_name}/{val_split}",
    )
    _configure_repository_difficulty_shadow_resources(
        config,
        train_dataset,
        label=f"train dataset {train_name}/{train_split}",
    )
    # Validation deliberately retains its established resource/lifecycle
    # path. Difficulty-adaptive shadow leases are a training-only admission
    # feature and must not alter validation or independent evaluation.
    if sandbox_backend == "minisandbox":
        records = _validate_minisandbox_task_contract(
            str(
                OmegaConf.select(
                    config, "swe.minisandbox.shared_cache_dir"
                )
            ),
            train_dataset,
            val_dataset,
            dataset_name=str(train_name),
        )
        OmegaConf.update(
            config,
            "swe.minisandbox.canary_task_id",
            sorted(records)[0],
            merge=False,
            force_add=True,
        )
        # Refresh the resolved audit now that exact cache coverage and the
        # deterministic real-image canary have been selected.
        _save_training_params(config)
    _validate_milestone_training_contract(config, train_dataset)
    rollout_log_path = OmegaConf.select(config, "swe.rollout_log_path", default=None)
    if rollout_log_path is not None:
        os.environ["RLLM_ROLLOUT_LOG_PATH"] = str(rollout_log_path)

    agent_flow = load_agent(agent_name)

    if agent_name == "codeflow":
        configure_tool_set = getattr(
            agent_flow,
            "configure_codeflow_tool_mode",
            None,
        )
        if not callable(configure_tool_set):
            raise ValueError("Agent 'codeflow' does not support swe.codeflow_tool_mode")
        configure_tool_set(codeflow_tool_mode)
        configure_tool_mode = getattr(
            agent_flow,
            "configure_codeflow_tool_restricted_mode",
            None,
        )
        if not callable(configure_tool_mode):
            raise ValueError("Agent 'codeflow' does not support swe.codeflow_tool_restricted_mode")
        configure_tool_mode(list(codeflow_tool_restricted_mode))

    protocol = OmegaConf.select(config, "swe.protocol", default=None)
    if protocol is not None:
        set_protocol = getattr(agent_flow, "set_protocol", None)
        if callable(set_protocol):
            set_protocol(str(protocol))
        elif hasattr(agent_flow, "protocol"):
            agent_flow.protocol = str(protocol)
        else:
            raise ValueError(f"Agent '{agent_name}' does not support swe.protocol={protocol!r}")

    max_turns = OmegaConf.select(config, "swe.max_turns", default=None)
    if max_turns is not None:
        agent_flow.max_turns = int(max_turns)
    if hasattr(agent_flow, "command_timeout"):
        agent_flow.command_timeout = _to_positive_float(
            OmegaConf.select(config, "swe.command_timeout", default=getattr(agent_flow, "command_timeout", 120.0)),
            name="swe.command_timeout",
        )
    agent_flow.limit_termination_success_reward = _limit_termination_success_reward(config)
    agent_flow.limit_termination_outcome_mode = _limit_termination_outcome_mode(config)
    agent_flow.denovo_background_finalize_enable = bool(
        denovo_background_finalize_enable
    )
    agent_flow.denovo_shadow_sandbox_count = int(
        denovo_shadow_sandbox_count
    )
    agent_flow.denovo_shadow_probe_merge_enable = bool(
        denovo_shadow_probe_merge_enable
    )
    agent_flow.denovo_shadow_probe_merge_max_steps = int(
        denovo_shadow_probe_merge_max_steps
    )
    if hasattr(agent_flow, "shadow_finalize_stall_timeout"):
        agent_flow.shadow_finalize_stall_timeout = shadow_finalize_stall_timeout
    if hasattr(agent_flow, "bug_repair_verification_potential_mode"):
        agent_flow.bug_repair_verification_potential_mode = (
            bug_repair_verification_potential_mode
        )
        agent_flow.shadow_partition_timeout_recovery = (
            shadow_partition_timeout_recovery
        )
        agent_flow.shadow_partition_recovery_command_timeout = (
            shadow_partition_recovery_command_timeout
        )
        agent_flow.shadow_partition_recovery_max_commands = (
            shadow_partition_recovery_max_commands
        )
    hooks = SandboxTaskHooks(
        sandbox_backend=sandbox_backend,
        shadow_sandbox_resources={
            "cpus": shadow_sandbox_cpus,
            "memory_mb": shadow_sandbox_memory_mb,
        },
    )
    trainer: AgentTrainer = AgentTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        backend="verl",
        agent_flow=agent_flow,
        hooks=hooks,
        sandbox_backend=sandbox_backend,
    )
    trainer.train()


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="unified",
    version_base=None,
)
def main(config: DictConfig) -> None:
    experiment_root = os.environ.get("EXP_ROOT")
    if experiment_root:
        backup_root = Path(experiment_root).expanduser()
    else:
        # Default launcher layout: EXP_ROOT/checkpoints/training.
        backup_root = Path(str(config.trainer.default_local_dir)).expanduser().parent.parent
    _save_launcher_backup(Path(__file__).with_suffix(".sh"), backup_root / "train_backup.sh")
    _run_training(config)


if __name__ == "__main__":
    main()
