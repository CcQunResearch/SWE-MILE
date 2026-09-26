"""Helpers for the VerlBackend that previously came from RayPPOTrainer."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import tempfile
from typing import Any, Literal

import torch
from omegaconf import DictConfig, OmegaConf
from verl import DataProto
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance

from rllm.hf_checkpoint_keys import repair_qwen35_hf_checkpoint
from rllm.trainer.algorithms.config import _explicit_override_keys, _plain, sync_shared_keys

logger = logging.getLogger(__name__)


# (verl_native_path, rllm_path) — value is the same on both sides; only the
# location differs. Used by ``sync_config`` to keep both
# namespaces in sync regardless of which side the user typed on the CLI.
_SHARED_KEYS: list[tuple[str, str]] = [
    ("algorithm.adv_estimator", "rllm.algorithm.adv_estimator"),
    ("algorithm.norm_adv_by_std_in_grpo", "rllm.algorithm.norm_adv_by_std_in_grpo"),
    ("algorithm.rollout_correction.bypass_mode", "rllm.algorithm.rollout_correction.bypass_mode"),
    ("algorithm.rollout_correction.rollout_is", "rllm.algorithm.rollout_correction.tis_mode"),
    ("algorithm.rollout_correction.rollout_is_threshold", "rllm.algorithm.rollout_correction.tis_cap"),
    ("actor_rollout_ref.actor.kl_loss_coef", "rllm.algorithm.kl_beta"),
    ("actor_rollout_ref.actor.policy_loss.loss_mode", "rllm.algorithm.loss_fn"),
    ("actor_rollout_ref.actor.loss_agg_mode", "rllm.algorithm.loss_agg_mode"),
    ("actor_rollout_ref.actor.optim.lr_warmup_steps", "rllm.algorithm.warmup_steps"),
    ("actor_rollout_ref.actor.optim.lr_warmup_steps_ratio", "rllm.algorithm.warmup_steps_ratio"),
    ("actor_rollout_ref.actor.clip_ratio_high", "rllm.algorithm.eps_clip_high"),
    ("actor_rollout_ref.actor.router_replay.mode", "rllm.algorithm.router_replay"),
    ("actor_rollout_ref.rollout.n", "rllm.rollout.n"),
    ("actor_rollout_ref.rollout.val_kwargs.n", "rllm.rollout.n_val"),
    ("actor_rollout_ref.rollout.temperature", "rllm.rollout.train.temperature"),
    ("actor_rollout_ref.rollout.top_p", "rllm.rollout.train.top_p"),
    ("actor_rollout_ref.rollout.top_k", "rllm.rollout.train.top_k"),
    ("actor_rollout_ref.rollout.response_length", "rllm.rollout.train.max_tokens"),
    ("actor_rollout_ref.rollout.val_kwargs.temperature", "rllm.rollout.val.temperature"),
    ("actor_rollout_ref.rollout.val_kwargs.top_p", "rllm.rollout.val.top_p"),
    ("actor_rollout_ref.rollout.val_kwargs.top_k", "rllm.rollout.val.top_k"),
    ("trainer.save_freq", "rllm.trainer.save_freq"),
    ("trainer.test_freq", "rllm.trainer.test_freq"),
    ("trainer.val_before_train", "rllm.trainer.val_before_train"),
    ("trainer.val_only", "rllm.trainer.val_only"),
    ("trainer.total_epochs", "rllm.trainer.total_epochs"),
    ("trainer.logger", "rllm.trainer.logger"),
    ("trainer.project_name", "rllm.trainer.project_name"),
    ("trainer.experiment_name", "rllm.trainer.experiment_name"),
    ("data.train_batch_size", "rllm.data.train_batch_size"),
    ("data.max_prompt_length", "rllm.data.max_prompt_length"),
    ("data.max_response_length", "rllm.data.max_response_length"),
]

_TOTAL_TRAINING_STEPS_KEY: tuple[str, str] = ("trainer.total_training_steps", "rllm.trainer.total_batches")

_HF_WEIGHT_FILENAMES = {
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
}

_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
_SAFETENSORS_FLOAT_DTYPES = {"F16", "BF16", "F32", "F64", "F8_E4M3", "F8_E5M2"}
_HF_SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
_HF_SAFE_WEIGHTS_NAME = "model.safetensors"


def _select_optional_config(config: DictConfig, *paths: str) -> Any:
    for path in paths:
        value = OmegaConf.select(config, path)
        if value is not None:
            return value
    return None


def _resolve_hf_checkpoint_save_dir(config: DictConfig) -> str | None:
    """Return an absolute optional root for standalone HF checkpoints."""
    hf_save_dir = _select_optional_config(config, "trainer.hf_save_dir", "rllm.trainer.hf_save_dir")
    if not hf_save_dir:
        return None

    hf_save_dir = os.path.expanduser(str(hf_save_dir))
    if not os.path.isabs(hf_save_dir):
        hf_save_dir = os.path.join(os.getcwd(), hf_save_dir)
    return os.path.abspath(hf_save_dir)


def _resolve_hf_checkpoint_save_dtype(config: DictConfig) -> str | None:
    dtype = _select_optional_config(config, "trainer.hf_save_dtype", "rllm.trainer.hf_save_dtype")
    if dtype is None:
        return None
    dtype = str(dtype).strip().lower()
    if dtype in {"", "none", "null", "false"}:
        return None
    return dtype


def _resolve_hf_checkpoint_max_shard_size(config: DictConfig) -> int | None:
    value = _select_optional_config(
        config,
        "trainer.hf_save_max_shard_size",
        "rllm.trainer.hf_save_max_shard_size",
    )
    if value is None:
        return None
    return _parse_size_to_bytes(value)


def _parse_size_to_bytes(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)

    text = str(value).strip()
    if text.lower() in {"", "none", "null"}:
        raise ValueError("HF checkpoint max shard size cannot be empty")

    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]+)?", text)
    if not match:
        raise ValueError(f"Invalid HF checkpoint max shard size: {value!r}")

    amount = float(match.group(1))
    unit = (match.group(2) or "B").lower()
    units = {
        "b": 1,
        "byte": 1,
        "bytes": 1,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "tb": 1000**4,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
        "tib": 1024**4,
    }
    if unit not in units:
        raise ValueError(f"Unsupported HF checkpoint max shard size unit: {unit!r}")
    return int(amount * units[unit])


def _contains_hf_model_weights(hf_path: str) -> bool:
    if not os.path.isdir(hf_path):
        return False

    for root, _, files in os.walk(hf_path):
        for name in files:
            if name in _HF_WEIGHT_FILENAMES:
                return True
            if name.startswith(("model-", "pytorch_model-", "adapter_model-")) and name.endswith(
                (".safetensors", ".bin")
            ):
                return True
    return False


def _parse_global_step_dir(name: str) -> int | None:
    prefix = "global_step_"
    if not name.startswith(prefix):
        return None
    try:
        return int(name[len(prefix) :])
    except ValueError:
        return None


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value.lower() in {"", "none", "null"}:
            return None
        return int(value)
    return int(value)


def _prune_old_hf_checkpoints(hf_save_root: str, max_to_keep: Any) -> None:
    max_to_keep = _coerce_optional_int(max_to_keep)
    if max_to_keep is None or max_to_keep <= 0 or not os.path.isdir(hf_save_root):
        return

    candidates: list[tuple[int, str]] = []
    for name in os.listdir(hf_save_root):
        step = _parse_global_step_dir(name)
        path = os.path.join(hf_save_root, name)
        if step is not None and os.path.isdir(path):
            candidates.append((step, path))

    candidates.sort()
    for _, old_path in candidates[: max(0, len(candidates) - max_to_keep)]:
        shutil.rmtree(old_path, ignore_errors=True)
        logger.info("Removed old HuggingFace checkpoint mirror: %s", old_path)


def _safe_tensor_header(path: str) -> dict[str, Any]:
    import struct

    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(header_len))


def _safe_tensor_nbytes(info: dict[str, Any], target_dtype: str | None) -> int:
    dtype = str(info["dtype"])
    shape = info.get("shape", [])
    numel = math.prod(int(dim) for dim in shape)
    if target_dtype == "bfloat16" and dtype in _SAFETENSORS_FLOAT_DTYPES:
        return numel * _SAFETENSORS_DTYPE_BYTES["BF16"]
    return numel * _SAFETENSORS_DTYPE_BYTES[dtype]


def _safe_tensor_payload_nbytes(path: str) -> int:
    header = _safe_tensor_header(path)
    return sum(
        info["data_offsets"][1] - info["data_offsets"][0]
        for key, info in header.items()
        if key != "__metadata__"
    )


def _model_safetensor_sources(hf_path: str) -> tuple[list[tuple[str, str]], set[str]]:
    index_path = os.path.join(hf_path, _HF_SAFE_WEIGHTS_INDEX_NAME)
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        source_files = {os.path.join(hf_path, filename) for filename in weight_map.values()}
        missing = [path for path in source_files if not os.path.isfile(path)]
        single_path = os.path.join(hf_path, _HF_SAFE_WEIGHTS_NAME)
        if missing and os.path.isfile(single_path) and os.path.getmtime(single_path) >= os.path.getmtime(index_path):
            # Re-saving a normalized checkpoint can replace its shards with a
            # single Transformers export while leaving our old index behind.
            # Never recover a partial export or silently change the key set.
            header = _safe_tensor_header(single_path)
            keys = [key for key in header if key != "__metadata__"]
            import struct

            with open(single_path, "rb") as handle:
                header_size = struct.unpack("<Q", handle.read(8))[0]
            payload_size = max((header[key]["data_offsets"][1] for key in keys), default=0)
            if set(keys) == set(weight_map) and keys and os.path.getsize(single_path) == 8 + header_size + payload_size:
                logger.warning("Ignoring stale HF shard index in %s; using complete newer single-file export", hf_path)
                return [(key, single_path) for key in keys], {single_path}
        return [(key, os.path.join(hf_path, filename)) for key, filename in weight_map.items()], source_files

    single_path = os.path.join(hf_path, _HF_SAFE_WEIGHTS_NAME)
    if os.path.exists(single_path):
        header = _safe_tensor_header(single_path)
        keys = [key for key in header if key != "__metadata__"]
        return [(key, single_path) for key in keys], {single_path}

    return [], set()


def _plan_hf_safetensor_shards(
    hf_path: str,
    target_dtype: str | None,
    max_shard_size: int | None,
) -> tuple[list[list[tuple[str, str]]], set[str]]:
    entries, source_files = _model_safetensor_sources(hf_path)
    if not entries:
        return [], set()

    headers = {path: _safe_tensor_header(path) for path in source_files}
    max_shard_size = max_shard_size or sum(
        _safe_tensor_nbytes(headers[path][key], target_dtype) for key, path in entries
    )

    shards: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    current_size = 0
    for key, path in entries:
        tensor_size = _safe_tensor_nbytes(headers[path][key], target_dtype)
        if current and current_size + tensor_size > max_shard_size:
            shards.append(current)
            current = []
            current_size = 0
        current.append((key, path))
        current_size += tensor_size
    if current:
        shards.append(current)
    return shards, source_files


def _torch_dtype_from_hf_save_dtype(dtype: str | None):
    if dtype is None:
        return None
    normalized = dtype.lower()
    if normalized in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "torch.float16"}:
        return torch.float16
    if normalized in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported HF checkpoint save dtype: {dtype!r}")


def _canonical_hf_save_dtype(dtype: str | None) -> str | None:
    torch_dtype = _torch_dtype_from_hf_save_dtype(dtype)
    if torch_dtype is torch.bfloat16:
        return "bfloat16"
    if torch_dtype is torch.float16:
        return "float16"
    if torch_dtype is torch.float32:
        return "float32"
    return None


def _update_hf_config_dtype(hf_path: str, dtype: str | None) -> None:
    if dtype is None:
        return

    config_path = os.path.join(hf_path, "config.json")
    if not os.path.exists(config_path):
        return

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    config["dtype"] = dtype
    if "torch_dtype" in config:
        config["torch_dtype"] = dtype
    for nested_key in ("text_config", "vision_config"):
        nested = config.get(nested_key)
        if isinstance(nested, dict):
            nested["dtype"] = dtype

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def _normalize_hf_checkpoint_weights(
    hf_path: str,
    save_dtype: str | None,
    max_shard_size: int | None,
) -> None:
    target_dtype = _canonical_hf_save_dtype(save_dtype)
    if target_dtype is None and max_shard_size is None:
        return

    torch_dtype = _torch_dtype_from_hf_save_dtype(target_dtype)
    shards, source_files = _plan_hf_safetensor_shards(hf_path, target_dtype, max_shard_size)
    if not shards:
        logger.warning("Skipping HuggingFace checkpoint normalization: no safetensors weights in %s", hf_path)
        return

    from safetensors import safe_open
    from safetensors.torch import save_file

    tmp_path = os.path.join(hf_path, f".tmp_hf_weights_{os.getpid()}")
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    os.makedirs(tmp_path, exist_ok=True)

    total_shards = len(shards)
    weight_map: dict[str, str] = {}
    try:
        for shard_idx, shard_entries in enumerate(shards, start=1):
            shard_name = f"model.safetensors-{shard_idx:05d}-of-{total_shards:05d}.safetensors"
            shard_state: dict[str, torch.Tensor] = {}
            handles = {path: safe_open(path, framework="pt", device="cpu") for _, path in shard_entries}
            try:
                for key, path in shard_entries:
                    tensor = handles[path].get_tensor(key)
                    if torch_dtype is not None and tensor.is_floating_point():
                        tensor = tensor.to(dtype=torch_dtype)
                    shard_state[key] = tensor.contiguous()
                    weight_map[key] = shard_name
                save_file(shard_state, os.path.join(tmp_path, shard_name), metadata={"format": "pt"})
            finally:
                for handle in handles.values():
                    del handle
                del shard_state

        index = {
            "metadata": {
                "total_size": sum(_safe_tensor_payload_nbytes(os.path.join(tmp_path, name)) for name in set(weight_map.values())),
            },
            "weight_map": weight_map,
        }
        with open(os.path.join(tmp_path, _HF_SAFE_WEIGHTS_INDEX_NAME), "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
            f.write("\n")

        for path in source_files:
            if os.path.exists(path):
                os.remove(path)
        old_index = os.path.join(hf_path, _HF_SAFE_WEIGHTS_INDEX_NAME)
        if os.path.exists(old_index):
            os.remove(old_index)

        for name in os.listdir(tmp_path):
            os.replace(os.path.join(tmp_path, name), os.path.join(hf_path, name))
        _update_hf_config_dtype(hf_path, target_dtype)
        logger.info(
            "Normalized HuggingFace checkpoint at %s to dtype=%s max_shard_size=%s",
            hf_path,
            target_dtype or "original",
            max_shard_size,
        )
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def _copy_hf_checkpoint_for_step(
    actor_local_path: str,
    hf_save_root: str,
    global_steps: int,
    max_to_keep: Any = None,
    save_dtype: str | None = None,
    max_shard_size: int | None = None,
) -> str | None:
    """Copy VERL's actor/huggingface export into a standalone global_step_N tree."""
    source_path = os.path.join(actor_local_path, "huggingface")
    if not _contains_hf_model_weights(source_path):
        logger.warning(
            "Skipping HuggingFace checkpoint mirror for step %s: %s does not contain model weights. "
            "Make sure actor_rollout_ref.actor.checkpoint.save_contents includes 'hf_model'.",
            global_steps,
            source_path,
        )
        return None

    _normalize_hf_checkpoint_weights(source_path, save_dtype=save_dtype, max_shard_size=max_shard_size)
    key_repair = repair_qwen35_hf_checkpoint(source_path)
    if key_repair.applicable:
        logger.info(
            "Validated Qwen3.5 Hugging Face checkpoint keys at %s "
            "(changed=%s visual_keys=%d)",
            source_path,
            key_repair.changed,
            key_repair.canonical_visual_keys,
        )

    hf_save_root = os.path.abspath(hf_save_root)
    target_path = os.path.abspath(os.path.join(hf_save_root, f"global_step_{global_steps}"))
    if os.path.commonpath([hf_save_root, target_path]) != hf_save_root:
        raise ValueError(f"Invalid HuggingFace checkpoint target outside save root: {target_path}")
    if os.path.realpath(source_path) == os.path.realpath(target_path):
        return target_path

    os.makedirs(hf_save_root, exist_ok=True)
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    shutil.copytree(source_path, target_path)
    logger.info("Mirrored HuggingFace checkpoint for step %s to %s", global_steps, target_path)
    _prune_old_hf_checkpoints(hf_save_root, max_to_keep)
    return target_path


def sync_config(config: DictConfig, hydra_overrides: list[str] | None = None) -> None:
    """Keep verl-native and rllm-namespaced config in sync.

    Precedence per shared key: rllm CLI explicit > verl CLI explicit > rllm
    yaml default > verl yaml default. ``None`` rllm values are treated as
    "no rllm default", letting verl's yaml default stand. Verl-native shared
    CLI overrides still work for backward compatibility, but warn because
    shared keys should be set through their rllm.* paths going forward.

    ``hydra_overrides`` is the list of CLI overrides captured in the main
    Hydra-decorated process (``HydraConfig.get().overrides.task``). It must be
    passed in when this function runs inside a Ray actor (Hydra context isn't
    available across processes). Outside Ray, it can be omitted.

    Also derives ``actor.use_kl_loss = (kl_beta > 0)`` from rllm values.
    """
    explicit = _explicit_override_keys(hydra_overrides)

    dynamic_sequence_budget = bool(
        OmegaConf.select(config, "rllm.data.dynamic_sequence_budget", default=False)
    )
    if dynamic_sequence_budget:
        legacy_length_keys = {
            "rllm.data.max_prompt_length",
            "rllm.data.max_response_length",
            "data.max_prompt_length",
            "data.max_response_length",
        }
        conflicting = sorted(legacy_length_keys & explicit)
        if conflicting:
            raise ValueError(
                "rllm.data.dynamic_sequence_budget=true conflicts with explicit "
                "legacy sequence partitions: "
                + ", ".join(conflicting)
                + ". Remove those overrides and configure only "
                "actor_rollout_ref.rollout.max_model_len."
            )

        max_model_len = OmegaConf.select(
            config, "actor_rollout_ref.rollout.max_model_len"
        )
        if (
            isinstance(max_model_len, bool)
            or not isinstance(max_model_len, int)
            or max_model_len <= 0
        ):
            raise ValueError(
                "rllm.data.dynamic_sequence_budget=true requires "
                "actor_rollout_ref.rollout.max_model_len to be a positive integer"
            )

        gateway_max_context = OmegaConf.select(
            config, "rllm.gateway.max_context_tokens"
        )
        if (
            gateway_max_context is not None
            and int(gateway_max_context) != max_model_len
        ):
            raise ValueError(
                "dynamic sequence budgeting uses "
                "actor_rollout_ref.rollout.max_model_len as its sole model "
                "window; rllm.gateway.max_context_tokens must be unset or "
                "equal to max_model_len"
            )

        use_remove_padding = OmegaConf.select(
            config, "actor_rollout_ref.model.use_remove_padding"
        )
        if use_remove_padding is not True:
            raise ValueError(
                "rllm.data.dynamic_sequence_budget=true requires "
                "actor_rollout_ref.model.use_remove_padding=true"
            )

        per_gpu_limit = OmegaConf.select(
            config, "actor_rollout_ref.actor.ppo_max_token_len_per_gpu"
        )
        sequence_parallel_size = OmegaConf.select(
            config,
            "actor_rollout_ref.actor.ulysses_sequence_parallel_size",
            default=1,
        )
        if (
            isinstance(per_gpu_limit, bool)
            or not isinstance(per_gpu_limit, int)
            or per_gpu_limit <= 0
            or isinstance(sequence_parallel_size, bool)
            or not isinstance(sequence_parallel_size, int)
            or sequence_parallel_size <= 0
        ):
            raise ValueError(
                "dynamic sequence budgeting requires positive integer "
                "actor_rollout_ref.actor.ppo_max_token_len_per_gpu and "
                "ulysses_sequence_parallel_size values"
            )
        distributed_limit = per_gpu_limit * sequence_parallel_size
        if distributed_limit < max_model_len:
            raise ValueError(
                "dynamic sequence budget exceeds actor sequence-parallel token "
                f"capacity: max_model_len={max_model_len}, "
                f"ppo_max_token_len_per_gpu={per_gpu_limit}, "
                f"ulysses_sequence_parallel_size={sequence_parallel_size}, "
                f"capacity={distributed_limit}"
            )

    def warn_verl_override(verl_path: str, rllm_path: str, conflict: bool = False) -> None:
        if conflict:
            logger.warning(
                "Verl-native shared config %s conflicts with %s; using %s. Setting shared rLLM/verl knobs through Verl-native paths is deprecated; use %s=... instead.",
                verl_path,
                rllm_path,
                rllm_path,
                rllm_path,
            )
            return
        logger.warning(
            "Verl-native shared config %s is deprecated and will be removed in a future release; use %s=... instead.",
            verl_path,
            rllm_path,
        )

    def maybe_warn_verl_override(verl_path: str, rllm_path: str, *, conflict: bool = False) -> None:
        if verl_path in explicit:
            warn_verl_override(verl_path, rllm_path, conflict=conflict)

    def same(left: Any, right: Any) -> bool:
        return _plain(left) == _plain(right)

    def sync_total_training_steps() -> None:
        verl_path, rllm_path = _TOTAL_TRAINING_STEPS_KEY

        def to_verl(value: int | None) -> int | None:
            return None if value is None or value <= 0 else value

        def to_rllm(value: int | None) -> int:
            return -1 if value is None else value

        if rllm_path in explicit:
            rllm_value = OmegaConf.select(config, rllm_path)
            verl_value = OmegaConf.select(config, verl_path)
            rllm_as_verl = to_verl(rllm_value)
            maybe_warn_verl_override(verl_path, rllm_path, conflict=not same(verl_value, rllm_as_verl))
            OmegaConf.update(config, verl_path, rllm_as_verl, merge=False)
        elif verl_path in explicit:
            maybe_warn_verl_override(verl_path, rllm_path)
            OmegaConf.update(config, rllm_path, to_rllm(OmegaConf.select(config, verl_path)), merge=False)
        else:
            OmegaConf.update(config, verl_path, to_verl(OmegaConf.select(config, rllm_path)), merge=False)

    def sync_do_sample_temperature() -> None:
        """Force temperature to 0.0 when do_sample is False, on both sides."""
        for do_sample_path, train_temp_paths in (
            ("actor_rollout_ref.rollout.do_sample", ("rllm.rollout.train.temperature", "actor_rollout_ref.rollout.temperature")),
            ("actor_rollout_ref.rollout.val_kwargs.do_sample", ("rllm.rollout.val.temperature", "actor_rollout_ref.rollout.val_kwargs.temperature")),
        ):
            if OmegaConf.select(config, do_sample_path) is False:
                for path in train_temp_paths:
                    OmegaConf.update(config, path, 0.0, merge=False)

    def sync_clip_ratio() -> None:
        eps_clip_path = "rllm.algorithm.eps_clip"
        clip_ratio_path = "actor_rollout_ref.actor.clip_ratio"
        clip_ratio_low_path = "actor_rollout_ref.actor.clip_ratio_low"

        if eps_clip_path in explicit:
            eps_clip = OmegaConf.select(config, eps_clip_path)
            for verl_path in (clip_ratio_low_path, clip_ratio_path):
                maybe_warn_verl_override(
                    verl_path,
                    eps_clip_path,
                    conflict=not same(OmegaConf.select(config, verl_path), eps_clip),
                )
            OmegaConf.update(config, clip_ratio_path, eps_clip, merge=False)
            OmegaConf.update(config, clip_ratio_low_path, eps_clip, merge=False)
        elif clip_ratio_low_path in explicit:
            maybe_warn_verl_override(clip_ratio_low_path, eps_clip_path)
            if clip_ratio_path in explicit:
                maybe_warn_verl_override(clip_ratio_path, eps_clip_path)
            OmegaConf.update(config, eps_clip_path, OmegaConf.select(config, clip_ratio_low_path), merge=False)
        elif clip_ratio_path in explicit:
            maybe_warn_verl_override(clip_ratio_path, eps_clip_path)
            OmegaConf.update(config, eps_clip_path, OmegaConf.select(config, clip_ratio_path), merge=False)
        else:
            eps_clip = OmegaConf.select(config, eps_clip_path)
            if eps_clip is not None:
                OmegaConf.update(config, clip_ratio_path, eps_clip, merge=False)
                OmegaConf.update(config, clip_ratio_low_path, eps_clip, merge=False)

        eps_clip = OmegaConf.select(config, eps_clip_path)
        eps_clip_high = OmegaConf.select(config, "rllm.algorithm.eps_clip_high")
        if eps_clip_high is None:
            eps_clip_high = eps_clip
        if eps_clip_high is not None:
            OmegaConf.update(config, "actor_rollout_ref.actor.clip_ratio_high", eps_clip_high, merge=False)

    def sync_lr_schedule() -> None:
        optim = OmegaConf.select(config, "actor_rollout_ref.actor.optim")
        if optim is None:
            return
        # different training backends use a different config key
        for key in ("lr_scheduler_type", "lr_decay_style", "decay_type"):
            if key in optim:
                sync_shared_keys(config, [(f"actor_rollout_ref.actor.optim.{key}", "rllm.algorithm.lr_schedule")], explicit=explicit, on_native_override=warn_verl_override)
                return

    sync_shared_keys(config, _SHARED_KEYS, explicit=explicit, on_native_override=warn_verl_override)

    sync_lr_schedule()
    sync_total_training_steps()

    # Derived verl-only keys
    if "actor_rollout_ref.actor.use_kl_loss" not in explicit:
        kl_beta = OmegaConf.select(config, "rllm.algorithm.kl_beta")
        if kl_beta is None:
            kl_beta = 0.0
        OmegaConf.update(config, "actor_rollout_ref.actor.use_kl_loss", kl_beta > 0, merge=False)

    # Router replay: derive verl's rollout-side flag from the rllm mode (R3 records at rollout).
    router_replay_mode = config.rllm.algorithm.get("router_replay", "disabled")
    if router_replay_mode == "R3":
        OmegaConf.update(config, "actor_rollout_ref.rollout.enable_rollout_routing_replay", True, merge=False)

    # clip_ratio family: verl uses clip_ratio_{low,high} when set, else falls back to clip_ratio.
    # Mirror the effective low bound to/from rllm.algorithm.eps_clip.
    sync_clip_ratio()

    # When do_sample=False, force the effective sampling temperature to 0.0
    sync_do_sample_temperature()

    # Async / separated mode toggles. Colocated needs hybrid_engine + naive ckpt;
    # separated needs hybrid_engine=False + nccl ckpt + the async_training mini-batch sizing.
    is_separated = config.rllm.get("async_training", {}).get("enable", False)
    rollout = config.actor_rollout_ref.rollout
    actor = config.actor_rollout_ref.actor
    ckpt_backend = OmegaConf.select(rollout, "checkpoint_engine.backend")
    if is_separated:
        config.actor_rollout_ref.hybrid_engine = False
        if ckpt_backend == "naive":
            logger.info("Async training enabled; overriding checkpoint_engine.backend 'naive' → 'nccl' (naive is colocated-only).")
            rollout.checkpoint_engine.backend = "nccl"
        actor.ppo_mini_batch_size = config.rllm.async_training.mini_batch_size
        config.async_training.partial_rollout = config.rllm.async_training.partial_rollout
    else:
        config.actor_rollout_ref.hybrid_engine = True
        if ckpt_backend is not None and ckpt_backend != "naive":
            logger.info(f"Sync training; overriding checkpoint_engine.backend '{ckpt_backend}' → 'naive' (hybrid engine requires naive).")
            rollout.checkpoint_engine.backend = "naive"


CheckpointKind = Literal["transient", "periodic", "final"]
_TRAINING_CHECKPOINT_PATTERN = re.compile(r"^global_step_(\d+)$")


def _checkpoint_async_save_enabled(config: DictConfig) -> bool:
    ckpt_cfg = config.actor_rollout_ref.actor.get("checkpoint", {})
    if isinstance(ckpt_cfg, dict):
        return bool(ckpt_cfg.get("async_save", False))
    return bool(getattr(ckpt_cfg, "async_save", False))


def _write_latest_checkpoint_marker(root: str, global_steps: int) -> None:
    """Atomically publish the newest fully-written recovery checkpoint."""
    os.makedirs(root, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".latest_checkpointed_iteration.",
        suffix=".tmp",
        dir=root,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(str(global_steps))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, os.path.join(root, "latest_checkpointed_iteration.txt"))
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _save_dataloader_state_atomic(path: str, state: Any) -> None:
    """Write dataloader state without exposing a partially rewritten file."""
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=parent,
    )
    os.close(descriptor)
    try:
        torch.save(state, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _training_checkpoint_steps(root: str) -> dict[int, str]:
    try:
        names = os.listdir(root)
    except FileNotFoundError:
        return {}
    checkpoints: dict[int, str] = {}
    for name in names:
        match = _TRAINING_CHECKPOINT_PATTERN.fullmatch(name)
        if match is None:
            continue
        path = os.path.join(root, name)
        if os.path.isdir(path):
            checkpoints[int(match.group(1))] = path
    return checkpoints


def _prune_training_checkpoints(
    root: str,
    *,
    latest_step: int,
    save_freq: int,
    max_periodic_to_keep: Any,
) -> list[int]:
    """Keep periodic history plus only the latest interval's recovery steps."""
    if save_freq <= 0:
        return []

    checkpoints = _training_checkpoint_steps(root)
    committed_steps = sorted(step for step in checkpoints if step <= latest_step)
    periodic_steps = [step for step in committed_steps if step % save_freq == 0]
    try:
        periodic_limit = int(max_periodic_to_keep) if max_periodic_to_keep is not None else None
    except (TypeError, ValueError):
        periodic_limit = None
    if periodic_limit is not None and periodic_limit > 0:
        periodic_steps = periodic_steps[-periodic_limit:]

    interval_start = latest_step - (latest_step % save_freq) + 1
    transient_steps = {
        step
        for step in committed_steps
        if interval_start <= step <= latest_step and step % save_freq != 0
    }
    keep = set(periodic_steps) | transient_steps
    removed: list[int] = []
    root_real = os.path.realpath(root)
    for step in committed_steps:
        if step in keep:
            continue
        path = checkpoints[step]
        if os.path.commonpath([root_real, os.path.realpath(path)]) != root_real:
            logger.warning("Refusing to prune checkpoint outside training root: %s", path)
            continue
        try:
            shutil.rmtree(path)
        except OSError:
            logger.warning("Failed to prune training checkpoint %s; will retry after a later save", path, exc_info=True)
        else:
            removed.append(step)
    if removed:
        logger.info("Pruned training checkpoints after step %d: %s", latest_step, removed)
    return removed


def save_checkpoint(
    config: DictConfig,
    global_steps: int,
    actor_rollout_wg,
    train_dataloader=None,
    *,
    checkpoint_kind: CheckpointKind = "periodic",
) -> None:
    """Save one committed actor/dataloader checkpoint and enforce retention."""
    from verl.utils.fs import local_mkdir_safe

    if checkpoint_kind not in {"transient", "periodic", "final"}:
        raise ValueError(f"unsupported checkpoint_kind={checkpoint_kind!r}")
    save_freq = int(config.trainer.save_freq)
    async_save = _checkpoint_async_save_enabled(config)
    if async_save and save_freq > 0:
        raise ValueError(
            "actor_rollout_ref.actor.checkpoint.async_save=true is incompatible "
            "with committed per-step recovery checkpoints"
        )

    training_root = os.path.abspath(str(config.trainer.default_local_dir))
    local_global_step_folder = os.path.join(training_root, f"global_step_{global_steps}")
    print(f"local_global_step_folder: {local_global_step_folder}")

    actor_local_path = os.path.join(local_global_step_folder, "actor")
    actor_remote_path = None if config.trainer.default_hdfs_dir is None else os.path.join(config.trainer.default_hdfs_dir, f"global_step_{global_steps}", "actor")

    remove_previous = config.trainer.get("remove_previous_ckpt_in_save", False)
    if remove_previous:
        print("Warning: remove_previous_ckpt_in_save is deprecated, set max_actor_ckpt_to_keep=1 instead")
    max_actor_ckpt = config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous else 1
    include_hf_model = checkpoint_kind != "transient"

    # RLLM owns retention because transient recovery checkpoints must not count
    # against the periodic checkpoint limit maintained by VERL. When rolling
    # saves are disabled, preserve VERL's existing final-checkpoint retention.
    actor_rollout_wg.save_checkpoint(
        actor_local_path,
        actor_remote_path,
        global_steps,
        max_ckpt_to_keep=None if save_freq > 0 else max_actor_ckpt,
        save_hf_model=include_hf_model,
    )

    if train_dataloader is not None:
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        _save_dataloader_state_atomic(
            dataloader_local_path,
            train_dataloader.state_dict(),
        )

    hf_save_dir = _resolve_hf_checkpoint_save_dir(config)
    if include_hf_model and hf_save_dir:
        max_hf_ckpt = config.trainer.get("max_hf_ckpt_to_keep", max_actor_ckpt)
        hf_save_dtype = _resolve_hf_checkpoint_save_dtype(config)
        hf_save_max_shard_size = _resolve_hf_checkpoint_max_shard_size(config)
        _copy_hf_checkpoint_for_step(
            actor_local_path,
            hf_save_dir,
            global_steps,
            max_to_keep=max_hf_ckpt,
            save_dtype=hf_save_dtype,
            max_shard_size=hf_save_max_shard_size,
        )

    if not async_save:
        _write_latest_checkpoint_marker(training_root, global_steps)
        _prune_training_checkpoints(
            training_root,
            latest_step=global_steps,
            save_freq=save_freq,
            max_periodic_to_keep=max_actor_ckpt,
        )


def load_checkpoint(
    config: DictConfig,
    actor_rollout_wg,
    train_dataloader=None,
) -> int:
    """Load checkpoint and return global step to resume from (0 if training from scratch)."""
    from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path

    if config.trainer.resume_mode == "disable":
        return 0

    if config.trainer.default_hdfs_dir is not None:
        raise NotImplementedError("load from hdfs is not implemented yet")

    checkpoint_folder = config.trainer.default_local_dir
    if not os.path.isabs(checkpoint_folder):
        checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
    global_step_folder = find_latest_ckpt_path(checkpoint_folder)

    if config.trainer.resume_mode == "auto":
        if global_step_folder is None:
            print("Training from scratch")
            return 0
    elif config.trainer.resume_mode == "resume_path":
        assert isinstance(config.trainer.resume_from_path, str), "resume ckpt must be str type"
        assert "global_step_" in config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
        global_step_folder = config.trainer.resume_from_path
        if not os.path.isabs(global_step_folder):
            global_step_folder = os.path.join(os.getcwd(), global_step_folder)

    print(f"Load from checkpoint folder: {global_step_folder}")
    global_steps = int(global_step_folder.split("global_step_")[-1])
    print(f"Setting global step to {global_steps}")

    actor_path = os.path.join(global_step_folder, "actor")
    actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=config.trainer.del_local_ckpt_after_load)

    if train_dataloader is not None:
        dataloader_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_path):
            state_dict = torch.load(dataloader_path, weights_only=False)
            train_dataloader.load_state_dict(state_dict)
            pending_count = len(state_dict.get("pending_dispatches", []))
            print(
                "Loaded dataloader state from "
                f"{dataloader_path}: epoch={state_dict.get('epoch')} "
                f"cursor={state_dict.get('cursor')} "
                f"schema_version={state_dict.get('schema_version', 1)} "
                f"pending_replay={pending_count}"
            )
        else:
            print(f"Warning: No dataloader state found at {dataloader_path}, will start from scratch")

    return global_steps


def balance_batch(
    batch: DataProto,
    actor_rollout_wg,
    metrics: dict,
    use_prefix_grouper: bool = False,
    logging_prefix: str = "global_seqlen",
) -> None:
    """Reorder the batch so each DP rank gets similar total tokens.

    Mutates ``batch`` in-place via ``batch.reorder()``. Mirrors the semantics of
    ``RayPPOTrainer._balance_batch``.
    """
    attention_mask = batch.batch["attention_mask"]
    batch_size = attention_mask.shape[0]
    global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1)
    workload_lst = calculate_workload(global_seqlen_lst)

    role_key = "actor"
    if role_key not in actor_rollout_wg._dispatch_info:
        dp_rank_mapping = actor_rollout_wg._query_dispatch_info(role_key)
        actor_rollout_wg._dispatch_info[role_key] = dp_rank_mapping
    else:
        dp_rank_mapping = actor_rollout_wg._dispatch_info[role_key]
    dp_size = max(dp_rank_mapping) + 1

    if use_prefix_grouper and "uid" in batch.non_tensor_batch:
        from verl.utils.seqlen_balancing import get_group_balanced_partitions

        uid_list = list(batch.non_tensor_batch["uid"])
        seqlen_list = global_seqlen_lst.tolist()
        num_groups = len(set(uid_list))
        if num_groups % dp_size != 0:
            raise ValueError(f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) % dp_size ({dp_size}) == 0.")
        global_partition_lst = get_group_balanced_partitions(
            seqlen_list=seqlen_list,
            uid_list=uid_list,
            k_partitions=dp_size,
        )
    else:
        global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)

    if not use_prefix_grouper:
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (workload_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition

    global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
    batch.reorder(global_idx)
    global_balance_stats = log_seqlen_unbalance(
        seqlen_list=global_seqlen_lst.tolist(),
        partitions=global_partition_lst,
        prefix=logging_prefix,
    )
    metrics.update(global_balance_stats)


def start_profiling(global_steps, actor_rollout_wg, ref_policy_wg=None, use_reference_policy=False) -> None:
    actor_rollout_wg.start_profile(role="e2e", profile_step=global_steps)
    if use_reference_policy and ref_policy_wg is not None:
        ref_policy_wg.start_profile(profile_step=global_steps)


def stop_profiling(actor_rollout_wg, ref_policy_wg=None, use_reference_policy=False) -> None:
    actor_rollout_wg.stop_profile()
    if use_reference_policy and ref_policy_wg is not None:
        ref_policy_wg.stop_profile()


def build_wg_kwargs(config: DictConfig, device_name: str) -> dict[str, Any]:
    """Build kwargs for RayWorkerGroup construction."""
    wg_kwargs: dict[str, Any] = {"device_name": device_name}
    if OmegaConf.select(config.trainer, "ray_wait_register_center_timeout") is not None:
        wg_kwargs["ray_wait_register_center_timeout"] = config.trainer.ray_wait_register_center_timeout
    if OmegaConf.select(config, "global_profiler.steps") is not None:
        wg_kwargs["profile_steps"] = OmegaConf.select(config, "global_profiler.steps")
        if OmegaConf.select(config, "global_profiler.tool") == "nsys":
            assert OmegaConf.select(config, "global_profiler.global_tool_config.nsys.worker_nsight_options") is not None
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(OmegaConf.select(config, "global_profiler.global_tool_config.nsys.worker_nsight_options"))
    return wg_kwargs
