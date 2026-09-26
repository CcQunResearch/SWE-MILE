"""Startup policy for full-weight, separated VERL training.

The engine kwargs are deliberately set on a rollout-only config copy: VERL
0.8 changes standalone ``rollout.load_format=dummy`` back to auto, but applies
engine kwargs afterwards. No global vLLM/VERL behaviour is patched.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path

from omegaconf import OmegaConf, open_dict

logger = logging.getLogger(__name__)


def rollout_startup_config(config):
    """Return an isolated server config; only full FSDP sync may skip disk IO.

    The caller must await checkpoint restoration and the initial weight sync
    before exposing these servers to training or validation.
    """
    result = deepcopy(config)
    ar = result.actor_rollout_ref
    model, rollout, actor = ar.model, ar.rollout, ar.actor
    options = result.rllm.get("async_training", {})
    engine = OmegaConf.select(result, "actor_rollout_ref.rollout.engine_kwargs.vllm") or {}
    reason = None
    if not options.get("enable", False) or not options.get("skip_initial_rollout_weights", True):
        reason = "disabled"
    elif rollout.get("name") != "vllm" or actor.get("strategy") not in ("fsdp", "fsdp2"):
        reason = "requires vLLM and full FSDP weight synchronization"
    elif rollout.get("checkpoint_engine", {}).get("backend") != "nccl":
        reason = "requires NCCL checkpoint engine"
    elif model.get("lora_rank", 0) or model.get("lora", {}).get("rank", 0) or model.get("lora_adapter_path"):
        reason = "adapter training"
    elif rollout.get("quantization") or engine.get("quantization") or actor.get("fsdp_config", {}).get("qat", {}).get("enable", False):
        reason = "quantized weights"
    elif (model.get("mtp") or {}).get("enable", False) or (rollout.get("mtp") or {}).get("enable", False) or engine.get("speculative_config"):
        reason = "draft/MTP weights"
    elif engine.get("load_format") is not None or rollout.get("load_format", "dummy") not in ("auto", "dummy"):
        reason = "explicit weight loader"
    elif engine.get("hf_overrides") or model.get("external_lib"):
        reason = "custom model implementation"
    if reason is None:
        # Initially enable only the Qwen3.5-family implementations used by SWE
        # (Qwen3.6-35B-A3B also uses qwen3_5_moe). Unknown/quantized snapshots
        # keep their loader rather than assuming a complete compatible sync.
        try:
            hf_path = model.get("hf_config_path") or model.path
            hf = json.loads((Path(hf_path) / "config.json").read_text())
            if hf.get("model_type") not in {"qwen3_5", "qwen3_5_moe"}:
                reason = "model implementation not validated for initial full sync"
            elif hf.get("quantization_config") or model.get("override_config", {}).get("quantization_config"):
                reason = "quantized model snapshot"
        except (OSError, ValueError, TypeError):
            reason = "model config unavailable"
    if reason is not None:
        logger.info("Rollout startup keeps checkpoint loading: %s", reason)
        return result
    with open_dict(result):
        OmegaConf.update(result, "actor_rollout_ref.rollout.engine_kwargs.vllm.load_format", "dummy", force_add=True)
    logger.info("Rollout startup uses dummy weights; initial full checkpoint sync is required before sampling")
    return result
