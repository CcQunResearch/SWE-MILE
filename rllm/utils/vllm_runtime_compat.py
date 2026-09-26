"""Build-time backports for the pinned vLLM runtime.

Backport vllm-project/vllm#34571 without changing scheduler capacity, model
context, or attention kernels. The patch caps graph capture sizes after the
actual Mamba cache allocation and before graph dispatch keys are initialized.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import subprocess
from pathlib import Path


def apply_mamba_cudagraph_patch() -> str:
    distribution = importlib.metadata.distribution("vllm")
    if distribution.version != "0.17.0":
        raise RuntimeError(f"Review the Mamba CUDA graph backport for vLLM {distribution.version}; expected 0.17.0")
    root = Path(distribution.locate_file(""))
    patch = Path(__file__).resolve().parents[1] / "patches" / "vllm-0.17.0-mamba-cudagraph-capacity.patch"
    command = ["git", "-C", str(root), "apply"]
    # Runtime images may be built over an already patched base image.
    applied = subprocess.run([*command, "--reverse", "--check", str(patch)], capture_output=True, text=True)
    if applied.returncode == 0:
        return "already applied"
    check = subprocess.run([*command, "--check", str(patch)], capture_output=True, text=True)
    if check.returncode:
        raise RuntimeError(f"vLLM source does not match the reviewed Mamba backport: {check.stderr}")
    subprocess.run([*command, str(patch)], check=True)
    subprocess.run([*command, "--reverse", "--check", str(patch)], check=True)
    return "applied"


def apply_pinned_memory_patches() -> dict[str, str]:
    """Patch the two reviewed host allocation sites in the pinned runtime.

    Installed source is patched so Ray's spawned EngineCore and Worker
    processes, as well as Inductor's lazy imports, all receive the fix.
    """
    patches = (
        ("vllm", "0.17.0", "vllm-0.17.0-pinned-memory-fallback.patch"),
        ("torch", "2.10.0", "torch-2.10.0-autotune-pinned-memory-fallback.patch"),
    )
    pending = []
    results = {}
    # Validate both sources before modifying either package.
    for name, version, filename in patches:
        distribution = importlib.metadata.distribution(name)
        if distribution.version.split("+", 1)[0] != version:
            raise RuntimeError(
                f"Review the pinned-memory fallback for {name} {distribution.version}; expected {version}"
            )
        root = Path(distribution.locate_file(""))
        patch = Path(__file__).resolve().parents[1] / "patches" / filename
        command = ["git", "-C", str(root), "apply"]
        applied = subprocess.run([*command, "--reverse", "--check", str(patch)], capture_output=True, text=True)
        if applied.returncode == 0:
            results[name] = "already applied"
            continue
        check = subprocess.run([*command, "--check", str(patch)], capture_output=True, text=True)
        if check.returncode:
            raise RuntimeError(f"{name} source does not match the reviewed pinned-memory fallback: {check.stderr}")
        pending.append((name, command, patch))
    for name, command, patch in pending:
        subprocess.run([*command, str(patch)], check=True)
        subprocess.run([*command, "--reverse", "--check", str(patch)], check=True)
        results[name] = "applied"
    return results


def apply_training_runtime_patches() -> dict[str, str]:
    """Version/source-checked FSDP fallback and Ray lock observation."""
    targets = (
        ("torch_fsdp", "torch", "2.10.0", "torch-2.10.0-fsdp-pinned-memory-fallback.patch"),
        ("ray_lock", "ray", "2.54.0", "ray-2.54.0-deserialization-lock-diagnostics.patch"),
    )
    pending, result = [], {}
    for key, package, version, filename in targets:
        distribution = importlib.metadata.distribution(package)
        if distribution.version.split("+", 1)[0] != version:
            raise RuntimeError(f"Review training runtime patch for {package} {distribution.version}; expected {version}")
        patch = Path(__file__).resolve().parents[1] / "patches" / filename
        command = ["git", "-C", str(distribution.locate_file("")), "apply"]
        if subprocess.run([*command, "--reverse", "--check", str(patch)], capture_output=True).returncode == 0:
            result[key] = "already applied"
            continue
        check = subprocess.run([*command, "--check", str(patch)], capture_output=True, text=True)
        if check.returncode:
            raise RuntimeError(f"{package} source does not match reviewed training runtime patch: {check.stderr}")
        pending.append((key, command, patch))
    for key, command, patch in pending:
        subprocess.run([*command, str(patch)], check=True)
        subprocess.run([*command, "--reverse", "--check", str(patch)], check=True)
        result[key] = "applied"
    return result


def verify_qwen_runtime() -> None:
    """Verify the shared Dense/MoE stack without requiring a GPU or weights."""
    import inspect

    import causal_conv1d  # noqa: F401
    import cupy
    import flash_attn  # noqa: F401
    import flashinfer  # noqa: F401
    import tilelang  # noqa: F401
    import torch
    from fla.ops.common.chunk_o import chunk_bwd_dqkwg  # noqa: F401
    from renderers import config_from_name
    from transformers import AutoConfig, AutoModelForImageTextToText
    from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5_moe import Qwen3_5MoeForConditionalGeneration
    from verl.models.transformers.monkey_patch import apply_monkey_patch
    from verl.workers.rollout.llm_server import LLMServerClient, LLMServerManager  # noqa: F401
    from vllm.model_executor.models.registry import ModelRegistry

    for package, expected in (("transformers", "5.8.0"), ("vllm", "0.17.0"), ("renderers", "0.1.11"),
                              ("apache-tvm-ffi", "0.1.9"), ("flash-linear-attention", "0.5.1")):
        actual = importlib.metadata.version(package)
        assert actual == expected, f"{package} must be {expected}, got {actual}"
    assert importlib.metadata.version("verl").startswith("0.8.0")
    assert torch.__version__.startswith("2.10.") and torch.version.cuda == "12.9"
    assert cupy.__version__ == "13.6.0"
    for family in ("qwen3.5", "qwen3.6", "qwen3.8"):
        assert config_from_name(family).name == family
    for model_type, cls in (("qwen3_5", Qwen3_5ForConditionalGeneration), ("qwen3_5_moe", Qwen3_5MoeForConditionalGeneration)):
        assert AutoModelForImageTextToText._model_mapping[type(AutoConfig.for_model(model_type))] is cls
        assert cls.__name__ in ModelRegistry.get_supported_archs()
    assert "Qwen3_5MoeForConditionalGeneration" in inspect.getsource(apply_monkey_patch)
    print("Qwen3.5/3.6/3.8 Dense/MoE, VERL API and CUDA dependencies verified")


def apply_optimizer_offload_patch() -> str:
    """Patch the installed (possibly editable) VERL tree, never an import alias."""
    return _apply_verl_offload_patch("verl-0.8.0-optimizer-offload.patch")


def apply_fsdp_parameter_offload_patch() -> str:
    """Cover the model parameter transfer used during FSDP initialization."""
    return _apply_verl_offload_patch("verl-0.8.0-fsdp-parameter-offload.patch")


def apply_engine_startup_patch() -> str:
    """Keep FSDP isolated from optional backends and diagnose worker CUDA init."""
    return _apply_verl_offload_patch("verl-0.8.0-engine-startup.patch")


def _apply_verl_offload_patch(filename: str) -> str:
    version = importlib.metadata.version("verl")
    if version not in ("0.8.0", "0.8.0.dev0"):
        raise RuntimeError(f"Review {filename} for VERL {version}; expected 0.8.0 source")
    spec = importlib.util.find_spec("verl")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Cannot locate installed VERL source")
    root = Path(next(iter(spec.submodule_search_locations))).parent
    patch = Path(__file__).resolve().parents[1] / "patches" / filename
    command = ["git", "-C", str(root), "apply"]
    if subprocess.run([*command, "--reverse", "--check", str(patch)], capture_output=True).returncode == 0:
        return "already applied"
    check = subprocess.run([*command, "--check", str(patch)], capture_output=True, text=True)
    if check.returncode:
        raise RuntimeError(f"VERL source does not match reviewed {filename}: {check.stderr}")
    subprocess.run([*command, str(patch)], check=True)
    subprocess.run([*command, "--reverse", "--check", str(patch)], check=True)
    return "applied"


if __name__ == "__main__":
    print(f"vLLM Mamba CUDA graph capacity backport: {apply_mamba_cudagraph_patch()}")
    print(f"vLLM/Inductor pinned-memory fallback: {apply_pinned_memory_patches()}")
    print(f"FSDP/Ray training runtime patches: {apply_training_runtime_patches()}")
    print(f"VERL optimizer offload patch: {apply_optimizer_offload_patch()}")
    print(f"VERL FSDP parameter offload patch: {apply_fsdp_parameter_offload_patch()}")
    print(f"VERL engine startup patch: {apply_engine_startup_patch()}")
