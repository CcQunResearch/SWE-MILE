"""Narrow fallbacks for reviewed CUDA host-memory pinning sites."""

from __future__ import annotations

import os
from typing import Any, Callable

from rllm.utils.diagnostic_events import emit_diagnostic


def allocate_pinned_cpu(factory: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Retry only a CPU pinned allocation rejected with cudaErrorInvalidValue.

    This is called at reviewed vLLM/Inductor buffer allocation sites, not a
    global torch patch. Callers must use blocking copies for the pageable
    result. OOM, device allocation errors, and all other failures propagate.
    A failed pageable allocation also propagates, retaining the pinning error
    in its exception context.
    """
    try:
        return factory(*args, **kwargs)
    except RuntimeError as exc:
        device = kwargs.get("device")
        device_type = getattr(device, "type", str(device).split(":", 1)[0])
        if (
            kwargs.get("pin_memory") is not True
            or device_type != "cpu"
            or "CUDA error: invalid argument" not in str(exc)
        ):
            raise
        result = factory(*args, **{**kwargs, "pin_memory": False})
        emit_diagnostic(
            "CUDA_PIN_MEMORY_FALLBACK", factory=getattr(factory, "__name__", type(factory).__name__),
            dtype=str(kwargs.get("dtype")), error=str(exc).partition("\n")[0],
        )
        return result


def pin_fsdp_cpu_tensor(tensor, *, purpose: str, device):
    """Reviewed FSDP-only fallback; a broken CUDA stream must still fail.

    Synchronize before pinning to surface earlier asynchronous kernel errors,
    and again after a rejected pin before permitting a pageable fallback.
    No new CPU allocation or retry of model computation is performed.
    """
    import torch

    if tensor.device.type != "cpu" or tensor.layout != torch.strided:
        raise ValueError("FSDP pinning fallback requires a strided CPU tensor")
    torch.cuda.synchronize(device)
    try:
        return tensor.pin_memory()
    except RuntimeError as exc:
        if str(exc).partition("\n")[0].strip() != "CUDA error: invalid argument":
            raise
        torch.cuda.synchronize(device)
        emit_diagnostic("FSDP_PIN_MEMORY_FALLBACK", **tensor_diagnostics(tensor, purpose=purpose, device=device))
        return tensor


def tensor_diagnostics(tensor, *, purpose, device=None):
    """Scalar metadata only; memory-limit I/O is deferred to the writer."""
    import torch
    rank = os.environ.get("RANK")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    return {"purpose": purpose, "bytes": tensor.numel() * tensor.element_size(),
            "dtype": str(tensor.dtype), "device": str(device or tensor.device),
            "rank": rank, "torch": torch.__version__, "cuda": torch.version.cuda}
