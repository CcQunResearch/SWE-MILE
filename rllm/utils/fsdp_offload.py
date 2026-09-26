"""Explicit, checked host allocation for VERL's FSDP1 parameter offload."""
from __future__ import annotations

import torch

from rllm.utils.diagnostic_events import emit_diagnostic
from rllm.utils.pinned_memory import tensor_diagnostics

FSDP_OFFLOAD_REVISION = 1


def _diagnostic(event, value, stage, error=None):
    try:
        fields = tensor_diagnostics(value, purpose="parameter_shard")
        if error is not None:
            fields.update(error_type=type(error).__name__, error=str(error)[:1500])
        emit_diagnostic(event, stage=stage, **fields)
    except Exception:
        pass


@torch.no_grad()
def offload_fsdp_handle(handle):
    """Publish a CPU shard only after its copy succeeds; never retry CUDA copies.

    This matches FlatParamHandle.flat_param_to's view refresh contract in
    PyTorch 2.10. Pinned-host allocation is separate from the D2H operation,
    so cudaErrorInvalidValue is recoverable only before any copy is issued.
    """
    flat_param = handle.flat_param
    value = flat_param.data
    if value.device.type == "cpu":
        return
    if type(value) is not torch.Tensor or value.device.type != "cuda" or value.layout != torch.strided:
        handle.flat_param_to(torch.device("cpu"), non_blocking=False)
        return
    stage = "pre_offload_sync"
    try:
        torch.cuda.synchronize(value.device)
        stage = "host_allocation"
        try:
            target = torch.empty_like(value, device="cpu", pin_memory=True, memory_format=torch.preserve_format)
        except RuntimeError as error:
            if str(error).partition("\n")[0].strip() != "CUDA error: invalid argument":
                raise
            stage = "post_allocation_failure_sync"
            torch.cuda.synchronize(value.device)
            stage = "pageable_host_allocation"
            target = torch.empty_like(value, device="cpu", pin_memory=False, memory_format=torch.preserve_format)
            _diagnostic("FSDP_PARAMETER_PIN_MEMORY_FALLBACK", value, stage, error)
        stage = "device_to_host_copy"
        target.copy_(value, non_blocking=target.is_pinned())
        stage = "post_offload_sync"
        torch.cuda.synchronize(value.device)
    except BaseException as error:
        _diagnostic("FSDP_PARAMETER_TRANSFER_ERROR", value, stage, error)
        raise
    flat_param.data = target
    if handle._use_orig_params:
        if handle.is_sharded(flat_param):
            handle._use_sharded_views()
        else:
            handle._use_unsharded_views(as_params=True)
