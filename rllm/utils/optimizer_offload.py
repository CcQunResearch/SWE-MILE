"""Reviewed VERL optimizer transfers; only host allocation may fall back."""
from __future__ import annotations

import torch

from rllm.utils.diagnostic_events import emit_diagnostic
from rllm.utils.pinned_memory import tensor_diagnostics


def _ordinary_cuda(value):
    # DTensor and other subclasses keep their original dispatch semantics.
    return type(value) is torch.Tensor and value.device.type == "cuda" and value.layout == torch.strided


def _entries(optimizer):
    for group in optimizer.param_groups:
        for param in group["params"]:
            for key, value in optimizer.state[param].items():
                if isinstance(value, torch.Tensor):
                    yield optimizer.state[param], key, value


def _failure(value, key, stage, error):
    try:
        if getattr(error, "_rllm_optimizer_diagnostic_reported", False):
            return
        error._rllm_optimizer_diagnostic_reported = True
        emit_diagnostic("OPTIMIZER_TRANSFER_ERROR", stage=stage, state_key=str(key),
                        error_type=type(error).__name__, error=str(error)[:1500],
                        **tensor_diagnostics(value, purpose="optimizer_state"))
    except Exception:
        pass


def _host_buffer(value, key):
    # No copy has been issued when this allocation throws. A copy error must
    # never be mistaken for an allocation error or retried on another buffer.
    try:
        return torch.empty_like(value, device="cpu", pin_memory=True, memory_format=torch.preserve_format)
    except RuntimeError as error:
        if str(error).partition("\n")[0].strip() != "CUDA error: invalid argument":
            raise
        try:
            torch.cuda.synchronize(value.device)
        except BaseException as sync_error:
            _failure(value, key, "post_allocation_failure_sync", sync_error)
            raise
        result = torch.empty_like(value, device="cpu", pin_memory=False, memory_format=torch.preserve_format)
        emit_diagnostic("OPTIMIZER_PIN_MEMORY_FALLBACK", state_key=str(key),
                        **tensor_diagnostics(value, purpose="optimizer_state"))
        return result


@torch.no_grad()
def offload_optimizer_state(optimizer):
    if not optimizer.state:
        return
    entries = list(_entries(optimizer))
    devices = {}
    for _, key, value in entries:
        if _ordinary_cuda(value):
            devices.setdefault(value.device, (key, value))
    for device, (key, value) in devices.items():
        try:
            torch.cuda.synchronize(device)
        except BaseException as error:
            _failure(value, key, "pre_offload_sync", error)
            raise
    replacements = []
    # Retain original GPU references until every transfer has completed. On
    # failure, no partially offloaded optimizer is published for checkpointing.
    for state, key, value in entries:
        if value.device.type == "cpu":
            continue
        stage = "host_allocation"
        try:
            if _ordinary_cuda(value):
                target = _host_buffer(value, key)
                stage = "device_to_host_copy"
                target.copy_(value, non_blocking=target.is_pinned())
            else:
                stage = "native_offload"
                target = value.to("cpu", non_blocking=True)
            replacements.append((state, key, target))
        except BaseException as error:
            _failure(value, key, stage, error)
            raise
    for device, (key, value) in devices.items():
        try:
            torch.cuda.synchronize(device)
        except BaseException as error:
            _failure(value, key, "post_offload_sync", error)
            raise
    for state, key, target in replacements:
        state[key] = target


@torch.no_grad()
def load_optimizer_state(optimizer, device_id):
    if not optimizer.state:
        return
    for state, key, value in _entries(optimizer):
        non_blocking = True
        if type(value) is torch.Tensor and value.device.type == "cpu" and value.layout == torch.strided:
            non_blocking = value.is_pinned()
        try:
            state[key] = value.to(device_id, non_blocking=non_blocking)
        except BaseException as error:
            _failure(value, key, "host_to_device_copy", error)
            raise


def cleanup_preserving_error(steps, primary_error):
    """Run each exit step; retain an earlier body/cleanup exception."""
    first = primary_error
    for stage, function in steps:
        try:
            function()
        except BaseException as error:
            emit_diagnostic("ENGINE_CLEANUP_ERROR", stage=stage, error_type=type(error).__name__, error=str(error)[:1500])
            if first is None:
                first = error
            elif first is not error:
                detail = {"stage": stage, "type": type(error).__name__, "message": str(error)[:1500]}
                try:
                    errors = getattr(first, "cleanup_errors", [])
                    first.cleanup_errors = [*errors, detail]
                    first.add_note(f"Additional engine cleanup error at {stage}: {type(error).__name__}: {str(error)[:500]}")
                except Exception:
                    pass
    if primary_error is None and first is not None:
        raise first
