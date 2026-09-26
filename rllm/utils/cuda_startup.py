"""Validate the assigned CUDA device before VERL imports engines or joins ranks."""

from __future__ import annotations

import json
import os
import socket
import sys

from rllm.utils.diagnostic_events import emit_diagnostic

CUDA_STARTUP_REVISION = 1


def initialize_cuda_worker() -> None:
    """Initialize only this worker's visible local device; never remap or retry it.

    NVML/device enumeration alone cannot establish CUDA runtime health. The
    allocation and synchronization also detect failures before NCCL collectives.
    This runs in GPU workers after Ray/VERL have established their visibility.
    """
    import torch

    evidence = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "rank": os.environ.get("RANK"),
        "local_rank": os.environ.get("LOCAL_RANK", "0"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
    }
    stage = "select_device"
    try:
        device = int(evidence["local_rank"])
        torch.cuda.set_device(device)
        stage = "allocate_and_fill"
        probe = torch.empty(1, device=f"cuda:{device}")
        probe.zero_()
        stage = "synchronize"
        torch.cuda.synchronize(device)
        del probe
    except Exception as exc:
        evidence.update(stage=stage, error_type=type(exc).__name__, error=str(exc))
        try:
            ray = sys.modules.get("ray")
            if ray is not None and ray.is_initialized():
                context = ray.get_runtime_context()
                evidence["ray_node_id"] = context.get_node_id()
                evidence["ray_accelerator_ids"] = context.get_accelerator_ids()
        except Exception:
            pass
        # Reporting must not replace the first CUDA failure, even if the
        # diagnostic sink cannot start on a damaged node.
        try:
            emit_diagnostic("CUDA_WORKER_INITIALIZATION_FAILED", **evidence)
        except Exception:
            pass
        raise RuntimeError(
            "CUDA worker initialization failed before engine import/collectives: "
            + json.dumps(evidence, ensure_ascii=True, sort_keys=True)
        ) from exc
    try:
        emit_diagnostic("CUDA_WORKER_INITIALIZED", **evidence)
    except Exception:
        pass
