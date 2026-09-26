"""Small, import-safe runtime identity checks for evaluation replicas."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import socket
from pathlib import Path

SOURCE_PATHS = (
    "rllm/utils/runtime_provenance.py",
    "rllm/utils/cuda_startup.py",
    "rllm/patches/verl-0.8.0-engine-startup.patch",
    "rllm/utils/model_request_abort.py",
    "rllm/utils/vllm_runtime_compat.py",
    "rllm/utils/pinned_memory.py",
    "rllm/utils/diagnostic_events.py",
    "rllm/utils/rpc_diagnostics.py",
    "rllm/utils/optimizer_offload.py",
    "rllm/patches/verl-0.8.0-optimizer-offload.patch",
    "rllm/utils/ray_lock_diagnostics.py",
    "rllm/patches/torch-2.10.0-fsdp-pinned-memory-fallback.patch",
    "rllm/patches/ray-2.54.0-deserialization-lock-diagnostics.patch",
    "rllm/utils/infrastructure_diagnostics.py",
    "rllm/utils/node_diagnostics.py",
    "rllm/utils/shutdown.py",
    "rllm/sandbox/file_snapshot.py",
    "rllm/eval/verifier_artifacts.py",
    "rllm/types.py",
    "rllm/trainer/buffer.py",
    "rllm/trainer/algorithms/transform.py",
    "rllm/trainer/algorithms/config.py",
    "rllm/trainer/unified_trainer.py",
    "rllm/trainer/verl/patch.py",
    "rllm/trainer/verl/verl_launcher.py",
    "rllm/hf_checkpoint_keys.py",
    "rllm-model-gateway/src/rllm_model_gateway/proxy.py",
    "rllm-model-gateway/src/rllm_model_gateway/client.py",
    "rllm-model-gateway/src/rllm_model_gateway/server.py",
    "rllm-model-gateway/src/rllm_model_gateway/store/memory_store.py",
    "rllm-model-gateway/src/rllm_model_gateway/store/sqlite_store.py",
    "rllm-model-gateway/src/rllm_model_gateway/http_client.py",
    "rllm-model-gateway/src/rllm_model_gateway/control_transport.py",
    "rllm-model-gateway/src/rllm_model_gateway/token_accumulator.py",
    "rllm/patches/vllm-0.17.0-pinned-memory-fallback.patch",
    "rllm/patches/torch-2.10.0-autotune-pinned-memory-fallback.patch",
    "rllm/patches/vllm-0.17.0-mamba-cudagraph-capacity.patch",
    "rllm/sandbox/backends/k8s.py",
    "rllm/eval/script_evaluator.py",
    "rllm/eval/repo_generation.py",
    "rllm/sandbox/backends/minisandbox.py",
    "rllm/sandbox/structured_exec.py",
    "rllm/sandbox/verifier_assets.py",
    "rllm/utils/bounded_cleanup.py",
    "rllm/data/assets/denovoswe_verifier.py",
    "rllm/data/assets/denovoswe_clean.sh",
    "rllm/sandbox/nl2repo_setup.py",
    "rllm/sandbox/nl2repo_environment.py",
    "rllm/data/assets/nl2repo_environment.py",
    "rllm/eval/repo_generation_status.py",
    "rllm/sandbox/repo_generation_environment.py",
    "rllm/data/assets/repo_generation_environment.py",
    "rllm/sandbox/git_bundle.py",
    "rllm/eval/_resolution.py",
    "rllm/data/r2egym_builder.py",
    "rllm/data/swebench_pro_builder.py",
    "rllm/data/repo_generation_builder.py",
    "rllm/data/assets/doc2repo_verifier.sh",
    "rllm/eval/checkpoint_evaluation.py",
    "rllm/eval/scheduling.py",
    "rllm/eval/node_runtime.py",
    "rllm/eval/gateway_supervision.py",
    "rllm/engine/agentflow_engine.py",
    "rllm/gateway/manager.py",
    "rllm/hooks.py",
    "rllm/harnesses/swe_scaffold.py",
    "rllm/harnesses/repository_state.py",
    "rllm/harnesses/shadow_sandbox.py",
    "rllm/harnesses/partition_probe.py",
    "rllm/harnesses/non_python_verification.py",
    "rllm/trainer/verl/server_lifecycle.py",
    "launch/train.py",
    "launch/train.sh",
    "launch/eval.py",
    "launch/eval.sh",
)


def runtime_provenance(*, verify_build: bool = True) -> dict:
    root = Path(__file__).resolve().parents[2]
    sources = {name: root / name for name in SOURCE_PATHS}
    # Hash the package actually imported by this process, including wheel
    # installations whose files differ from a mounted source checkout.
    try:
        gateway_spec = importlib.util.find_spec("rllm_model_gateway")
    except (ImportError, ValueError):
        gateway_spec = None
    if gateway_spec and gateway_spec.origin:
        gateway_root = Path(gateway_spec.origin).parent
        prefix = "rllm-model-gateway/src/rllm_model_gateway/"
        for name in SOURCE_PATHS:
            if name.startswith(prefix):
                sources[name] = gateway_root / name.removeprefix(prefix)
    try:
        vllm_distribution = importlib.metadata.distribution("vllm")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        for name in ("vllm/config/compilation.py", "vllm/v1/worker/gpu_model_runner.py", "vllm/v1/utils.py", "vllm/v1/executor/multiproc_executor.py", "vllm/v1/worker/worker_base.py"):
            sources[name] = Path(vllm_distribution.locate_file(name))
    try:
        torch_distribution = importlib.metadata.distribution("torch")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        for name in ("torch/_inductor/runtime/triton_heuristics.py", "torch/distributed/fsdp/_flat_param.py", "torch/distributed/fsdp/_runtime_utils.py"):
            sources[name] = Path(torch_distribution.locate_file(name))
    try:
        distribution = importlib.metadata.distribution("ray")
        sources["ray/_private/worker.py"] = Path(distribution.locate_file("ray/_private/worker.py"))
    except importlib.metadata.PackageNotFoundError:
        pass
    for module in ("rllm.sandbox.minisandbox_runtime.oci_runtime", "rllm.sandbox.minisandbox_runtime.oci_helper"):
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            spec = None
        if spec and spec.origin:
            sources[module] = Path(spec.origin).resolve()
    try:
        spec = importlib.util.find_spec("verl")
        if spec and spec.submodule_search_locations:
            verl_root = Path(next(iter(spec.submodule_search_locations))).parent
            for name in ("verl/utils/fsdp_utils.py", "verl/workers/engine/base.py", "verl/workers/engine/__init__.py", "verl/workers/engine_workers.py", "verl/workers/engine/fsdp/transformer_impl.py"):
                sources[name] = verl_root / name
    except (ImportError, ValueError):
        pass
    files = {
        name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in sources.items() if path.is_file()
    }
    versions = {}
    for package in ("rllm", "verl", "vllm", "transformers", "renderers", "torch", "ray", "requests", "urllib3", "httpx", "httpcore"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    hashes = {name: row["sha256"] for name, row in files.items()}
    source_id = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    build_path = root / "runtime-build.json"
    build = json.loads(build_path.read_text()) if build_path.is_file() else None
    if verify_build and build and build["source_id"] != source_id:
        raise RuntimeError("Runtime sources differ from the image build manifest; rebuild or remove the stale source mount")
    return {"schema_version": 1, "hostname": socket.gethostname(), "source_id": source_id,
            "kernel_release": os.uname().release, "kernel_version": os.uname().version,
            "files": files, "versions": versions, "build_manifest_present": build is not None,
            "image_digest": os.environ.get("RLLM_RUNTIME_IMAGE_DIGEST"),
            "image_tag": os.environ.get("RLLM_RUNTIME_IMAGE_TAG"),
            "exec_transport": "task_metadata_or_backend_default",
            "exec_protocol": "single_post_v2", "verifier_protocol": "bound_snapshot_v3"}


def verify_cluster_runtime(ray_module, expected: dict, timeout: float = 60, *, node_ids=None, check: bool = True) -> list[dict]:
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    probe = ray_module.remote(num_cpus=0)(runtime_provenance)
    nodes = [node for node in ray_module.nodes() if node.get("Alive") and (node_ids is None or node["NodeID"] in node_ids)]
    refs = [probe.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False)).remote()
            for node in nodes]
    try:
        records = ray_module.get(refs, timeout=timeout)
    except BaseException:
        for ref in refs:
            ray_module.cancel(ref, force=True)
        raise
    for node, record in zip(nodes, records, strict=True):
        record["node_id"] = node["NodeID"]
        record["node_address"] = node.get("NodeManagerAddress")
        if check and (record["source_id"] != expected["source_id"] or record["versions"] != expected["versions"]):
            raise RuntimeError(f"Evaluation runtime mismatch on node {record['node_id']}: {record['source_id']}")
    return records


def verify_training_runtime(ray_module, config) -> list[dict]:
    """Verify and persist runtime identity before training workers allocate GPUs."""
    expected = runtime_provenance()
    nodes = [node["NodeID"] for node in ray_module.nodes()
             if node.get("Alive") and float(node.get("Resources", {}).get("GPU", 0)) > 0]
    if not nodes:
        raise RuntimeError("Training runtime verification found no live GPU nodes")
    records = verify_cluster_runtime(ray_module, expected, node_ids=nodes, check=False)
    missing = sorted(set(nodes) - {record["node_id"] for record in records})
    mismatches = [record["node_id"] for record in records
                  if record["source_id"] != expected["source_id"] or record["versions"] != expected["versions"]]
    params_path = os.environ.get("RLLM_TRAINING_PARAMS_PATH")
    directory = Path(params_path).parent if params_path else Path(str(config.rllm.trainer.wandb_dir))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "training_runtime_manifest.json"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps({"schema_version": 1, "driver": expected, "nodes": records,
                                     "status": "failed" if missing or mismatches else "verified",
                                     "missing_nodes": missing, "mismatched_nodes": mismatches}, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    if missing or mismatches:
        raise RuntimeError(f"Training runtime verification failed: missing={missing}, mismatched={mismatches}; evidence={path}")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-build", type=Path)
    args = parser.parse_args()
    record = runtime_provenance(verify_build=not bool(args.write_build))
    if args.write_build:
        args.write_build.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True))
