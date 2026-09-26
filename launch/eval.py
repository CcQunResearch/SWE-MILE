"""Multi-checkpoint Codeflow pass@n evaluation on SWE-Bench Verified.

The driver owns durable planning and aggregation.  A short-lived Ray actor
owns one standalone VERL ``LLMServerManager`` at a time and explicitly removes
its vLLM actors and placement groups before the next checkpoint is loaded.  No
trainer, actor policy worker, optimizer, or reference worker is created by this
entrypoint.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import shlex
import socket
import sys
import threading
import time
import uuid
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from launch.train import (
    _configure_model_profile,
    _validate_model_tensor_parallelism,
    _load_sandbox_dataset,
    _override_sandbox_resources,
    _reject_removed_rollout_log_probability,
    _resolved_codeflow_tool_mode,
    _resolved_codeflow_tool_restricted_mode,
    _save_launcher_backup,
    _validate_codeflow_tool_configuration,
)
from rllm.data.r2egym_builder import (
    ensure_swebench_verified_verifier,
    swebench_verified_verifier_fingerprint,
)
from rllm.data.repo_generation_exclusions import NL2REPO_EXCLUDED_TASK_IDS
from rllm.data.swebench_pro_builder import swebench_pro_verifier_fingerprint
from rllm.eval.checkpoint_evaluation import (
    AttemptSpec,
    EvaluationModel,
    EvaluationStore,
    aggregate_model,
    discover_evaluation_models,
    fingerprint_hf_model,
    infrastructure_failure_reason,
    semantic_config_id,
    semantic_configs_compatible,
    write_summary,
)
from rllm.harnesses.codeflow_shell_policy import (
    normalize_codeflow_tool_mode,
    normalize_codeflow_tool_restricted_mode,
)
from rllm.harnesses.swe_scaffold import (
    CODEFLOW_EVAL_NO_PROGRESS_POLICY_VERSION,
)
from rllm.trainer.verl.server_lifecycle import (
    create_llm_server_manager_with_retry as _shared_create_llm_server_manager_with_retry,
)
from rllm.trainer.verl.server_lifecycle import (
    initialize_llm_server_manager as _shared_initialize_llm_server_manager,
)
from rllm.trainer.verl.server_lifecycle import (
    is_retryable_server_init_error,
)
from rllm.types import RolloutInfrastructureError, Task

logger = logging.getLogger(__name__)

_DATASET_NAME = "swe_bench_verified"
_DATASET_SPLIT = "test"
_EXPECTED_TASKS = 500
_SWEBENCH_PRO_PUBLIC_EXPECTED_TASKS = 731
_BUG_REPAIR_PROFILES = frozenset({"swebench_verified", "swebench_pro_public"})
_REPO_GENERATION_PROFILES = frozenset({"nl2repo", "doc2repo"})
_DEFAULT_VERIFIER_TIMEOUT_SECONDS = 1800.0
_AGENT_SHELL_INIT = "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed"
_AGENT_ENVIRONMENT_POLICY = "preconfigured_eval_v1"
_CURRENT_METRICS_TASK_INTERVAL = 10
_DEFAULT_SERVER_INIT_MAX_ATTEMPTS = 3
_DEFAULT_SERVER_INIT_RETRY_QUIESCENCE_SECONDS = 15.0
_DEFAULT_GPU_QUIESCENCE_SECONDS = 30.0
_DEFAULT_STANDALONE_MASTER_PORT_BASE = 20000
_DEFAULT_STANDALONE_MASTER_PORT_SPAN = 128
_DEFAULT_GATEWAY_START_MAX_ATTEMPTS = 3
_MINISANDBOX_EVALUATION_DATASETS = {
    "nl2repo": "nl2repo",
    "doc2repo": "doc2repo",
    "swebench_verified": "swebench_verified",
    "swebench_pro_public": "swebench_pro_public",
}

# 45 SWE-Bench Verified tasks: 44 with action/verifier timeout evidence in
# 3,499 saved attempt_0000 records (base, steps 5/10/15/20/25/30) from
# evaluation/runs/20260907-124800, plus the existing historical timeout task
# matplotlib-20488. Action evidence includes tool timeout and exit code 124
# from an agent-issued timeout command. This is an execution-only allow-list.
_VERIFIER_TIMEOUT_RESOURCE_TASK_IDS = frozenset(
    {
        "astropy__astropy-14369",
        "astropy__astropy-14598",
        "django__django-10097",
        "django__django-11490",
        "django__django-12155",
        "django__django-12965",
        "django__django-13406",
        "django__django-13410",
        "django__django-13417",
        "django__django-13807",
        "django__django-13809",
        "django__django-14011",
        "django__django-14539",
        "django__django-15127",
        "django__django-15128",
        "django__django-15268",
        "django__django-15375",
        "django__django-16145",
        "django__django-16642",
        "django__django-9296",
        "matplotlib__matplotlib-20488",
        "matplotlib__matplotlib-22719",
        "psf__requests-1724",
        "psf__requests-1766",
        "psf__requests-1921",
        "psf__requests-2317",
        "psf__requests-5414",
        "pydata__xarray-4966",
        "pydata__xarray-6744",
        "pylint-dev__pylint-4551",
        "pytest-dev__pytest-5809",
        "scikit-learn__scikit-learn-14629",
        "scikit-learn__scikit-learn-14710",
        "scikit-learn__scikit-learn-25747",
        "sphinx-doc__sphinx-10435",
        "sphinx-doc__sphinx-11445",
        "sphinx-doc__sphinx-11510",
        "sphinx-doc__sphinx-7590",
        "sphinx-doc__sphinx-7985",
        "sphinx-doc__sphinx-8269",
        "sphinx-doc__sphinx-8475",
        "sphinx-doc__sphinx-9320",
        "sympy__sympy-13878",
        "sympy__sympy-18199",
        "sympy__sympy-21612",
    }
)
_VERIFIER_TIMEOUT_RESOURCE_CPUS = 8
_VERIFIER_TIMEOUT_RESOURCE_MEMORY_MB = 32 * 1024

# These are the only SWE-Bench Verified verifier suites in the materialized
# 500-task split whose selected tests require successful public HTTP access.
# The runtime probes exact upstreams before spending the full verifier budget;
# the socket guard only bounds calls that otherwise have no timeout.  Neither
# policy changes the target tests or the semantic evaluation identity.
_PUBLIC_NETWORK_VERIFIER_TASK_IDS = frozenset(
    {
        "psf__requests-1142",
        "psf__requests-1724",
        "psf__requests-1766",
        "psf__requests-1921",
        "psf__requests-2317",
        "sphinx-doc__sphinx-7985",
        "sphinx-doc__sphinx-8269",
        "sphinx-doc__sphinx-8475",
    }
)
_REQUESTS_NETWORK_PROBE_URLS = ["https://httpbin.org/get"]
_SPHINX_NETWORK_PROBE_URLS = [
    "https://www.google.com/",
    "http://www.sphinx-doc.org/en/1.7/intro.html",
]
_VERIFIER_PROXY_URL = os.environ.get("SWE_MILE_VERIFIER_PROXY_URL", "")
_VERIFIER_NETWORK_POLICY_VERSION = 1


def _benchmark_config(config: DictConfig) -> dict[str, Any]:
    profile = str(_config_value(config, "eval.benchmark_profile", "swebench_verified")).strip()
    defaults = {
        "swebench_verified": (_DATASET_NAME, _DATASET_SPLIT, _EXPECTED_TASKS),
        "swebench_pro_public": (
            "swebench_pro_public",
            "test",
            _SWEBENCH_PRO_PUBLIC_EXPECTED_TASKS,
        ),
        "nl2repo": (
            "nl2repo-bench",
            "test",
            104 - len(NL2REPO_EXCLUDED_TASK_IDS),
        ),
        "doc2repo": ("beyondswe-doc2repo", "test", 50),
    }
    if profile not in defaults:
        supported = ", ".join(defaults)
        raise ValueError(
            f"eval.benchmark_profile must be one of {supported}, got {profile!r}"
        )
    default_name, default_split, default_count = defaults[profile]
    expected = _config_value(config, "eval.expected_tasks", default_count)
    if expected is None:
        expected = default_count
    return {
        "profile": profile,
        "dataset_name": str(_config_value(config, "eval.dataset_name", default_name)),
        "dataset_split": str(_config_value(config, "eval.dataset_split", default_split)),
        "expected_tasks": _positive_int(expected, "eval.expected_tasks"),
        "bug_repair": profile in _BUG_REPAIR_PROFILES,
        "repo_generation": profile in _REPO_GENERATION_PROFILES,
    }


def _configure_minisandbox_evaluation(
    config: DictConfig,
    benchmark: dict[str, Any],
) -> str:
    """Resolve the opt-in evaluation cache without changing backend defaults."""

    backend = str(
        OmegaConf.select(config, "swe.sandbox_backend", default="minisandbox")
    ).strip().lower()
    if backend == "k8s":
        from rllm.sandbox.backends.k8s import check_k8s_adapter
        check_k8s_adapter()
    elif backend != "minisandbox":
        raise ValueError("swe.sandbox_backend must be minisandbox or k8s")
    if backend != "minisandbox":
        return backend
    profile = str(benchmark["profile"])
    dataset_key = _MINISANDBOX_EVALUATION_DATASETS.get(profile)
    if dataset_key is None:
        supported = ", ".join(sorted(_MINISANDBOX_EVALUATION_DATASETS))
        raise ValueError(
            "swe.sandbox_backend=minisandbox only supports evaluation profiles "
            f"{supported}; got {profile!r}"
        )
    from rllm.data.minisandbox_cache import (
        default_minisandbox_cache_dir,
        minisandbox_dataset_spec,
    )

    spec = minisandbox_dataset_spec(dataset_key)
    if benchmark["dataset_name"] != spec.physical_name:
        raise ValueError(
            "MiniSandbox evaluation dataset override does not match the profile: "
            f"expected={spec.physical_name} actual={benchmark['dataset_name']}"
        )
    if benchmark["dataset_split"] != spec.split:
        raise ValueError(
            "MiniSandbox evaluation split override does not match the profile: "
            f"expected={spec.split} actual={benchmark['dataset_split']}"
        )
    warm_size = OmegaConf.select(
        config, "rllm.workflow.warm_queue_size", default=0
    )
    if int(warm_size) != 0:
        raise ValueError(
            "MiniSandbox evaluation requires rllm.workflow.warm_queue_size=0; "
            "job-scoped node services perform their own cache-aware routing"
        )
    shared_cache_dir = OmegaConf.select(
        config, "swe.minisandbox.shared_cache_dir", default=None
    )
    if shared_cache_dir in (None, ""):
        data_root = os.environ.get("DATA_ROOT")
        if not data_root:
            raise ValueError(
                "DATA_ROOT is required to derive the default "
                "swe.minisandbox.shared_cache_dir"
            )
        OmegaConf.update(
            config,
            "swe.minisandbox.shared_cache_dir",
            str(default_minisandbox_cache_dir(data_root, spec)),
            merge=False,
            force_add=True,
        )
    if OmegaConf.select(config, "swe.minisandbox.local_root", default=None) in (
        None,
        "",
    ):
        OmegaConf.update(
            config,
            "swe.minisandbox.local_root",
            "/tmp/swe-minisandbox",
            merge=False,
            force_add=True,
        )
    OmegaConf.update(
        config,
        "swe.minisandbox.dataset",
        dataset_key,
        merge=False,
        force_add=True,
    )
    return backend


def _validate_minisandbox_evaluation_cache(
    config: DictConfig,
    tasks: Sequence[Task],
    benchmark_profile: str,
) -> dict[str, dict[str, Any]]:
    from rllm.data.minisandbox_cache import validate_minisandbox_oci_cache

    dataset_key = _MINISANDBOX_EVALUATION_DATASETS[benchmark_profile]
    try:
        records = validate_minisandbox_oci_cache(
            str(OmegaConf.select(config, "swe.minisandbox.shared_cache_dir")),
            tasks,
            dataset=dataset_key,
        )
        if len(records) != len({task.id for task in tasks}):
            raise ValueError(
                "MiniSandbox OCI cache does not exactly cover the evaluation tasks"
            )
    except RolloutInfrastructureError:
        raise
    except Exception as exc:
        raise RolloutInfrastructureError(
            "minisandbox_cache_invalid",
            f"MiniSandbox evaluation cache preflight failed: {exc}",
            retryable=False,
            stage="preflight",
            retry_scope="full_rollout",
        ) from exc
    OmegaConf.update(
        config,
        "swe.minisandbox.canary_task_id",
        sorted(records)[0],
        merge=False,
        force_add=True,
    )
    logger.info(
        "Validated %s MiniSandbox OCI cache for %d evaluation task(s)",
        dataset_key,
        len(records),
    )
    return records


def _filter_excluded_benchmark_tasks(tasks: Sequence[Task], benchmark_profile: str) -> list[Task]:
    excluded = NL2REPO_EXCLUDED_TASK_IDS if benchmark_profile == "nl2repo" else frozenset()
    if not excluded:
        return list(tasks)
    filtered = [task for task in tasks if task.id not in excluded]
    removed = sorted({task.id for task in tasks} & excluded)
    if removed:
        logger.warning(
            "Excluded %d inaccessible %s task(s): %s",
            len(removed),
            benchmark_profile,
            ", ".join(removed),
        )
    return filtered


def _filter_benchmark_tasks_by_id_file(
    tasks: Sequence[Task],
    path: Path,
) -> list[Task]:
    """Select an exact ordered task set for paired n=1/n=8 canaries."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read eval.task_ids_file {path}: {exc}") from exc
    if isinstance(payload, list):
        task_ids = payload
    elif isinstance(payload, dict):
        task_ids = next(
            (
                payload[key]
                for key in ("task_ids", "fixed_task_ids", "zero_signal_task_ids")
                if key in payload
            ),
            None,
        )
    else:
        task_ids = None
    if (
        not isinstance(task_ids, list)
        or not task_ids
        or any(not isinstance(task_id, str) or not task_id for task_id in task_ids)
    ):
        raise ValueError(
            "eval.task_ids_file must contain a non-empty JSON task-id list "
            "or an object with task_ids/fixed_task_ids"
        )
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("eval.task_ids_file contains duplicate task ids")
    by_id = {task.id: task for task in tasks}
    unknown = [task_id for task_id in task_ids if task_id not in by_id]
    if unknown:
        raise ValueError(
            f"eval.task_ids_file contains unknown task ids: {unknown[:10]}"
        )
    return [by_id[task_id] for task_id in task_ids]


def _teardown_llm_server_manager(
    manager: Any,
    ray_module: Any,
) -> dict[str, int]:
    """Release every Ray resource created by standalone ``LLMServerManager``.

    VERL 0.8 creates one ``RayResourcePool`` (and therefore one or more Ray
    placement groups) per standalone rollout replica, but exposes no public
    manager close method.  Killing this evaluation's owner actor is not enough:
    non-detached placement groups remain scoped to the Ray job and continue to
    reserve their GPU bundles while the driver loads the next checkpoint.

    Actor termination and placement-group removal are both idempotent requests.
    Cleanup is best effort here; the driver subsequently verifies that all GPU
    resources returned before it starts another model.
    """

    if manager is None:
        return {
            "load_balancers": 0,
            "servers": 0,
            "workers": 0,
            "placement_groups": 0,
            "errors": 0,
        }

    load_balancers = []
    load_balancer = getattr(manager, "global_load_balancer", None)
    if load_balancer is not None:
        load_balancers.append(load_balancer)

    servers = []
    workers = []
    placement_groups = []
    for replica in getattr(manager, "rollout_replicas", ()) or ():
        servers.extend(getattr(replica, "servers", ()) or ())
        workers.extend(getattr(replica, "workers", ()) or ())
        resource_pool = getattr(replica, "resource_pool", None)
        placement_groups.extend(getattr(resource_pool, "pgs", ()) or ())

    from rllm.trainer.verl.server_lifecycle import request_graceful_server_shutdown

    request_graceful_server_shutdown(servers, ray_module)
    errors = 0
    # HTTP server actors own vLLM multiprocessing children and should be
    # stopped before their checkpoint-engine workers and GPU reservations.
    for kind, handles in (
        ("vLLM server", servers),
        ("vLLM worker", workers),
        ("load balancer", load_balancers),
    ):
        for handle in handles:
            try:
                ray_module.kill(handle, no_restart=True)
            except Exception:  # pragma: no cover - exercised against real Ray
                errors += 1
                logger.exception("Failed to terminate %s actor during model teardown", kind)

    remove_placement_group = getattr(
        getattr(ray_module, "util", None),
        "remove_placement_group",
        None,
    )
    if placement_groups and remove_placement_group is None:
        errors += len(placement_groups)
        logger.error(
            "Ray has no remove_placement_group API; %d rollout placement group(s) may retain GPUs",
            len(placement_groups),
        )
    else:
        for placement_group in placement_groups:
            try:
                remove_placement_group(placement_group)
            except Exception:  # pragma: no cover - exercised against real Ray
                errors += 1
                logger.exception("Failed to remove rollout placement group during model teardown")

    summary = {
        "load_balancers": len(load_balancers),
        "servers": len(servers),
        "workers": len(workers),
        "placement_groups": len(placement_groups),
        "errors": errors,
    }
    logger.info("Requested standalone vLLM teardown: %s", summary)
    return summary


def _standalone_master_port_range(
    replica_rank: int,
    *,
    base: int,
    span: int,
) -> tuple[int, int]:
    """Return a disjoint Gloo/TCPStore port range for one rollout replica."""

    if replica_rank < 0:
        raise ValueError(f"replica_rank must be non-negative, got {replica_rank}")
    if not 1024 <= base <= 65535:
        raise ValueError(f"eval.standalone_master_port_base must be in [1024, 65535], got {base}")
    if span <= 0:
        raise ValueError(f"eval.standalone_master_port_span must be positive, got {span}")
    start = base + replica_rank * span
    end = start + span
    if end > 65536:
        raise ValueError(
            "standalone rollout replica port ranges exceed TCP port 65535: "
            f"replica_rank={replica_rank}, range=[{start}, {end})"
        )
    return start, end


def _install_verl_standalone_port_ranges(config: DictConfig) -> None:
    """Give concurrent standalone replicas non-overlapping TCPStore ranges.

    VERL discovers a free ``MASTER_PORT`` by binding port zero and immediately
    closing the probe socket.  With many replicas starting concurrently, two
    probes on one node can select the same port before either TCPStore binds.
    ``RayWorkerGroup`` already supports a bounded port range, but standalone
    rollout replicas do not pass one.  This process-local adapter supplies a
    deterministic, disjoint range based on the replica rank.
    """

    from verl.single_controller.ray.base import RayWorkerGroup

    base = _positive_int(
        _config_value(
            config,
            "eval.standalone_master_port_base",
            _DEFAULT_STANDALONE_MASTER_PORT_BASE,
        ),
        "eval.standalone_master_port_base",
    )
    span = _positive_int(
        _config_value(
            config,
            "eval.standalone_master_port_span",
            _DEFAULT_STANDALONE_MASTER_PORT_SPAN,
        ),
        "eval.standalone_master_port_span",
    )
    original_init = getattr(RayWorkerGroup.__init__, "_rllm_unpatched_init", RayWorkerGroup.__init__)

    def init_with_standalone_port_range(self, *args, **kwargs):
        name_prefix = str(kwargs.get("name_prefix") or "")
        match = re.fullmatch(r"rollout_(?:reward_|teacher_)?standalone_(\d+)(?:_.*)?", name_prefix)
        if match is not None and "master_port_range" not in kwargs:
            replica_rank = int(match.group(1))
            kwargs["master_port_range"] = list(
                _standalone_master_port_range(
                    replica_rank,
                    base=base,
                    span=span,
                )
            )
        return original_init(self, *args, **kwargs)

    init_with_standalone_port_range._rllm_unpatched_init = original_init  # type: ignore[attr-defined]
    RayWorkerGroup.__init__ = init_with_standalone_port_range
    logger.info(
        "Installed standalone rollout TCPStore port partition: base=%d span=%d",
        base,
        span,
    )


def _is_address_in_use_error(exc: BaseException) -> bool:
    """Recognize retryable TCPStore/engine startup failures through Ray errors."""

    return is_retryable_server_init_error(exc)


def _initialize_llm_server_manager(manager: Any) -> Any:
    """Initialize a manager while retaining its handle on partial failure."""

    return _shared_initialize_llm_server_manager(manager)


def _create_llm_server_manager_with_retry(
    manager_class: Any,
    *,
    config: DictConfig,
    ray_module: Any,
    model_key: str,
    isolation: Any = None,
) -> Any:
    """Create standalone vLLM, cleaning partial actors before port retries."""

    max_attempts = _positive_int(
        _config_value(
            config,
            "eval.server_init_max_attempts",
            _DEFAULT_SERVER_INIT_MAX_ATTEMPTS,
        ),
        "eval.server_init_max_attempts",
    )
    retry_quiescence = _nonnegative_float(
        _config_value(
            config,
            "eval.server_init_retry_quiescence_seconds",
            _DEFAULT_SERVER_INIT_RETRY_QUIESCENCE_SECONDS,
        ),
        "eval.server_init_retry_quiescence_seconds",
    )
    release_timeout = _positive_float(
        _config_value(config, "eval.gpu_release_timeout", 300),
        "eval.gpu_release_timeout",
    )
    def teardown(manager, ray_module):
        result = _teardown_llm_server_manager(manager, ray_module)
        if isolation is not None:
            isolation.cleanup()
        return result

    return _shared_create_llm_server_manager_with_retry(
        manager_class,
        config=config,
        ray_module=ray_module,
        model_key=model_key,
        max_attempts=max_attempts,
        retry_quiescence_seconds=retry_quiescence,
        gpu_release_timeout_seconds=release_timeout,
        teardown=teardown,
        wait_for_reclamation=_wait_for_gpu_reclamation,
    )


def _config_value(config: DictConfig, path: str, default: Any = None) -> Any:
    return OmegaConf.select(config, path, default=default)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if parsed <= 0 or str(value).strip() not in {str(parsed), f"+{parsed}"}:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return parsed


def _float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, got {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc


def _positive_float(value: Any, name: str) -> float:
    parsed = _float(value, name)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return parsed


def _nonnegative_float(value: Any, name: str) -> float:
    parsed = _float(value, name)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return parsed


def _boolean(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(f"{name} must be true or false, got {value!r}")


_LEGACY_SEQUENCE_PARTITION_KEYS = frozenset(
    {
        "rllm.data.max_prompt_length",
        "rllm.data.max_response_length",
        "data.max_prompt_length",
        "data.max_response_length",
    }
)


def _hydra_task_overrides() -> list[str]:
    """Return normalized task overrides when running under Hydra."""

    try:
        from hydra.core.hydra_config import HydraConfig

        return [str(value) for value in HydraConfig.get().overrides.task]
    except (AttributeError, ValueError):
        return []


def _sequence_budget_config(
    config: DictConfig,
    *,
    explicit_overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """Resolve and validate the standalone evaluation sequence-budget contract."""

    dynamic = _boolean(
        _config_value(config, "rllm.data.dynamic_sequence_budget", False),
        "rllm.data.dynamic_sequence_budget",
    )
    raw_model_window = _config_value(
        config,
        "actor_rollout_ref.rollout.max_model_len",
        None,
    )
    if isinstance(raw_model_window, bool) or not isinstance(raw_model_window, int) or raw_model_window <= 0:
        raise ValueError("actor_rollout_ref.rollout.max_model_len must be a positive integer")
    model_window = int(raw_model_window)
    per_turn = _positive_int(
        _config_value(config, "rllm.rollout.val.max_tokens", 2048),
        "rllm.rollout.val.max_tokens",
    )
    if per_turn > model_window:
        raise ValueError("rllm.rollout.val.max_tokens cannot exceed actor_rollout_ref.rollout.max_model_len")

    if dynamic:
        override_keys = {str(value).split("=", 1)[0].lstrip("+") for value in explicit_overrides}
        conflicting = sorted(_LEGACY_SEQUENCE_PARTITION_KEYS & override_keys)
        if conflicting:
            raise ValueError(
                "rllm.data.dynamic_sequence_budget=true conflicts with explicit legacy sequence partitions: " + ", ".join(conflicting) + ". Configure only actor_rollout_ref.rollout.max_model_len."
            )
        if not _boolean(
            _config_value(config, "rllm.gateway.cumulative_token_mode", False),
            "rllm.gateway.cumulative_token_mode",
        ):
            raise ValueError("dynamic evaluation sequence budgeting requires rllm.gateway.cumulative_token_mode=true")
        gateway_window = _config_value(
            config,
            "rllm.gateway.max_context_tokens",
            None,
        )
        if gateway_window is not None and int(gateway_window) != model_window:
            raise ValueError("dynamic evaluation sequence budgeting uses max_model_len as its sole window; rllm.gateway.max_context_tokens must be unset or equal to max_model_len")
        return {
            "mode": "dynamic_model_window",
            "dynamic": True,
            "model_window_tokens": model_window,
            "max_tokens_per_turn": per_turn,
        }

    prompt = _positive_int(
        _config_value(config, "rllm.data.max_prompt_length", 4096),
        "rllm.data.max_prompt_length",
    )
    response = _positive_int(
        _config_value(config, "rllm.data.max_response_length", 94208),
        "rllm.data.max_response_length",
    )
    if prompt + response > model_window:
        raise ValueError(f"fixed evaluation sequence partitions exceed max_model_len: prompt={prompt}, response={response}, window={model_window}")
    return {
        "mode": "fixed_partitions",
        "dynamic": False,
        "model_window_tokens": model_window,
        "initial_prompt_tokens": prompt,
        "cumulative_response_tokens": response,
        "max_tokens_per_turn": per_turn,
    }


def _configure_evaluation_server_generation_limit(
    config: DictConfig,
    sequence_budget: dict[str, Any],
) -> None:
    """Make the standalone vLLM server honor the evaluation per-turn cap.

    Training runs call VERL's ``sync_config``, which mirrors
    ``rllm.rollout.train.max_tokens`` into the server-side ``response_length``.
    Standalone evaluation intentionally has no trainer and therefore does not
    run that synchronization.  Without this explicit assignment, VERL keeps
    its 512-token default and vLLM silently clamps requests that correctly ask
    for a larger ``rllm.rollout.val.max_tokens`` value.
    """

    per_turn = _positive_int(
        sequence_budget.get("max_tokens_per_turn"),
        "sequence_budget.max_tokens_per_turn",
    )
    with open_dict(config):
        config.actor_rollout_ref.rollout.response_length = per_turn


def _sampling_config(config: DictConfig) -> dict[str, Any]:
    temperature = _float(
        _config_value(config, "eval.temperature", 1.0),
        "eval.temperature",
    )
    top_p = _float(_config_value(config, "eval.top_p", 1.0), "eval.top_p")
    raw_top_k = _config_value(config, "eval.top_k", -1)
    if isinstance(raw_top_k, bool):
        raise ValueError(f"eval.top_k must be -1 or a positive integer, got {raw_top_k!r}")
    try:
        top_k = int(raw_top_k)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"eval.top_k must be -1 or a positive integer, got {raw_top_k!r}") from exc
    if temperature < 0:
        raise ValueError("eval.temperature must be non-negative")
    if not 0 < top_p <= 1:
        raise ValueError("eval.top_p must be in (0, 1]")
    if top_k != -1 and top_k <= 0:
        raise ValueError("eval.top_k must be -1 or a positive integer")
    sampling: dict[str, Any] = {"temperature": temperature, "top_p": top_p, "top_k": top_k}
    template_kwargs = _config_value(config, "rllm.rollout.val.chat_template_kwargs", None)
    if template_kwargs is not None:
        sampling["chat_template_kwargs"] = OmegaConf.to_container(template_kwargs, resolve=True)
    return sampling


def _evaluation_session_sampling_params(config: DictConfig, base: dict[str, Any]) -> dict[str, Any]:
    """Use the same effective template controls for requests and cache identity."""
    return {
        **base,
        **_sampling_config(config),
        "max_tokens": int(config.rllm.rollout.val.max_tokens),
    }


def _task_fingerprint(tasks: Sequence[Task], *, benchmark_profile: str = "swebench_verified") -> str:
    import hashlib

    rows = []
    for index, task in enumerate(tasks):
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        environment = metadata.get("environment") if isinstance(metadata.get("environment"), dict) else {}
        row = {
            "index": index,
            "id": task.id,
            "instruction": task.instruction,
            "repo_name": metadata.get("repo_name"),
            "commit_hash": metadata.get("commit_hash"),
            "docker_image": metadata.get("docker_image") or environment.get("docker_image"),
        }
        if benchmark_profile == "swebench_verified":
            # Include the current verification metadata in the task identity.
            row["shadow_verification_contract"] = {
                "result_parser": metadata.get("result_parser"),
                "log_parser": metadata.get("log_parser"),
                "result_adapter_version": metadata.get("result_adapter_version"),
                "baseline_output_json": metadata.get("baseline_output_json"),
                "target_output_json": metadata.get("target_output_json"),
                "FAIL_TO_PASS": metadata.get("FAIL_TO_PASS"),
                "PASS_TO_PASS": metadata.get("PASS_TO_PASS"),
            }
        if benchmark_profile != "swebench_verified":
            row.update(
                {
                    "task_profile": metadata.get("task_profile") or (metadata.get("rllm") or {}).get("task_profile"),
                    "source_revision": metadata.get("source_revision"),
                    "source_fingerprint": metadata.get("source_fingerprint"),
                    "verifier_profile": metadata.get("verifier_profile"),
                }
            )
        rows.append(row)
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _semantic_config(config: DictConfig, tasks: Sequence[Task]) -> dict[str, Any]:
    """Only fields that change trajectory semantics belong in this hash."""

    sampling = _sampling_config(config)
    benchmark = _benchmark_config(config)
    repo_generation = bool(benchmark["repo_generation"])
    if repo_generation:
        import hashlib

        verifier_digest = hashlib.sha256()
        verifier_digest.update(benchmark["profile"].encode("utf-8"))
        verifier_sources = [
            Path(__file__).parents[1] / "rllm" / "eval" / "repo_generation.py",
            Path(__file__).parents[1] / "rllm" / "eval" / "script_evaluator.py",
        ]
        if benchmark["profile"] == "doc2repo":
            verifier_sources.append(Path(__file__).parents[1] / "rllm/data/assets/doc2repo_verifier.sh")
        if benchmark["profile"] == "nl2repo":
            verifier_sources.extend(Path(__file__).parents[1] / relative for relative in (
                "rllm/eval/repo_generation_status.py", "rllm/sandbox/nl2repo_environment.py",
                "rllm/data/assets/nl2repo_environment.py", "rllm/sandbox/repo_generation_environment.py",
                "rllm/data/assets/repo_generation_environment.py", "rllm/harnesses/swe_scaffold.py",
            ))
        for source in verifier_sources:
            verifier_digest.update(source.read_bytes())
        verifier_sha256 = verifier_digest.hexdigest()
    elif benchmark["profile"] == "swebench_verified":
        verifier_sha256 = swebench_verified_verifier_fingerprint()
    else:
        verifier_sha256 = swebench_pro_verifier_fingerprint()
    sequence_budget = _sequence_budget_config(config)
    return {
        "schema_version": 12,
        "execution": {"sandbox_backend": str(_config_value(config, "swe.sandbox_backend", "minisandbox"))},
        "dataset": {
            "name": benchmark["dataset_name"],
            "split": benchmark["dataset_split"],
            **(
                {"benchmark_profile": benchmark["profile"]}
                if benchmark["profile"] != "swebench_verified"
                else {}
            ),
            "tasks": len(tasks),
            "ordered_tasks_sha256": _task_fingerprint(tasks, benchmark_profile=benchmark["profile"]),
        },
        "agent": {
            "name": "codeflow",
            "protocol": "native_tool_call",
            "shell_environment": (
                "conda_testbed_v1"
                if benchmark["profile"] == "swebench_verified"
                else "repository_image_default"
            ),
            "environment_policy": (
                _AGENT_ENVIRONMENT_POLICY
                if benchmark["profile"] == "swebench_verified"
                else (
                    "repo_generation_package"
                    if repo_generation
                    else "swebench_pro_image_default_v1"
                )
            ),
            "request_protocol": "idempotent_turn_v1",
            "structured_helper_environment": "utf8_v1",
            "structured_helper_runtime": "legacy_subprocess_infrastructure_v1",
            "repository_checkpoint_failure_policy": "infrastructure_v1",
            **({"nl2repo_setup": "offline_git_primary_cleanup_image_contract_v4",
                "repo_generation_prompt": "generation_environment_v2",
                "repo_generation_proxy_url": str(_config_value(config, "eval.repo_generation_proxy_url", "") or "")} if benchmark["profile"] == "nl2repo" else {}),
            **({"nl2repo_execution_policy": "bounded_bash_no_progress_v1"} if benchmark["profile"] == "nl2repo" else {}),
            **({"doc2repo_setup": "offline_git_apt_v1"} if benchmark["profile"] == "doc2repo" else {}),
            **({"execution_policy": "bounded_bash_no_progress_v1"} if benchmark["profile"] in {"doc2repo", "swebench_pro_public"} else {}),
            "no_progress_policy": CODEFLOW_EVAL_NO_PROGRESS_POLICY_VERSION,
            "codeflow_tool_mode": normalize_codeflow_tool_mode(
                _config_value(
                    config,
                    "swe.codeflow_tool_mode",
                    None,
                )
            ),
            "codeflow_tool_restricted_mode": list(
                normalize_codeflow_tool_restricted_mode(
                    _config_value(
                        config,
                        "swe.codeflow_tool_restricted_mode",
                        None,
                    )
                )
            ),
        },
        "sampling": sampling,
        "limits": {
            "max_turns": int(_config_value(config, "swe.max_turns", 100)),
            "sequence_budget": {key: value for key, value in sequence_budget.items() if key != "dynamic"},
            "limit_termination_success_reward": 1.0,
            **({"limit_termination_outcome_mode": "verifier_outcome"} if repo_generation else {}),
        },
        # The verifier deadline is an execution policy, not a sampling
        # parameter.  It is recorded in the run's resolved config and each
        # timeout result, but intentionally excluded here so an interrupted
        # evaluation can reuse attempts completed under the former generous
        # deadline.  Completed attempts do not change when the cap is lowered;
        # only previously unresolved long-tail attempts use the new deadline.
        "gateway": {
            "cumulative_token_mode": bool(_config_value(config, "rllm.gateway.cumulative_token_mode", True)),
            "renderer_family": str(_config_value(config, "rllm.gateway.renderer_family", "qwen3.5")),
            "reasoning_parser": str(
                _config_value(
                    config,
                    "actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning_parser",
                    "qwen3",
                )
            ),
            "tool_call_parser": str(
                _config_value(
                    config,
                    "actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser",
                    "qwen3_coder",
                )
            ),
        },
        # Verifier semantics affect correctness just as much as sampling
        # parameters.  Including the digest prevents parser fixes from
        # silently reusing attempts scored by an older, incompatible parser.
        "verifier": {
            **(
                {"profile": benchmark["profile"]}
                if benchmark["profile"] != "swebench_verified"
                else {}
            ),
            "sha256": verifier_sha256,
        },
    }




def _configure_tasks(
    tasks: Sequence[Task],
    verifier_timeout: float = _DEFAULT_VERIFIER_TIMEOUT_SECONDS,
    benchmark_profile: str = "swebench_verified",
    repo_generation_proxy_url: str = "",
) -> None:
    """Apply evaluation-only task policy without changing materialized data.

    SWE-Bench Verified uses the materialized builders' conservative 1,800
    second timeout.  A task may opt into a different budget via
    ``task.metadata["rllm"]["eval_verifier_timeout"]``.  R2E-Gym training
    tasks never pass through this function and retain their existing timeout.
    """

    default_timeout = _positive_float(verifier_timeout, "swe.verifier_timeout")
    for task in tasks:
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        rllm_metadata = dict(metadata.get("rllm") or {})
        task_override = rllm_metadata.get("eval_verifier_timeout")
        resolved_timeout = (
            default_timeout
            if task_override is None
            else _positive_float(
                task_override,
                f"{task.id}.metadata.rllm.eval_verifier_timeout",
            )
        )
        # Evaluation scores only the primary verifier outcome.  Explicitly
        # disable any shadow flag inherited from materialized training data.
        rllm_metadata["shadow_enabled"] = False
        rllm_metadata["verifier_timeout_is_failure"] = True
        if benchmark_profile in {"nl2repo", "doc2repo", "swebench_pro_public"}:
            rllm_metadata["eval_no_progress"] = True
        else:
            rllm_metadata.pop("eval_no_progress", None)
        if benchmark_profile in {"doc2repo", "swebench_pro_public"}:
            # Stage current host-authored verifiers inside each sandbox. Do
            # not rewrite shared/possibly read-only materialized task trees.
            rllm_metadata["eval_verifier_profile"] = benchmark_profile
            metadata["setup_failure_mode"] = "raise"
        else:
            rllm_metadata.pop("eval_verifier_profile", None)
        if benchmark_profile == "swebench_verified":
            rllm_metadata["agent_shell_init"] = _AGENT_SHELL_INIT
            rllm_metadata["agent_environment_policy"] = _AGENT_ENVIRONMENT_POLICY
            if task.id in _PUBLIC_NETWORK_VERIFIER_TASK_IDS:
                probe_urls = (
                    _REQUESTS_NETWORK_PROBE_URLS
                    if task.id.startswith("psf__requests-")
                    else _SPHINX_NETWORK_PROBE_URLS
                )
                rllm_metadata["verifier_network_policy"] = {
                    "schema_version": _VERIFIER_NETWORK_POLICY_VERSION,
                    "probe_urls": list(probe_urls),
                    "probe_attempts": 3,
                    "probe_timeout_seconds": 10.0,
                    "probe_retry_delay_seconds": 1.0,
                    "default_socket_timeout_seconds": 20.0,
                    "python_executable": "/opt/miniconda3/envs/testbed/bin/python",
                    "proxy_env": ({
                        "http_proxy": _VERIFIER_PROXY_URL,
                        "https_proxy": _VERIFIER_PROXY_URL,
                        "all_proxy": _VERIFIER_PROXY_URL,
                        "no_proxy": "localhost,127.0.0.1,::1",
                    } if _VERIFIER_PROXY_URL else {}),
                    **(
                        {"verifier_env": {"HTTPBIN_URL": "https://httpbin.org"}}
                        if task.id.startswith("psf__requests-")
                        else {}
                    ),
                }
                # Apply the same network policy to tests launched by the
                # agent, not only to the final verifier. Keep these exports
                # inside the task shell so host-side service clients
                # never inherit a public-network proxy.
                policy = rllm_metadata["verifier_network_policy"]
                agent_env = dict(policy["proxy_env"])
                agent_env.update(policy.get("verifier_env", {}))
                agent_env.update({key.upper(): value for key, value in policy["proxy_env"].items()})
                exports = " ".join(f"{key}={shlex.quote(value)}" for key, value in agent_env.items())
                rllm_metadata["agent_shell_init"] = f"{_AGENT_SHELL_INIT} && export {exports}" if exports else _AGENT_SHELL_INIT
            else:
                rllm_metadata.pop("verifier_network_policy", None)
        else:
            rllm_metadata.pop("agent_shell_init", None)
            rllm_metadata.pop("agent_environment_policy", None)
            rllm_metadata.pop("verifier_network_policy", None)
        if benchmark_profile == "nl2repo":
            from rllm.sandbox.repo_generation_environment import repo_generation_proxy_exports
            repo_generation_proxy_exports(repo_generation_proxy_url)
            rllm_metadata["repo_generation_proxy_url"] = repo_generation_proxy_url.strip()
        # Keep the materialized timeout intact: it is part of the durable task
        # fingerprint used by existing evaluation stores.  The evaluator
        # resolver consumes this runtime-only value, so lowering the timeout
        # does not invalidate already completed attempts from the same run.
        rllm_metadata["resolved_verifier_timeout"] = resolved_timeout
        metadata["rllm"] = rllm_metadata
        task.metadata = metadata


def _configure_sandbox_resources(
    dataset: Any,
    *,
    benchmark_profile: str,
    cpus: int,
    memory_mb: int,
) -> None:
    """Apply evaluation-wide CPU/memory overrides to every benchmark task."""

    _override_sandbox_resources(dataset, cpus, memory_mb)
    if benchmark_profile != "swebench_pro_public":
        return

    mismatches: list[str] = []
    for item in dataset.data:
        metadata = item.metadata if hasattr(item, "metadata") else item
        environment = (
            metadata.get("environment") if isinstance(metadata, dict) else None
        )
        actual = (
            environment.get("cpus") if isinstance(environment, dict) else None,
            environment.get("memory_mb")
            if isinstance(environment, dict)
            else None,
        )
        if actual != (cpus, memory_mb):
            item_id = getattr(item, "id", "<unknown>")
            mismatches.append(f"{item_id}={actual[0]!r}/{actual[1]!r}")
    if mismatches:
        examples = ", ".join(mismatches[:10])
        raise ValueError(
            "SWE-bench Pro Public sandbox resource override failed closed: "
            f"expected every task to use {cpus} CPU/{memory_mb} MiB, "
            f"mismatches={len(mismatches)}, examples=[{examples}]"
        )


def _sandbox_resource_histogram(
    tasks: Sequence[Task],
) -> list[dict[str, int | None]]:
    """Return a stable task-count histogram of final sandbox resources."""

    counts: Counter[tuple[int | None, int | None, int | None]] = Counter()
    for task in tasks:
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        environment = metadata.get("environment")
        environment = environment if isinstance(environment, dict) else {}
        values: list[int | None] = []
        for key in ("cpus", "memory_mb", "storage_mb"):
            raw = environment.get(key)
            try:
                values.append(int(raw) if raw is not None else None)
            except (TypeError, ValueError):
                values.append(None)
        counts[(values[0], values[1], values[2])] += 1
    return [
        {
            "cpus": cpus,
            "memory_mb": memory_mb,
            "storage_mb": storage_mb,
            "tasks": task_count,
        }
        for (cpus, memory_mb, storage_mb), task_count in sorted(
            counts.items(),
            key=lambda item: tuple(
                -1 if value is None else value for value in item[0]
            ),
        )
    ]




def _override_verifier_timeout_task_resources(
    tasks: Sequence[Task],
    benchmark_profile: str = "swebench_verified",
) -> list[str]:
    """Pin 8 CPU/32 GiB for known action/verifier-timeout evaluation tasks.

    This runs after the evaluation-wide sandbox override so the selected tasks
    always receive their pinned capacity regardless of Hydra or shell
    overrides. CPU and memory are the only task fields changed.
    """

    if benchmark_profile != "swebench_verified":
        return []

    overridden: list[str] = []
    for task in tasks:
        if task.id not in _VERIFIER_TIMEOUT_RESOURCE_TASK_IDS:
            continue
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        environment = dict(metadata.get("environment") or {})
        environment["cpus"] = _VERIFIER_TIMEOUT_RESOURCE_CPUS
        environment["memory_mb"] = _VERIFIER_TIMEOUT_RESOURCE_MEMORY_MB
        metadata["environment"] = environment
        task.metadata = metadata
        overridden.append(task.id)
    return overridden


def _validate_tasks(tasks: Sequence[Task], benchmark_profile: str = "swebench_verified") -> None:
    seen: set[str] = set()
    errors: list[str] = []
    for task in tasks:
        if task.id in seen:
            errors.append(f"{task.id}: duplicate task ID")
        seen.add(task.id)
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        environment = metadata.get("environment") if isinstance(metadata.get("environment"), dict) else {}
        if not (metadata.get("docker_image") or environment.get("docker_image")):
            errors.append(f"{task.id}: docker image is missing")
        if not (task.task_dir / "tests" / "test.sh").is_file():
            errors.append(f"{task.id}: tests/test.sh is missing")
        if benchmark_profile == "swebench_verified" and not (task.task_dir / "tests" / "run_tests.sh").is_file():
            errors.append(f"{task.id}: tests/run_tests.sh is missing")
        rllm_metadata = metadata.get("rllm") or {}
        task_profile = str(metadata.get("task_profile") or rllm_metadata.get("task_profile") or "")
        if benchmark_profile == "nl2repo":
            if task_profile != "repo_generation_nl2repo":
                errors.append(f"{task.id}: NL2Repo task profile is missing")
            contract = task.task_dir / "tests" / "instance.json"
            if not contract.is_file():
                errors.append(f"{task.id}: tests/instance.json is missing")
        elif benchmark_profile == "doc2repo":
            if task_profile != "repo_generation_doc2repo":
                errors.append(f"{task.id}: Doc2Repo task profile is missing")
            for relative in ("tests/test_suite.zip", "tests/score_pytest.py", "environment/setup.sh", "environment/files/repo_document.md"):
                if not (task.task_dir / relative).is_file():
                    errors.append(f"{task.id}: {relative} is missing")
        elif benchmark_profile == "swebench_pro_public":
            for relative in (
                "tests/run_script.sh",
                "tests/parser.py",
                "tests/instance.json",
                ".materialized.json",
            ):
                if not (task.task_dir / relative).is_file():
                    errors.append(f"{task.id}: {relative} is missing")
            if metadata.get("verifier_profile") != "swebench_pro_public":
                errors.append(f"{task.id}: SWE-bench Pro verifier profile is missing")
    if errors:
        label = "SWE-Bench Verified" if benchmark_profile == "swebench_verified" else benchmark_profile
        raise ValueError(f"{label} materialization failed validation for {len(errors)} condition(s); examples: {'; '.join(errors[:5])}")


def _upgrade_task_verifiers(tasks: Sequence[Task], benchmark_profile: str = "swebench_verified") -> int:
    """Install the current verifier in already-materialized task trees."""

    if benchmark_profile != "swebench_verified":
        return 0
    return sum(ensure_swebench_verified_verifier(task.task_dir) for task in tasks)


def _sandbox_image_summary(
    tasks: Sequence[Task],
    backend: str,
) -> tuple[int, int, list[tuple[str, str]]]:
    """Return unique/resolved image counts plus a few rewrite examples."""

    from rllm.eval._resolution import _resolve_image

    resolved: set[str] = set()
    rewrites: list[tuple[str, str]] = []
    for task in tasks:
        environment = task.metadata.get("environment", {}) or {}
        source = str(environment.get("docker_image", "python:3.11-slim"))
        target = _resolve_image(task, backend)
        resolved.add(target)
        if source != target and len(rewrites) < 5:
            rewrites.append((source, target))
    return len({str((task.metadata.get("environment", {}) or {}).get("docker_image")) for task in tasks}), len(resolved), rewrites


def _find_free_port() -> int:
    """Return a port available on the same wildcard address the gateway binds."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("0.0.0.0", 0))
        return int(handle.getsockname()[1])


def _start_evaluation_gateway(
    config: DictConfig,
    rollout_engine: Any,
    gateway_factory: Any,
    *,
    max_attempts: int = _DEFAULT_GATEWAY_START_MAX_ATTEMPTS,
) -> Any:
    """Allocate the gateway port at launch time and retry a raced subprocess.

    Standalone vLLM initialization can take several minutes.  Selecting the
    gateway port before that work leaves a large TOCTOU window in which another
    process on the shared Ray node can claim it.  Keep discovery adjacent to
    ``GatewayManager.start`` and retry the specific early-process-exit failure
    with a newly selected port.
    """

    if (
        isinstance(max_attempts, bool)
        or not isinstance(max_attempts, int)
        or max_attempts <= 0
    ):
        raise ValueError("max_attempts must be a positive integer")

    for attempt in range(1, max_attempts + 1):
        port = _find_free_port()
        with open_dict(config):
            config.rllm.gateway.port = port
        gateway = gateway_factory(config, mode="process")
        try:
            gateway.start(rollout_engine)
        except RuntimeError as exc:
            try:
                gateway.stop()
            except Exception:
                logger.exception(
                    "Gateway cleanup failed after startup attempt %d/%d",
                    attempt,
                    max_attempts,
                )
            retryable = "Gateway process exited unexpectedly" in str(exc)
            if not retryable or attempt >= max_attempts:
                raise
            logger.warning(
                "Gateway subprocess exited during startup on port %d; "
                "retrying with a fresh port (attempt %d/%d)",
                port,
                attempt + 1,
                max_attempts,
            )
            continue
        logger.info("Evaluation gateway started on port %d", port)
        return gateway

    raise AssertionError("gateway startup retry loop exited unexpectedly")


def _evaluation_model_from_dict(payload: dict[str, Any]) -> EvaluationModel:
    return EvaluationModel(
        key=str(payload["key"]),
        step=int(payload["step"]),
        path=Path(payload["path"]),
        kind=str(payload["kind"]),
        fingerprint=str(payload["fingerprint"]),
    )


def _attempt_from_dict(payload: dict[str, Any]) -> AttemptSpec:
    return AttemptSpec(
        task_index=int(payload["task_index"]),
        task_id=str(payload["task_id"]),
        attempt_index=int(payload["attempt_index"]),
    )


async def _durable_evaluation_call(function: Any, *args: Any, **kwargs: Any) -> Any:
    """Drain an already-started write before allowing a recovery rescan."""
    pending = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(pending)
    except asyncio.CancelledError:
        await pending
        raise


async def _evaluate_attempt_queue(
    *,
    engine: Any,
    store: EvaluationStore,
    model: EvaluationModel,
    tasks: list[Task],
    attempts: list[AttemptSpec],
    run_id: str,
    budget: int | None = None,
    coordinator: Any = None,
    progress_owner: str | None = None,
    supervisor: Any = None,
) -> dict[str, int]:
    """Run a model's missing attempts through one rolling scheduler window.

    Transient verifier failures receive one queue-tail retry.  The engine has
    already exhausted its internal rollout retries before returning
    ``rollout_retry_exhausted``.  Once the applicable retries are exhausted,
    the attempt is persisted as a terminal failed rollout so later resumes do
    not load the model only to repeat known-bad work.
    """

    from rllm.eval.gateway_supervision import EvaluationTransportHealth

    transport_health = EvaluationTransportHealth()
    effective_budget = budget or max(
        (spec.attempt_index + 1 for spec in attempts),
        default=1,
    )
    persisted_attempts = store.scan_attempts(model, tasks)
    completed_attempts = {task_index: {attempt_index for attempt_index in task_attempts if attempt_index < effective_budget} for task_index, task_attempts in persisted_attempts.items()}
    valid = 0
    failure_records = 0
    terminal_failures = 0
    executions = 0
    verifier_timeouts = sum(
        payload.get("metrics", {}).get("verifier_status") == "timeout"
        for task_attempts in persisted_attempts.values()
        for attempt_index, payload in task_attempts.items()
        if attempt_index < effective_budget
    )
    deferred: list[AttemptSpec] = []
    unresolved = {(spec.task_index, spec.task_id, spec.attempt_index) for spec in attempts}
    executions_by_attempt: dict[tuple[int, str, int], int] = {}
    current_metrics: dict[str, Any] | None = None
    current_metrics_interval = _CURRENT_METRICS_TASK_INTERVAL
    last_current_metrics_threshold = 0

    def completed_task_count() -> int:
        return sum(all(attempt_index in completed_attempts.get(task_index, set()) for attempt_index in range(effective_budget)) for task_index in range(len(tasks)))

    scheduler_state: dict[str, Any] = {
        "active": 0,
        "queued": len(attempts),
        "active_rollouts": [],
        "active_by_stage": {},
        "oldest_active_uid": None,
        "oldest_active_seconds": 0.0,
        "oldest_active_turn_index": None,
        "oldest_active_turn_number": None,
        "oldest_active_stage": None,
    }
    phase = "sampling"
    pass_number = 0
    session_namespace = getattr(engine, "session_namespace", "")

    async def write_progress(*, status: str, force: bool = False) -> None:
        if not force and executions != 1 and executions % 10 != 0:
            return
        complete_tasks = completed_task_count()
        valid_attempts_total = sum(map(len, completed_attempts.values()))
        expected_attempts = len(tasks) * effective_budget
        payload = {
            "status": status,
            "model": model.to_dict(),
            "phase": phase,
            "planned_attempts": len(attempts),
            "executions": executions,
            "valid": valid,
            "failure_records": failure_records,
            "active": int(scheduler_state.get("active", 0)),
            "queued": int(scheduler_state.get("queued", 0)),
            "deferred_retry": len(deferred),
            "unresolved": len(unresolved),
            "verifier_timeouts": verifier_timeouts,
            "valid_attempts_total": valid_attempts_total,
            "expected_attempts": expected_attempts,
            "attempt_coverage": (valid_attempts_total / expected_attempts if expected_attempts else 1.0),
            "observed_task_count": complete_tasks,
            "total_task_count": len(tasks),
            "current_metrics": current_metrics,
            "oldest_active_uid": scheduler_state.get("oldest_active_uid"),
            "oldest_active_seconds": float(scheduler_state.get("oldest_active_seconds", 0.0)),
            "oldest_active_turn_index": scheduler_state.get("oldest_active_turn_index"),
            "oldest_active_turn_number": scheduler_state.get("oldest_active_turn_number"),
            "oldest_active_stage": scheduler_state.get("oldest_active_stage"),
            "active_by_stage": dict(scheduler_state.get("active_by_stage") or {}),
            "active_rollouts": list(scheduler_state.get("active_rollouts") or []),
        }
        try:
            if coordinator is None:
                await asyncio.to_thread(store.write_run_progress, run_id, payload)
            else:
                await asyncio.to_thread(store.write_run_progress, run_id, payload, model_key=model.key)
                await coordinator.update_progress.remote(progress_owner, payload)
        except Exception:
            # progress.json is observability only. Formal rollout persistence
            # below remains strict and is the resume source of truth.
            logger.exception("%s: failed to update evaluation progress", model.key)

    async def refresh_current_metrics(
        *,
        force: bool = False,
        reason: str,
    ) -> bool:
        """Refresh pass@1..n after each 50 newly complete legal tasks."""

        nonlocal current_metrics, last_current_metrics_threshold
        complete_tasks = completed_task_count()
        reached_threshold = (complete_tasks // current_metrics_interval) * current_metrics_interval
        if not force and (reached_threshold < current_metrics_interval or reached_threshold <= last_current_metrics_threshold):
            return False

        result = await _durable_evaluation_call(
            aggregate_model,
            store,
            model,
            tasks,
            effective_budget,
            allow_partial=True,
        )
        current_metrics = {
            "schema_version": 2,
            "sample_budget": effective_budget,
            "completed_task_count": int(result["observed_task_count"]),
            "total_task_count": len(tasks),
            "task_interval": current_metrics_interval,
            "reached_task_threshold": reached_threshold,
            "refresh_reason": reason,
            "selection": "tasks_with_all_legal_attempts",
            "metrics_status": result.get("metrics_status"),
            "selection_warning": result.get("selection_warning"),
            "capability_complete": result.get("capability_complete"),
            "effective_valid_attempts": result.get("effective_valid_attempts"),
            "terminal_infrastructure_failure_tasks": result.get("terminal_infrastructure_failure_tasks"),
            "pass_at_k": result.get("observed_pass_at_k"),
            "mean_sample_accuracy": result.get("observed_mean_sample_accuracy"),
            "average_reward": result.get("observed_average_reward"),
            "average_steps": result.get("observed_average_steps"),
            "average_steps_by_outcome": result.get("observed_average_steps_by_outcome"),
            "average_trajectory_output_tokens": result.get(
                "observed_average_trajectory_output_tokens"
            ),
            "average_trajectory_token_length": result.get(
                "observed_average_trajectory_token_length"
            ),
            "per_sample_metrics": result.get("observed_per_sample_metrics"),
            "official_full_dataset_metrics": bool(result.get("full_dataset_metrics_available")),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        last_current_metrics_threshold = max(
            last_current_metrics_threshold,
            reached_threshold,
        )
        return True

    async def run_pass(
        pass_attempts: list[AttemptSpec],
        *,
        allow_deferred: bool,
    ) -> None:
        nonlocal executions, failure_records, scheduler_state, valid, terminal_failures
        nonlocal pass_number
        if not pass_attempts:
            return
        generation = supervisor.generation if supervisor is not None else None
        if session_namespace:
            engine.session_namespace = f"{session_namespace}-p{pass_number}"
            pass_number += 1

        async def persist(
            task_id: str,
            rollout_index: int,
            result_index: int,
            episode: Any,
        ) -> None:
            nonlocal executions, failure_records, terminal_failures, valid, verifier_timeouts
            spec = pass_attempts[result_index]
            if task_id != spec.task_id or rollout_index != spec.attempt_index:
                raise RuntimeError(f"engine callback identity mismatch: got {task_id}:{rollout_index}, expected {spec.task_id}:{spec.attempt_index}")
            reason = infrastructure_failure_reason(episode)
            from rllm.eval.gateway_supervision import EvaluationRuntimeError, runtime_failure_action

            diagnostics = (episode.metadata or {}).get("infrastructure_failure") or (episode.metrics or {}).get("infrastructure_failure") or {}
            runtime_action = runtime_failure_action(reason, diagnostics)
            pro_task = (tasks[spec.task_index].metadata.get("rllm") or {}).get("eval_verifier_profile") == "swebench_pro_public"
            outage = transport_health.observe(task_id, reason) if pro_task else None
            if outage is not None:
                runtime_action = "abort"
                diagnostics = outage

            if supervisor is not None and (
                not supervisor.accepting_results or supervisor.generation != generation
            ):
                raise asyncio.CancelledError("stale evaluation generation")
            if runtime_action is not None and supervisor is not None:
                supervisor.report_failure(
                    "evaluation_transport_outage" if outage else reason,
                    {"task_id": task_id, "attempt_index": rollout_index, "failure": diagnostics},
                    recoverable=runtime_action == "recover",
                )
            # A write that has started must finish before recovery rescans the
            # directory; cancellation of to_thread alone cannot stop the write.
            await _durable_evaluation_call(
                store.save_episode,
                model,
                tasks[spec.task_index],
                spec,
                episode,
            )
            if runtime_action is not None:
                raise EvaluationRuntimeError(f"{model.key}: {'evaluation_transport_outage' if outage else reason}", diagnostics, recoverable=runtime_action == "recover")
            executions += 1
            key = (spec.task_index, spec.task_id, spec.attempt_index)
            executions_by_attempt[key] = executions_by_attempt.get(key, 0) + 1
            if reason is None:
                valid += 1
                unresolved.discard(key)
                completed_attempts.setdefault(spec.task_index, set()).add(spec.attempt_index)
                if (episode.metrics or {}).get("verifier_status") == "timeout":
                    verifier_timeouts += 1
            else:
                failure_records += 1
                should_defer = (
                    allow_deferred
                    and reason
                    in {
                        "verifier_reward_missing",
                        "verifier_execution_failed",
                        "verifier_sandbox_lost",
                        "verifier_network_unavailable",
                        "verifier_result_read_failed",
                        "verifier_result_invalid",
                        "verifier_result_unconfirmed",
                    }
                    and executions_by_attempt[key] < 2
                )
                if should_defer:
                    deferred.append(spec)
                else:
                    records_seen = len(
                        await asyncio.to_thread(store.failure_records, model, spec)
                    )
                    await _durable_evaluation_call(
                        store.save_terminal_failure,
                        model,
                        tasks[spec.task_index],
                        spec,
                        episode,
                        failure_records_seen=records_seen,
                    )
                    terminal_failures += 1
                    unresolved.discard(key)
                    completed_attempts.setdefault(spec.task_index, set()).add(
                        spec.attempt_index
                    )
                logger.warning(
                    "%s: infrastructure failure %s for %s attempt %d "
                    "(execution %d, terminal=%s)",
                    model.key,
                    reason,
                    task_id,
                    rollout_index,
                    executions,
                    not should_defer,
                )
            metrics_refreshed = await refresh_current_metrics(
                reason="task_interval",
            )
            if metrics_refreshed:
                await write_progress(status="running", force=True)
            if executions == 1 or executions % 10 == 0:
                logger.info(
                    "%s progress: executions=%d, valid=%d/%d, failure_records=%d, deferred=%d, unresolved=%d",
                    model.key,
                    executions,
                    valid,
                    len(attempts),
                    failure_records,
                    len(deferred),
                    len(unresolved),
                )

        async def scheduler_update(state: dict[str, Any]) -> None:
            nonlocal scheduler_state
            scheduler_state = dict(state)
            event = str(state.get("event", ""))
            await write_progress(
                status="running",
                force=event in {"started", "heartbeat", "finished"},
            )

        await engine.execute_tasks(
            [tasks[spec.task_index] for spec in pass_attempts],
            task_ids=[spec.task_id for spec in pass_attempts],
            rollout_indices=[spec.attempt_index for spec in pass_attempts],
            is_validation=True,
            on_episode_complete=persist,
            collect_results=False,
            on_scheduler_update=scheduler_update,
        )

    await refresh_current_metrics(force=True, reason="resume_scan")
    await write_progress(status="running", force=True)
    await run_pass(attempts, allow_deferred=True)
    if deferred:
        phase = "deferred_retry"
        logger.info(
            "%s: retrying %d infrastructure attempt(s) at queue tail",
            model.key,
            len(deferred),
        )
        await write_progress(status="running", force=True)
        await run_pass(list(deferred), allow_deferred=False)
    phase = "complete"
    await refresh_current_metrics(force=True, reason="model_complete")
    scheduler_state = {
        "active": 0,
        "queued": 0,
        "active_rollouts": [],
        "active_by_stage": {},
        "oldest_active_uid": None,
        "oldest_active_seconds": 0.0,
        "oldest_active_turn_index": None,
        "oldest_active_turn_number": None,
        "oldest_active_stage": None,
    }
    await write_progress(
        status="complete" if not unresolved else "incomplete",
        force=True,
    )
    return {
        "sampled": len(attempts),
        "executions": executions,
        "valid": valid,
        "failures": failure_records,
        "terminal_failures": terminal_failures,
        "deferred_retries": len(deferred),
        "unresolved": len(unresolved),
        "verifier_timeouts": verifier_timeouts,
    }


class _CurrentRayJobExecutionBackend:
    """Minimal hook backend used by standalone checkpoint evaluation."""

    @staticmethod
    def get_execution_node_ids() -> list[str]:
        from rllm.trainer.verl.ray_nodes import (
            discover_current_job_gpu_node_ids,
        )

        return discover_current_job_gpu_node_ids()


def _shutdown_checkpoint_runtime(
    *,
    warm_queue: Any,
    engine: Any,
    hooks: Any,
    gateway: Any,
    llm_server_manager: Any,
    ray_module: Any,
) -> None:
    """Release one checkpoint's resources in dependency-safe order."""

    if warm_queue is not None:
        try:
            warm_queue.shutdown(join_timeout=60.0)
        except Exception:
            logger.exception("Warm queue shutdown failed during model teardown")
    if engine is not None:
        try:
            engine.shutdown()
        except Exception:
            logger.exception(
                "AgentFlowEngine shutdown failed during model teardown"
            )
    if hooks is not None:
        try:
            hooks.shutdown_backend()
        except Exception:
            logger.exception(
                "MiniSandbox backend shutdown failed during model teardown"
            )
    if gateway is not None:
        try:
            gateway.stop()
        except Exception:
            logger.exception("Gateway shutdown failed during model teardown")
    # VERL 0.8 has no public LLMServerManager close API. Its standalone
    # replicas create job-scoped placement groups, so killing only this owner
    # actor leaves all GPU bundles reserved.
    _teardown_llm_server_manager(llm_server_manager, ray_module)


class CheckpointEvaluationRunner:
    """Ray-owned lifecycle for one model's standalone rollout replicas."""

    def __init__(self):
        self._cancel_requested = threading.Event()
        self._evaluation_loop = None
        self._evaluation_task = None
        self._node_context = None
        self._coordinator = None
        self._isolation = None

    def cancel(self):
        self._cancel_requested.set()
        loop, task = self._evaluation_loop, self._evaluation_task
        if loop is not None and task is not None and not loop.is_closed():
            loop.call_soon_threadsafe(task.cancel)

    def run(self, *args, node_context=None, coordinator=None, **kwargs):
        if node_context is None:
            return self._run(*args, **kwargs)
        import ray

        from rllm.eval.node_runtime import NodeRuntimeIsolation, wait_for_owned_resources

        self._node_context, self._coordinator = node_context, coordinator
        config = args[0] if args else kwargs["config"]
        with NodeRuntimeIsolation(ray, coordinator, node_context["owner"], node_context) as isolation:
            self._isolation = isolation
            try:
                return self._run(*args, **kwargs)
            finally:
                failed = sys.exc_info()[0] is not None
                try:
                    isolation.cleanup()
                    resources = ray.get(coordinator.get_resources.remote(node_context["owner"]))
                    wait_for_owned_resources(
                        ray, resources, node_context,
                        timeout=float(_config_value(config, "eval.gpu_release_timeout", 300)),
                        stable=float(_config_value(config, "eval.gpu_quiescence_seconds", 0)),
                    )
                except Exception:
                    if not failed:
                        raise
                    logger.exception("Node cleanup failed; preserving original runner failure")

    def _run(
        self,
        config: DictConfig,
        model_payload: dict[str, Any],
        tasks: list[Task],
        attempt_payloads: list[dict[str, Any]],
        store_root: str,
        semantic_id: str,
        semantic_config: dict[str, Any],
        runs_root: str,
        run_log_path: str,
        attach_file_logging: bool = True,
    ) -> dict[str, Any]:
        if attach_file_logging:
            _attach_run_logging(Path(run_log_path))
        import ray
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.fs import copy_to_local
        from verl.workers.rollout.llm_server import LLMServerManager

        from rllm.engine.agentflow_engine import AgentFlowEngine
        from rllm.engine.rollout import VerlEngine
        from rllm.eval.agent_loader import load_agent
        from rllm.gateway.manager import GatewayManager
        from rllm.hooks import SandboxTaskHooks
        from rllm.sandbox.snapshot import install_script_for
        from rllm.sandbox.warm_queue import WarmQueue

        model = _evaluation_model_from_dict(model_payload)
        _validate_model_tensor_parallelism(config, model_path=str(model.path))
        runtime_fingerprint = fingerprint_hf_model(model.path)
        if runtime_fingerprint != model.fingerprint:
            raise ValueError(f"model {model.key} changed after evaluation preflight; refusing to sample mixed weights")
        attempts = [_attempt_from_dict(payload) for payload in attempt_payloads]
        if not attempts:
            return {"sampled": 0, "valid": 0, "failures": 0}
        logger.info("Runner starting %s: %d trajectories", model.key, len(attempts))
        if self._cancel_requested.is_set():
            raise RuntimeError("evaluation cancelled before model startup")
        # Never inherit the sampled training-rollout logger into evaluation;
        # the EvaluationStore below is the only trajectory writer here.
        for variable in (
            "RLLM_ROLLOUT_LOG_PATH",
            "RLLM_ROLLOUT_LOG_START_INDEX",
        ):
            os.environ.pop(variable, None)

        # This actor receives its own config copy.  Mutating the served path
        # cannot affect the driver or the next model actor.
        with open_dict(config):
            config.actor_rollout_ref.model.path = str(model.path)
            if "model" not in config:
                config.model = {}
            config.model.name = str(model.path)
            config.actor_rollout_ref.rollout.nnodes = int(config.rollout.nnodes)
            config.actor_rollout_ref.rollout.n_gpus_per_node = int(config.rollout.n_gpus_per_node)
            # The separated trainer intentionally starts rollout with dummy
            # weights and fills them through checkpoint-engine sync.  This
            # evaluation runner has no trainer, so vLLM must load HF weights.
            if str(config.actor_rollout_ref.rollout.load_format).lower() == "dummy":
                raise ValueError("standalone checkpoint evaluation cannot use actor_rollout_ref.rollout.load_format=dummy; use auto")
            sequence_budget = _sequence_budget_config(config)
            _configure_evaluation_server_generation_limit(
                config,
                sequence_budget,
            )
            if not sequence_budget["dynamic"]:
                config.data.max_prompt_length = int(sequence_budget["initial_prompt_tokens"])
                config.data.max_response_length = int(sequence_budget["cumulative_response_tokens"])
            config.rllm.gateway.tunnel = None
            config.actor_rollout_ref.rollout.val_kwargs.do_sample = float(config.eval.temperature) > 0
            config.actor_rollout_ref.rollout.val_kwargs.temperature = float(config.eval.temperature)
            config.actor_rollout_ref.rollout.val_kwargs.top_p = float(config.eval.top_p)
            config.actor_rollout_ref.rollout.val_kwargs.top_k = int(config.eval.top_k)
        OmegaConf.resolve(config)

        local_path = copy_to_local(
            str(model.path),
            use_shm=bool(config.actor_rollout_ref.model.get("use_shm", False)),
        )
        trust_remote_code = bool(config.data.get("trust_remote_code", False))
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(
            local_path,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )

        _install_verl_standalone_port_ranges(config)
        from rllm.trainer.verl.server_lifecycle import install_graceful_vllm_shutdown

        install_graceful_vllm_shutdown()

        llm_server_manager = None
        gateway = None
        engine = None
        warm_queue = None
        hooks = None
        try:
            logger.info(
                "%s: creating standalone vLLM servers with per-turn generation cap=%d",
                model.key,
                int(config.actor_rollout_ref.rollout.response_length),
            )
            manager_class = LLMServerManager
            manager_ray = ray
            if self._isolation is not None:
                from rllm.eval.node_runtime import NodeRayResources
                manager_class = self._isolation.manager_class(LLMServerManager)
                manager_ray = NodeRayResources(ray, self._node_context["node_id"])
            llm_server_manager = _create_llm_server_manager_with_retry(
                manager_class,
                config=config,
                ray_module=manager_ray,
                model_key=model.key,
                isolation=self._isolation,
            )
            rollout_engine = VerlEngine(
                config=config,
                server_manager=llm_server_manager.get_client(),
                tokenizer=tokenizer,
                processor=processor,
            )
            rollout_engine.server_addresses = llm_server_manager.get_addresses()
            logger.info(
                "%s: vLLM ready at %d endpoint(s)",
                model.key,
                len(rollout_engine.server_addresses),
            )
            sandbox_backend = str(config.swe.sandbox_backend).strip().lower()
            hooks = SandboxTaskHooks(
                sandbox_backend=sandbox_backend,
            )
            # vLLM placement is now final. Discover only GPU actors owned by
            # this Ray job and start one privileged MiniSandbox service on
            # each of their nodes before the first rollout is admitted.
            backend = _CurrentRayJobExecutionBackend()
            if self._node_context is not None:
                from types import SimpleNamespace
                backend = SimpleNamespace(get_execution_node_ids=lambda: [self._node_context["node_id"]])
            hooks.initialize_backend(backend, config)
            pass_at_k_group_size = _positive_int(
                _config_value(config, "eval.pass_n"),
                "eval.pass_n",
            )
            logger.info(
                "%s: pass@%d attempts use %s routing across %d rollout replica(s)",
                model.key,
                pass_at_k_group_size,
                str(
                    _config_value(
                        config,
                        "rllm.gateway.routing.mode",
                        "sticky_least_loaded",
                    )
                ),
                len(rollout_engine.server_addresses),
            )
            val_params = _evaluation_session_sampling_params(config, dict(rollout_engine.val_sampling_params))
            logger.info("%s: effective session sampling parameters: %s", model.key, val_params)

            gateway = _start_evaluation_gateway(
                config,
                rollout_engine,
                self._isolation.gateway_class(GatewayManager) if self._isolation is not None else GatewayManager,
            )
            logger.info("%s: cumulative gateway ready", model.key)

            agent_flow = load_agent("codeflow")
            agent_flow.set_protocol("native_tool_call")
            agent_flow.configure_codeflow_tool_mode(
                normalize_codeflow_tool_mode(
                    _config_value(
                        config,
                        "swe.codeflow_tool_mode",
                        None,
                    )
                )
            )
            agent_flow.configure_codeflow_tool_restricted_mode(
                list(
                    normalize_codeflow_tool_restricted_mode(
                        _config_value(
                            config,
                            "swe.codeflow_tool_restricted_mode",
                            None,
                        )
                    )
                )
            )
            agent_flow.max_turns = int(config.swe.max_turns)
            agent_flow.command_timeout = float(config.swe.command_timeout)
            agent_flow.limit_termination_success_reward = 1.0
            agent_flow.limit_termination_outcome_mode = str(
                _config_value(
                    config,
                    "swe.limit_termination_outcome_mode",
                    "discount_success",
                )
            )
            warm_size = int(config.rllm.workflow.warm_queue_size)
            if warm_size != 0:
                schedule = [tasks[spec.task_index] for spec in attempts]
                if warm_size < 0:
                    warm_size = int(config.rllm.workflow.n_parallel_tasks)
                warm_size = min(warm_size, len(schedule))
                if warm_size > 0:
                    warm_queue = WarmQueue(
                        schedule=schedule,
                        backend=str(config.swe.sandbox_backend),
                        size=warm_size,
                        install_script=install_script_for(agent_flow),
                    )
                    hooks.warm_queue = warm_queue
                    warm_queue.start()

            admission = None
            if self._coordinator is not None:
                from rllm.eval.scheduling import RayEvaluationAdmission
                admission = RayEvaluationAdmission(self._coordinator, self._node_context["owner"])
            engine = AgentFlowEngine(
                agent_flow=agent_flow,
                evaluator=None,
                gateway=gateway,
                model=str(model.path),
                n_parallel_tasks=int(config.rllm.workflow.n_parallel_tasks),
                retry_limit=int(config.rllm.workflow.retry_limit),
                raise_on_error=False,
                hooks=hooks,
                val_sampling_params=val_params,
                milestone_reward_config={"enable": False},
                model_window_tokens=(int(sequence_budget["model_window_tokens"]) if sequence_budget["dynamic"] else None),
                rollout_startup_window=config.rllm.workflow.get(
                    "rollout_startup_window",
                    None,
                ),
                # Evaluation creates pass@k attempts explicitly instead of
                # using rllm.rollout.n_val.  Keep the attempt index as the
                # stable stripe slot and advertise the requested k as the
                # routing group size, including when a resumed run schedules
                # only a subset of missing attempts.
                rollout_group_size=pass_at_k_group_size,
                evaluation_admission=admission,
            )
            store = EvaluationStore(
                store_root,
                semantic_id=semantic_id,
                semantic_config=semantic_config,
                runs_root=runs_root,
            )
            store.ensure_model(model)

            async def evaluate() -> dict[str, int]:
                self._evaluation_loop = asyncio.get_running_loop()
                self._evaluation_task = asyncio.current_task()
                if self._cancel_requested.is_set():
                    raise asyncio.CancelledError("evaluation cancelled during model startup")
                from rllm.eval.gateway_supervision import EvaluationGatewaySupervisor

                run_id = Path(run_log_path).parent.name
                namespace = "eval-" + uuid.uuid4().hex

                async def record_event(payload):
                    await asyncio.to_thread(store.write_runtime_event, run_id, model.key, payload)

                supervisor = EvaluationGatewaySupervisor(gateway, engine, record_event=record_event)

                async def run_generation(generation):
                    engine.session_namespace = f"{namespace}-g{generation}"
                    engine.evaluation_session_ids.clear()
                    missing = await asyncio.to_thread(store.missing_attempts, model, tasks, int(config.eval.pass_n))
                    return await _evaluate_attempt_queue(
                        engine=engine,
                        store=store,
                        model=model,
                        tasks=tasks,
                        attempts=missing,
                        run_id=run_id,
                        budget=int(config.eval.pass_n),
                        coordinator=self._coordinator,
                        progress_owner=(self._node_context["owner"] if self._node_context else None),
                        supervisor=supervisor,
                    )

                return await supervisor.run(run_generation)

            from rllm.eval.gateway_supervision import run_evaluation_coroutine

            return run_evaluation_coroutine(
                evaluate(), cleanup_timeout=gateway.supervision_config.recovery_stall_timeout_seconds,
            )
        finally:
            _shutdown_checkpoint_runtime(
                warm_queue=warm_queue,
                engine=engine,
                hooks=hooks,
                gateway=gateway,
                llm_server_manager=llm_server_manager,
                ray_module=ray,
            )
            llm_server_manager = None


def _attach_run_logging(log_path: Path) -> None:
    """Attach the shared run log in both the driver and the remote owner actor."""

    resolved = str(log_path.resolve())
    root_logger = logging.getLogger()
    if any(getattr(handler, "_rllm_eval_log_path", None) == resolved for handler in root_logger.handlers):
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler._rllm_eval_log_path = resolved  # type: ignore[attr-defined]
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root_logger.addHandler(handler)


def _setup_run_logging(
    store: EvaluationStore,
    run_id: str,
    console_log_path: str | Path | None = None,
) -> Path:
    run_dir = store.runs_dir / run_id
    log_path = run_dir / "eval.log"
    if console_log_path is None:
        _attach_run_logging(log_path)
        return log_path

    physical_path = Path(console_log_path).expanduser().resolve()
    if physical_path != log_path.resolve():
        raise ValueError(f"eval.console_log_path must be the canonical run log path {log_path}, got {physical_path}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path.touch(exist_ok=True)
    # The launcher captures the complete process stdout/stderr with tee.
    # Attaching a FileHandler here would duplicate Python and Ray-forwarded
    # logging in the same physical file.
    return log_path


def _wait_for_gpu_reclamation(
    ray_module,
    baseline_available: float,
    *,
    timeout_seconds: float,
    stable_seconds: float = 0.0,
) -> None:
    stable_seconds = _nonnegative_float(stable_seconds, "stable_seconds")
    deadline = time.monotonic() + timeout_seconds
    stable_since: float | None = None
    while time.monotonic() < deadline:
        available = float(ray_module.available_resources().get("GPU", 0.0))
        if available + 1e-6 >= baseline_available:
            if stable_seconds == 0:
                return
            now = time.monotonic()
            if stable_since is None:
                stable_since = now
            elif now - stable_since >= stable_seconds:
                return
        else:
            stable_since = None
        time.sleep(2.0)
    available = float(ray_module.available_resources().get("GPU", 0.0))
    raise TimeoutError(f"rollout GPU resources were not reclaimed after model teardown: available={available:g}, expected_at_least={baseline_available:g}")


def _verify_evaluation_runtime(config, ray) -> None:
    from rllm.utils.runtime_provenance import runtime_provenance, verify_cluster_runtime

    provenance = runtime_provenance()
    node_provenance = verify_cluster_runtime(ray, provenance)
    run_id = str(config.eval.run_id)
    provenance_path = Path(str(config.eval.output_root)) / "runs" / run_id / "runtime_provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps({"driver": provenance, "nodes": node_provenance}, indent=2) + "\n")
    logger.info("Verified evaluation runtime on %d nodes: source_id=%s image_digest=%s", len(node_provenance), provenance["source_id"], provenance["image_digest"])


def _scheduling_mode(config: DictConfig) -> str:
    mode = str(_config_value(config, "eval.scheduling_mode", "sequential"))
    if mode not in {"sequential", "node_parallel"}:
        raise ValueError("eval.scheduling_mode must be sequential or node_parallel")
    return mode


def _run_models_node_parallel(config, models, tasks, store, budget, run_log_path, *, ray, console_captured):
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from rllm.eval.node_runtime import available_per_node, cleanup_resources, wait_for_owned_resources
    from rllm.eval.scheduling import EvaluationCoordinator, select_evaluation_nodes
    from rllm.eval.gateway_supervision import EvaluationRuntimeError, minisandbox_node_cleanup_failure

    rollout = config.actor_rollout_ref.rollout
    if bool(OmegaConf.select(rollout, "disaggregation.enabled", default=False)):
        raise ValueError("node_parallel does not support cross-node prefill/decode disaggregation")
    if int(_config_value(config, "rllm.workflow.warm_queue_size", 0)) != 0:
        raise ValueError("node_parallel requires rllm.workflow.warm_queue_size=0 so sandbox setup uses shared admission")
    replica_gpus = math.prod(_positive_int(rollout.get(key, 1), key) for key in (
        "tensor_model_parallel_size", "data_parallel_size", "pipeline_model_parallel_size",
    ))
    nodes = select_evaluation_nodes(
        ray.nodes(), available_per_node(ray),
        _positive_int(config.rollout.nnodes, "rollout.nnodes"),
        _positive_int(config.rollout.n_gpus_per_node, "rollout.n_gpus_per_node"),
        replica_gpus,
    )
    task_limit = _positive_int(config.rllm.workflow.n_parallel_tasks, "rllm.workflow.n_parallel_tasks")
    startup = _config_value(config, "rllm.workflow.rollout_startup_window", None)
    startup_limit = task_limit if startup is None else _positive_int(startup, "rllm.workflow.rollout_startup_window")
    release_timeout = _positive_float(_config_value(config, "eval.gpu_release_timeout", 300), "eval.gpu_release_timeout")
    _nonnegative_float(_config_value(config, "eval.gpu_quiescence_seconds", 0), "eval.gpu_quiescence_seconds")
    queue = [(model, store.missing_attempts(model, tasks, budget)) for model in sorted(models, key=lambda item: item.step)]
    queue = [(model, missing) for model, missing in queue if missing]
    actor_class = ray.remote(num_cpus=1, max_concurrency=2)(CheckpointEvaluationRunner)
    initial_owners = ["eval_" + uuid.uuid4().hex for _ in range(min(len(nodes), len(queue)))]
    coordinator = ray.remote(num_cpus=0, max_concurrency=100_000)(EvaluationCoordinator).remote(
        task_limit, startup_limit, initial_owners,
    )
    active = {}
    quarantined = {}
    states = {}
    run_id = str(_config_value(config, "eval.run_id", run_log_path.parent.name if run_log_path else "evaluation"))
    last_progress = 0.0

    def launch(node, owner):
        if not queue:
            return
        model, missing = queue.pop(0)
        context = {**node, "owner": owner}
        local_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        with open_dict(local_config):
            local_config.rollout.nnodes = 1
        runner = actor_class.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node["node_id"], soft=False),
            name=owner,
        ).remote()
        state = {"model": model.key, "node_id": node["node_id"], "status": "loading"}
        states[owner] = state
        # Register locally before submitting run, so submission failures clean it.
        entry = {"runner": runner, "context": context, "model": model, "ref": None}
        active[owner] = entry
        entry["ref"] = runner.run.remote(
            local_config, model.to_dict(), tasks,
            [spec.__dict__ for spec in missing], str(store.root.parent),
            store.semantic_id, store.semantic_config, str(store.runs_dir),
            str(run_log_path or (store.runs_dir / run_id / "eval.log")),
            not console_captured, node_context=context, coordinator=coordinator,
        )
        logger.info("Assigned %s to evaluation node %s: %d missing attempts", model.key, node["node_id"], len(missing))

    def progress(status, *, force=False):
        nonlocal last_progress
        if not force and time.monotonic() - last_progress < 5:
            return
        snapshot = ray.get(coordinator.snapshot.remote())
        for owner, model_progress in snapshot["models"].items():
            if status == "running" and owner in active:
                states[owner]["status"] = "reclaiming" if model_progress.get("status") == "complete" else "running"
        store.write_run_progress(run_id, {
            "schema_version": 4, "scheduling_mode": "node_parallel", "status": status,
            "nodes": nodes, "model_runtimes": dict(states),
            "queued_models": [model.key for model, _missing in queue],
            "shared_admission": snapshot,
        })
        last_progress = time.monotonic()

    try:
        for node, owner in zip(nodes, initial_owners, strict=False):
            launch(node, owner)
        progress("running", force=True)
        while active:
            refs = [entry["ref"] for entry in active.values()]
            ready, _ = ray.wait(refs, num_returns=1, timeout=1)
            if not ready:
                progress("running")
                continue
            owner = next(owner for owner, entry in active.items() if entry["ref"] == ready[0])
            entry = active[owner]
            try:
                result = ray.get(ready[0])  # The runner has already reclaimed its node.
            except Exception as exc:
                incident = (minisandbox_node_cleanup_failure(exc)
                            if _config_value(config, "swe.sandbox_backend", "minisandbox") == "minisandbox" else None)
                if incident is None:
                    raise
                # Each owner in this mode uses a disjoint GPU/sandbox node.
                # Keep the failed node fatal and reserved, but let other nodes
                # finish their independent model evaluations. Never refill it.
                capacity_available = ray.get(coordinator.quarantine_owner.remote(owner))
                states[owner].update(status="quarantined", failure=incident)
                quarantined[owner] = active.pop(owner)
                if not capacity_available:
                    raise EvaluationRuntimeError(
                        "Quarantined nodes exhausted evaluation admission capacity", incident,
                        recoverable=False,
                    ) from exc
                logger.error("Quarantined MiniSandbox evaluation node %s (%s); other nodes continue",
                             entry["context"]["node_id"], entry["model"].key)
                progress("running", force=True)
                continue
            ray.kill(entry["runner"], no_restart=True)
            successor = "eval_" + uuid.uuid4().hex if queue else None
            # Keep the slot reserved throughout cleanup and successor loading;
            # do not temporarily lend its permits to a still-running model.
            ray.get(coordinator.close_owner.remote(owner, successor))
            states[owner]["status"] = "complete"
            del active[owner]
            logger.info("Completed %s on node %s: %s", entry["model"].key, entry["context"]["node_id"], result)
            if successor is not None:
                launch(entry["context"], successor)
            store.update_manifest_models(models, tasks, budget)
            write_summary(store, models, tasks, budget)
            progress("running", force=True)
        if quarantined:
            raise EvaluationRuntimeError(
                "MiniSandbox cleanup unconfirmed on quarantined evaluation nodes; healthy nodes finished",
                {"failed_nodes": {owner: states[owner] for owner in quarantined},
                 "queued_models": [model.key for model, _ in queue]}, recoverable=False,
            )
        progress("complete", force=True)
    except BaseException:
        # Stop every owner first, then bound the combined grace period. Cleanup
        # of one node must not leave another runner generating during failure.
        active.update(quarantined)
        for entry in active.values():
            try:
                entry["runner"].cancel.remote()
            except Exception:
                logger.exception("Could not request evaluation cancellation")
        pending = [entry["ref"] for entry in active.values() if entry["ref"] is not None]
        deadline = time.monotonic() + release_timeout
        while pending and time.monotonic() < deadline:
            _done, pending = ray.wait(pending, num_returns=1, timeout=min(1, max(0, deadline - time.monotonic())))
        for owner, entry in active.items():
            if owner not in quarantined:
                states[owner]["status"] = "failed_or_cancelled"
            try:
                ray.kill(entry["runner"], no_restart=True)
                resources = ray.get(coordinator.get_resources.remote(owner))
                cleanup_resources(ray, resources, node_id=entry["context"]["node_id"], graceful=True)
                wait_for_owned_resources(ray, resources, entry["context"], timeout=release_timeout, stable=0)
                if owner not in quarantined:
                    ray.get(coordinator.close_owner.remote(owner))
            except Exception:
                logger.exception("Failed cleaning evaluation runtime %s; preserving original error", owner)
        try:
            progress("failed", force=True)
            store.update_manifest_models(models, tasks, budget)
            write_summary(store, models, tasks, budget)
        except Exception:
            logger.exception("Failed updating evaluation failure progress")
        raise
    finally:
        failed = sys.exc_info()[0] is not None
        try:
            ray.kill(coordinator, no_restart=True)
        except Exception:
            if not failed:
                raise
            logger.exception("Coordinator cleanup failed; preserving original evaluation error")


def _run_models(
    config: DictConfig,
    models: Sequence[EvaluationModel],
    tasks: list[Task],
    store: EvaluationStore,
    budget: int,
    run_log_path: Path | None = None,
    *,
    console_captured: bool = False,
) -> None:
    mode = _scheduling_mode(config)
    if not any(store.missing_attempts(model, tasks, budget) for model in models):
        logger.info(
            "All selected models already have complete pass@%d trajectories; no Ray/vLLM runtime will be started",
            budget,
        )
        return

    import ray

    from rllm.trainer.ray_init_utils import get_ray_init_settings
    from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env

    own_ray = False
    if not ray.is_initialized():
        ray.init(
            runtime_env=get_ppo_ray_runtime_env(),
            **get_ray_init_settings(config),
        )
        own_ray = True

    actor_class = ray.remote(num_cpus=1)(CheckpointEvaluationRunner)
    try:
        _verify_evaluation_runtime(config, ray)
        if mode == "node_parallel":
            _run_models_node_parallel(
                config, models, tasks, store, budget, run_log_path,
                ray=ray, console_captured=console_captured,
            )
            return
        for model in sorted(models, key=lambda item: item.step):
            missing = store.missing_attempts(model, tasks, budget)
            if not missing:
                logger.info(
                    "Skipping %s: pass@%d trajectories already complete",
                    model.key,
                    budget,
                )
                store.update_manifest_models(models, tasks, budget)
                write_summary(store, models, tasks, budget)
                continue

            logger.info(
                "Evaluating %s from %s: %d missing trajectories",
                model.key,
                model.path,
                len(missing),
            )
            baseline_available = float(ray.available_resources().get("GPU", 0.0))
            runner = actor_class.remote()
            runner_failed = False
            try:
                result = ray.get(
                    runner.run.remote(
                        config,
                        model.to_dict(),
                        tasks,
                        [spec.__dict__ for spec in missing],
                        str(store.root.parent),
                        store.semantic_id,
                        store.semantic_config,
                        str(store.runs_dir),
                        str(run_log_path or (store.root / "eval.log")),
                        not console_captured,
                    )
                )
                logger.info("Completed %s: %s", model.key, result)
            except BaseException:
                runner_failed = True
                raise
            finally:
                ray.kill(runner, no_restart=True)
                try:
                    _wait_for_gpu_reclamation(
                        ray,
                        baseline_available,
                        timeout_seconds=_positive_float(
                            _config_value(config, "eval.gpu_release_timeout", 300),
                            "eval.gpu_release_timeout",
                        ),
                        stable_seconds=_nonnegative_float(
                            _config_value(config, "eval.gpu_quiescence_seconds", 0.0),
                            "eval.gpu_quiescence_seconds",
                        ),
                    )
                except TimeoutError:
                    if not runner_failed:
                        raise
                    logger.exception(
                        "Rollout GPU cleanup also timed out after %s failed; preserving the original runner error",
                        model.key,
                    )

            store.update_manifest_models(models, tasks, budget)
            write_summary(store, models, tasks, budget)
    finally:
        if own_ray:
            ray.shutdown()


def _run_evaluation(config: DictConfig) -> None:
    _configure_model_profile(config, evaluation=True)
    _validate_model_tensor_parallelism(config)
    _scheduling_mode(config)
    _reject_removed_rollout_log_probability(config)
    tool_mode = _resolved_codeflow_tool_mode(config)
    restricted_mode = _resolved_codeflow_tool_restricted_mode(
        config,
        tool_mode=tool_mode,
    )
    _validate_codeflow_tool_configuration(tool_mode, restricted_mode)
    pass_n = _positive_int(_config_value(config, "eval.pass_n"), "eval.pass_n")
    routing_mode = str(
        _config_value(
            config,
            "rllm.gateway.routing.mode",
            "sticky_least_loaded",
        )
    )
    if pass_n > 1 and routing_mode != "group_striped_adaptive":
        raise ValueError(
            "eval.pass_n > 1 requires "
            "rllm.gateway.routing.mode=group_striped_adaptive so one task's "
            "pass@k attempts are striped across rollout replicas"
        )
    sequence_budget = _sequence_budget_config(
        config,
        explicit_overrides=_hydra_task_overrides(),
    )
    checkpoint_stride = _positive_int(
        _config_value(config, "eval.checkpoint_stride", 1),
        "eval.checkpoint_stride",
    )
    checkpoint_interval = _positive_int(
        _config_value(config, "eval.checkpoint_interval", 10),
        "eval.checkpoint_interval",
    )
    evaluate_base = _boolean(
        _config_value(config, "eval.evaluate_base", True),
        "eval.evaluate_base",
    )
    experiment_root = Path(str(_config_value(config, "eval.experiment_root"))).expanduser().resolve()
    output_root = Path(str(_config_value(config, "eval.output_root"))).expanduser().resolve()

    benchmark = _benchmark_config(config)
    sandbox_backend = _configure_minisandbox_evaluation(config, benchmark)
    if sandbox_backend == "k8s":
        from rllm.sandbox.backends.k8s import check_k8s_adapter
        check_k8s_adapter()
    dataset_name = benchmark["dataset_name"]
    dataset_split = benchmark["dataset_split"]
    dataset = _load_sandbox_dataset(dataset_name, dataset_split)
    tasks = _filter_excluded_benchmark_tasks(list(dataset.data), benchmark["profile"])
    max_tasks = _config_value(config, "eval.max_tasks", None)
    task_ids_file = _config_value(config, "eval.task_ids_file", None)
    if max_tasks is not None and task_ids_file is not None:
        raise ValueError("eval.max_tasks and eval.task_ids_file are mutually exclusive")
    if task_ids_file is not None:
        tasks = _filter_benchmark_tasks_by_id_file(
            tasks,
            Path(str(task_ids_file)).expanduser(),
        )
    elif max_tasks is not None:
        tasks = tasks[: _positive_int(max_tasks, "eval.max_tasks")]
    elif len(tasks) != benchmark["expected_tasks"]:
        raise ValueError(f"expected {benchmark['expected_tasks']} {dataset_name}/{dataset_split} tasks, got {len(tasks)}")
    if not tasks or not all(isinstance(task, Task) for task in tasks):
        raise TypeError(f"materialized {dataset_name} rows must load as Task objects")
    _configure_sandbox_resources(
        dataset,
        benchmark_profile=benchmark["profile"],
        cpus=int(_config_value(config, "swe.sandbox_cpus", 2)),
        memory_mb=int(_config_value(config, "swe.sandbox_memory_mb", 8192)),
    )
    timeout_resource_overrides = _override_verifier_timeout_task_resources(
        tasks,
        benchmark["profile"],
    )
    verifier_timeout = _positive_float(
        _config_value(
            config,
            "swe.verifier_timeout",
            _DEFAULT_VERIFIER_TIMEOUT_SECONDS,
        ),
        "swe.verifier_timeout",
    )
    _configure_tasks(
        tasks,
        verifier_timeout,
        benchmark["profile"],
        repo_generation_proxy_url=str(_config_value(config, "eval.repo_generation_proxy_url", "") or ""),
    )
    network_verifier_tasks = [
        task.id
        for task in tasks
        if isinstance(task.metadata, dict)
        and isinstance(task.metadata.get("rllm"), dict)
        and task.metadata["rllm"].get("verifier_network_policy")
    ]
    upgraded_verifiers = _upgrade_task_verifiers(tasks, benchmark["profile"])
    _validate_tasks(tasks, benchmark["profile"])
    if sandbox_backend == "minisandbox":
        _validate_minisandbox_evaluation_cache(
            config,
            tasks,
            benchmark["profile"],
        )

    models = discover_evaluation_models(
        experiment_root,
        base_model_path=_config_value(config, "eval.base_model_path", None),
        evaluate_base=evaluate_base,
        checkpoint_stride=checkpoint_stride,
        checkpoint_interval=checkpoint_interval,
    )
    semantic_config = _semantic_config(config, tasks)
    store_parent = output_root / dataset_name / "codeflow-native"
    semantic_id = semantic_config_id(semantic_config)
    store = EvaluationStore(
        store_parent,
        semantic_id=semantic_id,
        semantic_config=semantic_config,
        runs_root=output_root / "runs",
    )
    run_id = str(_config_value(config, "eval.run_id", time.strftime("%Y%m%d-%H%M%S")))

    with store.lock():
        _save_launcher_backup(Path(__file__).with_suffix(".sh"), store.root / "eval_backup.sh")
        console_log_path = _config_value(config, "eval.console_log_path", None)
        log_path = _setup_run_logging(store, run_id, console_log_path)
        store.initialize_manifest(run_id=run_id, requested_budget=pass_n)
        run_dir = store.runs_dir / run_id
        resolved = OmegaConf.to_container(config, resolve=True, enum_to_str=True)
        (run_dir / "resolved_config.json").write_text(
            json.dumps(resolved, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        logger.info("Evaluation output: %s", store.root)
        logger.info("Evaluation log: %s", log_path)
        logger.info("Selected models: %s", [model.key for model in models])
        logger.info(
            "%s verifier profile: upgraded_tasks=%d",
            benchmark["profile"],
            upgraded_verifiers,
        )
        task_timeouts = sorted({float(task.metadata["rllm"]["resolved_verifier_timeout"]) for task in tasks})
        logger.info(
            "%s verifier timeout: default=%.1fs, resolved=%s",
            benchmark["profile"],
            verifier_timeout,
            task_timeouts,
        )
        if timeout_resource_overrides:
            logger.info(
                "Pinned %d action/verifier-timeout task(s) to %d CPUs/%d MiB: %s",
                len(timeout_resource_overrides),
                _VERIFIER_TIMEOUT_RESOURCE_CPUS,
                _VERIFIER_TIMEOUT_RESOURCE_MEMORY_MB,
                timeout_resource_overrides,
            )
        if network_verifier_tasks:
            logger.info(
                "Enabled bounded public-network verifier policy v%d for %d task(s): %s",
                _VERIFIER_NETWORK_POLICY_VERSION,
                len(network_verifier_tasks),
                network_verifier_tasks,
            )
        logger.info(
            "Sandbox resource histogram (CPU, memory_mb, storage_mb): %s",
            _sandbox_resource_histogram(tasks),
        )
        logger.info("Evaluation sequence budget: %s", sequence_budget)
        source_images, resolved_images, rewrites = _sandbox_image_summary(
            tasks,
            str(config.swe.sandbox_backend),
        )
        logger.info(
            "Sandbox images: %d source image(s), %d resolved image(s), examples=%s",
            source_images,
            resolved_images,
            rewrites,
        )
        for model in models:
            store.ensure_model(model)
            # Scan before allocating GPUs so corrupt/mixed results fail fast.
            store.scan_attempts(model, tasks)
            promoted = store.promote_exhausted_failures(model, tasks, pass_n)
            if promoted:
                logger.info(
                    "Promoted %d exhausted infrastructure failure(s) for %s "
                    "to terminal failed attempts; future resumes will skip them",
                    len(promoted),
                    model.key,
                )
        store.update_manifest_models(models, tasks, pass_n)
        write_summary(store, models, tasks, pass_n)
        _run_models(
            config,
            models,
            tasks,
            store,
            pass_n,
            log_path,
            console_captured=console_log_path is not None,
        )
        store.update_manifest_models(models, tasks, pass_n)
        summary = write_summary(store, models, tasks, pass_n)
        complete_models = sum(result.get("metrics_status") == "complete" for result in summary["models"])
        partial_models = sum(result.get("metrics_status") == "partial" for result in summary["models"])
        capability_models = sum(bool(result.get("capability_complete")) for result in summary["models"])
        effective_attempts = sum(int(result.get("effective_valid_attempts", 0)) for result in summary["models"])
        expected_attempts = sum(int(result.get("expected_attempts", 0)) for result in summary["models"])
        logger.info(
            "Evaluation scheduling complete: %d complete and %d partial model(s) out of %d selected for pass@%d; "
            "capability_complete_models=%d effective_valid_attempts=%d/%d",
            complete_models,
            partial_models,
            len(models),
            pass_n,
            capability_models,
            effective_attempts,
            expected_attempts,
        )


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="unified",
    version_base=None,
)
def main(config: DictConfig) -> None:
    _run_evaluation(config)


if __name__ == "__main__":
    main()
