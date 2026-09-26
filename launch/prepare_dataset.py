#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from rllm.data import DatasetRegistry  # noqa: E402
from rllm.data.denovoswe_minisandbox_cache import (  # noqa: E402
    DEFAULT_IMAGE_WORKERS,
    materialize_denovoswe_oci_cache,
)
from rllm.data.materialization import (  # noqa: E402
    DEFAULT_MATERIALIZATION_WORKERS,
    MAX_MATERIALIZATION_WORKERS,
)
from rllm.data.minisandbox_cache import (  # noqa: E402
    materialize_minisandbox_oci_cache,
)
from rllm.data.r2egym_builder import build_benchmark  # noqa: E402
from rllm.data.repo_generation_builder import (  # noqa: E402
    build_denovoswe,
    build_doc2repo,
    build_nl2repo,
)
from rllm.data.swebench_pro_builder import build_benchmark as build_swebench_pro  # noqa: E402
from rllm.data.swebench_pro_builder import materialization_complete as swebench_pro_complete  # noqa: E402
from rllm.data.swerebench_v2_builder import build_benchmark as build_swerebench_v2  # noqa: E402
from rllm.data.swerebench_v2_builder import materialization_complete as swerebench_v2_complete  # noqa: E402

MILESTONE_REQUIRED_FIELDS = {
    "expected_output_json",
    "baseline_output_json",
    "target_output_json",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "modified_files",
    "relevant_files",
    "modified_entity_summaries",
    "test_file_names",
}


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"ERROR: {name} is not set. Please configure it in prepare_dataset.sh.")
    return value


def env(name: str, default: str = "") -> str:
    return os.environ.get(name) or default


def enabled(name: str, default: str = "1") -> bool:
    return env(name, default).lower() in {"1", "true", "yes", "y"}


def positive_int_from_env(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {name} must be a positive integer when set; got '{value}'.") from exc
    if parsed <= 0:
        raise SystemExit(f"ERROR: {name} must be greater than 0 when set.")
    return parsed


def materialization_workers_from_env() -> int:
    workers = positive_int_from_env("RLLM_MATERIALIZATION_WORKERS")
    if workers is None:
        workers = DEFAULT_MATERIALIZATION_WORKERS
    if workers > MAX_MATERIALIZATION_WORKERS:
        raise SystemExit(
            "ERROR: RLLM_MATERIALIZATION_WORKERS must be an integer in "
            f"[1, {MAX_MATERIALIZATION_WORKERS}]."
        )
    return workers


def csv_values_from_env(name: str) -> tuple[str, ...]:
    """Parse a stable, de-duplicated comma-separated environment setting."""

    return tuple(
        sorted(
            {
                value.strip()
                for value in os.environ.get(name, "").split(",")
                if value.strip()
            }
        )
    )


def configure_paths() -> tuple[Path, Path, Path]:
    exp_root = Path(required_env("EXP_ROOT"))
    data_root = Path(required_env("DATA_ROOT"))
    tasks_root = Path(required_env("TASKS_ROOT"))

    Path(required_env("RLLM_HOME"), "datasets").mkdir(parents=True, exist_ok=True)
    Path(required_env("HF_HOME")).mkdir(parents=True, exist_ok=True)
    Path(required_env("HF_DATASETS_CACHE")).mkdir(parents=True, exist_ok=True)
    tasks_root.mkdir(parents=True, exist_ok=True)
    return exp_root, data_root, tasks_root


def clean_data_root_extras(data_root: Path) -> None:
    if not enabled("CLEAN_DATA_ROOT_EXTRAS"):
        return

    data_root = data_root.resolve()
    allowed = {
        Path(os.environ["TASKS_ROOT"]).resolve(),
        Path(os.environ["RLLM_HOME"]).resolve(),
        Path(os.environ["HF_HOME"]).resolve(),
        # Every default MiniSandbox cache view lives below this shared root.
        # Keep it even when the DeNovo cache is explicitly overridden
        # outside DATA_ROOT.
        (data_root / "minisandbox").resolve(),
    }
    cache_dir_vars = (
        "DENOVOSWE_MINISANDBOX_CACHE_DIR",
        "SWEREBENCH_V2_PYTHON_MINISANDBOX_CACHE_DIR",
        "SWEREBENCH_V2_PROMATCHED_MINISANDBOX_CACHE_DIR",
        "R2EGYM_MINISANDBOX_CACHE_DIR",
        "NL2REPO_MINISANDBOX_CACHE_DIR",
        "DOC2REPO_MINISANDBOX_CACHE_DIR",
        "SWEBENCH_VERIFIED_MINISANDBOX_CACHE_DIR",
    )
    for variable in cache_dir_vars:
        configured = os.environ.get(variable)
        if not configured:
            continue
        cache_path = Path(configured).expanduser().resolve()
        try:
            cache_relative = cache_path.relative_to(data_root)
        except ValueError:
            continue
        if cache_relative.parts:
            allowed.add((data_root / cache_relative.parts[0]).resolve())
    if not data_root.exists():
        return

    for child in data_root.iterdir():
        if child.resolve() in allowed:
            continue
        if child.is_dir():
            print(f"Removing extra DATA_ROOT directory: {child}")
            shutil.rmtree(child)


_COMMON_TASK_FILES = (
    "task.toml",
    "instruction.md",
    "environment/Dockerfile",
    "tests/test.sh",
    "tests/instance.json",
    "solution/gold.patch",
    "solution/solve.sh",
)


def registered_dataset_complete(
    *, dataset_name: str, split: str, out_dir: str | Path, materialize_milestone_metadata: bool
) -> bool:
    """Recognize complete registered R2E/SWE-bench task inventories.

    The registry parquet is the authoritative task inventory.  Checking a
    bounded sample of task trees avoids thousands of high-latency shared-filesystem stats
    while still rejecting an obviously stale or mis-rooted registration.
    """

    out = Path(out_dir).expanduser().resolve()
    if not (out / "dataset.toml").is_file():
        return False
    info = DatasetRegistry.get_dataset_info(dataset_name)
    if not info or split not in info.get("splits", {}):
        return False
    split_info = info["splits"][split]
    dataset_path = Path(DatasetRegistry._resolve_path(split_info["path"]))
    if not dataset_path.is_file():
        return False
    if not Path(DatasetRegistry._verl_path_for(str(dataset_path))).is_file():
        return False
    fields = set(split_info.get("fields") or [])
    if not {"id", "task_path"}.issubset(fields):
        return False
    if materialize_milestone_metadata and MILESTONE_REQUIRED_FIELDS.difference(fields):
        return False
    try:
        df = pd.read_parquet(dataset_path, columns=["id", "task_path"])
    except Exception:
        return False
    if df.empty:
        return False
    if split_info.get("num_examples") not in {None, len(df)}:
        return False
    if df["id"].nunique(dropna=False) != len(df) or df["task_path"].nunique(dropna=False) != len(df):
        return False
    task_paths = [
        Path(os.path.abspath(os.path.expanduser(value)))
        for value in df["task_path"].astype(str)
    ]
    if any(path.parent != out for path in task_paths):
        return False
    try:
        task_directory_count = sum(
            entry.is_dir(follow_symlinks=False)
            for entry in os.scandir(out)
            if not entry.name.startswith(".")
        )
    except OSError:
        return False
    if task_directory_count != len(task_paths):
        return False

    # These datasets use registry inventory as their completion marker. The registry,
    # parquet row count, one-to-one task paths, and directory count establish
    # the complete inventory.  Probe a deterministic spread of task contents
    # without turning every resume startup into tens of thousands of shared-filesystem
    # metadata requests.
    sample_indexes = sorted({0, len(task_paths) // 4, len(task_paths) // 2, 3 * len(task_paths) // 4, len(task_paths) - 1})
    for index in sample_indexes:
        task_path = task_paths[index]
        if any(not (task_path / relative).is_file() for relative in _COMMON_TASK_FILES):
            return False
    return True


def build_one(
    *,
    enabled_var: str,
    name_var: str,
    split_var: str,
    repo_var: str,
    hf_split_var: str,
    out_dir_var: str,
    agent_var: str,
    limit_var: str,
    max_workers: int | None = None,
    milestone_metadata_var: str | None = None,
) -> tuple[str, str, bool] | None:
    if not enabled(enabled_var):
        print(f"Skipping {name_var} because {enabled_var}=0")
        return None

    dataset_name = required_env(name_var)
    split = required_env(split_var)
    repo = required_env(repo_var)
    hf_split = required_env(hf_split_var)
    out_dir = required_env(out_dir_var)
    agent = required_env(agent_var)
    limit = positive_int_from_env(limit_var)
    materialize_milestone_metadata = enabled(milestone_metadata_var, "0") if milestone_metadata_var else False

    if not enabled("CLEAN") and registered_dataset_complete(
        dataset_name=dataset_name,
        split=split,
        out_dir=out_dir,
        materialize_milestone_metadata=materialize_milestone_metadata,
    ):
        print(f"Skipping {dataset_name}/{split}: existing materialization is complete")
        return dataset_name, split, materialize_milestone_metadata

    print()
    print(f"Preparing {dataset_name}/{split}")
    print(f"  HF source: {repo}:{hf_split}")
    print(f"  task dir:  {out_dir}")
    print(f"  limit:     {limit if limit is not None else 'all'}")
    print(f"  milestone metadata: {'enabled' if materialize_milestone_metadata else 'disabled'}")

    build_benchmark(
        name=dataset_name,
        split=split,
        out_dir=out_dir,
        hf_repo_id=repo,
        hf_split=hf_split,
        limit=limit,
        default_agent=agent,
        clean=enabled("CLEAN"),
        register=True,
        materialize_milestone_metadata=materialize_milestone_metadata,
        max_workers=max_workers,
    )
    return dataset_name, split, materialize_milestone_metadata


def build_swerebench_dataset(
    max_workers: int | None = None,
) -> tuple[str, str, bool] | None:
    prefix = "SWEREBENCH_V2"
    enabled_var = f"BUILD_{prefix}"
    if not enabled(enabled_var):
        print(f"Skipping {prefix} because {enabled_var}=0")
        return None

    dataset_name = required_env(f"{prefix}_NAME")
    split = required_env(f"{prefix}_SPLIT")
    repo = required_env(f"{prefix}_HF_REPO")
    revision = required_env(f"{prefix}_HF_REVISION")
    hf_split = required_env(f"{prefix}_HF_SPLIT")
    out_dir = required_env(f"{prefix}_OUT_DIR")
    agent = required_env(f"{prefix}_DEFAULT_AGENT")
    limit = positive_int_from_env(f"{prefix}_LIMIT")
    shadow_resource_override_task_ids = csv_values_from_env(
        f"{prefix}_SHADOW_RESOURCE_OVERRIDE_TASK_IDS"
    )

    complete_kwargs = {
        "name": dataset_name,
        "split": split,
        "out_dir": out_dir,
        "hf_repo_id": repo,
        "hf_revision": revision,
        "hf_split": hf_split,
        "limit": limit,
        "language": None,
        "shadow_resource_override_task_ids": shadow_resource_override_task_ids,
    }
    if not enabled("CLEAN") and swerebench_v2_complete(**complete_kwargs):
        print(f"Skipping {dataset_name}/{split}: existing materialization is complete")
        return dataset_name, split, True

    print()
    print(f"Preparing {dataset_name}/{split}")
    print(f"  HF source: {repo}:{hf_split}@{revision}")
    print("  languages:  all")
    print(f"  task dir:  {out_dir}")
    print(f"  limit:     {limit if limit is not None else 'all'}")
    print("  milestone metadata: enabled")
    print(
        "  shadow 4 CPU/16 GB overrides: "
        f"{len(shadow_resource_override_task_ids)} task(s)"
    )

    build_swerebench_v2(
        name=dataset_name,
        split=split,
        out_dir=out_dir,
        hf_repo_id=repo,
        hf_revision=revision,
        hf_split=hf_split,
        limit=limit,
        language=None,
        default_agent=agent,
        clean=enabled("CLEAN"),
        register=True,
        shadow_resource_override_task_ids=shadow_resource_override_task_ids,
        max_workers=max_workers,
    )
    return dataset_name, split, True


def build_swebench_pro_public_dataset(
    max_workers: int | None = None,
) -> tuple[str, str, bool] | None:
    prefix = "SWEBENCH_PRO_PUBLIC"
    if not enabled(f"BUILD_{prefix}"):
        print(f"Skipping {prefix} because BUILD_{prefix}=0")
        return None

    name = required_env(f"{prefix}_NAME")
    split = required_env(f"{prefix}_SPLIT")
    hf_repo = required_env(f"{prefix}_HF_REPO")
    hf_revision = required_env(f"{prefix}_HF_REVISION")
    hf_split = required_env(f"{prefix}_HF_SPLIT")
    scripts_repo = required_env(f"{prefix}_SCRIPTS_REPO")
    scripts_revision = required_env(f"{prefix}_SCRIPTS_REVISION")
    out_dir = required_env(f"{prefix}_OUT_DIR")
    agent = required_env(f"{prefix}_DEFAULT_AGENT")
    limit = positive_int_from_env(f"{prefix}_LIMIT")
    complete_kwargs = {
        "name": name,
        "split": split,
        "out_dir": out_dir,
        "hf_repo_id": hf_repo,
        "hf_revision": hf_revision,
        "hf_split": hf_split,
        "scripts_repo_url": scripts_repo,
        "scripts_revision": scripts_revision,
        "limit": limit,
        "default_agent": agent,
    }
    if not enabled("CLEAN"):
        print(
            f"Checking {name}/{split} existing materialization...",
            flush=True,
        )
        if swebench_pro_complete(**complete_kwargs):
            print(f"Skipping {name}/{split}: existing materialization is complete")
            return name, split, False

    print()
    print(f"Preparing {name}/{split}")
    print(f"  HF source:      {hf_repo}:{hf_split}@{hf_revision}")
    print(f"  scripts source: {scripts_repo}@{scripts_revision}")
    print(f"  task dir:       {out_dir}")
    print(f"  limit:          {limit if limit is not None else 'all'}")
    build_swebench_pro(
        **complete_kwargs,
        clean=enabled("CLEAN"),
        register=True,
        max_workers=max_workers,
    )
    return name, split, False


def build_repo_generation_dataset(
    prefix: str,
    builder,
    *,
    has_filename: bool,
    max_workers: int | None = None,
) -> tuple[str, str, bool] | None:
    enabled_var = f"BUILD_{prefix}"
    if not enabled(enabled_var):
        print(f"Skipping {prefix} because {enabled_var}=0")
        return None

    name = required_env(f"{prefix}_NAME")
    split = required_env(f"{prefix}_SPLIT")
    repo = required_env(f"{prefix}_HF_REPO")
    revision = required_env(f"{prefix}_HF_REVISION")
    out_dir = required_env(f"{prefix}_OUT_DIR")
    agent = required_env(f"{prefix}_DEFAULT_AGENT")
    limit = positive_int_from_env(f"{prefix}_LIMIT")
    kwargs = {
        "name": name,
        "split": split,
        "out_dir": out_dir,
        "hf_repo_id": repo,
        "hf_revision": revision,
        "limit": limit,
        "default_agent": agent,
        "clean": enabled("CLEAN"),
        "register": True,
        "max_workers": max_workers,
    }
    if has_filename:
        kwargs["filename"] = required_env(f"{prefix}_HF_FILENAME")

    print()
    print(f"Preparing {name}/{split}")
    print(f"  HF source: {repo}@{revision}")
    print(f"  task dir:  {out_dir}")
    print(f"  limit:     {limit if limit is not None else 'all'}")
    builder(**kwargs)
    # Repo-generation metadata has its own contract and intentionally does
    # not claim the bug-repair MILESTONE_REQUIRED_FIELDS set.
    return name, split, False


def build_denovoswe_minisandbox_cache() -> Path | None:
    """Materialize the optional daemonless OCI cache after task generation."""

    if not enabled("BUILD_DENOVOSWE_MINISANDBOX_CACHE", "0"):
        print("Skipping DeNovoSWE MiniSandbox OCI cache")
        return None
    workers = positive_int_from_env("DENOVOSWE_MINISANDBOX_IMAGE_WORKERS")
    if workers is None:
        workers = DEFAULT_IMAGE_WORKERS
    tasks_root = Path(required_env("DENOVOSWE_OUT_DIR")).expanduser().resolve()
    cache_dir = Path(
        required_env("DENOVOSWE_MINISANDBOX_CACHE_DIR")
    ).expanduser().resolve()
    image_map = env("DENOVOSWE_MINISANDBOX_IMAGE_MAP_FILE") or None
    auth_file = env("DENOVOSWE_MINISANDBOX_REGISTRY_AUTH_FILE") or None
    print()
    print("Preparing DeNovoSWE MiniSandbox OCI cache")
    print(f"  task dir:      {tasks_root}")
    print(f"  cache dir:     {cache_dir}")
    print(f"  image workers: {workers}")
    print(f"  image map:     {image_map or 'none (original registry fallback)'}")
    manifest = materialize_denovoswe_oci_cache(
        tasks_root=tasks_root,
        cache_dir=cache_dir,
        image_map_file=image_map,
        registry_auth_file=auth_file,
        max_workers=workers,
    )
    print(f"  cache manifest: {manifest}")
    return manifest


_MINISANDBOX_CACHE_BUILDS = (
    (
        "SWEREBENCH_V2_PROMATCHED",
        "swe-rebench-v2-filtered-verified-promatched",
    ),
    (
        "SWEREBENCH_V2_PYTHON",
        "swe-rebench-v2-filtered-verified-python",
    ),
    ("R2EGYM", "r2egym"),
    ("NL2REPO", "nl2repo"),
    ("DOC2REPO", "doc2repo"),
    ("SWEBENCH_VERIFIED", "swebench_verified"),
    ("SWEBENCH_PRO_PUBLIC", "swebench_pro_public"),
)


def build_minisandbox_caches() -> list[Path]:
    """Build independently enabled schema-v2 OCI caches.

    Dataset selection comes from the canonical registry view rather than a
    external image-availability audit. Public images are used unless the user
    supplies an explicit image mapping and registry authentication file.
    """

    workers = positive_int_from_env("MINISANDBOX_IMAGE_WORKERS")
    if workers is None:
        workers = DEFAULT_IMAGE_WORKERS
    image_map = env("MINISANDBOX_IMAGE_MAP_FILE") or None
    auth_file = env("MINISANDBOX_REGISTRY_AUTH_FILE") or None
    manifests: list[Path] = []
    for prefix, dataset_name in _MINISANDBOX_CACHE_BUILDS:
        enabled_var = f"BUILD_{prefix}_MINISANDBOX_CACHE"
        if not enabled(enabled_var, "0"):
            print(f"Skipping {dataset_name} MiniSandbox OCI cache")
            continue
        cache_dir = Path(
            required_env(f"{prefix}_MINISANDBOX_CACHE_DIR")
        ).expanduser().resolve()
        print()
        print(f"Preparing {dataset_name} MiniSandbox OCI cache")
        print(f"  cache dir:     {cache_dir}")
        print(f"  image workers: {workers}")
        print(
            "  image map:     "
            f"{image_map or 'public image references'}"
        )
        manifest = materialize_minisandbox_oci_cache(
            dataset=dataset_name,
            cache_dir=cache_dir,
            image_map_file=image_map,
            registry_auth_file=auth_file,
            max_workers=workers,
            inventory_workers=min(workers, 64),
            strict_inventory=True,
        )
        print(f"  cache manifest: {manifest}")
        manifests.append(manifest)
    return manifests


def validate_registry(built: list[tuple[str, str, bool]]) -> None:
    print()
    print("Registry validation")
    for dataset_name, split, materialize_milestone_metadata in built:
        info = DatasetRegistry.get_dataset_info(dataset_name)
        if not info or split not in info.get("splits", {}):
            raise RuntimeError(f"{dataset_name}/{split} was not registered")

        dataset_path = Path(DatasetRegistry._resolve_path(info["splits"][split]["path"]))
        columns = ["id", "task_path"]
        if materialize_milestone_metadata:
            columns.extend(sorted(MILESTONE_REQUIRED_FIELDS))
        try:
            df = pd.read_parquet(dataset_path, columns=columns)
        except Exception as exc:
            raise RuntimeError(
                f"{dataset_name}/{split} registry parquet is missing required fields"
            ) from exc
        id_unique = df["id"].nunique(dropna=False)
        task_path_unique = df["task_path"].nunique(dropna=False)
        print(
            f"  {dataset_name}/{split}: "
            f"rows={len(df)} id_unique={id_unique} task_path_unique={task_path_unique}"
        )
        if id_unique != len(df) or task_path_unique != len(df):
            raise RuntimeError(f"{dataset_name}/{split} has duplicate id/task_path values")
        if "unknown" in set(df["id"].astype(str)):
            raise RuntimeError(f"{dataset_name}/{split} contains id='unknown'")
        if materialize_milestone_metadata:
            missing_fields = MILESTONE_REQUIRED_FIELDS.difference(df.columns)
            if missing_fields:
                raise RuntimeError(f"{dataset_name}/{split} is missing milestone fields: {sorted(missing_fields)}")
            for field in ("expected_output_json", "baseline_output_json", "target_output_json"):
                parsed = df[field].map(json.loads)
                if not parsed.map(lambda value: isinstance(value, dict) and bool(value)).all():
                    raise RuntimeError(f"{dataset_name}/{split} contains invalid or empty {field}")
            for field in ("FAIL_TO_PASS", "PASS_TO_PASS"):
                parsed = df[field].map(json.loads)
                if not parsed.map(lambda value: isinstance(value, list)).all():
                    raise RuntimeError(f"{dataset_name}/{split} contains invalid {field}")
            for field in ("modified_files", "relevant_files", "modified_entity_summaries", "test_file_names"):
                if not df[field].map(lambda value: value is not None and len(value) > 0).all():
                    raise RuntimeError(f"{dataset_name}/{split} contains empty {field}")


def main() -> None:
    exp_root, data_root, tasks_root = configure_paths()
    max_workers = materialization_workers_from_env()

    print(f"EXP_ROOT: {exp_root}")
    print(f"DATA_ROOT: {data_root}")
    print(f"TASKS_ROOT: {tasks_root}")
    print(f"RLLM_HOME: {os.environ['RLLM_HOME']}")
    print(f"HF_HOME: {os.environ['HF_HOME']}")
    print(f"CLEAN: {required_env('CLEAN')}")
    print(f"CLEAN_DATA_ROOT_EXTRAS: {required_env('CLEAN_DATA_ROOT_EXTRAS')}")
    print(
        "Materialization workers (all datasets): "
        f"{max_workers}"
    )

    built: list[tuple[str, str, bool]] = []
    specs = (
        {
            "enabled_var": "BUILD_R2EGYM",
            "name_var": "R2EGYM_NAME",
            "split_var": "R2EGYM_SPLIT",
            "repo_var": "R2EGYM_HF_REPO",
            "hf_split_var": "R2EGYM_HF_SPLIT",
            "out_dir_var": "R2EGYM_OUT_DIR",
            "agent_var": "R2EGYM_DEFAULT_AGENT",
            "limit_var": "R2EGYM_LIMIT",
            "milestone_metadata_var": "R2EGYM_MATERIALIZE_MILESTONE_METADATA",
        },
        {
            "enabled_var": "BUILD_VERIFIED",
            "name_var": "VERIFIED_NAME",
            "split_var": "VERIFIED_SPLIT",
            "repo_var": "VERIFIED_HF_REPO",
            "hf_split_var": "VERIFIED_HF_SPLIT",
            "out_dir_var": "VERIFIED_OUT_DIR",
            "agent_var": "VERIFIED_DEFAULT_AGENT",
            "limit_var": "VERIFIED_LIMIT",
        },
    )

    for spec in specs:
        result = build_one(**spec, max_workers=max_workers)
        if result is not None:
            built.append(result)

    swebench_pro_result = build_swebench_pro_public_dataset(max_workers)
    if swebench_pro_result is not None:
        built.append(swebench_pro_result)

    swerebench_result = build_swerebench_dataset(max_workers)
    if swerebench_result is not None:
        built.append(swerebench_result)

    repo_generation_specs = (
        ("DENOVOSWE", build_denovoswe, True),
        ("NL2REPO", build_nl2repo, True),
        ("DOC2REPO", build_doc2repo, False),
    )
    for prefix, builder, has_filename in repo_generation_specs:
        result = build_repo_generation_dataset(
            prefix,
            builder,
            has_filename=has_filename,
            max_workers=max_workers,
        )
        if result is not None:
            built.append(result)

    validate_registry(built)
    build_denovoswe_minisandbox_cache()
    build_minisandbox_caches()

    print()
    print("Done.")

    clean_data_root_extras(data_root)

    print()
    print("DATA_ROOT direct directories:")
    for child in sorted(data_root.iterdir()):
        if child.is_dir():
            print(f"  {child.name}")

    print()
    print("You can now run:")
    print("  bash launch/eval.sh")


if __name__ == "__main__":
    main()
