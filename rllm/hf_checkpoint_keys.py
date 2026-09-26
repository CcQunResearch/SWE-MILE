"""Repair canonical Hugging Face weight names without rewriting tensor payloads.

VERL's Qwen3.5 actor export can place the visual tower below the language
model namespace even though the Hugging Face configuration describes the
full conditional-generation model.  The tensor payloads are valid; only the
names embedded in the safetensors headers (and shard index) are wrong.

This module intentionally depends only on the Python standard library so it
can repair large checkpoints in place on a host without importing torch or
safetensors.  Safetensors headers have a fixed allocated length, and the
canonical prefix is shorter than the bad prefix, so the payload offsets and
the multi-gigabyte tensor data remain untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_BAD_QWEN35_VISUAL_PREFIX = "model.language_model.visual."
_CANONICAL_QWEN35_VISUAL_PREFIX = "model.visual."
_SAFETENSORS_INDEX = "model.safetensors.index.json"
_SINGLE_SAFETENSORS = "model.safetensors"
_HEADER_BACKUP_SUFFIX = ".rllm-qwen35-header.bak"


@dataclass(frozen=True)
class Qwen35CheckpointRepair:
    path: str
    applicable: bool
    changed: bool
    shard_count: int
    header_keys_renamed: int
    index_keys_renamed: int
    canonical_visual_keys: int


def canonicalize_qwen35_weight_name(name: str) -> str:
    """Return the canonical HF name for a Qwen3.5 actor-exported weight."""

    if name.startswith(_BAD_QWEN35_VISUAL_PREFIX):
        return _CANONICAL_QWEN35_VISUAL_PREFIX + name[len(_BAD_QWEN35_VISUAL_PREFIX) :]
    return name


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _is_full_qwen35_checkpoint(path: Path) -> bool:
    config_path = path / "config.json"
    if not config_path.is_file():
        return False
    config = _load_json(config_path)
    architectures = config.get("architectures")
    expected = {
        "qwen3_5": "Qwen3_5ForConditionalGeneration",
        "qwen3_5_moe": "Qwen3_5MoeForConditionalGeneration",
    }.get(config.get("model_type"))
    return expected is not None and isinstance(architectures, list) and expected in architectures


def _read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    size = path.stat().st_size
    with path.open("rb") as file:
        raw_length = file.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"truncated safetensors length header: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length <= 0 or header_length > size - 8:
            raise ValueError(
                f"invalid safetensors header length {header_length} for {path} ({size} bytes)"
            )
        raw_header = file.read(header_length)
        if len(raw_header) != header_length:
            raise ValueError(f"truncated safetensors JSON header: {path}")
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid safetensors JSON header in {path}: {error}") from error
    if not isinstance(header, dict):
        raise ValueError(f"expected an object safetensors header in {path}")
    return header_length, header


def _rename_mapping_keys(
    mapping: dict[str, Any],
    *,
    source: Path,
) -> tuple[dict[str, Any], int]:
    renamed = 0
    result: dict[str, Any] = {}
    for name, value in mapping.items():
        canonical = canonicalize_qwen35_weight_name(name)
        if canonical != name:
            renamed += 1
        if canonical in result:
            raise ValueError(
                f"Qwen3.5 weight-name collision in {source}: {name!r} maps to existing {canonical!r}"
            )
        result[canonical] = value
    return result, renamed


def _restore_pending_header_backup(path: Path) -> None:
    backup_path = path.with_name(path.name + _HEADER_BACKUP_SUFFIX)
    if not backup_path.exists():
        return
    original = backup_path.read_bytes()
    if len(original) < 9:
        raise ValueError(f"invalid pending safetensors header backup: {backup_path}")
    header_length = struct.unpack("<Q", original[:8])[0]
    if len(original) != 8 + header_length:
        raise ValueError(f"truncated pending safetensors header backup: {backup_path}")
    with path.open("r+b", buffering=0) as file:
        written = os.pwrite(file.fileno(), original, 0)
        if written != len(original):
            raise OSError(f"short restore write for {path}: {written}/{len(original)}")
        os.fsync(file.fileno())
    backup_path.unlink()


def _rewrite_safetensors_header(path: Path) -> int:
    _restore_pending_header_backup(path)
    header_length, header = _read_safetensors_header(path)
    canonical, renamed = _rename_mapping_keys(header, source=path)
    if renamed == 0:
        return 0

    raw_header = json.dumps(canonical, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(raw_header) > header_length:
        raise ValueError(
            f"canonical safetensors header no longer fits in {path}: "
            f"{len(raw_header)} > {header_length}"
        )
    padded_header = raw_header + b" " * (header_length - len(raw_header))

    backup_path = path.with_name(path.name + _HEADER_BACKUP_SUFFIX)
    with path.open("rb") as file:
        original = file.read(8 + header_length)
    with backup_path.open("xb") as backup:
        backup.write(original)
        backup.flush()
        os.fsync(backup.fileno())

    try:
        with path.open("r+b", buffering=0) as file:
            written = os.pwrite(file.fileno(), padded_header, 8)
            if written != len(padded_header):
                raise OSError(f"short safetensors header write for {path}: {written}/{len(padded_header)}")
            os.fsync(file.fileno())
        _, verified = _read_safetensors_header(path)
        if any(name.startswith(_BAD_QWEN35_VISUAL_PREFIX) for name in verified):
            raise ValueError(f"bad Qwen3.5 visual prefix remains in {path}")
        for name in canonical:
            if name not in verified:
                raise ValueError(f"canonical weight {name!r} is missing after rewriting {path}")
    except BaseException:
        _restore_pending_header_backup(path)
        raise
    backup_path.unlink()
    return renamed


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def repair_qwen35_hf_checkpoint(path: str | os.PathLike[str]) -> Qwen35CheckpointRepair:
    """Repair a full Qwen3.5 HF checkpoint in place and verify the result."""

    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"HF checkpoint directory does not exist: {checkpoint}")
    if not _is_full_qwen35_checkpoint(checkpoint):
        return Qwen35CheckpointRepair(str(checkpoint), False, False, 0, 0, 0, 0)

    index_path = checkpoint / _SAFETENSORS_INDEX
    single_path = checkpoint / _SINGLE_SAFETENSORS
    index: dict[str, Any] | None = None
    original_weight_map: dict[str, str] | None = None
    if index_path.is_file():
        index = _load_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"missing or invalid weight_map in {index_path}")
        if not all(isinstance(name, str) and isinstance(shard, str) for name, shard in weight_map.items()):
            raise ValueError(f"invalid weight_map entry in {index_path}")
        original_weight_map = weight_map
        shard_paths = [checkpoint / name for name in sorted(set(weight_map.values()))]
    elif single_path.is_file():
        shard_paths = [single_path]
    else:
        raise FileNotFoundError(f"safetensors weights are missing from {checkpoint}")

    for shard_path in shard_paths:
        if not shard_path.is_file() or shard_path.stat().st_size <= 8:
            raise FileNotFoundError(f"missing or empty safetensors shard: {shard_path}")

    header_renamed = sum(_rewrite_safetensors_header(shard_path) for shard_path in shard_paths)

    index_renamed = 0
    if index is not None and original_weight_map is not None:
        canonical_weight_map, index_renamed = _rename_mapping_keys(
            original_weight_map,
            source=index_path,
        )
        header_cache = {path.name: _read_safetensors_header(path)[1] for path in shard_paths}
        for old_name, shard_name in original_weight_map.items():
            expected_name = canonicalize_qwen35_weight_name(old_name)
            if expected_name not in header_cache[shard_name]:
                raise ValueError(
                    f"index/header mismatch after Qwen3.5 repair: {expected_name!r} "
                    f"is absent from {shard_name}"
                )
        if index_renamed:
            index["weight_map"] = canonical_weight_map
            _atomic_write_json(index_path, index)

    entries: Iterable[str]
    if index_path.is_file():
        repaired_index = _load_json(index_path)
        repaired_weight_map = repaired_index.get("weight_map")
        if not isinstance(repaired_weight_map, dict):
            raise ValueError(f"invalid repaired weight_map in {index_path}")
        entries = repaired_weight_map
    else:
        entries = _read_safetensors_header(single_path)[1]
    names = [name for name in entries if name != "__metadata__"]
    if any(name.startswith(_BAD_QWEN35_VISUAL_PREFIX) for name in names):
        raise ValueError(f"bad Qwen3.5 visual weight names remain in {checkpoint}")
    canonical_count = sum(name.startswith(_CANONICAL_QWEN35_VISUAL_PREFIX) for name in names)
    if canonical_count == 0:
        raise ValueError(f"full Qwen3.5 checkpoint has no canonical visual weights: {checkpoint}")

    return Qwen35CheckpointRepair(
        path=str(checkpoint),
        applicable=True,
        changed=bool(header_renamed or index_renamed),
        shard_count=len(shard_paths),
        header_keys_renamed=header_renamed,
        index_keys_renamed=index_renamed,
        canonical_visual_keys=canonical_count,
    )


def _discover_checkpoints(paths: Iterable[str]) -> list[Path]:
    checkpoints: dict[str, Path] = {}
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if (path / "config.json").is_file():
            checkpoints[str(path)] = path
            continue
        if not path.is_dir():
            raise FileNotFoundError(f"checkpoint path does not exist: {path}")
        for candidate in path.iterdir():
            if candidate.is_dir() and candidate.name.startswith("global_step_"):
                checkpoints[str(candidate.resolve())] = candidate.resolve()
    return [checkpoints[key] for key in sorted(checkpoints)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair Qwen3.5 HF visual weight names in place")
    parser.add_argument("paths", nargs="+", help="HF checkpoint or directory containing global_step_* checkpoints")
    args = parser.parse_args()

    checkpoints = _discover_checkpoints(args.paths)
    if not checkpoints:
        raise SystemExit("no Hugging Face checkpoints found")
    results = [repair_qwen35_hf_checkpoint(path) for path in checkpoints]
    print(json.dumps([asdict(result) for result in results], indent=2))


if __name__ == "__main__":
    main()
