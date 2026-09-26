"""Opt-in node-local HF snapshot cache, shared by VERL workers on that node.

Set RLLM_MODEL_CACHE_DIR to a node-local disk directory (same path on all nodes).
No default disk allocation is made. Existing use_shm requests retain precedence.
"""

from __future__ import annotations

import fcntl
import functools
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)
_NETWORK_FS = {"ceph", "nfs", "nfs4", "lustre", "gpfs", "fuse.ceph", "fuse.sshfs"}


def _filesystem(path: Path) -> str:
    """Use the longest enclosing mount; unknown storage is not a safe cache."""
    path = path.resolve()
    best = (-1, "unknown")
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        mount = Path(left.split()[4].replace("\\040", " ").replace("\\134", "\\"))
        if path == mount or mount in path.parents:
            if len(str(mount)) > best[0]:
                best = (len(str(mount)), right.split()[0])
    return best[1]


def _manifest(source: Path) -> dict:
    entries = {}
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part.startswith(".") for part in relative.parts) or not path.is_file():
            continue
        stat = path.stat()
        entries[str(relative)] = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    index = json.loads((source / "model.safetensors.index.json").read_text())
    for name in set(index["weight_map"].values()):
        if Path(name).is_absolute() or ".." in Path(name).parts or name not in entries:
            raise ValueError(f"Incomplete or invalid model shard: {name}")
    return entries


def cache_model_snapshot(source: str, cache_root: str) -> str:
    """Publish a complete snapshot once per node; failed copies never go live."""
    source_path = Path(source).resolve()
    if not (source_path / "model.safetensors.index.json").is_file():
        return source
    root = Path(cache_root).resolve()
    if root in source_path.parents:
        return source  # downstream copy_to_local(local_path) must not recache
    if root == source_path or source_path in root.parents:
        raise ValueError("Model cache must be outside the source snapshot")
    root.mkdir(parents=True, exist_ok=True)
    filesystem = _filesystem(root)
    if filesystem in _NETWORK_FS or filesystem in {"tmpfs", "ramfs", "unknown"} or filesystem.startswith("fuse."):
        logger.warning("Model cache disabled on non-local-disk filesystem %s: %s", filesystem, root)
        return source
    # One node-wide lock also prevents two different snapshots racing the free
    # space check. Never evict snapshots that another process might be mapping.
    with (root / ".copy.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = _manifest(source_path)
        identity = json.dumps([str(source_path), manifest], sort_keys=True).encode()
        key = hashlib.sha256(identity).hexdigest()
        destination = root / key
        marker = destination / ".complete.json"
        if marker.is_file():
            try:
                if json.loads(marker.read_text()) == manifest and all(
                    (destination / name).stat().st_size == info[0] and (destination / name).stat().st_mtime_ns == info[1] for name, info in manifest.items()
                ):
                    logger.info("Model cache hit: %s", destination)
                    return str(destination)
            except (OSError, ValueError):
                pass
        if destination.exists():
            # Do not replace a possibly mmap'ed snapshot on a cache hit failure.
            raise ValueError(f"Invalid published model cache: {destination}")
        required = sum(info[0] for info in manifest.values())
        if shutil.disk_usage(root).free < required * 1.05 + (5 << 30):
            logger.warning("Insufficient disk space for model cache (%d bytes plus reserve); using %s", required, source)
            return source
        # Remove only abandoned, unpublished copies of this exact snapshot.
        for abandoned in root.glob(f".{key}.partial-*"):
            shutil.rmtree(abandoned)
        temporary = Path(tempfile.mkdtemp(prefix=f".{key}.partial-", dir=root))
        try:
            logger.info("Staging model snapshot once on this node: %s -> %s", source, destination)
            for name, info in manifest.items():
                target = temporary / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path / name, target)
                if target.stat().st_size != info[0]:
                    raise ValueError(f"Incomplete copied shard: {name}")
            if _manifest(source_path) != manifest:
                raise ValueError("Source model changed during cache copy")
            (temporary / ".complete.json").write_text(json.dumps(manifest, sort_keys=True))
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        logger.info("Model cache ready: %s (%d bytes)", destination, required)
        return str(destination)


def install_verl_model_cache() -> None:
    """Install before HFModelConfig construction in every Ray worker."""
    root = os.environ.get("RLLM_MODEL_CACHE_DIR")
    if not root:
        return
    import verl.utils.fs as fs

    original = fs.copy_to_local
    if getattr(original, "_rllm_model_cache", False):
        return

    @functools.wraps(original)
    def copy_to_local(*args, **kwargs):
        path = original(*args, **kwargs)
        use_shm = kwargs.get("use_shm", args[5] if len(args) > 5 else False)
        if use_shm:
            return path
        try:
            return cache_model_snapshot(path, root)
        except (OSError, ValueError, KeyError, TypeError):
            logger.warning("Model cache unavailable; retaining original model path %s", path, exc_info=True)
            return path

    copy_to_local._rllm_model_cache = True
    # Earlier worker imports may already have bound the original helper.
    for name, module in list(sys.modules.items()):
        if name.startswith("verl.") and getattr(module, "copy_to_local", None) is original:
            module.copy_to_local = copy_to_local
    fs.copy_to_local = copy_to_local
