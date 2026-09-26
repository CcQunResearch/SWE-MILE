"""Bundle trusted worker Git and its ELF dependencies for offline primary setup.

The task image's Python, libc, package database, and verifier stay untouched.
ELF programs use the bundled loader instead of the task image's older libc.
"""

import functools
import hashlib
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import threading

ROOT = "/opt/rllm-nl2repo-git"
_lock = threading.Lock()
logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _bundle():
    temporary = tempfile.TemporaryDirectory(prefix="rllm-git-bundle-")
    root = Path(temporary.name)
    payload = root / "payload"
    (payload / "lib").mkdir(parents=True)
    git = shutil.which("git")
    if not git:
        raise RuntimeError("Repository-generation offline setup requires Git in the RLLM worker image")
    env = {**os.environ, "LD_PRELOAD": "", "LD_LIBRARY_PATH": ""}
    core = Path(subprocess.check_output([git, "--exec-path"], text=True, env=env).strip())
    loader = None
    programs = {}

    def copy_program(source, relative):
        nonlocal loader
        destination = payload / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as stream:
            elf = stream.read(4) == b"\x7fELF"
        if not elf:
            shutil.copy2(source, destination)
            return
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest in programs:
            destination.symlink_to(os.path.relpath(programs[digest], destination.parent))
            return
        programs[digest] = destination
        result = subprocess.run(["ldd", str(source)], env=env, text=True, capture_output=True, check=True, timeout=30)
        if "not found" in result.stdout:
            raise RuntimeError(f"Incomplete worker Git dependencies: {result.stdout}")
        paths = []
        for line in result.stdout.splitlines():
            match = re.match(r"\s*[\w.+-]+\s+=>\s+(/\S+)", line)
            direct = re.match(r"\s*(/\S*ld[^/\s]*\.so[^\s]*)\s+\(", line)
            if match:
                paths.append(Path(match[1]))
            if direct:
                loader = Path(direct[1]).name
                paths.append(Path(direct[1]))
        for library in paths:
            target = payload / "lib" / library.name
            if not target.exists():
                shutil.copy2(library, target)
        real = destination.with_name(destination.name + ".elf")
        shutil.copy2(source, real)
        if loader is None:
            raise RuntimeError("Cannot locate the worker Git ELF loader")
        destination.write_text(
            "#!/bin/sh\n"
            f"export GIT_EXEC_PATH={ROOT}/libexec\n"
            f"export GIT_TEMPLATE_DIR={ROOT}/templates\n"
            f'exec {ROOT}/lib/{loader} --argv0 "$0" --library-path {ROOT}/lib '
            f'{ROOT}/{relative}.elf "$@"\n'
        )
        destination.chmod(0o755)

    copy_program(Path(git), "bin/git")
    # Git's exec-path is a trusted worker image directory, never task content.
    for source in core.iterdir():
        if source.is_file():
            if source.samefile(git):
                target = payload / "libexec" / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to("../bin/git")
            else:
                copy_program(source, "libexec/" + source.name)
    templates = Path("/usr/share/git-core/templates")
    if templates.is_dir():
        shutil.copytree(templates, payload / "templates", symlinks=False)
    else:
        (payload / "templates").mkdir()
    archive = root / "git.tar.gz"
    with tarfile.open(archive, "w:gz", dereference=False) as stream:
        stream.add(payload, arcname=".")
    # Retain the temporary directory for this worker's lifetime.
    return temporary, archive


def ensure_nl2repo_git(sandbox):
    executor = getattr(sandbox, "exec_setup", None) or sandbox.exec
    status = executor(
        "if command -v git >/dev/null 2>&1; then git --version; else echo RLLM_GIT_MISSING; fi",
        timeout=30,
    )
    if "RLLM_GIT_MISSING" not in status:
        return
    with _lock:
        _, archive = _bundle()
    logger.info("Repository-generation setup: injecting offline worker Git (%d bytes)", archive.stat().st_size)
    sandbox.upload_file(str(archive), "/tmp/rllm-nl2repo-git.tar.gz")
    executor(
        f"set -eu; mkdir -p {ROOT}; tar -xzf /tmp/rllm-nl2repo-git.tar.gz -C {ROOT}; "
        f"ln -s {ROOT}/bin/git /usr/bin/git; "
        "rm /tmp/rllm-nl2repo-git.tar.gz; git --version",
        timeout=120,
    )
