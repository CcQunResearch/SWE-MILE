"""Shared, Python 3.6-compatible repository helpers uploaded into sandboxes.

Links are opaque filesystem objects.  Only archive member *names* and their
ancestors are paths to access; a symbolic link's target is never followed.
Keep this module stdlib-only: the task interpreter need not have rllm installed.
"""

# capture_output was introduced in Python 3.7; this source runs on 3.6.
# ruff: noqa: UP022
import base64  # noqa: F401 -- shared with the appended helper entrypoint
import hashlib
import io
import json  # noqa: F401 -- shared with the appended helper entrypoint
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

REPOSITORY_STATE_VERSION = 2


def helper_source():
    return Path(__file__).read_text(encoding="utf-8") + "\n"


def repo_unlink(path):
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass


def repo_relative(value, nested_git=False):
    if not isinstance(value, str) or "\0" in value:
        raise RuntimeError("invalid repository path: {!r}".format(value))
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts or pure.parts[0] == ".git" or (not nested_git and ".git" in pure.parts):
        raise RuntimeError("unsafe repository path: {!r}".format(value))
    return pure.as_posix()


def repo_path(root, value, nested_git=False, planned_directories=()):
    relative = repo_relative(value, nested_git)
    pure = PurePosixPath(relative)
    current = Path(root)
    for index, part in enumerate(pure.parts[:-1], 1):
        current = current / part
        prefix = PurePosixPath(*pure.parts[:index]).as_posix()
        # An explicit directory member will replace this object before any
        # children are restored. Never inspect the old directory's children.
        if prefix in planned_directories:
            break
        try:
            info = current.lstat()
        except FileNotFoundError:
            break
        if not stat.S_ISDIR(info.st_mode):
            raise RuntimeError("repository path traverses non-directory: {}".format(relative))
    return Path(root) / relative


def repo_remove(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISDIR(info.st_mode):
        shutil.rmtree(path)
    else:
        path.unlink()


def repo_lstat(root, relative):
    """Read a leaf without traversing links, including replaced parent trees."""
    current = Path(root)
    parts = PurePosixPath(repo_relative(relative, nested_git=True)).parts
    for index, part in enumerate(parts):
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            return None
        if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            return None
    return info


def repo_walk(root, relative, exclude_git=False):
    relative = repo_relative(relative, nested_git=True)
    path = repo_path(root, relative, nested_git=True)
    info = path.lstat()
    if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)):
        raise RuntimeError("unsupported repository path type: {}".format(relative))
    yield relative, info
    if stat.S_ISDIR(info.st_mode):
        for name in sorted(os.listdir(path), key=os.fsencode):
            if name == ".git" and exclude_git:
                continue
            child = path / name
            if name == ".git" and child.is_symlink():
                raise RuntimeError("unsupported symlink Git control path: {}".format(child))
            yield from repo_walk(root, relative + "/" + name, exclude_git)


def repo_directory_fingerprint(root, candidate, include_git=False):
    digest, size, count = hashlib.sha256(), 0, 0
    relative = candidate.relative_to(root).as_posix()
    try:
        for name, info in sorted(repo_walk(root, relative, exclude_git=not include_git), key=lambda entry: os.fsencode(entry[0])):
            if name == relative:
                continue
            path = Path(root) / name
            local = path.relative_to(candidate)
            mode = stat.S_IMODE(info.st_mode)
            if stat.S_ISLNK(info.st_mode):
                kind = b"symlink"
                content = hashlib.sha256(os.fsencode(os.readlink(path))).digest()
            elif stat.S_ISDIR(info.st_mode):
                kind, content = b"directory", hashlib.sha256(b"").digest()
            else:
                kind, hashed = b"file", hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        size += len(chunk)
                        hashed.update(chunk)
                content = hashed.digest()
            digest.update(os.fsencode(local.as_posix()) + b"\0" + kind + b"\0" + str(mode).encode("ascii") + b"\0" + content)
            count += 1
    except (OSError, RuntimeError):
        return None
    return digest.hexdigest(), size, count


def repo_filesystem_metadata(root, relatives):
    """Git omits directory/read modes and hard-link relationships."""
    entries = {}
    for relative in relatives:
        relative = repo_relative(relative)
        info = repo_lstat(root, relative)
        if info is None:
            continue
        if stat.S_ISDIR(info.st_mode):
            entries.update(repo_walk(root, relative, exclude_git=True))
        else:
            entries[relative] = info
        for parent in PurePosixPath(relative).parents:
            if parent.parts:
                entries[parent.as_posix()] = (root / parent.as_posix()).lstat()
    directories, modes, inodes = {}, {}, {}
    for name, info in entries.items():
        mode = stat.S_IMODE(info.st_mode)
        if stat.S_ISDIR(info.st_mode):
            directories[name] = mode
        elif stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            if stat.S_ISREG(info.st_mode) and mode != (0o755 if mode & stat.S_IXUSR else 0o644):
                modes[name] = mode
            if info.st_nlink > 1:
                inodes.setdefault((info.st_dev, info.st_ino), []).append(name)
    return {"directories": directories, "file_modes": modes, "hardlinks": sorted(sorted(names) for names in inodes.values() if len(names) > 1)}


def repo_archive(root, archive_path, relatives, max_bytes=None, exclude_git_roots=(), gitlink_roots=(), parent_directories=()):
    entries, sources, overrides = {}, {}, {}
    for relative in sorted(set(relatives), key=os.fsencode):
        relative = repo_relative(relative)
        for name, info in repo_walk(root, relative, relative in exclude_git_roots):
            entries[name] = info
            sources[name] = repo_path(root, name, nested_git=True)
    parents = set(parent_directories)
    for name in list(entries):
        parents.update(parent.as_posix() for parent in PurePosixPath(name).parents if parent.parts)
    for name in parents:
        if name not in entries:
            path = repo_path(root, name, nested_git=True)
            info = path.lstat()
            if not stat.S_ISDIR(info.st_mode):
                raise RuntimeError("repository parent is not a directory")
            entries[name], sources[name] = info, path
    for relative in gitlink_roots:
        control_name = relative + "/.git"
        if control_name not in entries or not stat.S_ISREG(entries[control_name].st_mode):
            continue
        # A submodule's .git file points into the outer repository's admin
        # directory. Transfer only that submodule's control tree, represented
        # as a self-contained nested .git directory in the destination.
        nested = repo_path(root, relative)
        raw = os.fsdecode(repo_git(nested, "rev-parse", "--git-dir").strip())
        control = (nested / raw).resolve()
        try:
            control.relative_to(root / ".git" / "modules")
        except ValueError as exc:
            raise RuntimeError("gitlink control is outside repository modules: {}".format(relative)) from exc
        entries[control_name], sources[control_name] = control.lstat(), control
        for child in sorted(os.listdir(str(control)), key=os.fsencode):
            for name, info in repo_walk(control, child):
                arcname = control_name + "/" + name
                entries[arcname] = info
                sources[arcname] = repo_path(control, name, nested_git=True)
        config_name = control_name + "/config"
        if config_name in sources:
            with tempfile.NamedTemporaryFile() as temporary_config:
                temporary_config.write(sources[config_name].read_bytes())
                temporary_config.flush()
                configured = subprocess.run(["git", "config", "--file", temporary_config.name, "--unset-all", "core.worktree"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if configured.returncode not in (0, 5):
                    raise RuntimeError("could not make submodule config portable")
                overrides[config_name] = Path(temporary_config.name).read_bytes()
    total = 0
    for name, info in entries.items():
        total += len(os.fsencode(os.readlink(sources[name]))) if stat.S_ISLNK(info.st_mode) else info.st_size if stat.S_ISREG(info.st_mode) else 0
        if max_bytes is not None and total > max_bytes:
            raise RuntimeError("repository delta exceeds {} bytes before archiving".format(max_bytes))
    try:
        with tarfile.open(str(archive_path), "w", dereference=False) as archive:
            inodes = {}
            for name in sorted(entries, key=os.fsencode):
                path = sources[name]
                info = path.lstat()
                member = archive.gettarinfo(str(path), arcname=name)
                inode = (info.st_dev, info.st_ino)
                if not stat.S_ISDIR(info.st_mode) and info.st_nlink > 1:
                    if inode in inodes:
                        member.type, member.linkname, member.size = tarfile.LNKTYPE, inodes[inode], 0
                    else:
                        inodes[inode] = name
                if member.isfile():
                    if name in overrides:
                        member.size = len(overrides[name])
                        archive.addfile(member, io.BytesIO(overrides[name]))
                    else:
                        descriptor = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW)
                        with os.fdopen(descriptor, "rb") as source:
                            archive.addfile(member, source)
                else:
                    archive.addfile(member)
        if max_bytes is not None and archive_path.stat().st_size > max_bytes:
            raise RuntimeError("repository delta archive exceeds {} bytes".format(max_bytes))
    except BaseException:
        repo_unlink(archive_path)
        raise
    return sorted(entries, key=os.fsencode)


def repo_validate_archive(archive, root, expected=None, max_bytes=None):
    members, total = {}, 0
    for member in archive.getmembers():
        name = repo_relative(member.name, nested_git=True)
        if name != member.name or name in members:
            raise RuntimeError("duplicate or noncanonical repository member: {}".format(member.name))
        if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
            raise RuntimeError("unsupported repository archive member: {}".format(name))
        if PurePosixPath(name).name == ".git" and not (member.isdir() or member.isfile()):
            raise RuntimeError("unsupported Git control archive member: {}".format(name))
        members[name] = member
        total += member.size
        if max_bytes is not None and total > max_bytes:
            raise RuntimeError("repository delta exceeds {} bytes when unpacked".format(max_bytes))
    if expected is not None and set(members) != set(expected):
        raise RuntimeError("repository delta member set mismatch")
    directories = {name for name, member in members.items() if member.isdir()}
    for name, member in members.items():
        parts = PurePosixPath(name).parts
        for index in range(1, len(parts)):
            ancestor = PurePosixPath(*parts[:index]).as_posix()
            if ancestor in members and ancestor not in directories:
                raise RuntimeError("repository member traverses archive non-directory: {}".format(name))
            # Nested Git metadata must belong to an explicitly archived tree.
            if parts[index] == ".git" and ancestor not in directories:
                raise RuntimeError("unowned nested Git archive member: {}".format(name))
        repo_path(root, name, nested_git=True, planned_directories=directories)
        if member.islnk():
            target = repo_relative(member.linkname, nested_git=True)
            seen = {name}
            while True:
                if target in seen or target not in members:
                    raise RuntimeError("unsafe repository hard link: {}".format(name))
                seen.add(target)
                other = members[target]
                if other.islnk():
                    target = repo_relative(other.linkname, nested_git=True)
                elif other.isfile() or other.issym():
                    break
                else:
                    raise RuntimeError("unsupported repository hard link target: {}".format(name))
    return members


def repo_extract_archive(archive, root, members=None):
    members = repo_validate_archive(archive, root) if members is None else members
    directories = [name for name, member in members.items() if member.isdir()]
    for name in sorted(directories, key=lambda value: (value.count("/"), os.fsencode(value))):
        path = repo_path(root, name, nested_git=True)
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            repo_remove(path)
        path.mkdir(parents=True, exist_ok=True)
    pending = []
    for name, member in members.items():
        if member.isdir():
            continue
        path = repo_path(root, name, nested_git=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        repo_remove(path)
        if member.issym():
            os.symlink(member.linkname, str(path))
        elif member.islnk():
            pending.append((name, member))
        else:
            # O_NOFOLLOW also prevents a late symlink replacement from turning
            # a repository restore into a write to the link target.
            descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(descriptor, "wb") as destination, archive.extractfile(member) as source:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            os.chmod(str(path), stat.S_IMODE(member.mode), follow_symlinks=False)
    while pending:
        deferred = []
        pending_names = {name for name, _member in pending}
        for name, member in pending:
            if member.linkname in pending_names:
                deferred.append((name, member))
                continue
            source = repo_path(root, member.linkname, nested_git=True)
            target = repo_path(root, name, nested_git=True)
            os.link(str(source), str(target), follow_symlinks=False)
        if len(deferred) == len(pending):
            raise RuntimeError("cyclic repository hard links")
        pending = deferred
    for name in sorted(directories, key=lambda value: value.count("/"), reverse=True):
        os.chmod(str(repo_path(root, name, nested_git=True)), stat.S_IMODE(members[name].mode))


def repo_empty_directories(root, ignored=()):
    result = []
    for current, directories, files in os.walk(str(root), followlinks=False):
        path = Path(current)
        directories[:] = [name for name in directories if name != ".git" and not (path / name).is_symlink() and (path / name).relative_to(root).as_posix() not in ignored]
        if path != root and not files and not os.listdir(str(path)):
            result.append(path.relative_to(root).as_posix())
    return result


def repo_git(root, *args):
    result = subprocess.run(["git", "-c", "safe.directory=" + str(root), "-C", str(root)] + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        error = RuntimeError((result.stderr or result.stdout).decode("utf-8", errors="replace")[-1000:])
        error.exit_code = result.returncode
        raise error
    return result.stdout


def repo_export_delta(payload):
    root, baseline, bundle = Path(payload["root"]).resolve(), str(payload["baseline_ref"]), Path(payload["bundle"])
    max_bytes = int(payload["max_bytes"])
    if repo_git(root, "rev-parse", "HEAD").decode("ascii").strip() != baseline:
        raise RuntimeError("repository HEAD changed")
    changed = repo_git(root, "diff", "--name-only", "--no-renames", "-z", baseline, "--")
    untracked = repo_git(root, "ls-files", "--others", "--exclude-standard", "-z")
    ignored = {os.fsdecode(value[:-1]) for value in repo_git(root, "ls-files", "--others", "--ignored", "--exclude-standard", "--directory", "-z").split(b"\0") if value.endswith(b"/")}
    selected = {repo_relative(os.fsdecode(value)) for value in (changed + untracked).split(b"\0") if value}
    tracked = repo_git(root, "ls-files", "--cached", "-z")
    filesystem = repo_filesystem_metadata(root, {os.fsdecode(value) for value in (tracked + untracked).split(b"\0") if value})
    selected.update(filesystem["file_modes"])
    for names in filesystem["hardlinks"]:
        selected.update(names)
    gitlinks = {os.fsdecode(record.partition(b"\t")[2]) for record in repo_git(root, "ls-files", "--stage", "-z").split(b"\0") if record.startswith(b"160000 ")}
    selected.update(repo_empty_directories(root, ignored))
    present, deleted = [], []
    for relative in sorted(selected, key=os.fsencode):
        if repo_lstat(root, relative) is None:
            deleted.append(relative)
        else:
            present.append(relative)
    bundle.parent.mkdir(parents=True, exist_ok=True)
    temporary = bundle.with_suffix(bundle.suffix + ".tmp")
    repo_unlink(temporary)
    index_path = Path(os.fsdecode(repo_git(root, "rev-parse", "--git-path", "index").strip()))
    if not index_path.is_absolute():
        index_path = root / index_path
    if index_path.is_symlink():
        raise RuntimeError("repository index is a symlink")
    index_size = index_path.stat().st_size if index_path.is_file() else 0
    if index_size >= max_bytes:
        raise RuntimeError("repository delta metadata exceeds {} bytes".format(max_bytes))
    # The index can reference blobs absent from the baseline object database.
    # Keep binary metadata inside the chunked bundle, never in shell argv.
    with tempfile.TemporaryDirectory(prefix="rllm-repo-export-") as scratch:
        scratch = Path(scratch)
        contents = scratch / "contents.tar"
        members = repo_archive(root, contents, present, max_bytes=max_bytes - index_size, gitlink_roots=gitlinks, parent_directories=filesystem["directories"])
        saved = [contents]
        state = scratch / "state.json"
        state.write_text(json.dumps({"complete_directories": [name for name in present if stat.S_ISDIR(repo_lstat(root, name).st_mode)]}))
        saved.append(state)
        if index_size:
            shutil.copyfile(str(index_path), str(scratch / "index"))
            saved.append(scratch / "index")
        baseline_objects = {record.split()[2] for record in repo_git(root, "ls-tree", "-r", "-z", baseline).split(b"\0") if record}
        index_objects = {record.partition(b"\t")[0].split()[1] for record in repo_git(root, "ls-files", "--stage", "-z").split(b"\0") if record and not record.startswith(b"160000 ")}
        needed = sorted(index_objects - baseline_objects - {b"0" * 40})
        if needed:
            identifiers = b"\n".join(needed) + b"\n"
            sizes = subprocess.run(["git", "-C", str(root), "cat-file", "--batch-check=%(objectsize)"], input=identifiers, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            if sum(int(size) for size in sizes.stdout.splitlines()) + index_size + contents.stat().st_size > max_bytes:
                raise RuntimeError("repository delta including index objects exceeds {} bytes".format(max_bytes))
            packed = scratch / "index-objects.pack"
            with packed.open("wb") as handle:
                subprocess.run(["git", "-C", str(root), "pack-objects", "--stdout"], input=identifiers, stdout=handle, stderr=subprocess.PIPE, check=True)
            saved.append(packed)
        try:
            with tarfile.open(str(temporary), "w") as archive:
                for path in saved:
                    archive.add(str(path), arcname=path.name, recursive=False)
            if temporary.stat().st_size > max_bytes:
                raise RuntimeError("repository delta archive exceeds {} bytes".format(max_bytes))
            os.replace(str(temporary), str(bundle))
        finally:
            repo_unlink(temporary)
    digest = hashlib.sha256()
    with bundle.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"ok": True, "size": bundle.stat().st_size, "sha256": digest.hexdigest(), "present": members, "deleted": deleted, "state_version": REPOSITORY_STATE_VERSION}


def repo_import_delta(payload):
    if payload.get("state_version", 1) == 1:
        return repo_import_worktree(payload)
    if payload["state_version"] != REPOSITORY_STATE_VERSION:
        raise RuntimeError("unsupported repository bundle version")
    bundle = Path(payload["bundle"])
    limit = int(payload["max_bytes"])
    if bundle.stat().st_size > limit:
        raise RuntimeError("repository delta archive exceeds {} bytes".format(limit))
    with tempfile.TemporaryDirectory(prefix="rllm-repo-import-") as scratch:
        scratch = Path(scratch)
        with tarfile.open(str(bundle), "r") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if (
                len(set(names)) != len(names)
                or "contents.tar" not in names
                or "state.json" not in names
                or set(names) - {"contents.tar", "index", "index-objects.pack", "state.json"}
                or any(not member.isfile() for member in members)
                or sum(member.size for member in members) > limit
            ):
                raise RuntimeError("invalid repository bundle envelope")
            for member in members:
                with archive.extractfile(member) as source, (scratch / member.name).open("wb") as destination:
                    shutil.copyfileobj(source, destination)
        index = scratch / "index"
        pack = scratch / "index-objects.pack"
        state = json.loads((scratch / "state.json").read_text())
        complete = state.get("complete_directories")
        if not isinstance(complete, list) or any(not isinstance(name, str) for name in complete):
            raise RuntimeError("invalid repository directory manifest")
        result = repo_import_worktree(
            dict(payload, bundle=str(scratch / "contents.tar"), complete_directories=complete), index.read_bytes() if index.exists() else None, pack if pack.exists() else None
        )
    repo_unlink(bundle)
    return result


def repo_import_worktree(payload, index_bytes=None, object_pack=None):
    root, baseline, bundle = Path(payload["root"]).resolve(), str(payload["baseline_ref"]), Path(payload["bundle"])
    if repo_git(root, "rev-parse", "HEAD").decode("ascii").strip() != baseline:
        raise RuntimeError("shadow baseline mismatch")
    if index_bytes is not None and not index_bytes.startswith(b"DIRC"):
        raise RuntimeError("invalid repository index header")
    index_path = Path(os.fsdecode(repo_git(root, "rev-parse", "--git-path", "index").strip()))
    if not index_path.is_absolute():
        index_path = root / index_path
    if index_path.is_symlink():
        raise RuntimeError("repository index is a symlink")
    # Validate the complete payload before reset/clean/deletion changes state.
    with tarfile.open(str(bundle), "r") as archive:
        members = repo_validate_archive(archive, root, expected=payload["present"], max_bytes=int(payload["max_bytes"]) - len(index_bytes or b""))
        directories = {name for name, member in members.items() if member.isdir()}
        complete = {repo_relative(name) for name in payload.get("complete_directories", directories)}
        if complete - directories:
            raise RuntimeError("repository directory manifest mismatch")
        deleted = [repo_relative(value) for value in payload["deleted"]]
        # A complete archived parent supersedes tombstones beneath it (e.g.
        # a tracked directory replaced by a file or an opaque symlink).
        deleted = [
            name for name in deleted if not any(parent.as_posix() in complete or (parent.as_posix() in members and not members[parent.as_posix()].isdir()) for parent in PurePosixPath(name).parents)
        ]
        for name in deleted:
            repo_path(root, name, planned_directories=directories)
        if object_pack is not None:
            with object_pack.open("rb") as source:
                subprocess.run(["git", "-C", str(root), "index-pack", "--stdin"], stdin=source, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        repo_git(root, "reset", "--hard", baseline)
        repo_git(root, "clean", "-ffd")
        # Directory payloads are complete trees, including ignored descendants.
        # Remove old trees once, before extraction, so stale files cannot survive.
        roots = [name for name in complete if not any(parent.as_posix() in complete for parent in PurePosixPath(name).parents)]
        for name in roots:
            repo_remove(repo_path(root, name))
        for name in deleted:
            repo_remove(repo_path(root, name))
        repo_extract_archive(archive, root, members)
    if index_bytes is not None:
        descriptor = os.open(str(index_path), os.O_WRONLY | os.O_TRUNC | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(index_bytes)
        if index_path.read_bytes() != index_bytes:
            raise RuntimeError("repository index copy mismatch")
    repo_unlink(bundle)
    return {"ok": True, "files": len(members), "deleted": len(deleted)}
