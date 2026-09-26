"""Single-threaded privileged helpers for the OCI MiniSandbox runtime.

The Ray node service is intentionally multi-threaded.  Namespace and chroot
operations must never mutate that process, so every privileged transition is
performed by this small, short-lived helper instead.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import errno
import hashlib
import json
import os
import pwd
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

OCI_HELPER_REVISION = 13

# Linux memory devices: never bind the node's device inodes into task roots.
_PRIVATE_DEVICES = {"null": (1, 3), "zero": (1, 5), "random": (1, 8), "urandom": (1, 9)}


class MiniSandboxDeviceError(RuntimeError):
    """A private runtime device could not be prepared before agent admission."""


def _prepare_devices(root: Path) -> None:
    for name, (major, minor) in _PRIVATE_DEVICES.items():
        path = root / "dev" / name
        try:
            os.mknod(path, stat.S_IFCHR | 0o666, os.makedev(major, minor))
            # mknod obeys the helper's inherited umask; normalize explicitly.
            os.chown(path, 0, 0)
            os.chmod(path, 0o666)
            # A read-only mount protects inode metadata, not character-device
            # I/O. Do NOT set nodev: shells and Git must still open /dev/null.
            # Self-bind our private inode, never the outer Pod's /dev inode.
            _mount("--bind", str(path), str(path))
            _mount("-o", "remount,bind,ro,nosuid,noexec", str(path))
            fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
            try:
                info = os.fstat(fd)
                if (not stat.S_ISCHR(info.st_mode)
                        or info.st_rdev != os.makedev(major, minor)
                        or stat.S_IMODE(info.st_mode) != 0o666
                        or (info.st_uid, info.st_gid) != (0, 0)
                        or not os.fstatvfs(fd).f_flag & os.ST_RDONLY):
                    raise RuntimeError("device identity, permissions or read-only mount invalid")
            finally:
                os.close(fd)
        except Exception as exc:
            raise MiniSandboxDeviceError(f"private device setup failed: /dev/{name}: {exc}") from exc

# Trusted task setup sometimes has to remove image-owned build artifacts before
# the agent starts. Keep only the three filesystem capabilities required for
# that host-supplied script; ordinary agent/verifier commands keep none.
TRUSTED_SETUP_CAPABILITIES = (0, 1, 3)  # CHOWN, DAC_OVERRIDE, FOWNER


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=check, capture_output=True, text=True)


def _mount(*args: str) -> None:
    result = _run(["mount", *args], check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"mount {' '.join(args)} failed: {(result.stderr or result.stdout).strip()}"
        )


def _bind_readonly(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        target.unlink()
    if source.is_dir():
        if target.exists() and not target.is_dir():
            target.unlink()
        target.mkdir(parents=True, exist_ok=True)
    else:
        if target.is_dir():
            target.rmdir()
        target.touch(exist_ok=True)
    _mount("--bind", str(source), str(target))
    _mount("-o", "remount,bind,ro,nosuid,nodev,noexec", str(target))


def _host_pid() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("NSpid:"):
                values = [int(value) for value in line.split()[1:]]
                if values:
                    return values[0]
    except OSError:
        pass
    return os.getpid()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_spec(path: str) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("MiniSandbox runtime spec must be a JSON object")
    return value


def launch(spec_path: str) -> None:
    """Join pre-created cgroups, then exec unshare so descendants inherit them."""

    spec = _load_spec(spec_path)
    for tasks_path in spec.get("cgroup_tasks", []):
        Path(tasks_path).write_text(f"{os.getpid()}\n", encoding="ascii")
    argv = [
        "unshare",
        "--fork",
        "--kill-child=SIGKILL",
        "--pid",
        "--mount",
        "--net",
        "--uts",
        "--ipc",
        "--cgroup",
        sys.executable,
        "-m",
        "rllm.sandbox.minisandbox_runtime.oci_helper",
        "init",
        "--spec",
        spec_path,
    ]
    os.execvp(argv[0], argv)


def enter_runtime(
    spec_path: str,
    target_pid: int,
    operation: str,
    operation_arguments_json: str,
) -> None:
    """Join the sandbox cgroups before entering all of its namespaces."""
    if operation not in {"exec", "write-file", "extract", "read-files"}:
        raise ValueError(f"invalid MiniSandbox helper operation: {operation}")
    arguments = json.loads(operation_arguments_json)
    if not isinstance(arguments, list) or not all(
        isinstance(value, str) for value in arguments
    ):
        raise ValueError("MiniSandbox helper arguments must be a string list")
    spec = _load_spec(spec_path)
    for tasks_path in spec.get("cgroup_tasks", []):
        Path(tasks_path).write_text(f"{os.getpid()}\n", encoding="ascii")
    argv = [
        "nsenter",
        "--target",
        str(target_pid),
        "--mount",
        "--pid",
        "--net",
        "--uts",
        "--ipc",
        "--cgroup",
        # A separate helper process group lets the host terminate the command
        # without killing nsenter's outer monitor before it reaps its child.
        "setsid",
        sys.executable,
        "-m",
        "rllm.sandbox.minisandbox_runtime.oci_helper",
        operation,
        "--spec",
        spec_path,
        *arguments,
    ]
    os.execvp(argv[0], argv)


def _prepare_mounts(spec: dict) -> None:
    session = Path(spec["session_root"])
    rootfs = Path(spec["base_rootfs"])
    upper = session / "upper"
    work = session / "work"
    merged = session / "merged"
    for path in (upper, work, merged):
        path.mkdir(parents=True, exist_ok=True)

    _mount("--make-rprivate", "/")
    _mount(
        "-t",
        "overlay",
        "overlay",
        "-o",
        f"lowerdir={rootfs},upperdir={upper},workdir={work}",
        str(merged),
    )

    # Some imported rootfs caches have lost the standard usrmerge links even
    # though the OCI layer contains the corresponding /usr directories.  Do
    # not mutate the immutable lowerdir: repair only absent canonical entries
    # in this session's overlay upperdir.  Existing files, directories, and
    # even dangling image-owned links remain untouched.
    for link_name, target_name in (
        ("bin", "usr/bin"),
        ("sbin", "usr/sbin"),
        ("lib", "usr/lib"),
        ("lib64", "usr/lib64"),
    ):
        link = merged / link_name
        target = merged / target_name
        if not os.path.lexists(link) and target.is_dir():
            link.symlink_to(target_name)

    for relative in ("proc", "dev", "dev/shm", "dev/pts", "run", "sys/fs/cgroup"):
        (merged / relative).mkdir(parents=True, exist_ok=True)
    _mount("-t", "proc", "-o", "nosuid,nodev,noexec", "proc", str(merged / "proc"))
    _mount("-t", "tmpfs", "-o", "mode=0755,nosuid", "tmpfs", str(merged / "dev"))
    (merged / "dev/shm").mkdir(exist_ok=True)
    (merged / "dev/pts").mkdir(exist_ok=True)
    _mount(
        "-t", "tmpfs", "-o", "mode=1777,nosuid,nodev", "tmpfs", str(merged / "dev/shm")
    )
    # devpts is optional for non-interactive command execution.  Keep the
    # sandbox usable on kernels that reject a nested newinstance mount.
    _run(
        [
            "mount",
            "-t",
            "devpts",
            "-o",
            "newinstance,ptmxmode=0666,mode=0620,nosuid,noexec",
            "devpts",
            str(merged / "dev/pts"),
        ],
        check=False,
    )
    _mount("-t", "tmpfs", "-o", "mode=0755,nosuid,nodev", "tmpfs", str(merged / "run"))

    _prepare_devices(merged)
    for name, target in (
        ("fd", "/proc/self/fd"),
        ("stdin", "/proc/self/fd/0"),
        ("stdout", "/proc/self/fd/1"),
        ("stderr", "/proc/self/fd/2"),
        ("ptmx", "pts/ptmx"),
    ):
        link = merged / "dev" / name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target)

    # Expose only this sandbox's cgroup directories.  The rest of /sys stays
    # absent, so host devices and GPU topology cannot be discovered through it.
    for controller, source_raw in spec.get("cgroup_views", {}).items():
        source = Path(source_raw)
        target = merged / "sys/fs/cgroup" / controller
        _bind_readonly(source, target)

    for name in ("hosts", "resolv.conf"):
        source = session / "etc" / name
        if source.is_file():
            _bind_readonly(source, merged / "etc" / name)
    tmp = merged / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    tmp.chmod(0o1777)


def init_runtime(spec_path: str) -> None:
    spec = _load_spec(spec_path)
    session = Path(spec["session_root"])
    error_path = session / "error.json"
    try:
        _prepare_mounts(spec)
        hostname = str(spec.get("hostname") or "minisandbox")[:63]
        result = _run(["hostname", hostname], check=False)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout).strip())
        _write_json(
            session / "state.json",
            {"host_pid": _host_pid(), "namespace_pid": os.getpid()},
        )
        network_gate = session / "network.ready"
        deadline = time.monotonic() + float(spec.get("startup_timeout", 60.0))
        while not network_gate.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("network namespace configuration timed out")
            time.sleep(0.02)

        merged = session / "merged"
        # This path is visible in the node service's mount namespace.  A
        # marker under ``merged`` would only be visible inside this namespace.
        (session / "ready").write_text("ready\n", encoding="ascii")
        os.chroot(merged)
        os.chdir("/")

        stopping = False

        def stop(_signum, _frame) -> None:
            nonlocal stopping
            stopping = True

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        while not stopping:
            try:
                while True:
                    child, _status = os.waitpid(-1, os.WNOHANG)
                    if child == 0:
                        break
            except ChildProcessError:
                pass
            time.sleep(0.1)
    except BaseException as exc:
        try:
            _write_json(
                error_path,
                {"type": type(exc).__name__, "message": str(exc),
                 "stage": "private_devices" if isinstance(exc, MiniSandboxDeviceError) else "runtime_start"},
            )
        except Exception:
            pass
        raise


PR_SET_NO_NEW_PRIVS = 38
PR_CAPBSET_DROP = 24
SECCOMP_ACT_ALLOW = 0x7FFF0000


def _restrict_capabilities(allowed: tuple[int, ...]) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    allowed_set = set(allowed)
    for capability in range(64):
        if capability not in allowed_set:
            libc.prctl(PR_CAPBSET_DROP, capability, 0, 0, 0)

    class CapHeader(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]

    class CapData(ctypes.Structure):
        _fields_ = [
            ("effective", ctypes.c_uint32),
            ("permitted", ctypes.c_uint32),
            ("inheritable", ctypes.c_uint32),
        ]

    header = CapHeader(0x20080522, 0)
    data = (CapData * 2)()
    for capability in allowed_set:
        if not 0 <= capability < 64:
            raise ValueError(f"invalid Linux capability: {capability}")
        index = capability // 32
        mask = 1 << (capability % 32)
        data[index].effective |= mask
        data[index].permitted |= mask
    if libc.capset(ctypes.byref(header), ctypes.byref(data)) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _drop_capabilities() -> None:
    _restrict_capabilities(())


def _load_seccomp_library():
    """Load the trusted host libseccomp before entering a task rootfs."""

    library_path = ctypes.util.find_library("seccomp") or "libseccomp.so.2"
    try:
        return ctypes.CDLL(library_path, use_errno=True)
    except OSError as exc:
        raise RuntimeError(
            "host libseccomp is required for MiniSandbox command execution"
        ) from exc


def _install_seccomp(library=None) -> None:
    if library is None:
        library = _load_seccomp_library()
    library.seccomp_init.argtypes = [ctypes.c_uint32]
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_release.argtypes = [ctypes.c_void_p]
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int
    library.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]

    class ScmpArgCmp(ctypes.Structure):
        _fields_ = [
            ("arg", ctypes.c_uint),
            ("op", ctypes.c_int),
            ("datum_a", ctypes.c_uint64),
            ("datum_b", ctypes.c_uint64),
        ]

    library.seccomp_rule_add_array.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(ScmpArgCmp),
    ]
    library.seccomp_load.argtypes = [ctypes.c_void_p]

    context = library.seccomp_init(SECCOMP_ACT_ALLOW)
    if not context:
        raise RuntimeError("seccomp_init failed")
    deny_action = 0x00050000 | (errno.EPERM & 0xFFFF)
    denied = (
        "mount",
        "umount2",
        "pivot_root",
        "open_tree",
        "move_mount",
        "fsopen",
        "fsconfig",
        "fsmount",
        "fspick",
        "mount_setattr",
        "setns",
        "unshare",
        "bpf",
        "perf_event_open",
        "kexec_load",
        "kexec_file_load",
        "open_by_handle_at",
        "init_module",
        "finit_module",
        "delete_module",
        "reboot",
        "swapon",
        "swapoff",
        "ptrace",
        "process_vm_readv",
        "process_vm_writev",
        "keyctl",
        "add_key",
        "request_key",
        "userfaultfd",
        "io_uring_setup",
        "io_uring_register",
        "quotactl",
        "acct",
    )
    try:
        for name in denied:
            number = library.seccomp_syscall_resolve_name(name.encode("ascii"))
            if (
                number >= 0
                and library.seccomp_rule_add(context, deny_action, number, 0) != 0
            ):
                raise RuntimeError(f"could not add seccomp rule for {name}")
        # Blocking unshare(2) is not sufficient: clone(2) can create every
        # namespace as well.  Keep ordinary process/thread creation working,
        # but reject clone calls carrying a namespace flag. Linux/amd64 places
        # clone flags in argument 0 (the only supported platform here).
        clone_number = library.seccomp_syscall_resolve_name(b"clone")
        namespace_flags = (
            0x00000080,  # CLONE_NEWTIME
            0x00020000,  # CLONE_NEWNS
            0x02000000,  # CLONE_NEWCGROUP
            0x04000000,  # CLONE_NEWUTS
            0x08000000,  # CLONE_NEWIPC
            0x10000000,  # CLONE_NEWUSER
            0x20000000,  # CLONE_NEWPID
            0x40000000,  # CLONE_NEWNET
        )
        if clone_number >= 0:
            for flag in namespace_flags:
                comparison = ScmpArgCmp(0, 7, flag, flag)  # MASKED_EQ
                if (
                    library.seccomp_rule_add_array(
                        context,
                        deny_action,
                        clone_number,
                        1,
                        ctypes.byref(comparison),
                    )
                    != 0
                ):
                    raise RuntimeError("could not add seccomp clone namespace rule")
        clone3_number = library.seccomp_syscall_resolve_name(b"clone3")
        if clone3_number >= 0:
            # ENOSYS makes glibc fall back to clone(2), where namespace flags
            # are filtered without breaking pthread/process creation.
            enosys_action = 0x00050000 | (errno.ENOSYS & 0xFFFF)
            if library.seccomp_rule_add(context, enosys_action, clone3_number, 0) != 0:
                raise RuntimeError("could not add seccomp clone3 fallback rule")
        if library.seccomp_load(context) != 0:
            raise RuntimeError("seccomp_load failed")
    finally:
        library.seccomp_release(context)


def _safe_inside_path(value: str, *, allow_root: bool = False) -> str:
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe sandbox path: {value!r}")
    if not allow_root and str(path) == "/":
        raise ValueError("sandbox path may not be root")
    return str(path)


def _select_command_shell(root: Path | None = None) -> str:
    """Select Bash without assuming a particular usrmerge entry point."""

    root = Path("/") if root is None else Path(root)
    for shell in ("/bin/bash", "/usr/bin/bash"):
        candidate = root / shell.removeprefix("/")
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return shell
    raise RuntimeError(
        "task rootfs has no executable Bash at /bin/bash or /usr/bin/bash"
    )


def _enter_root(
    spec: dict,
    user: str | None,
    *,
    trusted_setup: bool = False,
) -> None:
    # Load libseccomp while the helper still sees the trusted host filesystem.
    # DeNovoSWE task rootfs images are intentionally minimal and need not ship
    # this host runtime dependency.  Keeping the handle across chroot also
    # prevents an untrusted task image from supplying a replacement library.
    seccomp_library = _load_seccomp_library()
    os.chroot(Path(spec["session_root"]) / "merged")
    working_dir = _safe_inside_path(
        str(spec.get("working_dir") or "/"), allow_root=True
    )
    try:
        os.chdir(working_dir)
    except FileNotFoundError:
        os.chdir("/")

    uid = 0
    gid = 0
    username = "root"
    if trusted_setup and user and user not in {"root", "0"}:
        raise ValueError("trusted MiniSandbox setup must execute as root")
    if user and user not in {"root", "0"}:
        try:
            if str(user).isdigit():
                entry = pwd.getpwuid(int(user))
            else:
                entry = pwd.getpwnam(str(user))
        except KeyError as exc:
            raise ValueError(f"unknown sandbox user: {user}") from exc
        uid, gid, username = entry.pw_uid, entry.pw_gid, entry.pw_name
        os.initgroups(username, gid)
        os.setgid(gid)
        os.setuid(uid)
    if trusted_setup:
        _restrict_capabilities(TRUSTED_SETUP_CAPABILITIES)
    else:
        _drop_capabilities()
    _install_seccomp(seccomp_library)


def execute(
    spec_path: str,
    command: str,
    user: str | None,
    *,
    trusted_setup: bool = False,
) -> None:
    spec = _load_spec(spec_path)
    # Open and unlink the output file while the host session directory is still
    # visible.  Minimal task rootfs images are not required to provide /tmp,
    # and the open descriptor remains usable after chroot and privilege drop.
    # A regular file also prevents a deliberately persistent background process
    # from keeping the node service's RPC stdout pipe open after its parent shell
    # exits.  Any descendant retaining stdout merely retains the anonymous file.
    with tempfile.TemporaryFile(
        mode="w+b",
        prefix=".minisandbox-exec-",
        dir=spec["session_root"],
    ) as output:
        _enter_root(spec, user, trusted_setup=trusted_setup)
        shell = _select_command_shell()
        environment = {
            str(key): str(value)
            for key, value in (spec.get("environment") or {}).items()
        }
        environment.setdefault(
            "PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        )
        environment.setdefault("HOME", "/root")
        environment.setdefault("USER", "root" if not user else str(user))
        environment["CUDA_VISIBLE_DEVICES"] = ""
        returncode = subprocess.call(
            [shell, "-c", command],
            env=environment,
            stdin=sys.stdin.buffer,
            stdout=output,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )
        output.flush()
        output.seek(0)
        shutil.copyfileobj(output, sys.stdout.buffer, length=1024 * 1024)
        sys.stdout.buffer.flush()
    raise SystemExit(returncode)


def write_file(
    spec_path: str, destination: str, mode: int,
    expected_sha256: str | None = None, expected_size: int | None = None,
) -> None:
    spec = _load_spec(spec_path)
    _enter_root(spec, "root")
    destination = _safe_inside_path(destination)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.minisandbox-", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        digest = hashlib.sha256()
        size = 0
        with os.fdopen(descriptor, "wb") as handle:
            while chunk := sys.stdin.buffer.read(1024 * 1024):
                handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if ((expected_size is not None and size != expected_size)
                or (expected_sha256 is not None and digest.hexdigest() != expected_sha256)):
            raise ValueError("MiniSandbox upload checksum/size mismatch; original file preserved")
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_files(spec_path: str, paths: list[str], user: str | None = None, max_bytes: int = 1048576) -> None:
    """Read only regular files after entering the task's root and credentials.

    The worker interpreter is loaded before chroot; task images need no Python
    for this control operation. O_NONBLOCK prevents FIFO substitution hangs.
    """
    if type(max_bytes) is not int or not 0 < max_bytes <= 8388608:
        raise ValueError("max_bytes must be an integer in [1, 8388608]")
    spec = _load_spec(spec_path)
    paths = [_safe_inside_path(path) for path in paths]
    _enter_root(spec, user)
    records = {}
    for path in paths:
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as handle:
                if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                    records[path] = {"state": "invalid", "error": "not_regular_file"}
                    continue
                size = os.fstat(handle.fileno()).st_size
                content = handle.read(max_bytes + 1)
            if len(content) > max_bytes:
                records[path] = {"state": "invalid", "error": "file_too_large", "size": max(size, len(content)), "max_bytes": max_bytes}
            else:
                records[path] = {"state": "present", "content": content.decode("utf-8"),
                                 "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        except FileNotFoundError:
            records[path] = {"state": "missing"}
        except UnicodeDecodeError:
            records[path] = {"state": "invalid", "error": "invalid_utf8"}
        except Exception as exc:
            records[path] = {"state": "unreadable", "error": type(exc).__name__}
    print(json.dumps(records))


def extract_archive(spec_path: str, destination_parent: str) -> None:
    spec = _load_spec(spec_path)
    _enter_root(spec, "root")
    destination_parent = _safe_inside_path(destination_parent, allow_root=True)
    target = Path(destination_parent)
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=sys.stdin.buffer, mode="r|*") as archive:
        for member in archive:
            member_path = PurePosixPath(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"unsafe archive member: {member.name!r}")
            archive.extract(member, path=target, set_attrs=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for operation in ("launch", "init"):
        child = subparsers.add_parser(operation)
        child.add_argument("--spec", required=True)
    child = subparsers.add_parser("enter")
    child.add_argument("--spec", required=True)
    child.add_argument("--target-pid", type=int, required=True)
    child.add_argument("--helper-operation", required=True)
    child.add_argument("--operation-arguments-json", required=True)
    child = subparsers.add_parser("exec")
    child.add_argument("--spec", required=True)
    child.add_argument("--command", required=True)
    child.add_argument("--user")
    child.add_argument("--trusted-setup", action="store_true")
    child = subparsers.add_parser("write-file")
    child.add_argument("--spec", required=True)
    child.add_argument("--destination", required=True)
    child.add_argument("--mode", type=lambda value: int(value, 8), default=0o644)
    child.add_argument("--expected-sha256")
    child.add_argument("--expected-size", type=int)
    child = subparsers.add_parser("read-files")
    child.add_argument("--spec", required=True)
    child.add_argument("--paths-json", required=True)
    child.add_argument("--user")
    child.add_argument("--max-bytes", type=int, default=1048576)
    child = subparsers.add_parser("extract")
    child.add_argument("--spec", required=True)
    child.add_argument("--destination-parent", required=True)
    args = parser.parse_args()
    if args.operation == "launch":
        launch(args.spec)
    elif args.operation == "init":
        init_runtime(args.spec)
    elif args.operation == "enter":
        enter_runtime(
            args.spec,
            args.target_pid,
            args.helper_operation,
            args.operation_arguments_json,
        )
    elif args.operation == "exec":
        execute(
            args.spec,
            args.command,
            args.user,
            trusted_setup=args.trusted_setup,
        )
    elif args.operation == "write-file":
        write_file(args.spec, args.destination, args.mode, args.expected_sha256, args.expected_size)
    elif args.operation == "read-files":
        read_files(args.spec, json.loads(args.paths_json), args.user, args.max_bytes)
    else:
        extract_archive(args.spec, args.destination_parent)


if __name__ == "__main__":
    main()
