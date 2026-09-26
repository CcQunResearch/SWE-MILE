"""OCI-rootfs MiniSandbox runtime for high-concurrency local execution.

This module deliberately has no SWE-bench, R2E-Gym, Ray, or Docker daemon
dependency.  It provides the Linux isolation primitive used by RLLM's
node-local service: immutable OCI rootfs + per-session overlay, namespaces,
cgroup v1 limits, and a restricted public network.
"""

from __future__ import annotations

import hashlib
import errno
import ipaddress
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OCI_RUNTIME_REVISION = 19


def _pid_identity(pid: int) -> str | None:
    """Start time is stable across exec and distinguishes a reused PID."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


class MiniSandboxRuntimeError(RuntimeError):
    def __init__(self, message: str, *, diagnostics: dict | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


class MiniSandboxCommandInterrupted(MiniSandboxRuntimeError):
    """A running helper was explicitly terminated by this runtime's close."""


def _direct_child_pids(pid: int, *, proc_root: Path = Path("/proc")) -> tuple[int, ...]:
    """Return children created by the process's main thread."""

    try:
        value = (proc_root / str(pid) / "task" / str(pid) / "children").read_text(
            encoding="ascii"
        )
    except OSError:
        return ()
    result = []
    for item in value.split():
        try:
            child = int(item)
        except ValueError:
            continue
        if child > 0:
            result.append(child)
    return tuple(result)


def _namespace_init_pid(
    launcher_pid: int,
    spec_path: str | Path,
    *,
    proc_root: Path = Path("/proc"),
) -> int:
    """Resolve the sandbox init PID in the node service's PID namespace.

    The init helper's ``NSpid`` mapping is not portable across nested container
    and PID-namespace layouts.  The node service already owns the ``unshare``
    launcher PID, so only accept an exact init-helper match below that process.
    This also prevents a stale or reused PID from becoming an ``nsenter``
    target.
    """

    expected_spec = os.fsencode(str(Path(spec_path).resolve()))
    pending = list(_direct_child_pids(launcher_pid, proc_root=proc_root))
    seen: set[int] = set()
    matches: list[int] = []
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        pending.extend(_direct_child_pids(pid, proc_root=proc_root))
        try:
            arguments = tuple(
                item
                for item in (proc_root / str(pid) / "cmdline").read_bytes().split(b"\0")
                if item
            )
        except OSError:
            continue
        if (
            b"rllm.sandbox.minisandbox_runtime.oci_helper" in arguments
            and b"init" in arguments
            and expected_spec in arguments
        ):
            matches.append(pid)
    if len(matches) != 1:
        raise MiniSandboxRuntimeError(
            "could not uniquely resolve MiniSandbox init below unshare launcher "
            f"pid {launcher_pid}: candidates={sorted(matches)}"
        )
    return matches[0]


def _run(
    command: list[str],
    *,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if check and completed.returncode != 0:
        raise MiniSandboxRuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{(completed.stderr or completed.stdout).strip()[-2000:]}"
        )
    return completed


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="ascii")


def _parse_cpu_set(value: str) -> tuple[int, ...]:
    result: list[int] = []
    for item in value.strip().split(","):
        if not item:
            continue
        if "-" in item:
            start, end = (int(part) for part in item.split("-", 1))
            result.extend(range(start, end + 1))
        else:
            result.append(int(item))
    return tuple(dict.fromkeys(result))


def _format_cpu_set(values: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values)


def _nearest_nonempty(path: Path, filename: str) -> str:
    current = path
    while True:
        candidate = current / filename
        try:
            value = candidate.read_text(encoding="ascii").strip()
        except OSError:
            value = ""
        if value:
            return value
        if current == current.parent:
            return ""
        current = current.parent


def _cgroup_mounts() -> dict[str, Path]:
    mounts: dict[str, Path] = {}
    try:
        lines = Path("/proc/mounts").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise MiniSandboxRuntimeError("cannot inspect cgroup mounts") from exc
    wanted = {"cpu", "cpuacct", "memory", "pids", "cpuset"}
    for line in lines:
        fields = line.split()
        if len(fields) < 4 or fields[2] != "cgroup":
            continue
        mountpoint = Path(fields[1].replace("\\040", " "))
        options = set(fields[3].split(","))
        for controller in wanted & options:
            mounts[controller] = mountpoint
    missing = wanted - mounts.keys()
    if missing:
        raise MiniSandboxRuntimeError(
            "MiniSandbox requires writable cgroup v1 controllers: "
            + ", ".join(sorted(missing))
        )
    return mounts


@dataclass
class CgroupLease:
    paths: dict[str, Path]
    task_files: tuple[Path, ...]

    @property
    def views(self) -> dict[str, str]:
        unique: dict[Path, list[str]] = {}
        for controller, path in self.paths.items():
            unique.setdefault(path, []).append(controller)
        return {
            ",".join(sorted(controllers)): str(path)
            for path, controllers in unique.items()
        }

    def close(self, timeout: float = 5.0) -> None:
        # Namespace exit and cgroup removal are asynchronous. Retrying only the
        # later residue check cannot remove a directory after an initial EBUSY.
        pending = set(self.paths.values())
        deadline = time.monotonic() + timeout
        errors = {}
        while pending:
            for path in sorted(pending, key=lambda item: len(str(item)), reverse=True):
                try:
                    path.rmdir()
                except FileNotFoundError:
                    pending.discard(path)
                except OSError as exc:
                    errors[path] = str(exc)
                else:
                    pending.discard(path)
            if not pending:
                return
            if time.monotonic() >= deadline:
                raise MiniSandboxRuntimeError("cgroup cleanup incomplete: " + "; ".join(
                    f"{path}: {errors[path]}" for path in sorted(pending)))
            time.sleep(0.05)


class CgroupV1Manager:
    """Create non-exclusive per-sandbox limits below the current Pod cgroup."""

    def __init__(self, run_id: str) -> None:
        self.mounts = _cgroup_mounts()
        normalized_run = re.sub(r"[^a-zA-Z0-9_.-]", "-", run_id)
        run_digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:12]
        safe_run = f"{normalized_run[:32]}-{run_digest}"
        self.parents: dict[str, Path] = {}
        for controller, mount in self.mounts.items():
            parent = mount / f"rllm-minisandbox-{safe_run}"
            if controller in {"cpu", "cpuacct"}:
                parent = self.parents.get("cpu", parent)
            if not parent.exists():
                parent.mkdir()
            else:
                for stale in parent.iterdir():
                    if not stale.is_dir():
                        continue
                    tasks = stale / "tasks"
                    try:
                        active = tasks.read_text(encoding="ascii").strip()
                    except OSError:
                        active = "unknown"
                    if not active:
                        try:
                            stale.rmdir()
                        except OSError:
                            pass
            self.parents[controller] = parent
        cpuset_parent = self.parents["cpuset"]
        for filename in ("cpuset.cpus", "cpuset.mems"):
            target = cpuset_parent / filename
            if not target.read_text(encoding="ascii").strip():
                value = _nearest_nonempty(cpuset_parent.parent, filename)
                if not value:
                    raise MiniSandboxRuntimeError(f"cannot resolve {filename}")
                _write_text(target, value)
        self.cpus = _parse_cpu_set(
            (cpuset_parent / "cpuset.cpus").read_text(encoding="ascii")
        )
        if not self.cpus:
            raise MiniSandboxRuntimeError("current cgroup has no available CPUs")
        self._next_cpu = 0
        self._lock = threading.Lock()

    def create(self, sandbox_id: str, *, cpus: int, memory_mb: int) -> CgroupLease:
        if cpus <= 0 or memory_mb <= 0:
            raise ValueError("MiniSandbox cpus and memory_mb must be positive")
        if cpus > len(self.cpus):
            raise ValueError(
                f"MiniSandbox requested {cpus} CPUs but only {len(self.cpus)} are available"
            )
        safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "-", sandbox_id)[:80]
        paths: dict[str, Path] = {}
        created: set[Path] = set()
        try:
            for controller, parent in self.parents.items():
                path = parent / safe_id
                if path not in created:
                    path.mkdir()
                    created.add(path)
                paths[controller] = path

            with self._lock:
                start = self._next_cpu
                selected = tuple(
                    self.cpus[(start + offset) % len(self.cpus)]
                    for offset in range(cpus)
                )
                self._next_cpu = (start + cpus) % len(self.cpus)
            cpuset_path = paths["cpuset"]
            _write_text(
                cpuset_path / "cpuset.mems",
                _nearest_nonempty(self.parents["cpuset"], "cpuset.mems"),
            )
            _write_text(cpuset_path / "cpuset.cpus", _format_cpu_set(selected))

            cpu_path = paths["cpu"]
            period = 100_000
            _write_text(cpu_path / "cpu.cfs_period_us", str(period))
            _write_text(cpu_path / "cpu.cfs_quota_us", str(cpus * period))
            shares = cpu_path / "cpu.shares"
            if shares.exists():
                _write_text(shares, str(max(2, cpus * 1024)))

            memory_path = paths["memory"]
            limit = int(memory_mb) * 1024 * 1024
            _write_text(memory_path / "memory.limit_in_bytes", str(limit))
            memsw = memory_path / "memory.memsw.limit_in_bytes"
            if memsw.exists():
                try:
                    _write_text(memsw, str(limit))
                except OSError:
                    # Many Kubernetes cgroup-v1 nodes disable swap accounting.
                    pass
            swappiness = memory_path / "memory.swappiness"
            if swappiness.exists():
                try:
                    _write_text(swappiness, "0")
                except OSError:
                    pass

            pids_path = paths["pids"]
            _write_text(pids_path / "pids.max", str(max(1024, min(8192, cpus * 1024))))
            task_files = tuple(dict.fromkeys(path / "tasks" for path in paths.values()))
            return CgroupLease(paths=paths, task_files=task_files)
        except BaseException:
            for path in sorted(created, key=lambda item: len(str(item)), reverse=True):
                try:
                    path.rmdir()
                except OSError:
                    pass
            raise

    def close(self) -> None:
        for path in sorted(
            set(self.parents.values()), key=lambda item: len(str(item)), reverse=True
        ):
            try:
                path.rmdir()
            except OSError:
                pass


@dataclass(frozen=True)
class NetworkLease:
    address_index: int
    host_veth: str


class BridgeNetworkManager:
    """One scoped bridge/NAT per node service, one veth per sandbox."""

    def __init__(self, run_id: str, blocked_addresses: tuple[str, ...] = ()) -> None:
        digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
        scope = int(digest[:4], 16)
        second = 64 + scope % 64
        third = ((scope // 64) % 16) * 16
        self.prefix = 20
        self.subnet = f"10.{second}.{third}.0/{self.prefix}"
        self.gateway = f"10.{second}.{third}.1"
        self._network = ipaddress.IPv4Network(self.subnet)
        # Reserve network, gateway, and broadcast; 4093 live leases per service.
        self.address_capacity = self._network.num_addresses - 3
        self.bridge = f"msb{digest[:7]}"[:15]
        self.comment = f"rllm-minisandbox-{digest[:7]}"
        self._next_address = 2
        self._leased: set[int] = set()
        self._lock = threading.Lock()
        self.nameservers = self._nameservers()
        self.blocked_addresses = tuple(
            dict.fromkeys([*blocked_addresses, *self._host_addresses()])
        )

    @staticmethod
    def _host_addresses() -> tuple[str, ...]:
        completed = _run(["ip", "-j", "address", "show"], check=False)
        if completed.returncode != 0:
            return ()
        try:
            interfaces = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return ()
        result: list[str] = []
        for interface in interfaces if isinstance(interfaces, list) else []:
            for address in interface.get("addr_info", []):
                if address.get("family") == "inet":
                    local = str(address.get("local") or "")
                    if local and not local.startswith("127."):
                        result.append(local)
        return tuple(dict.fromkeys(result))

    @staticmethod
    def _nameservers() -> tuple[str, ...]:
        result: list[str] = []
        try:
            lines = Path("/etc/resolv.conf").read_text(encoding="utf-8").splitlines()
        except OSError:
            return ()
        for line in lines:
            fields = line.split()
            if len(fields) == 2 and fields[0] == "nameserver" and ":" not in fields[1]:
                result.append(fields[1])
        return tuple(dict.fromkeys(result))

    def _iptables_rule(self, table: str | None, chain: str, rule: list[str]) -> None:
        base = ["iptables"] + (["-t", table] if table else [])
        check = _run(base + ["-C", chain, *rule], check=False)
        if check.returncode != 0:
            _run(base + ["-A", chain, *rule])

    def start(self) -> None:
        if _run(["ip", "link", "show", self.bridge], check=False).returncode != 0:
            _run(["ip", "link", "add", self.bridge, "type", "bridge"])
            _run(
                [
                    "ip",
                    "addr",
                    "add",
                    f"{self.gateway}/{self.prefix}",
                    "dev",
                    self.bridge,
                ]
            )
        _run(["ip", "link", "set", self.bridge, "up"])
        forwarding = Path("/proc/sys/net/ipv4/ip_forward").read_text().strip()
        if forwarding != "1":
            raise MiniSandboxRuntimeError(
                "net.ipv4.ip_forward must already be enabled; MiniSandbox will not mutate the node setting"
            )
        self._iptables_rule(
            "nat",
            "POSTROUTING",
            [
                "-s",
                self.subnet,
                "!",
                "-d",
                self.subnet,
                "-m",
                "comment",
                "--comment",
                self.comment,
                "-j",
                "MASQUERADE",
            ],
        )
        self._iptables_rule(
            None,
            "FORWARD",
            [
                "-i",
                self.bridge,
                "-m",
                "comment",
                "--comment",
                self.comment,
                "-j",
                "ACCEPT",
            ],
        )
        self._iptables_rule(
            None,
            "FORWARD",
            [
                "-o",
                self.bridge,
                "-m",
                "conntrack",
                "--ctstate",
                "RELATED,ESTABLISHED",
                "-m",
                "comment",
                "--comment",
                self.comment,
                "-j",
                "ACCEPT",
            ],
        )

    def _address(self, index: int) -> str:
        return str(self._network.network_address + index)

    def attach(self, sandbox_id: str, host_pid: int) -> NetworkLease:
        with self._lock:
            for _ in range(self.address_capacity):
                index = self._next_address
                self._next_address = 2 if index >= self.address_capacity + 1 else index + 1
                if index not in self._leased:
                    self._leased.add(index)
                    break
            else:
                raise MiniSandboxRuntimeError(
                    f"MiniSandbox bridge address pool exhausted: {len(self._leased)}/{self.address_capacity} leases in {self.subnet}"
                )
        digest = hashlib.sha256(sandbox_id.encode("utf-8")).hexdigest()[:10]
        host_veth = f"mh{digest}"[:15]
        peer_veth = f"mp{digest}"[:15]
        address = self._address(index)
        try:
            _run(
                [
                    "ip",
                    "link",
                    "add",
                    host_veth,
                    "type",
                    "veth",
                    "peer",
                    "name",
                    peer_veth,
                ]
            )
            _run(["ip", "link", "set", host_veth, "master", self.bridge])
            _run(["ip", "link", "set", host_veth, "up"])
            _run(["ip", "link", "set", peer_veth, "netns", str(host_pid)])
            ns = ["nsenter", "--target", str(host_pid), "--net"]
            _run([*ns, "ip", "link", "set", "lo", "up"])
            _run([*ns, "ip", "link", "set", peer_veth, "name", "eth0"])
            _run([*ns, "ip", "addr", "add", f"{address}/{self.prefix}", "dev", "eth0"])
            _run([*ns, "ip", "link", "set", "eth0", "up"])
            _run([*ns, "ip", "route", "add", "default", "via", self.gateway])
            self._apply_namespace_firewall(ns)
            return NetworkLease(address_index=index, host_veth=host_veth)
        except BaseException:
            _run(["ip", "link", "del", host_veth], check=False)
            with self._lock:
                self._leased.discard(index)
            raise

    def _apply_namespace_firewall(self, ns: list[str]) -> None:
        def iptables(*args: str) -> None:
            _run([*ns, "iptables", *args])

        iptables("-P", "INPUT", "DROP")
        iptables("-P", "FORWARD", "DROP")
        iptables("-P", "OUTPUT", "ACCEPT")
        iptables("-A", "INPUT", "-i", "lo", "-j", "ACCEPT")
        iptables(
            "-A",
            "INPUT",
            "-m",
            "conntrack",
            "--ctstate",
            "RELATED,ESTABLISHED",
            "-j",
            "ACCEPT",
        )
        iptables("-A", "OUTPUT", "-o", "lo", "-j", "ACCEPT")
        for nameserver in self.nameservers:
            for protocol in ("udp", "tcp"):
                iptables(
                    "-A",
                    "OUTPUT",
                    "-d",
                    nameserver,
                    "-p",
                    protocol,
                    "--dport",
                    "53",
                    "-j",
                    "ACCEPT",
                )
        for address in self.blocked_addresses:
            iptables("-A", "OUTPUT", "-d", address, "-j", "REJECT")
        for cidr in (
            "10.0.0.0/8",
            "100.64.0.0/10",
            "127.0.0.0/8",
            "169.254.0.0/16",
            "172.16.0.0/12",
            "192.168.0.0/16",
            "224.0.0.0/4",
        ):
            iptables("-A", "OUTPUT", "-d", cidr, "-j", "REJECT")

    def release(self, lease: NetworkLease | None) -> None:
        if lease is None:
            return
        _run(["ip", "link", "del", lease.host_veth], check=False)
        with self._lock:
            self._leased.discard(lease.address_index)

    def close(self) -> None:
        with self._lock:
            if self._leased:
                raise MiniSandboxRuntimeError(
                    "cannot stop MiniSandbox bridge while sandbox veths are active"
                )
        rules = (
            (
                "nat",
                "POSTROUTING",
                [
                    "-s",
                    self.subnet,
                    "!",
                    "-d",
                    self.subnet,
                    "-m",
                    "comment",
                    "--comment",
                    self.comment,
                    "-j",
                    "MASQUERADE",
                ],
            ),
            (
                None,
                "FORWARD",
                [
                    "-i",
                    self.bridge,
                    "-m",
                    "comment",
                    "--comment",
                    self.comment,
                    "-j",
                    "ACCEPT",
                ],
            ),
            (
                None,
                "FORWARD",
                [
                    "-o",
                    self.bridge,
                    "-m",
                    "conntrack",
                    "--ctstate",
                    "RELATED,ESTABLISHED",
                    "-m",
                    "comment",
                    "--comment",
                    self.comment,
                    "-j",
                    "ACCEPT",
                ],
            ),
        )
        for table, chain, rule in rules:
            base = ["iptables"] + (["-t", table] if table else [])
            while _run(base + ["-C", chain, *rule], check=False).returncode == 0:
                _run(base + ["-D", chain, *rule], check=False)
        _run(["ip", "link", "set", self.bridge, "down"], check=False)
        _run(["ip", "link", "del", self.bridge, "type", "bridge"], check=False)


def check_runtime_requirements(local_root: str | Path) -> None:
    from .oci_helper import OCI_HELPER_REVISION

    if OCI_HELPER_REVISION < 13:
        raise MiniSandboxRuntimeError("MiniSandbox requires OCI helper revision 13+ for image environment preservation, bounded file snapshots and private devices")
    if sys.platform != "linux" or os.geteuid() != 0:
        raise MiniSandboxRuntimeError("OCI MiniSandbox requires Linux root privileges")
    required = (
        "setsid",
        "unshare",
        "nsenter",
        "mount",
        "umount",
        "chroot",
        "ip",
        "iptables",
        "skopeo",
        "umoci",
    )
    missing = [name for name in required if shutil.which(name) is None]
    if missing:
        raise MiniSandboxRuntimeError(
            "MiniSandbox runtime tools are missing: " + ", ".join(missing)
        )
    root = Path(local_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "unshare",
            "--fork",
            "--pid",
            "--net",
            "--mount",
            "--uts",
            "--ipc",
            "--cgroup",
            "--mount-proc",
            "true",
        ],
        timeout=20,
    )
    probe = Path(tempfile.mkdtemp(prefix=".overlay-probe-", dir=root))
    try:
        for name in ("lower", "upper", "work", "merged"):
            (probe / name).mkdir()
        (probe / "lower/marker").write_text("ok\n", encoding="ascii")
        script = (
            "set -eu; mount --make-rprivate /; "
            f"mount -t overlay overlay -o lowerdir={probe / 'lower'},upperdir={probe / 'upper'},workdir={probe / 'work'} {probe / 'merged'}; "
            f'test "$(cat {probe / "merged/marker"})" = ok; '
            f"umount {probe / 'merged'}"
        )
        _run(["unshare", "--mount", "sh", "-c", script], timeout=20)
    finally:
        shutil.rmtree(probe, ignore_errors=True)


class OciRootfsRuntime:
    """One persistent namespace with independent command executions."""

    def __init__(
        self,
        *,
        sandbox_id: str,
        base_rootfs: str | Path,
        session_root: str | Path,
        cpus: int,
        memory_mb: int,
        working_dir: str,
        environment: Mapping[str, str] | None,
        allow_internet: bool,
        cgroups: CgroupV1Manager,
        network: BridgeNetworkManager,
        startup_timeout: float = 60.0,
    ) -> None:
        self.sandbox_id = sandbox_id
        self.base_rootfs = Path(base_rootfs).resolve()
        self.session_root = Path(session_root).resolve()
        self.working_dir = working_dir
        self.environment = dict(environment or {})
        self.allow_internet = bool(allow_internet)
        self.startup_timeout = float(startup_timeout)
        self._cgroup = cgroups.create(
            sandbox_id, cpus=int(cpus), memory_mb=int(memory_mb)
        )
        self._network_manager = network
        self._network_lease: NetworkLease | None = None
        self._process: subprocess.Popen | None = None
        self._host_pid: int | None = None
        self._host_pid_identity: str | None = None
        self._host_pidfd: int | None = None
        self._closed = False
        self._cleanup_complete = False
        self._invocations: set[subprocess.Popen] = set()
        self._invocation_stops: dict[subprocess.Popen, dict] = {}
        self._lock = threading.RLock()
        self.spec_path = self.session_root / "runtime.json"

    def start(self) -> None:
        if not self.base_rootfs.is_dir():
            raise FileNotFoundError(self.base_rootfs)
        self.session_root.mkdir(parents=True, exist_ok=False)
        (self.session_root / "etc").mkdir()
        resolv = Path("/etc/resolv.conf")
        (self.session_root / "etc/resolv.conf").write_text(
            resolv.read_text(encoding="utf-8") if resolv.is_file() else "",
            encoding="utf-8",
        )
        hostname = "msb-" + hashlib.sha256(self.sandbox_id.encode()).hexdigest()[:12]
        (self.session_root / "etc/hosts").write_text(
            f"127.0.0.1 localhost {hostname}\n::1 localhost\n", encoding="ascii"
        )
        spec = {
            "schema_version": 1,
            "session_root": str(self.session_root),
            "base_rootfs": str(self.base_rootfs),
            "hostname": hostname,
            "working_dir": self.working_dir,
            "environment": self.environment,
            "startup_timeout": self.startup_timeout,
            "cgroup_tasks": [str(path) for path in self._cgroup.task_files],
            "cgroup_views": self._cgroup.views,
        }
        self.spec_path.write_text(
            json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        command = [
            sys.executable,
            "-m",
            "rllm.sandbox.minisandbox_runtime.oci_helper",
            "launch",
            "--spec",
            str(self.spec_path),
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            state = self._wait_json(self.session_root / "state.json")
            # Resolve from the launcher's actual descendant tree as observed by
            # this node service. ``state[\"host_pid\"]`` is only diagnostic:
            # its NSpid-derived value is ambiguous under nested PID namespaces.
            reported_host_pid = int(state["host_pid"])
            self._host_pid = _namespace_init_pid(
                self._process.pid,
                self.spec_path,
            )
            self._host_pid_identity = _pid_identity(self._host_pid)
            if self._host_pid_identity is None:
                raise MiniSandboxRuntimeError("namespace init exited during PID registration")
            if hasattr(os, "pidfd_open"):
                try:
                    self._host_pidfd = os.pidfd_open(self._host_pid)
                    if _pid_identity(self._host_pid) != self._host_pid_identity:
                        os.close(self._host_pidfd)
                        self._host_pidfd = None
                        raise MiniSandboxRuntimeError("namespace init exited during PID registration")
                except ProcessLookupError:
                    raise MiniSandboxRuntimeError("namespace init exited during PID registration")
                except OSError as exc:
                    if exc.errno not in {errno.ENOSYS, errno.EINVAL, errno.EPERM}:
                        raise
                    # Older/filtered hosts retain identity-checked waiting;
                    # signals go only to the owned, unreaped launcher group.
                    self._host_pidfd = None
            state["reported_host_pid"] = reported_host_pid
            state["host_pid"] = self._host_pid
            state_path = self.spec_path.with_name("state.json")
            temporary_state = state_path.with_name(f".{state_path.name}.resolved.tmp")
            temporary_state.write_text(
                json.dumps(state, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_state, state_path)
            if self.allow_internet:
                self._network_lease = self._network_manager.attach(
                    self.sandbox_id, self._host_pid
                )
            else:
                _run(
                    [
                        "nsenter",
                        "--target",
                        str(self._host_pid),
                        "--net",
                        "ip",
                        "link",
                        "set",
                        "lo",
                        "up",
                    ]
                )
            (self.session_root / "network.ready").touch()
            self._wait_ready()
        except BaseException:
            self.close()
            raise

    def _wait_json(self, path: Path) -> dict[str, Any]:
        deadline = time.monotonic() + self.startup_timeout
        while True:
            error = self.session_root / "error.json"
            if error.is_file():
                detail = json.loads(error.read_text(encoding="utf-8"))
                raise MiniSandboxRuntimeError(str(detail), diagnostics={"runtime_startup": detail})
            if path.is_file():
                value = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    return value
            if self._process is not None and self._process.poll() is not None:
                _stdout, stderr = self._process.communicate(timeout=1)
                raise MiniSandboxRuntimeError(
                    f"MiniSandbox namespace exited during startup: {stderr[-2000:]}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError("MiniSandbox namespace startup timed out")
            time.sleep(0.02)

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        ready = self.session_root / "ready"
        while not ready.is_file():
            error = self.session_root / "error.json"
            if error.is_file():
                detail = json.loads(error.read_text(encoding="utf-8"))
                raise MiniSandboxRuntimeError(str(detail), diagnostics={"runtime_startup": detail})
            if self._process is not None and self._process.poll() is not None:
                raise MiniSandboxRuntimeError(
                    "MiniSandbox namespace exited before ready"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError("MiniSandbox readiness timed out")
            time.sleep(0.02)

    def _helper_command(self, operation: str, *arguments: str) -> list[str]:
        if self._host_pid is None:
            raise MiniSandboxRuntimeError("MiniSandbox is not started")
        return [
            sys.executable,
            "-m",
            "rllm.sandbox.minisandbox_runtime.oci_helper",
            "enter",
            "--spec",
            str(self.spec_path),
            "--target-pid",
            str(self._host_pid),
            "--helper-operation",
            operation,
            "--operation-arguments-json",
            json.dumps(list(arguments)),
        ]

    def _invoke(
        self,
        command: list[str],
        *,
        payload: bytes | None = None,
        timeout: float | None = None,
    ) -> str:
        with self._lock:
            if self._closed or not self.is_alive():
                raise MiniSandboxRuntimeError(
                    f"MiniSandbox {self.sandbox_id} is not alive"
                )
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE if payload is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            # Launch and registration are atomic with close's admission gate.
            self._invocations.add(process)
        try:
            output, _ = process.communicate(input=payload, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            self._terminate_invocation(process)
            try:
                output, _ = process.communicate(timeout=5)
            except subprocess.TimeoutExpired as cleanup_error:
                # Retain this process for close(); never block a Ray worker
                # forever or report an unknown execution result as completed.
                raise MiniSandboxRuntimeError(
                    "exec cleanup incomplete after SIGKILL: " +
                    str(self._process_diagnostics(process.pid))
                ) from cleanup_error
            raise TimeoutError(
                f"MiniSandbox command exceeded {timeout}s: "
                + output.decode("utf-8", errors="replace")[-1000:]
            ) from exc
        finally:
            with self._lock:
                stop = getattr(self, "_invocation_stops", {}).get(process)
                if process.poll() is not None:
                    self._invocations.discard(process)
                    getattr(self, "_invocation_stops", {}).pop(process, None)
        text = output.decode("utf-8", errors="replace")
        if process.returncode != 0:
            diagnostics = {
                "sandbox_id": self.sandbox_id, "stage": "command_exit",
                "helper_pid": process.pid, "returncode": process.returncode,
                "execution_started": True, "termination_request": stop,
            }
            if stop is not None:
                raise MiniSandboxCommandInterrupted(
                    f"MiniSandbox command interrupted by sandbox close: {self.sandbox_id}",
                    diagnostics=diagnostics,
                )
            if process.returncode < 0:
                # An unexpected signal is not automatically OOM or a close.
                # Preserve available evidence without masking the exit itself.
                try:
                    diagnostics["runtime_evidence"] = self._cleanup_diagnostics("command_exit")
                except Exception as exc:
                    diagnostics["evidence_error"] = type(exc).__name__ + ": " + str(exc)
            raise MiniSandboxRuntimeError(
                f"MiniSandbox command failed with exit {process.returncode}:\n{text[-2000:]}",
                diagnostics=diagnostics,
            )
        return text

    def exec(
        self,
        command: str,
        *,
        timeout: float | None = None,
        user: str | None = None,
        trusted_setup: bool = False,
    ) -> str:
        arguments = ["--command", command]
        if user:
            arguments.extend(["--user", user])
        if trusted_setup:
            arguments.append("--trusted-setup")
        return self._invoke(self._helper_command("exec", *arguments), timeout=timeout)

    def write_file(
        self, destination: str, payload: bytes, *, mode: int = 0o644,
        expected_sha256: str | None = None, expected_size: int | None = None,
    ) -> None:
        self._invoke(
            self._helper_command(
                "write-file",
                "--destination",
                destination,
                "--mode",
                f"{mode:o}",
                "--expected-sha256",
                expected_sha256 if expected_sha256 is not None else hashlib.sha256(payload).hexdigest(),
                "--expected-size",
                str(expected_size if expected_size is not None else len(payload)),
            ),
            payload=payload,
            timeout=300,
        )

    def read_files(self, paths: list[str], *, timeout: float = 30, user: str | None = None, max_bytes: int = 1048576) -> dict:
        if type(max_bytes) is not int or not 0 < max_bytes <= 8388608:
            raise ValueError("max_bytes must be an integer in [1, 8388608]")
        arguments = ["--paths-json", json.dumps(paths), "--max-bytes", str(max_bytes)]
        if user:
            arguments.extend(["--user", user])
        return json.loads(self._invoke(self._helper_command("read-files", *arguments), timeout=timeout))

    def extract_archive(self, destination_parent: str, payload: bytes) -> None:
        self._invoke(
            self._helper_command("extract", "--destination-parent", destination_parent),
            payload=payload,
            timeout=600,
        )

    def is_alive(self) -> bool:
        process = self._process
        return (
            not self._closed
            and process is not None
            and process.poll() is None
            and self._host_pid is not None
            and Path(f"/proc/{self._host_pid}").exists()
        )

    @staticmethod
    def _process_diagnostics(pid: int) -> dict:
        diagnostics = {"pid": pid}
        for name in ("status", "wchan", "stack", "syscall"):
            try:
                with Path(f"/proc/{pid}/{name}").open() as stream:
                    diagnostics[name] = stream.read(4096)
            except OSError as exc:
                diagnostics[name] = str(exc)
        return diagnostics

    def _cleanup_diagnostics(self, stage: str) -> dict:
        """Capture kernel/resource evidence without reading task content."""
        result = {"stage": stage, "sandbox_id": getattr(self, "sandbox_id", None),
                  "kernel_release": os.uname().release, "cgroups": {}}
        result["pressure"] = {}
        for name in ("cpu", "io", "memory"):
            try:
                result["pressure"][name] = Path(f"/proc/pressure/{name}").read_text()[:2048]
            except OSError as exc:
                result["pressure"][name] = str(exc)
        process = self._process
        if process is not None:
            result["launcher"] = self._process_diagnostics(process.pid)
        if getattr(self, "_host_pid", None):
            result["namespace_init"] = self._process_diagnostics(self._host_pid)
        for controller, path in getattr(self._cgroup, "paths", {}).items():
            names = {
                "memory": ("memory.usage_in_bytes", "memory.limit_in_bytes",
                           "memory.max_usage_in_bytes", "memory.failcnt", "memory.oom_control"),
                "pids": ("pids.current", "pids.max"),
            }.get(controller, ())
            values = {}
            for name in names:
                try:
                    with (path / name).open() as stream:
                        values[name] = stream.read(2048)
                except OSError as exc:
                    values[name] = str(exc)
            if values:
                result["cgroups"][controller] = values
            if controller == "memory":
                result["memory_ancestors"] = []
                for parent in list(path.parents)[:8]:
                    ancestor = {"path": str(parent)}
                    for name in names:
                        try:
                            ancestor[name] = (parent / name).read_text()[:2048]
                        except OSError:
                            pass
                    if len(ancestor) > 1:
                        result["memory_ancestors"].append(ancestor)
        # Cgroup membership includes tasks whose parent is outside the PID
        # namespace. Parent stacks explain unreaped nsenter children.
        pids = set()
        for path in set(getattr(self._cgroup, "paths", {}).values()):
            try:
                pids.update(int(pid) for pid in (path / "cgroup.procs").read_text().split())
            except (OSError, ValueError):
                pass
        result["remaining_process_count"] = len(pids)
        result["remaining_processes"] = []
        for pid in sorted(pids)[:32]:
            evidence = self._process_diagnostics(pid)
            match = re.search(r"^PPid:\s*(\d+)", evidence.get("status", ""), re.MULTILINE)
            if match:
                evidence["parent"] = self._process_diagnostics(int(match[1]))
            result["remaining_processes"].append(evidence)
        stack = result.get("launcher", {}).get("stack", "")
        if "put_ipc_ns" in stack and "unregister_memcg_shrinker" in stack:
            result["blocked_at"] = "kernel_ipc_shrinker_teardown"
        return result

    @staticmethod
    def _terminate_invocation(process: subprocess.Popen, *, deadline: float | None = None) -> bool:
        # nsenter --pid forks a monitor outside the target PID namespace.
        # Killing that monitor before it reaps its child can leave namespace
        # init stuck in zap_pid_ns_processes. Preserve the monitor and kill
        # the helper's separate process group, including ordinary descendants.
        if process.poll() is not None:
            return False
        stopped = []
        killed = False

        def stop(pid):
            os.kill(pid, signal.SIGSTOP)
            stopped.append(pid)
            stop_deadline = min(time.monotonic() + 1, deadline) if deadline is not None else time.monotonic() + 1
            while True:
                try:
                    state = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1][0]
                except FileNotFoundError:
                    return False
                if state in {"T", "t"}:
                    return True
                if state in {"Z", "X"}:
                    return False
                if time.monotonic() >= stop_deadline:
                    raise MiniSandboxRuntimeError(
                        "exec cleanup incomplete: monitor did not stop: " +
                        str(OciRootfsRuntime._process_diagnostics(pid))
                    )
                time.sleep(0.01)

        try:
            if not stop(process.pid):
                return False
            # The stopped monitor cannot fork or reap/reuse these child PIDs.
            children = _direct_child_pids(process.pid)
            if not children:
                os.kill(process.pid, signal.SIGKILL)
                killed = True
            for pid in children:
                try:
                    if not stop(pid):
                        continue
                    try:
                        os.killpg(pid, signal.SIGKILL)
                        killed = True
                    except ProcessLookupError:
                        # nsenter may not yet have reached setsid; a stopped
                        # pre-exec helper has not launched a command subtree.
                        os.kill(pid, signal.SIGKILL)
                        killed = True
                except ProcessLookupError:
                    pass
        except ProcessLookupError:
            pass
        finally:
            for pid in reversed(stopped):
                try:
                    os.kill(pid, signal.SIGCONT)
                except ProcessLookupError:
                    pass
        return killed

    def _drain_invocations(self) -> None:
        # close owns _lock: no new helper may join a namespace being torn down.
        processes = tuple(getattr(self, "_invocations", ()))
        for process in processes:
            if self._terminate_invocation(process, deadline=self._cleanup_deadline):
                if not hasattr(self, "_invocation_stops"):
                    self._invocation_stops = {}
                self._invocation_stops[process] = {
                    "reason": "sandbox_close", "signal": int(signal.SIGKILL),
                    "monotonic": time.monotonic(),
                }
        deadline = self._cleanup_deadline
        for process in processes:
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired as exc:
                raise MiniSandboxRuntimeError(
                    "exec cleanup incomplete before namespace teardown: " +
                    str(self._process_diagnostics(process.pid))
                ) from exc
            self._invocations.discard(process)

    def _namespace_init_alive(self) -> bool:
        pid = getattr(self, "_host_pid", None)
        if pid is None:
            return False
        current = _pid_identity(pid)
        original = getattr(self, "_host_pid_identity", None)
        # Unknown identity retains the ledger but is never a signal target.
        return current is not None and (original is None or current == original)

    def close(self, *, reclaim_filesystem: bool = True) -> None:
        with self._lock:
            self._close_isolation()
            if reclaim_filesystem and not self._cleanup_complete:
                try:
                    shutil.rmtree(self.session_root)
                except FileNotFoundError:
                    if self.session_root.exists():
                        raise
                self._cleanup_complete = True

    def _close_isolation(self) -> None:
        # Serialize the entire cleanup, but allow retry after any failed stage.
        with self._lock:
            if getattr(self, "_isolation_closed", False):
                return
            self._closed = True
            # Both backend close attempts share the original budget. Reserve
            # five seconds of its 60s RPC deadline for transport/confirmation.
            if not hasattr(self, "_cleanup_deadline"):
                self._cleanup_deadline = time.monotonic() + 55.0
            self._drain_invocations()
            process = self._process
            if process is not None and process.poll() is None:
                if not getattr(self, "_termination_requested", False):
                    try:
                        if getattr(self, "_host_pidfd", None) is not None:
                            # Let PID 1 reap/exit before terminating unshare;
                            # unshare --kill-child would otherwise skip this.
                            signal.pidfd_send_signal(self._host_pidfd, signal.SIGTERM)
                        else:
                            os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    self._termination_requested = True
                    try:
                        process.wait(timeout=min(5, max(0, self._cleanup_deadline - time.monotonic())))
                    except subprocess.TimeoutExpired:
                        pass
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    # SIGKILL cannot interrupt a kernel IPC/shrinker wait.
                    # Use the remaining original budget rather than failing
                    # after two fixed waits while time is still available.
                    try:
                        process.wait(timeout=max(0, self._cleanup_deadline - time.monotonic()))
                    except subprocess.TimeoutExpired as exc:
                        diagnostics = self._cleanup_diagnostics("launcher_exit")
                        raise MiniSandboxRuntimeError(
                            f"launcher cleanup incomplete after SIGKILL: pid={process.pid}; "
                            f"diagnostics={diagnostics!r}", diagnostics=diagnostics,
                        ) from exc
            # A launcher exit alone does not prove namespace cleanup. Keep
            # resources pinned if namespace init still exists (including an
            # init blocked waiting for an externally parented zombie).
            while self._namespace_init_alive():
                remaining = self._cleanup_deadline - time.monotonic()
                if remaining <= 0:
                    diagnostics = self._cleanup_diagnostics("namespace_exit")
                    raise MiniSandboxRuntimeError(
                        "namespace init cleanup incomplete after launcher exit: " + str(diagnostics),
                        diagnostics=diagnostics,
                    )
                time.sleep(min(0.05, remaining))
            self._cgroup.close(timeout=max(0, min(5, self._cleanup_deadline - time.monotonic())))
            self._network_manager.release(self._network_lease)
            self._network_lease = None
            if getattr(self, "_host_pidfd", None) is not None:
                os.close(self._host_pidfd)
                self._host_pidfd = None
            self._isolation_closed = True



__all__ = [
    "BridgeNetworkManager",
    "CgroupV1Manager",
    "MiniSandboxRuntimeError",
    "MiniSandboxCommandInterrupted",
    "OciRootfsRuntime",
    "check_runtime_requirements",
]
