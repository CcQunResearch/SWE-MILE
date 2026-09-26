#!/usr/bin/env python3
"""Pinned, sandbox-side DeNovoSWE acceptance verifier.

The orchestration and ordering mirror AweAgent's DeNovoSWE evaluator at
``ed7865c57e821fd35f500e6ba1a078160da6a983``.  It intentionally has no
dependency on AweAgent so the benchmark image only needs Python and pytest.
"""

from __future__ import annotations

import argparse
import ast
import codecs
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

RESULTS = Path("/tmp/rllm/test_results.json")
REWARD = Path("/tmp/rllm/reward.json")
OUTPUT = Path("/tmp/rllm/denovoswe_verifier.log")
ANSI = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
STATUS_VALUES = ("PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL", "XPASS")
COMMAND_CLEANUP_CONFIRMED = True

# A dedicated interpreter owns only this command's descendants. Establishing
# subreaping in the caller would also adopt unrelated verifier/host children.
_COMMAND_RUNNER = r'''
import ctypes, json, os, signal, subprocess, sys, time
request = json.load(sys.stdin)
process = None
output = b""
timed_out = False
error = None
cleanup_confirmed = False

def descendants():
    pending, seen, result = [os.getpid()], set(), []
    while pending:
        parent = pending.pop()
        try:
            children = open("/proc/{0}/task/{0}/children".format(parent)).read().split()
        except FileNotFoundError:
            continue
        for raw in children:
            pid = int(raw)
            if pid in seen:
                continue
            seen.add(pid)
            try:
                fields = open("/proc/%d/stat" % pid).read().rsplit(") ", 1)[1].split()
            except FileNotFoundError:
                continue
            result.append((pid, fields[19], fields[0]))
            pending.append(pid)
    return result

def interrupted(signum, frame):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    raise InterruptedError("verifier command interrupted")

try:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot establish verifier child ownership")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    process = subprocess.Popen(request["command"], cwd=request["cwd"], env=request["env"],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               start_new_session=True)
    try:
        output, _ = process.communicate(timeout=request["timeout"])
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        output = exc.output or b""
except BaseException as exc:
    error = type(exc).__name__ + ": " + str(exc)
finally:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    started = time.monotonic()
    deadline = started + 5.5
    drained = process is None or process.stdout.closed
    try:
        while True:
            if process is not None:
                process.poll()
            remaining = descendants()
            for pid, identity, state in remaining:
                if state == "Z" and (process is None or pid != process.pid):
                    try:
                        os.waitpid(pid, os.WNOHANG)
                    except ChildProcessError:
                        pass
                    continue
                fd = None
                try:
                    if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
                        fd = os.pidfd_open(pid)
                    current = open("/proc/%d/stat" % pid).read().rsplit(") ", 1)[1].split()[19]
                    if current != identity:
                        continue
                    sig = signal.SIGTERM if time.monotonic() - started < .5 else signal.SIGKILL
                    if fd is not None:
                        signal.pidfd_send_signal(fd, sig)
                    else:
                        os.kill(pid, sig)
                except (FileNotFoundError, ProcessLookupError):
                    pass
                finally:
                    if fd is not None:
                        os.close(fd)
            left = deadline - time.monotonic()
            if left <= 0:
                break
            if process is not None and not drained:
                try:
                    output, _ = process.communicate(timeout=min(.05, left))
                    drained = True
                except subprocess.TimeoutExpired as exc:
                    output = exc.output or output
            if drained and not descendants():
                cleanup_confirmed = True
                break
            time.sleep(min(.01, max(0, deadline - time.monotonic())))
    except BaseException as exc:
        error = type(exc).__name__ + ": " + str(exc)
result = {"stdout": output.decode("utf-8", "replace"), "returncode": process.returncode if process else None,
          "timed_out": timed_out, "error": error, "cleanup_confirmed": cleanup_confirmed}
print(json.dumps(result))
'''


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True))
    os.replace(temporary, path)


def run(
    command: list[str],
    *,
    cwd: Path,
    timeout: int,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    global COMMAND_CLEANUP_CONFIRMED
    COMMAND_CLEANUP_CONFIRMED = False
    # The payload travels over stdin, never through a possibly huge argv.
    try:
        execution = subprocess.run(
            [sys.executable, "-c", _COMMAND_RUNNER],
            input=json.dumps({"command": command, "cwd": str(cwd), "timeout": timeout, "env": env}),
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout + 7,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("verifier command supervisor did not confirm cleanup") from exc
    result = json.loads(execution.stdout)
    COMMAND_CLEANUP_CONFIRMED = result.get("cleanup_confirmed") is True
    if execution.returncode != 0 or not COMMAND_CLEANUP_CONFIRMED or result.get("error"):
        raise RuntimeError("verifier command cleanup failed: " + str(result.get("error") or execution.stderr)[-2000:])
    with OUTPUT.open("a", encoding="utf-8") as handle:
        handle.write("\n$ " + " ".join(command) + "\n")
        handle.write(result["stdout"])
    if result["timed_out"]:
        raise subprocess.TimeoutExpired(command, timeout, output=result["stdout"])
    return subprocess.CompletedProcess(command, result["returncode"], result["stdout"])


def safe_relative(raw: str) -> Path:
    pure = PurePosixPath(raw)
    if not pure.parts or pure.is_absolute() or ".." in pure.parts or "\0" in raw:
        raise RuntimeError(f"unsafe verifier path: {raw!r}")
    return Path(*pure.parts)


def unlink_if_present(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def clean_agent_tests(workdir: Path) -> None:
    names = {"test", "tests", "testsuite", "testsuites", "testing", "test_suite"}
    directories = sorted(
        (path for path in workdir.rglob("*") if path.is_dir() and path.name.casefold() in names),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for path in directories:
        if path.is_symlink():
            unlink_if_present(path)
        else:
            shutil.rmtree(path, ignore_errors=True)
    root_patterns = (
        "test_*.py",
        "*_test.py",
        "*_tests.py",
        "conftest.py",
        "verify_*.py",
        "check_*.py",
    )
    for pattern in root_patterns:
        for path in workdir.glob(pattern):
            unlink_if_present(path)
    for cache_name in (".pytest_cache", "__pycache__"):
        for path in workdir.rglob(cache_name):
            shutil.rmtree(path, ignore_errors=True)


def files_added_by_patch(patch: str) -> list[Path]:
    """Include header-only empty additions, which have no /dev/null hunk."""
    lines = patch.splitlines()
    result: list[Path] = []
    header = ""

    def add(raw: str) -> None:
        if raw.startswith('"') and raw.endswith('"'):
            # Git quotes paths with C escapes, including octal UTF-8 bytes.
            raw = os.fsdecode(codecs.escape_decode(raw[1:-1].encode("utf-8"))[0])
        if not raw.startswith("b/"):
            raise RuntimeError(f"unsupported added-file patch path: {raw!r}")
        relative = safe_relative(raw[2:])
        if relative not in result:
            result.append(relative)

    for index, line in enumerate(lines):
        if line.startswith("diff --git "):
            header = line
        elif line.startswith("new file mode "):
            # New files have identical source/destination names. Matching both
            # avoids splitting an unquoted filename containing spaces or b/.
            match = re.fullmatch(
                r'diff --git (?:a/(?P<plain>.+) b/(?P=plain)|'
                r'"a/(?P<quoted>(?:[^"\\]|\\.)*)" "b/(?P=quoted)")',
                header,
            )
            if match is None:
                raise RuntimeError(f"unsupported added-file diff header: {header!r}")
            if match.group("plain") is not None:
                add("b/" + match.group("plain"))
            else:
                add('"b/' + match.group("quoted") + '"')
        elif line == "--- /dev/null" and index + 1 < len(lines):
            destination = lines[index + 1]
            if destination.startswith("+++ "):
                add(destination[4:].split("\t", 1)[0])
    return result


def prepare_test_patch(workdir: Path, patch_path: Path) -> None:
    patch = patch_path.read_text(encoding="utf-8")
    if not patch.strip():
        return
    for relative in files_added_by_patch(patch):
        target = workdir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            unlink_if_present(target)
    checked = run(
        ["git", "apply", "--check", "--recount", str(patch_path)],
        cwd=workdir,
        timeout=120,
    )
    if checked.returncode != 0:
        raise RuntimeError(
            f"test_patch failed git apply --check (exit_code={checked.returncode}): "
            f"{(checked.stdout or '')[-1000:]}"
        )
    applied = run(
        ["git", "apply", "--recount", "--whitespace=nowarn", str(patch_path)],
        cwd=workdir,
        timeout=120,
    )
    if applied.returncode != 0:
        raise RuntimeError(
            f"test_patch failed to apply (exit_code={applied.returncode}): "
            f"{(applied.stdout or '')[-1000:]}"
        )


def _node_matches(stack: list[ast.AST], node: ast.AST, parts: list[str]) -> bool:
    wanted = [part.split("[", 1)[0] for part in parts]
    names = [
        item.name
        for item in [*stack, node]
        if isinstance(item, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    return len(names) >= len(wanted) and names[-len(wanted) :] == wanted


def remove_failed_tests(workdir: Path, failed: list[str]) -> None:
    grouped: dict[Path, list[list[str]]] = {}
    for node_id in failed:
        parts = node_id.split("::")
        if len(parts) >= 2:
            grouped.setdefault(safe_relative(parts[0]), []).append(parts[1:])
    for relative, targets in grouped.items():
        path = workdir / relative
        if not path.is_file():
            continue
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        ranges: list[tuple[int, int]] = []

        def visit(
            node: ast.AST,
            stack: list[ast.AST],
            matched_targets: list[list[str]] = targets,
            deletion_ranges: list[tuple[int, int]] = ranges,
        ) -> None:
            removable = isinstance(
                node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
            ) and any(
                _node_matches(stack, node, target) for target in matched_targets
            )
            if removable:
                decorators = getattr(node, "decorator_list", ())
                start = min(
                    [node.lineno, *[item.lineno for item in decorators]],
                )
                deletion_ranges.append(
                    (start, getattr(node, "end_lineno", node.lineno))
                )
                return
            next_stack = (
                [*stack, node]
                if isinstance(
                    node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
                )
                else stack
            )
            for child in ast.iter_child_nodes(node):
                visit(child, next_stack)

        visit(tree, [])
        if not ranges:
            continue
        lines = source.splitlines(keepends=True)
        for start, end in sorted(ranges, reverse=True):
            del lines[start - 1 : end]
        path.write_text("".join(lines), encoding="utf-8")


def extract_binary_archive(workdir: Path, archive: Path) -> None:
    if not archive.is_file():
        return
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            safe_relative(member.name)
            if member.isdev() or member.isfifo():
                raise RuntimeError(f"unsafe fixture archive member: {member.name!r}")
            if member.issym() or member.islnk():
                safe_relative(member.linkname)
        try:
            handle.extractall(workdir, members=members, filter="data")
        except TypeError:  # Python 3.10 compatibility after manual validation.
            handle.extractall(workdir, members=members)


def uninstall_target(workdir: Path, names: list[str]) -> None:
    candidates: list[Path] = []
    for root in (Path("/usr/bin"), Path("/usr/local/bin"), Path("/opt")):
        if root.is_dir():
            candidates.extend(root.glob("python*"))
    seen: set[str] = set()
    interpreters: list[str] = []
    for path in [Path(shutil.which("python3") or ""), *candidates]:
        raw = str(path)
        if not raw or raw in seen or not path.is_file() or not os.access(path, os.X_OK):
            continue
        if not re.fullmatch(r"python(?:2|3)?(?:\.\d+)?", path.name):
            continue
        seen.add(raw)
        interpreters.append(raw)
    for name in dict.fromkeys(item for item in names if item):
        for interpreter in interpreters:
            try:
                run(
                    [interpreter, "-m", "pip", "uninstall", "-y", name],
                    cwd=workdir,
                    timeout=120,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass


def collect_ids(workdir: Path, test_file: str) -> set[str] | None:
    result = run(
        [
            "python3",
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "--no-header",
            "-o",
            "addopts=",
            "--rootdir=.",
            test_file,
        ],
        cwd=workdir,
        timeout=180,
    )
    ids = {
        line.strip()
        for line in ANSI.sub("", result.stdout or "").splitlines()
        if "::" in line and not line.lstrip().startswith(("ERROR", "WARNING", "="))
    }
    return ids if ids or result.returncode in {0, 5} else None


def parse_statuses(output: str, expected: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    # Match against the authoritative node ids instead of tokenising on
    # whitespace: parametrized pytest ids are allowed to contain spaces.
    expected_by_length = sorted(set(expected), key=len, reverse=True)
    for raw in ANSI.sub("", output).splitlines():
        line = raw.strip()
        matched = False

        # Normal verbose progress line: ``path::node PASSED [ 50%]``.
        for node in expected_by_length:
            if not line.startswith(node):
                continue
            tail = line[len(node) :]
            if not tail or not tail[0].isspace():
                continue
            status = tail.lstrip().split(None, 1)[0]
            if status in STATUS_VALUES:
                result[node] = status
                matched = True
                break
        if matched:
            continue

        # ``-rA`` summary line: ``PASSED path::node - optional detail``.
        for status in STATUS_VALUES:
            prefix = status + " "
            if not line.startswith(prefix):
                continue
            tail = line[len(prefix) :]
            for node in expected_by_length:
                if tail == node or tail.startswith(node + " - "):
                    result[node] = status
                    matched = True
                    break
            if matched:
                break
    return result


class TestExecutionTimeout(Exception):
    """A bounded benchmark test execution, distinct from setup failure."""

    def __init__(self, total: int, statuses: dict[str, str], error: Exception):
        self.total = total
        self.statuses = dict(statuses)
        super().__init__(str(error))


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    instance = json.loads(Path(args.instance).read_text(encoding="utf-8"))
    workdir = Path(instance["workdir"])
    passed_ptp = [str(item) for item in instance["passed_ptp"]]
    if not workdir.is_dir() or not (workdir / ".git").is_dir():
        raise RuntimeError(f"invalid verifier workdir: {workdir}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("", encoding="utf-8")

    clean_agent_tests(workdir)
    prepare_test_patch(workdir, Path(args.test_patch))
    extract_binary_archive(workdir, Path(args.binary_archive))
    remove_failed_tests(workdir, [str(item) for item in instance.get("failed_ptp", [])])
    uninstall_target(
        workdir,
        [
            str(instance.get("pypi_name") or ""),
            *[str(item) for item in instance.get("pypi_name_candidates", [])],
        ],
    )
    installed = run(
        ["python3", "-m", "pip", "install", "-e", "."],
        cwd=workdir,
        timeout=int(os.environ.get("RLLM_DENOVO_INSTALL_TIMEOUT", "1200")),
    )
    statuses = {node: "NOT_RUN" for node in passed_ptp}
    diagnostics: dict[str, object] = {"install_returncode": installed.returncode}
    if installed.returncode == 0:
        groups: dict[str, list[str]] = {}
        for node in passed_ptp:
            groups.setdefault(node.split("::", 1)[0], []).append(node)
        for test_file, expected in groups.items():
            try:
                collected = collect_ids(workdir, test_file)
            except subprocess.TimeoutExpired as exc:
                raise TestExecutionTimeout(len(passed_ptp), statuses, exc) from exc
            runnable = expected if collected is None else [node for node in expected if node in collected]
            if not runnable:
                continue
            try:
                result = run(
                    [
                        "python3",
                        "-m",
                        "pytest",
                        "-vv",
                        "-rA",
                        "--tb=short",
                        "-o",
                        "addopts=",
                        "--rootdir=.",
                        *runnable,
                    ],
                    cwd=workdir,
                    timeout=int(os.environ.get("RLLM_DENOVO_TEST_FILE_TIMEOUT", "600")),
                )
            except subprocess.TimeoutExpired as exc:
                raise TestExecutionTimeout(len(passed_ptp), statuses, exc) from exc
            statuses.update(parse_statuses(result.stdout or "", runnable))
            for node in runnable:
                if statuses[node] == "NOT_RUN" and result.returncode != 0:
                    statuses[node] = "ERROR"
    passed_count = sum(status == "PASSED" for status in statuses.values())
    failed_count = sum(status in {"FAILED", "SKIPPED", "XFAIL", "XPASS"} for status in statuses.values())
    error_count = sum(status in {"ERROR", "NOT_RUN"} for status in statuses.values())
    total_count = len(passed_ptp)
    pass_rate = passed_count / total_count
    return {
        "schema_version": 1,
        "parser": "denovoswe_official_v1",
        "test_results": statuses,
        "passed_count": passed_count,
        "failed_count": failed_count,
        "error_count": error_count,
        "total_count": total_count,
        "pass_rate": pass_rate,
        "is_correct": passed_count == total_count and failed_count == 0 and error_count == 0,
        "outcome_source": "shadow_verifier",
        "verifier_profile": "denovoswe_official",
        "diagnostics": diagnostics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)
    parser.add_argument("--test-patch", required=True)
    parser.add_argument("--binary-archive", required=True)
    args = parser.parse_args()
    try:
        outcome = evaluate(args)
    except TestExecutionTimeout as exc:
        # Match the primary verifier's existing all-zero timeout policy. The
        # partial vector is audit-only, never a fabricated completed probe.
        outcome = {
            "schema_version": 1, "parser": "denovoswe_official_v1",
            "test_results": {}, "passed_count": 0, "failed_count": 0,
            "error_count": exc.total, "total_count": exc.total,
            "pass_rate": 0.0, "is_correct": False,
            "outcome_source": "shadow_verifier",
            "verifier_profile": "denovoswe_official",
            "verifier_status": "timeout", "partial_test_results": exc.statuses,
            "timeout_error": str(exc)[-2000:],
        }
    except BaseException as exc:
        outcome = {
            "schema_version": 1,
            "parser": "denovoswe_official_v1",
            "test_results": {},
            "passed_count": 0,
            "failed_count": 0,
            "error_count": 1,
            "total_count": 0,
            "pass_rate": 0.0,
            "is_correct": False,
            "outcome_source": "shadow_verifier",
            "verifier_profile": "denovoswe_official",
            "infrastructure_error": f"{type(exc).__name__}: {exc}",
        }
    outcome["process_cleanup_confirmed"] = COMMAND_CLEANUP_CONFIRMED
    atomic_json(RESULTS, outcome)
    atomic_json(
        REWARD,
        {
            "reward": outcome["pass_rate"],
            "is_correct": outcome["is_correct"],
            "signals": {"acceptance_pass_rate": outcome["pass_rate"]},
            "metadata": outcome,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
