"""Evaluators specific to repository-generation benchmarks."""

from __future__ import annotations

import base64
import json
import logging
import math
import posixpath
import re
import shlex
import tarfile
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from rllm.eval.repo_generation_status import command_failure, infrastructure_metadata
from rllm.eval.types import EvalOutput, Signal
from rllm.sandbox.protocol import Sandbox
from rllm.sandbox.structured_exec import frame_structured_command, parse_structured_command_output
from rllm.types import Episode, RolloutInfrastructureError, Task

logger = logging.getLogger(__name__)

_PACKAGE_FILES = (
    "setup.py",
    "pyproject.toml",
    "setup.cfg",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-test.txt",
    "tox.ini",
    "pytest.ini",
    "poetry.lock",
    "Pipfile",
    "Pipfile.lock",
    "environment.yml",
    "conda-env.yaml",
    "manifest.in",
    "MANIFEST.in",
)
_PASSED_RE = re.compile(r"(\d+) passed")
_FAILED_RE = re.compile(r"(\d+) failed")
_ERROR_RE = re.compile(r"(\d+) error")
_PYTEST_SUMMARY_RE = re.compile(
    r"=+\s*(?:.*\d+\s+(?:passed|failed|error).*)\s+in\s+[\d.]+s.*=+"
)
_PYTEST_FILE_RE = re.compile(r"(?:^|/)(?:test_[^/]*\.py|[^/]*_test\.py|conftest\.py)$")


class UnsafeWorkspaceArchive(ValueError):
    """The agent workspace cannot safely be overlaid into an eval image."""


def _safe_relative_path(value: Any, *, field: str) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    pure = PurePosixPath(raw)
    if not raw or pure.is_absolute() or ".." in pure.parts or raw in {".", "./"}:
        raise ValueError(f"{field} contains unsafe path {value!r}")
    return str(pure)


def _safe_workdir(value: Any) -> str:
    raw = str(value or "/workspace").strip()
    pure = PurePosixPath(raw)
    if not pure.is_absolute() or ".." in pure.parts or raw == "/":
        raise ValueError(f"unsafe NL2Repo workdir {value!r}")
    return str(pure)


def _string_list(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{field} must be a non-empty string list")
    return [item.strip() for item in value]


def analyze_nl2repo_pytest_results(
    command_results: list[dict[str, Any]],
    total_test_cases: int,
) -> dict[str, int | float]:
    """Port the official loose pytest-summary parser used by AweAgent."""

    passed = failed = errors = 0
    for result in command_results:
        if "pytest" not in str(result.get("command", "")).lower():
            continue
        output = str(result.get("output", ""))
        summary_line = next(
            (
                line
                for line in reversed(output.splitlines())
                if _PYTEST_SUMMARY_RE.search(line)
            ),
            None,
        )
        lines = [summary_line] if summary_line is not None else output.splitlines()
        for line in lines:
            match = _PASSED_RE.search(line)
            if match:
                passed += int(match.group(1))
            match = _FAILED_RE.search(line)
            if match:
                failed += int(match.group(1))
            match = _ERROR_RE.search(line)
            if match:
                errors += int(match.group(1))
    rate = min(passed / total_test_cases, 1.0) if total_test_cases > 0 else 0.0
    return {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "total": total_test_cases,
        "success_rate": rate,
    }


class NL2RepoFreshSandboxEvaluator:
    """Transfer the whole agent workspace into a pristine evaluation image."""

    def __init__(
        self,
        primary_sandbox: Sandbox,
        *,
        timeout: float = 1800.0,
        archive_chunk_bytes: int = 768 * 1024,
        max_archive_bytes: int = 2 * 1024**3,
        max_unpacked_bytes: int = 16 * 1024**3,
        max_members: int = 250_000,
    ) -> None:
        self.primary_sandbox = primary_sandbox
        self.timeout = float(timeout)
        self.archive_chunk_bytes = int(archive_chunk_bytes)
        self.max_archive_bytes = int(max_archive_bytes)
        self.max_unpacked_bytes = int(max_unpacked_bytes)
        self.max_members = int(max_members)
        self._fresh_sandbox_factory: Callable[[], Sandbox] | None = None

    def configure_fresh_sandbox(self, factory: Callable[[], Sandbox]) -> None:
        self._fresh_sandbox_factory = factory

    @staticmethod
    def _load_contract(task: Task) -> dict[str, Any]:
        path = task.task_dir / "tests" / "instance.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid NL2Repo verifier contract {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise RuntimeError(f"invalid NL2Repo verifier contract {path}: expected object")
        commands = _string_list(raw.get("verify_cmd"), field="verify_cmd")
        files = [
            _safe_relative_path(value, field="verify_files")
            for value in _string_list(raw.get("verify_files"), field="verify_files")
        ]
        count = raw.get("test_cases_num")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise RuntimeError("test_cases_num must be a positive integer")
        image = str(raw.get("evaluation_image") or "").strip()
        task_image = str(
            task.metadata.get("docker_image")
            or (task.metadata.get("environment") or {}).get("docker_image")
            or ""
        ).strip()
        if not image or image != task_image:
            raise RuntimeError(
                "NL2Repo evaluation_image does not match the task sandbox image"
            )
        return {
            **raw,
            "verify_cmd": commands,
            "verify_files": files,
            "test_cases_num": count,
            "workdir": _safe_workdir(raw.get("workdir")),
        }

    @staticmethod
    def _archive_member_name(value: str) -> str:
        normalized = posixpath.normpath(value.replace("\\", "/"))
        while normalized.startswith("./"):
            normalized = normalized[2:]
        if normalized in {"", "."}:
            return ""
        pure = PurePosixPath(normalized)
        if pure.is_absolute() or ".." in pure.parts:
            raise UnsafeWorkspaceArchive(f"unsafe archive path {value!r}")
        if any(part == ".git" for part in pure.parts):
            raise UnsafeWorkspaceArchive("workspace archive contains Git metadata")
        return str(pure)

    @staticmethod
    def _path_overlaps(left: str, right: str) -> bool:
        return left == right or left.startswith(right + "/") or right.startswith(left + "/")

    def _validate_archive(self, path: Path, verify_files: list[str]) -> dict[str, int]:
        try:
            with tarfile.open(path, "r:gz") as archive:
                members = archive.getmembers()
        except (OSError, tarfile.TarError) as exc:
            raise UnsafeWorkspaceArchive(f"invalid gzip tar archive: {exc}") from exc
        if len(members) > self.max_members:
            raise UnsafeWorkspaceArchive(
                f"workspace archive has too many members: {len(members)}"
            )
        names: set[str] = set()
        normalized_members: list[tuple[tarfile.TarInfo, str]] = []
        unpacked_bytes = 0
        for member in members:
            name = self._archive_member_name(member.name)
            if not name:
                continue
            if name in names:
                raise UnsafeWorkspaceArchive(f"duplicate archive path {name!r}")
            names.add(name)
            normalized_members.append((member, name))
            if member.isreg():
                unpacked_bytes += max(0, int(member.size))
            elif not (member.isdir() or member.issym() or member.islnk()):
                raise UnsafeWorkspaceArchive(
                    f"unsupported special archive member {name!r}"
                )
            if unpacked_bytes > self.max_unpacked_bytes:
                raise UnsafeWorkspaceArchive("workspace archive expands beyond the size limit")

        def protected(value: str) -> bool:
            basename = PurePosixPath(value).name
            return (
                basename in _PACKAGE_FILES
                or bool(_PYTEST_FILE_RE.search(value))
                or any(self._path_overlaps(value, item) for item in verify_files)
            )

        for member, name in normalized_members:
            if not (member.issym() or member.islnk()):
                continue
            link = str(member.linkname or "").replace("\\", "/")
            if not link or PurePosixPath(link).is_absolute():
                raise UnsafeWorkspaceArchive(f"unsafe archive link {name!r} -> {link!r}")
            base = "" if member.islnk() else posixpath.dirname(name)
            target = self._archive_member_name(posixpath.join(base, link))
            if (
                not target
                or target not in names
                or protected(name)
                or protected(target)
            ):
                raise UnsafeWorkspaceArchive(f"unsafe archive link {name!r} -> {link!r}")
        return {
            "members": len(normalized_members),
            "unpacked_bytes": unpacked_bytes,
        }

    def _export_primary_workspace(self, remote_path: str, local_path: Path) -> int:
        chunk = self.archive_chunk_bytes
        command = (
            "set -e; "
            f"tar --format=posix --exclude='./.git' --exclude='./.git/*' "
            "--exclude='*/.git' --exclude='*/.git/*' "
            f"-czf {shlex.quote(remote_path)} -C /workspace .; "
            f"stat -c %s {shlex.quote(remote_path)}"
        )
        raw_size = self._export_command(command)
        match = re.search(r"(\d+)\s*$", str(raw_size))
        if match is None:
            raise RuntimeError("primary workspace export did not report an archive size")
        size = int(match.group(1))
        if size <= 0 or size > self.max_archive_bytes:
            raise UnsafeWorkspaceArchive(
                f"compressed workspace archive size {size} is outside allowed limits"
            )
        with local_path.open("wb") as handle:
            for index, offset in enumerate(range(0, size, chunk)):
                expected = min(chunk, size - offset)
                encoded = self._export_command(
                    "set -o pipefail; "
                    f"dd if={shlex.quote(remote_path)} bs={chunk} skip={index} "
                    "count=1 status=none | base64 -w0",
                )
                try:
                    payload = base64.b64decode(str(encoded).strip(), validate=True)
                except (ValueError, TypeError) as exc:
                    raise RuntimeError(
                        f"invalid base64 workspace chunk at offset {offset}"
                    ) from exc
                if len(payload) != expected:
                    raise RuntimeError(
                        f"short workspace chunk at offset {offset}: "
                        f"expected {expected}, got {len(payload)}"
                    )
                handle.write(payload)
        return size

    def _export_command(self, command: str) -> str:
        # Login shells can print banners or service diagnostics (e.g. Xvfb).
        # Only decode the command's bound stdout, never the mixed RPC output.
        framed = frame_structured_command(command)
        raw = self.primary_sandbox.exec(framed.command, timeout=max(60.0, self.timeout))
        result = parse_structured_command_output(raw, framed.nonce)
        if result.exit_code != 0:
            raise RuntimeError(
                f"workspace export command exited {result.exit_code}: {result.stderr[-1000:]}"
            )
        return result.stdout

    @staticmethod
    def _cleanup_staging_command(staging: str, verify_files: list[str]) -> str:
        package_expr = " -o ".join(
            f"-name {shlex.quote(name)}" for name in _PACKAGE_FILES
        )
        commands = [
            f"find -P {shlex.quote(staging)} "
            r"\( -type f -o -type l \) "
            f"\\( {package_expr} \\) -delete",
            f"find -P {shlex.quote(staging)} -type f "
            r"\( -name 'test_*.py' -o -name '*_test.py' -o -name 'conftest.py' \) "
            "-delete",
        ]
        commands.extend(
            f"rm -rf -- {shlex.quote(staging + '/' + value)}"
            for value in verify_files
        )
        return " && ".join(commands)

    def _stage_in_fresh(
        self,
        sandbox: Sandbox,
        local_archive: Path,
        contract: dict[str, Any],
    ) -> None:
        archive_path = "/tmp/rllm-nl2repo-workspace.tar.gz"
        staging = "/tmp/rllm-agent-workspace"
        sandbox.upload_file(str(local_archive), archive_path)
        cleanup = self._cleanup_staging_command(staging, contract["verify_files"])
        workdir = contract["workdir"]
        command = (
            "set -e; "
            f"rm -rf -- {shlex.quote(staging)}; "
            f"mkdir -p -- {shlex.quote(staging)} {shlex.quote(workdir)}; "
            f"tar --no-same-owner --no-same-permissions -xzf {shlex.quote(archive_path)} "
            f"-C {shlex.quote(staging)}; "
            f"{cleanup}; "
            f"cp -a {shlex.quote(staging)}/. {shlex.quote(workdir)}/"
        )
        # Pristine images can retain a non-root-owned /workspace. Primary
        # setup normalizes it, but must never run in this verifier sandbox:
        # it deletes the golden tests. Use filesystem-only setup privileges
        # just for the validated overlay, then run tests capability-free.
        setup_exec = getattr(sandbox, "exec_setup", None)
        executor = setup_exec if callable(setup_exec) else sandbox.exec
        executor(command, timeout=self.timeout, user="root")

    def _run_verifier_commands(
        self,
        sandbox: Sandbox,
        contract: dict[str, Any],
        *,
        trusted_install: bool = True,
        proxy_url: str = "",
    ) -> list[dict[str, Any]]:
        from rllm.sandbox.repo_generation_environment import repo_generation_environment_exports
        from rllm.sandbox.nl2repo_environment import compatible_nl2repo_command

        prefix = repo_generation_environment_exports(proxy_url)
        results: list[dict[str, Any]] = []
        workdir = contract["workdir"]
        command_timeout = max(1, int(math.ceil(self.timeout)))
        for index, command in enumerate(contract["verify_cmd"]):
            executed_command = compatible_nl2repo_command(command, contract) if trusted_install else command
            marker = f"__RLLM_NL2REPO_RC_{uuid.uuid4().hex}__="
            inner = (
                f"cd {shlex.quote(workdir)} || exit 125; "
                f"export PYTHONPATH={shlex.quote(workdir)}:${{PYTHONPATH-}}; "
                f"{prefix}set +e; "
                f"timeout -s TERM -k 10 {command_timeout} "
                f"bash -c {shlex.quote(executed_command)} 2>&1; "
                "_rllm_rc=$?; "
                f"printf '\\n{marker}%s\\n' \"$_rllm_rc\"; exit 0"
            )
            output = sandbox.exec(
                f"/bin/bash -o pipefail -c {shlex.quote(inner)}",
                timeout=self.timeout + 30,
            )
            before, separator, after = str(output).rpartition(marker)
            if not separator:
                raise RuntimeError(
                    f"NL2Repo verifier command {index} returned no status marker"
                )
            rc_match = re.match(r"\s*(-?\d+)", after)
            if rc_match is None:
                raise RuntimeError(
                    f"NL2Repo verifier command {index} returned an invalid status marker"
                )
            exit_code = int(rc_match.group(1))
            status = command_failure(command, exit_code, before, trusted_install=trusted_install)
            results.append(
                {
                    "command": command,
                    "executed_command": executed_command,
                    "exit_code": exit_code,
                    **status,
                    "output": before.rstrip(),
                }
            )
        return results

    def evaluate(self, task: Task, episode: Episode) -> EvalOutput:  # noqa: ARG002
        if self._fresh_sandbox_factory is None:
            raise RolloutInfrastructureError(
                "nl2repo_fresh_sandbox_unconfigured",
                "NL2Repo fresh-sandbox factory was not configured",
                retryable=False,
            )
        try:
            contract = self._load_contract(task)
        except Exception as exc:
            raise RolloutInfrastructureError(
                "nl2repo_verifier_contract_invalid",
                f"NL2Repo verifier contract is invalid: {exc}",
                retryable=False,
            ) from exc
        remote_archive = f"/tmp/rllm-workspace-{uuid.uuid4().hex}.tar.gz"
        temporary = tempfile.NamedTemporaryFile(
            prefix="rllm-nl2repo-", suffix=".tar.gz", delete=False
        )
        local_archive = Path(temporary.name)
        temporary.close()
        fresh: Sandbox | None = None
        archive_size = 0
        archive_info: dict[str, int] = {}
        try:
            try:
                archive_size = self._export_primary_workspace(
                    remote_archive, local_archive
                )
                archive_info = self._validate_archive(
                    local_archive, contract["verify_files"]
                )
            except UnsafeWorkspaceArchive as exc:
                return EvalOutput(
                    reward=0.0,
                    is_correct=False,
                    metadata={
                        "verifier_status": "invalid_submission",
                        "error": str(exc),
                        "verifier_profile": "nl2repo-fresh-sandbox",
                    },
                )
            except RolloutInfrastructureError:
                raise
            except Exception as exc:
                raise RolloutInfrastructureError(
                    "nl2repo_workspace_export_failed",
                    f"NL2Repo primary workspace export failed: {exc}",
                    retryable=True,
                ) from exc
            try:
                fresh = self._fresh_sandbox_factory()
                from rllm.sandbox.nl2repo_environment import prepare_nl2repo_image
                image_preparation = prepare_nl2repo_image(fresh, contract)
                self._stage_in_fresh(fresh, local_archive, contract)
                proxy_url = (task.metadata.get("rllm") or {}).get("repo_generation_proxy_url", "")
                command_results = self._run_verifier_commands(fresh, contract, **({"proxy_url": proxy_url} if proxy_url else {}))
            except RolloutInfrastructureError:
                raise
            except Exception as exc:
                raise RolloutInfrastructureError(
                    "nl2repo_fresh_sandbox_failed",
                    f"NL2Repo fresh-sandbox evaluation failed: {exc}",
                    retryable=True,
                ) from exc

            failures = [result for result in command_results if result.get("status") != "completed"]
            infrastructure = next((r for r in failures if r["status"] == "infrastructure_failure"), None)
            timed_out = any(r["status"] == "timeout" for r in failures)
            status = ("infrastructure_failure" if infrastructure else "timeout" if timed_out
                      else failures[0]["status"] if failures else "completed")
            parsed = analyze_nl2repo_pytest_results(
                command_results, contract["test_cases_num"]
            )
            passed = int(parsed["passed"])
            failed = int(parsed["failed"])
            errors = int(parsed["errors"])
            actual_total = passed + failed + errors
            count_mismatch = actual_total != contract["test_cases_num"]
            pass_rate = float(parsed["success_rate"])
            is_correct = (
                not failures
                and passed >= contract["test_cases_num"]
                and failed == 0
                and errors == 0
            )
            if infrastructure or timed_out:
                pass_rate = 0.0
                is_correct = False
            verifier_outcome = {
                "passed_count": passed,
                "total_count": contract["test_cases_num"],
                "pass_rate": pass_rate,
                "outcome_source": "fresh_sandbox_verifier",
                "verifier_profile": "nl2repo-fresh-sandbox",
            }
            return EvalOutput(
                reward=pass_rate,
                is_correct=is_correct,
                signals=[
                    Signal(name="pass_rate", value=pass_rate),
                    Signal(name="full_success", value=float(is_correct)),
                ],
                metadata={
                    "verifier_status": status,
                    "verifier_diagnostics": {"image_preparation": image_preparation},
                    "verifier_timed_out": timed_out,
                    "scoring_protocol": "aweagent_nl2repo_fixed_denominator_v2",
                    **({"infrastructure_failure": infrastructure_metadata(infrastructure, infrastructure["output"])}
                       if infrastructure else {}),
                    "verifier_profile": "nl2repo-fresh-sandbox",
                    "passed_count": passed,
                    "failed_count": failed,
                    "error_count": errors,
                    "expected_test_count": contract["test_cases_num"],
                    "actual_test_count": actual_total,
                    "count_mismatch": count_mismatch,
                    "count_mismatch_detail": (
                        f"expected {contract['test_cases_num']}, got {actual_total}"
                        if count_mismatch
                        else None
                    ),
                    "workspace_archive_bytes": archive_size,
                    "workspace_archive_members": archive_info.get("members", 0),
                    "workspace_unpacked_bytes": archive_info.get(
                        "unpacked_bytes", 0
                    ),
                    "fresh_sandbox_count": 1,
                    "command_results": [
                        {
                            "command": result["command"],
                            "executed_command": result.get("executed_command", result["command"]),
                            "exit_code": result["exit_code"],
                            "output": str(result["output"])[-4000:],
                            "status": result["status"],
                            "stage": result["stage"],
                            "reason": result.get("reason"),
                        }
                        for result in command_results
                    ],
                    "verifier_outcome": verifier_outcome,
                },
            )
        finally:
            try:
                self.primary_sandbox.exec(
                    f"rm -f -- {shlex.quote(remote_archive)}", timeout=30
                )
            except Exception:
                logger.debug("failed to remove primary NL2Repo archive", exc_info=True)
            if fresh is not None:
                try:
                    fresh.close()
                except Exception:
                    logger.exception("failed to close NL2Repo fresh sandbox")
            try:
                local_archive.unlink(missing_ok=True)
            except OSError:
                logger.debug("failed to remove local NL2Repo archive", exc_info=True)
