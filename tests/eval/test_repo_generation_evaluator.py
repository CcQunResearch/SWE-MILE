from __future__ import annotations

import base64
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path

import pytest

from rllm.eval._resolution import _detect_verifier, _resolve_evaluator
from rllm.eval.repo_generation import (
    NL2RepoFreshSandboxEvaluator,
    UnsafeWorkspaceArchive,
    analyze_nl2repo_pytest_results,
)
from rllm.types import Episode, RolloutInfrastructureError, Task


def _archive(entries: dict[str, bytes], *, unsafe_name: str | None = None) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, payload in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        if unsafe_name is not None:
            info = tarfile.TarInfo(unsafe_name)
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
    return buffer.getvalue()


class _PrimarySandbox:
    def __init__(self, artifact: bytes) -> None:
        self.artifact = artifact
        self.commands: list[str] = []
        self.uploads = 0

    def exec(self, command: str, timeout=None, user=None) -> str:  # noqa: ANN001, ARG002
        self.commands.append(command)
        if "tar --format=posix" in command:
            output = str(len(self.artifact))
        elif "dd if=" in command:
            block = int(re.search(r"bs=(\d+)", command).group(1))
            index = int(re.search(r"skip=(\d+)", command).group(1))
            payload = self.artifact[index * block : (index + 1) * block]
            output = base64.b64encode(payload).decode("ascii")
        elif command.startswith("rm -f --"):
            return ""
        else:
            raise AssertionError(f"unexpected primary command: {command}")
        nonce = re.search(r"__RLLM_STRUCTURED_BEGIN_([0-9a-f]+)__", command).group(1)
        envelope = {
            "exit_code": 0,
            "stdout": base64.b64encode(output.encode()).decode(),
            "stderr": "",
        }
        encoded = base64.b64encode(json.dumps(envelope).encode()).decode()
        return f"shell startup diagnostic\n__RLLM_STRUCTURED_BEGIN_{nonce}__{encoded}__RLLM_STRUCTURED_END_{nonce}__\n"

    def upload_file(self, *_args) -> None:
        self.uploads += 1
        raise AssertionError("golden verifier assets must never be uploaded primary")


class _FailingExportPrimarySandbox(_PrimarySandbox):
    def exec(self, command: str, timeout=None, user=None) -> str:  # noqa: ANN001, ARG002
        if "tar --format=posix" in command:
            self.commands.append(command)
            raise RuntimeError("primary provider unavailable")
        return super().exec(command, timeout=timeout, user=user)


class _FreshSandbox:
    def __init__(self, command_outputs: list[str]) -> None:
        self.command_outputs = list(command_outputs)
        self.commands: list[str] = []
        self.uploaded_archive: bytes | None = None
        self.close_calls = 0

    def upload_file(self, source: str, destination: str) -> None:
        assert destination == "/tmp/rllm-nl2repo-workspace.tar.gz"
        self.uploaded_archive = Path(source).read_bytes()

    def exec(self, command: str, timeout=None, user=None) -> str:  # noqa: ANN001, ARG002
        self.commands.append(command)
        if "Filesystem-only compatibility" in command:
            assert self.uploaded_archive is None, "image preparation must precede generated-source overlay"
            nonce = re.search(r"__RLLM_STRUCTURED_BEGIN_([0-9a-f]+)__", command).group(1)
            audit = {"schema_version": 1, "ok": True, "repairs": []}
            payload = {"exit_code": 0, "stdout": base64.b64encode(json.dumps(audit).encode()).decode(), "stderr": ""}
            return f"__RLLM_STRUCTURED_BEGIN_{nonce}__{base64.b64encode(json.dumps(payload).encode()).decode()}__RLLM_STRUCTURED_END_{nonce}__"
        marker = re.search(r"(__RLLM_NL2REPO_RC_[0-9a-f]+__=)", command)
        if marker is None:
            return ""
        output = self.command_outputs.pop(0)
        return f"{output}\n{marker.group(1)}0\n"

    def close(self) -> None:
        self.close_calls += 1


def test_nl2repo_export_frames_noisy_shell_output_and_multiple_chunks(tmp_path):
    source = tmp_path / "workspace"
    source.mkdir()
    (source / "binary").write_bytes(bytes(range(256)) * 20)

    class Shell:
        def exec(self, command, timeout=None):
            result = subprocess.run(
                ["bash", "-c", command.replace("-C /workspace .", f"-C {source} .")],
                capture_output=True, text=True, timeout=timeout, check=True,
            )
            # The transport merges stderr and login-shell output. Neither is
            # part of the binary archive, including Base64-looking banners.
            return "Xvfb startup warning\nYWJj\n" + result.stdout + "shell logout\n"

    evaluator = NL2RepoFreshSandboxEvaluator(Shell(), archive_chunk_bytes=96)
    archive = tmp_path / "received.tar.gz"
    size = evaluator._export_primary_workspace(str(tmp_path / "remote.tar.gz"), archive)
    assert size == archive.stat().st_size
    with tarfile.open(archive) as result:
        assert result.extractfile("./binary").read() == (source / "binary").read_bytes()


def test_nl2repo_export_does_not_accept_stderr_or_failed_pipeline(tmp_path):
    class Shell:
        def exec(self, command, timeout=None):
            return subprocess.run(
                ["bash", "-c", command], capture_output=True, text=True, check=True,
            ).stdout

    evaluator = NL2RepoFreshSandboxEvaluator(Shell())
    with pytest.raises(RuntimeError, match="workspace export command exited"):
        evaluator._export_command(f"set -o pipefail; dd if={tmp_path / 'missing'} status=none | base64 -w0")


def test_nl2repo_fresh_overlay_uses_only_trusted_filesystem_setup(tmp_path):
    task = _task(tmp_path)
    primary = _PrimarySandbox(_archive({"demo.py": b"answer = 42\n"}))

    class Fresh(_FreshSandbox):
        def __init__(self):
            super().__init__(["==== 4 passed in 0.1s ====", ""])
            self.setup_commands = []

        def exec_setup(self, command, timeout=None, user=None):
            assert user == "root"
            self.setup_commands.append(command)
            if "Filesystem-only compatibility" in command:
                return super().exec(command, timeout=timeout, user=user)
            assert "cp -a" in command and command.startswith("set -e;")
            assert "/tmp/rllm_setup.sh" not in command
            assert "pytest --" not in command

        def exec(self, command, timeout=None, user=None):
            assert "cp -a" not in command, "capability-free copy would fail for aiofiles"
            return super().exec(command, timeout=timeout, user=user)

    fresh = Fresh()
    evaluator = NL2RepoFreshSandboxEvaluator(primary)
    evaluator.configure_fresh_sandbox(lambda: fresh)
    result = evaluator.evaluate(task, Episode(id="e", task=task.id))
    assert result.reward == 1.0
    assert len(fresh.setup_commands) == 2
    assert len(fresh.commands) == 3


def _task(tmp_path: Path) -> Task:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (tests / "instance.json").write_text(
        json.dumps(
            {
                "instance_id": "demo",
                "evaluation_image": "example/nl2:latest",
                "workdir": "/workspace",
                "package_name": "demo",
                "verify_cmd": ["python -m pytest tests/test_a.py", "pytest -q"],
                "verify_files": ["tests/test_a.py"],
                "test_cases_num": 4,
            }
        ),
        encoding="utf-8",
    )
    return Task(
        id="demo",
        instruction="build it",
        dataset_dir=tmp_path,
        metadata={
            "docker_image": "example/nl2:latest",
            "environment": {"docker_image": "example/nl2:latest"},
            "rllm": {"verifier_kind": "nl2repo-fresh-sandbox"},
        },
    )


def test_nl2repo_fresh_evaluator_transfers_workspace_and_scores_fractionally(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path)
    artifact = _archive(
        {
            "pkg/core.py": b"value = 1\n",
            "setup.py": b"raise RuntimeError('must be stripped')\n",
            "tests/test_agent.py": b"def test_fake(): assert True\n",
            "tests/test_a.py": b"def test_fake(): assert True\n",
        }
    )
    primary = _PrimarySandbox(artifact)
    fresh = _FreshSandbox(
        [
            "======= 2 passed, 1 failed in 0.20s =======",
            "======= 1 passed in 0.10s =======",
        ]
    )
    evaluator = NL2RepoFreshSandboxEvaluator(primary)
    evaluator.configure_fresh_sandbox(lambda: fresh)

    result = evaluator.evaluate(task, Episode(id="e", task=task.id))

    assert result.reward == pytest.approx(0.75)
    assert result.is_correct is False
    assert result.metadata["passed_count"] == 3
    assert result.metadata["count_mismatch"] is False
    assert result.metadata["fresh_sandbox_count"] == 1
    assert result.metadata["verifier_outcome"]["pass_rate"] == pytest.approx(0.75)
    assert fresh.uploaded_archive == artifact
    assert fresh.close_calls == 1
    assert primary.uploads == 0
    assert any("setup.py" in command and "test_*.py" in command for command in fresh.commands)
    verifier_commands = [
        command for command in fresh.commands if "__RLLM_NL2REPO_RC_" in command
    ]
    assert len(verifier_commands) == 2
    assert all("timeout -s TERM -k 10 " in command for command in verifier_commands)
    assert all("--signal" not in command for command in verifier_commands)
    assert "tests/test_a.py" in verifier_commands[0]
    assert "pytest -q" in verifier_commands[1]
    assert primary.commands[-1].startswith("rm -f --")


def test_nl2repo_rejects_malicious_tar_without_starting_fresh_sandbox(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path)
    primary = _PrimarySandbox(_archive({}, unsafe_name="../escape"))
    evaluator = NL2RepoFreshSandboxEvaluator(primary)
    started = 0

    def factory():
        nonlocal started
        started += 1
        return _FreshSandbox([])

    evaluator.configure_fresh_sandbox(factory)
    result = evaluator.evaluate(task, Episode(id="e", task=task.id))

    assert result.reward == 0.0
    assert result.metadata["verifier_status"] == "invalid_submission"
    assert "unsafe archive path" in result.metadata["error"]
    assert started == 0
    assert primary.commands[-1].startswith("rm -f --")


def test_nl2repo_rejects_link_to_golden_test(tmp_path: Path) -> None:
    evaluator = NL2RepoFreshSandboxEvaluator(_PrimarySandbox(b"unused"))
    archive_path = tmp_path / "link.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        directory = tarfile.TarInfo("tests")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        target = tarfile.TarInfo("tests/test_a.py")
        target.size = 1
        archive.addfile(target, io.BytesIO(b"x"))
        link = tarfile.TarInfo("pkg/upstream.py")
        link.type = tarfile.SYMTYPE
        link.linkname = "../tests/test_a.py"
        archive.addfile(link)

    with pytest.raises(UnsafeWorkspaceArchive, match="unsafe archive link"):
        evaluator._validate_archive(archive_path, ["tests/test_a.py"])


def test_nl2repo_rejects_link_at_golden_test_path(tmp_path: Path) -> None:
    evaluator = NL2RepoFreshSandboxEvaluator(_PrimarySandbox(b"unused"))
    archive_path = tmp_path / "link-name.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        source = tarfile.TarInfo("pkg/fake.py")
        source.size = 1
        archive.addfile(source, io.BytesIO(b"x"))
        link = tarfile.TarInfo("tests/test_a.py")
        link.type = tarfile.SYMTYPE
        link.linkname = "../pkg/fake.py"
        archive.addfile(link)

    with pytest.raises(UnsafeWorkspaceArchive, match="unsafe archive link"):
        evaluator._validate_archive(archive_path, ["tests/test_a.py"])


def test_nl2repo_fresh_sandbox_failure_is_infrastructure_error(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path)
    primary = _PrimarySandbox(_archive({"pkg/core.py": b"x = 1\n"}))
    evaluator = NL2RepoFreshSandboxEvaluator(primary)
    evaluator.configure_fresh_sandbox(
        lambda: (_ for _ in ()).throw(RuntimeError("provider unavailable"))
    )

    with pytest.raises(RolloutInfrastructureError) as error:
        evaluator.evaluate(task, Episode(id="e", task=task.id))

    assert error.value.reason == "nl2repo_fresh_sandbox_failed"
    assert primary.commands[-1].startswith("rm -f --")


def test_nl2repo_primary_export_failure_is_infrastructure_error(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path)
    primary = _FailingExportPrimarySandbox(b"unused")
    evaluator = NL2RepoFreshSandboxEvaluator(primary)
    evaluator.configure_fresh_sandbox(lambda: _FreshSandbox([]))

    with pytest.raises(RolloutInfrastructureError) as error:
        evaluator.evaluate(task, Episode(id="e", task=task.id))

    assert error.value.reason == "nl2repo_workspace_export_failed"
    assert primary.commands[-1].startswith("rm -f --")


def test_nl2repo_parser_uses_final_summary_and_reports_count_mismatch() -> None:
    parsed = analyze_nl2repo_pytest_results(
        [
            {
                "command": "pytest -q",
                "output": (
                    "======= 99 passed in 0.01s =======\n"
                    "======= 2 passed, 1 failed, 1 error in 1.00s ======="
                ),
            },
            {"command": "python setup.py check", "output": "100 passed"},
        ],
        5,
    )

    assert parsed == {
        "passed": 2,
        "failed": 1,
        "errors": 1,
        "total": 5,
        "success_rate": pytest.approx(0.4),
    }


def test_nl2repo_verifier_kind_preempts_placeholder_test_script(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path)
    primary = _PrimarySandbox(_archive({"pkg/core.py": b"x = 1\n"}))

    kind, config = _detect_verifier(task)
    evaluator = _resolve_evaluator(task, primary, kind, config)

    assert kind == "nl2repo-fresh-sandbox"
    assert isinstance(evaluator, NL2RepoFreshSandboxEvaluator)

@pytest.mark.parametrize("command,rc,output,status,reason", [
    ("pip install -e .", 1, "Network is unreachable", "infrastructure_failure", "verifier_network_unavailable"),
    ("pip install -e .", 1, "No matching distribution found for invented-package", "install_failed", None),
    ("pip install -e .", 1, "FileNotFoundError: [Errno 2] No such file or directory: '/records'", "infrastructure_failure", "verifier_install_path_missing"),
    ("pytest -q", 127, "/bin/bash: pytest: command not found", "infrastructure_failure", "verifier_toolchain_missing"),
    ("pytest -q", 124, "2 passed", "timeout", None),
    ("pytest -q", 2, "==== 1 error in 1.0s ====", "collection_failed", None),
])
def test_nl2repo_failure_stages_are_not_false_completion(tmp_path, command, rc, output, status, reason):
    task = _task(tmp_path)
    path = task.task_dir / "tests/instance.json"
    contract = json.loads(path.read_text())
    contract["verify_cmd"] = [command]
    path.write_text(json.dumps(contract))

    class Fresh(_FreshSandbox):
        def exec(self, command, timeout=None, user=None):
            text = super().exec(command, timeout=timeout, user=user)
            return re.sub(r"(__RLLM_NL2REPO_RC_[0-9a-f]+__=)0", lambda m: m[1] + str(rc), text)

    fresh = Fresh([output])
    evaluator = NL2RepoFreshSandboxEvaluator(_PrimarySandbox(_archive({"module.py": b"pass"})))
    evaluator.configure_fresh_sandbox(lambda: fresh)
    result = evaluator.evaluate(task, Episode(id="e"))
    assert result.metadata["verifier_status"] == status
    assert result.reward == 0 and not result.is_correct
    assert result.metadata["command_results"][0]["exit_code"] == rc
    assert all("bash -lc" not in c for c in fresh.commands)
    if reason:
        assert result.metadata["infrastructure_failure"]["reason"] == reason
    else:
        assert "infrastructure_failure" not in result.metadata
    assert fresh.close_calls == 1



def test_nl2repo_command_proxy_applies_to_fresh_shell_only(tmp_path, monkeypatch):
    import os
    monkeypatch.setenv("https_proxy", "http://unrelated-host-proxy:80")

    class Shell:
        def exec(self, command, timeout=None):
            return subprocess.check_output(["bash", "-c", command], text=True, timeout=timeout)

    shell = Shell()
    contract = {"workdir": str(tmp_path), "verify_cmd": [
        "test \"$https_proxy\" = http://selected:11113 && test \"$HTTPS_PROXY\" = \"$https_proxy\" && printf '1 passed in 1s'"
    ]}
    result = NL2RepoFreshSandboxEvaluator(shell)._run_verifier_commands(shell, contract, proxy_url="http://selected:11113")
    assert result[0]["exit_code"] == 0
    assert "1 passed" in result[0]["output"]
    assert os.environ["https_proxy"] == "http://unrelated-host-proxy:80"
