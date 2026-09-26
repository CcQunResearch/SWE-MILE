import importlib.util
import json
import os
import subprocess
import sys

import pytest

from rllm.sandbox.repo_generation_environment import PROBE_PATH, inspect_repo_generation_environment
from rllm.types import RolloutInfrastructureError


def test_snapshot_uses_image_path_and_reports_measured_editable_state(tmp_path, monkeypatch):
    pip = tmp_path / "pip"
    pip.write_text("#!/bin/sh\ncase \"$1\" in\n --version) echo image-pip;;\n freeze) echo setuptools==79.0.1;;\n list) echo '[]';;\n *) exit 99;;\nesac\n")
    pip.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])

    class Shell:
        def exec(self, command, timeout=None):
            result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=timeout, check=True)
            return "shell banner\n" + result.stdout

    snapshot = inspect_repo_generation_environment(Shell())
    assert snapshot["tools"]["pip"] == str(pip)
    assert snapshot["pip"]["output"].strip() == "image-pip"
    assert snapshot["packages"]["output"].strip() == "setuptools==79.0.1"
    assert snapshot["editable"]["output"].strip() == "[]"
    assert "package_index" not in snapshot


def test_snapshot_unframed_output_is_infrastructure():
    class Broken:
        def exec(self, *args, **kwargs):
            return "lost result"
    with pytest.raises(RolloutInfrastructureError) as caught:
        inspect_repo_generation_environment(Broken())
    assert caught.value.reason == "repo_generation_environment_probe_failed"


def test_probe_redacts_credentials_and_query_parameters():
    spec = importlib.util.spec_from_file_location("environment_probe", PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text = module.redact("index https://name:secret@pypi.example/simple?q=token\npackage==1\n")
    assert text == "index https://pypi.example/simple\npackage==1\n"


def test_network_probe_downloads_uncached_wheel_without_installing(tmp_path, monkeypatch):
    pip = tmp_path / "pip"
    log = tmp_path / "args"
    pip.write_text("#!" + sys.executable + "\nimport json,sys\nfrom pathlib import Path\np=Path(" + repr(str(log)) + ")\nwith p.open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\nprint('[]')\n")
    pip.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    subprocess.run([sys.executable, str(PROBE_PATH), "--check-network"], check=True, capture_output=True)
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    query = commands[-1]
    assert query[:5] == ["--disable-pip-version-check", "download", "--no-deps", "--only-binary=:all:", "--no-cache-dir"]
    assert query[-5:] == ["--retries", "0", "--timeout", "10", "setuptools"]
    assert "--dest" in query
    assert all("install" not in command and "--index-url" not in command for command in commands)


@pytest.mark.parametrize("value", ["socks5://proxy:80", "http://user:secret@proxy", "http://proxy:bad", "http://proxy/x", "http://pro xy"])
def test_proxy_validation_rejects_unsupported_configuration(value):
    from rllm.sandbox.repo_generation_environment import repo_generation_proxy_exports
    with pytest.raises(ValueError):
        repo_generation_proxy_exports(value)


def test_proxy_exports_are_task_local_and_override_stale_no_proxy(monkeypatch):
    from rllm.sandbox.repo_generation_environment import repo_generation_proxy_exports
    monkeypatch.setenv("https_proxy", "http://old:80")
    prefix = repo_generation_proxy_exports("http://selected:11113")
    result = subprocess.check_output(["bash", "-c", prefix + "printf '%s|%s|%s' \"$https_proxy\" \"$HTTPS_PROXY\" \"$NO_PROXY\""], text=True)
    assert result == "http://selected:11113|http://selected:11113|localhost,127.0.0.1,::1"
    assert os.environ["https_proxy"] == "http://old:80"
    assert repo_generation_proxy_exports("") == ""
