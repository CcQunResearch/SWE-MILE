"""Measured package environment for generation prompts and targeted preflight."""

import json
import shlex
from pathlib import Path

from rllm.sandbox.structured_exec import frame_structured_command, parse_structured_command_output

PROMPT_VERSION = "generation_environment_v2"
LEGACY_PROMPT_VERSION = "legacy_issue_v1"
PROBE_PATH = Path(__file__).parents[1] / "data/assets/repo_generation_environment.py"


def validate_prompt_version(value: str) -> str:
    if value not in (PROMPT_VERSION, LEGACY_PROMPT_VERSION):
        raise ValueError(f"unsupported repo_generation_prompt: {value!r}")
    return value


def repo_generation_proxy_exports(value):
    """Explicit task-shell proxy; never mutate driver/Gateway/Ray environment."""
    from urllib.parse import urlsplit

    value = str(value or "").strip()
    if not value:
        return ""
    url = urlsplit(value)
    if (url.scheme not in ("http", "https") or not url.hostname
            or url.username is not None or url.password is not None
            or url.query or url.fragment or url.path not in ("", "/")
            or any(c.isspace() for c in value)):
        raise ValueError("repo_generation_proxy_url must be an HTTP(S) proxy URL without credentials")
    # Validate the port even though urllib otherwise accepts a non-numeric value.
    _ = url.port
    values = {name: value for name in ("http_proxy", "https_proxy", "all_proxy")}
    values["no_proxy"] = "localhost,127.0.0.1,::1"
    values.update({key.upper(): item for key, item in list(values.items())})
    return "export " + " ".join(key + "=" + shlex.quote(item) for key, item in values.items()) + "; "


def repo_generation_environment_exports(proxy_url=""):
    # Two released NL2Repo images embed their publisher's private index.
    # Preserve every other configured index; do not bypass dependency resolution.
    index = (
        'case "${PIP_INDEX_URL-}" in '
        'https://bytedpypi.byted.org/simple|https://bytedpypi.byted.org/simple/) '
        'export PIP_INDEX_URL=https://pypi.org/simple;; esac; '
    )
    return index + repo_generation_proxy_exports(proxy_url)


def inspect_repo_generation_environment(sandbox, *, check_network=False, proxy_url=""):
    source = PROBE_PATH.read_text(encoding="utf-8")
    command = repo_generation_environment_exports(proxy_url) + "python3 -c " + shlex.quote(source)
    if check_network:
        command += " --check-network"
    framed = frame_structured_command(command)
    from rllm.types import RolloutInfrastructureError

    try:
        result = parse_structured_command_output(sandbox.exec(framed.command, timeout=160), framed.nonce)
        if result.exit_code:
            raise RuntimeError(f"repository environment probe exited {result.exit_code}")
        snapshot = json.loads(result.stdout)
        if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 1:
            raise ValueError("invalid repository environment snapshot")
        return snapshot
    except RolloutInfrastructureError:
        raise
    except Exception as exc:
        raise RolloutInfrastructureError(
            "repo_generation_environment_probe_failed",
            f"Cannot read the initial package environment: {type(exc).__name__}",
            retryable=True, stage="agent_environment",
        ) from exc


def generation_environment_prompt(snapshot, profile):
    text = (
        "\n\nMeasured initial package environment (command failures are shown explicitly):\n"
        + json.dumps(snapshot, ensure_ascii=False, indent=2)
        + "\nUse the observed package and editable-install information; do not assume an installation succeeded."
    )
    if profile == "repo_generation_doc2repo":
        text += (
            "\nFinal evaluation runs pip install -e . with dependency resolution, then the supplied tests. "
            "Implement the API and import paths in repo_document.md. Manage runtime dependencies in setup.py; "
            "reuse packages shown above without redundant declarations unless a different version is required. "
            "Pin new dependencies explicitly. Do not substitute requirements.txt for setup.py."
        )
    else:
        text += (
            "\nFinal evaluation overlays your implementation into a fresh benchmark image while retaining "
            "its golden packaging files and acceptance tests. Changes to the primary environment do not "
            "carry over. Implement the package from the specification and verify your public interfaces."
        )
    return text
