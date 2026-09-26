"""Host-owned NL2Repo image/command compatibility, shared by primary and verifier."""
import json
from pathlib import Path
import shlex

from rllm.sandbox.structured_exec import frame_structured_command, parse_structured_command_output
from rllm.types import RolloutInfrastructureError

VERSION = "nl2repo_image_contract_v1"
ASSET_PATH = Path(__file__).parents[1] / "data/assets/nl2repo_environment.py"


def prepare_nl2repo_image(sandbox, contract):
    # All inputs come from the materialized benchmark, before agent execution
    # or source overlay. No generated file is executed with setup privileges.
    config = {key: contract.get(key, "") for key in ("workdir", "instance_id", "package_name")}
    framed = frame_structured_command("python3 -I -S -c " + shlex.quote(ASSET_PATH.read_text()) + " " + shlex.quote(json.dumps(config)))
    setup = getattr(sandbox, "exec_setup", None)
    executor = setup if callable(setup) else sandbox.exec
    try:
        result = parse_structured_command_output(executor(framed.command.replace("python3 -c ", "python3 -I -S -c "), timeout=60, user="root"), framed.nonce)
        if result.exit_code:
            raise ValueError("image preparation exited " + str(result.exit_code))
        audit = json.loads(result.stdout)
        if audit.get("schema_version") != 1:
            raise ValueError("invalid image preparation result")
    except RolloutInfrastructureError:
        raise
    except Exception as exc:
        raise RolloutInfrastructureError("nl2repo_image_preparation_failed", str(exc), retryable=False, stage="image_setup") from exc
    if not audit.get("ok"):
        raise RolloutInfrastructureError(audit.get("reason", "nl2repo_image_preparation_failed"), audit.get("detail", "image preparation failed"), retryable=False, stage="image_setup", diagnostics=audit)
    return {"version": VERSION, **audit}


def compatible_nl2repo_command(command, contract):
    # This released command double-escapes its Python string and writes invalid
    # syntax. Match the exact trusted asset; never unescape arbitrary commands.
    if contract.get("instance_id") == "tqdm":
        broken = "echo __version__ = " + "\\" * 2 + "'0.0.1" + "\\" * 2 + "' > tqdm/version.py"
        if command == broken:
            return "printf '%s\\n' " + shlex.quote("__version__ = '0.0.1'") + " > tqdm/version.py"
    return command
