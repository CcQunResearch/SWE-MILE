"""Stage and validate current materialized verifier assets."""

import hashlib
import json
from pathlib import Path


def upload_verifier_assets(sandbox, task, tests_dir):
    """Upload freshly materialized trusted assets and record their exact hashes."""
    sandbox.upload_dir(str(tests_dir), "/tests")
    provenance = {}
    for name in ("grader.py", "log_parsers.py", "denovoswe_verifier.py"):
        asset = Path(tests_dir) / name
        if asset.is_file():
            digest = hashlib.sha256(asset.read_bytes()).hexdigest()
            provenance[name] = {"original_sha256": digest, "executed_sha256": digest}
    sandbox._rllm_verifier_asset_provenance = provenance
    return provenance


def validate_training_verifier_assets(task, dataset_name):
    """Fail deterministic trusted-contract faults before any rollout admission."""
    if dataset_name not in {"denovoswe", "swe-rebench-v2-filtered-verified-python", "swe-rebench-v2-filtered-verified-promatched"}:
        return
    from rllm.types import RolloutInfrastructureError

    directory = task.task_dir / "tests"
    names = ["test.sh", "instance.json", "test.patch", "denovoswe_verifier.py" if dataset_name == "denovoswe" else "grader.py"]
    try:
        missing = [name for name in names if not (directory / name).is_file()]
        if missing:
            raise ValueError("missing trusted assets: " + ", ".join(missing))
        instance = json.loads((directory / "instance.json").read_text())
        if not isinstance(instance, dict) or not instance.get("workdir"):
            raise ValueError("instance contract requires workdir")
        if dataset_name == "denovoswe" and not instance.get("passed_ptp"):
            raise ValueError("empty authoritative acceptance test set")
    except (OSError, ValueError) as exc:
        raise RolloutInfrastructureError(
            "verifier_contract_invalid", f"{task.id}: {exc}",
            retryable=False, stage="preflight",
            diagnostics={"fatal": True, "failure_scope": "task"},
        ) from exc
