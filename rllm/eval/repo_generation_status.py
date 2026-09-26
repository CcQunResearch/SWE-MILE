"""Stdlib-only verifier diagnostics, also uploaded into Doc2Repo sandboxes."""

import re

_COUNT = re.compile(r"\b(\d+)\s+(passed|failed|errors?|skipped|xfailed|xpassed|deselected)\b")
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_NETWORK = re.compile(
    r"network is unreachable|temporary failure in name resolution|"
    r"name or service not known|failed to establish a new connection|"
    r"connection (?:refused|reset)|proxyerror|cannot connect to proxy|"
    r"readtimeouterror|connecttimeouterror|certificate_verify_failed",
    re.IGNORECASE,
)


def parse_doc2repo_summary(output: str) -> dict:
    """Use the final summary, retaining pytest-sugar's multiline count block.

    The pinned AweAgent parser takes the last matching line. Its passed count
    is preserved for normal summaries; retaining the adjacent count lines also
    makes failed/errors auditable without the Harbor scorer's false full pass.
    """
    lines = _ANSI.sub("", output).splitlines()
    counts = dict.fromkeys(("passed", "failed", "errors", "skipped", "xfailed", "xpassed", "deselected"), 0)
    matches = [i for i, line in enumerate(lines) if _COUNT.search(line)]
    if not matches:
        return {**counts, "summary_found": False}
    end = matches[-1]
    start = end
    if re.fullmatch(r"\s*\d+\s+(?:passed|failed|errors?|skipped|xfailed|xpassed|deselected)\s*", lines[end]):
        while start > 0 and re.fullmatch(
            r"\s*\d+\s+(?:passed|failed|errors?|skipped|xfailed|xpassed|deselected)\s*", lines[start - 1]
        ):
            start -= 1
    for line in lines[start:end + 1]:
        for count, label in _COUNT.findall(line):
            counts["errors" if label == "error" else label] = int(count)
    return {**counts, "summary_found": True}


def command_failure(command: str, exit_code: int, output: str, *, trusted_install: bool = False) -> dict:
    """Classify explicit execution evidence, never 'no matching distribution' alone."""
    install = bool(re.search(r"\bpip\d*\b.*\binstall\b", command))
    stage = "install" if install else "tests"
    if exit_code in (124, 137):
        return {"status": "timeout", "stage": stage, "reason": "verifier_timeout"}
    if not exit_code:
        return {"status": "completed", "stage": stage}
    reason = None
    if install and _NETWORK.search(output):
        reason = "verifier_network_unavailable"
    elif re.search(r"(?:pytest|pip|python)[\w. -]*: (?:command )?not found|No module named ['\"]?(?:pytest|pip)\b", output):
        reason = "verifier_toolchain_missing"
    elif install and re.search(r"(?:No such file or directory|FileNotFoundError).*['\"]/[^'\"\n]+", output):
        # Only trusted NL2Repo install commands opt into the stale image-path
        # diagnosis. A generated setup.py may itself reference a bad path.
        if trusted_install and re.search(r"""['"]/(?!workspace['"])[A-Za-z0-9_.-]+['"]""", output):
            reason = "verifier_install_path_missing"
    if reason:
        return {"status": "infrastructure_failure", "stage": stage, "reason": reason}
    if install:
        return {"status": "install_failed", "stage": stage}
    if exit_code == 2 or "Interrupted:" in output and "collection" in output:
        return {"status": "collection_failed", "stage": "collection"}
    # Pytest uses exit 4 when importing a conftest fails. Missing generated
    # APIs and syntax errors are submission failures, not a broken runner.
    if exit_code == 4 and "ImportError while loading conftest" in output:
        return {"status": "collection_failed", "stage": "collection"}
    if exit_code in (3, 4) or exit_code in (126, 127):
        return {"status": "infrastructure_failure", "stage": stage, "reason": "verifier_test_startup_failed"}
    if exit_code == 5:
        return {"status": "no_tests_collected", "stage": "collection"}
    return {"status": "test_failed", "stage": stage}


def infrastructure_metadata(failure: dict, output: str = "") -> dict:
    reason = failure["reason"]
    return {
        "reason": reason, "stage": "verifier_" + failure["stage"],
        "exception_type": reason, "error_summary": output[-2000:] or reason,
        "retryable": reason == "verifier_network_unavailable",
    }
