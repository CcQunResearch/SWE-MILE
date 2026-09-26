"""Dependency-free aggregate test-count collector for shadow verification.

The hash-bound runner contract maps authenticated native events to
materialized test ids so partial observations retain an exact F2P/P2P
breakdown. This module is uploaded into heterogeneous benchmark images, so
its syntax and runtime operations remain compatible with Python 3.7.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

COUNT_OBSERVATION_SCHEMA_VERSION = 4
COUNT_RUNNER_PLAN_SCHEMA_VERSION = 3
COUNT_COLLECTOR_VERSION = 7
_ANSI_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_STATUS_VALUES = {"PASSED", "FAILED", "ERROR", "SKIPPED"}
_SWEREBENCH_COUNT_PLAN_LOG_PARSERS = {
    "python": frozenset({"parse_log_pytest"}),
    "go": frozenset({"parse_log_gotest"}),
    "js": frozenset({"parse_log_js_4"}),
    "ts": frozenset({"parse_log_js_4"}),
}
_SWEREBENCH_COUNT_PLAN_LANGUAGE_ALIASES = {
    "javascript": "js",
    "typescript": "ts",
}


def supports_swerebench_count_plan(language: str, log_parser: str) -> bool:
    """Return whether the collector supports this SWE-rebench V2 contract.

    Keep this dependency-free capability check beside the collector itself so
    image builds and runtime contract construction cannot silently disagree.
    The accepted pairs are deliberately strict: accepting a parser merely
    because another language uses it could make partial count observations
    look authoritative when the runner-specific evidence is unsupported.
    """

    normalized_language = language.strip().casefold()
    normalized_language = _SWEREBENCH_COUNT_PLAN_LANGUAGE_ALIASES.get(
        normalized_language,
        normalized_language,
    )
    supported_parsers = _SWEREBENCH_COUNT_PLAN_LOG_PARSERS.get(
        normalized_language,
        frozenset(),
    )
    return log_parser.strip() in supported_parsers


@dataclass(frozen=True)
class CountResult:
    passed: int
    failed: int = 0
    errored: int = 0
    skipped: int = 0
    unclassified: int = 0
    collector: str = "unknown"
    terminal: bool = True

    @property
    def reported(self) -> int:
        return (
            self.passed
            + self.failed
            + self.errored
            + self.skipped
            + self.unclassified
        )


def _official_vector(
    artifact: Any,
    contract: dict[str, Any],
    expected: int,
) -> CountResult | None:
    expected_parser = str(contract.get("result_parser") or "")
    if (
        not isinstance(artifact, dict)
        or artifact.get("schema_version") != 3
        or expected_parser
        not in {"swerebench_v2_official_v2", "swerebench_v2_pytest_v1"}
        or artifact.get("parser") != expected_parser
        or artifact.get("result_adapter_version") != 6
        or str(artifact.get("log_parser") or "")
        != str(contract.get("log_parser") or "")
        or str(artifact.get("language") or "").casefold()
        != str(contract.get("language") or "").casefold()
    ):
        return None
    execution = artifact.get("execution")
    command_evidence = (
        execution.get("commands") if isinstance(execution, dict) else None
    )
    if (
        not isinstance(command_evidence, list)
        or len(command_evidence) != len(contract.get("commands") or [])
        or execution.get("timed_out") is not False
        or execution.get("resource_exhausted") is not False
    ):
        return None
    values = artifact.get("test_results")
    complete = artifact.get("complete") is True
    if (
        type(artifact.get("complete")) is not bool
        or not isinstance(values, dict)
        or len(values) > expected
        or (complete and len(values) != expected)
    ):
        return None
    statuses = [str(value).upper() for value in values.values()]
    if any(status not in _STATUS_VALUES for status in statuses):
        return None
    return CountResult(
        passed=sum(status == "PASSED" for status in statuses),
        failed=sum(status == "FAILED" for status in statuses),
        errored=sum(status == "ERROR" for status in statuses),
        skipped=sum(status == "SKIPPED" for status in statuses),
        collector=(
            "official_complete_vector"
            if complete
            else "official_partial_vector"
        ),
        terminal=complete,
    )


def _official_named_result(
    artifact: Any,
    contract: dict[str, Any],
    command: dict[str, Any],
) -> dict[str, Any] | None:
    """Map an authenticated grader vector onto one command's test contract."""

    authenticated = _official_vector(
        artifact,
        contract,
        int(contract.get("expected") or 0),
    )
    if authenticated is None or len(contract.get("commands") or []) != 1:
        return None
    values = artifact.get("test_results")
    if not isinstance(values, dict):
        return None
    command_id = str(command["command_id"])
    states: dict[str, tuple[str, str]] = {}
    extras: list[str] = []
    for raw_name, raw_status in values.items():
        name = str(raw_name)
        status = str(raw_status).upper()
        expected, conflict = _match_expected_test(name, None, command)
        if conflict:
            return {"ok": False, "failure_type": conflict, "extra": name}
        if expected is None:
            extras.append(name)
            continue
        conflict = _merge_owned_state(states, expected, status, command_id)
        if conflict:
            return {"ok": False, "failure_type": conflict, "extra": name}
    if extras:
        return {
            "ok": False,
            "failure_type": "independent_extra_tests",
            "states": states,
            "extra_count": len(set(extras)),
            "extras": sorted(set(extras))[:50],
        }
    return {
        "ok": True,
        "states": states,
        "ignored": [],
        # The authenticated grader command finished even when its named vector
        # is partial, so missing F2P/P2P semantics may be applied safely.
        "terminal": True,
        "started": bool(states),
        "collector": authenticated.collector + "_named",
    }


def _regions(log: str, command_count: int) -> list[str] | None:
    regions: list[str] = []
    for index in range(command_count):
        start = f"__RLLM_COUNT_COMMAND_START__:{index}"
        end = f"__RLLM_COUNT_COMMAND_END__:{index}:"
        if log.count(start) != 1 or log.count(end) != 1:
            return None
        body = log.split(start, 1)[1].split(end, 1)[0]
        regions.append(body)
    return regions


def _normalized_test_name(value: str) -> str:
    value = re.sub(
        r"(?:\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]|"
        r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)|"
        r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\))$",
        "",
        value.strip(),
        flags=re.IGNORECASE,
    )
    # Only spaced hierarchy separators are structural. Literal ``<div>`` and
    # comparison titles such as ``value >0`` are part of the test name.
    value = re.sub(r"\s+(?:>|›)\s+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _plan_commands(contract: dict[str, Any]) -> list[dict[str, Any]] | None:
    runner_plan = contract.get("runner_plan")
    if not isinstance(runner_plan, dict):
        return None
    commands = runner_plan.get("commands")
    expected_tests = contract.get("expected_tests")
    schema_version = runner_plan.get("schema_version")
    if (
        schema_version != COUNT_RUNNER_PLAN_SCHEMA_VERSION
        or runner_plan.get("plan_hash") != contract.get("plan_hash")
        or runner_plan.get("expected") != contract.get("expected")
        or runner_plan.get("expected_tests") != expected_tests
        or not isinstance(expected_tests, list)
        or not isinstance(commands, list)
        or len(commands) != len(contract.get("commands") or [])
    ):
        return None
    seen_ids: set[str] = set()
    seen_outputs: set[str] = set()
    covered_tests: set[str] = set()
    for index, command in enumerate(commands):
        if not isinstance(command, dict):
            return None
        command_id = command.get("command_id")
        expected_tests = command.get("expected_tests")
        expected_aliases = command.get("expected_aliases")
        if (
            not isinstance(command_id, str)
            or not command_id
            or command_id in seen_ids
            or command.get("command") != contract["commands"][index]
            or not isinstance(command.get("runner"), str)
            or not isinstance(expected_tests, list)
            or not isinstance(expected_aliases, list)
            or len(expected_tests) != len(expected_aliases)
            or any(not isinstance(item, str) or not item for item in expected_tests)
            or any(not isinstance(item, str) or not item for item in expected_aliases)
            or len(set(expected_tests)) != len(expected_tests)
            or not _aliases_have_unique_ownership(expected_tests, expected_aliases)
            or any(item not in contract["expected_tests"] for item in expected_tests)
        ):
            return None
        output_path = command.get("output_path")
        if output_path is not None:
            if (
                not isinstance(output_path, str)
                or not output_path
                or output_path in seen_outputs
            ):
                return None
            seen_outputs.add(output_path)
        if (
            command.get("result_channel")
            not in {"native", "native_json", "tap", "official_artifact"}
            or command.get("selection_strategy")
            not in {
                "exact_selector",
                "pytest_parser_alias_file_scope",
                "node_file_scope",
                "full_suite_fallback",
            }
            or not isinstance(command.get("runner_resolution"), str)
            or not command.get("runner_resolution")
            or type(command.get("wrapped_command_bytes")) is not int
            or command.get("wrapped_command_bytes") <= 0
            or command.get("wrapped_command_bytes") > 64 * 1024
        ):
            return None
        selection_strategy = str(command["selection_strategy"])
        structural_extra_policy = command.get("structural_extra_policy")
        test_files = command.get("test_files")
        if (
            structural_extra_policy
            not in {"none", "go_parent_subtest_v1", "scoped_non_contract_v1"}
            or not isinstance(test_files, list)
            or any(not isinstance(item, str) or not item for item in test_files)
            or (
                selection_strategy
                in {"pytest_parser_alias_file_scope", "node_file_scope"}
                and (
                    not test_files
                    or structural_extra_policy != "scoped_non_contract_v1"
                )
            )
            or (
                command.get("runner") == "go"
                and structural_extra_policy != "go_parent_subtest_v1"
            )
        ):
            return None
        if command.get("runner_resolution") == "package_script_expanded":
            script_hash = command.get("script_hash")
            if (
                not isinstance(command.get("script_path"), str)
                or not command.get("script_path")
                or not isinstance(command.get("script_name"), str)
                or not command.get("script_name")
                or not isinstance(script_hash, str)
                or len(script_hash) != 64
                or any(character not in "0123456789abcdef" for character in script_hash)
            ):
                return None
        seen_ids.add(command_id)
        covered_tests.update(expected_tests)
    if covered_tests != set(contract["expected_tests"]):
        return None
    hash_payload = {
        "schema_version": schema_version,
        "expected_tests": contract["expected_tests"],
        "commands": commands,
    }
    f2p_tests = runner_plan.get("f2p_tests")
    p2p_tests = runner_plan.get("p2p_tests")
    if f2p_tests is not None or p2p_tests is not None:
        if (
            not isinstance(f2p_tests, list)
            or not isinstance(p2p_tests, list)
            or any(not isinstance(item, str) or not item for item in f2p_tests)
            or any(not isinstance(item, str) or not item for item in p2p_tests)
            or f2p_tests + p2p_tests != contract["expected_tests"]
            or len(set(f2p_tests + p2p_tests)) != len(f2p_tests + p2p_tests)
        ):
            return None
        hash_payload["f2p_tests"] = f2p_tests
        hash_payload["p2p_tests"] = p2p_tests
    observed_hash = hashlib.sha256(
        json.dumps(
            hash_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if observed_hash != contract.get("plan_hash"):
        return None
    return commands


def _partition_manifest(
    contract: dict[str, Any],
) -> tuple[list[str], list[str]] | None:
    """Return the hash-bound F2P/P2P manifest for missing-test semantics."""

    runner_plan = contract.get("runner_plan")
    if not isinstance(runner_plan, dict):
        return None
    f2p_tests = runner_plan.get("f2p_tests")
    p2p_tests = runner_plan.get("p2p_tests")
    expected_tests = contract.get("expected_tests")
    if (
        not isinstance(f2p_tests, list)
        or not isinstance(p2p_tests, list)
        or f2p_tests + p2p_tests != expected_tests
        or contract.get("f2p_tests") != f2p_tests
        or contract.get("p2p_tests") != p2p_tests
    ):
        return None
    return f2p_tests, p2p_tests


def _aliases_have_unique_ownership(
    expected_tests: list[str],
    expected_aliases: list[str],
) -> bool:
    ownership: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    for index, alias in enumerate(expected_aliases):
        counts[alias] = counts.get(alias, 0) + 1
        test_id = expected_tests[index]
        path = test_id.split("::", 1)[0] if "::" in test_id else ""
        if path:
            ownership.setdefault(alias, set()).add(path.replace("\\", "/"))
    return all(
        count == 1 or len(ownership.get(alias, set())) == count
        for alias, count in counts.items()
    )


def _match_expected_test(
    candidate: str,
    file_name: str | None,
    command: dict[str, Any],
) -> tuple[str | None, str | None]:
    normalized = _normalized_test_name(candidate)
    path = str(file_name or "").replace("\\", "/").strip()
    selection_strategy = str(command.get("selection_strategy") or "")
    if (
        not path
        and selection_strategy == "pytest_parser_alias_file_scope"
        and "::" in candidate
    ):
        path = candidate.split("::", 1)[0].replace("\\", "/").strip()
    tests = command["expected_tests"]
    aliases = command["expected_aliases"]
    records: list[tuple[str, str, bool, bool]] = []
    for index, raw_alias in enumerate(aliases):
        alias = _normalized_test_name(str(raw_alias))
        expected = str(tests[index])
        expected_path = (
            expected.split("::", 1)[0].replace("\\", "/")
            if "::" in expected
            else ""
        )
        expected_basename = expected_path.rsplit("/", 1)[-1]
        basename_is_unique = bool(
            expected_path
            and sum(
                1
                for value in tests
                if "::" in str(value)
                and str(value)
                .split("::", 1)[0]
                .replace("\\", "/")
                .rsplit("/", 1)[-1]
                == expected_basename
            )
            == 1
        )
        file_authenticates = bool(
            path
            and expected_path
            and (
                path.endswith(expected_path)
                or expected_path.endswith(path)
                or basename_is_unique
                and path.rsplit("/", 1)[-1] == expected_basename
            )
        )
        path_matches = bool(
            not path
            or not expected_path
            or file_authenticates
        )
        records.append((expected, alias, path_matches, file_authenticates))

    def unique(
        matches: list[tuple[str, str, bool, bool]],
    ) -> tuple[str | None, str | None]:
        identities = sorted(set(record[0] for record in matches))
        if len(identities) == 1:
            return identities[0], None
        if len(identities) > 1:
            return None, "ambiguous_test_ownership"
        return None, None

    # 1. Exact normalized names always outrank suffix aliases. A file hint,
    # when present, still has to agree with the materialized test id.
    exact = [record for record in records if record[1] == normalized and record[2]]
    result, conflict = unique(exact)
    if result is not None or conflict is not None:
        return result, conflict

    # SWE-rebench's materializer historically tokenized pytest output on
    # whitespace, truncating parametrized nodeids such as ``test[x, y]`` to
    # ``test[x,``.  File-scoped execution lets the native plugin recover the
    # full nodeid; reproduce exactly that lossy parser alias only for a
    # hash-bound command whose nodeid path authenticates the selected file.
    if selection_strategy == "pytest_parser_alias_file_scope":
        collapsed = _normalized_test_name(candidate.split(maxsplit=1)[0])
        parser_alias = [
            record
            for record in records
            if record[1] == collapsed and record[3]
        ]
        result, conflict = unique(parser_alias)
        if result is not None or conflict is not None:
            return result, conflict

    # 2. Full runner names often prepend describe blocks. Prefer the longest
    # unique expected alias, so ``suite nested leaf`` owns a candidate instead
    # of the shorter ``leaf`` suffix.
    forward = [
        record
        for record in records
        if record[2] and normalized.endswith(" " + record[1])
    ]
    if forward:
        longest = max(len(record[1]) for record in forward)
        return unique([record for record in forward if len(record[1]) == longest])

    # 3. The reverse direction is inherently lossy and is accepted only when
    # the runner supplies a file path that uniquely authenticates ownership.
    if path:
        reverse = [
            record
            for record in records
            if record[3] and record[1].endswith(" " + normalized)
        ]
        if reverse:
            longest = max(len(record[1]) for record in reverse)
            return unique([record for record in reverse if len(record[1]) == longest])
    return None, None


def _file_in_command_scope(file_name: str, command: dict[str, Any]) -> bool:
    """Prove that a native event belongs to this command's selected files."""

    observed = file_name.replace("\\", "/").strip()
    test_files = [
        str(value).replace("\\", "/").strip()
        for value in command.get("test_files") or []
        if isinstance(value, str) and value.strip()
    ]
    if not observed or not test_files:
        return False
    if any(
        observed.endswith(expected) or expected.endswith(observed)
        for expected in test_files
    ):
        return True
    if "/" in observed:
        return False
    basename = observed.rsplit("/", 1)[-1]
    matching = [
        expected
        for expected in test_files
        if expected.rsplit("/", 1)[-1] == basename
    ]
    return len(matching) == 1


def _merge_owned_state(
    states: dict[str, tuple[str, str]],
    test_id: str,
    status: str,
    command_id: str,
) -> str | None:
    previous = states.get(test_id)
    if previous is not None:
        if previous[1] != command_id:
            return "duplicate_test_ownership"
        priority = {"PASSED": 0, "SKIPPED": 1, "FAILED": 2, "ERROR": 3}
        if priority.get(status, -1) > priority.get(previous[0], -1):
            states[test_id] = (status, command_id)
        return None
    states[test_id] = (status, command_id)
    return None


def _go_package_in_scope(package: str, command: dict[str, Any]) -> bool:
    scopes = [
        value.strip().replace("\\", "/").lstrip("./")
        for value in str(command.get("package_scope") or "").split(",")
        if value.strip()
    ]
    observed = package.strip().replace("\\", "/").lstrip("./")
    if not scopes or any(scope in {"...", "./..."} for scope in scopes):
        return True
    return bool(
        observed
        and any(
            observed == scope
            or observed.endswith("/" + scope)
            or scope.endswith("/" + observed)
            for scope in scopes
        )
    )


def _go_plan_result(
    text: str,
    command: dict[str, Any],
) -> dict[str, Any]:
    states: dict[str, tuple[str, str]] = {}
    ignored: list[str] = []
    extras: list[str] = []
    package_terminals = 0
    aliases = [_normalized_test_name(str(value)) for value in command["expected_aliases"]]
    command_id = str(command["command_id"])
    for raw_line in text.splitlines():
        try:
            event = json.loads(raw_line.strip())
        except Exception:
            continue
        if not isinstance(event, dict):
            continue
        action = str(event.get("Action") or "")
        package = str(event.get("Package") or "")
        test = str(event.get("Test") or "")
        if not test and package and action in {"pass", "fail"}:
            package_terminals += 1
            continue
        if action not in {"pass", "fail", "skip"} or not test:
            continue
        if not _go_package_in_scope(package, command):
            extras.append(f"{package}:{test}" if package else test)
            continue
        expected, conflict = _match_expected_test(test, None, command)
        if conflict:
            return {"ok": False, "failure_type": conflict, "extra": test}
        if expected is None:
            normalized = _normalized_test_name(test)
            parent = normalized.split("/", 1)[0]
            structural = any(
                alias == parent
                or alias.startswith(parent + "/")
                or parent.startswith(alias + "/")
                for alias in aliases
            )
            if structural:
                ignored.append(f"{package}:{test}" if package else test)
            else:
                extras.append(f"{package}:{test}" if package else test)
            continue
        status = {"pass": "PASSED", "fail": "FAILED", "skip": "SKIPPED"}[action]
        conflict = _merge_owned_state(states, expected, status, command_id)
        if conflict:
            return {"ok": False, "failure_type": conflict, "extra": test}
    if extras:
        return {
            "ok": False,
            "failure_type": "independent_extra_tests",
            "extras": sorted(set(extras))[:50],
        }
    observation = {
        "ok": True,
        "states": states,
        "ignored": sorted(set(ignored)),
        "terminal": package_terminals > 0,
        "started": bool(states or package_terminals),
        "collector": "go_test_json_contract",
    }
    return observation


def _node_json_entries(payload: Any) -> list[tuple[str, str, str]] | None:
    entries: list[tuple[str, str, str]] = []
    if isinstance(payload, dict) and isinstance(payload.get("tests"), list):
        for item in payload["tests"]:
            if not isinstance(item, dict):
                return None
            name = item.get("fullTitle") or item.get("fullName") or item.get("title")
            status = item.get("status")
            file_name = item.get("file") or ""
            if not isinstance(name, str) or not isinstance(status, str):
                return None
            entries.append((name, str(file_name), status))
        return entries
    if not isinstance(payload, dict) or not isinstance(payload.get("testResults"), list):
        return None
    for suite in payload["testResults"]:
        if not isinstance(suite, dict):
            return None
        file_name = str(suite.get("name") or suite.get("testFilePath") or "")
        assertions = suite.get("assertionResults") or suite.get("testResults")
        if not isinstance(assertions, list):
            continue
        for item in assertions:
            if not isinstance(item, dict):
                return None
            name = item.get("fullName") or item.get("fullTitle") or item.get("title")
            status = item.get("status")
            if not isinstance(name, str) or not isinstance(status, str):
                return None
            entries.append((name, file_name, status))
    return entries


def _node_plan_result(command: dict[str, Any]) -> dict[str, Any]:
    output_path = command.get("output_path")
    if not isinstance(output_path, str) or not output_path:
        return {"ok": False, "failure_type": "structured_output_missing"}
    try:
        payload = json.loads(Path(output_path).read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "ok": False,
            "failure_type": "structured_output_missing",
            "error": str(exc),
        }
    entries = _node_json_entries(payload)
    if entries is None:
        return {"ok": False, "failure_type": "structured_output_invalid"}
    states: dict[str, tuple[str, str]] = {}
    ignored: list[str] = []
    extras: list[str] = []
    command_id = str(command["command_id"])
    status_map = {
        "passed": "PASSED",
        "pass": "PASSED",
        "failed": "FAILED",
        "fail": "FAILED",
        "pending": "SKIPPED",
        "skipped": "SKIPPED",
        "skip": "SKIPPED",
        "disabled": "SKIPPED",
        "todo": "SKIPPED",
    }
    for name, file_name, raw_status in entries:
        normalized_status = status_map.get(raw_status.casefold())
        if normalized_status is None:
            return {
                "ok": False,
                "failure_type": "structured_status_unknown",
                "status": raw_status,
            }
        expected, conflict = _match_expected_test(name, file_name, command)
        if conflict:
            return {"ok": False, "failure_type": conflict, "extra": name}
        if expected is None:
            # Jest/Vitest include tests excluded by testNamePattern as pending.
            # They are auditable selector structure, not executed extras.
            if normalized_status == "SKIPPED" or (
                command.get("selection_strategy") == "node_file_scope"
                and _file_in_command_scope(file_name, command)
            ):
                ignored.append(name)
            else:
                extras.append(name)
            continue
        conflict = _merge_owned_state(
            states,
            expected,
            normalized_status,
            command_id,
        )
        if conflict:
            return {"ok": False, "failure_type": conflict, "extra": name}
    if extras:
        return {
            "ok": False,
            "failure_type": "independent_extra_tests",
            "extras": sorted(set(extras))[:50],
        }
    return {
        "ok": True,
        "states": states,
        "ignored": sorted(set(ignored)),
        "terminal": True,
        "started": True,
        "collector": str(command["runner"]) + "_json_contract",
    }


def _tap_plan_result(text: str, command: dict[str, Any]) -> dict[str, Any]:
    states: dict[str, tuple[str, str]] = {}
    extras: list[str] = []
    command_id = str(command["command_id"])
    plan_matches = list(
        re.finditer(r"(?m)^(?P<indent>[ \t]*)1\.\.(?P<n>\d+)\s*$", text)
    )
    point_matches = list(
        re.finditer(
            r"(?m)^(?P<indent>[ \t]*)(?P<not>not\s+)?ok\s+\d+\s*-?\s*(?P<name>[^\n#]*)(?P<tail>[^\n]*)$",
            text,
        )
    )
    # AVA/Node TAP diagnostic assertions are nested under a top-level test.
    # Only the least-indented plan owns acceptance-test names.
    indentation = [len(match.group("indent").expandtabs(8)) for match in point_matches]
    minimum_indent = min(indentation) if indentation else None
    selected_matches = [
        match
        for match in point_matches
        if minimum_indent is not None
        and len(match.group("indent").expandtabs(8)) == minimum_indent
    ]
    selected_plans = [
        match
        for match in plan_matches
        if minimum_indent is None
        or len(match.group("indent").expandtabs(8)) == minimum_indent
    ]
    plan = selected_plans[-1] if selected_plans else None
    for match in selected_matches:
        name = match.group("name").strip()
        if not name:
            continue
        expected, conflict = _match_expected_test(name, None, command)
        if conflict:
            return {"ok": False, "failure_type": conflict, "extra": name}
        if expected is None:
            extras.append(name)
            continue
        tail = match.group("tail").casefold()
        status = (
            "SKIPPED"
            if "# skip" in tail or "# todo" in tail
            else "FAILED"
            if match.group("not")
            else "PASSED"
        )
        conflict = _merge_owned_state(states, expected, status, command_id)
        if conflict:
            return {"ok": False, "failure_type": conflict, "extra": name}
    if extras:
        return {
            "ok": False,
            "failure_type": "independent_extra_tests",
            "extras": sorted(set(extras))[:50],
        }
    return {
        "ok": True,
        "states": states,
        "ignored": [],
        "terminal": plan is not None,
        "started": bool(states or plan is not None),
        "collector": "tap_contract",
    }


def _pytest_plan_events(
    path: str,
    plan_hash: str,
    commands: list[dict[str, Any]],
) -> dict[str, Any]:
    by_id = {str(command["command_id"]): command for command in commands}
    results = {
        command_id: {
            "ok": True,
            "states": {},
            "ignored": [],
            "terminal": False,
            "started": False,
            "collector": "pytest_event_contract",
        }
        for command_id in by_id
    }
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {"ok": False, "failure_type": "structured_output_missing"}
    for line in lines:
        try:
            event = json.loads(line)
        except Exception:
            continue
        if not isinstance(event, dict) or event.get("plan_hash") != plan_hash:
            continue
        command_id = str(event.get("command_id") or "")
        command = by_id.get(command_id)
        current = results.get(command_id)
        if command is None or current is None:
            continue
        kind = str(event.get("event") or "")
        if kind in {"collector_start", "session_start"}:
            current["started"] = True
        if kind == "session_finish":
            current["terminal"] = True
        name = event.get("nodeid")
        status = str(event.get("status") or "").upper()
        if not isinstance(name, str) or status not in _STATUS_VALUES:
            continue
        expected, conflict = _match_expected_test(name, None, command)
        if conflict:
            return {"ok": False, "failure_type": conflict, "extra": name}
        if expected is None:
            if (
                command.get("selection_strategy")
                == "pytest_parser_alias_file_scope"
                and _file_in_command_scope(name.split("::", 1)[0], command)
            ):
                current["ignored"].append(name)
                continue
            return {
                "ok": False,
                "failure_type": "independent_extra_tests",
                "extras": [name],
            }
        conflict = _merge_owned_state(
            current["states"],
            expected,
            status,
            command_id,
        )
        if conflict == "duplicate_test_result":
            # setup/call/teardown updates for one node are collapsed by the
            # plugin before emission; exact repeats from xdist are harmless.
            previous = current["states"].get(expected)
            if previous is not None and previous[0] == status:
                continue
        if conflict:
            return {"ok": False, "failure_type": conflict, "extra": name}
    return {"ok": True, "commands": results}


def _pytest_abort_candidate(
    log: str,
    states: dict[str, tuple[str, str]],
) -> dict[str, Any] | None:
    signatures = (
        (
            "pytest_config_error",
            r"ERROR:\s+usage:.*?(?:pytest|__main__\.py)|"
            r"(?:pytest|__main__\.py):\s+error:",
        ),
        (
            "test_module_import_error",
            r"ImportError while importing test module|Failed to import test module|"
            r"ERROR collecting|(?:SyntaxError|IndentationError|TabError|ImportError|"
            r"ModuleNotFoundError):",
        ),
        ("collection_error", r"Interrupted:\s+\d+\s+errors?\s+during collection"),
    )
    kind = None
    for candidate, pattern in signatures:
        if re.search(pattern, log, flags=re.IGNORECASE | re.DOTALL):
            kind = candidate
            break
    if kind is None:
        return None
    paths: set[str] = set()
    patterns = (
        r"File [\"'](?P<path>[^\"']+)[\"']",
        r"ERROR collecting\s+(?P<path>[^\s:]+)",
        r"ImportError while importing test module [\"']?(?P<path>[^\"'\n]+)",
        r"(?:configfile:|in config file)\s*[\"']?(?P<path>[^\"'\s:]+)",
        r"ERROR:\s+[\"']?(?P<path>[^\"'\s:]+\.(?:ini|toml|cfg))(?::|\s)",
        # ImportError messages commonly identify the production module only
        # in parentheses, for example ``from 'pkg.mod' (/repo/pkg/mod.py)``.
        # Retaining that path lets the shadow prove that an agent edit caused
        # the collection abort instead of discarding a safe partial result.
        r"\((?P<path>[^()\n]+\.py)\)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, log):
            value = str(match.group("path")).strip().replace("\\", "/")
            if value:
                paths.add(value)
    return {
        "kind": kind,
        "paths": sorted(paths)[:100],
        "reported": len(states),
        "bootstrap_authenticated": True,
        "repo_failure_authenticated": True,
    }


def _deterministic_abort_candidate(
    language: str,
    log: str,
    states: dict[str, tuple[str, str]],
) -> dict[str, Any] | None:
    normalized = language.strip().casefold()
    if normalized == "python":
        return _pytest_abort_candidate(log, states)

    # Infrastructure and invocation failures are never converted into a
    # partial code result, even when they contain incidental compiler words.
    infrastructure = (
        r"timed?\s*out|deadline exceeded|signal:\s*(?:killed|terminated)|"
        r"out of memory|oomkill|ENOMEM|ENOSPC|E2BIG|argument list too long|"
        r"ECONN|ENETUNREACH|getaddrinfo|network (?:is )?unreachable|"
        r"RLLM verifier requires Node Yarn|unknown (?:option|argument)|"
        r"unrecognized (?:option|argument)|invalid (?:option|argument)|"
        r"reporter.*(?:unknown|invalid|not found)"
    )
    if re.search(infrastructure, log, flags=re.IGNORECASE):
        return None

    kind = None
    extensions = ""
    if normalized == "go":
        signatures = (
            ("go_setup_failure", r"\[setup failed\]|setup failed"),
            (
                "go_build_failure",
                r"^(?:[^\s:]+/)*[^\s:]+\.go:\d+(?::\d+)?:\s+.+$|"
                r"^#\s+[^\s]+\s*$.*(?:undefined:|syntax error|cannot use|"
                r"imported and not used|too many arguments|not enough arguments)",
            ),
        )
        extensions = r"go|mod|sum"
    elif normalized in {"js", "javascript", "ts", "typescript"}:
        signatures = (
            (
                "typescript_compile_failure",
                r"\bTS\d{4}:|TSError:|TypeScript error|ts-node.*(?:error|failed)|"
                r"(?:esbuild|swc|babel).*(?:transform|compile).*(?:failed|error)",
            ),
            (
                "node_bootstrap_failure",
                r"Test suite failed to run|Exception during run:|"
                r"SyntaxError:\s|(?:Cannot find module|Module not found:).*(?:src|lib|packages)/",
            ),
            (
                "lint_failure",
                r"^\s*[✖×]\s*\d+\s+problems?\b|"
                r"^.+\.(?:ts|tsx|js|jsx):\d+:\d+\s+(?:error|fatal)\b",
            ),
        )
        extensions = r"ts|tsx|mts|cts|js|jsx|mjs|cjs|json"
    else:
        return None
    for candidate, pattern in signatures:
        if re.search(
            pattern,
            log,
            flags=re.IGNORECASE | re.DOTALL | re.MULTILINE,
        ):
            kind = candidate
            break
    if kind is None:
        return None

    paths: set[str] = set()
    path_patterns = (
        r"File [\"'](?P<path>[^\"']+)[\"']",
        r"(?m)^(?:\s*(?:at\s+)?)?(?P<path>(?:\.?\.?/)?[^\s():]+\.(?:"
        + extensions
        + r"))(?::\d+)(?::\d+)?",
        r"\((?P<path>[^()\n]+\.(?:" + extensions + r"))(?::\d+)?(?::\d+)?\)",
    )
    for pattern in path_patterns:
        for match in re.finditer(pattern, log, flags=re.IGNORECASE):
            value = str(match.group("path")).strip().replace("\\", "/")
            if value:
                paths.add(value)
    return {
        "kind": kind,
        "paths": sorted(paths)[:100],
        "reported": len(states),
        "bootstrap_authenticated": True,
        "repo_failure_authenticated": True,
    }


def _structured_failure_reason(
    failure_type: str,
    log: str,
    artifact: Any,
) -> str:
    if "RLLM verifier requires Node Yarn" in log:
        return "node_toolchain_unavailable"
    if re.search(
        r"unknown (?:option|argument)|unrecognized (?:option|argument)|"
        r"invalid (?:option|argument)|reporter.*(?:unknown|invalid|not found)",
        log,
        flags=re.IGNORECASE,
    ):
        return "runner_cli_contract_error"
    if isinstance(artifact, dict) and (
        artifact.get("grader_exception")
        or artifact.get("exception")
        or artifact.get("error_type") == "grader_exception"
    ):
        return "grader_exception"
    return failure_type


def _count_observation(
    contract: dict[str, Any],
    states: dict[str, tuple[str, str]],
    *,
    collector: str,
    command_evidence: list[dict[str, Any]],
    ignored_tests: list[str],
    termination: str,
    partial_reason: str | None,
) -> dict[str, Any]:
    expected = int(contract["expected"])
    statuses = [value[0] for value in states.values()]
    reported = len(statuses)
    complete = reported == expected
    partition_counts = None
    partition_manifest = _partition_manifest(contract)
    if (
        partition_manifest is not None
        and set(states).issubset(set(contract.get("expected_tests") or []))
    ):
        f2p_tests, p2p_tests = partition_manifest

        def partition(names: list[str]) -> dict[str, int]:
            partition_statuses = [
                states[name][0]
                for name in names
                if name in states
            ]
            return {
                "expected": len(names),
                "passed": sum(status == "PASSED" for status in partition_statuses),
                "failed": sum(status == "FAILED" for status in partition_statuses),
                "errored": sum(status == "ERROR" for status in partition_statuses),
                "skipped": sum(status == "SKIPPED" for status in partition_statuses),
                "not_run": len(names) - len(partition_statuses),
            }

        partition_counts = {
            "f2p": partition(f2p_tests),
            "p2p": partition(p2p_tests),
        }
    ownership_rows = [
        {
            "test_id": test_id,
            "command_id": owner,
            "status": status,
        }
        for test_id, (status, owner) in sorted(states.items())
    ]
    ownership_sha256 = hashlib.sha256(
        json.dumps(
            ownership_rows,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    native_state_rows = [
        {"test_id": test_id, "status": status}
        for test_id, (status, _owner) in sorted(states.items())
    ]
    native_state_sha256 = hashlib.sha256(
        json.dumps(
            native_state_rows,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    observation = {
        "schema_version": COUNT_OBSERVATION_SCHEMA_VERSION,
        "scope": "materialized_acceptance_contract",
        "expected": expected,
        "passed": sum(status == "PASSED" for status in statuses),
        "failed": sum(status == "FAILED" for status in statuses),
        "errored": sum(status == "ERROR" for status in statuses),
        "skipped": sum(status == "SKIPPED" for status in statuses),
        "unclassified": 0,
        "reported": reported,
        "not_run": expected - reported,
        "complete": complete,
        "collector": collector,
        "collector_version": COUNT_COLLECTOR_VERSION,
        "command_count": len(contract.get("commands") or []),
        "command_evidence": command_evidence,
        "partition_counts": partition_counts,
        "plan_hash": contract["plan_hash"],
        "termination": "complete" if complete else termination,
        "partial_reason": None if complete else partial_reason,
        "ignored_tests": sorted(set(ignored_tests))[:200],
        "ownership_observed": len(ownership_rows),
        "ownership_sha256": ownership_sha256,
        "ownership_evidence": ownership_rows[:50],
        "native_state_sha256": native_state_sha256,
        # Transport-only input. The host authenticates this map against the
        # digest, then persists only the aggregate observation.
    }
    observation["observed_test_states"] = {
        test_id: status for test_id, (status, _owner) in states.items()
    }
    return observation


def _collect_count(contract: dict[str, Any]) -> dict[str, Any]:
    expected = contract.get("expected")
    commands = contract.get("commands")
    plan_hash = contract.get("plan_hash")
    plan_commands = _plan_commands(contract)
    if (
        type(expected) is not int
        or expected <= 0
        or not isinstance(commands, list)
        or not commands
        or not isinstance(plan_hash, str)
        or len(plan_hash) != 64
        or plan_commands is None
    ):
        return {
            "ok": False,
            "failure_type": "count_contract_error",
            "error": "invalid count collector runner contract",
        }
    expected_tests = contract.get("expected_tests")
    if (
        not isinstance(expected_tests, list)
        or len(expected_tests) != expected
        or len(set(expected_tests)) != expected
    ):
        return {
            "ok": False,
            "failure_type": "count_contract_error",
            "error": "invalid global expected-test manifest",
        }
    try:
        artifact = json.loads(
            Path(str(contract["artifact_path"])).read_text(encoding="utf-8")
        )
    except Exception:
        artifact = None
    try:
        log = Path(str(contract["log_path"])).read_text(
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        log = ""
    clean_log = _ANSI_RE.sub("", log)
    official_named = (
        _official_named_result(artifact, contract, plan_commands[0])
        if len(plan_commands) == 1
        else None
    )
    official_partial_observation = None
    official_partial_states: dict[str, tuple[str, str]] = {}
    official_named_failure: dict[str, Any] | None = None
    if official_named is not None:
        if not official_named.get("ok"):
            official_named_failure = {
                **official_named,
                "error": "authenticated grader names violate the runner contract",
                "log_tail": log[-2000:],
            }
            official_partial_states = dict(
                official_named_failure.get("states") or {}
            )
        else:
            named_states = dict(official_named.get("states") or {})
            named_observation = _count_observation(
                contract,
                named_states,
                collector=str(official_named.get("collector") or "official_named_vector"),
                command_evidence=[
                    {
                        "command_index": 0,
                        "command_id": plan_commands[0]["command_id"],
                        "runner_family": plan_commands[0]["runner"],
                        "collector": official_named.get("collector"),
                        "reported": len(named_states),
                        "terminal": True,
                        "started": bool(official_named.get("started")),
                        "expected": len(plan_commands[0]["expected_tests"]),
                        "structured": True,
                        "result_channel": "official_artifact",
                    }
                ],
                ignored_tests=[],
                termination=("complete" if len(named_states) == expected else "interrupted"),
                partial_reason=(
                    None
                    if len(named_states) == expected
                    else "selector_contract_incomplete"
                ),
            )
            # Official evidence is fallback/cross-check. Continue to
            # parse a hash-bound runner-native artifact even when this vector
            # is complete.
            if named_observation.get("partition_counts") is not None:
                official_partial_observation = named_observation
                official_partial_states = named_states

    def official_partial_result() -> dict[str, Any] | None:
        if official_partial_observation is None:
            if official_named_failure is None:
                return None
            return {
                key: value
                for key, value in official_named_failure.items()
                if key != "states"
            }
        return {
            "ok": True,
            "observation": official_partial_observation,
            "log_tail": log[-2000:],
        }

    def evidence_conflict(
        conflicting_tests: list[str],
        *,
        native_collector: str,
    ) -> dict[str, Any]:
        return {
            "ok": False,
            "failure_type": "count_contract_error",
            "error": "official and runner-native test states conflict",
            "failure_evidence": {
                "plan_hash": plan_hash,
                "native_collector": native_collector,
                "conflicting_tests": sorted(set(conflicting_tests))[:50],
            },
            "log_tail": log[-2000:],
        }

    def checked_official_partial(
        native_states: dict[str, tuple[str, str]],
        *,
        native_collector: str,
    ) -> dict[str, Any] | None:
        fallback = official_partial_result()
        if fallback is None:
            return None
        conflicts = [
            test_id
            for test_id in set(official_partial_states).intersection(native_states)
            if official_partial_states[test_id][0] != native_states[test_id][0]
        ]
        if conflicts:
            return evidence_conflict(
                conflicts,
                native_collector=native_collector,
            )
        return fallback

    def select_named_observation(
        native_observation: dict[str, Any],
        native_states: dict[str, tuple[str, str]],
    ) -> dict[str, Any]:
        native_complete = bool(native_observation.get("complete"))
        official_extras = (
            list(official_named_failure.get("extras") or [])
            if official_named_failure is not None
            and official_named_failure.get("failure_type")
            == "independent_extra_tests"
            else []
        )
        conflicts = [
            test_id
            for test_id in set(official_partial_states).intersection(native_states)
            if official_partial_states[test_id][0] != native_states[test_id][0]
        ]
        if native_complete:
            selected = dict(native_observation)
            notes = list(selected.get("evidence_notes") or [])
            if official_extras:
                notes.append("official_extra_tests_ignored")
                selected["official_extra_count"] = int(
                    official_named_failure.get("extra_count")
                    or len(official_extras)
                )
                selected["official_extra_sample"] = official_extras[:50]
            if conflicts:
                notes.append("official_native_disagreement")
                selected["official_native_disagreement_count"] = len(conflicts)
                selected["official_native_disagreement_sample"] = sorted(conflicts)[:50]
            selected["evidence_notes"] = sorted(set(notes))[:20]
            return {
                "ok": True,
                "observation": selected,
                "log_tail": log[-2000:],
            }
        if official_named_failure is not None:
            # Never combine two partial vectors. If the official parser saw
            # independent tests, only a complete native contract can override
            # that failure.
            return {
                key: value
                for key, value in official_named_failure.items()
                if key != "states"
            }
        fallback = checked_official_partial(
            native_states,
            native_collector=str(native_observation.get("collector") or "structured"),
        )
        if fallback is None:
            return {
                "ok": True,
                "observation": native_observation,
                "log_tail": log[-2000:],
            }
        if not fallback.get("ok"):
            return fallback
        # Evidence selection is monotonic.  Equal or smaller native coverage
        # keeps the authenticated fallback; strictly larger coverage upgrades
        # to the runner-native observation.  We deliberately do not union two
        # partial vectors because their independent termination semantics do
        # not prove that a synthetic union is a single coherent observation.
        if len(native_states) <= len(official_partial_states):
            return fallback
        return {
            "ok": True,
            "observation": native_observation,
            "log_tail": log[-2000:],
        }
    regions = _regions(clean_log, len(commands))
    if regions is None:
        fallback = official_partial_result()
        if fallback is not None:
            return fallback
        unavailable_reason = _structured_failure_reason(
            "count_unavailable",
            clean_log,
            artifact,
        )
        return {
            "ok": False,
            "failure_type": unavailable_reason,
            "error": "count command boundaries are incomplete",
            "failure_evidence": {
                "expected": expected,
                "plan_hash": plan_hash,
                "command_count": len(commands),
            },
            "log_tail": log[-2000:],
        }

    pytest_commands = [
        command
        for command in plan_commands
        if command.get("runner") in {"pytest", "pytest_wrapper"}
    ]
    pytest_results: dict[str, Any] = {}
    pytest_stream = None
    if pytest_commands:
        pytest_stream = _pytest_plan_events(
            str(contract.get("pytest_events_path") or ""),
            plan_hash,
            pytest_commands,
        )
        if pytest_stream.get("ok"):
            pytest_results = dict(pytest_stream.get("commands") or {})
        elif pytest_stream.get("failure_type") not in {"structured_output_missing"}:
            return {
                **pytest_stream,
                "error": "pytest event stream violates the runner contract",
                "log_tail": log[-2000:],
            }

    global_states: dict[str, tuple[str, str]] = {}
    ignored_tests: list[str] = []
    evidence: list[dict[str, Any]] = []
    command_results: list[dict[str, Any]] = []
    structured_failure: dict[str, Any] | None = None
    for index, command in enumerate(plan_commands):
        runner = str(command["runner"])
        if runner in {"pytest", "pytest_wrapper"}:
            current = pytest_results.get(str(command["command_id"]))
            if current is None:
                current = {
                    "ok": False,
                    "failure_type": "structured_output_missing",
                }
        elif runner == "go":
            current = _go_plan_result(regions[index], command)
        elif runner in {"jest", "vitest", "mocha"}:
            current = _node_plan_result(command)
        elif runner in {"ava", "node", "tap", "borp"}:
            current = _tap_plan_result(regions[index], command)
        elif runner == "hardhat":
            current = {
                "ok": False,
                "failure_type": "structured_output_missing",
                "collector": "hardhat_official_artifact",
            }
        else:
            current = {
                "ok": False,
                "failure_type": "structured_runner_unsupported",
            }
        if not current.get("ok"):
            structured_failure = {
                **current,
                "command_id": command.get("command_id"),
                "runner": runner,
            }
            break
        command_results.append(current)
        local_states = current.get("states") or {}
        for test_id, value in local_states.items():
            status, owner = value
            conflict = _merge_owned_state(
                global_states,
                str(test_id),
                str(status),
                str(owner),
            )
            if conflict:
                return {
                    "ok": False,
                    "failure_type": conflict,
                    "error": f"{test_id} was reported by multiple commands",
                    "failure_evidence": {
                        "test_id": test_id,
                        "plan_hash": plan_hash,
                    },
                    "log_tail": log[-2000:],
                }
        ignored_tests.extend(str(value) for value in current.get("ignored") or [])
        evidence.append(
            {
                "command_index": index,
                "command_id": command["command_id"],
                "runner_family": runner,
                "collector": current.get("collector"),
                "reported": len(local_states),
                "terminal": bool(current.get("terminal")),
                "started": bool(current.get("started")),
                "expected": len(command["expected_tests"]),
                "structured": True,
                "result_channel": command.get("result_channel"),
                "runner_resolution": command.get("runner_resolution"),
                "script_path": command.get("script_path"),
                "script_name": command.get("script_name"),
                "script_hash": command.get("script_hash"),
                "script_chain": list(command.get("script_chain") or [])[:4],
                "package_scope": command.get("package_scope"),
                "workspace_scope": command.get("workspace_scope"),
                "test_files": list(command.get("test_files") or [])[:20],
            }
        )

    abort_candidate = _deterministic_abort_candidate(
        str(contract.get("language") or ""),
        clean_log,
        global_states,
    )
    if abort_candidate is not None and len(global_states) < expected:
        candidate_observation = _count_observation(
            contract,
            global_states,
            collector=(
                str(contract.get("language") or "unknown").strip().casefold()
                + "_deterministic_abort_candidate"
            ),
            command_evidence=evidence,
            ignored_tests=ignored_tests,
            termination="deterministic_abort",
            partial_reason=str(abort_candidate["kind"]),
        )
        fallback = checked_official_partial(
            global_states,
            native_collector=str(
                candidate_observation.get("collector") or "deterministic_abort"
            ),
        )
        if fallback is not None:
            return fallback
        return {
            "ok": False,
            "failure_type": "deterministic_abort_candidate",
            "error": "test runner aborted deterministically before the acceptance contract completed",
            "abort_candidate": {
                **abort_candidate,
                "observation": candidate_observation,
            },
            "failure_evidence": {
                "expected": expected,
                "reported": len(global_states),
                "plan_hash": plan_hash,
                "abort_kind": abort_candidate["kind"],
                "abort_paths": abort_candidate["paths"],
            },
            "log_tail": log[-2000:],
        }

    if structured_failure is not None:
        failure_type = str(
            structured_failure.get("failure_type") or "count_unavailable"
        )
        if failure_type not in {
            "ambiguous_test_ownership",
            "duplicate_test_ownership",
            "independent_extra_tests",
            "structured_status_unknown",
        }:
            fallback = checked_official_partial(
                global_states,
                native_collector=str(
                    structured_failure.get("collector")
                    or structured_failure.get("runner")
                    or "structured_failure"
                ),
            )
            if fallback is not None:
                return fallback
        if len(commands) > 1 and failure_type in {
            "structured_output_missing",
            "structured_runner_unsupported",
        }:
            failure_type = "runner_test_assignment_ambiguous"
        failure_type = _structured_failure_reason(
            failure_type,
            clean_log,
            artifact,
        )
        return {
            "ok": False,
            "failure_type": failure_type,
            "error": "runner-native structured output is unavailable or invalid",
            "failure_evidence": {
                "expected": expected,
                "reported": len(global_states),
                "plan_hash": plan_hash,
                "command_failure": structured_failure,
            },
            "log_tail": log[-2000:],
        }

    if len(global_states) > expected:
        return {
            "ok": False,
            "failure_type": "count_total_mismatch",
            "error": f"collector mapped {len(global_states)} tests for expected contract {expected}",
            "failure_evidence": {
                "expected": expected,
                "reported": len(global_states),
                "plan_hash": plan_hash,
            },
            "log_tail": log[-2000:],
        }
    all_terminal = all(bool(result.get("terminal")) for result in command_results)
    any_started = any(bool(result.get("started")) for result in command_results)
    if len(global_states) < expected and all_terminal:
        observation = _count_observation(
            contract,
            global_states,
            collector="+".join(
                str(result.get("collector") or "structured")
                for result in command_results
            ),
            command_evidence=evidence,
            ignored_tests=ignored_tests,
            termination="interrupted",
            partial_reason="selector_contract_incomplete",
        )
        # A terminal under-N vector is usable only when every missing test can
        # be assigned to the hash-bound F2P/P2P contract.  The reward layer can
        # then apply its conservative rule: missing F2P remains unpassed while
        # missing P2P does not create a new regression.
        if observation.get("partition_counts") is not None:
            return select_named_observation(observation, global_states)
        return {
            "ok": False,
            "failure_type": "selector_contract_incomplete",
            "error": (
                f"normally terminated runner plan reported {len(global_states)}/"
                f"{expected} acceptance tests"
            ),
            "failure_evidence": {
                "expected": expected,
                "reported": len(global_states),
                "plan_hash": plan_hash,
                "command_evidence": evidence,
            },
            "log_tail": log[-2000:],
        }
    if len(global_states) < expected and not any_started:
        fallback = checked_official_partial(
            global_states,
            native_collector="runner_native_empty",
        )
        if fallback is not None:
            return fallback
        return {
            "ok": False,
            "failure_type": "count_unavailable",
            "error": "no runner-native execution evidence was produced",
            "failure_evidence": {
                "expected": expected,
                "plan_hash": plan_hash,
                "command_evidence": evidence,
            },
            "log_tail": log[-2000:],
        }
    observation = _count_observation(
        contract,
        global_states,
        collector="+".join(
            str(result.get("collector") or "structured")
            for result in command_results
        ),
        command_evidence=evidence,
        ignored_tests=ignored_tests,
        termination=("complete" if len(global_states) == expected else "interrupted"),
        partial_reason=(None if len(global_states) == expected else "runner_interrupted"),
    )
    return select_named_observation(observation, global_states)


def collect(contract: dict[str, Any]) -> dict[str, Any]:
    result = _collect_count(contract)
    if not result.get("ok"):
        evidence = (
            dict(result.get("failure_evidence"))
            if isinstance(result.get("failure_evidence"), dict)
            else {}
        )
        runner_plan = contract.get("runner_plan")
        plan_commands = (
            runner_plan.get("commands")
            if isinstance(runner_plan, dict)
            and isinstance(runner_plan.get("commands"), list)
            else []
        )
        evidence.setdefault(
            "count_plan_fallback_reason",
            str(contract.get("count_plan_fallback_reason") or "")[:200] or None,
        )
        evidence.setdefault(
            "runner_resolution",
            [
                {
                    "command_id": str(command.get("command_id") or "")[:100],
                    "runner": str(command.get("runner") or "")[:50],
                    "resolution": str(command.get("runner_resolution") or "")[:100],
                    "script_hash": str(command.get("script_hash") or "")[:64] or None,
                }
                for command in plan_commands[:10]
                if isinstance(command, dict)
            ],
        )
        result["failure_evidence"] = evidence
    return result


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print(json.dumps({"ok": False, "error": "usage: test_count_collector.py CONTRACT"}))
        return 0
    try:
        contract = json.loads(Path(args[0]).read_text(encoding="utf-8"))
        result = collect(contract) if isinstance(contract, dict) else {"ok": False, "error": "contract is not an object"}
    except Exception as exc:
        result = {
            "ok": False,
            "failure_type": "count_collector_crash",
            "error": f"{type(exc).__name__}: {exc}",
            "failure_evidence": {
                "exception_type": type(exc).__name__,
                "collector_version": COUNT_COLLECTOR_VERSION,
            },
        }
    # ASCII-safe JSON also serializes isolated UTF-16 surrogate code points;
    # emitting them directly through a UTF-8 stdout stream raises
    # UnicodeEncodeError and used to lose the structured tail marker.
    print(json.dumps(result, ensure_ascii=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
