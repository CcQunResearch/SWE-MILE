"""Conservative F2P/P2P selector adapters for SWE-rebench V2.

The adapter is intentionally calibration-gated: command generation only says
that a runner can express a selector.  The shadow runtime accepts it only when
both dataset partitions report the exact expected baseline totals and states.
"""

from __future__ import annotations

import copy
import fnmatch
import hashlib
import json
import math
import re
import shlex
from dataclasses import dataclass
from typing import Any

from rllm.harnesses.action_event import TestPartitionCounts

# Linux limits each individual argv string to roughly 128 KiB even when the
# aggregate ARG_MAX is larger.  The selector is quoted twice (once by the
# count wrapper and once by the materialized grader), so a 120 KiB naked
# selector can still fail with E2BIG.  Keep the selector itself small and also
# verify the final wrapped command below.
MAX_SELECTOR_BYTES = 32 * 1024
MAX_WRAPPED_COUNT_COMMAND_BYTES = 64 * 1024
COUNT_RUNNER_PLAN_SCHEMA_VERSION = 3
UNKNOWN_TEST_ID = "---NO TEST NAME FOUND YET---"


class PartitionAdapterUnsupported(ValueError):
    """A selector cannot be generated without guessing about the test runner."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class PartitionShard:
    test_ids: tuple[str, ...]
    instance: dict[str, Any]
    runner: str


@dataclass(frozen=True)
class RunnerInvocation:
    runner: str
    command_index: int
    resolution: str = "direct"
    script_path: str | None = None
    script_name: str | None = None
    script_hash: str | None = None
    script_chain: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunnerPlan:
    invocations: tuple[RunnerInvocation, ...]

    @property
    def name(self) -> str:
        return "+".join(invocation.runner for invocation in self.invocations)


@dataclass(frozen=True)
class RunnerCommandPlan:
    """One auditable command in a scalar acceptance-count probe."""

    command_id: str
    runner: str
    source_command_index: int
    shard_index: int
    shard_count: int
    command: str
    selector: str
    expected_tests: tuple[str, ...]
    expected_aliases: tuple[str, ...]
    selection_strategy: str = "exact_selector"
    output_path: str | None = None
    package_scope: str | None = None
    workspace_scope: str | None = None
    test_files: tuple[str, ...] = ()
    structural_extra_policy: str = "none"
    result_channel: str = "native"
    runner_resolution: str = "direct"
    wrapped_command_bytes: int = 0
    script_path: str | None = None
    script_name: str | None = None
    script_hash: str | None = None
    script_chain: tuple[str, ...] = ()

    def as_contract(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "runner": self.runner,
            "source_command_index": self.source_command_index,
            "shard_index": self.shard_index,
            "shard_count": self.shard_count,
            "command": self.command,
            "selector": self.selector,
            "expected_tests": list(self.expected_tests),
            "expected_aliases": list(self.expected_aliases),
            "selection_strategy": self.selection_strategy,
            "output_path": self.output_path,
            "package_scope": self.package_scope,
            "workspace_scope": self.workspace_scope,
            "test_files": list(self.test_files),
            "structural_extra_policy": self.structural_extra_policy,
            "result_channel": self.result_channel,
            "runner_resolution": self.runner_resolution,
            "wrapped_command_bytes": self.wrapped_command_bytes,
            "script_path": self.script_path,
            "script_name": self.script_name,
            "script_hash": self.script_hash,
            "script_chain": list(self.script_chain),
        }


@dataclass(frozen=True)
class CountRunnerPlan:
    """Hash-bound command/ownership contract used by the count collector."""

    commands: tuple[RunnerCommandPlan, ...]
    expected_tests: tuple[str, ...]
    f2p_tests: tuple[str, ...]
    p2p_tests: tuple[str, ...]
    plan_hash: str

    def as_contract(self) -> dict[str, Any]:
        return {
            "schema_version": COUNT_RUNNER_PLAN_SCHEMA_VERSION,
            "plan_hash": self.plan_hash,
            "expected": len(self.expected_tests),
            "expected_tests": list(self.expected_tests),
            "f2p_tests": list(self.f2p_tests),
            "p2p_tests": list(self.p2p_tests),
            "commands": [command.as_contract() for command in self.commands],
        }


@dataclass(frozen=True)
class PartitionAdapter:
    name: str
    f2p: tuple[str, ...]
    p2p: tuple[str, ...]
    f2p_shards: tuple[PartitionShard, ...]
    p2p_shards: tuple[PartitionShard, ...]
    combined_shards: tuple[PartitionShard, ...]
    plan: RunnerPlan

    @property
    def f2p_instance(self) -> dict[str, Any]:
        """Compatibility view for callers testing a single-shard adapter."""

        return self.f2p_shards[0].instance

    @property
    def p2p_instance(self) -> dict[str, Any] | None:
        return self.p2p_shards[0].instance if self.p2p_shards else None

    def shards(self, partition: str) -> tuple[PartitionShard, ...]:
        if partition == "f2p":
            return self.f2p_shards
        if partition == "p2p":
            return self.p2p_shards
        if partition == "combined":
            return self.combined_shards
        raise ValueError(f"unknown partition: {partition}")


@dataclass(frozen=True)
class PartitionObservation:
    counts: TestPartitionCounts
    named_results: dict[str, str]
    extra_named_tests: tuple[str, ...]
    reported: int
    evidence_source: str = "named_results"
    selector_verified: bool = False


PARTITION_OBSERVATION_SCHEMA_VERSION = 1
PARTITION_RESULT_ADAPTER_VERSION = 6
_TRUSTED_PARTITION_COLLECTORS = {
    "pytest": frozenset({"pytest_summary"}),
    "go": frozenset({"go_test_json"}),
    "jest": frozenset({"jest_json", "jest_summary"}),
    "mocha": frozenset({"mocha_json", "mocha_summary"}),
    "ava": frozenset({"ava_tap"}),
    "vitest": frozenset({"vitest_summary"}),
    "tap": frozenset({"tap_contract"}),
    "borp": frozenset({"tap_contract"}),
}


_RUNNER_PATTERNS: dict[str, re.Pattern[str]] = {
    "pytest": re.compile(
        r"(?<![\w.-])(?:(?:python(?:\d+(?:\.\d+)*)?|py)\s+-m\s+)?(?:[\w./-]+/)?pytest\b"
    ),
    "go": re.compile(r"(?<![\w.-])(?:[\w./-]+/)?go\s+test\b"),
    # A colon is excluded from the left boundary so package scripts such as
    # ``npm run test:mocha`` are not mistaken for two runner invocations. The
    # optional path covers old repositories that invoke node_modules/.bin
    # directly instead of using npx.
    # Runner names must be complete shell words.  A trailing ``\b`` also
    # matches dots, which previously treated ``mocha.bootstrap.js`` and
    # ``jest.config.ts`` as extra runner invocations.
    "jest": re.compile(
        r"(?<![\w./:-])(?:npx\s+)?(?:[\w./-]+/)?jest(?:\.js)?(?=\s|$|[;&|])"
    ),
    "mocha": re.compile(
        r"(?<![\w./:-])(?:npx\s+)?(?:[\w./-]+/)?(?:mocha|_mocha)(?=\s|$|[;&|])"
    ),
    "vitest": re.compile(
        r"(?<![\w./:-])(?:npx\s+)?(?:[\w./-]+/)?vitest(?=\s|$|[;&|])(?:\s+run\b)?"
    ),
    "ava": re.compile(
        r"(?<![\w./:-])(?:npx\s+)?(?:[\w./-]+/)?ava(?=\s|$|[;&|])"
    ),
    "node": re.compile(
        r"(?<![\w./:-])(?:[\w./-]+/)?node\s+--test(?=\s|$|[;&|])"
    ),
    "tap": re.compile(
        r"(?<![\w./:-])(?<!reporter\s)(?:npx\s+)?(?:[\w./-]+/)?tap(?=\s|$|[;&|])"
    ),
    "hardhat": re.compile(
        r"(?<![\w./:-])(?:npx\s+)?(?:[\w./-]+/)?hardhat\s+test(?=\s|$|[;&|])"
    ),
    "borp": re.compile(
        r"(?<![\w./:-])(?:npx\s+)?(?:[\w./-]+/)?borp(?=\s|$|[;&|])"
    ),
    "maven": re.compile(r"(?<![\w.-])(?:[\w./-]+/)?mvnw?\b"),
    "gradle": re.compile(r"(?<![\w.-])(?:[\w./-]+/)?gradlew?\b"),
    "phpunit": re.compile(r"(?<![\w./-])(?:[\w./-]+/)?phpunit\b"),
    "pest": re.compile(r"(?<![\w./-])(?:[\w./-]+/)?pest\b"),
}
_PACKAGE_TEST = re.compile(
    r"(?<![\w./-])(?P<tool>npm|pnpm|yarn)\s+"
    r"(?:(?:run|run-script)\s+)?test(?:[:.-][\w.-]+)?\b"
)
_PACKAGE_SCRIPT = re.compile(
    r"(?<![\w./-])(?P<tool>npm|pnpm|yarn)\s+"
    r"(?:(?:workspace\s+(?P<workspace_scope>[^\s;&|]+)\s+)|"
    r"(?:(?:--workspace|--filter|-w)(?:=|\s+)[^\s;&|]+\s+))*"
    r"(?:(?P<verb>run|run-script)\s+)?(?P<script>[\w.:_-]+)\b"
)
_WRAPPER_RUNNERS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?<![\w./-])(?:npx\s+)?tsdx\s+test(?=\s|$|[;&|])"), "jest"),
    (re.compile(r"(?<![\w./-])(?:npx\s+)?test-ava(?=\s|$|[;&|])"), "ava"),
)
_MAKE_COMMAND = re.compile(r"(?<![\w./-])make(?:\s+-[\w-]+)*(?:\s+[\w./-]+)+\b")


def _shell_mask(command: str) -> str:
    """Blank quoted contents while preserving offsets for safe insertion."""

    masked = list(command)
    quote: str | None = None
    escaped = False
    for index, character in enumerate(command):
        if escaped:
            masked[index] = " "
            escaped = False
            continue
        if character == "\\" and quote != "'":
            masked[index] = " "
            escaped = True
            continue
        if quote is not None:
            masked[index] = " "
            if character == quote:
                quote = None
            continue
        if character in {"'", '"'}:
            masked[index] = " "
            quote = character
    if quote is not None or escaped:
        raise PartitionAdapterUnsupported("unsafe_shell_syntax")
    masked_command = "".join(masked)
    # A small audited family of Go commands expands package lists using
    # ``go list``.  Blank those trusted substitutions for command-boundary
    # analysis while continuing to reject arbitrary substitutions.
    audited = re.sub(
        r"\$\(\s*go\s+list\s+[^()]*\)",
        lambda match: " " * len(match.group(0)),
        masked_command,
    )
    audited = re.sub(
        r"`\s*go\s+list\s+[^`]*`",
        lambda match: " " * len(match.group(0)),
        audited,
    )
    if "`" in audited or "$(" in audited:
        raise PartitionAdapterUnsupported("unsafe_shell_syntax")
    return audited


def _runner_token_count(masked_command: str) -> int:
    package_matches = list(_PACKAGE_SCRIPT.finditer(masked_command))

    def inside_package(match: re.Match[str]) -> bool:
        return any(
            package.start() <= match.start() < package.end()
            for package in package_matches
        )

    direct = sum(
        sum(not inside_package(match) for match in pattern.finditer(masked_command))
        for pattern in _RUNNER_PATTERNS.values()
    )
    wrappers = sum(
        sum(not inside_package(match) for match in pattern.finditer(masked_command))
        for pattern, _ in _WRAPPER_RUNNERS
    )
    package = len(package_matches)
    return direct + wrappers + package


def _expand_compound_runner_command(command: str) -> list[str]:
    """Split a simple ``&&`` runner chain into cwd-preserving invocations."""

    masked = _shell_mask(command)
    if _runner_token_count(masked) <= 1:
        return [command]
    if re.search(r"\|\||(?<!\|)\|(?!\|)|;|\n", masked):
        raise PartitionAdapterUnsupported("multiple_test_runners")
    boundaries = list(re.finditer(r"\s*&&\s*", masked))
    if not boundaries:
        raise PartitionAdapterUnsupported("multiple_test_runners")
    starts = [0, *(match.end() for match in boundaries)]
    ends = [*(match.start() for match in boundaries), len(command)]
    segments = [
        command[start:end].strip()
        for start, end in zip(starts, ends, strict=True)
    ]
    prefix: list[str] = []
    invocations: list[str] = []
    for segment in segments:
        segment_masked = _shell_mask(segment)
        count = _runner_token_count(segment_masked)
        if count == 1:
            invocations.append(" && ".join((*prefix, segment)))
            continue
        if count > 1:
            raise PartitionAdapterUnsupported("multiple_test_runners")
        if re.fullmatch(r"cd\s+[^;&|]+", segment_masked.strip()):
            prefix.append(segment)
            continue
        raise PartitionAdapterUnsupported("compound_command_unsupported")
    if len(invocations) <= 1:
        raise PartitionAdapterUnsupported("multiple_test_runners")
    return invocations


def _commands(instance: dict[str, Any]) -> list[str]:
    config = instance.get("install_config")
    if not isinstance(config, dict):
        raise PartitionAdapterUnsupported("missing_install_config")
    raw = config.get("test_cmd")
    commands = [raw] if isinstance(raw, str) else raw
    if not isinstance(commands, list) or not commands or any(not isinstance(item, str) or not item.strip() for item in commands):
        raise PartitionAdapterUnsupported("invalid_test_cmd")
    return [
        invocation
        for command in commands
        for invocation in _expand_compound_runner_command(command)
    ]


def expand_runner_commands(instance: dict[str, Any]) -> tuple[str, ...]:
    """Return auditable command boundaries for a shadow-only runner plan.

    Simple multi-runner ``&&`` chains are expanded while retaining their
    working-directory prefixes. Unsupported shell constructs fail closed so a
    caller can keep the original command as an unavailable aggregate plan.
    """

    return tuple(_commands(instance))


def _wrapper_match(masked: str, runner: str) -> re.Match[str] | None:
    for pattern, family in _WRAPPER_RUNNERS:
        if family == runner:
            match = pattern.search(masked)
            if match is not None:
                return match
    return None


def wrap_count_command(command: str, index: int) -> str:
    """Return the exact command string passed to the materialized grader."""

    return (
        f"printf '\n__RLLM_COUNT_COMMAND_START__:{index}\n'; "
        f"bash -o pipefail -c {shlex.quote(command)}; "
        "__rllm_count_rc=$?; "
        f"printf '\n__RLLM_COUNT_COMMAND_END__:{index}:%s\n' "
        '"$__rllm_count_rc"; exit "$__rllm_count_rc"'
    )


def _append_runner_arguments(command: str, runner: str, arguments: str) -> str:
    """Append shadow-only reporter arguments inside one runner boundary."""

    masked = _shell_mask(command)
    match = _RUNNER_PATTERNS[runner].search(masked)
    package_match = None
    if match is None and runner in {"jest", "mocha", "vitest", "ava"}:
        package_match = _PACKAGE_SCRIPT.search(masked)
        match = package_match
    if match is None:
        match = _wrapper_match(masked, runner)
    if match is None:
        raise PartitionAdapterUnsupported("runner_invocation_not_found")
    boundary = re.search(
        r"&&|\|\||(?<!\|)\|(?!\|)|;|\n",
        masked[match.end() :],
    )
    insertion = (
        match.end() + boundary.start()
        if boundary is not None
        else len(command)
    )
    prefix = command[:insertion].rstrip()
    suffix = command[insertion:].lstrip()
    if package_match is not None:
        # npm/pnpm/yarn options must follow the script argument separator;
        # otherwise the package manager consumes the reporter option.
        tail = masked[package_match.end() : insertion]
        separator = " " if re.search(r"(?:^|\s)--(?:\s|$)", tail) else " -- "
        prefix += separator + arguments
    else:
        prefix += " " + arguments
    return prefix + ((" " + suffix) if suffix else "")


def _with_count_reporter(
    command: str,
    runner: str,
    command_number: int,
) -> tuple[str, str | None, str]:
    """Give each structured Node runner a collision-free result artifact."""

    root = "/tmp/rllm/shadow-count"
    if runner == "jest":
        output = f"{root}/jest-{command_number}.json"
        command = re.sub(r"(?<!\S)--json(?=\s|$)", "", command)
        command = re.sub(
            r"(?<!\S)--outputFile(?:=|\s+)(?:'[^']*'|\"[^\"]*\"|[^\s;&|]+)",
            "",
            command,
        )
        command = _append_runner_arguments(
            command,
            runner,
            "--json --outputFile=" + shlex.quote(output),
        )
        return command, output, "native_json"
    if runner == "vitest":
        output = f"{root}/vitest-{command_number}.json"
        command = re.sub(
            r"(?<!\S)--reporter(?:=|\s+)(?:'[^']*'|\"[^\"]*\"|[^\s;&|]+)",
            "",
            command,
        )
        command = re.sub(
            r"(?<!\S)--outputFile(?:=|\s+)(?:'[^']*'|\"[^\"]*\"|[^\s;&|]+)",
            "",
            command,
        )
        command = _append_runner_arguments(
            command,
            runner,
            "--reporter=json --outputFile=" + shlex.quote(output),
        )
        return command, output, "native_json"
    if runner == "mocha":
        # Reporter arguments are placed after the package-manager ``--``
        # separator, so every command owns an isolated native output file.
        # Mocha applies the explicit CLI reporter over package configuration.
        output = f"{root}/mocha-{command_number}.json"
        reporter = f"{root}/mocha-reporter.js"
        command = re.sub(
            r"(?<!\S)(?:--reporter|-R)(?:=|\s+)(?:'[^']*'|\"[^\"]*\"|[^\s;&|]+)",
            "",
            command,
        )
        command = _append_runner_arguments(
            command,
            runner,
            "--reporter " + shlex.quote(reporter),
        )
        command = (
            "export RLLM_SHADOW_NODE_RESULT_PATH="
            + shlex.quote(output)
            + "; "
            + command
        )
        return command, output, "native_json"
    if runner == "hardhat":
        # Hardhat owns Mocha's environment/bootstrap. Its supported test-file
        # and grep selectors are injected above, while the authenticated
        # grader vector remains the structured result channel. Replacing the
        # runner with raw Mocha would silently invalidate fixtures/plugins.
        return command, None, "official_artifact"
    return command, None, "tap"


def _command_ownership_hints(
    command: str,
    runner: str,
    test_ids: tuple[str, ...],
) -> tuple[str | None, str | None, tuple[str, ...]]:
    """Extract audit-only package/workspace/file ownership hints."""

    test_files = tuple(
        sorted(
            {
                name.split("::", 1)[0].replace("\\", "/")
                for name in test_ids
                if "::" in name and name.split("::", 1)[0]
            }
        )
    )
    package_scope = None
    if runner == "go":
        packages = []
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = []
        for index in range(len(tokens) - 1):
            if tokens[index].rsplit("/", 1)[-1] == "go" and tokens[index + 1] == "test":
                packages = [
                    token
                    for token in tokens[index + 2 :]
                    if not token.startswith(("-", "^"))
                    and (
                        token in {"."}
                        or token.startswith(("./", "../"))
                        or "/" in token
                    )
                ]
                break
        if packages:
            package_scope = ",".join(packages)
    workspace_scope = None
    if runner in {"jest", "mocha", "vitest", "ava", "node", "tap", "hardhat", "borp"}:
        match = re.search(
            r"(?:--(?:workspace|project|filter)(?:=|\s+)|(?:^|\s)-w\s+|"
            r"(?:^|\s)workspace\s+)([^\s;&|]+)",
            _shell_mask(command),
        )
        if match is not None:
            workspace_scope = match.group(1)
        else:
            cwd_match = re.search(
                r"(?:^|&&\s*)cd\s+([^\s;&|]+)\s+&&",
                _shell_mask(command),
            )
            if cwd_match is not None:
                workspace_scope = cwd_match.group(1)
    return package_scope, workspace_scope, test_files


def _aliases_have_unique_ownership(
    test_ids: tuple[str, ...],
    aliases: tuple[str, ...],
) -> bool:
    """Allow duplicate display names only when distinct files disambiguate."""

    ownership: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    for test_id, alias in zip(test_ids, aliases, strict=True):
        counts[alias] = counts.get(alias, 0) + 1
        path = test_id.split("::", 1)[0] if "::" in test_id else ""
        if path:
            ownership.setdefault(alias, set()).add(path.replace("\\", "/"))
    return all(
        count == 1 or len(ownership.get(alias, set())) == count
        for alias, count in counts.items()
    )


def build_count_runner_plan(
    instance: dict[str, Any],
    package_script_resolutions: dict[int, dict[str, Any]] | None = None,
) -> CountRunnerPlan:
    """Build one shadow-only plan for the union of acceptance-test ids.

    Pass-count probes do not need F2P and P2P as separate evidence domains,
    but they *do* need the runner to execute exactly the materialized
    acceptance contract.  Running the repository's unfiltered suite makes a
    terminal Jest/Mocha summary count unrelated extra tests, while dropping
    Go subtests makes the same summary smaller than ``N``.  Reuse the audited
    selector adapters here and flatten all selector shards into one verifier
    invocation plan.  Each acceptance id belongs to the union exactly once;
    multiple runner invocations remain independently bounded commands.

    The returned commands are used only by the derived shadow instance.  The
    primary verifier command in the materialized task is never modified.
    """

    f2p = tuple(str(name) for name in instance.get("FAIL_TO_PASS") or [])
    p2p = tuple(str(name) for name in instance.get("PASS_TO_PASS") or [])
    test_ids = (*f2p, *p2p)
    if not test_ids:
        raise PartitionAdapterUnsupported("empty_acceptance_contract")
    if any(
        not name.strip()
        or name == UNKNOWN_TEST_ID
        or UNKNOWN_TEST_ID in name
        for name in test_ids
    ):
        raise PartitionAdapterUnsupported("unusable_test_id")
    if len(set(test_ids)) != len(test_ids):
        raise PartitionAdapterUnsupported("ambiguous_test_id")

    commands = _commands(instance)
    config = instance.get("install_config")
    assert isinstance(config, dict)
    plan = _detect_runner_plan(
        commands,
        str(config.get("log_parser") or ""),
        package_script_resolutions=package_script_resolutions,
    )
    assignments = _assign_test_ids(plan, commands, test_ids)
    command_plans: list[RunnerCommandPlan] = []
    for invocation in plan.invocations:
        names = assignments.get(invocation)
        if not names:
            continue
        shards = _derived_shards(
            instance,
            commands,
            invocation.command_index,
            invocation.runner,
            names,
            partition="combined",
            # A count plan always flattens each runner/shard to a separate
            # command boundary.  Keeping unrelated original commands in a
            # shard would execute tests twice and invalidate aggregate N.
            isolate_command=True,
        )
        for shard_index, shard in enumerate(shards):
            command = shard.instance["install_config"].get("test_cmd")
            if not isinstance(command, str) or not command.strip():
                raise PartitionAdapterUnsupported("invalid_count_command")
            command_number = len(command_plans)
            command, output_path, result_channel = _with_count_reporter(
                command,
                invocation.runner,
                command_number,
            )
            if invocation.runner == "pytest" and not re.search(
                r"(?:^|\s)-p(?:=|\s+)rllm_shadow_pytest_plugin(?:\s|$)",
                _shell_mask(command),
            ):
                command = _append_runner_arguments(
                    command,
                    invocation.runner,
                    "-p rllm_shadow_pytest_plugin",
                )
            command_id = f"command-{command_number}"
            command = (
                "export RLLM_SHADOW_COUNT_COMMAND_ID="
                + shlex.quote(command_id)
                + "; "
                + command
            )
            wrapped_command_bytes = len(
                wrap_count_command(command, command_number).encode("utf-8")
            )
            if wrapped_command_bytes > MAX_WRAPPED_COUNT_COMMAND_BYTES:
                raise PartitionAdapterUnsupported("wrapped_command_too_large")
            aliases = tuple(
                _selector_test_id(invocation.runner, name)
                for name in shard.test_ids
            )
            if not _aliases_have_unique_ownership(shard.test_ids, aliases):
                raise PartitionAdapterUnsupported("ambiguous_normalized_test_id")
            package_scope, workspace_scope, test_files = _command_ownership_hints(
                command,
                invocation.runner,
                shard.test_ids,
            )
            command_plans.append(
                RunnerCommandPlan(
                    command_id=command_id,
                    runner=invocation.runner,
                    source_command_index=invocation.command_index,
                    shard_index=shard_index,
                    shard_count=len(shards),
                    command=command,
                    selector=_selector(invocation.runner, shard.test_ids),
                    expected_tests=shard.test_ids,
                    expected_aliases=aliases,
                    selection_strategy=_selection_strategy(
                        invocation.runner,
                        shard.test_ids,
                    ),
                    output_path=output_path,
                    package_scope=package_scope,
                    workspace_scope=workspace_scope,
                    test_files=test_files,
                    result_channel=result_channel,
                    runner_resolution=invocation.resolution,
                    wrapped_command_bytes=wrapped_command_bytes,
                    script_path=invocation.script_path,
                    script_name=invocation.script_name,
                    script_hash=invocation.script_hash,
                    script_chain=invocation.script_chain,
                    structural_extra_policy=(
                        "go_parent_subtest_v1"
                        if invocation.runner == "go"
                        else "scoped_non_contract_v1"
                        if _selection_strategy(
                            invocation.runner,
                            shard.test_ids,
                        )
                        in {
                            "pytest_parser_alias_file_scope",
                            "node_file_scope",
                        }
                        else "none"
                    ),
                )
            )
    if not command_plans:
        raise PartitionAdapterUnsupported("empty_count_plan")
    payload = {
        "schema_version": COUNT_RUNNER_PLAN_SCHEMA_VERSION,
        "expected_tests": list(test_ids),
        "f2p_tests": list(f2p),
        "p2p_tests": list(p2p),
        "commands": [command.as_contract() for command in command_plans],
    }
    plan_hash = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return CountRunnerPlan(
        commands=tuple(command_plans),
        expected_tests=test_ids,
        f2p_tests=f2p,
        p2p_tests=p2p,
        plan_hash=plan_hash,
    )


def _runner_from_log_parser(log_parser: str) -> str | None:
    normalized = log_parser.strip().casefold()
    exact = {
        # SWE-rebench's js_2 parser is Mocha's spec output; js_3 is TAP,
        # which the materialized JavaScript tasks use for AVA.
        "parse_log_js_2": "mocha",
        "parse_log_js_3": "ava",
    }
    if normalized in exact:
        return exact[normalized]
    mappings = {
        "pytest": "pytest",
        "jest": "jest",
        "mocha": "mocha",
        "vitest": "vitest",
        "ava": "ava",
        "golang": "go",
        "go": "go",
        "maven": "maven",
        "mvn": "maven",
        "gradle": "gradle",
        "phpunit": "phpunit",
        "pest": "pest",
    }
    candidates = {
        runner for marker, runner in mappings.items() if marker in normalized
    }
    return next(iter(candidates)) if len(candidates) == 1 else None


def _runner_from_package_contract(log_parser: str, masked_command: str) -> str | None:
    """Infer an opaque package script only from its materialized contract.

    ``parse_log_js_4`` is shared by old Jest and Mocha repositories.  Their
    command-line contracts still distinguish the one Mocha family in the
    supported JS/TS slice (explicit test globs plus Yeoman's ``--no-insight``);
    the remaining package scripts use Jest.  The selector remains
    calibration-gated, so an incorrect future inference cannot become a
    trusted result.
    """

    package = _PACKAGE_SCRIPT.search(masked_command)
    script = package.group("script").casefold() if package is not None else ""
    named_script = re.search(
        r"(?:^|[:._-])(?P<runner>jest|mocha|vitest|ava|tap|hardhat|borp)(?:$|[:._-])",
        script,
    )
    if named_script is not None:
        return named_script.group("runner")
    if script in {"jest", "mocha", "vitest", "ava", "tap", "hardhat", "borp"}:
        return script
    # Explicit CLI contracts are more reliable than the shared js_4 parser.
    if re.search(r"(?:^|\s)(?:--reporter|-R)(?:=|\s+)spec(?:\s|$)", masked_command):
        return "mocha"
    if re.search(
        r"(?:^|\s)(?:--maxWorkers|--runInBand|--testNamePattern|"
        r"--passWithNoTests|--detectOpenHandles)(?:=|\s|$)",
        masked_command,
    ):
        return "jest"
    inferred = _runner_from_log_parser(log_parser)
    if inferred is not None:
        return inferred
    # The generic js parser is the legacy Mocha/spec parser in the
    # SWE-rebench task family. Runtime named-result validation still gates the
    # inferred selector.
    if log_parser.strip().casefold() == "parse_log_js":
        return "mocha"
    if log_parser.strip().casefold() != "parse_log_js_4":
        return None
    if "--no-insight" in masked_command and re.search(
        r"(?:^|\s)test/[^\s;&|]*\*", masked_command
    ):
        return "mocha"
    return "jest"


def _detect_runner_plan(
    commands: list[str],
    log_parser: str,
    *,
    package_script_resolutions: dict[int, dict[str, Any]] | None = None,
) -> RunnerPlan:
    hits: list[RunnerInvocation] = []
    for index, command in enumerate(commands):
        masked = _shell_mask(command)
        command_hits: list[tuple[str, str]] = []
        package_matches = list(_PACKAGE_SCRIPT.finditer(masked))
        package_ranges = [match.span() for match in package_matches]
        for runner, pattern in _RUNNER_PATTERNS.items():
            matches = [
                match
                for match in pattern.finditer(masked)
                if not any(start <= match.start() < end for start, end in package_ranges)
            ]
            if len(matches) > 1:
                raise PartitionAdapterUnsupported("multiple_test_runners")
            if matches:
                command_hits.append((runner, "direct"))
        for pattern, runner in _WRAPPER_RUNNERS:
            matches = pattern.findall(masked)
            if len(matches) > 1:
                raise PartitionAdapterUnsupported("multiple_test_runners")
            if matches:
                command_hits.append((runner, "wrapper"))
        if len(package_matches) > 1:
            raise PartitionAdapterUnsupported("multiple_test_runners")
        if package_matches:
            resolved = (
                package_script_resolutions.get(index)
                if package_script_resolutions is not None
                else None
            )
            package_match = package_matches[0]
            direct_binary = bool(
                package_match.group("tool") == "yarn"
                and package_match.group("verb") is None
                and package_match.group("script")
                in {"jest", "mocha", "vitest", "ava", "borp"}
                and resolved is not None
                and resolved.get("reason") == "package_script_missing"
            )
            if resolved is not None and resolved.get("status") != "ok" and not direct_binary:
                raise PartitionAdapterUnsupported(
                    str(resolved.get("reason") or "package_script_resolution_failed")
                )
            inferred = (
                str(package_match.group("script"))
                if direct_binary
                else str(resolved.get("runner"))
                if resolved is not None and resolved.get("runner")
                else _runner_from_package_contract(log_parser, masked)
            )
            if inferred is not None and inferred not in {
                "jest",
                "mocha",
                "vitest",
                "ava",
                "node",
                "tap",
                "hardhat",
                "borp",
            }:
                raise PartitionAdapterUnsupported("package_script_runner_unsupported")
            if inferred is not None:
                command_hits = [
                    (
                        inferred,
                        (
                            "direct_binary"
                            if direct_binary
                            else
                            "package_script_expanded"
                            if resolved is not None
                            else "package_script"
                        ),
                    )
                ]
        if _MAKE_COMMAND.search(masked) and not command_hits:
            inferred = _runner_from_log_parser(log_parser)
            if inferred == "go":
                command_hits.append((inferred, "make_wrapper"))
        if not command_hits and _runner_from_log_parser(log_parser) == "pytest":
            if re.search(
                r"(?<![\w./-])(?:tox|nox)(?:\s|$)|"
                r"(?<![\w./-])(?:python(?:\d+(?:\.\d+)*)?|bash|sh)\s+[^;&|]+|"
                r"(?:^|\s)\.?/?[^\s;&|]+\.(?:py|sh)(?:\s|$)",
                masked,
            ):
                command_hits.append(("pytest_wrapper", "python_wrapper"))
        families = {runner for runner, _ in command_hits}
        if len(families) > 1:
            raise PartitionAdapterUnsupported("multiple_test_runners")
        if command_hits:
            runner, resolution = command_hits[0]
            resolved = (
                package_script_resolutions.get(index)
                if package_script_resolutions is not None
                else None
            )
            hits.append(
                RunnerInvocation(
                    runner=runner,
                    command_index=index,
                    resolution=resolution,
                    script_path=(resolved.get("path") if resolved else None),
                    script_name=(resolved.get("name") if resolved else None),
                    script_hash=(resolved.get("hash") if resolved else None),
                    script_chain=tuple(
                        str(item.get("name"))
                        for item in (resolved.get("chain") or [])
                        if isinstance(item, dict) and item.get("name")
                    ) if resolved else (),
                )
            )
    if not hits:
        raise PartitionAdapterUnsupported("runner_not_recognized")
    return RunnerPlan(invocations=tuple(hits))


def _selector(runner: str, test_ids: tuple[str, ...]) -> str:
    if not test_ids:
        return ""
    normalized_ids = tuple(_selector_test_id(runner, name) for name in test_ids)
    if runner == "pytest_wrapper":
        return ""
    regex = "^(?:" + "|".join(re.escape(name) for name in normalized_ids) + ")$"
    if runner == "pytest" and not any(
        _pytest_test_id_is_parser_truncated(name) for name in test_ids
    ):
        return " ".join(shlex.quote(name) for name in normalized_ids)
    if runner == "go":
        # Official Go contracts commonly include both ``TestParent`` and all
        # of its subtests. Selecting the parent is exact for that contract and
        # avoids a selector that grows with every child name.
        normalized_set = set(normalized_ids)
        normalized_ids = tuple(
            name
            for name in normalized_ids
            if "/" not in name or name.split("/", 1)[0] not in normalized_set
        )
        split_names = [name.split("/") for name in normalized_ids]
        max_depth = max(len(parts) for parts in split_names)
        components = []
        for depth in range(max_depth):
            values = sorted(
                {
                    parts[depth]
                    for parts in split_names
                    if depth < len(parts)
                }
            )
            # Go's RE2 syntax does not support non-capturing ``(?:...)``.
            components.append(
                "^(" + "|".join(re.escape(value) for value in values) + ")$"
            )
        regex = "/".join(components)
        return "-run " + shlex.quote(regex)
    test_files = tuple(
        sorted(
            {
                name.split("::", 1)[0].replace("\\", "/")
                for name in test_ids
                if "::" in name and name.split("::", 1)[0]
            }
        )
    )
    file_selector = " ".join(shlex.quote(path) for path in test_files)
    if runner == "pytest":
        if not file_selector:
            raise PartitionAdapterUnsupported(
                "pytest_parser_alias_test_file_unavailable"
            )
        return file_selector
    if runner in {"jest", "vitest"} and file_selector:
        return (
            "--runTestsByPath " + file_selector + " --runInBand"
            if runner == "jest"
            else file_selector
        )
    if runner in {"jest", "vitest"}:
        suffix_regex = "(?:^|.*\\s)(?:" + "|".join(re.escape(name) for name in normalized_ids) + ")$"
        selector = (
            ("--runTestsByPath " + file_selector + " " if runner == "jest" and file_selector else "")
            + (file_selector + " " if runner == "vitest" and file_selector else "")
            + "--testNamePattern "
            + shlex.quote(suffix_regex)
        )
        if runner == "jest":
            selector += " --json --outputFile=/tmp/rllm/partition-jest-results.json --runInBand"
        return selector
    if runner in {"mocha", "hardhat"} and file_selector:
        return file_selector
    if runner in {"mocha", "hardhat"}:
        suffix_regex = "(?:^|.*\\s)(?:" + "|".join(re.escape(name) for name in normalized_ids) + ")$"
        # Do not force a second reporter here. A number of materialized npm
        # scripts already configure Mocha's reporter internally and older
        # Mocha versions reject duplicate --reporter options. The private
        # grader records either native JSON or the isolated summary counts.
        return (file_selector + " " if file_selector else "") + "--grep " + shlex.quote(suffix_regex)
    if runner == "tap":
        greps = " ".join(
            "--grep=" + shlex.quote("^(?:" + re.escape(name) + ")$")
            for name in normalized_ids
        )
        return " ".join(value for value in (file_selector, greps, "--reporter=tap") if value)
    if runner == "borp":
        if not file_selector:
            raise PartitionAdapterUnsupported("borp_test_file_unavailable")
        return file_selector + " --reporter=tap"
    if runner == "ava":
        # AVA's --match consumes title patterns rather than a regular
        # expression. Repeating an exact (non-wildcard) title is the native
        # way to select a finite contract without accidentally treating regex
        # punctuation as literal title text.
        return " ".join(
            "--match " + shlex.quote(name) for name in normalized_ids
        ) + " --tap"
    if runner == "node":
        suffix_regex = "(?:^|.*\\s)(?:" + "|".join(re.escape(name) for name in normalized_ids) + ")$"
        return (
            "--test-name-pattern "
            + shlex.quote(suffix_regex)
            + " --test-reporter=tap"
        )
    if runner == "maven":
        return "-Dtest=" + shlex.quote(",".join(normalized_ids))
    if runner == "gradle":
        return " ".join("--tests " + shlex.quote(name) for name in normalized_ids)
    if runner in {"phpunit", "pest"}:
        return "--filter " + shlex.quote(regex)
    raise PartitionAdapterUnsupported("runner_not_supported")


_TIMING_SUFFIX = re.compile(
    r"(?:\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]|"
    r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)|"
    r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\))$",
    flags=re.IGNORECASE,
)


def _pytest_test_id_is_parser_truncated(name: str) -> bool:
    """Detect dataset ids truncated at whitespace inside parametrization."""

    normalized = _TIMING_SUFFIX.sub("", name).strip()
    return normalized.count("[") != normalized.count("]")


def _selection_strategy(runner: str, test_ids: tuple[str, ...]) -> str:
    if runner == "pytest" and any(
        _pytest_test_id_is_parser_truncated(name) for name in test_ids
    ):
        return "pytest_parser_alias_file_scope"
    has_files = any("::" in name and name.split("::", 1)[0] for name in test_ids)
    if runner in {"jest", "vitest", "mocha", "hardhat"} and has_files:
        return "node_file_scope"
    return "exact_selector"


def _selector_test_id(runner: str, name: str) -> str:
    normalized = _TIMING_SUFFIX.sub("", name).strip()
    if runner == "go":
        # Official logs may prefix the package.  ``go test -run`` selects the
        # test/subtest hierarchy inside a package, so retain only that suffix.
        match = re.search(r"((?:Test|Example|Fuzz)[^\s:]*(?:/[^\s]+)*)$", normalized)
        return match.group(1) if match else normalized
    if runner in {"jest", "mocha", "vitest", "ava", "node", "tap", "hardhat", "borp"}:
        if "::" in normalized:
            normalized = normalized.split("::", 1)[1]
        # Convert common official-parser hierarchy separators to the runner's
        # full-name separator.  The selector itself is suffix-anchored because
        # Jest/Vitest/Mocha prepend describe blocks to the leaf test name.
        # Only spaced hierarchy delimiters are structural. Literal comparison
        # operators and titles such as ``<div>`` or ``>0 index`` must survive.
        return re.sub(r"\s+(?:>|›)\s+", " ", normalized).strip()
    if runner in {"maven", "gradle"}:
        parenthesized = re.fullmatch(r"(.+?)\(\)\s+\(([^()]+)\)", normalized)
        if parenthesized:
            method, class_name = parenthesized.groups()
            return f"{class_name}#{method}" if runner == "maven" else f"{class_name}.{method}"
        display = re.fullmatch(r"([^>]+?)\s*>\s*(.+?)(?:\(\))?", normalized)
        if display:
            class_name, method = (part.strip() for part in display.groups())
            return f"{class_name}#{method}" if runner == "maven" else f"{class_name}.{method}"
        kotlin_display = re.fullmatch(r"(.+?)\s+\(([^()]+)\)", normalized)
        if kotlin_display:
            display_name, class_name = kotlin_display.groups()
            return f"{class_name}#{display_name}" if runner == "maven" else f"{class_name}.{display_name}"
    return normalized


def _insert_selector(
    command: str,
    runner: str,
    selector: str,
    *,
    known_test_paths: frozenset[str] = frozenset(),
) -> str:
    pattern = _RUNNER_PATTERNS[runner]
    masked = _shell_mask(command)
    if runner == "node":
        command = re.sub(
            r"(?<!\S)--test-reporter(?:=|\s+)(?:'[^']*'|\"[^\"]*\"|[^\s;&|]+)",
            "",
            command,
        )
        command = re.sub(
            r"(?<!\S)--test-reporter-destination(?:=|\s+)(?:'[^']*'|\"[^\"]*\"|[^\s;&|]+)",
            "",
            command,
        )
        masked = _shell_mask(command)
    if runner == "jest" and re.search(
        r"(?:^|\s)(?:--runInBand\b|--maxWorkers(?:=|\s))",
        masked,
    ):
        # Jest rejects --runInBand together with --maxWorkers. Preserve the
        # materialized concurrency choice when one is already explicit.
        selector = selector.replace(" --runInBand", "")
    match = pattern.search(masked)
    if match is not None:
        if runner == "pytest":
            boundary = re.search(r"&&|\|\||(?<!\|)\|(?!\|)|;|\n", masked[match.end() :])
            insertion = match.end() + boundary.start() if boundary is not None else len(command)
            prefix = command[: match.start()]
            segment = command[match.start() : insertion]
            suffix = command[insertion:]
            try:
                tokens = shlex.split(segment)
            except ValueError as exc:
                raise PartitionAdapterUnsupported("unsafe_shell_syntax") from exc
            retained: list[str] = []
            for token in tokens:
                test_path = token.split("::", 1)[0]
                if test_path in known_test_paths:
                    continue
                retained.append(token)
            return prefix + shlex.join(retained) + " " + selector + (" " if suffix else "") + suffix.lstrip()
        if runner == "go":
            existing = re.search(
                r"(?P<prefix>(?:^|\s)-run(?:=|\s+))"
                r"(?P<value>'[^']*'|\"[^\"]*\"|[^\s;&|]+)",
                command,
            )
            if existing is not None:
                try:
                    replacement = shlex.quote(shlex.split(selector)[1])
                except (IndexError, ValueError) as exc:
                    raise PartitionAdapterUnsupported("invalid_go_selector") from exc
                return (
                    command[: existing.start("value")]
                    + replacement
                    + command[existing.end("value") :]
                )
        if runner == "maven":
            existing = re.search(r"(?:^|\s)-Dtest=([^\s;&|]+)", masked)
            if existing is not None:
                replacement = selector.split("=", 1)[1]
                return command[: existing.start(1)] + replacement + command[existing.end(1) :]
        if runner == "gradle":
            if re.search(r"(?:^|\s)--tests(?:=|\s)", masked):
                raise PartitionAdapterUnsupported("existing_selector")
            boundary = re.search(r"&&|\|\||;|\n", masked[match.end() :])
            insertion = match.end() + boundary.start() if boundary is not None else len(command)
            return command[:insertion].rstrip() + " " + selector + " " + command[insertion:].lstrip()
        if runner in {"jest", "vitest"} and "--testNamePattern" in masked:
            raise PartitionAdapterUnsupported("existing_selector")
        if runner in {"mocha", "hardhat", "tap"} and re.search(r"(?:^|\s)--grep(?:=|\s)", masked):
            raise PartitionAdapterUnsupported("existing_selector")
        if runner == "ava" and re.search(r"(?:^|\s)--match(?:=|\s)", masked):
            raise PartitionAdapterUnsupported("existing_selector")
        if runner == "node" and re.search(
            r"(?:^|\s)--test-name-pattern(?:=|\s)", masked
        ):
            raise PartitionAdapterUnsupported("existing_selector")
        if runner in {"phpunit", "pest"} and re.search(r"(?:^|\s)--filter(?:=|\s)", masked):
            raise PartitionAdapterUnsupported("existing_selector")
        return command[: match.end()] + " " + selector + command[match.end() :]
    wrapper = _wrapper_match(masked, runner)
    if wrapper is not None:
        boundary = re.search(
            r"&&|\|\||(?<!\|)\|(?!\|)|;|\n",
            masked[wrapper.end() :],
        )
        insertion = wrapper.end() + boundary.start() if boundary else len(command)
        return command[:insertion].rstrip() + " " + selector + " " + command[insertion:].lstrip()
    package = _PACKAGE_SCRIPT.search(masked)
    if package is None or runner not in {
        "jest",
        "mocha",
        "vitest",
        "ava",
        "tap",
        "hardhat",
        "borp",
    }:
        if runner == "go" and _MAKE_COMMAND.search(masked):
            try:
                go_regex = shlex.split(selector)[1]
            except (IndexError, ValueError) as exc:
                raise PartitionAdapterUnsupported("invalid_go_selector") from exc
            return (
                "GOFLAGS="
                + shlex.quote("-json -run=" + go_regex)
                + " "
                + command
            )
        raise PartitionAdapterUnsupported("runner_invocation_not_found")
    tool = package.group("tool")
    invocation = command[package.start() : package.end()]
    # JS/TS shadow probes install a PATH-scoped Node-Yarn guard before the
    # materialized grader runs. Keep the source invocation intact here so old
    # Node images without Corepack can still use their native Yarn binary.
    prefix = command[: package.start()] + invocation
    tail = masked[package.end() :]
    existing_separator = re.match(r"\s+--(?:\s+|$)", tail)
    if existing_separator is not None:
        insertion = package.end() + existing_separator.end()
        return prefix + command[package.end() : insertion] + selector + " " + command[insertion:]
    separator = " --" if tool in {"npm", "pnpm", "yarn"} else ""
    return prefix + separator + " " + selector + command[package.end() :]


def _derived_instance(
    instance: dict[str, Any],
    commands: list[str],
    command_index: int,
    runner: str,
    test_ids: tuple[str, ...],
    *,
    partition: str,
    shard_index: int = 0,
    shard_count: int = 1,
    isolate_command: bool = False,
    selector_override: str | None = None,
) -> dict[str, Any]:
    selector = _selector(runner, test_ids) if selector_override is None else selector_override
    if len(selector.encode("utf-8")) > MAX_SELECTOR_BYTES:
        raise PartitionAdapterUnsupported("selector_too_long")
    derived = copy.deepcopy(instance)
    selected_commands = list(commands)
    all_ids = tuple(str(name) for name in (instance.get("FAIL_TO_PASS") or [])) + tuple(
        str(name) for name in (instance.get("PASS_TO_PASS") or [])
    )
    milestone = instance.get("milestone") if isinstance(instance.get("milestone"), dict) else {}
    known_test_paths = {
        str(name).split("::", 1)[0]
        for name in (*all_ids, *(milestone.get("test_file_names") or []))
        if isinstance(name, str) and name
    }
    if runner != "pytest_wrapper":
        selected_commands[command_index] = _insert_selector(
            selected_commands[command_index],
            runner,
            selector,
            known_test_paths=frozenset(known_test_paths),
        )
    if runner in {"pytest", "pytest_wrapper"}:
        # If the source suite already opts into xdist, make the bounded
        # selector probe serial. This is the safe PyBaMM recovery variant and
        # also avoids treating xdist internal errors as test outcomes.
        selected_commands[command_index] = re.sub(
            r"(?<!\S)(?:-n|--numprocesses)(?:=|\s+)(?:auto|logical|\d+)",
            "-n 0",
            selected_commands[command_index],
        )
    if runner == "go" and " -json" not in _shell_mask(selected_commands[command_index]):
        go_match = _RUNNER_PATTERNS["go"].search(
            _shell_mask(selected_commands[command_index])
        )
        if go_match is not None:
            selected_commands[command_index] = (
                selected_commands[command_index][: go_match.end()]
                + " -json"
                + selected_commands[command_index][go_match.end() :]
            )
    original = derived["install_config"].get("test_cmd")
    if isolate_command:
        derived["install_config"]["test_cmd"] = selected_commands[command_index]
    else:
        derived["install_config"]["test_cmd"] = selected_commands[0] if isinstance(original, str) else selected_commands
    if partition == "combined":
        selected = set(test_ids)
        derived["FAIL_TO_PASS"] = [
            str(name)
            for name in instance.get("FAIL_TO_PASS") or []
            if str(name) in selected
        ]
        derived["PASS_TO_PASS"] = [
            str(name)
            for name in instance.get("PASS_TO_PASS") or []
            if str(name) in selected
        ]
    else:
        derived["FAIL_TO_PASS"] = list(test_ids) if partition == "f2p" else []
        derived["PASS_TO_PASS"] = list(test_ids) if partition == "p2p" else []
    derived["rllm_partition_probe"] = {
        "schema_version": 3,
        "adapter": runner,
        "partition": partition,
        "expected": len(test_ids),
        "shard_index": shard_index,
        "shard_count": shard_count,
    }
    return derived


def _selector_shards(runner: str, test_ids: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    """Split selectors at test-id boundaries below the argv cap.

    Constructing the full selector after every appended id is quadratic for
    contracts such as FHIR's ~1,500 Jest tests. Once Go parent ids are ordered
    before their children, selector size is monotone; binary-search each shard
    boundary so planning remains O(N log N).
    """

    if not test_ids:
        return ()
    # Go's ``-run`` treats slash-delimited components hierarchically. Group
    # children under one top-level test so parent/child events are collected
    # once and cannot inflate counts across shards.
    if runner == "go" and any("/" in _selector_test_id(runner, name) for name in test_ids):
        by_parent: dict[str, list[str]] = {}
        for name in test_ids:
            normalized = _selector_test_id(runner, name)
            parent = normalized.split("/", 1)[0]
            by_parent.setdefault(parent, []).append(name)

        # Parent-present groups are exact because Go executes their complete
        # hierarchy and the contract already names the parent. Put a parent
        # before its children so adding ids cannot later shrink the selector.
        packable: list[str] = []
        for parent, names in by_parent.items():
            packable.extend(
                sorted(
                    names,
                    key=lambda name: _selector_test_id(runner, name) != parent,
                )
            )
        ordered_ids = tuple(packable)
    else:
        ordered_ids = test_ids

    groups: list[tuple[str, ...]] = []
    start = 0
    while start < len(ordered_ids):
        single = ordered_ids[start : start + 1]
        if len(_selector(runner, single).encode("utf-8")) > MAX_SELECTOR_BYTES:
            raise PartitionAdapterUnsupported("selector_too_long")
        low = start + 1
        high = len(ordered_ids)
        best = start + 1
        while low <= high:
            middle = (low + high) // 2
            candidate = ordered_ids[start:middle]
            if len(_selector(runner, candidate).encode("utf-8")) <= MAX_SELECTOR_BYTES:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        groups.append(ordered_ids[start:best])
        start = best
    candidates = tuple(groups)
    if any(len(_selector(runner, group).encode("utf-8")) > MAX_SELECTOR_BYTES for group in candidates):
        raise PartitionAdapterUnsupported("selector_too_long")
    return candidates


def _balanced_selector_shards(
    runner: str,
    test_ids: tuple[str, ...],
    target_count: int,
) -> tuple[tuple[str, ...], ...]:
    """Split an already-safe selector plan into bounded recovery shards.

    Normal probes retain the minimum number of shards imposed by argv limits.
    Timeout recovery may request more shards so one slow partition is not
    retried as another near-full-suite command. Go hierarchy selectors are
    deliberately left intact because splitting a parent from its children can
    execute the same subtests twice.
    """

    groups = list(_selector_shards(runner, test_ids))
    if runner == "go":
        return tuple(groups)
    target_count = max(len(groups), min(int(target_count), len(test_ids)))
    while len(groups) < target_count:
        candidates = [
            (len(group), index)
            for index, group in enumerate(groups)
            if len(group) > 1
        ]
        if not candidates:
            break
        _, index = max(candidates)
        group = groups[index]
        split_at = (len(group) + 1) // 2
        groups[index : index + 1] = [group[:split_at], group[split_at:]]
    return tuple(groups)


def _derived_shards(
    instance: dict[str, Any],
    commands: list[str],
    command_index: int,
    runner: str,
    test_ids: tuple[str, ...],
    *,
    partition: str,
    isolate_command: bool = False,
    target_shard_count: int | None = None,
) -> tuple[PartitionShard, ...]:
    groups = (
        _balanced_selector_shards(runner, test_ids, target_shard_count)
        if target_shard_count is not None
        else _selector_shards(runner, test_ids)
    )
    return tuple(
        PartitionShard(
            test_ids=group,
            instance=_derived_instance(
                instance,
                commands,
                command_index,
                runner,
                group,
                partition=partition,
                shard_index=index,
                shard_count=len(groups),
                isolate_command=isolate_command,
            ),
            runner=runner,
        )
        for index, group in enumerate(groups)
    )


def _assign_test_ids(
    plan: RunnerPlan,
    commands: list[str],
    test_ids: tuple[str, ...],
) -> dict[RunnerInvocation, tuple[str, ...]]:
    if len(plan.invocations) == 1:
        return {plan.invocations[0]: test_ids}

    assigned: dict[RunnerInvocation, list[str]] = {
        invocation: [] for invocation in plan.invocations
    }
    go_invocations = [
        invocation for invocation in plan.invocations if invocation.runner == "go"
    ]
    non_go_invocations = [
        invocation for invocation in plan.invocations if invocation.runner != "go"
    ]
    for name in test_ids:
        go_name = _selector_test_id("go", name)
        if (
            len(go_invocations) == 1
            and len(non_go_invocations) == 1
            and re.match(r"^(?:Test|Example|Fuzz)", go_name)
        ):
            assigned[go_invocations[0]].append(name)
            continue
        if len(go_invocations) == 1 and len(non_go_invocations) == 1:
            assigned[non_go_invocations[0]].append(name)
            continue

        path = name.split("::", 1)[0].strip()
        normalized_path = path.replace("\\", "/").lstrip("./")
        go_package_match = re.match(
            r"^(?P<package>\S+)\s+(?:Test|Example|Fuzz)",
            name,
        )
        go_package = (
            go_package_match.group("package").replace("\\", "/").lstrip("./")
            if go_package_match is not None
            else ""
        )
        path_matches: list[RunnerInvocation] = []
        for invocation in plan.invocations:
            command = _shell_mask(commands[invocation.command_index]).replace(
                "\\", "/"
            )
            script_root = (
                invocation.script_path.rsplit("/", 1)[0].lstrip("./")
                if invocation.script_path and "/" in invocation.script_path
                else ""
            )
            command_globs = [
                token.strip("'\"").lstrip("./")
                for token in command.split()
                if any(character in token for character in "*?[")
            ]
            owns_path = bool(
                go_package
                and invocation.runner == "go"
                and (
                    go_package in command
                    or any(
                        token.lstrip("./").rstrip("/") == go_package.rstrip("/")
                        for token in command.split()
                    )
                )
                or normalized_path
                and (
                    path in command
                    or normalized_path in command
                    or script_root
                    and (
                        normalized_path == script_root
                        or normalized_path.startswith(script_root.rstrip("/") + "/")
                    )
                    or any(
                        fnmatch.fnmatch(normalized_path, pattern)
                        for pattern in command_globs
                    )
                )
            )
            if owns_path:
                path_matches.append(invocation)
        if len(path_matches) == 1:
            assigned[path_matches[0]].append(name)
            continue
        raise PartitionAdapterUnsupported("runner_test_assignment_ambiguous")
    return {
        invocation: tuple(names)
        for invocation, names in assigned.items()
        if names
    }


def build_partition_adapter(
    instance: dict[str, Any],
    *,
    baseline_log: str = "",
    recovery_command_budget: int | None = None,
) -> PartitionAdapter:
    # ``baseline_log`` remains in the public signature for compatibility, but
    # runner selection intentionally never consults failure output.
    del baseline_log
    f2p = tuple(str(name) for name in instance.get("FAIL_TO_PASS") or [])
    p2p = tuple(str(name) for name in instance.get("PASS_TO_PASS") or [])
    if not f2p:
        raise PartitionAdapterUnsupported("empty_f2p")
    all_ids = (*f2p, *p2p)
    if any(not name.strip() or name == UNKNOWN_TEST_ID or UNKNOWN_TEST_ID in name for name in all_ids):
        raise PartitionAdapterUnsupported("unusable_test_id")
    if len(set(all_ids)) != len(all_ids):
        raise PartitionAdapterUnsupported("ambiguous_test_id")
    commands = _commands(instance)
    config = instance.get("install_config")
    assert isinstance(config, dict)
    plan = _detect_runner_plan(
        commands,
        str(config.get("log_parser") or ""),
    )
    isolate_command = len(plan.invocations) > 1
    f2p_assignments = _assign_test_ids(plan, commands, f2p)
    p2p_assignments = _assign_test_ids(plan, commands, p2p)
    assignments = [
        ("f2p", invocation, names)
        for invocation, names in f2p_assignments.items()
    ] + [
        ("p2p", invocation, names)
        for invocation, names in p2p_assignments.items()
    ]
    target_shards: dict[tuple[str, RunnerInvocation], int] = {
        (partition, invocation): len(_selector_shards(invocation.runner, names))
        for partition, invocation, names in assignments
    }
    if recovery_command_budget is not None:
        if isinstance(recovery_command_budget, bool) or recovery_command_budget <= 0:
            raise ValueError("recovery_command_budget must be a positive integer")
        # Ten tests per command is intentionally a planning target rather than
        # a correctness boundary. The hard upper bound remains the configured
        # command budget, and normal non-timeout adapters are unchanged.
        while sum(target_shards.values()) < recovery_command_budget:
            candidates = [
                (
                    len(names) / target_shards[(partition, invocation)],
                    partition,
                    invocation,
                )
                for partition, invocation, names in assignments
                if invocation.runner != "go"
                and target_shards[(partition, invocation)] < len(names)
                and target_shards[(partition, invocation)]
                < math.ceil(len(names) / 10)
            ]
            if not candidates:
                break
            _, partition, invocation = max(
                candidates,
                key=lambda candidate: candidate[0],
            )
            target_shards[(partition, invocation)] += 1
    f2p_shards = tuple(
        shard
        for invocation, names in f2p_assignments.items()
        for shard in _derived_shards(
            instance,
            commands,
            invocation.command_index,
            invocation.runner,
            names,
            partition="f2p",
            isolate_command=isolate_command,
            target_shard_count=target_shards[("f2p", invocation)],
        )
    )
    p2p_shards = tuple(
        shard
        for invocation, names in p2p_assignments.items()
        for shard in _derived_shards(
            instance,
            commands,
            invocation.command_index,
            invocation.runner,
            names,
            partition="p2p",
            isolate_command=isolate_command,
            target_shard_count=target_shards[("p2p", invocation)],
        )
    )
    return PartitionAdapter(
        name=plan.name,
        f2p=f2p,
        p2p=p2p,
        f2p_shards=f2p_shards,
        p2p_shards=p2p_shards,
        # Kept as an empty compatibility view. Runtime probes deliberately do
        # not construct or execute a combined selector: F2P and P2P are
        # independent evidence domains.
        combined_shards=(),
        plan=plan,
    )


def _compact_literal_regex(names: tuple[str, ...]) -> str:
    """A trie factors literal prefixes without broadening the selected set."""
    if any(len(name) > 512 for name in names):
        return "(?:" + "|".join(re.escape(name) for name in names) + ")"
    trie: dict[str, Any] = {}
    for name in names:
        node = trie
        for character in name:
            node = node.setdefault(character, {})
        node[""] = {}

    def render(node: dict[str, Any]) -> str:
        suffixes: dict[str, list[str]] = {}
        for character, child in sorted(node.items()):
            if character:
                suffixes.setdefault(render(child), []).append(re.escape(character))
        alternatives = []
        for suffix, prefixes in suffixes.items():
            prefix = prefixes[0] if len(prefixes) == 1 else "(?:" + "|".join(prefixes) + ")"
            alternatives.append(prefix + suffix)
        value = alternatives[0] if len(alternatives) == 1 else "(?:" + "|".join(alternatives) + ")" if alternatives else ""
        return "(?:" + value + ")?" if "" in node and value else value

    return render(trie)


def build_non_python_partition_adapter(
    instance: dict[str, Any], *, ava_test_files: tuple[str, ...] = (),
) -> PartitionAdapter:
    """Opt-in normalized shadow planner; default/Python plans are unchanged."""
    legacy = build_partition_adapter(instance)
    if legacy.name == "go":
        return legacy
    commands = _commands(instance)
    f2p = legacy.f2p
    p2p = legacy.p2p
    outputs: dict[str, list[PartitionShard]] = {"f2p": [], "p2p": []}
    for partition, names in (("f2p", f2p), ("p2p", p2p)):
        assignments = _assign_test_ids(legacy.plan, commands, names)
        for invocation, selected in assignments.items():
            runner = invocation.runner
            if runner not in {"jest", "vitest", "mocha", "ava"}:
                raise PartitionAdapterUnsupported("non_python_partition_runner_unsupported")
            groups: list[tuple[tuple[str, ...], str]] = []
            if runner == "ava":
                by_file: dict[str, list[tuple[str, str]]] = {}
                for name in selected:
                    matches = []
                    for path in ava_test_files:
                        stem = re.sub(r"\.[cm]?[jt]sx?$", "", path)
                        parts = stem.split("/")
                        # AVA displays paths relative to the common test root.
                        for offset in (0, 1):
                            prefix = " › ".join(parts[offset:]) + " › "
                            if name.startswith(prefix):
                                matches.append((path, name[len(prefix):]))
                    matches = list(dict.fromkeys(matches))
                    if len(matches) != 1:
                        raise PartitionAdapterUnsupported("ava_test_file_ownership_unavailable")
                    path, title = matches[0]
                    if any(char in title for char in "*?[]{}!\\"):
                        raise PartitionAdapterUnsupported("ava_title_glob_ambiguous")
                    by_file.setdefault(path, []).append((name, title))
                for path, rows in by_file.items():
                    selector = shlex.quote(path) + " " + " ".join("--match " + shlex.quote(title) for _, title in rows) + " --tap"
                    groups.append((tuple(name for name, _ in rows), selector))
            else:
                # Pack compressed selectors, rather than reverting to the
                # old uncompressed shards when the entire partition is large.
                def compressed_selector(ids: tuple[str, ...], runner: str = runner) -> str:
                    aliases = tuple(_selector_test_id(runner, name) for name in ids)
                    regex = "(?:^|.*\\s)(?:" + _compact_literal_regex(aliases) + ")$"
                    value = ("--grep " if runner == "mocha" else "--testNamePattern ") + shlex.quote(regex)
                    if runner == "jest":
                        value += " --json --outputFile=/tmp/rllm/partition-jest-results.json --runInBand"
                    return value

                start = 0
                while start < len(selected):
                    low, high = start + 1, len(selected)
                    best, best_selector = start, ""
                    while low <= high:
                        middle = (low + high) // 2
                        selector = compressed_selector(selected[start:middle])
                        if len(selector.encode()) <= MAX_SELECTOR_BYTES:
                            best, best_selector = middle, selector
                            low = middle + 1
                        else:
                            high = middle - 1
                    if best == start:
                        raise PartitionAdapterUnsupported("selector_too_long")
                    groups.append((selected[start:best], best_selector))
                    start = best
            for index, (ids, selector) in enumerate(groups):
                derived = _derived_instance(
                    instance, commands, invocation.command_index, runner, ids,
                    partition=partition, shard_index=index, shard_count=len(groups),
                    isolate_command=len(legacy.plan.invocations) > 1, selector_override=selector,
                )
                outputs[partition].append(PartitionShard(ids, derived, runner))
    return PartitionAdapter(legacy.name, f2p, p2p, tuple(outputs["f2p"]), tuple(outputs["p2p"]), (), legacy.plan)


def _integer(pattern: str, log: str) -> int:
    matches = re.findall(pattern, log, flags=re.IGNORECASE | re.MULTILINE)
    return int(matches[-1]) if matches else 0


def _aggregate_counts(runner: str, log: str) -> tuple[int, int, int, int] | None:
    """Return passed, failed, errored, skipped from a runner summary."""

    if runner == "pytest":
        # Pytest's final summary may survive a third-party teardown failure.
        summary_lines = re.findall(r"^=+\s*(.*?)\s*=+\s*$", log, flags=re.MULTILINE)
        if summary_lines:
            line = summary_lines[-1]
            passed = _integer(r"(\d+)\s+passed", line)
            failed = _integer(r"(\d+)\s+failed", line)
            errored = _integer(r"(\d+)\s+errors?", line)
            skipped = _integer(r"(\d+)\s+skipped", line)
            if passed + failed + errored + skipped:
                return passed, failed, errored, skipped
    if runner == "go":
        passed = len(re.findall(r"^\s*--- PASS:", log, flags=re.MULTILINE))
        failed = len(re.findall(r"^\s*--- FAIL:", log, flags=re.MULTILINE))
        skipped = len(re.findall(r"^\s*--- SKIP:", log, flags=re.MULTILINE))
        return (passed, failed, 0, skipped) if passed + failed + skipped else None
    if runner == "jest":
        passed = _integer(r"Tests:[^\n]*?(\d+)\s+passed", log)
        failed = _integer(r"Tests:[^\n]*?(\d+)\s+failed", log)
        skipped = _integer(r"Tests:[^\n]*?(\d+)\s+skipped", log)
        total = _integer(r"Tests:.*?(\d+)\s+total", log)
        if total:
            return passed, failed, max(0, total - passed - failed - skipped), skipped
    if runner == "mocha":
        passed = _integer(r"^\s*(\d+)\s+passing\b", log)
        failed = _integer(r"^\s*(\d+)\s+failing\b", log)
        skipped = _integer(r"^\s*(\d+)\s+pending\b", log)
        return (passed, failed, 0, skipped) if passed + failed + skipped else None
    if runner == "ava":
        passed = _integer(r"(?:^|\n)\s*(\d+)\s+(?:tests?\s+)?passed\b", log)
        failed = _integer(r"(?:^|\n)\s*(\d+)\s+(?:tests?\s+)?failed\b", log)
        skipped = _integer(r"(?:^|\n)\s*(\d+)\s+(?:tests?\s+)?skipped\b", log)
        return (passed, failed, 0, skipped) if passed + failed + skipped else None
    if runner == "vitest":
        summary = re.findall(r"^\s*Tests\s+(.+)$", log, flags=re.MULTILINE)
        if summary:
            line = summary[-1]
            return (
                _integer(r"(\d+)\s+passed", line),
                _integer(r"(\d+)\s+failed", line),
                _integer(r"(\d+)\s+error", line),
                _integer(r"(\d+)\s+skipped", line),
            )
    if runner in {"maven", "gradle"}:
        rows = re.findall(
            r"Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+),\s*Skipped:\s*(\d+)",
            log,
            flags=re.IGNORECASE,
        )
        if rows:
            total, failed, errored, skipped = map(int, rows[-1])
            return max(0, total - failed - errored - skipped), failed, errored, skipped
    if runner in {"phpunit", "pest"}:
        rows = re.findall(r"Tests:\s*(\d+)([^\n]*)", log, flags=re.IGNORECASE)
        if rows:
            total = int(rows[-1][0])
            detail = rows[-1][1]
            failed = _integer(r"Failures:\s*(\d+)", detail)
            errored = _integer(r"Errors:\s*(\d+)", detail)
            skipped = _integer(r"Skipped:\s*(\d+)", detail)
            skipped += _integer(r"Incomplete:\s*(\d+)", detail)
            return max(0, total - failed - errored - skipped), failed, errored, skipped
    return None


def observe_partition(
    runner: str,
    expected_names: tuple[str, ...],
    artifact: dict[str, Any] | None,
    log: str,
    *,
    partition: str | None = None,
) -> PartitionObservation:
    expected = set(expected_names)
    raw_results = artifact.get("test_results") if isinstance(artifact, dict) else None
    if not isinstance(raw_results, dict) and runner == "go":
        raw_results = {
            name: {"PASS": "PASSED", "FAIL": "FAILED", "SKIP": "SKIPPED"}[status]
            for status, name in re.findall(
                r"^\s*--- (PASS|FAIL|SKIP):\s+(.+?)\s+\([^\n]+\)\s*$",
                log,
                flags=re.MULTILINE,
            )
        }
    all_named_results = {
        str(name): str(status).upper() for name, status in (raw_results.items() if isinstance(raw_results, dict) else []) if str(status).upper() in {"PASSED", "FAILED", "ERROR", "SKIPPED"}
    }
    named_results = {name: status for name, status in all_named_results.items() if name in expected}
    statuses = list(named_results.values())
    named = (
        statuses.count("PASSED"),
        statuses.count("FAILED"),
        statuses.count("ERROR"),
        statuses.count("SKIPPED"),
    )
    aggregate: tuple[int, int, int, int] | None = None
    aggregate_source: str | None = None
    diagnostics = artifact.get("diagnostics") if isinstance(artifact, dict) else None
    evidence = diagnostics.get("partition_observation") if isinstance(diagnostics, dict) else None
    if isinstance(evidence, dict):
        if artifact.get("result_adapter_version") != PARTITION_RESULT_ADAPTER_VERSION:
            raise ValueError("partition_observation_result_adapter_mismatch")
        if evidence.get("schema_version") != PARTITION_OBSERVATION_SCHEMA_VERSION:
            raise ValueError("partition_observation_schema_mismatch")
        if evidence.get("runner") != runner:
            raise ValueError("partition_observation_runner_mismatch")
        if partition is not None and evidence.get("partition") != partition:
            raise ValueError("partition_observation_partition_mismatch")
        if evidence.get("expected") != len(expected_names):
            raise ValueError("partition_observation_expected_mismatch")
        if evidence.get("selector_verified") is not True:
            raise ValueError("partition_observation_selector_unverified")
        collector = evidence.get("collector")
        if collector not in _TRUSTED_PARTITION_COLLECTORS.get(runner, frozenset()):
            raise ValueError("partition_observation_collector_untrusted")
        raw_counts = evidence.get("counts")
        if not isinstance(raw_counts, dict):
            raise ValueError("partition_observation_counts_invalid")
        values: list[int] = []
        for key in ("passed", "failed", "errored", "skipped"):
            value = raw_counts.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("partition_observation_counts_invalid")
            values.append(value)
        aggregate = tuple(values)  # type: ignore[assignment]
        aggregate_reported = sum(aggregate)
        if evidence.get("reported") != aggregate_reported:
            raise ValueError("partition_observation_reported_mismatch")
        if aggregate_reported > len(expected_names):
            raise ValueError(
                "partition_reported_count_exceeds_expected:"
                f"{aggregate_reported}>{len(expected_names)}"
            )
        # Any names that did map are a useful consistency check. Unmapped
        # names are intentionally tolerated here: the isolated selector and
        # baseline calibration, rather than display-name equality, establish
        # partition membership.
        if any(named[index] > aggregate[index] for index in range(4)):
            raise ValueError("partition_observation_named_count_conflict")
        aggregate_source = f"structured_aggregate:{collector}"

    # Raw textual totals alone cannot prove selector precision. Aggregate
    # counts are accepted only when the materialized private grader emits the
    # authenticated partition_observation envelope above.
    selected = aggregate if aggregate is not None else named
    reported = sum(selected)
    if reported > len(expected_names):
        raise ValueError(f"partition_reported_count_exceeds_expected:{reported}>{len(expected_names)}")
    extra_named = set(all_named_results) - expected
    if runner == "go":
        # Running one Go subtest also emits the containing TestX result.  That
        # parent is execution structure, not an accidentally selected test.
        extra_named = {
            name
            for name in extra_named
            if not any(
                expected_name.startswith(name + "/")
                or name.startswith(expected_name + "/")
                for expected_name in expected
            )
        }
    return PartitionObservation(
        counts=TestPartitionCounts(
            expected=len(expected_names),
            passed=selected[0],
            failed=selected[1],
            errored=selected[2],
            skipped=selected[3],
            not_run=len(expected_names) - reported,
        ),
        named_results=named_results,
        # A verified isolated aggregate does not require display-name
        # correspondence. Without that evidence, independent extras remain a
        # hard failure as before.
        extra_named_tests=(
            () if aggregate is not None else tuple(sorted(extra_named))
        ),
        reported=reported,
        evidence_source=aggregate_source or "named_results",
        selector_verified=aggregate is not None,
    )


def encode_instance(instance: dict[str, Any]) -> str:
    """Canonical JSON used for upload identity and deterministic tests."""

    return json.dumps(instance, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
