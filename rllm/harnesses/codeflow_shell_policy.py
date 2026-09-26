"""Best-effort policy checks for Codeflow ``execute_bash`` commands.

This is a training-policy guard, not a security boundary.  Arbitrary Python
can reproduce a file search, so the goal is to reject the ordinary shell
forms agents actually emit while preserving legitimate test-output filters
and searches outside the repository.
"""

from __future__ import annotations

import ast
import fnmatch
import json
import posixpath
import re
import shlex
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit

_CONTROL_TOKENS = {";", ";;", "&&", "||", "|", "|&", "&"}
_GROUP_OPEN = {"("}
_GROUP_CLOSE = {")"}
_REDIRECTIONS = {"<", "<<", "<<<", ">", ">>", ">&", "<&", "<>"}
_SEARCH_EXECUTABLES = {
    "ack",
    "ag",
    "egrep",
    "fd",
    "fgrep",
    "find",
    "grep",
    "locate",
    "rg",
    "ripgrep",
}
_GREP_EXECUTABLES = {"ack", "ag", "egrep", "fgrep", "grep", "rg", "ripgrep"}
CODEFLOW_TOOL_BEHAVIOR_ORDER = ("search", "read", "edit")
DEFAULT_CODEFLOW_TOOL_RESTRICTED_MODE = CODEFLOW_TOOL_BEHAVIOR_ORDER
CodeflowToolBehavior = Literal["search", "read", "edit"]
CODEFLOW_TOOL_MODES = ("structured", "bash_only")
DEFAULT_CODEFLOW_TOOL_MODE = "structured"
CodeflowToolMode = Literal["structured", "bash_only"]

_STREAM_READERS = {"bat", "cat", "head", "less", "more", "nl", "tac", "tail"}
_LIST_EXECUTABLES = {"ls", "tree"}
_SHELL_CONTROL_PREFIXES = {
    "!",
    "{",
    "}",
    "case",
    "do",
    "elif",
    "else",
    "fi",
    "for",
    "if",
    "in",
    "then",
    "until",
    "while",
}
_PYTHON_EXECUTABLES = {"python", "python3", "pypy", "pypy3"}
_CONDA_EXECUTABLES = {"conda", "mamba", "micromamba"}
_SYSTEM_PACKAGE_EXECUTABLES = {
    "apk",
    "apt",
    "apt-get",
    "dnf",
    "pacman",
    "yum",
}
_LOCAL_NETWORK_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


@dataclass(frozen=True)
class RepositorySearchViolation:
    code: str
    message: str
    executable: str | None = None


@dataclass(frozen=True)
class ShellBehaviorAnalysis:
    """Best-effort semantic classification of repository-facing shell work.

    ``repository_paths`` contains only literal paths that can be normalized
    without shell expansion.  Callers must still intersect them with a trusted
    repository snapshot before treating them as exposure provenance.
    """

    behaviors: tuple[CodeflowToolBehavior, ...] = ()
    search_paths: tuple[str, ...] = ()
    read_paths: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    operations: tuple[ShellOperation, ...] = ()


@dataclass(frozen=True)
class ShellOperation:
    command_index: int | None
    behavior: CodeflowToolBehavior
    executable: str
    paths: tuple[str, ...] = ()
    output_kind: Literal["path", "content", "none"] = "none"
    stdout_redirected: bool = False


@dataclass(frozen=True)
class ShellFileAttribution:
    path: str
    behavior: Literal["search", "read"]
    exposure_kind: Literal["path", "content"]
    source: Literal["bash_rendered_path", "bash_single_operation"]
    confidence: Literal["high", "medium", "low"]


def normalize_codeflow_tool_restricted_mode(
    value: Any = None,
) -> tuple[CodeflowToolBehavior, ...]:
    """Validate and canonicalize the public Codeflow file-tool policy list."""

    if value is None:
        return DEFAULT_CODEFLOW_TOOL_RESTRICTED_MODE
    is_list_config = (
        value.__class__.__name__ == "ListConfig"
        and value.__class__.__module__.startswith("omegaconf.")
    )
    if not isinstance(value, list) and not is_list_config:
        raise ValueError(
            "swe.codeflow_tool_restricted_mode must be a list containing only "
            "'search', 'read', and 'edit'"
        )
    requested: set[str] = set()
    for item in value:
        if not isinstance(item, str) or item not in CODEFLOW_TOOL_BEHAVIOR_ORDER:
            raise ValueError(
                "swe.codeflow_tool_restricted_mode must be a list containing only "
                "'search', 'read', and 'edit'"
            )
        requested.add(item)
    return tuple(
        behavior
        for behavior in CODEFLOW_TOOL_BEHAVIOR_ORDER
        if behavior in requested
    )


def normalize_codeflow_tool_mode(value: Any = None) -> CodeflowToolMode:
    if value is None:
        return DEFAULT_CODEFLOW_TOOL_MODE
    if not isinstance(value, str) or value not in CODEFLOW_TOOL_MODES:
        raise ValueError(
            "swe.codeflow_tool_mode must be 'structured' or 'bash_only'"
        )
    return value


@dataclass(frozen=True)
class EnvironmentPolicyViolation:
    code: str
    subtype: str
    message: str
    executable: str | None = None


@dataclass(frozen=True)
class _ShellCommand:
    words: tuple[str, ...]
    stdin_paths: tuple[str, ...]
    stdout_paths: tuple[str, ...]
    separator_before: str | None
    cwd: str


def _tokenize(command: str) -> list[str]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()<>")
    lexer.whitespace_split = True
    lexer.commenters = ""
    raw = list(lexer)
    tokens: list[str] = []
    operators = ("<<<", "&&", "||", "|&", ";;", ">>", "<<", ">&", "<&", "<>")
    for token in raw:
        if not token or any(character not in ";&|()<>" for character in token):
            tokens.append(token)
            continue
        index = 0
        while index < len(token):
            operator = next((candidate for candidate in operators if token.startswith(candidate, index)), None)
            if operator is None:
                operator = token[index]
            tokens.append(operator)
            index += len(operator)
    return tokens


def canonicalize_shell_command(command: str) -> str:
    """Return a stable fingerprint input for ordinary shell commands.

    The policy parser already tokenizes shell operators and quoting without
    executing the command.  Rejoining those tokens makes harmless whitespace
    and quote spelling differences compare equal.  Malformed shell is still a
    model action, so callers fall back to a minimally normalized raw string
    instead of dropping it from repeat accounting.
    """

    value = str(command or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not value:
        return ""
    try:
        return shlex.join(_tokenize(value))
    except ValueError:
        return value


def _normalized_root(repository_root: str) -> str:
    value = posixpath.normpath(str(repository_root or "/testbed"))
    return value if value.startswith("/") else f"/{value}"


def _path_scope(path: str, *, cwd: str, repository_root: str) -> str:
    """Return ``repo``, ``external``, or ``unknown`` for one shell path."""
    value = str(path or "").strip()
    if not value or value == "-":
        return "unknown"
    if any(marker in value for marker in ("$", "`", "${", "$(")):
        return "unknown"
    if value.startswith("~"):
        return "unknown"
    normalized = posixpath.normpath(value if value.startswith("/") else posixpath.join(cwd, value))
    root_prefix = repository_root.rstrip("/") + "/"
    return "repo" if normalized == repository_root or normalized.startswith(root_prefix) else "external"


def _strip_redirections(words: list[str]) -> tuple[list[str], list[str], list[str]]:
    command_words: list[str] = []
    stdin_paths: list[str] = []
    stdout_paths: list[str] = []
    index = 0
    while index < len(words):
        token = words[index]
        if token in _REDIRECTIONS:
            target = words[index + 1] if index + 1 < len(words) else ""
            if token.startswith("<"):
                if token in {"<<", "<<<"}:
                    pass
                elif token != "<&" or not target.lstrip("&").isdigit():
                    stdin_paths.append(target)
            elif token.startswith(">"):
                if token != ">&" or not target.lstrip("&").isdigit():
                    stdout_paths.append(target)
            index += 2
            continue
        if token.isdigit() and index + 1 < len(words) and words[index + 1] in _REDIRECTIONS:
            index += 1
            continue
        command_words.append(token)
        index += 1
    return command_words, stdin_paths, stdout_paths


def _commands(
    tokens: list[str],
    repository_root: str,
    *,
    initial_cwd: str | None = None,
) -> list[_ShellCommand]:
    commands: list[_ShellCommand] = []
    words: list[str] = []
    cwd = initial_cwd or repository_root
    cwd_stack: list[str] = []
    separator_before: str | None = None

    def flush(next_separator: str | None = None) -> None:
        nonlocal words, cwd, separator_before
        if words:
            command_words, stdin_paths, stdout_paths = _strip_redirections(words)
            commands.append(
                _ShellCommand(
                    words=tuple(command_words),
                    stdin_paths=tuple(stdin_paths),
                    stdout_paths=tuple(stdout_paths),
                    separator_before=separator_before,
                    cwd=cwd,
                )
            )
            if command_words and posixpath.basename(command_words[0]) == "cd" and len(command_words) >= 2 and next_separator != "|":
                target = command_words[1]
                if not any(marker in target for marker in ("$", "`", "~")):
                    cwd = posixpath.normpath(target if target.startswith("/") else posixpath.join(cwd, target))
        words = []
        separator_before = next_separator

    for token in tokens:
        if token in _GROUP_OPEN:
            flush()
            cwd_stack.append(cwd)
            separator_before = None
        elif token in _GROUP_CLOSE:
            flush()
            if cwd_stack:
                cwd = cwd_stack.pop()
            separator_before = None
        elif token in _CONTROL_TOKENS:
            flush(token)
        elif token == "$" and not words:
            # ``shlex`` emits the dollar from ``$(...)`` separately.  It is
            # syntax, not an executable word.
            continue
        else:
            words.append(token)
    flush()
    return commands


def _skip_options(words: list[str], index: int) -> int:
    while index < len(words) and words[index].startswith("-"):
        index += 1
    return index


def _unwrap_command(words: tuple[str, ...]) -> tuple[str, list[str]]:
    values = list(words)
    index = 0
    while index < len(values) and "=" in values[index] and not values[index].startswith(("/", "./")):
        index += 1

    while index < len(values):
        executable = posixpath.basename(values[index])
        if executable in _SHELL_CONTROL_PREFIXES:
            index += 1
            continue
        if executable == "env":
            index += 1
            while index < len(values) and values[index].startswith("-"):
                consumes_value = values[index] in {"-u", "--unset"}
                index += 2 if consumes_value else 1
            while index < len(values) and "=" in values[index] and not values[index].startswith(("/", "./")):
                index += 1
            continue
        if executable in {"command", "builtin", "nohup"}:
            index = _skip_options(values, index + 1)
            continue
        if executable == "nice":
            index += 1
            while index < len(values) and values[index].startswith("-"):
                consumes_value = values[index] in {"-n", "--adjustment"}
                index += 2 if consumes_value else 1
            continue
        if executable == "stdbuf":
            index += 1
            while index < len(values) and values[index].startswith("-"):
                consumes_value = values[index] in {"-i", "-o", "-e", "--input", "--output", "--error"}
                index += 2 if consumes_value else 1
            continue
        if executable == "sudo":
            index = _skip_options(values, index + 1)
            continue
        if executable == "timeout":
            index += 1
            while index < len(values) and values[index].startswith("-"):
                consumes_value = values[index] in {"-k", "-s", "--kill-after", "--signal"}
                index += 2 if consumes_value else 1
            if index < len(values):
                index += 1  # duration
            continue
        return executable, values[index + 1 :]
    return "", []


def _option_value(option: str, options_with_values: set[str]) -> bool:
    return option in options_with_values


def _grep_targets(executable: str, arguments: list[str]) -> tuple[list[str], bool, bool]:
    """Return explicit file targets, recursive flag, and ambiguity."""
    targets: list[str] = []
    recursive = False
    pattern_seen = False
    pattern_from_option = False
    ambiguous = False
    files_mode = executable in {"rg", "ripgrep"} and "--files" in arguments
    value_options = {
        "-A",
        "-B",
        "-C",
        "-D",
        "-d",
        "-m",
        "--after-context",
        "--before-context",
        "--binary-files",
        "--context",
        "--devices",
        "--directories",
        "--exclude",
        "--exclude-dir",
        "--include",
        "--label",
        "--max-count",
    }
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if token == "--":
            remainder = arguments[index + 1 :]
            if not pattern_seen and not pattern_from_option and remainder:
                pattern_seen = True
                remainder = remainder[1:]
            targets.extend(remainder)
            break
        if token in {"-r", "-R", "--recursive"} or (token.startswith("-") and not token.startswith("--") and "r" in token[1:]):
            recursive = True
            index += 1
            continue
        if token in {"-e", "--regexp"}:
            pattern_from_option = True
            pattern_seen = True
            index += 2
            continue
        if token.startswith("--regexp="):
            pattern_from_option = True
            pattern_seen = True
            index += 1
            continue
        if token in {"-f", "--file"}:
            if index + 1 < len(arguments):
                targets.append(arguments[index + 1])
            else:
                ambiguous = True
            index += 2
            continue
        if token.startswith("--file="):
            targets.append(token.split("=", 1)[1])
            index += 1
            continue
        if token.startswith("-"):
            if _option_value(token, value_options):
                index += 2
            else:
                index += 1
            continue
        if files_mode:
            targets.append(token)
            index += 1
            continue
        if not pattern_seen and not pattern_from_option:
            pattern_seen = True
        else:
            targets.append(token)
        index += 1

    if executable in {"rg", "ripgrep", "ag", "ack"} and not targets:
        # These tools search cwd when they are not consuming piped stdin.
        ambiguous = True
    return targets, recursive, ambiguous


def _find_targets(arguments: list[str]) -> tuple[list[str], bool]:
    targets: list[str] = []
    for token in arguments:
        if token in {"!", "("} or token.startswith("-"):
            break
        targets.append(token)
    return (targets or ["."], False)


def _find_exec_commands(arguments: list[str]) -> list[list[str]]:
    commands: list[list[str]] = []
    index = 0
    while index < len(arguments):
        if arguments[index] not in {"-exec", "-execdir"}:
            index += 1
            continue
        index += 1
        nested: list[str] = []
        while index < len(arguments) and arguments[index] not in {";", "+"}:
            nested.append(arguments[index])
            index += 1
        if nested:
            commands.append(nested)
        index += 1
    return commands


def _fd_targets(arguments: list[str]) -> tuple[list[str], bool]:
    positionals: list[str] = []
    index = 0
    value_options = {"-E", "-e", "-t", "-x", "-X", "--exclude", "--extension", "--type", "--exec", "--exec-batch"}
    while index < len(arguments):
        token = arguments[index]
        if token == "--":
            positionals.extend(arguments[index + 1 :])
            break
        if token.startswith("-"):
            index += 2 if token in value_options else 1
            continue
        positionals.append(token)
        index += 1
    # fd syntax is PATTERN [PATH...].  With one positional, PATH defaults to cwd.
    return (positionals[1:] if len(positionals) > 1 else ["."], False)


def _repo_target(targets: list[str], *, cwd: str, repository_root: str) -> tuple[bool, bool]:
    saw_unknown = False
    for target in targets:
        scope = _path_scope(target, cwd=cwd, repository_root=repository_root)
        if scope == "repo":
            return True, saw_unknown
        if scope == "unknown":
            saw_unknown = True
    return False, saw_unknown


def _stream_reader_reads_repo(executable: str, arguments: list[str], *, cwd: str, repository_root: str) -> bool:
    if executable not in _STREAM_READERS:
        return False
    candidates = [value for value in arguments if value and not value.startswith("-")]
    return _repo_target(candidates, cwd=cwd, repository_root=repository_root)[0]


def _nested_shell(arguments: list[str]) -> str | None:
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if (
            token.startswith("-")
            and not token.startswith("--")
            and "c" in token[1:]
            and index + 1 < len(arguments)
        ):
            return arguments[index + 1]
        index += 1
    return None


def _wrapped_run_command(
    executable: str,
    arguments: list[str],
) -> list[str] | None:
    """Extract commands executed by package/environment ``run`` wrappers."""
    if executable not in {"conda", "mamba", "micromamba", "poetry", "uv"}:
        return None
    try:
        index = arguments.index("run") + 1
    except ValueError:
        return None
    options_with_values = {
        "-n",
        "--name",
        "-p",
        "--prefix",
        "--directory",
        "--project",
        "--python",
    }
    while index < len(arguments):
        token = arguments[index]
        if token == "--":
            index += 1
            break
        if token in options_with_values:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        break
    return arguments[index:] or None


def _python_module_command(executable: str, arguments: list[str]) -> tuple[str, list[str]]:
    """Normalize ``python -m <module>`` into an executable-like command."""
    if executable not in _PYTHON_EXECUTABLES:
        return executable, arguments
    try:
        module_index = arguments.index("-m")
    except ValueError:
        return executable, arguments
    if module_index + 1 >= len(arguments):
        return executable, arguments
    return posixpath.basename(arguments[module_index + 1]), arguments[module_index + 2 :]


def _local_editable_pip_install(
    arguments: list[str],
    *,
    cwd: str,
    repository_root: str,
) -> bool:
    """Whether this is the one permitted install: local editable, no deps."""
    if not arguments or arguments[0] != "install" or "--no-deps" not in arguments:
        return False
    if any(
        token in {"-r", "--requirement", "-c", "--constraint"}
        or token.startswith(("--requirement=", "--constraint=", "git+", "http://", "https://"))
        for token in arguments
    ):
        return False

    editable_targets: list[str] = []
    index = 1
    options_with_values = {
        "-e",
        "--editable",
        "--config-settings",
        "--src",
    }
    consumed_values: set[int] = set()
    while index < len(arguments):
        token = arguments[index]
        if token in {"-e", "--editable"}:
            if index + 1 >= len(arguments):
                return False
            editable_targets.append(arguments[index + 1])
            consumed_values.add(index + 1)
            index += 2
            continue
        if token.startswith("--editable="):
            editable_targets.append(token.split("=", 1)[1])
            index += 1
            continue
        if token in options_with_values:
            if index + 1 >= len(arguments):
                return False
            consumed_values.add(index + 1)
            index += 2
            continue
        index += 1

    if len(editable_targets) != 1:
        return False
    target = editable_targets[0]
    if _path_scope(target, cwd=cwd, repository_root=repository_root) != "repo":
        return False
    normalized_target = posixpath.normpath(
        target if target.startswith("/") else posixpath.join(cwd, target)
    )
    if normalized_target != repository_root:
        return False

    # A second bare positional is another distribution, even if the editable
    # target itself is local. Conservatively reject options we cannot prove do
    # not introduce another package source.
    safe_value_options = {"--config-settings", "--src"}
    index = 1
    while index < len(arguments):
        token = arguments[index]
        if index in consumed_values:
            index += 1
            continue
        if token in {"-e", "--editable"}:
            index += 2
            continue
        if token in safe_value_options:
            index += 2
            continue
        if token.startswith("--editable="):
            index += 1
            continue
        if token.startswith("-"):
            index += 1
            continue
        return False
    return True


def _network_target_is_local(value: str) -> bool:
    target = str(value or "").strip()
    if not target or any(marker in target for marker in ("$", "`", "$(")):
        return False
    if target.startswith("[") and "]" in target:
        return target[1 : target.index("]")].lower() in _LOCAL_NETWORK_HOSTS
    candidate = target if "://" in target else f"//{target}"
    try:
        host = (urlsplit(candidate).hostname or "").lower()
    except ValueError:
        return False
    return host in _LOCAL_NETWORK_HOSTS


def _download_targets(arguments: list[str]) -> list[str]:
    """Extract URL/host candidates from ordinary curl/wget invocations."""
    value_options = {
        "-A",
        "-b",
        "-d",
        "-e",
        "-F",
        "-H",
        "-o",
        "-O",
        "-T",
        "-u",
        "--data",
        "--data-binary",
        "--header",
        "--output",
        "--referer",
        "--user-agent",
    }
    targets: list[str] = []
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if token == "--":
            targets.extend(arguments[index + 1 :])
            break
        if token in value_options:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        targets.append(token)
        index += 1
    return targets


def _command_substitutions(value: str) -> list[str]:
    """Extract ordinary ``$(...)`` and backtick substitutions.

    ``shlex`` deliberately does not parse shell expansions.  This small
    scanner handles the common nested forms without trying to execute or
    fully interpret the command.  Single-quoted substitutions may be
    conservatively inspected after ``shlex`` removes quote provenance; this
    guard is a policy aid rather than a complete shell security parser.
    """
    fragments: list[str] = []
    index = 0
    while index < len(value):
        if value.startswith("$(", index):
            start = index + 2
            cursor = start
            depth = 1
            quote: str | None = None
            escaped = False
            while cursor < len(value):
                character = value[cursor]
                if escaped:
                    escaped = False
                elif character == "\\" and quote != "'":
                    escaped = True
                elif quote is not None:
                    if character == quote:
                        quote = None
                elif character in {"'", '"', "`"}:
                    quote = character
                elif value.startswith("$(", cursor):
                    depth += 1
                    cursor += 1
                elif character == "(":
                    depth += 1
                elif character == ")":
                    depth -= 1
                    if depth == 0:
                        fragments.append(value[start:cursor])
                        index = cursor + 1
                        break
                cursor += 1
            else:
                # An incomplete expansion is left to the existing shell
                # validation/execution error path.
                index += 2
            continue
        if value[index] == "`":
            cursor = index + 1
            escaped = False
            while cursor < len(value):
                character = value[cursor]
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == "`":
                    fragments.append(value[index + 1 : cursor])
                    index = cursor + 1
                    break
                cursor += 1
            else:
                index += 1
            continue
        index += 1
    return fragments


def _literal_repo_path(
    value: str,
    *,
    cwd: str,
    repository_root: str,
) -> str | None:
    if _path_scope(value, cwd=cwd, repository_root=repository_root) != "repo":
        return None
    normalized = posixpath.normpath(
        value if value.startswith("/") else posixpath.join(cwd, value)
    )
    relative = posixpath.relpath(normalized, repository_root)
    return "." if relative == "." else relative


def _literal_repo_paths(
    values: Sequence[str],
    *,
    cwd: str,
    repository_root: str,
) -> list[str]:
    paths: list[str] = []
    for value in values:
        path = _literal_repo_path(
            value,
            cwd=cwd,
            repository_root=repository_root,
        )
        if path is not None and path not in paths:
            paths.append(path)
    return paths


def _option_free_arguments(
    arguments: Sequence[str],
    *,
    value_options: set[str] | None = None,
) -> list[str]:
    """Return ordinary positional arguments without executing shell parsing."""

    values: list[str] = []
    consumes = value_options or set()
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if token == "--":
            values.extend(arguments[index + 1 :])
            break
        if token in consumes:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        values.append(token)
        index += 1
    return values


def _sed_awk_program_targets(
    executable: str,
    arguments: Sequence[str],
) -> tuple[str, list[str]]:
    programs: list[str] = []
    positionals: list[str] = []
    index = 0
    value_options = (
        {"-e", "-f", "--expression", "--file"}
        if executable == "sed"
        else {"-F", "-f", "-v", "--assign", "--field-separator", "--file"}
    )
    program_options = {"-e", "--expression"}
    while index < len(arguments):
        token = arguments[index]
        if token == "--":
            positionals.extend(arguments[index + 1 :])
            break
        if token in value_options:
            if index + 1 < len(arguments) and token in program_options:
                programs.append(arguments[index + 1])
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        positionals.append(token)
        index += 1
    if programs:
        return "\n".join(programs), positionals
    if not positionals:
        return "", []
    return positionals[0], positionals[1:]


def _python_literal_path(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Call) and node.args:
        is_path_constructor = (
            isinstance(node.func, ast.Name) and node.func.id == "Path"
        ) or (
            isinstance(node.func, ast.Attribute) and node.func.attr == "Path"
        )
        if is_path_constructor:
            return _python_literal_path(node.args[0])
    return None


def _python_call_name(node: ast.Call) -> str:
    parts: list[str] = []
    current: ast.AST = node.func
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _python_behavior(
    arguments: Sequence[str],
    *,
    cwd: str,
    repository_root: str,
) -> tuple[set[CodeflowToolBehavior], list[str], list[str], list[str]]:
    try:
        index = list(arguments).index("-c")
    except ValueError:
        return set(), [], [], []
    if index + 1 >= len(arguments):
        return set(), [], [], []
    try:
        # Model-generated one-liners can contain invalid string escapes (for
        # example, a backslash followed by a backtick). Python 3.12+ reports
        # them as SyntaxWarning while
        # still producing a usable AST.  Keep that diagnostic local to the
        # untrusted snippet instead of flooding the trainer logs.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(arguments[index + 1])
    except SyntaxError:
        return set(), [], [], []

    behaviors: set[CodeflowToolBehavior] = set()
    search_paths: list[str] = []
    read_paths: list[str] = []
    evidence: list[str] = []

    def repository_scope(node: ast.AST | None, *, default_repo: bool = False) -> tuple[bool, str | None]:
        literal = _python_literal_path(node)
        if literal is None:
            return default_repo or node is not None, None
        path = _literal_repo_path(literal, cwd=cwd, repository_root=repository_root)
        return path is not None, path

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _python_call_name(node)
        short = name.rsplit(".", 1)[-1]
        receiver: ast.AST | None = None
        if isinstance(node.func, ast.Attribute):
            receiver = node.func.value

        if short in {"glob", "iglob", "rglob", "walk", "listdir", "scandir"}:
            target = node.args[0] if node.args else receiver
            applies, path = repository_scope(target, default_repo=True)
            if applies:
                behaviors.add("search")
                evidence.append(f"python:{short}")
                if path is not None:
                    search_paths.append(path)
            continue

        if short == "open" or name in {"open", "io.open"}:
            target = node.args[0] if node.args else None
            applies, path = repository_scope(target)
            if not applies:
                continue
            mode = "r"
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                mode = str(node.args[1].value)
            for keyword in node.keywords:
                if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
                    mode = str(keyword.value.value)
            behavior: CodeflowToolBehavior = (
                "edit" if any(marker in mode for marker in "wax+") else "read"
            )
            behaviors.add(behavior)
            evidence.append(f"python:open:{behavior}")
            if behavior == "read" and path is not None:
                read_paths.append(path)
            continue

        if short in {"read_text", "read_bytes"}:
            applies, path = repository_scope(receiver)
            if applies:
                behaviors.add("read")
                evidence.append(f"python:{short}")
                if path is not None:
                    read_paths.append(path)
            continue

        if short in {"copy", "copy2", "copyfile"} and node.args:
            applies, path = repository_scope(node.args[0])
            if applies:
                behaviors.add("read")
                evidence.append(f"python:{short}:read")
                if path is not None:
                    read_paths.append(path)

        if short in {
            "chmod",
            "copy",
            "copy2",
            "copyfile",
            "link",
            "makedirs",
            "mkdir",
            "move",
            "remove",
            "rename",
            "replace",
            "rmdir",
            "rmtree",
            "symlink",
            "touch",
            "unlink",
            "write_bytes",
            "write_text",
        }:
            target = receiver if short in {"chmod", "mkdir", "rename", "replace", "rmdir", "touch", "unlink", "write_bytes", "write_text"} else (node.args[-1] if node.args else None)
            applies, _ = repository_scope(target)
            if applies:
                behaviors.add("edit")
                evidence.append(f"python:{short}")

    return behaviors, search_paths, read_paths, evidence


def _resolve_literal_shell_assignments(command: str) -> str:
    """Resolve simple scalar assignments without executing shell expansion."""

    values: dict[str, str] = {}
    assignment = re.compile(
        r"(?:^|(?<=[;&|()]))\s*([A-Za-z_][A-Za-z0-9_]*)="
        r"(?:'([^']*)'|\"([^\"$`]*)\"|([^\s;&|()]+))"
    )
    for match in assignment.finditer(command):
        single_quoted, double_quoted, bare = match.groups()[1:]
        value = next(
            (
                item
                for item in (single_quoted, double_quoted, bare)
                if item is not None
            ),
            "",
        )
        # Single quotes make every byte literal. Double-quoted and bare
        # assignments are accepted only when no shell expansion remains.
        if single_quoted is not None or not any(
            marker in value for marker in ("$", "`", "~")
        ):
            values[match.group(1)] = value
    resolved = command
    for name, value in values.items():
        replacement = shlex.quote(value)
        resolved = re.sub(
            rf"\$(?:\{{{re.escape(name)}\}}|{re.escape(name)}\b)",
            lambda _match, replacement=replacement: replacement,
            resolved,
        )
    return resolved


def analyze_repository_shell_behaviors(
    command: str,
    repository_root: str,
) -> ShellBehaviorAnalysis:
    """Classify explicit repository search/read/edit operations in Bash.

    The classifier intentionally describes the model's shell operation, not
    incidental file I/O performed by programs such as pytest.  Repository
    snapshots remain authoritative for actual edits.
    """

    root = _normalized_root(repository_root)
    behaviors: set[CodeflowToolBehavior] = set()
    search_paths: list[str] = []
    read_paths: list[str] = []
    evidence: list[str] = []
    operations: list[ShellOperation] = []
    pending_python_operations: list[ShellOperation] = []
    command = _resolve_literal_shell_assignments(command)

    # Preserve the mature, conservative repository-search decision while the
    # walk below gathers richer multi-label provenance.
    search_violation = detect_repository_search(command, root)
    if search_violation is not None:
        behaviors.add("search")
        evidence.append(f"search:{search_violation.executable or 'shell'}")

    for match in re.finditer(
        r"(?ms)\b(?:python|python3|pypy|pypy3)\b[^\n]*<<-?\s*['\"]?(?P<tag>[A-Za-z_][A-Za-z0-9_]*)['\"]?\s*\n"
        r"(?P<body>.*?)\n(?P=tag)\s*(?:\n|$)",
        command,
    ):
        python_behaviors, python_search, python_read, python_evidence = (
            _python_behavior(
                ["-c", match.group("body")],
                cwd=root,
                repository_root=root,
            )
        )
        behaviors.update(python_behaviors)
        search_paths.extend(
            path for path in python_search if path not in search_paths
        )
        read_paths.extend(path for path in python_read if path not in read_paths)
        evidence.extend(python_evidence)
        redirected = ">" in match.group(0).splitlines()[0]
        if python_search:
            pending_python_operations.append(
                ShellOperation(
                    command_index=None,
                    behavior="search",
                    executable="python",
                    paths=tuple(dict.fromkeys(python_search)),
                    output_kind="path",
                    stdout_redirected=redirected,
                )
            )
        if python_read:
            pending_python_operations.append(
                ShellOperation(
                    command_index=None,
                    behavior="read",
                    executable="python",
                    paths=tuple(dict.fromkeys(python_read)),
                    output_kind="content",
                    stdout_redirected=redirected,
                )
            )

    def add_path(destination: list[str], value: str) -> None:
        if value not in destination:
            destination.append(value)

    def record_operation(
        command_index: int,
        behavior: CodeflowToolBehavior,
        executable: str,
        paths: Sequence[str],
        *,
        output_kind: Literal["path", "content", "none"],
        stdout_redirected: bool,
    ) -> None:
        candidate = ShellOperation(
            command_index=command_index,
            behavior=behavior,
            executable=executable,
            paths=tuple(dict.fromkeys(paths)),
            output_kind=output_kind,
            stdout_redirected=stdout_redirected,
        )
        if candidate not in operations:
            operations.append(candidate)

    def inspect(fragment: str, initial_cwd: str) -> None:
        nonlocal next_command_index
        try:
            commands = _commands(
                _tokenize(fragment),
                root,
                initial_cwd=initial_cwd,
            )
        except ValueError:
            return
        for shell_command in commands:
            command_index = next_command_index
            next_command_index += 1
            executable, arguments = _unwrap_command(shell_command.words)
            if not executable:
                continue

            for nested in _command_substitutions(" ".join(shell_command.words)):
                inspect(nested, shell_command.cwd)
            if executable in {"bash", "dash", "sh", "zsh"}:
                nested = _nested_shell(arguments)
                if nested:
                    inspect(nested, shell_command.cwd)
                continue
            wrapped = _wrapped_run_command(executable, arguments)
            if wrapped is not None:
                inspect(shlex.join(wrapped), shell_command.cwd)
                continue
            if executable == "xargs":
                nested_index = next(
                    (i for i, value in enumerate(arguments) if not value.startswith("-")),
                    None,
                )
                if nested_index is not None:
                    inspect(shlex.join(arguments[nested_index:]), shell_command.cwd)

            stdin_paths = _literal_repo_paths(
                shell_command.stdin_paths,
                cwd=shell_command.cwd,
                repository_root=root,
            )
            if stdin_paths:
                behaviors.add("read")
                evidence.append(f"read:{executable}:stdin")
                for path in stdin_paths:
                    add_path(read_paths, path)

            output_paths = _literal_repo_paths(
                shell_command.stdout_paths,
                cwd=shell_command.cwd,
                repository_root=root,
            )
            if output_paths:
                behaviors.add("edit")
                evidence.append(f"edit:{executable}:redirect")

            if executable in _LIST_EXECUTABLES:
                targets = _option_free_arguments(arguments) or ["."]
                paths = _literal_repo_paths(
                    targets, cwd=shell_command.cwd, repository_root=root
                )
                if paths:
                    behaviors.add("search")
                    evidence.append(f"search:{executable}")
                    for path in paths:
                        add_path(search_paths, path)
                    record_operation(
                        command_index,
                        "search",
                        executable,
                        paths,
                        output_kind="path",
                        stdout_redirected=bool(shell_command.stdout_paths),
                    )

            if executable in _SEARCH_EXECUTABLES:
                if executable == "find":
                    targets, _ = _find_targets(arguments)
                    nested_commands = _find_exec_commands(arguments)
                    for nested_words in nested_commands:
                        inspect(shlex.join(nested_words), shell_command.cwd)
                elif executable == "fd":
                    targets, _ = _fd_targets(arguments)
                else:
                    targets, _, _ = _grep_targets(executable, arguments)
                resolved_search_paths = _literal_repo_paths(
                    targets, cwd=shell_command.cwd, repository_root=root
                )
                if (
                    not resolved_search_paths
                    and executable in {"rg", "ripgrep", "ag", "ack"}
                    and not stdin_paths
                ):
                    resolved_search_paths = ["."]
                for path in resolved_search_paths:
                    add_path(search_paths, path)
                if resolved_search_paths:
                    record_operation(
                        command_index,
                        "search",
                        executable,
                        resolved_search_paths,
                        output_kind="path",
                        stdout_redirected=bool(shell_command.stdout_paths),
                    )

            if executable == "git":
                operation = next(
                    (value for value in arguments if value and not value.startswith("-")),
                    "",
                )
                if operation == "grep":
                    behaviors.add("search")
                    evidence.append("search:git grep")
                    add_path(search_paths, ".")
                    record_operation(
                        command_index,
                        "search",
                        "git grep",
                        ["."],
                        output_kind="path",
                        stdout_redirected=bool(shell_command.stdout_paths),
                    )
                elif operation in {"diff", "show", "status", "blame", "log"}:
                    behaviors.add("read")
                    evidence.append(f"read:git {operation}")
                    separator = arguments.index("--") if "--" in arguments else -1
                    targets = arguments[separator + 1 :] if separator >= 0 else []
                    if separator < 0 and operation == "blame":
                        operation_index = arguments.index(operation)
                        targets = _option_free_arguments(
                            arguments[operation_index + 1 :],
                            value_options={"-L", "--contents", "--date"},
                        )
                    for path in _literal_repo_paths(
                        targets, cwd=shell_command.cwd, repository_root=root
                    ):
                        add_path(read_paths, path)
                    git_paths = _literal_repo_paths(
                        targets, cwd=shell_command.cwd, repository_root=root
                    )
                    if operation == "show" and not git_paths:
                        for value in arguments:
                            if ":" not in value or value.startswith("-"):
                                continue
                            _, candidate = value.split(":", 1)
                            path = _literal_repo_path(
                                candidate,
                                cwd=shell_command.cwd,
                                repository_root=root,
                            )
                            if path is not None:
                                git_paths.append(path)
                                add_path(read_paths, path)
                    record_operation(
                        command_index,
                        "read",
                        f"git {operation}",
                        git_paths,
                        output_kind="path" if operation == "status" else "content",
                        stdout_redirected=bool(shell_command.stdout_paths),
                    )
                elif operation == "apply":
                    behaviors.add("edit")
                    evidence.append("edit:git apply")

            if executable in _STREAM_READERS | {
                "cmp",
                "cut",
                "diff",
                "file",
                "stat",
                "strings",
                "wc",
            }:
                value_options = {
                    "-n",
                    "--bytes",
                    "--format",
                    "--lines",
                    "--max-unchanged-stats",
                    "--sleep-interval",
                    "-c",
                    "-F",
                }
                targets = _option_free_arguments(arguments, value_options=value_options)
                paths = _literal_repo_paths(
                    targets, cwd=shell_command.cwd, repository_root=root
                )
                if paths:
                    behaviors.add("read")
                    evidence.append(f"read:{executable}")
                    for path in paths:
                        add_path(read_paths, path)
                    content_readers = _STREAM_READERS | {"cut", "diff", "strings"}
                    record_operation(
                        command_index,
                        "read",
                        executable,
                        paths,
                        output_kind="content" if executable in content_readers else "none",
                        stdout_redirected=bool(shell_command.stdout_paths),
                    )

            if executable in {"sed", "awk", "gawk", "mawk", "nawk"}:
                in_place = executable == "sed" and any(
                    token == "--in-place"
                    or token.startswith("--in-place=")
                    or (token.startswith("-") and "i" in token[1:])
                    for token in arguments
                )
                program, targets = _sed_awk_program_targets(
                    executable,
                    arguments,
                )
                paths = _literal_repo_paths(
                    targets, cwd=shell_command.cwd, repository_root=root
                )
                if in_place and paths:
                    behaviors.add("edit")
                    evidence.append("edit:sed")
                elif paths:
                    is_search = bool(
                        re.search(r"/(?:[^/\\]|\\.)+/", program)
                        or re.search(r"(?:^|\s)(?:match\s*\(|[^\s]+\s*[~!]=?\s*)", program)
                    )
                    behavior: CodeflowToolBehavior = "search" if is_search else "read"
                    behaviors.add(behavior)
                    evidence.append(f"{behavior}:{executable}")
                    for path in paths:
                        add_path(search_paths if is_search else read_paths, path)
                    record_operation(
                        command_index,
                        behavior,
                        executable,
                        paths,
                        output_kind="path" if is_search else "content",
                        stdout_redirected=bool(shell_command.stdout_paths),
                    )

            edit_value_options = {
                "chmod": {"--reference"},
                "chown": {"--reference"},
                "cp": {"--target-directory", "-t"},
                "install": {"--target-directory", "-t"},
                "ln": {"--target-directory", "-t"},
                "mv": {"--target-directory", "-t"},
                "rm": set(),
                "rmdir": set(),
                "touch": {"--date", "--reference", "-d", "-r", "-t"},
                "truncate": {"--size", "-s", "--reference", "-r"},
            }
            if executable in edit_value_options:
                targets = _option_free_arguments(
                    arguments,
                    value_options=edit_value_options[executable],
                )
                if executable in {"cp", "install"} and len(targets) >= 2:
                    source_paths = _literal_repo_paths(
                        targets[:-1],
                        cwd=shell_command.cwd,
                        repository_root=root,
                    )
                    if source_paths:
                        behaviors.add("read")
                        evidence.append(f"read:{executable}")
                        for path in source_paths:
                            add_path(read_paths, path)
                        record_operation(
                            command_index,
                            "read",
                            executable,
                            source_paths,
                            output_kind="none",
                            stdout_redirected=bool(shell_command.stdout_paths),
                        )
                if executable in {"chmod", "chown"} and targets:
                    targets = targets[1:]
                elif executable in {"cp", "install", "ln"} and targets:
                    targets = targets[-1:]
                if _literal_repo_paths(
                    targets, cwd=shell_command.cwd, repository_root=root
                ):
                    behaviors.add("edit")
                    evidence.append(f"edit:{executable}")
            if executable in {"patch", "apply_patch"} and _path_scope(
                ".", cwd=shell_command.cwd, repository_root=root
            ) == "repo":
                behaviors.add("edit")
                evidence.append(f"edit:{executable}")

            if executable in _PYTHON_EXECUTABLES:
                python_behaviors, python_search, python_read, python_evidence = (
                    _python_behavior(
                        arguments,
                        cwd=shell_command.cwd,
                        repository_root=root,
                    )
                )
                behaviors.update(python_behaviors)
                for path in python_search:
                    add_path(search_paths, path)
                for path in python_read:
                    add_path(read_paths, path)
                evidence.extend(python_evidence)
                if python_search:
                    record_operation(
                        command_index,
                        "search",
                        executable,
                        python_search,
                        output_kind="path",
                        stdout_redirected=bool(shell_command.stdout_paths),
                    )
                if python_read:
                    record_operation(
                        command_index,
                        "read",
                        executable,
                        python_read,
                        output_kind="content",
                        stdout_redirected=bool(shell_command.stdout_paths),
                    )

    next_command_index = 0
    inspect(command, root)
    for operation in pending_python_operations:
        if operation not in operations:
            operations.append(operation)
    ordered = tuple(
        behavior
        for behavior in CODEFLOW_TOOL_BEHAVIOR_ORDER
        if behavior in behaviors
    )
    return ShellBehaviorAnalysis(
        behaviors=ordered,
        search_paths=tuple(search_paths),
        read_paths=tuple(read_paths),
        evidence=tuple(dict.fromkeys(evidence)),
        operations=tuple(operations),
    )


def is_bounded_read_only_head_pipeline(
    command: str,
    repository_root: str,
) -> bool:
    """Recognize the narrow pipeline whose producer may observe SIGPIPE.

    This is intentionally stricter than general shell classification: the
    final command must be ``head`` consuming stdin, it must not write through
    redirection, and all positionals must be option values rather than files.
    Repository snapshots and policy evidence remain authoritative at the
    replay call site.
    """

    root = _normalized_root(repository_root)
    try:
        commands = _commands(_tokenize(command), root)
    except ValueError:
        return False
    if len(commands) < 2:
        return False
    consumer = commands[-1]
    if consumer.separator_before not in {"|", "|&"} or consumer.stdout_paths:
        return False
    executable, arguments = _unwrap_command(consumer.words)
    if executable != "head":
        return False
    index = 0
    while index < len(arguments):
        value = arguments[index]
        if value == "--":
            return index == len(arguments) - 1
        if re.fullmatch(r"-\d+", value) or re.fullmatch(r"--(?:lines|bytes)=\d+", value):
            index += 1
            continue
        if value in {"-n", "--lines", "-c", "--bytes"}:
            if index + 1 >= len(arguments) or not re.fullmatch(r"\d+", arguments[index + 1]):
                return False
            index += 2
            continue
        if value.startswith("-"):
            # Formatting switches do not turn head into a file reader.
            index += 1
            continue
        return False
    return True


def is_proven_side_effect_free_shell_command(
    command: str,
    repository_root: str,
) -> bool:
    """Recognize observation-only shell commands for shadow replay.

    This predicate is deliberately narrower than the general behavior
    analyzer.  It is used only when primary and shadow produced different
    exit/status observations but their Git-visible state is identical.  Every
    command in the shell expression must be drawn from a small allow-list and
    may not write through a redirection or invoke mutation-capable variants.

    Package installation, downloads, arbitrary Python, nested shells and
    command substitutions are intentionally rejected: those operations can
    change non-Git environment state even when the repository fingerprint is
    stable.
    """

    value = str(command or "").strip()
    if not value or _command_substitutions(value):
        return False
    root = _normalized_root(repository_root)
    try:
        commands = _commands(_tokenize(value), root)
    except ValueError:
        return False
    if not commands:
        return False

    allowed_simple = {
        "cd",
        "cut",
        "env",
        "file",
        "grep",
        "egrep",
        "fgrep",
        "head",
        "ls",
        "nl",
        "ps",
        "pwd",
        "stat",
        "tail",
        "tree",
        "wc",
        "which",
    }
    read_only_git = {
        "diff",
        "grep",
        "log",
        "ls-files",
        "rev-parse",
        "show",
        "status",
    }
    read_only_pip = {"freeze", "list", "show"}

    for shell_command in commands:
        if shell_command.stdin_paths or any(
            path != "/dev/null" for path in shell_command.stdout_paths
        ):
            return False
        if shell_command.separator_before not in {None, "&&", "|", "|&"}:
            return False
        executable, arguments = _unwrap_command(shell_command.words)
        if not executable:
            return False
        if executable in allowed_simple:
            continue
        if executable == "find":
            if any(
                token in {"-delete", "-exec", "-execdir", "-fls", "-fprint", "-fprint0"}
                for token in arguments
            ):
                return False
            continue
        if executable == "git":
            index = 0
            while index < len(arguments):
                token = arguments[index]
                if token in {"-C", "-c", "--git-dir", "--work-tree"}:
                    index += 2
                    continue
                if token.startswith("-"):
                    index += 1
                    continue
                break
            if index >= len(arguments) or arguments[index] not in read_only_git:
                return False
            continue
        if executable in {"pip", "pip3"}:
            operation = next(
                (token for token in arguments if token and not token.startswith("-")),
                "",
            )
            if operation not in read_only_pip:
                return False
            continue
        if executable in _PYTHON_EXECUTABLES:
            if arguments not in (["--version"], ["-V"]):
                return False
            continue
        return False
    return True


def _bash_visible_stdout(observation: str) -> str:
    value = str(observation or "")
    marker = "\n[stdout]\n"
    if marker not in value:
        return value
    stdout = value.split(marker, 1)[1]
    if "\n[stderr]\n" in stdout:
        stdout = stdout.split("\n[stderr]\n", 1)[0]
    return stdout


def shell_file_attributions(
    analysis: ShellBehaviorAnalysis,
    observation: str,
    repository_root: str,
    repository_files: Sequence[str],
) -> tuple[ShellFileAttribution, ...]:
    """Resolve file-level Bash evidence against a complete repository inventory."""

    known = {posixpath.normpath(str(path)) for path in repository_files}
    if not known:
        return ()
    stdout = _bash_visible_stdout(observation)
    visible_stdout = stdout.strip()
    attributions: list[ShellFileAttribution] = []

    def normalize(candidate: str) -> str | None:
        value = str(candidate or "").strip().strip("'\"")
        if not value:
            return None
        if value.startswith(repository_root.rstrip("/") + "/"):
            value = posixpath.relpath(value, repository_root)
        value = posixpath.normpath(value.removeprefix("./"))
        if value in known:
            return value
        if "/" not in value:
            basename_matches = [
                path for path in known if posixpath.basename(path) == value
            ]
            if len(basename_matches) == 1:
                return basename_matches[0]
        return None

    def operation_files(operation: ShellOperation) -> set[str]:
        resolved: set[str] = set()
        for target in operation.paths:
            value = posixpath.normpath(str(target).removeprefix("./"))
            if value == ".":
                continue
            if any(marker in value for marker in ("*", "?", "[")):
                resolved.update(path for path in known if fnmatch.fnmatch(path, value))
            elif value in known:
                resolved.add(value)
        return resolved

    def operation_matches(operation: ShellOperation, path: str) -> bool:
        if not operation.paths:
            return True
        for target in operation.paths:
            value = posixpath.normpath(str(target).removeprefix("./"))
            if value == "." or path == value or path.startswith(value.rstrip("/") + "/"):
                return True
            if any(marker in value for marker in ("*", "?", "[")) and fnmatch.fnmatch(path, value):
                return True
        return False

    def add(
        path: str,
        operation: ShellOperation,
        *,
        source: Literal["bash_rendered_path", "bash_single_operation"],
        confidence: Literal["high", "medium", "low"],
    ) -> None:
        if operation.behavior not in {"search", "read"}:
            return
        kind: Literal["path", "content"] = (
            "content"
            if operation.behavior == "read" and operation.output_kind == "content"
            else "path"
        )
        candidate = ShellFileAttribution(
            path=path,
            behavior=operation.behavior,
            exposure_kind=kind,
            source=source,
            confidence=confidence,
        )
        if candidate not in attributions:
            attributions.append(candidate)

    # Common grep/rg/find/git-status/diff output formats expose a path at the
    # beginning of a complete line.  Intersecting with the trusted snapshot
    # prevents arbitrary stdout from claiming unrelated repository files.
    rendered_paths: list[str] = []
    ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    for line in stdout.splitlines():
        stripped = ansi.sub("", line).strip()
        if not stripped or stripped.startswith("... [truncated"):
            continue
        tree_path = re.sub(r"^(?:[|`+\\\-\s]|├|└|─|│)+", "", stripped)
        candidates = (
            [stripped.split(":", 1)[0]]
            if ":" in stripped
            else [stripped]
        )
        if tree_path and tree_path != stripped:
            candidates.append(tree_path)
        if re.match(r"^[bcdlps-][rwxStTs-]{9}[.+@]?\s+", stripped):
            candidates.extend(
                suffix
                for path in known
                for suffix in (path, posixpath.basename(path))
                if stripped.endswith(" " + suffix)
            )
        if len(stripped) > 3 and stripped[:2].strip() in {"M", "A", "D", "R", "??"}:
            candidates.append(stripped[3:])
        if stripped.startswith(("+++ ", "--- ")):
            candidates.append(stripped[4:].removeprefix("a/").removeprefix("b/"))
        for candidate in candidates:
            path = normalize(candidate)
            if path is not None and path not in rendered_paths:
                rendered_paths.append(path)

    operations = [
        operation
        for operation in analysis.operations
        if operation.behavior in {"search", "read"}
    ]
    for path in rendered_paths:
        matching = [operation for operation in operations if operation_matches(operation, path)]
        behaviors = {operation.behavior for operation in matching}
        if len(behaviors) != 1:
            continue
        candidates = [
            operation
            for operation in matching
            if not operation.stdout_redirected and operation.output_kind != "none"
        ]
        if not candidates:
            continue
        add(path, candidates[0], source="bash_rendered_path", confidence="high")

    eligible = [
        operation
        for operation in operations
        if not operation.stdout_redirected and operation.output_kind != "none"
    ]
    if visible_stdout and len(eligible) == 1:
        operation = eligible[0]
        files = operation_files(operation)
        if len(files) == 1:
            add(
                next(iter(files)),
                operation,
                source="bash_single_operation",
                confidence="high",
            )

    high_keys = {(item.path, item.behavior) for item in attributions if item.confidence == "high"}
    for operation in operations:
        for path in sorted(operation_files(operation)):
            if (path, operation.behavior) in high_keys:
                continue
            add(path, operation, source="bash_single_operation", confidence="medium")
    return tuple(attributions)


def shell_exposed_repository_files(
    analysis: ShellBehaviorAnalysis,
    observation: str,
    repository_root: str,
    repository_files: Sequence[str],
) -> tuple[str, ...]:
    """Compatibility projection of high-confidence Bash file exposure."""

    return tuple(
        dict.fromkeys(
            item.path
            for item in shell_file_attributions(
                analysis,
                observation,
                repository_root,
                repository_files,
            )
            if item.confidence == "high"
        )
    )


def detect_preconfigured_environment_violation(
    command: str,
    repository_root: str,
) -> EnvironmentPolicyViolation | None:
    """Reject ordinary environment mutation/download commands in eval.

    This is deliberately a high-coverage policy aid, not a network security
    boundary.  The sandbox platform remains responsible for actual egress
    controls; this guard gives a non-adversarial agent immediate feedback
    instead of letting it spend many turns attempting dependency installs.
    """
    root = _normalized_root(repository_root)
    return _detect_preconfigured_environment_violation(command, root, root)


def detect_denovo_detached_process_violation(
    command: str,
    repository_root: str,
) -> EnvironmentPolicyViolation | None:
    """Reject agent-owned processes that could outlive the vLLM session.

    This guard is enabled only for the deferred-primary-verifier DeNovo path;
    bug-repair command policy remains unchanged.
    """

    root = _normalized_root(repository_root)
    try:
        tokens = _tokenize(command)
        commands = _commands(tokens, root)
    except ValueError:
        return None
    for index, token in enumerate(tokens):
        if token == "&" and (
            index + 1 >= len(tokens) or tokens[index + 1] != ">"
        ):
            return EnvironmentPolicyViolation(
                code="bash_detached_process_attempt",
                subtype="detached_process",
                message=(
                    "execute_bash cannot start a background process while "
                    "DeNovo primary verification is deferred"
                ),
            )
    banned = {"nohup", "setsid", "disown"}
    for shell_command in commands:
        words = list(shell_command.words)
        if any(posixpath.basename(word) in banned for word in words[:8]):
            return EnvironmentPolicyViolation(
                code="bash_detached_process_attempt",
                subtype="detached_process",
                message=(
                    "execute_bash cannot use nohup, setsid, or disown while "
                    "DeNovo primary verification is deferred"
                ),
            )
        executable, arguments = _unwrap_command(shell_command.words)
        if executable in {"bash", "dash", "sh", "zsh"}:
            nested = _nested_shell(arguments)
            if nested:
                violation = detect_denovo_detached_process_violation(
                    nested,
                    root,
                )
                if violation is not None:
                    return violation
    return None


def detect_repo_generation_package_violation(
    command: str,
    metadata: dict[str, Any],
) -> EnvironmentPolicyViolation | None:
    """Block retrieval or introspection of the package being regenerated.

    Unlike the preconfigured-evaluation policy, this deliberately permits
    third-party dependency installation.  Only identifiers belonging to the
    target distribution/repository (plus unconditional package-index source
    downloads) are rejected, matching the DeNovoSWE/NL2Repo threat model.
    """

    raw_identifiers: list[str] = []

    def add(value: Any) -> None:
        if not isinstance(value, str):
            return
        value = value.strip()
        if not value:
            return
        value = re.sub(r"^git\+", "", value, flags=re.IGNORECASE)
        vcs_match = re.search(
            r"(?:github|gitlab|bitbucket|codeberg)\.(?:com|org)/"
            r"([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?(?:/|$|[?#])",
            value,
            flags=re.IGNORECASE,
        )
        if vcs_match:
            owner, repo = vcs_match.groups()
            raw_identifiers.extend((repo, f"{owner}/{repo}"))
            return
        value = value.split("[", 1)[0].removesuffix(".git").strip("/")
        if "/" in value and not value.startswith(("/", "./", "../")):
            parts = value.split("/")
            if len(parts) == 2:
                raw_identifiers.extend((value, parts[-1]))
                return
        raw_identifiers.append(value)

    for key in (
        "pypi_name",
        "package_name",
        "repo_name",
        "repo",
    ):
        add(metadata.get(key))
    for key in ("pypi_name_candidates", "import_names"):
        values = metadata.get(key) or []
        if isinstance(values, str):
            try:
                values = json.loads(values)
            except (TypeError, ValueError):
                values = [values]
        if isinstance(values, Sequence) and not isinstance(values, str | bytes):
            for value in values:
                add(value.get("name") if isinstance(value, dict) else value)

    variants: set[str] = set()
    for identifier in raw_identifiers:
        for value in {
            identifier,
            identifier.replace("-", "_"),
            identifier.replace("_", "-"),
            identifier.replace(".", "_"),
            identifier.replace(".", "-"),
        }:
            if len(value) >= 3 and not any(char.isspace() for char in value):
                variants.add(value)
    if not variants:
        return None

    target = "(?:" + "|".join(
        rf"(?<![A-Za-z0-9_]){re.escape(value)}(?![A-Za-z0-9_])"
        for value in sorted(variants, key=lambda item: (-len(item), item))
    ) + ")"
    package_fetch = (
        r"(?:\bpip\d*\b|\bpython\d*\s+-m\s+pip\b|\bpipx\b|"
        r"\buv(?:\s+pip)?\b|\buvx\b|\bpoetry\b|"
        r"\b(?:conda|mamba|micromamba)\b|\bpdm\b|\bhatch\b|"
        r"\beasy_install\d*\b)[^\n]*(?:install|download|show|wheel|"
        r"fetch|add|sync|search|run|index\s+versions)[^\n]*"
    )
    downloader = r"\b(?:curl|wget|aria2c?|http|httpie|lwp-download|fetch)\b"
    distribution_host = (
        r"(?:pypi\.org|pypi\.python\.org|test\.pypi\.org|"
        r"files\.pythonhosted\.org|pythonhosted\.org|pypistats\.org|"
        r"conda\.anaconda\.org|anaconda\.org|repo\.anaconda\.com)"
    )
    checks: tuple[tuple[str, str], ...] = (
        (rf"{package_fetch}{target}", "target package manager access"),
        (
            rf"\bgit\s+(?:clone|submodule\s+add)\b[^\n]*{target}",
            "target repository clone",
        ),
        (rf"{downloader}[^\n]*{target}", "target source download"),
        (rf"{downloader}[^\n]*{distribution_host}", "package-index download"),
        (
            rf"\bpython\d*\b[^\n]*\s-c\b[^\n]*{target}[^\n]*"
            r"(?:inspect\.getsource|__file__|importlib|find_spec|get_data|pkgutil)",
            "installed target source introspection",
        ),
        (
            rf"\bpython\d*\b[^\n]*\s-c\b[^\n]*"
            r"(?:inspect\.getsource|__file__|importlib|find_spec|get_data|pkgutil)"
            rf"[^\n]*{target}",
            "installed target source introspection",
        ),
        (
            rf"\b(?:unzip|tar|gunzip)\b[^\n]*{target}[^\n]*"
            r"\.(?:whl|tar\.gz|tgz|zip)\b",
            "cached target archive extraction",
        ),
    )
    for pattern, operation in checks:
        if re.search(pattern, command, flags=re.IGNORECASE):
            return EnvironmentPolicyViolation(
                code="bash_target_package_source_attempt",
                subtype="target_source_access",
                message=(
                    f"execute_bash cannot perform {operation}; implement the "
                    "target repository only from the supplied specification"
                ),
            )
    return None


def _detect_preconfigured_environment_violation(
    command: str,
    repository_root: str,
    initial_cwd: str,
) -> EnvironmentPolicyViolation | None:
    try:
        commands = _commands(
            _tokenize(command),
            repository_root,
            initial_cwd=initial_cwd,
        )
    except ValueError:
        lowered = command.lower()
        suspicious = (
            "pip install",
            "conda install",
            "mamba install",
            "apt install",
            "apt-get install",
            "curl ",
            "wget ",
        )
        if any(fragment in lowered for fragment in suspicious):
            return _environment_violation(
                "bash_environment_mutation_attempt",
                "environment_mutation",
                "unparseable dependency installation or download command",
            )
        return None

    for shell_command in commands:
        executable, arguments = _unwrap_command(shell_command.words)
        if not executable:
            continue

        for nested in _command_substitutions(" ".join(shell_command.words)):
            violation = _detect_preconfigured_environment_violation(
                nested,
                repository_root,
                shell_command.cwd,
            )
            if violation is not None:
                return violation

        if executable in {"bash", "sh", "dash", "zsh"}:
            nested = _nested_shell(arguments)
            if nested:
                violation = _detect_preconfigured_environment_violation(
                    nested,
                    repository_root,
                    shell_command.cwd,
                )
                if violation is not None:
                    return violation
            continue

        if executable == "find":
            for nested_words in _find_exec_commands(arguments):
                violation = _detect_preconfigured_environment_violation(
                    shlex.join(nested_words),
                    repository_root,
                    shell_command.cwd,
                )
                if violation is not None:
                    return violation

        if executable == "xargs":
            nested_index = next(
                (
                    index
                    for index, value in enumerate(arguments)
                    if not value.startswith("-")
                ),
                None,
            )
            if nested_index is not None:
                violation = _detect_preconfigured_environment_violation(
                    shlex.join(arguments[nested_index:]),
                    repository_root,
                    shell_command.cwd,
                )
                if violation is not None:
                    return violation
            continue

        executable, arguments = _python_module_command(executable, arguments)
        if executable in _PYTHON_EXECUTABLES and arguments:
            script = posixpath.basename(arguments[0])
            operation = arguments[1] if len(arguments) > 1 else ""
            if script == "setup.py" and operation in {
                "develop",
                "easy_install",
                "install",
            }:
                return _environment_violation(
                    "bash_environment_mutation_attempt",
                    "environment_mutation",
                    f"{script} {operation}",
                    executable,
                )
        if executable in {"pip", "pip3"}:
            operation = arguments[0] if arguments else ""
            if operation in {"install", "uninstall", "download", "wheel"}:
                if _local_editable_pip_install(
                    arguments,
                    cwd=shell_command.cwd,
                    repository_root=repository_root,
                ):
                    continue
                return _environment_violation(
                    "bash_environment_mutation_attempt",
                    "environment_mutation",
                    f"{executable} {operation}",
                    executable,
                )

        if executable in _CONDA_EXECUTABLES:
            operation = next(
                (value for value in arguments if value and not value.startswith("-")),
                "",
            )
            if operation in {"create", "env", "install", "remove", "uninstall", "update", "upgrade"}:
                return _environment_violation(
                    "bash_environment_mutation_attempt",
                    "environment_mutation",
                    f"{executable} {operation}",
                    executable,
                )

        wrapped_command = _wrapped_run_command(executable, arguments)
        if wrapped_command is not None:
            violation = _detect_preconfigured_environment_violation(
                shlex.join(wrapped_command),
                repository_root,
                shell_command.cwd,
            )
            if violation is not None:
                return violation

        if executable in {"easy_install", "easy_install3", "pipx"}:
            return _environment_violation(
                "bash_environment_mutation_attempt",
                "environment_mutation",
                executable,
                executable,
            )

        if executable in _SYSTEM_PACKAGE_EXECUTABLES:
            return _environment_violation(
                "bash_environment_mutation_attempt",
                "environment_mutation",
                executable,
                executable,
            )

        package_operations = {
            "npm": {"ci", "install", "update"},
            "pnpm": {"add", "install", "update"},
            "poetry": {"add", "install", "update"},
            "uv": {"add", "lock", "sync"},
            "yarn": {"add", "install", "up", "upgrade"},
        }
        if executable in package_operations:
            operation = arguments[0] if arguments else ""
            if executable == "uv" and operation == "pip":
                operation = arguments[1] if len(arguments) > 1 else ""
                pip_arguments = arguments[1:]
                if operation in {"install", "uninstall", "download", "wheel"}:
                    if _local_editable_pip_install(
                        pip_arguments,
                        cwd=shell_command.cwd,
                        repository_root=repository_root,
                    ):
                        continue
                    return _environment_violation(
                        "bash_environment_mutation_attempt",
                        "environment_mutation",
                        f"uv pip {operation}",
                        executable,
                    )
            elif operation in package_operations[executable]:
                return _environment_violation(
                    "bash_environment_mutation_attempt",
                    "environment_mutation",
                    f"{executable} {operation}",
                    executable,
                )

        if executable in {"curl", "wget"}:
            targets = _download_targets(arguments)
            if not targets or any(not _network_target_is_local(target) for target in targets):
                return _environment_violation(
                    "bash_external_download_attempt",
                    "external_download",
                    executable,
                    executable,
                )

        if executable == "git":
            operation = next(
                (value for value in arguments if value and not value.startswith("-")),
                "",
            )
            network_operation = operation in {"clone", "fetch", "pull"}
            if operation == "submodule" and "update" in arguments:
                network_operation = True
            if network_operation:
                return _environment_violation(
                    "bash_external_download_attempt",
                    "external_download",
                    f"git {operation}",
                    executable,
                )

    return None


def _environment_violation(
    code: str,
    subtype: str,
    operation: str,
    executable: str | None = None,
) -> EnvironmentPolicyViolation:
    if subtype == "external_download":
        message = (
            f"execute_bash cannot use {operation} for external downloads in this preconfigured "
            "evaluation environment; use the installed testbed dependencies"
        )
    else:
        message = (
            f"execute_bash cannot run {operation} in this preconfigured evaluation environment; "
            "the testbed environment is already provisioned"
        )
    return EnvironmentPolicyViolation(
        code=code,
        subtype=subtype,
        message=message,
        executable=executable,
    )


def detect_repository_search(command: str, repository_root: str) -> RepositorySearchViolation | None:
    """Detect ordinary attempts to bypass Codeflow's repository search tool."""
    root = _normalized_root(repository_root)
    return _detect_repository_search(command, root, root)


def _detect_repository_search(
    command: str,
    repository_root: str,
    initial_cwd: str,
) -> RepositorySearchViolation | None:
    try:
        commands = _commands(
            _tokenize(command),
            repository_root,
            initial_cwd=initial_cwd,
        )
    except ValueError:
        lowered = command.lower()
        if any(name in lowered for name in _SEARCH_EXECUTABLES):
            return RepositorySearchViolation(
                code="bash_repository_search_attempt",
                message="execute_bash contains an unparseable repository-search command; use search instead",
            )
        return None

    pipeline_scope = "none"
    for shell_command in commands:
        if shell_command.separator_before != "|":
            pipeline_scope = "none"
        executable, arguments = _unwrap_command(shell_command.words)
        if not executable:
            continue

        for nested in _command_substitutions(" ".join(shell_command.words)):
            violation = _detect_repository_search(
                nested,
                repository_root,
                shell_command.cwd,
            )
            if violation is not None:
                return violation

        if executable in {"bash", "sh", "dash", "zsh"}:
            nested = _nested_shell(arguments)
            if nested:
                violation = _detect_repository_search(
                    nested,
                    repository_root,
                    shell_command.cwd,
                )
                if violation is not None:
                    return violation
            continue

        if _stream_reader_reads_repo(
            executable,
            arguments,
            cwd=shell_command.cwd,
            repository_root=repository_root,
        ):
            pipeline_scope = "repo"

        stdin_repo, stdin_unknown = _repo_target(
            list(shell_command.stdin_paths),
            cwd=shell_command.cwd,
            repository_root=repository_root,
        )

        if executable == "git" and "grep" in arguments:
            git_cwd = shell_command.cwd
            if "-C" in arguments:
                position = arguments.index("-C")
                if position + 1 < len(arguments):
                    target = arguments[position + 1]
                    if not any(marker in target for marker in ("$", "`", "~")):
                        git_cwd = posixpath.normpath(target if target.startswith("/") else posixpath.join(git_cwd, target))
            if (
                _path_scope(
                    ".",
                    cwd=git_cwd,
                    repository_root=repository_root,
                )
                != "external"
            ):
                return _violation("git grep")
            continue

        if executable == "xargs":
            search_index = next(
                (
                    index
                    for index, value in enumerate(arguments)
                    if posixpath.basename(value) in _GREP_EXECUTABLES
                ),
                None,
            )
            if search_index is not None:
                nested_arguments = [
                    "/tmp/__rllm_xargs_item__" if value == "{}" else value
                    for value in arguments[search_index + 1 :]
                ]
                targets, _, _ = _grep_targets(
                    posixpath.basename(arguments[search_index]),
                    nested_arguments,
                )
                repo, unknown = _repo_target(
                    targets,
                    cwd=shell_command.cwd,
                    repository_root=repository_root,
                )
                if pipeline_scope != "external" or repo or unknown:
                    return _violation("xargs search")
            continue

        if executable not in _SEARCH_EXECUTABLES:
            continue
        if executable == "locate":
            return _violation(executable)

        if executable == "find":
            targets, _ = _find_targets(arguments)
            repo, unknown = _repo_target(
                targets,
                cwd=shell_command.cwd,
                repository_root=repository_root,
            )
            if repo or unknown:
                return _violation(executable)
            for nested_words in _find_exec_commands(arguments):
                normalized_words = [
                    "/tmp/__rllm_find_result__" if value == "{}" else value
                    for value in nested_words
                ]
                violation = _detect_repository_search(
                    shlex.join(normalized_words),
                    repository_root,
                    shell_command.cwd,
                )
                if violation is not None:
                    return violation
            pipeline_scope = "external"
            continue

        if executable == "fd":
            targets, _ = _fd_targets(arguments)
            repo, unknown = _repo_target(
                targets,
                cwd=shell_command.cwd,
                repository_root=repository_root,
            )
            if repo or unknown:
                return _violation(executable)
            pipeline_scope = "external"
            continue

        targets, recursive, ambiguous = _grep_targets(executable, arguments)
        repo, unknown = _repo_target(
            targets,
            cwd=shell_command.cwd,
            repository_root=repository_root,
        )
        piped_input = shell_command.separator_before == "|"
        if stdin_repo or stdin_unknown or repo or unknown or pipeline_scope == "repo":
            return _violation(executable)
        if recursive and not targets:
            return _violation(executable)
        if ambiguous and not piped_input:
            return _violation(executable)

    return None


def _violation(executable: str) -> RepositorySearchViolation:
    return RepositorySearchViolation(
        code="bash_repository_search_attempt",
        executable=executable,
        message=(
            f"execute_bash cannot use {executable} to search repository files; "
            "use search(mode='text'|'symbol'|'file'|'directory') instead"
        ),
    )
