"""Noise-tolerant framing for JSON-producing sandbox helper commands."""

from __future__ import annotations

import base64
import json
import re
import secrets
import shlex
from dataclasses import dataclass

# Task code and test runners routinely use (and occasionally replace) /tmp.
# Keep the short-lived transport files on the sandbox tmpfs so deleting a
# task's scratch tree cannot break RLLM's control protocol mid-command. Larger
# repository checkpoints live in /var/tmp instead. Sandboxes without a usable
# shared-memory mount fall back to /var/tmp after an explicit writability check.
STRUCTURED_COMMAND_TMPDIR = "/dev/shm"
STRUCTURED_COMMAND_FALLBACK_TMPDIR = "/var/tmp"


@dataclass(frozen=True)
class StructuredCommand:
    command: str
    nonce: str


@dataclass(frozen=True)
class StructuredCommandResult:
    exit_code: int
    stdout: str
    stderr: str


class StructuredCommandProtocolError(RuntimeError):
    """The helper may have run, but its framed response was not trustworthy."""


def frame_structured_command(command: str) -> StructuredCommand:
    """Wrap one command so stdout/stderr survive noisy combined transports."""

    nonce = secrets.token_hex(8)
    temp_dir = STRUCTURED_COMMAND_TMPDIR
    fallback_temp_dir = STRUCTURED_COMMAND_FALLBACK_TMPDIR
    encoder = (
        "import base64,json,sys;"
        "p={'exit_code':int(sys.argv[3]),"
        "'stdout':base64.b64encode(open(sys.argv[1],'rb').read()).decode('ascii'),"
        "'stderr':base64.b64encode(open(sys.argv[2],'rb').read()).decode('ascii')};"
        "print('__RLLM_STRUCTURED_BEGIN_" + nonce + "__'+"
        "base64.b64encode(json.dumps(p,separators=(',',':')).encode()).decode('ascii')+"
        "'__RLLM_STRUCTURED_END_" + nonce + "__')"
    )
    wrapped = (
        "_rllm_ensure_dir() { "
        '[ -d "$1" ] || mkdir -p -m 1777 "$1" 2>/dev/null || return 1; '
        '[ -w "$1" ]; }; '
        f"_rllm_tmpdir={shlex.quote(temp_dir)}; "
        'if ! _rllm_ensure_dir "$_rllm_tmpdir"; then '
        f"_rllm_tmpdir={shlex.quote(fallback_temp_dir)}; "
        '_rllm_ensure_dir "$_rllm_tmpdir" || exit 70; fi; '
        # The same command can be retried while a timed-out RPC is still
        # executing. The protocol nonce is not an execution-unique filename.
        f'_rllm_capture=$(mktemp -d "$_rllm_tmpdir/rllm-structured-{nonce}.XXXXXX") || exit 70; '
        '_rllm_stdout="$_rllm_capture/stdout"; '
        '_rllm_stderr="$_rllm_capture/stderr"; '
        '_rllm_cleanup() { rm -f "$_rllm_stdout" "$_rllm_stderr"; rmdir "$_rllm_capture"; }; '
        "trap _rllm_cleanup 0 1 2 15; "
        f"(\n{command}\n) >\"$_rllm_stdout\" 2>\"$_rllm_stderr\"; "
        "_rllm_rc=$?; "
        f"python3 -c {shlex.quote(encoder)} "
        '"$_rllm_stdout" "$_rllm_stderr" "$_rllm_rc"; '
        "_rllm_frame_rc=$?; _rllm_cleanup; trap - 0 1 2 15; "
        'exit "$_rllm_frame_rc"'
    )
    return StructuredCommand(command=wrapped, nonce=nonce)


def parse_structured_command_output(raw: object, nonce: str) -> StructuredCommandResult:
    """Extract exactly one framed response, ignoring transport noise around it."""

    text = str(raw)
    begin = f"__RLLM_STRUCTURED_BEGIN_{nonce}__"
    end = f"__RLLM_STRUCTURED_END_{nonce}__"
    matches = re.findall(
        re.escape(begin) + r"([A-Za-z0-9+/=]+)" + re.escape(end),
        text,
    )
    if len(matches) != 1:
        raise StructuredCommandProtocolError(
            f"expected exactly one structured response frame, found {len(matches)}"
        )
    try:
        payload = json.loads(base64.b64decode(matches[0], validate=True).decode("utf-8"))
        exit_code = int(payload["exit_code"])
        stdout = base64.b64decode(payload["stdout"], validate=True).decode(
            "utf-8", errors="replace"
        )
        stderr = base64.b64decode(payload["stderr"], validate=True).decode(
            "utf-8", errors="replace"
        )
    except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StructuredCommandProtocolError(
            f"invalid structured response frame: {exc}"
        ) from exc
    return StructuredCommandResult(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
    )
