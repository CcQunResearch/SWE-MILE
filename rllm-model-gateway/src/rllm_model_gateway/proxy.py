"""httpx-based reverse proxy with streaming SSE support.

Reference: miles ``MilesRouter._do_proxy()``
(``miles/router/router.py`` lines 138-166).
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import httpx
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from rllm_model_gateway.data_process import (
    build_trace_record,
    build_trace_record_from_chunks,
    extract_completion_token_ids,
    extract_prompt_token_ids,
    strip_vllm_fields,
)
from rllm_model_gateway.http_client import shared_ssl_context
from rllm_model_gateway.models import TraceRecord
from rllm_model_gateway.session_router import (
    NoHealthyWorkersError,
    RouteSelection,
    SessionRouter,
)
from rllm_model_gateway.store.base import TraceStore
from rllm_model_gateway.token_accumulator import (
    TokenAccumulator,
    extract_new_messages,
)

logger = logging.getLogger(__name__)

_CONTEXT_METADATA_KEY = "rllm_context"
_CONTEXT_SCHEMA_VERSION = 1
_DISCARD_CONTEXT_TERMINAL = "discard_context_terminal"
_REQUEST_METADATA_KEY = "rllm_request"
_REQUEST_METADATA_SCHEMA_VERSION = 1
_ROUTING_METADATA_KEY = "rllm_routing"
_REQUEST_ID_HEADER = "x-rllm-request-id"
_TURN_INDEX_HEADER = "x-rllm-turn-index"
_RETRYABLE_TRANSPORT_ERRORS = (
    httpx.ReadError,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.TimeoutException,
)


class CumulativeResponseError(RuntimeError):
    """A cumulative response could not be translated without corrupting semantics."""


def _requested_output_tokens(request_body: dict[str, Any]) -> tuple[str, int] | None:
    """Return the OpenAI output-limit field and its validated value."""
    for key in ("max_tokens", "max_completion_tokens"):
        if key not in request_body:
            continue
        value = request_body[key]
        if isinstance(value, bool):
            raise CumulativeResponseError(f"{key} must be a positive integer")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise CumulativeResponseError(f"{key} must be a positive integer") from exc
        if parsed <= 0:
            raise CumulativeResponseError(f"{key} must be a positive integer")
        return key, parsed
    return None


def _normalise_prompt_token_ids(rendered: Any) -> list[int]:
    """Extract one prompt's token IDs from Hugging Face return shapes.

    ``PreTrainedTokenizerBase.apply_chat_template`` normally returns
    ``list[int]`` when ``tokenize=True``. Some tokenizer/transformers versions
    instead return a ``BatchEncoding`` (a mapping containing ``input_ids``),
    while tensor-backed variants expose ``tolist()``. Keep the gateway
    independent of transformers and tensor libraries by accepting those shapes
    through their public duck-typed interfaces.
    """

    rendered_type = type(rendered).__name__
    if isinstance(rendered, Mapping):
        if "input_ids" not in rendered:
            raise TypeError(
                "apply_chat_template returned "
                f"{rendered_type} without an input_ids field"
            )
        rendered = rendered["input_ids"]
    elif not isinstance(rendered, list | tuple) and hasattr(
        rendered, "input_ids"
    ):
        rendered = rendered.input_ids

    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    if isinstance(rendered, tuple):
        rendered = list(rendered)

    # Tokenizers may retain a batch dimension even for one chat prompt.
    if isinstance(rendered, list) and len(rendered) == 1:
        first = rendered[0]
        if hasattr(first, "tolist"):
            first = first.tolist()
        if isinstance(first, tuple):
            first = list(first)
        if isinstance(first, list):
            rendered = first

    if not isinstance(rendered, list):
        raise TypeError(
            "apply_chat_template returned "
            f"{rendered_type}; expected list[int] or input_ids"
        )
    if any(
        isinstance(token_id, bool) or not isinstance(token_id, Integral)
        for token_id in rendered
    ):
        raise TypeError(
            "apply_chat_template returned "
            f"{rendered_type} with non-integer or batched token IDs"
        )
    return [int(token_id) for token_id in rendered]


def _context_metadata(
    *,
    max_context_tokens: int,
    input_tokens: int,
    requested_output_tokens: int,
    effective_output_tokens: int,
    input_exhausted: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": _CONTEXT_SCHEMA_VERSION,
        "max_context_tokens": max_context_tokens,
        "input_tokens": input_tokens,
        "requested_output_tokens": requested_output_tokens,
        "effective_output_tokens": effective_output_tokens,
        "output_capped": effective_output_tokens < requested_output_tokens,
        "input_exhausted": input_exhausted,
        "upstream_finish_reason": None,
        "completion_tokens": 0,
        "training_disposition": "keep",
    }


def _context_exhausted_body(
    request_body: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Build a valid chat response without invoking the inference worker."""
    now = int(time.time())
    input_tokens = int(context["input_tokens"])
    return {
        "id": f"chatcmpl-rllm-context-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": now,
        "model": request_body.get("model", ""),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": None},
                "finish_reason": "length",
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": 0,
            "total_tokens": input_tokens,
        },
        _CONTEXT_METADATA_KEY: context,
    }


def _parsed_field(parsed: Any, name: str, default: Any = None) -> Any:
    if isinstance(parsed, dict):
        return parsed.get(name, default)
    return getattr(parsed, name, default)


def _renderer_tool_specs(tools: Any) -> list[dict[str, Any]] | None:
    """Unwrap OpenAI tools into the ToolSpec shape expected by renderers."""
    if not tools:
        return None
    if not isinstance(tools, list):
        raise CumulativeResponseError("tools must be a list")

    specs: list[dict[str, Any]] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise CumulativeResponseError(f"tools[{index}] must be an object")
        function = tool.get("function") if tool.get("type") == "function" else tool
        if not isinstance(function, dict):
            raise CumulativeResponseError(f"tools[{index}].function must be an object")
        specs.append(function)
    return specs


def _renderer_tool_call_status(raw_tool_call: Any) -> str | None:
    status = _parsed_field(raw_tool_call, "status")
    if status is None:
        return None
    value = getattr(status, "value", status)
    return str(value)


def _normalise_renderer_tool_calls(raw_tool_calls: Any) -> list[dict[str, Any]]:
    """Convert renderer tool calls to the OpenAI chat-completions shape.

    ``renderers`` 0.1.8 returns ``ParsedToolCall`` dataclasses, while older
    adapters and tests may return lightweight dictionaries. The agent-facing
    API needs a stable call id, a type, and JSON-encoded arguments.
    """
    if not isinstance(raw_tool_calls, list):
        raise CumulativeResponseError("renderer tool_calls must be a list")

    normalised: list[dict[str, Any]] = []
    for index, raw_tool_call in enumerate(raw_tool_calls):
        raw_function = _parsed_field(raw_tool_call, "function")
        if raw_function is None:
            raw_function = raw_tool_call
        if not isinstance(raw_function, dict) and not hasattr(raw_function, "name"):
            raise CumulativeResponseError(f"renderer tool_calls[{index}].function must be an object")
        name = _parsed_field(raw_function, "name")
        if not isinstance(name, str) or not name:
            raise CumulativeResponseError(f"renderer tool_calls[{index}] has no function name")

        arguments = _parsed_field(raw_function, "arguments", {})
        if arguments is None:
            arguments = {}
        if isinstance(arguments, str):
            try:
                json.loads(arguments)
            except (json.JSONDecodeError, ValueError) as exc:
                raise CumulativeResponseError(f"renderer tool_calls[{index}] has invalid JSON arguments") from exc
            arguments_json = arguments
        else:
            try:
                arguments_json = json.dumps(arguments, ensure_ascii=False)
            except (TypeError, ValueError) as exc:
                raise CumulativeResponseError(f"renderer tool_calls[{index}] has non-serializable arguments") from exc

        call_id = _parsed_field(raw_tool_call, "id")
        if not isinstance(call_id, str) or not call_id:
            call_id = f"chatcmpl-tool-{uuid.uuid4().hex}"
        call_type = _parsed_field(raw_tool_call, "type", "function")
        if call_type != "function":
            raise CumulativeResponseError(f"renderer tool_calls[{index}] has unsupported type {call_type!r}")

        normalised.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments_json},
            }
        )
    return normalised


def _translate_cumulative_message(
    renderer: Any,
    raw_text: str,
    completion_token_ids: list[int],
    *,
    tools: Any,
) -> tuple[dict[str, Any], str | None]:
    """Translate a text-completion result back to a structured chat message.

    ``/v1/completions`` does not run vLLM's tool parser.  In cumulative mode
    the renderer is the authority for decoding the exact sampled token IDs;
    re-parsing decoded text would lose the token-level contract that motivated
    cumulative forwarding in the first place.
    """
    plain_message: dict[str, Any] = {"role": "assistant", "content": raw_text}
    if not tools:
        return plain_message, None
    if renderer is None or not callable(getattr(renderer, "parse_response", None)):
        raise CumulativeResponseError("cumulative tool-call translation requires renderer.parse_response")
    if not completion_token_ids:
        raise CumulativeResponseError("cumulative tool-call response is missing completion token IDs")

    try:
        parsed = renderer.parse_response(
            completion_token_ids,
            tools=_renderer_tool_specs(tools),
        )
    except Exception as exc:
        raise CumulativeResponseError("renderer.parse_response failed for a cumulative tool-call response") from exc

    raw_tool_calls = _parsed_field(parsed, "tool_calls")
    if not raw_tool_calls:
        # A model response without one complete tool call remains a normal
        # response.  The Codeflow harness will apply its existing format error;
        # the gateway must not fuzzily repair malformed or truncated XML.
        return plain_message, None

    # ParsedToolCall includes unsuccessful parse attempts as provenance. Such
    # attempts are model-format errors, not gateway infrastructure errors. Do
    # not preserve only the successful subset (which could turn a mixed invalid
    # response into an accepted single call); return the sampled text unchanged
    # and let the Codeflow harness apply its normal missing-call penalty.
    statuses = [_renderer_tool_call_status(tool_call) for tool_call in raw_tool_calls]
    if any(status not in (None, "ok") for status in statuses):
        return plain_message, None

    message: dict[str, Any] = {
        "role": "assistant",
        "content": _parsed_field(parsed, "content", "") or None,
        "tool_calls": _normalise_renderer_tool_calls(raw_tool_calls),
    }
    reasoning = _parsed_field(parsed, "reasoning_content")
    if reasoning:
        message["reasoning"] = reasoning
    return message, "tool_calls"


def _translate_initial_cumulative_chat_response(
    renderer: Any,
    request_body: dict[str, Any],
    response_body: dict[str, Any],
) -> None:
    """Give turn zero the same renderer semantics as cumulative turns.

    vLLM's chat endpoint may parse tool calls without parsing Qwen reasoning,
    while rewritten turn 1+ completions are parsed by ``renderers``.  That
    leaves the initial ``</think>`` marker and its preceding reasoning in
    ``content``.  Re-parse the exact sampled token IDs before tracing or
    returning the response so every turn has the same content/reasoning split.
    """
    if not request_body.get("tools"):
        return
    choices = response_body.get("choices") or []
    if not choices:
        return

    first_choice = choices[0]
    raw_message = first_choice.get("message") or {}
    raw_text = raw_message.get("content", "") or ""
    message, finish_reason = _translate_cumulative_message(
        renderer,
        raw_text,
        extract_completion_token_ids(response_body),
        tools=request_body.get("tools"),
    )
    first_choice["message"] = message
    if finish_reason is not None:
        first_choice["finish_reason"] = finish_reason

# Headers that should not be forwarded verbatim
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "transfer-encoding",
        "te",
        "trailer",
        "upgrade",
        "content-length",
        "content-encoding",
        "host",
        _REQUEST_ID_HEADER,
        _TURN_INDEX_HEADER,
    }
)


@dataclass(frozen=True)
class _ResponseSnapshot:
    body: bytes
    status_code: int
    media_type: str | None
    headers: dict[str, str]

    @classmethod
    def from_response(cls, response: Response) -> "_ResponseSnapshot":
        headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower() not in {"content-length", "content-type"}
        }
        return cls(
            body=bytes(response.body),
            status_code=int(response.status_code),
            media_type=response.media_type,
            headers=headers,
        )

    def response(self) -> Response:
        return Response(
            content=self.body,
            status_code=self.status_code,
            media_type=self.media_type,
            headers=self.headers,
        )


@dataclass
class _IdempotentRequest:
    body_sha256: str
    turn_index: int
    task: asyncio.Task[_ResponseSnapshot]


def _strip_logprobs(response: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *response* with ``logprobs`` removed from each choice.

    Called when the gateway injected ``logprobs=True`` but the original
    client request did not ask for them — keeps the proxy transparent.

    Returns a new dict so that the original (used for trace capture) is
    never mutated.
    """
    if "choices" not in response:
        return response
    return {
        **response,
        "choices": [{k: v for k, v in choice.items() if k != "logprobs"} for choice in response["choices"]],
    }


def _trace_metadata(
    request: Request,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = dict(extra or {})
    request_id = getattr(request.state, "rllm_request_id", None)
    if request_id:
        metadata[_REQUEST_METADATA_KEY] = {
            "schema_version": _REQUEST_METADATA_SCHEMA_VERSION,
            "request_id": str(request_id),
            "turn_index": int(request.state.rllm_turn_index),
            "request_body_sha256": str(
                request.state.rllm_request_body_sha256
            ),
        }
    routing = getattr(request.state, "rllm_routing", None)
    if isinstance(routing, dict):
        metadata[_ROUTING_METADATA_KEY] = dict(routing)
    return metadata


class ReverseProxy:
    """Forward requests to inference workers, capture traces.

    Non-streaming requests are fully buffered so that the complete response
    can be inspected for token IDs and logprobs.

    Streaming (SSE) requests are forwarded chunk-by-chunk in real time.
    Chunks are buffered internally so that a ``TraceRecord`` can be assembled
    after ``[DONE]``.
    """

    def __init__(
        self,
        router: SessionRouter,
        store: TraceStore,
        *,
        strip_vllm: bool = True,
        capture_raw_payloads: bool = True,
        sync_traces: bool = False,
        max_retries: int = 2,
        local_handler: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] | None = None,
        cumulative_token_mode: bool = False,
        dynamic_sequence_budget: bool = False,
        max_context_tokens: int | None = None,
        worker_recovery_timeout: float = 600.0,
        renderer: Any = None,
        tokenizer: Any = None,
    ) -> None:
        self.router = router
        self.store = store
        self.strip_vllm = strip_vllm
        self.capture_raw_payloads = capture_raw_payloads
        self.sync_traces = sync_traces
        self.max_retries = max_retries
        self.local_handler = local_handler
        self.cumulative_token_mode = cumulative_token_mode
        self.dynamic_sequence_budget = dynamic_sequence_budget
        self.max_context_tokens = max_context_tokens
        if self.max_context_tokens is not None and self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be a positive integer")
        self.worker_recovery_timeout = float(worker_recovery_timeout)
        if self.worker_recovery_timeout <= 0:
            raise ValueError("worker_recovery_timeout must be positive")
        self.renderer = renderer
        self.tokenizer = tokenizer
        self.weight_version: int | None = None
        self._http: httpx.AsyncClient | None = None
        self._pending_traces: set[asyncio.Task[None]] = set()
        self._pending_traces_by_session: dict[str, set[asyncio.Task[None]]] = {}
        # A session is sealed before its traces are deleted.  Fire-and-forget
        # persistence scheduled by a late streaming finalizer must not recreate
        # a session after DELETE has returned.
        self._sealed_sessions: set[str] = set()
        self._accumulators: dict[str, TokenAccumulator] = {}
        self._idempotent_requests: dict[
            tuple[str, str], _IdempotentRequest
        ] = {}
        self._idempotent_requests_by_session: dict[
            str, set[tuple[str, str]]
        ] = {}
        self._latest_idempotent_turn_by_session: dict[str, int] = {}
        self._idempotent_lock = asyncio.Lock()
        self._started_monotonic = time.monotonic()
        self._last_proxy_success_monotonic: float | None = None
        self.on_proxy_success: Callable[[float], None] | None = None

    async def _route_worker(
        self,
        session_id: str | None,
        *,
        input_tokens: int | None = None,
    ) -> RouteSelection:
        """Wait through a temporary all-workers-unhealthy interval."""

        started = time.monotonic()
        try:
            selection = self.router.route_request(
                session_id,
                input_tokens=input_tokens,
            )
        except NoHealthyWorkersError:
            logger.warning(
                "No healthy inference workers available; waiting up to %.0fs for recovery",
                self.worker_recovery_timeout,
            )
            try:
                selection = await self.router.route_request_when_available(
                    session_id,
                    input_tokens=input_tokens,
                    timeout=self.worker_recovery_timeout,
                )
            except TimeoutError as timeout_exc:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "No healthy inference workers recovered within "
                        f"{self.worker_recovery_timeout:g} seconds"
                    ),
                ) from timeout_exc
            logger.info(
                "Inference worker recovered after %.1fs; resuming request on %s",
                time.monotonic() - started,
                selection.worker.url,
            )
        return selection

    @staticmethod
    def _request_input_tokens(
        request: Request,
        request_body: dict[str, Any],
    ) -> int | None:
        initial_context = getattr(request.state, "rllm_initial_context", None)
        if isinstance(initial_context, dict):
            value = initial_context.get("input_tokens")
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        prompt = request_body.get("prompt")
        if isinstance(prompt, list) and all(
            isinstance(token, int) and not isinstance(token, bool) for token in prompt
        ):
            return len(prompt)
        return None

    def open_session(self, session_id: str) -> None:
        """Allow trace writes for a newly created or implicitly reopened session."""
        self._sealed_sessions.discard(session_id)

    def seal_session(self, session_id: str) -> None:
        """Prevent future trace writes while a session is being deleted."""
        self._sealed_sessions.add(session_id)

    def is_session_sealed(self, session_id: str) -> bool:
        return session_id in self._sealed_sessions

    def discard_session_state(
        self,
        session_id: str,
    ) -> tuple[asyncio.Task[_ResponseSnapshot], ...]:
        """Detach heavy session state immediately and return active requests.

        This method deliberately contains no ``await``.  It runs atomically on
        the gateway event loop, so a batch tombstone can release accumulators
        and completed idempotent response snapshots before the bounded physical
        reaper starts waiting for upstream request cancellation.
        """
        self._accumulators.pop(session_id, None)
        self._latest_idempotent_turn_by_session.pop(session_id, None)
        keys = self._idempotent_requests_by_session.pop(session_id, set())
        active: list[asyncio.Task[_ResponseSnapshot]] = []
        for key in keys:
            entry = self._idempotent_requests.pop(key, None)
            if entry is not None and not entry.task.done():
                entry.task.cancel()
                active.append(entry.task)
        return tuple(active)

    async def cancel_session_requests(self, session_id: str) -> None:
        """Cancel and drain server-owned requests during explicit deletion."""
        async with self._idempotent_lock:
            entries = [
                self._idempotent_requests[key]
                for key in self._idempotent_requests_by_session.get(
                    session_id,
                    set(),
                )
                if key in self._idempotent_requests
            ]
            for entry in entries:
                if not entry.task.done():
                    entry.task.cancel()
        if entries:
            await asyncio.gather(
                *(entry.task for entry in entries),
                return_exceptions=True,
            )

    def runtime_stats(self) -> dict[str, int | float | None]:
        """Return lightweight state counts for the external supervisor."""
        now = time.monotonic()
        last_success = self._last_proxy_success_monotonic
        idempotent_in_flight = sum(
            not entry.task.done()
            for entry in self._idempotent_requests.values()
        )
        idempotent_replayable = sum(
            entry.task.done()
            and not entry.task.cancelled()
            and entry.task.exception() is None
            and entry.task.result().status_code != 429
            and entry.task.result().status_code < 500
            for entry in self._idempotent_requests.values()
        )
        return {
            "accumulators": len(self._accumulators),
            "idempotent_requests": len(self._idempotent_requests),
            "idempotent_in_flight": idempotent_in_flight,
            "idempotent_replayable": idempotent_replayable,
            "pending_traces": len(self._pending_traces),
            "sealed_sessions": len(self._sealed_sessions),
            "seconds_since_last_proxy_success": (
                max(0.0, now - self._started_monotonic)
                if last_success is None
                else max(0.0, now - last_success)
            ),
            "uptime_seconds": max(0.0, now - self._started_monotonic),
        }

    @property
    def last_proxy_success_monotonic(self) -> float | None:
        """Timestamp copied into the out-of-band supervision heartbeat."""
        return self._last_proxy_success_monotonic

    def _note_proxy_response(self, response: Response) -> Response:
        if response.status_code < 500:
            success_monotonic = time.monotonic()
            self._last_proxy_success_monotonic = success_monotonic
            if self.on_proxy_success is not None:
                self.on_proxy_success(success_monotonic)
        return response

    def _track_trace_task(self, session_id: str, task: asyncio.Task[None]) -> None:
        """Index one background trace write globally and by owning session."""
        self._pending_traces.add(task)
        session_tasks = self._pending_traces_by_session.setdefault(session_id, set())
        session_tasks.add(task)

        def _discard(done: asyncio.Task[None]) -> None:
            self._pending_traces.discard(done)
            pending = self._pending_traces_by_session.get(session_id)
            if pending is None:
                return
            pending.discard(done)
            if not pending:
                self._pending_traces_by_session.pop(session_id, None)

        task.add_done_callback(_discard)

    def _schedule_trace_store(
        self,
        trace_id: str,
        session_id: str,
        data: dict[str, Any],
    ) -> asyncio.Task[None]:
        task = asyncio.create_task(self._safe_store(trace_id, session_id, data))
        self._track_trace_task(session_id, task)
        return task

    async def wait_for_pending_traces(
        self,
        session_ids: set[str] | None = None,
    ) -> None:
        """Drain trace writes for selected sessions, or all sessions.

        The loop is intentional: a streaming response can enqueue its trace
        while an earlier write is being awaited.  Session deletion seals the
        target first, so any still-later write becomes a no-op instead of
        resurrecting deleted memory-store state.
        """
        while True:
            if session_ids is None:
                pending = set(self._pending_traces)
            else:
                pending = {
                    task
                    for session_id in session_ids
                    for task in self._pending_traces_by_session.get(session_id, ())
                }
            if not pending:
                return
            await asyncio.gather(*pending, return_exceptions=True)

    def _get_accumulator(self, session_id: str, template_kwargs: dict[str, Any] | None = None) -> TokenAccumulator:
        """Return the TokenAccumulator for *session_id*, creating if needed."""
        if session_id not in self._accumulators:
            self._accumulators[session_id] = TokenAccumulator(self.renderer)
        acc = self._accumulators[session_id]
        renderer_config = getattr(self.renderer, "config", None)
        if template_kwargs is not None and getattr(renderer_config, "name", None) == "qwen3.6":
            # Train/validation sessions can have different template controls.
            # Bind a renderer per session; changing the global renderer would
            # silently switch neighboring trajectories' generation prefixes.
            from renderers import create_renderer

            if not isinstance(template_kwargs, dict):
                raise HTTPException(status_code=400, detail="chat_template_kwargs must be an object")
            fields = renderer_config._template_fields
            controls = {key: value for key, value in template_kwargs.items() if key in fields}
            if controls != acc.template_controls:
                if acc.should_rewrite() and acc.template_controls is not None:
                    raise HTTPException(status_code=400, detail="chat template controls cannot change during a cumulative session")
                values = renderer_config.model_dump()
                values.update(controls)
                typed_config = type(renderer_config).model_validate(values)
                acc.renderer = create_renderer(self.tokenizer, typed_config)
                acc.template_controls = controls
        return acc

    def _prepare_initial_context_request(
        self,
        request_body: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
        """Cap a fresh chat turn before forwarding it to the model worker.

        Later cumulative turns already have exact token IDs. For turn zero we
        render with the same Hugging Face tokenizer loaded from the served
        checkpoint. The vLLM-returned prompt IDs remain authoritative for the
        persisted trace and final context metadata.
        """
        if self.max_context_tokens is None or self.tokenizer is None:
            return request_body, None, None
        output_limit = _requested_output_tokens(request_body)
        if output_limit is None:
            return request_body, None, None

        try:
            template_kwargs = request_body.get("chat_template_kwargs") or {}
            if not isinstance(template_kwargs, dict):
                template_kwargs = {}
            prompt_token_ids = self.tokenizer.apply_chat_template(
                request_body.get("messages") or [],
                tools=request_body.get("tools"),
                tokenize=True,
                add_generation_prompt=True,
                **template_kwargs,
            )
            prompt_token_ids = _normalise_prompt_token_ids(prompt_token_ids)
        except Exception as exc:
            # Preserve compatibility with custom chat templates. The serving
            # worker still enforces max_model_len, and the returned exact token
            # IDs are validated by the training transform.
            logger.warning(
                "Could not pre-render initial prompt for context budgeting; "
                "falling back to worker-side model-window enforcement: %s",
                exc,
            )
            return request_body, None, None

        output_key, requested_output_tokens = output_limit
        input_tokens = len(prompt_token_ids)
        remaining = self.max_context_tokens - input_tokens
        effective_output_tokens = min(
            requested_output_tokens, max(remaining, 0)
        )
        context = _context_metadata(
            max_context_tokens=self.max_context_tokens,
            input_tokens=input_tokens,
            requested_output_tokens=requested_output_tokens,
            effective_output_tokens=effective_output_tokens,
            input_exhausted=remaining <= 0,
        )
        if remaining <= 0:
            context["training_disposition"] = _DISCARD_CONTEXT_TERMINAL
            return request_body, context, _context_exhausted_body(
                request_body, context
            )

        capped_body = dict(request_body)
        capped_body[output_key] = effective_output_tokens
        return capped_body, context, None

    def _finalize_initial_context_metadata(
        self,
        context: dict[str, Any],
        response_body: dict[str, Any],
        upstream_finish_reason: Any,
    ) -> bool:
        """Reconcile turn-zero estimates with vLLM's authoritative IDs."""
        prompt_token_ids = extract_prompt_token_ids(response_body)
        completion_token_ids = extract_completion_token_ids(response_body)
        if prompt_token_ids:
            input_tokens = len(prompt_token_ids)
            requested_output_tokens = int(context["requested_output_tokens"])
            remaining = self.max_context_tokens - input_tokens
            effective_output_tokens = min(
                requested_output_tokens, max(remaining, 0)
            )
            context.update(
                {
                    "input_tokens": input_tokens,
                    "effective_output_tokens": effective_output_tokens,
                    "output_capped": (
                        effective_output_tokens < requested_output_tokens
                    ),
                    "input_exhausted": remaining <= 0,
                }
            )
        context["upstream_finish_reason"] = upstream_finish_reason
        context["completion_tokens"] = len(completion_token_ids)

        choices = response_body.get("choices") or []
        message = choices[0].get("message") or {} if choices else {}
        exceeds_window = (
            int(context["input_tokens"]) + len(completion_token_ids)
            > self.max_context_tokens
        )
        capped_length = bool(
            context.get("output_capped")
            and upstream_finish_reason == "length"
        )
        discard = bool(
            context.get("input_exhausted")
            or exceeds_window
            or (capped_length and not message.get("tool_calls"))
        )
        if discard:
            context["training_disposition"] = _DISCARD_CONTEXT_TERMINAL
        response_body[_CONTEXT_METADATA_KEY] = context
        return discard

    async def start(self) -> None:
        self._http = httpx.AsyncClient(
            verify=shared_ssl_context(),
            timeout=httpx.Timeout(timeout=None),  # no timeout — LLM calls can be long
            limits=httpx.Limits(max_connections=500, max_keepalive_connections=100),
            follow_redirects=True,
        )

    async def stop(self) -> None:
        # Shutdown owns all remaining server-side generations. Client
        # disconnects intentionally do not cancel them during normal service,
        # but process shutdown must not leave task warnings or retain request
        # snapshots until the event loop disappears.
        idempotent_tasks = {
            entry.task for entry in self._idempotent_requests.values()
        }
        for task in idempotent_tasks:
            if not task.done():
                task.cancel()
        if idempotent_tasks:
            await asyncio.gather(*idempotent_tasks, return_exceptions=True)
        self._idempotent_requests.clear()
        self._idempotent_requests_by_session.clear()
        self._latest_idempotent_turn_by_session.clear()
        # Drain pending trace writes before closing
        if self._pending_traces:
            logger.info("Draining %d pending trace writes...", len(self._pending_traces))
            await self.wait_for_pending_traces()
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------

    async def _ensure_started(self) -> None:
        if self._http is None:
            await self.start()

    async def handle(self, request: Request) -> Response:
        """Proxy *request* to an inference worker, capture trace, return response."""
        await self._ensure_started()
        session_id: str | None = request.state.session_id
        if session_id and self.is_session_sealed(session_id):
            return Response(
                content=json.dumps({"error": f"Session {session_id} is being deleted"}),
                status_code=409,
                media_type="application/json",
            )
        originally_requested_logprobs: bool = getattr(request.state, "originally_requested_logprobs", False)
        body = await request.body()

        try:
            request_body = json.loads(body) if body else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            request_body = {}

        is_stream = request_body.get("stream", False)
        request_id = str(request.headers.get(_REQUEST_ID_HEADER) or "").strip()
        turn_index_raw = str(request.headers.get(_TURN_INDEX_HEADER) or "").strip()
        if request_id:
            if len(request_id) > 128:
                return Response(
                    content=json.dumps({"error": "X-RLLM-Request-ID is too long"}),
                    status_code=400,
                    media_type="application/json",
                )
            try:
                turn_index = int(turn_index_raw)
                if turn_index < 0:
                    raise ValueError
            except ValueError:
                return Response(
                    content=json.dumps(
                        {"error": "X-RLLM-Turn-Index must be a non-negative integer"}
                    ),
                    status_code=400,
                    media_type="application/json",
                )
            request_fingerprint = hashlib.sha256(
                request.method.encode("utf-8")
                + b"\0"
                + str(request.url.path).encode("utf-8")
                + b"?"
                + str(request.url.query).encode("utf-8")
                + b"\0"
                + body
            ).hexdigest()
            request.state.rllm_request_id = request_id
            request.state.rllm_turn_index = turn_index
            request.state.rllm_request_body_sha256 = request_fingerprint

            if session_id and not is_stream:
                key = (session_id, request_id)
                async with self._idempotent_lock:
                    latest_turn = self._latest_idempotent_turn_by_session.get(
                        session_id
                    )
                    if latest_turn is not None and turn_index < latest_turn:
                        return Response(
                            content=json.dumps(
                                {
                                    "error": (
                                        "stale turn cannot be replayed after the "
                                        f"session advanced to turn {latest_turn}"
                                    )
                                }
                            ),
                            status_code=409,
                            media_type="application/json",
                        )
                    if latest_turn is None or turn_index > latest_turn:
                        prior_entries = [
                            (prior_key, self._idempotent_requests[prior_key])
                            for prior_key in self._idempotent_requests_by_session.get(
                                session_id, set()
                            )
                            if prior_key in self._idempotent_requests
                        ]
                        if any(
                            not prior_entry.task.done()
                            for _, prior_entry in prior_entries
                        ):
                            return Response(
                                content=json.dumps(
                                    {
                                        "error": (
                                            "cannot advance turn while the previous "
                                            "request is still in flight"
                                        )
                                    }
                                ),
                                status_code=409,
                                media_type="application/json",
                            )
                        for prior_key, _ in prior_entries:
                            self._drop_idempotent_request(prior_key)
                        self._latest_idempotent_turn_by_session[session_id] = (
                            turn_index
                        )
                    entry = self._idempotent_requests.get(key)
                    if entry is not None and entry.body_sha256 != request_fingerprint:
                        return Response(
                            content=json.dumps(
                                {
                                    "error": (
                                        "request id was reused with a different request body"
                                    )
                                }
                            ),
                            status_code=409,
                            media_type="application/json",
                        )
                    if entry is None:
                        same_turn_entry = next(
                            (
                                prior_entry
                                for prior_key in self._idempotent_requests_by_session.get(
                                    session_id, set()
                                )
                                if prior_key in self._idempotent_requests
                                and (
                                    prior_entry := self._idempotent_requests[
                                        prior_key
                                    ]
                                ).turn_index
                                == turn_index
                            ),
                            None,
                        )
                        if same_turn_entry is not None:
                            return Response(
                                content=json.dumps(
                                    {
                                        "error": (
                                            "turn already has a different request id"
                                        )
                                    }
                                ),
                                status_code=409,
                                media_type="application/json",
                            )
                        task = asyncio.create_task(
                            self._execute_idempotent_request(
                                request,
                                body,
                                request_body,
                                session_id,
                                originally_requested_logprobs,
                            )
                        )
                        entry = _IdempotentRequest(
                            body_sha256=request_fingerprint,
                            turn_index=turn_index,
                            task=task,
                        )
                        self._idempotent_requests[key] = entry
                        self._idempotent_requests_by_session.setdefault(
                            session_id,
                            set(),
                        ).add(key)
                        task.add_done_callback(
                            lambda done, request_key=key: self._forget_failed_idempotent_request(
                                request_key,
                                done,
                            )
                        )
                try:
                    snapshot = await asyncio.shield(entry.task)
                except asyncio.CancelledError:
                    # Explicit DELETE owns cancellation of the server-side
                    # generation. Convert that expected lifecycle event into a
                    # stable response instead of emitting an ASGI traceback for
                    # every speculative rollout that was reclaimed.
                    if session_id and self.is_session_sealed(session_id):
                        return Response(
                            content=json.dumps(
                                {"error": f"Session {session_id} is closed"}
                            ),
                            status_code=410,
                            media_type="application/json",
                        )
                    raise
                return self._note_proxy_response(snapshot.response())

        return self._note_proxy_response(
            await self._handle_once(
                request,
                body,
                request_body,
                session_id,
                originally_requested_logprobs,
            )
        )

    def _forget_failed_idempotent_request(
        self,
        key: tuple[str, str],
        task: asyncio.Task[_ResponseSnapshot],
    ) -> None:
        entry = self._idempotent_requests.get(key)
        if entry is None or entry.task is not task:
            return
        if not task.cancelled() and task.exception() is None:
            # A retryable HTTP response is not a completed logical model
            # request.  Keeping it in the idempotency cache would make every
            # harness retry with the same request id replay the stale 429/5xx
            # forever, even after workers recover from a weight switch.
            snapshot = task.result()
            latest_turn = self._latest_idempotent_turn_by_session.get(key[0])
            if (
                snapshot.status_code != 429
                and snapshot.status_code < 500
                and entry.turn_index == latest_turn
            ):
                return
        self._drop_idempotent_request(key)

    def _drop_idempotent_request(self, key: tuple[str, str]) -> None:
        """Remove one cache entry and keep the reverse index consistent."""
        self._idempotent_requests.pop(key, None)
        session_keys = self._idempotent_requests_by_session.get(key[0])
        if session_keys is not None:
            session_keys.discard(key)
            if not session_keys:
                self._idempotent_requests_by_session.pop(key[0], None)

    async def _execute_idempotent_request(
        self,
        request: Request,
        body: bytes,
        request_body: dict[str, Any],
        session_id: str,
        originally_requested_logprobs: bool,
    ) -> _ResponseSnapshot:
        response = await self._handle_once(
            request,
            body,
            request_body,
            session_id,
            originally_requested_logprobs,
        )
        if isinstance(response, StreamingResponse):
            raise RuntimeError("streaming responses cannot use request idempotency")
        return _ResponseSnapshot.from_response(response)

    async def _handle_once(
        self,
        request: Request,
        body: bytes,
        request_body: dict[str, Any],
        session_id: str | None,
        originally_requested_logprobs: bool,
    ) -> Response:
        try:
            await self.router.wait_for_admission(
                session_id,
                timeout=self.worker_recovery_timeout,
            )
        except TimeoutError as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Inference maintenance did not finish within "
                    f"{self.worker_recovery_timeout:g} seconds"
                ),
            ) from exc
        # Stamp the version only after the maintenance barrier.  A request
        # queued during a weight update therefore belongs to the new policy,
        # not the version that happened to be active when its HTTP body arrived.
        request.state.weight_version = self.weight_version
        is_stream = request_body.get("stream", False)

        # Cumulative token mode interception: if enabled and past first turn,
        # rewrite to /v1/completions with pre-tokenized prompt to avoid drift.
        if self.cumulative_token_mode and session_id and request.url.path.endswith("/chat/completions"):
            acc = self._get_accumulator(session_id, request_body.get("chat_template_kwargs") or {})
            if acc.should_rewrite():
                messages = request_body.get("messages", [])
                if not acc.is_cumulative(messages):
                    # Message history diverged — reset and fall through to
                    # normal chat path (treated as fresh turn-0).
                    acc.reset(
                        reason=f"message_prefix_diverged(request_messages={len(messages)})",
                        session_id=session_id,
                    )
                else:
                    new_messages = extract_new_messages(messages, acc.message_count)
                    token_ids = None
                    if new_messages:
                        token_ids = acc.build_next_prompt(new_messages, tools=request_body.get("tools"))
                    if token_ids is not None:
                        return await self._handle_cumulative_turn(
                            request,
                            request_body,
                            session_id,
                            acc,
                            token_ids,
                            originally_requested_logprobs,
                        )
                    # No new messages, or the renderer couldn't prove the
                    # prefix-extension contract (e.g. DefaultRenderer, or an
                    # assistant message in the new slice). Reset so this turn is
                    # re-ingested as a fresh turn-0 on the chat path; otherwise
                    # the stale prefix would drop this turn's completion tokens
                    # from the next cumulative prompt and break prefix-extension.
                    roles = [str(message.get("role") or "") for message in new_messages]
                    reason = "no_new_messages" if not new_messages else f"renderer_bridge_rejected(roles={roles})"
                    acc.reset(reason=reason, session_id=session_id)

        initial_context: dict[str, Any] | None = None
        if (
            self.cumulative_token_mode
            and self.dynamic_sequence_budget
            and request.url.path.endswith("/chat/completions")
            and (not session_id or not self._get_accumulator(session_id).should_rewrite())
        ):
            request_body, initial_context, exhausted_body = (
                self._prepare_initial_context_request(request_body)
            )
            if exhausted_body is not None:
                if is_stream:
                    return self._context_exhausted_streaming_response(
                        exhausted_body
                    )
                return Response(
                    content=json.dumps(exhausted_body),
                    status_code=200,
                    media_type="application/json",
                )
            if initial_context is not None:
                body = json.dumps(request_body).encode()
                request.state.rllm_initial_context = initial_context

        if is_stream:
            return await self._handle_streaming(request, body, request_body, session_id, originally_requested_logprobs)
        return await self._handle_non_streaming(request, body, request_body, session_id, originally_requested_logprobs)

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    async def _handle_non_streaming(
        self,
        request: Request,
        raw_body: bytes,
        request_body: dict[str, Any],
        session_id: str | None,
        originally_requested_logprobs: bool = False,
    ) -> Response:
        t0 = time.perf_counter()

        if self.local_handler is not None:
            # In-process path: call handler directly, no HTTP
            response_body = await self.local_handler(request_body)
            status_code = 200
        else:
            # HTTP proxy path
            selection = await self._route_worker(
                session_id,
                input_tokens=self._request_input_tokens(request, request_body),
            )
            request.state.rllm_routing = selection.metadata
            worker = selection.worker
            url = self._build_url(worker.api_url, request.url.path, str(request.url.query))
            headers = self._forward_headers(request)
            try:
                resp = await self._send_with_retry(
                    method=request.method,
                    url=url,
                    content=raw_body,
                    headers=headers,
                    session_id=session_id,
                )
                content = resp.content
                status_code = resp.status_code
            finally:
                self.router.release(worker.url, session_id)

            # Parse response for trace extraction
            try:
                response_body = json.loads(content)
            except (json.JSONDecodeError, UnicodeDecodeError):
                response_body = {}

        latency_ms = (time.perf_counter() - t0) * 1000

        initial_context = getattr(
            request.state, "rllm_initial_context", None
        )
        upstream_finish_reason = None
        if initial_context is not None:
            choices = response_body.get("choices") or []
            if choices:
                upstream_finish_reason = choices[0].get("finish_reason")

        if (
            self.cumulative_token_mode
            and request.url.path.endswith("/chat/completions")
            and 200 <= status_code < 300
            and response_body
        ):
            _translate_initial_cumulative_chat_response(
                self._get_accumulator(session_id).renderer if session_id else self.renderer,
                request_body,
                response_body,
            )

        discard_initial_context = False
        if initial_context is not None and response_body:
            discard_initial_context = self._finalize_initial_context_metadata(
                initial_context,
                response_body,
                upstream_finish_reason,
            )

        # Persist trace
        # Retryable HTTP failures are not model generations.  Persisting one
        # under a stable request id and then persisting the successful retry
        # would create two conflicting traces for one agent step.
        if (
            session_id
            and response_body
            and status_code != 429
            and status_code < 500
        ):
            trace = build_trace_record(
                session_id,
                request_body,
                response_body,
                latency_ms,
                metadata=_trace_metadata(
                    request,
                    {_CONTEXT_METADATA_KEY: initial_context}
                    if initial_context is not None
                    else None,
                ),
                weight_version=request.state.weight_version,
                capture_raw_payloads=self.capture_raw_payloads,
            )
            await self._persist(trace)

            # Ingest first turn into accumulator for cumulative token mode
            if self.cumulative_token_mode and request.url.path.endswith("/chat/completions"):
                acc = self._get_accumulator(session_id)
                if acc.turn_count == 0 and not discard_initial_context:
                    prompt_ids = extract_prompt_token_ids(response_body)
                    completion_ids = extract_completion_token_ids(response_body)
                    if prompt_ids or completion_ids:
                        acc.ingest_turn(prompt_ids, completion_ids)
                        acc.update_prefix(request_body.get("messages", []))

        # Sanitise response
        needs_strip_vllm = self.strip_vllm
        needs_strip_logprobs = not originally_requested_logprobs

        sanitized = response_body
        if isinstance(response_body, dict) and response_body:
            if needs_strip_vllm:
                sanitized = strip_vllm_fields(response_body)
            if needs_strip_logprobs:
                sanitized = _strip_logprobs(sanitized)

        return Response(
            content=json.dumps(sanitized),
            status_code=status_code,
            media_type="application/json",
        )

    # ------------------------------------------------------------------
    # Cumulative token mode
    # ------------------------------------------------------------------

    async def _handle_cumulative_turn(
        self,
        request: Request,
        request_body: dict[str, Any],
        session_id: str,
        acc: TokenAccumulator,
        token_ids: list[int],
        originally_requested_logprobs: bool = False,
    ) -> Response:
        """Rewrite chat/completions to /v1/completions with pre-tokenized prompt.

        ``token_ids`` is the full bridge-extended prompt for this turn, built
        by ``acc.build_next_prompt`` in ``handle()``.

        Respects the original stream setting: if the client requested streaming,
        we stream from vLLM and translate completions chunks to chat format in
        real-time.
        """
        is_stream = request_body.get("stream", False)

        context: dict[str, Any] | None = None
        output_limit = _requested_output_tokens(request_body)
        if self.max_context_tokens is not None and output_limit is not None:
            output_key, requested_output_tokens = output_limit
            input_tokens = len(token_ids)
            remaining = self.max_context_tokens - input_tokens
            effective_output_tokens = min(requested_output_tokens, max(remaining, 0))
            context = _context_metadata(
                max_context_tokens=self.max_context_tokens,
                input_tokens=input_tokens,
                requested_output_tokens=requested_output_tokens,
                effective_output_tokens=effective_output_tokens,
                input_exhausted=remaining <= 0,
            )
            if remaining <= 0:
                body = _context_exhausted_body(request_body, context)
                if is_stream:
                    return self._context_exhausted_streaming_response(body)
                return Response(
                    content=json.dumps(body),
                    status_code=200,
                    media_type="application/json",
                )
        else:
            output_key = None

        # Construct completions request: forward everything except chat-specific fields
        completions_body = {k: v for k, v in request_body.items() if k not in ("messages", "stream", "stream_options", "tools", "tool_choice", "chat_template_kwargs")}
        completions_body["prompt"] = token_ids
        completions_body["add_special_tokens"] = False
        if context is not None and output_key is not None:
            completions_body.pop("max_completion_tokens", None)
            completions_body["max_tokens"] = context["effective_output_tokens"]

        if is_stream:
            return await self._handle_cumulative_streaming(
                request,
                request_body,
                completions_body,
                session_id,
                acc,
                token_ids,
                context,
            )
        return await self._handle_cumulative_non_streaming(
            request,
            request_body,
            completions_body,
            session_id,
            acc,
            token_ids,
            originally_requested_logprobs,
            context,
        )

    @staticmethod
    def _context_exhausted_streaming_response(body: dict[str, Any]) -> StreamingResponse:
        async def event_generator():
            choice = body["choices"][0]
            chunk = {
                "id": body["id"],
                "object": "chat.completion.chunk",
                "created": body["created"],
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": choice["finish_reason"],
                    }
                ],
                "usage": body["usage"],
                _CONTEXT_METADATA_KEY: body[_CONTEXT_METADATA_KEY],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream", status_code=200)

    async def _handle_cumulative_non_streaming(
        self,
        request: Request,
        request_body: dict[str, Any],
        completions_body: dict[str, Any],
        session_id: str,
        acc: TokenAccumulator,
        token_ids: list[int],
        originally_requested_logprobs: bool = False,
        context: dict[str, Any] | None = None,
    ) -> Response:
        """Non-streaming cumulative turn: send non-streaming to vLLM, return JSON."""
        t0 = time.perf_counter()

        selection = await self._route_worker(
            session_id,
            input_tokens=len(token_ids),
        )
        request.state.rllm_routing = selection.metadata
        worker = selection.worker
        url = self._build_url(worker.api_url, "/v1/completions", "")
        headers = self._forward_headers(request)
        raw_body = json.dumps(completions_body).encode()
        try:
            resp = await self._send_with_retry(
                method="POST",
                url=url,
                content=raw_body,
                headers=headers,
                session_id=session_id,
            )
            content = resp.content
            status_code = resp.status_code
        finally:
            self.router.release(worker.url, session_id)

        try:
            response_body = json.loads(content)
        except (json.JSONDecodeError, UnicodeDecodeError):
            response_body = {}

        latency_ms = (time.perf_counter() - t0) * 1000

        prompt_token_ids = extract_prompt_token_ids(response_body) or token_ids
        completion_token_ids = extract_completion_token_ids(response_body)

        # Translate to chat format
        choices = response_body.get("choices") or []
        discard_terminal_completion = False
        if choices:
            first_choice = choices[0]
            upstream_finish_reason = first_choice.get("finish_reason")
            raw_text = first_choice.pop("text", "")
            capped_length = bool(
                context is not None
                and context.get("output_capped")
                and upstream_finish_reason == "length"
            )
            if capped_length and not completion_token_ids:
                # A zero-token completion at the context boundary is a normal
                # terminal condition, not a renderer/infrastructure failure.
                message, finish_reason = {"role": "assistant", "content": raw_text}, None
            else:
                message, finish_reason = _translate_cumulative_message(
                    acc.renderer,
                    raw_text,
                    completion_token_ids,
                    tools=request_body.get("tools"),
                )
            first_choice["message"] = message
            if finish_reason is not None:
                first_choice["finish_reason"] = finish_reason
            discard_terminal_completion = bool(
                capped_length and not message.get("tool_calls")
            )
            if context is not None:
                context["upstream_finish_reason"] = upstream_finish_reason
                context["completion_tokens"] = len(completion_token_ids)
                if discard_terminal_completion:
                    context["training_disposition"] = _DISCARD_CONTEXT_TERMINAL
        response_body["object"] = "chat.completion"
        if context is not None:
            response_body[_CONTEXT_METADATA_KEY] = context

        # Do not advance the cumulative prefix until response translation has
        # succeeded.  A renderer/instrumentation failure must make the request
        # retryable from the previous known-good turn rather than silently
        # poisoning all subsequent prompts in this session.
        if 200 <= status_code < 300 and choices and not discard_terminal_completion:
            acc.ingest_turn(prompt_token_ids, completion_token_ids)
            acc.update_prefix(request_body.get("messages", []))

        if (
            session_id
            and response_body
            and status_code != 429
            and status_code < 500
        ):
            trace = build_trace_record(
                session_id,
                request_body,
                response_body,
                latency_ms,
                metadata=_trace_metadata(
                    request,
                    {_CONTEXT_METADATA_KEY: context}
                    if context is not None
                    else None,
                ),
                weight_version=request.state.weight_version,
                capture_raw_payloads=self.capture_raw_payloads,
            )
            await self._persist(trace)

        sanitized = response_body
        if isinstance(response_body, dict) and response_body:
            if self.strip_vllm:
                sanitized = strip_vllm_fields(response_body)
            if not originally_requested_logprobs:
                sanitized = _strip_logprobs(sanitized)

        return Response(
            content=json.dumps(sanitized),
            status_code=status_code,
            media_type="application/json",
        )

    async def _handle_cumulative_streaming(
        self,
        request: Request,
        request_body: dict[str, Any],
        completions_body: dict[str, Any],
        session_id: str,
        acc: TokenAccumulator,
        token_ids: list[int],
        context: dict[str, Any] | None = None,
    ) -> StreamingResponse:
        """Streaming cumulative turn: stream from vLLM, translate chunks to chat format."""
        completions_body["stream"] = True

        selection = await self._route_worker(
            session_id,
            input_tokens=len(token_ids),
        )
        request.state.rllm_routing = selection.metadata
        worker = selection.worker
        url = self._build_url(worker.api_url, "/v1/completions", "")
        headers = self._forward_headers(request)
        raw_body = json.dumps(completions_body).encode()

        assert self._http is not None
        upstream = self._http.stream(
            method="POST",
            url=url,
            content=raw_body,
            headers=headers,
        )
        retry_client: httpx.AsyncClient | None = None
        try:
            resp = await upstream.__aenter__()
        except _RETRYABLE_TRANSPORT_ERRORS as first_exc:
            logger.warning(
                "Cumulative streaming connection error to %s (type=%s). Retrying.",
                url,
                type(first_exc).__name__,
            )
            await self.router.wait_before_retry(
                session_id,
                attempt=1,
                timeout=self.worker_recovery_timeout,
            )
            retry_client = httpx.AsyncClient(
                timeout=httpx.Timeout(timeout=None),
                verify=shared_ssl_context(),
                limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
                follow_redirects=True,
            )
            retry_upstream = retry_client.stream(
                method="POST",
                url=url,
                content=raw_body,
                headers=headers,
            )
            try:
                resp = await retry_upstream.__aenter__()
                upstream = retry_upstream
            except Exception:
                await retry_client.aclose()
                self.router.release(worker.url, session_id)
                raise

        t0 = time.perf_counter()
        chunks: list[dict[str, Any]] = []

        async def event_generator():
            trace: TraceRecord | None = None
            raw_text_parts: list[str] = []
            upstream_finish_reason: str | None = None
            last_usage: dict[str, Any] | None = None
            saw_done = False
            buffer_for_tool_parsing = bool(request_body.get("tools"))

            def _chat_chunk(
                *,
                delta: dict[str, Any] | None = None,
                finish_reason: str | None = None,
                usage: dict[str, Any] | None = None,
            ) -> dict[str, Any]:
                source = chunks[-1] if chunks else {}
                chunk: dict[str, Any] = {
                    "id": source.get("id", ""),
                    "object": "chat.completion.chunk",
                    "created": source.get("created", 0),
                    "model": source.get("model", ""),
                    "choices": [],
                }
                if delta is not None or finish_reason is not None:
                    chunk["choices"] = [
                        {
                            "index": 0,
                            "delta": delta or {},
                            "finish_reason": finish_reason,
                        }
                    ]
                if usage is not None:
                    chunk["usage"] = usage
                if context is not None:
                    chunk[_CONTEXT_METADATA_KEY] = context
                return strip_vllm_fields(chunk) if self.strip_vllm else chunk

            def _finalize_trace() -> tuple[TraceRecord, dict[str, Any], str | None]:
                nonlocal trace
                if trace is not None:
                    return trace, trace.response_message, trace.finish_reason

                latency_ms = (time.perf_counter() - t0) * 1000
                trace = build_trace_record_from_chunks(
                    session_id,
                    request_body,
                    chunks,
                    latency_ms,
                    metadata=_trace_metadata(
                        request,
                        {_CONTEXT_METADATA_KEY: context}
                        if context is not None
                        else None,
                    ),
                    weight_version=request.state.weight_version,
                    capture_raw_payloads=self.capture_raw_payloads,
                )
                capped_length = bool(
                    context is not None
                    and context.get("output_capped")
                    and upstream_finish_reason == "length"
                )
                raw_text = "".join(raw_text_parts)
                if capped_length and not trace.completion_token_ids:
                    message, finish_override = {"role": "assistant", "content": raw_text}, None
                else:
                    message, finish_override = _translate_cumulative_message(
                        acc.renderer,
                        raw_text,
                        trace.completion_token_ids,
                        tools=request_body.get("tools"),
                    )
                trace.response_message = message
                trace.finish_reason = finish_override or upstream_finish_reason
                discard_terminal_completion = bool(
                    capped_length and not message.get("tool_calls")
                )
                if context is not None:
                    context["upstream_finish_reason"] = upstream_finish_reason
                    context["completion_tokens"] = len(trace.completion_token_ids)
                    if discard_terminal_completion:
                        context["training_disposition"] = _DISCARD_CONTEXT_TERMINAL
                    trace.metadata[_CONTEXT_METADATA_KEY] = dict(context)

                if 200 <= resp.status_code < 300 and not discard_terminal_completion:
                    prompt_ids = trace.prompt_token_ids or token_ids
                    acc.ingest_turn(prompt_ids, trace.completion_token_ids)
                    acc.update_prefix(request_body.get("messages", []))

                self._schedule_trace_store(
                    trace.trace_id,
                    trace.session_id,
                    trace.model_dump(),
                )
                return trace, message, trace.finish_reason

            try:
                first_chunk_sent = False
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        if line and not buffer_for_tool_parsing:
                            yield line + "\n"
                        continue

                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        saw_done = True
                        continue

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    chunks.append(chunk)
                    if chunk.get("usage"):
                        last_usage = chunk["usage"]

                    # Translate completions chunk → chat chunk
                    choices = chunk.get("choices", [])
                    if choices:
                        c = choices[0]
                        text = c.get("text", "")
                        if text:
                            raw_text_parts.append(text)
                        if c.get("finish_reason"):
                            upstream_finish_reason = c["finish_reason"]

                    if buffer_for_tool_parsing:
                        # Tool XML cannot be emitted as assistant content and
                        # later retracted.  Buffer tool-enabled cumulative
                        # streams until the exact completion IDs can be parsed;
                        # ordinary text-only streams retain real-time delivery.
                        continue

                    chat_chunk: dict[str, Any] = {
                        "id": chunk.get("id", ""),
                        "object": "chat.completion.chunk",
                        "created": chunk.get("created", 0),
                        "model": chunk.get("model", ""),
                        "choices": [],
                    }
                    if choices:
                        c = choices[0]
                        delta: dict[str, Any] = {}
                        if not first_chunk_sent:
                            delta["role"] = "assistant"
                            first_chunk_sent = True
                        text = c.get("text", "")
                        if text:
                            delta["content"] = text
                        chat_chunk["choices"] = [
                            {
                                "index": 0,
                                "delta": delta,
                                "finish_reason": c.get("finish_reason"),
                            }
                        ]
                    elif not chunk.get("usage"):
                        # Empty chunk with no usage either — nothing to forward
                        continue

                    if chunk.get("usage"):
                        chat_chunk["usage"] = chunk["usage"]
                    if context is not None:
                        chat_chunk[_CONTEXT_METADATA_KEY] = context

                    sanitized = strip_vllm_fields(chat_chunk) if self.strip_vllm else chat_chunk
                    yield f"data: {json.dumps(sanitized)}\n\n"

                if chunks:
                    _, message, finish_reason = _finalize_trace()
                    if buffer_for_tool_parsing:
                        delta: dict[str, Any] = {"role": "assistant"}
                        if message.get("content") is not None:
                            delta["content"] = message["content"]
                        if message.get("reasoning"):
                            delta["reasoning"] = message["reasoning"]
                        if message.get("tool_calls"):
                            delta["tool_calls"] = [
                                {"index": index, **tool_call}
                                for index, tool_call in enumerate(message["tool_calls"])
                            ]
                        yield f"data: {json.dumps(_chat_chunk(delta=delta))}\n\n"
                        yield f"data: {json.dumps(_chat_chunk(finish_reason=finish_reason, usage=last_usage))}\n\n"

                if saw_done:
                    yield "data: [DONE]\n\n"

            finally:
                await upstream.__aexit__(None, None, None)
                if retry_client is not None:
                    await retry_client.aclose()
                self.router.release(worker.url, session_id)
                if chunks and trace is None:
                    _finalize_trace()

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            status_code=resp.status_code,
        )

    # ------------------------------------------------------------------
    # Streaming (SSE)
    # ------------------------------------------------------------------

    async def _handle_streaming(
        self,
        request: Request,
        raw_body: bytes,
        request_body: dict[str, Any],
        session_id: str | None,
        originally_requested_logprobs: bool = False,
    ) -> StreamingResponse:
        if self.local_handler is not None:
            return await self._handle_streaming_local(request_body, session_id, originally_requested_logprobs, request.state.weight_version)

        selection = await self._route_worker(
            session_id,
            input_tokens=self._request_input_tokens(request, request_body),
        )
        request.state.rllm_routing = selection.metadata
        worker = selection.worker
        url = self._build_url(worker.api_url, request.url.path, str(request.url.query))
        headers = self._forward_headers(request)

        assert self._http is not None
        upstream = self._http.stream(
            method=request.method,
            url=url,
            content=raw_body,
            headers=headers,
        )
        # Retry is needed because pooled TCP connections can go stale during the
        # weight-update idle window: VPC silently drops idle sockets, and the next
        # request on that socket fails with httpx.ReadError / RemoteProtocolError
        # ("Server disconnected without sending a response") / ConnectError.
        # Without retry, these transient failures propagate as failed rollouts and
        # surface as ASGI exceptions in the agent loop.  The retry uses a fresh
        # single-use client (no pool) so it cannot hit another stale socket.
        # retry_client is non-None only when we fell back; event_generator's
        # finally block closes it after streaming completes.
        retry_client: httpx.AsyncClient | None = None
        try:
            resp = await upstream.__aenter__()
        except _RETRYABLE_TRANSPORT_ERRORS as first_exc:
            logger.warning(
                "Connection error to %s (type=%s, msg=%s). Retrying with a fresh connection.",
                url,
                type(first_exc).__name__,
                first_exc,
            )

            await self.router.wait_before_retry(
                session_id,
                attempt=1,
                timeout=self.worker_recovery_timeout,
            )

            retry_client = httpx.AsyncClient(
                timeout=httpx.Timeout(timeout=None),
                verify=shared_ssl_context(),
                limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
                follow_redirects=True,
            )
            retry_upstream = retry_client.stream(
                method=request.method,
                url=url,
                content=raw_body,
                headers=headers,
            )
            try:
                resp = await retry_upstream.__aenter__()
                upstream = retry_upstream
            except Exception:
                await retry_client.aclose()
                self.router.release(worker.url, session_id)
                raise

        t0 = time.perf_counter()
        chunks: list[dict[str, Any]] = []
        needs_strip_vllm = self.strip_vllm
        needs_strip_logprobs = not originally_requested_logprobs

        async def event_generator():
            try:
                async for line in resp.aiter_lines():
                    # Parse SSE data lines for trace capture and sanitization
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            yield "data: [DONE]\n\n"
                            continue
                        try:
                            chunk = json.loads(data_str)
                            chunks.append(chunk)
                            if not needs_strip_vllm and not needs_strip_logprobs:
                                yield f"data: {data_str}\n\n"
                            else:
                                sanitized = strip_vllm_fields(chunk) if needs_strip_vllm else chunk
                                if needs_strip_logprobs:
                                    sanitized = _strip_logprobs(sanitized)
                                yield f"data: {json.dumps(sanitized)}\n\n"
                            continue
                        except json.JSONDecodeError:
                            pass
                    # Skip blank lines — SSE separators are already included
                    # in the \n\n suffix above
                    if not line:
                        continue
                    yield line + "\n"
            finally:
                await upstream.__aexit__(None, None, None)
                if retry_client is not None:
                    await retry_client.aclose()
                self.router.release(worker.url, session_id)

                latency_ms = (time.perf_counter() - t0) * 1000
                # Build trace from accumulated chunks.
                # NOTE: We use create_task instead of await because this
                # finally block may run during GeneratorExit, where await
                # on real async I/O (e.g. aiosqlite) is not reliable.
                if session_id and chunks:
                    trace = build_trace_record_from_chunks(
                        session_id,
                        request_body,
                        chunks,
                        latency_ms,
                        metadata=_trace_metadata(request),
                        weight_version=request.state.weight_version,
                        capture_raw_payloads=self.capture_raw_payloads,
                    )
                    self._schedule_trace_store(
                        trace.trace_id,
                        trace.session_id,
                        trace.model_dump(),
                    )

                    # Ingest first turn into accumulator for cumulative token mode
                    if self.cumulative_token_mode:
                        acc = self._get_accumulator(session_id)
                        if acc.turn_count == 0:
                            prompt_ids = trace.prompt_token_ids
                            completion_ids = trace.completion_token_ids
                            if prompt_ids or completion_ids:
                                acc.ingest_turn(prompt_ids, completion_ids)
                                acc.update_prefix(request_body.get("messages", []))

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            status_code=resp.status_code,
        )

    async def _handle_streaming_local(
        self,
        request_body: dict[str, Any],
        session_id: str | None,
        originally_requested_logprobs: bool = False,
        weight_version: int | None = None,
    ) -> StreamingResponse:
        """Handle streaming when using a local handler (fake-streaming)."""
        assert self.local_handler is not None
        t0 = time.perf_counter()
        response_body = await self.local_handler(request_body)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Persist trace from the full response
        if session_id and response_body:
            trace = build_trace_record(
                session_id,
                request_body,
                response_body,
                latency_ms,
                weight_version=weight_version,
                capture_raw_payloads=self.capture_raw_payloads,
            )
            await self._persist(trace)

        needs_strip_vllm = self.strip_vllm
        needs_strip_logprobs = not originally_requested_logprobs

        # Build SSE chunks from the complete response
        chat_id = response_body.get("id", "chatcmpl-local")
        created = response_body.get("created", int(time.time()))
        model = response_body.get("model", "")
        choices = response_body.get("choices", [])
        first_choice = choices[0] if choices else {}
        message = first_choice.get("message", {})
        finish_reason = first_choice.get("finish_reason", "stop")

        def _sanitize_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
            sanitized = strip_vllm_fields(chunk) if needs_strip_vllm else chunk
            if needs_strip_logprobs:
                sanitized = _strip_logprobs(sanitized)
            return sanitized

        async def event_generator():
            def _sse(data: str) -> str:
                return f"data: {data}\n\n"

            # Chunk 1: role
            yield _sse(
                json.dumps(
                    _sanitize_chunk(
                        {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
                        }
                    )
                )
            )

            # Chunk 2: full content + token data
            delta: dict[str, Any] = {}
            if message.get("content"):
                delta["content"] = message["content"]
            if message.get("reasoning"):
                delta["reasoning"] = message["reasoning"]
            if message.get("tool_calls"):
                delta["tool_calls"] = message["tool_calls"]

            content_chunk: dict[str, Any] = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": None,
                        "token_ids": first_choice.get("token_ids", []),
                        "logprobs": first_choice.get("logprobs"),
                    }
                ],
                "prompt_token_ids": response_body.get("prompt_token_ids", []),
            }
            yield _sse(json.dumps(_sanitize_chunk(content_chunk)))

            # Chunk 3: finish + usage
            yield _sse(
                json.dumps(
                    _sanitize_chunk(
                        {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                            "usage": response_body.get("usage", {}),
                        }
                    )
                )
            )

            yield _sse("[DONE]")

        return StreamingResponse(event_generator(), media_type="text/event-stream", status_code=200)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _send_with_retry(
        self,
        method: str,
        url: str,
        content: bytes,
        headers: dict[str, str],
        session_id: str | None = None,
    ) -> httpx.Response:
        """Send one buffered request, retrying transient transport failures.

        The first attempt uses the shared high-throughput pool.  A retry uses
        a fresh single-use client because the common production failure is a
        stale keep-alive socket being closed before response headers arrive.
        HTTP error responses are returned unchanged; only failures without a
        usable response are retried.
        """
        assert self._http is not None
        last_exc: Exception | None = None
        for attempt in range(1 + self.max_retries):
            retry_client: httpx.AsyncClient | None = None
            client = self._http
            if attempt > 0:
                retry_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(timeout=None),
                    verify=shared_ssl_context(),
                    limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
                    follow_redirects=True,
                )
                client = retry_client
            try:
                return await client.request(method, url, content=content, headers=headers)
            except _RETRYABLE_TRANSPORT_ERRORS as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    logger.warning(
                        "Transient upstream transport error to %s "
                        "(type=%s, attempt %d/%d): %s; retrying with a fresh connection",
                        url,
                        type(exc).__name__,
                        attempt + 1,
                        self.max_retries + 1,
                        exc,
                    )
                    await self.router.wait_before_retry(
                        session_id,
                        attempt=attempt + 1,
                        timeout=self.worker_recovery_timeout,
                    )
            finally:
                if retry_client is not None:
                    await retry_client.aclose()
        raise last_exc  # type: ignore[misc]

    async def _persist(self, trace: TraceRecord) -> None:
        try:
            data = trace.model_dump()
            if self.sync_traces:
                await self._safe_store(trace.trace_id, trace.session_id, data)
            else:
                self._schedule_trace_store(trace.trace_id, trace.session_id, data)
        except Exception:
            logger.exception("Failed to persist trace %s", trace.trace_id)

    async def _safe_store(self, trace_id: str, session_id: str, data: dict[str, Any]) -> None:
        if self.is_session_sealed(session_id):
            logger.debug(
                "Skipping late trace %s for deleted session %s",
                trace_id,
                session_id,
            )
            return
        try:
            await self.store.store_trace(trace_id, session_id, data)
        except Exception:
            logger.exception("Failed to persist trace %s", trace_id)

    @staticmethod
    def _build_url(worker_url: str, path: str, query: str, *, gateway_prefix: str = "/v1") -> str:
        base = worker_url.rstrip("/")
        # Strip the gateway's own prefix to get the tail (e.g. /chat/completions).
        # The gateway always exposes routes under /v1/{path}, so request paths
        # arrive as /v1/... regardless of the worker's actual api_path.
        if path.startswith(gateway_prefix):
            path = path[len(gateway_prefix) :]
        url = f"{base}{path}"
        if query:
            url = f"{url}?{query}"
        return url

    @staticmethod
    def _forward_headers(request: Request) -> dict[str, str]:
        return {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}
