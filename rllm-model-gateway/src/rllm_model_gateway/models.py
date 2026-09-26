"""Pydantic data models for the rllm-model-gateway."""

import math
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator


class TraceRecord(BaseModel):
    """A single captured LLM call with full token-level data."""

    trace_id: str
    session_id: str
    model: str = ""
    # Input
    messages: list[dict[str, Any]] = Field(default_factory=list)
    prompt_token_ids: list[int] = Field(default_factory=list)
    # Output
    response_message: dict[str, Any] = Field(default_factory=dict)
    completion_token_ids: list[int] = Field(default_factory=list)
    logprobs: list[float] | None = None
    routing_matrices: list[str] | None = None
    finish_reason: str | None = None
    weight_version: int | None = None
    # Metadata
    latency_ms: float = 0.0
    token_counts: dict[str, int] = Field(default_factory=dict)
    timestamp: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw_request: dict[str, Any] | None = None
    raw_response: dict[str, Any] | None = None


def _split_worker_url(raw: str) -> dict[str, str]:
    """Split ``http://host:port/v1`` into base URL + api_path.

    If the URL contains a path component (e.g. ``/v1``), it is separated
    out so that health checks can use the bare ``scheme://host:port`` while
    proxying uses ``scheme://host:port + api_path``.
    """
    parsed = urlparse(raw.rstrip("/"))
    if parsed.path and parsed.path != "/":
        base = f"{parsed.scheme}://{parsed.netloc}"
        return {"url": base, "api_path": parsed.path}
    return {"url": raw.rstrip("/"), "api_path": "/v1"}


class WorkerConfig(BaseModel):
    """Configuration for a single inference worker."""

    worker_id: str = ""
    url: str  # base URL, e.g. "http://localhost:4000"
    api_path: str = "/v1"  # API version prefix, appended for proxying
    model_name: str | None = None
    weight: int = 1

    @model_validator(mode="before")
    @classmethod
    def _auto_split_url(cls, values: Any) -> Any:
        """Backward compat: auto-split url with path into url + api_path."""
        if isinstance(values, dict):
            url = values.get("url", "")
            # Only auto-split if api_path was NOT explicitly provided
            if url and "api_path" not in values:
                parts = _split_worker_url(url)
                values["url"] = parts["url"]
                values["api_path"] = parts["api_path"]
        return values


class WorkerInfo(BaseModel):
    """Runtime info for a worker including health state."""

    worker_id: str
    url: str  # base URL
    api_path: str = "/v1"
    model_name: str | None = None
    weight: int = 1
    healthy: bool = True
    active_requests: int = 0
    pinned_sessions: int = 0
    pinned_context_tokens: int = 0

    @model_validator(mode="before")
    @classmethod
    def _auto_split_url(cls, values: Any) -> Any:
        """Auto-split url with path into url + api_path."""
        if isinstance(values, dict):
            url = values.get("url", "")
            if url and "api_path" not in values:
                parts = _split_worker_url(url)
                values["url"] = parts["url"]
                values["api_path"] = parts["api_path"]
        return values

    @property
    def api_url(self) -> str:
        """Full URL for API proxying: base + api_path."""
        return self.url.rstrip("/") + self.api_path


class SessionInfo(BaseModel):
    """Session metadata returned by session management APIs."""

    session_id: str
    trace_count: int = 0
    created_at: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GatewayRoutingConfig(BaseModel):
    """Built-in session routing configuration."""

    mode: Literal["sticky_least_loaded", "group_striped_adaptive"] = (
        "sticky_least_loaded"
    )
    migration_min_active_gap: int = Field(default=4, gt=0)
    migration_sustain_seconds: float = Field(default=30.0, gt=0)
    max_migrations_per_session: int = Field(default=1, ge=0)
    metrics_interval_seconds: float = Field(default=30.0, gt=0)

    @field_validator(
        "migration_min_active_gap",
        "max_migrations_per_session",
        mode="before",
    )
    @classmethod
    def _validate_integer_fields(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("routing integer thresholds must be integers")
        return value

    @field_validator(
        "migration_sustain_seconds",
        "metrics_interval_seconds",
        mode="before",
    )
    @classmethod
    def _validate_duration_fields(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("routing durations must be finite numbers")
        if not math.isfinite(float(value)):
            raise ValueError("routing durations must be finite numbers")
        return value


class GatewayConfig(BaseModel):
    """Top-level gateway configuration."""

    host: str = "0.0.0.0"
    port: int = 9090
    # Internal process-supervision channel.  These fields are populated by the
    # training-side GatewayManager and are intentionally not part of Hydra or
    # user-facing launch configuration.
    supervision_port: int | None = Field(default=None, ge=1, le=65535)
    supervision_generation: str | None = None
    workers: list[WorkerConfig] = Field(default_factory=list)
    db_path: str | None = None
    store_worker: str = "memory"
    add_logprobs: bool = True
    add_return_token_ids: bool = True
    strip_vllm_fields: bool = True
    # Raw provider payloads duplicate the normalized trace fields and can be
    # disabled for high-concurrency training while preserving API compatibility.
    capture_raw_payloads: bool = True
    routing_policy: str | None = None
    routing: GatewayRoutingConfig = Field(default_factory=GatewayRoutingConfig)
    health_check_interval: float = Field(default=10.0, gt=0)
    # vLLM's HTTP event loop can take longer than five seconds to answer
    # /health while a large chunked-prefill wave is active.  A longer probe
    # timeout avoids treating ordinary saturation as process death.
    worker_health_check_timeout: float = Field(default=15.0, gt=0)
    worker_health_check_failure_threshold: int = Field(default=3, gt=0)
    # Weight updates intentionally abort in-flight generation.  While the
    # fleet is in maintenance, and briefly after it leaves maintenance,
    # failed health probes must not evict every worker from the routing ring.
    maintenance_health_grace_seconds: float = Field(default=120.0, ge=0)
    # Requests already waiting at the maintenance barrier are released in a
    # bounded ramp.  This is a recovery-only limiter; normal steady-state
    # admission remains unlimited.
    recovery_admission_rate_per_second: float = Field(default=64.0, gt=0)
    recovery_admission_burst: int = Field(default=64, gt=0)
    recovery_admission_jitter_seconds: float = Field(default=2.0, ge=0)
    # A brief fleet-wide health-check fluctuation should pause requests rather
    # than fail all active rollouts. This remains below the agent's 3600s model
    # request timeout so a permanent outage still surfaces as a retryable 503.
    worker_recovery_timeout: float = Field(default=600.0, gt=0)
    # Session DELETE tombstones synchronously and reclaims model/trace state in
    # this bounded background pool. This protects the control plane from a
    # speculative-wave cancellation burst without reducing rollout concurrency.
    session_cleanup_concurrency: int = Field(default=32, gt=0)
    session_cleanup_phase_timeout: float = Field(default=30.0, gt=0)
    session_cleanup_shutdown_timeout: float = Field(default=30.0, gt=0)
    session_cleanup_max_pending: int = Field(default=4096, gt=0)
    session_cleanup_stall_timeout: float = Field(default=600.0, gt=0)
    log_level: str = "INFO"
    sync_traces: bool = False
    model: str | None = None  # When set, overrides ``body.model``
    cumulative_token_mode: bool = False
    # Enables turn-zero pre-rendering and per-trajectory model-window budgets.
    # Kept separate so fixed-partition callers retain their historical path.
    dynamic_sequence_budget: bool = False
    # Total prompt + completion token capacity enforced for cumulative turns.
    # ``None`` preserves the historical transparent-proxy behaviour.
    max_context_tokens: int | None = Field(default=None, gt=0)
    # renderers family for the cumulative-mode bridge. Check supported model families
    # in MODEL_RENDERER_MAP of https://github.com/PrimeIntellect-ai/renderers/blob/main/renderers/base.py
    renderer_family: str = "auto"

    @field_validator(
        "worker_health_check_failure_threshold",
        "recovery_admission_burst",
        "session_cleanup_concurrency",
        "session_cleanup_max_pending",
        mode="before",
    )
    @classmethod
    def _validate_gateway_integer_fields(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("gateway integer thresholds must be integers")
        return value

    @field_validator(
        "health_check_interval",
        "worker_health_check_timeout",
        "maintenance_health_grace_seconds",
        "recovery_admission_rate_per_second",
        "recovery_admission_jitter_seconds",
        "session_cleanup_phase_timeout",
        "session_cleanup_shutdown_timeout",
        "session_cleanup_stall_timeout",
        mode="before",
    )
    @classmethod
    def _validate_gateway_duration_fields(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("gateway durations and rates must be finite numbers")
        if not math.isfinite(float(value)):
            raise ValueError("gateway durations and rates must be finite numbers")
        return value

    @model_validator(mode="after")
    def _validate_routing(self) -> "GatewayConfig":
        if (self.supervision_port is None) != (
            self.supervision_generation is None
        ):
            raise ValueError(
                "supervision_port and supervision_generation must be set together"
            )
        if self.supervision_generation is not None and not (
            self.supervision_generation.strip()
        ):
            raise ValueError("supervision_generation must be non-empty")
        if self.routing.mode == "group_striped_adaptive":
            if not self.cumulative_token_mode:
                raise ValueError(
                    "routing.mode='group_striped_adaptive' requires "
                    "cumulative_token_mode=True"
                )
            if self.routing_policy:
                raise ValueError(
                    "routing_policy cannot be combined with "
                    "routing.mode='group_striped_adaptive'"
                )
        return self
