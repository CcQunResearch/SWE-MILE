"""Session-sticky worker routing with pluggable policies.

Reference implementations:
- verl ``AsyncLLMServerManager._choose_server()`` — LRU cache for session stickiness
- miles ``MilesRouter._use_url()`` / ``_finish_url()`` — least-loaded selection
"""

import asyncio
import hashlib
import logging
import time
from collections import Counter, OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from rllm_model_gateway.models import GatewayRoutingConfig, WorkerInfo

logger = logging.getLogger(__name__)

_ADAPTIVE_EVENT_LOG_MIN_INTERVAL_SECONDS = 60.0


class NoHealthyWorkersError(RuntimeError):
    """No inference worker is currently eligible for routing."""


# ------------------------------------------------------------------
# RoutingPolicy protocol
# ------------------------------------------------------------------


class RoutingPolicy(Protocol):
    """Pluggable worker selection strategy."""

    def select_worker(
        self,
        workers: list[WorkerInfo],
        session_id: str | None,
        active_counts: dict[str, int],
    ) -> WorkerInfo: ...

    def on_worker_change(self, workers: list[WorkerInfo]) -> None: ...


# ------------------------------------------------------------------
# Default policy: sticky sessions with least-loaded fallback
# ------------------------------------------------------------------


class LRUCache:
    """Minimal LRU cache backed by ``OrderedDict``."""

    def __init__(self, maxsize: int = 10_000) -> None:
        self._data: OrderedDict[str, str] = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: str) -> str | None:
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        return None

    def put(self, key: str, value: str) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        if len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def clear(self) -> None:
        self._data.clear()


class StickyLeastLoadedPolicy:
    """Sticky sessions with least-loaded fallback.

    Mirrors verl ``AsyncLLMServerManager._choose_server()`` (LRU cache mapping
    session_id → worker) with a least-loaded tiebreaker for new sessions.
    """

    def __init__(self, max_cache_size: int = 10_000) -> None:
        self._cache = LRUCache(maxsize=max_cache_size)

    def select_worker(
        self,
        workers: list[WorkerInfo],
        session_id: str | None,
        active_counts: dict[str, int],
    ) -> WorkerInfo:
        worker_urls = {w.url for w in workers}

        # Sticky: return cached worker if still alive
        if session_id:
            cached_url = self._cache.get(session_id)
            if cached_url and cached_url in worker_urls:
                return next(w for w in workers if w.url == cached_url)

        # Fallback: least-loaded
        worker = min(workers, key=lambda w: active_counts.get(w.url, 0))
        if session_id:
            self._cache.put(session_id, worker.url)
        return worker

    def on_worker_change(self, workers: list[WorkerInfo]) -> None:
        self._cache.clear()


@dataclass
class SessionBinding:
    session_id: str
    worker_url: str
    group_id: str | None = None
    slot: int | None = None
    group_size: int | None = None
    latest_input_tokens: int = 0
    migrations: int = 0
    active_requests: int = 0
    routed_requests: int = 0
    initial_reason: str = "power_of_two"
    bound_at: float = 0.0


@dataclass(frozen=True)
class RouteSelection:
    worker: WorkerInfo
    metadata: dict[str, Any]


class AdaptiveGroupStickyPolicy:
    """Group-striped initial placement with bounded sticky work stealing."""

    def __init__(
        self,
        config: GatewayRoutingConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._clock = clock
        self._bindings: dict[str, SessionBinding] = {}
        self._hints: dict[str, tuple[str, int, int] | None] = {}
        self._pinned_sessions: Counter[str] = Counter()
        self._pinned_context_tokens: Counter[str] = Counter()
        self._overloaded_since: dict[str, float] = {}
        self._metrics: Counter[str] = Counter()

    @staticmethod
    def _stable_int(value: str) -> int:
        digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big")

    @staticmethod
    def _routing_hint(metadata: dict[str, Any] | None) -> tuple[str, int, int] | None:
        if not metadata:
            return None
        raw = metadata.get("routing")
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError("session metadata.routing must be an object")
        if raw.get("schema_version") != 1:
            raise ValueError("session metadata.routing.schema_version must be 1")
        group_id = raw.get("group_id")
        slot = raw.get("slot")
        group_size = raw.get("group_size")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("session metadata.routing.group_id must be non-empty")
        if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
            raise ValueError("session metadata.routing.slot must be a non-negative integer")
        if (
            isinstance(group_size, bool)
            or not isinstance(group_size, int)
            or group_size <= 0
        ):
            raise ValueError("session metadata.routing.group_size must be a positive integer")
        if slot >= group_size:
            raise ValueError("session metadata.routing.slot must be smaller than group_size")
        return group_id, slot, group_size

    def register_session(
        self,
        session_id: str,
        metadata: dict[str, Any] | None,
        workers: list[WorkerInfo],
        dead_workers: set[str],
        active_counts: dict[str, int],
    ) -> None:
        hint = self._routing_hint(metadata)
        if session_id in self._hints and self._hints[session_id] != hint:
            raise ValueError(f"conflicting routing metadata for session {session_id}")
        self._hints[session_id] = hint
        if session_id not in self._bindings:
            healthy = [w for w in workers if w.url not in dead_workers]
            if healthy:
                self._bind(session_id, healthy, active_counts)

    def close_session(self, session_id: str) -> None:
        binding = self._bindings.pop(session_id, None)
        self._hints.pop(session_id, None)
        if binding is None:
            return
        self._pinned_sessions[binding.worker_url] = max(
            0, self._pinned_sessions[binding.worker_url] - 1
        )
        self._pinned_context_tokens[binding.worker_url] = max(
            0,
            self._pinned_context_tokens[binding.worker_url]
            - binding.latest_input_tokens,
        )
        self._metrics["sessions_closed"] += 1

    def _score(
        self,
        worker: WorkerInfo,
        active_counts: dict[str, int],
    ) -> tuple[int, int, int, str]:
        return (
            active_counts.get(worker.url, 0),
            self._pinned_context_tokens.get(worker.url, 0),
            self._pinned_sessions.get(worker.url, 0),
            worker.url,
        )

    def _power_of_two(
        self,
        session_id: str,
        workers: list[WorkerInfo],
        active_counts: dict[str, int],
    ) -> WorkerInfo:
        if len(workers) == 1:
            return workers[0]
        first = self._stable_int(f"{session_id}:a") % len(workers)
        second = self._stable_int(f"{session_id}:b") % len(workers)
        if second == first:
            second = (second + 1) % len(workers)
        return min(
            (workers[first], workers[second]),
            key=lambda worker: self._score(worker, active_counts),
        )

    def _bind(
        self,
        session_id: str,
        workers: list[WorkerInfo],
        active_counts: dict[str, int],
    ) -> SessionBinding:
        hint = self._hints.get(session_id)
        if hint is not None:
            group_id, slot, group_size = hint
            offset = self._stable_int(group_id) % len(workers)
            worker = workers[(offset + slot) % len(workers)]
            reason = "group_striped"
            self._metrics["group_striped_bindings"] += 1
        else:
            group_id = None
            slot = None
            group_size = None
            worker = self._power_of_two(session_id, workers, active_counts)
            reason = "power_of_two"
            self._metrics["power_of_two_bindings"] += 1
        binding = SessionBinding(
            session_id=session_id,
            worker_url=worker.url,
            group_id=group_id,
            slot=slot,
            group_size=group_size,
            initial_reason=reason,
            bound_at=self._clock(),
        )
        self._bindings[session_id] = binding
        self._pinned_sessions[worker.url] += 1
        self._metrics["initial_bindings"] += 1
        return binding

    def _least_loaded(
        self,
        workers: list[WorkerInfo],
        active_counts: dict[str, int],
        *,
        exclude: str | None = None,
    ) -> WorkerInfo | None:
        candidates = [worker for worker in workers if worker.url != exclude]
        if not candidates:
            return None
        return min(candidates, key=lambda worker: self._score(worker, active_counts))

    def _move(
        self,
        binding: SessionBinding,
        target: WorkerInfo,
        *,
        adaptive: bool,
    ) -> None:
        source_url = binding.worker_url
        if source_url == target.url:
            return
        self._pinned_sessions[source_url] = max(
            0, self._pinned_sessions[source_url] - 1
        )
        self._pinned_context_tokens[source_url] = max(
            0,
            self._pinned_context_tokens[source_url]
            - binding.latest_input_tokens,
        )
        binding.worker_url = target.url
        binding.bound_at = self._clock()
        self._pinned_sessions[target.url] += 1
        self._pinned_context_tokens[target.url] += binding.latest_input_tokens
        if adaptive:
            binding.migrations += 1
            self._metrics["adaptive_migrations"] += 1
        else:
            self._metrics["failover_rebindings"] += 1

    def _update_context(self, binding: SessionBinding, input_tokens: int | None) -> None:
        if input_tokens is None:
            return
        if (
            isinstance(input_tokens, bool)
            or not isinstance(input_tokens, int)
            or input_tokens < 0
        ):
            raise ValueError("input_tokens must be a non-negative integer")
        delta = input_tokens - binding.latest_input_tokens
        binding.latest_input_tokens = input_tokens
        self._pinned_context_tokens[binding.worker_url] = max(
            0, self._pinned_context_tokens[binding.worker_url] + delta
        )

    def select(
        self,
        workers: list[WorkerInfo],
        session_id: str | None,
        active_counts: dict[str, int],
        *,
        input_tokens: int | None = None,
    ) -> RouteSelection:
        if not workers:
            raise NoHealthyWorkersError("No healthy workers available")
        if not session_id:
            worker = min(workers, key=lambda item: self._score(item, active_counts))
            return RouteSelection(
                worker=worker,
                metadata={"schema_version": 1, "reason": "anonymous"},
            )

        if session_id not in self._hints:
            self._hints[session_id] = None
        binding = self._bindings.get(session_id)
        if binding is None:
            binding = self._bind(session_id, workers, active_counts)

        self._update_context(binding, input_tokens)
        route_reason = binding.initial_reason if binding.routed_requests == 0 else "sticky"
        source_url: str | None = None
        prefix_rebuild_tokens = 0
        worker_by_url = {worker.url: worker for worker in workers}

        if binding.worker_url not in worker_by_url:
            target = self._least_loaded(workers, active_counts)
            assert target is not None
            source_url = binding.worker_url
            self._move(binding, target, adaptive=False)
            route_reason = "failover"
            prefix_rebuild_tokens = binding.latest_input_tokens
            self._metrics["prefix_rebuild_tokens"] += prefix_rebuild_tokens
        elif (
            binding.active_requests == 0
            and binding.migrations < self.config.max_migrations_per_session
        ):
            target = self._least_loaded(
                workers,
                active_counts,
                exclude=binding.worker_url,
            )
            overloaded_at = self._overloaded_since.get(binding.worker_url)
            if target is not None and overloaded_at is not None:
                gap = active_counts.get(binding.worker_url, 0) - active_counts.get(
                    target.url, 0
                )
                if (
                    gap >= self.config.migration_min_active_gap
                    and self._clock() - overloaded_at
                    >= self.config.migration_sustain_seconds
                ):
                    source_url = binding.worker_url
                    self._move(binding, target, adaptive=True)
                    route_reason = "adaptive_migration"
                    prefix_rebuild_tokens = binding.latest_input_tokens
                    self._metrics["prefix_rebuild_tokens"] += prefix_rebuild_tokens

        binding.active_requests += 1
        binding.routed_requests += 1
        worker = next(worker for worker in workers if worker.url == binding.worker_url)
        return RouteSelection(
            worker=worker,
            metadata={
                "schema_version": 1,
                "worker_id": worker.worker_id,
                "reason": route_reason,
                "migration_count": binding.migrations,
                "source_worker_url": source_url,
                "prefix_rebuild_tokens": prefix_rebuild_tokens,
            },
        )

    def release_request(self, session_id: str | None) -> None:
        if not session_id:
            return
        binding = self._bindings.get(session_id)
        if binding is not None:
            binding.active_requests = max(0, binding.active_requests - 1)

    def observe_load(
        self,
        workers: list[WorkerInfo],
        dead_workers: set[str],
        active_counts: dict[str, int],
    ) -> None:
        healthy = [worker for worker in workers if worker.url not in dead_workers]
        now = self._clock()
        healthy_urls = {worker.url for worker in healthy}
        for worker in healthy:
            target = self._least_loaded(healthy, active_counts, exclude=worker.url)
            gap = (
                active_counts.get(worker.url, 0)
                - active_counts.get(target.url, 0)
                if target is not None
                else 0
            )
            if gap >= self.config.migration_min_active_gap:
                self._overloaded_since.setdefault(worker.url, now)
            else:
                self._overloaded_since.pop(worker.url, None)
        for url in list(self._overloaded_since):
            if url not in healthy_urls:
                self._overloaded_since.pop(url, None)

    def pinned_sessions(self, worker_url: str) -> int:
        return self._pinned_sessions.get(worker_url, 0)

    def pinned_context_tokens(self, worker_url: str) -> int:
        return self._pinned_context_tokens.get(worker_url, 0)

    def stats(
        self,
        workers: list[WorkerInfo],
        dead_workers: set[str],
        active_counts: dict[str, int],
    ) -> dict[str, Any]:
        per_worker = [
            {
                "worker_id": worker.worker_id,
                "url": worker.url,
                "healthy": worker.url not in dead_workers,
                "active_requests": active_counts.get(worker.url, 0),
                "pinned_sessions": self.pinned_sessions(worker.url),
                "pinned_context_tokens": self.pinned_context_tokens(worker.url),
            }
            for worker in workers
        ]
        aggregate_workers = [item for item in per_worker if item["healthy"]]
        if not aggregate_workers:
            aggregate_workers = per_worker
        active = [item["active_requests"] for item in aggregate_workers] or [0]
        pinned = [item["pinned_sessions"] for item in aggregate_workers] or [0]
        tokens = [item["pinned_context_tokens"] for item in aggregate_workers] or [0]
        return {
            "mode": "group_striped_adaptive",
            "workers": per_worker,
            "initial_bindings": self._metrics["initial_bindings"],
            "group_striped_bindings": self._metrics["group_striped_bindings"],
            "power_of_two_bindings": self._metrics["power_of_two_bindings"],
            "adaptive_migrations": self._metrics["adaptive_migrations"],
            "failover_rebindings": self._metrics["failover_rebindings"],
            "prefix_rebuild_tokens": self._metrics["prefix_rebuild_tokens"],
            "pinned_sessions_max": max(pinned),
            "pinned_sessions_min": min(pinned),
            "pinned_context_tokens_max": max(tokens),
            "pinned_context_tokens_min": min(tokens),
            "active_requests_max": max(active),
            "active_requests_min": min(active),
        }


# ------------------------------------------------------------------
# SessionRouter — manages worker pool + delegates to policy
# ------------------------------------------------------------------


class SessionRouter:
    """Worker pool manager with pluggable routing policy.

    Reference: miles ``MilesRouter._use_url()`` / ``_finish_url()``
    (``miles/router/router.py`` lines 214-235).
    """

    def __init__(
        self,
        policy: RoutingPolicy | None = None,
        routing_config: GatewayRoutingConfig | None = None,
        health_check_interval: float = 10.0,
        health_check_timeout: float = 15.0,
        failure_threshold: int = 3,
        maintenance_health_grace_seconds: float = 120.0,
        recovery_admission_rate_per_second: float = 64.0,
        recovery_admission_burst: int = 64,
        recovery_admission_jitter_seconds: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.routing_config = routing_config or GatewayRoutingConfig()
        self.workers: list[WorkerInfo] = []
        self.dead_workers: set[str] = set()
        self.active_counts: dict[str, int] = {}
        self.failure_counts: dict[str, int] = {}
        self.policy: RoutingPolicy = policy or StickyLeastLoadedPolicy()
        self.adaptive_policy: AdaptiveGroupStickyPolicy | None = None
        if self.routing_config.mode == "group_striped_adaptive":
            if policy is not None:
                raise ValueError(
                    "custom routing policy cannot be combined with "
                    "group_striped_adaptive"
                )
            self.adaptive_policy = AdaptiveGroupStickyPolicy(
                self.routing_config,
                clock=clock,
            )
        self._clock = clock
        self._health_interval = health_check_interval
        self._health_timeout = health_check_timeout
        self._failure_threshold = failure_threshold
        self._maintenance_health_grace_seconds = (
            maintenance_health_grace_seconds
        )
        self._recovery_admission_rate = recovery_admission_rate_per_second
        self._recovery_admission_burst = recovery_admission_burst
        self._recovery_admission_jitter = recovery_admission_jitter_seconds
        self._maintenance_active = False
        self._maintenance_transition_id: str | None = None
        self._maintenance_transition_status: dict[str, Any] | None = None
        self._maintenance_started_at: float | None = None
        self._health_grace_until = 0.0
        self._admission_available = asyncio.Event()
        self._admission_available.set()
        self._maintenance_waiters = 0
        self._recovery_release_started_at = 0.0
        self._recovery_admission_slot = 0
        self._recovery_admission_lock = asyncio.Lock()
        self._health_task: asyncio.Task[None] | None = None
        self._http: httpx.AsyncClient | None = None
        self._healthy_workers_available = asyncio.Event()
        self._last_adaptive_log_at: float | None = None
        self._last_adaptive_log_stats: dict[str, Any] | None = None
        self._last_adaptive_imbalance: bool | None = None
        self._pending_adaptive_log_reasons: set[str] = set()

    # -- Maintenance / admission ------------------------------------------

    def set_maintenance(
        self,
        *,
        active: bool,
        transition_id: str,
        health_grace_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Idempotently close or reopen inference admission.

        Administrative endpoints remain available while this barrier is
        closed.  Only model requests wait, so a weight-version update cannot
        be starved behind the retry wave caused by the weight update itself.
        """

        if not transition_id:
            raise ValueError("maintenance transition_id must be non-empty")
        if (
            transition_id == self._maintenance_transition_id
            and active != self._maintenance_active
        ):
            raise ValueError(
                "maintenance transition_id was reused with a different state"
            )
        if (
            transition_id == self._maintenance_transition_id
            and active == self._maintenance_active
        ):
            assert self._maintenance_transition_status is not None
            return dict(self._maintenance_transition_status)

        grace = (
            self._maintenance_health_grace_seconds
            if health_grace_seconds is None
            else float(health_grace_seconds)
        )
        if grace < 0:
            raise ValueError("maintenance health grace must be non-negative")

        now = self._clock()
        self._maintenance_transition_id = transition_id
        self._health_grace_until = max(self._health_grace_until, now + grace)
        # A probe failure observed during a controlled transition must not be
        # carried across the grace boundary as part of a consecutive streak.
        for url in self.failure_counts:
            self.failure_counts[url] = 0

        if active:
            self._maintenance_active = True
            self._maintenance_started_at = now
            self._admission_available.clear()
            logger.info(
                "Inference maintenance entered: transition=%s grace=%.1fs",
                transition_id,
                grace,
            )
        else:
            self._maintenance_active = False
            self._maintenance_started_at = None
            self._recovery_release_started_at = now
            self._recovery_admission_slot = 0
            self._admission_available.set()
            logger.info(
                "Inference maintenance exited: transition=%s waiters=%d "
                "recovery_rate=%.1f/s burst=%d jitter=%.1fs grace=%.1fs",
                transition_id,
                self._maintenance_waiters,
                self._recovery_admission_rate,
                self._recovery_admission_burst,
                self._recovery_admission_jitter,
                grace,
            )
        status = self.maintenance_status()
        self._maintenance_transition_status = dict(status)
        return status

    def maintenance_status(self) -> dict[str, Any]:
        now = self._clock()
        return {
            "active": self._maintenance_active,
            "transition_id": self._maintenance_transition_id,
            "waiters": self._maintenance_waiters,
            "health_grace_remaining_seconds": max(
                0.0, self._health_grace_until - now
            ),
            "recovery_admissions": self._recovery_admission_slot,
        }

    @staticmethod
    def _stable_fraction(value: str) -> float:
        digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") / float((1 << 64) - 1)

    async def wait_for_admission(
        self,
        session_id: str | None = None,
        *,
        timeout: float | None = None,
    ) -> bool:
        """Wait for maintenance to finish and ramp only the queued cohort.

        Returns ``True`` when this request actually waited at the barrier.
        Requests arriving after maintenance has ended bypass this recovery
        limiter, preserving the configured steady-state rollout concurrency.
        """

        if self._admission_available.is_set():
            return False

        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        self._maintenance_waiters += 1
        try:
            if deadline is None:
                await self._admission_available.wait()
            else:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for maintenance exit")
                try:
                    await asyncio.wait_for(
                        self._admission_available.wait(), timeout=remaining
                    )
                except TimeoutError as exc:
                    raise TimeoutError(
                        "timed out waiting for maintenance exit"
                    ) from exc

            async with self._recovery_admission_lock:
                slot = self._recovery_admission_slot
                self._recovery_admission_slot += 1

            rate_delay = max(
                0.0,
                (
                    slot - self._recovery_admission_burst + 1
                )
                / self._recovery_admission_rate,
            )
            jitter_key = (
                f"{self._maintenance_transition_id}:"
                f"{session_id or 'anonymous'}:{slot}"
            )
            jitter = (
                self._stable_fraction(jitter_key)
                * self._recovery_admission_jitter
            )
            target = self._recovery_release_started_at + rate_delay + jitter
            delay = max(0.0, target - self._clock())
            if delay:
                if deadline is not None and loop.time() + delay > deadline:
                    raise TimeoutError(
                        "timed out during maintenance recovery admission"
                    )
                await asyncio.sleep(delay)
            return True
        finally:
            self._maintenance_waiters = max(0, self._maintenance_waiters - 1)

    async def wait_before_retry(
        self,
        session_id: str | None,
        *,
        attempt: int,
        timeout: float | None = None,
    ) -> None:
        """Coordinate upstream retries with maintenance and add jitter."""

        waited = await self.wait_for_admission(session_id, timeout=timeout)
        if waited or self._recovery_admission_jitter <= 0:
            return
        key = f"retry:{session_id or 'anonymous'}:{attempt}"
        delay = self._stable_fraction(key) * self._recovery_admission_jitter
        if delay:
            await asyncio.sleep(delay)

    # -- Worker management -------------------------------------------------

    def add_worker(self, worker: WorkerInfo) -> None:
        if any(w.url == worker.url for w in self.workers):
            return
        self.workers.append(worker)
        self.active_counts.setdefault(worker.url, 0)
        self.failure_counts.setdefault(worker.url, 0)
        self.dead_workers.discard(worker.url)
        self._refresh_healthy_workers_event()
        if self.adaptive_policy is None:
            self.policy.on_worker_change(self.workers)

    def remove_worker(self, worker_url: str) -> None:
        self.workers = [w for w in self.workers if w.url != worker_url]
        self.active_counts.pop(worker_url, None)
        self.failure_counts.pop(worker_url, None)
        self.dead_workers.discard(worker_url)
        self._refresh_healthy_workers_event()
        if self.adaptive_policy is None:
            self.policy.on_worker_change(self.workers)

    def get_workers(self) -> list[WorkerInfo]:
        result: list[WorkerInfo] = []
        for w in self.workers:
            result.append(
                WorkerInfo(
                    worker_id=w.worker_id,
                    url=w.url,
                    api_path=w.api_path,
                    model_name=w.model_name,
                    weight=w.weight,
                    healthy=w.url not in self.dead_workers,
                    active_requests=self.active_counts.get(w.url, 0),
                    pinned_sessions=(
                        self.adaptive_policy.pinned_sessions(w.url)
                        if self.adaptive_policy is not None
                        else 0
                    ),
                    pinned_context_tokens=(
                        self.adaptive_policy.pinned_context_tokens(w.url)
                        if self.adaptive_policy is not None
                        else 0
                    ),
                )
            )
        return result

    # -- Routing -----------------------------------------------------------

    def route(self, session_id: str | None = None) -> WorkerInfo:
        return self.route_request(session_id).worker

    def route_request(
        self,
        session_id: str | None = None,
        *,
        input_tokens: int | None = None,
    ) -> RouteSelection:
        healthy = [w for w in self.workers if w.url not in self.dead_workers]
        if not healthy:
            raise NoHealthyWorkersError("No healthy workers available")
        if self.adaptive_policy is not None:
            selection = self.adaptive_policy.select(
                healthy,
                session_id,
                self.active_counts,
                input_tokens=input_tokens,
            )
            worker = selection.worker
        else:
            worker = self.policy.select_worker(healthy, session_id, self.active_counts)
            selection = RouteSelection(
                worker=worker,
                metadata={
                    "schema_version": 1,
                    "worker_id": worker.worker_id,
                    "reason": "sticky_least_loaded",
                    "migration_count": 0,
                    "source_worker_url": None,
                    "prefix_rebuild_tokens": 0,
                },
            )
        self.active_counts[worker.url] = self.active_counts.get(worker.url, 0) + 1
        self._observe_adaptive_load()
        return selection

    async def route_when_available(
        self,
        session_id: str | None = None,
        *,
        timeout: float | None = None,
    ) -> WorkerInfo:
        """Route now or wait until a health check observes a recovered worker.

        A fleet-wide transient health-check failure must not turn every in-flight
        rollout into an immediate HTTP 500.  The loop rechecks after clearing the
        event to close the recovery-between-check-and-wait race.
        """

        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        while True:
            try:
                return self.route(session_id)
            except NoHealthyWorkersError:
                self._healthy_workers_available.clear()
                if any(w.url not in self.dead_workers for w in self.workers):
                    self._healthy_workers_available.set()
                    continue

                if deadline is None:
                    await self._healthy_workers_available.wait()
                    continue

                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TimeoutError(
                        "timed out waiting for a healthy inference worker"
                    ) from None
                try:
                    await asyncio.wait_for(
                        self._healthy_workers_available.wait(),
                        timeout=remaining,
                    )
                except TimeoutError as exc:
                    raise TimeoutError(
                        "timed out waiting for a healthy inference worker"
                    ) from exc

    async def route_request_when_available(
        self,
        session_id: str | None = None,
        *,
        input_tokens: int | None = None,
        timeout: float | None = None,
    ) -> RouteSelection:
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        while True:
            try:
                return self.route_request(session_id, input_tokens=input_tokens)
            except NoHealthyWorkersError:
                self._healthy_workers_available.clear()
                if any(w.url not in self.dead_workers for w in self.workers):
                    self._healthy_workers_available.set()
                    continue
                if deadline is None:
                    await self._healthy_workers_available.wait()
                    continue
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TimeoutError(
                        "timed out waiting for a healthy inference worker"
                    ) from None
                try:
                    await asyncio.wait_for(
                        self._healthy_workers_available.wait(),
                        timeout=remaining,
                    )
                except TimeoutError as exc:
                    raise TimeoutError(
                        "timed out waiting for a healthy inference worker"
                    ) from exc

    def release(self, worker_url: str, session_id: str | None = None) -> None:
        self.active_counts[worker_url] = max(0, self.active_counts.get(worker_url, 0) - 1)
        if self.adaptive_policy is not None:
            self.adaptive_policy.release_request(session_id)
        self._observe_adaptive_load()

    def register_session(
        self,
        session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.adaptive_policy is not None:
            self.adaptive_policy.register_session(
                session_id,
                metadata,
                self.workers,
                self.dead_workers,
                self.active_counts,
            )

    def close_session(self, session_id: str) -> None:
        if self.adaptive_policy is not None:
            self.adaptive_policy.close_session(session_id)

    def routing_stats(self) -> dict[str, Any]:
        if self.adaptive_policy is None:
            workers = self.get_workers()
            active = [worker.active_requests for worker in workers] or [0]
            stats = {
                "mode": "sticky_least_loaded",
                "workers": [worker.model_dump() for worker in workers],
                "initial_bindings": 0,
                "adaptive_migrations": 0,
                "failover_rebindings": 0,
                "prefix_rebuild_tokens": 0,
                "pinned_sessions_max": 0,
                "pinned_sessions_min": 0,
                "pinned_context_tokens_max": 0,
                "pinned_context_tokens_min": 0,
                "active_requests_max": max(active),
                "active_requests_min": min(active),
            }
        else:
            stats = self.adaptive_policy.stats(
                self.workers,
                self.dead_workers,
                self.active_counts,
            )
        stats["maintenance"] = self.maintenance_status()
        return stats

    def _observe_adaptive_load(self) -> None:
        if self.adaptive_policy is not None:
            self.adaptive_policy.observe_load(
                self.workers,
                self.dead_workers,
                self.active_counts,
            )

    def _refresh_healthy_workers_event(self) -> None:
        if any(w.url not in self.dead_workers for w in self.workers):
            self._healthy_workers_available.set()
        else:
            self._healthy_workers_available.clear()

    def _adaptive_log_reason(
        self,
        stats: dict[str, Any],
        *,
        now: float,
    ) -> str | None:
        """Return why an adaptive summary is due, with noisy events coalesced."""

        active_gap = int(stats["active_requests_max"]) - int(
            stats["active_requests_min"]
        )
        imbalanced = (
            active_gap >= self.routing_config.migration_min_active_gap
        )
        if (
            self._last_adaptive_imbalance is not None
            and imbalanced != self._last_adaptive_imbalance
        ):
            self._pending_adaptive_log_reasons.add(
                "imbalance_started" if imbalanced else "imbalance_recovered"
            )
        self._last_adaptive_imbalance = imbalanced

        previous = self._last_adaptive_log_stats
        if previous is None or self._last_adaptive_log_at is None:
            reasons = ["initial"]
        else:
            if int(stats["adaptive_migrations"]) > int(
                previous["adaptive_migrations"]
            ):
                self._pending_adaptive_log_reasons.add("migration")
            failover = int(stats["failover_rebindings"]) > int(
                previous["failover_rebindings"]
            )
            elapsed = max(0.0, now - self._last_adaptive_log_at)
            if failover:
                self._pending_adaptive_log_reasons.add("failover")
                reasons = sorted(self._pending_adaptive_log_reasons)
            elif (
                self._pending_adaptive_log_reasons
                and elapsed >= _ADAPTIVE_EVENT_LOG_MIN_INTERVAL_SECONDS
            ):
                reasons = sorted(self._pending_adaptive_log_reasons)
            elif elapsed >= self.routing_config.metrics_interval_seconds:
                reasons = ["heartbeat"]
            else:
                return None

        self._last_adaptive_log_at = now
        self._last_adaptive_log_stats = dict(stats)
        self._pending_adaptive_log_reasons.clear()
        return "+".join(reasons)

    def _log_adaptive_summary(
        self,
        stats: dict[str, Any],
        *,
        reason: str,
    ) -> None:
        logger.info(
            "Adaptive routing fleet: reason=%s active=%d..%d pinned=%d..%d "
            "context_tokens=%d..%d migrations=%d failovers=%d "
            "prefix_rebuild_tokens=%d",
            reason,
            stats["active_requests_min"],
            stats["active_requests_max"],
            stats["pinned_sessions_min"],
            stats["pinned_sessions_max"],
            stats["pinned_context_tokens_min"],
            stats["pinned_context_tokens_max"],
            stats["adaptive_migrations"],
            stats["failover_rebindings"],
            stats["prefix_rebuild_tokens"],
        )

    # -- Health checks -----------------------------------------------------

    async def start_health_checks(self) -> None:
        if self._health_task is not None:
            return
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(self._health_timeout)
        )
        self._health_task = asyncio.create_task(self._health_loop())

    async def stop_health_checks(self) -> None:
        if self._health_task is not None:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            self._health_task = None
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def _health_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._health_interval)
                # Check healthy workers
                alive_urls = [w.url for w in self.workers if w.url not in self.dead_workers]
                # Also check dead workers so they can recover
                dead_urls = [w.url for w in self.workers if w.url in self.dead_workers]
                all_urls = alive_urls + dead_urls
                if not all_urls:
                    continue

                results = await asyncio.gather(*(self._check(u) for u in all_urls), return_exceptions=True)
                now = self._clock()
                suppress_failures = (
                    self._maintenance_active or now < self._health_grace_until
                )
                for url, ok in results:
                    if not ok:
                        if suppress_failures:
                            self.failure_counts[url] = 0
                            continue
                        count = self.failure_counts.get(url, 0) + 1
                        self.failure_counts[url] = count
                        if count >= self._failure_threshold:
                            if url not in self.dead_workers:
                                logger.warning(
                                    "Worker %s failed %d health checks — marking dead",
                                    url,
                                    count,
                                )
                            self.dead_workers.add(url)
                            self._refresh_healthy_workers_event()
                    else:
                        if url in self.dead_workers:
                            logger.info("Worker %s recovered — marking healthy", url)
                            self.dead_workers.discard(url)
                            self._refresh_healthy_workers_event()
                        self.failure_counts[url] = 0
                self._observe_adaptive_load()
                if self.adaptive_policy is not None:
                    stats = self.routing_stats()
                    reason = self._adaptive_log_reason(stats, now=now)
                    if reason is not None:
                        self._log_adaptive_summary(
                            stats,
                            reason=reason,
                        )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Health check loop error")
                await asyncio.sleep(5)

    async def _check(self, url: str) -> tuple[str, bool]:
        assert self._http is not None
        try:
            resp = await self._http.get(f"{url.rstrip('/')}/health")
            return url, resp.status_code == 200
        except Exception:
            return url, False
