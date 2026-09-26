"""Backend-agnostic, stateful dataloader yielding rllm task dicts.

Replaces the per-backend dataloaders (verl's StatefulDataLoader over an
RLHFDataset, tinker's plain torch DataLoader). Yields ``list[dict]`` batches of
rllm task dicts directly, shuffles deterministically per epoch, and checkpoints
its position so training can resume mid-epoch.

Fully-async training uses dispatch tickets.  Version-2 checkpoints store both
the cursor after the last dispatch and the exact set of dispatched batches that
were not yet committed to an optimizer step.  Those batches are replayed before
new data after resume, avoiding the permanent sample holes caused by saving only
the dispatched cursor.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
from collections import Counter, deque
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rllm.data import Dataset

logger = logging.getLogger(__name__)

STATE_SCHEMA_VERSION = 2
DYNAMIC_STATE_SCHEMA_VERSION = 3


@dataclass(frozen=True, order=True)
class DataloaderBatchTicket:
    """Stable position of one batch in the deterministic per-epoch order."""

    epoch: int
    start: int
    end: int

    def to_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DataloaderBatchTicket:
        if not isinstance(value, dict):
            raise ValueError(f"dataloader ticket must be a mapping, got {type(value).__name__}")
        fields = []
        for key in ("epoch", "start", "end"):
            item = value.get(key)
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError(f"dataloader ticket {key} must be an integer")
            fields.append(item)
        ticket = cls(*fields)
        if ticket.epoch < 0 or ticket.start < 0 or ticket.end <= ticket.start:
            raise ValueError(f"invalid dataloader ticket: {value}")
        return ticket


class StatefulTaskDataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self._dataset = dataset
        self._batch_size = int(batch_size)
        self._shuffle = bool(shuffle)
        self._seed = int(seed)
        self._drop_last = bool(drop_last)
        # State: next new sample to dispatch is ``_order(epoch)[cursor]``.
        self._epoch = 0
        self._cursor = 0
        # Restored pending tickets wait here until re-dispatched. Newly
        # dispatched tickets remain active until the trainer/filter resolves
        # them. Ordered dict semantics preserve original dispatch order.
        self._replay_tickets: deque[DataloaderBatchTicket] = deque()
        self._active_tickets: dict[DataloaderBatchTicket, None] = {}
        self._replayed_count = 0
        self._resolved_count = 0

    def __len__(self) -> int:
        """Batches per epoch."""
        n = len(self._dataset)
        return n // self._batch_size if self._drop_last else math.ceil(n / self._batch_size)

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def pending_tickets(self) -> tuple[DataloaderBatchTicket, ...]:
        return tuple(sorted((*self._replay_tickets, *self._active_tickets)))

    def _order(self, epoch: int) -> list[int]:
        indices = list(range(len(self._dataset)))
        if self._shuffle:
            random.Random(self._seed + epoch).shuffle(indices)
        return indices

    def _batch_for_ticket(self, ticket: DataloaderBatchTicket) -> list[dict[str, Any]]:
        n = len(self._dataset)
        if ticket.end > n:
            raise ValueError(
                f"dataloader ticket {ticket.to_dict()} exceeds dataset size {n}"
            )
        order = self._order(ticket.epoch)
        return [self._dataset[i] for i in order[ticket.start : ticket.end]]

    def __iter__(self) -> Iterator[list[dict[str, Any]]]:
        """Yield ordinary (non-ticketed) batches for one epoch.

        Fully-async callers must use :meth:`next_dispatch` so pending work can
        be checkpointed and replayed.  The conventional path remains unchanged
        for synchronous trainers and existing callers.
        """
        if self._replay_tickets or self._active_tickets:
            raise RuntimeError(
                "dataloader has pending fully-async dispatch tickets; "
                "resume must use next_dispatch()"
            )
        order = self._order(self._epoch)
        n = len(order)
        pos = self._cursor
        while pos < n:
            end = min(pos + self._batch_size, n)
            if end - pos < self._batch_size and self._drop_last:
                break
            batch = [self._dataset[i] for i in order[pos:end]]
            pos = end
            self._cursor = pos
            yield batch
        # Epoch exhausted: advance to the next epoch for the next pass.
        self._epoch += 1
        self._cursor = 0

    def next_dispatch(
        self,
        total_epochs: int,
    ) -> tuple[list[dict[str, Any]], DataloaderBatchTicket] | None:
        """Return and register the next fully-async batch.

        Restored pending batches are emitted first.  New cursor state advances
        only when this method actually returns a batch, allowing the caller to
        wait for dispatch quota before touching the dataloader.
        """
        if self._replay_tickets:
            ticket = self._replay_tickets.popleft()
            self._active_tickets[ticket] = None
            self._replayed_count += 1
            return self._batch_for_ticket(ticket), ticket

        n = len(self._dataset)
        while self._epoch < total_epochs:
            start = self._cursor
            if start >= n or (
                self._drop_last and start + self._batch_size > n
            ):
                self._epoch += 1
                self._cursor = 0
                continue
            end = min(start + self._batch_size, n)
            ticket = DataloaderBatchTicket(self._epoch, start, end)
            self._cursor = end
            self._active_tickets[ticket] = None
            return self._batch_for_ticket(ticket), ticket
        return None

    def mark_resolved(self, ticket: DataloaderBatchTicket | None) -> None:
        """Commit one dispatched batch after training or intentional filtering."""
        if ticket is None:
            return
        if ticket not in self._active_tickets:
            raise ValueError(f"dataloader ticket is not active: {ticket.to_dict()}")
        del self._active_tickets[ticket]
        self._resolved_count += 1

    def mark_resolved_many(
        self,
        tickets: list[DataloaderBatchTicket | None],
    ) -> None:
        for ticket in tickets:
            self.mark_resolved(ticket)

    def _dispatched_batch_count(self) -> int:
        batches_per_epoch = len(self)
        completed_epochs = self._epoch * batches_per_epoch
        current_batches = math.ceil(self._cursor / self._batch_size)
        return completed_epochs + min(batches_per_epoch, current_batches)

    def stats(self) -> dict[str, int]:
        pending = len(self._replay_tickets) + len(self._active_tickets)
        dispatched = self._dispatched_batch_count()
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "dispatched": dispatched,
            "resolved": max(0, dispatched - pending),
            "pending": pending,
            "replay_remaining": len(self._replay_tickets),
            "replayed": self._replayed_count,
            "resolved_session": self._resolved_count,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "epoch": self._epoch,
            "cursor": self._cursor,
            "seed": self._seed,
            "dataset_size": len(self._dataset),
            "batch_size": self._batch_size,
            "shuffle": self._shuffle,
            "drop_last": self._drop_last,
            "pending_dispatches": [
                ticket.to_dict() for ticket in self.pending_tickets
            ],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("loader_type") == "dynamic_sampling":
            raise ValueError(
                "cannot load a dynamic-sampling checkpoint into "
                "StatefulTaskDataLoader"
            )
        schema_version = state.get("schema_version", 1)
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise ValueError("dataloader checkpoint schema_version must be an integer")
        if schema_version != STATE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported dataloader checkpoint schema_version={schema_version}"
            )

        expected = {
            "dataset_size": len(self._dataset),
            "batch_size": self._batch_size,
            "shuffle": self._shuffle,
            "drop_last": self._drop_last,
            "seed": self._seed,
        }
        for key, current_value in expected.items():
            saved_value = state.get(key)
            if saved_value != current_value:
                raise ValueError(
                    f"dataloader checkpoint {key} mismatch: "
                    f"saved={saved_value!r}, current={current_value!r}"
                )

        epoch = state.get("epoch")
        cursor = state.get("cursor")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError(f"invalid dataloader epoch: {epoch!r}")
        if (
            isinstance(cursor, bool)
            or not isinstance(cursor, int)
            or cursor < 0
            or cursor > len(self._dataset)
        ):
            raise ValueError(f"invalid dataloader cursor: {cursor!r}")
        if cursor % self._batch_size != 0 and not (
            not self._drop_last and cursor == len(self._dataset)
        ):
            raise ValueError(
                "dataloader cursor is not aligned to the saved batch "
                f"configuration: {cursor}"
            )

        raw_pending = state.get("pending_dispatches", [])
        if not isinstance(raw_pending, list):
            raise ValueError("dataloader pending_dispatches must be a list")
        pending = [DataloaderBatchTicket.from_dict(item) for item in raw_pending]
        if len(set(pending)) != len(pending):
            raise ValueError("dataloader pending_dispatches contains duplicates")
        for ticket in pending:
            if ticket.end > len(self._dataset):
                raise ValueError(
                    f"pending dataloader ticket exceeds dataset size: {ticket.to_dict()}"
                )
            expected_end = min(
                ticket.start + self._batch_size,
                len(self._dataset),
            )
            if (
                ticket.start % self._batch_size != 0
                or ticket.end != expected_end
                or (
                    self._drop_last
                    and ticket.end - ticket.start < self._batch_size
                )
            ):
                raise ValueError(
                    "pending dataloader ticket is not aligned to the saved batch "
                    f"configuration: {ticket.to_dict()}"
                )
            if ticket.epoch > epoch or (
                ticket.epoch == epoch and ticket.end > cursor
            ):
                raise ValueError(
                    "pending dataloader ticket is beyond the dispatched cursor: "
                    f"{ticket.to_dict()} vs epoch={epoch}, cursor={cursor}"
                )

        self._epoch = epoch
        self._cursor = cursor
        self._replay_tickets = deque(sorted(pending))
        self._active_tickets.clear()
        self._replayed_count = 0
        self._resolved_count = 0

    def clone(self) -> StatefulTaskDataLoader:
        """A detached copy at the same position — including pending replay work."""
        twin = StatefulTaskDataLoader(
            self._dataset,
            self._batch_size,
            shuffle=self._shuffle,
            seed=self._seed,
            drop_last=self._drop_last,
        )
        twin.load_state_dict(self.state_dict())
        return twin


@dataclass(frozen=True)
class DynamicSamplingResolution:
    """Result of classifying one dynamically sampled task attempt."""

    status: str
    dropped_easy: bool = False
    dropped_hard: bool = False
    dropped_total: bool = False

    @property
    def requeued(self) -> bool:
        return self.status == "requeued"

    @property
    def dropped(self) -> bool:
        return self.status == "dropped"


@dataclass
class _DynamicEpochState:
    epoch: int
    available: list[int]
    deferred: list[int]
    uniform_counts: dict[int, int]
    easy_counts: dict[int, int]
    hard_counts: dict[int, int]
    total_filter_counts: dict[int, int]
    metrics: Counter[str]
    dispatch_exhausted: bool = False
    summary_emitted: bool = False


class DynamicSamplingTaskDataLoader(StatefulTaskDataLoader):
    """Stateful one-task dispatcher with outcome-driven random requeueing.

    Dataset positions, rather than task IDs, are the identity boundary. This
    allows duplicate task IDs and gives every task independent easy/hard
    counters in every epoch. Tickets accepted for training stay pending until
    the optimizer commits them, matching the ordinary fully-async loader's
    checkpoint durability contract.
    """

    _AWAITING = "awaiting_outcome"
    _ACCEPTED = "accepted"

    def __init__(
        self,
        dataset: Dataset,
        *,
        shuffle: bool = True,
        seed: int = 0,
        max_easy_rejections: int = 3,
        max_hard_rejections: int = 5,
        outcome_mode: str = "reward_uniform",
        easy_pass_rate_threshold: float = 0.9,
        hard_pass_rate_threshold: float = 0.1,
        target_batch_size: int = 1,
        defer_requeues_until_next_wave: bool = False,
        checkpoint_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            dataset,
            batch_size=1,
            shuffle=shuffle,
            seed=seed,
            drop_last=True,
        )
        for name, value in (
            ("max_easy_rejections", max_easy_rejections),
            ("max_hard_rejections", max_hard_rejections),
            ("target_batch_size", target_batch_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        self._max_easy_rejections = max_easy_rejections
        self._max_hard_rejections = max_hard_rejections
        if outcome_mode not in {"reward_uniform", "verifier_pass_count"}:
            raise ValueError(f"unsupported outcome_mode={outcome_mode!r}")
        if not 0.0 <= hard_pass_rate_threshold <= easy_pass_rate_threshold <= 1.0:
            raise ValueError("dynamic sampling pass-rate thresholds are invalid")
        self._outcome_mode = outcome_mode
        self._easy_pass_rate_threshold = float(easy_pass_rate_threshold)
        self._hard_pass_rate_threshold = float(hard_pass_rate_threshold)
        self._max_total_rejections = max(
            self._max_easy_rejections, self._max_hard_rejections
        )
        self._target_batch_size = target_batch_size
        self._defer_requeues_until_next_wave = bool(
            defer_requeues_until_next_wave
        )
        self._checkpoint_config = dict(checkpoint_config or {})
        # Keep pool admission independent from StatefulTaskDataLoader._order so
        # its exact random-without-replacement sequence can be checkpointed.
        self._pool_rng = random.Random(self._seed ^ 0xD1A6_5A6D)
        self._epochs: dict[int, _DynamicEpochState] = {}
        self._created_through_epoch = -1
        self._active_tickets: dict[DataloaderBatchTicket, str] = {}
        self._replay_tickets: deque[DataloaderBatchTicket] = deque()
        self._step_metrics: Counter[str] = Counter()
        self._completed_epoch_summaries: deque[dict[str, int]] = deque()
        self._sampling_wave_state: dict[str, Any] | None = None
        self._state_changed = asyncio.Event()
        self._create_epoch(0)

    def __iter__(self) -> Iterator[list[dict[str, Any]]]:
        raise RuntimeError(
            "DynamicSamplingTaskDataLoader requires feedback-aware next_dispatch()"
        )

    def _create_epoch(self, epoch: int) -> _DynamicEpochState:
        state = _DynamicEpochState(
            epoch=epoch,
            available=list(range(len(self._dataset))),
            deferred=[],
            uniform_counts={},
            easy_counts={},
            hard_counts={},
            total_filter_counts={},
            metrics=Counter(),
        )
        self._epochs[epoch] = state
        self._created_through_epoch = max(self._created_through_epoch, epoch)
        self._epoch = self._created_through_epoch
        self._state_changed.set()
        return state

    def _state_for_ticket(self, ticket: DataloaderBatchTicket) -> _DynamicEpochState:
        state = self._epochs.get(ticket.epoch)
        if state is None:
            raise ValueError(
                f"dynamic sampling ticket references unknown epoch: {ticket.to_dict()}"
            )
        if ticket.end != ticket.start + 1 or not 0 <= ticket.start < len(self._dataset):
            raise ValueError(f"invalid dynamic sampling ticket: {ticket.to_dict()}")
        return state

    def _dispatch_ticket(
        self,
        ticket: DataloaderBatchTicket,
    ) -> tuple[list[dict[str, Any]], DataloaderBatchTicket]:
        state = self._state_for_ticket(ticket)
        self._active_tickets[ticket] = self._AWAITING
        state.metrics["task_rollout_attempts"] += 1
        self._step_metrics["tasks_sampled"] += 1
        return self._batch_for_ticket(ticket), ticket

    def _awaiting_feedback(self) -> bool:
        return any(status == self._AWAITING for status in self._active_tickets.values())

    def _active_in_epoch(self, epoch: int) -> bool:
        return any(ticket.epoch == epoch for ticket in self._active_tickets)

    def _mark_dispatch_exhausted_epochs(self) -> None:
        for epoch, state in self._epochs.items():
            if state.available:
                continue
            if any(
                ticket.epoch == epoch and status == self._AWAITING
                for ticket, status in self._active_tickets.items()
            ):
                continue
            state.dispatch_exhausted = True
            self._maybe_finalize_epoch(epoch)

    def next_dispatch(
        self,
        total_epochs: int,
    ) -> tuple[list[dict[str, Any]], DataloaderBatchTicket] | None:
        if total_epochs < 0:
            raise ValueError(f"total_epochs must be non-negative, got {total_epochs}")
        if total_epochs == 0:
            return None

        if self._replay_tickets:
            ticket = self._replay_tickets.popleft()
            self._replayed_count += 1
            return self._dispatch_ticket(ticket)

        while True:
            for epoch in sorted(self._epochs):
                state = self._epochs[epoch]
                if state.available:
                    index = (
                        self._pool_rng.randrange(len(state.available))
                        if self._shuffle
                        else 0
                    )
                    position = state.available.pop(index)
                    return self._dispatch_ticket(
                        DataloaderBatchTicket(epoch, position, position + 1)
                    )

            # Preserve task diversity within a rejection pass. The admission
            # controller explicitly promotes deferred tasks either when the
            # current wave needs another pass or when a successor wave starts.
            if self._defer_requeues_until_next_wave and any(
                state.deferred for state in self._epochs.values()
            ):
                self._mark_dispatch_exhausted_epochs()
                return None

            self._mark_dispatch_exhausted_epochs()
            if self._awaiting_feedback():
                return None

            next_epoch = self._created_through_epoch + 1
            if next_epoch < total_epochs:
                self._create_epoch(next_epoch)
                continue
            return None

    def generation_exhausted(self, total_epochs: int) -> bool:
        if total_epochs <= 0:
            return True
        self._mark_dispatch_exhausted_epochs()
        if self._replay_tickets or any(
            state.available or state.deferred for state in self._epochs.values()
        ):
            return False
        if self._awaiting_feedback():
            return False
        return self._created_through_epoch + 1 >= total_epochs

    async def wait_for_feedback(self) -> None:
        """Wait until a completed rollout changes pool dispatchability."""
        self._state_changed.clear()
        if (
            self._replay_tickets
            or any(state.available for state in self._epochs.values())
            or not self._awaiting_feedback()
        ):
            return
        await self._state_changed.wait()

    def _pop_awaiting(
        self,
        ticket: DataloaderBatchTicket,
    ) -> _DynamicEpochState:
        status = self._active_tickets.get(ticket)
        if status != self._AWAITING:
            raise ValueError(
                "dynamic sampling ticket is not awaiting an outcome: "
                f"{ticket.to_dict()} status={status!r}"
            )
        return self._state_for_ticket(ticket)


    def record_filtered_outcome(
        self,
        ticket: DataloaderBatchTicket,
        *,
        uniform: bool,
        easy: bool,
        hard: bool,
        rollout_count: int,
    ) -> DynamicSamplingResolution:
        """Requeue or threshold-drop one outcome-filtered task attempt."""
        if not uniform and not easy and not hard:
            raise ValueError("filtered outcome has no classification reason")
        state = self._pop_awaiting(ticket)
        position = ticket.start
        del self._active_tickets[ticket]

        state.metrics["rollouts_generated"] += rollout_count
        state.total_filter_counts[position] = (
            state.total_filter_counts.get(position, 0) + 1
        )
        state.metrics["total_filtered_attempts"] += 1
        self._step_metrics["tasks_completed"] += 1
        self._step_metrics["tasks_filtered"] += 1
        self._step_metrics["total_filtered"] += 1
        if uniform:
            state.uniform_counts[position] = state.uniform_counts.get(position, 0) + 1
            state.metrics["uniform_filtered_attempts"] += 1
            self._step_metrics["uniform_filtered"] += 1
        if easy:
            state.easy_counts[position] = state.easy_counts.get(position, 0) + 1
            state.metrics["easy_filtered_attempts"] += 1
            self._step_metrics["easy_filtered"] += 1
        if hard:
            state.hard_counts[position] = state.hard_counts.get(position, 0) + 1
            state.metrics["hard_filtered_attempts"] += 1
            self._step_metrics["hard_filtered"] += 1

        dropped_easy = easy and state.easy_counts[position] >= self._max_easy_rejections
        dropped_hard = hard and state.hard_counts[position] >= self._max_hard_rejections
        dropped_total = (
            self._outcome_mode == "verifier_pass_count"
            and state.total_filter_counts[position] >= self._max_total_rejections
        )
        if dropped_easy or dropped_hard or dropped_total:
            state.metrics["tasks_dropped"] += 1
            self._step_metrics["tasks_dropped"] += 1
            if dropped_easy:
                state.metrics["tasks_dropped_easy"] += 1
                self._step_metrics["tasks_dropped_easy"] += 1
            if dropped_hard:
                state.metrics["tasks_dropped_hard"] += 1
                self._step_metrics["tasks_dropped_hard"] += 1
            if dropped_total:
                state.metrics["tasks_dropped_total"] += 1
                self._step_metrics["tasks_dropped_total"] += 1
            if uniform:
                state.metrics["tasks_dropped_uniform"] += 1
                self._step_metrics["tasks_dropped_uniform"] += 1
            resolution = DynamicSamplingResolution(
                status="dropped",
                dropped_easy=dropped_easy,
                dropped_hard=dropped_hard,
                dropped_total=dropped_total,
            )
        else:
            target_pool = (
                state.deferred
                if self._defer_requeues_until_next_wave
                else state.available
            )
            target_pool.append(position)
            state.metrics["tasks_requeued"] += 1
            self._step_metrics["tasks_requeued"] += 1
            resolution = DynamicSamplingResolution(status="requeued")

        self._state_changed.set()
        self._maybe_finalize_epoch(ticket.epoch)
        return resolution

    def return_unconsumed(
        self,
        ticket: DataloaderBatchTicket | None,
        *,
        rollout_count: int = 0,
        defer_until_next_wave: bool = True,
    ) -> None:
        """Return a speculative task without classifying or consuming it.

        This path is used for wave-close cancellation and the completion race
        after the accepted target has already been reserved. It deliberately
        does not affect easy/hard, filtered, accepted, consumed, or dropped
        counters. Infrastructure recovery sets ``defer_until_next_wave=False``
        so the interrupted candidate can refill the current wave.
        """
        if ticket is None:
            return
        state = self._pop_awaiting(ticket)
        del self._active_tickets[ticket]
        target_pool = (
            state.deferred if defer_until_next_wave else state.available
        )
        target_pool.append(ticket.start)
        if not defer_until_next_wave:
            state.dispatch_exhausted = False
        state.metrics["rollouts_generated"] += max(0, int(rollout_count))
        state.metrics["speculative_returns"] += 1
        self._step_metrics["tasks_unconsumed"] += 1
        self._state_changed.set()

    def advance_sampling_wave(self) -> int:
        """Make prior-wave returns eligible and return the promoted count."""
        promoted = 0
        for state in self._epochs.values():
            if not state.deferred:
                continue
            promoted += len(state.deferred)
            state.available.extend(state.deferred)
            state.deferred.clear()
            state.dispatch_exhausted = False
        if promoted:
            self._state_changed.set()
        return promoted

    def finalize_deferred_tail(self) -> int:
        """Consume tasks that cannot be retried without repeating this wave."""
        finalized = 0
        for state in self._epochs.values():
            count = len(state.deferred)
            if not count:
                continue
            state.deferred.clear()
            state.metrics["tasks_untrained_tail"] += count
            state.metrics["tasks_wave_exhausted_tail"] += count
            self._resolved_count += count
            finalized += count
            self._maybe_finalize_epoch(state.epoch)
        if finalized:
            self._step_metrics["tasks_wave_exhausted_tail"] += finalized
            self._state_changed.set()
        return finalized

    def sampling_pool_sizes(self) -> tuple[int, int]:
        return (
            sum(len(state.available) for state in self._epochs.values()),
            sum(len(state.deferred) for state in self._epochs.values()),
        )

    def set_sampling_wave_state(self, state: dict[str, Any]) -> None:
        # The controller owns validation; the loader merely makes the state
        # part of the same atomic trainer checkpoint as pool/ticket state.
        self._sampling_wave_state = dict(state)

    def sampling_wave_state(self) -> dict[str, Any] | None:
        return (
            dict(self._sampling_wave_state)
            if self._sampling_wave_state is not None
            else None
        )

    def mark_accepted(
        self,
        ticket: DataloaderBatchTicket,
        *,
        rollout_count: int,
    ) -> None:
        state = self._pop_awaiting(ticket)
        self._active_tickets[ticket] = self._ACCEPTED
        state.metrics["rollouts_generated"] += rollout_count
        state.metrics["tasks_accepted"] += 1
        self._step_metrics["tasks_completed"] += 1
        self._step_metrics["tasks_accepted"] += 1
        self._state_changed.set()

    def mark_consumed_other(
        self,
        ticket: DataloaderBatchTicket | None,
        *,
        rollout_count: int,
    ) -> None:
        if ticket is None:
            return
        state = self._pop_awaiting(ticket)
        del self._active_tickets[ticket]
        state.metrics["rollouts_generated"] += rollout_count
        state.metrics["tasks_consumed_other"] += 1
        self._step_metrics["tasks_completed"] += 1
        self._resolved_count += 1
        self._state_changed.set()
        self._maybe_finalize_epoch(ticket.epoch)

    def mark_resolved(self, ticket: DataloaderBatchTicket | None) -> None:
        if ticket is None:
            return
        status = self._active_tickets.get(ticket)
        if status == self._AWAITING:
            self.mark_consumed_other(ticket, rollout_count=0)
            return
        if status != self._ACCEPTED:
            raise ValueError(
                f"dynamic sampling ticket is not active: {ticket.to_dict()}"
            )
        state = self._state_for_ticket(ticket)
        del self._active_tickets[ticket]
        state.metrics["tasks_committed"] += 1
        self._resolved_count += 1
        self._state_changed.set()
        self._maybe_finalize_epoch(ticket.epoch)

    def mark_untrained_tail_many(
        self,
        tickets: list[DataloaderBatchTicket | None],
    ) -> None:
        for ticket in tickets:
            if ticket is None:
                continue
            if self._active_tickets.get(ticket) != self._ACCEPTED:
                raise ValueError(
                    "untrained-tail ticket was not accepted: "
                    f"{ticket.to_dict()}"
                )
            state = self._state_for_ticket(ticket)
            del self._active_tickets[ticket]
            state.metrics["tasks_untrained_tail"] += 1
            self._resolved_count += 1
            self._maybe_finalize_epoch(ticket.epoch)
        self._state_changed.set()

    def _maybe_finalize_epoch(self, epoch: int) -> None:
        state = self._epochs[epoch]
        if (
            state.summary_emitted
            or not state.dispatch_exhausted
            or state.available
            or state.deferred
            or self._active_in_epoch(epoch)
            or any(ticket.epoch == epoch for ticket in self._replay_tickets)
        ):
            return
        summary = {
            "index": epoch,
            "tasks_total": len(self._dataset),
            "task_rollout_attempts": state.metrics["task_rollout_attempts"],
            "rollouts_generated": state.metrics["rollouts_generated"],
            "tasks_accepted": state.metrics["tasks_accepted"],
            "tasks_consumed_other": state.metrics["tasks_consumed_other"],
            "uniform_filtered_attempts": state.metrics[
                "uniform_filtered_attempts"
            ],
            "total_filtered_attempts": state.metrics["total_filtered_attempts"],
            "easy_filtered_attempts": state.metrics["easy_filtered_attempts"],
            "hard_filtered_attempts": state.metrics["hard_filtered_attempts"],
            "tasks_dropped": state.metrics["tasks_dropped"],
            "tasks_dropped_easy": state.metrics["tasks_dropped_easy"],
            "tasks_dropped_hard": state.metrics["tasks_dropped_hard"],
            "tasks_dropped_uniform": state.metrics["tasks_dropped_uniform"],
            "tasks_dropped_total": state.metrics["tasks_dropped_total"],
            "tasks_untrained_tail": state.metrics["tasks_untrained_tail"],
        }
        state.summary_emitted = True
        self._completed_epoch_summaries.append(summary)

    def drain_step_metrics(self) -> dict[str, int]:
        names = (
            "tasks_sampled",
            "tasks_completed",
            "tasks_accepted",
            "tasks_filtered",
            "tasks_requeued",
            "tasks_dropped",
            "tasks_dropped_uniform",
            "tasks_dropped_easy",
            "tasks_dropped_hard",
            "tasks_dropped_total",
            "uniform_filtered",
            "easy_filtered",
            "hard_filtered",
            "total_filtered",
            "tasks_unconsumed",
            "tasks_wave_exhausted_tail",
        )
        metrics = {
            f"dynamic_sampling/step/{name}": int(self._step_metrics[name])
            for name in names
        }
        self._step_metrics.clear()
        metrics["dynamic_sampling/step/pool_remaining"] = sum(
            len(state.available) + len(state.deferred)
            for state in self._epochs.values()
        )
        eligible, deferred = self.sampling_pool_sizes()
        metrics["dynamic_sampling/step/pool_eligible"] = eligible
        metrics["dynamic_sampling/step/pool_deferred"] = deferred
        metrics["dynamic_sampling/step/in_flight"] = len(
            self._active_tickets
        ) + len(self._replay_tickets)
        return metrics

    def drain_epoch_summaries(self) -> list[dict[str, int]]:
        summaries = list(self._completed_epoch_summaries)
        self._completed_epoch_summaries.clear()
        return summaries

    def stats(self) -> dict[str, int]:
        resolved = sum(
            state.metrics["tasks_committed"]
            + state.metrics["tasks_consumed_other"]
            + state.metrics["tasks_dropped"]
            + state.metrics["tasks_untrained_tail"]
            for state in self._epochs.values()
        )
        return {
            "schema_version": DYNAMIC_STATE_SCHEMA_VERSION,
            "dispatched": sum(
                state.metrics["task_rollout_attempts"]
                for state in self._epochs.values()
            ),
            "resolved": resolved,
            "pending": len(self._active_tickets) + len(self._replay_tickets),
            "replay_remaining": len(self._replay_tickets),
            "replayed": self._replayed_count,
            "resolved_session": self._resolved_count,
            "pool_remaining": sum(
                len(state.available) + len(state.deferred)
                for state in self._epochs.values()
            ),
            "pool_eligible": self.sampling_pool_sizes()[0],
            "pool_deferred": self.sampling_pool_sizes()[1],
            "tasks_dropped": sum(
                state.metrics["tasks_dropped"] for state in self._epochs.values()
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        active_dispatches = [
            {"ticket": ticket.to_dict(), "status": status}
            for ticket, status in self._active_tickets.items()
        ]
        active_dispatches.extend(
            {"ticket": ticket.to_dict(), "status": self._AWAITING}
            for ticket in self._replay_tickets
        )
        return {
            "loader_type": "dynamic_sampling",
            "schema_version": DYNAMIC_STATE_SCHEMA_VERSION,
            "epoch": self._epoch,
            "cursor": 0,
            "seed": self._seed,
            "dataset_size": len(self._dataset),
            "batch_size": 1,
            "shuffle": self._shuffle,
            "drop_last": True,
            "max_easy_rejections": self._max_easy_rejections,
            "max_hard_rejections": self._max_hard_rejections,
            "max_total_rejections": self._max_total_rejections,
            "outcome_mode": self._outcome_mode,
            "easy_pass_rate_threshold": self._easy_pass_rate_threshold,
            "hard_pass_rate_threshold": self._hard_pass_rate_threshold,
            "target_batch_size": self._target_batch_size,
            "defer_requeues_until_next_wave": (
                self._defer_requeues_until_next_wave
            ),
            "checkpoint_config": self._checkpoint_config,
            "pool_rng_state": self._pool_rng.getstate(),
            "created_through_epoch": self._created_through_epoch,
            "epochs": [
                {
                    "epoch": epoch,
                    "available": list(state.available),
                    "deferred": list(state.deferred),
                    "uniform_counts": dict(state.uniform_counts),
                    "easy_counts": dict(state.easy_counts),
                    "hard_counts": dict(state.hard_counts),
                    "total_filter_counts": dict(state.total_filter_counts),
                    "metrics": dict(state.metrics),
                    "dispatch_exhausted": state.dispatch_exhausted,
                    "summary_emitted": state.summary_emitted,
                }
                for epoch, state in sorted(self._epochs.items())
            ],
            "pending_dispatches": [
                item["ticket"] for item in active_dispatches
            ],
            "active_dispatches": active_dispatches,
            "step_metrics": dict(self._step_metrics),
            "completed_epoch_summaries": list(
                self._completed_epoch_summaries
            ),
            "sampling_wave_state": self._sampling_wave_state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("loader_type") != "dynamic_sampling":
            raise ValueError(
                "cannot load a non-dynamic dataloader checkpoint into "
                "DynamicSamplingTaskDataLoader"
            )
        saved_schema = state.get("schema_version")
        if saved_schema != DYNAMIC_STATE_SCHEMA_VERSION:
            raise ValueError(
                "unsupported dynamic dataloader checkpoint schema_version="
                f"{state.get('schema_version')!r}"
            )
        expected_checkpoint_config = self._checkpoint_config
        expected = {
            "dataset_size": len(self._dataset),
            "batch_size": 1,
            "shuffle": self._shuffle,
            "drop_last": True,
            "seed": self._seed,
            "max_easy_rejections": self._max_easy_rejections,
            "max_hard_rejections": self._max_hard_rejections,
            "target_batch_size": self._target_batch_size,
            "defer_requeues_until_next_wave": (
                self._defer_requeues_until_next_wave
            ),
            "checkpoint_config": expected_checkpoint_config,
        }
        if saved_schema == DYNAMIC_STATE_SCHEMA_VERSION:
            expected.update(
                {
                    "max_total_rejections": self._max_total_rejections,
                    "outcome_mode": self._outcome_mode,
                    "easy_pass_rate_threshold": self._easy_pass_rate_threshold,
                    "hard_pass_rate_threshold": self._hard_pass_rate_threshold,
                }
            )
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(
                    f"dynamic dataloader checkpoint {key} mismatch: "
                    f"saved={state.get(key)!r}, current={value!r}"
                )

        epochs: dict[int, _DynamicEpochState] = {}
        for raw in state.get("epochs", []):
            epoch = int(raw["epoch"])
            available = [int(value) for value in raw.get("available", [])]
            deferred = [int(value) for value in raw.get("deferred", [])]
            if any(
                not 0 <= value < len(self._dataset)
                for value in available + deferred
            ):
                raise ValueError(f"invalid dynamic pool position in epoch {epoch}")
            epochs[epoch] = _DynamicEpochState(
                epoch=epoch,
                available=available,
                deferred=deferred,
                uniform_counts={
                    int(k): int(v)
                    for k, v in raw.get("uniform_counts", {}).items()
                },
                easy_counts={int(k): int(v) for k, v in raw.get("easy_counts", {}).items()},
                hard_counts={int(k): int(v) for k, v in raw.get("hard_counts", {}).items()},
                total_filter_counts={
                    int(k): int(v)
                    for k, v in raw.get("total_filter_counts", {}).items()
                },
                metrics=Counter(
                    {str(k): int(v) for k, v in raw.get("metrics", {}).items()}
                ),
                dispatch_exhausted=bool(raw.get("dispatch_exhausted", False)),
                summary_emitted=bool(raw.get("summary_emitted", False)),
            )
        if not epochs:
            raise ValueError("dynamic dataloader checkpoint has no epoch states")

        replay: list[DataloaderBatchTicket] = []
        seen: set[DataloaderBatchTicket] = set()
        for item in state.get("active_dispatches", []):
            ticket = DataloaderBatchTicket.from_dict(item["ticket"])
            if ticket in seen:
                raise ValueError("dynamic active_dispatches contains duplicates")
            seen.add(ticket)
            if ticket.epoch not in epochs:
                raise ValueError(
                    f"dynamic pending ticket has no epoch state: {ticket.to_dict()}"
                )
            status = item.get("status")
            if status not in {self._AWAITING, self._ACCEPTED}:
                raise ValueError(f"invalid dynamic ticket status: {status!r}")
            if status == self._ACCEPTED:
                epochs[ticket.epoch].metrics["tasks_accepted"] -= 1
            epochs[ticket.epoch].dispatch_exhausted = False
            epochs[ticket.epoch].summary_emitted = False
            replay.append(ticket)

        self._epochs = epochs
        self._created_through_epoch = int(state["created_through_epoch"])
        if self._created_through_epoch != max(epochs):
            raise ValueError("dynamic checkpoint created_through_epoch mismatch")
        self._epoch = self._created_through_epoch
        self._cursor = 0
        self._active_tickets.clear()
        self._replay_tickets = deque(replay)
        try:
            self._pool_rng.setstate(state["pool_rng_state"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid dynamic pool RNG checkpoint state") from exc
        self._step_metrics = Counter(
            {str(k): int(v) for k, v in state.get("step_metrics", {}).items()}
        )
        self._completed_epoch_summaries = deque(
            dict(summary)
            for summary in state.get("completed_epoch_summaries", [])
        )
        raw_wave_state = state.get("sampling_wave_state")
        if raw_wave_state is not None and not isinstance(raw_wave_state, dict):
            raise ValueError("invalid dynamic sampling wave checkpoint state")
        self._sampling_wave_state = (
            dict(raw_wave_state) if raw_wave_state is not None else None
        )
        self._replayed_count = 0
        self._resolved_count = 0
        self._state_changed.set()

    def preview_batches(
        self,
        *,
        total_epochs: int,
        remaining_batches: int = -1,
    ) -> list[list[dict[str, Any]]]:
        """Return only the currently predictable first-pass dispatch order."""
        tickets = list(self._replay_tickets)
        preview_rng = random.Random()
        preview_rng.setstate(self._pool_rng.getstate())
        for epoch, state in sorted(self._epochs.items()):
            available = list(state.available)
            while available:
                index = preview_rng.randrange(len(available)) if self._shuffle else 0
                position = available.pop(index)
                tickets.append(DataloaderBatchTicket(epoch, position, position + 1))
        for epoch in range(self._created_through_epoch + 1, total_epochs):
            available = list(range(len(self._dataset)))
            while available:
                index = preview_rng.randrange(len(available)) if self._shuffle else 0
                position = available.pop(index)
                tickets.append(DataloaderBatchTicket(epoch, position, position + 1))
        if remaining_batches > 0:
            tickets = tickets[:remaining_batches]
        return [self._batch_for_ticket(ticket) for ticket in tickets]

    def clone(self) -> DynamicSamplingTaskDataLoader:
        twin = DynamicSamplingTaskDataLoader(
            self._dataset,
            shuffle=self._shuffle,
            seed=self._seed,
            max_easy_rejections=self._max_easy_rejections,
            max_hard_rejections=self._max_hard_rejections,
            outcome_mode=self._outcome_mode,
            easy_pass_rate_threshold=self._easy_pass_rate_threshold,
            hard_pass_rate_threshold=self._hard_pass_rate_threshold,
            target_batch_size=self._target_batch_size,
            defer_requeues_until_next_wave=(
                self._defer_requeues_until_next_wave
            ),
            checkpoint_config=self._checkpoint_config,
        )
        twin.load_state_dict(self.state_dict())
        return twin
