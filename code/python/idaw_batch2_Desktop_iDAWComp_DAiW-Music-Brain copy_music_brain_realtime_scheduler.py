"""
Deterministic event scheduler for the realtime engine.

Uses a min-heap keyed by `ScheduledEvent.start_tick` so we can efficiently
fetch all events that should be emitted within a lookahead window.
"""

from __future__ import annotations

import heapq
from typing import Iterable, List, Optional

from music_brain.realtime.events import ScheduledEvent


class EventScheduler:
    """Priority queue wrapper for scheduled note events."""

    def __init__(self) -> None:
        self._queue: List[ScheduledEvent] = []

    def clear(self) -> None:
        self._queue.clear()

    def __len__(self) -> int:
        return len(self._queue)

    # ------------------------------------------------------------------ #
    # Scheduling
    # ------------------------------------------------------------------ #
    def schedule(self, event: ScheduledEvent) -> None:
        """Add a single event to the queue."""

        heapq.heappush(self._queue, event)

    def schedule_many(self, events: Iterable[ScheduledEvent]) -> None:
        """Bulk insert events (more efficient than repeated pushes)."""

        for event in events:
            heapq.heappush(self._queue, event)

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def peek_next_tick(self) -> Optional[int]:
        """Return the tick of the next event without removing it."""

        return self._queue[0].start_tick if self._queue else None

    def pop_due_events(self, current_tick: int, lookahead_ticks: int) -> List[ScheduledEvent]:
        """
        Return all events scheduled to start within the lookahead window.

        Args:
            current_tick: Current transport position in ticks.
            lookahead_ticks: Events with start_tick <= current_tick + lookahead
                are considered due and will be removed from the queue.
        """

        due: List[ScheduledEvent] = []
        window = current_tick + lookahead_ticks

        while self._queue and self._queue[0].start_tick <= window:
            due.append(heapq.heappop(self._queue))

        return due

