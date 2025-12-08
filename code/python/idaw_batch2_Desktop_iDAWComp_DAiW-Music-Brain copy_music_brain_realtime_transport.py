"""
Transport abstractions for emitting scheduled events.

The realtime engine can host multiple transports at once (e.g., send MIDI to a
virtual port and simultaneously broadcast JSON over WebSocket). Each transport
only needs to implement `emit` and `close`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional

from music_brain.realtime.events import ScheduledEvent


class BaseTransport(ABC):
    """Abstract base for realtime transports."""

    @abstractmethod
    def emit(self, events: Iterable[ScheduledEvent]) -> None:
        """Send scheduled events to the underlying output."""

    def emit_metrics(self, *_args, **_kwargs) -> None:  # pragma: no cover - optional
        """Hook for subclasses that want telemetry (no-op by default)."""

    @abstractmethod
    def close(self) -> None:
        """Release resources (MIDI ports, sockets, etc.)."""


class MidiTransport(BaseTransport):
    """
    Simple MIDI transport powered by `mido.open_output`.

    Any import errors are deferred until instantiation so the realtime module
    can be imported on systems without mido (e.g., Cloud CI).
    """

    def __init__(self, port_name: Optional[str] = None) -> None:
        try:
            import mido
        except ImportError as exc:  # pragma: no cover - fallback path
            raise RuntimeError("MidiTransport requires the 'mido' package") from exc

        self._mido = mido
        self._port = mido.open_output(port_name) if port_name else mido.open_output()

    def emit(self, events: Iterable[ScheduledEvent]) -> None:
        for scheduled in events:
            note = scheduled.note_event
            self._port.send(
                self._mido.Message(
                    "note_on",
                    note=note.pitch,
                    velocity=note.velocity,
                    channel=scheduled.channel,
                    time=0,
                )
            )
            self._port.send(
                self._mido.Message(
                    "note_off",
                    note=note.pitch,
                    velocity=0,
                    channel=scheduled.channel,
                    time=note.duration_ticks,
                )
            )

    def close(self) -> None:
        self._port.close()

