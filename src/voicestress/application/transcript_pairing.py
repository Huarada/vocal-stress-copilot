"""Pairs finalized turn audio with its transcript, accounting for STT processing lag.

Real finding, 2026-09-05 (ARCHITECTURE.md §7b): on a live run, turn `t0007` had audio
duration 300ms (`t_start_ms: 36300` → `t_end_ms: 36600`) but was logged with the
transcript `"And I'm telling the truth right now, I think."` — an eight-word sentence
that cannot fit in 300ms. `transcript.user` (final) for a turn can arrive AFTER
`input.speech.stopped` for that same turn — sometimes not until the NEXT turn's audio
has already finished — so grabbing "whatever transcript text we've seen most recently"
at `speech.stopped` time silently attaches the wrong turn's words. This queue fixes that
by pairing FIFO instead: transcripts for a linear conversation still arrive in the same
order as their turns, just not necessarily before that turn's `speech.stopped`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

T = TypeVar("T")
P = TypeVar("P")


@dataclass
class TranscriptPairer(Generic[T, P]):
    """`T` is whatever the caller wants paired with a transcript — in practice
    `SyncBus`'s `PendingTurn`. `P` is the transcript payload type — plain `str` by
    default, generalized 2026-09-06 so a caller can pair a richer payload (text +
    word-level confidence/timestamps — see `application/asr_signals.py`) through the
    exact same FIFO logic instead of duplicating it. Both generic, so this stays
    testable without depending on `SyncBus` or any AssemblyAI event shape.
    """

    # Self-healing ceiling: if a turn's transcript never arrives (e.g. wordless audio —
    # the empty-transcript t0001 seen on every live run so far is plausibly this, not a
    # bug), the queue must not stall pairing for every turn after it forever.
    max_pending: int = 2
    # Value used to pair with a turn whose transcript never arrived (queue overflow or
    # session-end flush). Defaults to "" for the original str-payload behavior —
    # existing callers/tests are unaffected; pass e.g. `TranscriptPayload` for a richer
    # payload type.
    default_payload_factory: Callable[[], P] = field(default=lambda: "")  # type: ignore[assignment]
    _queue: list[T] = field(default_factory=list)

    def submit_turn(self, turn: T) -> tuple[T, P] | None:
        """Call when a turn's audio is finalized (after SyncBus.on_speech_stopped()
        returns non-None). Returns `(turn, default_payload)` immediately if this
        submission pushed the queue past `max_pending` — an older turn's transcript
        never showed up and must be flushed rather than blocking every later pairing.
        Otherwise queues the turn and returns None (finalization happens later, from
        `submit_transcript`)."""
        self._queue.append(turn)
        if len(self._queue) > self.max_pending:
            oldest = self._queue.pop(0)
            return (oldest, self.default_payload_factory())
        return None

    def submit_transcript(self, payload: P) -> tuple[T, P] | None:
        """Call when a transcript.user (final) event arrives. Pairs it with the oldest
        still-queued turn (FIFO — see module docstring for why this is correct even
        though transcripts can lag speech.stopped). Returns None if nothing is queued
        (a stray/duplicate transcript event with no turn waiting, e.g. from the agent's
        own speech being misrouted, or a second transcript.user for an already-paired
        turn)."""
        if not self._queue:
            return None
        turn = self._queue.pop(0)
        return (turn, payload)

    def flush_all(self) -> list[tuple[T, P]]:
        """Call at session end: any turns still queued never got a transcript and
        should be finalized with the default (empty) payload rather than silently
        dropped."""
        flushed = [(turn, self.default_payload_factory()) for turn in self._queue]
        self._queue.clear()
        return flushed

    @property
    def pending_count(self) -> int:
        return len(self._queue)
