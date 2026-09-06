"""Turn segmentation use-case (ARCHITECTURE.md §2, §3): the single place raw audio
samples get sliced into per-turn chunks, keyed by the Voice Agent's
`input.speech.started` / `input.speech.stopped` events.

Centralizing this here — rather than letting each consumer guess turn boundaries from
whatever event happens to fire — is what ARCHITECTURE.md calls the single synchronization
authority. Everything downstream (EvidenceService, the dashboard) consumes its output and
never re-derives turn boundaries itself.

Pure Python, no audio I/O, no websocket — testable with synthetic numpy arrays and
simulated events, independent of any real microphone or live connection.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class SyncBusError(RuntimeError):
    pass


@dataclass
class PendingTurn:
    turn_id: str
    samples: np.ndarray
    t_start_ms: int
    t_end_ms: int


@dataclass
class SyncBus:
    sample_rate_hz: int
    # Guards against spurious VAD blips (mic click / room noise triggering
    # input.speech.started+stopped within ~100ms — observed for real on 2026-09-05,
    # first live run of scripts/run_interview.py: a 100ms "turn" with near-constant
    # pitch and an empty transcript reached the acoustic pipeline and got logged as
    # real evidence). Default 0 preserves the original no-filtering behavior for
    # existing callers/tests; live composition roots should pass a real floor.
    min_turn_duration_ms: int = 0
    # BUFFER STRATEGY REWRITTEN 2026-09-06 after a real live session: the original
    # implementation stored every fed chunk in a growing list and re-concatenated ALL
    # of them (the entire session's audio, from t=0) on every single `_slice()` call —
    # O(session_length) work on every turn, O(n^2) over a session of n turns. On a
    # ~14-turn live run this compounded into growing per-turn latency that stalled the
    # asyncio event loop (see ADR-028), and the resulting backlog caused the LOCAL
    # SyncBus's sample bookkeeping to drift out of sync with the audio AssemblyAI's
    # server was actually processing in real time: turns 10-14 all showed degenerate
    # prosody (f0_std≈0, HNR collapsed) and near-zero arousal despite normal duration
    # and CORRECT transcripts — the transcript comes from AssemblyAI's own real-time
    # pipeline on the same stream, unaffected by our local backlog, while the locally
    # sliced audio fed to the CNN/Praat was no longer the same content.
    #
    # Fixed with a pre-allocated, amortized-doubling buffer (the same growth strategy
    # CPython uses for `list` — O(1) amortized append, O(k) slice for a k-sample
    # request, never O(total audio fed so far) for either).
    _buffer: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    _write_pos: int = 0  # samples actually written; may be < len(_buffer) (spare capacity)
    _turn_start_sample: int | None = None
    _turn_counter: int = 0
    _turns_discarded_too_short: int = 0

    def feed_audio(self, chunk: np.ndarray) -> None:
        """Call for every raw audio chunk from the mic fan-out (ARCHITECTURE.md §2's
        Ingest component), regardless of whether a turn is currently open — samples
        outside an open turn are kept (cheaply) so slicing math stays correct, and
        dropped lazily once no longer reachable by any future turn boundary."""
        chunk = np.asarray(chunk, dtype=np.float32)
        needed = self._write_pos + len(chunk)
        if needed > len(self._buffer):
            new_capacity = max(needed, max(1, len(self._buffer)) * 2)
            grown = np.zeros(new_capacity, dtype=np.float32)
            grown[: self._write_pos] = self._buffer[: self._write_pos]
            self._buffer = grown
        self._buffer[self._write_pos : self._write_pos + len(chunk)] = chunk
        self._write_pos += len(chunk)

    def on_speech_started(self) -> None:
        if self._turn_start_sample is not None:
            raise SyncBusError(
                "on_speech_started called while a turn was already open — the Voice "
                "Agent sent two input.speech.started events without a stopped between "
                "them, or the caller is misusing this class."
            )
        self._turn_start_sample = self._write_pos

    def on_speech_stopped(self) -> PendingTurn | None:
        """Returns None (instead of a PendingTurn) when the segment is shorter than
        `min_turn_duration_ms` — a likely VAD false-positive rather than real speech.
        Discarded turns still consume a turn_id's worth of the counter's underlying
        sample bookkeeping but do NOT increment `_turn_counter`, so real turns keep
        clean sequential ids (t0001, t0002, ...) unpolluted by blips.
        """
        if self._turn_start_sample is None:
            raise SyncBusError(
                "on_speech_stopped called with no open turn — check that "
                "on_speech_started was called first for every input.speech.started event."
            )
        start_sample = self._turn_start_sample
        end_sample = self._write_pos
        self._turn_start_sample = None

        to_ms = 1000.0 / self.sample_rate_hz
        duration_ms = (end_sample - start_sample) * to_ms
        if duration_ms < self.min_turn_duration_ms:
            self._turns_discarded_too_short += 1
            return None

        samples = self._slice(start_sample, end_sample)
        self._turn_counter += 1
        turn_id = f"t{self._turn_counter:04d}"
        return PendingTurn(
            turn_id=turn_id,
            samples=samples,
            t_start_ms=int(start_sample * to_ms),
            t_end_ms=int(end_sample * to_ms),
        )

    def _slice(self, start_sample: int, end_sample: int) -> np.ndarray:
        return self._buffer[start_sample:end_sample].copy()

    @property
    def has_open_turn(self) -> bool:
        return self._turn_start_sample is not None

    @property
    def turns_discarded_too_short(self) -> int:
        return self._turns_discarded_too_short
