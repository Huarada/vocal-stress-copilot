"""ASR-derived signals from the Voice Agent's `transcript.user` word-level data
(confidence, timestamps) — confirmed available in AssemblyAI's streaming docs
(2026-09-06): each word carries `confidence` (0-1) and `start`/`end` millisecond
timestamps, alongside the plain text this project was already using.

These are independent of the local acoustic pipeline (Praat DSP, the CNN) — they come
from AssemblyAI's own recognition model, not our own signal processing. Pure functions,
no I/O, so they're testable without audio or a live connection; `scripts/run_interview.py`
is the only caller, and it wires the results into `ProsodyFeatures` (which already had a
precedent for externally-populated fields — see `onset_latency_ms`'s original design).
"""
from __future__ import annotations

WordDict = dict  # {"text": str, "start": int, "end": int, "confidence": float, ...}


def mean_word_confidence(words: list[WordDict] | tuple[WordDict, ...]) -> float | None:
    """Average ASR confidence across a turn's recognized words — a secondary,
    ASR-derived hesitation/mumbling proxy, independent of the Praat-based prosody
    pipeline (different failure modes: this can be low even when Praat's pitch
    tracking succeeds, and vice versa). None if there are no words with a confidence
    value (e.g. a turn with no transcribed speech)."""
    confidences = [w.get("confidence") for w in words if w.get("confidence") is not None]
    if not confidences:
        return None
    return float(sum(confidences) / len(confidences))


def onset_latency_ms(turn_start_ms: int, words: list[WordDict] | tuple[WordDict, ...]) -> float | None:
    """Leading silence within a turn's segmented audio: the gap between the turn's
    recorded start (`SyncBus`'s `t_start_ms`, derived from the Voice Agent's
    `input.speech.started`) and the first recognized word's `start` timestamp.

    ASSUMPTION (untested against live data — flag this if it produces implausible
    numbers on a real session): this treats AssemblyAI's word `start` timestamp and
    `SyncBus`'s locally-derived `t_start_ms` as sharing a comparable origin (both
    anchored near stream start). This is narrower than ARCHITECTURE.md's original
    onset_latency_ms description — the full "agent finished asking -> user began
    answering" gap, which would need the agent's `reply.done` timestamp, not wired up
    in this pass — it measures silence at the START of this turn's own audio, not the
    full inter-turn gap.

    Returns None if there are no words, the first word has no `start` timestamp, or
    the computed value is negative (a sign the origin assumption doesn't hold for this
    turn — reporting an impossible negative latency would be worse than reporting
    nothing, consistent with how `f0_detected`/`arousal_score_reliable` handle their
    own "don't fabricate a number" cases).
    """
    if not words:
        return None
    first_start = words[0].get("start")
    if first_start is None:
        return None
    latency = float(first_start) - float(turn_start_ms)
    return latency if latency >= 0 else None
