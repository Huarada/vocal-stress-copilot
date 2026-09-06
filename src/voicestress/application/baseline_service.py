"""Baseline calibration use-case (ARCHITECTURE.md §7): accumulate prosodic features from
a session's neutral opening turns, then produce a `BaselineProfile` other services use to
convert later turns' raw prosody into z-deviations.

This is the piece that keeps the whole system honest: without it, every claim is "this
person sounds tense" (confounded by voice, accent, mic, room); with it, the claim narrows
to "this answer deviated from how this same person sounded five minutes ago."
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from voicestress.domain.value_objects import BaselineProfile, ProsodyFeatures, SpeakerId

_TRACKED_FIELDS = (
    "f0_mean_hz",
    "f0_std_hz",
    "jitter_local_pct",
    "shimmer_local_pct",
    "hnr_db",
    "speech_rate_syll_s",
)


@dataclass
class BaselineCalibrator:
    """Stateful accumulator for one speaker's calibration window. Call `.add_turn()`
    for each neutral opening turn, then `.build_profile()` once enough have arrived.
    Kept separate from `InterviewSession` (domain) because *when* calibration is
    considered sufficient, and how it degrades gracefully, is an application-level
    policy — the domain object (`BaselineProfile`) only knows how to use the result.
    """

    speaker: SpeakerId
    _samples: dict[str, list[float]] = field(default_factory=lambda: {f: [] for f in _TRACKED_FIELDS})
    _n_turns: int = 0
    _n_turns_skipped_no_pitch: int = 0

    def add_turn(self, features: ProsodyFeatures) -> None:
        # GUARD ADDED 2026-09-06: a real live session had Praat detect zero voiced
        # frames on several calibration turns, which — before `f0_detected` existed —
        # silently fed fabricated 0.0 values for f0/jitter/shimmer/HNR into the running
        # baseline mean/std, corrupting every later z-score computed against it. Skip
        # the whole turn (not just the pitch fields) rather than partially average in
        # fake data: jitter/shimmer/HNR are computed from the same Praat pitch pass, so
        # a failed pitch detection makes all four Praat-derived features unreliable
        # together, not just f0. speech_rate_syll_s (librosa onset-based, pitch-
        # independent) is lost for this turn too — an acceptable simplification over
        # partially-corrupt baseline statistics.
        if not features.f0_detected:
            self._n_turns_skipped_no_pitch += 1
            return

        raw = features.as_dict()
        for name in _TRACKED_FIELDS:
            value = raw.get(name)
            if value is not None:
                self._samples[name].append(float(value))
        self._n_turns += 1

    @property
    def n_turns_skipped_no_pitch(self) -> int:
        return self._n_turns_skipped_no_pitch

    def build_profile(self) -> BaselineProfile:
        means = {name: float(np.mean(values)) if values else 0.0 for name, values in self._samples.items()}
        stds = {name: float(np.std(values)) if values else 0.0 for name, values in self._samples.items()}
        return BaselineProfile(
            speaker=self.speaker,
            means=means,
            stds=stds,
            n_calibration_turns=self._n_turns,
        )
