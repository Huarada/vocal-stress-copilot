"""Domain value objects.

Everything in this module is immutable (frozen dataclasses) and has zero dependency on
TensorFlow, librosa, or any I/O library. This is what makes the domain layer testable in
milliseconds and reusable regardless of which model/DSP library backs the ports.

See ARCHITECTURE.md §4 for the evidence contract this module implements in typed form.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class ArousalLabel(str, Enum):
    """Binary arousal target. Deliberately not named 'stress' at the label level —
    the model predicts *arousal*; the word 'stress' is a downstream, human-facing framing
    applied only after baseline calibration (see SKILLS.md Hard Rule #1)."""

    LOW = "low"
    HIGH = "high"

    @property
    def as_int(self) -> int:
        return 0 if self is ArousalLabel.LOW else 1

    @staticmethod
    def from_int(value: int) -> "ArousalLabel":
        return ArousalLabel.HIGH if int(value) == 1 else ArousalLabel.LOW


@dataclass(frozen=True, slots=True)
class SpeakerId:
    """Opaque speaker identity, scoped to whichever corpus it came from.

    Corpus-scoping matters: RAVDESS 'Actor_07' and a live interview candidate must never
    collide, and grouped cross-validation depends on this id being a true speaker boundary.
    """

    corpus: str
    raw_id: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.corpus}:{self.raw_id}"


@dataclass(frozen=True, slots=True)
class AudioClipRef:
    """A pointer to an audio clip on disk plus the metadata needed to preprocess it
    correctly. Does not hold samples — keeps the domain object cheap to pass around."""

    path: str
    speaker: SpeakerId
    sample_rate_hz: int
    duration_s: float | None = None


@dataclass(frozen=True, slots=True)
class LabeledClip:
    """One training/eval example: a clip plus its ground-truth arousal label and the
    corpus-specific provenance needed to audit the label later."""

    clip: AudioClipRef
    label: ArousalLabel
    source_annotation: str  # e.g. "ravdess_emotion=06" or "mustard_arousal=8.0"


@dataclass(frozen=True, slots=True)
class ProsodyFeatures:
    """Interpretable acoustic descriptors for one turn. Raw units — not yet baseline-relative.

    These are exactly the numbers a human reviewer can sanity-check against intuition
    (e.g. "pitch went up and got more variable"), which is why they exist as a first-class
    object distinct from the CNN's opaque spectrogram embedding.
    """

    f0_mean_hz: float
    f0_std_hz: float
    jitter_local_pct: float
    shimmer_local_pct: float
    hnr_db: float
    speech_rate_syll_s: float
    onset_latency_ms: float | None = None
    # ADDED 2026-09-06: a real live session showed f0_mean=0.0/f0_std=0.0/hnr≈-6.5 on 6
    # of 10 turns — Praat's pitch tracker found zero voiced frames (short and/or quiet
    # utterances), and the extractor was silently returning a ProsodyFeatures full of
    # fallback zeros indistinguishable from a genuine (nonsensical) 0 Hz measurement.
    # This field lets consumers (BaselineCalibrator, the Analyst Agent) tell the two
    # apart instead of averaging fabricated zeros into real statistics. True unless the
    # extractor explicitly detected zero voiced frames.
    f0_detected: bool = True
    # ADDED 2026-09-06 (ARCHITECTURE.md's item-1 follow-up): AssemblyAI's streaming
    # transcript returns per-word `confidence` (0-1) alongside the text — an ASR-
    # derived hesitation/mumbling proxy, independent of the Praat-based fields above
    # (different source: their recognition model, not our local DSP). Populated by
    # `application/asr_signals.py`, not by ProsodyExtractorPort (which only ever sees
    # raw audio, never transcript data) — same reason `onset_latency_ms` above is
    # externally-populated rather than Praat-derived.
    asr_mean_confidence: float | None = None

    def as_dict(self) -> dict[str, float | None | bool]:
        return {
            "f0_mean_hz": self.f0_mean_hz,
            "f0_std_hz": self.f0_std_hz,
            "jitter_local_pct": self.jitter_local_pct,
            "shimmer_local_pct": self.shimmer_local_pct,
            "hnr_db": self.hnr_db,
            "speech_rate_syll_s": self.speech_rate_syll_s,
            "onset_latency_ms": self.onset_latency_ms,
            "f0_detected": self.f0_detected,
            "asr_mean_confidence": self.asr_mean_confidence,
        }

    @staticmethod
    def from_dict(d: dict) -> "ProsodyFeatures":
        """Inverse of `as_dict` — reconstructs a ProsodyFeatures from a persisted
        evidence record (e.g. `artifacts/sessions/*.json`), so the dashboard backend can
        rehydrate real TurnEvidence objects and reuse `tool_definitions.build_tool_handlers`
        unchanged instead of duplicating its logic against raw dicts."""
        return ProsodyFeatures(
            f0_mean_hz=d["f0_mean_hz"],
            f0_std_hz=d["f0_std_hz"],
            jitter_local_pct=d["jitter_local_pct"],
            shimmer_local_pct=d["shimmer_local_pct"],
            hnr_db=d["hnr_db"],
            speech_rate_syll_s=d["speech_rate_syll_s"],
            onset_latency_ms=d.get("onset_latency_ms"),
            # LEGACY-SAFE DEFAULT: sessions recorded before `f0_detected` existed have
            # no such key, and defaulting them to True would present the extractor's
            # fabricated 0.0 fallbacks to the dashboard and the Analyst Agent as if
            # they were real measurements — exactly the confusion ADR-026 exists to
            # prevent. A genuine 0 Hz mean pitch is physically impossible, so zeros
            # are inferred as "pitch was never detected" rather than trusted.
            f0_detected=d.get(
                "f0_detected",
                not (float(d["f0_mean_hz"]) == 0.0 and float(d["f0_std_hz"]) == 0.0),
            ),
            asr_mean_confidence=d.get("asr_mean_confidence"),
        )


# Maps a ProsodyFeatures raw field name to its short evidence-contract key
# (ARCHITECTURE.md §4's `baseline_deviation` object: f0_mean_z, f0_std_z, jitter_z, ...).
# Explicit table instead of string-munging the unit suffix off the field name — the
# munging approach (`.replace('_hz','').replace('_pct','')...`) silently produced
# "jitter_local_z"/"shimmer_local_z"/"speech_rate_syll_s_z" instead of the documented
# "jitter_z"/"shimmer_z"/"speech_rate_z"; caught by test_baseline_profile.py.
_PROSODY_FIELD_TO_SHORT_KEY: dict[str, str] = {
    "f0_mean_hz": "f0_mean",
    "f0_std_hz": "f0_std",
    "jitter_local_pct": "jitter",
    "shimmer_local_pct": "shimmer",
    "hnr_db": "hnr",
    "speech_rate_syll_s": "speech_rate",
}


@dataclass(frozen=True, slots=True)
class BaselineProfile:
    """Per-speaker, per-session distribution of prosodic features gathered from neutral
    calibration turns (ARCHITECTURE.md §7). Feature-wise mean/std, used to convert a new
    turn's raw prosody into z-deviations.
    """

    speaker: SpeakerId
    means: Mapping[str, float]
    stds: Mapping[str, float]
    n_calibration_turns: int

    MIN_CALIBRATION_TURNS: int = field(default=3, repr=False, compare=False)

    @property
    def is_reliable(self) -> bool:
        """False if there isn't enough (or varied enough) calibration data to support
        a z-score claim. Consumers MUST check this before reporting deviations —
        see SKILLS.md Hard Rule about not reporting unsupported z-scores."""
        if self.n_calibration_turns < self.MIN_CALIBRATION_TURNS:
            return False
        degenerate = any(std < 1e-6 for std in self.stds.values())
        return not degenerate

    def deviation(self, features: ProsodyFeatures) -> dict[str, float]:
        """z = (x - mean) / std for every known feature. Raises if not reliable —
        callers must check `.is_reliable` first and branch explicitly rather than
        silently receiving misleading zeros."""
        if not self.is_reliable:
            raise ValueError(
                "BaselineProfile is not reliable enough to compute deviations "
                f"(n={self.n_calibration_turns}); check .is_reliable before calling."
            )
        raw = features.as_dict()
        out: dict[str, float] = {}
        for field_name, short_key in _PROSODY_FIELD_TO_SHORT_KEY.items():
            value = raw.get(field_name)
            if value is None:
                continue
            mean = self.means.get(field_name, 0.0)
            std = self.stds.get(field_name, 1.0) or 1.0
            out[f"{short_key}_z"] = (value - mean) / std
        return out


@dataclass(frozen=True, slots=True)
class ArousalPrediction:
    """The acoustic model's output for one clip/turn, plus the metadata needed to trust
    (or distrust) it. `probability` is P(HIGH arousal)."""

    probability: float
    model_version: str

    @property
    def label(self) -> ArousalLabel:
        return ArousalLabel.HIGH if self.probability >= 0.5 else ArousalLabel.LOW

    @property
    def confidence(self) -> float:
        """Distance from the decision boundary, rescaled to [0, 1]. 0 = coin flip."""
        return abs(self.probability - 0.5) * 2.0


@dataclass(frozen=True, slots=True)
class FeatureAttribution:
    feature: str
    z_score: float

    @property
    def direction(self) -> str:
        return "up" if self.z_score >= 0 else "down"


DECEPTION_VOCABULARY_DISCLAIMER = (
    "Vocal stress signal relative to this speaker's own baseline. "
    "Not a determination of truthfulness."
)

# Single source of truth for the vocabulary ADR-001 forbids anywhere in this system.
# Previously duplicated inside tests/unit/test_entities.py; promoted here in 2026-09-06
# when it became clear the *agent's own output* needed checking too, not just the
# evidence data — a 4B model (the only one this account can reach, ADR-024/ADR-030)
# does not reliably honour the prompt-level constraint. See
# `application/vocabulary_guard.py`.
BANNED_DECEPTION_TERMS: frozenset[str] = frozenset(
    {
        "lying", "lie", "lied", "lies",
        "deceptive", "deception", "deceit",
        "guilty",
        # The honesty axis, added 2026-09-06: an answer contrasting the signal with
        # "an honest answer" slipped through when only "truthful" was listed. Positive
        # forms count too — asserting someone WAS honest is equally a determination
        # this system has no evidence for.
        "truthful", "truthfulness", "untruthful",
        "honest", "honesty", "dishonest", "dishonesty",
    }
)
