"""Mutable domain entities/aggregates. Unlike `value_objects.py`, these accumulate state
over the lifetime of an interview session — but they still hold no I/O, no TensorFlow,
no websocket. Application services mutate these through explicit methods only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from voicestress.domain.value_objects import (
    DECEPTION_VOCABULARY_DISCLAIMER,
    ArousalPrediction,
    FeatureAttribution,
    ProsodyFeatures,
    SpeakerId,
)

# ADDED 2026-09-06 after a real live session: turns under
# infrastructure/audio/spectrogram.py's MIN_CLIP_SECONDS (1.0s) get zero-padded before
# spectrogram extraction, producing a mostly-silent image the CNN — trained exclusively
# on ~3-4s unpadded RAVDESS clips — never saw during training. Observed directly: every
# turn <=500ms scored 0.67-0.97 regardless of spoken content (confident vs. hesitant
# tone), while turns >=1200ms stayed in a lower, more stable 0.00-0.68 band. The
# threshold here is set at the padding boundary rather than at RAVDESS's native
# duration (which would flag nearly every real interview turn) — a defensible middle
# ground, not a claim that turns above it are fully in-distribution.
MIN_RELIABLE_AROUSAL_DURATION_MS = 1000


@dataclass(slots=True)
class TurnEvidence:
    """Everything gathered about one interview turn — the aggregate that
    `application/evidence_service.py` assembles and that the Analyst Agent's tools read
    from. Mirrors the JSON contract in ARCHITECTURE.md §4."""

    turn_id: str
    session_id: str
    t_start_ms: int
    t_end_ms: int
    is_baseline_turn: bool
    transcript: str

    arousal: ArousalPrediction
    prosody_raw: ProsodyFeatures
    baseline_deviation: dict[str, float] | None  # None if baseline not yet reliable
    top_contributors: list[FeatureAttribution] = field(default_factory=list)
    gradcam_path: str | None = None
    spectrogram_path: str | None = None
    # ADDED 2026-09-06 (ADR-032): structured facts derived from the Grad-CAM heatmap —
    # which frequency band and time region drove the score, and how much of the model's
    # attention landed on zero-padded silence. Before this, the only Grad-CAM output
    # reaching the Analyst Agent was `gradcam_path`, a file path a text model cannot
    # open, which is why it kept answering "I don't have enough information to explain
    # the score". See application/gradcam_summary.py.
    gradcam_summary: dict | None = None
    gradcam_description: str | None = None

    disclaimer: str = DECEPTION_VOCABULARY_DISCLAIMER

    @property
    def duration_ms(self) -> int:
        return self.t_end_ms - self.t_start_ms

    @property
    def arousal_score_reliable(self) -> bool:
        """False when this turn's audio was shorter than the spectrogram extractor's
        padding threshold — see the MIN_RELIABLE_AROUSAL_DURATION_MS comment above.
        Consumers (the Analyst Agent, the dashboard) should lead with this caveat
        rather than presenting a short-clip score with the same confidence as a
        full-length one."""
        return self.duration_ms >= MIN_RELIABLE_AROUSAL_DURATION_MS

    @staticmethod
    def from_evidence_dict(d: dict) -> "TurnEvidence":
        """Inverse of `to_evidence_dict` — reconstructs a full TurnEvidence from a
        persisted record. Used by the dashboard backend to rehydrate sessions saved by
        `scripts/run_interview.py` / `scripts/build_demo_session.py`, so
        `tool_definitions.build_tool_handlers` works identically whether the session is
        live or loaded from disk."""
        baseline_deviation = d.get("baseline_deviation") or None
        return TurnEvidence(
            turn_id=d["turn_id"],
            session_id=d["session_id"],
            t_start_ms=d["t_start_ms"],
            t_end_ms=d["t_end_ms"],
            is_baseline_turn=d["is_baseline_turn"],
            transcript=d["transcript"],
            arousal=ArousalPrediction(
                probability=d["stress"]["score"], model_version=d["stress"]["model_version"]
            ),
            prosody_raw=ProsodyFeatures.from_dict(d["prosody_raw"]),
            baseline_deviation=baseline_deviation,
            top_contributors=[
                FeatureAttribution(feature=c["feature"], z_score=c["z"])
                for c in d["xai"]["top_contributors"]
            ],
            gradcam_path=d["xai"]["gradcam_png"],
            spectrogram_path=d["xai"]["spectrogram_png"],
            gradcam_summary=d["xai"].get("gradcam_summary"),
            gradcam_description=d["xai"].get("gradcam_description"),
            disclaimer=d.get("disclaimer", DECEPTION_VOCABULARY_DISCLAIMER),
        )

    def to_evidence_dict(self) -> dict:
        """The exact object the Analyst Agent's tools return — ARCHITECTURE.md §4."""
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "t_start_ms": self.t_start_ms,
            "t_end_ms": self.t_end_ms,
            "is_baseline_turn": self.is_baseline_turn,
            "transcript": self.transcript,
            "stress": {
                "score": self.arousal.probability,
                "label": self.arousal.label.value,
                "model_version": self.arousal.model_version,
                "calibrated": self.baseline_deviation is not None,
                "duration_ms": self.duration_ms,
                "score_reliable": self.arousal_score_reliable,
            },
            "baseline_deviation": self.baseline_deviation or {},
            "prosody_raw": self.prosody_raw.as_dict(),
            "xai": {
                "gradcam_png": self.gradcam_path,
                "spectrogram_png": self.spectrogram_path,
                "gradcam_summary": self.gradcam_summary,
                "gradcam_description": self.gradcam_description,
                "top_contributors": [
                    {"feature": c.feature, "z": c.z_score, "direction": c.direction}
                    for c in self.top_contributors
                ],
            },
            "disclaimer": self.disclaimer,
        }


@dataclass(slots=True)
class InterviewSession:
    """Aggregate root for one interview. The Interview Agent adapter creates one of
    these per call; the Sync Bus appends turns to it as they complete.

    Deliberately has no method that lets a turn's evidence influence how later turns are
    conducted — see ARCHITECTURE.md §2 on why the Interview Agent stays blind to scores.
    This class only *accumulates*; nothing here can steer the live conversation.
    """

    session_id: str
    speaker: SpeakerId
    turns: list[TurnEvidence] = field(default_factory=list)
    # ADR-044: the ONE thing the Interview Agent may ever report back — a candidate-
    # reported audio/connection problem, via the ONE tool it's configured with
    # (`flag_technical_issue`). Data-quality metadata, not analysis; the field name
    # itself is a small structural reminder that nothing analytical belongs here.
    technical_flags: list[str] = field(default_factory=list)

    @staticmethod
    def from_session_dict(d: dict, speaker: SpeakerId) -> "InterviewSession":
        """Rehydrates a whole session from the JSON shape `scripts/run_interview.py`
        and `scripts/build_demo_session.py` persist (`{"session_id": ..., "turns": [...]}`)."""
        session = InterviewSession(
            session_id=d["session_id"],
            speaker=speaker,
            technical_flags=list(d.get("technical_flags", [])),  # absent in pre-ADR-044 sessions
        )
        for turn_dict in d["turns"]:
            session.add_turn(TurnEvidence.from_evidence_dict(turn_dict))
        return session

    def add_turn(self, turn: TurnEvidence) -> None:
        if turn.session_id != self.session_id:
            raise ValueError(
                f"Turn {turn.turn_id} belongs to session {turn.session_id}, "
                f"not {self.session_id}"
            )
        self.turns.append(turn)

    @property
    def baseline_turns(self) -> list[TurnEvidence]:
        return [t for t in self.turns if t.is_baseline_turn]

    @property
    def scored_turns(self) -> list[TurnEvidence]:
        return [t for t in self.turns if not t.is_baseline_turn]

    def flagged_turns(self, min_score: float = 0.6) -> list[TurnEvidence]:
        return [t for t in self.scored_turns if t.arousal.probability >= min_score]

    def get_turn(self, turn_id: str) -> TurnEvidence | None:
        for turn in self.turns:
            if turn.turn_id == turn_id:
                return turn
        return None
