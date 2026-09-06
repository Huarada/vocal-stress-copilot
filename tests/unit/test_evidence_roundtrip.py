import pytest

from voicestress.domain.entities import InterviewSession, TurnEvidence
from voicestress.domain.value_objects import (
    ArousalPrediction,
    FeatureAttribution,
    ProsodyFeatures,
    SpeakerId,
)  # noqa: F401  (ProsodyFeatures used by the legacy-inference tests below)


def _turn() -> TurnEvidence:
    return TurnEvidence(
        turn_id="t1",
        session_id="s1",
        t_start_ms=100,
        t_end_ms=3200,
        is_baseline_turn=False,
        transcript="Q: why? A: because.",
        arousal=ArousalPrediction(probability=0.73, model_version="v1-warm_started"),
        prosody_raw=ProsodyFeatures(
            f0_mean_hz=181.2, f0_std_hz=22.4, jitter_local_pct=1.1, shimmer_local_pct=5.6,
            hnr_db=14.3, speech_rate_syll_s=4.2, onset_latency_ms=850,
        ),
        baseline_deviation={"f0_mean_z": 1.8, "jitter_z": 0.4},
        top_contributors=[
            FeatureAttribution(feature="f0_mean_z", z_score=1.8),
            FeatureAttribution(feature="jitter_z", z_score=0.4),
        ],
        gradcam_path="artifacts/turn_artifacts/s1/t1_gradcam.png",
        spectrogram_path="artifacts/turn_artifacts/s1/t1_spec.png",
    )


def test_turn_evidence_roundtrips_through_dict():
    original = _turn()
    restored = TurnEvidence.from_evidence_dict(original.to_evidence_dict())

    assert restored.turn_id == original.turn_id
    assert restored.session_id == original.session_id
    assert restored.t_start_ms == original.t_start_ms
    assert restored.t_end_ms == original.t_end_ms
    assert restored.is_baseline_turn == original.is_baseline_turn
    assert restored.transcript == original.transcript
    assert restored.arousal.probability == pytest.approx(original.arousal.probability)
    assert restored.arousal.model_version == original.arousal.model_version
    assert restored.prosody_raw.as_dict() == original.prosody_raw.as_dict()
    assert restored.baseline_deviation == original.baseline_deviation
    assert [(c.feature, c.z_score) for c in restored.top_contributors] == [
        (c.feature, c.z_score) for c in original.top_contributors
    ]
    assert restored.gradcam_path == original.gradcam_path
    assert restored.spectrogram_path == original.spectrogram_path
    # roundtripping twice must be stable (idempotent)
    assert restored.to_evidence_dict() == original.to_evidence_dict()


def test_turn_evidence_roundtrip_with_no_baseline():
    original = TurnEvidence(
        turn_id="t2",
        session_id="s1",
        t_start_ms=0,
        t_end_ms=1000,
        is_baseline_turn=True,
        transcript="hello",
        arousal=ArousalPrediction(probability=0.2, model_version="v1"),
        prosody_raw=ProsodyFeatures(170, 15, 0.9, 4.5, 16, 3.8),
        baseline_deviation=None,
    )
    restored = TurnEvidence.from_evidence_dict(original.to_evidence_dict())
    assert restored.baseline_deviation is None
    assert restored.top_contributors == []


def test_legacy_session_without_f0_detected_infers_false_from_zero_pitch():
    """Sessions recorded before `f0_detected` existed carry Praat's fabricated 0.0
    fallbacks with no flag. Defaulting those to True would hand the dashboard and the
    Analyst Agent fake measurements as if they were real (ADR-026). A 0 Hz mean pitch
    is physically impossible, so it must be inferred as 'never detected'."""
    legacy = {
        "f0_mean_hz": 0.0,
        "f0_std_hz": 0.0,
        "jitter_local_pct": 0.0,
        "shimmer_local_pct": 0.0,
        "hnr_db": -6.5,
        "speech_rate_syll_s": 6.0,
        # no "f0_detected" key at all — this is the legacy shape
    }
    restored = ProsodyFeatures.from_dict(legacy)
    assert restored.f0_detected is False


def test_legacy_session_with_real_pitch_infers_true():
    legacy = {
        "f0_mean_hz": 157.8,
        "f0_std_hz": 9.2,
        "jitter_local_pct": 0.91,
        "shimmer_local_pct": 6.48,
        "hnr_db": 14.4,
        "speech_rate_syll_s": 2.0,
    }
    assert ProsodyFeatures.from_dict(legacy).f0_detected is True


def test_explicit_f0_detected_always_wins_over_inference():
    """A modern session's explicit flag must never be second-guessed by the heuristic."""
    d = {
        "f0_mean_hz": 0.0, "f0_std_hz": 0.0, "jitter_local_pct": 0.0,
        "shimmer_local_pct": 0.0, "hnr_db": 0.0, "speech_rate_syll_s": 0.0,
        "f0_detected": True,
    }
    assert ProsodyFeatures.from_dict(d).f0_detected is True


def test_interview_session_roundtrips_through_dict():
    session = InterviewSession(session_id="s1", speaker=SpeakerId("demo", "c1"))
    session.add_turn(_turn())

    payload = {
        "session_id": session.session_id,
        "turns": [t.to_evidence_dict() for t in session.turns],
    }
    restored = InterviewSession.from_session_dict(payload, speaker=SpeakerId("demo", "c1"))

    assert restored.session_id == "s1"
    assert len(restored.turns) == 1
    assert restored.get_turn("t1") is not None
