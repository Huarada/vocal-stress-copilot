"""End-to-end wiring test: real audio file -> spectrogram -> prosody -> classifier ->
Grad-CAM -> TurnEvidence -> tool_definitions handlers -> evidence dict. Uses a freshly
built (untrained) model, since this test validates PLUMBING, not accuracy — accuracy is
covered separately by tests/quality_gates/test_minimum_accuracy_gate.py against the real
trained model.
"""
import os
from pathlib import Path

import pytest
import soundfile as sf

from voicestress.application.evidence_service import EvidenceService
from voicestress.domain.entities import InterviewSession
from voicestress.domain.value_objects import SpeakerId
from voicestress.infrastructure.agents.tool_definitions import build_tool_handlers
from voicestress.infrastructure.audio.prosody import PraatProsodyExtractor
from voicestress.infrastructure.audio.resampling import (
    ANALYSIS_SAMPLE_RATE_HZ,
    peak_normalize,
    resample_to_analysis_rate,
)
from voicestress.infrastructure.audio.spectrogram import NarrowbandSpectrogramExtractor
from voicestress.infrastructure.models.gradcam import GradCAMExplainer
from voicestress.infrastructure.models.keras_arousal_classifier import KerasArousalClassifier
from voicestress.infrastructure.models.resnet_light import build_resnet_light

# Sibling of this repo, gitignored (ARCHITECTURE.md §8) — env-overridable rather than a
# literal personal path (fixed 2026-09-06; see scripts/build_features.py's comment).
RAVDESS_ROOT = (
    Path(os.environ.get("VOICESTRESS_HACKATHON_ROOT", Path(__file__).resolve().parents[3]))
    / "databaseAudio"
    / "audioSpeech"
)

pytestmark = pytest.mark.integration


@pytest.fixture
def evidence_service(tmp_path) -> EvidenceService:
    model = build_resnet_light(input_shape=(96, 128, 3), num_classes=2)
    classifier = KerasArousalClassifier(model=model, version="untrained-wiring-test")
    explainer = GradCAMExplainer(model=model)
    return EvidenceService(
        spectrogram_extractor=NarrowbandSpectrogramExtractor(),
        prosody_extractor=PraatProsodyExtractor(),
        classifier=classifier,
        explainer=explainer,
        artifact_dir=tmp_path,
    )


def _load_real_clip():
    if not RAVDESS_ROOT.exists():
        pytest.skip(f"RAVDESS not found at {RAVDESS_ROOT}")
    wav_path = next(RAVDESS_ROOT.rglob("*.wav"))
    y, sr = sf.read(str(wav_path), always_2d=False)
    return peak_normalize(resample_to_analysis_rate(y, sr))


def test_build_turn_evidence_produces_complete_record_with_artifacts(evidence_service, tmp_path):
    samples = _load_real_clip()

    evidence = evidence_service.build_turn_evidence(
        session_id="s_test",
        turn_id="t1",
        t_start_ms=0,
        t_end_ms=3000,
        is_baseline_turn=False,
        transcript="This is a test answer.",
        samples=samples,
        sample_rate_hz=ANALYSIS_SAMPLE_RATE_HZ,
        baseline_profile=None,  # no baseline yet -> deviation must be None, not crash
    )

    assert evidence.turn_id == "t1"
    assert 0.0 <= evidence.arousal.probability <= 1.0
    assert evidence.baseline_deviation is None
    assert evidence.top_contributors == []
    assert evidence.gradcam_path is not None and Path(evidence.gradcam_path).exists()
    assert evidence.spectrogram_path is not None and Path(evidence.spectrogram_path).exists()


def test_evidence_flows_correctly_into_tool_handlers(evidence_service):
    """Full loop: build evidence -> attach to a session -> Analyst Agent tool handlers
    -> evidence dict, exactly as the Analyst Agent would receive it via tool.call."""
    samples = _load_real_clip()
    evidence = evidence_service.build_turn_evidence(
        session_id="s_test",
        turn_id="t1",
        t_start_ms=0,
        t_end_ms=3000,
        is_baseline_turn=False,
        transcript="This is a test answer.",
        samples=samples,
        sample_rate_hz=ANALYSIS_SAMPLE_RATE_HZ,
        baseline_profile=None,
    )

    session = InterviewSession(session_id="s_test", speaker=SpeakerId("interview", "c1"))
    session.add_turn(evidence)

    handlers = build_tool_handlers(session)
    result = handlers["get_turn_evidence"]({"turn_id": "t1"})

    assert result["turn_id"] == "t1"
    assert "disclaimer" in result
    assert result["xai"]["gradcam_png"] is not None
