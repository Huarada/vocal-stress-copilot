"""Builds a demo InterviewSession from real RAVDESS clips run through the real trained
model — no AssemblyAI connection needed. This is what the dashboard (Frente 2) renders
and is fully testable/runnable today, unlike `scripts/run_interview.py` (Frente 1),
which needs a live API key this implementation pass didn't have.

Picks a mix of low- and high-arousal real clips from one RAVDESS actor, treats the first
3 as baseline (calm) turns, and the rest as "interview answers" with synthetic
transcripts — clearly labeled as synthetic in the output, never presented as a real call.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from voicestress.application.baseline_service import BaselineCalibrator
from voicestress.application.evidence_service import EvidenceService
from voicestress.domain.entities import InterviewSession
from voicestress.domain.value_objects import SpeakerId
from voicestress.infrastructure.audio.prosody import PraatProsodyExtractor
from voicestress.infrastructure.audio.resampling import (
    ANALYSIS_SAMPLE_RATE_HZ,
    peak_normalize,
    resample_to_analysis_rate,
)
from voicestress.infrastructure.audio.spectrogram import NarrowbandSpectrogramExtractor
from voicestress.infrastructure.models.gradcam import GradCAMExplainer
from voicestress.infrastructure.models.keras_arousal_classifier import KerasArousalClassifier

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "artifacts" / "models" / "arousal_resnet_light.keras"
SESSIONS_DIR = ROOT / "artifacts" / "sessions"
ARTIFACT_DIR = ROOT / "artifacts" / "turn_artifacts"
# Sibling of this repo, gitignored, not committed (ARCHITECTURE.md §8) — see
# scripts/build_features.py's comment for why this is env-overridable rather than a
# literal personal path (fixed 2026-09-06, was leaking a local username/folder layout).
RAVDESS_ROOT = (
    Path(os.environ.get("VOICESTRESS_HACKATHON_ROOT", ROOT.parent))
    / "databaseAudio"
    / "audioSpeech"
)

DEMO_SESSION_ID = "demo_session_001"

# (relative filename under Actor_01/, synthetic question, synthetic "answer" transcript,
#  is_baseline)
SCRIPT = [
    ("03-01-01-01-01-01-01.wav", "Could you confirm your name for me?", "Yes, it's Alex Carter.", True),
    ("03-01-01-01-02-01-01.wav", "How's your connection, can you hear me okay?", "Yep, sounds good on my end.", True),
    ("03-01-02-01-01-01-01.wav", "Tell me a bit about your current role.", "I lead a small backend team, mostly Python services.", True),
    ("03-01-02-01-02-01-01.wav", "Why are you looking to leave your current position?", "I'd like more ownership over architecture decisions.", False),
    ("03-01-05-01-01-01-01.wav", "Can you walk me through a conflict you had with a teammate?", "There was a disagreement about a deployment rollback, we argued about it in standup.", False),
    ("03-01-06-01-01-01-01.wav", "Have you ever missed a deadline? What happened?", "I— yes, once, the client requirements changed very late.", False),
    ("03-01-08-01-01-01-01.wav", "What salary range are you expecting?", "Oh — I hadn't really thought about a specific number yet.", False),
]


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f"{MODEL_PATH} not found — run scripts/train_arousal_model.py first.")

    speaker = SpeakerId(corpus="demo", raw_id="alex_carter")
    session = InterviewSession(session_id=DEMO_SESSION_ID, speaker=speaker)
    calibrator = BaselineCalibrator(speaker=speaker)

    classifier = KerasArousalClassifier.from_checkpoint(MODEL_PATH, version="v1-warm_started")
    evidence_service = EvidenceService(
        spectrogram_extractor=NarrowbandSpectrogramExtractor(),
        prosody_extractor=PraatProsodyExtractor(),
        classifier=classifier,
        explainer=GradCAMExplainer(model=classifier.model),
        artifact_dir=ARTIFACT_DIR,
    )

    t_cursor_ms = 0
    baseline_ready = False

    for i, (filename, question, answer, is_baseline) in enumerate(SCRIPT, start=1):
        wav_path = RAVDESS_ROOT / "Actor_01" / filename
        if not wav_path.exists():
            print(f"[skip] {wav_path} not found")
            continue

        y, sr = sf.read(str(wav_path), always_2d=False)
        y = peak_normalize(resample_to_analysis_rate(y, sr))
        duration_ms = int(1000 * len(y) / ANALYSIS_SAMPLE_RATE_HZ)

        profile = calibrator.build_profile() if baseline_ready else None
        evidence = evidence_service.build_turn_evidence(
            session_id=DEMO_SESSION_ID,
            turn_id=f"t{i:04d}",
            t_start_ms=t_cursor_ms,
            t_end_ms=t_cursor_ms + duration_ms,
            is_baseline_turn=is_baseline,
            transcript=f"Q: {question}\nA: {answer}",
            samples=y,
            sample_rate_hz=ANALYSIS_SAMPLE_RATE_HZ,
            baseline_profile=profile,
        )
        session.add_turn(evidence)

        if is_baseline:
            calibrator.add_turn(evidence.prosody_raw)
            if sum(1 for _, _, _, b in SCRIPT[:i] if b) >= 3:
                baseline_ready = True

        t_cursor_ms += duration_ms + 500  # small gap between turns
        print(
            f"[{evidence.turn_id}] baseline={is_baseline} "
            f"arousal={evidence.arousal.probability:.3f} "
            f"calibrated={'yes' if profile and profile.is_reliable else 'no'}"
        )

    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SESSIONS_DIR / f"{DEMO_SESSION_ID}.json"
    data = {
        "session_id": DEMO_SESSION_ID,
        "is_synthetic_demo": True,
        "note": (
            "Synthetic demo: real speaker audio (RAVDESS actor 1) paired with scripted "
            "transcripts for illustration. Not a real interview."
        ),
        "turns": [t.to_evidence_dict() for t in session.turns],
    }
    out_path.write_text(json.dumps(data, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
