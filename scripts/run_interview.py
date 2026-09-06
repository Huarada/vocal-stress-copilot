"""Composition root: a live interview session against the real AssemblyAI Voice Agent
API, with the acoustic analysis side-channel wired in per ARCHITECTURE.md — captured
from a real local microphone. For a browser-based capture surface (no local Python/mic
drivers required), see web/backend.py's `/ws/interview` and web/static/capture.html.

Requires ASSEMBLYAI_API_KEY (see .env.example) and a working microphone.

HONESTY NOTE (ARCHITECTURE.md §7b): connectivity and protocol shape are CONFIRMED
against a real account (scripts/check_live_connection.py, 2026-09-05) — auth works,
`session.update`/`session.updated` round-trip correctly, and a full live conversation
has run end-to-end (candidate mic -> Voice Agent -> SyncBus -> EvidenceService -> saved
session). What changed 2026-09-05 (ADR-041): the actual orchestration used to live
entirely in this file; it is now `application/live_interview_runner.py`, audio-source-
agnostic, so this script and the browser capture path share one implementation instead
of two copies of ~15 documented bug fixes (ADR.md Appendix A). This file is now just the
mic-specific composition root.

ALSO CHANGED 2026-09-05 (ADR-040): every live run before this date sent the candidate's
mic audio to the Voice Agent and never played or displayed its reply — `reply.audio`
is now wired (application/live_interview_runner.py), but this terminal script still has
no speaker output; it prints the agent's transcribed reply text (`[agent] ...`) instead
of playing its voice. A terminal script has no natural place to route synthesized audio
without a new dependency; the browser capture page does, and plays it back for real. If
you need to actually hear the agent from this script, that's the remaining gap here —
not upstream: `on_agent_reply_audio` is available on `LiveInterviewRunner` and simply
isn't connected to anything in this composition root.

Usage:
    python scripts/run_interview.py
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from voicestress.application.live_interview_runner import (
    CHUNK_SAMPLES,
    SAMPLE_RATE_HZ,
    LiveInterviewRunner,
)
from voicestress.config import AssemblyAISettings
from voicestress.infrastructure.audio.mic_source import MicAudioSource

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "artifacts" / "models" / "arousal_resnet_light.keras"


async def main() -> None:
    settings = AssemblyAISettings.from_env()
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"{MODEL_PATH} not found — run scripts/train_arousal_model.py first."
        )
    runner = LiveInterviewRunner(settings=settings, model_path=MODEL_PATH)
    mic = MicAudioSource(sample_rate_hz=SAMPLE_RATE_HZ, chunk_samples=CHUNK_SAMPLES)
    await runner.run(audio_source=mic)


if __name__ == "__main__":
    asyncio.run(main())
