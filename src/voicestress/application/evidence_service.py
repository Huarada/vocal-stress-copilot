"""Evidence-assembly use-case: raw audio + transcript -> `TurnEvidence`
(ARCHITECTURE.md §4's contract, in typed form).

This is the seam where the whole acoustic pipeline (spectrogram, prosody, classifier,
XAI) meets one interview turn. It depends only on domain ports — see
`voicestress.domain.ports` — so it is testable with fakes and swappable at the
composition root without touching this file.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from voicestress.application.gradcam_summary import describe_gradcam, summarize_gradcam
from voicestress.domain.entities import TurnEvidence
from voicestress.domain.ports import (
    ArousalClassifierPort,
    ExplainerPort,
    ProsodyExtractorPort,
    SpectrogramExtractorPort,
)
from voicestress.domain.value_objects import BaselineProfile, FeatureAttribution

TOP_CONTRIBUTORS_COUNT = 3


@dataclass
class EvidenceService:
    spectrogram_extractor: SpectrogramExtractorPort
    prosody_extractor: ProsodyExtractorPort
    classifier: ArousalClassifierPort
    explainer: ExplainerPort | None = None
    artifact_dir: Path | None = None

    def build_turn_evidence(
        self,
        session_id: str,
        turn_id: str,
        t_start_ms: int,
        t_end_ms: int,
        is_baseline_turn: bool,
        transcript: str,
        samples: np.ndarray,
        sample_rate_hz: int,
        baseline_profile: BaselineProfile | None,
    ) -> TurnEvidence:
        spectrogram = self.spectrogram_extractor.extract(samples, sample_rate_hz)
        prosody = self.prosody_extractor.extract(samples, sample_rate_hz)
        prediction = self.classifier.predict(spectrogram)

        baseline_deviation: dict[str, float] | None = None
        top_contributors: list[FeatureAttribution] = []
        if baseline_profile is not None and baseline_profile.is_reliable:
            baseline_deviation = baseline_profile.deviation(prosody)
            ranked = sorted(baseline_deviation.items(), key=lambda kv: abs(kv[1]), reverse=True)
            top_contributors = [
                FeatureAttribution(feature=name, z_score=z)
                for name, z in ranked[:TOP_CONTRIBUTORS_COUNT]
            ]

        # ADDED 2026-09-06 (ADR-032): the heatmap is now computed unconditionally (not
        # only when saving PNGs) and summarized into citable facts, because the Analyst
        # Agent previously received only a file path it could not open — leaving it
        # genuinely unable to explain any score.
        heatmap = self.explainer.explain(spectrogram) if self.explainer is not None else None

        gradcam_summary: dict | None = None
        gradcam_description: str | None = None
        if heatmap is not None:
            real_duration_s = len(samples) / sample_rate_hz if sample_rate_hz else 0.0
            padding_threshold_s = getattr(self.spectrogram_extractor, "min_clip_seconds", 1.0)
            real_audio_fraction = (
                min(1.0, real_duration_s / padding_threshold_s) if padding_threshold_s else 1.0
            )
            gradcam_summary = summarize_gradcam(
                heatmap, real_audio_fraction=real_audio_fraction
            )
            gradcam_description = describe_gradcam(gradcam_summary)

        gradcam_path, spectrogram_path = self._maybe_save_artifacts(
            session_id, turn_id, spectrogram, heatmap
        )

        return TurnEvidence(
            turn_id=turn_id,
            session_id=session_id,
            t_start_ms=t_start_ms,
            t_end_ms=t_end_ms,
            is_baseline_turn=is_baseline_turn,
            transcript=transcript,
            arousal=prediction,
            prosody_raw=prosody,
            baseline_deviation=baseline_deviation,
            top_contributors=top_contributors,
            gradcam_path=gradcam_path,
            spectrogram_path=spectrogram_path,
            gradcam_summary=gradcam_summary,
            gradcam_description=gradcam_description,
        )

    def _maybe_save_artifacts(
        self,
        session_id: str,
        turn_id: str,
        spectrogram: np.ndarray,
        heatmap: np.ndarray | None,
    ) -> tuple[str | None, str | None]:
        if self.artifact_dir is None:
            return None, None

        session_dir = self.artifact_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        spectrogram_path = session_dir / f"{turn_id}_spec.png"
        Image.fromarray((np.clip(spectrogram, 0, 1) * 255).astype(np.uint8)).save(spectrogram_path)

        gradcam_path: str | None = None
        if heatmap is not None:
            gradcam_img = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
            gradcam_out = session_dir / f"{turn_id}_gradcam.png"
            Image.fromarray(gradcam_img).save(gradcam_out)
            gradcam_path = str(gradcam_out)

        return gradcam_path, str(spectrogram_path)
