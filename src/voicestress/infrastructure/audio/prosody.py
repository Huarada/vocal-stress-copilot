"""Interpretable prosodic descriptors, via Praat (through parselmouth) for the voice-
quality measures it does best (F0, jitter, shimmer, HNR) and librosa for a speech-rate
proxy. This is the human-readable half of the acoustic pipeline (ARCHITECTURE.md §4) —
distinct from the CNN's opaque spectrogram embedding, and what the Analyst Agent actually
quotes when explaining a turn.

Implements `voicestress.domain.ports.ProsodyExtractorPort`.
"""
from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call

from voicestress.domain.value_objects import ProsodyFeatures

F0_MIN_HZ = 75.0
F0_MAX_HZ = 500.0


class ProsodyExtractionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PraatProsodyExtractor:
    f0_min_hz: float = F0_MIN_HZ
    f0_max_hz: float = F0_MAX_HZ

    def extract(self, samples: np.ndarray, sample_rate_hz: int) -> ProsodyFeatures:
        if samples.size < int(0.05 * sample_rate_hz):
            raise ProsodyExtractionError("Clip too short for prosodic analysis (<50ms)")

        sound = parselmouth.Sound(samples.astype(np.float64), sampling_frequency=sample_rate_hz)

        pitch = call(sound, "To Pitch", 0.0, self.f0_min_hz, self.f0_max_hz)
        f0_values = pitch.selected_array["frequency"]
        voiced = f0_values[f0_values > 0]
        # CONFIRMED 2026-09-06 on a real live session: Praat's pitch tracker found zero
        # voiced frames on 6 of 10 turns (mostly short, ≤500ms, utterances) — not
        # necessarily a bug in this extractor, plausibly the clips genuinely being too
        # short/quiet/breathy for autocorrelation-based pitch tracking. What WAS a bug:
        # falling back to f0_mean=f0_std=0.0 with no signal that extraction had failed,
        # indistinguishable from an (impossible) real 0 Hz measurement. f0_detected
        # below is what lets BaselineCalibrator and the Analyst Agent tell the difference.
        f0_detected = voiced.size > 0
        f0_mean = float(np.mean(voiced)) if voiced.size else 0.0
        f0_std = float(np.std(voiced)) if voiced.size else 0.0

        point_process = call(
            sound, "To PointProcess (periodic, cc)", self.f0_min_hz, self.f0_max_hz
        )

        if voiced.size >= 2:
            jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
            shimmer = call(
                [sound, point_process],
                "Get shimmer (local)",
                0,
                0,
                0.0001,
                0.02,
                1.3,
                1.6,
            ) * 100
        else:
            jitter, shimmer = 0.0, 0.0

        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, self.f0_min_hz, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        if not np.isfinite(hnr):
            hnr = 0.0

        speech_rate = self._onset_rate(samples, sample_rate_hz)

        return ProsodyFeatures(
            f0_mean_hz=f0_mean,
            f0_std_hz=f0_std,
            jitter_local_pct=float(jitter) if np.isfinite(jitter) else 0.0,
            shimmer_local_pct=float(shimmer) if np.isfinite(shimmer) else 0.0,
            hnr_db=float(hnr),
            speech_rate_syll_s=speech_rate,
            f0_detected=f0_detected,
        )

    @staticmethod
    def _onset_rate(samples: np.ndarray, sample_rate_hz: int) -> float:
        """Onset-detection-based syllable-rate proxy — a cheap, standard stand-in for
        true syllable counting (which needs a phonetic aligner we don't have here)."""
        duration_s = len(samples) / sample_rate_hz
        if duration_s <= 0:
            return 0.0
        onset_frames = librosa.onset.onset_detect(
            y=samples, sr=sample_rate_hz, units="frames", backtrack=False
        )
        return float(len(onset_frames)) / duration_s
