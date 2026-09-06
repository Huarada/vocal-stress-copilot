"""Sample-rate normalization.

SKILLS.md Hard Rule #5: assert sample rate at every boundary. This module is the one
place resampling happens, so every other module can assume 16 kHz mono without re-checking.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

ANALYSIS_SAMPLE_RATE_HZ = 16_000  # matches the sarcasm-model training pipeline


def to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples.astype(np.float32, copy=False)
    return samples.mean(axis=1).astype(np.float32, copy=False)


def resample_to_analysis_rate(samples: np.ndarray, source_rate_hz: int) -> np.ndarray:
    """Polyphase resample to ANALYSIS_SAMPLE_RATE_HZ. No-ops if already at that rate.

    Deliberately independent of whatever rate the Voice Agent stream runs at (24 kHz) —
    ARCHITECTURE.md §5 calls out that the analysis fork must be produced from its own
    resample, not derived from audio already resampled for the agent's purposes.
    """
    samples = to_mono(samples)
    if source_rate_hz == ANALYSIS_SAMPLE_RATE_HZ:
        return samples
    resampled = resample_poly(samples, ANALYSIS_SAMPLE_RATE_HZ, source_rate_hz)
    return resampled.astype(np.float32, copy=False)


def peak_normalize(samples: np.ndarray, target_peak: float = 1.0) -> np.ndarray:
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak <= 1e-9:
        return samples
    return (samples / peak * target_peak).astype(np.float32, copy=False)


class DegenerateAudioError(ValueError):
    """Raised when a clip is too short or too quiet to analyze meaningfully."""


def assert_analyzable(samples: np.ndarray, sample_rate_hz: int, min_duration_s: float = 0.3) -> None:
    duration_s = len(samples) / sample_rate_hz if sample_rate_hz else 0.0
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    if duration_s < min_duration_s:
        raise DegenerateAudioError(f"Clip too short: {duration_s:.3f}s < {min_duration_s}s")
    if rms < 1e-4:
        raise DegenerateAudioError(f"Clip is effectively silent (rms={rms:.2e})")
