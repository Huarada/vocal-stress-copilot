"""Ports (Protocols) the application layer depends on.

This is the dependency-inversion seam of the whole codebase: `application/` imports only
from here and from `domain/`, never from `infrastructure/`. Concrete adapters live in
`infrastructure/` and are wired in at the composition root (the `scripts/*.py` entry points).

Swapping the CNN for a wav2vec2 head, or swapping AssemblyAI for another STT vendor, means
writing a new adapter against these ports — the application/domain layers do not change.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from voicestress.domain.value_objects import (
    ArousalPrediction,
    AudioClipRef,
    ProsodyFeatures,
)


@runtime_checkable
class SpectrogramExtractorPort(Protocol):
    """Raw audio samples -> normalized narrowband spectrogram image, ready for the CNN."""

    # How many seconds an implementation pads short clips up to before analysis.
    # Exposed on the port (rather than the application layer importing the concrete
    # extractor's module constant, which would invert ADR-005's dependency arrow) so
    # `EvidenceService` can compute what fraction of an analysed clip is real audio
    # versus fabricated silence — the input to ADR-032's `attention_on_padding`.
    min_clip_seconds: float

    def extract(self, samples: np.ndarray, sample_rate_hz: int) -> np.ndarray:
        """Returns an HxWx3 float32 array in [0, 1]. Must raise ValueError (not return
        garbage) on degenerate input (silence, too-short clip) — see
        infrastructure/audio/spectrogram.py for the contrast guard."""
        ...


@runtime_checkable
class ProsodyExtractorPort(Protocol):
    def extract(self, samples: np.ndarray, sample_rate_hz: int) -> ProsodyFeatures: ...


@runtime_checkable
class ArousalClassifierPort(Protocol):
    """Wraps whatever model backs arousal prediction. The application layer only ever
    sees this interface, never a Keras/PyTorch object directly."""

    version: str

    def predict(self, spectrogram: np.ndarray) -> ArousalPrediction: ...

    def predict_batch(self, spectrograms: np.ndarray) -> list[ArousalPrediction]: ...


@runtime_checkable
class ExplainerPort(Protocol):
    """Produces a saliency map over the input spectrogram for one prediction (Grad-CAM or
    equivalent). Returns an HxW float32 array in [0, 1], same spatial size as the input."""

    def explain(self, spectrogram: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class AudioLoaderPort(Protocol):
    """Reads a clip from disk into mono float32 samples at its native rate. Kept separate
    from resampling so callers control the resample target explicitly (16 kHz analysis
    fork vs whatever the caller needs) rather than an adapter silently picking one."""

    def load(self, clip: AudioClipRef) -> tuple[np.ndarray, int]: ...
