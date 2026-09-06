"""Runs the extractor against real RAVDESS files on disk. Skips gracefully if the
dataset isn't present on this machine (e.g. CI), so the suite stays runnable everywhere
while still exercising real audio locally — see SKILLS.md §3 for where the data lives.
"""
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voicestress.infrastructure.audio.resampling import (
    resample_to_analysis_rate,
    peak_normalize,
    ANALYSIS_SAMPLE_RATE_HZ,
)
from voicestress.infrastructure.audio.spectrogram import (
    NarrowbandSpectrogramExtractor,
    FlatSpectrogramError,
)

# Sibling of this repo, gitignored (ARCHITECTURE.md §8) — env-overridable rather than a
# literal personal path (fixed 2026-09-06; see scripts/build_features.py's comment).
RAVDESS_ROOT = (
    Path(os.environ.get("VOICESTRESS_HACKATHON_ROOT", Path(__file__).resolve().parents[3]))
    / "databaseAudio"
    / "audioSpeech"
)

pytestmark = pytest.mark.integration


def _first_n_wavs(n: int) -> list[Path]:
    if not RAVDESS_ROOT.exists():
        pytest.skip(f"RAVDESS not found at {RAVDESS_ROOT}")
    files = sorted(RAVDESS_ROOT.rglob("*.wav"))[:n]
    if not files:
        pytest.skip("No .wav files found under RAVDESS root")
    return files


def test_extract_shape_and_range_on_real_clip():
    wav_path = _first_n_wavs(1)[0]
    y, sr = sf.read(str(wav_path), always_2d=False)
    y = resample_to_analysis_rate(y, sr)
    y = peak_normalize(y)
    assert sr == 48000 or sr == ANALYSIS_SAMPLE_RATE_HZ  # RAVDESS ships 48kHz originals

    extractor = NarrowbandSpectrogramExtractor()
    img = extractor.extract(y, ANALYSIS_SAMPLE_RATE_HZ)

    assert img.shape == (extractor.output_height, extractor.output_width, 3)
    assert img.dtype == np.float32
    assert np.isfinite(img).all()
    assert 0.0 <= img.min() and img.max() <= 1.0
    # not a flat/degenerate image
    assert img.std() > 0.01


def test_extract_is_deterministic():
    wav_path = _first_n_wavs(1)[0]
    y, sr = sf.read(str(wav_path), always_2d=False)
    y = peak_normalize(resample_to_analysis_rate(y, sr))
    extractor = NarrowbandSpectrogramExtractor()
    img_a = extractor.extract(y, ANALYSIS_SAMPLE_RATE_HZ)
    img_b = extractor.extract(y, ANALYSIS_SAMPLE_RATE_HZ)
    np.testing.assert_array_equal(img_a, img_b)


def test_extract_raises_on_pure_silence():
    silence = np.zeros(ANALYSIS_SAMPLE_RATE_HZ * 2, dtype=np.float32)
    extractor = NarrowbandSpectrogramExtractor()
    with pytest.raises(FlatSpectrogramError):
        extractor.extract(silence, ANALYSIS_SAMPLE_RATE_HZ)


def test_extract_across_several_real_clips_has_variance_between_clips():
    """Sanity check that different clips actually produce different images — catches
    accidental constant-output bugs that a single-file test would miss."""
    wavs = _first_n_wavs(6)
    extractor = NarrowbandSpectrogramExtractor()
    images = []
    for wav_path in wavs:
        y, sr = sf.read(str(wav_path), always_2d=False)
        y = peak_normalize(resample_to_analysis_rate(y, sr))
        try:
            images.append(extractor.extract(y, ANALYSIS_SAMPLE_RATE_HZ))
        except FlatSpectrogramError:
            continue
    assert len(images) >= 3
    stacked = np.stack(images)
    # per-pixel variance across clips should be well above zero
    assert stacked.var(axis=0).mean() > 1e-4
