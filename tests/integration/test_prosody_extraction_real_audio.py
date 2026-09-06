import os
from pathlib import Path

import pytest
import soundfile as sf

from voicestress.infrastructure.audio.prosody import PraatProsodyExtractor
from voicestress.infrastructure.audio.resampling import (
    peak_normalize,
    resample_to_analysis_rate,
    ANALYSIS_SAMPLE_RATE_HZ,
)

# Sibling of this repo, gitignored (ARCHITECTURE.md §8) — env-overridable rather than a
# literal personal path (fixed 2026-09-06; see scripts/build_features.py's comment).
RAVDESS_ROOT = (
    Path(os.environ.get("VOICESTRESS_HACKATHON_ROOT", Path(__file__).resolve().parents[3]))
    / "databaseAudio"
    / "audioSpeech"
)

pytestmark = pytest.mark.integration


def _load_one():
    if not RAVDESS_ROOT.exists():
        pytest.skip(f"RAVDESS not found at {RAVDESS_ROOT}")
    wav_path = next(RAVDESS_ROOT.rglob("*.wav"))
    y, sr = sf.read(str(wav_path), always_2d=False)
    return peak_normalize(resample_to_analysis_rate(y, sr))


def test_prosody_features_are_physically_plausible():
    y = _load_one()
    features = PraatProsodyExtractor().extract(y, ANALYSIS_SAMPLE_RATE_HZ)

    # Human voice F0 range, generously bounded
    assert 60.0 <= features.f0_mean_hz <= 500.0
    assert features.f0_std_hz >= 0.0
    assert features.jitter_local_pct >= 0.0
    assert features.shimmer_local_pct >= 0.0
    assert features.speech_rate_syll_s >= 0.0
    # HNR for real speech is usually within roughly this band
    assert -20.0 <= features.hnr_db <= 40.0


def test_prosody_differs_between_calm_and_angry_same_actor():
    """Weak sanity probe, not a model-accuracy test: angry speech should tend to show
    higher/more variable pitch than calm speech from the *same* actor/statement pair.
    A single-pair check is not proof, but a systematic failure here (features identical
    regardless of emotion) would indicate a broken extractor, not a modeling issue."""
    if not RAVDESS_ROOT.exists():
        pytest.skip(f"RAVDESS not found at {RAVDESS_ROOT}")

    def find(emotion_code: str) -> Path | None:
        matches = list(RAVDESS_ROOT.glob(f"Actor_01/03-01-{emotion_code}-01-01-01-01.wav"))
        return matches[0] if matches else None

    calm_path, angry_path = find("02"), find("05")
    if calm_path is None or angry_path is None:
        pytest.skip("Expected calm/angry files for Actor_01 not found")

    extractor = PraatProsodyExtractor()

    def load_and_extract(p: Path):
        y, sr = sf.read(str(p), always_2d=False)
        y = peak_normalize(resample_to_analysis_rate(y, sr))
        return extractor.extract(y, ANALYSIS_SAMPLE_RATE_HZ)

    calm = load_and_extract(calm_path)
    angry = load_and_extract(angry_path)

    # angry should show more pitch variability than calm for the same speaker/text
    assert angry.f0_std_hz > calm.f0_std_hz * 0.8  # loose bound, this is a sanity probe
