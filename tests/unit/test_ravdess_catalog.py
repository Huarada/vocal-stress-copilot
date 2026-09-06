from pathlib import Path

import pytest

from voicestress.domain.value_objects import ArousalLabel
from voicestress.infrastructure.datasets.ravdess_catalog import (
    RavdessFilenameError,
    parse_ravdess_filename,
    training_subset,
)


@pytest.mark.parametrize(
    "filename,expected_emotion,expected_actor,expected_female",
    [
        ("03-01-01-01-01-01-01.wav", "neutral", "01", False),
        ("03-01-05-02-02-01-12.wav", "angry", "12", True),
        ("03-01-08-01-02-01-12.wav", "surprised", "12", True),
        ("03-01-03-01-01-01-07.wav", "happy", "07", False),
    ],
)
def test_parses_known_ravdess_filenames(
    filename, expected_emotion, expected_actor, expected_female
):
    info = parse_ravdess_filename(Path(filename))
    assert info.emotion_name == expected_emotion
    assert info.actor == expected_actor
    assert info.is_female is expected_female
    assert info.actor_id == f"Actor_{expected_actor}"


def test_rejects_malformed_filename():
    with pytest.raises(RavdessFilenameError):
        parse_ravdess_filename(Path("not-a-ravdess-file.wav"))


def test_rejects_unknown_emotion_code():
    with pytest.raises(RavdessFilenameError):
        parse_ravdess_filename(Path("03-01-99-01-01-01-01.wav"))


@pytest.mark.parametrize(
    "filename,expected_label",
    [
        ("03-01-01-01-01-01-01.wav", ArousalLabel.LOW),   # neutral
        ("03-01-02-01-01-01-01.wav", ArousalLabel.LOW),   # calm
        ("03-01-04-01-01-01-01.wav", ArousalLabel.LOW),   # sad
        ("03-01-05-01-01-01-01.wav", ArousalLabel.HIGH),  # angry
        ("03-01-06-01-01-01-01.wav", ArousalLabel.HIGH),  # fearful
        ("03-01-07-01-01-01-01.wav", ArousalLabel.HIGH),  # disgust
        ("03-01-08-01-01-01-01.wav", ArousalLabel.HIGH),  # surprised
    ],
)
def test_arousal_binarization(filename, expected_label):
    info = parse_ravdess_filename(Path(filename))
    assert info.arousal_label == expected_label


def test_happy_is_excluded_not_mislabeled():
    info = parse_ravdess_filename(Path("03-01-03-01-01-01-01.wav"))
    assert info.arousal_label is None


def test_training_subset_drops_none_labels():
    import pandas as pd

    manifest = pd.DataFrame(
        {
            "path": ["a", "b", "c"],
            "arousal_label": ["low", None, "high"],
        }
    )
    subset = training_subset(manifest)
    assert len(subset) == 2
    assert set(subset["arousal_label"]) == {"low", "high"}
