import pandas as pd

from voicestress.infrastructure.datasets.mustard_catalog import (
    HIGH_AROUSAL_MIN,
    LOW_AROUSAL_MAX,
    eval_subset,
    _binarize,
)


def test_binarize_boundaries():
    assert _binarize(LOW_AROUSAL_MAX) == "low"
    assert _binarize(LOW_AROUSAL_MAX + 0.5) is None
    assert _binarize(HIGH_AROUSAL_MIN - 0.5) is None
    assert _binarize(HIGH_AROUSAL_MIN) == "high"


def test_binarize_matches_known_distribution_shape():
    # Regression pin for the real MUStARD++ Arousal histogram (value_counts checked
    # against the CSV on disk during architecture design): 3:7, 4:21, 5:226, 6:297,
    # 7:400, 8:240, 9:10. Low/high split should land in the low-hundreds each, not near
    # a degenerate 0 or the full 1201.
    values = (
        [3.0] * 7
        + [4.0] * 21
        + [5.0] * 226
        + [6.0] * 297
        + [7.0] * 400
        + [8.0] * 240
        + [9.0] * 10
    )
    df = pd.DataFrame({"Arousal": values})
    labels = df["Arousal"].apply(_binarize)
    counts = labels.value_counts(dropna=True)
    assert 200 <= counts.get("low", 0) <= 300
    assert 200 <= counts.get("high", 0) <= 300
    assert labels.isna().sum() == 697  # the 6-7 dead zone


def test_eval_subset_drops_dead_zone():
    manifest = pd.DataFrame(
        {"path": ["a", "b", "c"], "arousal_label": ["low", None, "high"]}
    )
    subset = eval_subset(manifest)
    assert len(subset) == 2
