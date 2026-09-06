"""Reads artifacts/metrics/latest.json (produced by scripts/train_arousal_model.py) and
enforces the minimum accuracy bar this project needs to be an honest claim, per
ARCHITECTURE.md §6.

Threshold rationale (not arbitrary — each is anchored to a concrete baseline):

- RAVDESS OOF accuracy/F1 >= 0.65: the training set's majority-class baseline (always
  predicting HIGH) is ~61.5% (768/1248). Requiring 0.65 means the model has to be doing
  more than exploiting class imbalance — a small but real margin above the trivial
  baseline, appropriate for a 6.5k-parameter CNN on ~1200 samples.
- Cross-corpus (MUStARD++) accuracy must beat ITS OWN majority-class baseline (computed
  fresh from the eval set, not hardcoded) — this is the non-negotiable floor, because
  cross-corpus is genuinely hard (acted studio speech -> natural TV dialogue) and a model
  that can't clear its own eval set's trivial baseline is reporting noise, not signal.
- Cross-corpus accuracy >= 0.55 as an absolute floor on top of that: beating a
  near-50/50 majority baseline by even one sample would technically pass the relative
  check above without being remotely reliable; 0.55 requires an actual margin.

If a run misses these, SKILLS.md's guidance is explicit: report the honest number in the
README/video, do not weaken the threshold to make a run pass, and treat a near-miss as
the concept-implementation-validation loop finding something real to fix (features,
architecture, warm-start choice, label binarization thresholds) rather than a gate to
route around.
"""
import json
from pathlib import Path

import pytest

METRICS_PATH = Path(__file__).resolve().parent.parent.parent / "artifacts" / "metrics" / "latest.json"

RAVDESS_OOF_ACCURACY_MIN = 0.65
RAVDESS_OOF_F1_MIN = 0.65
CROSS_CORPUS_ACCURACY_ABSOLUTE_MIN = 0.55

pytestmark = pytest.mark.quality_gate


@pytest.fixture(scope="module")
def report() -> dict:
    if not METRICS_PATH.exists():
        pytest.skip(
            f"{METRICS_PATH} not found — run `python scripts/train_arousal_model.py` first."
        )
    return json.loads(METRICS_PATH.read_text())


def test_winning_variant_clears_ravdess_oof_bar(report):
    winner = report["winner"]
    variant_metrics = report["variants"][winner]
    assert variant_metrics["oof_accuracy"] >= RAVDESS_OOF_ACCURACY_MIN, (
        f"{winner} OOF accuracy {variant_metrics['oof_accuracy']:.3f} is below the "
        f"{RAVDESS_OOF_ACCURACY_MIN} floor — see this file's docstring for why that "
        "floor was chosen before weakening it."
    )
    assert variant_metrics["oof_f1"] >= RAVDESS_OOF_F1_MIN


def test_cross_corpus_beats_its_own_majority_baseline(report):
    cross_corpus = report["cross_corpus"]
    assert cross_corpus["accuracy"] > cross_corpus["baseline_majority_class_accuracy"], (
        f"Cross-corpus accuracy {cross_corpus['accuracy']:.3f} does not even beat the "
        f"MUStARD++ eval set's own majority-class baseline "
        f"({cross_corpus['baseline_majority_class_accuracy']:.3f}) — the model is not "
        "generalizing past RAVDESS at all; this is a real finding, not noise to explain away."
    )


def test_cross_corpus_clears_absolute_floor(report):
    cross_corpus = report["cross_corpus"]
    assert cross_corpus["accuracy"] >= CROSS_CORPUS_ACCURACY_ABSOLUTE_MIN, (
        f"Cross-corpus accuracy {cross_corpus['accuracy']:.3f} is below the absolute "
        f"floor of {CROSS_CORPUS_ACCURACY_ABSOLUTE_MIN} — report this honestly rather "
        "than treating it as passing."
    )


def test_both_variants_were_actually_trained_and_compared(report):
    """Guards against a config regression that silently skips the A/B warm-start
    experiment ARCHITECTURE.md §6 calls for."""
    assert set(report["variants"].keys()) == {"random_init", "warm_started"}
    assert report["variants"]["warm_started"]["layers_transplanted"] > 0, (
        "warm_started variant reports 0 layers transplanted — the checkpoint transplant "
        "silently failed (see tests/integration/test_checkpoint_transplant.py) and this "
        "run's 'warm_started' result is actually just another random-init run."
    )
