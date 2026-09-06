import os
from pathlib import Path

import pytest

from voicestress.infrastructure.models.resnet_light import (
    build_resnet_light,
    load_compatible_weights,
)

# Sibling of this repo, gitignored (ARCHITECTURE.md §8) — env-overridable rather than a
# literal personal path (fixed 2026-09-06; see scripts/build_features.py's comment).
_HACKATHON_ROOT = Path(
    os.environ.get("VOICESTRESS_HACKATHON_ROOT", Path(__file__).resolve().parents[3])
)
CHECKPOINT = (
    _HACKATHON_ROOT / "sarcasmoVoz" / "DetectarSarcasmoPorVoz-main" / "resnet1_prelayer1.keras"
)

pytestmark = pytest.mark.integration


def test_transplant_covers_almost_all_weighted_layers():
    if not CHECKPOINT.exists():
        pytest.skip(f"Inherited checkpoint not found at {CHECKPOINT}")

    model = build_resnet_light(input_shape=(96, 128, 3), num_classes=2)
    weighted_layers = [l for l in model.layers if l.get_weights()]

    copied = load_compatible_weights(model, str(CHECKPOINT))

    # Every layer that actually has weights should have matched by name+shape.
    # If this regresses, warm-start silently degrades to random init — see
    # ARCHITECTURE.md's corrected warm-start note and SKILLS.md Hard Rule discipline
    # around not silently misrepresenting what was actually reused.
    assert copied == len(weighted_layers), (
        f"Only {copied}/{len(weighted_layers)} weighted layers transplanted — "
        "checkpoint layer names likely no longer match build_resnet_light's output."
    )


def test_transplant_still_works_after_several_models_built_in_same_process():
    """Reproduces the actual k-fold training scenario: several `build_resnet_light()`
    models get constructed in one process before the transplant is attempted. This is
    the exact scenario that broke the original name-keyed matching (Keras' global
    layer-naming counter advances with every model built), caught here so it can't
    silently regress back to that bug.
    """
    if not CHECKPOINT.exists():
        pytest.skip(f"Inherited checkpoint not found at {CHECKPOINT}")

    for _ in range(5):  # simulate 5 StratifiedGroupKFold folds each building a fresh model
        build_resnet_light(input_shape=(96, 128, 3), num_classes=2)

    late_model = build_resnet_light(input_shape=(96, 128, 3), num_classes=2)
    weighted_layers = [l for l in late_model.layers if l.get_weights()]
    copied = load_compatible_weights(late_model, str(CHECKPOINT))
    assert copied == len(weighted_layers)
