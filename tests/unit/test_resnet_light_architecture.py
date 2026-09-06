import numpy as np

from voicestress.infrastructure.models.resnet_light import build_resnet_light


def test_forward_pass_shape_and_probability_simplex():
    model = build_resnet_light(input_shape=(96, 128, 3), num_classes=2)
    batch = np.random.rand(4, 96, 128, 3).astype("float32")
    preds = model.predict(batch, verbose=0)
    assert preds.shape == (4, 2)
    # softmax output: each row sums to ~1, all entries in [0, 1]
    np.testing.assert_allclose(preds.sum(axis=1), 1.0, atol=1e-5)
    assert (preds >= 0).all() and (preds <= 1).all()


def test_architecture_is_input_resolution_agnostic():
    """Fully-convolutional + GAP means the same function must build successfully at
    any input H/W, and produce the same total parameter count regardless of H/W —
    this is the property `load_compatible_weights` (checkpoint transplant) depends on."""
    small = build_resnet_light(input_shape=(32, 32, 3), num_classes=2)
    large = build_resnet_light(input_shape=(96, 128, 3), num_classes=2)
    assert small.count_params() == large.count_params()


def test_param_count_matches_known_inherited_checkpoint():
    """Regression pin: the inherited resnet1_prelayer1.keras checkpoint has 6,578
    weight params (`model.count_params()`; the 19,384 figure seen in `.summary()` output
    also includes Adam's 12,806 optimizer-state slots, which are not model weights).
    If this drifts, the architecture was accidentally changed and the checkpoint
    transplant (load_compatible_weights) will silently stop matching shapes."""
    model = build_resnet_light(input_shape=(96, 128, 3), num_classes=2)
    assert model.count_params() == 6_578
