import numpy as np
import pytest

from voicestress.infrastructure.models.gradcam import GradCAMExplainer
from voicestress.infrastructure.models.resnet_light import build_resnet_light


def test_gradcam_heatmap_shape_and_range():
    model = build_resnet_light(input_shape=(96, 128, 3), num_classes=2)
    explainer = GradCAMExplainer(model=model)
    img = np.random.RandomState(0).rand(96, 128, 3).astype("float32")

    heatmap = explainer.explain(img)

    assert heatmap.shape == (96, 128)
    assert heatmap.dtype == np.float32
    assert np.isfinite(heatmap).all()
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6


def test_gradcam_predict_probability_matches_model_predict():
    model = build_resnet_light(input_shape=(96, 128, 3), num_classes=2)
    explainer = GradCAMExplainer(model=model, target_class=1)
    img = np.random.RandomState(1).rand(96, 128, 3).astype("float32")

    prob_from_explainer = explainer.predict_probability(img)
    prob_from_model = model.predict(np.expand_dims(img, 0), verbose=0)[0, 1]

    assert prob_from_explainer == pytest.approx(float(prob_from_model))


def test_gradcam_survives_several_models_built_first():
    """Same class of risk as the checkpoint-transplant bug: build several models first,
    then confirm Grad-CAM still finds the GAP layer structurally rather than by name."""
    for _ in range(3):
        build_resnet_light(input_shape=(96, 128, 3), num_classes=2)
    model = build_resnet_light(input_shape=(64, 64, 3), num_classes=2)
    explainer = GradCAMExplainer(model=model)
    img = np.random.RandomState(2).rand(64, 64, 3).astype("float32")
    heatmap = explainer.explain(img)
    assert heatmap.shape == (64, 64)
