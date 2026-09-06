"""Grad-CAM over the arousal CNN's last convolutional feature map.

This is what turns "the model says 0.71" into "the model says 0.71 because of this
region of the spectrogram" — the visual half of the XAI layer (ARCHITECTURE.md §2, §6).
Cheap to add here specifically because `build_resnet_light` is a plain
conv -> ... -> GlobalAveragePooling2D -> Dense CNN, which is exactly the architecture
Grad-CAM was designed for.

Implements `voicestress.domain.ports.ExplainerPort`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D


class NoConvFeatureMapError(RuntimeError):
    pass


def _find_last_conv_output(model: Model):
    """Locates the tensor feeding the model's GlobalAveragePooling2D layer — i.e. the
    last spatial feature map before pooling collapses it. Structural lookup (by layer
    type + graph position), not by name, for the same reason `load_compatible_weights`
    matches by type: names are not a stable contract across model-build calls."""
    for layer in model.layers:
        if isinstance(layer, GlobalAveragePooling2D):
            return layer.input
    raise NoConvFeatureMapError(
        "No GlobalAveragePooling2D layer found — Grad-CAM as implemented here assumes "
        "the resnet_light architecture's conv -> GAP -> Dense shape."
    )


@dataclass
class GradCAMExplainer:
    """`voicestress.domain.ports.ExplainerPort` implementation, bound to one trained
    Keras model at construction time (building the grad-model graph is not free, so we
    do it once rather than per call).

    Not `slots=True` (unlike the domain value objects): `__post_init__` attaches
    `_grad_model`, a derived attribute with no corresponding constructor field, which a
    slotted dataclass rejects at assignment time — caught while smoke-testing this class
    against a real model instead of shipping it broken.
    """

    model: Model
    target_class: int = 1  # index 1 = HIGH arousal in our label encoding

    def __post_init__(self) -> None:
        conv_output = _find_last_conv_output(self.model)
        self._grad_model = Model(inputs=self.model.inputs, outputs=[conv_output, self.model.output])

    def explain(self, spectrogram: np.ndarray) -> np.ndarray:
        """`spectrogram`: HxWx3 float32 in [0,1]. Returns an HxW float32 heatmap in
        [0,1], resized (nearest-neighbor via broadcast-friendly resize) to match the
        input's spatial size."""
        x = np.expand_dims(spectrogram.astype(np.float32), axis=0)

        with tf.GradientTape() as tape:
            conv_out, predictions = self._grad_model(x)
            loss = predictions[:, self.target_class]

        grads = tape.gradient(loss, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (channels,)

        conv_out = conv_out[0]  # drop batch dim -> (h, w, channels)
        heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0)  # ReLU

        max_val = tf.reduce_max(heatmap)
        heatmap = heatmap / max_val if max_val > 0 else heatmap
        heatmap = heatmap.numpy().astype(np.float32)

        return _resize_heatmap(heatmap, spectrogram.shape[0], spectrogram.shape[1])

    def predict_probability(self, spectrogram: np.ndarray) -> float:
        """Convenience: P(HIGH arousal) for the same input, without a second forward
        pass through the un-instrumented model."""
        x = np.expand_dims(spectrogram.astype(np.float32), axis=0)
        probs = self.model.predict(x, verbose=0)[0]
        return float(probs[self.target_class])


def _resize_heatmap(heatmap: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    h, w = heatmap.shape
    y_old, y_new = np.linspace(0.0, 1.0, h), np.linspace(0.0, 1.0, out_h)
    x_old, x_new = np.linspace(0.0, 1.0, w), np.linspace(0.0, 1.0, out_w)
    tmp = np.empty((h, out_w), dtype=np.float32)
    for i in range(h):
        tmp[i, :] = np.interp(x_new, x_old, heatmap[i, :])
    out = np.empty((out_h, out_w), dtype=np.float32)
    for j in range(out_w):
        out[:, j] = np.interp(y_new, y_old, tmp[:, j])
    return out
