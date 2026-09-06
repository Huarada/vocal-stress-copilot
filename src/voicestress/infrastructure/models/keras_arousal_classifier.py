"""Adapts a trained `build_resnet_light` Keras model to `ArousalClassifierPort`, so the
application layer only ever depends on the port — not on Keras, not on a file path.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from voicestress.domain.value_objects import ArousalPrediction

HIGH_AROUSAL_CLASS_INDEX = 1  # matches ArousalLabel.HIGH.as_int


@dataclass
class KerasArousalClassifier:
    """`voicestress.domain.ports.ArousalClassifierPort` implementation."""

    model: tf.keras.Model
    version: str

    @classmethod
    def from_checkpoint(cls, path: str | Path, version: str) -> "KerasArousalClassifier":
        model = tf.keras.models.load_model(str(path))
        return cls(model=model, version=version)

    def predict(self, spectrogram: np.ndarray) -> ArousalPrediction:
        return self.predict_batch(np.expand_dims(spectrogram, axis=0))[0]

    def predict_batch(self, spectrograms: np.ndarray) -> list[ArousalPrediction]:
        probs = self.model.predict(spectrograms, verbose=0)
        return [
            ArousalPrediction(probability=float(p[HIGH_AROUSAL_CLASS_INDEX]), model_version=self.version)
            for p in probs
        ]
