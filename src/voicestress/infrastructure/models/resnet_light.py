"""The ResNet-light architecture, ported verbatim (topology, filter counts, dropout
rates, regularization) from `resnet_prelayer1.py` / notebook cell 20 in the inherited
sarcasm-detection project (SKILLS.md §3). Only the *task* changes — arousal instead of
sarcasm — not the network.

Fully convolutional + GlobalAveragePooling before the Dense head means every weight in
this network (conv kernels, BatchNorm params, the final Dense) has a shape independent of
input height/width. That is what makes `load_compatible_weights` below possible: the
inherited checkpoint was trained at 32x32 (a CIFAR-10 pretraining experiment — see
ARCHITECTURE.md's corrected note; it is not a spectrogram-domain checkpoint), and it can
still be loaded into a model built at 96x128 layer-by-layer.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    SpatialDropout2D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def resnet_layer(
    inputs,
    num_filters: int = 16,
    kernel_size: int = 3,
    strides: int = 1,
    activation: str | None = "relu",
    batch_normalization: bool = True,
):
    x = Conv2D(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(1e-4),
    )(inputs)
    if batch_normalization:
        x = BatchNormalization()(x)
    if activation is not None:
        x = Activation(activation)(x)
    return x


def build_resnet_light(
    input_shape: tuple[int, int, int] = (96, 128, 3),
    num_classes: int = 2,
    base_filters: int = 8,
    dropout: float = 0.5,
    spatial_dropout: float = 0.25,
) -> Model:
    inputs = Input(shape=input_shape)
    num_filters = base_filters
    x = resnet_layer(inputs=inputs, num_filters=num_filters)

    for _ in range(2):
        y = resnet_layer(inputs=x, num_filters=num_filters)
        y = SpatialDropout2D(spatial_dropout)(y)
        y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
        y = SpatialDropout2D(spatial_dropout)(y)
        x = tf.keras.layers.add([x, y])
        x = Activation("relu")(x)

    num_filters *= 2
    y = resnet_layer(inputs=x, num_filters=num_filters, strides=2)
    y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
    x = resnet_layer(
        inputs=x,
        num_filters=num_filters,
        kernel_size=1,
        strides=2,
        activation=None,
        batch_normalization=False,
    )
    x = tf.keras.layers.add([x, y])
    x = Activation("relu")(x)

    x = GlobalAveragePooling2D(name="gap_embedding")(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    outputs = Dense(num_classes, activation="softmax", kernel_regularizer=l2(1e-4))(x)

    model = Model(inputs, outputs, name="resnet_light_arousal")
    model.compile(
        optimizer=Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def load_compatible_weights(target_model: Model, source_checkpoint_path: str) -> int:
    """Copies weights from `source_checkpoint_path` into `target_model`, matched
    **positionally by layer type** (the Nth Conv2D in the source goes to the Nth Conv2D
    in the target, and so on for each layer class) rather than by layer name.

    Name-based matching was the first version of this function and it is broken in any
    long-running process: Keras' layer auto-naming is a *global* counter, so the second
    `build_resnet_light()` call in the same interpreter produces names like `conv2d_32`
    instead of `conv2d`, and name lookup silently returns 0 layers copied. That is exactly
    the failure mode a k-fold training loop hits from fold 2 onward — caught by
    `tests/integration/test_checkpoint_transplant.py`, which is why this function looks
    the way it does instead of the simpler name-keyed version.

    Returns the count of layers actually copied, so callers can assert the transplant
    wasn't silently a partial or total no-op.
    """
    source_model = tf.keras.models.load_model(source_checkpoint_path)

    source_by_type: dict[type, list] = {}
    for layer in source_model.layers:
        source_by_type.setdefault(type(layer), []).append(layer)
    cursors = {layer_type: 0 for layer_type in source_by_type}

    copied = 0
    for layer in target_model.layers:
        candidates = source_by_type.get(type(layer))
        if not candidates:
            continue
        cursor = cursors[type(layer)]
        if cursor >= len(candidates):
            continue  # source has fewer layers of this type than the target
        source_layer = candidates[cursor]
        cursors[type(layer)] += 1

        source_weights = source_layer.get_weights()
        target_weights = layer.get_weights()
        if not source_weights:
            continue
        if len(source_weights) != len(target_weights):
            continue
        if any(sw.shape != tw.shape for sw, tw in zip(source_weights, target_weights)):
            continue
        layer.set_weights(source_weights)
        copied += 1
    return copied
