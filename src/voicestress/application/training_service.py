"""Training use-case: grouped k-fold cross-validation for the arousal classifier.

This is the application layer — it orchestrates the domain concept ("train an arousal
classifier and report honest, leak-free metrics") using the concrete
`build_resnet_light` / `load_compatible_weights` infrastructure. It knows nothing about
where the features came from (RAVDESS vs. anything else) — it takes arrays in.

SKILLS.md Hard Rule #4: grouping is not optional. `groups` must carry true speaker
identity or these metrics are meaningless.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from voicestress.infrastructure.models.resnet_light import (
    build_resnet_light,
    load_compatible_weights,
)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    n_splits: int = 5
    epochs: int = 60
    batch_size: int = 16
    early_stopping_patience: int = 10
    random_state: int = 42
    warm_start_checkpoint: str | None = None
    input_shape: tuple[int, int, int] = (96, 128, 3)


@dataclass(slots=True)
class FoldResult:
    fold: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    n_train: int
    n_val: int
    confusion: list[list[int]]


@dataclass(slots=True)
class CrossValidationResult:
    config: TrainingConfig
    warm_started: bool
    layers_transplanted: int
    fold_results: list[FoldResult] = field(default_factory=list)
    oof_predictions: np.ndarray | None = None
    oof_accuracy: float = 0.0
    oof_precision: float = 0.0
    oof_recall: float = 0.0
    oof_f1: float = 0.0
    oof_confusion: list[list[int]] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "warm_started": self.warm_started,
            "layers_transplanted": self.layers_transplanted,
            "n_splits": self.config.n_splits,
            "fold_accuracy_mean": float(np.mean([f.accuracy for f in self.fold_results])),
            "fold_f1_mean": float(np.mean([f.f1 for f in self.fold_results])),
            "oof_accuracy": self.oof_accuracy,
            "oof_precision": self.oof_precision,
            "oof_recall": self.oof_recall,
            "oof_f1": self.oof_f1,
            "oof_confusion": self.oof_confusion,
            "per_fold": [
                {
                    "fold": f.fold,
                    "accuracy": f.accuracy,
                    "precision": f.precision,
                    "recall": f.recall,
                    "f1": f.f1,
                    "n_train": f.n_train,
                    "n_val": f.n_val,
                }
                for f in self.fold_results
            ],
        }


def _build_model(config: TrainingConfig):
    model = build_resnet_light(input_shape=config.input_shape, num_classes=2)
    transplanted = 0
    if config.warm_start_checkpoint:
        weighted_layers = [l for l in model.layers if l.get_weights()]
        transplanted = load_compatible_weights(model, config.warm_start_checkpoint)
        if transplanted == 0:
            raise RuntimeError(
                "warm_start_checkpoint given but 0 layers transplanted — "
                "the checkpoint transplant silently failed; do not train believing "
                "it warm-started. See tests/integration/test_checkpoint_transplant.py."
            )
        if transplanted < len(weighted_layers):
            # Partial transplant is suspicious but not necessarily wrong (e.g. output
            # head legitimately differs) — surface it instead of hiding it.
            print(
                f"  [warn] only {transplanted}/{len(weighted_layers)} weighted layers "
                "transplanted from checkpoint"
            )
    return model, transplanted


def run_cross_validation(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, config: TrainingConfig
) -> CrossValidationResult:
    """`X`: (N, H, W, 3) float32 in [0,1]. `y`: (N,) int in {0,1}. `groups`: (N,) speaker
    ids — StratifiedGroupKFold guarantees no speaker appears in both train and val for
    any fold (SKILLS.md Hard Rule #4)."""
    cv = StratifiedGroupKFold(
        n_splits=config.n_splits, shuffle=True, random_state=config.random_state
    )

    y_cat = to_categorical(y, num_classes=2)
    oof_pred = np.zeros((len(X), 2), dtype=np.float32)

    result = CrossValidationResult(
        config=config, warm_started=bool(config.warm_start_checkpoint), layers_transplanted=0
    )

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), start=1):
        train_groups, val_groups = set(groups[train_idx]), set(groups[val_idx])
        overlap = train_groups & val_groups
        if overlap:
            raise RuntimeError(
                f"Fold {fold}: speaker leakage detected — groups {overlap} appear in "
                "both train and val. StratifiedGroupKFold should make this impossible; "
                "if this fires, the `groups` array itself is wrong upstream."
            )

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_cat[train_idx], y_cat[val_idx]
        y_train_int, y_val_int = y[train_idx], y[val_idx]

        class_weights = compute_class_weight(
            "balanced", classes=np.unique(y_train_int), y=y_train_int
        )
        class_weight_dict = dict(enumerate(class_weights))

        model, transplanted = _build_model(config)
        result.layers_transplanted = transplanted

        callbacks = [
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
            EarlyStopping(
                monitor="val_loss",
                patience=config.early_stopping_patience,
                restore_best_weights=True,
            ),
        ]
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=config.epochs,
            batch_size=config.batch_size,
            verbose=0,
            class_weight=class_weight_dict,
            callbacks=callbacks,
        )

        val_probs = model.predict(X_val, verbose=0)
        oof_pred[val_idx] = val_probs
        val_pred_labels = val_probs.argmax(axis=1)

        fold_result = FoldResult(
            fold=fold,
            accuracy=float(accuracy_score(y_val_int, val_pred_labels)),
            precision=float(precision_score(y_val_int, val_pred_labels, zero_division=0)),
            recall=float(recall_score(y_val_int, val_pred_labels, zero_division=0)),
            f1=float(f1_score(y_val_int, val_pred_labels, zero_division=0)),
            n_train=len(train_idx),
            n_val=len(val_idx),
            confusion=confusion_matrix(y_val_int, val_pred_labels).tolist(),
        )
        result.fold_results.append(fold_result)
        print(
            f"  fold {fold}: acc={fold_result.accuracy:.3f} f1={fold_result.f1:.3f} "
            f"(n_train={fold_result.n_train}, n_val={fold_result.n_val})"
        )

    oof_labels = oof_pred.argmax(axis=1)
    result.oof_predictions = oof_pred
    result.oof_accuracy = float(accuracy_score(y, oof_labels))
    result.oof_precision = float(precision_score(y, oof_labels, zero_division=0))
    result.oof_recall = float(recall_score(y, oof_labels, zero_division=0))
    result.oof_f1 = float(f1_score(y, oof_labels, zero_division=0))
    result.oof_confusion = confusion_matrix(y, oof_labels).tolist()
    return result


def train_final_model(X: np.ndarray, y: np.ndarray, config: TrainingConfig):
    """Trains one model on *all* available data (no held-out split) — used only after
    cross-validation has already produced trustworthy generalization estimates, to
    produce the artifact that actually ships. Returns the fitted Keras model."""
    y_cat = to_categorical(y, num_classes=2)
    class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
    class_weight_dict = dict(enumerate(class_weights))

    model, _ = _build_model(config)
    callbacks = [
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
        EarlyStopping(monitor="loss", patience=config.early_stopping_patience, restore_best_weights=True),
    ]
    model.fit(
        X,
        y_cat,
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=0,
        class_weight=class_weight_dict,
        callbacks=callbacks,
    )
    return model
