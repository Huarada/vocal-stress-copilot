"""Cross-corpus evaluation use-case: RAVDESS-trained model -> MUStARD++ Arousal.

This is the number that matters (ARCHITECTURE.md §6 tier 2). In-corpus RAVDESS accuracy
answers "did the model memorize 24 actors"; this answers "does it generalize past acted
studio speech to natural dialogue" — the actual claim the product needs to make.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


@dataclass(slots=True)
class CrossCorpusResult:
    n_samples: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: list[list[int]]
    baseline_majority_class_accuracy: float

    def summary(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "confusion": self.confusion,
            "baseline_majority_class_accuracy": self.baseline_majority_class_accuracy,
            "beats_majority_baseline": self.accuracy > self.baseline_majority_class_accuracy,
        }


def evaluate(model, X: np.ndarray, y: np.ndarray) -> CrossCorpusResult:
    probs = model.predict(X, verbose=0)
    pred = probs.argmax(axis=1)

    majority_class = int(np.round(np.mean(y)))
    majority_baseline = float(np.mean(y == majority_class))

    return CrossCorpusResult(
        n_samples=len(y),
        accuracy=float(accuracy_score(y, pred)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        f1=float(f1_score(y, pred, zero_division=0)),
        confusion=confusion_matrix(y, pred).tolist(),
        baseline_majority_class_accuracy=majority_baseline,
    )
