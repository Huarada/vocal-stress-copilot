"""Composition root for the model track: runs the concept -> implementation ->
validation loop described in ARCHITECTURE.md §6.

1. Grouped 5-fold CV on RAVDESS, TWICE — random-init vs. warm-started from the inherited
   CIFAR-10-pretrained checkpoint (ARCHITECTURE.md's corrected warm-start note: it is an
   empirical A/B, not an assumed win, because the domain gap — natural photos vs.
   spectrograms — makes the transfer benefit genuinely uncertain).
2. Whichever variant wins on OOF F1 is retrained once on the full RAVDESS training set.
3. That final model is evaluated cross-corpus on MUStARD++ Arousal (never trained on).
4. Everything lands in artifacts/metrics/latest.json, which
   tests/quality_gates/test_minimum_accuracy_gate.py reads and enforces thresholds against.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from voicestress.application.cross_corpus_eval_service import evaluate as evaluate_cross_corpus
from voicestress.application.training_service import (
    TrainingConfig,
    run_cross_validation,
    train_final_model,
)

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = ROOT / "artifacts" / "manifests"
METRICS_DIR = ROOT / "artifacts" / "metrics"
MODELS_DIR = ROOT / "artifacts" / "models"
# Sibling of this repo, gitignored (ARCHITECTURE.md §8) — see build_features.py's
# comment for why this is env-overridable, not a literal personal path (fixed
# 2026-09-06, was leaking a local username/folder layout into committed source).
CHECKPOINT = (
    Path(os.environ.get("VOICESTRESS_HACKATHON_ROOT", ROOT.parent))
    / "sarcasmoVoz"
    / "DetectarSarcasmoPorVoz-main"
    / "resnet1_prelayer1.keras"
)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    ravdess = np.load(MANIFEST_DIR / "ravdess_features.npz", allow_pickle=True)
    X, y, groups = ravdess["X"], ravdess["y"], ravdess["groups"]
    print(f"RAVDESS: X={X.shape} y_high_frac={y.mean():.3f} n_actors={len(set(groups))}")

    mustard = np.load(MANIFEST_DIR / "mustard_features.npz", allow_pickle=True)
    X_mustard, y_mustard = mustard["X"], mustard["y"]
    print(f"MUStARD++: X={X_mustard.shape} y_high_frac={y_mustard.mean():.3f}")

    variants = {
        "random_init": TrainingConfig(warm_start_checkpoint=None),
        "warm_started": TrainingConfig(warm_start_checkpoint=str(CHECKPOINT)),
    }

    report: dict = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "variants": {}}

    for name, config in variants.items():
        print(f"\n=== Cross-validation: {name} ===")
        t0 = time.time()
        cv_result = run_cross_validation(X, y, groups, config)
        elapsed = time.time() - t0
        print(
            f"  OOF: acc={cv_result.oof_accuracy:.3f} f1={cv_result.oof_f1:.3f} "
            f"({elapsed:.1f}s, transplanted={cv_result.layers_transplanted})"
        )
        report["variants"][name] = cv_result.summary()
        report["variants"][name]["cv_wall_time_s"] = elapsed

    winner = max(report["variants"], key=lambda k: report["variants"][k]["oof_f1"])
    print(f"\n=== Winner: {winner} (by OOF F1) ===")
    report["winner"] = winner

    winner_config = variants[winner]
    print("Training final model on full RAVDESS training set...")
    final_model = train_final_model(X, y, winner_config)
    final_model_path = MODELS_DIR / "arousal_resnet_light.keras"
    final_model.save(final_model_path)
    print(f"Saved {final_model_path}")

    print("\n=== Cross-corpus evaluation: RAVDESS -> MUStARD++ Arousal ===")
    cross_corpus = evaluate_cross_corpus(final_model, X_mustard, y_mustard)
    print(json.dumps(cross_corpus.summary(), indent=2))
    report["cross_corpus"] = cross_corpus.summary()
    report["final_model_variant"] = winner
    report["final_model_path"] = str(final_model_path)

    out_path = METRICS_DIR / "latest.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
