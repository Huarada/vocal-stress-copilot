"""Composition root: builds cached feature arrays (spectrogram images + labels + group
ids) for both RAVDESS (training) and MUStARD++ (cross-corpus eval), from the manifests
produced by the dataset catalogs. Output goes to artifacts/manifests/*.npz — gitignored,
regenerable from raw audio at any time (ARCHITECTURE.md §8).

Usage:
    python scripts/build_features.py --corpus ravdess [--limit N]
    python scripts/build_features.py --corpus mustard [--limit N]
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from voicestress.infrastructure.audio.resampling import (
    ANALYSIS_SAMPLE_RATE_HZ,
    peak_normalize,
    resample_to_analysis_rate,
)
from voicestress.infrastructure.audio.spectrogram import (
    FlatSpectrogramError,
    NarrowbandSpectrogramExtractor,
)
from voicestress.infrastructure.datasets.mustard_catalog import (
    eval_subset as mustard_eval_subset,
    existing_only,
    load_manifest as load_mustard_manifest,
)
from voicestress.infrastructure.datasets.ravdess_catalog import (
    build_manifest as build_ravdess_manifest,
    training_subset as ravdess_training_subset,
)

# The two source datasets are gitignored siblings of this repo, not inside it
# (ARCHITECTURE.md §8) — never hardcode a personal path to them in committed source
# (found and fixed 2026-09-06: the literal path used to be here, leaking a local
# username and folder layout into a public repo). Defaults to the sibling-directory
# layout this project's own machine uses; override with VOICESTRESS_HACKATHON_ROOT if
# your copy of the datasets lives somewhere else.
_HACKATHON_ROOT = Path(
    os.environ.get("VOICESTRESS_HACKATHON_ROOT", Path(__file__).resolve().parents[2])
)
RAVDESS_ROOT = _HACKATHON_ROOT / "databaseAudio" / "audioSpeech"
MUSTARD_CSV = (
    _HACKATHON_ROOT
    / "sarcasmoVoz"
    / "DetectarSarcasmoPorVoz-main"
    / "audio_extracted_16k"
    / "MUStARD_plusplus_with_audio_16k.csv"
)
MUSTARD_AUDIO_ROOT = (
    _HACKATHON_ROOT / "sarcasmoVoz" / "DetectarSarcasmoPorVoz-main" / "audio_extracted_16k"
)
OUT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "manifests"

LABEL_TO_INT = {"low": 0, "high": 1}


def _extract_all(paths: list[str], extractor: NarrowbandSpectrogramExtractor):
    images, ok_mask = [], []
    n_fail = 0
    t0 = time.time()
    for i, path in enumerate(paths):
        try:
            y, sr = sf.read(path, always_2d=False)
            y = peak_normalize(resample_to_analysis_rate(y, sr))
            img = extractor.extract(y, ANALYSIS_SAMPLE_RATE_HZ)
            images.append(img)
            ok_mask.append(True)
        except (FlatSpectrogramError, Exception) as e:  # noqa: BLE001 - log & continue
            n_fail += 1
            ok_mask.append(False)
        if (i + 1) % 100 == 0 or (i + 1) == len(paths):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{len(paths)}] {rate:.1f} clips/s, {n_fail} failed so far")
    return np.stack(images).astype(np.float32) if images else np.empty((0,)), np.array(ok_mask)


def build_ravdess(limit: int | None) -> None:
    manifest = build_ravdess_manifest(RAVDESS_ROOT)
    manifest = ravdess_training_subset(manifest)
    if limit:
        manifest = manifest.sample(n=min(limit, len(manifest)), random_state=42).reset_index(drop=True)

    print(f"RAVDESS training subset: {len(manifest)} clips")
    extractor = NarrowbandSpectrogramExtractor()
    X, ok_mask = _extract_all(manifest["path"].tolist(), extractor)
    manifest = manifest[ok_mask].reset_index(drop=True)

    y = manifest["arousal_label"].map(LABEL_TO_INT).to_numpy()
    groups = manifest["actor_id"].to_numpy()

    out_path = OUT_DIR / "ravdess_features.npz"
    np.savez_compressed(out_path, X=X, y=y, groups=groups)
    print(f"Saved {out_path}  X={X.shape} y={y.shape} unique_groups={len(set(groups))}")


def build_mustard(limit: int | None) -> None:
    manifest = load_mustard_manifest(MUSTARD_CSV, audio_root=MUSTARD_AUDIO_ROOT)
    manifest = mustard_eval_subset(manifest)
    manifest = existing_only(manifest)
    if limit:
        manifest = manifest.sample(n=min(limit, len(manifest)), random_state=42).reset_index(drop=True)

    print(f"MUStARD++ eval subset: {len(manifest)} clips")
    extractor = NarrowbandSpectrogramExtractor()
    X, ok_mask = _extract_all(manifest["path"].tolist(), extractor)
    manifest = manifest[ok_mask].reset_index(drop=True)

    y = manifest["arousal_label"].map(LABEL_TO_INT).to_numpy()
    groups = manifest["speaker"].to_numpy()

    out_path = OUT_DIR / "mustard_features.npz"
    np.savez_compressed(out_path, X=X, y=y, groups=groups)
    print(f"Saved {out_path}  X={X.shape} y={y.shape} unique_groups={len(set(groups))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=["ravdess", "mustard"], required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.corpus == "ravdess":
        build_ravdess(args.limit)
    else:
        build_mustard(args.limit)
