"""MUStARD++ manifest loading — used exclusively as the cross-corpus evaluation set
(ARCHITECTURE.md §6). Never trained on: it exists to answer "does a model trained on
RAVDESS's acted studio speech generalize to natural TV dialogue?" honestly.

The CSV ships an `Arousal` column on a ~3-9 rating scale, peaked hard at 6-7 (see the
value_counts this module's tests pin down). We binarize with a deliberate dead zone
around the peak rather than a median split, which on this distribution would produce a
near-degenerate class balance:

    LOW  = Arousal <= LOW_MAX   (<=5)
    HIGH = Arousal >= HIGH_MIN  (>=8)
    dropped: 6 <= Arousal <= 7  (the ambiguous majority band)

This trades away recall on middling cases for label reliability on the ones we keep —
standard practice when binarizing a Likert-type target, and it is documented here instead
of silently baked into a threshold nobody could reconstruct later.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from voicestress.domain.value_objects import ArousalLabel

LOW_AROUSAL_MAX = 5.0
HIGH_AROUSAL_MIN = 8.0


def _binarize(arousal: float) -> str | None:
    if arousal <= LOW_AROUSAL_MAX:
        return ArousalLabel.LOW.value
    if arousal >= HIGH_AROUSAL_MIN:
        return ArousalLabel.HIGH.value
    return None


def load_manifest(csv_path: Path, audio_root: Path | None = None) -> pd.DataFrame:
    """Returns rows with: path, speaker (SPEAKER column, used as the CV/eval group key),
    arousal_raw, arousal_label ('low'/'high'/None for the dropped middle band).

    `audio_root` lets the caller repoint `audio_path` (which is stored relative to
    wherever the original notebook's cwd was) at the actual location on this machine.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["audio_path", "Arousal"]).copy()

    if audio_root is not None:
        audio_root = Path(audio_root)
        df["path"] = df["audio_path"].apply(
            lambda rel: str(audio_root / Path(str(rel)).name)
        )
    else:
        df["path"] = df["audio_path"].astype(str)

    df["arousal_raw"] = df["Arousal"].astype(float)
    df["arousal_label"] = df["arousal_raw"].apply(_binarize)
    df["speaker"] = df["SPEAKER"].astype(str)

    keep_cols = [
        "path",
        "speaker",
        "KEY",
        "SENTENCE",
        "arousal_raw",
        "arousal_label",
        "Sarcasm",
    ]
    return df[keep_cols].reset_index(drop=True)


def eval_subset(manifest: pd.DataFrame) -> pd.DataFrame:
    """Drops the dropped-middle-band rows — this is the frame cross-corpus eval runs on."""
    return manifest[manifest["arousal_label"].notna()].reset_index(drop=True)


def existing_only(manifest: pd.DataFrame) -> pd.DataFrame:
    mask = manifest["path"].apply(lambda p: Path(p).exists())
    return manifest[mask].reset_index(drop=True)
