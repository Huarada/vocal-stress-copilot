"""Turns a Grad-CAM heatmap into structured facts the Analyst Agent can actually cite.

The XAI chain had a gap at its last metre: `GradCAMExplainer` computes *which region of
the spectrogram drove the score*, `EvidenceService` saves it as a PNG, and the dashboard
shows it to a human — but the Analyst Agent, whose entire job is to verbalise the
explanation, received only a **file path it cannot open**. Asked "what impacted the high
arousal in turn 4?", it correctly answered that it didn't have enough information,
because it genuinely didn't: the one artifact that explains the score was never given to
it in a form it could read.

This module closes that gap. The heatmap is a (n_mels x n_frames) array; its rows are
mel bins (→ frequency) and its columns are time frames (→ position in the clip), so
"where the model looked" is directly derivable as numbers.

The third output is the one that matters most for this project: `attention_on_padding`.
Clips shorter than the spectrogram extractor's padding threshold get zero-padded before
analysis (ADR-025), and if the model's attention falls on that fabricated silence, the
score is an artifact rather than a reading of the speaker — which is exactly the failure
mode short turns kept exhibiting. That's now a measurable, reportable fact instead of an
inference.

Pure functions, no I/O — testable without a model or audio.
"""
from __future__ import annotations

import numpy as np


def _mel_bin_to_hz(bin_index: float, n_mels: int, fmin_hz: float, fmax_hz: float) -> float:
    """Approximate centre frequency of a mel bin, using the standard HTK mel scale
    librosa uses. Kept local rather than importing librosa so this module stays a pure,
    dependency-light application-layer function."""
    def hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min, mel_max = hz_to_mel(fmin_hz), hz_to_mel(fmax_hz)
    fraction = bin_index / max(n_mels - 1, 1)
    return float(mel_to_hz(mel_min + fraction * (mel_max - mel_min)))


def summarize_gradcam(
    heatmap: np.ndarray,
    fmin_hz: float = 50.0,
    fmax_hz: float = 8000.0,
    real_audio_fraction: float = 1.0,
) -> dict[str, float | list[float]]:
    """`heatmap`: (H, W) float array in [0,1] — H = mel bins (row 0 = lowest frequency),
    W = time frames spanning the whole analysed clip *including any zero-padding*.

    `real_audio_fraction`: how much of the analysed clip is genuine audio rather than
    padding (1.0 = no padding). Used to compute `attention_on_padding`.

    Returns a dict of plain numbers, safe to serialize into the evidence contract and
    to read aloud in an explanation.
    """
    if heatmap.ndim != 2 or heatmap.size == 0:
        raise ValueError(f"heatmap must be a non-empty 2-D array, got shape {heatmap.shape}")

    n_mels, n_frames = heatmap.shape
    total = float(heatmap.sum())

    # --- frequency: which mel band carries the most attention ---
    per_bin = heatmap.mean(axis=1)
    peak_bin = int(np.argmax(per_bin))
    peak_frequency_hz = _mel_bin_to_hz(peak_bin, n_mels, fmin_hz, fmax_hz)

    # Band = the contiguous span around the peak that stays above half the peak value
    # (a half-maximum width, the standard way to describe "where the energy is").
    half = per_bin[peak_bin] / 2.0
    low_bin = peak_bin
    while low_bin > 0 and per_bin[low_bin - 1] >= half:
        low_bin -= 1
    high_bin = peak_bin
    while high_bin < n_mels - 1 and per_bin[high_bin + 1] >= half:
        high_bin += 1

    # --- time: where in the clip the attention sits ---
    per_frame = heatmap.mean(axis=0)
    peak_time_fraction = float(np.argmax(per_frame)) / max(n_frames - 1, 1)

    # --- padding: how much attention landed on fabricated silence ---
    real_frames = max(1, int(round(n_frames * max(0.0, min(1.0, real_audio_fraction)))))
    padding_mass = float(heatmap[:, real_frames:].sum()) if real_frames < n_frames else 0.0
    attention_on_padding = padding_mass / total if total > 0 else 0.0

    # --- concentration: focused (near 1) vs smeared across everything (near 0) ---
    flat = heatmap.ravel()
    top_decile = int(max(1, flat.size // 10))
    concentration = (
        float(np.sort(flat)[-top_decile:].sum() / total) if total > 0 else 0.0
    )

    return {
        "peak_frequency_hz": round(peak_frequency_hz, 1),
        "band_hz": [
            round(_mel_bin_to_hz(low_bin, n_mels, fmin_hz, fmax_hz), 1),
            round(_mel_bin_to_hz(high_bin, n_mels, fmin_hz, fmax_hz), 1),
        ],
        "peak_time_fraction": round(peak_time_fraction, 3),
        "attention_on_padding": round(attention_on_padding, 3),
        "concentration": round(concentration, 3),
    }


def describe_gradcam(summary: dict) -> str:
    """Renders a `summarize_gradcam` result as one plain-language sentence for the
    Analyst Agent's context — the form in which the explanation finally reaches the
    component that has to explain it."""
    band = summary["band_hz"]
    where = summary["peak_time_fraction"]
    position = "early in" if where < 0.33 else ("the middle of" if where < 0.66 else "late in")

    parts = [
        f"the model's attention concentrated around {summary['peak_frequency_hz']:.0f} Hz "
        f"(band ~{band[0]:.0f}-{band[1]:.0f} Hz), {position} the clip"
    ]
    if summary["attention_on_padding"] >= 0.2:
        parts.append(
            f"WARNING: {summary['attention_on_padding'] * 100:.0f}% of that attention fell on "
            "zero-padded silence rather than real audio — this score is largely an artifact "
            "of the clip being too short, not a reading of the speaker"
        )
    if summary["concentration"] >= 0.5:
        parts.append("attention was tightly focused on that region")
    elif summary["concentration"] <= 0.25:
        parts.append("attention was diffuse across the spectrogram rather than focused")
    return "; ".join(parts)
