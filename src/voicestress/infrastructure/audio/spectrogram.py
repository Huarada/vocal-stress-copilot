"""Narrowband spectrogram extraction — ported from the sarcasm-detection pipeline
(`DetectarSarcasmoDataAgumentationMustardPlusPrecisaoFscore.ipynb`, "CÉLULA SUBSTITUTA
SAFE") with the exact same channel encoding, since that is the tested, working part of
the inherited project (SKILLS.md §3).

Channel encoding (not RGB colors — three different acoustic views stacked as channels):
    R = log-mel spectrogram      (narrowband: 80ms window / 10ms hop -> harmonic detail)
    G = first-order delta MFCC   (spectral envelope *rate of change*)
    B = second-order delta MFCC  (spectral envelope *acceleration*)

Each channel is globally z-scored and clipped to [-3, 3] -> [0, 1] independently
(`_zscore_global`), which is what makes the CNN's job tractable — without per-channel
normalization the log-mel channel dominates purely by having larger raw magnitude.

Implements `voicestress.domain.ports.SpectrogramExtractorPort`.
"""
from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
from PIL import Image

DEFAULT_N_MELS = 96
DEFAULT_N_MFCC = 20
DEFAULT_WIN_MS = 80
DEFAULT_HOP_MS = 10
DEFAULT_FMIN_HZ = 50
MIN_DYNAMIC_RANGE_DB = 3.0
MIN_CLIP_SECONDS = 1.0  # shorter clips are zero-padded, matching the original notebook


class FlatSpectrogramError(ValueError):
    """Raised when the log-mel dynamic range is too low to be informative (a.k.a. the
    original notebook's 'tapete cinza' / gray-carpet guard)."""


def _zscore_global(matrix: np.ndarray) -> np.ndarray:
    z = (matrix - np.mean(matrix)) / (np.std(matrix) + 1e-8)
    z = np.clip(z, -3, 3)
    return (z + 3) / 6.0  # -> [0, 1]


def _resize_2d_linear(mat: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    h, w = mat.shape
    x_old, x_new = np.linspace(0.0, 1.0, w), np.linspace(0.0, 1.0, new_w)
    tmp = np.empty((h, new_w), dtype=np.float32)
    for i in range(h):
        tmp[i, :] = np.interp(x_new, x_old, mat[i, :])
    y_old, y_new = np.linspace(0.0, 1.0, h), np.linspace(0.0, 1.0, new_h)
    out = np.empty((new_h, new_w), dtype=np.float32)
    for j in range(new_w):
        out[:, j] = np.interp(y_new, y_old, tmp[:, j])
    return out


def _log_mel_db(
    y: np.ndarray,
    sr: int,
    n_mels: int,
    n_fft: int,
    hop: int,
    fmin: int = DEFAULT_FMIN_HZ,
) -> np.ndarray:
    fmax = sr // 2
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0
    )
    S = np.maximum(S, 1e-12)
    log_s = librosa.power_to_db(S, ref=1.0 if np.max(S) <= 1e-10 else np.max(S))

    p2, p98 = np.percentile(log_s, 2), np.percentile(log_s, 98)
    if (p98 - p2) >= 5.0:
        log_s = np.clip(log_s, p2, p98)
    else:
        log_s = librosa.power_to_db(S, ref=1.0)
    return log_s


@dataclass(frozen=True, slots=True)
class NarrowbandSpectrogramExtractor:
    """`voicestress.domain.ports.SpectrogramExtractorPort` implementation."""

    n_mels: int = DEFAULT_N_MELS
    n_mfcc: int = DEFAULT_N_MFCC
    win_ms: int = DEFAULT_WIN_MS
    hop_ms: int = DEFAULT_HOP_MS
    output_height: int = 96
    output_width: int = 128
    # Satisfies SpectrogramExtractorPort.min_clip_seconds — clips shorter than this get
    # zero-padded in `extract_raw_channels`, and EvidenceService needs to know that to
    # compute how much of an analysed clip is fabricated silence (ADR-032).
    min_clip_seconds: float = MIN_CLIP_SECONDS

    def extract_raw_channels(self, samples: np.ndarray, sample_rate_hz: int) -> np.ndarray:
        """Returns the HxWx3 float32 [0,1] image at native (n_mels x n_frames)
        resolution, before the final resize. Exposed separately so XAI/Grad-CAM can
        map back to real mel-bin / time-frame coordinates without a second resize."""
        y = samples
        min_len = int(sample_rate_hz * MIN_CLIP_SECONDS)
        if len(y) < min_len:
            y = np.pad(y, (0, min_len - len(y)))

        n_fft = int(sample_rate_hz * (self.win_ms / 1000.0))
        hop = int(sample_rate_hz * (self.hop_ms / 1000.0))

        log_s = _log_mel_db(y, sample_rate_hz, self.n_mels, n_fft, hop)

        mfcc = librosa.feature.mfcc(S=log_s, n_mfcc=self.n_mfcc)
        d1 = librosa.feature.delta(mfcc, order=1)
        d2 = librosa.feature.delta(mfcc, order=2)

        r = _zscore_global(log_s)
        g = _zscore_global(d1)
        b = _zscore_global(d2)

        t = min(r.shape[1], g.shape[1], b.shape[1])
        r, g, b = r[:, :t], g[:, :t], b[:, :t]
        if g.shape[0] != r.shape[0]:
            g = _resize_2d_linear(g, r.shape[0], r.shape[1])
        if b.shape[0] != r.shape[0]:
            b = _resize_2d_linear(b, r.shape[0], r.shape[1])

        dyn_db = float(np.nanmax(log_s) - np.nanmin(log_s))
        if dyn_db < MIN_DYNAMIC_RANGE_DB:
            raise FlatSpectrogramError(
                f"log-mel dynamic range too low ({dyn_db:.2f} dB < {MIN_DYNAMIC_RANGE_DB} dB) "
                "— likely near-silent or corrupted clip."
            )

        return np.stack([r, g, b], axis=-1).astype(np.float32)

    def extract(self, samples: np.ndarray, sample_rate_hz: int) -> np.ndarray:
        """Full pipeline: raw channels -> resize to (output_height, output_width, 3),
        the fixed size the CNN expects. Uses PIL bilinear resize on the uint8-quantized
        image, matching how the original notebook round-tripped through PNG files."""
        raw = self.extract_raw_channels(samples, sample_rate_hz)
        img8 = (np.clip(raw, 0, 1) * 255).astype(np.uint8)
        pil_img = Image.fromarray(img8).resize(
            (self.output_width, self.output_height), Image.BILINEAR
        )
        return (np.asarray(pil_img).astype(np.float32)) / 255.0
