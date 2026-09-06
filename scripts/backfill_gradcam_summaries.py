"""Backfills `gradcam_summary`/`gradcam_description` into sessions recorded before
ADR-032 added them, by re-deriving the summary from each turn's already-saved Grad-CAM
PNG.

Why this exists: without it, older sessions render "model attention: NOT AVAILABLE" in
the Analyst Agent's context — honest, but it means every pre-ADR-032 session (including
every session recorded during this project's live testing) can't be explained properly
in a demo. The heatmap was saved as an image at the time, so the information isn't lost,
just stored in a form the summary functions never saw.

Idempotent: turns that already carry a summary are left alone.

Usage:
    python scripts/backfill_gradcam_summaries.py            # all sessions
    python scripts/backfill_gradcam_summaries.py --dry-run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from voicestress.application.gradcam_summary import describe_gradcam, summarize_gradcam

ROOT = Path(__file__).resolve().parent.parent
SESSIONS_DIR = ROOT / "artifacts" / "sessions"
PADDING_THRESHOLD_MS = 1000  # matches NarrowbandSpectrogramExtractor.min_clip_seconds


def _real_audio_fraction(turn: dict) -> float:
    duration_ms = turn["t_end_ms"] - turn["t_start_ms"]
    return min(1.0, duration_ms / PADDING_THRESHOLD_MS) if PADDING_THRESHOLD_MS else 1.0


def backfill_session(path: Path, dry_run: bool) -> tuple[int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    filled = skipped = 0

    for turn in data.get("turns", []):
        xai = turn.setdefault("xai", {})
        if xai.get("gradcam_description"):
            skipped += 1
            continue

        gradcam_png = xai.get("gradcam_png")
        if not gradcam_png or not Path(gradcam_png).exists():
            skipped += 1
            continue

        # The saved PNG is the heatmap quantized to uint8; /255 recovers the [0,1]
        # array summarize_gradcam expects. Quantization costs precision, not meaning.
        heatmap = np.asarray(Image.open(gradcam_png).convert("L"), dtype=np.float32) / 255.0
        summary = summarize_gradcam(heatmap, real_audio_fraction=_real_audio_fraction(turn))

        xai["gradcam_summary"] = summary
        xai["gradcam_description"] = describe_gradcam(summary)
        filled += 1

    if filled and not dry_run:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    return filled, skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not SESSIONS_DIR.exists():
        raise SystemExit(f"{SESSIONS_DIR} not found")

    total_filled = 0
    for path in sorted(SESSIONS_DIR.glob("*.json")):
        filled, skipped = backfill_session(path, args.dry_run)
        total_filled += filled
        status = "would fill" if args.dry_run else "filled"
        print(f"{path.name}: {status} {filled}, skipped {skipped}")

    print(f"\n{'Would backfill' if args.dry_run else 'Backfilled'} {total_filled} turns.")


if __name__ == "__main__":
    main()
