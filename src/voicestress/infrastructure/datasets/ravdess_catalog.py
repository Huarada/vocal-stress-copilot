"""RAVDESS filename parsing and manifest construction.

Filename convention (7 dash-separated fields), per the official Zenodo record
https://zenodo.org/records/1188976:

    modality-vocalChannel-emotion-intensity-statement-repetition-actor.wav

Emotion codes: 01 neutral, 02 calm, 03 happy, 04 sad, 05 angry, 06 fearful,
07 disgust, 08 surprised. Odd actor id = male, even = female.

Arousal binarization (ARCHITECTURE.md §6):
  HIGH = angry, fearful, disgust, surprised
  LOW  = neutral, calm, sad
  EXCLUDED = happy (high arousal + positive valence — would teach the model an
             entangled arousal/valence signal; kept as a held-out probe instead)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from voicestress.domain.value_objects import ArousalLabel

RAVDESS_FILENAME_RE = re.compile(
    r"^(?P<modality>\d{2})-(?P<vocal_channel>\d{2})-(?P<emotion>\d{2})-"
    r"(?P<intensity>\d{2})-(?P<statement>\d{2})-(?P<repetition>\d{2})-"
    r"(?P<actor>\d{2})$"
)

EMOTION_CODE_TO_NAME = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

HIGH_AROUSAL_EMOTIONS = frozenset({"angry", "fearful", "disgust", "surprised"})
LOW_AROUSAL_EMOTIONS = frozenset({"neutral", "calm", "sad"})
EXCLUDED_EMOTIONS = frozenset({"happy"})

# Vocal channel / modality codes we accept for the *speech* manifest.
SPEECH_VOCAL_CHANNEL = "01"
AUDIO_ONLY_MODALITY = "03"


class RavdessFilenameError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RavdessFileInfo:
    path: Path
    modality: str
    vocal_channel: str
    emotion_code: str
    emotion_name: str
    intensity: str
    statement: str
    repetition: str
    actor: str
    is_female: bool

    @property
    def actor_id(self) -> str:
        return f"Actor_{self.actor}"

    @property
    def arousal_label(self) -> ArousalLabel | None:
        """None means 'excluded from binary training' (currently only `happy`)."""
        if self.emotion_name in EXCLUDED_EMOTIONS:
            return None
        if self.emotion_name in HIGH_AROUSAL_EMOTIONS:
            return ArousalLabel.HIGH
        if self.emotion_name in LOW_AROUSAL_EMOTIONS:
            return ArousalLabel.LOW
        raise RavdessFilenameError(f"Unmapped emotion: {self.emotion_name}")


def parse_ravdess_filename(path: Path) -> RavdessFileInfo:
    stem = path.stem
    m = RAVDESS_FILENAME_RE.match(stem)
    if not m:
        raise RavdessFilenameError(f"Filename does not match RAVDESS pattern: {path.name}")
    g = m.groupdict()
    emotion_name = EMOTION_CODE_TO_NAME.get(g["emotion"])
    if emotion_name is None:
        raise RavdessFilenameError(f"Unknown emotion code {g['emotion']!r} in {path.name}")
    actor_num = int(g["actor"])
    return RavdessFileInfo(
        path=path,
        modality=g["modality"],
        vocal_channel=g["vocal_channel"],
        emotion_code=g["emotion"],
        emotion_name=emotion_name,
        intensity=g["intensity"],
        statement=g["statement"],
        repetition=g["repetition"],
        actor=g["actor"],
        is_female=(actor_num % 2 == 0),
    )


def scan_ravdess_speech_dir(root: Path) -> list[RavdessFileInfo]:
    """Walks `root` (expects `Actor_XX/*.wav` layout) and parses every speech file.
    Song files and non-conforming names are skipped with no error — RAVDESS ships both
    audio-only and audio-video zips side by side in some distributions."""
    root = Path(root)
    infos: list[RavdessFileInfo] = []
    for wav_path in sorted(root.rglob("*.wav")):
        try:
            info = parse_ravdess_filename(wav_path)
        except RavdessFilenameError:
            continue
        if info.vocal_channel != SPEECH_VOCAL_CHANNEL:
            continue
        infos.append(info)
    return infos


def build_manifest(root: Path) -> pd.DataFrame:
    """Returns a DataFrame with one row per usable speech clip:
    path, actor_id, emotion_name, intensity, arousal_label ('low'/'high'/None).

    Rows with arousal_label == None (currently: happy) are kept in the frame but should
    be filtered out by callers that build a *training* set — they remain useful as a
    held-out probe (see ARCHITECTURE.md §6).
    """
    infos = scan_ravdess_speech_dir(root)
    if not infos:
        raise RavdessFilenameError(f"No RAVDESS speech .wav files found under {root}")
    rows = []
    for info in infos:
        label = info.arousal_label
        rows.append(
            {
                "path": str(info.path),
                "actor_id": info.actor_id,
                "is_female": info.is_female,
                "emotion_code": info.emotion_code,
                "emotion_name": info.emotion_name,
                "intensity": info.intensity,
                "statement": info.statement,
                "repetition": info.repetition,
                "arousal_label": None if label is None else label.value,
            }
        )
    return pd.DataFrame(rows)


def training_subset(manifest: pd.DataFrame) -> pd.DataFrame:
    """Drops excluded-emotion rows (arousal_label is None). This is the frame that
    actually goes into StratifiedGroupKFold."""
    return manifest[manifest["arousal_label"].notna()].reset_index(drop=True)
