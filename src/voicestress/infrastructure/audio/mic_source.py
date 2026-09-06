"""Real microphone `AudioSource` (application/live_interview_runner.py's `AudioSource`
Protocol) backed by `sounddevice`. Moved out of scripts/run_interview.py (2026-09-05,
ADR-041) when the runner itself became audio-source-agnostic — this is the one adapter
that genuinely needs local mic hardware and cannot be exercised in a unit test; the
websocket-backed adapter in web/backend.py is its counterpart for a browser client.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import sounddevice as sd


class MicAudioSource:
    """Starts a real `sounddevice.InputStream` and calls `feed` with each mono int16
    PCM chunk as it arrives, on sounddevice's own audio callback thread."""

    def __init__(self, sample_rate_hz: int, chunk_samples: int):
        self._sample_rate_hz = sample_rate_hz
        self._chunk_samples = chunk_samples
        self._stream: sd.InputStream | None = None

    def start(self, feed: Callable[[np.ndarray], None]) -> None:
        def _callback(indata, frames, time_info, status) -> None:
            if status:
                print(f"[mic status] {status}")
            feed(indata[:, 0].copy())

        self._stream = sd.InputStream(
            samplerate=self._sample_rate_hz,
            channels=1,
            dtype="int16",
            blocksize=self._chunk_samples,
            callback=_callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
