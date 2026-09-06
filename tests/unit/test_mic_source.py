"""MicAudioSource, extracted from scripts/run_interview.py's inline sounddevice wiring
(ADR-041). sounddevice.InputStream itself needs real hardware, so this stubs the class
and asserts the wiring around it — the mono-channel extraction and stream parameters —
which is exactly the kind of thing that silently breaks in an extraction (wrong
blocksize, wrong dtype, forgetting the `[:, 0]` mono slice) without a symptom until a
real mic session sends garbled audio.
"""
import numpy as np

from voicestress.infrastructure.audio.mic_source import MicAudioSource


class FakeInputStream:
    """Records its constructor args; `start()`/`stop()`/`close()` are no-ops, and the
    test drives `callback` directly instead of real audio hardware."""

    instances = []

    def __init__(self, samplerate, channels, dtype, blocksize, callback):
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.blocksize = blocksize
        self.callback = callback
        self.started = False
        self.stopped = False
        self.closed = False
        FakeInputStream.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


def test_start_configures_the_stream_for_mono_pcm16_at_the_given_rate(monkeypatch):
    monkeypatch.setattr(
        "voicestress.infrastructure.audio.mic_source.sd.InputStream", FakeInputStream
    )
    source = MicAudioSource(sample_rate_hz=24_000, chunk_samples=2_400)

    source.start(feed=lambda chunk: None)

    stream = FakeInputStream.instances[-1]
    assert stream.samplerate == 24_000
    assert stream.channels == 1
    assert stream.dtype == "int16"
    assert stream.blocksize == 2_400
    assert stream.started is True


def test_callback_extracts_the_mono_channel_and_forwards_a_copy(monkeypatch):
    """sounddevice hands the callback a (frames, channels) array even for a mono
    stream. Forgetting `[:, 0]` would forward a 2-D array the rest of the pipeline
    (SyncBus.feed_audio) does not expect."""
    monkeypatch.setattr(
        "voicestress.infrastructure.audio.mic_source.sd.InputStream", FakeInputStream
    )
    received = []
    source = MicAudioSource(sample_rate_hz=24_000, chunk_samples=4)
    source.start(feed=received.append)
    stream = FakeInputStream.instances[-1]

    indata = np.array([[1], [2], [3], [4]], dtype=np.int16)
    stream.callback(indata, frames=4, time_info=None, status=None)

    assert len(received) == 1
    assert received[0].ndim == 1
    assert list(received[0]) == [1, 2, 3, 4]


def test_callback_forwards_a_copy_not_a_view(monkeypatch):
    """sounddevice reuses its internal buffer across callbacks; forwarding a view
    instead of a copy would let a later chunk silently overwrite an earlier one still
    sitting in the audio queue."""
    monkeypatch.setattr(
        "voicestress.infrastructure.audio.mic_source.sd.InputStream", FakeInputStream
    )
    received = []
    source = MicAudioSource(sample_rate_hz=24_000, chunk_samples=2)
    source.start(feed=received.append)
    stream = FakeInputStream.instances[-1]

    indata = np.array([[10], [20]], dtype=np.int16)
    stream.callback(indata, frames=2, time_info=None, status=None)
    indata[:, 0] = [99, 99]  # mutate the source buffer after the callback returns

    assert list(received[0]) == [10, 20]


def test_stop_stops_and_closes_the_stream(monkeypatch):
    monkeypatch.setattr(
        "voicestress.infrastructure.audio.mic_source.sd.InputStream", FakeInputStream
    )
    source = MicAudioSource(sample_rate_hz=24_000, chunk_samples=2_400)
    source.start(feed=lambda chunk: None)
    stream = FakeInputStream.instances[-1]

    source.stop()

    assert stream.stopped is True
    assert stream.closed is True


def test_stop_before_start_does_not_raise():
    MicAudioSource(sample_rate_hz=24_000, chunk_samples=2_400).stop()
