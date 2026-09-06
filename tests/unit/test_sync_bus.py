import numpy as np
import pytest

from voicestress.application.sync_bus import SyncBus, SyncBusError


def test_simple_turn_slices_correct_samples():
    bus = SyncBus(sample_rate_hz=1000)  # 1 sample = 1ms, easy to reason about
    bus.feed_audio(np.zeros(500, dtype=np.float32))  # pre-turn silence

    bus.on_speech_started()
    bus.feed_audio(np.ones(300, dtype=np.float32))  # the turn itself
    turn = bus.on_speech_stopped()

    assert turn.turn_id == "t0001"
    assert len(turn.samples) == 300
    np.testing.assert_array_equal(turn.samples, np.ones(300, dtype=np.float32))
    assert turn.t_start_ms == 500
    assert turn.t_end_ms == 800


def test_turn_can_span_multiple_feed_calls():
    bus = SyncBus(sample_rate_hz=1000)
    bus.on_speech_started()
    bus.feed_audio(np.full(100, 1.0, dtype=np.float32))
    bus.feed_audio(np.full(100, 2.0, dtype=np.float32))
    bus.feed_audio(np.full(100, 3.0, dtype=np.float32))
    turn = bus.on_speech_stopped()

    assert len(turn.samples) == 300
    assert turn.samples[0] == 1.0
    assert turn.samples[100] == 2.0
    assert turn.samples[200] == 3.0


def test_sequential_turns_get_incrementing_ids_and_correct_slices():
    bus = SyncBus(sample_rate_hz=1000)

    bus.on_speech_started()
    bus.feed_audio(np.full(100, 1.0, dtype=np.float32))
    turn1 = bus.on_speech_stopped()

    bus.feed_audio(np.zeros(50, dtype=np.float32))  # inter-turn silence

    bus.on_speech_started()
    bus.feed_audio(np.full(200, 2.0, dtype=np.float32))
    turn2 = bus.on_speech_stopped()

    assert turn1.turn_id == "t0001"
    assert turn2.turn_id == "t0002"
    assert len(turn1.samples) == 100
    assert len(turn2.samples) == 200
    assert (turn1.samples == 1.0).all()
    assert (turn2.samples == 2.0).all()
    assert turn2.t_start_ms == 150  # 100 (turn1) + 50 (silence)
    assert turn2.t_end_ms == 350


def test_double_speech_started_without_stopped_raises():
    bus = SyncBus(sample_rate_hz=1000)
    bus.on_speech_started()
    with pytest.raises(SyncBusError):
        bus.on_speech_started()


def test_speech_stopped_without_started_raises():
    bus = SyncBus(sample_rate_hz=1000)
    with pytest.raises(SyncBusError):
        bus.on_speech_stopped()


def test_short_blip_below_minimum_duration_is_discarded():
    """Reproduces the real 2026-09-05 finding: a 100ms input.speech.started/stopped
    pair (mic click / room noise) must not reach the caller as a real turn."""
    bus = SyncBus(sample_rate_hz=1000, min_turn_duration_ms=300)
    bus.on_speech_started()
    bus.feed_audio(np.ones(100, dtype=np.float32))  # 100ms — below the 300ms floor
    result = bus.on_speech_stopped()

    assert result is None
    assert bus.turns_discarded_too_short == 1
    assert bus.has_open_turn is False  # turn state still closes cleanly


def test_turn_at_or_above_minimum_duration_is_kept():
    bus = SyncBus(sample_rate_hz=1000, min_turn_duration_ms=300)
    bus.on_speech_started()
    bus.feed_audio(np.ones(300, dtype=np.float32))  # exactly at the floor
    result = bus.on_speech_stopped()

    assert result is not None
    assert result.turn_id == "t0001"
    assert bus.turns_discarded_too_short == 0


def test_discarded_blips_do_not_consume_turn_ids():
    bus = SyncBus(sample_rate_hz=1000, min_turn_duration_ms=300)

    bus.on_speech_started()
    bus.feed_audio(np.ones(50, dtype=np.float32))
    assert bus.on_speech_stopped() is None  # blip 1, discarded

    bus.on_speech_started()
    bus.feed_audio(np.ones(50, dtype=np.float32))
    assert bus.on_speech_stopped() is None  # blip 2, discarded

    bus.on_speech_started()
    bus.feed_audio(np.ones(500, dtype=np.float32))
    real_turn = bus.on_speech_stopped()

    assert real_turn.turn_id == "t0001"  # not t0003 — blips don't pollute the sequence
    assert bus.turns_discarded_too_short == 2


def test_default_min_duration_is_zero_and_keeps_everything():
    """Backward-compat guard: existing callers/tests that never set
    min_turn_duration_ms must see identical behavior to before this guard existed."""
    bus = SyncBus(sample_rate_hz=1000)
    bus.on_speech_started()
    bus.feed_audio(np.ones(1, dtype=np.float32))  # 1ms — would be discarded at any real floor
    result = bus.on_speech_stopped()
    assert result is not None
    assert bus.turns_discarded_too_short == 0


def test_many_turns_stay_correct_at_scale():
    """Reproduces the real 2026-09-06 finding at the scale it actually occurred: a
    14-turn live session where turns 10-14 got degenerate/wrong audio due to the old
    O(n^2) re-concatenate-everything `_slice()`. Feeds ~14 turns' worth of chunks (as
    `feed_audio` would receive them, ~100ms each) and asserts every turn — including
    the last ones — still slices exactly the samples it should, with no cross-turn
    bleed from stale buffer state."""
    bus = SyncBus(sample_rate_hz=1000, min_turn_duration_ms=0)
    expected: list[tuple[str, float]] = []

    for turn_num in range(1, 15):
        bus.on_speech_started()
        value = float(turn_num)  # distinct fill value per turn -> easy to verify
        for _ in range(20):  # 20 x 10-sample chunks per turn = 200 samples/turn
            bus.feed_audio(np.full(10, value, dtype=np.float32))
        turn = bus.on_speech_stopped()
        expected.append((turn.turn_id, value))
        assert len(turn.samples) == 200
        assert (turn.samples == value).all(), f"turn {turn_num} got contaminated samples"
        # inter-turn gap, so turns don't sit back-to-back (matches real mic timing)
        bus.feed_audio(np.zeros(15, dtype=np.float32))

    assert [tid for tid, _ in expected] == [f"t{i:04d}" for i in range(1, 15)]


def test_buffer_capacity_grows_amortized_not_per_chunk():
    """Structural proxy for the O(1)-amortized-append property the rewrite depends on:
    capacity should grow in geometric jumps, not track write position 1:1 (which is
    the signature of a fresh full-size allocation on every feed)."""
    bus = SyncBus(sample_rate_hz=1000)
    capacities_seen = set()
    for _ in range(50):
        bus.feed_audio(np.zeros(10, dtype=np.float32))
        capacities_seen.add(len(bus._buffer))
    # 50 feeds of 10 samples = 500 samples written; far fewer than 50 distinct
    # capacities means growth wasn't happening on every single feed call
    assert len(capacities_seen) < 15


def test_has_open_turn_reflects_state():
    bus = SyncBus(sample_rate_hz=1000)
    assert bus.has_open_turn is False
    bus.on_speech_started()
    assert bus.has_open_turn is True
    bus.feed_audio(np.zeros(10, dtype=np.float32))
    bus.on_speech_stopped()
    assert bus.has_open_turn is False
