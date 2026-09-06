import pytest

from voicestress.application.asr_signals import mean_word_confidence, onset_latency_ms


def test_mean_word_confidence_averages_correctly():
    words = [{"confidence": 0.9}, {"confidence": 0.7}, {"confidence": 0.8}]
    assert mean_word_confidence(words) == pytest.approx(0.8)


def test_mean_word_confidence_none_when_no_words():
    assert mean_word_confidence([]) is None


def test_mean_word_confidence_ignores_words_missing_confidence():
    words = [{"confidence": 0.9}, {"text": "uh"}]  # second word has no confidence key
    assert mean_word_confidence(words) == pytest.approx(0.9)


def test_mean_word_confidence_none_when_all_missing_confidence():
    words = [{"text": "uh"}, {"text": "um"}]
    assert mean_word_confidence(words) is None


def test_onset_latency_computes_gap_between_turn_start_and_first_word():
    words = [{"start": 1500, "confidence": 0.9}]
    assert onset_latency_ms(turn_start_ms=1000, words=words) == pytest.approx(500)


def test_onset_latency_uses_only_first_word():
    words = [{"start": 1500}, {"start": 1800}, {"start": 2100}]
    assert onset_latency_ms(turn_start_ms=1000, words=words) == pytest.approx(500)


def test_onset_latency_none_when_no_words():
    assert onset_latency_ms(turn_start_ms=1000, words=[]) is None


def test_onset_latency_none_when_first_word_missing_start():
    words = [{"text": "hello", "confidence": 0.9}]  # no "start" key
    assert onset_latency_ms(turn_start_ms=1000, words=words) is None


def test_onset_latency_none_when_negative():
    """A word starting before the turn's recorded start means the clock-alignment
    assumption didn't hold for this turn — report None, not a nonsensical negative
    latency (same discipline as f0_detected/arousal_score_reliable elsewhere)."""
    words = [{"start": 500}]  # earlier than turn_start_ms
    assert onset_latency_ms(turn_start_ms=1000, words=words) is None


def test_onset_latency_zero_is_valid():
    words = [{"start": 1000}]  # word starts exactly at turn start -> no leading silence
    assert onset_latency_ms(turn_start_ms=1000, words=words) == pytest.approx(0.0)
