from dataclasses import dataclass, field

from voicestress.application.transcript_pairing import TranscriptPairer


@dataclass(frozen=True)
class _RichPayload:
    """Stand-in for run_interview.py's real TranscriptPayload (text + words) —
    exercises the generic P type parameter without depending on that script."""

    text: str = ""
    words: tuple = field(default_factory=tuple)


def test_generic_payload_type_pairs_correctly():
    pairer: TranscriptPairer[str, _RichPayload] = TranscriptPairer(
        default_payload_factory=_RichPayload
    )
    assert pairer.submit_turn("t1") is None
    payload = _RichPayload(text="hi there", words=({"text": "hi", "confidence": 0.9},))
    turn, result_payload = pairer.submit_transcript(payload)
    assert turn == "t1"
    assert result_payload.text == "hi there"
    assert result_payload.words[0]["confidence"] == 0.9


def test_generic_payload_type_flushes_with_default_factory():
    pairer: TranscriptPairer[str, _RichPayload] = TranscriptPairer(
        default_payload_factory=_RichPayload
    )
    pairer.submit_turn("t1")
    flushed = pairer.flush_all()
    assert flushed == [("t1", _RichPayload())]


def test_transcript_arriving_after_next_turn_still_pairs_with_correct_turn():
    """Reproduces the real 2026-09-05 bug directly: turn A's audio finishes, turn B's
    audio also finishes, and ONLY THEN does turn A's transcript arrive. FIFO pairing
    must attach it to A, not B."""
    pairer = TranscriptPairer[str, str]()

    assert pairer.submit_turn("turn_A") is None  # queued, no transcript yet
    assert pairer.submit_turn("turn_B") is None  # queued, still no transcript for A

    # Turn A's transcript finally arrives, after B's audio already ended
    result = pairer.submit_transcript("And I'm telling the truth right now, I think.")
    assert result == ("turn_A", "And I'm telling the truth right now, I think.")

    result = pairer.submit_transcript("Can you hear me?")
    assert result == ("turn_B", "Can you hear me?")


def test_simple_in_order_pairing():
    pairer = TranscriptPairer[str, str]()
    assert pairer.submit_turn("t1") is None
    assert pairer.submit_transcript("hello") == ("t1", "hello")


def test_stray_transcript_with_nothing_queued_returns_none():
    pairer = TranscriptPairer[str, str]()
    assert pairer.submit_transcript("unexpected") is None


def test_queue_overflow_flushes_oldest_with_empty_transcript():
    pairer = TranscriptPairer[str, str](max_pending=2)
    assert pairer.submit_turn("t1") is None
    assert pairer.submit_turn("t2") is None
    # t1's transcript never arrived; a third turn finishing pushes the queue past
    # max_pending, so t1 must be flushed now rather than blocking forever
    flushed = pairer.submit_turn("t3")
    assert flushed == ("t1", "")
    assert pairer.pending_count == 2  # t2, t3 still waiting


def test_flush_all_drains_remaining_queue_with_empty_transcripts():
    pairer = TranscriptPairer[str, str]()
    pairer.submit_turn("t1")
    pairer.submit_turn("t2")
    result = pairer.flush_all()
    assert result == [("t1", ""), ("t2", "")]
    assert pairer.pending_count == 0


def test_pending_count_reflects_queue_state():
    pairer = TranscriptPairer[str, str]()
    assert pairer.pending_count == 0
    pairer.submit_turn("t1")
    assert pairer.pending_count == 1
    pairer.submit_transcript("hi")
    assert pairer.pending_count == 0
