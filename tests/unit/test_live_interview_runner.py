"""LiveInterviewRunner, moved out of scripts/run_interview.py (ADR-041) and made
audio-source-agnostic. `run()`'s own live-protocol loop still hardcodes a real
WebsocketTransport.connect(...) — exactly like the pre-refactor script did, and exactly
like `websocket_transport.py`'s own docstring says is the one boundary that genuinely
needs a live ASSEMBLYAI_API_KEY. What's newly testable, and newly added, is everything
below that boundary: turn finalization with the new observer hooks (ADR-040/041), and
the reply.* handlers this pass wired up for the first time.
"""
import asyncio

import pytest

from voicestress.application.live_interview_runner import (
    FLAG_TECHNICAL_ISSUE_SCHEMA,
    LiveInterviewRunner,
    TranscriptPayload,
)
from voicestress.application.sync_bus import PendingTurn
from voicestress.config import AssemblyAISettings
from voicestress.infrastructure.models.keras_arousal_classifier import KerasArousalClassifier
from voicestress.infrastructure.models.resnet_light import build_resnet_light


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def model_path(tmp_path):
    model = build_resnet_light(input_shape=(96, 128, 3), num_classes=2)
    path = tmp_path / "model.keras"
    model.save(str(path))
    return path


@pytest.fixture
def runner(model_path):
    settings = AssemblyAISettings(api_key="fake-key-not-sent-anywhere")
    return LiveInterviewRunner(settings=settings, model_path=model_path)


def _make_runner(model_path, **hooks) -> LiveInterviewRunner:
    settings = AssemblyAISettings(api_key="fake-key-not-sent-anywhere")
    return LiveInterviewRunner(settings=settings, model_path=model_path, **hooks)


# --- feed_audio_chunk: the public replacement for the old sounddevice-only callback --


def test_feed_audio_chunk_enqueues_for_the_audio_pump(runner):
    import numpy as np

    chunk = np.zeros(2400, dtype=np.int16)
    runner.feed_audio_chunk(chunk)
    assert runner._audio_queue.qsize() == 1
    assert np.array_equal(runner._audio_queue.get(), chunk)


# --- _status: single choke point for print + on_status -----------------------------


def test_status_forwards_to_on_status_hook(model_path):
    received = []
    r = _make_runner(model_path, on_status=received.append)
    r._status("hello")
    assert received == ["hello"]


def test_status_does_not_require_a_hook(runner):
    runner._status("no hook registered, must not raise")  # should simply print


# --- turn finalization: the new on_turn_finalized observer --------------------------


def _pending(turn_id="t1", n_samples=24_000 * 2):
    import numpy as np

    return PendingTurn(
        turn_id=turn_id,
        samples=np.random.default_rng(0).uniform(-0.2, 0.2, n_samples).astype(np.float32),
        t_start_ms=0,
        t_end_ms=2000,
    )


def test_finalize_turn_invokes_on_turn_finalized(model_path):
    received = []
    r = _make_runner(model_path, on_turn_finalized=received.append)

    run(r._finalize_turn(_pending(), TranscriptPayload(text="hello there")))

    assert len(received) == 1
    assert received[0].transcript == "hello there"
    assert received[0].turn_id == "t1"


def test_finalize_turn_supports_a_sync_on_turn_finalized_hook(model_path):
    """Callback contract mirrors VoiceAgentSession's EventHandler: sync or async, both
    supported (`result = handler(...); if result is not None: await result`)."""
    calls = []

    def on_turn(evidence):
        calls.append(evidence.turn_id)
        return None  # sync hook: no awaitable returned

    r = _make_runner(model_path, on_turn_finalized=on_turn)
    run(r._finalize_turn(_pending(), TranscriptPayload()))
    assert calls == ["t1"]


def test_first_three_turns_are_baseline_and_calibrate(model_path):
    r = _make_runner(model_path)
    for i in range(3):
        run(r._finalize_turn(_pending(turn_id=f"t{i}"), TranscriptPayload(text="ok")))
    assert r.baseline_ready is True
    assert r.interview_session.turns[0].is_baseline_turn is True
    assert r.interview_session.turns[2].is_baseline_turn is True


def test_fourth_turn_is_scored_not_baseline(model_path):
    r = _make_runner(model_path)
    for i in range(4):
        run(r._finalize_turn(_pending(turn_id=f"t{i}"), TranscriptPayload(text="ok")))
    assert r.interview_session.turns[3].is_baseline_turn is False


# --- the agent's reply, wired for the first time (ADR-040) --------------------------


def test_on_agent_transcript_forwards_text_to_hook(model_path):
    received = []
    r = _make_runner(model_path, on_agent_reply_text=received.append)

    run(r._on_agent_transcript({"type": "transcript.agent", "text": "Tell me about yourself."}))

    assert received == ["Tell me about yourself."]


def test_on_agent_transcript_ignores_empty_text(model_path):
    received = []
    r = _make_runner(model_path, on_agent_reply_text=received.append)
    run(r._on_agent_transcript({"type": "transcript.agent", "text": ""}))
    assert received == []


def test_on_reply_audio_decodes_base64_under_the_confirmed_real_key(model_path):
    """ADR-045: confirmed 2026-09-06 against AssemblyAI's own docs (tool-calling guide
    + 5-minute walkthrough's example handler) that reply.audio's payload lives under
    "data" — not "audio", the original guess ADR-043 found wrong. This is now a known
    fact, not one candidate among several; test_on_reply_audio_decodes_base64_under_
    every_other_candidate_key covers the remaining ones as a safety net."""
    import base64

    received = []
    r = _make_runner(model_path, on_agent_reply_audio=received.append)
    raw_pcm = bytes([7, 7, 7, 200])

    run(r._on_reply_audio({"type": "reply.audio", "data": base64.b64encode(raw_pcm).decode()}))

    assert received == [raw_pcm]


def test_on_reply_audio_decodes_base64_under_the_original_guessed_key(model_path):
    import base64

    received = []
    r = _make_runner(model_path, on_agent_reply_audio=received.append)
    raw_pcm = bytes([10, 20, 30, 255])

    run(r._on_reply_audio({"type": "reply.audio", "audio": base64.b64encode(raw_pcm).decode()}))

    assert received == [raw_pcm]


@pytest.mark.parametrize("key", ["delta", "data", "chunk", "pcm", "audio_delta"])
def test_on_reply_audio_decodes_base64_under_every_other_candidate_key(model_path, key):
    """ADR-043: a real session confirmed the original "audio" guess wrong — every
    reply.audio event missed it, and the resulting fallback dumped a raw multi-
    kilobyte event as a status line, freezing the browser tab. Rather than ship a
    second single guess, _on_reply_audio tries several plausible keys; this pins that
    every one of them actually works, not just whichever is eventually confirmed."""
    import base64

    received = []
    r = _make_runner(model_path, on_agent_reply_audio=received.append)
    raw_pcm = bytes([1, 2, 3, 4, 250])

    run(r._on_reply_audio({"type": "reply.audio", key: base64.b64encode(raw_pcm).decode()}))

    assert received == [raw_pcm]


def test_on_reply_audio_prefers_the_first_matching_candidate_key(model_path):
    """If more than one candidate key is present (shouldn't happen, but nothing
    guarantees it can't), the match must be deterministic, not whichever dict
    iteration order happens to produce."""
    import base64

    received = []
    r = _make_runner(model_path, on_agent_reply_audio=received.append)
    real_pcm = bytes([9, 9, 9])
    decoy_pcm = bytes([1, 1, 1])

    run(
        r._on_reply_audio(
            {
                "type": "reply.audio",
                "audio": base64.b64encode(real_pcm).decode(),  # earlier in the tuple than "delta"
                "delta": base64.b64encode(decoy_pcm).decode(),
            }
        )
    )

    assert received == [real_pcm]


def test_on_reply_audio_logs_only_keys_and_shapes_when_no_candidate_key_matches(model_path):
    """Guards the exact defect ADR-043 fixes: the old fallback interpolated the whole
    raw event (including a giant base64 value) into a status line. This must report
    enough to diagnose the real key WITHOUT ever including a field's actual value —
    only its name and a bounded description of its shape."""
    statuses = []
    r = _make_runner(model_path, on_status=statuses.append)
    huge_value = "A" * 50_000  # simulates the real base64 payload's actual size

    run(r._on_reply_audio({"type": "reply.audio", "mystery_field": huge_value}))

    assert len(statuses) == 1
    assert huge_value not in statuses[0], "the raw payload value leaked into the log"
    assert "mystery_field" in statuses[0]
    assert "len=50000" in statuses[0] or "50000" in statuses[0]


def test_on_reply_audio_logs_instead_of_crashing_on_bad_base64(model_path):
    statuses = []
    r = _make_runner(model_path, on_status=statuses.append)

    run(r._on_reply_audio({"type": "reply.audio", "audio": "not valid base64!!"}))

    assert any("failed to decode" in s for s in statuses)


def test_status_truncates_a_very_long_message(model_path):
    """ADR-043's general fix: whatever produced an oversized message (a mis-keyed
    reply.audio event, or anything else in the future), _status() is the one choke
    point that must never forward it unbounded to print() or a web caller's log."""
    forwarded = []
    r = _make_runner(model_path, on_status=forwarded.append)

    r._status("x" * 10_000)

    assert len(forwarded[0]) < 1000
    assert "truncated" in forwarded[0]


def test_on_reply_started_and_done_do_not_raise(runner):
    run(runner._on_reply_started({"type": "reply.started"}))
    run(runner._on_reply_done({"type": "reply.done"}))


# --- ADR-042: run() must save even when cancelled before its event loop starts -----


class _StallingTransport:
    """A fake AssemblyAI transport whose `send()` (called by `configure()`) never
    returns — lets a test suspend `run()` at that exact point and cancel it there,
    reproducing the real race: a client that sends "end_session" within milliseconds
    of "session_started", before `run()` has even reached its own event loop."""

    def __init__(self):
        self.closed = False

    async def send(self, message: str) -> None:
        await asyncio.Event().wait()  # never set: blocks forever until cancelled

    async def recv(self) -> str:
        await asyncio.Event().wait()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        self.closed = True


class _NullAudioSource:
    def start(self, feed):
        pass

    def stop(self):
        pass


def test_run_saves_the_session_even_when_cancelled_before_the_event_loop_starts(model_path, tmp_path):
    """Reproduces a real bug found live-testing web/live_capture.py (2026-09-05): the
    old `finally` sat only around the `while True: handle_next_event()` loop.
    Cancelling run() while it was still inside `configure()` (awaiting a `send()` that
    never returns) skipped that `finally` entirely — no flush, no save — while the web
    endpoint unconditionally reported `session_saved` anyway. A file must now exist
    regardless of when the cancellation lands."""
    import voicestress.application.live_interview_runner as runner_module

    monkeypatch_dir = tmp_path / "sessions"
    monkeypatch_dir.mkdir()
    original_sessions_dir = runner_module.SESSIONS_DIR
    runner_module.SESSIONS_DIR = monkeypatch_dir
    try:
        r = _make_runner(model_path)

        async def connect():
            return _StallingTransport()

        async def scenario():
            task = asyncio.create_task(
                r.run(audio_source=_NullAudioSource(), connect=connect)
            )
            await asyncio.sleep(0.05)  # let run() reach the stalled configure() send()
            task.cancel()
            await task

        run(scenario())

        saved_path = monkeypatch_dir / f"{r.session_id}.json"
        assert saved_path.exists(), "session was never saved despite an early cancellation"
    finally:
        runner_module.SESSIONS_DIR = original_sessions_dir


# --- ADR-044: the one tool the Interview Agent may call ----------------------------


def test_flag_technical_issue_schema_names_nothing_analytical():
    """Structural guard mirroring test_voice_agent_session_has_no_stress_score_input_
    path's spirit: the ONE tool ever offered to the Interview Agent must not, even by
    a future edit, grow a field that smuggles analysis in either direction."""
    import json

    schema_text = json.dumps(FLAG_TECHNICAL_ISSUE_SCHEMA).lower()
    for banned in ("stress", "arousal", "score", "deceptio", "lying", "truthful"):
        assert banned not in schema_text, f"{banned!r} found in the tool schema"


def test_on_tool_call_records_the_flag_and_forwards_it(model_path):
    statuses = []
    r = _make_runner(model_path, on_status=statuses.append)

    run(
        r._on_tool_call(
            {
                "type": "tool.call",
                "call_id": "call_1",
                "name": "flag_technical_issue",
                "arguments": {"issue": "candidate reported an echo"},
            }
        )
    )

    assert r.interview_session.technical_flags == ["candidate reported an echo"]
    assert any("echo" in s for s in statuses)


def test_on_tool_call_sends_a_tool_result_back_when_a_session_is_attached(model_path):
    sent = []

    class FakeVoiceSession:
        async def send_tool_result(self, call_id, result):
            sent.append((call_id, result))

    r = _make_runner(model_path)
    r._voice_session = FakeVoiceSession()

    run(
        r._on_tool_call(
            {"type": "tool.call", "call_id": "call_42", "name": "flag_technical_issue",
             "arguments": {"issue": "cutting out"}}
        )
    )

    assert sent == [("call_42", {"logged": True})]


def test_on_tool_call_does_not_crash_with_no_session_attached(model_path):
    r = _make_runner(model_path)
    assert r._voice_session is None
    run(
        r._on_tool_call(
            {"type": "tool.call", "call_id": "call_1", "name": "flag_technical_issue",
             "arguments": {"issue": "test"}}
        )
    )
    assert r.interview_session.technical_flags == ["test"]


def test_on_tool_call_rejects_an_unexpected_tool_name(model_path):
    """ADR-044: this dispatch must refuse a tool name it doesn't recognize rather than
    acting on it blindly — the guard that keeps a future second tool from being wired
    up silently by whatever the server happens to send."""
    statuses = []
    r = _make_runner(model_path, on_status=statuses.append)

    run(
        r._on_tool_call(
            {"type": "tool.call", "call_id": "call_1", "name": "get_arousal_score",
             "arguments": {}}
        )
    )

    assert r.interview_session.technical_flags == []
    assert any("unexpected tool name" in s for s in statuses)
