"""Exercises the Voice Agent protocol framing/dispatch with a fake in-memory transport
— no network, no API key. This is the ceiling of what's testable without a live
AssemblyAI connection; see ARCHITECTURE.md's note on the live-credential boundary.
"""
import asyncio
import base64
import json

import pytest

from voicestress.infrastructure.agents.voice_agent_client import VoiceAgentSession


class FakeTransport:
    """Records every outgoing message; replays a scripted queue of incoming ones."""

    def __init__(self, incoming: list[dict] | None = None):
        self.sent: list[dict] = []
        self._incoming = list(incoming or [])

    async def send(self, message: str) -> None:
        self.sent.append(json.loads(message))

    async def recv(self) -> str:
        if not self._incoming:
            raise ConnectionError("FakeTransport: no more scripted incoming messages")
        return json.dumps(self._incoming.pop(0))


def run(coro):
    return asyncio.run(coro)


def test_configure_sends_tools_and_turn_detection():
    transport = FakeTransport()
    session = VoiceAgentSession(transport=transport)

    run(session.configure(tools=[{"type": "function", "name": "x"}], system_prompt="be helpful"))

    assert len(transport.sent) == 1
    msg = transport.sent[0]
    assert msg["type"] == "session.update"
    assert msg["session"]["system_prompt"] == "be helpful"
    assert msg["session"]["tools"] == [{"type": "function", "name": "x"}]
    assert msg["session"]["input"]["turn_detection"]["interrupt_response"] is True
    assert "greeting" not in msg["session"]  # optional field, omitted when not passed


def test_send_audio_chunk_base64_roundtrips():
    transport = FakeTransport()
    session = VoiceAgentSession(transport=transport)
    raw_pcm = bytes([0, 1, 2, 3, 255, 254])

    run(session.send_audio_chunk(raw_pcm))

    msg = transport.sent[0]
    assert msg["type"] == "input.audio"
    assert base64.b64decode(msg["audio"]) == raw_pcm


def test_session_ready_captures_session_id():
    transport = FakeTransport(incoming=[{"type": "session.ready", "session_id": "sess_123"}])
    session = VoiceAgentSession(transport=transport)

    run(session.handle_next_event())

    assert session.session_id == "sess_123"


def test_session_updated_captures_session_id_from_nested_config():
    """The real server (confirmed 2026-09-05) never sends session.ready in this
    account's flow — session.updated, with the id nested at config.id, is what
    actually arrives. See the comment in voice_agent_client.py's _dispatch."""
    transport = FakeTransport(
        incoming=[
            {
                "type": "session.updated",
                "config": {"id": "sess_d9fba13ee0e3413d9fc445922ee28c52", "system_prompt": "hi"},
            }
        ]
    )
    session = VoiceAgentSession(transport=transport)

    run(session.handle_next_event())

    assert session.session_id == "sess_d9fba13ee0e3413d9fc445922ee28c52"


def test_tool_call_dispatches_to_handler():
    received = []

    async def on_tool_call(event):
        received.append(event)

    transport = FakeTransport(
        incoming=[
            {
                "type": "tool.call",
                "call_id": "call_1",
                "name": "get_turn_evidence",
                "arguments": {"turn_id": "t1"},
            }
        ]
    )
    session = VoiceAgentSession(transport=transport, on_tool_call=on_tool_call)

    run(session.handle_next_event())

    assert len(received) == 1
    assert received[0]["name"] == "get_turn_evidence"


def test_send_tool_result_includes_call_id():
    transport = FakeTransport()
    session = VoiceAgentSession(transport=transport)

    run(session.send_tool_result("call_1", {"score": 0.7}))

    msg = transport.sent[0]
    assert msg["type"] == "tool.result"
    assert msg["call_id"] == "call_1"
    assert json.loads(msg["result"]) == {"score": 0.7}


def test_reply_started_dispatches_to_handler():
    """ADR-040: SKILLS.md documented `reply.started|audio|done` on day one; nothing
    wired it until every live session had already run mute — the candidate's mic
    worked, the agent's reply never reached a human. This and the two tests below
    close that gap at the dispatch level."""
    received = []

    async def on_reply_started(event):
        received.append(event)

    transport = FakeTransport(incoming=[{"type": "reply.started"}])
    session = VoiceAgentSession(transport=transport, on_reply_started=on_reply_started)

    run(session.handle_next_event())

    assert len(received) == 1


def test_reply_audio_dispatches_the_raw_event_to_handler():
    """The audio payload's exact key is unconfirmed against a live connection (see the
    ASSUMPTION in live_interview_runner.py) — this dispatch layer stays agnostic and
    hands the whole event to the handler, exactly as on_tool_call already does."""
    received = []

    async def on_reply_audio(event):
        received.append(event)

    transport = FakeTransport(incoming=[{"type": "reply.audio", "audio": "c29tZWJ5dGVz"}])
    session = VoiceAgentSession(transport=transport, on_reply_audio=on_reply_audio)

    run(session.handle_next_event())

    assert received[0]["audio"] == "c29tZWJ5dGVz"


def test_reply_done_dispatches_to_handler():
    received = []

    async def on_reply_done(event):
        received.append(event)

    transport = FakeTransport(incoming=[{"type": "reply.done"}])
    session = VoiceAgentSession(transport=transport, on_reply_done=on_reply_done)

    run(session.handle_next_event())

    assert len(received) == 1


def test_unhandled_event_type_does_not_raise():
    transport = FakeTransport(incoming=[{"type": "session.updated"}])  # no callback registered
    session = VoiceAgentSession(transport=transport)
    run(session.handle_next_event())  # should not raise


def test_voice_agent_session_has_no_stress_score_input_path():
    """Structural guard for ARCHITECTURE.md §2 / SKILLS.md Hard Rule #2: nothing in this
    class's public surface accepts a stress/arousal score. If a future change adds e.g.
    `set_stress_score()` or a `stress`-named constructor field, this test should be the
    one that makes a reviewer stop and ask why."""
    import inspect

    members = {name for name, _ in inspect.getmembers(VoiceAgentSession)}
    fields = {f for f in VoiceAgentSession.__dataclass_fields__}
    suspicious = {m for m in members | fields if "stress" in m.lower() or "arousal" in m.lower()}
    assert suspicious == set()
