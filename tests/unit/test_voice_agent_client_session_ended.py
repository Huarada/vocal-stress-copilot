import asyncio
import json

from voicestress.infrastructure.agents.voice_agent_client import VoiceAgentSession


class FakeTransport:
    def __init__(self, incoming):
        self.sent = []
        self._incoming = list(incoming)

    async def send(self, message):
        self.sent.append(json.loads(message))

    async def recv(self):
        return json.dumps(self._incoming.pop(0))


def run(coro):
    return asyncio.run(coro)


def test_session_ended_event_reaches_registered_handler():
    """Regression test for the 2026-09-05 finding: a real live run disconnected after
    one turn with zero diagnostic output, because 'session.ended' had no dispatch
    entry at all and was silently dropped. This must never regress back to silence."""
    received = []

    async def on_ended(event):
        received.append(event)

    transport = FakeTransport(
        incoming=[{"type": "session.ended", "reason": "trial_session_limit_reached"}]
    )
    session = VoiceAgentSession(transport=transport, on_session_ended=on_ended)

    run(session.handle_next_event())

    assert len(received) == 1
    assert received[0]["reason"] == "trial_session_limit_reached"


def test_session_ended_without_handler_does_not_raise():
    transport = FakeTransport(incoming=[{"type": "session.ended", "reason": "x"}])
    session = VoiceAgentSession(transport=transport)  # no on_session_ended registered
    run(session.handle_next_event())  # must not raise
