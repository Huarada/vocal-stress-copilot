"""AssemblyAI Voice Agent API client (ARCHITECTURE.md §5) — the Interview Agent adapter.

Protocol details verified against https://www.assemblyai.com/docs/voice-agents/voice-agent-api/api-spec/voice-agent-websocket
on 2026-09-04 (SKILLS.md §3): wss://agents.assemblyai.com/v1/ws, PCM16 24kHz base64
frames in JSON, session.update / input.audio / tool.result / reply.create client-side,
session.ready / input.speech.started|stopped / transcript.user / reply.* / tool.call /
session.error server-side.

The websocket connection itself is injected as a `Transport` protocol rather than opened
directly with the `websockets` library inline — this is what lets
tests/unit/test_voice_agent_client.py exercise the full message-framing and event-
dispatch logic with a fake, in-memory transport, with no network and no API key. Swap in
a real `websockets.connect(...)` at the composition root (scripts/run_interview.py, not
yet written — requires a live key to exercise end-to-end) for production use.

ARCHITECTURE.md §2 / SKILLS.md Hard Rule #2: this client exposes callbacks for
transcripts and speech events, but deliberately has NO callback or code path that lets
external stress/arousal scores flow back in and influence `reply.create`. Wiring a score
into this class's conversation control is the one change this file must never grow.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

PCM16_24K_MIME = "audio/pcm"
SAMPLE_RATE_HZ = 24_000


class Transport(Protocol):
    """Minimal async duck-type a real `websockets` connection satisfies, and that a
    fake can satisfy in tests without any network."""

    async def send(self, message: str) -> None: ...
    async def recv(self) -> str: ...


EventHandler = Callable[[dict[str, Any]], Awaitable[None] | None]


@dataclass
class VoiceAgentSession:
    """One live interview turn-taking session. Construct with an already-connected
    `Transport` (see module docstring); this class only knows the message protocol.
    """

    transport: Transport
    on_transcript_final: EventHandler | None = None
    on_speech_started: EventHandler | None = None
    on_speech_stopped: EventHandler | None = None
    on_agent_transcript: EventHandler | None = None
    on_tool_call: EventHandler | None = None
    on_session_ready: EventHandler | None = None
    on_session_updated: EventHandler | None = None
    on_session_ended: EventHandler | None = None
    on_error: EventHandler | None = None
    # ADDED 2026-09-05 (ADR-040): "reply.started" / "reply.audio" / "reply.done" were
    # documented in SKILLS.md's protocol notes from day one but never wired here or by
    # any composition root — every live run to date sent the candidate's mic audio to
    # the Voice Agent and never surfaced its reply, spoken or written, to a human. See
    # ADR-040 for why this is more than a missing feature.
    on_reply_started: EventHandler | None = None
    on_reply_audio: EventHandler | None = None
    on_reply_done: EventHandler | None = None

    session_id: str | None = field(default=None, init=False)

    async def configure(
        self,
        tools: list[dict[str, Any]],
        system_prompt: str,
        greeting: str | None = None,
        vad_threshold: float = 0.5,
        # DEFAULTS CORRECTED 2026-09-05: this method's original defaults (500ms /
        # 2000ms) were guessed, not sourced. The live docs state the platform's own
        # defaults as min_silence=1000ms, max_silence=3000ms — this session's
        # 500ms was more aggressive than AssemblyAI's own default and plausibly
        # contributed to the turn fragmentation observed on real runs ("I" / "I
        # think." / "gonna lie now." logged as three separate turns instead of one
        # sentence). Now matches the documented platform defaults.
        min_silence_ms: int = 1000,
        max_silence_ms: int = 3000,
        interrupt_response: bool = True,
    ) -> None:
        # Field name CONFIRMED against a real session.error response on 2026-09-05:
        # the first version of this method sent "instructions", which the live server
        # rejected with {"code": "invalid_format", "message": "Invalid message format
        # for type 'session.update'"}. "system_prompt" (and the rest of this shape) is
        # verified correct against a real "Build a Voice Agent in 5 Minutes" code sample
        # AND a real connection via scripts/check_live_connection.py.
        session: dict[str, Any] = {
            "system_prompt": system_prompt,
            "tools": tools,
            "input": {
                "turn_detection": {
                    "vad_threshold": vad_threshold,
                    "min_silence": min_silence_ms,
                    "max_silence": max_silence_ms,
                    "interrupt_response": interrupt_response,
                }
            },
        }
        if greeting is not None:
            session["greeting"] = greeting

        message = {"type": "session.update", "session": session}
        await self.transport.send(json.dumps(message))

    async def send_audio_chunk(self, pcm16_bytes: bytes) -> None:
        message = {
            "type": "input.audio",
            "audio": base64.b64encode(pcm16_bytes).decode("ascii"),
        }
        await self.transport.send(json.dumps(message))

    async def send_tool_result(self, call_id: str, result: Any) -> None:
        message = {
            "type": "tool.result",
            "call_id": call_id,
            "result": json.dumps(result) if not isinstance(result, str) else result,
        }
        await self.transport.send(json.dumps(message))

    async def request_reply(self, instructions: str | None = None) -> None:
        message: dict[str, Any] = {"type": "reply.create"}
        if instructions:
            message["instructions"] = instructions
        await self.transport.send(json.dumps(message))

    async def end_session(self) -> None:
        await self.transport.send(json.dumps({"type": "session.end"}))

    async def handle_next_event(self) -> dict[str, Any]:
        """Receives one server event and dispatches it to the matching callback.
        Callers drive the session by awaiting this in a loop. Returns the parsed event
        so a caller (e.g. the Sync Bus) can also inspect it directly."""
        raw = await self.transport.recv()
        event = json.loads(raw)
        await self._dispatch(event)
        return event

    async def _dispatch(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        handler = {
            "session.ready": self.on_session_ready,
            "session.updated": self.on_session_updated,
            # ADDED after a real disconnect (2026-09-05) that produced NO diagnostic
            # output: "session.ended" is a documented server->client event
            # (ARCHITECTURE.md §5's "session.error/ended: Error and teardown events")
            # that this dispatch table never handled at all — the event was silently
            # dropped (handler=None -> no-op), so a graceful server-initiated close
            # (trial session limit, policy, etc.) looked exactly like an unexplained
            # crash. Always wire on_session_ended in any live composition root.
            "session.ended": self.on_session_ended,
            "transcript.user": self.on_transcript_final,
            "input.speech.started": self.on_speech_started,
            "input.speech.stopped": self.on_speech_stopped,
            "transcript.agent": self.on_agent_transcript,
            "tool.call": self.on_tool_call,
            "session.error": self.on_error,
            "reply.started": self.on_reply_started,
            "reply.audio": self.on_reply_audio,
            "reply.done": self.on_reply_done,
        }.get(event_type)

        if event_type == "session.ready":
            self.session_id = event.get("session_id")
        elif event_type == "session.updated":
            # CONFIRMED against a real connection on 2026-09-05 (scripts/check_live_connection.py):
            # this account's flow never emits "session.ready" at all — the first (and
            # only, absent a greeting) server event is "session.updated", sent in
            # response to our session.update, with the session id nested at
            # config.id rather than top-level session_id. Docs describe both event
            # types; only this one was observed in practice, so both are handled.
            config = event.get("config") or {}
            session_id = config.get("id")
            if session_id:
                self.session_id = session_id

        if handler is None:
            return
        result = handler(event)
        if result is not None:  # supports both sync and async callbacks
            await result
