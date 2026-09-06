"""Browser-based capture composition root (ADR-041): bridges a FastAPI WebSocket (the
candidate's mic in, the agent's reply out) to `LiveInterviewRunner` — the SAME
orchestration `scripts/run_interview.py`'s real-microphone path uses, so this file
duplicates none of that module's ~15 documented bug fixes (ADR.md Appendix A).

Framing over the one WebSocket connection to the browser:
  browser -> server, binary frames : mono int16 PCM at SAMPLE_RATE_HZ (candidate audio)
  browser -> server, text (JSON)   : {"type": "handshake", "sample_rate_hz": N} — REQUIRED
                                      as the very first message; {"type": "end_session"}
                                      at any later point to stop gracefully
  server -> browser, text (JSON)   : {"type": "status"|"turn"|"agent_text"|"error"
                                      |"session_started"|"session_saved", ...}
  server -> browser, binary frames : the agent's synthesized reply audio — raw PCM16
                                      bytes, no base64 (that encoding exists only inside
                                      AssemblyAI's own JSON protocol; this is our own,
                                      simpler framing for the browser hop)

All outgoing frames go through one `asyncio.Queue` drained by a single writer task.
`LiveInterviewRunner`'s callbacks (on_status/on_turn_finalized/on_agent_reply_*) fire
from `runner.run()`'s own asyncio Task — a different task than this handler's receive
loop — and two tasks writing to one ASGI WebSocket concurrently is not safe without
serializing them (the protocol assumes one sender in flight at a time per connection).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Awaitable, Callable

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from voicestress.application.live_interview_runner import SAMPLE_RATE_HZ, LiveInterviewRunner
from voicestress.config import AssemblyAISettings

HANDSHAKE_TIMEOUT_S = 10


class WebSocketAudioSource:
    """The `AudioSource` the browser capture path hands to `LiveInterviewRunner.run()`.
    Unlike `MicAudioSource` (which owns a real audio thread), this one owns nothing —
    the FastAPI receive loop already reads the socket, so `start()` just records where
    to send each chunk, and `push()` is called from that loop directly."""

    def __init__(self) -> None:
        self._feed: Callable[[np.ndarray], None] | None = None

    def start(self, feed: Callable[[np.ndarray], None]) -> None:
        self._feed = feed

    def stop(self) -> None:
        self._feed = None

    def push(self, chunk: np.ndarray) -> None:
        if self._feed is not None:
            self._feed(chunk)


async def run_capture_session(
    ws: WebSocket,
    settings: AssemblyAISettings,
    model_path: Path,
    connect: Callable[[], Awaitable[Any]] | None = None,
) -> None:
    """Owns one browser capture session end to end: handshake, runner lifecycle,
    receive loop, and graceful shutdown on either a client "end_session" message or a
    real disconnect. `connect` overrides how `LiveInterviewRunner.run()` reaches the
    Voice Agent API — production code leaves it as the default (a real
    `WebsocketTransport.connect(...)`); tests inject a fake so this whole function is
    exercisable without a live ASSEMBLYAI_API_KEY (see test_live_capture.py).
    """
    await ws.accept()

    outgoing: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    def enqueue(kind: str, payload: Any) -> None:
        outgoing.put_nowait((kind, payload))

    # --- handshake -------------------------------------------------------------
    # SKILLS.md Hard Rule 5: assert sample rate at every boundary. capture.js asks the
    # browser's AudioContext for SAMPLE_RATE_HZ directly, but not every browser honours
    # a requested context rate (Safari in particular can silently ignore it and hand
    # back the hardware rate instead). Trust nothing the client claims implicitly —
    # require it to state its actual rate explicitly, and reject cleanly rather than
    # feed possibly-wrong-rate audio into a pipeline that assumes SAMPLE_RATE_HZ
    # everywhere downstream (SyncBus, EvidenceService, VoiceAgentSession).
    try:
        handshake = await asyncio.wait_for(ws.receive_json(), timeout=HANDSHAKE_TIMEOUT_S)
    except asyncio.TimeoutError:
        await _reject(ws, f"No handshake received within {HANDSHAKE_TIMEOUT_S}s.")
        return
    except (json.JSONDecodeError, WebSocketDisconnect) as e:
        await _reject(ws, f"Malformed or missing handshake: {type(e).__name__}: {e}")
        return

    client_rate = handshake.get("sample_rate_hz")
    if client_rate != SAMPLE_RATE_HZ:
        await _reject(
            ws,
            f"Capture rate mismatch: browser reports {client_rate!r}Hz, this pipeline "
            f"requires exactly {SAMPLE_RATE_HZ}Hz. Resample client-side before sending.",
        )
        return

    if not model_path.exists():
        await _reject(ws, f"Arousal model not found at {model_path} — cannot start a session.")
        return

    try:
        runner = LiveInterviewRunner(
            settings=settings,
            model_path=model_path,
            on_status=lambda msg: enqueue("status", msg),
            on_turn_finalized=lambda evidence: enqueue("turn", evidence.to_evidence_dict()),
            on_agent_reply_text=lambda text: enqueue("agent_text", text),
            on_agent_reply_audio=lambda pcm: enqueue("agent_audio", pcm),
        )
    except Exception as e:  # noqa: BLE001 — printed per ADR-027, never swallowed
        await _reject(ws, f"Failed to initialize the session: {type(e).__name__}: {e}")
        return

    audio_source = WebSocketAudioSource()
    writer_task = asyncio.create_task(_writer_loop(ws, outgoing))
    run_task = asyncio.create_task(runner.run(audio_source=audio_source, connect=connect))

    await ws.send_json({"type": "session_started", "session_id": runner.session_id})

    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break

            data_bytes = message.get("bytes")
            data_text = message.get("text")

            if data_bytes is not None:
                if len(data_bytes) % 2 != 0:
                    # An odd byte count cannot be a whole number of int16 samples —
                    # np.frombuffer would raise. Drop the frame with a diagnostic
                    # rather than crash the whole session over one malformed chunk.
                    enqueue(
                        "status",
                        f"[capture] dropped a {len(data_bytes)}-byte frame: not a "
                        f"whole number of int16 samples",
                    )
                    continue
                audio_source.push(np.frombuffer(data_bytes, dtype=np.int16))

            elif data_text is not None:
                try:
                    control = json.loads(data_text)
                except json.JSONDecodeError:
                    enqueue(
                        "status",
                        f"[capture] ignored malformed control message: {data_text[:200]!r}",
                    )
                    continue
                if control.get("type") == "end_session":
                    break
                enqueue(
                    "status",
                    f"[capture] ignored unknown control message type: {control.get('type')!r}",
                )
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001 — ADR-027: every branch prints, never bare-passes
        enqueue("status", f"[capture] receive loop ended unexpectedly: {type(e).__name__}: {e}")
    finally:
        # Cancelling run_task raises CancelledError inside runner.run()'s own try
        # block — that method's `except asyncio.CancelledError` branch does NOT
        # re-raise (ADR-041), so awaiting it here returns normally once its own
        # `finally` (flush pending turns, save the session) has actually completed.
        run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001 — the runner's own catch-all should have
            # already handled everything; this is a last-resort net so a truly
            # unexpected failure here is still visible instead of vanishing into an
            # unretrieved-task-exception warning.
            enqueue("status", f"[capture] runner task ended unexpectedly: {type(e).__name__}: {e}")

        # ADR-042: this used to claim "session_saved" unconditionally the moment
        # run_task finished — which was true whenever a cancellation landed inside
        # run()'s own try/finally, but a live test found a cancellation landing BEFORE
        # that finally (during configure(), before the event loop even started) that
        # skipped the save entirely while this still reported success. run()'s finally
        # now wraps its whole body, closing that specific race, but claiming success
        # is still worth verifying rather than assuming — this checks the file the
        # runner actually claims to have written, not just that its task returned.
        # Imported locally, not at module load: SESSIONS_DIR is monkeypatched per-test
        # (test_live_capture.py's `sessions_dir` fixture) precisely because
        # backend.py's and this module's copy resolve to the same real path in
        # production (Appendix A row 38) — a top-level import here would bind the
        # ORIGINAL value before any test could patch it.
        from voicestress.application.live_interview_runner import SESSIONS_DIR

        expected_path = SESSIONS_DIR / f"{runner.session_id}.json"
        outcome = (
            {"type": "session_saved", "session_id": runner.session_id}
            if expected_path.exists()
            else {
                "type": "error",
                "message": (
                    f"Session ended but no file was written to {expected_path}. "
                    "Check the status log above for what the runner reported."
                ),
            }
        )
        try:
            await ws.send_json(outcome)
        except Exception:  # noqa: BLE001 — client is very likely already gone; nothing
            # more to send, and this is cleanup code, not a fresh failure to surface.
            pass

        # Let the writer drain whatever the runner's own shutdown enqueued (its
        # "Saved session evidence to ..." status, the flushed final turns) before
        # tearing it down, but don't hang forever if a send is stuck.
        try:
            await asyncio.wait_for(outgoing.join(), timeout=2)
        except asyncio.TimeoutError:
            pass
        writer_task.cancel()
        try:
            await writer_task
        except asyncio.CancelledError:
            pass


async def _writer_loop(ws: WebSocket, outgoing: asyncio.Queue[tuple[str, Any]]) -> None:
    kind_to_message = {
        "status": lambda payload: {"type": "status", "message": payload},
        "turn": lambda payload: {"type": "turn", "turn": payload},
        "agent_text": lambda payload: {"type": "agent_text", "text": payload},
    }
    while True:
        kind, payload = await outgoing.get()
        try:
            if kind == "agent_audio":
                await ws.send_bytes(payload)
            else:
                await ws.send_json(kind_to_message[kind](payload))
        except Exception as e:  # noqa: BLE001 — the connection is very likely gone;
            # print (ADR-027) and stop rather than spin retrying sends that will keep
            # failing the same way.
            print(f"[capture writer] send failed, stopping: {type(e).__name__}: {e}")
            outgoing.task_done()
            return
        outgoing.task_done()


async def _reject(ws: WebSocket, message: str) -> None:
    print(f"[capture] rejected: {message}")
    try:
        await ws.send_json({"type": "error", "message": message})
    except Exception:  # noqa: BLE001 — best-effort notice; closing below is what matters
        pass
    await ws.close(code=1008)
